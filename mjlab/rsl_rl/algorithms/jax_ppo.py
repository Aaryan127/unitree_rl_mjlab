"""Rollout functions using lax.scan for efficient trajectory collection.

This module provides JIT-compiled rollout functions that use JAX's lax.scan
for fully GPU-accelerated parallel trajectory collection.
"""

import jax
import jax.numpy as jp
from jax import lax, random
from typing import NamedTuple, Tuple, Any
import orbax.checkpoint as ocp

from mjlab.rsl_rl.networks.jax_networks import (
    ActorCriticRNN,
    sample_actions,
    get_value,
    get_action_mean,
)


class RolloutStorage(NamedTuple):
    """Storage for rollout data.

    Contains all data needed for PPO training.
    """

    observations: jp.ndarray
    """Observation history (num_steps, batch, obs_dim)."""

    actions: jp.ndarray
    """Actions taken (num_steps, batch, action_dim)."""

    rewards: jp.ndarray
    """Rewards received (num_steps, batch)."""

    dones: jp.ndarray
    """Done flags (num_steps, batch)."""

    values: jp.ndarray
    """Value estimates (num_steps, batch)."""

    log_probs: jp.ndarray
    """Log probabilities (num_steps, batch)."""

    episode_returns: jp.ndarray
    """Cumulative returns for episode (batch,)."""

    episode_lengths: jp.ndarray
    """Episode lengths (batch,)."""


def rollout(
    env_reset_fn,
    env_step_fn,
    policy_fn,
    value_fn,
    num_steps: int,
    num_envs: int,
    key: random.PRNGKey,
) -> Tuple[RolloutStorage, Any]:
    """Collect rollout using lax.scan for efficiency.

    Args:
        env_reset_fn: Function to reset environment.
        env_step_fn: Function to step environment.
        policy_fn: Function to get action from policy.
        value_fn: Function to get value from critic.
        num_steps: Number of steps per rollout.
        num_envs: Number of parallel environments.
        key: JAX random key.

    Returns:
        Tuple of (RolloutStorage, final_env_state).
    """

    def rollout_step(carry, step_idx):
        state, key = carry

        key, policy_key = random.split(key)

        obs = state.data.qpos

        actions, log_probs = policy_fn(obs, policy_key)

        values = value_fn(obs)

        key, env_key = random.split(key)
        new_state, rewards, dones, _ = env_step_fn(state, actions, env_key)

        return (new_state, key), (obs, actions, rewards, dones, values, log_probs)

    key, init_key = random.split(key)
    state = env_reset_fn(init_key)

    (final_state, _), (observations, actions, rewards, dones, values, log_probs) = (
        lax.scan(
            rollout_step,
            (state, key),
            None,
            length=num_steps,
        )
    )

    episode_returns = rewards.sum(axis=0)
    episode_lengths = jp.full(num_envs, num_steps)

    storage = RolloutStorage(
        observations=observations,
        actions=actions,
        rewards=rewards,
        dones=dones,
        values=values,
        log_probs=log_probs,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )

    return storage, final_state


def compute_gae(
    rewards: jp.ndarray,
    values: jp.ndarray,
    dones: jp.ndarray,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards (num_steps, batch).
        values: Value estimates (num_steps, batch).
        dones: Done flags (num_steps, batch).
        gamma: Discount factor.
        lam: GAE lambda parameter.

    Returns:
        Tuple of (advantages, returns).
    """
    num_steps = rewards.shape[0]
    advantages = jp.zeros_like(rewards)
    gae = 0

    for t in reversed(range(num_steps)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values[:-1]

    return advantages, returns


class PPOConfig(NamedTuple):
    """PPO configuration."""

    num_steps_per_env: int = 24
    num_mini_batches: int = 4
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    use_clipped_value_loss: bool = True


def ppo_loss(
    params: jp.ndarray,
    network: ActorCriticRNN,
    batch: RolloutStorage,
    advantages: jp.ndarray,
    returns: jp.ndarray,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
) -> Tuple[jp.ndarray, dict]:
    """Compute PPO loss.

    Args:
        params: Network parameters.
        network: Actor-critic network.
        batch: Rollout storage.
        advantages: Advantages (batch,).
        returns: Returns (batch,).
        clip_epsilon: PPO clipping epsilon.
        value_loss_coef: Value loss coefficient.
        entropy_coef: Entropy coefficient.

    Returns:
        Tuple of (total_loss, loss_dict).
    """
    observations = batch.observations.reshape(-1, batch.observations.shape[-1])
    actions = batch.actions.reshape(-1, batch.actions.shape[-1])
    old_log_probs = batch.log_probs.reshape(-1)
    old_values = batch.values.reshape(-1)
    advantages_flat = advantages.reshape(-1)
    returns_flat = returns.reshape(-1)

    mean, value_pred, log_std = network.apply(params, observations)

    std = jp.exp(jp.clip(log_std, -20, 2))
    log_probs = -0.5 * (
        ((actions - mean) ** 2) / (std**2) + 2 * log_std + jp.log(2 * jp.pi)
    ).sum(axis=-1)

    ratio = jp.exp(log_probs - old_log_probs)

    surr1 = ratio * advantages_flat
    surr2 = jp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_flat
    policy_loss = -jp.minimum(surr1, surr2).mean()

    value_pred_clipped = old_values + jp.clip(
        value_pred.squeeze() - old_values, -clip_epsilon, clip_epsilon
    )
    value_loss1 = (value_pred.squeeze() - returns_flat) ** 2
    value_loss2 = (value_pred_clipped - returns_flat) ** 2
    value_loss = jp.maximum(value_loss1, value_loss2).mean()

    entropy_loss = -0.5 * (1 + jp.log(2 * jp.pi * std**2)).sum(axis=-1).mean()

    total_loss = (
        policy_loss + value_loss_coef * value_loss + entropy_coef * entropy_loss
    )

    loss_dict = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
    }

    return total_loss, loss_dict


def train_step(
    train_state,
    network: ActorCriticRNN,
    batch: RolloutStorage,
    config: PPOConfig,
) -> Tuple[train_state, dict]:
    """Single PPO training step.

    Args:
        train_state: Training state with params and optimizer.
        network: Actor-critic network.
        batch: Rollout storage with collected data.
        config: PPO configuration.

    Returns:
        Tuple of (new_train_state, metrics).
    """
    advantages, returns = compute_gae(
        batch.rewards,
        batch.values,
        batch.dones,
        config.gamma,
        config.lam,
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    num_samples = batch.observations.shape[1]
    mini_batch_size = num_samples // config.num_mini_batches

    def update_epoch(carry, _):
        (train_state,) = carry

        key = random.PRNGKey(0)
        key, perm_key = random.split(key)
        indices = random.permutation(perm_key, num_samples)

        mini_batches = jax.lax.map(
            lambda i: indices[i * mini_batch_size : (i + 1) * mini_batch_size],
            jp.arange(config.num_mini_batches),
        )

        def update_mini_batch(carry, idx):
            (train_state,) = carry

            mb_obs = batch.observations[:, idx, :]
            mb_actions = batch.actions[:, idx, :]
            mb_old_log_probs = batch.log_probs[:, idx]
            mb_old_values = batch.values[:, idx]
            mb_advantages = advantages[:, idx]
            mb_returns = returns[:, idx]

            mb_batch = RolloutStorage(
                observations=mb_obs,
                actions=mb_actions,
                rewards=batch.rewards[:, idx],
                dones=batch.dones[:, idx],
                values=mb_old_values,
                log_probs=mb_old_log_probs,
                episode_returns=batch.episode_returns[idx],
                episode_lengths=batch.episode_lengths[idx],
            )

            def loss_fn(params):
                return ppo_loss(
                    params,
                    network,
                    mb_batch,
                    mb_advantages,
                    mb_returns,
                    config.clip_epsilon,
                    config.value_loss_coef,
                    config.entropy_coef,
                )

            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )

            grads = jax.lax.clamp(grads, -config.max_grad_norm, config.max_grad_norm)

            updates, new_opt_state = train_state.opt_state.update(grads)
            new_params = optax.apply_updates(train_state.params, updates)

            new_train_state = train_state.replace(
                params=new_params,
                opt_state=new_opt_state,
            )

            return (new_train_state,), metrics

        (train_state,), metrics = jax.lax.scan(
            update_mini_batch,
            (train_state,),
            None,
            length=config.num_mini_batches,
        )

        return (train_state,), metrics

    train_state, metrics = jax.lax.scan(
        update_epoch,
        (train_state,),
        None,
        length=config.num_epochs,
    )

    return train_state, metrics
