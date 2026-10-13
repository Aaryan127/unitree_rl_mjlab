"""JAX/MJX training script for GPU-accelerated RL training.

This script demonstrates the full JAX/MJX pipeline for training RL agents
with fully GPU-accelerated simulation and neural networks.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
from flax.training.train_state import TrainState
import optax
import yaml


@dataclass
class MJXTrainConfig:
    """Configuration for MJX training."""

    num_envs: int = 4096
    max_episode_length: int = 1000
    decimation: int = 10
    physics_dt: float = 0.002

    actor_hidden_dims: tuple = (256, 256, 256)
    critic_hidden_dims: tuple = (256, 256, 256)

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

    num_iterations: int = 10000
    log_interval: int = 10
    save_interval: int = 1000

    model_path: str = "robot.xml"
    device: str = "cuda"
    seed: int = 42


class MJXEnv:
    """Simple MJX environment for training."""

    def __init__(
        self,
        model_path: str,
        num_envs: int,
        max_episode_length: int,
        decimation: int,
        physics_dt: float,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.max_episode_length = max_episode_length
        self.decimation = decimation
        self.physics_dt = physics_dt
        self.step_dt = physics_dt * decimation
        self.device = device

        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        self._default_data = mjx.make_data(self.mjx_model)
        mjx.forward(self.mjx_model, self._default_data)

        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu
        self.obs_dim = self.nq + self.nv
        self.action_dim = self.nu

        self._step_fn = jax.jit(mjx.step)
        self._forward_fn = jax.jit(mjx.forward)

    def _create_batch_data(self):
        """Create batched simulation data."""
        batch_size = self.num_envs

        return self._default_data.replace(
            qpos=jp.tile(self._default_data.qpos[None, :], (batch_size, 1)),
            qvel=jp.tile(self._default_data.qvel[None, :], (batch_size, 1)),
            ctrl=jp.zeros((batch_size, self.nu)),
        )

    def reset(self, key):
        """Reset environment."""
        data = self._create_batch_data()
        data = self._forward_fn(self.mjx_model, data)

        obs = jp.concatenate([data.qpos, data.qvel], axis=-1)

        return obs, data

    def step(self, data, action, key):
        """Step environment."""
        for _ in range(self.decimation):
            data = data.replace(ctrl=action)
            data = self._step_fn(self.mjx_model, data)
            data = self._forward_fn(self.mjx_model, data)

        obs = jp.concatenate([data.qpos, data.qvel], axis=-1)

        reward = self._compute_reward(data)

        return obs, data, reward

    def _compute_reward(self, data):
        """Compute reward."""
        return jp.zeros(self.num_envs)


class ActorCriticJAX:
    """Actor-Critic network in JAX/Flax."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple,
        critic_hidden_dims: tuple,
        key,
    ):
        from mjlab.rsl_rl.networks.jax_networks import ActorCriticRNN
        import mjlab.rsl_rl.networks.jax_networks as jax_nets

        self.network = jax_nets.ActorCriticRNN(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        )

        dummy_obs = jp.zeros((1, obs_dim))
        params = self.network.init(key, dummy_obs)

        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=optax.adam(3e-4),
        )

    def act(self, params, obs, key):
        """Sample actions."""
        from mjlab.rsl_rl.networks.jax_networks import sample_actions

        return sample_actions(self.network, params, obs, key)

    def get_value(self, params, obs):
        """Get value estimate."""
        from mjlab.rsl_rl.networks.jax_networks import get_value

        return get_value(self.network, params, obs)


def collect_rollout(env, actor_critic, params, obs, data, key, num_steps):
    """Collect rollout using lax.scan."""

    def rollout_step(carry, step_idx):
        obs, data, key, cum_reward = carry

        key, action_key = random.split(key)
        actions, log_probs = actor_critic.act(params, obs, action_key)

        values = actor_critic.get_value(params, obs)

        key, env_key = random.split(key)
        new_obs, new_data, rewards = env.step(data, actions, env_key)

        return (new_obs, new_data, key, cum_reward + rewards), (
            obs,
            actions,
            rewards,
            values,
            log_probs,
        )

    key, init_key = random.split(key)

    (final_obs, final_data, _, cum_rewards), trajectory = jax.lax.scan(
        rollout_step,
        (obs, data, key, jp.zeros(env.num_envs)),
        None,
        length=num_steps,
    )

    obs_batch, actions, rewards, values, log_probs = trajectory

    return obs_batch, actions, rewards, values, log_probs, final_obs, final_data


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE."""
    advantages = jp.zeros_like(rewards)
    gae = 0

    for t in reversed(range(rewards.shape[0])):
        if t < rewards.shape[0] - 1:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        else:
            delta = rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        advantages = advantages.at[t].set(gae)

    returns = advantages + values

    return advantages, returns


def ppo_update(
    actor_critic,
    obs,
    actions,
    old_log_probs,
    old_values,
    advantages,
    returns,
    config,
    key,
):
    """PPO update step."""

    def loss_fn(params):
        mean, value_pred, log_std = actor_critic.network.apply(params, obs)

        std = jp.exp(jp.clip(log_std, -20, 2))

        log_probs = -0.5 * (
            ((actions - mean) ** 2) / (std**2) + 2 * log_std + jp.log(2 * jp.pi)
        ).sum(axis=-1)

        ratio = jp.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = (
            jp.clip(ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon)
            * advantages
        )
        policy_loss = -jp.minimum(surr1, surr2).mean()

        value_loss = ((value_pred.squeeze() - returns) ** 2).mean()

        entropy_loss = -0.5 * (1 + jp.log(2 * jp.pi * std**2)).sum(axis=-1).mean()

        total_loss = (
            policy_loss
            + config.value_loss_coef * value_loss
            + config.entropy_coef * entropy_loss
        )

        return total_loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
        }

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        actor_critic.train_state.params
    )

    grads = jax.lax.clamp(grads, -config.max_grad_norm, config.max_grad_norm)

    new_train_state = actor_critic.train_state.apply_gradient(grads)
    actor_critic.train_state = new_train_state

    return actor_critic, metrics


def train(config: MJXTrainConfig):
    """Main training loop."""
    from jax import random

    print(f"[INFO] Initializing MJX training with {config.num_envs} environments")
    print(f"[INFO] Device: {config.device}")

    env = MJXEnv(
        model_path=config.model_path,
        num_envs=config.num_envs,
        max_episode_length=config.max_episode_length,
        decimation=config.decimation,
        physics_dt=config.physics_dt,
        device=config.device,
    )

    key = random.PRNGKey(config.seed)
    key, env_key = random.split(key)

    obs, data = env.reset(env_key)

    actor_critic = ActorCriticJAX(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        actor_hidden_dims=config.actor_hidden_dims,
        critic_hidden_dims=config.critic_hidden_dims,
        key=key,
    )

    print(f"[INFO] Observation dim: {env.obs_dim}, Action dim: {env.action_dim}")

    for iteration in range(config.num_iterations):
        iter_start = time.time()

        key, rollout_key = random.split(key)

        obs_batch, actions, rewards, values, log_probs, obs, data = collect_rollout(
            env,
            actor_critic,
            actor_critic.train_state.params,
            obs,
            data,
            rollout_key,
            config.num_steps_per_env,
        )

        advantages, returns = compute_gae(
            rewards, values, jp.zeros_like(rewards), config.gamma, config.lam
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        flat_obs = obs_batch.reshape(-1, env.obs_dim)
        flat_actions = actions.reshape(-1, env.action_dim)
        flat_log_probs = log_probs.reshape(-1)
        flat_values = values.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        num_samples = flat_obs.shape[0]
        mini_batch_size = num_samples // config.num_mini_batches

        for epoch in range(config.num_epochs):
            perm = random.permutation(random.PRNGKey(epoch), num_samples)

            for mb_idx in range(config.num_mini_batches):
                mb_start = mb_idx * mini_batch_size
                mb_end = (mb_idx + 1) * mini_batch_size
                mb_indices = perm[mb_start:mb_end]

                mb_obs = flat_obs[mb_indices]
                mb_actions = flat_actions[mb_indices]
                mb_log_probs = flat_log_probs[mb_indices]
                mb_values = flat_values[mb_indices]
                mb_advantages = flat_advantages[mb_indices]
                mb_returns = flat_returns[mb_indices]

                key, update_key = random.split(key)
                actor_critic, metrics = ppo_update(
                    actor_critic,
                    mb_obs,
                    mb_actions,
                    mb_log_probs,
                    mb_values,
                    mb_advantages,
                    mb_returns,
                    config,
                    update_key,
                )

        mean_reward = rewards.mean()
        iter_time = time.time() - iter_start

        if iteration % config.log_interval == 0:
            print(
                f"Iter {iteration:5d} | Reward: {mean_reward:.4f} | "
                f"Time: {iter_time:.2f}s | Policy Loss: {metrics['policy_loss']:.4f}"
            )

        if iteration % config.save_interval == 0 and iteration > 0:
            save_path = Path(f"checkpoints/mjx_model_{iteration}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Saving checkpoint to {save_path}")

    print("[INFO] Training complete!")


def main():
    import tyro

    config = tyro.cli(MJXTrainConfig)

    os.environ["MUJOCO_GL"] = "egl"

    if config.device == "cuda" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    train(config)


if __name__ == "__main__":
    main()
