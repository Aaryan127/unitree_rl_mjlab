"""Flax (JAX) implementations of neural networks for RL.

This module provides JAX/Flax versions of the neural networks
for fully GPU-accelerated training.
"""

import jax
import jax.numpy as jp
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Sequence, Callable, Any, Tuple
import optax


class MLP(nn.Module):
    """Multi-layer perceptron in Flax.

    A feedforward neural network with configurable hidden layers
    and activation functions.
    """

    hidden_dims: Sequence[int]
    """Dimensions of hidden layers."""

    activation: Callable = nn.leaky_relu
    """Activation function. Defaults to leaky_relu."""

    activate_final: bool = False
    """Whether to apply activation to final layer."""

    use_bias: bool = True
    """Whether to use bias in linear layers."""

    dtype: Any = jp.float32
    """Data type for computations."""

    @nn.compact
    def __call__(self, x: jp.ndarray) -> jp.ndarray:
        """Forward pass.

        Args:
            x: Input array.

        Returns:
            Output array.
        """
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, use_bias=self.use_bias, dtype=self.dtype)(x)
            x = self.activation(x)

        return x


class ActorCriticRNN(nn.Module):
    """Actor-Critic network with optional recurrent processing.

    This network computes policy (actor) and value (critic) outputs
    from observations. Supports state-dependent noise for exploration.
    """

    num_actor_obs: int
    """Dimension of actor observations."""

    num_critic_obs: int
    """Dimension of critic observations."""

    num_actions: int
    """Number of actions."""

    actor_hidden_dims: Sequence[int] = (256, 256, 256)
    """Hidden layer dimensions for actor."""

    critic_hidden_dims: Sequence[int] = (256, 256, 256)
    """Hidden layer dimensions for critic."""

    activation: Callable = nn.leaky_relu
    """Activation function."""

    state_dependent_std: bool = False
    """Whether to use state-dependent standard deviation."""

    init_noise_std: float = 1.0
    """Initial noise standard deviation."""

    @nn.compact
    def __call__(
        self,
        observations: jp.ndarray,
        actions: jp.ndarray | None = None,
        hidden_state: jp.ndarray | None = None,
    ) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray]:
        """Forward pass.

        Args:
            observations: Observation array (batch, obs_dim).
            actions: Actions for evaluation (batch, action_dim). Optional.
            hidden_state: RNN hidden state. Optional.

        Returns:
            Tuple of (actions, values, log_stds).
        """
        actor_obs = observations[:, : self.num_actor_obs]
        critic_obs = observations[:, : self.num_critic_obs]

        actor_mean = MLP(
            hidden_dims=self.actor_hidden_dims,
            activation=self.activation,
            activate_final=False,
        )(actor_obs)

        actor_mean = nn.Dense(self.num_actions, use_bias=True)(actor_mean)

        if self.state_dependent_std:
            log_std = MLP(
                hidden_dims=self.actor_hidden_dims,
                activation=self.activation,
                activate_final=False,
            )(actor_obs)
            log_std = nn.Dense(self.num_actions, use_bias=True)(log_std)
            log_std = jp.clip(log_std, -20, 2)
        else:
            log_std = jp.full((self.num_actions,), jp.log(self.init_noise_std))

        if actions is None:
            return actor_mean, jp.zeros((observations.shape[0], 1)), log_std

        critic = MLP(
            hidden_dims=self.critic_hidden_dims,
            activation=self.activation,
            activate_final=False,
        )(critic_obs)
        value = nn.Dense(1, use_bias=False)(critic)

        return actor_mean, value, log_std


class Normal(nn.Module):
    """Normal distribution for policy.

    Provides differentiable sampling and log probability computation.
    """

    @nn.compact
    def __call__(
        self, mean: jp.ndarray, log_std: jp.ndarray
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """Create normal distribution and sample.

        Args:
            mean: Mean of distribution.
            log_std: Log standard deviation.

        Returns:
            Tuple of (sample, log_prob).
        """
        std = jp.exp(log_std)

        noise = jax.random.normal(jax.random.PRNGKey(0), mean.shape)

        sample = mean + std * noise

        log_prob = -0.5 * (
            ((sample - mean) ** 2) / (std**2) + 2 * log_std + jp.log(2 * jp.pi)
        )

        return sample, log_prob.sum(axis=-1)


def create_actor_critic(
    num_actor_obs: int,
    num_critic_obs: int,
    num_actions: int,
    actor_hidden_dims: Sequence[int] = (256, 256, 256),
    critic_hidden_dims: Sequence[int] = (256, 256, 256),
    state_dependent_std: bool = False,
    init_noise_std: float = 1.0,
    key: jp.ndarray | None = None,
) -> Tuple[ActorCriticRNN, jp.ndarray]:
    """Create actor-critic networks with initialized parameters.

    Args:
        num_actor_obs: Dimension of actor observations.
        num_critic_obs: Dimension of critic observations.
        num_actions: Number of actions.
        actor_hidden_dims: Hidden layer dimensions for actor.
        critic_hidden_dims: Hidden layer dimensions for critic.
        state_dependent_std: Whether to use state-dependent std.
        init_noise_std: Initial noise standard deviation.
        key: JAX random key.

    Returns:
        Tuple of (network, params).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    network = ActorCriticRNN(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        state_dependent_std=state_dependent_std,
        init_noise_std=init_noise_std,
    )

    dummy_obs = jp.zeros((1, num_actor_obs))
    params = network.init(key, dummy_obs)

    return network, params


def sample_actions(
    network: ActorCriticRNN,
    params: jp.ndarray,
    observations: jp.ndarray,
    key: jax.random.PRNGKey,
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Sample actions from policy.

    Args:
        network: Actor-critic network.
        params: Network parameters.
        observations: Observations (batch, obs_dim).
        key: JAX random key.

    Returns:
        Tuple of (actions, log_probs).
    """
    mean, value, log_std = network.apply(params, observations)

    std = jp.exp(jp.clip(log_std, -20, 2))

    noise = jax.random.normal(key, mean.shape)
    actions = mean + std * noise

    log_probs = -0.5 * (
        ((actions - mean) ** 2) / (std**2) + 2 * log_std + jp.log(2 * jp.pi)
    ).sum(axis=-1)

    return actions, log_probs


def get_value(
    network: ActorCriticRNN,
    params: jp.ndarray,
    observations: jp.ndarray,
) -> jp.ndarray:
    """Get value estimate from critic.

    Args:
        network: Actor-critic network.
        params: Network parameters.
        observations: Observations.

    Returns:
        Value estimates.
    """
    mean, value, log_std = network.apply(params, observations)
    return value.squeeze(-1)


def get_action_mean(
    network: ActorCriticRNN,
    params: jp.ndarray,
    observations: jp.ndarray,
) -> jp.ndarray:
    """Get deterministic action (mean) from policy.

    Args:
        network: Actor-critic network.
        params: Network parameters.
        observations: Observations.

    Returns:
        Action means.
    """
    mean, _, _ = network.apply(params, observations)
    return mean


class TrainStateWithAux(TrainState):
    """Extended train state that can hold additional info."""

    extra: Any = None
