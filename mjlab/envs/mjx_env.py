"""JAX-based RL environment using MJX for GPU-accelerated simulation.

This module provides a functional, pure JAX implementation of the RL environment
that supports batched parallel simulation with vmap and lax.scan.
"""

import jax
import jax.numpy as jp
from jax import random
from typing import NamedTuple, Tuple, Any
from dataclasses import dataclass

import mujoco
from mujoco import mjx


class EnvState(NamedTuple):
    """Environment state containing all simulation data and metadata.

    This is an immutable state object that gets passed through the environment.
    """

    data: mjx.Data
    step_count: jp.ndarray
    episode_length: jp.ndarray
    reward: jp.ndarray
    done: jp.ndarray


class EnvParams(NamedTuple):
    """Environment parameters that don't change during rollout."""

    num_envs: int
    max_episode_length: int
    decimation: int
    physics_dt: float


class EnvConfig(NamedTuple):
    """Static environment configuration."""

    mj_model: mujoco.MjModel
    mjx_model: mjx.Model
    num_envs: int
    max_episode_length: int
    decimation: int
    physics_dt: float
    action_scale: jp.ndarray
    default_qpos: jp.ndarray
    default_qvel: jp.ndarray


@dataclass
class MJXEnvCfg:
    """Configuration for MJX environment."""

    num_envs: int = 1
    max_episode_length: int = 1000
    decimation: int = 10
    physics_dt: float = 0.002
    action_scale: float = 0.5


class MJXEnv:
    """MJX-based RL environment with functional API.

    This environment provides a pure JAX interface for:
    - Batched parallel simulation
    - JIT-compiled reset and step functions
    - Integration with lax.scan for efficient rollouts
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        cfg: MJXEnvCfg,
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.num_envs

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model)

        self._default_data = mjx.make_data(self._mjx_model)
        mjx.forward(self._mjx_model, self._default_data)

        self._default_qpos = self._default_data.qpos
        self._default_qvel = self._default_data.qvel

        self.action_scale = jp.array(cfg.action_scale)

        self.max_episode_length = cfg.max_episode_length
        self.decimation = cfg.decimation
        self.physics_dt = cfg.physics_dt
        self.step_dt = cfg.physics_dt * cfg.decimation

        self.nu = mj_model.nu
        self.nq = mj_model.nq
        self.nv = mj_model.nv
        self.nu = mj_model.nu
        self.nsensordata = mj_model.nsensordata

        self._step_fn = jax.jit(mjx.step)
        self._forward_fn = jax.jit(mjx.forward)

        self._reset_fn = jax.jit(self._reset, static_argnums=(1,))
        self._step_env_fn = jax.jit(self._step_env, static_argnums=(2,))

    def _create_batch_data(self) -> mjx.Data:
        """Create batched simulation data."""
        batch_size = self.num_envs

        def tile_array(arr):
            if arr is None:
                return None
            return jp.tile(arr[None, :], (batch_size, 1))

        return self._default_data.replace(
            qpos=tile_array(self._default_qpos),
            qvel=tile_array(self._default_qvel),
            ctrl=jp.zeros((batch_size, self.nu)),
        )

    def _reset(self, key: random.PRNGKey, batch_size: int) -> EnvState:
        """Reset environment to initial state.

        Args:
            key: JAX random key
            batch_size: Number of environments

        Returns:
            Initial environment state
        """
        data = self._create_batch_data()

        qpos = jp.tile(self._default_qpos[None, :], (batch_size, 1))
        qvel = jp.tile(self._default_qvel[None, :], (batch_size, 1))

        data = data.replace(qpos=qpos, qvel=qvel)

        data = self._forward_fn(self._mjx_model, data)

        obs = self._get_observation(data)

        state = EnvState(
            data=data,
            step_count=jp.zeros(batch_size, dtype=jp.int32),
            episode_length=jp.zeros(batch_size, dtype=jp.int32),
            reward=jp.zeros(batch_size),
            done=jp.zeros(batch_size, dtype=jp.bool_),
        )

        return state

    def _step_env(
        self,
        state: EnvState,
        action: jp.ndarray,
        key: random.PRNGKey,
    ) -> Tuple[EnvState, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
        """Step the environment forward one step (with decimation).

        Args:
            state: Current environment state
            action: Actions to apply (batch_size, nu)
            key: JAX random key

        Returns:
            Tuple of (new_state, observation, reward, done, info)
        """
        key, subkey = random.split(key)

        data = state.data

        total_reward = jp.zeros(self.num_envs)

        for _ in range(self.decimation):
            ctrl = action * self.action_scale
            data = data.replace(ctrl=ctrl)
            data = self._step_fn(self._mjx_model, data)
            data = self._forward_fn(self._mjx_model, data)

        obs = self._get_observation(data)
        reward = self._compute_reward(data)

        state = EnvState(
            data=data,
            step_count=state.step_count + 1,
            episode_length=state.episode_length + 1,
            reward=reward,
            done=jp.zeros(self.num_envs, dtype=jp.bool_),
        )

        return state, obs, reward, state.done, {}

    def _get_observation(self, data: mjx.Data) -> jp.ndarray:
        """Compute observation from simulation data.

        Args:
            data: MJX simulation data

        Returns:
            Observation array (batch_size, obs_dim)
        """
        qpos = data.qpos
        qvel = data.qvel

        obs = jp.concatenate(
            [
                qpos.flatten(),
                qvel.flatten(),
            ]
        )

        return obs

    def _compute_reward(self, data: mjx.Data) -> jp.ndarray:
        """Compute reward for current state.

        Args:
            data: MJX simulation data

        Returns:
            Reward array (batch_size,)
        """
        return jp.zeros(self.num_envs)

    def reset(self, key: random.PRNGKey) -> EnvState:
        """Public reset function."""
        return self._reset_fn(key, self.num_envs)

    def step(
        self, state: EnvState, action: jp.ndarray, key: random.PRNGKey
    ) -> Tuple[EnvState, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
        """Public step function."""
        return self._step_env_fn(state, action, key)

    @property
    def obs_dim(self) -> int:
        """Get observation dimension."""
        return self.nq + self.nv

    @property
    def action_dim(self) -> int:
        """Get action dimension."""
        return self.nu


def create_env(
    model: mujoco.MjModel,
    cfg: MJXEnvCfg,
    device: str = "cuda",
) -> MJXEnv:
    """Factory function to create MJX environment.

    Args:
        model: Compiled MuJoCo model
        cfg: Environment configuration
        device: Device for computation

    Returns:
        MJXEnv instance
    """
    return MJXEnv(model, cfg, device)


def batch_reset(env: MJXEnv, keys: random.PRNGKeyArray) -> EnvState:
    """Reset multiple environments with different keys.

    Args:
        env: MJX environment
        keys: Array of JAX random keys (num_envs,)

    Returns:
        Batched environment state
    """
    return jax.vmap(env.reset)(keys)


def batch_step(
    env: MJXEnv,
    state: EnvState,
    actions: jp.ndarray,
    key: random.PRNGKey,
) -> Tuple[EnvState, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """Step multiple environments in parallel.

    Args:
        env: MJX environment
        state: Current environment state
        actions: Actions (batch_size, action_dim)
        key: JAX random key

    Returns:
        Tuple of (new_state, observations, rewards, dones, infos)
    """
    return jax.vmap(env.step, in_axes=(0, 0, None))(
        jax.tree.map(lambda x: x, state), actions, key
    )
