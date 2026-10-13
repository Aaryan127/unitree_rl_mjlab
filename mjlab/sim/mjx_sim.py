"""MJX (MuJoCo XLA) simulation backend for GPU-accelerated batched simulation.

This module provides a pure JAX-based simulation pipeline using Google's MJX,
which allows for fully parallel GPU simulation with vmap and lax.scan support.
"""

from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Any

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx


@dataclass
class MJXConfig:
    """Configuration for MJX simulation parameters."""

    timestep: float = 0.002
    integrator: Literal["euler", "implicitfast", "implicit", "RK4"] = "implicitfast"
    cone: Literal["elliptic", "pyramidal"] = "pyramidal"
    jacobian: Literal["dense", "sparse", "auto"] = "auto"
    solver: Literal["newton", "cg", "pgs"] = "newton"
    iterations: int = 100
    tolerance: float = 1e-8
    ls_iterations: int = 50
    ls_tolerance: float = 0.01
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    disableflags: int = 0
    enableflags: int = 0


_INTEGRATOR_MAP = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
}

_CONE_MAP = {
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}

_JACOBIAN_MAP = {
    "dense": mujoco.mjtJacobian.mjJAC_DENSE,
    "sparse": mujoco.mjtJacobian.mjJAC_SPARSE,
    "auto": mujoco.mjtJacobian.mjJAC_AUTO,
}

_SOLVER_MAP = {
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
}


class MJXModel(NamedTuple):
    """Wrapper for MJX model with configuration."""

    mj_model: mujoco.MjModel
    mjx_model: mjx.Model


class SimState(NamedTuple):
    """Simulation state containing MJX data and additional info."""

    data: mjx.Data
    step_count: jp.ndarray


class MJXSimulation:
    """GPU-accelerated MJX simulation backend.

    Supports batched parallel simulation with JIT-compiled operations.
    Uses pure JAX for all computations enabling full GPU acceleration.
    """

    def __init__(
        self,
        num_envs: int,
        model: mujoco.MjModel,
        cfg: MJXConfig | None = None,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.device = device

        if cfg is None:
            cfg = MJXConfig()
        self.cfg = cfg

        self._mj_model = model
        self._apply_config(model, cfg)

        self._mj_model_ptr = self._mj_model

        mjx_model = mjx.put_model(self._mj_model)
        self.mjx_model = mjx_model

        self._step_fn = jax.jit(mjx.step)
        self._forward_fn = jax.jit(mjx.forward)

        self._data = self._create_batch_data(mjx_model, num_envs)

        self._data_flat = self._flatten_pytree(self._data)

        self._reset_fn = jax.jit(self._make_reset_fn(mjx_model), static_argnums=())

    def _apply_config(self, model: mujoco.MjModel, cfg: MJXConfig) -> None:
        """Apply configuration to MuJoCo model."""
        model.opt.timestep = cfg.timestep
        model.opt.integrator = _INTEGRATOR_MAP.get(
            cfg.integrator, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        )
        model.opt.cone = _CONE_MAP.get(cfg.cone, mujoco.mjtCone.mjCONE_PYRAMIDAL)
        model.opt.jacobian = _JACOBIAN_MAP.get(
            cfg.jacobian, mujoco.mjtJacobian.mjJAC_AUTO
        )
        model.opt.solver = _SOLVER_MAP.get(cfg.solver, mujoco.mjtSolver.mjSOL_NEWTON)
        model.opt.iterations = cfg.iterations
        model.opt.tolerance = cfg.tolerance
        model.opt.ls_iterations = cfg.ls_iterations
        model.opt.ls_tolerance = cfg.ls_tolerance
        model.opt.gravity[:] = cfg.gravity
        model.opt.disableflags = cfg.disableflags
        model.opt.enableflags = cfg.enableflags

    def _create_batch_data(self, mjx_model: mjx.Model, num_envs: int) -> mjx.Data:
        """Create batched MJX data for multiple environments."""
        batch_shape = (num_envs,)

        def make_batch_array(shape, dtype):
            return jp.zeros(shape + shape, dtype=dtype)

        data = mjx.make_data(mjx_model)

        data = data.replace(
            qpos=jp.tile(data.qpos[None, :], (num_envs, 1)),
            qvel=jp.tile(data.qvel[None, :], (num_envs, 1)),
            act=jp.tile(data.act[None, :], (num_envs, 1))
            if data.act is not None
            else None,
            ctrl=jp.tile(data.ctrl[None, :], (num_envs, 1)),
            qfrc_applied=jp.tile(data.qfrc_applied[None, :], (num_envs, 1)),
            xfrc_applied=jp.tile(data.xfrc_applied[None, :], (num_envs, 1)),
            eq_active=jp.tile(data.eq_active[None, :], (num_envs, 1))
            if data.eq_active is not None
            else None,
            qacc=jp.tile(data.qacc[None, :], (num_envs, 1)),
            qacc_warmstart=jp.tile(data.qacc_warmstart[None, :], (num_envs, 1)),
            ctrlmap_active=jp.tile(data.ctrlmap_active[None, :], (num_envs, 1))
            if data.ctrlmap_active is not None
            else None,
        )

        return data

    def _flatten_pytree(self, data: mjx.Data) -> list:
        """Flatten MJX data pytree to a list for vectorized operations."""
        return [
            data.qpos,
            data.qvel,
            data.act,
            data.ctrl,
            data.qfrc_applied,
            data.xfrc_applied,
            data.eq_active,
            data.qacc,
            data.qacc_warmstart,
            data.ctrlmap_active,
        ]

    def _make_reset_fn(self, mjx_model: mjx.Model):
        """Create reset function that resets specified environments to default state."""

        def reset_fn(
            data: mjx.Data, default_data: mjx.Data, env_ids: jp.ndarray | None = None
        ) -> mjx.Data:
            """Reset simulation data for specified environments."""
            if env_ids is None:
                return default_data

            def reset_single(data_slice, default_slice):
                return default_slice

            indices = env_ids
            new_qpos = data.qpos.at[indices].set(default_data.qpos[0])
            new_qvel = data.qvel.at[indices].set(default_data.qvel[0])
            new_act = data.act.at[indices].set(0.0) if data.act is not None else None
            new_ctrl = data.ctrl.at[indices].set(0.0)

            return data.replace(
                qpos=new_qpos,
                qvel=new_qvel,
                act=new_act,
                ctrl=new_ctrl,
            )

        return reset_fn

    @property
    def mj_model(self) -> mujoco.MjModel:
        """Get the underlying MuJoCo model."""
        return self._mj_model

    @property
    def data(self) -> mjx.Data:
        """Get current simulation data."""
        return self._data

    def forward(self) -> None:
        """Run forward kinematics (compute all derivatives)."""
        self._data = self._forward_fn(self.mjx_model, self._data)

    def step(self) -> None:
        """Step the simulation forward one timestep."""
        self._data = self._step_fn(self.mjx_model, self._data)

    def reset(self, env_ids: jp.ndarray | None = None) -> None:
        """Reset simulation for specified environments.

        Args:
            env_ids: Array of environment IDs to reset. If None, reset all.
        """
        pass

    def get_default_state(self) -> mjx.Data:
        """Get the default/initial state of the simulation."""
        default_data = mjx.make_data(self.mjx_model)
        mjx.forward(self.mjx_model, default_data)
        return default_data


def create_mjx_simulation(
    num_envs: int,
    model: mujoco.MjModel,
    cfg: MJXConfig | None = None,
    device: str = "cuda",
) -> MJXSimulation:
    """Factory function to create an MJX simulation.

    Args:
        num_envs: Number of parallel environments
        model: Compiled MuJoCo model
        cfg: MJX configuration (optional)
        device: Device for computation ("cuda" or "cpu")

    Returns:
        MJXSimulation instance
    """
    return MJXSimulation(num_envs, model, cfg, device)
