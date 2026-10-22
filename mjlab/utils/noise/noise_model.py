from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import override

if TYPE_CHECKING:
  from mjlab.utils.noise import noise_cfg


class NoiseModel:
  """Base class for noise models."""

  def __init__(
    self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str
  ):
    self._noise_model_cfg = noise_model_cfg
    self._num_envs = num_envs
    self._device = device

    # Validate configuration.
    if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
      raise ValueError("NoiseModelCfg must have a valid noise_cfg")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset noise model state. Override in subclasses if needed."""

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to input data."""
    assert self._noise_model_cfg.noise_cfg is not None
    return self._noise_model_cfg.noise_cfg.apply(data)


class NoiseModelWithAdditiveBias(NoiseModel):
  """Noise model with additional additive bias that is constant for the duration
  of the entire episode."""

  def __init__(
    self,
    noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
    num_envs: int,
    device: str,
  ):
    super().__init__(noise_model_cfg, num_envs, device)

    # Validate bias configuration.
    if (
      not hasattr(noise_model_cfg, "bias_noise_cfg")
      or noise_model_cfg.bias_noise_cfg is None
    ):
      raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

    self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
    self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    # Initialize bias tensor.
    self._bias = torch.zeros((num_envs, 1), device=self._device)
    self._num_components: int | None = None
    self._bias_initialized = False

  @override
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset bias values for specified environments."""
    indices = slice(None) if env_ids is None else env_ids
    # Sample new bias values.
    self._bias[indices] = self._bias_noise_cfg.apply(self._bias[indices])

  def _initialize_bias_shape(self, data_shape: torch.Size) -> None:
    """Initialize bias tensor shape based on data and configuration."""
    if self._sample_bias_per_component and not self._bias_initialized:
      *_, self._num_components = data_shape
      # Expand bias to match number of components.
      self._bias = self._bias.repeat(1, self._num_components)
      self._bias_initialized = True
      # Resample bias with new shape.
      self.reset()

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise and additive bias to input data."""
    self._initialize_bias_shape(data.shape)
    noisy_data = super().__call__(data)
    return noisy_data + self._bias


class HeightScanNoiseModel(NoiseModel):
  """Height-map corruption for the student policy (Miki et al. 2022 style).

  Models what a real LiDAR elevation map does wrong:
    - per-step Gaussian jitter on every cell (noise_cfg)
    - a per-episode vertical offset of the whole map that drifts every step
    - outlier cells (random spikes)
    - missing cells, reported as 0 (= "flat ground at nominal height")
    - whole episodes without any map ("blind"), also reported as 0
    - "noisy" episodes with much stronger jitter and more outliers (Miki's NOISY type)
    - a per-episode horizontal shift of the map by whole cells (map misaligned with the
      robot; the cpu branch's Map_noise_cfg). Needs `grid_shape`.

  0 is used for missing data because the real robot sends all zeros when no height
  map is available, so the student learns that case explicitly.
  """

  def __init__(
    self,
    noise_model_cfg: noise_cfg.HeightScanNoiseModelCfg,
    num_envs: int,
    device: str,
  ):
    super().__init__(noise_model_cfg, num_envs, device)
    self._cfg = noise_model_cfg
    self._offset = torch.zeros((num_envs, 1), device=device)
    self._blind = torch.zeros((num_envs, 1), dtype=torch.bool, device=device)
    self._noisy = torch.zeros((num_envs, 1), dtype=torch.bool, device=device)
    self._shift = torch.zeros((num_envs, 2), dtype=torch.long, device=device)  # (rows, cols)
    self.reset()

  @override
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    cfg = self._cfg
    indices = slice(None) if env_ids is None else env_ids
    n = self._offset[indices].shape[0]
    self._offset[indices] = torch.randn((n, 1), device=self._device) * cfg.offset_std
    self._blind[indices] = torch.rand((n, 1), device=self._device) < cfg.blind_prob
    self._noisy[indices] = torch.rand((n, 1), device=self._device) < cfg.noisy_episode_prob
    shift = torch.randint(
      -cfg.max_shift_cells, cfg.max_shift_cells + 1, (n, 2), device=self._device
    )
    shifted = torch.rand((n, 1), device=self._device) < cfg.shift_prob
    self._shift[indices] = shift * shifted

  def _shift_map(self, data: torch.Tensor) -> torch.Tensor:
    rows, cols = self._cfg.grid_shape  # type: ignore[misc]
    b = data.shape[0]
    grid = data.view(b, rows, cols)
    r = (torch.arange(rows, device=data.device)[None] - self._shift[:, :1]).clamp(0, rows - 1)
    c = (torch.arange(cols, device=data.device)[None] - self._shift[:, 1:]).clamp(0, cols - 1)
    batch = torch.arange(b, device=data.device)[:, None, None]
    return grid[batch, r[:, :, None], c[:, None, :]].reshape(b, -1)

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    cfg = self._cfg
    if cfg.grid_shape is not None and cfg.max_shift_cells > 0:
      data = self._shift_map(data)

    self._offset += torch.randn_like(self._offset) * cfg.drift_std
    self._offset.clamp_(-cfg.max_offset, cfg.max_offset)

    jitter = super().__call__(data) - data  # noise_cfg applied to the map
    jitter = torch.where(self._noisy, jitter * cfg.noisy_scale, jitter)
    noisy = data + jitter + self._offset

    outlier_prob = torch.where(
      self._noisy,
      torch.full_like(self._offset, cfg.outlier_prob * cfg.noisy_scale),
      torch.full_like(self._offset, cfg.outlier_prob),
    )
    outliers = torch.rand_like(data) < outlier_prob
    noisy = torch.where(
      outliers,
      (torch.rand_like(data) * 2.0 - 1.0) * cfg.outlier_range,
      noisy,
    )
    missing = (torch.rand_like(data) < cfg.missing_prob) | self._blind
    return torch.where(missing, torch.zeros_like(noisy), noisy)
