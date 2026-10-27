from copy import deepcopy

import mujoco

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.terrains.terrain_importer import TerrainImporter, TerrainImporterCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=(0.0, 0.1),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
      proportion=0.1,
      slope_range=(0.0, 1.0),
      platform_width=2.0,
      border_width=0.25,
      inverted=True,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.1,
      noise_range=(0.02, 0.10),
      noise_step=0.02,
      border_width=0.25,
    ),
    "wave_terrain": terrain_gen.HfWaveTerrainCfg(
      proportion=0.1,
      amplitude_range=(0.0, 0.2),
      num_waves=4,
      border_width=0.25,
    ),
  },
  add_lights=True,
)


# PGTT (Ntagkas et al. 2025) trains on stairs with step height 1-13 cm and step
# width 0.30-0.45 m. Rows are difficulty levels, so step height grows from 1 cm
# (row 0) to 13 cm (row 9). The stairs generator takes a single step width, so each
# stairs type is added twice with two widths.
_PGTT_STAIRS_HEIGHT = (0.01, 0.13)

PGTT_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
    "stairs_up_30": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=_PGTT_STAIRS_HEIGHT,
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_up_40": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=_PGTT_STAIRS_HEIGHT,
      step_width=0.40,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_down_30": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=_PGTT_STAIRS_HEIGHT,
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_down_40": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.2,
      step_height_range=_PGTT_STAIRS_HEIGHT,
      step_width=0.40,
      platform_width=3.0,
      border_width=1.0,
    ),
  },
  add_lights=True,
)

# One policy is trained on all deployment surfaces. Keep the original stairs-only
# recipe available for controlled evaluation and experiments.
PGTT_MIXED_TERRAINS_CFG = deepcopy(PGTT_STAIRS_TERRAINS_CFG)
for name, terrain in PGTT_MIXED_TERRAINS_CFG.sub_terrains.items():
  terrain.proportion = 0.20 if name == "flat" else 0.125
PGTT_MIXED_TERRAINS_CFG.sub_terrains.update({
  "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
    proportion=0.10, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.25,
  ),
  "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
    proportion=0.075, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25,
  ),
  "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
    proportion=0.075, slope_range=(0.0, 0.25), platform_width=2.0,
    border_width=0.25, inverted=True,
  ),
  "waves": terrain_gen.HfWaveTerrainCfg(
    proportion=0.05, amplitude_range=(0.0, 0.05), num_waves=4, border_width=0.25,
  ),
})


if __name__ == "__main__":
  import mujoco.viewer
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"

  terrain_cfg = TerrainImporterCfg(
    terrain_type="generator",
    terrain_generator=ROUGH_TERRAINS_CFG,
  )
  terrain = TerrainImporter(terrain_cfg, device=device)
  mujoco.viewer.launch(terrain.spec.compile())
