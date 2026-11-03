# What this file is for, and how to change it
# --------------------------------------------
# The terrain RECIPES: which kinds of ground tiles are in the training world and how often.
# - PGTT_MIXED_TERRAINS_CFG (built at the bottom from PGTT_STAIRS_TERRAINS_CFG) is what rung 7
#   and the teacher train on: flat, 4 stair types, random_rough, 2 slopes, waves.
# - The grid: size=(8.0, 8.0) m per tile, num_rows=10 difficulty levels (row 0 easiest),
#   num_cols=20 tiles per level, border_width around the whole grid.
# - How often a kind appears: its `proportion`. They don't need to add up to 1 (the generator
#   rescales them).
# - How hard it gets: each kind's range, e.g. step_height_range=(0.01, 0.13) m for the stairs
#   (_PGTT_STAIRS_HEIGHT). Each row gets a difficulty from 0 to 1: row r gets
#   (r + a random 0-1) / num_rows (terrain_generator.py), and a tile uses
#   low + difficulty x (high - low). So the top row's stairs are 12-13 cm, row 0's 1-2 cm.
#   random_rough ignores the difficulty unless scale_with_difficulty=True (rung 9's recipe).
# - Rung 9 trains on PGTT_HARD_TERRAINS_CFG (bottom of this file): wider ranges + boulders.
# - More levels: num_rows (10). The robots start on rows 0-max_init_terrain_level (3, set in
#   velocity_env_cfg.py). More rows or columns means more tiles, more boxes, slower ray casts.
# - A new kind of tile: write its class in primitive_terrains.py (boxes) or
#   heightfield_terrains.py (a height grid), export it in mjlab/terrains/__init__.py, then add
#   "<name>": terrain_gen.<YourCfg>(proportion=...) to the recipe here. For rung 7 and the
#   teacher only: inside `PGTT_MIXED_TERRAINS_CFG.sub_terrains.update({` at the bottom. Added to
#   PGTT_STAIRS_TERRAINS_CFG's sub_terrains instead, it is in both (mixed is copied from it).
# - Watch it: the viewer of any rung 7 run (COMMANDS.md 4e), or play.py with --terrain mixed.
# Every terrain box costs ray-cast time (each height-scan ray is tested against every box):
# prefer a few large boxes over many small ones. Changing the recipe changes the experiment:
# give the run its own name and note it in doc/experiments.md.

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


# Rung 9: the same kinds of ground, each spanning easy to hard along the rows, plus boulders.
# Row 0 stays gentle (2 cm steps, near-flat rough ground, 3 cm boulders) and the last row
# reaches the top of each range, so every robot meets small and big obstacles over training.
# The top step is what a trotting Go2 can climb, not more (doc/experiments.md, rung 9).
PGTT_HARD_MAX_STEP = 0.18  # tallest stair step on the last row [m]
PGTT_HARD_TERRAINS_CFG = deepcopy(PGTT_MIXED_TERRAINS_CFG)
for name, terrain in PGTT_HARD_TERRAINS_CFG.sub_terrains.items():
  if name.startswith("stairs"):
    terrain.step_height_range = (0.02, PGTT_HARD_MAX_STEP)
PGTT_HARD_TERRAINS_CFG.sub_terrains["random_rough"] = terrain_gen.HfRandomUniformTerrainCfg(
  proportion=0.10, noise_range=(0.0, 0.10), noise_step=0.01, border_width=0.25,
  downsampled_scale=0.2,  # 20 cm "stones": larger than a foot, so it can stand on one
  scale_with_difficulty=True,  # near-flat on row 0, bumps up to 10 cm on the last row
)
PGTT_HARD_TERRAINS_CFG.sub_terrains["boulders"] = terrain_gen.BoxBouldersTerrainCfg(
  proportion=0.10, height_range=(0.03, 0.15), size_range=(0.3, 0.8), num_boulders=30,
)

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
