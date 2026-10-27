"""Deployment-surface selection shared by training, play and evaluation."""

from copy import deepcopy
from typing import Literal

from mjlab.terrains.config import PGTT_MIXED_TERRAINS_CFG, PGTT_STAIRS_TERRAINS_CFG

TerrainProfile = Literal["flat", "stairs", "rough", "mixed"]


def set_terrain_profile(cfg, profile: TerrainProfile) -> None:
  if profile not in {"flat", "stairs", "rough", "mixed"}:
    raise ValueError(f"Unknown terrain profile: {profile}")
  terrain = cfg.scene.terrain
  if terrain is None:
    raise ValueError("This task has no terrain configuration")
  if profile == "flat":
    terrain.terrain_type = "plane"
    terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)
    return
  previous = terrain.terrain_generator
  generator = deepcopy(PGTT_STAIRS_TERRAINS_CFG if profile == "stairs" else PGTT_MIXED_TERRAINS_CFG)
  if profile == "rough":
    generator.sub_terrains = {
      key: value for key, value in generator.sub_terrains.items()
      if key in {"flat", "random_rough", "slope_up", "slope_down", "waves"}
    }
  if previous is not None:
    for key in ("curriculum", "seed", "num_rows", "num_cols", "border_width", "difficulty_range"):
      setattr(generator, key, getattr(previous, key))
  total = sum(t.proportion for t in generator.sub_terrains.values())
  for subterrain in generator.sub_terrains.values():
    subterrain.proportion /= total
  terrain.terrain_type = "generator"
  terrain.terrain_generator = generator
