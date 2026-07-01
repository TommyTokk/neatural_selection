from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import unittest

for optional_module in ("arcade", "neat"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = ModuleType("pymunk")

pymunk = sys.modules["pymunk"]
if not hasattr(pymunk, "Space"):
    pymunk.Space = lambda: SimpleNamespace(
        gravity=(0.0, 0.0),
        damping=0.0,
        iterations=0,
        add=lambda *args: None,
        remove=lambda *args: None,
    )
if not hasattr(pymunk, "Shape"):
    pymunk.Shape = object

from configs.sim_config import SpeciationConfig, build_sim_config
import src.world as world_module
from src.world import World


class FakeNeatBrainController:
    def __init__(
        self,
        config_path: str,
        compatibility_threshold: float = 3.0,
    ) -> None:
        self.config_path = config_path
        self.compatibility_threshold = compatibility_threshold
        self.assigned_creature_ids: list[int] = []

    def assign_initial_brains(self, creature_ids: list[int]) -> None:
        self.assigned_creature_ids = creature_ids


class WorldControllerConfigTest(unittest.TestCase):
    def test_default_config_uses_neat_brains(self) -> None:
        config = build_sim_config()

        self.assertTrue(config.controller.use_neat_brains)

    def test_speciation_config_defaults_and_accepts_custom_threshold(self) -> None:
        config = build_sim_config()

        self.assertEqual(config.speciation.compatibility_threshold, 3.0)
        self.assertEqual(
            SpeciationConfig(compatibility_threshold=4.25).compatibility_threshold,
            4.25,
        )

    def test_world_honors_baseline_controller_config(self) -> None:
        config = build_sim_config()
        config.controller.use_neat_brains = False
        config.population.initial_creatures = 0
        config.food.initial_food_items = 0

        original_rebuild_boundaries = World._rebuild_boundaries
        original_spawn_creatures = World._spawn_creatures
        original_neat_controller = world_module.NeatBrainController

        World._rebuild_boundaries = lambda self: None
        World._spawn_creatures = lambda self: []
        world_module.NeatBrainController = FakeNeatBrainController

        try:
            world = World(config)
        finally:
            World._rebuild_boundaries = original_rebuild_boundaries
            World._spawn_creatures = original_spawn_creatures
            world_module.NeatBrainController = original_neat_controller

        self.assertFalse(world.use_neat_brains)
        self.assertEqual(world.neat_controller.compatibility_threshold, 3.0)


if __name__ == "__main__":
    unittest.main()
