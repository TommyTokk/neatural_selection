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
        phenotypic_weight: float = 2.0,
        trait_config: object | None = None,
        vision_config: object | None = None,
        flocking_trait_distance_coefficient: float = 1.0,
    ) -> None:
        self.config_path = config_path
        self.compatibility_threshold = compatibility_threshold
        self.phenotypic_weight = phenotypic_weight
        self.trait_config = trait_config
        self.vision_config = vision_config
        self.flocking_trait_distance_coefficient = (
            flocking_trait_distance_coefficient
        )
        self.assigned_creatures: list[object] = []

    def assign_initial_brains(self, creatures: list[object]) -> None:
        self.assigned_creatures = creatures


class NeatControllerWiringTest(unittest.TestCase):
    def test_config_has_no_controller_selection(self) -> None:
        config = build_sim_config()

        self.assertFalse(hasattr(config, "controller"))
        with self.assertRaises(AttributeError):
            config.controller = SimpleNamespace(use_neat_brains=False)

    def test_speciation_config_defaults_and_accepts_custom_threshold(self) -> None:
        config = build_sim_config()

        self.assertEqual(config.speciation.compatibility_threshold, 3.5)
        self.assertEqual(config.speciation.phenotypic_weight, 2.0)
        self.assertEqual(
            config.speciation.flocking_trait_distance_coefficient,
            1.0,
        )
        self.assertEqual(
            SpeciationConfig(compatibility_threshold=4.25).compatibility_threshold,
            4.25,
        )

    def test_world_always_initializes_neat_controller(self) -> None:
        config = build_sim_config()
        config.population.initial_creatures = 0
        config.food.initial_food_items = 0
        config.persistence.enable_telemetry = False

        original_rebuild_boundaries = World._rebuild_boundaries
        original_spawn_creatures = World._spawn_creatures
        original_neat_controller = world_module.NeatBrainController

        World._rebuild_boundaries = lambda self: None
        World._spawn_creatures = lambda self: []
        world_module.NeatBrainController = FakeNeatBrainController

        try:
            world = World(config, simulation_paths=SimpleNamespace())
        finally:
            World._rebuild_boundaries = original_rebuild_boundaries
            World._spawn_creatures = original_spawn_creatures
            world_module.NeatBrainController = original_neat_controller

        self.assertFalse(hasattr(world, "use_neat_brains"))
        self.assertFalse(hasattr(world, "baseline_controller"))
        self.assertEqual(world.neat_controller.assigned_creatures, world.creatures)
        self.assertEqual(world.neat_controller.compatibility_threshold, 3.5)
        self.assertEqual(world.neat_controller.phenotypic_weight, 2.0)
        self.assertIs(world.neat_controller.trait_config, config.trait)
        self.assertIs(world.neat_controller.vision_config, config.vision)
        self.assertEqual(
            world.neat_controller.flocking_trait_distance_coefficient,
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
