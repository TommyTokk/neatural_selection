from __future__ import annotations

from dataclasses import replace
from random import Random
from threading import RLock
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

from configs.sim_config import LiveFoodConfig, SpeciationConfig, build_sim_config
from src.biome import Biome, BiomeGenerationHandler
from src.food_spawner import FoodSpawner
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
        random_seed: int | None = None,
    ) -> None:
        self.config_path = config_path
        self.compatibility_threshold = compatibility_threshold
        self.phenotypic_weight = phenotypic_weight
        self.trait_config = trait_config
        self.vision_config = vision_config
        self.flocking_trait_distance_coefficient = (
            flocking_trait_distance_coefficient
        )
        self.random_seed = random_seed
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
        self.assertEqual(world.neat_controller.random_seed, config.random_seed)

    def test_world_accepts_a_separate_brain_initialization_seed(self) -> None:
        config = build_sim_config()
        config.population.initial_creatures = 0
        config.food.initial_food_items = 0
        config.persistence.enable_telemetry = False
        brain_seed = 987_654_321

        original_rebuild_boundaries = World._rebuild_boundaries
        original_spawn_creatures = World._spawn_creatures
        original_neat_controller = world_module.NeatBrainController

        World._rebuild_boundaries = lambda self: None
        World._spawn_creatures = lambda self: []
        world_module.NeatBrainController = FakeNeatBrainController

        try:
            world = World(
                config,
                simulation_paths=SimpleNamespace(),
                brain_initialization_seed=brain_seed,
            )
        finally:
            World._rebuild_boundaries = original_rebuild_boundaries
            World._spawn_creatures = original_spawn_creatures
            world_module.NeatBrainController = original_neat_controller

        self.assertEqual(world.brain_initialization_seed, brain_seed)
        self.assertEqual(world.neat_controller.random_seed, brain_seed)


class LiveFoodConfigTest(unittest.TestCase):
    def make_runtime_world(self) -> tuple[World, object]:
        config = build_sim_config()
        world = object.__new__(World)
        world._checkpoint_state_lock = RLock()
        world.biome_map = BiomeGenerationHandler(config.biome).generate(
            (-100.0, -100.0, 100.0, 100.0)
        )
        world.food_spawner = FoodSpawner(
            replace(config.food),
            Random(7),
            world.biome_map,
        )
        world._live_food_config = LiveFoodConfig.from_configs(
            config.biome,
            config.food,
        )
        return world, config

    def test_runtime_values_do_not_mutate_startup_configuration(self) -> None:
        world, config = self.make_runtime_world()
        old_map = world.biome_map
        old_density_grid = old_map._expected_density_grid
        old_sensor_grid = old_map._sensor_richness_grid
        original_max_food = config.food.max_food_items
        original_burst_items = config.food.low_food_burst_items
        original_forest_weight = config.biome.forest_spawn_weight

        world.food_spawner._spawn_credit = 2.5
        world.food_spawner._low_food_burst_credit = 0.75
        world.food_spawner._pending_low_food_burst_items = 12
        existing_foods = [object() for _ in range(25)]
        world.foods = existing_foods
        world.set_live_food_config_value("forest_spawn_weight", 4.5)
        world.set_live_food_config_value("max_food_items", 111)
        world.set_live_food_config_value("low_food_burst_items", 33)
        world.set_live_food_config_value("max_food_items", 10)

        self.assertEqual(world.live_food_config.forest_spawn_weight, 4.5)
        self.assertEqual(world.food_spawner.config.max_food_items, 10)
        self.assertEqual(world.food_spawner.config.low_food_burst_items, 33)
        self.assertEqual(config.food.max_food_items, original_max_food)
        self.assertEqual(config.food.low_food_burst_items, original_burst_items)
        self.assertEqual(
            config.biome.forest_spawn_weight,
            original_forest_weight,
        )
        self.assertIs(world.biome_map.biome_ids, old_map.biome_ids)
        self.assertIs(world.biome_map.render_rgba, old_map.render_rgba)
        self.assertIsNot(
            world.biome_map._expected_density_grid,
            old_density_grid,
        )
        self.assertIsNot(
            world.biome_map._sensor_richness_grid,
            old_sensor_grid,
        )
        self.assertEqual(
            world.biome_map.spawn_weights[Biome.FOREST],
            4.5,
        )
        self.assertIs(world.food_spawner.biome_map, world.biome_map)
        self.assertEqual(world.food_spawner._spawn_credit, 2.5)
        self.assertEqual(world.food_spawner._low_food_burst_credit, 0.0)
        self.assertEqual(world.food_spawner._pending_low_food_burst_items, 0)
        self.assertIs(world.foods, existing_foods)
        self.assertEqual(len(world.foods), 25)
        self.assertEqual(
            world.food_spawner.update(
                1.0,
                (-100.0, -100.0, 100.0, 100.0),
                current_food_count=len(world.foods),
                active_species_count=1,
                available_biomass=10_000.0,
            ),
            [],
        )

    def test_runtime_ratio_edits_clamp_the_edited_value(self) -> None:
        world, _config = self.make_runtime_world()

        world.set_live_food_config_value("critical_food_ratio", 0.8)
        self.assertEqual(world.live_food_config.critical_food_ratio, 0.5)

        world.set_live_food_config_value("critical_food_ratio", 0.4)
        world.set_live_food_config_value("low_food_pressure_threshold", 0.2)
        self.assertEqual(world.live_food_config.low_food_pressure_threshold, 0.4)


if __name__ == "__main__":
    unittest.main()
