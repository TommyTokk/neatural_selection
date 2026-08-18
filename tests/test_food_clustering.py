from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from configs.sim_config import FoodClusterConfig, FoodConfig, build_sim_config
from src.biome import Biome, BiomeMap
from src.food_clustering import FoodClusterManager, FoodPatch
from src.food_spawner import FoodSpawner


BOUNDS = (0.0, 0.0, 1000.0, 1000.0)


def biome_map(ids: np.ndarray) -> BiomeMap:
    counts = np.bincount(ids.reshape(-1), minlength=len(Biome))
    return BiomeMap(
        biome_ids=ids.astype(np.uint8),
        render_rgba=np.zeros((*ids.shape, 4), dtype=np.uint8),
        world_bounds=BOUNDS,
        area_shares={
            biome: float(counts[int(biome)] / ids.size) for biome in Biome
        },
        spawn_weights={
            Biome.PRAIRIE: 0.25,
            Biome.BUSHES: 1.25,
            Biome.FOREST: 2.75,
        },
        uniform_spawn_chance=0.0,
        max_spawn_attempts=32,
    )


class FakeWorld:
    def __init__(self, map_: BiomeMap, food_config: FoodConfig) -> None:
        self.foods = []
        self.food_spawner = FoodSpawner(food_config, Random(7), map_)

    def _add_foods(self, foods) -> None:
        self.foods.extend(foods)

    def _remove_food(self, food) -> None:
        if food in self.foods:
            self.foods.remove(food)


class FoodClusterManagerTest(unittest.TestCase):
    def test_mode_targets_are_exact_for_experiment_ratios(self) -> None:
        self.assertEqual(FoodClusterManager.mode_targets(100, 0.30), (70, 30))
        self.assertEqual(FoodClusterManager.mode_targets(101, 0.50), (50, 51))
        self.assertEqual(FoodClusterManager.mode_targets(100, 0.60), (40, 60))

    def test_default_inventory_is_three_bush_patches_per_forest_patch(self) -> None:
        ids = np.full((20, 20), int(Biome.BUSHES), dtype=np.uint8)
        ids[:, 12:] = int(Biome.FOREST)
        map_ = biome_map(ids)

        for seed in (1, 7, 19):
            for cluster_share in (0.30, 0.50, 0.60):
                manager = FoodClusterManager(
                    FoodClusterConfig(),
                    FoodConfig(),
                    map_,
                    seed,
                )
                _independent, target = manager.mode_targets(363, cluster_share)

                manager._ensure_patch_capacity(target)

                counts = Counter(
                    patch.biome_type for patch in manager.patches.values()
                )
                self.assertEqual(counts[Biome.PRAIRIE], 0)
                self.assertEqual(counts[Biome.BUSHES], 3 * counts[Biome.FOREST])
                self.assertGreater(counts[Biome.FOREST], 0)
                self.assertEqual(
                    sum(
                        patch.target_food_count
                        for patch in manager.patches.values()
                    ),
                    target,
                )
                self.assertTrue(
                    all(
                        3 <= patch.target_food_count <= 6
                        for patch in manager.patches.values()
                        if patch.biome_type is Biome.BUSHES
                    )
                )
                self.assertTrue(
                    all(
                        12 <= patch.target_food_count <= 25
                        for patch in manager.patches.values()
                        if patch.biome_type is Biome.FOREST
                    )
                )

    def test_default_world_bootstrap_has_both_cluster_and_single_modes(self) -> None:
        from src.persistence import SimulationPaths
        from src.world import World

        config = build_sim_config()
        config.population.initial_creatures = 0
        config.persistence.enable_telemetry = False
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        with TemporaryDirectory() as directory:
            world = World(
                config,
                simulation_paths=SimulationPaths(Path(directory)),
            )
            try:
                patch_counts = Counter(
                    patch.biome_type
                    for patch in world.food_cluster_manager.patches.values()
                )
                independent_biomes = {
                    world.biome_map.biome_at(*food.position)
                    for food in world.foods
                    if food.cluster_id is None
                }

                self.assertEqual(patch_counts[Biome.BUSHES], 9)
                self.assertEqual(patch_counts[Biome.FOREST], 3)
                self.assertEqual(patch_counts[Biome.PRAIRIE], 0)
                self.assertIn(Biome.BUSHES, independent_biomes)
                self.assertIn(Biome.FOREST, independent_biomes)
                self.assertEqual(len(world.foods), config.food.initial_food_items)
            finally:
                world.close()

    def test_small_cluster_budgets_never_create_singleton_padding(self) -> None:
        ids = np.full((10, 10), int(Biome.BUSHES), dtype=np.uint8)
        ids[:, 6:] = int(Biome.FOREST)
        map_ = biome_map(ids)

        for target in (1, 2):
            manager = FoodClusterManager(
                FoodClusterConfig(), FoodConfig(), map_, target
            )
            manager._ensure_patch_capacity(target)
            self.assertEqual(manager.patches, {})

        for target in (3, 7, 14):
            manager = FoodClusterManager(
                FoodClusterConfig(), FoodConfig(), map_, target
            )
            manager._ensure_patch_capacity(target)
            self.assertEqual(
                sum(p.target_food_count for p in manager.patches.values()),
                target,
            )
            self.assertTrue(
                all(p.biome_type is Biome.BUSHES for p in manager.patches.values())
            )

        manager = FoodClusterManager(FoodClusterConfig(), FoodConfig(), map_, 15)
        manager._ensure_patch_capacity(15)
        self.assertEqual(
            {patch.biome_type for patch in manager.patches.values()},
            {Biome.BUSHES, Biome.FOREST},
        )

    def test_depleted_surplus_forest_patch_is_replaced_by_bush_inventory(self) -> None:
        ids = np.full((20, 20), int(Biome.BUSHES), dtype=np.uint8)
        ids[:, 12:] = int(Biome.FOREST)
        map_ = biome_map(ids)
        manager = FoodClusterManager(FoodClusterConfig(), FoodConfig(), map_, 23)
        for patch_id in range(1, 4):
            manager.patches[patch_id] = FoodPatch(
                id=patch_id,
                biome_type=Biome.FOREST,
                center_pos=(800.0, 500.0),
                configured_radius=50.0,
                effective_radius=50.0,
                target_food_count=20,
                is_depleted=patch_id == 3,
                has_spawned=True,
            )
        manager._next_patch_id = 4

        manager._ensure_patch_capacity(60)

        counts = Counter(p.biome_type for p in manager.patches.values())
        self.assertEqual(counts[Biome.BUSHES], 6)
        self.assertEqual(counts[Biome.FOREST], 2)
        self.assertIn(1, manager.patches)
        self.assertIn(2, manager.patches)
        self.assertNotIn(3, manager.patches)
        self.assertEqual(
            sum(p.target_food_count for p in manager.patches.values()),
            60,
        )

    def test_missing_forest_uses_only_complete_bush_clusters(self) -> None:
        map_ = biome_map(np.full((10, 10), int(Biome.BUSHES)))
        manager = FoodClusterManager(FoodClusterConfig(), FoodConfig(), map_, 29)

        manager._ensure_patch_capacity(30)

        self.assertEqual(
            sum(p.target_food_count for p in manager.patches.values()),
            30,
        )
        self.assertTrue(
            all(p.biome_type is Biome.BUSHES for p in manager.patches.values())
        )

    def test_forest_patch_has_rich_shared_food_inside_its_biome(self) -> None:
        map_ = biome_map(np.full((10, 10), int(Biome.FOREST)))
        food_config = FoodConfig(min_food_radius=2.0, max_food_radius=2.0)
        manager = FoodClusterManager(FoodClusterConfig(), food_config, map_, 9)
        world = FakeWorld(map_, food_config)
        patch = FoodPatch(
            id=1,
            biome_type=Biome.FOREST,
            center_pos=(500.0, 500.0),
            configured_radius=80.0,
            effective_radius=80.0,
            target_food_count=12,
        )

        foods = manager.spawn_patch_food(patch, world)

        self.assertEqual(len(foods), 12)
        self.assertTrue(all(food.cluster_id == patch.id for food in foods))
        self.assertTrue(all(food.bite_capacity >= 2 for food in foods))
        self.assertTrue(
            all(map_.biome_at(*food.position) is Biome.FOREST for food in foods)
        )
        standard_energy = np.pi * food_config.max_food_radius**2 * food_config.energy_density
        self.assertTrue(all(food.max_energy >= 2.0 * standard_energy for food in foods))

    def test_narrow_corridor_reduces_effective_radius(self) -> None:
        ids = np.full((20, 50), int(Biome.PRAIRIE), dtype=np.uint8)
        ids[:, 25] = int(Biome.FOREST)
        map_ = biome_map(ids)
        config = replace(
            FoodClusterConfig(),
            max_sampling_attempts=1,
            center_sampling_attempts=1,
        )
        food_config = FoodConfig(min_food_radius=2.0, max_food_radius=2.0)
        manager = FoodClusterManager(config, food_config, map_, 11)
        radii = np.full(12, 2.0)

        sampled = manager._sample_positions(
            (510.0, 500.0),
            Biome.FOREST,
            80.0,
            radii,
        )

        self.assertIsNotNone(sampled)
        points, effective_radius = sampled
        self.assertLess(effective_radius, 80.0)
        self.assertTrue(
            all(map_.biome_at(float(x), float(y)) is Biome.FOREST for x, y in points)
        )
        self.assertGreater(manager.diagnostics()["radius_reductions"], 0)

    def test_sparse_patch_enters_cooldown_and_relocates(self) -> None:
        map_ = biome_map(np.full((10, 10), int(Biome.FOREST)))
        config = replace(FoodClusterConfig(), cooldown_ticks=(1, 1))
        food_config = FoodConfig(min_food_radius=2.0, max_food_radius=2.0)
        manager = FoodClusterManager(config, food_config, map_, 13)
        world = FakeWorld(map_, food_config)
        patch = FoodPatch(
            id=1,
            biome_type=Biome.FOREST,
            center_pos=(500.0, 500.0),
            configured_radius=50.0,
            effective_radius=50.0,
            target_food_count=12,
            has_spawned=True,
            is_depleted=False,
        )
        manager.patches[patch.id] = patch
        foods = manager.spawn_patch_food(patch, world)
        for food in foods[:-1]:
            world.foods.remove(food)
        patch.active_food_ids = {foods[-1].id}
        previous = patch.center_pos

        manager._deplete_sparse_patches(world)
        manager._advance_cooldowns(1)

        self.assertEqual(world.foods, [])
        self.assertTrue(patch.is_depleted)
        self.assertEqual(patch.cooldown_timer, 0)
        self.assertNotEqual(patch.center_pos, previous)
        self.assertIs(map_.biome_at(*patch.center_pos), Biome.FOREST)

    def test_manager_state_restores_patch_and_rng_continuation(self) -> None:
        map_ = biome_map(np.full((4, 4), int(Biome.BUSHES)))
        food_config = FoodConfig(min_food_radius=2.0, max_food_radius=2.0)
        first = FoodClusterManager(FoodClusterConfig(), food_config, map_, 17)
        first._cluster_spawn_credit = 7.5
        first._ensure_patch_capacity(6)
        state = first.state_dict()

        restored = FoodClusterManager(FoodClusterConfig(), food_config, map_, 999)
        restored.restore_state(state)

        self.assertEqual(restored._cluster_spawn_credit, 7.5)
        self.assertEqual(restored.state_dict()["patches"], state["patches"])
        self.assertEqual(first.rng.random(), restored.rng.random())

    def test_restore_tolerates_duplicate_patch_entries_but_serializes_once(self) -> None:
        map_ = biome_map(np.full((4, 4), int(Biome.BUSHES)))
        manager = FoodClusterManager(FoodClusterConfig(), FoodConfig(), map_, 31)
        manager._ensure_patch_capacity(6)
        state = manager.state_dict()
        state["patches"] = state["patches"] * 2

        restored = FoodClusterManager(FoodClusterConfig(), FoodConfig(), map_, 32)
        restored.restore_state(state)

        self.assertEqual(
            len(restored.state_dict()["patches"]),
            len(restored.patches),
        )


if __name__ == "__main__":
    unittest.main()
