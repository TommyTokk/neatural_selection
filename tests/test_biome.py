from __future__ import annotations

from dataclasses import replace
from random import Random
import unittest

import numpy as np

from configs.sim_config import BiomeConfig, FoodConfig
from src.biome import Biome, BiomeGenerationHandler, BiomeMap
from src.food_spawner import FoodSpawner


WORLD_BOUNDS = (-1600.0, -1100.0, 1600.0, 1100.0)


class BiomeGenerationHandlerTest(unittest.TestCase):
    def test_same_seed_generates_same_biome_map(self) -> None:
        config = BiomeConfig(seed=7)
        first = BiomeGenerationHandler(config).generate(WORLD_BOUNDS)
        second = BiomeGenerationHandler(config).generate(WORLD_BOUNDS)

        self.assertTrue(np.array_equal(first.biome_ids, second.biome_ids))
        self.assertTrue(np.array_equal(first.render_rgba, second.render_rgba))

    def test_different_seed_changes_biome_map(self) -> None:
        first = BiomeGenerationHandler(BiomeConfig(seed=7)).generate(WORLD_BOUNDS)
        second = BiomeGenerationHandler(BiomeConfig(seed=8)).generate(WORLD_BOUNDS)

        self.assertFalse(np.array_equal(first.biome_ids, second.biome_ids))

    def test_generated_area_shares_follow_configured_targets(self) -> None:
        config = BiomeConfig(
            seed=7,
            forest_target_share=0.25,
            bushes_target_share=0.40,
            prairie_target_share=0.35,
        )
        biome_map = BiomeGenerationHandler(config).generate(WORLD_BOUNDS)

        self.assertAlmostEqual(biome_map.area_shares[Biome.FOREST], 0.25, delta=0.02)
        self.assertAlmostEqual(biome_map.area_shares[Biome.BUSHES], 0.40, delta=0.02)
        self.assertAlmostEqual(biome_map.area_shares[Biome.PRAIRIE], 0.35, delta=0.02)

    def test_biome_lookup_clamps_to_world_bounds(self) -> None:
        biome_ids = np.array(
            [
                [Biome.PRAIRIE, Biome.BUSHES],
                [Biome.FOREST, Biome.BUSHES],
            ],
            dtype=np.uint8,
        )
        biome_map = BiomeMap(
            biome_ids=biome_ids,
            render_rgba=np.zeros((2, 2, 4), dtype=np.uint8),
            world_bounds=(0.0, 0.0, 20.0, 10.0),
            area_shares={biome: 0.0 for biome in Biome},
            spawn_weights={biome: 1.0 for biome in Biome},
            uniform_spawn_chance=0.0,
            max_spawn_attempts=4,
        )

        self.assertEqual(biome_map.biome_at(-100.0, -100.0), Biome.PRAIRIE)
        self.assertEqual(biome_map.biome_at(100.0, 100.0), Biome.BUSHES)
        self.assertEqual(biome_map.biome_at(5.0, 8.0), Biome.FOREST)

    def test_fertility_uses_resource_spawn_weights(self) -> None:
        biome_map = self._three_vertical_biome_map()

        prairie = biome_map.fertility_at(-1000.0, 0.0)
        bushes = biome_map.fertility_at(0.0, 0.0)
        forest = biome_map.fertility_at(1000.0, 0.0)

        self.assertGreaterEqual(prairie, 0.0)
        self.assertLessEqual(forest, 1.0)
        self.assertLess(prairie, bushes)
        self.assertLess(bushes, forest)

    def test_fertility_equal_spawn_weights_use_safe_fallback(self) -> None:
        biome_map = replace(
            self._three_vertical_biome_map(),
            spawn_weights={biome: 1.0 for biome in Biome},
        )

        self.assertEqual(biome_map.fertility_at(-1000.0, 0.0), 1.0)
        self.assertEqual(biome_map.fertility_at(0.0, 0.0), 1.0)
        self.assertEqual(biome_map.fertility_at(1000.0, 0.0), 1.0)

    def _three_vertical_biome_map(self) -> BiomeMap:
        biome_ids = np.array(
            [[Biome.PRAIRIE, Biome.BUSHES, Biome.FOREST]],
            dtype=np.uint8,
        )
        return BiomeMap(
            biome_ids=biome_ids,
            render_rgba=np.zeros((1, 3, 4), dtype=np.uint8),
            world_bounds=WORLD_BOUNDS,
            area_shares={
                Biome.PRAIRIE: 1.0 / 3.0,
                Biome.BUSHES: 1.0 / 3.0,
                Biome.FOREST: 1.0 / 3.0,
            },
            spawn_weights={
                Biome.FOREST: 2.75,
                Biome.BUSHES: 1.25,
                Biome.PRAIRIE: 0.25,
            },
            uniform_spawn_chance=0.0,
            max_spawn_attempts=32,
        )


class FoodSpawnerBiomePlacementTest(unittest.TestCase):
    def test_biome_weighted_spawn_positions_favor_food_rich_regions(self) -> None:
        biome_map = self._three_vertical_biome_map()
        spawner = FoodSpawner(
            FoodConfig(min_food_radius=1.0, max_food_radius=1.0),
            Random(4),
            biome_map,
        )
        counts = {biome: 0 for biome in Biome}

        for _ in range(2500):
            x, y = spawner._spawn_position(WORLD_BOUNDS, radius=1.0)
            counts[biome_map.biome_at(x, y)] += 1

        self.assertGreater(counts[Biome.FOREST], counts[Biome.BUSHES])
        self.assertGreater(counts[Biome.BUSHES], counts[Biome.PRAIRIE])
        self.assertGreater(counts[Biome.PRAIRIE], 0)

    def test_default_biome_weights_create_stronger_density_profile(self) -> None:
        config = BiomeConfig()
        biome_map = BiomeGenerationHandler(config).generate(WORLD_BOUNDS)
        spawner = FoodSpawner(
            FoodConfig(min_food_radius=1.0, max_food_radius=1.0),
            Random(11),
            biome_map,
        )
        counts = {biome: 0 for biome in Biome}
        sample_count = 8000

        for _ in range(sample_count):
            x, y = spawner._spawn_position(WORLD_BOUNDS, radius=1.0)
            counts[biome_map.biome_at(x, y)] += 1

        density_index = {
            biome: (counts[biome] / sample_count) / biome_map.area_shares[biome]
            for biome in Biome
        }

        self.assertGreater(density_index[Biome.FOREST], 1.8)
        self.assertGreater(density_index[Biome.FOREST], density_index[Biome.BUSHES])
        self.assertGreater(density_index[Biome.BUSHES], density_index[Biome.PRAIRIE])
        self.assertGreater(density_index[Biome.PRAIRIE], 0.0)
        self.assertLess(density_index[Biome.PRAIRIE], 0.5)

    def test_biome_weighted_spawn_positions_stay_inside_bounds(self) -> None:
        biome_map = self._three_vertical_biome_map()
        spawner = FoodSpawner(
            FoodConfig(min_food_radius=8.0, max_food_radius=8.0),
            Random(5),
            replace(biome_map, max_spawn_attempts=1),
        )

        for _ in range(100):
            x, y = spawner._spawn_position(WORLD_BOUNDS, radius=8.0)
            self.assertGreaterEqual(x, WORLD_BOUNDS[0] + 8.0)
            self.assertLessEqual(x, WORLD_BOUNDS[2] - 8.0)
            self.assertGreaterEqual(y, WORLD_BOUNDS[1] + 8.0)
            self.assertLessEqual(y, WORLD_BOUNDS[3] - 8.0)

    def _three_vertical_biome_map(self) -> BiomeMap:
        biome_ids = np.array(
            [[Biome.PRAIRIE, Biome.BUSHES, Biome.FOREST]],
            dtype=np.uint8,
        )
        return BiomeMap(
            biome_ids=biome_ids,
            render_rgba=np.zeros((1, 3, 4), dtype=np.uint8),
            world_bounds=WORLD_BOUNDS,
            area_shares={
                Biome.PRAIRIE: 1.0 / 3.0,
                Biome.BUSHES: 1.0 / 3.0,
                Biome.FOREST: 1.0 / 3.0,
            },
            spawn_weights={
                Biome.FOREST: 2.75,
                Biome.BUSHES: 1.25,
                Biome.PRAIRIE: 0.25,
            },
            uniform_spawn_chance=0.0,
            max_spawn_attempts=32,
        )


if __name__ == "__main__":
    unittest.main()
