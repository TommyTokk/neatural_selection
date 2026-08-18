from __future__ import annotations

from dataclasses import replace
from math import floor
from random import Random
import unittest

import numpy as np

from configs.sim_config import BiomeConfig, FoodClusterConfig, FoodConfig
from src.biome import Biome, BiomeGenerationHandler, BiomeMap
from src.food_spawner import FoodSpawner


WORLD_BOUNDS = (-1600.0, -1100.0, 1600.0, 1100.0)


class BiomeGenerationHandlerTest(unittest.TestCase):
    def test_explicit_normalized_thresholds_are_exposed(self) -> None:
        cluster_config = FoodClusterConfig(prairie_max=0.2, bush_max=0.8)

        biome_map = BiomeGenerationHandler(
            BiomeConfig(seed=7),
            cluster_config,
        ).generate(WORLD_BOUNDS)

        self.assertEqual(biome_map.prairie_max, 0.2)
        self.assertEqual(biome_map.bush_max, 0.8)

    @staticmethod
    def _reference_density(biome_map: BiomeMap, x: float, y: float) -> float:
        weights = [
            max(0.0, weight) for weight in biome_map.spawn_weights.values()
        ]
        maximum = max(weights, default=0.0)
        if maximum <= 0.0:
            return 1.0
        uniform_probability = max(
            0.0,
            min(1.0, biome_map.uniform_spawn_chance),
        )
        left, bottom, right, top = biome_map.world_bounds
        cell_width = max(0.0001, right - left) / biome_map.grid_width
        cell_height = max(0.0001, top - bottom) / biome_map.grid_height
        grid_x = (x - left) / cell_width - 0.5
        grid_y = (y - bottom) / cell_height - 0.5
        column0 = floor(grid_x)
        row0 = floor(grid_y)
        column1 = column0 + 1
        row1 = row0 + 1
        u = grid_x - column0
        v = grid_y - row0
        column0 = max(0, min(biome_map.grid_width - 1, column0))
        column1 = max(0, min(biome_map.grid_width - 1, column1))
        row0 = max(0, min(biome_map.grid_height - 1, row0))
        row1 = max(0, min(biome_map.grid_height - 1, row1))

        def sample(column: int, row: int) -> float:
            biome = Biome(int(biome_map.biome_ids[row, column]))
            normalized_weight = (
                max(0.0, biome_map.spawn_weights[biome]) / maximum
            )
            return uniform_probability + (
                1.0 - uniform_probability
            ) * normalized_weight

        c00 = sample(column0, row0)
        c10 = sample(column1, row0)
        c01 = sample(column0, row1)
        c11 = sample(column1, row1)
        result = (
            c00 * (1.0 - u) * (1.0 - v)
            + c10 * u * (1.0 - v)
            + c01 * (1.0 - u) * v
            + c11 * u * v
        )
        return max(0.0, min(1.0, result))

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

    def test_density_uses_resource_spawn_weights(self) -> None:
        biome_map = self._three_vertical_biome_map()

        prairie = biome_map.expected_food_density_at(-1000.0, 0.0)
        bushes = biome_map.expected_food_density_at(0.0, 0.0)
        forest = biome_map.expected_food_density_at(1000.0, 0.0)

        self.assertGreaterEqual(prairie, 0.0)
        self.assertLessEqual(forest, 1.0)
        self.assertLess(prairie, bushes)
        self.assertLess(bushes, forest)

    def test_density_equal_spawn_weights_use_safe_fallback(self) -> None:
        biome_map = replace(
            self._three_vertical_biome_map(),
            spawn_weights={biome: 1.0 for biome in Biome},
        )

        self.assertEqual(
            biome_map.expected_food_density_at(-1000.0, 0.0),
            1.0,
        )
        self.assertEqual(biome_map.expected_food_density_at(0.0, 0.0), 1.0)
        self.assertEqual(
            biome_map.expected_food_density_at(1000.0, 0.0),
            1.0,
        )

    def test_density_bilinearly_interpolates_between_cell_centers(self) -> None:
        biome_map = BiomeMap(
            biome_ids=np.array(
                [
                    [Biome.PRAIRIE, Biome.FOREST],
                    [Biome.PRAIRIE, Biome.FOREST],
                ],
                dtype=np.uint8,
            ),
            render_rgba=np.zeros((2, 2, 4), dtype=np.uint8),
            world_bounds=(10.0, 20.0, 30.0, 40.0),
            area_shares={biome: 0.0 for biome in Biome},
            spawn_weights={
                Biome.PRAIRIE: 0.0,
                Biome.BUSHES: 0.5,
                Biome.FOREST: 1.0,
            },
            uniform_spawn_chance=0.0,
            max_spawn_attempts=4,
        )

        self.assertEqual(biome_map.expected_food_density_at(15.0, 25.0), 0.0)
        self.assertEqual(biome_map.expected_food_density_at(25.0, 35.0), 1.0)
        self.assertEqual(biome_map.expected_food_density_at(20.0, 30.0), 0.5)

    def test_density_interpolation_clamps_outside_world_bounds(self) -> None:
        biome_map = BiomeMap(
            biome_ids=np.array(
                [[Biome.PRAIRIE, Biome.FOREST]],
                dtype=np.uint8,
            ),
            render_rgba=np.zeros((1, 2, 4), dtype=np.uint8),
            world_bounds=(10.0, 20.0, 30.0, 40.0),
            area_shares={biome: 0.0 for biome in Biome},
            spawn_weights={
                Biome.PRAIRIE: 0.0,
                Biome.BUSHES: 0.5,
                Biome.FOREST: 1.0,
            },
            uniform_spawn_chance=0.0,
            max_spawn_attempts=4,
        )

        self.assertEqual(
            biome_map.expected_food_density_at(-100.0, -100.0),
            0.0,
        )
        self.assertEqual(
            biome_map.expected_food_density_at(100.0, 100.0),
            1.0,
        )

    def test_cached_density_matches_reference_near_boundaries(self) -> None:
        biome_map = self._three_vertical_biome_map()
        rng = Random(4421)
        points = [
            (-1600.0, -1100.0),
            (1600.0, 1100.0),
            (-1066.6666666667, 0.0),
            (0.0, 0.0),
            (1066.6666666667, 0.0),
        ]
        for boundary in (-533.3333333333, 533.3333333333):
            points.extend(
                (boundary + offset, 0.0)
                for offset in (-1e-9, 0.0, 1e-9)
            )
        points.extend(
            (
                rng.uniform(WORLD_BOUNDS[0], WORLD_BOUNDS[2]),
                rng.uniform(WORLD_BOUNDS[1], WORLD_BOUNDS[3]),
            )
            for _ in range(100)
        )

        for x, y in points:
            with self.subTest(x=x, y=y):
                expected = self._reference_density(biome_map, x, y)
                actual = biome_map.expected_food_density_at(x, y)
                self.assertEqual(actual, expected)
                self.assertEqual(actual <= 0.0, expected <= 0.0)
                self.assertEqual(actual >= 1.0, expected >= 1.0)

        self.assertEqual(biome_map._expected_density_grid.dtype, np.float64)
        self.assertFalse(biome_map._expected_density_grid.flags.writeable)

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
