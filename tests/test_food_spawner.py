from __future__ import annotations

from math import ceil, pi, sqrt
import sys
import types
import unittest
from random import Random


class _Position:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x = x
        self.y = y


class _Body:
    STATIC = object()

    def __init__(self, *args: object, body_type: object | None = None) -> None:
        self.args = args
        self.body_type = body_type
        self._position = _Position()

    @property
    def position(self) -> _Position:
        return self._position

    @position.setter
    def position(self, value: tuple[float, float] | _Position) -> None:
        if isinstance(value, _Position):
            self._position = value
        else:
            self._position = _Position(value[0], value[1])


class _Circle:
    def __init__(self, body: _Body, radius: float) -> None:
        self.body = body
        self.radius = radius
        self.sensor = False


class _ShapeFilter:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = types.SimpleNamespace(
        Body=_Body,
        Circle=_Circle,
        ShapeFilter=_ShapeFilter,
        moment_for_circle=lambda *args: 1.0,
    )

from configs.sim_config import FoodConfig
from src.food import Food
from src.food_spawner import FoodSpawner


BOUNDS = (0.0, 0.0, 1000.0, 1000.0)


def low_food_config(**overrides: object) -> FoodConfig:
    defaults = {
        "max_biomass_spawns_per_second": 0.0,
    }
    defaults.update(overrides)
    return FoodConfig(**defaults)


class FoodEnergyValueTest(unittest.TestCase):
    def test_food_energy_value_uses_area_formula(self) -> None:
        food = Food(
            id=1,
            x=0.0,
            y=0.0,
            radius=8.0,
            energy_density=0.002,
        )

        self.assertAlmostEqual(food.energy_value, pi * 8.0**2 * 0.002)

    def test_food_mass_is_slightly_reduced_before_and_after_resize(self) -> None:
        food = Food(
            id=1,
            x=0.0,
            y=0.0,
            radius=10.0,
            energy_density=0.002,
        )
        expected_initial_mass = (0.2 + 10.0 * 0.035) * 0.9
        original_energy = food.energy_value

        initial_mass = (
            food.body.mass if hasattr(food.body, "mass") else food.body.args[0]
        )
        self.assertAlmostEqual(initial_mass, expected_initial_mass)

        food.consume_energy(original_energy * 0.25, min_remainder_ratio=0.10)

        expected_resized_mass = (0.2 + food.radius * 0.035) * 0.9
        self.assertAlmostEqual(food.body.mass, expected_resized_mass)

    def test_average_radius_food_matches_spawner_average_energy(self) -> None:
        config = FoodConfig(min_food_radius=6.0, max_food_radius=10.0)
        average_radius = (config.min_food_radius + config.max_food_radius) * 0.5
        food = Food(
            id=1,
            x=0.0,
            y=0.0,
            radius=average_radius,
            energy_density=config.energy_density,
        )
        spawner = FoodSpawner(config, Random(1))

        self.assertAlmostEqual(food.energy_value, spawner.average_food_energy_value())

    def test_partial_food_consumption_shrinks_remaining_pellet(self) -> None:
        food = Food(
            id=1,
            x=0.0,
            y=0.0,
            radius=10.0,
            energy_density=0.002,
        )
        original_energy = food.energy_value

        result = food.consume_energy(original_energy * 0.25, min_remainder_ratio=0.10)

        self.assertFalse(result.depleted)
        self.assertAlmostEqual(result.energy_removed, original_energy * 0.25)
        self.assertAlmostEqual(food.energy_value, original_energy * 0.75)
        self.assertAlmostEqual(
            food.radius,
            sqrt(food.energy_value / (pi * food.energy_density)),
        )
        self.assertAlmostEqual(food.shape.radius, food.radius)

    def test_micro_food_remainder_is_depleted(self) -> None:
        food = Food(
            id=1,
            x=0.0,
            y=0.0,
            radius=10.0,
            energy_density=0.002,
        )
        original_energy = food.energy_value

        result = food.consume_energy(original_energy * 0.95, min_remainder_ratio=0.10)

        self.assertTrue(result.depleted)
        self.assertAlmostEqual(result.energy_removed, original_energy)
        self.assertEqual(food.energy_value, 0.0)

    def test_zero_requested_energy_does_not_consume_food(self) -> None:
        food = Food(
            id=1,
            x=0.0,
            y=0.0,
            radius=10.0,
            energy_density=0.002,
        )
        original_energy = food.energy_value

        result = food.consume_energy(0.0, min_remainder_ratio=0.10)

        self.assertFalse(result.depleted)
        self.assertEqual(result.energy_removed, 0.0)
        self.assertEqual(food.energy_value, original_energy)


class FoodSpawnerLogisticGrowthTest(unittest.TestCase):
    def test_high_species_count_does_not_block_normal_spawning(self) -> None:
        config = FoodConfig(
            max_food_items=100,
            max_biomass_spawns_per_second=10.0,
            low_food_burst_items=0,
        )
        spawner = FoodSpawner(config, Random(1))

        foods = spawner.update(
            1.0,
            BOUNDS,
            current_food_count=50,
            active_species_count=50,
            available_biomass=10_000.0,
        )

        self.assertGreater(len(foods), 0)

    def test_available_biomass_is_not_scaled_by_species_or_creatures(self) -> None:
        config = FoodConfig(
            max_food_items=100,
            max_biomass_spawns_per_second=100.0,
            low_food_burst_items=0,
        )
        spawner = FoodSpawner(config, Random(1))
        five_foods_biomass = 5 * spawner.average_food_energy_value()

        foods = spawner.update(
            10.0,
            BOUNDS,
            current_food_count=50,
            active_species_count=50,
            available_biomass=five_foods_biomass,
        )

        self.assertEqual(len(foods), 5)

    def test_zero_food_uses_seeding_pressure(self) -> None:
        config = FoodConfig(
            max_food_items=100,
            max_biomass_spawns_per_second=20.0,
            low_food_burst_items=0,
        )
        spawner = FoodSpawner(config, Random(1))

        foods = spawner.update(
            1.0,
            BOUNDS,
            current_food_count=0,
            active_species_count=4,
            available_biomass=10_000.0,
        )

        self.assertEqual(spawner.food_regrowth_pressure(0, config.max_food_items), 0.05)
        self.assertGreater(len(foods), 0)

    def test_full_capacity_blocks_spawning_and_resets_credits(self) -> None:
        config = FoodConfig(max_food_items=100)
        spawner = FoodSpawner(config, Random(1))
        spawner._spawn_credit = 3.0
        spawner._low_food_burst_credit = 0.5
        spawner._pending_low_food_burst_items = 4

        foods = spawner.update(
            1.0,
            BOUNDS,
            current_food_count=100,
            active_species_count=4,
            available_biomass=10_000.0,
        )

        self.assertEqual(foods, [])
        self.assertEqual(spawner._spawn_credit, 0.0)
        self.assertEqual(spawner._low_food_burst_credit, 0.0)
        self.assertEqual(spawner._pending_low_food_burst_items, 0)

    def test_species_count_scales_regular_spawn_rate(self) -> None:
        spawner = FoodSpawner(FoodConfig(), Random(1))

        low_species_rate = spawner._spawn_rate_per_second(
            active_species_count=1,
            spawn_pressure=1.0,
        )
        high_species_rate = spawner._spawn_rate_per_second(
            active_species_count=8,
            spawn_pressure=1.0,
        )

        self.assertGreater(high_species_rate, low_species_rate)


class FoodSpawnerLowFoodRecoveryTest(unittest.TestCase):
    def test_low_food_refill_does_not_wait_for_burst_interval(self) -> None:
        config = low_food_config()
        spawner = FoodSpawner(config, Random(1))
        emergency_target = ceil(
            config.max_food_items * config.critical_food_ratio
        )
        current_food_count = 20

        foods = spawner.update(
            0.0,
            BOUNDS,
            current_food_count=current_food_count,
            active_species_count=4,
            available_biomass=10_000.0,
        )

        expected_deficit = emergency_target - current_food_count
        self.assertEqual(
            len(foods),
            min(expected_deficit, spawner._burst_spawn_cap()),
        )

        carried_foods = spawner.update(
            0.0,
            BOUNDS,
            current_food_count=current_food_count + len(foods),
            active_species_count=4,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(foods) + len(carried_foods), expected_deficit)

    def test_lower_food_count_gets_larger_immediate_refill(self) -> None:
        config = low_food_config()

        nearly_empty_foods = FoodSpawner(config, Random(1)).update(
            0.0,
            BOUNDS,
            current_food_count=20,
            active_species_count=4,
            available_biomass=10_000.0,
        )
        low_foods = FoodSpawner(config, Random(1)).update(
            0.0,
            BOUNDS,
            current_food_count=300,
            active_species_count=4,
            available_biomass=10_000.0,
        )

        self.assertGreater(len(nearly_empty_foods), len(low_foods))

    def test_low_food_refill_waits_above_critical_ratio(self) -> None:
        config = low_food_config()
        current_food_count = ceil(
            config.max_food_items * config.critical_food_ratio
        ) + 1

        foods = FoodSpawner(config, Random(1)).update(
            0.0,
            BOUNDS,
            current_food_count=current_food_count,
            active_species_count=4,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(foods), 0)

    def test_low_food_refill_respects_biomass_limit(self) -> None:
        config = low_food_config()
        spawner = FoodSpawner(config, Random(1))
        five_foods_biomass = 5 * spawner.average_food_energy_value()

        foods = spawner.update(
            0.0,
            BOUNDS,
            current_food_count=20,
            active_species_count=4,
            available_biomass=five_foods_biomass,
        )

        self.assertEqual(len(foods), 5)

    def test_logistic_food_pressure_peaks_near_half_capacity(self) -> None:
        config = FoodConfig()
        spawner = FoodSpawner(config, Random(1))
        high_food_pressure = spawner.food_regrowth_pressure(
            600,
            config.max_food_items,
        )
        half_capacity_pressure = spawner.food_regrowth_pressure(
            config.max_food_items // 2,
            config.max_food_items,
        )
        low_food_pressure = spawner.food_regrowth_pressure(
            20,
            config.max_food_items,
        )

        self.assertGreater(half_capacity_pressure, high_food_pressure)
        self.assertGreater(half_capacity_pressure, low_food_pressure)


if __name__ == "__main__":
    unittest.main()
