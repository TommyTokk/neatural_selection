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


def low_creature_config(**overrides: object) -> FoodConfig:
    defaults = {
        "low_food_burst_items": 0,
        "max_biomass_spawns_per_second": 0.0,
    }
    defaults.update(overrides)
    return FoodConfig(**defaults)


def low_food_config(**overrides: object) -> FoodConfig:
    defaults = {
        "low_creature_burst_items": 0,
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


class FoodSpawnerLowCreatureBurstTest(unittest.TestCase):
    def test_low_creature_burst_uses_configured_interval(self) -> None:
        config = low_creature_config()
        spawner = FoodSpawner(config, Random(1))
        expected_burst_items = ceil(
            config.low_creature_burst_items
            * spawner.low_creature_shortage_ratio(9)
        )

        early_foods = spawner.update(
            config.low_creature_burst_interval - 0.01,
            BOUNDS,
            current_food_count=config.max_food_items,
            creature_count=9,
            available_biomass=10_000.0,
        )
        burst_foods = spawner.update(
            0.01,
            BOUNDS,
            current_food_count=config.max_food_items,
            creature_count=9,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(early_foods), 0)
        self.assertEqual(len(burst_foods), expected_burst_items)

    def test_burst_size_scales_with_low_creature_shortage(self) -> None:
        config = low_creature_config()

        nine_creature_spawner = FoodSpawner(config, Random(1))
        five_creature_spawner = FoodSpawner(config, Random(1))
        nine_creature_foods = nine_creature_spawner.update(
            config.low_creature_burst_interval,
            BOUNDS,
            current_food_count=config.max_food_items,
            creature_count=9,
            available_biomass=10_000.0,
        )
        five_creature_foods = five_creature_spawner.update(
            config.low_creature_burst_interval,
            BOUNDS,
            current_food_count=config.max_food_items,
            creature_count=5,
            available_biomass=10_000.0,
        )

        self.assertEqual(
            len(nine_creature_foods),
            ceil(
                config.low_creature_burst_items
                * nine_creature_spawner.low_creature_shortage_ratio(9)
            ),
        )
        self.assertEqual(
            len(five_creature_foods),
            ceil(
                config.low_creature_burst_items
                * five_creature_spawner.low_creature_shortage_ratio(5)
            ),
        )
        self.assertGreater(len(five_creature_foods), len(nine_creature_foods))

    def test_no_low_creature_burst_at_threshold(self) -> None:
        spawner = FoodSpawner(low_creature_config(), Random(1))

        foods = spawner.update(
            spawner.config.low_creature_burst_interval,
            BOUNDS,
            current_food_count=spawner.config.max_food_items - 100,
            creature_count=10,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(foods), 0)

    def test_low_creature_burst_respects_slots_and_biomass(self) -> None:
        spawner = FoodSpawner(low_creature_config(), Random(1))
        spawn_pressure = spawner.creature_pressure_factor(5)
        seven_foods_biomass = (
            7 * spawner.average_food_energy_value()
        ) / spawn_pressure

        biomass_limited_foods = spawner.update(
            spawner.config.low_creature_burst_interval,
            BOUNDS,
            current_food_count=spawner.config.max_food_items,
            creature_count=5,
            available_biomass=seven_foods_biomass,
        )
        slot_limited_spawner = FoodSpawner(low_creature_config(), Random(1))
        slot_limited_foods = slot_limited_spawner.update(
            slot_limited_spawner.config.low_creature_burst_interval,
            BOUNDS,
            current_food_count=slot_limited_spawner.food_capacity(5) - 2,
            creature_count=5,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(biomass_limited_foods), 7)
        self.assertEqual(len(slot_limited_foods), 2)


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
            creature_count=12,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(foods), emergency_target - current_food_count)

    def test_lower_food_count_gets_larger_immediate_refill(self) -> None:
        config = low_food_config()

        nearly_empty_foods = FoodSpawner(config, Random(1)).update(
            0.0,
            BOUNDS,
            current_food_count=20,
            creature_count=12,
            available_biomass=10_000.0,
        )
        low_foods = FoodSpawner(config, Random(1)).update(
            0.0,
            BOUNDS,
            current_food_count=300,
            creature_count=12,
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
            creature_count=12,
            available_biomass=10_000.0,
        )

        self.assertEqual(len(foods), 0)

    def test_low_food_refill_respects_biomass_limit(self) -> None:
        config = low_food_config()
        spawner = FoodSpawner(config, Random(1))
        spawn_pressure = spawner.creature_pressure_factor(12)
        five_foods_biomass = (
            5 * spawner.average_food_energy_value()
        ) / spawn_pressure

        foods = spawner.update(
            0.0,
            BOUNDS,
            current_food_count=20,
            creature_count=12,
            available_biomass=five_foods_biomass,
        )

        self.assertEqual(len(foods), 5)

    def test_spawn_rate_increases_as_food_count_lowers(self) -> None:
        config = FoodConfig()
        spawner = FoodSpawner(config, Random(1))
        spawn_pressure = spawner.creature_pressure_factor(12)
        high_food_pressure = spawner.food_regrowth_pressure(
            600,
            config.max_food_items,
        )
        low_food_pressure = spawner.food_regrowth_pressure(
            20,
            config.max_food_items,
        )

        high_food_rate = spawner._spawn_rate_per_second(
            spawn_pressure=spawn_pressure,
            food_pressure=high_food_pressure,
        )
        low_food_rate = spawner._spawn_rate_per_second(
            spawn_pressure=spawn_pressure,
            food_pressure=low_food_pressure,
        )

        self.assertGreater(low_food_rate, high_food_rate)


if __name__ == "__main__":
    unittest.main()
