from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys
import unittest

if "arcade" not in sys.modules:
    arcade = ModuleType("arcade")

    @dataclass(slots=True)
    class FakeRect:
        left: float
        bottom: float
        width: float
        height: float

        @property
        def right(self) -> float:
            return self.left + self.width

        @property
        def top(self) -> float:
            return self.bottom + self.height

        @property
        def center_x(self) -> float:
            return self.left + self.width / 2.0

        @property
        def center_y(self) -> float:
            return self.bottom + self.height / 2.0

    def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
        return FakeRect(left, bottom, width, height)

    arcade.Rect = FakeRect
    arcade.LBWH = fake_lbwh
    sys.modules["arcade"] = arcade

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import build_sim_config
from src.creature import PhysicalTraits, VisionTraits
from src.vision import (
    BoundarySnapshot,
    SensorSnapshot,
    VisionSenseResult,
    VisionTargetSnapshot,
)
from src.world import World


class FakeRng:
    def __init__(self, gaussian_values: list[float]) -> None:
        self.gaussian_values = gaussian_values

    def gauss(self, mean: float, deviation: float) -> float:
        del mean, deviation
        return self.gaussian_values.pop(0)


class FakeFitness:
    def __init__(self) -> None:
        self.age_seconds = 0.0
        self.discovered_food_ids: list[int] = []

    def record_food_discoveries(self, food_ids: list[int]) -> None:
        self.discovered_food_ids.extend(food_ids)


class FakeVisionSystem:
    def __init__(self) -> None:
        self.sense_with_visible_food_ids_calls = 0
        self.sense_calls = 0

    def sense_with_visible_food_ids(self, *args: object, **kwargs: object) -> VisionSenseResult:
        del args, kwargs
        self.sense_with_visible_food_ids_calls += 1
        return VisionSenseResult(
            snapshot=self._snapshot(),
            visible_food_ids=[101, 202],
        )

    def sense(self, *args: object, **kwargs: object) -> SensorSnapshot:
        del args, kwargs
        self.sense_calls += 1
        return self._snapshot()

    def visible_foods(self, *args: object, **kwargs: object) -> list[object]:
        raise AssertionError("visible_foods should not be called during hot sensing.")

    def _snapshot(self) -> SensorSnapshot:
        empty = VisionTargetSnapshot(
            visible=0.0,
            proximity=0.0,
            angle=0.0,
            density=0.0,
            count=0,
        )
        return SensorSnapshot(
            food=empty,
            creatures=empty,
            walls=empty,
            boundary=BoundarySnapshot(pressure=0.0, turn=0.0),
            energy=1.0,
            speed=0.0,
            vision_range=1.0,
            vision_angle=1.0,
            vision_energy_cost=0.0,
            maturity=0.0,
            visible_food_count=0.0,
            visible_creature_count=0.0,
            clock_tik_tok=1.0,
            clock_chronometer=0.0,
            clock_time_alive=0.0,
            is_grabbing=0.0,
        )


class WorldVisionMutationTest(unittest.TestCase):
    def make_world_with_mutations(self, mutations: list[float]) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.rng = FakeRng(mutations)
        return world

    def test_mutated_vision_inherits_parent_values_with_small_variation(self) -> None:
        world = self.make_world_with_mutations([6.0, -0.05])

        child_vision = world._mutated_vision(
            VisionTraits(
                range=120.0,
                angle=0.80,
            )
        )

        self.assertAlmostEqual(child_vision.range, 126.0)
        self.assertAlmostEqual(child_vision.angle, 0.75)

    def test_mutated_vision_clamps_to_config_bounds(self) -> None:
        world = self.make_world_with_mutations([-100.0, 100.0])

        child_vision = world._mutated_vision(
            VisionTraits(
                range=world.config.vision.min_range,
                angle=world.config.vision.max_angle,
            )
        )

        self.assertEqual(child_vision.range, world.config.vision.min_range)
        self.assertEqual(child_vision.angle, world.config.vision.max_angle)

    def test_mutated_physical_traits_inherit_parent_values_with_small_variation(self) -> None:
        world = self.make_world_with_mutations([2.0, -0.03])

        child_traits, delta = world._mutated_physical_traits(
            PhysicalTraits(
                radius=16.0,
                movement_cost_multiplier=1.0,
            )
        )

        self.assertAlmostEqual(child_traits.radius, 18.0)
        self.assertAlmostEqual(child_traits.movement_cost_multiplier, 0.97)
        self.assertAlmostEqual(delta.radius, 2.0)
        self.assertAlmostEqual(delta.movement_cost_multiplier, -0.03)

    def test_mutated_physical_traits_clamp_to_config_bounds(self) -> None:
        world = self.make_world_with_mutations([-100.0, 100.0])

        child_traits, delta = world._mutated_physical_traits(
            PhysicalTraits(
                radius=world.config.trait.min_radius,
                movement_cost_multiplier=world.config.trait.max_movement_cost_multiplier,
            )
        )

        self.assertEqual(child_traits.radius, world.config.trait.min_radius)
        self.assertEqual(
            child_traits.movement_cost_multiplier,
            world.config.trait.max_movement_cost_multiplier,
        )
        self.assertEqual(delta.radius, 0.0)
        self.assertEqual(delta.movement_cost_multiplier, 0.0)

    def test_sensor_snapshot_records_food_discoveries_from_single_vision_pass(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.creatures = []
        world._held_food_by_creature_id = {}
        world._chronometers = {}
        world._nearby_foods_for = lambda creature, radius: []
        world._ignored_food_ids_for = lambda creature: set()
        world.MAX_SPEED = 170.0
        world.vision = FakeVisionSystem()
        fitness = FakeFitness()
        creature = SimpleNamespace(
            creature_id=1,
            vision=SimpleNamespace(range=120.0),
            energy=1.0,
        )
        world.fitness = {1: fitness}

        snapshot = world._sensor_snapshot_for(
            creature,
            record_food_discoveries=True,
        )

        self.assertIsInstance(snapshot, SensorSnapshot)
        self.assertEqual(fitness.discovered_food_ids, [101, 202])
        self.assertEqual(world.vision.sense_with_visible_food_ids_calls, 1)
        self.assertEqual(world.vision.sense_calls, 0)


if __name__ == "__main__":
    unittest.main()
