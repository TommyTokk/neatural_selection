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
from src.action import Action
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
        self.last_creatures: object | None = None

    def sense_with_visible_food_ids(self, *args: object, **kwargs: object) -> VisionSenseResult:
        del kwargs
        self.sense_with_visible_food_ids_calls += 1
        self.last_creatures = args[2]
        return VisionSenseResult(
            snapshot=self._snapshot(),
            visible_food_ids=[101, 202],
        )

    def sense(self, *args: object, **kwargs: object) -> SensorSnapshot:
        del kwargs
        self.sense_calls += 1
        self.last_creatures = args[2]
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


class FakeBiomeMap:
    def __init__(self, fertility: float = 0.0) -> None:
        self.fertility = fertility

    def fertility_at(self, x: float, y: float) -> float:
        del x, y
        return self.fertility


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
        world._nearby_creatures_for = lambda creature, radius: world.creatures
        world._ignored_food_ids_for = lambda creature: set()
        world.MAX_SPEED = 170.0
        world.vision = FakeVisionSystem()
        fitness = FakeFitness()
        creature = SimpleNamespace(
            creature_id=1,
            position=(0.0, 0.0),
            heading=0.0,
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

    def test_sensor_snapshot_uses_nearby_creature_candidates(self) -> None:
        world = self.make_world_for_biome_sensors()
        observer = self.biome_sensor_creature()
        nearby = self.biome_sensor_creature(position=(10.0, 0.0))
        nearby.creature_id = 2
        world.creatures = [observer, nearby]
        world._nearby_creatures_for = lambda creature, radius: [observer, nearby]

        world._sensor_snapshot_for(observer, record_food_discoveries=False)

        self.assertEqual(world.vision.last_creatures, [observer, nearby])

    def test_biome_memory_activation_overwrites_reused_creature_id(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.7)
        world._previous_biome_here_by_creature_id = {1: 0.1}
        creature = self.biome_sensor_creature()

        world._activate_creature_biome_memory(creature)

        self.assertAlmostEqual(world._previous_biome_here_by_creature_id[1], 0.7)

    def test_first_biome_sensor_tick_delta_is_zero(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.8)
        creature = self.biome_sensor_creature()
        world._activate_creature_biome_memory(creature)

        snapshot = world._sensor_snapshot_for(
            creature,
            record_food_discoveries=False,
        )

        self.assertAlmostEqual(snapshot.biome.here, 0.8)
        self.assertAlmostEqual(snapshot.biome.delta, 0.0)

    def test_read_only_biome_sensor_snapshot_does_not_mutate_memory(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.9)
        creature = self.biome_sensor_creature()
        world._previous_biome_here_by_creature_id = {1: 0.2}

        first_snapshot = world._sensor_snapshot_for(
            creature,
            record_food_discoveries=False,
        )
        second_snapshot = world._sensor_snapshot_for(
            creature,
            record_food_discoveries=False,
        )

        self.assertAlmostEqual(world._previous_biome_here_by_creature_id[1], 0.2)
        self.assertAlmostEqual(first_snapshot.biome.delta, 1.0)
        self.assertAlmostEqual(second_snapshot.biome.delta, 1.0)

    def test_creature_intent_tick_advances_biome_memory_once(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.85)
        creature = self.biome_sensor_creature()
        world.creatures = [creature]
        world._previous_biome_here_by_creature_id = {1: 0.2}
        world.use_neat_brains = True
        world.neat_controller = SimpleNamespace(
            decide=lambda creature_id, snapshot: Action(
                accelerate=0.0,
                rotate=0.0,
                want_reproduce=0.0,
                want_eat=0.0,
                reset_chronometer=0.0,
                want_grab=0.0,
                want_release=0.0,
            )
        )
        world._last_actions = {}
        world._apply_carry_intent = lambda creature, action: None
        world._apply_action = lambda *args, **kwargs: None

        world._apply_creature_intents()

        self.assertAlmostEqual(world._previous_biome_here_by_creature_id[1], 0.85)

    def test_creature_intents_apply_cached_actions_every_tick(self) -> None:
        world = self.make_world_for_biome_sensors()
        creature = self.biome_sensor_creature()
        world.creatures = [creature]
        world.use_neat_brains = True
        world.physics_step_count = 1
        action = Action(
            accelerate=0.0,
            rotate=0.0,
            want_reproduce=0.0,
            want_eat=0.0,
            reset_chronometer=0.0,
            want_grab=0.0,
            want_release=0.0,
        )
        snapshot = world.vision._snapshot()
        world._last_actions = {creature.creature_id: action}
        world._last_sensor_snapshots = {creature.creature_id: snapshot}
        applied: list[tuple[object, object, object]] = []
        world.neat_controller = SimpleNamespace(
            decide=lambda creature_id, snapshot: self.fail("should not decide")
        )
        world._apply_carry_intent = lambda creature, action: self.fail(
            "should not replay carry intent"
        )
        world._apply_action = (
            lambda active_creature, active_action, active_snapshot, **kwargs: applied.append(
                (active_creature, active_action, active_snapshot)
            )
        )

        world._apply_creature_intents()

        self.assertEqual(applied, [(creature, action, snapshot)])
        self.assertEqual(world.vision.sense_with_visible_food_ids_calls, 0)

    def test_creature_intents_stagger_new_decisions_by_creature_id(self) -> None:
        world = self.make_world_for_biome_sensors()
        creatures = [
            self.biome_sensor_creature(position=(float(index), 0.0))
            for index in range(1, 6)
        ]
        for index, creature in enumerate(creatures, start=1):
            creature.creature_id = index
        world.creatures = creatures
        world.use_neat_brains = True
        world.physics_step_count = 0
        cached_action = Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        cached_snapshot = world.vision._snapshot()
        world._last_actions = {
            creature.creature_id: cached_action for creature in creatures
        }
        world._last_sensor_snapshots = {
            creature.creature_id: cached_snapshot for creature in creatures
        }
        decided_ids: list[int] = []
        world.neat_controller = SimpleNamespace(
            decide=lambda creature_id, snapshot: decided_ids.append(creature_id)
            or cached_action
        )
        world._apply_carry_intent = lambda creature, action: None
        world._apply_action = lambda *args, **kwargs: None

        world._apply_creature_intents()

        self.assertEqual(decided_ids, [5])

    def test_east_facing_biome_sensors_use_y_up_left_right_orientation(self) -> None:
        world = self.make_world_for_biome_sensors()
        creature = self.biome_sensor_creature(position=(0.0, 0.0), heading=0.0)

        _here, forward_left, forward_right = world.biome_sensor_positions_for(creature)

        self.assertGreater(forward_left[1], forward_right[1])

    def make_world_for_biome_sensors(self, fertility: float = 0.0) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.creatures = []
        world._held_food_by_creature_id = {}
        world._chronometers = {}
        world._nearby_foods_for = lambda creature, radius: []
        world._nearby_creatures_for = lambda creature, radius: world.creatures
        world._ignored_food_ids_for = lambda creature: set()
        world.MAX_SPEED = 170.0
        world.vision = FakeVisionSystem()
        world.fitness = {1: FakeFitness()}
        world.biome_map = FakeBiomeMap(fertility)
        world._previous_biome_here_by_creature_id = {}
        return world

    def biome_sensor_creature(
        self,
        *,
        position: tuple[float, float] = (0.0, 0.0),
        heading: float = 0.0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            creature_id=1,
            position=position,
            heading=heading,
            vision=SimpleNamespace(range=120.0),
            energy=1.0,
        )


if __name__ == "__main__":
    unittest.main()
