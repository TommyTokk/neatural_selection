from __future__ import annotations

from dataclasses import dataclass
from math import exp
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
    def __init__(
        self,
        gaussian_values: list[float],
        random_value: float = 1.0,
    ) -> None:
        self.gaussian_values = gaussian_values
        self.random_value = random_value

    def gauss(self, mean: float, deviation: float) -> float:
        del mean, deviation
        return self.gaussian_values.pop(0)

    def random(self) -> float:
        return self.random_value


class FakeFitness:
    def __init__(self) -> None:
        self.age_seconds = 0.0


class FakeVisionSystem:
    def __init__(self) -> None:
        self.sense_with_visible_food_ids_calls = 0
        self.sense_calls = 0
        self.last_creatures: object | None = None
        self.last_kwargs: dict[str, object] = {}

    def sense_with_visible_food_ids(self, *args: object, **kwargs: object) -> VisionSenseResult:
        self.last_kwargs = kwargs
        self.sense_with_visible_food_ids_calls += 1
        self.last_creatures = args[2]
        snapshot = self._snapshot()
        snapshot.reproductive_readiness = float(
            kwargs.get("reproductive_readiness", 0.0)
        )
        return VisionSenseResult(
            snapshot=snapshot,
            visible_food_ids=[101, 202],
        )

    def sense(self, *args: object, **kwargs: object) -> SensorSnapshot:
        self.last_kwargs = kwargs
        self.sense_calls += 1
        self.last_creatures = args[2]
        snapshot = self._snapshot()
        snapshot.reproductive_readiness = float(
            kwargs.get("reproductive_readiness", 0.0)
        )
        return snapshot

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
            reproductive_readiness=0.0,
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
    def test_communication_intents_commit_together_and_scale_deposits_by_dt(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world.creatures = [
            SimpleNamespace(creature_id=1, position=(10.0, 20.0)),
            SimpleNamespace(creature_id=2, position=(30.0, 40.0)),
        ]
        world._last_actions = {
            1: Action(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                emit_sound=0.8,
                sound_tone=-0.25,
                emit_trail_pheromone=0.5,
                emit_alarm_pheromone=0.25,
            ),
            2: Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
        captured_signals: list[object] = []
        deposits: list[tuple[tuple[float, float], float, float]] = []
        world.acoustics = SimpleNamespace(
            replace_signals=lambda signals: captured_signals.extend(signals)
        )
        world.pheromones = SimpleNamespace(
            deposit_many=lambda positions, trail_amounts, alarm_amounts: deposits.extend(
                (
                    (tuple(position), float(trail_amount), float(alarm_amount))
                    for position, trail_amount, alarm_amount in zip(
                        positions,
                        trail_amounts,
                        alarm_amounts,
                    )
                )
            )
        )

        world._commit_communication_intents(1.0 / 60.0)

        self.assertEqual(len(captured_signals), 1)
        self.assertEqual(captured_signals[0].emitter_id, 1)
        self.assertAlmostEqual(captured_signals[0].tone, -0.25)
        rate_per_step = world.config.communication.pheromone_deposit_rate / 60.0
        self.assertAlmostEqual(deposits[0][1], 0.5 * rate_per_step)
        self.assertAlmostEqual(deposits[0][2], 0.25 * rate_per_step)
        self.assertEqual(len(deposits), 1)

    def test_cached_acoustic_level_replaces_state_until_next_decision(
        self,
    ) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        creature = SimpleNamespace(creature_id=1, position=(10.0, 20.0))
        active = Action(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            emit_sound=0.8,
            sound_tone=0.25,
        )
        world.creatures = [creature]
        world._last_actions = {creature.creature_id: active}
        replacements: list[tuple[object, ...]] = []
        world.acoustics = SimpleNamespace(
            replace_signals=lambda signals: replacements.append(
                tuple(signals)
            )
        )
        world.pheromones = SimpleNamespace(
            deposit_many=lambda *_args: None
        )

        for _ in range(3):
            world._commit_communication_intents(1.0 / 60.0)
        world._last_actions[creature.creature_id] = Action(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        world._commit_communication_intents(1.0 / 60.0)

        self.assertEqual([len(signals) for signals in replacements], [1, 1, 1, 0])
        self.assertTrue(
            all(
                signals[0].emitter_id == creature.creature_id
                and signals[0].strength == 0.8
                and signals[0].tone == 0.25
                for signals in replacements[:3]
            )
        )

    def test_inactive_pheromone_population_skips_batch_deposition(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.creatures = [
            SimpleNamespace(creature_id=1, position=(10.0, 20.0)),
            SimpleNamespace(creature_id=2, position=(30.0, 40.0)),
        ]
        world._last_actions = {
            creature.creature_id: Action(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            for creature in world.creatures
        }
        calls: list[object] = []
        world.acoustics = SimpleNamespace(replace_signals=lambda signals: None)
        world.pheromones = SimpleNamespace(
            deposit_many=lambda *args: calls.append(args)
        )

        world._commit_communication_intents(1.0 / 60.0)

        self.assertEqual(calls, [])
        self.assertFalse(hasattr(world, "_communication_positions"))

    def make_world_with_mutations(
        self,
        mutations: list[float],
        *,
        random_value: float = 1.0,
    ) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.rng = FakeRng(mutations, random_value)
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

    def test_digestive_mutation_probability_zero_is_exact_inheritance(
        self,
    ) -> None:
        world = self.make_world_with_mutations(
            [0.0, 0.0],
            random_value=0.0,
        )
        world.config.trait.digestive_trait_mutation_rate = 0.0
        parent = PhysicalTraits(
            radius=16.0,
            movement_cost_multiplier=1.0,
            stomach_capacity=2.1,
            digestion_rate=0.27,
            digestion_efficiency=0.94,
        )

        child, delta = world._mutated_physical_traits(parent)

        self.assertEqual(child, parent)
        self.assertEqual(delta.stomach_capacity, 0.0)
        self.assertEqual(delta.digestion_rate, 0.0)
        self.assertEqual(delta.digestion_efficiency, 0.0)

    def test_digestive_mutation_probability_one_mutates_and_clamps_all(
        self,
    ) -> None:
        world = self.make_world_with_mutations(
            [0.0, 0.0, 100.0, -100.0, 100.0],
            random_value=0.0,
        )
        world.config.trait.digestive_trait_mutation_rate = 1.0

        child, delta = world._mutated_physical_traits(
            PhysicalTraits(radius=16.0)
        )

        self.assertEqual(
            child.stomach_capacity,
            world.config.trait.max_stomach_capacity,
        )
        self.assertEqual(
            child.digestion_rate,
            world.config.trait.min_digestion_rate,
        )
        self.assertEqual(
            child.digestion_efficiency,
            world.config.trait.max_digestion_efficiency,
        )
        self.assertNotEqual(delta.stomach_capacity, 0.0)
        self.assertNotEqual(delta.digestion_rate, 0.0)
        self.assertNotEqual(delta.digestion_efficiency, 0.0)

    def test_sensor_snapshot_uses_standard_vision_pass(self) -> None:
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

        snapshot = world._sensor_snapshot_for(creature)

        self.assertIsInstance(snapshot, SensorSnapshot)
        self.assertEqual(world.vision.sense_with_visible_food_ids_calls, 0)
        self.assertEqual(world.vision.sense_calls, 1)

    def test_sensor_snapshot_uses_nearby_creature_candidates(self) -> None:
        world = self.make_world_for_biome_sensors()
        observer = self.biome_sensor_creature()
        nearby = self.biome_sensor_creature(position=(10.0, 0.0))
        nearby.creature_id = 2
        world.creatures = [observer, nearby]
        world._nearby_creatures_for = lambda creature, radius: [observer, nearby]

        world._sensor_snapshot_for(observer)

        self.assertEqual(world.vision.last_creatures, [observer, nearby])

    def test_biome_memory_initializes_from_spawn_position(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.7)
        creature = self.biome_sensor_creature()
        creature.biome_fertility_ema = 0.1

        world._initialize_creature_biome_memory(creature)

        self.assertAlmostEqual(creature.biome_fertility_ema, 0.7)
        self.assertAlmostEqual(creature.biome_fertility_ema_updated_at, 0.0)

    def test_first_biome_sensor_tick_delta_is_zero(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.8)
        creature = self.biome_sensor_creature()
        world._initialize_creature_biome_memory(creature)

        snapshot = world._sensor_snapshot_for(creature)

        self.assertAlmostEqual(snapshot.biome.here, 0.8)
        self.assertAlmostEqual(snapshot.biome.trend, 0.0)

    def test_read_only_biome_sensor_snapshot_does_not_adapt_memory(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.9)
        creature = self.biome_sensor_creature()
        creature.biome_fertility_ema = 0.2

        first_snapshot = world._sensor_snapshot_for(creature)
        second_snapshot = world._sensor_snapshot_for(creature)

        self.assertAlmostEqual(creature.biome_fertility_ema, 0.2)
        self.assertAlmostEqual(first_snapshot.biome.trend, 0.7)
        self.assertAlmostEqual(second_snapshot.biome.trend, 0.7)

    def test_creature_intent_tick_adapts_biome_memory_by_elapsed_time(self) -> None:
        world = self.make_world_for_biome_sensors(fertility=0.85)
        creature = self.biome_sensor_creature()
        world.creatures = [creature]
        creature.biome_fertility_ema = 0.2
        creature.biome_fertility_ema_updated_at = 0.0
        world.elapsed_time = 1.5
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
        world._simulation_step = (
            creature.creature_id
            % world.config.scheduler.decision_period_steps
        )
        world._apply_carry_intent = lambda creature, action: None
        world._apply_action = lambda *args, **kwargs: None

        world._apply_creature_intents()

        expected = 0.2 + (0.85 - 0.2) * (1.0 - exp(-0.5))
        self.assertAlmostEqual(creature.biome_fertility_ema, expected)

    def test_chronometer_reset_uses_strict_centered_intent_threshold(self) -> None:
        def chronometer_after_intent(value: float) -> float:
            world = self.make_world_for_biome_sensors()
            creature = self.biome_sensor_creature()
            world.creatures = [creature]
            world._chronometers = {creature.creature_id: 4.0}
            world.neat_controller = SimpleNamespace(
                decide=lambda creature_id, snapshot: Action(
                    accelerate=0.0,
                    rotate=0.0,
                    want_reproduce=0.0,
                    want_eat=0.0,
                    reset_chronometer=value,
                    want_grab=0.0,
                    want_release=0.0,
                )
            )
            world._last_actions = {}
            world._simulation_step = (
                creature.creature_id
                % world.config.scheduler.decision_period_steps
            )
            world._apply_carry_intent = lambda creature, action: None
            world._apply_action = lambda *args, **kwargs: None

            world._apply_creature_intents()

            return world._chronometers[creature.creature_id]

        self.assertEqual(chronometer_after_intent(0.0), 4.0)
        self.assertEqual(chronometer_after_intent(0.1), 4.0)
        self.assertEqual(chronometer_after_intent(0.100001), 0.0)

    def test_biome_memory_is_rate_independent(self) -> None:
        def adapt_at_rate(step: float) -> float:
            world = self.make_world_for_biome_sensors(fertility=1.0)
            creature = self.biome_sensor_creature()
            creature.biome_fertility_ema = 0.0
            creature.biome_fertility_ema_updated_at = 0.0
            snapshot = world._sensor_snapshot_for(creature)
            elapsed = step
            while elapsed <= 3.0 + 1e-9:
                world.elapsed_time = min(elapsed, 3.0)
                world._adapt_creature_biome_memory(creature, snapshot)
                elapsed += step
            return creature.biome_fertility_ema

        self.assertAlmostEqual(adapt_at_rate(0.5), 1.0 - exp(-1.0))
        self.assertAlmostEqual(adapt_at_rate(1.0), adapt_at_rate(0.5))

    def test_biome_gradients_are_local_signed_differences(self) -> None:
        world = self.make_world_for_biome_sensors()
        world.biome_map = SimpleNamespace(
            fertility_at=lambda x, y: 0.5 + y / 200.0
        )
        creature = self.biome_sensor_creature()

        snapshot = world._sensor_snapshot_for(creature)

        self.assertAlmostEqual(snapshot.biome.here, 0.5)
        self.assertGreater(snapshot.biome.left_gradient, 0.0)
        self.assertLess(snapshot.biome.right_gradient, 0.0)

    def test_reproductive_readiness_is_independent_from_infant_cutoff(self) -> None:
        world = self.make_world_for_biome_sensors()
        creature = self.biome_sensor_creature()
        world.fitness[creature.creature_id].age_seconds = 11.9
        self.assertTrue(world._is_infant(creature))

        world.fitness[creature.creature_id].age_seconds = 12.0
        self.assertFalse(world._is_infant(creature))
        adolescent_snapshot = world._sensor_snapshot_for(creature)
        self.assertAlmostEqual(adolescent_snapshot.reproductive_readiness, 0.6)

        world.fitness[creature.creature_id].age_seconds = 20.0

        snapshot = world._sensor_snapshot_for(creature)

        self.assertEqual(snapshot.reproductive_readiness, 1.0)

    def test_creature_intents_apply_cached_actions_every_tick(self) -> None:
        world = self.make_world_for_biome_sensors()
        creature = self.biome_sensor_creature()
        world.creatures = [creature]
        world._simulation_step = 0
        action = Action(
            accelerate=0.0,
            rotate=0.0,
            want_reproduce=0.0,
            want_eat=0.0,
            reset_chronometer=0.0,
            want_grab=0.0,
            want_release=0.0,
            herding=0.15,
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
        self.assertEqual(applied[0][1].herding, 0.15)
        self.assertEqual(world.vision.sense_with_visible_food_ids_calls, 0)
        self.assertEqual(creature.biome_fertility_ema, 0.0)

    def test_cached_input_snapshots_are_captured_only_when_requested(self) -> None:
        world = self.make_world_for_biome_sensors()
        creature = self.biome_sensor_creature()
        world.creatures = [creature]
        world._simulation_step = 0
        action = Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        snapshot = world.vision._snapshot()
        world._last_actions = {creature.creature_id: action}
        world._last_sensor_snapshots = {creature.creature_id: snapshot}
        captures: list[int] = []
        runtime_flags: list[bool] = []
        world.neat_controller = SimpleNamespace(
            decide=lambda creature_id, snapshot: self.fail("should not decide"),
            capture_input_snapshot=lambda creature_id: captures.append(
                creature_id
            ),
        )
        world._apply_carry_intent = lambda creature, action: None
        world._apply_action = lambda *args, **kwargs: runtime_flags.append(
            kwargs["capture_runtime"]
        )

        world.selected_creature_id = None
        world._flocking_capture_due_this_step = False
        world._apply_creature_intents_with_spatial_cache()
        self.assertEqual(captures, [])
        self.assertEqual(runtime_flags, [False])

        world.selected_creature_id = creature.creature_id
        world._apply_creature_intents_with_spatial_cache()
        self.assertEqual(captures, [creature.creature_id])
        self.assertEqual(runtime_flags, [False, True])

        world.selected_creature_id = None
        world._flocking_capture_due_this_step = True
        world._apply_creature_intents_with_spatial_cache()
        self.assertEqual(captures, [creature.creature_id, creature.creature_id])
        self.assertEqual(runtime_flags, [False, True, True])

    def test_creature_intents_stagger_new_decisions_by_creature_id(self) -> None:
        world = self.make_world_for_biome_sensors()
        creatures = [
            self.biome_sensor_creature(position=(float(index), 0.0))
            for index in range(1, 6)
        ]
        for index, creature in enumerate(creatures, start=1):
            creature.creature_id = index
        world.creatures = creatures
        world._simulation_step = 0
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

        self.assertEqual(decided_ids, [3])

    def test_east_facing_biome_sensors_use_y_up_left_right_orientation(self) -> None:
        world = self.make_world_for_biome_sensors()
        creature = self.biome_sensor_creature(position=(0.0, 0.0), heading=0.0)

        _here, forward_left, forward_right = world.biome_sensor_positions_for(creature)

        self.assertEqual(forward_left, (96.0, 48.0))
        self.assertEqual(forward_right, (96.0, -48.0))
        self.assertGreater(forward_left[1], forward_right[1])

    def make_world_for_biome_sensors(self, fertility: float = 0.0) -> World:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world.elapsed_time = 0.0
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
            biome_fertility_ema=0.0,
            biome_fertility_ema_updated_at=0.0,
        )


if __name__ == "__main__":
    unittest.main()
