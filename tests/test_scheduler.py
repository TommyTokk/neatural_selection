from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

from configs.sim_config import SchedulerConfig, build_sim_config
from src.action import Action
from src.metabolism import FoodConsumption
from src.neat_brain import NeatBrain
from src.persistence import PersistenceManager, SimulationPaths
from src.world import (
    SimulationLagMetrics,
    World,
    _MouthExposureBuffer,
)


def neutral_action() -> Action:
    return Action(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class SchedulerConfigTest(unittest.TestCase):
    def test_defaults_expose_intended_frequencies(self) -> None:
        scheduler = SchedulerConfig()

        self.assertEqual(scheduler.decision_hz, 20)
        self.assertEqual(scheduler.biology_hz, 20)
        self.assertEqual(scheduler.statistics_hz, 5)

    def test_periods_must_divide_physics_frequency(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly divisible"):
            SchedulerConfig(decision_period_steps=7)

    def test_values_must_be_plain_positive_integers(self) -> None:
        for invalid in (True, 0, -1, 3.0):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    SchedulerConfig(max_steps_per_frame=invalid)

    def test_custom_physics_frequency_drives_derived_rates(self) -> None:
        scheduler = SchedulerConfig(
            physics_hz=120,
            decision_period_steps=4,
            biology_period_steps=6,
            statistics_period_steps=24,
        )

        self.assertEqual(scheduler.decision_hz, 30)
        self.assertEqual(scheduler.biology_hz, 20)
        self.assertEqual(scheduler.statistics_hz, 5)

    def test_world_timestep_is_derived_from_scheduler_frequency(self) -> None:
        config = build_sim_config()
        config.scheduler = SchedulerConfig(
            physics_hz=120,
            decision_period_steps=6,
            biology_period_steps=6,
            statistics_period_steps=24,
        )
        config.persistence.enable_telemetry = False
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        world = World(
            config,
            bootstrap=False,
            simulation_paths=SimulationPaths(Path(".").resolve()),
        )
        try:
            self.assertAlmostEqual(world.fixed_timestep, 1.0 / 120.0)
            self.assertAlmostEqual(World.FIXED_TIMESTEP, 1.0 / 60.0)
        finally:
            world.close()

    def test_backlog_must_cover_frame_work_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            SchedulerConfig(max_steps_per_frame=6, max_backlog_steps=5)


class SchedulerCadenceTest(unittest.TestCase):
    @staticmethod
    def clock_world() -> tuple[World, dict[str, list[float]]]:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world.simulation_speed = 1.0
        world.elapsed_time = 0.0
        world.fps = 0.0
        world.is_paused = False
        world._physics_accumulator = 0.0
        world._simulation_step = 0
        world.simulation_lag_metrics = SimulationLagMetrics()
        world.space = SimpleNamespace(step=lambda _dt: None)
        world.pheromones = SimpleNamespace(accumulate=lambda _dt: None)
        calls = {
            "motion": [],
            "biology": [],
            "survival": [],
            "flocking_fitness": [],
            "chronometers": [],
            "reproduction": [],
            "communication": [],
            "statistics": [],
            "contacts": [],
        }
        world._apply_creature_intents = lambda: calls["motion"].append(
            world.fixed_timestep
        )
        world._update_metabolism = lambda dt: calls["biology"].append(dt)
        world._update_fitness_survival = lambda dt: calls["survival"].append(dt)
        world._update_flocking_benchmark = lambda dt: calls[
            "flocking_fitness"
        ].append(dt)
        world._update_chronometers = lambda dt: calls["chronometers"].append(dt)
        world._update_reproduction = lambda dt: calls["reproduction"].append(dt)
        world._commit_communication_intents = lambda dt: calls[
            "communication"
        ].append(dt)
        world._refresh_stats = lambda: calls["statistics"].append(
            world.elapsed_time
        )
        world._accumulate_mouth_exposures = lambda dt: calls["contacts"].append(dt)
        for name in (
            "_update_speciation_threshold",
            "_apply_top_down_motion",
            "_spawn_foods",
            "_update_flocking_telemetry",
            "_update_persistence_timer",
        ):
            setattr(world, name, lambda *_args: None)
        for name in (
            "_settle_food_motion",
            "_limit_creature_motion",
            "_sync_carried_foods",
            "_apply_immediate_direct_damage",
            "_sample_selected_behavior",
            "_sample_selected_why",
            "_follow_selected_creature",
        ):
            setattr(world, name, lambda: None)
        return world, calls

    def test_sixty_step_boundaries(self) -> None:
        world, calls = self.clock_world()

        for _ in range(60):
            world.update(world.fixed_timestep)

        self.assertEqual(world._simulation_step, 60)
        self.assertEqual(len(calls["motion"]), 60)
        self.assertEqual(len(calls["contacts"]), 60)
        self.assertEqual(len(calls["biology"]), 20)
        self.assertEqual(len(calls["statistics"]), 5)
        for name in (
            "biology",
            "survival",
            "flocking_fitness",
            "chronometers",
            "reproduction",
        ):
            self.assertEqual(len(calls[name]), 20)
            self.assertAlmostEqual(sum(calls[name]), 1.0)
            self.assertTrue(
                all(abs(dt - 0.05) < 1e-12 for dt in calls[name])
            )
        self.assertEqual(len(calls["communication"]), 60)
        self.assertAlmostEqual(sum(calls["communication"]), 1.0)

    def test_compatibility_counter_is_read_only(self) -> None:
        world, _calls = self.clock_world()
        world._simulation_step = 17

        self.assertEqual(world.physics_step_count, 17)
        with self.assertRaises(AttributeError):
            world.physics_step_count = 18
        self.assertEqual(world._simulation_step, 17)

    def test_failed_fixed_step_does_not_increment_counter(self) -> None:
        world, _calls = self.clock_world()
        world._apply_creature_intents = lambda: (_ for _ in ()).throw(
            RuntimeError("injected")
        )

        with self.assertRaisesRegex(RuntimeError, "injected"):
            world.update(world.fixed_timestep)

        self.assertEqual(world._simulation_step, 0)

    def test_backlog_is_bounded_and_overflow_is_reported(self) -> None:
        world, _calls = self.clock_world()

        world.update(2.0)

        self.assertEqual(world._simulation_step, 5)
        self.assertAlmostEqual(
            world.simulation_lag_metrics.pending_seconds,
            55 * world.fixed_timestep,
        )
        self.assertAlmostEqual(
            world.simulation_lag_metrics.session_dropped_seconds,
            1.0,
        )
        self.assertAlmostEqual(
            world.simulation_lag_metrics.session_requested_seconds,
            2.0,
        )
        self.assertAlmostEqual(
            world.simulation_lag_metrics.session_completed_seconds,
            5 * world.fixed_timestep,
        )
        self.assertAlmostEqual(
            world.simulation_lag_metrics.effective_speed_multiplier,
            (5 * world.fixed_timestep) / 2.0,
        )
        self.assertAlmostEqual(
            world.simulation_lag_metrics.session_requested_seconds,
            world.simulation_lag_metrics.session_completed_seconds
            + world.simulation_lag_metrics.pending_seconds
            + world.simulation_lag_metrics.session_dropped_seconds,
        )
        self.assertTrue(world.simulation_lag_metrics.clamped_this_update)

    def test_biology_and_statistics_boundary_pipeline_order(self) -> None:
        world, _calls = self.clock_world()
        world._simulation_step = 11
        events: list[str] = []
        world._update_speciation_threshold = lambda _dt: events.append(
            "speciation"
        )
        world._apply_creature_intents = lambda: events.append("intent")
        world._commit_communication_intents = lambda _dt: events.append(
            "communication"
        )
        world.space.step = lambda _dt: events.append("physics")
        for name, label in (
            ("_settle_food_motion", "settle"),
            ("_limit_creature_motion", "limits"),
            ("_sync_carried_foods", "carry_sync"),
            ("_apply_immediate_direct_damage", "direct_damage"),
            ("_sample_selected_behavior", "observer"),
            ("_sample_selected_why", "counterfactual"),
        ):
            setattr(world, name, lambda label=label: events.append(label))
        for name, label in (
            ("_apply_top_down_motion", "post_physics_motion"),
            ("_accumulate_mouth_exposures", "contacts"),
            ("_update_fitness_survival", "survival_age"),
            ("_update_flocking_benchmark", "flocking_fitness"),
            ("_update_chronometers", "chronometers"),
            ("_update_reproduction", "reproduction_cadence"),
            ("_update_metabolism", "resource_biology"),
            ("_spawn_foods", "food_spawn"),
            ("_update_flocking_telemetry", "telemetry"),
        ):
            setattr(
                world,
                name,
                lambda _dt, label=label: events.append(label),
            )
        world.pheromones.accumulate = lambda _dt: events.append("pheromones")
        world._refresh_stats = lambda: events.append("statistics")

        world._run_fixed_step()

        self.assertEqual(
            events,
            [
                "speciation",
                "intent",
                "communication",
                "physics",
                "settle",
                "post_physics_motion",
                "limits",
                "carry_sync",
                "contacts",
                "direct_damage",
                "survival_age",
                "flocking_fitness",
                "chronometers",
                "reproduction_cadence",
                "resource_biology",
                "pheromones",
                "food_spawn",
                "telemetry",
                "observer",
                "counterfactual",
                "statistics",
            ],
        )

    def test_pre_biology_direct_death_removes_creature_from_biology(self) -> None:
        world, _calls = self.clock_world()
        world._simulation_step = 2
        creature = SimpleNamespace(
            creature_id=1,
            life=0.1,
            pending_direct_life_damage=0.2,
        )
        world.creatures = [creature]
        world.selected_creature_id = None
        world._immediate_dead_buffer = []
        world._apply_immediate_direct_damage = (
            World._apply_immediate_direct_damage.__get__(world)
        )
        world._remove_dead_creatures = lambda candidates, **_kwargs: [
            world.creatures.remove(candidate)
            for candidate in tuple(candidates)
            if candidate in world.creatures and candidate.life <= 0.0
        ]
        populations: list[tuple[str, int]] = []
        for name in (
            "_update_fitness_survival",
            "_update_flocking_benchmark",
            "_update_chronometers",
            "_update_reproduction",
            "_update_metabolism",
        ):
            setattr(
                world,
                name,
                lambda _dt, name=name: populations.append(
                    (name, len(world.creatures))
                ),
            )

        world._run_fixed_step()

        self.assertEqual(world.creatures, [])
        self.assertTrue(populations)
        self.assertTrue(all(population == 0 for _name, population in populations))

    def test_post_biology_death_precedes_later_timebases(self) -> None:
        world, _calls = self.clock_world()
        world._simulation_step = 2
        creature = SimpleNamespace(creature_id=1, life=1.0)
        world.creatures = [creature]
        world._update_metabolism = lambda _dt: world.creatures.clear()
        later_populations: list[int] = []
        world.pheromones.accumulate = lambda _dt: later_populations.append(
            len(world.creatures)
        )
        world._spawn_foods = lambda _dt: later_populations.append(
            len(world.creatures)
        )
        world._update_flocking_telemetry = lambda _dt: later_populations.append(
            len(world.creatures)
        )

        world._run_fixed_step()

        self.assertEqual(later_populations, [0, 0, 0])


class DecisionPhaseTest(unittest.TestCase):
    def test_phase_depends_only_on_persisted_integer_id(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        ids = [8, 3, 14, 1]
        expected = {creature_id: creature_id % 3 for creature_id in ids}

        first = {creature_id: world._decision_phase(creature_id) for creature_id in ids}
        reordered = {
            creature_id: world._decision_phase(creature_id)
            for creature_id in reversed(ids)
        }
        after_removal = {
            creature_id: world._decision_phase(creature_id)
            for creature_id in (14, 8, 1)
        }

        self.assertEqual(first, expected)
        self.assertEqual(reordered, expected)
        self.assertEqual(
            after_removal,
            {creature_id: expected[creature_id] for creature_id in (14, 8, 1)},
        )
        with self.assertRaisesRegex(TypeError, "stable integer"):
            world._decision_phase("8")

    def test_each_stable_id_thinks_twenty_times(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world._simulation_step = 0
        world.selected_creature_id = None
        world._flocking_capture_due_this_step = False
        world.pheromones = None
        world._chronometers = {}
        world._last_sensor_snapshots = {}
        world._last_actions = {}
        world._effective_actions = {}
        world._cached_social_intentions = {}
        world._motion_commands = {}
        world.creatures = [
            SimpleNamespace(creature_id=index, energy=1.0, last_action=None)
            for index in range(1, 7)
        ]
        counts = {creature.creature_id: 0 for creature in world.creatures}
        snapshot = object()
        world._sensor_snapshot_for = lambda *_args, **_kwargs: snapshot
        world._adapt_creature_biome_memory = lambda *_args: None
        world._apply_carry_intent = lambda *_args: None
        world._apply_action = lambda *_args, **_kwargs: None
        world.neat_controller = SimpleNamespace(
            decide=lambda creature_id, _snapshot, **_kwargs: (
                counts.__setitem__(creature_id, counts[creature_id] + 1)
                or neutral_action()
            )
        )

        for step in range(60):
            world._simulation_step = step
            world._apply_creature_intents_with_spatial_cache()

        self.assertEqual(set(counts.values()), {20})
        self.assertEqual(
            [world._decision_phase(index) for index in range(1, 7)],
            [1, 2, 0, 1, 2, 0],
        )

    def test_current_step_executes_the_matching_phase(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        expected = ({3, 6}, {1, 4}, {2, 5})

        for step in range(6):
            scheduled = {
                creature_id
                for creature_id in range(1, 7)
                if step % world.config.scheduler.decision_period_steps
                == world._decision_phase(creature_id)
            }
            self.assertEqual(
                scheduled,
                expected[step % 3],
            )

    def test_unscheduled_step_reuses_action_without_replaying_edges(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world._simulation_step = 0
        world.selected_creature_id = None
        world._flocking_capture_due_this_step = False
        world.pheromones = None
        creature = SimpleNamespace(creature_id=1, energy=1.0)
        world.creatures = [creature]
        action = Action(
            0.2,
            0.1,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        )
        world._chronometers = {1: 4.0}
        world._last_sensor_snapshots = {1: object()}
        world._last_actions = {1: action}
        world._effective_actions = {1: action}
        world._cached_social_intentions = {}
        world._motion_commands = {}
        carry_calls: list[Action] = []
        applied: list[Action] = []
        world.neat_controller = SimpleNamespace(
            decide=lambda *_args, **_kwargs: self.fail(
                "unscheduled creature must not activate NEAT"
            )
        )
        world._apply_carry_intent = lambda _creature, active: carry_calls.append(
            active
        )
        world._apply_action = lambda _creature, active, *_args, **_kwargs: (
            applied.append(active)
        )

        world._apply_creature_intents_with_spatial_cache()

        self.assertEqual(world._chronometers[1], 4.0)
        self.assertEqual(carry_calls, [])
        self.assertEqual(len(applied), 1)
        self.assertIs(applied[0], action)
        self.assertIs(world._last_actions[1], action)

    def test_fresh_decision_executes_reset_and_carry_edges_once(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world._simulation_step = 2
        world.selected_creature_id = None
        world._flocking_capture_due_this_step = False
        world.pheromones = None
        creature = SimpleNamespace(creature_id=1, energy=1.0, last_action=None)
        world.creatures = [creature]
        world._chronometers = {1: 4.0}
        world._last_sensor_snapshots = {}
        world._last_actions = {}
        world._effective_actions = {}
        world._cached_social_intentions = {}
        world._motion_commands = {}
        action = neutral_action()
        action.reset_chronometer = 1.0
        action.want_grab = 1.0
        action.want_release = 1.0
        world._sensor_snapshot_for = lambda *_args, **_kwargs: object()
        world._adapt_creature_biome_memory = lambda *_args: None
        world.neat_controller = SimpleNamespace(
            decide=lambda *_args, **_kwargs: action
        )
        edge_calls: list[Action] = []
        world._apply_carry_intent = (
            lambda _creature, active: edge_calls.append(active)
        )
        world._apply_action = lambda *_args, **_kwargs: None

        for step in (2, 3, 4):
            world._simulation_step = step
            world._apply_creature_intents_with_spatial_cache()

        self.assertEqual(world._chronometers[1], 0.0)
        self.assertEqual(edge_calls, [action])
        self.assertIs(world._last_actions[1], action)

    def test_unscheduled_depleted_action_reuses_gated_view(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world._simulation_step = 0
        world.selected_creature_id = None
        world._flocking_capture_due_this_step = False
        world.pheromones = None
        creature = SimpleNamespace(creature_id=1, energy=0.0)
        world.creatures = [creature]
        action = Action(0.2, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0)
        world._chronometers = {1: 4.0}
        world._last_sensor_snapshots = {1: object()}
        world._last_actions = {1: action}
        world._effective_actions = {1: action}
        world._cached_social_intentions = {}
        world._motion_commands = {}
        applied: list[Action] = []
        world.neat_controller = SimpleNamespace(
            decide=lambda *_args, **_kwargs: self.fail(
                "unscheduled creature must not activate NEAT"
            )
        )
        world._apply_carry_intent = lambda *_args: None
        world._apply_action = lambda _creature, active, *_args, **_kwargs: (
            applied.append(active)
        )

        world._apply_creature_intents_with_spatial_cache()
        first_gated = world._effective_actions[1]
        world._apply_creature_intents_with_spatial_cache()

        self.assertIsNot(first_gated, action)
        self.assertIs(world._effective_actions[1], first_gated)
        self.assertIs(applied[0], first_gated)
        self.assertIs(applied[1], first_gated)

    def test_selected_waiting_fallback_is_read_only(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world._simulation_step = 0
        world.selected_creature_id = 1
        world._flocking_capture_due_this_step = False
        world.pheromones = None
        creature = SimpleNamespace(
            creature_id=1,
            energy=1.0,
            last_action=None,
        )
        world.creatures = [creature]
        world._chronometers = {1: 4.0}
        world._last_sensor_snapshots = {}
        world._last_actions = {}
        world._effective_actions = {}
        world._cached_social_intentions = {}
        world._motion_commands = {}
        world._initialize_creature_runtime_state(creature)
        neutral = world._last_actions[1]
        social = world._cached_social_intentions[1]
        applied: list[tuple[Action, object, bool]] = []
        world.neat_controller = SimpleNamespace(
            decide=lambda *_args, **_kwargs: self.fail(
                "waiting fallback must not activate NEAT"
            ),
            capture_input_snapshot=lambda *_args: self.fail(
                "waiting fallback must not publish sensor inputs"
            ),
        )
        world._apply_carry_intent = lambda *_args: self.fail(
            "waiting fallback must not replay an edge action"
        )
        world._apply_action = (
            lambda _creature, action, snapshot, **kwargs: applied.append(
                (action, snapshot, kwargs["refresh_intention"])
            )
        )

        world._apply_creature_intents_with_spatial_cache()

        self.assertEqual(world._simulation_step, 0)
        self.assertEqual(world._chronometers[1], 4.0)
        self.assertEqual(world._last_sensor_snapshots, {})
        self.assertIs(world._last_actions[1], neutral)
        self.assertIs(world._cached_social_intentions[1], social)
        self.assertEqual(applied, [(neutral, None, False)])


class SchedulerSmoothingTest(unittest.TestCase):
    def test_herding_filter_matches_equal_elapsed_time(self) -> None:
        brain = NeatBrain(
            genome_id=1,
            genome=object(),
            network=object(),
            herding_decay_rate=0.15,
        )

        def response(durations: list[float]) -> float:
            value = 0.0
            for duration in durations:
                alpha = brain._elapsed_herding_alpha(duration)
                value += (1.0 - value) * alpha
            return value

        self.assertAlmostEqual(
            response([1.0 / 30.0] * 3),
            response([1.0 / 20.0] * 2),
        )

    def test_physics_filter_matches_equal_elapsed_time(self) -> None:
        world = object.__new__(World)
        reference = world._physics_rate_alpha(0.8, 1.0 / 60.0)
        half_step = world._physics_rate_alpha(0.8, 1.0 / 120.0)

        self.assertAlmostEqual(reference, 0.8)
        self.assertAlmostEqual(
            reference,
            1.0 - (1.0 - half_step) ** 2,
        )

    def test_turn_response_matches_equal_elapsed_time_across_physics_rates(
        self,
    ) -> None:
        def response(physics_hz: int) -> tuple[float, float]:
            world = object.__new__(World)
            world.config = build_sim_config()
            world.fixed_timestep = 1.0 / physics_hz
            body = SimpleNamespace(
                angular_velocity=0.0,
                torque=0.0,
                angle=0.0,
            )
            creature = SimpleNamespace(body=body)
            for _ in range(physics_hz // 5):
                world._apply_turn_control(
                    creature,
                    1.0,
                    max_angular_speed=1.0,
                )
                body.angle += body.angular_velocity * world.fixed_timestep
            return body.angular_velocity, body.angle

        sixty_velocity, sixty_angle = response(60)
        one_twenty_velocity, one_twenty_angle = response(120)

        self.assertAlmostEqual(sixty_velocity, one_twenty_velocity)
        self.assertAlmostEqual(sixty_angle, one_twenty_angle, delta=0.004)


class SchedulerBiologyRateTest(unittest.TestCase):
    @staticmethod
    def _real_world(action: Action) -> World:
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.persistence.quick_save_interval_seconds = 0.0
        config.persistence.archive_save_interval_seconds = 0.0
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        config.population.initial_creatures = 1
        config.food.initial_food_items = 0
        world = World(
            config,
            simulation_paths=SimulationPaths(Path(".").resolve()),
        )
        world.neat_controller.decide = lambda *_args, **_kwargs: action
        world._spawn_foods = lambda _dt: None
        return world

    def test_reproduction_cadence_becomes_due_once_per_second(self) -> None:
        world = object.__new__(World)
        world._reproduction_accumulator = 0.0
        due_count = 0

        for _ in range(20):
            world._update_reproduction(0.05)
            due_count += int(world._reproduction_due_this_step)

        self.assertEqual(due_count, 1)
        self.assertAlmostEqual(world._reproduction_accumulator, 0.0)

    def test_cached_reproduction_level_is_consumed_once_per_due_cadence(
        self,
    ) -> None:
        world = object.__new__(World)
        parent = SimpleNamespace(creature_id=1)
        action = neutral_action()
        action.want_reproduce = 1.0
        world.creatures = [parent]
        world.fitness = {parent.creature_id: object()}
        world.rt_neat = SimpleNamespace(
            eligible_parent_ids=[parent.creature_id]
        )
        world._last_actions = {parent.creature_id: action}
        world._effective_actions = {parent.creature_id: action}
        world._reproduction_due_this_step = True
        resource_checks: list[None] = []
        world._has_reproduction_resources = lambda: (
            resource_checks.append(None) or True
        )
        world._reproduction_cost_for = lambda _parent: 0.25

        first = world._prepare_reproduction_requests()
        second = world._prepare_reproduction_requests()

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].parent, parent)
        self.assertEqual(first[0].reserved_energy_cost, 0.25)
        self.assertEqual(second, [])
        self.assertEqual(resource_checks, [None])

    def test_nursing_rate_integrates_full_biology_duration(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        donor = SimpleNamespace(creature_id=1)
        target = SimpleNamespace(creature_id=2)
        action = neutral_action()
        action.want_nurse = 1.0
        world.creatures = [donor]
        world._last_actions = {1: action}
        world._effective_actions = {1: action}
        world._nearest_nursable_infant_for = lambda _donor: target

        requested = sum(
            world._prepare_nursing_requests(0.05)[0].requested_transfer
            for _ in range(20)
        )

        self.assertAlmostEqual(
            requested,
            world.config.population.nursing_energy_transfer_rate,
        )

    def test_real_one_second_integrates_digestion_communication_and_fitness(
        self,
    ) -> None:
        quiet = self._real_world(neutral_action())
        emitting_action = neutral_action()
        emitting_action.emit_sound = 1.0
        emitting_action.emit_trail_pheromone = 1.0
        emitting_action.emit_alarm_pheromone = 1.0
        emitting = self._real_world(emitting_action)
        try:
            for world in (quiet, emitting):
                creature = world.creatures[0]
                creature.energy = 0.4
                creature.stomach_energy = 0.2
                creature.stomach_difficulty_load = 0.2
            for _ in range(60):
                quiet.update(quiet.fixed_timestep)
                emitting.update(emitting.fixed_timestep)

            quiet_creature = quiet.creatures[0]
            emitting_creature = emitting.creatures[0]
            quiet_fitness = quiet.fitness[quiet_creature.creature_id]
            emitting_fitness = emitting.fitness[emitting_creature.creature_id]
            communication_cost = (
                emitting.config.communication.acoustic_energy_cost_per_second
                + 2.0
                * emitting.config.communication.pheromone_energy_cost_per_second
            )

            self.assertAlmostEqual(quiet_fitness.age_seconds, 1.0)
            self.assertAlmostEqual(
                quiet._chronometers[quiet_creature.creature_id],
                1.0,
            )
            self.assertLess(quiet_creature.stomach_energy, 0.2)
            self.assertAlmostEqual(
                quiet_creature.stomach_energy,
                emitting_creature.stomach_energy,
            )
            self.assertAlmostEqual(
                quiet_creature.energy - emitting_creature.energy,
                communication_cost,
            )
            self.assertAlmostEqual(
                emitting_fitness.trait_energy_cost
                - quiet_fitness.trait_energy_cost,
                communication_cost,
            )
        finally:
            quiet.close()
            emitting.close()

    def test_real_one_second_applies_unfunded_upkeep_to_life(self) -> None:
        world = self._real_world(neutral_action())
        try:
            creature = world.creatures[0]
            creature.energy = 0.0
            creature.stomach_energy = 0.0
            creature.stomach_difficulty_load = 0.0

            for _ in range(60):
                world.update(world.fixed_timestep)

            self.assertEqual(creature.energy, 0.0)
            self.assertLess(creature.life, world.config.metabolism.max_life)
            self.assertGreater(creature.life, 0.0)
            self.assertAlmostEqual(
                world.fitness[creature.creature_id].age_seconds,
                1.0,
            )
        finally:
            world.close()

    def test_real_starvation_death_precedes_every_downstream_timebase(
        self,
    ) -> None:
        world = self._real_world(neutral_action())
        try:
            victim = world.creatures[0]
            victim_id = victim.creature_id
            victim.energy = 0.0
            victim.stomach_energy = 0.0
            victim.stomach_difficulty_load = 0.0
            victim.life = 1e-12
            world._simulation_step = 2
            world._recover_extinct_population = lambda: None
            observed: list[tuple[str, tuple[int, ...]]] = []

            def record(label: str) -> None:
                observed.append(
                    (
                        label,
                        tuple(
                            creature.creature_id
                            for creature in world.creatures
                        ),
                    )
                )

            world.pheromones.accumulate = lambda _dt: record("pheromones")
            world._spawn_foods = lambda _dt: record("food_spawn")
            world._update_flocking_telemetry = lambda _dt: record(
                "telemetry"
            )
            world._sample_selected_behavior = lambda: record("observer")
            world._sample_selected_why = lambda: record("counterfactual")

            world._run_fixed_step()

            self.assertNotIn(victim, world.creatures)
            self.assertEqual(
                [label for label, _ids in observed],
                [
                    "pheromones",
                    "food_spawn",
                    "telemetry",
                    "observer",
                    "counterfactual",
                ],
            )
            self.assertTrue(
                all(victim_id not in creature_ids for _label, creature_ids in observed)
            )
        finally:
            world.close()


class SchedulerPersistenceTest(unittest.TestCase):
    def test_in_memory_restore_keeps_phase_state_and_resets_session_debt(
        self,
    ) -> None:
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.persistence.quick_save_interval_seconds = 0.0
        config.persistence.archive_save_interval_seconds = 0.0
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        config.population.initial_creatures = 2
        config.food.initial_food_items = 2
        paths = SimulationPaths(Path(".").resolve())
        world = World(config, simulation_paths=paths)
        restored = None
        try:
            creature = world.creatures[0]
            food = world.foods[0]
            world._simulation_step = 7
            world._physics_accumulator = 0.5
            world._speciation_adjustment_accumulator = 1.75
            world.simulation_lag_metrics.session_requested_seconds = 1.25
            world.simulation_lag_metrics.session_completed_seconds = 0.75
            world.simulation_lag_metrics.session_dropped_seconds = 0.25
            world._behavior_next_sample_time = 1.3
            world._why_next_probe_time = 1.4
            world._behavior_selection_generation = 9
            world._behavior_subject_generation_counter = 12
            world._behavior_food_consumption_count = 3
            world._behavior_food_consumed_energy_total = 0.75
            world._behavior_consumption_totals = {
                creature.creature_id: (3, 0.75)
            }
            world._behavior_active_subjects = {creature.creature_id: 9}
            world._behavior_cohort_dirty = False
            world._mouth_exposures.append(
                6,
                creature.creature_id,
                food.id,
                world.fixed_timestep,
            )
            world._last_actions[creature.creature_id].accelerate = 0.8
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )

            restored = PersistenceManager._restore_world(
                state,
                config,
                paths,
            )

            self.assertEqual(restored._simulation_step, 7)
            self.assertEqual(restored.physics_step_count, 7)
            self.assertEqual(
                restored._speciation_adjustment_accumulator,
                1.75,
            )
            self.assertEqual(restored._physics_accumulator, 0.0)
            self.assertEqual(
                restored.simulation_lag_metrics,
                SimulationLagMetrics(),
            )
            self.assertEqual(
                restored._mouth_exposures.state(),
                ((6, creature.creature_id, food.id, world.fixed_timestep),),
            )
            self.assertEqual(
                restored._last_actions[creature.creature_id].accelerate,
                0.8,
            )
            self.assertIs(
                restored._last_actions[creature.creature_id],
                restored._effective_actions[creature.creature_id],
            )
            self.assertEqual(restored._behavior_next_sample_time, 1.3)
            self.assertEqual(restored._why_next_probe_time, 1.4)
            self.assertEqual(restored._behavior_selection_generation, 9)
            self.assertEqual(restored._behavior_subject_generation_counter, 12)
            self.assertEqual(restored._behavior_food_consumption_count, 3)
            self.assertEqual(restored._behavior_food_consumed_energy_total, 0.75)
            self.assertEqual(
                restored._behavior_consumption_totals,
                {creature.creature_id: (3, 0.75)},
            )
            self.assertEqual(
                restored._behavior_active_subjects,
                {creature.creature_id: 9},
            )
            self.assertFalse(restored._behavior_cohort_dirty)
        finally:
            if restored is not None:
                restored.close()
            world.close()

    def test_legacy_restore_derives_step_and_ignores_saved_frame_debt(
        self,
    ) -> None:
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.persistence.quick_save_interval_seconds = 0.0
        config.persistence.archive_save_interval_seconds = 0.0
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        config.population.initial_creatures = 1
        config.food.initial_food_items = 1
        paths = SimulationPaths(Path(".").resolve())
        world = World(config, simulation_paths=paths)
        restored = None
        try:
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            expected_step = 13
            state["version"] = 18
            state["sim_time"] = expected_step * world.fixed_timestep
            state["world"].pop("simulation_step", None)
            state["world"].pop("mouth_exposures", None)
            state["world"]["physics_accumulator"] = 0.75

            restored = PersistenceManager._restore_world(
                state,
                config,
                paths,
            )

            self.assertEqual(restored._simulation_step, expected_step)
            self.assertEqual(restored.physics_step_count, expected_step)
            self.assertEqual(restored._mouth_exposures.state(), ())
            self.assertEqual(restored._physics_accumulator, 0.0)
            self.assertEqual(
                restored.simulation_lag_metrics,
                SimulationLagMetrics(),
            )
        finally:
            if restored is not None:
                restored.close()
            world.close()


class DeterministicSchedulerIntegrationTest(unittest.TestCase):
    @staticmethod
    def _world() -> World:
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.persistence.quick_save_interval_seconds = 0.0
        config.persistence.archive_save_interval_seconds = 0.0
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        config.population.initial_creatures = 4
        config.food.initial_food_items = 8
        return World(
            config,
            simulation_paths=SimulationPaths(Path(".").resolve()),
        )

    @staticmethod
    def _state(world: World) -> tuple[object, ...]:
        creatures = tuple(
            (
                creature.creature_id,
                creature.position,
                creature.heading,
                tuple(creature.body.velocity),
                creature.energy,
                creature.life,
                creature.stomach_energy,
                world.fitness[creature.creature_id],
                world._last_actions[creature.creature_id],
                world._decision_phase(creature.creature_id),
            )
            for creature in world.creatures
        )
        foods = tuple(
            (food.id, food.position, food.energy_value)
            for food in world.foods
        )
        return (
            world._simulation_step,
            world.elapsed_time,
            creatures,
            foods,
            world.rt_neat.stats.births,
            world.rt_neat.stats.deaths,
            world._mouth_exposures.state(),
            world.rng.getstate(),
            world.neat_controller.evolution_random_state(),
        )

    def test_duplicate_sixty_step_runs_match_authoritative_state(self) -> None:
        first = self._world()
        second = self._world()
        try:
            for _ in range(60):
                first.update(first.fixed_timestep)
                second.update(second.fixed_timestep)

            self.assertEqual(self._state(first), self._state(second))
        finally:
            first.close()
            second.close()

    def test_forced_birth_death_and_exposure_are_deterministic(self) -> None:
        first = self._world()
        second = self._world()
        try:
            consumed_food_id = first.foods[0].id
            initial_food_energy = first.foods[0].energy_value
            consumer_id = first.creatures[1].creature_id
            for world in (first, second):
                reproduction = neutral_action()
                reproduction.want_reproduce = 1.0
                parent = world.creatures[0]
                world.rt_neat.eligible_parent_ids = [parent.creature_id]
                world._last_actions[parent.creature_id] = reproduction
                world._effective_actions[parent.creature_id] = reproduction
                world._has_reproduction_resources = lambda: True
                self.assertTrue(world._try_reproduce())

                victim = world.creatures[0]
                victim.pending_direct_life_damage = victim.life + 1.0
                consumer = next(
                    creature
                    for creature in world.creatures
                    if creature.creature_id == consumer_id
                )
                consumer.stomach_energy = 0.0
                food = next(food for food in world.foods if food.id == consumed_food_id)
                world._mouth_exposures.append(
                    0,
                    consumer.creature_id,
                    food.id,
                    world.fixed_timestep,
                )

            for _ in range(3):
                first.update(first.fixed_timestep)
                second.update(second.fixed_timestep)

            self.assertEqual(self._state(first), self._state(second))
            self.assertEqual(first.rt_neat.stats.births, 1)
            self.assertEqual(first.rt_neat.stats.deaths, 1)
            first_consumer = next(
                creature
                for creature in first.creatures
                if creature.creature_id == consumer_id
            )
            self.assertGreater(
                first.fitness[first_consumer.creature_id].energy_gained,
                0.0,
            )
            first_food = next(
                food for food in first.foods if food.id == consumed_food_id
            )
            self.assertLess(first_food.energy_value, initial_food_energy)
        finally:
            first.close()
            second.close()


@dataclass(slots=True)
class FakeExposureCreature:
    creature_id: int
    life: float = 1.0
    stomach_energy: float = 0.0
    stomach_capacity: float = 1.0


@dataclass(slots=True)
class FakeExposureFood:
    id: int
    energy_value: float = 1.0


class FakeExposureMetabolism:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float]] = []

    def eat(self, creature, food, duration: float) -> FoodConsumption:
        self.calls.append((creature.creature_id, food.id, duration))
        amount = min(
            food.energy_value,
            duration,
            max(0.0, creature.stomach_capacity - creature.stomach_energy),
        )
        food.energy_value -= amount
        creature.stomach_energy += amount
        return FoodConsumption(
            creature_id=creature.creature_id,
            food=food,
            energy_swallowed=amount,
            depleted=food.energy_value <= 0.0,
        )


class MouthExposureBufferTest(unittest.TestCase):
    def test_interrupted_resolution_retains_active_records(self) -> None:
        world = object.__new__(World)
        creature = FakeExposureCreature(1)
        food = FakeExposureFood(10)
        world.creatures = [creature]
        world.foods = [food]
        world.metabolism = SimpleNamespace(
            eat=lambda *_args: (_ for _ in ()).throw(RuntimeError("injected"))
        )
        world._mouth_exposures = _MouthExposureBuffer()
        world._mouth_exposures.append(0, creature.creature_id, food.id, 0.1)

        with self.assertRaisesRegex(RuntimeError, "injected"):
            world._resolve_accumulated_mouth_exposures()

        self.assertEqual(
            world._mouth_exposures.state(),
            ((0, 1, 10, 0.1),),
        )
        self.assertEqual(creature.stomach_energy, 0.0)
        self.assertEqual(food.energy_value, 1.0)

    def test_partial_resolution_rolls_back_before_retry(self) -> None:
        world = object.__new__(World)
        creature = FakeExposureCreature(1)
        food_a = FakeExposureFood(10)
        food_b = FakeExposureFood(20)
        world.creatures = [creature]
        world.foods = [food_a, food_b]
        successful = FakeExposureMetabolism()
        call_count = 0

        def fail_second(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("second claim failed")
            return successful.eat(*args)

        world.metabolism = SimpleNamespace(eat=fail_second)
        world._mouth_exposures = _MouthExposureBuffer()
        world._mouth_exposures.append(0, 1, 10, 0.1)
        world._mouth_exposures.append(1, 1, 20, 0.1)

        with self.assertRaisesRegex(RuntimeError, "second claim failed"):
            world._resolve_accumulated_mouth_exposures()

        self.assertEqual(world._mouth_exposures.count, 2)
        self.assertEqual(creature.stomach_energy, 0.0)
        self.assertEqual([food_a.energy_value, food_b.energy_value], [1.0, 1.0])

        world.metabolism = FakeExposureMetabolism()
        world._resolve_accumulated_mouth_exposures()

        self.assertAlmostEqual(creature.stomach_energy, 0.2)
        self.assertEqual([food_a.energy_value, food_b.energy_value], [0.9, 0.9])

    def test_depleted_food_cannot_satisfy_later_claims(self) -> None:
        world = object.__new__(World)
        first = FakeExposureCreature(1)
        second = FakeExposureCreature(2)
        food = FakeExposureFood(10, energy_value=0.05)
        world.creatures = [first, second]
        world.foods = [food]
        world.metabolism = FakeExposureMetabolism()
        world._mouth_exposures = _MouthExposureBuffer()
        world._mouth_exposures.append(0, 1, 10, 0.1)
        world._mouth_exposures.append(1, 2, 10, 0.1)

        report = world._resolve_accumulated_mouth_exposures()

        self.assertEqual(world.metabolism.calls, [(1, 10, 0.1)])
        self.assertEqual(report.depleted_foods, [food])
        self.assertEqual([item.creature_id for item in report.food_consumptions], [1])
        self.assertEqual(food.energy_value, 0.0)

    def test_resolution_is_chronological_and_discards_missing_ids(self) -> None:
        world = object.__new__(World)
        first = FakeExposureCreature(1)
        second = FakeExposureCreature(2)
        food_a = FakeExposureFood(10)
        food_b = FakeExposureFood(20)
        world.creatures = [second, first]
        world.foods = [food_b, food_a]
        world.metabolism = FakeExposureMetabolism()
        world._mouth_exposures = _MouthExposureBuffer()
        world._mouth_exposures.append(2, 2, 20, 0.1)
        world._mouth_exposures.append(1, 2, 10, 0.1)
        world._mouth_exposures.append(1, 1, 10, 0.1)
        world._mouth_exposures.append(1, 2, 20, 0.1)
        world._mouth_exposures.append(0, 99, 10, 0.1)

        report = world._resolve_accumulated_mouth_exposures()

        self.assertEqual(
            world.metabolism.calls,
            [(1, 10, 0.1), (2, 20, 0.1), (2, 20, 0.1)],
        )
        self.assertEqual(len(report.food_consumptions), 3)
        self.assertAlmostEqual(first.stomach_energy, 0.1)
        self.assertAlmostEqual(second.stomach_energy, 0.2)
        self.assertEqual(world._mouth_exposures.count, 0)

    def test_invalid_first_claim_does_not_block_contested_food(self) -> None:
        world = object.__new__(World)
        full = FakeExposureCreature(1, stomach_energy=1.0)
        hungry = FakeExposureCreature(2)
        food = FakeExposureFood(10)
        world.creatures = [hungry, full]
        world.foods = [food]
        world.metabolism = FakeExposureMetabolism()
        world._mouth_exposures = _MouthExposureBuffer()
        world._mouth_exposures.append(0, full.creature_id, food.id, 0.1)
        world._mouth_exposures.append(0, hungry.creature_id, food.id, 0.1)

        report = world._resolve_accumulated_mouth_exposures()

        self.assertEqual(
            world.metabolism.calls,
            [(1, 10, 0.1), (2, 10, 0.1)],
        )
        self.assertEqual(
            [item.creature_id for item in report.food_consumptions],
            [2],
        )
        self.assertAlmostEqual(hungry.stomach_energy, 0.1)

    def test_one_creature_and_one_food_can_win_only_once_per_step(self) -> None:
        world = object.__new__(World)
        first = FakeExposureCreature(1)
        second = FakeExposureCreature(2)
        food_a = FakeExposureFood(10)
        food_b = FakeExposureFood(20)
        world.creatures = [first, second]
        world.foods = [food_a, food_b]
        world.metabolism = FakeExposureMetabolism()
        world._mouth_exposures = _MouthExposureBuffer()
        for creature in (second, first):
            for food in (food_b, food_a):
                world._mouth_exposures.append(
                    0,
                    creature.creature_id,
                    food.id,
                    0.1,
                )

        world._resolve_accumulated_mouth_exposures()

        self.assertEqual(
            world.metabolism.calls,
            [(1, 10, 0.1), (2, 20, 0.1)],
        )

    def test_boundary_exposure_is_visible_to_resource_evaluation(self) -> None:
        world = object.__new__(World)
        creature = FakeExposureCreature(1)
        food = FakeExposureFood(10)
        metabolism = FakeExposureMetabolism()
        metabolism.evaluate_candidate = lambda *_args, **_kwargs: None
        world.creatures = [creature]
        world.foods = [food]
        world.metabolism = metabolism
        world._mouth_exposures = _MouthExposureBuffer()
        world._mouth_exposures.append(2, creature.creature_id, food.id, 0.1)
        observed_stomach: list[float] = []

        def stop_after_exposure(_delta_time: float):
            observed_stomach.append(creature.stomach_energy)
            raise RuntimeError("stop after resource evaluation begins")

        world._resolve_resource_transactions = stop_after_exposure

        with self.assertRaisesRegex(RuntimeError, "resource evaluation"):
            world._update_metabolism(0.15)

        self.assertEqual(observed_stomach, [0.1])
        self.assertEqual(world._mouth_exposures.count, 1)
        self.assertEqual(creature.stomach_energy, 0.0)
        self.assertEqual(food.energy_value, 1.0)


if __name__ == "__main__":
    unittest.main()
