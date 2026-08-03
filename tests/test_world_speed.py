from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest

for optional_module in ("arcade", "neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from src.world import SimulationLagMetrics, World
from configs.sim_config import build_sim_config
from src.persistence import SimulationPaths


class WorldSimulationSpeedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = object.__new__(World)
        self.world.simulation_speed = 1.0

    def test_set_simulation_speed_clamps_to_five_x(self) -> None:
        self.world.set_simulation_speed(999.0)

        self.assertEqual(self.world.simulation_speed, 5.0)

    def test_repeated_speed_up_stops_at_five_x(self) -> None:
        for _ in range(100):
            self.world.increase_simulation_speed()

        self.assertEqual(self.world.simulation_speed, 5.0)

    def test_reset_returns_to_one_x(self) -> None:
        self.world.set_simulation_speed(5.0)

        self.world.reset_simulation_speed()

        self.assertEqual(self.world.simulation_speed, 1.0)

    def test_speed_rounds_to_quarter_step(self) -> None:
        self.world.set_simulation_speed(1.37)

        self.assertEqual(self.world.simulation_speed, 1.25)

    def _clock_world(self, speed: float) -> World:
        world = object.__new__(World)
        world.simulation_speed = speed
        world.config = build_sim_config()
        world.fixed_timestep = 1.0 / world.config.scheduler.physics_hz
        world.elapsed_time = 0.0
        world.fps = 0.0
        world.is_paused = False
        world._physics_accumulator = 0.0
        world._simulation_step = 0
        world.simulation_lag_metrics = SimulationLagMetrics()
        world.space = SimpleNamespace(step=lambda delta: None)
        world.pheromones = SimpleNamespace(accumulate=lambda delta: None)
        world.timebase_calls = {
            name: []
            for name in (
                "speciation",
                "food",
                "reproduction",
                "telemetry",
                "persistence",
            )
        }
        timed_methods = {
            "_update_speciation_threshold": "speciation",
            "_spawn_foods": "food",
            "_update_reproduction": "reproduction",
            "_update_flocking_telemetry": "telemetry",
            "_update_persistence_timer": "persistence",
        }
        for method_name, call_name in timed_methods.items():
            setattr(
                world,
                method_name,
                lambda delta, name=call_name: world.timebase_calls[name].append(
                    delta
                ),
            )
        for method_name in (
            "_apply_creature_intents",
            "_commit_communication_intents",
            "_settle_food_motion",
            "_apply_top_down_motion",
            "_limit_creature_motion",
            "_sync_carried_foods",
            "_accumulate_mouth_exposures",
            "_apply_immediate_direct_damage",
            "_update_fitness_survival",
            "_update_flocking_benchmark",
            "_update_chronometers",
            "_update_metabolism",
            "_refresh_stats",
            "_follow_selected_creature",
        ):
            setattr(world, method_name, lambda *args: None)
        return world

    def test_elapsed_time_advances_only_for_completed_steps(self) -> None:
        world = self._clock_world(5.0)

        world.update(1.0 / 30.0)

        self.assertEqual(world.physics_step_count, World.MAX_FRAME_STEPS)
        self.assertAlmostEqual(
            world.elapsed_time,
            World.MAX_FRAME_STEPS * World.FIXED_TIMESTEP,
        )

    def test_large_frame_retains_bounded_backlog(self) -> None:
        world = self._clock_world(1.0)

        world.update(1.0)

        self.assertEqual(world.physics_step_count, World.MAX_FRAME_STEPS)
        self.assertAlmostEqual(
            world.elapsed_time,
            World.MAX_FRAME_STEPS * World.FIXED_TIMESTEP,
        )
        self.assertLess(world.elapsed_time, 1.0)
        self.assertAlmostEqual(
            world.simulation_lag_metrics.pending_seconds,
            55 * World.FIXED_TIMESTEP,
        )
        self.assertEqual(
            world.simulation_lag_metrics.session_dropped_seconds,
            0.0,
        )

    def test_fractional_time_is_preserved_until_one_step_completes(self) -> None:
        world = self._clock_world(0.25)

        for _ in range(3):
            world.update(1.0 / 60.0)
        self.assertEqual(world.physics_step_count, 0)
        self.assertEqual(world.elapsed_time, 0.0)

        world.update(1.0 / 60.0)

        self.assertEqual(world.physics_step_count, 1)
        self.assertAlmostEqual(world.elapsed_time, World.FIXED_TIMESTEP)

    def test_negative_frame_delta_does_not_reverse_simulation_time(self) -> None:
        world = self._clock_world(5.0)

        world.update(-1.0)

        self.assertEqual(world.physics_step_count, 0)
        self.assertEqual(world.elapsed_time, 0.0)
        self.assertEqual(world._physics_accumulator, 0.0)

    def test_simulation_subsystems_receive_one_fixed_delta_per_step(
        self,
    ) -> None:
        world = self._clock_world(5.0)

        world.update(1.0 / 30.0)

        fixed_expected = [World.FIXED_TIMESTEP] * World.MAX_FRAME_STEPS
        self.assertEqual(world.timebase_calls["speciation"], fixed_expected)
        self.assertEqual(world.timebase_calls["food"], fixed_expected)
        self.assertEqual(
            world.timebase_calls["reproduction"],
            [World.FIXED_TIMESTEP * 3],
        )
        self.assertEqual(world.timebase_calls["telemetry"], fixed_expected)
        self.assertEqual(
            world.timebase_calls["persistence"],
            [1.0 / 30.0],
        )

    def test_wall_clock_timer_runs_without_a_completed_simulation_step(
        self,
    ) -> None:
        world = self._clock_world(0.25)

        world.update(1.0 / 60.0)

        self.assertEqual(world.physics_step_count, 0)
        self.assertEqual(world.timebase_calls["speciation"], [])
        self.assertEqual(world.timebase_calls["food"], [])
        self.assertEqual(world.timebase_calls["reproduction"], [])
        self.assertEqual(world.timebase_calls["telemetry"], [])
        self.assertEqual(
            world.timebase_calls["persistence"],
            [1.0 / 60.0],
        )

    def test_pause_stops_both_simulation_and_existing_save_timer_behavior(
        self,
    ) -> None:
        world = self._clock_world(5.0)
        world.is_paused = True

        world.update(1.0)

        self.assertEqual(world.elapsed_time, 0.0)
        self.assertEqual(world.physics_step_count, 0)
        self.assertTrue(
            all(not calls for calls in world.timebase_calls.values())
        )


class WorldTimebaseIntegrationTest(unittest.TestCase):
    def _world(self, speed: float) -> World:
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.persistence.quick_save_interval_seconds = 0.0
        config.persistence.archive_save_interval_seconds = 0.0
        config.population.initial_creatures = 2
        config.food.initial_food_items = 4
        world = World(
            config,
            simulation_paths=SimulationPaths(Path(".").resolve()),
        )
        world.set_simulation_speed(speed)
        return world

    @staticmethod
    def _biological_state(world: World) -> tuple[object, ...]:
        creatures = tuple(
            (
                creature.creature_id,
                creature.energy,
                creature.life,
                creature.stomach_energy,
                creature.stomach_difficulty_load,
                creature.rest_intent,
                creature.smoothed_rest,
                creature.effective_rest,
                creature.activity,
                creature.pending_direct_life_damage,
                creature.position,
                tuple(creature.body.velocity),
                creature.lineage.parent_id,
                creature.lineage.generation,
                creature.lineage.species_id,
                world.fitness[creature.creature_id].age_seconds,
                world._chronometers[creature.creature_id],
            )
            for creature in world.creatures
        )
        foods = tuple(
            (
                food.id,
                food.energy_value,
                food.radius,
                food.position,
            )
            for food in world.foods
        )
        return (
            world.elapsed_time,
            world.physics_step_count,
            world._reproduction_accumulator,
            world._speciation_adjustment_accumulator,
            world._flocking_telemetry_accumulator,
            world.food_spawner._spawn_credit,
            world.food_spawner._low_food_burst_credit,
            world.rng.getstate(),
            world.neat_controller.evolution_random_state(),
            world.neat_controller.evolution_allocator_state(),
            creatures,
            foods,
        )

    def _assert_grouping_equivalent(
        self,
        *,
        first_speed: float,
        first_frames: int,
        first_delta: float,
        second_speed: float,
        second_frames: int,
        second_delta: float,
    ) -> None:
        first = self._world(first_speed)
        second = self._world(second_speed)
        try:
            for _ in range(first_frames):
                first.update(first_delta)
            for _ in range(second_frames):
                second.update(second_delta)

            self.assertEqual(
                self._biological_state(first),
                self._biological_state(second),
            )
            self.assertAlmostEqual(
                first.elapsed_time,
                first.physics_step_count * World.FIXED_TIMESTEP,
            )
            self.assertAlmostEqual(
                second.elapsed_time,
                second.physics_step_count * World.FIXED_TIMESTEP,
            )
        finally:
            first.close()
            second.close()

    def test_stable_five_x_matches_equal_one_x_fixed_steps(self) -> None:
        self._assert_grouping_equivalent(
            first_speed=5.0,
            first_frames=60,
            first_delta=1.0 / 60.0,
            second_speed=1.0,
            second_frames=300,
            second_delta=1.0 / 60.0,
        )

    def test_quarter_speed_matches_equal_one_x_fixed_steps(self) -> None:
        self._assert_grouping_equivalent(
            first_speed=0.25,
            first_frames=240,
            first_delta=1.0 / 60.0,
            second_speed=1.0,
            second_frames=60,
            second_delta=1.0 / 60.0,
        )

    def test_low_fps_five_x_matches_completed_one_x_steps(self) -> None:
        self._assert_grouping_equivalent(
            first_speed=5.0,
            first_frames=30,
            first_delta=1.0 / 30.0,
            second_speed=1.0,
            second_frames=150,
            second_delta=1.0 / 60.0,
        )


if __name__ == "__main__":
    unittest.main()
