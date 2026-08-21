from __future__ import annotations

from dataclasses import replace
from itertools import count
import os
from pathlib import Path
from queue import Queue
from random import Random
from threading import Event, Thread
from types import SimpleNamespace
import unittest

from src.action import Action
from src.behavior_observer import BehaviorObserverService
from src.persistence import PersistenceManager, SimulationPaths
from tests.scheduler_validation import (
    AuthoritativeStateDigest,
    DeterministicSoakHarness,
    assert_authoritative_match,
    validation_config,
    validation_world,
)


def _action(*, reproduce: bool = False, eat: bool = False) -> Action:
    return Action(
        accelerate=0.0,
        rotate=0.0,
        want_reproduce=1.0 if reproduce else 0.0,
        want_eat=1.0 if eat else 0.0,
        reset_chronometer=0.0,
        want_grab=0.0,
        want_release=0.0,
    )


class _PassiveObserver:
    def __init__(self, *, accepts: bool = True) -> None:
        self.accepts = accepts
        self.latest_snapshot = None
        self.latest_snapshots = {}
        self.latest_why_snapshots = ()
        self.submissions = 0
        self.probes = 0
        self.subjects = ()

    def poll(self):
        return None

    def drain_progress_snapshots(self):
        return ()

    def drain_completed_bouts(self):
        return ()

    def set_subjects(self, subjects):
        self.subjects = tuple(subjects)

    def set_focus(self, *_args):
        return None

    def set_focal_brain(self, *_args):
        return self.accepts

    def submit_batch(self, observations):
        self.submissions += len(observations)
        return self.accepts

    def submit_why(self, _probe):
        self.probes += 1
        return self.accepts

    def finalize_subject(self, *_args):
        return self.accepts

    def close(self):
        return None


def _replace_observer(world, observer: _PassiveObserver) -> None:
    current = world.behavior_observer
    if current is not observer:
        current.close()
    world.behavior_observer = observer


def _restore(world):
    state = PersistenceManager._capture_state(
        world,
        world.neat_controller,
    )
    return PersistenceManager._restore_world(
        state,
        world.config,
        world.simulation_paths,
    )


class DeterministicDigestTest(unittest.TestCase):
    def test_duplicate_run_compares_every_sixty_steps(self) -> None:
        first = validation_world(seed=101)
        second = validation_world(seed=101)
        try:
            for _step in range(1, 181):
                first.update(first.fixed_timestep)
                second.update(second.fixed_timestep)
                if first._simulation_step % 60 == 0:
                    assert_authoritative_match(self, first, second)
        finally:
            first.close()
            second.close()

    def test_divergence_reports_completed_step_entity_and_field(self) -> None:
        first = validation_world(creatures=2)
        second = validation_world(creatures=2)
        try:
            second.creatures[1].energy -= 0.25
            difference = AuthoritativeStateDigest.capture(first).compare(
                AuthoritativeStateDigest.capture(second)
            )
            self.assertIsNotNone(difference)
            assert difference is not None
            self.assertEqual(difference.step, 0)
            self.assertIn("creatures[1].energy", difference.field)
        finally:
            first.close()
            second.close()


class SchedulerPhaseCheckpointTest(unittest.TestCase):
    def test_neat_innovation_history_survives_checkpoint(self) -> None:
        world = validation_world(seed=102)
        restored = None
        try:
            tracker = (
                world.neat_controller.config.genome_config.innovation_tracker
            )
            marker = (-999, 999, "scheduler_validation")
            tracker.generation_innovations[marker] = 10_001
            tracker.global_counter = 10_001
            restored = _restore(world)
            restored_tracker = (
                restored.neat_controller.config.genome_config.innovation_tracker
            )
            self.assertEqual(restored_tracker.global_counter, 10_001)
            self.assertEqual(
                restored_tracker.generation_innovations[marker],
                10_001,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_neat_node_allocator_survives_without_a_live_high_node(self) -> None:
        world = validation_world(seed=116)
        restored = None
        try:
            world.neat_controller.config.genome_config.node_indexer = count(123)
            restored = _restore(world)
            self.assertEqual(
                world.neat_controller.evolution_allocator_state()[
                    "next_node_id"
                ],
                123,
            )
            self.assertEqual(
                restored.neat_controller.evolution_allocator_state()[
                    "next_node_id"
                ],
                123,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_save_load_covers_all_scheduler_phase_boundaries(self) -> None:
        capture_points = {
            1: "after decision phase 0",
            2: "after decision phase 1 / before biology",
            3: "after decision phase 2 / after biology",
            5: "before observer deadline",
            6: "after observer deadline",
            11: "before statistics refresh",
            12: "after statistics refresh",
            14: "before pheromone boundary",
            15: "after pheromone boundary",
        }
        reference = validation_world(
            seed=103,
            behavior_enabled=True,
        )
        _replace_observer(reference, _PassiveObserver())
        try:
            for _ in range(24):
                reference.update(reference.fixed_timestep)
            for capture_step, label in capture_points.items():
                with self.subTest(capture_step=capture_step, boundary=label):
                    interrupted = validation_world(
                        seed=103,
                        behavior_enabled=True,
                    )
                    _replace_observer(interrupted, _PassiveObserver())
                    restored = None
                    try:
                        for _ in range(capture_step):
                            interrupted.update(interrupted.fixed_timestep)
                        restored = _restore(interrupted)
                        _replace_observer(restored, _PassiveObserver())
                        for _ in range(24 - capture_step):
                            restored.update(restored.fixed_timestep)
                        assert_authoritative_match(self, reference, restored)
                    finally:
                        interrupted.close()
                        if restored is not None:
                            restored.close()
        finally:
            reference.close()

    def test_repeated_save_load_schedules_do_not_drift(self) -> None:
        schedules = {
            "every_step": set(range(1, 13)),
            "seven_and_fifty_nine": {
                *range(7, 127, 7),
                59,
                118,
            },
            "seeded_pseudorandom": set(
                Random(911).sample(range(1, 127), 14)
            ),
        }
        for name, checkpoints in schedules.items():
            with self.subTest(schedule=name):
                target = max(checkpoints)
                reference = validation_world(seed=107)
                candidate = validation_world(seed=107)
                try:
                    for step in range(1, target + 1):
                        reference.update(reference.fixed_timestep)
                        candidate.update(candidate.fixed_timestep)
                        if step in checkpoints:
                            restored = _restore(candidate)
                            candidate.close()
                            candidate = restored
                            assert_authoritative_match(
                                self,
                                reference,
                                candidate,
                            )
                    assert_authoritative_match(self, reference, candidate)
                    self.assertEqual(candidate._physics_accumulator, 0.0)
                    self.assertEqual(
                        candidate.simulation_lag_metrics.pending_seconds,
                        0.0,
                    )
                finally:
                    reference.close()
                    candidate.close()


class MouthExposureCheckpointTest(unittest.TestCase):
    @staticmethod
    def _contested_world():
        world = validation_world(seed=109, creatures=2, foods=2)
        first, second = sorted(
            world.creatures,
            key=lambda creature: creature.creature_id,
        )
        food = min(world.foods, key=lambda item: item.id)
        first.stomach_energy = 0.0
        second.stomach_energy = 0.0
        food.energy_value = min(food.energy_value, 1e-9)
        food._resize_for_remaining_energy()
        world._held_food_by_creature_id[second.creature_id] = food.id
        world._carrier_by_food_id[food.id] = second.creature_id
        world._sync_carried_foods()
        world._mouth_exposures.append(
            0,
            second.creature_id,
            food.id,
            world.fixed_timestep,
        )
        world._mouth_exposures.append(
            0,
            first.creature_id,
            food.id,
            world.fixed_timestep,
        )
        world._mouth_exposures.append(
            1,
            second.creature_id,
            world.foods[1].id,
            world.fixed_timestep,
        )
        return world, first.creature_id, second.creature_id, food.id

    def test_pending_and_contested_exposure_replays_identically(self) -> None:
        reference, winner_id, carrier_id, contested_food_id = (
            self._contested_world()
        )
        interrupted, _winner, _carrier, _food = self._contested_world()
        restored = None
        try:
            expected_records = interrupted._mouth_exposures.state()
            restored = _restore(interrupted)
            self.assertEqual(restored._mouth_exposures.count, 3)
            self.assertEqual(
                restored._mouth_exposures.state(),
                expected_records,
            )

            reference._update_metabolism(3 * reference.fixed_timestep)
            restored._update_metabolism(3 * restored.fixed_timestep)
            assert_authoritative_match(self, reference, restored)
            self.assertEqual(restored._mouth_exposures.count, 0)
            self.assertNotIn(
                contested_food_id,
                restored._carrier_by_food_id,
            )
            self.assertNotIn(
                carrier_id,
                restored._held_food_by_creature_id,
            )
            self.assertEqual(restored.fitness[winner_id].food_eaten, 1)
        finally:
            reference.close()
            interrupted.close()
            if restored is not None:
                restored.close()

    def test_dead_creature_and_missing_food_records_are_retryable(self) -> None:
        reference = validation_world(seed=111, creatures=2, foods=1)
        interrupted = validation_world(seed=111, creatures=2, foods=1)
        restored = None
        try:
            for world in (reference, interrupted):
                victim = max(
                    world.creatures,
                    key=lambda creature: creature.creature_id,
                )
                survivor = min(
                    world.creatures,
                    key=lambda creature: creature.creature_id,
                )
                world._mouth_exposures.append(
                    0,
                    victim.creature_id,
                    world.foods[0].id,
                    world.fixed_timestep,
                )
                world._mouth_exposures.append(
                    1,
                    survivor.creature_id,
                    999_999,
                    world.fixed_timestep,
                )
                victim.life = 0.0
                world._remove_creature(victim, death_reason="injected")
            restored = _restore(interrupted)
            self.assertEqual(restored._mouth_exposures.count, 2)
            reference._update_metabolism(3 * reference.fixed_timestep)
            restored._update_metabolism(3 * restored.fixed_timestep)
            self.assertEqual(restored._mouth_exposures.count, 0)
            assert_authoritative_match(self, reference, restored)
        finally:
            reference.close()
            interrupted.close()
            if restored is not None:
                restored.close()


class AsynchronousCheckpointTest(unittest.TestCase):
    def test_concurrent_capture_is_completed_step_and_io_does_not_lock_world(
        self,
    ) -> None:
            paths = SimulationPaths(Path(".").resolve())
            world = validation_world(seed=113, paths=paths)
            restored = None
            entered_step = Event()
            release_step = Event()
            save_returned = Event()
            writer_started = Event()
            release_writer = Event()
            original_accumulate = world._accumulate_mouth_exposures
            written_states = []

            def blocked_accumulate(delta_time):
                original_accumulate(delta_time)
                entered_step.set()
                if not release_step.wait(2.0):
                    raise RuntimeError("test did not release fixed step")

            def blocked_write(*args, **kwargs):
                writer_started.set()
                if not release_writer.wait(2.0):
                    raise RuntimeError("test did not release writer")
                written_states.append(args[0])

            world._accumulate_mouth_exposures = blocked_accumulate
            world.persistence_manager._write_atomic = blocked_write

            update_thread = Thread(
                target=world.update,
                args=(world.fixed_timestep,),
            )

            def request_save():
                world.save_now()
                save_returned.set()

            save_thread = Thread(target=request_save)
            try:
                update_thread.start()
                self.assertTrue(entered_step.wait(1.0))
                save_thread.start()
                self.assertFalse(save_returned.wait(0.05))
                release_step.set()
                update_thread.join(2.0)
                save_thread.join(2.0)
                self.assertFalse(update_thread.is_alive())
                self.assertFalse(save_thread.is_alive())
                self.assertEqual(world._simulation_step, 1)
                self.assertTrue(writer_started.wait(1.0))

                world._accumulate_mouth_exposures = original_accumulate
                world.update(world.fixed_timestep)
                self.assertEqual(world._simulation_step, 2)
                release_writer.set()
                world.persistence_manager.flush()

                self.assertEqual(len(written_states), 1)
                restored = PersistenceManager._restore_world(
                    written_states[0],
                    world.config,
                    paths,
                )
                self.assertEqual(restored._simulation_step, 1)
                self.assertAlmostEqual(
                    restored.elapsed_time,
                    restored.fixed_timestep,
                )
                while restored._simulation_step < 24:
                    restored.update(restored.fixed_timestep)
                while world._simulation_step < 24:
                    world.update(world.fixed_timestep)
                assert_authoritative_match(self, world, restored)
            finally:
                release_step.set()
                release_writer.set()
                update_thread.join(2.0)
                save_thread.join(2.0)
                if restored is not None:
                    restored.close()
                world.close()


class ObserverBackpressureValidationTest(unittest.TestCase):
    def test_bounded_latest_wins_observer_and_counterfactual_queues(self) -> None:
        config = validation_config(behavior_enabled=True)
        service = BehaviorObserverService(
            config.behavior,
            config.counterfactual_why,
            config.behavior_history,
        )

        class Alive:
            exitcode = None

            @staticmethod
            def is_alive():
                return True

        service._process = Alive()
        service._input_queue = Queue(maxsize=1)
        service._why_probe_queue = Queue(maxsize=1)
        try:
            for index in range(100):
                service.submit(SimpleNamespace(index=index))
                service.submit_why(SimpleNamespace(index=index))
            self.assertEqual(service._input_queue.qsize(), 1)
            self.assertEqual(service._why_probe_queue.qsize(), 1)
            self.assertEqual(service._input_queue.get_nowait().index, 99)
            self.assertEqual(service._why_probe_queue.get_nowait().index, 99)
            self.assertGreater(service.diagnostics.samples_dropped, 0)
            self.assertGreater(
                service.counterfactual_diagnostics.probe_requests_dropped,
                0,
            )
        finally:
            service._closed = True

    def test_dropped_observations_do_not_change_authoritative_world(self) -> None:
        worlds = [
            validation_world(seed=127, behavior_enabled=True)
            for _ in range(3)
        ]
        observers = [
            _PassiveObserver(accepts=True),
            _PassiveObserver(accepts=False),
            _PassiveObserver(accepts=False),
        ]
        try:
            for world, observer in zip(worlds, observers):
                _replace_observer(world, observer)
            for _ in range(60):
                for world in worlds:
                    world.update(world.fixed_timestep)
            assert_authoritative_match(self, worlds[0], worlds[1])
            assert_authoritative_match(self, worlds[0], worlds[2])
            self.assertGreater(observers[0].submissions, 0)
            self.assertEqual(
                observers[0].submissions,
                observers[1].submissions,
            )
        finally:
            for world in worlds:
                world.close()


class FailureInjectionValidationTest(unittest.TestCase):
    exposure_points = (
        "exposure.before_validation",
        "exposure.before_mutation",
        "exposure.after_stomach_mutation",
        "exposure.after_food_mutation",
        "exposure.after_carried_food_mutation",
        "exposure.after_valid_claim",
        "exposure.before_buffer_clear",
    )

    @staticmethod
    def _exposure_world():
        world = validation_world(seed=131, creatures=1, foods=1)
        creature = world.creatures[0]
        food = world.foods[0]
        creature.stomach_energy = 0.0
        food.energy_value = min(food.energy_value, 1e-9)
        food._resize_for_remaining_energy()
        world._held_food_by_creature_id[creature.creature_id] = food.id
        world._carrier_by_food_id[food.id] = creature.creature_id
        world._sync_carried_foods()
        world._mouth_exposures.append(
            0,
            creature.creature_id,
            food.id,
            world.fixed_timestep,
        )
        return world

    def test_exposure_failure_points_roll_back_and_retry_cleanly(self) -> None:
        for point in self.exposure_points:
            with self.subTest(point=point):
                failed = self._exposure_world()
                clean = self._exposure_world()
                try:
                    before = AuthoritativeStateDigest.capture(failed)

                    def inject(candidate):
                        if candidate == point:
                            raise RuntimeError(point)

                    failed._scheduler_validation_failure_injector = inject
                    with self.assertRaisesRegex(RuntimeError, point):
                        failed._resolve_accumulated_mouth_exposures()
                    difference = before.compare(
                        AuthoritativeStateDigest.capture(failed)
                    )
                    self.assertIsNone(
                        difference,
                        None if difference is None else difference.describe(),
                    )
                    self.assertEqual(failed._simulation_step, 0)
                    self.assertEqual(failed._mouth_exposures.count, 1)

                    del failed._scheduler_validation_failure_injector
                    failed._resolve_accumulated_mouth_exposures()
                    clean._resolve_accumulated_mouth_exposures()
                    assert_authoritative_match(self, failed, clean)
                finally:
                    failed.close()
                    clean.close()

    def test_biology_failures_surface_without_advancing_completed_step(self) -> None:
        points = (
            "biology.resource_candidate_preparation",
            "biology.reproduction_preparation",
            "biology.nursing_preparation",
            "biology.digestion_evaluation",
            "biology.metabolism_evaluation",
            "biology.commit_boundary",
            "biology.after_creature_commit",
            "biology.post_death_processing",
            "biology.dependent_fitness_bookkeeping",
        )
        for point in points:
            with self.subTest(point=point):
                world = validation_world(seed=137, creatures=2, foods=2)
                try:
                    for _ in range(2):
                        world.update(world.fixed_timestep)
                    creature = world.creatures[0]
                    food = world.foods[0]
                    world._mouth_exposures.append(
                        world._simulation_step,
                        creature.creature_id,
                        food.id,
                        world.fixed_timestep,
                    )
                    count = world._mouth_exposures.count

                    def inject(candidate):
                        if candidate == point:
                            raise RuntimeError(point)

                    world._scheduler_validation_failure_injector = inject
                    with self.assertRaisesRegex(RuntimeError, point):
                        world.update(world.fixed_timestep)
                    self.assertEqual(world._simulation_step, 2)
                    self.assertGreaterEqual(world._mouth_exposures.count, count)
                    checkpoint = PersistenceManager._capture_state(
                        world,
                        world.neat_controller,
                    )
                    self.assertEqual(
                        checkpoint["world"]["simulation_step"],
                        2,
                    )
                finally:
                    world.close()


class PopulationChurnValidationTest(unittest.TestCase):
    @staticmethod
    def _scripted_churn(world, step: int) -> None:
        if step > 0 and step % 15 == 0 and world.creatures:
            parent = min(world.creatures, key=lambda item: item.creature_id)
            parent.energy = max(parent.energy, 1.0)
            parent.age_seconds = max(
                parent.age_seconds,
                world.config.population.min_reproduction_age,
            )
            parent.last_birth_time = (
                parent.age_seconds
                - world.config.population.reproduction_cooldown
            )
            reproduce = _action(reproduce=True)
            world._last_actions[parent.creature_id] = reproduce
            world._effective_actions[parent.creature_id] = reproduce
        if step % 20 == 10 and len(world.creatures) > 2:
            victim = max(world.creatures, key=lambda item: item.creature_id)
            if world.foods:
                world._mouth_exposures.append(
                    step,
                    victim.creature_id,
                    world.foods[0].id,
                    world.fixed_timestep,
                )
            victim.pending_direct_life_damage = victim.life + 1.0
        if step % 30 == 29 and len(world.creatures) > 2:
            victim = sorted(
                world.creatures,
                key=lambda item: item.creature_id,
            )[-2]
            victim.energy = 0.0
            victim.life = 1e-12
            victim.stomach_energy = 0.0
            victim.stomach_difficulty_load = 0.0
        # Pymunk's live contact/broad-phase caches are intentionally not part
        # of the checkpoint format. Keep this scheduler-focused churn fixture
        # contact-rich but impulse-free, so begin/separate callbacks and mouth
        # exposure are exercised without comparing non-serializable solver
        # warm-start state after a reload.
        for creature in world.creatures:
            creature.shape.sensor = True
        for food in world.foods:
            food.shape.sensor = True

    def test_birth_death_phase_and_checkpoint_churn_are_deterministic(self) -> None:
        reference = validation_world(seed=139, creatures=4, foods=8)
        candidate = validation_world(seed=139, creatures=4, foods=8)
        original_phases = {
            creature.creature_id: reference._decision_phase(
                creature.creature_id
            )
            for creature in reference.creatures
        }
        checkpoints = {7, 29, 30, 59, 60, 89, 90, 119}
        try:
            for step in range(120):
                self._scripted_churn(reference, step)
                self._scripted_churn(candidate, step)
                reference.update(reference.fixed_timestep)
                candidate.update(candidate.fixed_timestep)
                if candidate._simulation_step in checkpoints:
                    restored = _restore(candidate)
                    candidate.close()
                    candidate = restored
                if reference._simulation_step % 30 == 0:
                    assert_authoritative_match(self, reference, candidate)

            self.assertGreater(reference.rt_neat.stats.births, 0)
            self.assertGreater(reference.rt_neat.stats.deaths, 0)
            ids = [creature.creature_id for creature in candidate.creatures]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertLess(max(ids, default=0), candidate._next_creature_id_value)
            for creature in candidate.creatures:
                self.assertEqual(
                    candidate._decision_phase(creature.creature_id),
                    creature.creature_id
                    % candidate.config.scheduler.decision_period_steps,
                )
                if creature.creature_id in original_phases:
                    self.assertEqual(
                        candidate._decision_phase(creature.creature_id),
                        original_phases[creature.creature_id],
                    )
            live_ids = set(ids)
            live_food_ids = {food.id for food in candidate.foods}
            for _step, creature_id, food_id, _duration in (
                candidate._mouth_exposures.state()
            ):
                self.assertIn(creature_id, live_ids)
                self.assertIn(food_id, live_food_ids)
            assert_authoritative_match(self, reference, candidate)
        finally:
            reference.close()
            candidate.close()


class PauseAndDeltaValidationTest(unittest.TestCase):
    def test_pause_load_resume_keeps_exposure_but_resets_session_debt(self) -> None:
        world = validation_world(seed=149, creatures=1, foods=1)
        restored = None
        try:
            world.update(world.fixed_timestep / 2.0)
            creature = world.creatures[0]
            food = world.foods[0]
            world._mouth_exposures.append(
                0,
                creature.creature_id,
                food.id,
                world.fixed_timestep,
            )
            world.is_paused = True
            step = world._simulation_step
            elapsed = world.elapsed_time
            world.update(-1.0)
            world.update(0.0)
            world.update(10.0)
            self.assertEqual(world._simulation_step, step)
            self.assertEqual(world.elapsed_time, elapsed)

            restored = _restore(world)
            self.assertTrue(restored.is_paused)
            self.assertEqual(restored._physics_accumulator, 0.0)
            self.assertEqual(restored._mouth_exposures.count, 1)
            self.assertEqual(
                restored.simulation_lag_metrics.session_dropped_seconds,
                0.0,
            )
            restored.is_paused = False
            restored.update(restored.fixed_timestep)
            self.assertEqual(restored._simulation_step, step + 1)
        finally:
            world.close()
            if restored is not None:
                restored.close()


@unittest.skipUnless(
    os.environ.get("RUN_SCHEDULER_SOAK") == "1",
    "set RUN_SCHEDULER_SOAK=1 to run the 36,000-step scheduler soak",
)
class ExtendedSchedulerSoakTest(unittest.TestCase):
    def test_ten_simulated_minutes_with_checkpoints_churn_and_pressure(self) -> None:
        reference = validation_world(
            seed=151,
            creatures=6,
            foods=16,
            behavior_enabled=True,
        )
        candidate = validation_world(
            seed=151,
            creatures=6,
            foods=16,
            behavior_enabled=True,
        )
        _replace_observer(reference, _PassiveObserver(accepts=False))
        _replace_observer(candidate, _PassiveObserver(accepts=False))
        reference_harness = DeterministicSoakHarness(reference)
        candidate_harness = DeterministicSoakHarness(candidate)
        checkpoint_steps = {
            1,
            2,
            3,
            59,
            60,
            3599,
            3600,
            7199,
            7200,
            17999,
            18000,
            35999,
        }
        try:
            for target in range(60, 36001, 60):
                reference_harness.run_to(
                    target,
                    before_step=PopulationChurnValidationTest._scripted_churn,
                )
                while candidate_harness.world._simulation_step < target:
                    step = candidate_harness.world._simulation_step
                    PopulationChurnValidationTest._scripted_churn(
                        candidate_harness.world,
                        step,
                    )
                    candidate_harness.world.update(
                        candidate_harness.world.fixed_timestep
                    )
                    candidate_harness._record_metrics()
                    if candidate_harness.world._simulation_step in checkpoint_steps:
                        candidate_harness.checkpoint_reload()
                        _replace_observer(
                            candidate_harness.world,
                            _PassiveObserver(accepts=False),
                        )
                assert_authoritative_match(
                    self,
                    reference_harness.world,
                    candidate_harness.world,
                )
            self.assertEqual(reference_harness.metrics.fixed_steps, 36000)
            self.assertGreater(reference_harness.metrics.births, 0)
            self.assertGreater(reference_harness.metrics.deaths, 0)
            self.assertEqual(
                candidate_harness.metrics.checkpoints,
                len(checkpoint_steps),
            )
        finally:
            reference_harness.world.close()
            candidate_harness.world.close()


if __name__ == "__main__":
    unittest.main()
