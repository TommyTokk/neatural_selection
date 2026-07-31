from __future__ import annotations

from dataclasses import replace
import pickle
from queue import Queue
import time
from types import SimpleNamespace
import unittest

from configs.sim_config import BehaviorObserverConfig
from src.behavior_history import BehaviorTermination
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObservation,
    BehaviorObserverService,
    BehaviorSnapshot,
    BehaviorWorkerError,
    BoutStatus,
    TemporalBehaviorAnalyzer,
    _put_latest,
)


def observation(
    simulation_time: float,
    *,
    creature_id: int = 1,
    generation: int = 1,
    speed: float = 20.0,
    velocity_x: float | None = None,
    velocity_y: float = 0.0,
    angular_velocity: float = 0.0,
    food_id: int | None = None,
    food_distance: float | None = None,
    food_angle: float | None = None,
    consumption_count: int = 0,
    consumed_energy: float = 0.0,
    group_distance: float | None = None,
    group_velocity_x: float = 0.0,
    group_velocity_y: float = 0.0,
    personal_space: bool = False,
    alarm_here: float = 0.0,
    alarm_forward: float = 0.0,
) -> BehaviorObservation:
    return BehaviorObservation(
        creature_id=creature_id,
        selection_generation=generation,
        simulation_time=simulation_time,
        x=simulation_time * speed,
        y=0.0,
        heading=0.0,
        angular_velocity=angular_velocity,
        velocity_x=speed if velocity_x is None else velocity_x,
        velocity_y=velocity_y,
        speed=speed,
        nearest_food_id=food_id,
        food_visible=food_id is not None,
        food_distance=food_distance,
        food_relative_angle=food_angle,
        compatible_group_visible=group_distance is not None,
        compatible_group_count=2.0 if group_distance is not None else 0.0,
        compatible_group_distance=group_distance,
        compatible_group_direction=0.0 if group_distance is not None else None,
        group_velocity_x=group_velocity_x,
        group_velocity_y=group_velocity_y,
        personal_space_occupied=personal_space,
        alarm_here=alarm_here,
        alarm_forward_left=alarm_forward,
        alarm_forward_right=alarm_forward,
        food_consumption_count=consumption_count,
        food_consumed_energy_total=consumed_energy,
    )


def state_for(snapshot, behavior: BehaviorKind):
    return next(
        (
            state
            for state in snapshot.behaviors
            if state.behavior is behavior
        ),
        None,
    )


class TemporalBehaviorAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BehaviorObserverConfig()
        self.analyzer = TemporalBehaviorAnalyzer(self.config)

    def test_food_approach_emerges_then_activates(self) -> None:
        distances = [150, 140, 129, 117, 105, 92, 80, 68]
        statuses = []
        for index, distance in enumerate(distances):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=7,
                    food_distance=distance,
                    food_angle=0.0,
                )
            )
            state = state_for(snapshot, BehaviorKind.FOOD_APPROACH)
            statuses.append(None if state is None else state.status)

        self.assertIn(BoutStatus.EMERGING, statuses)
        self.assertEqual(statuses[-1], BoutStatus.ACTIVE)
        self.assertEqual(
            state_for(snapshot, BehaviorKind.FOOD_APPROACH).target_id,
            7,
        )

    def test_noisy_food_distance_never_becomes_stable_approach(self) -> None:
        distances = [100, 94, 101, 95, 103, 98, 104, 99]
        states = []
        for index, distance in enumerate(distances):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=7,
                    food_distance=distance,
                    food_angle=0.0,
                )
            )
            states.append(state_for(snapshot, BehaviorKind.FOOD_APPROACH))

        self.assertFalse(
            any(
                state is not None and state.status is BoutStatus.ACTIVE
                for state in states
            )
        )

    def test_converging_angle_activates_orientation(self) -> None:
        angles = [0.8, 0.68, 0.55, 0.43, 0.31, 0.20, 0.10, 0.04]
        final = None
        for index, angle in enumerate(angles):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=9,
                    food_distance=80.0,
                    food_angle=angle,
                    angular_velocity=0.8,
                )
            )
            final = state_for(snapshot, BehaviorKind.FOOD_ORIENTATION)

        self.assertIsNotNone(final)
        self.assertEqual(final.status, BoutStatus.ACTIVE)
        self.assertEqual(final.target_id, 9)

    def test_food_grace_state_does_not_retain_stale_target_id(self) -> None:
        snapshot = None
        for index, distance in enumerate(
            [150, 140, 129, 117, 105, 92, 80, 68]
        ):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=7,
                    food_distance=distance,
                    food_angle=0.0,
                )
            )
        active = state_for(snapshot, BehaviorKind.FOOD_APPROACH)
        self.assertIsNotNone(active)
        self.assertEqual(active.target_id, 7)

        grace_snapshot = self.analyzer.process(observation(0.8))
        grace = state_for(
            grace_snapshot,
            BehaviorKind.FOOD_APPROACH,
        )

        self.assertIsNotNone(grace)
        self.assertIs(grace.status, BoutStatus.ACTIVE)
        self.assertIsNone(grace.target_id)

    def test_fluctuating_angle_does_not_activate_orientation(self) -> None:
        angles = [0.6, 0.45, 0.62, 0.40, 0.58, 0.39, 0.55, 0.41]
        active = False
        for index, angle in enumerate(angles):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=9,
                    food_distance=80.0,
                    food_angle=angle,
                    angular_velocity=0.8,
                )
            )
            state = state_for(snapshot, BehaviorKind.FOOD_ORIENTATION)
            active |= state is not None and state.status is BoutStatus.ACTIVE

        self.assertFalse(active)

    def test_feeding_requires_real_consumption_event(self) -> None:
        first = self.analyzer.process(observation(0.0))
        second = self.analyzer.process(
            observation(
                0.1,
                consumption_count=1,
                consumed_energy=0.08,
            )
        )

        self.assertIsNone(state_for(first, BehaviorKind.FEEDING))
        feeding = state_for(second, BehaviorKind.FEEDING)
        self.assertIsNotNone(feeding)
        self.assertEqual(feeding.status, BoutStatus.ACTIVE)
        self.assertEqual(feeding.evidence_score, 1.0)

        persisted = self.analyzer.process(
            observation(
                0.8,
                consumption_count=1,
                consumed_energy=0.08,
            )
        )
        self.assertEqual(
            state_for(persisted, BehaviorKind.FEEDING).evidence_score,
            1.0,
        )

        expired = self.analyzer.process(
            observation(
                1.0,
                consumption_count=1,
                consumed_energy=0.08,
            )
        )
        self.assertIsNone(state_for(expired, BehaviorKind.FEEDING))

    def test_sustained_rest_activates_but_single_stop_does_not(self) -> None:
        single = self.analyzer.process(observation(0.0, speed=0.0))
        self.assertEqual(
            state_for(single, BehaviorKind.RESTING).status,
            BoutStatus.EMERGING,
        )
        moving = self.analyzer.process(observation(0.1, speed=30.0))
        self.assertIsNone(state_for(moving, BehaviorKind.RESTING))

        self.analyzer.reset(1, 2)
        final = None
        for index in range(7):
            final = self.analyzer.process(
                observation(
                    index * 0.1,
                    generation=2,
                    speed=0.5,
                )
            )
        resting = state_for(final, BehaviorKind.RESTING)
        self.assertIsNotNone(resting)
        self.assertEqual(resting.status, BoutStatus.ACTIVE)

    def test_active_bout_uses_grace_then_ends(self) -> None:
        snapshot = None
        for index in range(8):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=3,
                    food_distance=150 - index * 12,
                    food_angle=0.0,
                )
            )
        self.assertEqual(
            state_for(snapshot, BehaviorKind.FOOD_APPROACH).status,
            BoutStatus.ACTIVE,
        )

        short_gap = self.analyzer.process(observation(0.8))
        long_gap = self.analyzer.process(observation(1.2))
        self.assertEqual(
            state_for(short_gap, BehaviorKind.FOOD_APPROACH).status,
            BoutStatus.ACTIVE,
        )
        self.assertIsNone(
            state_for(long_gap, BehaviorKind.FOOD_APPROACH)
        )

    def test_focus_change_clears_history_and_bouts(self) -> None:
        for index in range(8):
            snapshot = self.analyzer.process(
                observation(index * 0.1, speed=0.0)
            )
        self.assertIsNotNone(state_for(snapshot, BehaviorKind.RESTING))

        changed = self.analyzer.process(
            observation(0.8, creature_id=2, generation=2, speed=30.0)
        )
        self.assertEqual(len(self.analyzer.history), 1)
        self.assertIsNone(state_for(changed, BehaviorKind.RESTING))

    def test_sliding_window_expires_samples_and_incremental_rest_totals(self) -> None:
        analyzer = TemporalBehaviorAnalyzer(
            BehaviorObserverConfig(
                window_seconds=0.5,
                bout_start_seconds=0.2,
            )
        )
        for index in range(6):
            analyzer.process(observation(index * 0.1, speed=0.5))

        snapshot = analyzer.process(observation(1.1, speed=20.0))

        self.assertEqual(len(analyzer.history), 1)
        self.assertEqual(analyzer._low_speed_count, 0)
        self.assertAlmostEqual(analyzer._speed_sum, 20.0)
        self.assertIsNone(state_for(snapshot, BehaviorKind.RESTING))

    def test_food_target_change_resets_orientation_trend(self) -> None:
        for index, angle in enumerate([0.8, 0.6, 0.4, 0.2]):
            self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=1,
                    food_distance=80.0,
                    food_angle=angle,
                    angular_velocity=0.8,
                )
            )

        changed = self.analyzer.process(
            observation(
                0.4,
                food_id=2,
                food_distance=80.0,
                food_angle=0.1,
                angular_velocity=0.8,
            )
        )

        self.assertIsNone(
            state_for(changed, BehaviorKind.FOOD_ORIENTATION)
        )

    def test_cohesion_uses_realized_closing_motion(self) -> None:
        snapshot = None
        for index, distance in enumerate([100, 94, 88, 82, 76, 70, 64, 58]):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    speed=20.0,
                    group_distance=distance,
                    group_velocity_x=18.0,
                )
            )
        cohesion = state_for(snapshot, BehaviorKind.COHESION)
        self.assertIsNotNone(cohesion)
        self.assertEqual(cohesion.status, BoutStatus.ACTIVE)

    def test_personal_space_separation_is_not_cohesion(self) -> None:
        active = False
        for index, distance in enumerate([50, 44, 38, 32, 26, 20, 14, 8]):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    speed=20.0,
                    group_distance=distance,
                    group_velocity_x=18.0,
                    personal_space=True,
                )
            )
            state = state_for(snapshot, BehaviorKind.COHESION)
            active |= state is not None and state.status is BoutStatus.ACTIVE
        self.assertFalse(active)

    def test_cohesion_uses_aligned_following_with_stable_separation(self) -> None:
        snapshot = None
        for index in range(8):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    speed=12.0,
                    group_distance=80.0,
                    group_velocity_x=9.0,
                )
            )

        cohesion = state_for(snapshot, BehaviorKind.COHESION)
        self.assertIsNotNone(cohesion)
        self.assertEqual(cohesion.status, BoutStatus.ACTIVE)
        alignment = next(
            item
            for item in cohesion.evidence
            if item.key == "group_velocity_alignment"
        )
        self.assertTrue(alignment.passed)

    def test_alarm_retreat_uses_alarm_gradient_not_neural_panic(self) -> None:
        snapshot = None
        for index in range(8):
            local_alarm = 0.30 - index * 0.015
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    speed=20.0,
                    alarm_here=local_alarm,
                    alarm_forward=local_alarm - 0.04,
                )
            )
        retreat = state_for(snapshot, BehaviorKind.ALARM_RETREAT)
        self.assertIsNotNone(retreat)
        self.assertEqual(retreat.status, BoutStatus.ACTIVE)

        no_alarm = TemporalBehaviorAnalyzer(self.config)
        for index in range(8):
            snapshot = no_alarm.process(
                observation(index * 0.1, speed=40.0)
            )
        self.assertIsNone(
            state_for(snapshot, BehaviorKind.ALARM_RETREAT)
        )

    def test_alarm_retreat_requires_spatial_and_temporal_decrease(self) -> None:
        no_spatial = TemporalBehaviorAnalyzer(self.config)
        no_temporal = TemporalBehaviorAnalyzer(self.config)
        for index in range(8):
            local_alarm = 0.30 - index * 0.015
            spatial_snapshot = no_spatial.process(
                observation(
                    index * 0.1,
                    speed=20.0,
                    alarm_here=local_alarm,
                    alarm_forward=local_alarm,
                )
            )
            temporal_snapshot = no_temporal.process(
                observation(
                    index * 0.1,
                    speed=20.0,
                    alarm_here=0.30,
                    alarm_forward=0.25,
                )
            )

        self.assertIsNone(
            state_for(spatial_snapshot, BehaviorKind.ALARM_RETREAT)
        )
        self.assertIsNone(
            state_for(temporal_snapshot, BehaviorKind.ALARM_RETREAT)
        )

    def test_primary_and_secondary_bouts_can_be_simultaneous(self) -> None:
        snapshot = None
        for index in range(8):
            local_alarm = 0.30 - index * 0.015
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    speed=20.0,
                    food_id=4,
                    food_distance=150.0 - index * 12.0,
                    food_angle=0.8 - index * 0.1,
                    angular_velocity=0.8,
                    group_distance=80.0,
                    group_velocity_x=18.0,
                    alarm_here=local_alarm,
                    alarm_forward=local_alarm - 0.04,
                )
            )

        kinds = {
            state.behavior
            for state in snapshot.behaviors
            if state.status is BoutStatus.ACTIVE
        }
        self.assertTrue(
            {
                BehaviorKind.FOOD_ORIENTATION,
                BehaviorKind.FOOD_APPROACH,
                BehaviorKind.COHESION,
                BehaviorKind.ALARM_RETREAT,
            }.issubset(kinds)
        )

    def test_evidence_scores_are_bounded_and_multi_label(self) -> None:
        snapshot = None
        for index in range(8):
            snapshot = self.analyzer.process(
                observation(
                    index * 0.1,
                    food_id=4,
                    food_distance=150 - index * 15,
                    food_angle=0.8 - index * 0.1,
                    angular_velocity=0.8,
                )
            )
        kinds = {state.behavior for state in snapshot.behaviors}
        self.assertIn(BehaviorKind.FOOD_ORIENTATION, kinds)
        self.assertIn(BehaviorKind.FOOD_APPROACH, kinds)
        self.assertTrue(
            all(0.0 <= state.evidence_score <= 1.0 for state in snapshot.behaviors)
        )


class BehaviorObserverServiceTest(unittest.TestCase):
    def test_contracts_are_spawn_picklable(self) -> None:
        restored = pickle.loads(pickle.dumps(observation(0.0)))

        self.assertEqual(restored, observation(0.0))
        self.assertIs(restored.__class__, BehaviorObservation)

    def test_queue_overload_discards_oldest_without_blocking(self) -> None:
        service = BehaviorObserverService(
            BehaviorObserverConfig(input_queue_capacity=1)
        )
        service._process = SimpleNamespace(
            is_alive=lambda: True,
            exitcode=None,
        )
        service._input_queue = Queue(maxsize=1)
        first = observation(0.0)
        second = replace(first, simulation_time=0.1)

        self.assertTrue(service.submit(first))
        self.assertTrue(service.submit(second))
        retained = service._input_queue.get_nowait()

        self.assertEqual(retained.simulation_time, 0.1)
        self.assertEqual(service.diagnostics.samples_dropped, 1)

    def test_result_queue_overload_retains_newest(self) -> None:
        queue = Queue(maxsize=1)
        old = BehaviorSnapshot(1, 1, 0.0, (), 1, time.monotonic())
        new = replace(old, simulation_time=0.1, observations_processed=2)
        queue.put_nowait(old)

        drops = _put_latest(queue, new, 0)

        self.assertEqual(drops, 1)
        self.assertEqual(queue.get_nowait().simulation_time, 0.1)

    def test_subject_replacement_finalizes_removed_analyzers_as_mode_switch(
        self,
    ) -> None:
        service = BehaviorObserverService(BehaviorObserverConfig())
        service._process = SimpleNamespace(
            is_alive=lambda: True,
            exitcode=None,
        )
        service._lifecycle_queue = Queue(maxsize=8)
        service._subjects = {(1, -1), (2, -2)}

        service.set_subjects(((3, 1),))

        requests = (
            service._lifecycle_queue.get_nowait(),
            service._lifecycle_queue.get_nowait(),
        )
        self.assertEqual(
            {request.creature_id for request in requests},
            {1, 2},
        )
        self.assertTrue(
            all(
                request.termination is BehaviorTermination.MODE_SWITCHED
                for request in requests
            )
        )

    def test_poll_rejects_stale_focus_and_reports_worker_errors(self) -> None:
        service = BehaviorObserverService(BehaviorObserverConfig())
        service._focus = (1, 2)
        service._process = SimpleNamespace(
            is_alive=lambda: True,
            exitcode=None,
        )
        service._stop_event = SimpleNamespace(is_set=lambda: False)
        service._result_queue = Queue()
        service._result_queue.put_nowait(
            BehaviorSnapshot(1, 1, 0.0, (), 1, time.monotonic())
        )
        service._result_queue.put_nowait(
            BehaviorWorkerError(1, 2, "synthetic worker failure")
        )
        newest = BehaviorSnapshot(1, 2, 0.1, (), 2, time.monotonic())
        service._result_queue.put_nowait(newest)

        with self.assertLogs(
            "src.behavior_observer",
            level="ERROR",
        ) as messages:
            self.assertIs(service.poll(), newest)
        self.assertEqual(
            service.diagnostics.last_error,
            "synthetic worker failure",
        )
        self.assertIn("synthetic worker failure", messages.output[0])

    def test_spawn_worker_round_trip_and_clean_shutdown(self) -> None:
        service = BehaviorObserverService(BehaviorObserverConfig())
        try:
            service.set_focus(1, 1)
            self.assertTrue(service.submit(observation(0.0)))
            deadline = time.monotonic() + 10.0
            snapshot = None
            while snapshot is None and time.monotonic() < deadline:
                snapshot = service.poll()
                time.sleep(0.01)
            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot.creature_id, 1)
        finally:
            service.close()
        self.assertFalse(service._process.is_alive())

    def test_spawn_worker_keeps_batched_subject_histories_independent(self) -> None:
        service = BehaviorObserverService(
            BehaviorObserverConfig(input_queue_capacity=32)
        )
        subjects = ((1, 11), (2, 22))
        try:
            service.set_subjects(subjects)
            for index in range(8):
                self.assertTrue(
                    service.submit_batch(
                        (
                            observation(
                                index * 0.1,
                                creature_id=1,
                                generation=11,
                                speed=0.1,
                            ),
                            observation(
                                index * 0.1,
                                creature_id=2,
                                generation=22,
                                food_id=9,
                                food_distance=100.0 - index * 10.0,
                                food_angle=0.0,
                            ),
                        )
                    )
                )
            deadline = time.monotonic() + 10.0
            while (
                (
                    len(service.latest_snapshots) < 2
                    or min(
                        snapshot.observations_processed
                        for snapshot in service.latest_snapshots.values()
                    )
                    < 8
                )
                and time.monotonic() < deadline
            ):
                service.poll()
                time.sleep(0.01)
            first = service.snapshot_for(1, 11)
            second = service.snapshot_for(2, 22)
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertEqual(first.observations_processed, 8)
            self.assertEqual(second.observations_processed, 8)
            self.assertIsNotNone(state_for(first, BehaviorKind.RESTING))
            self.assertIsNotNone(
                state_for(second, BehaviorKind.FOOD_APPROACH)
            )
        finally:
            service.close()


class BehaviorObserverConfigTest(unittest.TestCase):
    def test_rejects_invalid_background_representative_count(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorObserverConfig(background_representatives_per_species=-1)

    def test_rejects_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorObserverConfig(
                window_seconds=0.25,
                bout_start_seconds=0.5,
            )

    def test_rejects_invalid_evidence_ratio(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorObserverConfig(food_visibility_ratio=1.1)


if __name__ == "__main__":
    unittest.main()
