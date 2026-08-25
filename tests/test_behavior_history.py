from __future__ import annotations

from dataclasses import replace
import time
import unittest

from configs.sim_config import (
    BehaviorHistoryConfig,
    BehaviorObserverConfig,
    CounterfactualWhyConfig,
)
from src.behavior_history import (
    BehaviorOutcome,
    BehaviorTermination,
    BoundedMetricAccumulator,
    CompletedBehaviorBoutDraft,
    CompletedSemanticEffect,
    CompletedWhyExplanation,
    CreatureBehaviorHistoryStore,
    EffectDirectionCounts,
)
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObserverService,
    BoutStatus,
    TemporalBehaviorAnalyzer,
)
from src.counterfactual_neat import (
    EffectDirection,
    InfluenceLabel,
    SemanticIntervention,
)
from tests.test_behavior_observer import observation


def _feeding_observations(bout_count: int) -> tuple[object, ...]:
    samples = []
    count = 0
    energy = 0.0
    for index in range(bout_count):
        count += 1
        energy += 0.1
        start = index * 0.1
        samples.append(
            observation(
                start,
                consumption_count=count,
                consumed_energy=energy,
            )
        )
        samples.append(
            observation(
                start + 0.02,
                consumption_count=count,
                consumed_energy=energy,
            )
        )
    return tuple(samples)


def _draft(
    local_id: int,
    influence: float,
    probe_count: int,
    direction: EffectDirection,
    *,
    quantiles_estimated: bool = False,
) -> CompletedBehaviorBoutDraft:
    effect = CompletedSemanticEffect(
        intervention=SemanticIntervention.VISIBLE_FOOD_CUES,
        sample_count=probe_count,
        median_influence=influence,
        p25=influence,
        p75=influence,
        influence_label=(
            InfluenceLabel.STRONG
            if influence >= 0.6
            else InfluenceLabel.WEAK
        ),
        effect_direction=direction,
        direction_counts=EffectDirectionCounts(
            supportive=(
                probe_count
                if direction is EffectDirection.SUPPORTIVE
                else 0
            ),
            suppressive=(
                probe_count
                if direction is EffectDirection.SUPPRESSIVE
                else 0
            ),
        ),
        output_summaries=(),
        quantiles_estimated=quantiles_estimated,
    )
    return CompletedBehaviorBoutDraft(
        creature_id=7,
        selection_generation=3,
        behavior=BehaviorKind.FOOD_APPROACH,
        local_bout_id=local_id,
        start_time=float(local_id),
        end_time=float(local_id + 1),
        duration=1.0,
        evidence_summary=(),
        outcome=BehaviorOutcome.ABANDONED,
        termination=BehaviorTermination.NATURAL,
        why_summary=CompletedWhyExplanation(
            behavior=BehaviorKind.FOOD_APPROACH,
            bout_id=local_id,
            effects=(effect,),
        ),
    )


class BoundedMetricAccumulatorTest(unittest.TestCase):
    def test_short_history_is_exact(self) -> None:
        accumulator = BoundedMetricAccumulator(8)
        for value in range(8):
            accumulator.add(value, value % 2 == 0)

        median_value, p25, p75, estimated = accumulator.summary_values()

        self.assertEqual(median_value, 3.5)
        self.assertEqual(p25, 1.75)
        self.assertEqual(p75, 5.25)
        self.assertFalse(estimated)
        self.assertEqual(accumulator.passed_count, 4)
        self.assertEqual(accumulator.first_value, 0.0)
        self.assertEqual(accumulator.last_value, 7.0)

    def test_long_history_compacts_deterministically_and_stays_bounded(self) -> None:
        first = BoundedMetricAccumulator(16)
        second = BoundedMetricAccumulator(16)
        for value in range(200):
            first.add(value)
            second.add(value)

        self.assertEqual(first.summary_values(), second.summary_values())
        self.assertTrue(first.compacted)
        self.assertEqual(first.total_count, 200)
        self.assertEqual(first.first_value, 0.0)
        self.assertEqual(first.last_value, 199.0)
        self.assertLessEqual(len(first._samples), 16)
        retained_indexes = [index for index, _value in first._samples]
        self.assertEqual(retained_indexes[0], 0)
        self.assertEqual(retained_indexes[-1], 199)
        self.assertTrue(
            all(
                index % first._stride == 0
                for index in retained_indexes[1:-1]
            )
        )
        self.assertLessEqual(
            max(
                later - earlier
                for earlier, later in zip(
                    retained_indexes,
                    retained_indexes[1:],
                )
            ),
            first._stride,
        )

    def test_default_capacity_is_exact_through_512_samples(self) -> None:
        accumulator = BoundedMetricAccumulator(512)
        for value in range(512):
            accumulator.add(value, value < 400)

        median_value, p25, p75, estimated = (
            accumulator.summary_values()
        )

        self.assertEqual(median_value, 255.5)
        self.assertEqual(p25, 127.75)
        self.assertEqual(p75, 383.25)
        self.assertFalse(estimated)
        self.assertEqual(len(accumulator._samples), 512)
        self.assertEqual(accumulator.total_count, 512)
        self.assertEqual(accumulator.passed_count, 400)


class BehaviorHistoryConfigTest(unittest.TestCase):
    def test_capacities_must_be_positive_and_ordered(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorHistoryConfig(completion_queue_capacity=0)
        with self.assertRaises(ValueError):
            BehaviorHistoryConfig(
                completion_outbox_recovery_capacity=2,
                completion_outbox_soft_capacity=2,
                completion_outbox_hard_capacity=4,
            )
        with self.assertRaises(ValueError):
            BehaviorHistoryConfig(active_metric_sample_capacity=3)


class CompletedBoutFinalizationTest(unittest.TestCase):
    def test_feeding_finalizes_once_with_live_evidence_units(self) -> None:
        analyzer = TemporalBehaviorAnalyzer(
            BehaviorObserverConfig(feeding_display_seconds=0.01)
        )
        analyzer.process(
            observation(
                0.0,
                consumption_count=1,
                consumed_energy=0.2,
            )
        )
        analyzer.process(
            observation(
                0.02,
                consumption_count=1,
                consumed_energy=0.2,
            )
        )

        completed = analyzer.drain_completed_bouts()

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].behavior, BehaviorKind.FEEDING)
        self.assertEqual(
            completed[0].outcome,
            BehaviorOutcome.CONSUMPTION_EVENT,
        )
        self.assertEqual(
            {item.unit for item in completed[0].evidence_summary},
            {"events", "energy"},
        )
        evidence = {
            item.key: item for item in completed[0].evidence_summary
        }
        self.assertEqual(
            evidence["food_consumption_event"].total_value,
            1.0,
        )
        self.assertAlmostEqual(
            evidence["energy_swallowed"].total_value,
            0.2,
        )
        self.assertEqual(analyzer.drain_completed_bouts(), ())

    def test_focus_loss_finalizes_active_not_emerging(self) -> None:
        analyzer = TemporalBehaviorAnalyzer(BehaviorObserverConfig())
        for index, distance in enumerate(
            (150, 140, 129, 117, 105, 92, 80, 68)
        ):
            analyzer.process(
                observation(
                    index * 0.1,
                    food_id=4,
                    food_distance=distance,
                    food_angle=0.0,
                )
            )

        analyzer.force_finalize(BehaviorTermination.FOCUS_CHANGED)

        approach = next(
            item
            for item in analyzer.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_APPROACH
        )
        self.assertEqual(
            approach.termination,
            BehaviorTermination.FOCUS_CHANGED,
        )
        self.assertEqual(approach.outcome, BehaviorOutcome.INTERRUPTED)
        emerging = TemporalBehaviorAnalyzer(
            BehaviorObserverConfig(bout_start_seconds=0.5)
        )
        emerging.process(observation(0.0, speed=0.0))
        emerging.force_finalize(BehaviorTermination.CREATURE_DIED)
        self.assertEqual(emerging.drain_completed_bouts(), ())

        death = TemporalBehaviorAnalyzer(BehaviorObserverConfig())
        for index, distance in enumerate(
            (150, 140, 129, 117, 105, 92, 80, 68)
        ):
            death.process(
                observation(
                    index * 0.1,
                    food_id=4,
                    food_distance=distance,
                    food_angle=0.0,
                )
            )
        death.force_finalize(BehaviorTermination.CREATURE_DIED)
        death_approach = next(
            item
            for item in death.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_APPROACH
        )
        self.assertEqual(
            death_approach.termination,
            BehaviorTermination.CREATURE_DIED,
        )
        self.assertEqual(
            death_approach.outcome,
            BehaviorOutcome.INTERRUPTED,
        )

    def test_overlapping_orientation_can_end_into_active_approach(self) -> None:
        analyzer = TemporalBehaviorAnalyzer(
            BehaviorObserverConfig(bout_end_grace_seconds=0.25)
        )
        snapshot = None
        angles = (0.8, 0.68, 0.55, 0.43, 0.31, 0.20, 0.10, 0.04)
        for index, angle in enumerate(angles):
            snapshot = analyzer.process(
                observation(
                    index * 0.1,
                    food_id=9,
                    food_distance=150.0 - index * 12.0,
                    food_angle=angle,
                    angular_velocity=0.8,
                )
            )
        assert snapshot is not None
        active = {
            state.behavior
            for state in snapshot.behaviors
            if state.status is BoutStatus.ACTIVE
        }
        self.assertIn(BehaviorKind.FOOD_ORIENTATION, active)
        self.assertIn(BehaviorKind.FOOD_APPROACH, active)

        for index in range(8, 12):
            snapshot = analyzer.process(
                observation(
                    index * 0.1,
                    food_id=9,
                    food_distance=150.0 - index * 12.0,
                    food_angle=0.04,
                    angular_velocity=0.0,
                )
            )

        completed = analyzer.drain_completed_bouts()
        orientation = next(
            item
            for item in completed
            if item.behavior is BehaviorKind.FOOD_ORIENTATION
        )
        self.assertEqual(
            orientation.outcome,
            BehaviorOutcome.APPROACH_STARTED,
        )
        assert snapshot is not None
        self.assertTrue(
            any(
                state.behavior is BehaviorKind.FOOD_APPROACH
                and state.status is BoutStatus.ACTIVE
                for state in snapshot.behaviors
            )
        )
        analyzer.force_finalize(BehaviorTermination.FOCUS_CHANGED)
        approach = next(
            item
            for item in analyzer.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_APPROACH
        )
        self.assertLess(
            max(orientation.start_time, approach.start_time),
            min(orientation.end_time, approach.end_time),
        )

    def test_food_approach_natural_outcomes_use_final_live_context(self) -> None:
        def active_analyzer() -> TemporalBehaviorAnalyzer:
            analyzer = TemporalBehaviorAnalyzer(
                BehaviorObserverConfig(bout_end_grace_seconds=0.15)
            )
            for index, distance in enumerate(
                (150, 140, 129, 117, 105, 92, 80, 68)
            ):
                analyzer.process(
                    observation(
                        index * 0.1,
                        food_id=4,
                        food_distance=distance,
                        food_angle=0.0,
                    )
                )
            return analyzer

        target_lost = active_analyzer()
        target_lost.process(observation(0.8))
        target_lost.process(observation(1.0))
        lost = next(
            item
            for item in target_lost.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_APPROACH
        )
        self.assertEqual(lost.outcome, BehaviorOutcome.TARGET_LOST)
        evidence = {item.key: item for item in lost.evidence_summary}
        self.assertEqual(evidence["closing_speed"].unit, "px/s")
        self.assertEqual(evidence["movement_toward_food"].unit, "cos")
        self.assertEqual(
            set(evidence),
            {
                "food_visibility",
                "target_persistent",
                "closing_speed",
                "closing_consistency",
                "movement_toward_food",
            },
        )

        consumed = active_analyzer()
        consumed.process(
            observation(
                0.8,
                consumption_count=1,
                consumed_energy=0.1,
            )
        )
        consumed.process(
            observation(
                1.0,
                consumption_count=1,
                consumed_energy=0.1,
            )
        )
        eaten = next(
            item
            for item in consumed.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_APPROACH
        )
        self.assertEqual(eaten.outcome, BehaviorOutcome.FOOD_CONSUMED)

        abandoned = active_analyzer()
        for simulation_time in (0.8, 1.0):
            abandoned.process(
                observation(
                    simulation_time,
                    velocity_x=-20.0,
                    food_id=4,
                    food_distance=68.0,
                    food_angle=0.0,
                )
            )
        ended = next(
            item
            for item in abandoned.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_APPROACH
        )
        self.assertEqual(ended.outcome, BehaviorOutcome.ABANDONED)

    def test_orientation_and_pheromone_natural_outcomes(self) -> None:
        def active_orientation() -> TemporalBehaviorAnalyzer:
            analyzer = TemporalBehaviorAnalyzer(
                BehaviorObserverConfig(bout_end_grace_seconds=0.15)
            )
            for index, angle in enumerate(
                (0.8, 0.68, 0.55, 0.43, 0.31, 0.20, 0.10, 0.04)
            ):
                analyzer.process(
                    observation(
                        index * 0.1,
                        food_id=9,
                        food_distance=80.0,
                        food_angle=angle,
                        angular_velocity=0.8,
                    )
                )
            return analyzer

        without_approach = active_orientation()
        for simulation_time in (0.8, 1.0):
            without_approach.process(
                observation(
                    simulation_time,
                    food_id=9,
                    food_distance=80.0,
                    food_angle=0.04,
                    angular_velocity=0.0,
                )
            )
        ended = next(
            item
            for item in without_approach.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_ORIENTATION
        )
        self.assertEqual(
            ended.outcome,
            BehaviorOutcome.ENDED_WITHOUT_APPROACH,
        )
        orientation_evidence = {
            item.key: item for item in ended.evidence_summary
        }
        self.assertEqual(
            orientation_evidence["heading_error_reduction"].unit,
            "rad/s",
        )

        target_lost = active_orientation()
        target_lost.process(observation(0.8))
        target_lost.process(observation(1.0))
        lost = next(
            item
            for item in target_lost.drain_completed_bouts()
            if item.behavior is BehaviorKind.FOOD_ORIENTATION
        )
        self.assertEqual(lost.outcome, BehaviorOutcome.TARGET_LOST)

        reduced = TemporalBehaviorAnalyzer(
            BehaviorObserverConfig(bout_end_grace_seconds=0.15)
        )
        for index in range(8):
            local_red = 0.30 - index * 0.015
            reduced.process(
                observation(
                    index * 0.1,
                    red_here=local_red,
                    red_forward=local_red - 0.04,
                )
            )
        reduced.process(observation(0.8))
        reduced.process(observation(1.0))
        retreat = next(
            item
            for item in reduced.drain_completed_bouts()
            if item.behavior is BehaviorKind.PHEROMONE_GRADIENT_RESPONSE
        )
        self.assertEqual(
            retreat.outcome,
            BehaviorOutcome.PHEROMONE_DESCENT,
        )


class CreatureBehaviorHistoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = CreatureBehaviorHistoryStore(
            max_completed_bouts_per_creature=4,
            max_remembered_creatures=2,
            minimum_stable_bouts=3,
        )
        self.store.register_creature(7, "Seven", 0.0)

    def test_persistent_ids_and_one_lifetime_value_per_bout(self) -> None:
        first_draft = _draft(
            1,
            0.8,
            100,
            EffectDirection.SUPPORTIVE,
        )
        self.store.append_draft(first_draft)
        self.store.append_draft(
            _draft(2, 0.2, 5, EffectDirection.SUPPRESSIVE)
        )
        self.assertIsNone(self.store.append_draft(first_draft))

        report = self.store.report_for(7)
        assert report is not None
        self.assertEqual(
            [bout.bout_id for bout in report.completed_bouts],
            [1, 2],
        )
        why = report.summary.behaviors[0].why_summaries[0]
        self.assertEqual(why.median_bout_influence, 0.5)
        self.assertEqual(why.influence_label, InfluenceLabel.MODERATE)
        self.assertEqual(why.contributing_bout_count, 2)
        self.assertEqual(why.direction_counts.supportive, 1)
        self.assertEqual(why.direction_counts.suppressive, 1)
        diagnostics = self.store.diagnostics
        self.assertEqual(diagnostics.bout_finalizations, 2)
        self.assertEqual(diagnostics.why_summaries_finalized, 2)
        self.assertEqual(diagnostics.duplicate_completions_ignored, 1)

    def test_reselection_reuses_local_id_but_not_persistent_id(self) -> None:
        first = _draft(1, 0.8, 4, EffectDirection.SUPPORTIVE)
        second = replace(first, selection_generation=4)

        self.store.append_draft(first)
        self.store.append_draft(second)
        report = self.store.report_for(7)

        assert report is not None
        self.assertEqual(
            [bout.bout_id for bout in report.completed_bouts],
            [1, 2],
        )

    def test_prepared_summary_and_duplicate_memory_are_bounded(self) -> None:
        self.store.append_draft(
            _draft(1, 0.4, 3, EffectDirection.SUPPORTIVE)
        )
        first_report = self.store.report_for(7)
        second_report = self.store.report_for(7)

        assert first_report is not None and second_report is not None
        self.assertIs(first_report.summary, second_report.summary)
        for local_id in range(2, 30):
            self.store.append_draft(
                _draft(
                    local_id,
                    0.4,
                    3,
                    EffectDirection.SUPPORTIVE,
                )
            )
        self.assertLessEqual(
            len(self.store._seen_sources),
            self.store._seen_source_capacity,
        )

    def test_missing_why_uses_behavior_and_contributing_denominators(
        self,
    ) -> None:
        self.store.append_draft(
            _draft(1, 0.7, 5, EffectDirection.SUPPORTIVE)
        )
        self.store.append_draft(
            replace(
                _draft(2, 0.2, 5, EffectDirection.SUPPRESSIVE),
                why_summary=None,
            )
        )

        report = self.store.report_for(7)

        assert report is not None
        why = report.summary.behaviors[0].why_summaries[0]
        self.assertEqual(why.behavior_bout_count, 2)
        self.assertEqual(why.contributing_bout_count, 1)
        self.assertEqual(why.direction_counts.total, 1)

    def test_lifetime_why_propagates_estimated_quantile_provenance(self) -> None:
        self.store.append_draft(
            _draft(
                1,
                0.7,
                600,
                EffectDirection.SUPPORTIVE,
                quantiles_estimated=True,
            )
        )
        self.store.append_draft(
            _draft(2, 0.2, 5, EffectDirection.SUPPRESSIVE)
        )

        report = self.store.report_for(7)

        assert report is not None
        why = report.summary.behaviors[0].why_summaries[0]
        self.assertTrue(why.quantiles_estimated)

    def test_capacity_and_incomplete_state_round_trip(self) -> None:
        for local_id in range(1, 7):
            self.store.append_draft(
                _draft(
                    local_id,
                    0.4,
                    3,
                    EffectDirection.SUPPORTIVE,
                )
            )
        self.store.mark_deceased(7, 9.0)
        self.store.mark_incomplete(3)
        restored = CreatureBehaviorHistoryStore(
            max_completed_bouts_per_creature=4,
            max_remembered_creatures=2,
            minimum_stable_bouts=3,
        )

        restored.restore_state(self.store.state_dict())
        report = restored.report_for(7)

        assert report is not None
        self.assertEqual(len(report.completed_bouts), 4)
        self.assertEqual(report.completed_bouts[0].bout_id, 3)
        self.assertTrue(report.deceased)
        self.assertTrue(report.history_incomplete)
        self.assertEqual(report.history_completions_not_recorded, 3)
        restored.record_skipped_completions(2)
        self.assertEqual(
            restored.diagnostics.history_completions_not_recorded,
            5,
        )

    def test_least_recent_creature_is_evicted_first(self) -> None:
        self.store.register_creature(8, "Eight", 1.0)
        self.store.register_creature(7, "Seven", 2.0)
        self.store.register_creature(9, "Nine", 3.0)

        self.assertEqual(
            [entry.creature_id for entry in self.store.index],
            [9, 7],
        )
        self.assertIsNone(self.store.report_for(8))
        self.assertEqual(self.store.diagnostics.creatures_evicted, 1)

    def test_missing_legacy_state_restores_empty_complete_history(self) -> None:
        self.store.append_draft(
            _draft(1, 0.7, 5, EffectDirection.SUPPORTIVE)
        )
        self.store.mark_incomplete(2)

        self.store.restore_state({})

        self.assertEqual(self.store.index, ())
        self.assertFalse(self.store.diagnostics.history_incomplete)
        self.assertEqual(
            self.store.diagnostics.history_completions_not_recorded,
            0,
        )

    def test_species_coverage_progress_and_aggregate_rate(self) -> None:
        self.store.register_creature(
            7,
            "Seven",
            0.0,
            species_id=3,
            observation_mode="automatic",
            observation_generation=9,
            active=True,
        )
        self.store.record_observation_progress(7, 9, 0.0, 1)
        self.store.record_observation_progress(7, 9, 10.0, 101)
        self.store.append_draft(
            _draft(1, 0.4, 3, EffectDirection.SUPPORTIVE)
        )

        report = self.store.species_report(3)

        self.assertEqual(report.observed_creature_count, 1)
        self.assertEqual(report.total_observation_seconds, 10.0)
        self.assertEqual(report.completed_bout_count, 1)
        self.assertAlmostEqual(
            report.behaviors[0].bouts_per_creature_hour,
            360.0,
        )

    def test_late_progress_from_paused_session_is_counted_once(self) -> None:
        self.store.register_creature(
            7,
            "Seven",
            0.0,
            species_id=3,
            observation_mode="automatic",
            observation_generation=9,
            active=True,
        )
        self.store.record_observation_progress(7, 9, 0.0, 1)
        self.store.set_active_creatures(set())
        self.store.register_creature(
            7,
            "Seven",
            10.0,
            species_id=3,
            observation_mode="focal",
            observation_generation=10,
            active=True,
        )
        self.store.record_observation_progress(7, 10, 10.0, 1)
        self.store.record_observation_progress(7, 9, 8.0, 81)
        self.store.record_observation_progress(7, 9, 8.0, 81)
        self.store.record_observation_progress(7, 10, 12.0, 21)

        report = self.store.report_for(7)

        assert report is not None
        self.assertEqual(report.total_observation_seconds, 10.0)
        self.assertEqual(report.observation_session_count, 2)

    def test_legacy_species_is_inferred_or_stays_under_unknown_species(
        self,
    ) -> None:
        self.store.register_creature(7, "Seven", 0.0)
        self.store.register_creature(8, "Eight", 1.0)
        legacy_state = self.store.state_dict()
        for creature in legacy_state["creatures"]:
            creature.pop("species_id")
        restored = CreatureBehaviorHistoryStore(
            max_completed_bouts_per_creature=4,
            max_remembered_creatures=2,
            minimum_stable_bouts=3,
        )

        restored.restore_state(legacy_state)
        restored.assign_missing_species({7: 4})

        self.assertEqual(
            restored.species_report(None).observed_creature_count,
            1,
        )
        self.assertEqual(
            restored.species_report(4).observed_creature_count,
            1,
        )

    def test_active_records_are_protected_from_historical_retention(self) -> None:
        self.store.register_creature(
            8,
            "Eight",
            1.0,
            species_id=1,
            active=True,
            observation_generation=1,
        )
        self.store.register_creature(9, "Nine", 2.0)
        self.store.register_creature(10, "Ten", 3.0)
        self.store.register_creature(11, "Eleven", 4.0)

        self.assertIsNotNone(self.store.report_for(8))
        inactive = [entry for entry in self.store.index if not entry.active]
        self.assertLessEqual(len(inactive), 2)

    def test_unobserved_death_does_not_create_history_entry(self) -> None:
        self.store.mark_deceased(99, 4.0)
        self.assertIsNone(self.store.report_for(99))


class CompletionBackpressureTest(unittest.TestCase):
    def _service(self, hard_capacity: int) -> BehaviorObserverService:
        service = BehaviorObserverService(
            BehaviorObserverConfig(
                input_queue_capacity=64,
                result_queue_capacity=8,
                feeding_display_seconds=0.01,
            ),
            CounterfactualWhyConfig(enabled=False),
            BehaviorHistoryConfig(
                completion_queue_capacity=1,
                completion_outbox_soft_capacity=2,
                completion_outbox_hard_capacity=hard_capacity,
                completion_outbox_recovery_capacity=1,
            ),
        )
        service.set_focus(1, 1)
        return service

    @staticmethod
    def _submit_without_draining(
        service: BehaviorObserverService,
        count: int,
    ) -> None:
        for sample in _feeding_observations(count):
            service.submit(sample)
        time.sleep(0.35)

    @staticmethod
    def _drain(service: BehaviorObserverService) -> tuple[object, ...]:
        completed = []
        deadline = time.monotonic() + 5.0
        stable_empty_polls = 0
        while time.monotonic() < deadline:
            service.poll()
            completed.extend(service.drain_completed_bouts())
            diagnostics = service.diagnostics
            accepted_observations = (
                diagnostics.samples_produced
                - diagnostics.samples_dropped
            )
            if (
                diagnostics.completion_outbox_depth == 0
                and not diagnostics.completion_recording_suspended
                and diagnostics.observations_processed
                >= accepted_observations
            ):
                stable_empty_polls += 1
                if stable_empty_polls >= 5:
                    break
            else:
                stable_empty_polls = 0
            time.sleep(0.02)
        return tuple(completed)

    def test_temporary_saturation_recovers_fifo_without_loss(self) -> None:
        service = self._service(16)
        try:
            self._submit_without_draining(service, 4)
            service.poll()
            temporal_diagnostics = service.diagnostics
            self.assertEqual(tuple(service._completed_bouts), ())
            completed = self._drain(service)
            diagnostics = service.diagnostics
        finally:
            service.close()

        self.assertEqual(
            [item.local_bout_id for item in completed],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            len({item.local_bout_id for item in completed}),
            4,
        )
        self.assertEqual(diagnostics.history_completions_not_recorded, 0)
        self.assertTrue(diagnostics.completion_outbox_warning)
        self.assertGreaterEqual(
            temporal_diagnostics.observations_processed,
            8,
        )

    def test_hard_bound_counts_skips_and_recovers(self) -> None:
        service = self._service(4)
        try:
            self._submit_without_draining(service, 10)
            completed = list(self._drain(service))
            diagnostics = service.diagnostics
            for sample in _feeding_observations(1):
                service.submit(
                    observation(
                        sample.simulation_time + 2.0,
                        consumption_count=(
                            sample.food_consumption_count + 10
                        ),
                        consumed_energy=(
                            sample.food_consumed_energy_total + 1.0
                        ),
                    )
            )
            time.sleep(0.15)
            completed.extend(self._drain(service))
            recovered_diagnostics = service.diagnostics
        finally:
            service.close()

        ids = [item.local_bout_id for item in completed]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn(11, ids)
        self.assertGreater(
            diagnostics.history_completions_not_recorded,
            0,
        )
        self.assertLessEqual(
            diagnostics.completion_outbox_high_water,
            4,
        )
        self.assertGreaterEqual(diagnostics.observations_processed, 20)
        self.assertFalse(
            recovered_diagnostics.completion_recording_suspended
        )


if __name__ == "__main__":
    unittest.main()
