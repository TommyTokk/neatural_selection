from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from math import pi
import pickle
import time
import unittest
from types import SimpleNamespace

from configs.sim_config import (
    BehaviorObserverConfig,
    CounterfactualWhyConfig,
)
from src.action import ACTION_OUTPUT_COUNT, ACTION_OUTPUT_NAMES
from src.behavior_observer import (
    BehaviorKind,
    BehaviorObservation,
    BehaviorObserverService,
    BoutStatus,
)
from src.counterfactual_neat import (
    BEHAVIOR_EXPLANATION_SPECS,
    INTERVENTION_REPLACEMENTS,
    CounterfactualBoutAggregator,
    CounterfactualProbeInput,
    CounterfactualProbeJob,
    EffectDirection,
    FocalBrainUpdate,
    ProbeBehavior,
    PureNeatEvaluator,
    SemanticIntervention,
    apply_intervention,
    mapped_probe_behaviors,
    semantic_effect,
    steering_toward_target,
    validate_probe,
    _dominant_direction,
)
from src.neat_brain import NeatBrain
from src.vision import SENSOR_CONTRACT, SENSOR_INPUT_NAMES, SensorSnapshot


class FixedNetwork:
    def __init__(self, outputs: tuple[float, ...]) -> None:
        self.outputs = outputs
        self.calls = 0

    def activate(self, inputs) -> tuple[float, ...]:
        self.calls += 1
        return self.outputs


class SlowNetwork(FixedNetwork):
    def __init__(
        self,
        outputs: tuple[float, ...],
        delay_seconds: float,
    ) -> None:
        super().__init__(outputs)
        self.delay_seconds = delay_seconds

    def activate(self, inputs) -> tuple[float, ...]:
        time.sleep(self.delay_seconds)
        return super().activate(inputs)


class SemanticInterventionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = tuple(
            (index + 1) / 100.0 for index in range(len(SENSOR_INPUT_NAMES))
        )

    def test_each_intervention_changes_only_its_named_sensors(self) -> None:
        for intervention, replacements in INTERVENTION_REPLACEMENTS.items():
            with self.subTest(intervention=intervention):
                changed = apply_intervention(intervention, self.inputs)
                changed_names = {
                    name
                    for index, name in enumerate(SENSOR_INPUT_NAMES)
                    if changed[index] != self.inputs[index]
                }
                self.assertEqual(changed_names, set(replacements))

    def test_visible_food_and_resource_gradients_are_disjoint(self) -> None:
        visible = set(
            INTERVENTION_REPLACEMENTS[
                SemanticIntervention.VISIBLE_FOOD_CUES
            ]
        )
        resource = set(
            INTERVENTION_REPLACEMENTS[
                SemanticIntervention.RESOURCE_GRADIENT_CUES
            ]
        )

        self.assertEqual(
            visible,
            {"food_count", "food_proximity", "food_angle"},
        )
        self.assertEqual(
            resource,
            {"local_richness", "lateral_gradient", "forward_gradient"},
        )
        self.assertTrue(visible.isdisjoint(resource))

    def test_satiated_state_matches_live_feeding_drive_formula(self) -> None:
        replacements = INTERVENTION_REPLACEMENTS[
            SemanticIntervention.SATIATED_STATE
        ]
        expected_drive = (
            (1.0 - replacements["energy_percent"])
            * (1.0 - replacements["stomach_fullness"])
        )

        self.assertEqual(replacements["feeding_drive"], expected_drive)
        self.assertEqual(replacements["energy_percent"], 1.0)
        self.assertEqual(replacements["stomach_fullness"], 1.0)
        metabolic_names = {
            name
            for intervention in SemanticIntervention
            for name in INTERVENTION_REPLACEMENTS[intervention]
            if name in {"feeding_drive", "energy_percent", "stomach_fullness"}
        }
        self.assertEqual(
            metabolic_names,
            {"feeding_drive", "energy_percent", "stomach_fullness"},
        )


class PureEvaluationTest(unittest.TestCase):
    def test_evaluate_pure_does_not_mutate_live_brain_state(self) -> None:
        network = FixedNetwork(tuple([0.6] * ACTION_OUTPUT_COUNT))
        action = SimpleNamespace(marker="live")
        brain = NeatBrain(
            genome_id=1,
            genome=SimpleNamespace(),
            network=network,
            output_activations=["clamped"] * ACTION_OUTPUT_COUNT,
            last_inputs=[0.1, 0.2],
            last_outputs=[0.3] * ACTION_OUTPUT_COUNT,
            last_action=action,
        )
        brain.herding_state = 0.45
        brain.last_raw_herding = 0.55

        result = brain.evaluate_pure(tuple([0.0] * len(SENSOR_INPUT_NAMES)))

        self.assertEqual(result, tuple([0.6] * ACTION_OUTPUT_COUNT))
        self.assertEqual(brain.last_inputs, [0.1, 0.2])
        self.assertEqual(brain.last_outputs, [0.3] * ACTION_OUTPUT_COUNT)
        self.assertIs(brain.last_action, action)
        self.assertEqual(brain.herding_state, 0.45)
        self.assertEqual(brain.last_raw_herding, 0.55)


class CounterfactualScoringTest(unittest.TestCase):
    @staticmethod
    def outputs(**values: float) -> tuple[float, ...]:
        outputs = [0.0] * ACTION_OUTPUT_COUNT
        for name, value in values.items():
            outputs[ACTION_OUTPUT_NAMES.index(name)] = value
        return tuple(outputs)

    def test_food_orientation_scores_rotate_and_displays_acceleration(self) -> None:
        effect = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=0.8, accelerate=0.9),
            self.outputs(rotate=0.2, accelerate=-0.9),
            target_visible=True,
            food_relative_angle=0.4,
        )

        self.assertAlmostEqual(effect.influence_score, 0.3)
        by_name = {
            output.output_name: output for output in effect.output_effects
        }
        self.assertFalse(by_name["rotate"].secondary_context)
        self.assertTrue(by_name["accelerate"].secondary_context)

    def test_conflicting_relevant_outputs_produce_mixed_direction(self) -> None:
        effect = semantic_effect(
            BehaviorKind.FOOD_APPROACH,
            SemanticIntervention.SOCIAL_CUES,
            self.outputs(accelerate=0.70, rotate=0.20),
            self.outputs(accelerate=0.30, rotate=0.55),
            target_visible=True,
            food_relative_angle=0.4,
        )
        directions = {
            output.output_name: output.direction
            for output in effect.output_effects
        }

        self.assertEqual(
            directions,
            {
                "accelerate": EffectDirection.SUPPORTIVE,
                "rotate": EffectDirection.SUPPRESSIVE,
            },
        )
        self.assertIs(effect.effect_direction, EffectDirection.MIXED)

    def test_material_signed_reversal_has_aggregate_precedence(self) -> None:
        effect = semantic_effect(
            BehaviorKind.FOOD_APPROACH,
            SemanticIntervention.WALL_CUES,
            self.outputs(accelerate=0.7, rotate=0.6),
            self.outputs(accelerate=0.2, rotate=-0.4),
            target_visible=True,
            food_relative_angle=0.4,
        )

        self.assertIs(effect.effect_direction, EffectDirection.REVERSING)

    def test_target_steering_uses_project_sign_conventions(self) -> None:
        self.assertGreater(
            steering_toward_target(0.6, 0.3, 0.05),
            0.0,
        )
        self.assertGreater(
            steering_toward_target(-0.6, -0.3, 0.05),
            0.0,
        )
        self.assertLess(
            steering_toward_target(-0.6, 0.3, 0.05),
            0.0,
        )
        self.assertLess(
            steering_toward_target(0.6, -0.3, 0.05),
            0.0,
        )

    def test_food_steering_reverses_for_left_and_right_targets(self) -> None:
        for target_angle, factual, counterfactual in (
            (0.3, 0.7, -0.5),
            (-0.3, -0.7, 0.5),
        ):
            with self.subTest(target_angle=target_angle):
                effect = semantic_effect(
                    BehaviorKind.FOOD_ORIENTATION,
                    SemanticIntervention.VISIBLE_FOOD_CUES,
                    self.outputs(rotate=factual),
                    self.outputs(rotate=counterfactual),
                    target_visible=True,
                    food_relative_angle=target_angle,
                )
                rotate = next(
                    output
                    for output in effect.output_effects
                    if output.output_name == "rotate"
                )
                self.assertIs(
                    rotate.direction,
                    EffectDirection.REVERSING,
                )
                self.assertIs(
                    effect.effect_direction,
                    EffectDirection.REVERSING,
                )

    def test_centered_target_treats_small_rotation_as_stabilizing(self) -> None:
        effect = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=-0.04),
            self.outputs(rotate=0.86),
            target_visible=True,
            food_relative_angle=0.01,
        )
        rotate = next(
            output
            for output in effect.output_effects
            if output.output_name == "rotate"
        )

        self.assertEqual(rotate.actual, -0.04)
        self.assertEqual(rotate.counterfactual, 0.86)
        self.assertAlmostEqual(rotate.delta, -0.90)
        self.assertAlmostEqual(rotate.influence_score, 0.45)
        self.assertIs(rotate.direction, EffectDirection.SUPPORTIVE)
        self.assertIs(
            effect.effect_direction,
            EffectDirection.SUPPORTIVE,
        )

    def test_centered_target_stabilizes_against_either_turn_direction(
        self,
    ) -> None:
        for counterfactual in (-0.8, 0.8):
            with self.subTest(counterfactual=counterfactual):
                effect = semantic_effect(
                    BehaviorKind.FOOD_ORIENTATION,
                    SemanticIntervention.VISIBLE_FOOD_CUES,
                    self.outputs(rotate=0.0),
                    self.outputs(rotate=counterfactual),
                    target_visible=True,
                    food_relative_angle=0.0,
                )
                self.assertIs(
                    effect.effect_direction,
                    EffectDirection.SUPPORTIVE,
                )

    def test_counterfactual_can_improve_target_correction(self) -> None:
        effect = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=0.2),
            self.outputs(rotate=0.8),
            target_visible=True,
            food_relative_angle=0.3,
        )

        self.assertIs(
            effect.effect_direction,
            EffectDirection.SUPPRESSIVE,
        )

    def test_factual_steering_can_be_better_toward_or_less_bad_away(
        self,
    ) -> None:
        for factual, counterfactual in ((0.8, 0.2), (-0.2, -0.8)):
            with self.subTest(
                factual=factual,
                counterfactual=counterfactual,
            ):
                effect = semantic_effect(
                    BehaviorKind.FOOD_ORIENTATION,
                    SemanticIntervention.VISIBLE_FOOD_CUES,
                    self.outputs(rotate=factual),
                    self.outputs(rotate=counterfactual),
                    target_visible=True,
                    food_relative_angle=0.3,
                )
                self.assertIs(
                    effect.effect_direction,
                    EffectDirection.SUPPORTIVE,
                )

    def test_center_dead_zone_and_reversal_noise_boundaries(self) -> None:
        centered = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=0.0),
            self.outputs(rotate=0.8),
            target_visible=True,
            food_relative_angle=0.05,
            target_center_dead_zone=0.05,
        )
        outside = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=0.0),
            self.outputs(rotate=0.8),
            target_visible=True,
            food_relative_angle=0.050001,
            target_center_dead_zone=0.05,
        )
        below_noise = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=0.049),
            self.outputs(rotate=-0.5),
            target_visible=True,
            food_relative_angle=0.3,
        )
        at_noise = semantic_effect(
            BehaviorKind.FOOD_ORIENTATION,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(rotate=0.05),
            self.outputs(rotate=-0.5),
            target_visible=True,
            food_relative_angle=0.3,
        )

        self.assertIs(
            centered.effect_direction,
            EffectDirection.SUPPORTIVE,
        )
        self.assertIs(
            outside.effect_direction,
            EffectDirection.SUPPRESSIVE,
        )
        self.assertIs(
            below_noise.effect_direction,
            EffectDirection.SUPPORTIVE,
        )
        self.assertIs(
            at_noise.effect_direction,
            EffectDirection.REVERSING,
        )

    def test_resting_has_no_neural_explanation_mapping(self) -> None:
        self.assertNotIn(
            BehaviorKind.RESTING,
            BEHAVIOR_EXPLANATION_SPECS,
        )
        state = SimpleNamespace(
            behavior=BehaviorKind.RESTING,
            status=BoutStatus.ACTIVE,
            bout_id=2,
            duration_seconds=1.2,
        )
        self.assertEqual(mapped_probe_behaviors((state,)), ())

    def test_food_effect_requires_visible_factual_target_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "visible factual food target"):
            semantic_effect(
                BehaviorKind.FOOD_ORIENTATION,
                SemanticIntervention.VISIBLE_FOOD_CUES,
                self.outputs(rotate=0.5),
                self.outputs(rotate=-0.5),
            )

    def test_cohesion_movement_uses_factual_group_heading(self) -> None:
        effect = semantic_effect(
            BehaviorKind.COHESION,
            SemanticIntervention.SOCIAL_CUES,
            self.outputs(accelerate=0.7, rotate=0.2),
            self.outputs(accelerate=0.3, rotate=0.55),
            group_visible=True,
            group_relative_angle=0.4,
        )
        by_name = {
            output.output_name: output for output in effect.output_effects
        }

        self.assertIs(
            by_name["accelerate"].direction,
            EffectDirection.SUPPORTIVE,
        )
        self.assertIs(
            by_name["rotate"].direction,
            EffectDirection.SUPPRESSIVE,
        )
        self.assertIs(effect.effect_direction, EffectDirection.MIXED)

    def test_food_acceleration_is_aligned_with_factual_target(self) -> None:
        ahead = semantic_effect(
            BehaviorKind.FOOD_APPROACH,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(accelerate=0.7),
            self.outputs(accelerate=0.2),
            target_visible=True,
            food_relative_angle=0.0,
        )
        behind = semantic_effect(
            BehaviorKind.FOOD_APPROACH,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            self.outputs(accelerate=-0.7),
            self.outputs(accelerate=0.2),
            target_visible=True,
            food_relative_angle=pi,
        )

        ahead_acceleration = next(
            output
            for output in ahead.output_effects
            if output.output_name == "accelerate"
        )
        behind_acceleration = next(
            output
            for output in behind.output_effects
            if output.output_name == "accelerate"
        )
        self.assertIs(
            ahead_acceleration.direction,
            EffectDirection.SUPPORTIVE,
        )
        self.assertIs(
            behind_acceleration.direction,
            EffectDirection.REVERSING,
        )

    def test_cohesion_turning_handles_left_right_and_behind_groups(self) -> None:
        for group_angle, factual_rotate in (
            (pi / 2.0, 0.7),
            (-pi / 2.0, -0.7),
        ):
            with self.subTest(group_angle=group_angle):
                effect = semantic_effect(
                    BehaviorKind.COHESION,
                    SemanticIntervention.SOCIAL_CUES,
                    self.outputs(rotate=factual_rotate),
                    self.outputs(rotate=-factual_rotate),
                    group_visible=True,
                    group_relative_angle=group_angle,
                )
                rotate = next(
                    output
                    for output in effect.output_effects
                    if output.output_name == "rotate"
                )
                self.assertIs(rotate.direction, EffectDirection.REVERSING)

        behind = semantic_effect(
            BehaviorKind.COHESION,
            SemanticIntervention.SOCIAL_CUES,
            self.outputs(accelerate=-0.7),
            self.outputs(accelerate=0.3),
            group_visible=True,
            group_relative_angle=pi,
        )
        acceleration = next(
            output
            for output in behind.output_effects
            if output.output_name == "accelerate"
        )
        self.assertIs(acceleration.direction, EffectDirection.REVERSING)

        side_acceleration = semantic_effect(
            BehaviorKind.COHESION,
            SemanticIntervention.SOCIAL_CUES,
            self.outputs(accelerate=0.7),
            self.outputs(accelerate=-0.7),
            group_visible=True,
            group_relative_angle=pi / 2.0,
        )
        acceleration = next(
            output
            for output in side_acceleration.output_effects
            if output.output_name == "accelerate"
        )
        self.assertIs(acceleration.direction, EffectDirection.MINIMAL)

    def test_alarm_retreat_prefers_forward_acceleration_and_stable_heading(
        self,
    ) -> None:
        supportive = semantic_effect(
            BehaviorKind.ALARM_RETREAT,
            SemanticIntervention.ALARM_PHEROMONE_CUES,
            self.outputs(accelerate=0.7, rotate=0.1),
            self.outputs(accelerate=0.2, rotate=0.7),
        )
        suppressive = semantic_effect(
            BehaviorKind.ALARM_RETREAT,
            SemanticIntervention.ALARM_PHEROMONE_CUES,
            self.outputs(accelerate=0.2, rotate=0.7),
            self.outputs(accelerate=0.7, rotate=0.1),
        )

        self.assertIs(
            supportive.effect_direction,
            EffectDirection.SUPPORTIVE,
        )
        self.assertIs(
            suppressive.effect_direction,
            EffectDirection.SUPPRESSIVE,
        )

    def test_mapped_food_behavior_preserves_target_identity(self) -> None:
        state = SimpleNamespace(
            behavior=BehaviorKind.FOOD_APPROACH,
            status=BoutStatus.ACTIVE,
            bout_id=3,
            duration_seconds=1.2,
            target_id=17,
        )

        mapped = mapped_probe_behaviors((state,))

        self.assertEqual(mapped[0].target_id, 17)


class CounterfactualAggregationTest(unittest.TestCase):
    def test_history_is_bounded_and_uses_medians(self) -> None:
        aggregator = CounterfactualBoutAggregator(history_capacity=5)
        actual = CounterfactualScoringTest.outputs(rotate=0.8)
        observed = ProbeBehavior(
            BehaviorKind.FOOD_ORIENTATION,
            BoutStatus.ACTIVE,
            3,
            1.0,
            target_id=9,
        )
        values = (0.19, 0.16, 0.21, -0.15, 0.18, 0.17)
        latest = None
        for index, counterfactual_rotate in enumerate(values):
            probe = CounterfactualProbeInput(
                creature_id=1,
                selection_generation=2,
                brain_revision=4,
                simulation_time=index / 5.0,
                sensor_schema_version=SENSOR_CONTRACT.schema_version,
                behaviors=(observed,),
                actual_inputs=tuple([0.0] * len(SENSOR_INPUT_NAMES)),
                actual_outputs=actual,
                submitted_monotonic=time.monotonic(),
                target_visible=True,
                food_target_id=9,
                food_relative_angle=0.4,
            )
            job = CounterfactualProbeJob(
                probe,
                PureNeatEvaluator(
                    FixedNetwork(tuple([0.0] * ACTION_OUTPUT_COUNT)),
                    tuple(["clamped"] * ACTION_OUTPUT_COUNT),
                ),
            )
            counterfactual = CounterfactualScoringTest.outputs(
                rotate=counterfactual_rotate
            )
            job.outputs = {
                intervention: counterfactual
                for intervention in job.interventions
            }
            job._next_index = len(job.interventions)
            latest = aggregator.complete_job(job)[0]

        self.assertIsNotNone(latest)
        visible_food = next(
            effect
            for effect in latest.effects
            if effect.intervention
            is SemanticIntervention.VISIBLE_FOOD_CUES
        )
        self.assertEqual(visible_food.sample_count, 5)
        rotate = next(
            output
            for output in visible_food.output_effects
            if output.output_name == "rotate"
        )
        self.assertAlmostEqual(rotate.counterfactual, 0.17)

    def test_completed_summary_freezes_and_discards_probe_history(self) -> None:
        aggregator = CounterfactualBoutAggregator(history_capacity=8)
        observed = ProbeBehavior(
            BehaviorKind.FOOD_APPROACH,
            BoutStatus.ACTIVE,
            9,
            1.0,
            target_id=12,
        )
        for index, counterfactual_acceleration in enumerate((0.1, 0.2, 0.3)):
            probe = CounterfactualProbeInput(
                creature_id=3,
                selection_generation=4,
                brain_revision=5,
                simulation_time=index / 5.0,
                sensor_schema_version=SENSOR_CONTRACT.schema_version,
                behaviors=(observed,),
                actual_inputs=tuple([0.0] * len(SENSOR_INPUT_NAMES)),
                actual_outputs=CounterfactualScoringTest.outputs(
                    accelerate=0.8
                ),
                submitted_monotonic=time.monotonic(),
                target_visible=True,
                food_target_id=12,
                food_relative_angle=0.0,
            )
            job = CounterfactualProbeJob(
                probe,
                PureNeatEvaluator(
                    FixedNetwork(tuple([0.0] * ACTION_OUTPUT_COUNT)),
                    tuple(["clamped"] * ACTION_OUTPUT_COUNT),
                ),
            )
            counterfactual = CounterfactualScoringTest.outputs(
                accelerate=counterfactual_acceleration
            )
            job.outputs = {
                intervention: counterfactual
                for intervention in job.interventions
            }
            job._next_index = len(job.interventions)
            aggregator.complete_job(job)

        completed = aggregator.finalize_completed_bout(
            3,
            4,
            BehaviorKind.FOOD_APPROACH,
            9,
        )

        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertTrue(completed.effects)
        self.assertTrue(
            all(effect.sample_count == 3 for effect in completed.effects)
        )
        self.assertTrue(
            all(
                effect.direction_counts.total == 3
                for effect in completed.effects
            )
        )
        self.assertTrue(
            all(
                effect.p25 is not None and effect.p75 is not None
                for effect in completed.effects
            )
        )
        self.assertTrue(
            all(
                output.sample_count == 3
                and output.delta_p25 is not None
                and output.delta_p75 is not None
                for effect in completed.effects
                for output in effect.output_summaries
            )
        )
        self.assertIsNone(
            aggregator.finalize_completed_bout(
                3,
                4,
                BehaviorKind.FOOD_APPROACH,
                9,
            )
        )

    def test_alignment_medians_and_target_ids_isolate_histories(self) -> None:
        aggregator = CounterfactualBoutAggregator(history_capacity=8)

        def complete(
            target_id: int,
            target_angle: float,
            actual_rotate: float,
            counterfactual_rotate: float,
        ):
            observed = ProbeBehavior(
                BehaviorKind.FOOD_ORIENTATION,
                BoutStatus.ACTIVE,
                4,
                1.0,
                target_id=target_id,
            )
            probe = CounterfactualProbeInput(
                creature_id=1,
                selection_generation=2,
                brain_revision=4,
                simulation_time=1.0,
                sensor_schema_version=SENSOR_CONTRACT.schema_version,
                behaviors=(observed,),
                actual_inputs=tuple([0.0] * len(SENSOR_INPUT_NAMES)),
                actual_outputs=CounterfactualScoringTest.outputs(
                    rotate=actual_rotate
                ),
                submitted_monotonic=time.monotonic(),
                target_visible=True,
                food_target_id=target_id,
                food_relative_angle=target_angle,
            )
            job = CounterfactualProbeJob(
                probe,
                PureNeatEvaluator(
                    FixedNetwork(tuple([0.0] * ACTION_OUTPUT_COUNT)),
                    tuple(["clamped"] * ACTION_OUTPUT_COUNT),
                ),
            )
            counterfactual = CounterfactualScoringTest.outputs(
                rotate=counterfactual_rotate
            )
            job.outputs = {
                intervention: counterfactual
                for intervention in job.interventions
            }
            job._next_index = len(job.interventions)
            return aggregator.complete_job(job)[0]

        complete(9, 0.3, 0.8, 0.2)
        mirrored = complete(9, -0.3, -0.8, -0.2)
        visible = next(
            effect
            for effect in mirrored.effects
            if effect.intervention
            is SemanticIntervention.VISIBLE_FOOD_CUES
        )
        rotate = next(
            output
            for output in visible.output_effects
            if output.output_name == "rotate"
        )

        self.assertEqual(visible.sample_count, 2)
        self.assertAlmostEqual(rotate.actual, -0.8)
        self.assertAlmostEqual(rotate.counterfactual, -0.2)
        self.assertAlmostEqual(rotate.actual_target_alignment, 0.8)
        self.assertAlmostEqual(
            rotate.counterfactual_target_alignment,
            0.2,
        )
        self.assertIs(rotate.direction, EffectDirection.SUPPORTIVE)
        self.assertIs(
            visible.effect_direction,
            EffectDirection.SUPPORTIVE,
        )

        switched = complete(10, 0.3, 0.8, -0.2)
        switched_visible = next(
            effect
            for effect in switched.effects
            if effect.intervention
            is SemanticIntervention.VISIBLE_FOOD_CUES
        )
        self.assertEqual(switched.target_id, 10)
        self.assertEqual(switched_visible.sample_count, 1)

    def test_median_probe_preserves_one_coherent_transition(self) -> None:
        samples = deque(
            semantic_effect(
                BehaviorKind.FEEDING,
                SemanticIntervention.VISIBLE_FOOD_CUES,
                CounterfactualScoringTest.outputs(want_eat=actual),
                CounterfactualScoringTest.outputs(want_eat=counterfactual),
            )
            for actual, counterfactual in (
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
            )
        )

        aggregated = CounterfactualBoutAggregator._aggregate_samples(
            BehaviorKind.FEEDING,
            samples,
        )
        output = aggregated.output_effects[0]

        self.assertEqual(aggregated.sample_count, 3)
        self.assertEqual(aggregated.influence_score, output.influence_score)
        self.assertEqual(aggregated.effect_direction, output.direction)
        self.assertEqual(output.delta, output.actual - output.counterfactual)
        self.assertEqual((output.actual, output.counterfactual), (1.0, 1.0))

    def test_even_median_probe_tie_prefers_newest_sample(self) -> None:
        older = semantic_effect(
            BehaviorKind.FEEDING,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            CounterfactualScoringTest.outputs(want_eat=0.2),
            CounterfactualScoringTest.outputs(want_eat=0.0),
        )
        newer = semantic_effect(
            BehaviorKind.FEEDING,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            CounterfactualScoringTest.outputs(want_eat=0.8),
            CounterfactualScoringTest.outputs(want_eat=0.0),
        )

        aggregated = CounterfactualBoutAggregator._aggregate_samples(
            BehaviorKind.FEEDING,
            deque((older, newer)),
        )

        self.assertEqual(aggregated.influence_score, 0.8)
        self.assertEqual(aggregated.output_effects[0].actual, 0.8)

    def test_completed_direction_uses_dominant_non_minimal_count(self) -> None:
        self.assertIs(
            _dominant_direction(
                0.4,
                Counter(
                    {
                        EffectDirection.SUPPORTIVE: 2,
                        EffectDirection.SUPPRESSIVE: 1,
                    }
                ),
            ),
            EffectDirection.SUPPORTIVE,
        )
        self.assertIs(
            _dominant_direction(
                0.4,
                Counter(
                    {
                        EffectDirection.SUPPORTIVE: 1,
                        EffectDirection.SUPPRESSIVE: 1,
                    }
                ),
            ),
            EffectDirection.MIXED,
        )
        self.assertIs(
            _dominant_direction(
                0.05,
                Counter({EffectDirection.SUPPORTIVE: 10}),
            ),
            EffectDirection.MINIMAL,
        )


class CounterfactualProbeValidationTest(unittest.TestCase):
    @staticmethod
    def probe(
        behavior: BehaviorKind,
        *,
        behavior_target_id: int | None = None,
        target_visible: bool = False,
        food_target_id: int | None = None,
        food_relative_angle: float | None = None,
        group_visible: bool = False,
        group_relative_angle: float | None = None,
    ) -> CounterfactualProbeInput:
        return CounterfactualProbeInput(
            creature_id=1,
            selection_generation=1,
            brain_revision=1,
            simulation_time=0.0,
            sensor_schema_version=SENSOR_CONTRACT.schema_version,
            behaviors=(
                ProbeBehavior(
                    behavior,
                    BoutStatus.ACTIVE,
                    1,
                    0.5,
                    target_id=behavior_target_id,
                ),
            ),
            actual_inputs=tuple([0.0] * len(SENSOR_INPUT_NAMES)),
            actual_outputs=tuple([0.0] * ACTION_OUTPUT_COUNT),
            submitted_monotonic=time.monotonic(),
            target_visible=target_visible,
            food_target_id=food_target_id,
            food_relative_angle=food_relative_angle,
            group_visible=group_visible,
            group_relative_angle=group_relative_angle,
        )

    def test_target_behavior_requires_matching_visible_context(self) -> None:
        valid = self.probe(
            BehaviorKind.FOOD_APPROACH,
            behavior_target_id=8,
            target_visible=True,
            food_target_id=8,
            food_relative_angle=-0.2,
        )
        validate_probe(valid)

        for invalid in (
            self.probe(
                BehaviorKind.FOOD_APPROACH,
                behavior_target_id=8,
            ),
            self.probe(
                BehaviorKind.FOOD_APPROACH,
                behavior_target_id=8,
                target_visible=True,
                food_target_id=9,
                food_relative_angle=-0.2,
            ),
            self.probe(
                BehaviorKind.FOOD_APPROACH,
                behavior_target_id=8,
                target_visible=True,
                food_target_id=8,
                food_relative_angle=float("nan"),
            ),
        ):
            with self.subTest(probe=invalid):
                with self.assertRaises(ValueError):
                    validate_probe(invalid)

    def test_non_target_behavior_does_not_require_food_context(self) -> None:
        validate_probe(self.probe(BehaviorKind.FEEDING))

    def test_cohesion_requires_visible_flock_center_context(self) -> None:
        validate_probe(
            self.probe(
                BehaviorKind.COHESION,
                group_visible=True,
                group_relative_angle=-0.4,
            )
        )
        for invalid in (
            self.probe(BehaviorKind.COHESION),
            self.probe(
                BehaviorKind.COHESION,
                group_visible=True,
                group_relative_angle=float("nan"),
            ),
        ):
            with self.subTest(probe=invalid):
                with self.assertRaises(ValueError):
                    validate_probe(invalid)

    def test_intervention_keeps_factual_target_context_immutable(self) -> None:
        probe = self.probe(
            BehaviorKind.FOOD_ORIENTATION,
            behavior_target_id=8,
            target_visible=True,
            food_target_id=8,
            food_relative_angle=0.25,
        )
        job = CounterfactualProbeJob(
            probe,
            PureNeatEvaluator(
                FixedNetwork(tuple([0.0] * ACTION_OUTPUT_COUNT)),
                tuple(["clamped"] * ACTION_OUTPUT_COUNT),
            ),
        )

        job.advance()

        self.assertIs(job.probe, probe)
        self.assertTrue(job.probe.target_visible)
        self.assertEqual(job.probe.food_target_id, 8)
        self.assertEqual(job.probe.food_relative_angle, 0.25)


class SharedWorkerWhyTest(unittest.TestCase):
    @staticmethod
    def observation(simulation_time: float) -> BehaviorObservation:
        return BehaviorObservation(
            creature_id=1,
            selection_generation=1,
            simulation_time=simulation_time,
            x=simulation_time * 20.0,
            y=0.0,
            heading=0.0,
            angular_velocity=0.5,
            velocity_x=20.0,
            velocity_y=0.0,
            speed=20.0,
            nearest_food_id=5,
            food_visible=True,
            food_distance=150.0 - simulation_time * 12.0,
            food_relative_angle=max(0.0, 0.8 - simulation_time * 0.18),
        )

    def test_spawn_worker_processes_behavior_and_why(self) -> None:
        service = BehaviorObserverService(
            BehaviorObserverConfig(),
            CounterfactualWhyConfig(),
        )
        try:
            service.set_focus(1, 1)
            evaluator = PureNeatEvaluator(
                FixedNetwork(tuple([0.0] * ACTION_OUTPUT_COUNT)),
                tuple(["clamped"] * ACTION_OUTPUT_COUNT),
            )
            service.set_focal_brain(
                FocalBrainUpdate(
                    1,
                    1,
                    7,
                    pickle.dumps(evaluator),
                )
            )
            for index in range(8):
                service.submit(self.observation(index / 10.0))
            behavior = ProbeBehavior(
                BehaviorKind.FOOD_APPROACH,
                BoutStatus.ACTIVE,
                1,
                0.8,
                target_id=5,
            )
            probe = CounterfactualProbeInput(
                creature_id=1,
                selection_generation=1,
                brain_revision=7,
                simulation_time=0.8,
                sensor_schema_version=SENSOR_CONTRACT.schema_version,
                behaviors=(behavior,),
                actual_inputs=tuple([0.0] * len(SENSOR_INPUT_NAMES)),
                actual_outputs=tuple([0.5] * ACTION_OUTPUT_COUNT),
                submitted_monotonic=time.monotonic(),
                target_visible=True,
                food_target_id=5,
                food_relative_angle=0.4,
            )
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                service.submit_why(probe)
                service.poll()
                if (
                    service.latest_snapshot is not None
                    and service.latest_why_snapshots
                ):
                    break
                time.sleep(0.02)

            self.assertIsNotNone(service.latest_snapshot)
            self.assertTrue(service.latest_why_snapshots)
            self.assertEqual(
                service.latest_why_snapshots[0].brain_revision,
                7,
            )
            self.assertEqual(
                service.latest_why_snapshots[0].target_id,
                5,
            )
            visible_food = next(
                effect
                for effect in service.latest_why_snapshots[0].effects
                if effect.intervention
                is SemanticIntervention.VISIBLE_FOOD_CUES
            )
            rotate = next(
                output
                for output in visible_food.output_effects
                if output.output_name == "rotate"
            )
            self.assertIsNotNone(rotate.actual_target_alignment)
            self.assertIsNotNone(
                rotate.counterfactual_target_alignment
            )
        finally:
            service.close()

    def test_slow_why_work_yields_to_temporal_observations(self) -> None:
        service = BehaviorObserverService(
            BehaviorObserverConfig(input_queue_capacity=16),
            CounterfactualWhyConfig(probe_queue_capacity=1),
        )
        try:
            service.set_focus(1, 1)
            service.set_focal_brain(
                FocalBrainUpdate(
                    1,
                    1,
                    9,
                    pickle.dumps(
                        PureNeatEvaluator(
                            SlowNetwork(
                                tuple([0.0] * ACTION_OUTPUT_COUNT),
                                0.02,
                            ),
                            tuple(["clamped"] * ACTION_OUTPUT_COUNT),
                        )
                    ),
                )
            )
            time.sleep(0.05)
            service.submit_why(
                CounterfactualProbeInput(
                    creature_id=1,
                    selection_generation=1,
                    brain_revision=9,
                    simulation_time=0.0,
                    sensor_schema_version=SENSOR_CONTRACT.schema_version,
                    behaviors=(
                        ProbeBehavior(
                            BehaviorKind.FOOD_APPROACH,
                            BoutStatus.ACTIVE,
                            1,
                            0.0,
                            target_id=5,
                        ),
                    ),
                    actual_inputs=tuple(
                        [0.0] * len(SENSOR_INPUT_NAMES)
                    ),
                    actual_outputs=tuple(
                        [0.5] * ACTION_OUTPUT_COUNT
                    ),
                    submitted_monotonic=time.monotonic(),
                    target_visible=True,
                    food_target_id=5,
                    food_relative_angle=0.4,
                )
            )
            for index in range(12):
                service.submit(self.observation(index / 10.0))
                service.poll()
                time.sleep(0.006)

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                service.poll()
                if service.diagnostics.observations_processed >= 12:
                    break
                time.sleep(0.01)

            self.assertEqual(
                service.diagnostics.observations_processed,
                12,
            )
            self.assertEqual(service.diagnostics.samples_dropped, 0)
            states = {
                state.behavior
                for state in service.latest_snapshot.behaviors
            }
            self.assertIn(BehaviorKind.FOOD_APPROACH, states)
        finally:
            service.close()


class CounterfactualWhyConfigTest(unittest.TestCase):
    def test_rejects_invalid_probe_rate_and_capacity(self) -> None:
        with self.assertRaises(ValueError):
            CounterfactualWhyConfig(probe_hz=0.0)
        with self.assertRaises(ValueError):
            CounterfactualWhyConfig(history_capacity=0)

    def test_target_center_dead_zone_defaults_and_validation(self) -> None:
        self.assertEqual(
            CounterfactualWhyConfig().target_center_dead_zone_radians,
            0.05,
        )
        for value in (-0.01, pi, float("nan"), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CounterfactualWhyConfig(
                        target_center_dead_zone_radians=value
                    )


if __name__ == "__main__":
    unittest.main()
