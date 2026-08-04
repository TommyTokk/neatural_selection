from __future__ import annotations

from math import exp, log
import sqlite3
from types import SimpleNamespace
import unittest

from configs.sim_config import build_sim_config
from src.action import ACTION_OUTPUT_NAMES
from src.analysis import (
    BEHAVIOR_RADAR_LABELS,
    action_node_label,
    calculate_behavior_scores,
    classify_connection_transition,
    generate_inspector_report,
    generate_radar_chart_image,
    sensory_node_label,
)
from src.vision import SENSOR_INPUT_NAMES
from src.speciation import (
    NeuralShift,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)


def record(
    species_id: int,
    parent_species_id: int | None,
    traits: SpeciesTraitSnapshot,
    *,
    neural_shifts: tuple[object, ...] = (),
) -> SpeciesRecord:
    return SpeciesRecord(
        species_id=species_id,
        parent_species_id=parent_species_id,
        founder_creature_id=species_id,
        founder_genome_id=species_id,
        emerged_at=float(species_id),
        founder_color=(10, 20, 30),
        data_quality="exact",
        founder_traits=traits,
        trait_deltas=None,
        distances=SpeciesDistanceBreakdown(
            None, None, None, None, None, None, None, None, None, None
        ),
        emergence_food_ratio=0.25,
        emergence_pop_ratio=0.8,
        neural_shifts=neural_shifts,
    )


class InspectorAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.database = sqlite3.connect(":memory:")
        self.database.executescript(
            """
            CREATE TABLE species (
                species_id INTEGER PRIMARY KEY,
                parent_species_id INTEGER
            );
            CREATE TABLE creatures (
                creature_id INTEGER PRIMARY KEY,
                species_id INTEGER,
                birth_time REAL,
                death_time REAL
            );
            INSERT INTO species VALUES (1, NULL), (2, 1), (3, 2), (4, 1);
            INSERT INTO creatures VALUES
                (1, 2, 2.0, 12.0),
                (2, 2, 4.0, 24.0),
                (3, 2, 9.0, NULL);
            """
        )
        self.config = build_sim_config()
        self.parent = record(
            1,
            None,
            SpeciesTraitSnapshot(16.0, 100.0, 1.0, 1.0),
        )
        self.child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=(
                (7, -1, "added", 0.8),
                (99, -2, "weight", -0.7),
            ),
        )

    def tearDown(self) -> None:
        self.database.close()

    def test_report_groups_actions_and_queries_recursive_legacy(self) -> None:
        report = generate_inspector_report(
            self.child,
            self.parent,
            self.database,
            self.config,
            range(12),
        )

        self.assertEqual(report.species_id, 2)
        self.assertEqual(report.parent_species_id, 1)
        self.assertEqual(report.species_traits, self.child.founder_traits)
        self.assertNotEqual(report.species_traits, self.parent.founder_traits)
        self.assertEqual(len(report.direct_brain_changes), 1)
        self.assertEqual(report.direct_brain_changes[0].source_node_id, -1)
        self.assertEqual(report.direct_brain_changes[0].target_node_id, 7)
        self.assertEqual(len(report.neuro_integration_hubs), 1)
        self.assertEqual(report.neuro_integration_hubs[0].hub_id, 99)
        self.assertEqual(
            report.neuro_integration_hubs[0]
            .incoming_sensor_changes[0]
            .source_node_id,
            -2,
        )
        self.assertEqual(report.legacy.descendant_count, 1)
        self.assertEqual(report.legacy.average_lifespan, 15.0)
        self.assertEqual(report.food_scarcity, 0.75)
        self.assertEqual(report.population_density, 0.8)

    def test_direct_change_retains_load_carriage_to_panic_edge(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((8, -17, "added", 0.8),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        shift = report.direct_brain_changes[0]
        self.assertEqual((shift.source_node_id, shift.target_node_id), (-17, 8))
        self.assertEqual((shift.parent_weight, shift.child_weight), (None, 0.8))

    def test_stomach_fullness_sensor_has_satiety_lexicon_entry(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((3, -33, "added", 0.8),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertEqual(report.direct_brain_changes[0].source_node_id, -33)
        self.assertEqual(
            sensory_node_label("stomach_fullness").technical,
            "Stomach Fullness (Satiety)",
        )

    def test_effective_flockmate_count_has_analysis_lexicon_entry(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((3, -25, "added", 0.8),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertEqual(report.direct_brain_changes[0].source_node_id, -25)
        self.assertEqual(
            sensory_node_label("flock_effective_count").technical,
            "Target-Scaled Compatible Flockmate Count",
        )

    def test_species_reflex_summary_uses_canonical_input_order(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=tuple(
                (3, input_key, "added", 0.8)
                for input_key in range(-33, 0)
            ),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
            range(-1, -34, -1),
        )

        self.assertEqual(
            [shift.source_node_id for shift in report.direct_brain_changes],
            list(range(-1, -34, -1)),
        )
        self.assertEqual(report.direct_brain_changes[-1].source_node_id, -33)

    def test_species_hub_summary_uses_canonical_input_order(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=(
                (99, -33, "added", 0.8),
                (99, -11, "added", 0.8),
                (99, -1, "added", 0.8),
            ),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
            range(-1, -34, -1),
        )

        integrations = report.neuro_integration_hubs[0].incoming_sensor_changes
        self.assertEqual(
            [shift.source_node_id for shift in integrations],
            [-1, -11, -33],
        )

    def test_direct_negative_addition_retains_negative_child_weight(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((8, -17, "added", -0.8),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        shift = report.direct_brain_changes[0]
        self.assertEqual(shift.child_weight, -0.8)
        self.assertEqual(
            classify_connection_transition(None, shift.child_weight).label,
            "Negative influence added",
        )

    def test_legacy_removed_inhibitory_edge_is_classified_as_removed(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((8, -17, "removed", 0.4),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        shift = report.direct_brain_changes[0]
        self.assertEqual((shift.parent_weight, shift.child_weight), (-0.4, None))
        self.assertEqual(
            classify_connection_transition(shift.parent_weight, None).label,
            "Negative influence removed",
        )

    def test_hidden_sensor_shift_is_grouped_as_integration_hub(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((491, -17, "weight", 0.9),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertEqual(report.direct_brain_changes, ())
        self.assertEqual(len(report.neuro_integration_hubs), 1)
        hub = report.neuro_integration_hubs[0]
        self.assertEqual(hub.hub_id, 491)
        self.assertEqual(
            (
                hub.incoming_sensor_changes[0].source_node_id,
                hub.incoming_sensor_changes[0].target_node_id,
            ),
            (-17, 491),
        )

    def test_hidden_output_shift_is_grouped_as_behavioral_modulation(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((0, 491, "weight", -0.9),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertEqual(report.direct_brain_changes, ())
        self.assertEqual(len(report.neuro_integration_hubs), 1)
        self.assertEqual(
            (
                report.neuro_integration_hubs[0]
                .outgoing_action_changes[0]
                .source_node_id,
                report.neuro_integration_hubs[0]
                .outgoing_action_changes[0]
                .target_node_id,
            ),
            (491, 0),
        )

    def test_hub_changes_keep_exact_weights_and_ignore_hidden_to_hidden(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=(
                NeuralShift(-17, 53, "added", None, 0.41),
                NeuralShift(53, 8, "changed", 0.5, 2.74, 2.24),
                NeuralShift(53, 54, "added", None, 0.2),
            ),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertEqual(len(report.neuro_integration_hubs), 1)
        hub = report.neuro_integration_hubs[0]
        self.assertEqual(hub.hub_id, 53)
        self.assertEqual(
            (
                hub.incoming_sensor_changes[0].parent_weight,
                hub.incoming_sensor_changes[0].child_weight,
            ),
            (None, 0.41),
        )
        self.assertEqual(
            (
                hub.outgoing_action_changes[0].parent_weight,
                hub.outgoing_action_changes[0].child_weight,
                hub.outgoing_action_changes[0].weight_delta,
            ),
            (0.5, 2.74, 2.24),
        )
        self.assertEqual(report.direct_brain_changes, ())

    def test_action_labels_keep_acceleration_and_turn_distinct(self) -> None:
        self.assertEqual(action_node_label("accelerate").primary, "Accelerate")
        self.assertEqual(action_node_label("rotate").primary, "Turn")

    def test_semantic_registry_covers_contracts_and_has_safe_fallbacks(self) -> None:
        self.assertTrue(
            all(sensory_node_label(name).primary for name in SENSOR_INPUT_NAMES)
        )
        self.assertTrue(
            all(action_node_label(name).primary for name in ACTION_OUTPUT_NAMES)
        )
        self.assertEqual(
            sensory_node_label("future_sensor").primary,
            "Future Sensor",
        )
        self.assertEqual(
            action_node_label("future_action").technical,
            "Future Action",
        )

    def test_metabolic_profile_uses_exact_configured_formulas(self) -> None:
        report = generate_inspector_report(
            self.child,
            self.parent,
            None,
            self.config,
            range(12),
        )
        parent_traits = self.parent.founder_traits
        child_traits = self.child.founder_traits
        assert parent_traits is not None and child_traits is not None

        def idle(traits: SpeciesTraitSnapshot) -> float:
            vision = self.config.vision
            area = (
                traits.vision_angle
                / vision.max_angle
                * (traits.vision_range / vision.max_range) ** 2
            )
            return (
                self.config.metabolism.basic_metabolism_rate
                + vision.base_energy_cost
                + vision.area_energy_cost_factor * area
                + self.config.trait.body_metabolism_cost_factor
                * (traits.radius / self.config.trait.max_radius) ** 2
            )

        self.assertAlmostEqual(report.metabolism.parent_idle_cost, idle(parent_traits))
        self.assertAlmostEqual(report.metabolism.child_idle_cost, idle(child_traits))
        self.assertAlmostEqual(
            report.metabolism.child_active_cost,
            idle(child_traits)
            + self.config.metabolism.movement_energy_cost_factor
            * child_traits.movement_cost_multiplier,
        )

    def test_root_species_still_reports_its_own_metabolic_cost(self) -> None:
        report = generate_inspector_report(
            self.parent,
            None,
            None,
            self.config,
            range(12),
        )

        self.assertIsNotNone(report.metabolism.child_idle_cost)
        self.assertIsNotNone(report.metabolism.child_active_cost)
        self.assertIsNone(report.metabolism.parent_idle_cost)
        self.assertIsNone(report.metabolism.idle_percent_change)


class ConnectionClassificationTest(unittest.TestCase):
    def test_requested_weight_transitions(self) -> None:
        cases = (
            (None, 0.7, "Positive influence added"),
            (None, -0.7, "Negative influence added"),
            (0.8, None, "Positive influence removed"),
            (-0.8, None, "Negative influence removed"),
            (0.3, 0.8, "Positive influence strengthened"),
            (0.8, 0.3, "Positive influence weakened"),
            (-0.6, -1.2, "Negative influence strengthened"),
            (-1.2, -0.6, "Negative influence weakened"),
            (0.5, -0.3, "Influence changed from positive to negative"),
            (-0.5, 0.3, "Influence changed from negative to positive"),
        )
        for parent, child, expected in cases:
            with self.subTest(parent=parent, child=child):
                self.assertEqual(
                    classify_connection_transition(parent, child).label,
                    expected,
                )

    def test_zero_unchanged_and_unavailable_transitions(self) -> None:
        self.assertEqual(
            classify_connection_transition(None, 0.0).label,
            "Zero-weight connection added",
        )
        self.assertEqual(
            classify_connection_transition(0.4, 0.4).label,
            "Influence unchanged",
        )
        self.assertEqual(
            classify_connection_transition(None, None).label,
            "Historical weights unavailable",
        )


class BehaviorRadarAnalysisTest(unittest.TestCase):
    @staticmethod
    def genome(
        biases: dict[int, float],
        connections: tuple[tuple[int, int, float, bool], ...] = (),
    ) -> SimpleNamespace:
        return SimpleNamespace(
            nodes={
                key: SimpleNamespace(bias=bias)
                for key, bias in biases.items()
            },
            connections={
                (source, target): SimpleNamespace(
                    key=(source, target),
                    weight=weight,
                    enabled=enabled,
                )
                for source, target, weight, enabled in connections
            },
        )

    def test_extreme_drives_are_stable_and_bounded(self) -> None:
        genome = self.genome({0: 100.0, 8: -100.0})

        scores = calculate_behavior_scores(genome, range(12))

        self.assertAlmostEqual(scores[0], 1.0)
        self.assertLess(scores[5], 1e-40)

    def test_only_enabled_incoming_weights_contribute(self) -> None:
        genome = self.genome(
            {0: 0.0},
            (
                (-1, 0, 1.0, True),
                (-2, 0, 100.0, False),
            ),
        )

        scores = calculate_behavior_scores(genome, range(12))

        self.assertAlmostEqual(scores[0], 1.0 / (1.0 + exp(-1.0)))

    def test_output_keys_are_mapped_positionally(self) -> None:
        output_keys = tuple(range(100, 112))
        genome = self.genome({100: log(3.0), 108: -log(3.0)})

        scores = calculate_behavior_scores(genome, output_keys)

        self.assertAlmostEqual(scores[0], 0.75)
        self.assertAlmostEqual(scores[5], 0.25)

    def test_composite_axes_average_normalized_scores(self) -> None:
        magnitude = log(3.0)
        genome = self.genome(
            {
                3: magnitude,
                5: -magnitude,
                9: -magnitude,
                10: magnitude,
                11: -magnitude,
            }
        )

        scores = calculate_behavior_scores(genome, range(12))

        self.assertAlmostEqual(scores[1], 0.5)
        self.assertAlmostEqual(scores[2], 0.25)
        self.assertEqual(len(scores), len(BEHAVIOR_RADAR_LABELS))

    def test_missing_output_nodes_are_neutral(self) -> None:
        scores = calculate_behavior_scores(self.genome({}), range(12))

        self.assertEqual(scores, (0.5,) * 6)

    def test_radar_image_is_detached_and_figures_are_closed(self) -> None:
        import matplotlib.pyplot as plt

        before = tuple(plt.get_fignums())
        for parent_scores in (None, (0.4,) * 6):
            image = generate_radar_chart_image(
                (0.6,) * 6,
                parent_scores,
                BEHAVIOR_RADAR_LABELS,
            )
            image.load()
            self.assertEqual(image.mode, "RGBA")
            self.assertGreater(image.width, 0)
            self.assertGreater(image.height, 0)
            self.assertGreaterEqual(image.width, 600)
            self.assertEqual(image.width, image.height)
            self.assertEqual(tuple(plt.get_fignums()), before)


if __name__ == "__main__":
    unittest.main()
