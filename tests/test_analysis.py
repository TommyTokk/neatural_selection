from __future__ import annotations

from math import exp, log
import sqlite3
from types import SimpleNamespace
import unittest

from configs.sim_config import build_sim_config
from src.analysis import (
    BEHAVIOR_RADAR_LABELS,
    calculate_behavior_scores,
    generate_inspector_report,
    generate_radar_chart_image,
)
from src.speciation import (
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)


def record(
    species_id: int,
    parent_species_id: int | None,
    traits: SpeciesTraitSnapshot,
    *,
    neural_shifts: tuple[tuple[int, int, str, float], ...] = (),
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
        self.assertEqual(len(report.behavioral_ethogram), 1)
        self.assertIn(
            "Endogenous Baseline Drive",
            report.behavioral_ethogram[0].description,
        )
        self.assertEqual(len(report.neuro_integration_hubs), 1)
        self.assertEqual(report.neuro_integration_hubs[0].hub_id, 99)
        self.assertIn(
            "Feeding Drive (Satiety-Modulated)",
            report.neuro_integration_hubs[0].sensory_integrations[0],
        )
        self.assertEqual(report.legacy.descendant_count, 1)
        self.assertEqual(report.legacy.average_lifespan, 15.0)
        self.assertEqual(report.food_scarcity, 0.75)
        self.assertEqual(report.population_density, 0.8)

    def test_direct_reflex_translation_for_load_carriage_to_fleeing(self) -> None:
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

        self.assertEqual(
            report.behavioral_ethogram[0].description,
            "🟢 [Load Carriage State (Carrying Object)] now actively "
            "triggers/sensitizes [Threat Avoidance Reflexes]",
        )

    def test_stomach_fullness_sensor_has_satiety_lexicon_entry(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((3, -28, "added", 0.8),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertIn(
            "Stomach Fullness (Satiety)",
            report.behavioral_ethogram[0].description,
        )

    def test_effective_flockmate_count_has_analysis_lexicon_entry(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=((3, -27, "added", 0.8),),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
        )

        self.assertIn(
            "Effective Compatible Flockmate Count",
            report.behavioral_ethogram[0].description,
        )

    def test_species_reflex_summary_uses_canonical_input_order(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=tuple(
                (3, input_key, "added", 0.8)
                for input_key in range(-28, 0)
            ),
        )

        report = generate_inspector_report(
            child,
            self.parent,
            None,
            self.config,
            range(12),
            range(-1, -29, -1),
        )

        self.assertEqual(
            [reflex.source_node_id for reflex in report.behavioral_ethogram],
            list(range(-1, -29, -1)),
        )
        self.assertIn(
            "Stomach Fullness (Satiety)",
            report.behavioral_ethogram[-1].description,
        )

    def test_species_hub_summary_uses_canonical_input_order(self) -> None:
        child = record(
            2,
            1,
            SpeciesTraitSnapshot(20.0, 120.0, 1.2, 1.25),
            neural_shifts=(
                (99, -28, "added", 0.8),
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
            range(-1, -29, -1),
        )

        integrations = report.neuro_integration_hubs[0].sensory_integrations
        self.assertIn("Endogenous Baseline Drive", integrations[0])
        self.assertIn("Nearest Food Distance", integrations[1])
        self.assertIn("Stomach Fullness (Satiety)", integrations[2])

    def test_direct_reflex_translation_for_inhibitory_shift(self) -> None:
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

        self.assertEqual(
            report.behavioral_ethogram[0].description,
            "🔴 [Load Carriage State (Carrying Object)] now actively "
            "suppresses/brakes [Threat Avoidance Reflexes]",
        )

    def test_removed_reflex_translation(self) -> None:
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

        self.assertEqual(
            report.behavioral_ethogram[0].description,
            "⚪ Lost the instinct to trigger [Threat Avoidance Reflexes] "
            "in response to [Load Carriage State (Carrying Object)]",
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

        self.assertEqual(report.behavioral_ethogram, ())
        self.assertEqual(len(report.neuro_integration_hubs), 1)
        hub = report.neuro_integration_hubs[0]
        self.assertEqual(hub.hub_id, 491)
        self.assertIn(
            "Integration Hub 491 is now integrating "
            "[Load Carriage State (Carrying Object)]",
            hub.sensory_integrations[0],
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

        self.assertEqual(report.behavioral_ethogram, ())
        self.assertEqual(len(report.neuro_integration_hubs), 1)
        self.assertIn(
            "Kinetic / Locomotion Reflexes is now modulated by "
            "abstract concepts from [Integration Hub 491]",
            report.neuro_integration_hubs[0].behavioral_modulations[0],
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
            self.assertEqual(image.width, image.height)
            self.assertEqual(tuple(plt.get_fignums()), before)


if __name__ == "__main__":
    unittest.main()
