from __future__ import annotations

import sqlite3
import unittest

from configs.sim_config import build_sim_config
from src.analysis import generate_inspector_report
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

        groups = {group.action: group for group in report.behavioral_shifts}
        self.assertEqual(report.species_id, 2)
        self.assertEqual(report.parent_species_id, 1)
        self.assertEqual(report.species_traits, self.child.founder_traits)
        self.assertNotEqual(report.species_traits, self.parent.founder_traits)
        self.assertEqual(len(groups["Nursing"].excitatory), 1)
        self.assertEqual(
            len(groups["Sensory Processing (Hidden)"].inhibitory),
            1,
        )
        self.assertEqual(report.legacy.descendant_count, 1)
        self.assertEqual(report.legacy.average_lifespan, 15.0)
        self.assertEqual(report.food_scarcity, 0.75)
        self.assertEqual(report.population_density, 0.8)

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


if __name__ == "__main__":
    unittest.main()
