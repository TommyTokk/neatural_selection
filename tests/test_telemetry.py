from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from src.telemetry import TelemetryDatabase
from src.speciation import (
    NeatChangeSummary,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)


def species_record() -> SpeciesRecord:
    return SpeciesRecord(
        species_id=2,
        parent_species_id=1,
        founder_creature_id=9,
        founder_genome_id=12,
        emerged_at=4.5,
        founder_color=(10, 20, 30),
        data_quality="exact",
        founder_traits=SpeciesTraitSnapshot(
            17.0, 123.0, 1.2, 1.1, 0.8, 0.3, 0.6
        ),
        trait_deltas=SpeciesTraitSnapshot(
            1.0, 5.0, 0.1, 0.05, 0.1, -0.2, 0.05
        ),
        distances=SpeciesDistanceBreakdown(
            neat_distance=2.0,
            phenotypic_distance=0.6,
            weighted_phenotypic_distance=1.2,
            composite_distance=3.2,
            compatibility_threshold=3.0,
            phenotypic_weight=2.0,
            radius_component=0.1,
            vision_range_component=0.2,
            vision_angle_component=0.1,
            movement_cost_component=0.2,
            flocking_trait_distance=0.25,
            weighted_flocking_trait_distance=0.25,
            flocking_trait_distance_coefficient=1.0,
            separation_gene_component=0.1,
            alignment_gene_component=0.2,
            cohesion_gene_component=0.45,
        ),
        neat_changes=NeatChangeSummary(
            nodes_added=1,
            nodes_removed=0,
            connections_added=2,
            connections_removed=0,
            connections_enabled=0,
            connections_disabled=1,
            weights_changed=3,
            node_parameters_changed=1,
            key_changes=(),
        ),
        emergence_food_ratio=0.25,
        emergence_pop_ratio=0.8,
        neural_shifts=((7, -1, "added", 0.9),),
    )


class TelemetryDatabaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_file = (
            Path(self.temporary_directory.name) / "nested" / "telemetry.sqlite"
        )
        self.database = TelemetryDatabase(self.database_file)

    def tearDown(self) -> None:
        self.database.close()
        self.temporary_directory.cleanup()

    def test_creates_tables_and_enables_wal(self) -> None:
        journal_mode = self.database.connection.execute(
            "PRAGMA journal_mode;"
        ).fetchone()
        table_rows = self.database.connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = ? AND name IN (?, ?, ?, ?)
            """,
            (
                "table",
                "species",
                "creatures",
                "population_metrics",
                "species_history",
            ),
        ).fetchall()

        self.assertEqual(journal_mode, ("wal",))
        self.assertEqual(
            {row[0] for row in table_rows},
            {"species", "creatures", "population_metrics", "species_history"},
        )

    def test_parameterized_helpers_store_values_safely(self) -> None:
        malicious_reason = "starved'); DROP TABLE creatures; --"
        self.database.log_species(2, 1, 4.5)
        self.database.log_creature_birth(9, 2, 5.0, 123.0, 17.0)
        self.database.log_creature_death(9, 8.0, malicious_reason)
        self.database.log_metrics(8.0, 3, 40, 12.5)

        creature = self.database.connection.execute(
            """
            SELECT species_id, death_time, death_reason, vision_range, radius
            FROM creatures WHERE creature_id = ?
            """,
            (9,),
        ).fetchone()
        metrics = self.database.connection.execute(
            """
            SELECT alive_count, food_count, best_fitness
            FROM population_metrics WHERE sim_time = ?
            """,
            (8.0,),
        ).fetchone()
        creatures_table = self.database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
            ("table", "creatures"),
        ).fetchone()

        self.assertEqual(creature, (2, 8.0, malicious_reason, 123.0, 17.0))
        self.assertEqual(metrics, (3, 40, 12.5))
        self.assertEqual(creatures_table, ("creatures",))

    def test_load_species_end_times_respects_checkpoint_time(self) -> None:
        self.database.log_creature_birth(1, 1, 0.0, 100.0, 15.0)
        self.database.log_creature_death(1, 5.0, "starved")
        self.database.log_creature_birth(2, 2, 2.0, 100.0, 15.0)
        self.database.log_creature_death(2, 20.0, "old_age")
        self.database.log_creature_birth(3, 3, 30.0, 100.0, 15.0)

        end_times = self.database.load_species_end_times(up_to_time=10.0)

        self.assertEqual(end_times[1], 5.0)
        self.assertEqual(end_times[2], float("inf"))
        self.assertNotIn(3, end_times)

    def test_species_history_survives_database_reopen(self) -> None:
        record = species_record()
        self.database.log_species_record(record)
        self.database.close()
        self.database = TelemetryDatabase(self.database_file)

        row = self.database.connection.execute(
            """
            SELECT parent_species_id, founder_creature_id, founder_genome_id,
                   composite_distance, radius_delta, color_red, data_quality
            FROM species_history WHERE species_id = ?
            """,
            (2,),
        ).fetchone()

        self.assertEqual(row, (1, 9, 12, 3.2, 1.0, 10, "exact"))
        self.assertEqual(self.database.load_species_lineage(), [(2, 1, 4.5)])

    def test_load_species_records_round_trips_and_filters_future_events(
        self,
    ) -> None:
        current = species_record()
        future = replace(
            current,
            species_id=3,
            founder_creature_id=10,
            founder_genome_id=13,
            emerged_at=12.0,
            founder_color=(40, 50, 60),
        )
        self.database.log_species_record(current)
        self.database.log_species_record(future)

        records = self.database.load_species_records(up_to_time=8.0)

        self.assertEqual(records, {2: current})
        self.assertEqual(
            self.database.load_species_records(),
            {2: current, 3: future},
        )

    def test_existing_telemetry_database_gains_history_table(self) -> None:
        legacy_file = Path(self.temporary_directory.name) / "legacy.sqlite"
        connection = sqlite3.connect(legacy_file)
        connection.execute(
            """
            CREATE TABLE species (
                species_id INTEGER PRIMARY KEY,
                parent_species_id INTEGER,
                time_emerged REAL
            )
            """
        )
        connection.commit()
        connection.close()

        upgraded = TelemetryDatabase(legacy_file)
        try:
            table = upgraded.connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name = 'species_history'
                """
            ).fetchone()
        finally:
            upgraded.close()

        self.assertEqual(table, ("species_history",))

    def test_existing_history_table_gains_neat_changes_column(self) -> None:
        legacy_file = Path(self.temporary_directory.name) / "legacy_history.sqlite"
        connection = sqlite3.connect(legacy_file)
        connection.execute(
            "CREATE TABLE species_history (species_id INTEGER PRIMARY KEY)"
        )
        connection.commit()
        connection.close()

        upgraded = TelemetryDatabase(legacy_file)
        try:
            columns = {
                row[1]
                for row in upgraded.connection.execute(
                    "PRAGMA table_info(species_history)"
                ).fetchall()
            }
        finally:
            upgraded.close()

        self.assertIn("neat_changes_json", columns)
        self.assertIn("emergence_food_ratio", columns)
        self.assertIn("emergence_pop_ratio", columns)
        self.assertIn("neural_shifts_json", columns)

    def test_legacy_history_row_loads_with_unavailable_neat_changes(self) -> None:
        self.database.connection.execute(
            """
            INSERT INTO species_history (species_id, data_quality)
            VALUES (?, ?)
            """,
            (7, "legacy"),
        )
        self.database.connection.commit()

        record = self.database.load_species_records()[7]

        self.assertIsNone(record.neat_changes)

    def test_close_is_idempotent(self) -> None:
        self.database.close()
        self.database.close()


if __name__ == "__main__":
    unittest.main()
