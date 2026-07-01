from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

from src.telemetry import TelemetryDatabase
from src.speciation import (
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
        founder_traits=SpeciesTraitSnapshot(17.0, 123.0, 1.2, 1.1),
        trait_deltas=SpeciesTraitSnapshot(1.0, 5.0, 0.1, 0.05),
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
        ),
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

    def test_close_is_idempotent(self) -> None:
        self.database.close()
        self.database.close()


if __name__ == "__main__":
    unittest.main()
