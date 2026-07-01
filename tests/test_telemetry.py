from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.telemetry import TelemetryDatabase


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
            WHERE type = ? AND name IN (?, ?, ?)
            """,
            (
                "table",
                "species",
                "creatures",
                "population_metrics",
            ),
        ).fetchall()

        self.assertEqual(journal_mode, ("wal",))
        self.assertEqual(
            {row[0] for row in table_rows},
            {"species", "creatures", "population_metrics"},
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

    def test_close_is_idempotent(self) -> None:
        self.database.close()
        self.database.close()


if __name__ == "__main__":
    unittest.main()
