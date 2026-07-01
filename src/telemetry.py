from __future__ import annotations

from pathlib import Path
import sqlite3


class TelemetryDatabase:
    def __init__(self, database_file: str | Path) -> None:
        self.database_file = Path(database_file)
        self.database_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.database_file)
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA synchronous=NORMAL;")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS species (
                species_id INTEGER PRIMARY KEY,
                parent_species_id INTEGER,
                time_emerged REAL
            );

            CREATE TABLE IF NOT EXISTS creatures (
                creature_id INTEGER PRIMARY KEY,
                species_id INTEGER,
                birth_time REAL,
                death_time REAL,
                death_reason TEXT,
                vision_range REAL,
                radius REAL
            );

            CREATE TABLE IF NOT EXISTS population_metrics (
                sim_time REAL PRIMARY KEY,
                alive_count INTEGER,
                food_count INTEGER,
                best_fitness REAL
            );
            """
        )
        self.connection.commit()
        self._closed = False

    def log_species(
        self,
        species_id: int,
        parent_species_id: int | None,
        time_emerged: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO species
                (species_id, parent_species_id, time_emerged)
            VALUES (?, ?, ?)
            """,
            (species_id, parent_species_id, time_emerged),
        )
        self.connection.commit()

    def log_creature_birth(
        self,
        creature_id: int,
        species_id: int,
        birth_time: float,
        vision_range: float,
        radius: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO creatures
                (
                    creature_id,
                    species_id,
                    birth_time,
                    death_time,
                    death_reason,
                    vision_range,
                    radius
                )
            VALUES (?, ?, ?, NULL, NULL, ?, ?)
            """,
            (creature_id, species_id, birth_time, vision_range, radius),
        )
        self.connection.commit()

    def log_creature_death(
        self,
        creature_id: int,
        death_time: float,
        death_reason: str,
    ) -> None:
        self.connection.execute(
            """
            UPDATE creatures
            SET death_time = ?, death_reason = ?
            WHERE creature_id = ?
            """,
            (death_time, death_reason, creature_id),
        )
        self.connection.commit()

    def log_metrics(
        self,
        sim_time: float,
        alive_count: int,
        food_count: int,
        best_fitness: float,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO population_metrics
                (sim_time, alive_count, food_count, best_fitness)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(sim_time) DO UPDATE SET
                alive_count = excluded.alive_count,
                food_count = excluded.food_count,
                best_fitness = excluded.best_fitness
            """,
            (sim_time, alive_count, food_count, best_fitness),
        )
        self.connection.commit()

    def close(self) -> None:
        if self._closed:
            return
        self.connection.commit()
        self.connection.close()
        self._closed = True

    def __enter__(self) -> TelemetryDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
