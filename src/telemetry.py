from __future__ import annotations

from pathlib import Path
import sqlite3

from src.speciation import (
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)


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

            CREATE TABLE IF NOT EXISTS species_history (
                species_id INTEGER PRIMARY KEY,
                parent_species_id INTEGER,
                founder_creature_id INTEGER,
                founder_genome_id INTEGER,
                time_emerged REAL,
                color_red INTEGER,
                color_green INTEGER,
                color_blue INTEGER,
                data_quality TEXT NOT NULL,
                radius REAL,
                vision_range REAL,
                vision_angle REAL,
                movement_cost_multiplier REAL,
                radius_delta REAL,
                vision_range_delta REAL,
                vision_angle_delta REAL,
                movement_cost_delta REAL,
                neat_distance REAL,
                phenotypic_distance REAL,
                weighted_phenotypic_distance REAL,
                composite_distance REAL,
                compatibility_threshold REAL,
                phenotypic_weight REAL,
                radius_component REAL,
                vision_range_component REAL,
                vision_angle_component REAL,
                movement_cost_component REAL
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

    def log_species_record(self, record: SpeciesRecord) -> None:
        traits = record.founder_traits
        deltas = record.trait_deltas
        distances = record.distances
        color = record.founder_color or (None, None, None)
        self.connection.execute(
            """
            INSERT OR IGNORE INTO species
                (species_id, parent_species_id, time_emerged)
            VALUES (?, ?, ?)
            """,
            (
                record.species_id,
                record.parent_species_id,
                record.emerged_at,
            ),
        )
        self.connection.execute(
            """
            INSERT OR REPLACE INTO species_history (
                species_id, parent_species_id, founder_creature_id,
                founder_genome_id, time_emerged, color_red, color_green,
                color_blue, data_quality, radius, vision_range, vision_angle,
                movement_cost_multiplier, radius_delta, vision_range_delta,
                vision_angle_delta, movement_cost_delta, neat_distance,
                phenotypic_distance, weighted_phenotypic_distance,
                composite_distance, compatibility_threshold,
                phenotypic_weight, radius_component, vision_range_component,
                vision_angle_component, movement_cost_component
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                record.species_id,
                record.parent_species_id,
                record.founder_creature_id,
                record.founder_genome_id,
                record.emerged_at,
                *color,
                record.data_quality,
                None if traits is None else traits.radius,
                None if traits is None else traits.vision_range,
                None if traits is None else traits.vision_angle,
                None if traits is None else traits.movement_cost_multiplier,
                None if deltas is None else deltas.radius,
                None if deltas is None else deltas.vision_range,
                None if deltas is None else deltas.vision_angle,
                None if deltas is None else deltas.movement_cost_multiplier,
                distances.neat_distance,
                distances.phenotypic_distance,
                distances.weighted_phenotypic_distance,
                distances.composite_distance,
                distances.compatibility_threshold,
                distances.phenotypic_weight,
                distances.radius_component,
                distances.vision_range_component,
                distances.vision_angle_component,
                distances.movement_cost_component,
            ),
        )
        self.connection.commit()

    def load_species_lineage(self) -> list[tuple[int, int | None, float | None]]:
        rows = self.connection.execute(
            """
            SELECT species_id, parent_species_id, time_emerged
            FROM species
            ORDER BY species_id
            """
        ).fetchall()
        return [
            (int(species_id), parent_species_id, time_emerged)
            for species_id, parent_species_id, time_emerged in rows
        ]

    def load_species_records(
        self,
        *,
        up_to_time: float | None = None,
    ) -> dict[int, SpeciesRecord]:
        query = """
            SELECT
                species_id, parent_species_id, founder_creature_id,
                founder_genome_id, time_emerged, color_red, color_green,
                color_blue, data_quality, radius, vision_range, vision_angle,
                movement_cost_multiplier, radius_delta, vision_range_delta,
                vision_angle_delta, movement_cost_delta, neat_distance,
                phenotypic_distance, weighted_phenotypic_distance,
                composite_distance, compatibility_threshold,
                phenotypic_weight, radius_component, vision_range_component,
                vision_angle_component, movement_cost_component
            FROM species_history
        """
        parameters: tuple[float, ...] = ()
        if up_to_time is not None:
            query += " WHERE time_emerged IS NULL OR time_emerged <= ?"
            parameters = (float(up_to_time),)
        query += " ORDER BY species_id"

        records: dict[int, SpeciesRecord] = {}
        for row in self.connection.execute(query, parameters).fetchall():
            (
                species_id,
                parent_species_id,
                founder_creature_id,
                founder_genome_id,
                emerged_at,
                color_red,
                color_green,
                color_blue,
                data_quality,
                radius,
                vision_range,
                vision_angle,
                movement_cost_multiplier,
                radius_delta,
                vision_range_delta,
                vision_angle_delta,
                movement_cost_delta,
                neat_distance,
                phenotypic_distance,
                weighted_phenotypic_distance,
                composite_distance,
                compatibility_threshold,
                phenotypic_weight,
                radius_component,
                vision_range_component,
                vision_angle_component,
                movement_cost_component,
            ) = row
            records[int(species_id)] = SpeciesRecord(
                species_id=int(species_id),
                parent_species_id=parent_species_id,
                founder_creature_id=founder_creature_id,
                founder_genome_id=founder_genome_id,
                emerged_at=emerged_at,
                founder_color=(
                    None
                    if color_red is None
                    or color_green is None
                    or color_blue is None
                    else (int(color_red), int(color_green), int(color_blue))
                ),
                data_quality=str(data_quality),
                founder_traits=_trait_snapshot(
                    radius,
                    vision_range,
                    vision_angle,
                    movement_cost_multiplier,
                ),
                trait_deltas=_trait_snapshot(
                    radius_delta,
                    vision_range_delta,
                    vision_angle_delta,
                    movement_cost_delta,
                ),
                distances=SpeciesDistanceBreakdown(
                    neat_distance=neat_distance,
                    phenotypic_distance=phenotypic_distance,
                    weighted_phenotypic_distance=weighted_phenotypic_distance,
                    composite_distance=composite_distance,
                    compatibility_threshold=compatibility_threshold,
                    phenotypic_weight=phenotypic_weight,
                    radius_component=radius_component,
                    vision_range_component=vision_range_component,
                    vision_angle_component=vision_angle_component,
                    movement_cost_component=movement_cost_component,
                ),
            )
        return records

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


def _trait_snapshot(
    radius: float | None,
    vision_range: float | None,
    vision_angle: float | None,
    movement_cost_multiplier: float | None,
) -> SpeciesTraitSnapshot | None:
    if (
        radius is None
        or vision_range is None
        or vision_angle is None
        or movement_cost_multiplier is None
    ):
        return None
    return SpeciesTraitSnapshot(
        radius=float(radius),
        vision_range=float(vision_range),
        vision_angle=float(vision_angle),
        movement_cost_multiplier=float(movement_cost_multiplier),
    )
