from __future__ import annotations

import json
from pathlib import Path
import sqlite3

TELEMETRY_SCHEMA_VERSION = 27

from src.creature.speciation import (
    NeatChangeSummary,
    NeuralShift,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
    normalize_neural_shifts,
)


class TelemetryDatabase:
    def __init__(self, database_file: str | Path) -> None:
        self.database_file = Path(database_file)
        in_memory = str(database_file) == ":memory:"
        if not in_memory:
            self.database_file.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            ":memory:" if in_memory else self.database_file
        )
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
                stomach_capacity REAL,
                digestion_rate REAL,
                digestion_efficiency REAL,
                separation_gene REAL,
                alignment_gene REAL,
                cohesion_gene REAL,
                radius_delta REAL,
                vision_range_delta REAL,
                vision_angle_delta REAL,
                movement_cost_delta REAL,
                stomach_capacity_delta REAL,
                digestion_rate_delta REAL,
                digestion_efficiency_delta REAL,
                separation_gene_delta REAL,
                alignment_gene_delta REAL,
                cohesion_gene_delta REAL,
                neat_distance REAL,
                phenotypic_distance REAL,
                weighted_phenotypic_distance REAL,
                composite_distance REAL,
                compatibility_threshold REAL,
                phenotypic_weight REAL,
                radius_component REAL,
                vision_range_component REAL,
                vision_angle_component REAL,
                movement_cost_component REAL,
                stomach_capacity_component REAL,
                digestion_rate_component REAL,
                digestion_efficiency_component REAL,
                digestive_trait_component REAL,
                flocking_trait_distance REAL,
                weighted_flocking_trait_distance REAL,
                flocking_trait_distance_coefficient REAL,
                separation_gene_component REAL,
                alignment_gene_component REAL,
                cohesion_gene_component REAL,
                neat_changes_json TEXT,
                emergence_food_ratio REAL,
                emergence_pop_ratio REAL,
                neural_shifts_json TEXT
            );

            CREATE TABLE IF NOT EXISTS population_metrics (
                sim_time REAL PRIMARY KEY,
                alive_count INTEGER,
                food_count INTEGER,
                best_net_energy_balance REAL,
                red_avg REAL,
                green_avg REAL,
                blue_avg REAL
            );

            CREATE TABLE IF NOT EXISTS reproduction_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sim_time REAL NOT NULL,
                parent_creature_id INTEGER NOT NULL,
                species_id INTEGER NOT NULL,
                parent_investment REAL NOT NULL,
                child_endowment REAL NOT NULL,
                outcome TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS flocking_population_metrics (
                sim_time REAL PRIMARY KEY,
                population_size INTEGER,
                seeing_any_percent REAL,
                seeing_compatible_percent REAL,
                effective_count_ge_2_percent REAL,
                mean_effective_count REAL,
                max_effective_count REAL,
                mean_engagement REAL,
                mean_neural_herding REAL,
                mean_effective_herding REAL,
                mean_panic REAL,
                mean_panic_attenuation REAL,
                mean_separation_weight REAL,
                mean_alignment_weight REAL,
                mean_cohesion_weight REAL,
                mean_requested_social_force REAL,
                mean_accepted_social_force REAL,
                mean_social_blend REAL,
                mean_alignment_error REAL,
                mean_center_distance REAL,
                in_groups_ge_3_percent REAL,
                largest_group_size INTEGER,
                mean_group_lifetime REAL,
                fragmentation_count INTEGER,
                merger_count INTEGER,
                benchmark_reward_contribution REAL
            );

            CREATE INDEX IF NOT EXISTS idx_species_parent
                ON species(parent_species_id);
            CREATE INDEX IF NOT EXISTS idx_creatures_species
                ON creatures(species_id);
            CREATE INDEX IF NOT EXISTS idx_reproduction_events_time
                ON reproduction_events(sim_time);
            """
        )
        self._ensure_species_history_columns()
        self._ensure_flocking_population_metrics_columns()
        self._ensure_population_metrics_columns()
        self.connection.execute(f"PRAGMA user_version = {TELEMETRY_SCHEMA_VERSION}")
        self.connection.commit()
        self._closed = False

    def _ensure_population_metrics_columns(self) -> None:
        """Add thermodynamic metrics to telemetry databases in place."""
        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(population_metrics)"
            )
        }
        required = {
            "best_net_energy_balance": "REAL",
            "red_avg": "REAL",
            "green_avg": "REAL",
            "blue_avg": "REAL",
        }
        for name, column_type in required.items():
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE population_metrics ADD COLUMN {name} {column_type}"
                )

    def _ensure_species_history_columns(self) -> None:
        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(species_history)"
            ).fetchall()
        }
        required_columns = {
            "neat_changes_json": "TEXT",
            "emergence_food_ratio": "REAL",
            "emergence_pop_ratio": "REAL",
            "neural_shifts_json": "TEXT",
            "separation_gene": "REAL",
            "alignment_gene": "REAL",
            "cohesion_gene": "REAL",
            "separation_gene_delta": "REAL",
            "alignment_gene_delta": "REAL",
            "cohesion_gene_delta": "REAL",
            "flocking_trait_distance": "REAL",
            "weighted_flocking_trait_distance": "REAL",
            "flocking_trait_distance_coefficient": "REAL",
            "separation_gene_component": "REAL",
            "alignment_gene_component": "REAL",
            "cohesion_gene_component": "REAL",
            "stomach_capacity": "REAL",
            "digestion_rate": "REAL",
            "digestion_efficiency": "REAL",
            "stomach_capacity_delta": "REAL",
            "digestion_rate_delta": "REAL",
            "digestion_efficiency_delta": "REAL",
            "stomach_capacity_component": "REAL",
            "digestion_rate_component": "REAL",
            "digestion_efficiency_component": "REAL",
            "digestive_trait_component": "REAL",
        }
        for name, column_type in required_columns.items():
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE species_history ADD COLUMN {name} {column_type}"
                )

    def _ensure_flocking_population_metrics_columns(self) -> None:
        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(flocking_population_metrics)"
            ).fetchall()
        }
        required_columns = {
            "population_size": "INTEGER",
            "seeing_any_percent": "REAL",
            "seeing_compatible_percent": "REAL",
            "effective_count_ge_2_percent": "REAL",
            "mean_effective_count": "REAL",
            "max_effective_count": "REAL",
            "mean_engagement": "REAL",
            "mean_neural_herding": "REAL",
            "mean_effective_herding": "REAL",
            "mean_panic": "REAL",
            "mean_panic_attenuation": "REAL",
            "mean_separation_weight": "REAL",
            "mean_alignment_weight": "REAL",
            "mean_cohesion_weight": "REAL",
            "mean_requested_social_force": "REAL",
            "mean_accepted_social_force": "REAL",
            "mean_social_blend": "REAL",
            "mean_alignment_error": "REAL",
            "mean_center_distance": "REAL",
            "in_groups_ge_3_percent": "REAL",
            "largest_group_size": "INTEGER",
            "mean_group_lifetime": "REAL",
            "fragmentation_count": "INTEGER",
            "merger_count": "INTEGER",
            "benchmark_reward_contribution": "REAL",
        }
        for name, column_type in required_columns.items():
            if name not in columns:
                self.connection.execute(
                    "ALTER TABLE flocking_population_metrics "
                    f"ADD COLUMN {name} {column_type}"
                )
        self.connection.execute(
            "UPDATE flocking_population_metrics "
            "SET mean_effective_herding = mean_neural_herding "
            "WHERE mean_effective_herding IS NULL"
        )

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
        columns = (
            "species_id", "parent_species_id", "founder_creature_id",
            "founder_genome_id", "time_emerged", "color_red", "color_green",
            "color_blue", "data_quality", "radius", "vision_range",
            "vision_angle", "movement_cost_multiplier", "stomach_capacity",
            "digestion_rate", "digestion_efficiency", "separation_gene",
            "alignment_gene", "cohesion_gene", "radius_delta",
            "vision_range_delta", "vision_angle_delta", "movement_cost_delta",
            "stomach_capacity_delta", "digestion_rate_delta",
            "digestion_efficiency_delta",
            "separation_gene_delta", "alignment_gene_delta",
            "cohesion_gene_delta", "neat_distance", "phenotypic_distance",
            "weighted_phenotypic_distance", "composite_distance",
            "compatibility_threshold", "phenotypic_weight", "radius_component",
            "vision_range_component", "vision_angle_component",
            "movement_cost_component", "stomach_capacity_component",
            "digestion_rate_component", "digestion_efficiency_component",
            "digestive_trait_component", "flocking_trait_distance",
            "weighted_flocking_trait_distance",
            "flocking_trait_distance_coefficient", "separation_gene_component",
            "alignment_gene_component", "cohesion_gene_component",
            "neat_changes_json", "emergence_food_ratio", "emergence_pop_ratio",
            "neural_shifts_json",
        )
        values = (
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
            None if traits is None else traits.stomach_capacity,
            None if traits is None else traits.digestion_rate,
            None if traits is None else traits.digestion_efficiency,
            None if traits is None else traits.separation_gene,
            None if traits is None else traits.alignment_gene,
            None if traits is None else traits.cohesion_gene,
            None if deltas is None else deltas.radius,
            None if deltas is None else deltas.vision_range,
            None if deltas is None else deltas.vision_angle,
            None if deltas is None else deltas.movement_cost_multiplier,
            None if deltas is None else deltas.stomach_capacity,
            None if deltas is None else deltas.digestion_rate,
            None if deltas is None else deltas.digestion_efficiency,
            None if deltas is None else deltas.separation_gene,
            None if deltas is None else deltas.alignment_gene,
            None if deltas is None else deltas.cohesion_gene,
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
            distances.stomach_capacity_component,
            distances.digestion_rate_component,
            distances.digestion_efficiency_component,
            distances.digestive_trait_component,
            distances.flocking_trait_distance,
            distances.weighted_flocking_trait_distance,
            distances.flocking_trait_distance_coefficient,
            distances.separation_gene_component,
            distances.alignment_gene_component,
            distances.cohesion_gene_component,
            _serialize_neat_changes(getattr(record, "neat_changes", None)),
            getattr(record, "emergence_food_ratio", None),
            getattr(record, "emergence_pop_ratio", None),
            _serialize_neural_shifts(getattr(record, "neural_shifts", ())),
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO species_history "
            f"({', '.join(columns)}) VALUES "
            f"({', '.join('?' for _ in columns)})",
            values,
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
                movement_cost_multiplier, stomach_capacity, digestion_rate,
                digestion_efficiency, separation_gene, alignment_gene,
                cohesion_gene, radius_delta, vision_range_delta,
                vision_angle_delta, movement_cost_delta,
                stomach_capacity_delta, digestion_rate_delta,
                digestion_efficiency_delta, separation_gene_delta,
                alignment_gene_delta, cohesion_gene_delta, neat_distance,
                phenotypic_distance, weighted_phenotypic_distance,
                composite_distance, compatibility_threshold,
                phenotypic_weight, radius_component, vision_range_component,
                vision_angle_component, movement_cost_component,
                stomach_capacity_component, digestion_rate_component,
                digestion_efficiency_component, digestive_trait_component,
                flocking_trait_distance, weighted_flocking_trait_distance,
                flocking_trait_distance_coefficient, separation_gene_component,
                alignment_gene_component, cohesion_gene_component,
                neat_changes_json, emergence_food_ratio,
                emergence_pop_ratio, neural_shifts_json
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
                stomach_capacity,
                digestion_rate,
                digestion_efficiency,
                separation_gene,
                alignment_gene,
                cohesion_gene,
                radius_delta,
                vision_range_delta,
                vision_angle_delta,
                movement_cost_delta,
                stomach_capacity_delta,
                digestion_rate_delta,
                digestion_efficiency_delta,
                separation_gene_delta,
                alignment_gene_delta,
                cohesion_gene_delta,
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
                stomach_capacity_component,
                digestion_rate_component,
                digestion_efficiency_component,
                digestive_trait_component,
                flocking_trait_distance,
                weighted_flocking_trait_distance,
                flocking_trait_distance_coefficient,
                separation_gene_component,
                alignment_gene_component,
                cohesion_gene_component,
                neat_changes_json,
                emergence_food_ratio,
                emergence_pop_ratio,
                neural_shifts_json,
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
                    stomach_capacity,
                    digestion_rate,
                    digestion_efficiency,
                    separation_gene,
                    alignment_gene,
                    cohesion_gene,
                    gene_default=0.5,
                    bounded_genes=True,
                ),
                trait_deltas=_trait_snapshot(
                    radius_delta,
                    vision_range_delta,
                    vision_angle_delta,
                    movement_cost_delta,
                    stomach_capacity_delta,
                    digestion_rate_delta,
                    digestion_efficiency_delta,
                    separation_gene_delta,
                    alignment_gene_delta,
                    cohesion_gene_delta,
                    gene_default=0.0,
                    bounded_genes=False,
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
                    stomach_capacity_component=(
                        stomach_capacity_component
                    ),
                    digestion_rate_component=digestion_rate_component,
                    digestion_efficiency_component=(
                        digestion_efficiency_component
                    ),
                    digestive_trait_component=digestive_trait_component,
                    flocking_trait_distance=flocking_trait_distance,
                    weighted_flocking_trait_distance=(
                        weighted_flocking_trait_distance
                    ),
                    flocking_trait_distance_coefficient=(
                        flocking_trait_distance_coefficient
                    ),
                    separation_gene_component=separation_gene_component,
                    alignment_gene_component=alignment_gene_component,
                    cohesion_gene_component=cohesion_gene_component,
                ),
                neat_changes=_deserialize_neat_changes(neat_changes_json),
                emergence_food_ratio=_optional_float(emergence_food_ratio),
                emergence_pop_ratio=_optional_float(emergence_pop_ratio),
                neural_shifts=_deserialize_neural_shifts(neural_shifts_json),
            )
        return records

    def load_species_end_times(
        self,
        *,
        up_to_time: float | None = None,
    ) -> dict[int, float]:
        """Return inferred species end times from creature birth/death rows."""
        if up_to_time is None:
            rows = self.connection.execute(
                """
                SELECT
                    species_id,
                    SUM(CASE WHEN death_time IS NULL THEN 1 ELSE 0 END),
                    MAX(death_time)
                FROM creatures
                GROUP BY species_id
                """
            ).fetchall()
            end_times: dict[int, float] = {}
            for species_id, living_count, last_death in rows:
                if species_id is None:
                    continue
                if int(living_count or 0) > 0:
                    end_times[int(species_id)] = float("inf")
                elif last_death is not None:
                    end_times[int(species_id)] = float(last_death)
            return end_times

        cutoff = float(up_to_time)
        rows = self.connection.execute(
            """
            SELECT
                species_id,
                SUM(
                    CASE
                        WHEN death_time IS NULL OR death_time > ? THEN 1
                        ELSE 0
                    END
                ),
                MAX(
                    CASE
                        WHEN death_time <= ? THEN death_time
                        ELSE NULL
                    END
                )
            FROM creatures
            WHERE birth_time <= ?
            GROUP BY species_id
            """,
            (cutoff, cutoff, cutoff),
        ).fetchall()
        end_times: dict[int, float] = {}
        for species_id, living_count, last_death in rows:
            if species_id is None:
                continue
            end_times[int(species_id)] = (
                float("inf")
                if int(living_count or 0) > 0
                else cutoff if last_death is None else float(last_death)
            )
        return end_times

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
        best_net_energy_balance: float,
        red_avg: float = 0.0,
        green_avg: float = 0.0,
        blue_avg: float = 0.0,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO population_metrics
                (sim_time, alive_count, food_count, best_net_energy_balance,
                 red_avg, green_avg, blue_avg)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sim_time) DO UPDATE SET
                alive_count = excluded.alive_count,
                food_count = excluded.food_count,
                best_net_energy_balance = excluded.best_net_energy_balance,
                red_avg = excluded.red_avg,
                green_avg = excluded.green_avg,
                blue_avg = excluded.blue_avg
            """,
            (
                sim_time,
                alive_count,
                food_count,
                best_net_energy_balance,
                red_avg,
                green_avg,
                blue_avg,
            ),
        )
        self.connection.commit()

    def log_reproduction_events(
        self,
        events: list[dict[str, object]],
    ) -> None:
        """Store passive autonomous-birth outcomes."""
        if not events:
            return
        columns = (
            "sim_time",
            "parent_creature_id",
            "species_id",
            "parent_investment",
            "child_endowment",
            "outcome",
        )
        self.connection.executemany(
            "INSERT INTO reproduction_events "
            f"({', '.join(columns)}) VALUES "
            f"({', '.join('?' for _ in columns)})",
            [tuple(event[column] for column in columns) for event in events],
        )
        self.connection.commit()

    def log_flocking_metrics(
        self,
        metrics: dict[str, float | int],
    ) -> None:
        columns = tuple(metrics)
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column != "sim_time"
        )
        self.connection.execute(
            "INSERT INTO flocking_population_metrics "
            f"({', '.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(sim_time) DO UPDATE SET {updates}",
            tuple(metrics[column] for column in columns),
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
    stomach_capacity: float | None,
    digestion_rate: float | None,
    digestion_efficiency: float | None,
    separation_gene: float | None,
    alignment_gene: float | None,
    cohesion_gene: float | None,
    *,
    gene_default: float,
    bounded_genes: bool,
) -> SpeciesTraitSnapshot | None:
    if (
        radius is None
        or vision_range is None
        or vision_angle is None
        or movement_cost_multiplier is None
    ):
        return None
    genes = [
        gene_default if separation_gene is None else float(separation_gene),
        gene_default if alignment_gene is None else float(alignment_gene),
        gene_default if cohesion_gene is None else float(cohesion_gene),
    ]
    if bounded_genes:
        genes = [max(0.0, min(1.0, gene)) for gene in genes]
    digestive_defaults = (1.6, 0.2, 0.9) if bounded_genes else (0.0, 0.0, 0.0)
    return SpeciesTraitSnapshot(
        radius=float(radius),
        vision_range=float(vision_range),
        vision_angle=float(vision_angle),
        movement_cost_multiplier=float(movement_cost_multiplier),
        separation_gene=genes[0],
        alignment_gene=genes[1],
        cohesion_gene=genes[2],
        stomach_capacity=(
            digestive_defaults[0]
            if stomach_capacity is None
            else float(stomach_capacity)
        ),
        digestion_rate=(
            digestive_defaults[1]
            if digestion_rate is None
            else float(digestion_rate)
        ),
        digestion_efficiency=(
            digestive_defaults[2]
            if digestion_efficiency is None
            else float(digestion_efficiency)
        ),
    )


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _serialize_neural_shifts(shifts: object) -> str | None:
    try:
        raw_shifts = tuple(shifts)  # type: ignore[arg-type]
    except TypeError:
        return None
    normalized = normalize_neural_shifts(raw_shifts)
    if len(normalized) != len(raw_shifts):
        return None
    compact = [
        {
            "source_node_id": shift.source_node_id,
            "target_node_id": shift.target_node_id,
            "change_type": shift.change_type,
            "parent_weight": shift.parent_weight,
            "child_weight": shift.child_weight,
            "weight_delta": shift.weight_delta,
        }
        for shift in normalized
    ]
    return json.dumps(compact, separators=(",", ":"))


def _deserialize_neural_shifts(
    value: object,
) -> tuple[NeuralShift, ...]:
    if value is None:
        return ()
    try:
        rows = json.loads(str(value))
        return normalize_neural_shifts(rows)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()


def _serialize_neat_changes(summary: NeatChangeSummary | None) -> str | None:
    if summary is None:
        return None
    return json.dumps(
        {
            "nodes_added": summary.nodes_added,
            "nodes_removed": summary.nodes_removed,
            "connections_added": summary.connections_added,
            "connections_removed": summary.connections_removed,
            "connections_enabled": summary.connections_enabled,
            "connections_disabled": summary.connections_disabled,
            "weights_changed": summary.weights_changed,
            "node_parameters_changed": summary.node_parameters_changed,
            "key_changes": list(summary.key_changes),
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def _deserialize_neat_changes(value: object) -> NeatChangeSummary | None:
    if value is None:
        return None
    try:
        data = json.loads(str(value))
        return NeatChangeSummary(
            nodes_added=int(data["nodes_added"]),
            nodes_removed=int(data["nodes_removed"]),
            connections_added=int(data["connections_added"]),
            connections_removed=int(data["connections_removed"]),
            connections_enabled=int(data["connections_enabled"]),
            connections_disabled=int(data["connections_disabled"]),
            weights_changed=int(data["weights_changed"]),
            node_parameters_changed=int(data["node_parameters_changed"]),
            key_changes=(),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
