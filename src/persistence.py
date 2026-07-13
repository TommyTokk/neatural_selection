from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
import os
import pickle
import shutil
from threading import Condition, Thread
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from configs.sim_config import PersistenceConfig, SimConfig

if TYPE_CHECKING:
    from src.neat_controller import NeatBrainController
    from src.world import World


CHECKPOINT_VERSION = 9
LEGACY_CHECKPOINT_VERSIONS = {2, 3, 4, 5, 6, 7, 8}


def _checkpoint_rgb(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        channels = tuple(int(channel) for channel in value[:3])
    except (TypeError, ValueError):
        return None
    if any(channel < 0 or channel > 255 for channel in channels):
        return None
    return channels


class CheckpointError(RuntimeError):
    pass


class SavePriority(IntEnum):
    AUTO = 0
    MANUAL = 1


@dataclass(frozen=True, slots=True)
class CheckpointTarget:
    path: Path
    rotate_backup: bool


@dataclass(frozen=True, slots=True)
class SimulationPaths:
    simulation_directory: Path

    @property
    def simulation_id(self) -> str:
        return self.simulation_directory.name

    @property
    def quick_checkpoint(self) -> Path:
        return self.simulation_directory / "checkpoint.pkl"

    @property
    def telemetry_database(self) -> Path:
        return self.simulation_directory / "telemetry.sqlite"

    @property
    def hourly_directory(self) -> Path:
        return self.simulation_directory / "hourly"

    @classmethod
    def create_new(
        cls,
        config: PersistenceConfig,
        *,
        now: datetime | None = None,
        unique_suffix: str | None = None,
    ) -> SimulationPaths:
        root = Path(config.simulation_root_directory)
        timestamp = cls._timestamp(now)
        suffix = unique_suffix or uuid4().hex[:8]
        paths = cls(root / f"simulation_{timestamp}_{suffix}")
        paths.hourly_directory.mkdir(parents=True, exist_ok=False)
        return paths

    @classmethod
    def from_directory(cls, simulation_directory: str | Path) -> SimulationPaths:
        paths = cls(Path(simulation_directory))
        if not paths.simulation_directory.is_dir():
            raise CheckpointError(
                f"Simulation directory does not exist: "
                f"{paths.simulation_directory}"
            )
        paths.hourly_directory.mkdir(parents=True, exist_ok=True)
        return paths

    @classmethod
    def latest(cls, config: PersistenceConfig) -> SimulationPaths:
        root = Path(config.simulation_root_directory)
        directories = sorted(
            path
            for path in root.glob("simulation_*")
            if path.is_dir()
        )
        if not directories:
            raise CheckpointError(
                f"No simulation directories exist under {root}."
            )
        return cls.from_directory(directories[-1])

    def quick_target(self) -> CheckpointTarget:
        return CheckpointTarget(self.quick_checkpoint, rotate_backup=True)

    def hourly_target(
        self,
        *,
        now: datetime | None = None,
    ) -> CheckpointTarget:
        timestamp = self._timestamp(now)
        path = self.hourly_directory / f"checkpoint_{timestamp}.pkl"
        return CheckpointTarget(path, rotate_backup=False)

    def checkpoint_candidates(self) -> list[Path]:
        candidates = [
            self.quick_checkpoint,
            Path(f"{self.quick_checkpoint}.bak"),
            *self.hourly_directory.glob("checkpoint_*.pkl"),
        ]
        existing = [path for path in candidates if path.is_file()]
        return sorted(
            existing,
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )

    @staticmethod
    def _timestamp(now: datetime | None) -> str:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        return current.strftime("%Y%m%dT%H%M%S%fZ")


@dataclass(slots=True)
class _PendingSave:
    state: dict[str, Any]
    targets: tuple[CheckpointTarget, ...]
    priority: SavePriority


class PersistenceManager:
    def __init__(self) -> None:
        self._condition = Condition()
        self._pending_save: _PendingSave | None = None
        self._saving = False
        self._closing = False
        self._last_error: BaseException | None = None
        self._worker = Thread(
            target=self._run,
            name="checkpoint-writer",
            daemon=True,
        )
        self._worker.start()

    @property
    def last_error(self) -> BaseException | None:
        with self._condition:
            return self._last_error

    @property
    def is_busy(self) -> bool:
        with self._condition:
            return self._saving or self._pending_save is not None

    def save_simulation(
        self,
        world: World,
        neat_controller: NeatBrainController,
        targets: tuple[CheckpointTarget, ...],
        *,
        priority: SavePriority = SavePriority.AUTO,
    ) -> None:
        if not targets:
            return
        with self._condition:
            if self._closing:
                raise RuntimeError("PersistenceManager is closed.")
            pending = self._pending_save
            if pending is not None and pending.priority > priority:
                pending_targets = {
                    target.path: target
                    for target in pending.targets
                }
                pending_targets.update(
                    (target.path, target)
                    for target in targets
                )
                self._pending_save = _PendingSave(
                    state=pending.state,
                    targets=tuple(pending_targets.values()),
                    priority=pending.priority,
                )
                self._condition.notify_all()
                return

        state = self._capture_state(world, neat_controller)
        with self._condition:
            if self._closing:
                raise RuntimeError("PersistenceManager is closed.")
            pending_targets: dict[Path, CheckpointTarget] = {}
            pending = self._pending_save
            if pending is not None:
                pending_targets.update(
                    (target.path, target)
                    for target in pending.targets
                )
            pending_targets.update((target.path, target) for target in targets)
            if pending is not None and pending.priority > priority:
                queued_state = pending.state
                queued_priority = pending.priority
            else:
                queued_state = state
                queued_priority = priority
            self._pending_save = _PendingSave(
                state=queued_state,
                targets=tuple(pending_targets.values()),
                priority=queued_priority,
            )
            self._condition.notify_all()

    def flush(self) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._pending_save is None and not self._saving
            )
            error = self._last_error
            self._last_error = None
        if error is not None:
            raise CheckpointError("The checkpoint writer failed.") from error

    def close(self) -> None:
        with self._condition:
            if self._closing:
                return
        flush_error: BaseException | None = None
        try:
            self.flush()
        except BaseException as error:
            flush_error = error
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._worker.join()
        if flush_error is not None:
            raise flush_error

    @classmethod
    def load_simulation(
        cls,
        config: SimConfig,
        simulation_directory: str | Path | None = None,
    ) -> World:
        """Load a checkpoint into a World.

        Pickle can execute arbitrary code while loading. Only load checkpoint
        files created by this application and never accept untrusted files.
        """
        simulation_paths = (
            SimulationPaths.latest(config.persistence)
            if simulation_directory is None
            else SimulationPaths.from_directory(simulation_directory)
        )
        failures: list[tuple[Path, BaseException]] = []

        for candidate in simulation_paths.checkpoint_candidates():
            try:
                with candidate.open("rb") as checkpoint_stream:
                    state = pickle.load(checkpoint_stream)
                cls._validate_state(state)
                return cls._restore_world(state, config, simulation_paths)
            except (
                OSError,
                EOFError,
                pickle.PickleError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                CheckpointError,
            ) as error:
                failures.append((candidate, error))

        if not failures:
            raise CheckpointError(
                f"No checkpoints exist in "
                f"{simulation_paths.simulation_directory}."
            )
        detail = "; ".join(f"{path}: {error}" for path, error in failures)
        raise CheckpointError(f"No valid checkpoint could be loaded. {detail}")

    @classmethod
    def load_checkpoint(
        cls,
        config: SimConfig,
        checkpoint_file: str | Path,
    ) -> World:
        """Load exactly one checkpoint file into a World.

        Pickle can execute arbitrary code while loading. Only load checkpoint
        files created by this application and never accept untrusted files.
        """
        checkpoint = Path(checkpoint_file)
        simulation_directory = (
            checkpoint.parent.parent
            if checkpoint.parent.name == "hourly"
            else checkpoint.parent
        )
        try:
            simulation_paths = SimulationPaths.from_directory(
                simulation_directory
            )
            with checkpoint.open("rb") as checkpoint_stream:
                state = pickle.load(checkpoint_stream)
            cls._validate_state(state)
            return cls._restore_world(state, config, simulation_paths)
        except Exception as error:
            raise CheckpointError(
                f"Could not load checkpoint {checkpoint}: {error}"
            ) from error

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._pending_save is not None or self._closing
                )
                if self._pending_save is None and self._closing:
                    return
                save_request = self._pending_save
                self._pending_save = None
                self._saving = True

            assert save_request is not None
            state = save_request.state
            targets = save_request.targets
            try:
                target_errors: list[tuple[Path, BaseException]] = []
                for target in targets:
                    try:
                        self._write_atomic(
                            state,
                            target.path,
                            rotate_backup=target.rotate_backup,
                        )
                    except BaseException as error:
                        target_errors.append((target.path, error))
                if target_errors:
                    details = "; ".join(
                        f"{path}: {error}"
                        for path, error in target_errors
                    )
                    raise CheckpointError(
                        f"One or more checkpoint targets failed: {details}"
                    ) from target_errors[0][1]
            except BaseException as error:
                with self._condition:
                    self._last_error = error
            finally:
                with self._condition:
                    self._saving = False
                    self._condition.notify_all()
                state = None
                targets = ()
                save_request = None

    @staticmethod
    def _capture_state(
        world: World,
        neat_controller: NeatBrainController,
    ) -> dict[str, Any]:
        creatures: list[dict[str, Any]] = []
        for creature in world.creatures:
            body = creature.body
            creatures.append(
                {
                    "creature_id": creature.creature_id,
                    "name": creature.name,
                    "position": tuple(creature.position),
                    "heading": creature.heading,
                    "velocity": (body.velocity.x, body.velocity.y),
                    "angular_velocity": body.angular_velocity,
                    "energy": creature.energy,
                    "stomach_energy": max(
                        0.0,
                        float(getattr(creature, "stomach_energy", 0.0)),
                    ),
                    "vision": copy.deepcopy(creature.vision),
                    "physical_traits": copy.deepcopy(creature.physical_traits),
                    "color": tuple(creature.color),
                    "lineage": copy.deepcopy(creature.lineage),
                    "fitness": copy.deepcopy(
                        world.fitness.get(creature.creature_id)
                    ),
                    "chronometer": world._chronometers.get(
                        creature.creature_id, 0.0
                    ),
                    "fertility_baseline": creature.fertility_baseline,
                    "genome_id": neat_controller.genome_id_for(
                        creature.creature_id
                    ),
                }
            )

        foods: list[dict[str, Any]] = []
        for food in world.foods:
            foods.append(
                {
                    "id": food.id,
                    "position": tuple(food.position),
                    "velocity": (food.body.velocity.x, food.body.velocity.y),
                    "angle": food.body.angle,
                    "angular_velocity": food.body.angular_velocity,
                    "radius": food.radius,
                    "energy_density": food.energy_density,
                    "energy_value": food.energy_value,
                    "original_energy_value": food.original_energy_value,
                }
            )

        spawner = world.food_spawner
        species_manager = neat_controller.species_manager
        rt_neat = world.rt_neat
        evolution_state = copy.deepcopy(
            {
                "genomes": neat_controller.population.population,
                "representatives": species_manager.representatives,
            }
        )
        genome_config = getattr(
            getattr(neat_controller, "config", None),
            "genome_config",
            None,
        )
        input_count = 37 if genome_config is None else len(genome_config.input_keys)
        output_count = 16 if genome_config is None else len(genome_config.output_keys)
        next_creature_id = getattr(world, "_next_creature_id_value", None)
        if next_creature_id is None:
            next_creature_id = max(
                [
                    0,
                    *(creature.creature_id for creature in world.creatures),
                    *world.fitness,
                    *world.fitness_archive,
                ]
            ) + 1
        next_genome_id = getattr(
            neat_controller,
            "_next_genome_id_value",
            None,
        )
        if next_genome_id is None:
            next_genome_id = max(
                [
                    0,
                    *neat_controller.population.population,
                    *(
                        brain.genome_id
                        for brain in neat_controller.brains.values()
                        if hasattr(brain, "genome_id")
                    ),
                ]
            ) + 1
        allocator_state_getter = getattr(
            neat_controller,
            "evolution_allocator_state",
            None,
        )
        allocator_state = (
            allocator_state_getter()
            if allocator_state_getter is not None
            else {}
        )
        pheromones = getattr(world, "pheromones", None)
        acoustics = getattr(world, "acoustics", None)
        communication_state = None
        if pheromones is not None and acoustics is not None:
            communication_state = {
                "pheromone_accumulator": float(pheromones.accumulator),
                "trail": pheromones.trail.copy(),
                "alarm": pheromones.alarm.copy(),
                "acoustic_signals": copy.deepcopy(acoustics.signals),
            }

        return {
            "version": CHECKPOINT_VERSION,
            "brain_contract": {
                "inputs": input_count,
                "outputs": output_count,
            },
            "simulation_id": world.simulation_paths.simulation_id,
            "sim_time": world.elapsed_time,
            "rng_state": world.rng.getstate(),
            "world": {
                "physics_accumulator": world._physics_accumulator,
                "reproduction_accumulator": world._reproduction_accumulator,
                "next_creature_id": next_creature_id,
                "time_since_last_quick_save": (
                    world.time_since_last_quick_save
                ),
                "time_since_last_archive_save": (
                    world.time_since_last_archive_save
                ),
                "total_biomass_energy": world.total_biomass_energy,
                "simulation_speed": world.simulation_speed,
                "is_paused": world.is_paused,
                "selected_creature_id": world.selected_creature_id,
                "held_foods": copy.deepcopy(world._held_food_by_creature_id),
                "food_carriers": copy.deepcopy(world._carrier_by_food_id),
            },
            "creatures": creatures,
            "foods": foods,
            "communication": communication_state,
            "food_spawner": {
                "next_food_id": spawner._next_food_id,
                "spawn_credit": spawner._spawn_credit,
                "low_food_burst_credit": spawner._low_food_burst_credit,
                "pending_low_food_burst_items": (
                    spawner._pending_low_food_burst_items
                ),
            },
            "population": {
                "genomes": evolution_state["genomes"],
                "generation": neat_controller.population.generation,
                "next_genome_id": next_genome_id,
                **allocator_state,
            },
            "species_manager": {
                "compatibility_threshold": (
                    species_manager.compatibility_threshold
                ),
                "phenotypic_weight": species_manager.phenotypic_weight,
                "representatives": evolution_state["representatives"],
                "next_species_id": species_manager.next_species_id,
            },
            "species_history": copy.deepcopy(world.species_history),
            "fitness_archive": copy.deepcopy(world.fitness_archive),
            "archived_traits": copy.deepcopy(world._trait_archive_by_genome_id),
            "rt_neat": {
                "stats": copy.deepcopy(rt_neat.stats),
                "eligible_parent_ids": list(rt_neat.eligible_parent_ids),
                "lifespan_at_death_total": rt_neat._lifespan_at_death_total,
                "lifespan_at_death_count": rt_neat._lifespan_at_death_count,
            },
        }

    @staticmethod
    def _write_atomic(
        state: dict[str, Any],
        checkpoint_file: Path,
        *,
        rotate_backup: bool = True,
    ) -> None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{checkpoint_file}.tmp")
        backup = Path(f"{checkpoint_file}.bak")

        try:
            with temporary.open("wb") as checkpoint_stream:
                pickle.dump(state, checkpoint_stream, protocol=pickle.HIGHEST_PROTOCOL)
                checkpoint_stream.flush()
                os.fsync(checkpoint_stream.fileno())

            if rotate_backup and checkpoint_file.exists():
                os.replace(checkpoint_file, backup)
                PersistenceManager._sync_directory(checkpoint_file.parent)

            os.replace(temporary, checkpoint_file)
            PersistenceManager._sync_directory(checkpoint_file.parent)
        except BaseException:
            temporary.unlink(missing_ok=True)
            if (
                rotate_backup
                and not checkpoint_file.exists()
                and backup.exists()
            ):
                shutil.copy2(backup, checkpoint_file)
                with checkpoint_file.open("rb+") as restored_stream:
                    os.fsync(restored_stream.fileno())
                PersistenceManager._sync_directory(checkpoint_file.parent)
            raise

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        try:
            descriptor = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_state(state: Any) -> None:
        if not isinstance(state, dict):
            raise TypeError("Checkpoint root must be a dictionary.")
        version = state.get("version")
        if version != CHECKPOINT_VERSION and version not in LEGACY_CHECKPOINT_VERSIONS:
            raise ValueError(
                f"Unsupported checkpoint version {version!r}; "
                f"expected one of "
                f"{sorted({CHECKPOINT_VERSION, *LEGACY_CHECKPOINT_VERSIONS})}."
            )

    @staticmethod
    def _migrate_species_representatives(
        representatives: dict[int, Any],
        creature_states: list[dict[str, Any]],
        archived_traits: dict[int, Any],
    ) -> dict[int, Any]:
        living_traits = {
            creature_state["genome_id"]: (
                creature_state["physical_traits"],
                creature_state["vision"],
            )
            for creature_state in creature_states
            if creature_state.get("genome_id") is not None
        }
        migrated: dict[int, Any] = {}
        for species_id, representative in representatives.items():
            if isinstance(representative, tuple) and len(representative) == 3:
                migrated[species_id] = representative
                continue

            genome_id = getattr(representative, "key", None)
            traits = living_traits.get(genome_id)
            if traits is None:
                archived = archived_traits.get(genome_id)
                if archived is not None:
                    traits = (archived.physical_traits, archived.vision)
            if traits is None:
                raise CheckpointError(
                    "Cannot migrate species "
                    f"{species_id}: representative genome {genome_id!r} "
                    "has no living or archived phenotype."
                )

            physical_traits, vision = traits
            migrated[species_id] = (
                representative,
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
            )
        return migrated

    @staticmethod
    def _reconstruct_species_history(
        world: World,
        neat_controller: NeatBrainController,
        *,
        up_to_time: float | None = None,
    ) -> dict[int, Any]:
        from src.neat_controller import calculate_phenotypic_distance_components
        from src.speciation import (
            NeatChangeSummary,
            SpeciesDistanceBreakdown,
            SpeciesRecord,
            SpeciesTraitSnapshot,
            extract_neural_shifts,
        )

        telemetry = getattr(world, "telemetry", None)
        lineage_rows = telemetry.load_species_lineage() if telemetry is not None else []
        if up_to_time is not None:
            lineage_rows = [
                (species_id, parent_species_id, emerged_at)
                for species_id, parent_species_id, emerged_at in lineage_rows
                if emerged_at is None or emerged_at <= up_to_time
            ]
        lineage = {
            species_id: (parent_species_id, emerged_at)
            for species_id, parent_species_id, emerged_at in lineage_rows
        }
        manager = neat_controller.species_manager
        records: dict[int, SpeciesRecord] = {}
        for species_id, representative in sorted(manager.representatives.items()):
            genome, physical_traits, vision = representative
            parent_species_id, emerged_at = lineage.get(
                species_id,
                (None, 0.0 if species_id == 1 else None),
            )
            founder_genome_id = getattr(genome, "key", None)
            founder_creature_id = None
            founder_color = None
            for creature in world.creatures:
                if (
                    neat_controller.genome_id_for(creature.creature_id)
                    == founder_genome_id
                ):
                    founder_creature_id = creature.creature_id
                    founder_color = tuple(creature.color[:3])
                    break
            if founder_creature_id is None:
                archived = world._trait_archive_by_genome_id.get(
                    founder_genome_id
                )
                if archived is not None:
                    founder_creature_id = archived.creature_id
                    founder_color = tuple(archived.color[:3])

            zero = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
            deltas = None
            neat_changes = None
            neural_shifts = ()
            distances = SpeciesDistanceBreakdown(
                neat_distance=None,
                phenotypic_distance=None,
                weighted_phenotypic_distance=None,
                composite_distance=None,
                compatibility_threshold=manager.compatibility_threshold,
                phenotypic_weight=manager.phenotypic_weight,
                radius_component=None,
                vision_range_component=None,
                vision_angle_component=None,
                movement_cost_component=None,
            )
            can_reconstruct = (
                parent_species_id is not None
                and parent_species_id in manager.representatives
            )
            if species_id == 1:
                deltas = zero
                neat_changes = NeatChangeSummary.empty()
                distances = SpeciesDistanceBreakdown(
                    neat_distance=0.0,
                    phenotypic_distance=0.0,
                    weighted_phenotypic_distance=0.0,
                    composite_distance=0.0,
                    compatibility_threshold=manager.compatibility_threshold,
                    phenotypic_weight=manager.phenotypic_weight,
                    radius_component=0.0,
                    vision_range_component=0.0,
                    vision_angle_component=0.0,
                    movement_cost_component=0.0,
                )
            elif can_reconstruct:
                parent_genome, parent_physical, parent_vision = (
                    manager.representatives[parent_species_id]
                )
                components = calculate_phenotypic_distance_components(
                    physical_traits,
                    vision,
                    parent_physical,
                    parent_vision,
                    manager.trait_config,
                    manager.vision_config,
                )
                phenotypic_distance = (
                    components.radius
                    + components.vision_range
                    + components.vision_angle
                    + components.movement_cost_multiplier
                )
                neat_distance = genome.distance(
                    parent_genome,
                    neat_controller.config.genome_config,
                )
                neural_shifts = extract_neural_shifts(parent_genome, genome)
                weighted_distance = (
                    phenotypic_distance * manager.phenotypic_weight
                )
                deltas = SpeciesTraitSnapshot(
                    radius=physical_traits.radius - parent_physical.radius,
                    vision_range=vision.range - parent_vision.range,
                    vision_angle=vision.angle - parent_vision.angle,
                    movement_cost_multiplier=(
                        physical_traits.movement_cost_multiplier
                        - parent_physical.movement_cost_multiplier
                    ),
                )
                distances = SpeciesDistanceBreakdown(
                    neat_distance=neat_distance,
                    phenotypic_distance=phenotypic_distance,
                    weighted_phenotypic_distance=weighted_distance,
                    composite_distance=neat_distance + weighted_distance,
                    compatibility_threshold=manager.compatibility_threshold,
                    phenotypic_weight=manager.phenotypic_weight,
                    radius_component=components.radius,
                    vision_range_component=components.vision_range,
                    vision_angle_component=components.vision_angle,
                    movement_cost_component=(
                        components.movement_cost_multiplier
                    ),
                )

            records[species_id] = SpeciesRecord(
                species_id=species_id,
                parent_species_id=parent_species_id,
                founder_creature_id=founder_creature_id,
                founder_genome_id=founder_genome_id,
                emerged_at=emerged_at,
                founder_color=founder_color,
                data_quality=(
                    "reconstructed"
                    if species_id == 1 or can_reconstruct
                    else "partial"
                ),
                founder_traits=SpeciesTraitSnapshot.from_traits(
                    physical_traits,
                    vision,
                ),
                trait_deltas=deltas,
                distances=distances,
                neat_changes=neat_changes,
                emergence_food_ratio=None,
                emergence_pop_ratio=None,
                neural_shifts=neural_shifts,
            )
        return records

    @staticmethod
    def _normalize_species_record(record: Any) -> Any:
        from src.speciation import NeatChangeSummary, SpeciesRecord

        if not isinstance(record, SpeciesRecord):
            return record
        legacy_summary = getattr(record, "neat_changes", None)
        if legacy_summary is not None:
            legacy_summary = NeatChangeSummary(
                nodes_added=legacy_summary.nodes_added,
                nodes_removed=legacy_summary.nodes_removed,
                connections_added=legacy_summary.connections_added,
                connections_removed=legacy_summary.connections_removed,
                connections_enabled=legacy_summary.connections_enabled,
                connections_disabled=legacy_summary.connections_disabled,
                weights_changed=legacy_summary.weights_changed,
                node_parameters_changed=legacy_summary.node_parameters_changed,
                key_changes=(),
            )
        return SpeciesRecord(
            species_id=record.species_id,
            parent_species_id=record.parent_species_id,
            founder_creature_id=record.founder_creature_id,
            founder_genome_id=record.founder_genome_id,
            emerged_at=record.emerged_at,
            founder_color=_checkpoint_rgb(record.founder_color),
            data_quality=record.data_quality,
            founder_traits=record.founder_traits,
            trait_deltas=record.trait_deltas,
            distances=record.distances,
            neat_changes=legacy_summary,
            emergence_food_ratio=getattr(
                record,
                "emergence_food_ratio",
                None,
            ),
            emergence_pop_ratio=getattr(
                record,
                "emergence_pop_ratio",
                None,
            ),
            neural_shifts=tuple(getattr(record, "neural_shifts", ()) or ()),
        )

    @staticmethod
    def _enrich_species_neat_changes(
        history: dict[int, Any],
        neat_controller: NeatBrainController,
    ) -> dict[int, Any]:
        from src.speciation import (
            SpeciesRecord,
            extract_neural_shifts,
        )

        representatives = neat_controller.species_manager.representatives
        enriched: dict[int, Any] = {}
        for species_id, raw_record in history.items():
            record = PersistenceManager._normalize_species_record(raw_record)
            if not isinstance(record, SpeciesRecord) or record.neural_shifts:
                enriched[int(species_id)] = record
                continue
            if record.parent_species_id is None:
                enriched[int(species_id)] = record
                continue
            child = representatives.get(int(species_id))
            parent = representatives.get(int(record.parent_species_id))
            if (
                isinstance(child, tuple)
                and len(child) == 3
                and isinstance(parent, tuple)
                and len(parent) == 3
                and hasattr(child[0], "nodes")
                and hasattr(child[0], "connections")
                and hasattr(parent[0], "nodes")
                and hasattr(parent[0], "connections")
            ):
                record = replace(
                    record,
                    neural_shifts=extract_neural_shifts(parent[0], child[0]),
                )
            enriched[int(species_id)] = record
        return enriched

    @staticmethod
    def _enrich_species_colors(
        history: dict[int, Any],
        world: World,
        neat_controller: NeatBrainController,
    ) -> dict[int, Any]:
        """Recover canonical tree colors missing from legacy checkpoints."""
        from src.speciation import SpeciesRecord

        living_by_id = {
            int(creature.creature_id): creature
            for creature in getattr(world, "creatures", ())
        }
        living_by_species: dict[int, Any] = {}
        living_by_genome: dict[int, Any] = {}
        for creature in getattr(world, "creatures", ()):
            lineage = getattr(creature, "lineage", None)
            species_id = getattr(lineage, "species_id", None)
            if species_id is not None:
                living_by_species.setdefault(int(species_id), creature)
            genome_id = neat_controller.genome_id_for(creature.creature_id)
            if genome_id is not None:
                living_by_genome.setdefault(int(genome_id), creature)

        archived_by_genome = getattr(
            world,
            "_trait_archive_by_genome_id",
            {},
        )
        archived_by_species: dict[int, Any] = {}
        for archived in archived_by_genome.values():
            lineage = getattr(archived, "lineage", None)
            species_id = getattr(lineage, "species_id", None)
            if species_id is not None:
                archived_by_species.setdefault(int(species_id), archived)

        enriched: dict[int, Any] = {}
        for species_id, record in history.items():
            if (
                not isinstance(record, SpeciesRecord)
                or _checkpoint_rgb(getattr(record, "founder_color", None))
                is not None
            ):
                enriched[int(species_id)] = record
                continue

            source = None
            founder_creature_id = getattr(
                record,
                "founder_creature_id",
                None,
            )
            if founder_creature_id is not None:
                source = living_by_id.get(int(founder_creature_id))
            founder_genome_id = getattr(record, "founder_genome_id", None)
            if source is None and founder_genome_id is not None:
                source = living_by_genome.get(int(founder_genome_id))
            if source is None and founder_genome_id is not None:
                source = archived_by_genome.get(int(founder_genome_id))
            if source is None:
                source = living_by_species.get(int(species_id))
            if source is None:
                source = archived_by_species.get(int(species_id))

            recovered_color = _checkpoint_rgb(
                getattr(source, "color", None)
            )
            enriched[int(species_id)] = (
                record
                if recovered_color is None
                else replace(record, founder_color=recovered_color)
            )
        return enriched

    @staticmethod
    def _restore_species_history(
        world: World,
        neat_controller: NeatBrainController,
        saved_history: dict[int, Any] | None,
    ) -> dict[int, Any]:
        normalized_saved_history = {
            int(species_id): PersistenceManager._normalize_species_record(
                record
            )
            for species_id, record in (saved_history or {}).items()
        }
        history = PersistenceManager._enrich_species_neat_changes(
            copy.deepcopy(normalized_saved_history),
            neat_controller,
        )
        history = PersistenceManager._enrich_species_colors(
            history,
            world,
            neat_controller,
        )
        representative_ids = {
            int(species_id)
            for species_id in neat_controller.species_manager.representatives
        }
        next_species_id = max(
            2,
            int(
                getattr(
                    neat_controller.species_manager,
                    "next_species_id",
                    2,
                )
            ),
        )
        allocated_species_ids = set(range(1, next_species_id))
        living_species_ids = {
            int(creature.lineage.species_id)
            for creature in world.creatures
        }
        required_species_ids = (
            allocated_species_ids | representative_ids | living_species_ids
        )
        if saved_history is not None and required_species_ids <= history.keys():
            return history

        recovered = PersistenceManager._reconstruct_species_history(
            world,
            neat_controller,
            up_to_time=world.elapsed_time,
        )
        telemetry = getattr(world, "telemetry", None)
        load_records = getattr(telemetry, "load_species_records", None)
        if load_records is not None:
            recovered.update(load_records(up_to_time=world.elapsed_time))
        recovered.update(history)

        missing_living_ids = living_species_ids - recovered.keys()
        if missing_living_ids:
            from src.speciation import (
                SpeciesDistanceBreakdown,
                SpeciesRecord,
                SpeciesTraitSnapshot,
            )

            unknown_distances = SpeciesDistanceBreakdown(
                neat_distance=None,
                phenotypic_distance=None,
                weighted_phenotypic_distance=None,
                composite_distance=None,
                compatibility_threshold=(
                    neat_controller.species_manager.compatibility_threshold
                ),
                phenotypic_weight=(
                    neat_controller.species_manager.phenotypic_weight
                ),
                radius_component=None,
                vision_range_component=None,
                vision_angle_component=None,
                movement_cost_component=None,
            )
            for species_id in sorted(missing_living_ids):
                founder = next(
                    creature
                    for creature in world.creatures
                    if int(creature.lineage.species_id) == species_id
                )
                recovered[species_id] = SpeciesRecord(
                    species_id=species_id,
                    parent_species_id=None,
                    founder_creature_id=founder.creature_id,
                    founder_genome_id=neat_controller.genome_id_for(
                        founder.creature_id
                    ),
                    emerged_at=None,
                    founder_color=tuple(founder.color[:3]),
                    data_quality="partial",
                    founder_traits=SpeciesTraitSnapshot.from_traits(
                        founder.physical_traits,
                        founder.vision,
                    ),
                    trait_deltas=None,
                    distances=unknown_distances,
                    neat_changes=None,
                    emergence_food_ratio=None,
                    emergence_pop_ratio=None,
                    neural_shifts=(),
                )
        return PersistenceManager._enrich_species_colors(
            PersistenceManager._enrich_species_neat_changes(
                recovered,
                neat_controller,
            ),
            world,
            neat_controller,
        )

    @staticmethod
    def _reconcile_next_species_id(
        neat_controller: NeatBrainController,
        species_history: dict[int, Any],
        creatures: list[Any],
    ) -> None:
        manager = neat_controller.species_manager
        known_species_ids = {
            int(species_id) for species_id in manager.representatives
        }
        known_species_ids.update(
            int(species_id) for species_id in species_history
        )
        known_species_ids.update(
            int(creature.lineage.species_id) for creature in creatures
        )
        minimum_next_id = max(known_species_ids, default=1) + 1
        manager.next_species_id = max(
            2,
            int(manager.next_species_id),
            minimum_next_id,
        )

    @staticmethod
    def _restore_world(
        state: dict[str, Any],
        config: SimConfig,
        simulation_paths: SimulationPaths,
    ) -> World:
        from src.food import Food
        from src.world import World

        world = World(
            config,
            bootstrap=False,
            simulation_paths=simulation_paths,
        )
        try:
            if state["simulation_id"] != simulation_paths.simulation_id:
                raise CheckpointError(
                    f"Checkpoint belongs to {state['simulation_id']}, not "
                    f"{simulation_paths.simulation_id}."
                )
            world.elapsed_time = state["sim_time"]
            world.rng.setstate(state["rng_state"])

            runtime = state["world"]
            world._physics_accumulator = runtime["physics_accumulator"]
            world._reproduction_accumulator = runtime["reproduction_accumulator"]
            world.time_since_last_quick_save = runtime[
                "time_since_last_quick_save"
            ]
            world.time_since_last_archive_save = runtime[
                "time_since_last_archive_save"
            ]
            world.total_biomass_energy = runtime["total_biomass_energy"]
            world.simulation_speed = runtime["simulation_speed"]
            world.is_paused = runtime["is_paused"]
            world.selected_creature_id = runtime["selected_creature_id"]
            legacy_fertility_baselines = runtime.get("previous_biome", {})

            communication_state = state.get("communication")
            if communication_state:
                world.pheromones.restore(
                    communication_state["trail"],
                    communication_state["alarm"],
                    communication_state.get("pheromone_accumulator", 0.0),
                )
                saved_signals = communication_state.get("acoustic_signals", {})
                world.acoustics.replace_signals(
                    saved_signals.values()
                    if isinstance(saved_signals, dict)
                    else saved_signals
                )

            controller = world.neat_controller
            population_state = state["population"]
            controller.population.population = population_state["genomes"]
            controller.population.generation = population_state["generation"]
            species_state = state["species_manager"]
            controller.species_manager.compatibility_threshold = species_state[
                "compatibility_threshold"
            ]
            controller.species_manager.phenotypic_weight = species_state.get(
                "phenotypic_weight",
                controller.species_manager.phenotypic_weight,
            )
            controller.species_manager.representatives = (
                PersistenceManager._migrate_species_representatives(
                    species_state["representatives"],
                    state["creatures"],
                    state.get("archived_traits", {}),
                )
            )
            controller.species_manager.next_species_id = species_state[
                "next_species_id"
            ]
            contract = state.get("brain_contract", {"inputs": 23, "outputs": 8})
            if (
                int(contract.get("inputs", 23)) < len(
                    controller.config.genome_config.input_keys
                )
                or int(contract.get("outputs", 8)) < len(
                    controller.config.genome_config.output_keys
                )
            ):
                controller.migrate_legacy_brain_contract()
            controller.restore_evolution_allocators(
                population_state.get("next_node_id"),
                population_state.get("innovation_number"),
            )

            world.fitness = {}
            world._chronometers = {}
            for creature_state in state["creatures"]:
                creature = world._spawn_creature(
                    creature_state["creature_id"],
                    position=creature_state["position"],
                    heading=creature_state["heading"],
                    energy=creature_state["energy"],
                    color=creature_state["color"],
                    vision=creature_state["vision"],
                    physical_traits=creature_state["physical_traits"],
                    lineage=creature_state["lineage"],
                )
                creature.name = creature_state["name"]
                creature.stomach_energy = max(
                    0.0,
                    float(creature_state.get("stomach_energy", 0.0)),
                )
                creature.body.velocity = creature_state["velocity"]
                creature.body.angular_velocity = creature_state["angular_velocity"]
                creature.fertility_baseline = float(
                    creature_state.get(
                        "fertility_baseline",
                        legacy_fertility_baselines.get(
                            creature.creature_id,
                            world._biome_fertility_at(*creature.position),
                        ),
                    )
                )
                world.creatures.append(creature)
                fitness = creature_state["fitness"]
                if fitness is not None:
                    world.fitness[creature.creature_id] = fitness
                world._chronometers[creature.creature_id] = creature_state[
                    "chronometer"
                ]
                genome_id = creature_state["genome_id"]
                if genome_id is None:
                    raise CheckpointError(
                        f"Creature {creature.creature_id} has no saved genome ID."
                    )
                controller.restore_brain(creature.creature_id, genome_id)

            for food_state in state["foods"]:
                food = Food(
                    id=food_state["id"],
                    x=food_state["position"][0],
                    y=food_state["position"][1],
                    radius=food_state["radius"],
                    energy_density=food_state["energy_density"],
                )
                food.energy_value = food_state["energy_value"]
                food.original_energy_value = food_state["original_energy_value"]
                food.body.velocity = food_state["velocity"]
                food.body.angle = food_state["angle"]
                food.body.angular_velocity = food_state["angular_velocity"]
                world._add_foods([food])

            spawner_state = state["food_spawner"]
            spawner = world.food_spawner
            spawner._next_food_id = spawner_state["next_food_id"]
            spawner._spawn_credit = spawner_state["spawn_credit"]
            spawner._low_food_burst_credit = spawner_state.get(
                "low_food_burst_credit",
                0.0,
            )
            spawner._pending_low_food_burst_items = spawner_state.get(
                "pending_low_food_burst_items",
                0,
            )

            world._held_food_by_creature_id = runtime["held_foods"]
            world._carrier_by_food_id = runtime["food_carriers"]
            world._sync_carried_foods()
            world.fitness_archive = state["fitness_archive"]
            world._trait_archive_by_genome_id = state["archived_traits"]
            world.species_history = PersistenceManager._restore_species_history(
                world,
                controller,
                state.get("species_history"),
            )
            PersistenceManager._reconcile_next_species_id(
                controller,
                world.species_history,
                world.creatures,
            )
            if world.telemetry is not None:
                for record in world.species_history.values():
                    world.telemetry.log_species_record(record)

            rt_state = state["rt_neat"]
            world.rt_neat.stats = rt_state["stats"]
            world.rt_neat.eligible_parent_ids = rt_state["eligible_parent_ids"]
            world.rt_neat._lifespan_at_death_total = rt_state[
                "lifespan_at_death_total"
            ]
            world.rt_neat._lifespan_at_death_count = rt_state[
                "lifespan_at_death_count"
            ]
            next_creature_id = runtime.get("next_creature_id")
            if next_creature_id is None:
                next_creature_id = max(
                    [
                        0,
                        *(creature.creature_id for creature in world.creatures),
                        *world.fitness,
                        *world.fitness_archive,
                        *(
                            archived.creature_id
                            for archived in (
                                world._trait_archive_by_genome_id.values()
                            )
                        ),
                    ]
                ) + 1
            world._next_creature_id_value = next_creature_id

            next_genome_id = population_state.get("next_genome_id")
            if next_genome_id is None:
                next_genome_id = max(
                    [
                        0,
                        *controller.population.population,
                        *(
                            getattr(representative[0], "key", 0)
                            for representative in (
                                controller.species_manager.representatives.values()
                            )
                        ),
                    ]
                ) + 1
            controller._next_genome_id_value = next_genome_id
            world._prune_historical_archives()
            world._refresh_stats()
            return world
        except BaseException:
            world.close()
            raise
