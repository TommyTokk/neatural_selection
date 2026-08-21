from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
import os
import pickle
import logging
from math import isfinite
import shutil
from threading import Condition, Thread
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from configs.sim_config import LiveFoodConfig, PersistenceConfig, SimConfig
from src.creature.action import (
    ACTION_OUTPUT_COUNT,
    ACTION_OUTPUT_NAMES,
    ACTION_SCHEMA_VERSION,
    Action,
)
from src.creature import FlockingTraits
from src.creature.flocking import SocialRuntime
from src.creature.vision import SENSOR_INPUT_COUNT, SENSING_SCHEMA_VERSION

if TYPE_CHECKING:
    from src.creature.neat.controller import NeatBrainController
    from src.world import World


CHECKPOINT_VERSION = 25
LEGACY_CHECKPOINT_VERSIONS = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24,
}
LOGGER = logging.getLogger(__name__)


def _action_to_primitive(action: object) -> dict[str, float] | None:
    if action is None:
        return None
    return {
        name: float(getattr(action, name, 0.0))
        for name in ACTION_OUTPUT_NAMES
    }


def _action_from_checkpoint(value: object) -> Action | None:
    if value is None:
        return None
    if isinstance(value, Action):
        return value
    if not isinstance(value, dict):
        return None
    return Action(**{
        name: float(value.get(name, 0.0))
        for name in ACTION_OUTPUT_NAMES
    })


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


class CheckpointContractError(CheckpointError):
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
        *,
        allow_brain_contract_reset: bool = False,
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
                return cls._restore_world(
                    state,
                    config,
                    simulation_paths,
                    allow_brain_contract_reset=allow_brain_contract_reset,
                )
            except CheckpointContractError:
                # Contract rejection is a policy decision, not checkpoint
                # corruption. Never fall back to another checkpoint and make
                # a destructive neural reset appear implicit.
                raise
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
        *,
        allow_brain_contract_reset: bool = False,
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
            return cls._restore_world(
                state,
                config,
                simulation_paths,
                allow_brain_contract_reset=allow_brain_contract_reset,
            )
        except CheckpointContractError:
            raise
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
        lock = getattr(world, "_checkpoint_state_lock", None)
        if lock is None:
            return PersistenceManager._capture_state_unlocked(
                world,
                neat_controller,
            )
        with lock:
            return PersistenceManager._capture_state_unlocked(
                world,
                neat_controller,
            )

    @staticmethod
    def _capture_state_unlocked(
        world: World,
        neat_controller: NeatBrainController,
    ) -> dict[str, Any]:
        creatures: list[dict[str, Any]] = []
        for creature in world.creatures:
            body = creature.body
            creature_id = creature.creature_id
            brain_for = getattr(neat_controller, "brain_for", None)
            brain = (
                brain_for(creature_id)
                if callable(brain_for)
                else None
            )
            creatures.append(
                {
                    "creature_id": creature_id,
                    "name": creature.name,
                    "position": tuple(creature.position),
                    "heading": creature.heading,
                    "velocity": (body.velocity.x, body.velocity.y),
                    "angular_velocity": body.angular_velocity,
                    "energy": creature.energy,
                    "total_energy_gathered": max(
                        0.0,
                        float(
                            getattr(creature, "total_energy_gathered", 0.0)
                        ),
                    ),
                    "age_seconds": float(
                        getattr(creature, "age_seconds", 0.0)
                    ),
                    "last_birth_time": float(
                        getattr(creature, "last_birth_time", -1_000_000.0)
                    ),
                    "lifetime_offspring_count": int(
                        getattr(creature, "lifetime_offspring_count", 0)
                    ),
                    "life": float(getattr(creature, "life", 1.0)),
                    "stomach_energy": max(
                        0.0,
                        float(getattr(creature, "stomach_energy", 0.0)),
                    ),
                    "stomach_difficulty_load": max(
                        0.0,
                        float(
                            getattr(
                                creature,
                                "stomach_difficulty_load",
                                0.0,
                            )
                        ),
                    ),
                    "vision": copy.deepcopy(creature.vision),
                    "physical_traits": copy.deepcopy(creature.physical_traits),
                    "flocking_traits": copy.deepcopy(
                        getattr(creature, "flocking_traits", FlockingTraits())
                    ),
                    "color": tuple(creature.color),
                    "lineage": copy.deepcopy(creature.lineage),
                    "fitness": copy.deepcopy(
                        world.fitness.get(creature.creature_id)
                    ),
                    "chronometer": world._chronometers.get(
                        creature.creature_id, 0.0
                    ),
                    "scheduler_continuation": {
                        "raw_action": _action_to_primitive(
                            getattr(world, "_last_actions", {}).get(creature_id)
                        ),
                        "effective_action": _action_to_primitive(
                            getattr(world, "_effective_actions", {}).get(creature_id)
                        ),
                        "effective_action_is_raw": (
                            getattr(world, "_effective_actions", {}).get(
                                creature_id
                            )
                            is getattr(world, "_last_actions", {}).get(
                                creature_id
                            )
                        ),
                        "social_runtime": SocialRuntime.from_legacy(
                            getattr(
                                world,
                                "_cached_social_intentions",
                                {},
                            ).get(creature_id)
                        ).to_primitive(),
                        "flocking_benchmark_quality": copy.deepcopy(
                            getattr(
                                world,
                                "_flocking_benchmark_quality_by_creature_id",
                                {},
                            ).get(creature_id)
                        ),
                        "brain_herding_state": float(
                            getattr(brain, "herding_state", 0.0)
                        ),
                        "smoothed_rotation": float(
                            getattr(creature, "smoothed_rotation", 0.0)
                        ),
                        "smoothed_acceleration": float(
                            getattr(creature, "smoothed_acceleration", 0.0)
                        ),
                        "rest_intent": float(
                            getattr(creature, "rest_intent", 0.0)
                        ),
                        "smoothed_rest": float(
                            getattr(creature, "smoothed_rest", 0.0)
                        ),
                        "effective_rest": float(
                            getattr(creature, "effective_rest", 0.0)
                        ),
                        "activity": float(
                            getattr(creature, "activity", 0.0)
                        ),
                        "pending_direct_life_damage": float(
                            getattr(
                                creature,
                                "pending_direct_life_damage",
                                0.0,
                            )
                        ),
                        "effective_voluntary_motor_effort": float(
                            getattr(
                                creature,
                                "effective_voluntary_motor_effort",
                                0.0,
                            )
                        ),
                    },
                    "genome_id": neat_controller.genome_id_for(
                        creature_id
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
                    "original_radius": food.original_radius,
                    "cluster_id": getattr(food, "cluster_id", None),
                    "bite_capacity": int(getattr(food, "bite_capacity", 1)),
                    "max_energy": float(food.original_energy_value),
                    "energy_remaining": float(food.energy_value),
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
        input_count = (
            SENSOR_INPUT_COUNT
            if genome_config is None
            else len(genome_config.input_keys)
        )
        output_count = (
            ACTION_OUTPUT_COUNT
            if genome_config is None
            else len(genome_config.output_keys)
        )
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
                "pheromone_metadata": pheromones.state_metadata(),
                "acoustic_signals": copy.deepcopy(acoustics.signals),
            }
        world_config = getattr(world, "config", None)
        flocking_config = getattr(world_config, "flocking", None)
        compatibility_config = getattr(
            flocking_config,
            "compatibility",
            None,
        )
        compatibility_mode = getattr(
            compatibility_config,
            "mode",
            "legacy",
        )

        return {
            "version": CHECKPOINT_VERSION,
            "brain_contract": {
                "inputs": input_count,
                "outputs": output_count,
                "sensor_schema": getattr(
                    getattr(world, "vision", None),
                    "sensor_contract",
                    None,
                ).schema_version
                if getattr(
                    getattr(world, "vision", None),
                    "sensor_contract",
                    None,
                )
                is not None
                else SENSING_SCHEMA_VERSION,
                "action_schema": ACTION_SCHEMA_VERSION,
            },
            "flocking_contract": {
                "compatibility_mode": str(
                    getattr(
                        compatibility_mode,
                        "value",
                        compatibility_mode,
                    )
                ),
            },
            "simulation_id": world.simulation_paths.simulation_id,
            "sim_time": world.elapsed_time,
            "rng_state": world.rng.getstate(),
            "world": {
                "simulation_step": int(
                    getattr(world, "_simulation_step", 0)
                ),
                "mouth_exposures": getattr(
                    world,
                    "_mouth_exposures",
                    None,
                ).state()
                if getattr(world, "_mouth_exposures", None) is not None
                else (),
                "reproduction_accumulator": world._reproduction_accumulator,
                "speciation_adjustment_accumulator": float(
                    getattr(world, "_speciation_adjustment_accumulator", 0.0)
                ),
                "next_creature_id": next_creature_id,
                "time_since_last_quick_save": (
                    world.time_since_last_quick_save
                ),
                "time_since_last_archive_save": (
                    world.time_since_last_archive_save
                ),
                "total_biomass_energy": world.total_biomass_energy,
                "live_food_config": (
                    world.live_food_config.to_primitive()
                    if getattr(world, "live_food_config", None) is not None
                    else None
                ),
                "simulation_speed": world.simulation_speed,
                "is_paused": world.is_paused,
                "selected_creature_id": world.selected_creature_id,
                "behavior_history": copy.deepcopy(
                    getattr(world, "behavior_history", None).state_dict()
                    if getattr(world, "behavior_history", None) is not None
                    else {}
                ),
                "behavior_automatic_cohort": copy.deepcopy(
                    getattr(world, "_behavior_automatic_cohort", {})
                ),
                "behavior_next_sample_time": float(
                    getattr(world, "_behavior_next_sample_time", 0.0)
                ),
                "why_next_probe_time": float(
                    getattr(world, "_why_next_probe_time", 0.0)
                ),
                "behavior_selection_generation": int(
                    getattr(world, "_behavior_selection_generation", 0)
                ),
                "behavior_subject_generation_counter": int(
                    getattr(world, "_behavior_subject_generation_counter", 0)
                ),
                "behavior_food_consumption_count": int(
                    getattr(world, "_behavior_food_consumption_count", 0)
                ),
                "behavior_food_consumed_energy_total": float(
                    getattr(world, "_behavior_food_consumed_energy_total", 0.0)
                ),
                "behavior_consumption_totals": copy.deepcopy(
                    getattr(world, "_behavior_consumption_totals", {})
                ),
                "behavior_active_subjects": copy.deepcopy(
                    getattr(world, "_behavior_active_subjects", {})
                ),
                "behavior_cohort_dirty": bool(
                    getattr(world, "_behavior_cohort_dirty", True)
                ),
                "flocking_telemetry_accumulator": getattr(
                    world,
                    "_flocking_telemetry_accumulator",
                    0.0,
                ),
                "flocking_capture_origin": getattr(
                    world,
                    "_flocking_capture_origin",
                    0.0,
                ),
                "flocking_capture_ordinal": getattr(
                    world,
                    "_flocking_capture_ordinal",
                    1,
                ),
                "flocking_group_tracker": getattr(
                    world,
                    "_flocking_group_tracker",
                    None,
                ).state_dict()
                if getattr(world, "_flocking_group_tracker", None) is not None
                else None,
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
            "food_clusters": copy.deepcopy(
                getattr(world, "food_cluster_manager", None).state_dict()
                if getattr(world, "food_cluster_manager", None) is not None
                else None
            ),
            "population": {
                "genomes": evolution_state["genomes"],
                "generation": neat_controller.population.generation,
                "next_genome_id": next_genome_id,
                "evolution_rng_state": getattr(
                    neat_controller,
                    "evolution_random_state",
                    lambda: None,
                )(),
                **allocator_state,
            },
            "species_manager": {
                "compatibility_threshold": (
                    species_manager.compatibility_threshold
                ),
                "phenotypic_weight": species_manager.phenotypic_weight,
                "flocking_trait_distance_coefficient": (
                    getattr(
                        species_manager,
                        "flocking_trait_distance_coefficient",
                        1.0,
                    )
                ),
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
        default_flocking_traits: FlockingTraits | None = None,
    ) -> dict[int, Any]:
        default_flocking = default_flocking_traits or FlockingTraits()

        def normalized_flocking(value: object) -> FlockingTraits:
            return FlockingTraits(
                separation_gene=getattr(
                    value,
                    "separation_gene",
                    default_flocking.separation_gene,
                ),
                alignment_gene=getattr(
                    value,
                    "alignment_gene",
                    default_flocking.alignment_gene,
                ),
                cohesion_gene=getattr(
                    value,
                    "cohesion_gene",
                    default_flocking.cohesion_gene,
                ),
                social_tag_x=getattr(
                    value,
                    "social_tag_x",
                    default_flocking.social_tag_x,
                ),
                social_tag_y=getattr(
                    value,
                    "social_tag_y",
                    default_flocking.social_tag_y,
                ),
            )

        living_traits = {
            creature_state["genome_id"]: (
                creature_state["physical_traits"],
                creature_state["vision"],
                creature_state.get("flocking_traits", default_flocking),
            )
            for creature_state in creature_states
            if creature_state.get("genome_id") is not None
        }
        migrated: dict[int, Any] = {}
        for species_id, representative in representatives.items():
            if isinstance(representative, tuple) and len(representative) == 4:
                genome, physical_traits, vision, flocking = representative
                migrated[species_id] = (
                    genome,
                    copy.deepcopy(physical_traits),
                    copy.deepcopy(vision),
                    normalized_flocking(flocking),
                )
                continue

            if isinstance(representative, tuple) and len(representative) == 3:
                genome, physical_traits, vision = representative
                genome_id = getattr(genome, "key", None)
                known_traits = living_traits.get(genome_id)
                if known_traits is None:
                    archived = archived_traits.get(genome_id)
                    flocking = getattr(
                        archived,
                        "flocking_traits",
                        default_flocking,
                    )
                else:
                    flocking = known_traits[2]
                migrated[species_id] = (
                    genome,
                    copy.deepcopy(physical_traits),
                    copy.deepcopy(vision),
                    normalized_flocking(flocking),
                )
                continue

            genome_id = getattr(representative, "key", None)
            traits = living_traits.get(genome_id)
            if traits is None:
                archived = archived_traits.get(genome_id)
                if archived is not None:
                    traits = (
                        archived.physical_traits,
                        archived.vision,
                        getattr(
                            archived,
                            "flocking_traits",
                            default_flocking,
                        ),
                    )
            if traits is None:
                raise CheckpointError(
                    "Cannot migrate species "
                    f"{species_id}: representative genome {genome_id!r} "
                    "has no living or archived phenotype."
                )

            physical_traits, vision, flocking_traits = traits
            migrated[species_id] = (
                representative,
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
                normalized_flocking(flocking_traits),
            )
        return migrated

    @staticmethod
    def _reconstruct_species_history(
        world: World,
        neat_controller: NeatBrainController,
        *,
        up_to_time: float | None = None,
    ) -> dict[int, Any]:
        from src.creature.speciation import (
            NeatChangeSummary,
            SpeciesDistanceBreakdown,
            SpeciesRecord,
            SpeciesTraitSnapshot,
            calculate_flocking_trait_distance,
            calculate_phenotypic_distance_components,
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
        flocking_coefficient = getattr(
            manager,
            "flocking_trait_distance_coefficient",
            1.0,
        )

        def complete_representative(
            representative: tuple[Any, ...],
        ) -> tuple[Any, Any, Any, FlockingTraits]:
            if len(representative) == 4:
                genome, physical_traits, vision, flocking_traits = representative
                return genome, physical_traits, vision, flocking_traits
            genome, physical_traits, vision = representative
            return genome, physical_traits, vision, FlockingTraits()

        records: dict[int, SpeciesRecord] = {}
        for species_id, representative in sorted(manager.representatives.items()):
            genome, physical_traits, vision, flocking_traits = (
                complete_representative(representative)
            )
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

            zero = SpeciesTraitSnapshot(
                0.0,
                0.0,
                0.0,
                0.0,
                stomach_capacity=0.0,
                digestion_rate=0.0,
                digestion_efficiency=0.0,
            )
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
                flocking_trait_distance=None,
                weighted_flocking_trait_distance=None,
                flocking_trait_distance_coefficient=(
                    flocking_coefficient
                ),
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
                    flocking_trait_distance=0.0,
                    weighted_flocking_trait_distance=0.0,
                    flocking_trait_distance_coefficient=(
                        flocking_coefficient
                    ),
                    separation_gene_component=0.0,
                    alignment_gene_component=0.0,
                    cohesion_gene_component=0.0,
                    stomach_capacity_component=0.0,
                    digestion_rate_component=0.0,
                    digestion_efficiency_component=0.0,
                    digestive_trait_component=0.0,
                )
            elif can_reconstruct:
                (
                    parent_genome,
                    parent_physical,
                    parent_vision,
                    parent_flocking,
                ) = complete_representative(
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
                    + (
                        components.stomach_capacity
                        + components.digestion_rate
                        + components.digestion_efficiency
                    )
                    / 3.0
                )
                neat_distance = genome.distance(
                    parent_genome,
                    neat_controller.config.genome_config,
                )
                neural_shifts = extract_neural_shifts(parent_genome, genome)
                weighted_distance = (
                    phenotypic_distance * manager.phenotypic_weight
                )
                (
                    flocking_distance,
                    separation_component,
                    alignment_component,
                    cohesion_component,
                ) = calculate_flocking_trait_distance(
                    flocking_traits,
                    parent_flocking,
                )
                weighted_flocking_distance = (
                    flocking_distance
                    * flocking_coefficient
                )
                deltas = SpeciesTraitSnapshot(
                    radius=physical_traits.radius - parent_physical.radius,
                    vision_range=vision.range - parent_vision.range,
                    vision_angle=vision.angle - parent_vision.angle,
                    movement_cost_multiplier=(
                        physical_traits.movement_cost_multiplier
                        - parent_physical.movement_cost_multiplier
                    ),
                    separation_gene=(
                        flocking_traits.separation_gene
                        - parent_flocking.separation_gene
                    ),
                    alignment_gene=(
                        flocking_traits.alignment_gene
                        - parent_flocking.alignment_gene
                    ),
                    cohesion_gene=(
                        flocking_traits.cohesion_gene
                        - parent_flocking.cohesion_gene
                    ),
                    stomach_capacity=(
                        physical_traits.stomach_capacity
                        - parent_physical.stomach_capacity
                    ),
                    digestion_rate=(
                        physical_traits.digestion_rate
                        - parent_physical.digestion_rate
                    ),
                    digestion_efficiency=(
                        physical_traits.digestion_efficiency
                        - parent_physical.digestion_efficiency
                    ),
                )
                distances = SpeciesDistanceBreakdown(
                    neat_distance=neat_distance,
                    phenotypic_distance=phenotypic_distance,
                    weighted_phenotypic_distance=weighted_distance,
                    composite_distance=(
                        neat_distance
                        + weighted_distance
                        + weighted_flocking_distance
                    ),
                    compatibility_threshold=manager.compatibility_threshold,
                    phenotypic_weight=manager.phenotypic_weight,
                    radius_component=components.radius,
                    vision_range_component=components.vision_range,
                    vision_angle_component=components.vision_angle,
                    movement_cost_component=(
                        components.movement_cost_multiplier
                    ),
                    flocking_trait_distance=flocking_distance,
                    weighted_flocking_trait_distance=(
                        weighted_flocking_distance
                    ),
                    flocking_trait_distance_coefficient=(
                        flocking_coefficient
                    ),
                    separation_gene_component=separation_component,
                    alignment_gene_component=alignment_component,
                    cohesion_gene_component=cohesion_component,
                    stomach_capacity_component=(
                        components.stomach_capacity
                    ),
                    digestion_rate_component=components.digestion_rate,
                    digestion_efficiency_component=(
                        components.digestion_efficiency
                    ),
                    digestive_trait_component=(
                        components.stomach_capacity
                        + components.digestion_rate
                        + components.digestion_efficiency
                    )
                    / 3.0,
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
                    flocking_traits,
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
        from src.creature.speciation import (
            NeatChangeSummary,
            SpeciesDistanceBreakdown,
            SpeciesRecord,
            SpeciesTraitSnapshot,
            normalize_neural_shifts,
        )

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

        def normalize_traits(
            value: object | None,
            gene_default: float,
            *,
            bounded_genes: bool,
        ) -> SpeciesTraitSnapshot | None:
            if value is None:
                return None
            genes = [
                float(getattr(value, "separation_gene", gene_default)),
                float(getattr(value, "alignment_gene", gene_default)),
                float(getattr(value, "cohesion_gene", gene_default)),
            ]
            if bounded_genes:
                genes = [max(0.0, min(1.0, gene)) for gene in genes]
            return SpeciesTraitSnapshot(
                radius=float(getattr(value, "radius")),
                vision_range=float(getattr(value, "vision_range")),
                vision_angle=float(getattr(value, "vision_angle")),
                movement_cost_multiplier=float(
                    getattr(value, "movement_cost_multiplier")
                ),
                separation_gene=genes[0],
                alignment_gene=genes[1],
                cohesion_gene=genes[2],
                stomach_capacity=float(
                    getattr(
                        value,
                        "stomach_capacity",
                        0.0 if not bounded_genes else 1.6,
                    )
                ),
                digestion_rate=float(
                    getattr(
                        value,
                        "digestion_rate",
                        0.0 if not bounded_genes else 0.2,
                    )
                ),
                digestion_efficiency=float(
                    getattr(
                        value,
                        "digestion_efficiency",
                        0.0 if not bounded_genes else 0.9,
                    )
                ),
            )

        raw_distances = record.distances
        distances = SpeciesDistanceBreakdown(
            neat_distance=raw_distances.neat_distance,
            phenotypic_distance=raw_distances.phenotypic_distance,
            weighted_phenotypic_distance=(
                raw_distances.weighted_phenotypic_distance
            ),
            composite_distance=raw_distances.composite_distance,
            compatibility_threshold=raw_distances.compatibility_threshold,
            phenotypic_weight=raw_distances.phenotypic_weight,
            radius_component=raw_distances.radius_component,
            vision_range_component=raw_distances.vision_range_component,
            vision_angle_component=raw_distances.vision_angle_component,
            movement_cost_component=raw_distances.movement_cost_component,
            flocking_trait_distance=getattr(
                raw_distances,
                "flocking_trait_distance",
                None,
            ),
            weighted_flocking_trait_distance=getattr(
                raw_distances,
                "weighted_flocking_trait_distance",
                None,
            ),
            flocking_trait_distance_coefficient=getattr(
                raw_distances,
                "flocking_trait_distance_coefficient",
                None,
            ),
            separation_gene_component=getattr(
                raw_distances,
                "separation_gene_component",
                None,
            ),
            alignment_gene_component=getattr(
                raw_distances,
                "alignment_gene_component",
                None,
            ),
            cohesion_gene_component=getattr(
                raw_distances,
                "cohesion_gene_component",
                None,
            ),
            stomach_capacity_component=getattr(
                raw_distances,
                "stomach_capacity_component",
                None,
            ),
            digestion_rate_component=getattr(
                raw_distances,
                "digestion_rate_component",
                None,
            ),
            digestion_efficiency_component=getattr(
                raw_distances,
                "digestion_efficiency_component",
                None,
            ),
            digestive_trait_component=getattr(
                raw_distances,
                "digestive_trait_component",
                None,
            ),
        )
        return SpeciesRecord(
            species_id=record.species_id,
            parent_species_id=record.parent_species_id,
            founder_creature_id=record.founder_creature_id,
            founder_genome_id=record.founder_genome_id,
            emerged_at=record.emerged_at,
            founder_color=_checkpoint_rgb(record.founder_color),
            data_quality=record.data_quality,
            founder_traits=normalize_traits(
                record.founder_traits,
                0.5,
                bounded_genes=True,
            ),
            trait_deltas=normalize_traits(
                record.trait_deltas,
                0.0,
                bounded_genes=False,
            ),
            distances=distances,
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
            neural_shifts=normalize_neural_shifts(
                getattr(record, "neural_shifts", ()) or ()
            ),
        )

    @staticmethod
    def _enrich_species_neat_changes(
        history: dict[int, Any],
        neat_controller: NeatBrainController,
    ) -> dict[int, Any]:
        from src.creature.speciation import (
            SpeciesRecord,
            extract_neural_shifts,
        )

        representatives = neat_controller.species_manager.representatives
        enriched: dict[int, Any] = {}
        for species_id, raw_record in history.items():
            record = PersistenceManager._normalize_species_record(raw_record)
            if not isinstance(record, SpeciesRecord):
                enriched[int(species_id)] = record
                continue
            shifts_need_reconstruction = not record.neural_shifts or any(
                not shift.weights_complete for shift in record.neural_shifts
            )
            if not shifts_need_reconstruction:
                enriched[int(species_id)] = record
                continue
            if record.parent_species_id is None:
                enriched[int(species_id)] = record
                continue
            child = representatives.get(int(species_id))
            parent = representatives.get(int(record.parent_species_id))
            if (
                isinstance(child, tuple)
                and len(child) == 4
                and isinstance(parent, tuple)
                and len(parent) == 4
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
        from src.creature.speciation import SpeciesRecord

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
            from src.creature.speciation import (
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
                        getattr(founder, "flocking_traits", FlockingTraits()),
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
        *,
        allow_brain_contract_reset: bool = False,
    ) -> World:
        """Restore a versioned checkpoint into the composed runtime model.

        Parameters
        ----------
        state
            Deserialized version-24 flat checkpoint state.
        config
            Runtime configuration used to normalize legacy fallback values.
        simulation_paths
            Filesystem paths attached to the restored world.
        allow_brain_contract_reset
            Whether incompatible neural sensor contracts may be regenerated.

        Returns
        -------
        World
            Restored world with genotype aggregates and rebound runtime services.

        Raises
        ------
        ValueError
            If checkpoint data violates required persistence invariants.

        Notes
        -----
        Legacy normalization intentionally remains separate from live genotype
        validation because old checkpoints have different fallback semantics.
        """
        # Import compatibility symbols locally so old pickle paths remain valid.
        from src.creature import LineageInfo, PhysicalTraits, TraitMutationDelta
        from src.food import Food
        from src.world import ArchivedCreatureTraits, World

        trait_config = config.trait
        default_flocking_traits = FlockingTraits(
            separation_gene=trait_config.default_separation_gene,
            alignment_gene=trait_config.default_alignment_gene,
            cohesion_gene=trait_config.default_cohesion_gene,
            social_tag_x=trait_config.default_social_tag_x,
            social_tag_y=trait_config.default_social_tag_y,
        )

        def normalized_flocking(value: object | None) -> FlockingTraits:
            value = value or default_flocking_traits
            return FlockingTraits(
                separation_gene=getattr(
                    value,
                    "separation_gene",
                    default_flocking_traits.separation_gene,
                ),
                alignment_gene=getattr(
                    value,
                    "alignment_gene",
                    default_flocking_traits.alignment_gene,
                ),
                cohesion_gene=getattr(
                    value,
                    "cohesion_gene",
                    default_flocking_traits.cohesion_gene,
                ),
                social_tag_x=getattr(
                    value,
                    "social_tag_x",
                    default_flocking_traits.social_tag_x,
                ),
                social_tag_y=getattr(
                    value,
                    "social_tag_y",
                    default_flocking_traits.social_tag_y,
                ),
            )

        def normalized_lineage(value: object) -> LineageInfo:
            delta = getattr(value, "mutation_delta", None)
            return LineageInfo(
                parent_id=getattr(value, "parent_id", None),
                generation=int(getattr(value, "generation", 0)),
                species_id=int(getattr(value, "species_id", 1)),
                mutation_delta=TraitMutationDelta(
                    vision_range=float(getattr(delta, "vision_range", 0.0)),
                    vision_angle=float(getattr(delta, "vision_angle", 0.0)),
                    radius=float(getattr(delta, "radius", 0.0)),
                    movement_cost_multiplier=float(
                        getattr(delta, "movement_cost_multiplier", 0.0)
                    ),
                    stomach_capacity=float(
                        getattr(delta, "stomach_capacity", 0.0)
                    ),
                    digestion_rate=float(
                        getattr(delta, "digestion_rate", 0.0)
                    ),
                    digestion_efficiency=float(
                        getattr(delta, "digestion_efficiency", 0.0)
                    ),
                    separation_gene=float(
                        getattr(delta, "separation_gene", 0.0)
                    ),
                    alignment_gene=float(
                        getattr(delta, "alignment_gene", 0.0)
                    ),
                    cohesion_gene=float(getattr(delta, "cohesion_gene", 0.0)),
                    social_tag_x=float(getattr(delta, "social_tag_x", 0.0)),
                    social_tag_y=float(getattr(delta, "social_tag_y", 0.0)),
                ),
            )

        def normalized_physical(
            value: object,
            stomach_energy: float = 0.0,
        ) -> PhysicalTraits:
            def finite_or_default(raw: object, default: float) -> float:
                try:
                    parsed = float(raw)
                except (TypeError, ValueError):
                    return default
                return parsed if isfinite(parsed) else default

            radius = finite_or_default(
                getattr(value, "radius", trait_config.default_radius),
                trait_config.default_radius,
            )
            legacy_capacity = (
                radius * config.metabolism.stomach_capacity_per_radius
            )
            capacity = finite_or_default(
                getattr(value, "stomach_capacity", legacy_capacity),
                legacy_capacity,
            )
            saved_stomach = finite_or_default(stomach_energy, 0.0)
            capacity = max(capacity, max(0.0, saved_stomach))
            return PhysicalTraits(
                radius=max(
                    trait_config.min_radius,
                    min(trait_config.max_radius, radius),
                ),
                movement_cost_multiplier=max(
                    trait_config.min_movement_cost_multiplier,
                    min(
                        trait_config.max_movement_cost_multiplier,
                        finite_or_default(
                            getattr(
                                value,
                                "movement_cost_multiplier",
                                trait_config.default_movement_cost_multiplier,
                            ),
                            trait_config.default_movement_cost_multiplier,
                        ),
                    ),
                ),
                stomach_capacity=max(
                    trait_config.min_stomach_capacity,
                    min(trait_config.max_stomach_capacity, capacity),
                ),
                digestion_rate=max(
                    trait_config.min_digestion_rate,
                    min(
                        trait_config.max_digestion_rate,
                        finite_or_default(
                            getattr(
                                value,
                                "digestion_rate",
                                trait_config.default_digestion_rate,
                            ),
                            trait_config.default_digestion_rate,
                        ),
                    ),
                ),
                digestion_efficiency=max(
                    trait_config.min_digestion_efficiency,
                    min(
                        trait_config.max_digestion_efficiency,
                        finite_or_default(
                            getattr(
                                value,
                                "digestion_efficiency",
                                trait_config.default_digestion_efficiency,
                            ),
                            trait_config.default_digestion_efficiency,
                        ),
                    ),
                ),
            )

        for creature_state in state["creatures"]:
            saved_stomach_energy = max(
                0.0,
                float(creature_state.get("stomach_energy", 0.0)),
            )
            creature_state["physical_traits"] = normalized_physical(
                creature_state["physical_traits"],
                saved_stomach_energy,
            )
            creature_state["flocking_traits"] = normalized_flocking(
                creature_state.get("flocking_traits")
            )
            creature_state["lineage"] = normalized_lineage(
                creature_state["lineage"]
            )

        normalized_archives: dict[int, ArchivedCreatureTraits] = {}
        for genome_id, archived in state.get("archived_traits", {}).items():
            normalized_archives[int(genome_id)] = ArchivedCreatureTraits(
                creature_id=int(archived.creature_id),
                vision=copy.deepcopy(archived.vision),
                physical_traits=normalized_physical(
                    archived.physical_traits
                ),
                color=copy.deepcopy(archived.color),
                lineage=normalized_lineage(archived.lineage),
                flocking_traits=normalized_flocking(
                    getattr(archived, "flocking_traits", None)
                ),
            )
        state["archived_traits"] = normalized_archives

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
            saved_live_food_config = runtime.get("live_food_config")
            if saved_live_food_config is not None:
                world.apply_live_food_config(
                    LiveFoodConfig.from_primitive(
                        saved_live_food_config,
                        fallback=world.live_food_config,
                    )
                )
            # Render-loop debt and lag telemetry are session state. Never
            # replay wall-clock backlog captured by an earlier process.
            world._physics_accumulator = 0.0
            world.simulation_lag_metrics = type(
                world.simulation_lag_metrics
            )()
            saved_step = runtime.get("simulation_step")
            world._simulation_step = (
                max(0, int(saved_step))
                if saved_step is not None
                else max(
                    0,
                    int(round(world.elapsed_time / world.fixed_timestep)),
                )
            )
            world._mouth_exposures.restore(
                runtime.get("mouth_exposures", ())
            )
            world._reproduction_accumulator = runtime["reproduction_accumulator"]
            world._speciation_adjustment_accumulator = max(
                0.0,
                float(runtime.get("speciation_adjustment_accumulator", 0.0)),
            )
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
            behavior_history = getattr(world, "behavior_history", None)
            if behavior_history is not None:
                behavior_history.restore_state(
                    runtime.get("behavior_history", {})
                )
            world._behavior_automatic_cohort = {
                int(species_id): tuple(int(value) for value in creature_ids)
                for species_id, creature_ids in runtime.get(
                    "behavior_automatic_cohort",
                    {},
                ).items()
            }
            restore_behavior_runtime = (
                "behavior_next_sample_time" in runtime
                and "why_next_probe_time" in runtime
            )
            if restore_behavior_runtime:
                behavior_deadline = float(runtime["behavior_next_sample_time"])
                why_deadline = float(runtime["why_next_probe_time"])
                world._behavior_next_sample_time = (
                    behavior_deadline
                    if isfinite(behavior_deadline) and behavior_deadline >= 0.0
                    else world.elapsed_time
                )
                world._why_next_probe_time = (
                    why_deadline
                    if isfinite(why_deadline) and why_deadline >= 0.0
                    else world.elapsed_time
                )
                world._behavior_selection_generation = max(
                    0,
                    int(runtime.get("behavior_selection_generation", 0)),
                )
                world._behavior_subject_generation_counter = max(
                    0,
                    int(
                        runtime.get(
                            "behavior_subject_generation_counter",
                            0,
                        )
                    ),
                )
                world._behavior_food_consumption_count = max(
                    0,
                    int(runtime.get("behavior_food_consumption_count", 0)),
                )
                consumed_energy = float(
                    runtime.get("behavior_food_consumed_energy_total", 0.0)
                )
                world._behavior_food_consumed_energy_total = (
                    max(0.0, consumed_energy)
                    if isfinite(consumed_energy)
                    else 0.0
                )
                world._behavior_consumption_totals = {
                    int(creature_id): (
                        max(0, int(values[0])),
                        max(0.0, float(values[1])),
                    )
                    for creature_id, values in runtime.get(
                        "behavior_consumption_totals",
                        {},
                    ).items()
                    if isinstance(values, (tuple, list)) and len(values) == 2
                }
                world._behavior_active_subjects = {
                    int(creature_id): max(0, int(generation))
                    for creature_id, generation in runtime.get(
                        "behavior_active_subjects",
                        {},
                    ).items()
                }
                world._behavior_cohort_dirty = bool(
                    runtime.get("behavior_cohort_dirty", True)
                )
            world._flocking_telemetry_accumulator = float(
                runtime.get("flocking_telemetry_accumulator", 0.0)
            )
            if (
                "flocking_capture_origin" in runtime
                and "flocking_capture_ordinal" in runtime
            ):
                world._flocking_capture_origin = float(
                    runtime["flocking_capture_origin"]
                )
                world._flocking_capture_ordinal = max(
                    1,
                    int(runtime["flocking_capture_ordinal"]),
                )
            else:
                world._flocking_capture_origin = (
                    world.elapsed_time
                    - world._flocking_telemetry_accumulator
                )
                world._flocking_capture_ordinal = 1
            world._flocking_capture_due_this_step = False
            if not restore_behavior_runtime:
                world._behavior_cohort_dirty = True
            world._flocking_group_tracker.restore(
                runtime.get("flocking_group_tracker")
            )
            communication_state = state.get("communication")
            if communication_state:
                pheromone_metadata = communication_state.get(
                    "pheromone_metadata"
                )
                if state.get("version") == CHECKPOINT_VERSION and (
                    pheromone_metadata is None
                ):
                    raise ValueError(
                        "Current checkpoint is missing pheromone metadata."
                    )
                world.pheromones.restore(
                    communication_state["trail"],
                    communication_state["alarm"],
                    communication_state.get("pheromone_accumulator", 0.0),
                    pheromone_metadata,
                )
                saved_signals = communication_state.get("acoustic_signals", {})
                world.acoustics.replace_signals(
                    saved_signals.values()
                    if isinstance(saved_signals, dict)
                    else saved_signals
                )

            controller = world.neat_controller
            population_state = state["population"]
            contract = state.get("brain_contract", {"inputs": 23, "outputs": 8})
            saved_sensor_schema = int(contract.get("sensor_schema", 1))
            saved_action_schema = int(contract.get("action_schema", 0))
            saved_input_count = int(contract.get("inputs", 23))
            saved_output_count = int(contract.get("outputs", 8))
            reset_brain_epoch = (
                saved_sensor_schema
                != world.vision.sensor_contract.schema_version
                or saved_action_schema != ACTION_SCHEMA_VERSION
                or saved_input_count
                != len(controller.config.genome_config.input_keys)
                or saved_output_count != ACTION_OUTPUT_COUNT
            )
            world.brain_contract_reset_occurred = reset_brain_epoch
            if reset_brain_epoch:
                message = (
                    "Checkpoint brain contract "
                    f"(sensor schema {saved_sensor_schema}, action schema "
                    f"{saved_action_schema}, {saved_input_count} inputs, "
                    f"{saved_output_count} outputs) is incompatible with the "
                    "requested contract "
                    f"(sensor schema {world.vision.sensor_contract.schema_version}, "
                    f"action schema {ACTION_SCHEMA_VERSION}, "
                    f"{len(controller.config.genome_config.input_keys)} inputs, "
                    f"{ACTION_OUTPUT_COUNT} outputs)."
                )
                if not allow_brain_contract_reset:
                    raise CheckpointContractError(
                        message
                        + " Pass allow_brain_contract_reset=True to explicitly "
                        "discard evolved neural/species state and start a fresh epoch."
                    )
                LOGGER.warning(
                    "Checkpoint brain contract (sensor schema %s, action schema "
                    "%s, %s inputs, %s outputs) is incompatible with the current contract "
                    "(sensor schema %s, action schema %s, %s inputs, %s outputs); "
                    "preserving biological world state and starting a fresh "
                    "neural/species epoch.",
                    saved_sensor_schema,
                    saved_action_schema,
                    saved_input_count,
                    saved_output_count,
                    world.vision.sensor_contract.schema_version,
                    ACTION_SCHEMA_VERSION,
                    len(controller.config.genome_config.input_keys),
                    ACTION_OUTPUT_COUNT,
                )
            controller.population.population = population_state["genomes"]
            controller.population.generation = population_state["generation"]
            saved_evolution_rng_state = population_state.get(
                "evolution_rng_state"
            )
            restore_evolution_rng = getattr(
                controller,
                "restore_evolution_random_state",
                None,
            )
            if (
                saved_evolution_rng_state is not None
                and callable(restore_evolution_rng)
            ):
                restore_evolution_rng(saved_evolution_rng_state)
            species_state = state["species_manager"]
            controller.species_manager.compatibility_threshold = species_state[
                "compatibility_threshold"
            ]
            controller.species_manager.phenotypic_weight = species_state.get(
                "phenotypic_weight",
                controller.species_manager.phenotypic_weight,
            )
            controller.species_manager.flocking_trait_distance_coefficient = (
                float(
                    species_state.get(
                        "flocking_trait_distance_coefficient",
                        controller.species_manager.flocking_trait_distance_coefficient,
                    )
                )
            )
            migrated_representatives = (
                PersistenceManager._migrate_species_representatives(
                    species_state["representatives"],
                    state["creatures"],
                    state.get("archived_traits", {}),
                    default_flocking_traits,
                )
            )
            controller.species_manager.representatives = {
                species_id: (
                    genome,
                    normalized_physical(physical_traits),
                    vision,
                    flocking_traits,
                )
                for species_id, (
                    genome,
                    physical_traits,
                    vision,
                    flocking_traits,
                ) in migrated_representatives.items()
            }
            controller.species_manager.next_species_id = species_state[
                "next_species_id"
            ]
            if not reset_brain_epoch:
                controller.restore_evolution_allocators(
                    population_state.get("next_node_id"),
                    population_state.get("innovation_number"),
                    population_state.get("innovation_history"),
                )

            world.fitness = {}
            world._chronometers = {}
            for creature_state in state["creatures"]:
                creature = world._spawn_creature(
                    creature_state["creature_id"],
                    position=creature_state["position"],
                    heading=creature_state["heading"],
                    energy=creature_state["energy"],
                    life=creature_state.get(
                        "life",
                        config.metabolism.max_life
                        * config.metabolism.initial_life_fraction,
                    ),
                    color=creature_state["color"],
                    vision=creature_state["vision"],
                    physical_traits=creature_state["physical_traits"],
                    flocking_traits=creature_state["flocking_traits"],
                    lineage=creature_state["lineage"],
                )
                creature.name = creature_state["name"]
                raw_stomach_energy = float(
                    creature_state.get("stomach_energy", 0.0)
                )
                if not isfinite(raw_stomach_energy):
                    raw_stomach_energy = 0.0
                creature.stomach_energy = min(
                    creature.physical_traits.stomach_capacity,
                    max(0.0, raw_stomach_energy),
                )
                raw_difficulty_load = float(
                    creature_state.get(
                        "stomach_difficulty_load",
                        creature.stomach_energy,
                    )
                )
                if not isfinite(raw_difficulty_load):
                    raw_difficulty_load = creature.stomach_energy
                if creature.stomach_energy <= 0.0:
                    creature.stomach_difficulty_load = 0.0
                else:
                    creature.stomach_difficulty_load = max(
                        creature.stomach_energy
                        * config.metabolism.min_food_difficulty_multiplier,
                        min(
                            creature.stomach_energy
                            * config.metabolism.max_food_difficulty_multiplier,
                            raw_difficulty_load,
                        ),
                    )
                creature.body.velocity = creature_state["velocity"]
                creature.body.angular_velocity = creature_state["angular_velocity"]
                world.creatures.append(creature)
                world._register_living_creature(creature)
                world._initialize_creature_runtime_state(creature)
                fitness = creature_state["fitness"]
                creature.age_seconds = max(
                    0.0,
                    float(
                        creature_state.get(
                            "age_seconds",
                            getattr(fitness, "age_seconds", 0.0),
                        )
                    ),
                )
                creature.last_birth_time = float(
                    creature_state.get(
                        "last_birth_time",
                        getattr(
                            fitness,
                            "last_reproduction_age",
                            -1_000_000.0,
                        ),
                    )
                )
                creature.lifetime_offspring_count = max(
                    0,
                    int(
                        creature_state.get(
                            "lifetime_offspring_count",
                            getattr(fitness, "offspring_count", 0),
                        )
                    ),
                )
                saved_total = creature_state.get("total_energy_gathered")
                if saved_total is None:
                    saved_total = getattr(
                        fitness,
                        "_legacy_energy_gained",
                        getattr(fitness, "energy_gained", 0.0),
                    )
                try:
                    parsed_total = float(saved_total)
                except (TypeError, ValueError, OverflowError):
                    parsed_total = 0.0
                creature.total_energy_gathered = (
                    max(0.0, parsed_total) if isfinite(parsed_total) else 0.0
                )
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
                if not reset_brain_epoch:
                    brain = controller.restore_brain(
                        creature.creature_id,
                        genome_id,
                    )
                    continuation = creature_state.get(
                        "scheduler_continuation"
                    )
                    if isinstance(continuation, dict):
                        raw_action = _action_from_checkpoint(
                            continuation.get("raw_action")
                        )
                        effective_action = _action_from_checkpoint(
                            continuation.get("effective_action")
                        )
                        if raw_action is not None:
                            world._last_actions[creature.creature_id] = (
                                raw_action
                            )
                            creature.last_action = raw_action
                        if effective_action is not None:
                            world._effective_actions[
                                creature.creature_id
                            ] = (
                                raw_action
                                if continuation.get(
                                    "effective_action_is_raw",
                                    False,
                                )
                                and raw_action is not None
                                else effective_action
                            )
                        social_state = continuation.get("social_runtime")
                        if social_state is None:
                            social_state = continuation.get("social_intention")
                        world._cached_social_intentions[
                            creature.creature_id
                        ] = (
                            SocialRuntime.from_primitive(social_state)
                            if "social_runtime" in continuation
                            else SocialRuntime.from_legacy(social_state)
                        )
                        benchmark_quality = continuation.get(
                            "flocking_benchmark_quality"
                        )
                        if benchmark_quality is not None:
                            world._flocking_benchmark_quality_by_creature_id[
                                creature.creature_id
                            ] = float(benchmark_quality)
                        brain.herding_state = float(
                            continuation.get("brain_herding_state", 0.0)
                        )
                        for attribute in (
                            "smoothed_rotation",
                            "smoothed_acceleration",
                            "rest_intent",
                            "smoothed_rest",
                            "effective_rest",
                            "activity",
                            "pending_direct_life_damage",
                            "effective_voluntary_motor_effort",
                        ):
                            setattr(
                                creature,
                                attribute,
                                float(continuation.get(attribute, 0.0)),
                            )

            for food_state in state["foods"]:
                food = Food(
                    id=food_state["id"],
                    x=food_state["position"][0],
                    y=food_state["position"][1],
                    radius=food_state["radius"],
                    energy_density=food_state["energy_density"],
                    cluster_id=food_state.get("cluster_id"),
                    bite_capacity=int(food_state.get("bite_capacity", 1)),
                    max_energy=float(
                        food_state.get(
                            "max_energy",
                            food_state["original_energy_value"],
                        )
                    ),
                    energy_remaining=float(
                        food_state.get(
                            "energy_remaining",
                            food_state["energy_value"],
                        )
                    ),
                )
                food.energy_value = food_state["energy_value"]
                food.original_energy_value = food_state["original_energy_value"]
                food.original_radius = float(
                    food_state.get(
                        "original_radius",
                        (
                            food.original_radius
                            if food.energy_density <= 0.0
                            else (
                                food.original_energy_value
                                / (3.141592653589793 * food.energy_density)
                            )
                            ** 0.5
                        ),
                    )
                )
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
            world.food_cluster_manager.restore_state(
                state.get("food_clusters")
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

            if reset_brain_epoch:
                historical_species_ids = {
                    *(int(species_id) for species_id in world.species_history),
                    *(
                        int(creature.lineage.species_id)
                        for creature in world.creatures
                    ),
                    *(
                        int(species_id)
                        for species_id in controller.species_manager.representatives
                    ),
                }
                new_root_species_id = max(historical_species_ids, default=0) + 1
                world.start_new_sensing_epoch(new_root_species_id)
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

            if not reset_brain_epoch:
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
            if not reset_brain_epoch:
                if restore_behavior_runtime:
                    observer = getattr(world, "behavior_observer", None)
                    if observer is not None:
                        observer.set_focus(
                            world.selected_creature_id,
                            world._behavior_selection_generation,
                        )
                    world.behavior_history.set_active_creatures(
                        set(world._behavior_active_subjects)
                    )
                else:
                    world._reset_behavior_focus(world.selected_creature_id)
                selected = world.selected_creature
                if selected is not None:
                    from src.behavior_observer import ObservationMode

                    world.behavior_history.register_creature(
                        selected.creature_id,
                        selected.name,
                        world.elapsed_time,
                        species_id=selected.lineage.species_id,
                        observation_mode=ObservationMode.FOCAL,
                        observation_generation=(
                            world._behavior_selection_generation
                        ),
                        active=True,
                    )
            species_by_creature_id = {
                creature.creature_id: creature.lineage.species_id
                for creature in world.creatures
            }
            species_by_creature_id.update(
                {
                    archived.creature_id: archived.lineage.species_id
                    for archived in world._trait_archive_by_genome_id.values()
                }
            )
            world.behavior_history.assign_missing_species(
                species_by_creature_id
            )
            living_ids = {creature.creature_id for creature in world.creatures}
            world._behavior_automatic_cohort = {
                species_id: tuple(
                    creature_id
                    for creature_id in creature_ids
                    if creature_id in living_ids
                )
                for species_id, creature_ids in (
                    world._behavior_automatic_cohort.items()
                )
            }
            world._prune_historical_archives()
            if not reset_brain_epoch:
                # Reconcile once more after all genomes, representatives, and
                # archives have been restored and pruned.  The raw count
                # iterator must never lag behind a node already present in a
                # restored genome, even when the saved population contained
                # long-lived RT-NEAT history.
                final_allocator_state = (
                    controller.evolution_allocator_state()
                )
                controller.restore_evolution_allocators(
                    int(final_allocator_state["next_node_id"]),
                    int(final_allocator_state["innovation_number"]),
                    final_allocator_state.get("innovation_history"),
                )
            # Atomic restoration replaces containers; reconnect composed services.
            world.rebind_creature_services()
            world._refresh_stats()
            return world
        except BaseException:
            world.close()
            raise
