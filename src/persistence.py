from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
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


CHECKPOINT_VERSION = 2


class CheckpointError(RuntimeError):
    pass


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


class PersistenceManager:
    def __init__(self) -> None:
        self._condition = Condition()
        self._pending_save: (
            tuple[dict[str, Any], tuple[CheckpointTarget, ...]] | None
        ) = None
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

    def save_simulation(
        self,
        world: World,
        neat_controller: NeatBrainController,
        targets: tuple[CheckpointTarget, ...],
    ) -> None:
        if not targets:
            return
        state = self._capture_state(world, neat_controller)
        with self._condition:
            if self._closing:
                raise RuntimeError("PersistenceManager is closed.")
            pending_targets: dict[Path, CheckpointTarget] = {}
            if self._pending_save is not None:
                pending_targets.update(
                    (target.path, target)
                    for target in self._pending_save[1]
                )
            pending_targets.update((target.path, target) for target in targets)
            # Coalescing uses the newest state while retaining every distinct
            # quick/archive destination already waiting to be written.
            self._pending_save = (state, tuple(pending_targets.values()))
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
            state, targets = save_request
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
        return {
            "version": CHECKPOINT_VERSION,
            "simulation_id": world.simulation_paths.simulation_id,
            "sim_time": world.elapsed_time,
            "rng_state": world.rng.getstate(),
            "world": {
                "physics_accumulator": world._physics_accumulator,
                "reproduction_accumulator": world._reproduction_accumulator,
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
                "previous_biome": copy.deepcopy(
                    world._previous_biome_here_by_creature_id
                ),
                "held_foods": copy.deepcopy(world._held_food_by_creature_id),
                "food_carriers": copy.deepcopy(world._carrier_by_food_id),
            },
            "creatures": creatures,
            "foods": foods,
            "food_spawner": {
                "next_food_id": spawner._next_food_id,
                "spawn_credit": spawner._spawn_credit,
                "burst_credit": spawner._burst_credit,
                "low_food_burst_credit": spawner._low_food_burst_credit,
                "pending_burst_items": spawner._pending_burst_items,
                "pending_low_food_burst_items": (
                    spawner._pending_low_food_burst_items
                ),
            },
            "population": {
                "genomes": evolution_state["genomes"],
                "generation": neat_controller.population.generation,
            },
            "species_manager": {
                "compatibility_threshold": (
                    species_manager.compatibility_threshold
                ),
                "representatives": evolution_state["representatives"],
                "next_species_id": species_manager.next_species_id,
            },
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
        if version != CHECKPOINT_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {version!r}; "
                f"expected {CHECKPOINT_VERSION}."
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
            world._previous_biome_here_by_creature_id = runtime["previous_biome"]

            controller = world.neat_controller
            population_state = state["population"]
            controller.population.population = population_state["genomes"]
            controller.population.generation = population_state["generation"]
            species_state = state["species_manager"]
            controller.species_manager.compatibility_threshold = species_state[
                "compatibility_threshold"
            ]
            controller.species_manager.representatives = species_state[
                "representatives"
            ]
            controller.species_manager.next_species_id = species_state[
                "next_species_id"
            ]

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
                creature.body.velocity = creature_state["velocity"]
                creature.body.angular_velocity = creature_state["angular_velocity"]
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
            spawner._burst_credit = spawner_state["burst_credit"]
            spawner._low_food_burst_credit = spawner_state[
                "low_food_burst_credit"
            ]
            spawner._pending_burst_items = spawner_state["pending_burst_items"]
            spawner._pending_low_food_burst_items = spawner_state[
                "pending_low_food_burst_items"
            ]

            world._held_food_by_creature_id = runtime["held_foods"]
            world._carrier_by_food_id = runtime["food_carriers"]
            world._sync_carried_foods()
            world.fitness_archive = state["fitness_archive"]
            world._trait_archive_by_genome_id = state["archived_traits"]

            rt_state = state["rt_neat"]
            world.rt_neat.stats = rt_state["stats"]
            world.rt_neat.eligible_parent_ids = rt_state["eligible_parent_ids"]
            world.rt_neat._lifespan_at_death_total = rt_state[
                "lifespan_at_death_total"
            ]
            world.rt_neat._lifespan_at_death_count = rt_state[
                "lifespan_at_death_count"
            ]
            world._refresh_stats()
            return world
        except BaseException:
            world.close()
            raise
