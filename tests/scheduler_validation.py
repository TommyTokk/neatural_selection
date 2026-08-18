"""Deterministic state digest and fixed-step harness for scheduler validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
from math import isclose
from pathlib import Path
import pickle
from typing import Any, Callable, Iterable

import numpy as np

from configs.sim_config import SimConfig, build_sim_config
from src.persistence import PersistenceManager, SimulationPaths
from src.world import World


# Reloading reconstructs the Pymunk space from persisted body coordinates.
# Its subsequent floating-point integration can differ by a few 1e-10 while
# remaining on the same deterministic trajectory.  This tolerance is tight
# enough to expose action, cadence, resource, and allocator drift, while all
# discrete state continues to compare exactly.
FLOAT_ABSOLUTE_TOLERANCE = 1e-9
FLOAT_RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class StateDivergence:
    step: int
    field: str
    left: object
    right: object

    def describe(self) -> str:
        return (
            f"first divergence at completed step {self.step}: "
            f"{self.field}: {self.left!r} != {self.right!r}"
        )


@dataclass(frozen=True, slots=True)
class AuthoritativeStateDigest:
    completed_step: int
    value: dict[str, Any]
    compact_hash: str

    @classmethod
    def capture(cls, world: World) -> "AuthoritativeStateDigest":
        value = authoritative_state(world)
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return cls(
            completed_step=world._simulation_step,
            value=value,
            compact_hash=hashlib.blake2b(payload, digest_size=20).hexdigest(),
        )

    def compare(
        self,
        other: "AuthoritativeStateDigest",
    ) -> StateDivergence | None:
        difference = _first_difference(self.value, other.value, "state")
        if difference is None:
            return None
        field, left, right = difference
        return StateDivergence(
            step=min(self.completed_step, other.completed_step),
            field=field,
            left=left,
            right=right,
        )


@dataclass(slots=True)
class SoakMetrics:
    fixed_steps: int = 0
    checkpoints: int = 0
    peak_creatures: int = 0
    peak_pending_exposures: int = 0
    births: int = 0
    deaths: int = 0
    checkpoint_payload_bytes: int = 0


class DeterministicSoakHarness:
    """Run exact fixed-step schedules without consulting wall-clock time."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.metrics = SoakMetrics(
            peak_creatures=len(world.creatures),
            peak_pending_exposures=world._mouth_exposures.count,
        )

    def run_to(
        self,
        target_step: int,
        *,
        digest_steps: Iterable[int] = (),
        before_step: Callable[[World, int], None] | None = None,
    ) -> dict[int, AuthoritativeStateDigest]:
        requested = set(int(step) for step in digest_steps)
        digests: dict[int, AuthoritativeStateDigest] = {}
        while self.world._simulation_step < target_step:
            current = self.world._simulation_step
            if before_step is not None:
                before_step(self.world, current)
            self.world.update(self.world.fixed_timestep)
            self._record_metrics()
            completed = self.world._simulation_step
            if completed in requested:
                digests[completed] = AuthoritativeStateDigest.capture(
                    self.world
                )
        return digests

    def checkpoint_reload(self) -> World:
        state = PersistenceManager._capture_state(
            self.world,
            self.world.neat_controller,
        )
        self.metrics.checkpoints += 1
        self.metrics.checkpoint_payload_bytes = max(
            self.metrics.checkpoint_payload_bytes,
            len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)),
        )
        restored = PersistenceManager._restore_world(
            state,
            self.world.config,
            self.world.simulation_paths,
        )
        previous = self.world
        self.world = restored
        previous.close()
        self._record_metrics()
        return restored

    def _record_metrics(self) -> None:
        self.metrics.fixed_steps = self.world._simulation_step
        self.metrics.peak_creatures = max(
            self.metrics.peak_creatures,
            len(self.world.creatures),
        )
        self.metrics.peak_pending_exposures = max(
            self.metrics.peak_pending_exposures,
            self.world._mouth_exposures.count,
        )
        self.metrics.births = int(self.world.rt_neat.stats.births)
        self.metrics.deaths = int(self.world.rt_neat.stats.deaths)


def validation_config(
    *,
    seed: int = 11,
    creatures: int = 4,
    foods: int = 8,
    behavior_enabled: bool = False,
) -> SimConfig:
    config = build_sim_config()
    config.random_seed = seed
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    config.behavior.enabled = behavior_enabled
    config.counterfactual_why.enabled = behavior_enabled
    config.population.initial_creatures = creatures
    config.food.initial_food_items = foods
    # Scheduler determinism fixtures exercise ordinary-pellet behavior. Cluster
    # lifecycle and shared grazing have dedicated focused tests.
    config.food_clusters.cluster_spawn_share = 0.0
    return config


def validation_world(
    *,
    seed: int = 11,
    creatures: int = 4,
    foods: int = 8,
    behavior_enabled: bool = False,
    paths: SimulationPaths | None = None,
) -> World:
    return World(
        validation_config(
            seed=seed,
            creatures=creatures,
            foods=foods,
            behavior_enabled=behavior_enabled,
        ),
        simulation_paths=(
            paths
            if paths is not None
            else SimulationPaths(Path(".").resolve())
        ),
    )


def assert_authoritative_match(
    testcase: Any,
    left: World,
    right: World,
) -> None:
    left_digest = AuthoritativeStateDigest.capture(left)
    right_digest = AuthoritativeStateDigest.capture(right)
    difference = left_digest.compare(right_digest)
    testcase.assertIsNone(
        difference,
        None if difference is None else difference.describe(),
    )


def authoritative_state(world: World) -> dict[str, Any]:
    controller = world.neat_controller
    creatures = []
    for creature in sorted(world.creatures, key=lambda item: item.creature_id):
        creature_id = creature.creature_id
        brain = controller.brain_for(creature_id)
        genome_id = controller.genome_id_for(creature_id)
        genome = (
            None
            if genome_id is None
            else controller.population.population.get(genome_id)
        )
        creatures.append(
            {
                "creature_id": creature_id,
                "species_lineage": _canonical(creature.lineage),
                "position": tuple(creature.position),
                "velocity": tuple(creature.body.velocity),
                "heading": float(creature.heading),
                "angular_velocity": float(creature.body.angular_velocity),
                "energy": float(creature.energy),
                "total_energy_gathered": float(
                    creature.total_energy_gathered
                ),
                "life": float(creature.life),
                "stomach_energy": float(creature.stomach_energy),
                "stomach_difficulty_load": float(
                    creature.stomach_difficulty_load
                ),
                "age_fitness": _canonical(world.fitness.get(creature_id)),
                "chronometer": float(
                    world._chronometers.get(creature_id, 0.0)
                ),
                "smoothing": (
                    float(creature.smoothed_rotation),
                    float(creature.smoothed_acceleration),
                    float(creature.rest_intent),
                    float(creature.smoothed_rest),
                    float(creature.effective_rest),
                    float(creature.activity),
                    float(creature.effective_voluntary_motor_effort),
                ),
                "pending_direct_life_damage": float(
                    creature.pending_direct_life_damage
                ),
                "carried_food_id": getattr(
                    world,
                    "_held_food_by_creature_id",
                    {},
                ).get(creature_id),
                "phenotype": (
                    _canonical(creature.vision),
                    _canonical(creature.physical_traits),
                    _canonical(creature.flocking_traits),
                    tuple(creature.color),
                ),
                "genome_id": genome_id,
                "genome": _canonical(genome),
                "brain_herding_state": float(
                    getattr(brain, "herding_state", 0.0)
                ),
                "decision_phase": world._decision_phase(creature_id),
            }
        )

    foods = [
        {
            "food_id": food.id,
            "position": tuple(food.position),
            "velocity": tuple(food.body.velocity),
            "angle": float(food.body.angle),
            "angular_velocity": float(food.body.angular_velocity),
            "radius": float(food.radius),
            "energy_density": float(food.energy_density),
            "remaining_quantity": float(food.energy_value),
            "original_energy_value": float(food.original_energy_value),
            "original_radius": float(food.original_radius),
            "carrier_id": getattr(world, "_carrier_by_food_id", {}).get(
                food.id
            ),
        }
        for food in sorted(world.foods, key=lambda item: item.id)
    ]

    pheromones = world.pheromones
    acoustics = world.acoustics
    spawner = world.food_spawner
    species_manager = controller.species_manager
    return {
        "world": {
            "simulation_step": int(world._simulation_step),
            "completed_simulated_time": float(world.elapsed_time),
            "seed": int(world.config.random_seed),
            "rng_state": _canonical(world.rng.getstate()),
            "food_spawner": (
                int(spawner._next_food_id),
                float(spawner._spawn_credit),
                float(spawner._low_food_burst_credit),
                int(spawner._pending_low_food_burst_items),
            ),
            "pheromones": {
                "accumulator": float(pheromones.accumulator),
                "trail": np.array(pheromones.trail, copy=True),
                "alarm": np.array(pheromones.alarm, copy=True),
            },
            "acoustic_signals": _canonical(acoustics.signals),
            "observer_deadline": float(world._behavior_next_sample_time),
            "counterfactual_deadline": float(world._why_next_probe_time),
            "reproduction_accumulator": float(
                world._reproduction_accumulator
            ),
            "speciation_accumulator": float(
                world._speciation_adjustment_accumulator
            ),
            "pending_resource_transactions": (),
            "mouth_exposure_active_count": int(
                world._mouth_exposures.count
            ),
            "mouth_exposure_records": world._mouth_exposures.state(),
            "held_foods": tuple(
                sorted(world._held_food_by_creature_id.items())
            ),
            "food_carriers": tuple(sorted(world._carrier_by_food_id.items())),
            "next_creature_id": int(world._next_creature_id_value),
            "total_biomass_energy": float(world.total_biomass_energy),
            "rt_neat_events": (
                int(world.rt_neat.stats.births),
                int(world.rt_neat.stats.normal_replacements),
                int(world.rt_neat.stats.extinction_replacements),
                int(world.rt_neat.stats.deaths),
                float(world.rt_neat._lifespan_at_death_total),
                int(world.rt_neat._lifespan_at_death_count),
            ),
            "eligible_parent_ids": tuple(world.rt_neat.eligible_parent_ids),
            "species_threshold": float(
                species_manager.compatibility_threshold
            ),
            "next_species_id": int(species_manager.next_species_id),
            "evolution_rng_state": _canonical(
                controller.evolution_random_state()
            ),
            "evolution_allocators": _canonical(
                controller.evolution_allocator_state()
            ),
            "species_representatives": _canonical(
                species_manager.representatives
            ),
            "species_history": _canonical(world.species_history),
            "fitness_archive": _canonical(world.fitness_archive),
            "archived_traits": _canonical(
                world._trait_archive_by_genome_id
            ),
        },
        "creatures": tuple(creatures),
        "foods": tuple(foods),
    }


def _array_hash(value: np.ndarray) -> tuple[str, tuple[int, ...], str]:
    contiguous = np.ascontiguousarray(value)
    return (
        str(contiguous.dtype),
        tuple(contiguous.shape),
        hashlib.blake2b(contiguous.tobytes(), digest_size=20).hexdigest(),
    )


def _pickle_hash(value: object) -> str:
    return hashlib.blake2b(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
        digest_size=20,
    ).hexdigest()


def _structural_hash(value: object) -> str:
    # Pickle preserves aliasing, so equal value trees can produce different
    # bytes when one was reconstructed with a different object-sharing graph.
    # The canonical representation deliberately compares value, not aliasing.
    return hashlib.blake2b(
        repr(_canonical(value)).encode("utf-8"),
        digest_size=20,
    ).hexdigest()


def _canonical(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, np.ndarray):
        return _array_hash(value)
    if isinstance(value, dict):
        return tuple(
            (key, _canonical(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_canonical(item) for item in value), key=repr))
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return (
            type(value).__qualname__,
            _canonical(attributes),
        )
    slot_names: list[str] = []
    for cls in type(value).__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            slot_names.append(slots)
        else:
            slot_names.extend(slots)
    if slot_names:
        return (
            type(value).__qualname__,
            tuple(
                (name, _canonical(getattr(value, name)))
                for name in sorted(set(slot_names))
                if hasattr(value, name)
            ),
        )
    return repr(value)


def _first_difference(
    left: object,
    right: object,
    path: str,
) -> tuple[str, object, object] | None:
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype:
            return f"{path}.dtype", str(left.dtype), str(right.dtype)
        if left.shape != right.shape:
            return f"{path}.shape", left.shape, right.shape
        matches = np.isclose(
            left,
            right,
            rtol=FLOAT_RELATIVE_TOLERANCE,
            atol=FLOAT_ABSOLUTE_TOLERANCE,
            equal_nan=True,
        )
        if bool(np.all(matches)):
            return None
        first_index = tuple(int(value) for value in np.argwhere(~matches)[0])
        suffix = "".join(f"[{value}]" for value in first_index)
        return (
            f"{path}{suffix}",
            float(left[first_index]),
            float(right[first_index]),
        )
    if isinstance(left, float) and isinstance(right, float):
        if isclose(
            left,
            right,
            rel_tol=FLOAT_RELATIVE_TOLERANCE,
            abs_tol=FLOAT_ABSOLUTE_TOLERANCE,
        ):
            return None
        return path, left, right
    if type(left) is not type(right):
        return path, left, right
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}.keys", tuple(left), tuple(right)
        for key in left:
            difference = _first_difference(
                left[key],
                right[key],
                f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return f"{path}.length", len(left), len(right)
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_item,
                right_item,
                f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if left != right:
        return path, left, right
    return None
