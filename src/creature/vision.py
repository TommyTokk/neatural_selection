from __future__ import annotations

from collections.abc import Callable, MutableSequence, Sequence
from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin, sqrt

import numpy as np

from configs.sim_config import (
    FlockingConfig,
    MetabolismConfig,
    VisionConfig,
)
from src.creature.flocking import LongRangeSocialObservation
from src.creature.model import Creature
from src.creature.communication import AcousticObservation, PheromoneSnapshot
from src.food import Food

BIOME_GRADIENT_EPSILON = 0.001
PHEROMONE_GRADIENT_EPSILON = 0.001

SENSOR_INPUT_NAMES = (
    "constant",
    "feeding_drive",
    "reproductive_readiness",
    "energy_percent",
    "speed",
    "creature_count",
    "food_count",
    "clock_tik_tok",
    "clock_chronometer",
    "clock_time_alive",
    "food_proximity",
    "food_angle",
    "creature_proximity",
    "creature_angle",
    "wall_proximity",
    "wall_angle",
    "is_grabbing",
    "local_richness",
    "lateral_gradient",
    "forward_gradient",
    "own_infant_proximity",
    "own_infant_angle",
    "flock_presence",
    "flock_effective_count",
    "flock_center_forward",
    "flock_center_right",
    "flock_relative_velocity_forward",
    "flock_relative_velocity_right",
    "long_range_social_intensity",
    "long_range_social_direction_forward",
    "long_range_social_direction_right",
    "stomach_fullness",
    "sound_strength",
    "sound_dir_sin",
    "sound_dir_cos",
    "sound_tone",
    "pheromone_local_red",
    "pheromone_local_green",
    "pheromone_local_blue",
    "pheromone_lateral_red",
    "pheromone_lateral_green",
    "pheromone_lateral_blue",
    "pheromone_forward_red",
    "pheromone_forward_green",
    "pheromone_forward_blue",
    "life_normalized",
)


@dataclass(frozen=True, slots=True)
class SensorContract:
    schema_version: int
    input_names: tuple[str, ...]
    neat_config_path: str

    @property
    def input_count(self) -> int:
        """Execute input count behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
int
    Result produced by this creature-domain operation."""
        # Keep input count behavior explicit in its owning subsystem.
        return len(self.input_names)


SENSOR_CONTRACT = SensorContract(
    8,
    SENSOR_INPUT_NAMES,
    "configs/neat_herbivore.ini",
)


# Public aliases describe the only runtime sensor contract.
SENSOR_INPUT_COUNT = SENSOR_CONTRACT.input_count
SENSING_SCHEMA_VERSION = SENSOR_CONTRACT.schema_version

if len(SENSOR_INPUT_NAMES) != SENSOR_INPUT_COUNT:
    raise RuntimeError("Sensor names must match SensorSnapshot.as_inputs().")


@dataclass(slots=True)
class VisionTargetSnapshot:
    visible: float
    proximity: float
    angle: float
    density: float
    count: int
    nearest_id: int | None = None
    surface_distance: float | None = None
    relative_angle: float | None = None


@dataclass(slots=True)
class BoundarySnapshot:
    pressure: float
    turn: float


@dataclass(slots=True)
class BiomeSensorSnapshot:
    local_richness: float = 0.0
    lateral_gradient: float = 0.0
    forward_gradient: float = 0.0

    @classmethod
    def from_probe_samples(
        cls,
        here: float,
        forward_left: float,
        forward_right: float,
    ) -> BiomeSensorSnapshot:
        """Execute from probe samples behavior.

Parameters
----------
here
    Input used by this creature-domain operation.
forward_left
    Input used by this creature-domain operation.
forward_right
    Input used by this creature-domain operation.
Returns
-------
BiomeSensorSnapshot
    Result produced by this creature-domain operation."""
        # Keep from probe samples behavior explicit in its owning subsystem.
        local = cls._clamp(float(here), 0.0, 1.0)
        left = cls._clamp(float(forward_left), 0.0, 1.0)
        right = cls._clamp(float(forward_right), 0.0, 1.0)
        denominator = local + BIOME_GRADIENT_EPSILON
        lateral = cls._clamp((left - right) / denominator, -1.0, 1.0)
        ahead_average = (left + right) * 0.5
        forward = cls._clamp(
            (ahead_average - local) / denominator,
            -1.0,
            1.0,
        )
        return cls(
            local_richness=local,
            lateral_gradient=lateral,
            forward_gradient=forward,
        )

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Execute clamp behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
minimum
    Input used by this creature-domain operation.
maximum
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep clamp behavior explicit in its owning subsystem.
        return max(minimum, min(maximum, value))


@dataclass(slots=True)
class FlockSensorSnapshot:
    center_proximity: float = 0.0
    center_angle: float = 0.0
    average_relative_heading: float = 0.0
    flockmate_count: float = 0.0
    crowd_separation_absolute_angle: float = 0.0
    cohesion_absolute_angle: float = 0.0
    crowd_separation_strength: float = 0.0
    average_flockmate_proximity: float = 0.0
    average_flockmate_velocity: tuple[float, float] = (0.0, 0.0)
    actual_average_flockmate_velocity: tuple[float, float] = (0.0, 0.0)
    visible_creature_count: int = 0
    compatible_visible_count: int = 0
    visible_personal_space_count: int = 0
    social_presence: float = 0.0
    center_forward: float = 0.0
    center_right: float = 0.0
    relative_velocity_forward: float = 0.0
    relative_velocity_right: float = 0.0
    center_distance: float = 0.0
    mean_neighbor_distance: float = 0.0
    mean_heading_error: float = 0.0
    long_range: LongRangeSocialObservation = field(
        default_factory=LongRangeSocialObservation
    )


@dataclass(slots=True)
class _VisionCandidate:
    kind: str
    source: Food | Creature | None
    surface_distance: float
    signed_angle: float
    closeness: float
    interval: tuple[float, float]


class _CandidateView(Sequence[_VisionCandidate]):
    __slots__ = ("storage", "indices", "count")

    def __init__(
        self,
        storage: list[_VisionCandidate],
        indices: list[int],
        count: int,
    ) -> None:
        """Execute init behavior.

Parameters
----------
storage
    Input used by this creature-domain operation.
indices
    Input used by this creature-domain operation.
count
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.storage = storage
        self.indices = indices
        self.count = count

    def __len__(self) -> int:
        """Execute len behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
int
    Result produced by this creature-domain operation."""
        # Keep len behavior explicit in its owning subsystem.
        return self.count

    def __getitem__(self, item):
        """Execute getitem behavior.
        
        Parameters
        ----------
        item
            Input used by this creature-domain operation.
        Returns
        -------
        None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        IndexError
            If the requested index is outside the valid view.
        """
        # Keep getitem behavior explicit in its owning subsystem.
        if isinstance(item, slice):
            return [self[index] for index in range(*item.indices(self.count))]
        position = int(item)
        if position < 0:
            position += self.count
        if position < 0 or position >= self.count:
            raise IndexError(position)
        return self.storage[self.indices[position]]

    def __iter__(self):
        """Execute iter behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep iter behavior explicit in its owning subsystem.
        for position in range(self.count):
            yield self.storage[self.indices[position]]


@dataclass(slots=True)
class SensorSnapshot:
    food: VisionTargetSnapshot
    creatures: VisionTargetSnapshot
    walls: VisionTargetSnapshot
    boundary: BoundarySnapshot
    energy: float
    speed: float
    vision_range: float
    vision_angle: float
    vision_energy_cost: float
    reproductive_readiness: float
    visible_food_count: float
    visible_creature_count: float
    clock_tik_tok: float
    clock_chronometer: float
    clock_time_alive: float
    is_grabbing: float
    stomach_fullness: float = 0.0
    life_normalized: float = 1.0
    own_infants: VisionTargetSnapshot = field(
        default_factory=lambda: VisionTargetSnapshot(
            visible=0.0,
            proximity=0.0,
            angle=0.0,
            density=0.0,
            count=0,
        )
    )
    biome: BiomeSensorSnapshot = field(default_factory=BiomeSensorSnapshot)
    flock: FlockSensorSnapshot = field(default_factory=FlockSensorSnapshot)
    acoustic: AcousticObservation = field(default_factory=AcousticObservation)
    pheromones: PheromoneSnapshot = field(default_factory=PheromoneSnapshot)
    sensor_contract: SensorContract = SENSOR_CONTRACT
    flock_target_group_size: int = 4

    def write_inputs(self, output: MutableSequence[float]) -> None:
        """Write the ordered sensor contract into caller-owned storage.
        
        Parameters
        ----------
        output
            Input used by this creature-domain operation.
        Returns
        -------
        None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        RuntimeError
            If runtime state violates the callable invariant.
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep write inputs behavior explicit in its owning subsystem.
        expected = self.sensor_contract.input_count
        if len(output) != expected:
            raise ValueError(
                f"Expected an input buffer of length {expected}, got {len(output)}."
            )

        stomach_fullness = self._clamp01(self.stomach_fullness)
        energy_percent = self._clamp01(self.energy)
        energy_deficit = max(0.0, 1.0 - energy_percent)
        stomach_emptiness = max(0.0, 1.0 - stomach_fullness)
        feeding_drive = energy_deficit * stomach_emptiness
        output[0] = 1.0
        output[1] = feeding_drive
        output[2] = self.reproductive_readiness
        output[3] = energy_percent
        output[4] = self.speed
        output[5] = min(self.visible_creature_count / 5.0, 1.0)
        output[6] = min(self.visible_food_count / 10.0, 1.0)
        output[7] = self.clock_tik_tok
        output[8] = self.clock_chronometer
        output[9] = self.clock_time_alive
        output[10] = self.food.proximity
        output[11] = self.food.angle
        output[12] = self.creatures.proximity
        output[13] = self.creatures.angle
        output[14] = self.walls.proximity
        output[15] = self.walls.angle
        output[16] = self.is_grabbing
        output[17] = self.biome.local_richness
        output[18] = self.biome.lateral_gradient
        output[19] = self.biome.forward_gradient
        output[20] = self.own_infants.proximity
        output[21] = self.own_infants.angle
        index = 22
        if self.sensor_contract.schema_version >= 4:
            output[index] = self._clamp01(self.flock.social_presence)
            output[index + 1] = self._target_scaled_flockmate_count(
                self.flock.flockmate_count,
                self.flock_target_group_size,
            )
            output[index + 2] = self._clamp(
                self.flock.center_forward, -1.0, 1.0
            )
            output[index + 3] = self._clamp(
                self.flock.center_right, -1.0, 1.0
            )
            output[index + 4] = self._clamp(
                self.flock.relative_velocity_forward, -1.0, 1.0
            )
            output[index + 5] = self._clamp(
                self.flock.relative_velocity_right, -1.0, 1.0
            )
            output[index + 6] = self._clamp01(
                self.flock.long_range.intensity
            )
            output[index + 7] = self._clamp(
                self.flock.long_range.direction_forward, -1.0, 1.0
            )
            output[index + 8] = self._clamp(
                self.flock.long_range.direction_right, -1.0, 1.0
            )
            index += 9
        else:
            output[index] = self.flock.center_proximity
            output[index + 1] = self.flock.center_angle
            output[index + 2] = self.flock.average_relative_heading
            output[index + 3] = self._normalized_flockmate_count(
                self.flock.flockmate_count
            )
            index += 4

        output[index] = stomach_fullness
        output[index + 1] = self.acoustic.strength
        output[index + 2] = self.acoustic.direction_sin
        output[index + 3] = self.acoustic.direction_cos
        output[index + 4] = self.acoustic.tone
        local = np.clip(np.asarray(self.pheromones.local, dtype=np.float64), 0.0, 1.0)
        left = np.clip(
            np.asarray(self.pheromones.forward_left, dtype=np.float64), 0.0, 1.0
        )
        right = np.clip(
            np.asarray(self.pheromones.forward_right, dtype=np.float64), 0.0, 1.0
        )
        denominator = local + PHEROMONE_GRADIENT_EPSILON
        lateral = np.clip((left - right) / denominator, -1.0, 1.0)
        forward = np.clip(((left + right) * 0.5 - local) / denominator, -1.0, 1.0)
        output[index + 5 : index + 8] = local
        output[index + 8 : index + 11] = lateral
        output[index + 11 : index + 14] = forward
        output[index + 14] = self._clamp01(self.life_normalized)
        if index + 15 != expected:
            raise RuntimeError(
                "Sensor contract input names do not match the written values."
            )

    def as_inputs(self) -> list[float]:
        """Execute as inputs behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
list[float]
    Result produced by this creature-domain operation."""
        # Keep as inputs behavior explicit in its owning subsystem.
        values = [0.0] * self.sensor_contract.input_count
        self.write_inputs(values)
        return values

    @staticmethod
    def _clamp01(value: float) -> float:
        """Execute clamp01 behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep clamp01 behavior explicit in its owning subsystem.
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalized_flockmate_count(value: float) -> float:
        """Execute normalized flockmate count behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalized flockmate count behavior explicit in its owning subsystem.
        effective_count = max(0.0, float(value))
        return effective_count / (effective_count + 3.0)

    @staticmethod
    def _target_scaled_flockmate_count(value: float, target: int) -> float:
        """Execute target scaled flockmate count behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
target
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep target scaled flockmate count behavior explicit in its owning subsystem.
        return max(0.0, min(1.0, float(value) / max(1, int(target))))

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Execute clamp behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
minimum
    Input used by this creature-domain operation.
maximum
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep clamp behavior explicit in its owning subsystem.
        return max(minimum, min(maximum, value))


@dataclass(slots=True)
class VisionSenseResult:
    snapshot: SensorSnapshot
    visible_food_ids: Sequence[int]


class _VisibleFoodIdView(Sequence[int]):
    __slots__ = ("owner",)

    def __init__(self, owner: VisionSystem) -> None:
        """Execute init behavior.

Parameters
----------
owner
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.owner = owner

    def __len__(self) -> int:
        """Execute len behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
int
    Result produced by this creature-domain operation."""
        # Keep len behavior explicit in its owning subsystem.
        return self.owner._visible_food_id_count

    def __getitem__(self, index):
        """Execute getitem behavior.
        
        Parameters
        ----------
        index
            Input used by this creature-domain operation.
        Returns
        -------
        None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        IndexError
            If the requested index is outside the valid view.
        """
        # Keep getitem behavior explicit in its owning subsystem.
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = int(index)
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError(position)
        return self.owner._visible_food_ids[position]


class VisionSystem:
    def __init__(
        self,
        config: VisionConfig,
        eating_distance: float = MetabolismConfig().eating_distance,
        stomach_capacity_per_radius: float = (
            MetabolismConfig().stomach_capacity_per_radius
        ),
        max_life: float = MetabolismConfig().max_life,
        flock_compatibility_resolver: Callable[[Creature, Creature], float]
        | None = None,
        flocking_config: FlockingConfig | None = None,
    ) -> None:
        """Execute init behavior.

Parameters
----------
config
    Input used by this creature-domain operation.
eating_distance
    Input used by this creature-domain operation.
stomach_capacity_per_radius
    Input used by this creature-domain operation.
max_life
    Input used by this creature-domain operation.
flock_compatibility_resolver
    Input used by this creature-domain operation.
flocking_config
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.config = config
        self.eating_distance = eating_distance
        self.stomach_capacity_per_radius = stomach_capacity_per_radius
        self.max_life = max(0.0, float(max_life))
        self.flock_compatibility_resolver = flock_compatibility_resolver
        self.flocking_config = flocking_config or FlockingConfig()
        self.sensor_contract = SENSOR_CONTRACT
        self._candidate_storage: list[_VisionCandidate] = []
        self._candidate_count = 0
        self._visible_indices: list[int] = []
        self._visible_count = 0
        self._blocked_starts: list[float] = []
        self._blocked_ends: list[float] = []
        self._blocked_count = 0
        self._visible_food_ids: list[int] = []
        self._visible_food_id_count = 0
        self._visible_food_id_view = _VisibleFoodIdView(self)
        self._scratch_sense_result: VisionSenseResult | None = None
        self.candidate_buffer_growth = 0
        self.visible_index_growth = 0
        self.blocked_interval_growth = 0
        self.visible_food_id_growth = 0
        self.sense_result_growth = 0
        self.stable_sort_count = 0
        self.failure_injector = None

    def clear_scratch(self) -> None:
        """Execute clear scratch behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep clear scratch behavior explicit in its owning subsystem.
        self._candidate_count = 0
        self._visible_count = 0
        self._blocked_count = 0
        self._visible_food_id_count = 0

    def sense(
        self,
        creature: Creature,
        foods: list[Food],
        creatures: list[Creature],
        world_bounds: tuple[float, float, float, float],
        max_speed: float,
        reproductive_readiness: float = 0.0,
        clock_tik_tok: float = 0.0,
        clock_chronometer: float = 0.0,
        clock_time_alive: float = 0.0,
        is_grabbing: bool = False,
        ignored_food_ids: set[int] | None = None,
        own_infants: list[Creature] | None = None,
    ) -> SensorSnapshot:
        """Execute sense behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
foods
    Input used by this creature-domain operation.
creatures
    Input used by this creature-domain operation.
world_bounds
    Input used by this creature-domain operation.
max_speed
    Input used by this creature-domain operation.
reproductive_readiness
    Input used by this creature-domain operation.
clock_tik_tok
    Input used by this creature-domain operation.
clock_chronometer
    Input used by this creature-domain operation.
clock_time_alive
    Input used by this creature-domain operation.
is_grabbing
    Input used by this creature-domain operation.
ignored_food_ids
    Input used by this creature-domain operation.
own_infants
    Input used by this creature-domain operation.
Returns
-------
SensorSnapshot
    Result produced by this creature-domain operation."""
        # Keep sense behavior explicit in its owning subsystem.
        return self.sense_with_visible_food_ids(
            creature,
            foods,
            creatures,
            world_bounds,
            max_speed,
            reproductive_readiness=reproductive_readiness,
            clock_tik_tok=clock_tik_tok,
            clock_chronometer=clock_chronometer,
            clock_time_alive=clock_time_alive,
            is_grabbing=is_grabbing,
            ignored_food_ids=ignored_food_ids,
            own_infants=own_infants,
        ).snapshot

    def sense_with_visible_food_ids(
        self,
        creature: Creature,
        foods: list[Food],
        creatures: list[Creature],
        world_bounds: tuple[float, float, float, float],
        max_speed: float,
        reproductive_readiness: float = 0.0,
        clock_tik_tok: float = 0.0,
        clock_chronometer: float = 0.0,
        clock_time_alive: float = 0.0,
        is_grabbing: bool = False,
        ignored_food_ids: set[int] | None = None,
        own_infants: list[Creature] | None = None,
        reuse_scratch: bool = False,
    ) -> VisionSenseResult:
        """Execute sense with visible food ids behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
foods
    Input used by this creature-domain operation.
creatures
    Input used by this creature-domain operation.
world_bounds
    Input used by this creature-domain operation.
max_speed
    Input used by this creature-domain operation.
reproductive_readiness
    Input used by this creature-domain operation.
clock_tik_tok
    Input used by this creature-domain operation.
clock_chronometer
    Input used by this creature-domain operation.
clock_time_alive
    Input used by this creature-domain operation.
is_grabbing
    Input used by this creature-domain operation.
ignored_food_ids
    Input used by this creature-domain operation.
own_infants
    Input used by this creature-domain operation.
reuse_scratch
    Input used by this creature-domain operation.
Returns
-------
VisionSenseResult
    Result produced by this creature-domain operation."""
        # Keep sense with visible food ids behavior explicit in its owning subsystem.
        visible_targets = (
            self._visible_targets_reused
            if reuse_scratch
            else self._visible_targets
        )(
            creature,
            foods,
            creatures,
            ignored_food_ids=ignored_food_ids,
            own_infants=own_infants,
        )
        snapshot = self._sensor_snapshot_from_visible_targets(
            creature,
            visible_targets,
            creatures,
            world_bounds,
            max_speed,
            reproductive_readiness=reproductive_readiness,
            clock_tik_tok=clock_tik_tok,
            clock_chronometer=clock_chronometer,
            clock_time_alive=clock_time_alive,
            is_grabbing=is_grabbing,
        )
        if not reuse_scratch:
            visible_food_ids = [
                target.source.id
                for target in visible_targets
                if target.kind == "food" and isinstance(target.source, Food)
            ]
            return VisionSenseResult(
                snapshot=snapshot,
                visible_food_ids=visible_food_ids,
            )
        self._visible_food_id_count = 0
        for target in visible_targets:
            if target.kind != "food" or not isinstance(target.source, Food):
                continue
            if self._visible_food_id_count == len(self._visible_food_ids):
                self._visible_food_ids.extend(
                    [0] * max(8, len(self._visible_food_ids) or 8)
                )
                self.visible_food_id_growth += 1
            self._visible_food_ids[self._visible_food_id_count] = target.source.id
            self._visible_food_id_count += 1
        result = self._scratch_sense_result
        if result is None:
            result = VisionSenseResult(snapshot, self._visible_food_id_view)
            self._scratch_sense_result = result
            self.sense_result_growth += 1
        else:
            result.snapshot = snapshot
        return result

    def _sensor_snapshot_from_visible_targets(
        self,
        creature: Creature,
        visible_targets: Sequence[_VisionCandidate],
        nearby_creatures: list[Creature],
        world_bounds: tuple[float, float, float, float],
        max_speed: float,
        reproductive_readiness: float = 0.0,
        clock_tik_tok: float = 0.0,
        clock_chronometer: float = 0.0,
        clock_time_alive: float = 0.0,
        is_grabbing: bool = False,
    ) -> SensorSnapshot:
        """Execute sensor snapshot from visible targets behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
visible_targets
    Input used by this creature-domain operation.
nearby_creatures
    Input used by this creature-domain operation.
world_bounds
    Input used by this creature-domain operation.
max_speed
    Input used by this creature-domain operation.
reproductive_readiness
    Input used by this creature-domain operation.
clock_tik_tok
    Input used by this creature-domain operation.
clock_chronometer
    Input used by this creature-domain operation.
clock_time_alive
    Input used by this creature-domain operation.
is_grabbing
    Input used by this creature-domain operation.
Returns
-------
SensorSnapshot
    Result produced by this creature-domain operation."""
        # Keep sensor snapshot from visible targets behavior explicit in its owning subsystem.
        food_snapshot = self._snapshot_for_kind(creature, visible_targets, "food")
        creature_snapshot = self._snapshot_for_kind(
            creature,
            visible_targets,
            "creature",
        )
        wall_snapshot = self._sense_walls(creature, world_bounds)
        own_infant_snapshot = self._snapshot_for_kind(
            creature,
            visible_targets,
            "own_infant",
        )
        boundary_snapshot = self.sense_boundary(creature, world_bounds)
        flock_snapshot = self._flock_snapshot(
            creature,
            visible_targets,
            nearby_creatures,
            max_speed,
        )

        return SensorSnapshot(
            food=food_snapshot,
            creatures=creature_snapshot,
            own_infants=own_infant_snapshot,
            walls=wall_snapshot,
            boundary=boundary_snapshot,
            energy=self._clamp01(creature.energy),
            speed=self.normalized_speed(creature, max_speed),
            vision_range=self.normalized_range(creature),
            vision_angle=self.normalized_angle(creature),
            vision_energy_cost=self.normalized_energy_cost(creature),
            reproductive_readiness=reproductive_readiness,
            visible_food_count=food_snapshot.count,
            visible_creature_count=creature_snapshot.count,
            clock_tik_tok=clock_tik_tok,
            clock_chronometer=clock_chronometer,
            clock_time_alive=clock_time_alive,
            is_grabbing=1.0 if is_grabbing else 0.0,
            stomach_fullness=self.stomach_fullness(creature),
            life_normalized=(
                0.0
                if self.max_life <= 0.0
                else self._clamp01(
                    float(getattr(creature, "life", self.max_life))
                    / self.max_life
                )
            ),
            flock=flock_snapshot,
            sensor_contract=self.sensor_contract,
            flock_target_group_size=self.flocking_config.target_group_size,
        )

    def stomach_fullness(self, creature: Creature) -> float:
        """Execute stomach fullness behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep stomach fullness behavior explicit in its owning subsystem.
        traits = getattr(creature, "physical_traits", None)
        inherited_capacity = getattr(traits, "stomach_capacity", None)
        capacity = (
            max(0.0, float(inherited_capacity))
            if inherited_capacity is not None
            else max(
                0.0,
                creature.radius * self.stomach_capacity_per_radius,
            )
        )
        if capacity <= 0.0:
            return 0.0
        return self._clamp01(
            max(0.0, getattr(creature, "stomach_energy", 0.0)) / capacity
        )

    def _flock_snapshot(
        self,
        creature: Creature,
        visible_targets: list[_VisionCandidate],
        nearby_creatures: list[Creature] | None = None,
        max_speed: float = 1.0,
    ) -> FlockSensorSnapshot:
        """Execute flock snapshot behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
visible_targets
    Input used by this creature-domain operation.
nearby_creatures
    Input used by this creature-domain operation.
max_speed
    Input used by this creature-domain operation.
Returns
-------
FlockSensorSnapshot
    Result produced by this creature-domain operation."""
        # Keep flock snapshot behavior explicit in its owning subsystem.
        del visible_targets
        flocking_config = self.flocking_config
        perception_radius = max(0.0, flocking_config.perception_radius)
        perception_radius_squared = perception_radius * perception_radius
        personal_space = max(0.0, flocking_config.preferred_personal_space)
        personal_space_squared = personal_space * personal_space
        long_range_config = flocking_config.long_range
        long_range_enabled = (
            long_range_config.enabled and long_range_config.range > 0.0
        )
        long_range_squared = (
            long_range_config.range * long_range_config.range
            if long_range_enabled
            else 0.0
        )
        creature_x, creature_y = creature.position
        creature_heading = creature.heading

        separation_x = 0.0
        separation_y = 0.0
        personal_space_count = 0
        compatible_neighbor_count = 0
        effective_flockmate_count = 0.0
        weighted_dx = 0.0
        weighted_dy = 0.0
        weighted_velocity_x = 0.0
        weighted_velocity_y = 0.0
        weighted_proximity = 0.0
        weighted_neighbor_distance = 0.0
        moving_heading_weight = 0.0
        weighted_heading_error = 0.0
        long_range_x = 0.0
        long_range_y = 0.0
        long_range_total = 0.0

        if getattr(self, "_reused_flock_creature", None) is creature:
            (
                perception_radius,
                creature_heading,
                separation_x,
                separation_y,
                personal_space_count,
                compatible_neighbor_count,
                effective_flockmate_count,
                weighted_dx,
                weighted_dy,
                weighted_velocity_x,
                weighted_velocity_y,
                weighted_proximity,
                weighted_neighbor_distance,
                moving_heading_weight,
                weighted_heading_error,
                long_range_x,
                long_range_y,
                long_range_total,
            ) = self._reused_flock_totals
            nearby_creatures = None
            self._reused_flock_creature = None

        for neighbor in [] if nearby_creatures is None else nearby_creatures:
            if neighbor.creature_id == creature.creature_id:
                continue
            neighbor_x, neighbor_y = neighbor.position
            dx = neighbor_x - creature_x
            dy = neighbor_y - creature_y
            distance_squared = dx * dx + dy * dy
            within_boid_range = (
                perception_radius > 0.0
                and distance_squared <= perception_radius_squared
            )
            within_long_range = (
                long_range_enabled
                and 0.0 < distance_squared <= long_range_squared
            )
            if not within_boid_range and not within_long_range:
                continue

            compatibility = self._flock_compatibility(creature, neighbor)
            if compatibility <= 0.0:
                continue

            distance = self._linear_distance(distance_squared)
            if within_long_range and distance > 1e-12:
                long_range_weight = (
                    compatibility
                    * self._clamp01(
                        1.0 - distance / long_range_config.range
                    )
                    * long_range_config.strength
                )
                if long_range_weight > 0.0:
                    long_range_x += (dx / distance) * long_range_weight
                    long_range_y += (dy / distance) * long_range_weight
                    long_range_total += long_range_weight

            if not within_boid_range:
                continue

            compatible_neighbor_count += 1
            effective_flockmate_count += compatibility
            weighted_dx += dx * compatibility
            weighted_dy += dy * compatibility
            velocity_x = neighbor.body.velocity.x
            velocity_y = neighbor.body.velocity.y
            weighted_velocity_x += velocity_x * compatibility
            weighted_velocity_y += velocity_y * compatibility
            weighted_neighbor_distance += distance * compatibility
            weighted_proximity += compatibility * self._clamp01(
                1.0 - distance / perception_radius
            )

            velocity_squared = (
                velocity_x * velocity_x + velocity_y * velocity_y
            )
            if velocity_squared > 1e-24:
                moving_heading_weight += compatibility
                weighted_heading_error += (
                    abs(
                        self._signed_angle(
                            atan2(velocity_y, velocity_x)
                            - creature_heading
                        )
                    )
                    * compatibility
                )

            if (
                0.0 < distance_squared < personal_space_squared
                and distance > 1e-12
            ):
                personal_space_count += 1
                proximity = self._clamp01(
                    1.0 - distance / personal_space
                )
                separation_x -= (
                    dx / distance
                ) * proximity * compatibility
                separation_y -= (
                    dy / distance
                ) * proximity * compatibility

        crowd_separation_strength = self._clamp01(
            hypot(separation_x, separation_y)
        )
        crowd_separation_absolute_angle = (
            0.0
            if crowd_separation_strength <= 1e-12
            else atan2(separation_y, separation_x)
        )
        long_range = self._long_range_social_observation_from_totals(
            creature,
            long_range_x,
            long_range_y,
            long_range_total,
        )
        if effective_flockmate_count <= 0.0:
            return FlockSensorSnapshot(
                crowd_separation_absolute_angle=(
                    crowd_separation_absolute_angle
                ),
                crowd_separation_strength=crowd_separation_strength,
                visible_creature_count=0,
                compatible_visible_count=0,
                visible_personal_space_count=personal_space_count,
                long_range=long_range,
            )

        average_flockmate_proximity = (
            weighted_proximity / effective_flockmate_count
        )
        dx = weighted_dx / effective_flockmate_count
        dy = weighted_dy / effective_flockmate_count
        center_distance = hypot(dx, dy)
        center_proximity = (
            0.0
            if perception_radius <= 0.0
            else self._clamp01(1.0 - center_distance / perception_radius)
        )
        cohesion_absolute_angle = atan2(dy, dx)
        center_relative_angle = self._signed_angle(
            cohesion_absolute_angle - creature_heading
        )

        average_velocity_x = weighted_velocity_x / effective_flockmate_count
        average_velocity_y = weighted_velocity_y / effective_flockmate_count
        actual_average_velocity_x = average_velocity_x
        actual_average_velocity_y = average_velocity_y
        if (
            abs(average_velocity_x) <= 1e-12
            and abs(average_velocity_y) <= 1e-12
        ):
            average_velocity_x = creature.body.velocity.x
            average_velocity_y = creature.body.velocity.y

        alignment_absolute_angle = creature_heading
        if (
            abs(average_velocity_x) > 1e-12
            or abs(average_velocity_y) > 1e-12
        ):
            alignment_absolute_angle = atan2(
                average_velocity_y,
                average_velocity_x,
            )
        relative_heading = self._signed_angle(
            alignment_absolute_angle - creature_heading
        )
        forward_x = cos(creature_heading)
        forward_y = sin(creature_heading)
        right_x = forward_y
        right_y = -forward_x
        position_scale = max(1e-12, perception_radius)
        relative_velocity_x = (
            actual_average_velocity_x - creature.body.velocity.x
        )
        relative_velocity_y = (
            actual_average_velocity_y - creature.body.velocity.y
        )
        velocity_scale = max(1e-12, 2.0 * max_speed)
        mean_neighbor_distance = (
            weighted_neighbor_distance / effective_flockmate_count
        )
        mean_heading_error = (
            0.0
            if moving_heading_weight <= 1e-12
            else weighted_heading_error / moving_heading_weight
        )

        return FlockSensorSnapshot(
            center_proximity=center_proximity,
            center_angle=self._normalized_view_angle(
                center_relative_angle,
                2.0 * pi,
            ),
            average_relative_heading=self._clamp(relative_heading / pi, -1.0, 1.0),
            flockmate_count=effective_flockmate_count,
            crowd_separation_absolute_angle=(
                crowd_separation_absolute_angle
            ),
            cohesion_absolute_angle=cohesion_absolute_angle,
            crowd_separation_strength=crowd_separation_strength,
            average_flockmate_proximity=average_flockmate_proximity,
            average_flockmate_velocity=(
                average_velocity_x,
                average_velocity_y,
            ),
            actual_average_flockmate_velocity=(
                actual_average_velocity_x,
                actual_average_velocity_y,
            ),
            visible_creature_count=compatible_neighbor_count,
            compatible_visible_count=compatible_neighbor_count,
            visible_personal_space_count=personal_space_count,
            social_presence=self._clamp01(effective_flockmate_count),
            center_forward=self._clamp(
                (dx * forward_x + dy * forward_y) / position_scale,
                -1.0,
                1.0,
            ),
            center_right=self._clamp(
                (dx * right_x + dy * right_y) / position_scale,
                -1.0,
                1.0,
            ),
            relative_velocity_forward=self._clamp(
                (
                    relative_velocity_x * forward_x
                    + relative_velocity_y * forward_y
                )
                / velocity_scale,
                -1.0,
                1.0,
            ),
            relative_velocity_right=self._clamp(
                (
                    relative_velocity_x * right_x
                    + relative_velocity_y * right_y
                )
                / velocity_scale,
                -1.0,
                1.0,
            ),
            center_distance=center_distance,
            mean_neighbor_distance=mean_neighbor_distance,
            mean_heading_error=mean_heading_error,
            long_range=long_range,
        )

    def _long_range_social_observation_from_totals(
        self,
        creature: Creature,
        weighted_x: float,
        weighted_y: float,
        total: float,
    ) -> LongRangeSocialObservation:
        """Execute long range social observation from totals behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
weighted_x
    Input used by this creature-domain operation.
weighted_y
    Input used by this creature-domain operation.
total
    Input used by this creature-domain operation.
Returns
-------
LongRangeSocialObservation
    Result produced by this creature-domain operation."""
        # Keep long range social observation from totals behavior explicit in its owning subsystem.
        magnitude = hypot(weighted_x, weighted_y)
        if total <= 0.0 or magnitude <= 1e-12:
            return LongRangeSocialObservation(
                intensity=self._clamp01(total),
            )
        unit_x = weighted_x / magnitude
        unit_y = weighted_y / magnitude
        forward_x = cos(creature.heading)
        forward_y = sin(creature.heading)
        right_x = forward_y
        right_y = -forward_x
        return LongRangeSocialObservation(
            intensity=self._clamp01(total),
            direction_forward=self._clamp(
                unit_x * forward_x + unit_y * forward_y,
                -1.0,
                1.0,
            ),
            direction_right=self._clamp(
                unit_x * right_x + unit_y * right_y,
                -1.0,
                1.0,
            ),
        )

    @staticmethod
    def _linear_distance(distance_squared: float) -> float:
        """Execute linear distance behavior.

Parameters
----------
distance_squared
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep linear distance behavior explicit in its owning subsystem.
        return sqrt(distance_squared)

    def _flock_compatibility(
        self,
        creature: Creature,
        neighbor: Creature,
    ) -> float:
        """Execute flock compatibility behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
neighbor
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep flock compatibility behavior explicit in its owning subsystem.
        resolver = self.flock_compatibility_resolver
        if resolver is None:
            return (
                1.0
                if self._species_id(creature) == self._species_id(neighbor)
                else 0.0
            )
        return self._clamp01(float(resolver(creature, neighbor)))

    def _normalized_view_angle(self, angle: float, field_of_view: float) -> float:
        """Execute normalized view angle behavior.

Parameters
----------
angle
    Input used by this creature-domain operation.
field_of_view
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalized view angle behavior explicit in its owning subsystem.
        if field_of_view <= 0.0:
            return 0.0
        return self._clamp(angle / (field_of_view / 2.0), -1.0, 1.0)

    def _species_id(self, creature: Creature) -> int | None:
        """Execute species id behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
int | None
    Result produced by this creature-domain operation."""
        # Keep species id behavior explicit in its owning subsystem.
        lineage = getattr(creature, "lineage", None)
        return getattr(
            lineage,
            "species_id",
            getattr(creature, "species_id", None),
        )

    def sense_boundary(
        self,
        creature: Creature,
        world_bounds: tuple[float, float, float, float],
    ) -> BoundarySnapshot:
        """Execute sense boundary behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
world_bounds
    Input used by this creature-domain operation.
Returns
-------
BoundarySnapshot
    Result produced by this creature-domain operation."""
        # Keep sense boundary behavior explicit in its owning subsystem.
        left, bottom, right, top = world_bounds
        x, y = creature.position
        radius = creature.radius
        warning_distance = self.config.boundary_warning_distance
        if warning_distance <= 0:
            return BoundarySnapshot(pressure=0.0, turn=0.0)

        boundary_vectors = [
            (x - left - radius, (1.0, 0.0)),
            (right - x - radius, (-1.0, 0.0)),
            (y - bottom - radius, (0.0, 1.0)),
            (top - y - radius, (0.0, -1.0)),
        ]
        pressure = 0.0
        inward_x = 0.0
        inward_y = 0.0
        for distance, inward_vector in boundary_vectors:
            wall_pressure = self._clamp01(1.0 - max(0.0, distance) / warning_distance)
            pressure = max(pressure, wall_pressure)
            inward_x += inward_vector[0] * wall_pressure
            inward_y += inward_vector[1] * wall_pressure

        if pressure <= 0.0:
            return BoundarySnapshot(pressure=0.0, turn=0.0)

        inward_angle = atan2(inward_y, inward_x)
        signed_angle = self._signed_angle(inward_angle - creature.heading)
        return BoundarySnapshot(
            pressure=pressure,
            turn=self._clamp(signed_angle / pi, -1.0, 1.0),
        )

    def visible_foods(
        self,
        creature: Creature,
        foods: list[Food],
        creatures: list[Creature] | None = None,
        ignored_food_ids: set[int] | None = None,
    ) -> list[Food]:
        """Execute visible foods behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
foods
    Input used by this creature-domain operation.
creatures
    Input used by this creature-domain operation.
ignored_food_ids
    Input used by this creature-domain operation.
Returns
-------
list[Food]
    Result produced by this creature-domain operation."""
        # Keep visible foods behavior explicit in its owning subsystem.
        blockers = [] if creatures is None else creatures
        return [
            target.source
            for target in self._visible_targets(
                creature,
                foods,
                blockers,
                ignored_food_ids=ignored_food_ids,
            )
            if target.kind == "food"
        ]

    def visible_creatures(
        self,
        creature: Creature,
        creatures: list[Creature],
        foods: list[Food] | None = None,
        ignored_food_ids: set[int] | None = None,
    ) -> list[Creature]:
        """Execute visible creatures behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
creatures
    Input used by this creature-domain operation.
foods
    Input used by this creature-domain operation.
ignored_food_ids
    Input used by this creature-domain operation.
Returns
-------
list[Creature]
    Result produced by this creature-domain operation."""
        # Keep visible creatures behavior explicit in its owning subsystem.
        blockers = [] if foods is None else foods
        return [
            target.source
            for target in self._visible_targets(
                creature,
                blockers,
                creatures,
                ignored_food_ids=ignored_food_ids,
            )
            if target.kind == "creature"
        ]

    def _visible_targets(
        self,
        creature: Creature,
        foods: list[Food],
        creatures: list[Creature],
        ignored_food_ids: set[int] | None = None,
        own_infants: list[Creature] | None = None,
    ) -> list[_VisionCandidate]:
        """Execute visible targets behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
foods
    Input used by this creature-domain operation.
creatures
    Input used by this creature-domain operation.
ignored_food_ids
    Input used by this creature-domain operation.
own_infants
    Input used by this creature-domain operation.
Returns
-------
list[_VisionCandidate]
    Result produced by this creature-domain operation."""
        # Keep visible targets behavior explicit in its owning subsystem.
        if creature.vision.range <= 0 or creature.vision.angle <= 0:
            return []

        candidates: list[_VisionCandidate] = []
        ignored_food_ids = set() if ignored_food_ids is None else ignored_food_ids

        for food in foods:
            if food.id in ignored_food_ids:
                continue
            if self._food_touches_mouth(creature, food.position, food.radius):
                candidates.append(
                    self._mouth_contact_candidate(creature, "food", food)
                )
                continue
            candidate = self._vision_candidate(
                creature,
                "food",
                food,
                food.position,
                food.radius,
            )
            if candidate is not None:
                candidates.append(candidate)

        for other in creatures:
            if other.creature_id == creature.creature_id:
                continue
            candidate = self._vision_candidate(
                creature,
                "creature",
                other,
                other.position,
                other.radius,
            )
            if candidate is not None:
                candidates.append(candidate)

        for infant in [] if own_infants is None else own_infants:
            if infant.creature_id == creature.creature_id:
                continue
            candidate = self._vision_candidate(
                creature,
                "own_infant",
                infant,
                infant.position,
                infant.radius,
            )
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda candidate: candidate.surface_distance)
        return self._remove_targets_occluded_by_creatures(candidates)

    def _scratch_candidate(self) -> _VisionCandidate:
        """Execute scratch candidate behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
_VisionCandidate
    Result produced by this creature-domain operation."""
        # Keep scratch candidate behavior explicit in its owning subsystem.
        index = self._candidate_count
        if index == len(self._candidate_storage):
            self._candidate_storage.append(
                _VisionCandidate("", None, 0.0, 0.0, 0.0, (0.0, 0.0))
            )
            self.candidate_buffer_growth += 1
        self._candidate_count += 1
        return self._candidate_storage[index]

    def _append_visible_index(self, index: int) -> None:
        """Execute append visible index behavior.

Parameters
----------
index
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep append visible index behavior explicit in its owning subsystem.
        if self._visible_count == len(self._visible_indices):
            self._visible_indices.extend(
                [0] * max(16, len(self._visible_indices) or 16)
            )
            self.visible_index_growth += 1
        self._visible_indices[self._visible_count] = index
        self._visible_count += 1

    def _fill_candidate(
        self,
        creature: Creature,
        kind: str,
        source: Food | Creature,
        target_position: tuple[float, float],
        target_radius: float,
        *,
        mouth_contact: bool = False,
    ) -> bool:
        """Execute fill candidate behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
kind
    Input used by this creature-domain operation.
source
    Input used by this creature-domain operation.
target_position
    Input used by this creature-domain operation.
target_radius
    Input used by this creature-domain operation.
mouth_contact
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep fill candidate behavior explicit in its owning subsystem.
        origin_x, origin_y = self._vision_origin(creature)
        target_x, target_y = target_position
        return self._fill_candidate_values(
            kind,
            source,
            float(target_x),
            float(target_y),
            target_radius,
            origin_x,
            origin_y,
            creature.vision.range,
            creature.vision.angle / 2.0,
            creature.heading,
            mouth_contact=mouth_contact,
        )

    def _fill_candidate_values(
        self,
        kind: str,
        source: Food | Creature,
        target_x: float,
        target_y: float,
        target_radius: float,
        origin_x: float,
        origin_y: float,
        vision_range: float,
        half_cone: float,
        creature_heading: float,
        *,
        mouth_contact: bool = False,
    ) -> bool:
        """Execute fill candidate values behavior.

Parameters
----------
kind
    Input used by this creature-domain operation.
source
    Input used by this creature-domain operation.
target_x
    Input used by this creature-domain operation.
target_y
    Input used by this creature-domain operation.
target_radius
    Input used by this creature-domain operation.
origin_x
    Input used by this creature-domain operation.
origin_y
    Input used by this creature-domain operation.
vision_range
    Input used by this creature-domain operation.
half_cone
    Input used by this creature-domain operation.
creature_heading
    Input used by this creature-domain operation.
mouth_contact
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep fill candidate values behavior explicit in its owning subsystem.
        injector = self.failure_injector
        if callable(injector):
            injector("vision.filtering")
        if mouth_contact:
            candidate = self._scratch_candidate()
            candidate.kind = kind
            candidate.source = source
            candidate.surface_distance = 0.0
            candidate.signed_angle = 0.0
            candidate.closeness = 1.0
            candidate.interval = (-half_cone, half_cone)
            return True
        dx = target_x - origin_x
        dy = target_y - origin_y
        distance = hypot(dx, dy)
        surface_distance = max(0.0, distance - target_radius)
        if surface_distance > vision_range:
            return False
        signed_angle = self._signed_angle(
            atan2(dy, dx) - creature_heading
        )
        angular_radius = pi if distance <= 0 else atan2(target_radius, distance)
        interval_start = max(-half_cone, signed_angle - angular_radius)
        interval_end = min(half_cone, signed_angle + angular_radius)
        if interval_start > interval_end:
            return False
        candidate = self._scratch_candidate()
        candidate.kind = kind
        candidate.source = source
        candidate.surface_distance = surface_distance
        candidate.signed_angle = signed_angle
        candidate.closeness = 1.0 - surface_distance / vision_range
        candidate.interval = (interval_start, interval_end)
        return True

    @staticmethod
    def _candidate_stable_key(candidate: _VisionCandidate):
        """Execute candidate stable key behavior.

Parameters
----------
candidate
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep candidate stable key behavior explicit in its owning subsystem.
        source = candidate.source
        source_id = getattr(source, "id", getattr(source, "creature_id", -1))
        return candidate.surface_distance, candidate.kind, source_id

    def _sort_candidate_prefix(self) -> None:
        """Execute sort candidate prefix behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep sort candidate prefix behavior explicit in its owning subsystem.
        for cursor in range(1, self._candidate_count):
            candidate = self._candidate_storage[cursor]
            key = self._candidate_stable_key(candidate)
            insertion = cursor
            while (
                insertion > 0
                and self._candidate_stable_key(
                    self._candidate_storage[insertion - 1]
                ) > key
            ):
                self._candidate_storage[insertion] = (
                    self._candidate_storage[insertion - 1]
                )
                insertion -= 1
            self._candidate_storage[insertion] = candidate
        self.stable_sort_count += 1

    def _interval_blocked_scratch(self, interval: tuple[float, float]) -> bool:
        """Execute interval blocked scratch behavior.

Parameters
----------
interval
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep interval blocked scratch behavior explicit in its owning subsystem.
        epsilon = 1e-9
        start_value, end_value = interval
        if end_value - start_value <= epsilon:
            for index in range(self._blocked_count):
                if (
                    self._blocked_starts[index] - epsilon
                    <= start_value
                    <= self._blocked_ends[index] + epsilon
                ):
                    return True
            return False
        cursor = start_value
        for index in range(self._blocked_count):
            start = self._blocked_starts[index]
            end = self._blocked_ends[index]
            if end <= cursor + epsilon:
                continue
            if start > cursor + epsilon:
                return False
            cursor = max(cursor, end)
            if cursor >= end_value - epsilon:
                return True
        return cursor >= end_value - epsilon

    def _add_blocked_scratch(self, interval: tuple[float, float]) -> None:
        """Execute add blocked scratch behavior.

Parameters
----------
interval
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep add blocked scratch behavior explicit in its owning subsystem.
        start, end = interval
        insertion = 0
        while (
            insertion < self._blocked_count
            and self._blocked_ends[insertion] < start
        ):
            insertion += 1
        merge_end = insertion
        while (
            merge_end < self._blocked_count
            and self._blocked_starts[merge_end] <= end
        ):
            start = min(start, self._blocked_starts[merge_end])
            end = max(end, self._blocked_ends[merge_end])
            merge_end += 1
        removed = merge_end - insertion
        if removed == 0:
            if self._blocked_count == len(self._blocked_starts):
                growth = max(8, len(self._blocked_starts) or 8)
                self._blocked_starts.extend([0.0] * growth)
                self._blocked_ends.extend([0.0] * growth)
                self.blocked_interval_growth += 1
            for index in range(self._blocked_count, insertion, -1):
                self._blocked_starts[index] = self._blocked_starts[index - 1]
                self._blocked_ends[index] = self._blocked_ends[index - 1]
            self._blocked_count += 1
        elif removed > 1:
            shift = removed - 1
            for index in range(merge_end, self._blocked_count):
                self._blocked_starts[index - shift] = self._blocked_starts[index]
                self._blocked_ends[index - shift] = self._blocked_ends[index]
            self._blocked_count -= shift
        self._blocked_starts[insertion] = start
        self._blocked_ends[insertion] = end

    def _visible_targets_reused(
        self,
        creature: Creature,
        foods: list[Food],
        creatures: Sequence[Creature],
        ignored_food_ids: set[int] | None = None,
        own_infants: Sequence[Creature] | None = None,
    ) -> Sequence[_VisionCandidate]:
        """Execute visible targets reused behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
foods
    Input used by this creature-domain operation.
creatures
    Input used by this creature-domain operation.
ignored_food_ids
    Input used by this creature-domain operation.
own_infants
    Input used by this creature-domain operation.
Returns
-------
Sequence[_VisionCandidate]
    Result produced by this creature-domain operation."""
        # Keep visible targets reused behavior explicit in its owning subsystem.
        self.clear_scratch()
        if creature.vision.range <= 0 or creature.vision.angle <= 0:
            return _CandidateView(
                self._candidate_storage,
                self._visible_indices,
                0,
            )
        ignored = () if ignored_food_ids is None else ignored_food_ids
        flocking_config = self.flocking_config
        perception_radius = max(0.0, flocking_config.perception_radius)
        perception_radius_squared = perception_radius * perception_radius
        personal_space = max(
            0.0, flocking_config.preferred_personal_space
        )
        personal_space_squared = personal_space * personal_space
        long_range_config = flocking_config.long_range
        long_range_enabled = (
            long_range_config.enabled and long_range_config.range > 0.0
        )
        long_range_squared = (
            long_range_config.range * long_range_config.range
            if long_range_enabled
            else 0.0
        )
        creature_x, creature_y = creature.position
        creature_heading = creature.heading
        observer_radius = creature.radius
        origin_x = creature_x + cos(creature_heading) * observer_radius * 0.35
        origin_y = creature_y + sin(creature_heading) * observer_radius * 0.35
        vision_range = creature.vision.range
        half_cone = creature.vision.angle / 2.0
        separation_x = separation_y = 0.0
        personal_space_count = compatible_neighbor_count = 0
        effective_flockmate_count = 0.0
        weighted_dx = weighted_dy = 0.0
        weighted_velocity_x = weighted_velocity_y = 0.0
        weighted_proximity = weighted_neighbor_distance = 0.0
        moving_heading_weight = weighted_heading_error = 0.0
        long_range_x = long_range_y = long_range_total = 0.0
        for food in foods:
            if food.id in ignored:
                continue
            food_position = food.position
            food_x, food_y = food_position
            contact = self._food_touches_mouth(
                creature, food_position, food.radius
            )
            self._fill_candidate_values(
                "food",
                food,
                food_x,
                food_y,
                food.radius,
                origin_x,
                origin_y,
                vision_range,
                half_cone,
                creature_heading,
                mouth_contact=contact,
            )
        spatial_index = getattr(creatures, "index", None)
        spatial_slots = getattr(creatures, "slots", None)
        spatial_count = getattr(creatures, "count", 0)
        use_spatial_values = spatial_index is not None and spatial_slots is not None
        neighbor_count = spatial_count if use_spatial_values else len(creatures)
        for neighbor_position in range(neighbor_count):
            if use_spatial_values:
                slot = spatial_slots[neighbor_position]
                other = spatial_index.creature_for_slot(slot)
                neighbor_x = spatial_index.centres_x[slot]
                neighbor_y = spatial_index.centres_y[slot]
                neighbor_radius = spatial_index.radii[slot]
            else:
                other = creatures[neighbor_position]
                neighbor_x, neighbor_y = other.position
                neighbor_radius = other.radius
            if other is None:
                continue
            if other.creature_id != creature.creature_id:
                injector = self.failure_injector
                if callable(injector):
                    injector("flocking.accumulation")
                dx = neighbor_x - creature_x
                dy = neighbor_y - creature_y
                distance_squared = dx * dx + dy * dy
                within_boid_range = (
                    perception_radius > 0.0
                    and distance_squared <= perception_radius_squared
                )
                within_long_range = (
                    long_range_enabled
                    and 0.0 < distance_squared <= long_range_squared
                )
                if within_boid_range or within_long_range:
                    compatibility = self._flock_compatibility(
                        creature, other
                    )
                    if compatibility > 0.0:
                        distance = self._linear_distance(distance_squared)
                        if within_long_range and distance > 1e-12:
                            long_range_weight = (
                                compatibility
                                * self._clamp01(
                                    1.0
                                    - distance / long_range_config.range
                                )
                                * long_range_config.strength
                            )
                            if long_range_weight > 0.0:
                                long_range_x += (
                                    dx / distance
                                ) * long_range_weight
                                long_range_y += (
                                    dy / distance
                                ) * long_range_weight
                                long_range_total += long_range_weight
                        if within_boid_range:
                            compatible_neighbor_count += 1
                            effective_flockmate_count += compatibility
                            weighted_dx += dx * compatibility
                            weighted_dy += dy * compatibility
                            velocity_x = other.body.velocity.x
                            velocity_y = other.body.velocity.y
                            weighted_velocity_x += (
                                velocity_x * compatibility
                            )
                            weighted_velocity_y += (
                                velocity_y * compatibility
                            )
                            weighted_neighbor_distance += (
                                distance * compatibility
                            )
                            weighted_proximity += (
                                compatibility
                                * self._clamp01(
                                    1.0
                                    - distance / perception_radius
                                )
                            )
                            velocity_squared = (
                                velocity_x * velocity_x
                                + velocity_y * velocity_y
                            )
                            if velocity_squared > 1e-24:
                                moving_heading_weight += compatibility
                                weighted_heading_error += (
                                    abs(
                                        self._signed_angle(
                                            atan2(
                                                velocity_y,
                                                velocity_x,
                                            )
                                            - creature_heading
                                        )
                                    )
                                    * compatibility
                                )
                            if (
                                0.0
                                < distance_squared
                                < personal_space_squared
                                and distance > 1e-12
                            ):
                                personal_space_count += 1
                                proximity = self._clamp01(
                                    1.0 - distance / personal_space
                                )
                                separation_x -= (
                                    dx / distance
                                ) * proximity * compatibility
                                separation_y -= (
                                    dy / distance
                                ) * proximity * compatibility
                self._fill_candidate_values(
                    "creature",
                    other,
                    neighbor_x,
                    neighbor_y,
                    neighbor_radius,
                    origin_x,
                    origin_y,
                    vision_range,
                    half_cone,
                    creature_heading,
                )
        self._reused_flock_creature = creature
        self._reused_flock_totals = (
            perception_radius,
            creature_heading,
            separation_x,
            separation_y,
            personal_space_count,
            compatible_neighbor_count,
            effective_flockmate_count,
            weighted_dx,
            weighted_dy,
            weighted_velocity_x,
            weighted_velocity_y,
            weighted_proximity,
            weighted_neighbor_distance,
            moving_heading_weight,
            weighted_heading_error,
            long_range_x,
            long_range_y,
            long_range_total,
        )
        for infant in () if own_infants is None else own_infants:
            if infant.creature_id != creature.creature_id:
                if spatial_index is not None:
                    values = spatial_index.values_for(infant)
                else:
                    values = None
                if values is None:
                    infant_x, infant_y = infant.position
                    infant_radius = infant.radius
                else:
                    infant_x, infant_y, infant_radius = values
                self._fill_candidate_values(
                    "own_infant",
                    infant,
                    infant_x,
                    infant_y,
                    infant_radius,
                    origin_x,
                    origin_y,
                    vision_range,
                    half_cone,
                    creature_heading,
                )
        self._sort_candidate_prefix()
        candidate_index = 0
        epsilon = 1e-9
        injector = self.failure_injector
        if callable(injector):
            injector("vision.occlusion")
        while candidate_index < self._candidate_count:
            group_end = candidate_index + 1
            distance = self._candidate_storage[candidate_index].surface_distance
            while (
                group_end < self._candidate_count
                and self._candidate_storage[group_end].surface_distance
                <= distance + epsilon
            ):
                group_end += 1
            group_visible_start = self._visible_count
            for index in range(candidate_index, group_end):
                if not self._interval_blocked_scratch(
                    self._candidate_storage[index].interval
                ):
                    self._append_visible_index(index)
            for position in range(group_visible_start, self._visible_count):
                visible = self._candidate_storage[
                    self._visible_indices[position]
                ]
                if visible.kind in {"creature", "own_infant"}:
                    self._add_blocked_scratch(visible.interval)
            candidate_index = group_end
        return _CandidateView(
            self._candidate_storage,
            self._visible_indices,
            self._visible_count,
        )

    def _remove_targets_occluded_by_creatures(
        self,
        candidates: list[_VisionCandidate],
    ) -> list[_VisionCandidate]:
        """Execute remove targets occluded by creatures behavior.

Parameters
----------
candidates
    Input used by this creature-domain operation.
Returns
-------
list[_VisionCandidate]
    Result produced by this creature-domain operation."""
        # Keep remove targets occluded by creatures behavior explicit in its owning subsystem.
        blocked_intervals: list[tuple[float, float]] = []
        visible_targets: list[_VisionCandidate] = []
        candidate_index = 0
        distance_epsilon = 1e-9

        while candidate_index < len(candidates):
            group_end = candidate_index + 1
            group_distance = candidates[candidate_index].surface_distance
            while (
                group_end < len(candidates)
                and candidates[group_end].surface_distance
                <= group_distance + distance_epsilon
            ):
                group_end += 1

            visible_group = [
                candidate
                for candidate in candidates[candidate_index:group_end]
                if not self._interval_is_blocked(
                    candidate.interval,
                    blocked_intervals,
                )
            ]
            visible_targets.extend(visible_group)

            for candidate in visible_group:
                if candidate.kind in {"creature", "own_infant"}:
                    self._add_blocked_interval(
                        candidate.interval,
                        blocked_intervals,
                    )

            candidate_index = group_end

        return visible_targets

    def _interval_is_blocked(
        self,
        interval: tuple[float, float],
        blocked_intervals: list[tuple[float, float]],
    ) -> bool:
        """Execute interval is blocked behavior.

Parameters
----------
interval
    Input used by this creature-domain operation.
blocked_intervals
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep interval is blocked behavior explicit in its owning subsystem.
        epsilon = 1e-9
        if interval[1] - interval[0] <= epsilon:
            return any(
                start - epsilon <= interval[0] <= end + epsilon
                for start, end in blocked_intervals
            )

        cursor = interval[0]
        for start, end in blocked_intervals:
            if end <= cursor + epsilon:
                continue
            if start > cursor + epsilon:
                return False
            cursor = max(cursor, end)
            if cursor >= interval[1] - epsilon:
                return True

        return cursor >= interval[1] - epsilon

    def _add_blocked_interval(
        self,
        interval: tuple[float, float],
        blocked_intervals: list[tuple[float, float]],
    ) -> None:
        """Execute add blocked interval behavior.

Parameters
----------
interval
    Input used by this creature-domain operation.
blocked_intervals
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep add blocked interval behavior explicit in its owning subsystem.
        start, end = interval
        merged: list[tuple[float, float]] = []
        inserted = False

        for current_start, current_end in blocked_intervals:
            if current_end < start:
                merged.append((current_start, current_end))
                continue
            if end < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
                continue

            start = min(start, current_start)
            end = max(end, current_end)

        if not inserted:
            merged.append((start, end))

        blocked_intervals[:] = merged

    def _vision_candidate(
        self,
        creature: Creature,
        kind: str,
        source: Food | Creature,
        target_position: tuple[float, float],
        target_radius: float,
    ) -> _VisionCandidate | None:
        """Execute vision candidate behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
kind
    Input used by this creature-domain operation.
source
    Input used by this creature-domain operation.
target_position
    Input used by this creature-domain operation.
target_radius
    Input used by this creature-domain operation.
Returns
-------
_VisionCandidate | None
    Result produced by this creature-domain operation."""
        # Keep vision candidate behavior explicit in its owning subsystem.
        origin_x, origin_y = self._vision_origin(creature)
        target_x, target_y = target_position
        vision_range = creature.vision.range
        cone_angle = creature.vision.angle
        if vision_range <= 0 or cone_angle <= 0:
            return None

        dx = target_x - origin_x
        dy = target_y - origin_y
        distance = hypot(dx, dy)
        surface_distance = max(0.0, distance - target_radius)
        if surface_distance > vision_range:
            return None

        angle_to_target = atan2(dy, dx)
        signed_angle = self._signed_angle(angle_to_target - creature.heading)
        angular_radius = pi if distance <= 0 else atan2(target_radius, distance)
        half_cone = cone_angle / 2
        interval = (
            max(-half_cone, signed_angle - angular_radius),
            min(half_cone, signed_angle + angular_radius),
        )
        if interval[0] > interval[1]:
            return None

        closeness = 1.0 - (surface_distance / vision_range)
        return _VisionCandidate(
            kind=kind,
            source=source,
            surface_distance=surface_distance,
            signed_angle=signed_angle,
            closeness=closeness,
            interval=interval,
        )

    def _mouth_contact_candidate(
        self,
        creature: Creature,
        kind: str,
        source: Food,
    ) -> _VisionCandidate:
        """Execute mouth contact candidate behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
kind
    Input used by this creature-domain operation.
source
    Input used by this creature-domain operation.
Returns
-------
_VisionCandidate
    Result produced by this creature-domain operation."""
        # Keep mouth contact candidate behavior explicit in its owning subsystem.
        half_cone = creature.vision.angle / 2.0
        return _VisionCandidate(
            kind=kind,
            source=source,
            surface_distance=0.0,
            signed_angle=0.0,
            closeness=1.0,
            interval=(-half_cone, half_cone),
        )

    def _vision_origin(self, creature: Creature) -> tuple[float, float]:
        """Execute vision origin behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
tuple[float, float]
    Result produced by this creature-domain operation."""
        # Keep vision origin behavior explicit in its owning subsystem.
        creature_x, creature_y = creature.position
        return (
            creature_x + cos(creature.heading) * creature.radius * 0.35,
            creature_y + sin(creature.heading) * creature.radius * 0.35,
        )

    def _food_touches_mouth(
        self,
        creature: Creature,
        food_position: tuple[float, float],
        food_radius: float,
    ) -> bool:
        """Execute food touches mouth behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
food_position
    Input used by this creature-domain operation.
food_radius
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep food touches mouth behavior explicit in its owning subsystem.
        creature_x, creature_y = creature.position
        food_x, food_y = food_position
        dx = food_x - creature_x
        dy = food_y - creature_y

        contact_slop = max(1.0, min(3.0, self.eating_distance * 0.25))
        contact_range = creature.radius + food_radius + contact_slop
        if dx * dx + dy * dy > contact_range * contact_range:
            return False

        forward_x = cos(creature.heading)
        forward_y = sin(creature.heading)
        forward_distance = dx * forward_x + dy * forward_y
        if forward_distance < creature.radius - contact_slop:
            return False

        lateral_x = -forward_y
        lateral_y = forward_x
        lateral_distance = abs(dx * lateral_x + dy * lateral_y)
        mouth_half_width = max(2.0, creature.radius * 0.35)
        return lateral_distance <= food_radius + mouth_half_width

    def _snapshot_for_kind(
        self,
        creature: Creature,
        targets: Sequence[_VisionCandidate],
        kind: str,
    ) -> VisionTargetSnapshot:
        """Execute snapshot for kind behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
targets
    Input used by this creature-domain operation.
kind
    Input used by this creature-domain operation.
Returns
-------
VisionTargetSnapshot
    Result produced by this creature-domain operation."""
        # Keep snapshot for kind behavior explicit in its owning subsystem.
        nearest = None
        density = 0.0
        count = 0
        for target in targets:
            if target.kind != kind:
                continue
            count += 1
            density += target.closeness
            if (
                nearest is None
                or target.surface_distance < nearest.surface_distance
            ):
                nearest = target
        if nearest is None:
            return self._empty_target_snapshot()
        source = nearest.source
        nearest_id = getattr(
            source,
            "id",
            getattr(source, "creature_id", None),
        )

        return VisionTargetSnapshot(
            visible=1.0,
            proximity=nearest.closeness,
            angle=self._clamp(
                nearest.signed_angle / (creature.vision.angle / 2.0),
                -1.0,
                1.0,
            ),
            density=self._clamp01(density),
            count=count,
            nearest_id=nearest_id,
            surface_distance=nearest.surface_distance,
            relative_angle=nearest.signed_angle,
        )

    def _sense_walls(
        self,
        creature: Creature,
        world_bounds: tuple[float, float, float, float],
    ) -> VisionTargetSnapshot:
        """Execute sense walls behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
world_bounds
    Input used by this creature-domain operation.
Returns
-------
VisionTargetSnapshot
    Result produced by this creature-domain operation."""
        # Keep sense walls behavior explicit in its owning subsystem.
        creature_x, creature_y = creature.position
        vision_range = creature.vision.range
        cone_angle = creature.vision.angle
        if vision_range <= 0 or cone_angle <= 0:
            return self._empty_target_snapshot()

        left, bottom, right, top = world_bounds
        wall_segments = [
            ((left, bottom), (right, bottom)),
            ((right, bottom), (right, top)),
            ((right, top), (left, top)),
            ((left, top), (left, bottom)),
        ]

        visible_count = 0
        density = 0.0
        wall_candidates: list[_VisionCandidate] = []
        half_cone = cone_angle / 2.0

        for start, end in wall_segments:
            for point in self._wall_candidate_points(creature, start, end):
                dx = point[0] - creature_x
                dy = point[1] - creature_y
                distance = hypot(dx, dy)
                surface_distance = max(0.0, distance - creature.radius)
                if surface_distance > vision_range:
                    continue

                angle_to_wall = atan2(dy, dx)
                signed_angle = self._signed_angle(angle_to_wall - creature.heading)
                if abs(signed_angle) > cone_angle / 2:
                    continue

                closeness = 1.0 - (surface_distance / vision_range)
                proximity = self._clamp01(closeness)
                fuzziness = 0.05 + (0.1 * proximity)
                interval = (
                    max(-half_cone, signed_angle - fuzziness),
                    min(half_cone, signed_angle + fuzziness),
                )
                if interval[0] > interval[1]:
                    continue

                visible_count += 1
                density += closeness
                wall_candidates.append(
                    _VisionCandidate(
                        kind="wall",
                        source=None,
                        surface_distance=surface_distance,
                        signed_angle=signed_angle,
                        closeness=proximity,
                        interval=interval,
                    )
                )

        if visible_count == 0:
            return self._empty_target_snapshot()

        proximity, angle = self._nearest_proximity_and_angle(
            wall_candidates, cone_angle
        )
        return VisionTargetSnapshot(
            visible=1.0,
            proximity=proximity,
            angle=angle,
            density=self._clamp01(density),
            count=visible_count,
        )

    def _wall_candidate_points(
        self,
        creature: Creature,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Execute wall candidate points behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
start
    Input used by this creature-domain operation.
end
    Input used by this creature-domain operation.
Returns
-------
list[tuple[float, float]]
    Result produced by this creature-domain operation."""
        # Keep wall candidate points behavior explicit in its owning subsystem.
        left_ray_angle = creature.heading - creature.vision.angle / 2
        right_ray_angle = creature.heading + creature.vision.angle / 2
        candidates = [
            start,
            end,
            self._closest_point_on_segment(creature.position, start, end),
        ]

        for ray_angle in (left_ray_angle, right_ray_angle):
            intersection = self._ray_segment_intersection(
                creature.position,
                (cos(ray_angle), sin(ray_angle)),
                start,
                end,
            )
            if intersection is not None:
                candidates.append(intersection)

        return candidates

    def _closest_point_on_segment(
        self,
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        """Execute closest point on segment behavior.

Parameters
----------
point
    Input used by this creature-domain operation.
start
    Input used by this creature-domain operation.
end
    Input used by this creature-domain operation.
Returns
-------
tuple[float, float]
    Result produced by this creature-domain operation."""
        # Keep closest point on segment behavior explicit in its owning subsystem.
        point_x, point_y = point
        start_x, start_y = start
        end_x, end_y = end
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        length_squared = segment_x**2 + segment_y**2
        if length_squared <= 0.0:
            return start

        t = (
            (point_x - start_x) * segment_x + (point_y - start_y) * segment_y
        ) / length_squared
        t = self._clamp(t, 0.0, 1.0)
        return start_x + segment_x * t, start_y + segment_y * t

    def _ray_segment_intersection(
        self,
        ray_origin: tuple[float, float],
        ray_direction: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Execute ray segment intersection behavior.

Parameters
----------
ray_origin
    Input used by this creature-domain operation.
ray_direction
    Input used by this creature-domain operation.
start
    Input used by this creature-domain operation.
end
    Input used by this creature-domain operation.
Returns
-------
tuple[float, float] | None
    Result produced by this creature-domain operation."""
        # Keep ray segment intersection behavior explicit in its owning subsystem.
        origin_x, origin_y = ray_origin
        ray_x, ray_y = ray_direction
        start_x, start_y = start
        segment_x = end[0] - start_x
        segment_y = end[1] - start_y
        denominator = self._cross(ray_x, ray_y, segment_x, segment_y)
        if abs(denominator) <= 1e-9:
            return None

        offset_x = start_x - origin_x
        offset_y = start_y - origin_y
        ray_scale = self._cross(offset_x, offset_y, segment_x, segment_y) / denominator
        segment_scale = self._cross(offset_x, offset_y, ray_x, ray_y) / denominator
        if ray_scale < 0.0 or segment_scale < 0.0 or segment_scale > 1.0:
            return None

        return origin_x + ray_x * ray_scale, origin_y + ray_y * ray_scale

    def energy_cost_per_second(self, creature: Creature) -> float:
        """Execute energy cost per second behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep energy cost per second behavior explicit in its owning subsystem.
        range_ratio = creature.vision.range / self.config.max_range
        angle_ratio = creature.vision.angle / self.config.max_angle

        vision_area_ratio = angle_ratio * range_ratio**2

        return (
            self.config.base_energy_cost
            + self.config.area_energy_cost_factor * vision_area_ratio
        )

    def normalized_range(self, creature: Creature) -> float:
        """Execute normalized range behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalized range behavior explicit in its owning subsystem.
        return self._normalize(
            creature.vision.range,
            self.config.min_range,
            self.config.max_range,
        )

    def normalized_angle(self, creature: Creature) -> float:
        """Execute normalized angle behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalized angle behavior explicit in its owning subsystem.
        return self._normalize(
            creature.vision.angle,
            self.config.min_angle,
            self.config.max_angle,
        )

    def normalized_speed(self, creature: Creature, max_speed: float) -> float:
        """Execute normalized speed behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
max_speed
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalized speed behavior explicit in its owning subsystem.
        if max_speed <= 0:
            return 0.0

        return self._clamp01(creature.speed / max_speed)

    def normalized_energy_cost(self, creature: Creature) -> float:
        """Execute normalized energy cost behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalized energy cost behavior explicit in its owning subsystem.
        max_cost = self.config.base_energy_cost + self.config.area_energy_cost_factor

        if max_cost <= 0:
            return 0.0

        return self._clamp01(self.energy_cost_per_second(creature) / max_cost)

    def _signed_angle(self, angle: float) -> float:
        """Execute signed angle behavior.

Parameters
----------
angle
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep signed angle behavior explicit in its owning subsystem.
        wrapped = (angle + pi) % (2.0 * pi) - pi
        # Preserve the previous helper's inclusive positive endpoint.  The two
        # endpoints describe the same direction, but keeping +pi for positive
        # inputs avoids changing existing sensor semantics.
        return pi if wrapped == -pi and angle > 0.0 else wrapped

    def _nearest_proximity_and_angle(
        self,
        candidates: list[_VisionCandidate],
        fov: float,
    ) -> tuple[float, float]:
        """Execute nearest proximity and angle behavior.

Parameters
----------
candidates
    Input used by this creature-domain operation.
fov
    Input used by this creature-domain operation.
Returns
-------
tuple[float, float]
    Result produced by this creature-domain operation."""
        # Keep nearest proximity and angle behavior explicit in its owning subsystem.
        if fov <= 0.0 or not candidates:
            return 0.0, 0.0
        nearest = min(candidates, key=lambda candidate: candidate.surface_distance)
        normalized_angle = nearest.signed_angle / (fov / 2.0)
        return nearest.closeness, self._clamp(normalized_angle, -1.0, 1.0)

    def _empty_target_snapshot(self) -> VisionTargetSnapshot:
        """Execute empty target snapshot behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
VisionTargetSnapshot
    Result produced by this creature-domain operation."""
        # Keep empty target snapshot behavior explicit in its owning subsystem.
        return VisionTargetSnapshot(
            visible=0.0,
            proximity=0.0,
            angle=0.0,
            density=0.0,
            count=0,
        )

    def _cross(self, ax: float, ay: float, bx: float, by: float) -> float:
        """Execute cross behavior.

Parameters
----------
ax
    Input used by this creature-domain operation.
ay
    Input used by this creature-domain operation.
bx
    Input used by this creature-domain operation.
by
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep cross behavior explicit in its owning subsystem.
        return ax * by - ay * bx

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        """Execute normalize behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
minimum
    Input used by this creature-domain operation.
maximum
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep normalize behavior explicit in its owning subsystem.
        if maximum <= minimum:
            return 0.0

        return self._clamp01((value - minimum) / (maximum - minimum))

    def _clamp01(self, value: float) -> float:
        """Execute clamp01 behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep clamp01 behavior explicit in its owning subsystem.
        return self._clamp(value, 0.0, 1.0)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        """Execute clamp behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
minimum
    Input used by this creature-domain operation.
maximum
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep clamp behavior explicit in its owning subsystem.
        return max(minimum, min(maximum, value))
