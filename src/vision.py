from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin

from configs.sim_config import (
    FlockingConfig,
    MetabolismConfig,
    VisionConfig,
)
from src.flocking import LongRangeSocialObservation
from src.creature import Creature
from src.communication import AcousticObservation, PheromoneSnapshot
from src.food import Food

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
    "biome_fertility_here",
    "biome_fertility_left_gradient",
    "biome_fertility_right_gradient",
    "biome_fertility_trend",
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
    "trail_pheromone_here",
    "trail_pheromone_forward_left",
    "trail_pheromone_forward_right",
    "alarm_pheromone_here",
    "alarm_pheromone_forward_left",
    "alarm_pheromone_forward_right",
)


@dataclass(frozen=True, slots=True)
class SensorContract:
    schema_version: int
    input_names: tuple[str, ...]
    neat_config_path: str

    @property
    def input_count(self) -> int:
        return len(self.input_names)


SENSOR_CONTRACT = SensorContract(
    4,
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


@dataclass(slots=True)
class BoundarySnapshot:
    pressure: float
    turn: float


@dataclass(slots=True)
class BiomeSensorSnapshot:
    here: float = 0.0
    left_gradient: float = 0.0
    right_gradient: float = 0.0
    trend: float = 0.0


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

    def as_inputs(self) -> list[float]:
        stomach_fullness = self._clamp01(self.stomach_fullness)
        energy_percent = self._clamp01(self.energy)
        energy_deficit = max(0.0, 1.0 - energy_percent)
        stomach_emptiness = max(0.0, 1.0 - stomach_fullness)
        feeding_drive = energy_deficit * stomach_emptiness
        prefix = [
            1.0,  # constant
            feeding_drive,
            self.reproductive_readiness,
            energy_percent,
            self.speed,
            min(self.visible_creature_count / 5.0, 1.0),
            min(self.visible_food_count / 10.0, 1.0),
            self.clock_tik_tok,
            self.clock_chronometer,
            self.clock_time_alive,
            self.food.proximity,
            self.food.angle,
            self.creatures.proximity,
            self.creatures.angle,
            self.walls.proximity,
            self.walls.angle,
            self.is_grabbing,
            self.biome.here,
            self.biome.left_gradient,
            self.biome.right_gradient,
            self.biome.trend,
            self.own_infants.proximity,
            self.own_infants.angle,
        ]
        if self.sensor_contract.schema_version == 4:
            social = [
                self._clamp01(self.flock.social_presence),
                self._target_scaled_flockmate_count(
                    self.flock.flockmate_count,
                    self.flock_target_group_size,
                ),
                self._clamp(self.flock.center_forward, -1.0, 1.0),
                self._clamp(self.flock.center_right, -1.0, 1.0),
                self._clamp(
                    self.flock.relative_velocity_forward,
                    -1.0,
                    1.0,
                ),
                self._clamp(
                    self.flock.relative_velocity_right,
                    -1.0,
                    1.0,
                ),
                self._clamp01(self.flock.long_range.intensity),
                self._clamp(
                    self.flock.long_range.direction_forward,
                    -1.0,
                    1.0,
                ),
                self._clamp(
                    self.flock.long_range.direction_right,
                    -1.0,
                    1.0,
                ),
            ]
        else:
            social = [
                self.flock.center_proximity,
                self.flock.center_angle,
                self.flock.average_relative_heading,
                self._normalized_flockmate_count(self.flock.flockmate_count),
            ]
        suffix = [
            stomach_fullness,
            self.acoustic.strength,
            self.acoustic.direction_sin,
            self.acoustic.direction_cos,
            self.acoustic.tone,
            self.pheromones.trail_here,
            self.pheromones.trail_forward_left,
            self.pheromones.trail_forward_right,
            self.pheromones.alarm_here,
            self.pheromones.alarm_forward_left,
            self.pheromones.alarm_forward_right,
        ]
        return [*prefix, *social, *suffix]

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalized_flockmate_count(value: float) -> float:
        effective_count = max(0.0, float(value))
        return effective_count / (effective_count + 3.0)

    @staticmethod
    def _target_scaled_flockmate_count(value: float, target: int) -> float:
        return max(0.0, min(1.0, float(value) / max(1, int(target))))

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


@dataclass(slots=True)
class VisionSenseResult:
    snapshot: SensorSnapshot
    visible_food_ids: list[int]


class VisionSystem:
    def __init__(
        self,
        config: VisionConfig,
        eating_distance: float = MetabolismConfig().eating_distance,
        stomach_capacity_per_radius: float = (
            MetabolismConfig().stomach_capacity_per_radius
        ),
        flock_compatibility_resolver: Callable[[Creature, Creature], float]
        | None = None,
        flocking_config: FlockingConfig | None = None,
    ) -> None:
        self.config = config
        self.eating_distance = eating_distance
        self.stomach_capacity_per_radius = stomach_capacity_per_radius
        self.flock_compatibility_resolver = flock_compatibility_resolver
        self.flocking_config = flocking_config or FlockingConfig()
        self.sensor_contract = SENSOR_CONTRACT

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
    ) -> VisionSenseResult:
        visible_targets = self._visible_targets(
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
        visible_food_ids = [
            target.source.id
            for target in visible_targets
            if target.kind == "food" and isinstance(target.source, Food)
        ]
        return VisionSenseResult(
            snapshot=snapshot,
            visible_food_ids=visible_food_ids,
        )

    def _sensor_snapshot_from_visible_targets(
        self,
        creature: Creature,
        visible_targets: list[_VisionCandidate],
        nearby_creatures: list[Creature],
        world_bounds: tuple[float, float, float, float],
        max_speed: float,
        reproductive_readiness: float = 0.0,
        clock_tik_tok: float = 0.0,
        clock_chronometer: float = 0.0,
        clock_time_alive: float = 0.0,
        is_grabbing: bool = False,
    ) -> SensorSnapshot:
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
            flock=flock_snapshot,
            sensor_contract=self.sensor_contract,
            flock_target_group_size=self.flocking_config.target_group_size,
        )

    def stomach_fullness(self, creature: Creature) -> float:
        capacity = max(
            0.0,
            creature.radius * self.stomach_capacity_per_radius,
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
        visible_creatures = [
            target
            for target in visible_targets
            if target.kind == "creature" and target.source is not None
        ]
        # Crowd separation is deliberately species-independent. It is a local
        # personal-space response, not a same-species flocking response.
        separation_x = 0.0
        separation_y = 0.0
        personal_space_count = 0
        vision_range = creature.vision.range
        personal_space = max(
            0.0,
            self.flocking_config.preferred_personal_space,
        )
        for target in visible_creatures:
            neighbor = target.source
            away_x = creature.position[0] - neighbor.position[0]
            away_y = creature.position[1] - neighbor.position[1]
            distance = hypot(away_x, away_y)
            if distance <= 1e-12 or distance >= personal_space:
                continue
            personal_space_count += 1
            proximity = self._clamp01(1.0 - distance / personal_space)
            separation_x += (away_x / distance) * proximity
            separation_y += (away_y / distance) * proximity

        crowd_separation_strength = self._clamp01(
            hypot(separation_x, separation_y)
        )
        crowd_separation_absolute_angle = (
            0.0
            if crowd_separation_strength <= 1e-12
            else atan2(separation_y, separation_x)
        )
        long_range = self._long_range_social_observation(
            creature,
            [] if nearby_creatures is None else nearby_creatures,
        )

        compatible_flockmates: list[tuple[Creature, float]] = []
        for target in visible_creatures:
            neighbor = target.source
            compatibility = self._flock_compatibility(creature, neighbor)
            if compatibility > 1e-12:
                compatible_flockmates.append((neighbor, compatibility))

        effective_flockmate_count = sum(
            compatibility for _, compatibility in compatible_flockmates
        )
        if effective_flockmate_count <= 1e-12:
            return FlockSensorSnapshot(
                crowd_separation_absolute_angle=(
                    crowd_separation_absolute_angle
                ),
                crowd_separation_strength=crowd_separation_strength,
                visible_creature_count=len(visible_creatures),
                compatible_visible_count=0,
                visible_personal_space_count=personal_space_count,
                long_range=long_range,
            )

        average_flockmate_proximity = (
            0.0
            if vision_range <= 0.0
            else sum(
                compatibility
                * self._clamp01(
                    1.0
                    - (
                        hypot(
                            flockmate.position[0] - creature.position[0],
                            flockmate.position[1] - creature.position[1],
                        )
                        / vision_range
                    )
                )
                for flockmate, compatibility in compatible_flockmates
            )
            / effective_flockmate_count
        )

        center_x = (
            sum(
                flockmate.position[0] * compatibility
                for flockmate, compatibility in compatible_flockmates
            )
            / effective_flockmate_count
        )
        center_y = (
            sum(
                flockmate.position[1] * compatibility
                for flockmate, compatibility in compatible_flockmates
            )
            / effective_flockmate_count
        )
        dx = center_x - creature.position[0]
        dy = center_y - creature.position[1]
        center_distance = hypot(dx, dy)
        center_proximity = (
            0.0
            if creature.vision.range <= 0.0
            else self._clamp01(1.0 - center_distance / creature.vision.range)
        )
        cohesion_absolute_angle = atan2(dy, dx)
        center_relative_angle = self._signed_angle(
            cohesion_absolute_angle - creature.heading
        )

        average_velocity_x = (
            sum(
                flockmate.body.velocity.x * compatibility
                for flockmate, compatibility in compatible_flockmates
            )
            / effective_flockmate_count
        )
        average_velocity_y = (
            sum(
                flockmate.body.velocity.y * compatibility
                for flockmate, compatibility in compatible_flockmates
            )
            / effective_flockmate_count
        )
        actual_average_velocity_x = average_velocity_x
        actual_average_velocity_y = average_velocity_y
        if (
            abs(average_velocity_x) <= 1e-12
            and abs(average_velocity_y) <= 1e-12
        ):
            average_velocity_x = creature.body.velocity.x
            average_velocity_y = creature.body.velocity.y

        alignment_absolute_angle = creature.heading
        if (
            abs(average_velocity_x) > 1e-12
            or abs(average_velocity_y) > 1e-12
        ):
            alignment_absolute_angle = atan2(
                average_velocity_y,
                average_velocity_x,
            )
        relative_heading = self._signed_angle(
            alignment_absolute_angle - creature.heading
        )
        forward_x = cos(creature.heading)
        forward_y = sin(creature.heading)
        right_x = forward_y
        right_y = -forward_x
        position_scale = max(1e-12, creature.vision.range)
        relative_velocity_x = (
            actual_average_velocity_x - creature.body.velocity.x
        )
        relative_velocity_y = (
            actual_average_velocity_y - creature.body.velocity.y
        )
        velocity_scale = max(1e-12, 2.0 * max_speed)
        mean_neighbor_distance = (
            sum(
                hypot(
                    flockmate.position[0] - creature.position[0],
                    flockmate.position[1] - creature.position[1],
                )
                * compatibility
                for flockmate, compatibility in compatible_flockmates
            )
            / effective_flockmate_count
        )
        moving_heading_weight = sum(
            compatibility
            for flockmate, compatibility in compatible_flockmates
            if hypot(
                flockmate.body.velocity.x,
                flockmate.body.velocity.y,
            )
            > 1e-12
        )
        mean_heading_error = (
            0.0
            if moving_heading_weight <= 1e-12
            else sum(
                abs(
                    self._signed_angle(
                        atan2(
                            flockmate.body.velocity.y,
                            flockmate.body.velocity.x,
                        )
                        - creature.heading
                    )
                )
                * compatibility
                for flockmate, compatibility in compatible_flockmates
                if hypot(
                    flockmate.body.velocity.x,
                    flockmate.body.velocity.y,
                )
                > 1e-12
            )
            / moving_heading_weight
        )

        return FlockSensorSnapshot(
            center_proximity=center_proximity,
            center_angle=self._normalized_view_angle(
                center_relative_angle,
                creature.vision.angle,
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
            visible_creature_count=len(visible_creatures),
            compatible_visible_count=len(compatible_flockmates),
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

    def _long_range_social_observation(
        self,
        creature: Creature,
        nearby_creatures: list[Creature],
    ) -> LongRangeSocialObservation:
        config = self.flocking_config.long_range
        if not config.enabled or config.range <= 0.0:
            return LongRangeSocialObservation()
        weighted_x = 0.0
        weighted_y = 0.0
        total = 0.0
        for neighbor in nearby_creatures:
            if neighbor.creature_id == creature.creature_id:
                continue
            dx = neighbor.position[0] - creature.position[0]
            dy = neighbor.position[1] - creature.position[1]
            distance = hypot(dx, dy)
            if distance <= 1e-12 or distance > config.range:
                continue
            weight = (
                self._flock_compatibility(creature, neighbor)
                * self._clamp01(1.0 - distance / config.range)
                * config.strength
            )
            if weight <= 1e-12:
                continue
            weighted_x += (dx / distance) * weight
            weighted_y += (dy / distance) * weight
            total += weight
        magnitude = hypot(weighted_x, weighted_y)
        if total <= 1e-12 or magnitude <= 1e-12:
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

    def _flock_compatibility(
        self,
        creature: Creature,
        neighbor: Creature,
    ) -> float:
        resolver = self.flock_compatibility_resolver
        if resolver is None:
            return (
                1.0
                if self._species_id(creature) == self._species_id(neighbor)
                else 0.0
            )
        return self._clamp01(float(resolver(creature, neighbor)))

    def _normalized_view_angle(self, angle: float, field_of_view: float) -> float:
        if field_of_view <= 0.0:
            return 0.0
        return self._clamp(angle / (field_of_view / 2.0), -1.0, 1.0)

    def _species_id(self, creature: Creature) -> int | None:
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

    def _remove_targets_occluded_by_creatures(
        self,
        candidates: list[_VisionCandidate],
    ) -> list[_VisionCandidate]:
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
        targets: list[_VisionCandidate],
        kind: str,
    ) -> VisionTargetSnapshot:
        visible = [target for target in targets if target.kind == kind]
        if not visible:
            return self._empty_target_snapshot()

        proximity, angle = self._nearest_proximity_and_angle(
            visible, creature.vision.angle
        )

        return VisionTargetSnapshot(
            visible=1.0,
            proximity=proximity,
            angle=angle,
            density=self._clamp01(sum(target.closeness for target in visible)),
            count=len(visible),
        )

    def _sense_walls(
        self,
        creature: Creature,
        world_bounds: tuple[float, float, float, float],
    ) -> VisionTargetSnapshot:
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
        range_ratio = creature.vision.range / self.config.max_range
        angle_ratio = creature.vision.angle / self.config.max_angle

        vision_area_ratio = angle_ratio * range_ratio**2

        return (
            self.config.base_energy_cost
            + self.config.area_energy_cost_factor * vision_area_ratio
        )

    def normalized_range(self, creature: Creature) -> float:
        return self._normalize(
            creature.vision.range,
            self.config.min_range,
            self.config.max_range,
        )

    def normalized_angle(self, creature: Creature) -> float:
        return self._normalize(
            creature.vision.angle,
            self.config.min_angle,
            self.config.max_angle,
        )

    def normalized_speed(self, creature: Creature, max_speed: float) -> float:
        if max_speed <= 0:
            return 0.0

        return self._clamp01(creature.speed / max_speed)

    def normalized_energy_cost(self, creature: Creature) -> float:
        max_cost = self.config.base_energy_cost + self.config.area_energy_cost_factor

        if max_cost <= 0:
            return 0.0

        return self._clamp01(self.energy_cost_per_second(creature) / max_cost)

    def _signed_angle(self, angle: float) -> float:
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
        if fov <= 0.0 or not candidates:
            return 0.0, 0.0
        nearest = min(candidates, key=lambda candidate: candidate.surface_distance)
        normalized_angle = nearest.signed_angle / (fov / 2.0)
        return nearest.closeness, self._clamp(normalized_angle, -1.0, 1.0)

    def _empty_target_snapshot(self) -> VisionTargetSnapshot:
        return VisionTargetSnapshot(
            visible=0.0,
            proximity=0.0,
            angle=0.0,
            density=0.0,
            count=0,
        )

    def _cross(self, ax: float, ay: float, bx: float, by: float) -> float:
        return ax * by - ay * bx

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 0.0

        return self._clamp01((value - minimum) / (maximum - minimum))

    def _clamp01(self, value: float) -> float:
        return self._clamp(value, 0.0, 1.0)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))
