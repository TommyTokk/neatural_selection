from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, hypot, pi, sin

from configs.sim_config import MetabolismConfig, VisionConfig
from src.creature import Creature
from src.communication import AcousticSnapshot, PheromoneSnapshot
from src.food import Food

SENSOR_INPUT_COUNT = 37
SENSOR_INPUT_NAMES = (
    "constant",
    "hungriness",
    "maturity",
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
    "biome_fertility_forward_left",
    "biome_fertility_forward_right",
    "biome_fertility_delta",
    "own_infant_proximity",
    "own_infant_angle",
    "flock_center_proximity",
    "flock_center_angle",
    "flock_average_relative_heading",
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
    forward_left: float = 0.0
    forward_right: float = 0.0
    delta: float = 0.0


@dataclass(slots=True)
class FlockSensorSnapshot:
    center_proximity: float = 0.0
    center_angle: float = 0.0
    average_relative_heading: float = 0.0
    flockmate_count: int = 0
    separation_relative_heading: float = 0.0
    separation_strength: float = 0.0
    average_flockmate_proximity: float = 0.0


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
    maturity: float
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
    acoustic: AcousticSnapshot = field(default_factory=AcousticSnapshot)
    pheromones: PheromoneSnapshot = field(default_factory=PheromoneSnapshot)

    def as_inputs(self) -> list[float]:
        # Inputs 18-21 are body-relative biome smell samples, not a direct
        # mathematical gradient.
        return [
            1.0,  # constant
            1.0 - self.energy,  # hungriness
            self.maturity,
            self.energy,
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
            self.biome.forward_left,
            self.biome.forward_right,
            self.biome.delta,
            self.own_infants.proximity,
            self.own_infants.angle,
            self.flock.center_proximity,
            self.flock.center_angle,
            self.flock.average_relative_heading,
            self.stomach_fullness,
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
    ) -> None:
        self.config = config
        self.eating_distance = eating_distance
        self.stomach_capacity_per_radius = stomach_capacity_per_radius

    def sense(
        self,
        creature: Creature,
        foods: list[Food],
        creatures: list[Creature],
        world_bounds: tuple[float, float, float, float],
        max_speed: float,
        maturity: float = 0.0,
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
            maturity=maturity,
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
        maturity: float = 0.0,
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
            world_bounds,
            max_speed,
            maturity=maturity,
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
        world_bounds: tuple[float, float, float, float],
        max_speed: float,
        maturity: float = 0.0,
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
        flock_snapshot = self._flock_snapshot(creature, visible_targets)

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
            maturity=maturity,
            visible_food_count=food_snapshot.count,
            visible_creature_count=creature_snapshot.count,
            clock_tik_tok=clock_tik_tok,
            clock_chronometer=clock_chronometer,
            clock_time_alive=clock_time_alive,
            is_grabbing=1.0 if is_grabbing else 0.0,
            stomach_fullness=self.stomach_fullness(creature),
            flock=flock_snapshot,
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
    ) -> FlockSensorSnapshot:
        visible_creatures = [
            target
            for target in visible_targets
            if target.kind == "creature" and target.source is not None
        ]

        separation_x = 0.0
        separation_y = 0.0
        vision_range = creature.vision.range
        for target in visible_creatures:
            neighbor = target.source
            away_x = creature.position[0] - neighbor.position[0]
            away_y = creature.position[1] - neighbor.position[1]
            distance = hypot(away_x, away_y)
            if distance <= 1e-12:
                continue
            proximity = (
                0.0
                if vision_range <= 0.0
                else self._clamp01(1.0 - distance / vision_range)
            )
            separation_x += (away_x / distance) * proximity
            separation_y += (away_y / distance) * proximity

        if visible_creatures:
            separation_x /= len(visible_creatures)
            separation_y /= len(visible_creatures)
        separation_strength = self._clamp01(hypot(separation_x, separation_y))
        separation_relative_heading = (
            0.0
            if separation_strength <= 1e-12
            else self._signed_angle(
                atan2(separation_y, separation_x) - creature.heading
            )
        )

        species_id = self._species_id(creature)
        flockmate_targets = [
            target
            for target in visible_creatures
            if self._species_id(target.source) == species_id
        ]
        flockmates = [target.source for target in flockmate_targets]
        if not flockmates:
            return FlockSensorSnapshot(
                separation_relative_heading=separation_relative_heading,
                separation_strength=separation_strength,
            )

        average_flockmate_proximity = (
            0.0
            if vision_range <= 0.0
            else sum(
                self._clamp01(
                    1.0
                    - hypot(
                        flockmate.position[0] - creature.position[0],
                        flockmate.position[1] - creature.position[1],
                    )
                    / vision_range
                )
                for flockmate in flockmates
            )
            / len(flockmates)
        )

        center_x = sum(flockmate.position[0] for flockmate in flockmates) / len(
            flockmates
        )
        center_y = sum(flockmate.position[1] for flockmate in flockmates) / len(
            flockmates
        )
        dx = center_x - creature.position[0]
        dy = center_y - creature.position[1]
        center_distance = hypot(dx, dy)
        center_proximity = (
            0.0
            if creature.vision.range <= 0.0
            else self._clamp01(1.0 - center_distance / creature.vision.range)
        )
        center_relative_angle = self._signed_angle(atan2(dy, dx) - creature.heading)

        heading_x = sum(cos(flockmate.heading) for flockmate in flockmates)
        heading_y = sum(sin(flockmate.heading) for flockmate in flockmates)
        average_heading = (
            creature.heading
            if abs(heading_x) <= 1e-12 and abs(heading_y) <= 1e-12
            else atan2(heading_y, heading_x)
        )
        relative_heading = self._signed_angle(average_heading - creature.heading)

        return FlockSensorSnapshot(
            center_proximity=center_proximity,
            center_angle=self._normalized_view_angle(
                center_relative_angle,
                creature.vision.angle,
            ),
            average_relative_heading=self._clamp(relative_heading / pi, -1.0, 1.0),
            flockmate_count=len(flockmates),
            separation_relative_heading=separation_relative_heading,
            separation_strength=separation_strength,
            average_flockmate_proximity=average_flockmate_proximity,
        )

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

        proximity, angle = self._compute_proximity_and_angle(
            visible,
            creature.vision.angle,
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

        proximity, angle = self._compute_proximity_and_angle(
            wall_candidates,
            cone_angle,
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
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def _compute_proximity_and_angle(
        self,
        candidates: list[_VisionCandidate],
        fov: float,
    ) -> tuple[float, float]:
        if fov <= 0.0:
            return 0.0, 0.0

        max_proximity = 0.0
        total_weight = 0.0
        weighted_angle_sum = 0.0

        for candidate in candidates:
            angular_width = candidate.interval[1] - candidate.interval[0]
            weight = candidate.closeness * angular_width
            max_proximity = max(max_proximity, candidate.closeness)
            total_weight += weight
            weighted_angle_sum += candidate.signed_angle * weight

        if total_weight == 0.0:
            return max_proximity, 0.0

        avg_angle = weighted_angle_sum / total_weight
        normalized_angle = avg_angle / (fov / 2.0)
        return max_proximity, self._clamp(normalized_angle, -1.0, 1.0)

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
