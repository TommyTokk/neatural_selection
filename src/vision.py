from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

from configs.sim_config import VisionConfig
from src.creature import Creature
from src.food import Food

SENSOR_INPUT_COUNT = 16
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
)


@dataclass(slots=True)
class VisionTargetSnapshot:
    visible: float
    nearest_closeness: float
    nearest_angle: float
    density: float
    count: int


@dataclass(slots=True)
class BoundarySnapshot:
    pressure: float
    turn: float


@dataclass(slots=True)
class _VisionCandidate:
    kind: str
    source: Food | Creature
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

    def as_inputs(self) -> list[float]:
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
            self._target_proximity(self.food),
            self._target_angle(self.food),
            self._target_proximity(self.creatures),
            self._target_angle(self.creatures),
            self._target_proximity(self.walls),
            self._target_angle(self.walls),
        ]

    def _target_proximity(self, target: VisionTargetSnapshot) -> float:
        if target.visible <= 0.0:
            return 0.0
        return target.nearest_closeness

    def _target_angle(self, target: VisionTargetSnapshot) -> float:
        if target.visible <= 0.0:
            return 0.0
        return target.nearest_angle


class VisionSystem:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config

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
        ignored_food_ids: set[int] | None = None,
    ) -> SensorSnapshot:
        visible_targets = self._visible_targets(
            creature,
            foods,
            creatures,
            ignored_food_ids=ignored_food_ids,
        )
        food_snapshot = self._snapshot_for_kind(creature, visible_targets, "food")
        creature_snapshot = self._snapshot_for_kind(
            creature,
            visible_targets,
            "creature",
        )
        wall_snapshot = self._sense_walls(creature, world_bounds)
        boundary_snapshot = self.sense_boundary(creature, world_bounds)

        return SensorSnapshot(
            food=food_snapshot,
            creatures=creature_snapshot,
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
    ) -> list[_VisionCandidate]:
        candidates: list[_VisionCandidate] = []
        ignored_food_ids = set() if ignored_food_ids is None else ignored_food_ids

        for food in foods:
            if food.id in ignored_food_ids:
                continue
            if self._food_in_mouth_blind_zone(creature, food.position, food.radius):
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

        blocked_intervals: list[tuple[float, float]] = []
        visible_targets: list[_VisionCandidate] = []
        candidates.sort(key=lambda candidate: candidate.surface_distance)

        for candidate in candidates:
            if self._interval_is_blocked(candidate.interval, blocked_intervals):
                continue

            visible_targets.append(candidate)
            self._add_blocked_interval(candidate.interval, blocked_intervals)

        return visible_targets

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

    def _vision_origin(self, creature: Creature) -> tuple[float, float]:
        creature_x, creature_y = creature.position
        return (
            creature_x + cos(creature.heading) * creature.radius * 0.35,
            creature_y + sin(creature.heading) * creature.radius * 0.35,
        )

    def _food_in_mouth_blind_zone(
        self,
        creature: Creature,
        food_position: tuple[float, float],
        food_radius: float,
    ) -> bool:
        creature_x, creature_y = creature.position
        food_x, food_y = food_position
        dx = food_x - creature_x
        dy = food_y - creature_y

        forward_x = cos(creature.heading)
        forward_y = sin(creature.heading)
        lateral_x = -forward_y
        lateral_y = forward_x

        forward_distance = dx * forward_x + dy * forward_y
        lateral_distance = abs(dx * lateral_x + dy * lateral_y)
        mouth_local_forward = forward_distance - creature.radius

        if mouth_local_forward > 0.0:
            return False

        rear_length = max(4.0, creature.radius * 0.45) + food_radius
        half_width = max(2.0, creature.radius * 0.35) + food_radius
        half_width += max(1.0, creature.radius * 0.2)

        return (
            mouth_local_forward >= -rear_length
            and lateral_distance <= half_width
        )

    def _snapshot_for_kind(
        self,
        creature: Creature,
        targets: list[_VisionCandidate],
        kind: str,
    ) -> VisionTargetSnapshot:
        visible = [target for target in targets if target.kind == kind]
        if not visible:
            return self._empty_target_snapshot()

        nearest = min(visible, key=lambda target: target.surface_distance)
        normalized_angle = nearest.signed_angle / (creature.vision.angle / 2)

        return VisionTargetSnapshot(
            visible=1.0,
            nearest_closeness=self._clamp01(nearest.closeness),
            nearest_angle=self._clamp(normalized_angle, -1.0, 1.0),
            density=self._clamp01(sum(target.closeness for target in visible)),
            count=len(visible),
        )

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

        nearest_distance = vision_range
        nearest_angle = 0.0
        best_closeness = 0.0
        visible_count = 0
        density = 0.0

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
                visible_count += 1
                density += closeness

                if surface_distance < nearest_distance:
                    nearest_distance = surface_distance
                    nearest_angle = signed_angle
                    best_closeness = closeness

        if visible_count == 0:
            return self._empty_target_snapshot()

        normalized_angle = nearest_angle / (cone_angle / 2)
        return VisionTargetSnapshot(
            visible=1.0,
            nearest_closeness=self._clamp01(best_closeness),
            nearest_angle=self._clamp(normalized_angle, -1.0, 1.0),
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

    def _empty_target_snapshot(self) -> VisionTargetSnapshot:
        return VisionTargetSnapshot(
            visible=0.0, nearest_closeness=0.0, nearest_angle=0.0, density=0.0, count=0
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
