from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi

from configs.sim_config import VisionConfig
from src.creature import Creature
from src.food import Food

SENSOR_INPUT_COUNT = 5
SENSOR_INPUT_NAMES = (
    "food_closeness",
    "food_angle",
    "creature_closeness",
    "creature_angle",
    "energy",
)


@dataclass(slots=True)
class VisionTargetSnapshot:
    visible: float
    nearest_closeness: float
    nearest_angle: float
    density: float


@dataclass(slots=True)
class BoundarySnapshot:
    pressure: float
    turn: float


@dataclass(slots=True)
class SensorSnapshot:
    food: VisionTargetSnapshot
    creatures: VisionTargetSnapshot
    boundary: BoundarySnapshot
    energy: float
    speed: float
    vision_range: float
    vision_angle: float
    vision_energy_cost: float

    def as_inputs(self) -> list[float]:
        return [
            self._visible_closeness(self.food),
            self.food.nearest_angle,
            self._visible_closeness(self.creatures),
            self.creatures.nearest_angle,
            self.energy,
        ]

    def _visible_closeness(self, target: VisionTargetSnapshot) -> float:
        if target.visible <= 0.0:
            return -1.0
        return target.nearest_closeness


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
    ) -> SensorSnapshot:
        food_snapshot = self._sense_food(creature, foods)
        creature_snapshot = self._sense_creatures(creature, creatures)
        boundary_snapshot = self.sense_boundary(creature, world_bounds)

        return SensorSnapshot(
            food=food_snapshot,
            creatures=creature_snapshot,
            boundary=boundary_snapshot,
            energy=self._clamp01(creature.energy),
            speed=self.normalized_speed(creature, max_speed),
            vision_range=self.normalized_range(creature),
            vision_angle=self.normalized_angle(creature),
            vision_energy_cost=self.normalized_energy_cost(creature),
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

    def visible_foods(self, creature: Creature, foods: list[Food]) -> list[Food]:
        return [
            food
            for food in foods
            if self._target_is_visible(creature, food.position, food.radius)
        ]

    def visible_creatures(
        self,
        creature: Creature,
        creatures: list[Creature],
    ) -> list[Creature]:
        return [
            other
            for other in creatures
            if other.creature_id != creature.creature_id
            and self._target_is_visible(creature, other.position, other.radius)
        ]

    def _sense_food(
        self,
        creature: Creature,
        foods: list[Food],
    ) -> VisionTargetSnapshot:
        return self._sense_targets(
            creature,
            [(food.position, food.radius) for food in foods],
        )

    def _sense_creatures(
        self,
        creature: Creature,
        creatures: list[Creature],
    ) -> VisionTargetSnapshot:
        return self._sense_targets(
            creature,
            [
                (other.position, other.radius)
                for other in creatures
                if other.creature_id != creature.creature_id
            ],
        )

    def _sense_targets(
        self,
        creature: Creature,
        targets: list[tuple[tuple[float, float], float]],
    ) -> VisionTargetSnapshot:
        creature_x, creature_y = creature.position
        vision_range = creature.vision.range
        cone_angle = creature.vision.angle
        if vision_range <= 0 or cone_angle <= 0:
            return VisionTargetSnapshot(
                visible=0.0,
                nearest_closeness=0.0,
                nearest_angle=0.0,
                density=0.0,
            )

        nearest_distance = vision_range
        nearest_angle = 0.0
        density = 0.0
        visible_count = 0

        for (target_x, target_y), target_radius in targets:
            dx = target_x - creature_x
            dy = target_y - creature_y
            distance = hypot(dx, dy)
            surface_distance = max(0.0, distance - target_radius)

            if surface_distance > vision_range:
                continue

            angle_to_target = atan2(dy, dx)
            signed_angle = self._signed_angle(angle_to_target - creature.heading)
            angular_radius = pi if distance <= 0 else atan2(target_radius, distance)

            if abs(signed_angle) > cone_angle / 2 + angular_radius:
                continue

            closeness = 1.0 - (surface_distance / vision_range)

            density += closeness
            visible_count += 1

            if surface_distance < nearest_distance:
                nearest_distance = surface_distance
                nearest_angle = signed_angle

        if visible_count == 0:
            return VisionTargetSnapshot(
                visible=0.0,
                nearest_closeness=0.0,
                nearest_angle=0.0,
                density=0.0,
            )

        nearest_closeness = 1.0 - (nearest_distance / vision_range)
        normalized_angle = nearest_angle / (cone_angle / 2)

        return VisionTargetSnapshot(
            visible=1.0,
            nearest_closeness=self._clamp01(nearest_closeness),
            nearest_angle=self._clamp(normalized_angle, -1.0, 1.0),
            density=self._clamp01(density),
        )

    def _target_is_visible(
        self,
        creature: Creature,
        target_position: tuple[float, float],
        target_radius: float,
    ) -> bool:
        creature_x, creature_y = creature.position
        target_x, target_y = target_position
        vision_range = creature.vision.range
        cone_angle = creature.vision.angle
        if vision_range <= 0 or cone_angle <= 0:
            return False

        dx = target_x - creature_x
        dy = target_y - creature_y
        distance = hypot(dx, dy)
        surface_distance = max(0.0, distance - target_radius)
        if surface_distance > vision_range:
            return False

        angle_to_target = atan2(dy, dx)
        signed_angle = self._signed_angle(angle_to_target - creature.heading)
        angular_radius = pi if distance <= 0 else atan2(target_radius, distance)
        return abs(signed_angle) <= cone_angle / 2 + angular_radius

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
        max_cost = (
            self.config.base_energy_cost
            + self.config.area_energy_cost_factor
        )

        if max_cost <= 0:
            return 0.0

        return self._clamp01(self.energy_cost_per_second(creature) / max_cost)

    def _signed_angle(self, angle: float) -> float:
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def _normalize(self, value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 0.0

        return self._clamp01((value - minimum) / (maximum - minimum))

    def _clamp01(self, value: float) -> float:
        return self._clamp(value, 0.0, 1.0)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))
