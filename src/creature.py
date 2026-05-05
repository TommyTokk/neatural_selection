from __future__ import annotations

from dataclasses import dataclass
from math import cos, dist, sin

import arcade


Color = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(slots=True)
class Creature:
    creature_id: int  # ID of the creature
    name: str  # Name of the creature
    anchor_x: float  # X position ratio (0.0 to 1.0) where the creature is anchored in the environment
    anchor_y: float  # Y position ratio (0.0 to 1.0) where the creature is anchored in the environment
    drift_radius: float  # Maximum distance the creature can drift from its anchor point
    drift_speed: float  # Speed at which the creature drifts around its anchor point
    heading_speed: float  # Speed at which the creature changes its heading direction
    energy: float  # Current energy level of the creature, which may affect its behavior and appearance
    color: Color  # Color of the creature, which may be used for rendering and can be influenced by its energy level or other factors

    def get_screen_position(
        self, bounds: arcade.Rect, elapsed_time: float
    ) -> tuple[float, float]:
        offset_x = (
            sin(elapsed_time * self.drift_speed + self.creature_id) * self.drift_radius
        )
        offset_y = (
            cos(elapsed_time * self.drift_speed * 0.8 + self.creature_id)
            * self.drift_radius
        )

        x_ratio = min(0.93, max(0.007, self.anchor_x + offset_x))
        y_ratio = min(0.93, max(0.07, self.anchor_y + offset_y))

        return (
            bounds.left + bounds.width * x_ratio,
            bounds.bottom + bounds.height * y_ratio,
        )

    def get_heading(self, elapsed_time: float) -> float:
        return elapsed_time * self.heading_speed + self.creature_id * 0.9

    def get_radius(self, bounds: arcade.Rect) -> float:
        return max(12.0, min(18.0, bounds.width * 0.015))

    def contains_screen_point(
        self, x: float, y: float, bounds: arcade.Rect, elapsed_time: float
    ) -> bool:
        center = self.get_screen_position(bounds, elapsed_time)
        return dist(center, (x, y)) <= self.get_radius(bounds) + 4.0

    def triangle_points(
        self, bounds: arcade.Rect, elapsed_time: float
    ) -> list[tuple[float, float]]:
        x, y = self.get_screen_position(bounds, elapsed_time)
        angle = self.get_heading(elapsed_time)
        size = self.get_radius(bounds)
        head_center = (x + cos(angle) * size * 0.62, y + sin(angle) * size * 0.62)
        left_head = (
            head_center[0] + cos(angle + 1.57) * size * 0.72,
            head_center[1] + sin(angle + 1.57) * size * 0.72,
        )
        right_head = (
            head_center[0] + cos(angle - 1.57) * size * 0.72,
            head_center[1] + sin(angle - 1.57) * size * 0.72,
        )
        tail = (x - cos(angle) * size * 0.95, y - sin(angle) * size * 0.95)
        return [left_head, right_head, tail]

    def head_base_points(
        self, bounds: arcade.Rect, elapsed_time: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        points = self.triangle_points(bounds, elapsed_time)
        return points[0], points[1]

    def eye_positions(
        self, bounds: arcade.Rect, elapsed_time: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        left_head, right_head = self.head_base_points(bounds, elapsed_time)
        center_x = (left_head[0] + right_head[0]) / 2
        center_y = (left_head[1] + right_head[1]) / 2

        left_eye = (
            center_x + (left_head[0] - center_x) * 0.58,
            center_y + (left_head[1] - center_y) * 0.58,
        )
        right_eye = (
            center_x + (right_head[0] - center_x) * 0.58,
            center_y + (right_head[1] - center_y) * 0.58,
        )
        return left_eye, right_eye
