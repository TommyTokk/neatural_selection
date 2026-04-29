from __future__ import annotations

from dataclasses import dataclass
from math import cos, dist, sin

import arcade


Color = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(slots=True)
class Creature:
    creature_id: int  # ID of the creature
    name: str  # Name of the creature
    anchor_x: float
    anchor_y: float
    drift_radius: float
    drift_speed: float
    heading_speed: float
    energy: float
    color: Color

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
        front = (x + cos(angle) * size, y + sin(angle) * size)
        left = (
            x + cos(angle + 2.45) * size * 0.72,
            y + sin(angle + 2.45) * size * 0.72,
        )
        right = (
            x + cos(angle - 2.45) * size * 0.72,
            y + sin(angle - 2.45) * size * 0.72,
        )
        return [front, left, right]
