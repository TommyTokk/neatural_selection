from __future__ import annotations

from dataclasses import dataclass
from math import dist

import pymunk


Color = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(slots=True)
class Creature:
    creature_id: int
    name: str
    body: pymunk.Body
    shape: pymunk.Circle
    radius: float
    energy: float
    color: Color

    @property
    def position(self) -> tuple[float, float]:
        return self.body.position.x, self.body.position.y

    @property
    def heading(self) -> float:
        return self.body.angle

    @property
    def speed(self) -> float:
        return self.body.velocity.length

    def contains_point(self, x: float, y: float) -> bool:
        return dist(self.position, (x, y)) <= self.radius + 4.0
