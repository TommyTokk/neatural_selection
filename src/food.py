from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from src.collision import BOUNDARY_CATEGORY, CREATURE_CATEGORY, FOOD_CATEGORY
import pymunk


@dataclass(slots=True)
class Food:
    id: int
    x: float
    y: float
    radius: float
    energy_density: float

    body: pymunk.Body = field(init=False)
    shape: pymunk.Circle = field(init=False)
    energy_value: float = field(init=False)

    def __post_init__(self) -> None:
        self.energy_value = pi * self.radius * 2 * self.energy_density
        mass = 0.2 + self.radius * 0.035
        moment = pymunk.moment_for_circle(mass, 0.0, self.radius)

        self.body = pymunk.Body(mass, moment)
        self.body.position = self.x, self.y

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.sensor = False
        self.shape.elasticity = 0.01
        self.shape.friction = 0.12
        self.shape.filter = pymunk.ShapeFilter(
            categories=FOOD_CATEGORY,
            mask=CREATURE_CATEGORY | FOOD_CATEGORY | BOUNDARY_CATEGORY,
        )

    @property
    def position(self) -> tuple[float, float]:
        return self.body.position.x, self.body.position.y
