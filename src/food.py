from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

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
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = self.x, self.y

        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.sensor = True  # Make it a sensor so it doesn't affect physics

        self.energy_value = pi * self.radius**2 * self.energy_density

    @property
    def position(self) -> tuple[float, float]:
        return self.body.position.x, self.body.position.y
