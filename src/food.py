from __future__ import annotations

from dataclasses import dataclass, field
from math import pi, sqrt
from src.collision import BOUNDARY_CATEGORY, CREATURE_CATEGORY, FOOD_CATEGORY
import pymunk


@dataclass(frozen=True, slots=True)
class FoodConsumptionResult:
    energy_removed: float
    depleted: bool


@dataclass(slots=True)
class Food:
    id: int
    x: float
    y: float
    radius: float
    energy_density: float
    cluster_id: int | None = None
    bite_capacity: int = 1
    max_energy: float | None = None
    energy_remaining: float | None = None

    body: pymunk.Body = field(init=False)
    shape: pymunk.Circle = field(init=False)
    original_radius: float = field(init=False)

    def __post_init__(self) -> None:
        calculated_energy = pi * self.radius**2 * self.energy_density
        if self.max_energy is None:
            self.max_energy = calculated_energy
        else:
            self.max_energy = max(0.0, float(self.max_energy))
        if self.energy_remaining is None:
            self.energy_remaining = self.max_energy
        else:
            self.energy_remaining = max(
                0.0,
                min(self.max_energy, float(self.energy_remaining)),
            )
        self.bite_capacity = max(1, int(self.bite_capacity))
        self.original_radius = self.radius
        mass = self._mass_for_radius(self.radius)
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

    @property
    def energy_value(self) -> float:
        return float(self.energy_remaining or 0.0)

    @energy_value.setter
    def energy_value(self, value: float) -> None:
        self.energy_remaining = max(0.0, float(value))

    @property
    def original_energy_value(self) -> float:
        return float(self.max_energy or 0.0)

    @original_energy_value.setter
    def original_energy_value(self, value: float) -> None:
        self.max_energy = max(0.0, float(value))

    def consume_energy(
        self,
        requested_energy: float,
        min_remainder_ratio: float,
    ) -> FoodConsumptionResult:
        requested_energy = max(0.0, requested_energy)
        if requested_energy <= 0.0 or self.energy_value <= 0.0:
            return FoodConsumptionResult(energy_removed=0.0, depleted=False)

        min_remainder_ratio = max(0.0, min(1.0, min_remainder_ratio))
        previous_energy = self.energy_value
        remaining_energy = self.energy_value - requested_energy
        minimum_remainder = self.original_energy_value * min_remainder_ratio

        if remaining_energy <= minimum_remainder + 1e-12:
            self.energy_value = 0.0
            return FoodConsumptionResult(
                energy_removed=previous_energy,
                depleted=True,
            )

        self.energy_value = remaining_energy
        self._resize_for_remaining_energy()
        return FoodConsumptionResult(
            energy_removed=previous_energy - self.energy_value,
            depleted=False,
        )

    def _resize_for_remaining_energy(self) -> None:
        if self.energy_density <= 0.0:
            self.radius = 0.0
        else:
            self.radius = sqrt(self.energy_value / (pi * self.energy_density))

        mass = self._mass_for_radius(self.radius)
        moment = pymunk.moment_for_circle(mass, 0.0, self.radius)
        self.body.mass = mass
        self.body.moment = moment
        unsafe_set_radius = getattr(self.shape, "unsafe_set_radius", None)
        if unsafe_set_radius is None:
            self.shape.radius = self.radius
        else:
            unsafe_set_radius(self.radius)

    def _mass_for_radius(self, radius: float) -> float:
        return (0.2 + radius * 0.035) * 0.9
