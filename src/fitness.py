from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CreatureFitness:
    age_seconds: float = 0.0
    food_eaten: int = 0
    energy_gained: float = 0.0
    movement_effort: float = 0.0
    last_reproduction_age: float = -1_000_000.0
    offspring_count: int = 0

    def record_tick(self, delta_time: float, speed: float, max_speed: float) -> None:
        self.age_seconds += delta_time
        if max_speed > 0.0:
            self.movement_effort += min(speed / max_speed, 1.0) * delta_time

    def record_food(self, energy_gained: float) -> None:
        self.food_eaten += 1
        self.energy_gained += energy_gained

    def record_reproduction(self) -> None:
        self.last_reproduction_age = self.age_seconds
        self.offspring_count += 1

    def seconds_since_reproduction(self) -> float:
        return self.age_seconds - self.last_reproduction_age

    @property
    def score(self) -> float:
        return (
            self.age_seconds * 0.10
            + self.food_eaten * 10.0
            + self.energy_gained * 25.0
            - self.movement_effort * 0.03
        )
