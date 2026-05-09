from __future__ import annotations

from random import Random

from configs.sim_config import FoodConfig
from src.food import Food


class FoodSpawner:
    def __init__(self, config: FoodConfig, rng: Random) -> None:
        self.config = config
        self.rng = rng
        self._next_food_id = 1
        self._spawn_timer = 0.0

    def create_initial_foods(
        self, bounds: tuple[float, float, float, float]
    ) -> list[Food]:
        count = min(self.config.initial_food_items, self.config.max_food_items)
        return [self.create_food(bounds) for _ in range(count)]

    def update(
        self,
        delta_time: float,
        bounds: tuple[float, float, float, float],
        current_food_count: int,
    ) -> list[Food]:
        if current_food_count >= self.config.max_food_items:
            self._spawn_timer = 0.0
            return []

        self._spawn_timer += delta_time
        if self._spawn_timer < self.config.spawn_interval:
            return []

        spawn_count = int(self._spawn_timer // self.config.spawn_interval)
        self._spawn_timer %= self.config.spawn_interval
        available_slots = self.config.max_food_items - current_food_count
        return [
            self.create_food(bounds)
            for _ in range(min(spawn_count, available_slots))
        ]

    def create_food(self, bounds: tuple[float, float, float, float]) -> Food:
        left, bottom, right, top = bounds
        radius = self.rng.uniform(
            self.config.min_food_radius,
            self.config.max_food_radius,
        )
        return Food(
            id=self._claim_food_id(),
            x=self.rng.uniform(left + radius, right - radius),
            y=self.rng.uniform(bottom + radius, top - radius),
            radius=radius,
            energy_density=self.config.energy_density,
        )

    def _claim_food_id(self) -> int:
        food_id = self._next_food_id
        self._next_food_id += 1
        return food_id
