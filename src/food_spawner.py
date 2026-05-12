from __future__ import annotations

from math import exp, pi
from random import Random

from configs.sim_config import FoodConfig
from src.food import Food


class FoodSpawner:
    def __init__(self, config: FoodConfig, rng: Random) -> None:
        self.config = config
        self.rng = rng
        self._next_food_id = 1
        self._spawn_credit = 0.0

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
        creature_count: int,
        available_biomass: float,
    ) -> list[Food]:
        if current_food_count >= self.config.max_food_items:
            self._spawn_credit = 0.0
            return []

        spawn_pressure = self.creature_pressure_factor(creature_count)
        if spawn_pressure <= self.config.creature_pressure_spawn_cutoff:
            self._spawn_credit = 0.0
            return []

        plant_energy_value = self.average_food_energy_value()
        spawnable_biomass = available_biomass * spawn_pressure
        if spawnable_biomass <= plant_energy_value:
            return []

        available_slots = self.config.max_food_items - current_food_count
        biomass_slots = int(spawnable_biomass // plant_energy_value)
        allowed_spawns = min(available_slots, biomass_slots)
        if allowed_spawns <= 0:
            return []

        spawn_rate = self._spawn_rate_per_second(allowed_spawns, spawn_pressure)
        self._spawn_credit += max(0.0, delta_time) * spawn_rate
        spawn_count = min(allowed_spawns, int(self._spawn_credit))
        if spawn_count <= 0:
            return []

        self._spawn_credit -= spawn_count
        self._spawn_credit = min(
            self._spawn_credit,
            max(1.0, self.config.max_biomass_spawns_per_second),
        )
        return [
            self.create_food(bounds)
            for _ in range(spawn_count)
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

    def average_food_energy_value(self) -> float:
        average_radius = (
            self.config.min_food_radius + self.config.max_food_radius
        ) * 0.5
        return pi * average_radius**2 * self.config.energy_density

    def creature_pressure_factor(self, creature_count: int) -> float:
        steepness = max(0.001, self.config.creature_pressure_steepness)
        pressure = (creature_count - self.config.creature_pressure_midpoint) / steepness
        return 1.0 / (1.0 + exp(pressure))

    def _spawn_rate_per_second(
        self,
        allowed_spawns: int,
        spawn_pressure: float,
    ) -> float:
        density_pressure = min(1.0, allowed_spawns / self.config.max_food_items)
        biomass_pressure = density_pressure ** self.config.biomass_spawn_pressure_exponent
        return (
            self.config.max_biomass_spawns_per_second
            * biomass_pressure
            * spawn_pressure
        )
