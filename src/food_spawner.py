from __future__ import annotations

from collections.abc import Iterable
from math import ceil, exp, pi
from random import Random

from configs.sim_config import FoodConfig
from src.biome import Biome, BiomeMap
from src.food import Food


class FoodSpawner:
    def __init__(
        self,
        config: FoodConfig,
        rng: Random,
        biome_map: BiomeMap | None = None,
    ) -> None:
        self.config = config
        self.rng = rng
        self.biome_map = biome_map
        self._next_food_id = 1
        self._spawn_credit = 0.0
        self._low_food_burst_credit = 0.0
        self._pending_low_food_burst_items = 0

    def create_initial_foods(
        self,
        bounds: tuple[float, float, float, float],
        count: int | None = None,
        *,
        required_biomes: Iterable[Biome] = (),
    ) -> list[Food]:
        requested = self.config.initial_food_items if count is None else count
        spawn_count = min(max(0, requested), self.config.max_food_items)
        foods: list[Food] = []
        for biome in required_biomes:
            if len(foods) >= spawn_count:
                break
            food = self.create_food_in_biome(bounds, biome)
            if food is not None:
                foods.append(food)
        foods.extend(
            self.create_food(bounds) for _ in range(spawn_count - len(foods))
        )
        return foods

    def update(
        self,
        delta_time: float,
        bounds: tuple[float, float, float, float],
        current_food_count: int,
        active_species_count: int,
        available_biomass: float,
        *,
        capacity_override: int | None = None,
        spawn_rate_scale: float = 1.0,
        max_spawn_count: int | None = None,
    ) -> list[Food]:
        food_capacity = (
            self.food_capacity()
            if capacity_override is None
            else max(0, int(capacity_override))
        )
        if current_food_count >= food_capacity:
            self._reset_spawn_credits()
            return []

        spawn_pressure = self.food_regrowth_pressure(current_food_count, food_capacity)

        plant_energy_value = self.average_food_energy_value()
        if plant_energy_value <= 0.0:
            return []

        spawnable_biomass = max(0.0, available_biomass)

        available_slots = food_capacity - current_food_count
        biomass_slots = int(spawnable_biomass // plant_energy_value)
        allowed_spawns = min(biomass_slots, available_slots)
        if max_spawn_count is not None:
            allowed_spawns = min(allowed_spawns, max(0, int(max_spawn_count)))
        if allowed_spawns <= 0:
            return []

        spawn_rate = self._spawn_rate_per_second(
            active_species_count,
            spawn_pressure,
        ) * max(0.0, float(spawn_rate_scale))

        self._spawn_credit += max(0.0, delta_time) * spawn_rate
        regular_spawn_count = min(allowed_spawns, int(self._spawn_credit))

        burst_spawn_budget = self._burst_spawn_cap()
        burst_count = self._low_food_burst_count(
            delta_time,
            current_food_count,
            food_capacity,
            min(
                allowed_spawns - regular_spawn_count,
                burst_spawn_budget,
            ),
            allowed_spawns - regular_spawn_count,
        )

        spawn_count = regular_spawn_count + burst_count
        if spawn_count <= 0:
            return []

        self._spawn_credit -= regular_spawn_count
        self._spawn_credit = min(
            self._spawn_credit,
            max(1.0, spawn_rate),
        )

        return [self.create_food(bounds) for _ in range(spawn_count)]

    def create_food(self, bounds: tuple[float, float, float, float]) -> Food:
        radius = self.rng.uniform(
            self.config.min_food_radius,
            self.config.max_food_radius,
        )
        x, y = self._spawn_position(bounds, radius)
        return Food(
            id=self._claim_food_id(),
            x=x,
            y=y,
            radius=radius,
            energy_density=self.config.energy_density,
        )

    def create_food_in_biome(
        self,
        bounds: tuple[float, float, float, float],
        biome: Biome,
    ) -> Food | None:
        """Create one ordinary pellet in ``biome`` or fail without claiming an ID."""
        radius = self.rng.uniform(
            self.config.min_food_radius,
            self.config.max_food_radius,
        )
        position = self._spawn_position_in_biome(bounds, radius, biome)
        if position is None:
            return None
        return Food(
            id=self._claim_food_id(),
            x=position[0],
            y=position[1],
            radius=radius,
            energy_density=self.config.energy_density,
        )

    def _spawn_position_in_biome(
        self,
        bounds: tuple[float, float, float, float],
        radius: float,
        biome: Biome,
    ) -> tuple[float, float] | None:
        biome_map = self.biome_map
        if biome_map is None:
            return None
        rows, columns = (biome_map.biome_ids == int(biome)).nonzero()
        cell_count = len(rows)
        if cell_count == 0:
            return None

        map_left, map_bottom, map_right, map_top = biome_map.world_bounds
        cell_width = (map_right - map_left) / biome_map.grid_width
        cell_height = (map_top - map_bottom) / biome_map.grid_height
        left, bottom, right, top = bounds
        inner_left = left + radius
        inner_bottom = bottom + radius
        inner_right = right - radius
        inner_top = top - radius
        if inner_left >= inner_right or inner_bottom >= inner_top:
            return None

        # Visit every cached biome cell at most once, starting from a seeded
        # random offset. This is bounded but does not make narrow biomes lose
        # their bootstrap guarantee through rejection-sampling bad luck.
        start = self.rng.randrange(cell_count)
        for offset in range(cell_count):
            index = (start + offset) % cell_count
            row = int(rows[index])
            column = int(columns[index])
            x_low = max(inner_left, map_left + column * cell_width)
            x_high = min(inner_right, map_left + (column + 1) * cell_width)
            y_low = max(inner_bottom, map_bottom + row * cell_height)
            y_high = min(inner_top, map_bottom + (row + 1) * cell_height)
            if x_low >= x_high or y_low >= y_high:
                continue
            x = x_low + self.rng.random() * (x_high - x_low)
            y = y_low + self.rng.random() * (y_high - y_low)
            if biome_map.biome_at(x, y) is biome:
                return x, y
        return None

    def _spawn_position(
        self,
        bounds: tuple[float, float, float, float],
        radius: float,
    ) -> tuple[float, float]:
        biome_map = self.biome_map
        if biome_map is None or self.rng.random() < biome_map.uniform_spawn_chance:
            return self._uniform_spawn_position(bounds, radius)

        weights = tuple(max(0.0, weight) for weight in biome_map.spawn_weights.values())
        max_weight = max(weights, default=0.0)
        if max_weight <= 0.0:
            return self._uniform_spawn_position(bounds, radius)

        for _ in range(biome_map.max_spawn_attempts):
            x, y = self._uniform_spawn_position(bounds, radius)
            acceptance = biome_map.spawn_weight_at(x, y) / max_weight
            if self.rng.random() <= acceptance:
                return x, y

        return self._uniform_spawn_position(bounds, radius)

    def _uniform_spawn_position(
        self,
        bounds: tuple[float, float, float, float],
        radius: float,
    ) -> tuple[float, float]:
        left, bottom, right, top = bounds
        return (
            self.rng.uniform(left + radius, right - radius),
            self.rng.uniform(bottom + radius, top - radius),
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

    def food_regrowth_pressure(
        self, current_food_count: int, food_capacity: int
    ) -> float:
        if food_capacity <= 0:
            return 0.0

        food_ratio = max(0.0, min(1.0, current_food_count / food_capacity))
        logistic_food_pressure = 4.0 * food_ratio * (1.0 - food_ratio)
        seeding_pressure = 0.05 if current_food_count == 0 else 0.0
        return max(logistic_food_pressure, seeding_pressure)

    def food_capacity(self, active_species_count: int | None = None) -> int:
        return max(0, self.config.max_food_items)

    def low_food_shortage_ratio(
        self,
        current_food_count: int,
        food_capacity: int,
    ) -> float:
        threshold = max(0.001, self.config.low_food_pressure_threshold)
        food_ratio = current_food_count / max(1, food_capacity)
        return max(0.0, min(1.0, (threshold - food_ratio) / threshold))

    def _spawn_rate_per_second(
        self,
        active_species_count: int,
        spawn_pressure: float,
    ) -> float:
        active_max_rate = (
            self.config.max_biomass_spawns_per_second
            * self.species_growth_multiplier(active_species_count)
        )
        return active_max_rate * spawn_pressure

    def species_growth_multiplier(self, active_species_count: int) -> float:
        species_pressure = -0.6 * (max(0, active_species_count) - 4.0)
        return 0.5 + (1.5 / (1.0 + exp(species_pressure)))

    def _low_food_burst_count(
        self,
        delta_time: float,
        current_food_count: int,
        food_capacity: int,
        remaining_spawns: int,
        refill_limit: int | None = None,
    ) -> int:
        burst_items = max(0, self.config.low_food_burst_items)
        if burst_items <= 0 or remaining_spawns <= 0:
            return 0

        food_ratio = current_food_count / max(1, food_capacity)
        if (
            food_ratio > self.config.critical_food_ratio
            and self._pending_low_food_burst_items <= 0
        ):
            self._low_food_burst_credit = 0.0
            return 0

        shortage_ratio = self.low_food_shortage_ratio(
            current_food_count,
            food_capacity,
        )
        if shortage_ratio <= 0.0:
            self._low_food_burst_credit = 0.0
            self._pending_low_food_burst_items = 0
            return 0

        if self._pending_low_food_burst_items > 0:
            return self._spend_pending_low_food_burst(remaining_spawns)

        burst_count = self._immediate_low_food_refill_count(
            current_food_count,
            food_capacity,
            remaining_spawns if refill_limit is None else refill_limit,
        )
        if burst_count > 0:
            self._low_food_burst_credit = 0.0
            self._pending_low_food_burst_items += burst_count
            return self._spend_pending_low_food_burst(remaining_spawns)

        burst_interval = max(0.001, self.config.low_food_burst_interval)
        self._low_food_burst_credit += (
            max(0.0, delta_time) * shortage_ratio / burst_interval
        )
        burst_events = int(self._low_food_burst_credit)
        if burst_events > 0:
            self._pending_low_food_burst_items += burst_events * burst_items
            self._low_food_burst_credit -= burst_events
            self._low_food_burst_credit = min(self._low_food_burst_credit, 1.0)

        return self._spend_pending_low_food_burst(remaining_spawns)

    def _immediate_low_food_refill_count(
        self,
        current_food_count: int,
        food_capacity: int,
        remaining_spawns: int,
    ) -> int:
        emergency_target = ceil(
            food_capacity * max(0.0, self.config.critical_food_ratio)
        )
        emergency_deficit = max(0, emergency_target - current_food_count)
        if emergency_deficit <= 0:
            return 0

        burst_items = max(0, self.config.low_food_burst_items)
        return min(remaining_spawns, emergency_deficit, burst_items)

    def _spend_pending_low_food_burst(self, remaining_spawns: int) -> int:
        burst_count = min(remaining_spawns, self._pending_low_food_burst_items)
        self._pending_low_food_burst_items -= burst_count
        return burst_count

    def _burst_spawn_cap(self) -> int:
        return max(1, ceil(max(1, self.config.low_food_burst_items) * 0.25))

    def _reset_spawn_credits(self) -> None:
        self._spawn_credit = 0.0
        self.reset_low_food_burst_state()

    def reset_low_food_burst_state(self) -> None:
        """Discard burst progress created by an earlier configuration."""
        self._low_food_burst_credit = 0.0
        self._pending_low_food_burst_items = 0
