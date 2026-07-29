from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot, pi

from configs.sim_config import FlockingBenchmarkConfig, FitnessConfig
from src.flocking import SocialObservation


@dataclass(slots=True)
class CreatureFitness:
    age_seconds: float = 0.0
    food_eaten: int = 0
    food_discovered: int = 0
    energy_gained: float = 0.0
    movement_effort: float = 0.0
    distance_traveled: float = 0.0
    trait_energy_cost: float = 0.0
    last_reproduction_age: float = -1_000_000.0
    offspring_count: int = 0
    matured_offspring_ids: list[int] = field(default_factory=list)
    discovered_food_ids: set[int] = field(default_factory=set)
    evaluation_start_age_seconds: float = 0.0
    flocking_benchmark_reward: float = 0.0

    def record_tick(self, delta_time: float, speed: float, max_speed: float) -> None:
        self.age_seconds += delta_time
        self.distance_traveled += max(0.0, speed) * max(0.0, delta_time)
        if max_speed > 0.0:
            self.movement_effort += min(speed / max_speed, 1.0) * delta_time

    def record_food(self, energy_gained: float, depleted: bool = True) -> None:
        if depleted:
            self.food_eaten += 1
        self.energy_gained += energy_gained

    def record_food_discoveries(self, food_ids: list[int]) -> None:
        for food_id in food_ids:
            if food_id in self.discovered_food_ids:
                continue
            self.discovered_food_ids.add(food_id)
            self.food_discovered += 1

    def record_trait_cost(self, cost_per_second: float, delta_time: float) -> None:
        self.trait_energy_cost += max(0.0, cost_per_second) * max(0.0, delta_time)

    def record_reproduction(self) -> None:
        self.last_reproduction_age = self.age_seconds
        self.offspring_count += 1

    def seconds_since_reproduction(self) -> float:
        return self.age_seconds - self.last_reproduction_age

    def record_flocking_benchmark(
        self,
        observation: SocialObservation,
        delta_time: float,
        config: FlockingBenchmarkConfig,
    ) -> float:
        if not config.enabled or observation.effective_count <= 0.0:
            return 0.0
        group_presence = min(
            max(
                observation.effective_count
                / max(1, config.target_group_size - 1),
                0.0,
            ),
            1.0,
        )
        alignment_quality = min(
            max(1.0 - observation.mean_heading_error / pi, 0.0),
            1.0,
        )
        spacing_quality = exp(
            -(
                (
                    observation.mean_neighbor_distance
                    - config.target_spacing
                )
                / config.spacing_tolerance
            )
            ** 2
        )
        movement_quality = min(
            hypot(*observation.mean_group_velocity)
            / config.reference_speed,
            1.0,
        )
        quality = (
            group_presence
            * alignment_quality
            * spacing_quality
            * movement_quality
        )
        before = self.flocking_benchmark_reward
        self.flocking_benchmark_reward = min(
            config.max_per_evaluation,
            before
            + quality
            * max(0.0, delta_time)
            * config.reward_rate,
        )
        return self.flocking_benchmark_reward - before

    def average_speed(self) -> float:
        evaluation_age = self.evaluation_age_seconds()
        if evaluation_age <= 0.0:
            return 0.0
        return self.distance_traveled / evaluation_age

    def evaluation_age_seconds(self) -> float:
        return max(0.0, self.age_seconds - self.evaluation_start_age_seconds)

    def score(self, config: FitnessConfig) -> float:
        evaluation_age = self.evaluation_age_seconds()
        scoring_age = max(evaluation_age, config.efficiency_min_age_seconds)
        capped_discoveries = min(self.food_discovered, config.food_discovery_cap)
        energy_efficiency = self.energy_gained / scoring_age

        return (
            evaluation_age * config.age_weight
            + capped_discoveries * config.food_discovery_weight
            # + self.food_eaten * config.food_eaten_weight
            + self.energy_gained * config.energy_gained_weight
            + energy_efficiency * config.energy_efficiency_weight
            + self.offspring_count * config.offspring_weight
            + len(self.matured_offspring_ids) * config.matured_offspring_weight
            - self.movement_effort * config.movement_effort_penalty
            - self.trait_energy_cost * config.trait_energy_cost_penalty_weight
            + self.flocking_benchmark_reward
        )
