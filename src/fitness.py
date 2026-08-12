from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot, pi

from configs.sim_config import FlockingBenchmarkConfig, FitnessConfig
from src.flocking import SocialObservation


def flocking_benchmark_quality(
    observation: SocialObservation,
    config: FlockingBenchmarkConfig,
) -> float:
    """Calculate the instantaneous benchmark quality without retaining a snapshot."""
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
            (observation.mean_neighbor_distance - config.target_spacing)
            / config.spacing_tolerance
        )
        ** 2
    )
    movement_quality = min(
        hypot(*observation.mean_group_velocity) / config.reference_speed,
        1.0,
    )
    return (
        group_presence
        * alignment_quality
        * spacing_quality
        * movement_quality
    )


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
        quality = flocking_benchmark_quality(observation, config)
        return self.record_flocking_benchmark_quality(
            quality,
            delta_time,
            config,
        )

    def record_flocking_benchmark_quality(
        self,
        quality: float,
        delta_time: float,
        config: FlockingBenchmarkConfig,
    ) -> float:
        """Record a precomputed scalar quality without a diagnostic object."""
        if not config.enabled or quality <= 0.0:
            return 0.0
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
        """
        Calculates a scale-normalized, multi-tiered evolutionary fitness score.

        1. Base Survival & Activity Score (Bounded [0, 1]):
           - Longevity (Age ratio relative to target expected lifespan)
           - Net Energy Efficiency Ratio (Energy Gained vs Total Expenditure)
           - Exploration (Capped food discovery ratio)
           - Social Benchmark (Flocking quality ratio)

        2. Reproductive Success Multiplier:
           - Scales the Base Score upward based on offspring and matured offspring counts,
             ensuring evolutionary selection prioritizes gene propagation over pure gluttony.
        """
        evaluation_age = self.evaluation_age_seconds()
        if evaluation_age <= 0.0:
            return 0.0

        # --- 1. Normalized Longevity Component [0, 1] ---
        age_norm = min(evaluation_age / max(1.0, config.target_lifespan_seconds), 1.0)

        # --- 2. Bounded Net Energy Efficiency [0, 1] ---
        # Ratio of total energy gained over total expenditure (effort + traits)
        total_expenditure = self.movement_effort + self.trait_energy_cost + 1e-6
        energy_efficiency_norm = self.energy_gained / (self.energy_gained + total_expenditure)

        # --- 3. Bounded Exploration / Discovery Component [0, 1] ---
        capped_discoveries = min(self.food_discovered, config.food_discovery_cap)
        exploration_norm = capped_discoveries / max(1, config.food_discovery_cap)

        # --- 4. Bounded Social/Flocking Component [0, 1] ---
        flocking_norm = min(
            self.flocking_benchmark_reward / max(1e-6, config.max_flocking_reward_cap),
            1.0,
        )

        # --- Combine Normalized Sub-Scores into a Base Survival Score [0, 1] ---
        base_survival_score = (
            config.age_weight * age_norm
            + config.energy_efficiency_weight * energy_efficiency_norm
            + config.exploration_weight * exploration_norm
            + config.flocking_benchmark_weight * flocking_norm
        )

        # --- 5. Multiplicative Reproductive Boost ---
        # Early generations with 0 offspring rely on base_survival_score.
        # Reproducing creatures receive a heavy multiplier on top of their survival base.
        reproduction_multiplier = 1.0 + (
            self.offspring_count * config.offspring_weight
            + len(self.matured_offspring_ids) * config.matured_offspring_weight
        )

        return base_survival_score * reproduction_multiplier