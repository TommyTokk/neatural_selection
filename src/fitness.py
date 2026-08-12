from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot, isfinite, pi

from configs.sim_config import FlockingBenchmarkConfig
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
    distance_traveled: float = 0.0
    last_reproduction_age: float = -1_000_000.0
    offspring_count: int = 0
    matured_offspring_ids: list[int] = field(default_factory=list)
    evaluation_start_age_seconds: float = 0.0
    flocking_benchmark_reward: float = 0.0
    _legacy_energy_gained: float = field(
        default=0.0,
        repr=False,
        compare=False,
    )

    def record_tick(self, delta_time: float, speed: float) -> None:
        self.age_seconds += delta_time
        self.distance_traveled += max(0.0, speed) * max(0.0, delta_time)

    def record_food(self, depleted: bool = True) -> None:
        if depleted:
            self.food_eaten += 1

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

    def score(self, creature: object) -> float:
        """Return implicit fitness from lifetime world energy and an age tie-breaker."""
        gathered = float(getattr(creature, "total_energy_gathered", 0.0))
        if not isfinite(gathered):
            gathered = 0.0
        return max(0.0, gathered) + max(0.0, self.age_seconds) * 0.001

    def __getstate__(self) -> dict[str, object]:
        """Exclude one-shot legacy migration data from new checkpoints."""
        return {
            "age_seconds": self.age_seconds,
            "food_eaten": self.food_eaten,
            "distance_traveled": self.distance_traveled,
            "last_reproduction_age": self.last_reproduction_age,
            "offspring_count": self.offspring_count,
            "matured_offspring_ids": self.matured_offspring_ids,
            "evaluation_start_age_seconds": self.evaluation_start_age_seconds,
            "flocking_benchmark_reward": self.flocking_benchmark_reward,
        }

    def __setstate__(self, state: object) -> None:
        """Load current state and retain old energy totals for checkpoint migration."""
        values: dict[str, object]
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[1], dict)
        ):
            values = state[1]
        elif isinstance(state, dict):
            values = state
        else:
            values = {}

        self.age_seconds = float(values.get("age_seconds", 0.0))
        self.food_eaten = int(values.get("food_eaten", 0))
        self.distance_traveled = float(values.get("distance_traveled", 0.0))
        self.last_reproduction_age = float(
            values.get("last_reproduction_age", -1_000_000.0)
        )
        self.offspring_count = int(values.get("offspring_count", 0))
        self.matured_offspring_ids = list(
            values.get("matured_offspring_ids", [])
        )
        self.evaluation_start_age_seconds = float(
            values.get("evaluation_start_age_seconds", 0.0)
        )
        self.flocking_benchmark_reward = float(
            values.get("flocking_benchmark_reward", 0.0)
        )
        try:
            legacy_energy = float(values.get("energy_gained", 0.0))
        except (TypeError, ValueError, OverflowError):
            legacy_energy = 0.0
        self._legacy_energy_gained = (
            max(0.0, legacy_energy) if isfinite(legacy_energy) else 0.0
        )
