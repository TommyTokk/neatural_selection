from __future__ import annotations

from dataclasses import dataclass

from configs.sim_config import PopulationConfig
from src.creature import Creature
from src.fitness import CreatureFitness
from src.neat_controller import NeatBrainController


@dataclass(slots=True)
class RtNeatStats:
    best_fitness: float = 0.0
    average_fitness: float = 0.0
    worst_fitness: float = 0.0
    best_creature_id: int | None = None
    best_eligible_parent_id: int | None = None
    evaluated_count: int = 0
    eligible_parent_count: int = 0
    births: int = 0
    replacements: int = 0


class RtNeatManager:
    def __init__(self, brain_controller: NeatBrainController) -> None:
        self.brain_controller = brain_controller
        self.stats = RtNeatStats()
        self.eligible_parent_ids: list[int] = []

    def update_stats(
        self,
        creatures: list[Creature],
        fitness_by_creature_id: dict[int, CreatureFitness],
        population_config: PopulationConfig,
    ) -> None:
        live_scores: list[tuple[int, float]] = []
        eligible_scores: list[tuple[int, float]] = []

        for creature in creatures:
            fitness = fitness_by_creature_id.get(creature.creature_id)
            if fitness is not None:
                live_scores.append((creature.creature_id, fitness.score))
                if self.is_reproduction_eligible(
                    creature,
                    fitness,
                    population_config,
                ):
                    eligible_scores.append((creature.creature_id, fitness.score))

        if not live_scores:
            self.eligible_parent_ids = []
            self.stats = RtNeatStats(
                births=self.stats.births,
                replacements=self.stats.replacements,
            )
            return

        best_id, best_score = max(live_scores, key=lambda item: item[1])
        _, worst_score = min(live_scores, key=lambda item: item[1])
        average_score = sum(score for _, score in live_scores) / len(live_scores)
        self.eligible_parent_ids = [
            creature_id
            for creature_id, _ in sorted(
                eligible_scores,
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        self.stats.best_fitness = best_score
        self.stats.average_fitness = average_score
        self.stats.worst_fitness = worst_score
        self.stats.best_creature_id = best_id
        self.stats.evaluated_count = len(live_scores)
        self.stats.eligible_parent_count = len(eligible_scores)
        self.stats.best_eligible_parent_id = (
            self.eligible_parent_ids[0] if self.eligible_parent_ids else None
        )

    def is_reproduction_eligible(
        self,
        creature: Creature,
        fitness: CreatureFitness,
        population_config: PopulationConfig,
    ) -> bool:
        if fitness.age_seconds < population_config.min_reproduction_age:
            return False
        if creature.energy < population_config.reproduction_energy_threshold:
            return False
        return (
            fitness.seconds_since_reproduction()
            >= population_config.reproduction_cooldown
        )
