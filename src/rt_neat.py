from __future__ import annotations

from dataclasses import dataclass

from configs.sim_config import PopulationConfig, FitnessConfig
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
    normal_replacements: int = 0
    extinction_replacements: int = 0
    deaths: int = 0
    average_lifespan_at_death: float = 0.0
    average_brain_nodes: float = 0.0
    average_brain_enabled_connections: float = 0.0
    average_brain_connections: float = 0.0
    average_speed: float = 0.0
    average_distance_traveled: float = 0.0
    births_per_minute: float = 0.0
    deaths_per_minute: float = 0.0


class RtNeatManager:
    def __init__(self, brain_controller: NeatBrainController) -> None:
        self.brain_controller = brain_controller
        self.stats = RtNeatStats()
        self.eligible_parent_ids: list[int] = []
        self._lifespan_at_death_total = 0.0
        self._lifespan_at_death_count = 0

    def update_stats(
        self,
        creatures: list[Creature],
        fitness_by_creature_id: dict[int, CreatureFitness],
        population_config: PopulationConfig,
        fitness_config: FitnessConfig,
        elapsed_time: float = 0.0,
    ) -> None:
        live_scores: list[tuple[int, float]] = []
        eligible_scores: list[tuple[int, float]] = []
        live_fitnesses: list[CreatureFitness] = []
        species_counts: dict[int, int] = {}

        for creature in creatures:
            species_id = creature.lineage.species_id
            species_counts[species_id] = species_counts.get(species_id, 0) + 1

        for creature in creatures:
            fitness = fitness_by_creature_id.get(creature.creature_id)
            if fitness is not None:
                score = fitness.score(fitness_config)
                live_scores.append((creature.creature_id, score))
                live_fitnesses.append(fitness)
                if self.is_reproduction_eligible(creature, fitness, population_config):
                    species_size = max(
                        1,
                        species_counts[creature.lineage.species_id],
                    )
                    eligible_scores.append(
                        (creature.creature_id, score / species_size)
                    )

        if not live_scores:
            self.eligible_parent_ids = []
            self.stats.best_fitness = 0.0
            self.stats.average_fitness = 0.0
            self.stats.worst_fitness = 0.0
            self.stats.best_creature_id = None
            self.stats.best_eligible_parent_id = None
            self.stats.evaluated_count = 0
            self.stats.eligible_parent_count = 0
            self.stats.average_speed = 0.0
            self.stats.average_distance_traveled = 0.0
            self.stats.average_brain_nodes = 0.0
            self.stats.average_brain_enabled_connections = 0.0
            self.stats.average_brain_connections = 0.0
            self._update_event_rates(elapsed_time)
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
        self.stats.average_speed = sum(
            fitness.average_speed() for fitness in live_fitnesses
        ) / len(live_fitnesses)
        self.stats.average_distance_traveled = sum(
            fitness.distance_traveled for fitness in live_fitnesses
        ) / len(live_fitnesses)
        self._update_brain_size_stats(creatures)
        self._update_event_rates(elapsed_time)

    def record_normal_replacement(self) -> None:
        self.stats.births += 1
        self.stats.normal_replacements += 1

    def record_extinction_replacements(self, count: int) -> None:
        replacement_count = max(0, count)
        self.stats.births += replacement_count
        self.stats.extinction_replacements += replacement_count

    def record_death(self, fitness: CreatureFitness | None) -> None:
        self.stats.deaths += 1
        if fitness is None:
            return

        self._lifespan_at_death_total += fitness.age_seconds
        self._lifespan_at_death_count += 1
        self.stats.average_lifespan_at_death = (
            self._lifespan_at_death_total / self._lifespan_at_death_count
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

    def _update_brain_size_stats(self, creatures: list[Creature]) -> None:
        if self.brain_controller is None:
            self.stats.average_brain_nodes = 0.0
            self.stats.average_brain_enabled_connections = 0.0
            self.stats.average_brain_connections = 0.0
            return

        brain_sizes: list[tuple[int, int, int]] = []
        for creature in creatures:
            brain = self.brain_controller.brain_for(creature.creature_id)
            if brain is None:
                continue

            enabled_connections = sum(
                1
                for connection in brain.genome.connections.values()
                if connection.enabled
            )
            brain_sizes.append(
                (
                    len(brain.genome.nodes),
                    enabled_connections,
                    len(brain.genome.connections),
                )
            )

        if not brain_sizes:
            self.stats.average_brain_nodes = 0.0
            self.stats.average_brain_enabled_connections = 0.0
            self.stats.average_brain_connections = 0.0
            return

        self.stats.average_brain_nodes = sum(
            nodes for nodes, _, _ in brain_sizes
        ) / len(brain_sizes)
        self.stats.average_brain_enabled_connections = sum(
            enabled for _, enabled, _ in brain_sizes
        ) / len(brain_sizes)
        self.stats.average_brain_connections = sum(
            connections for _, _, connections in brain_sizes
        ) / len(brain_sizes)

    def _update_event_rates(self, elapsed_time: float) -> None:
        elapsed_minutes = max(0.0, elapsed_time) / 60.0
        if elapsed_minutes <= 0.0:
            self.stats.births_per_minute = 0.0
            self.stats.deaths_per_minute = 0.0
            return

        self.stats.births_per_minute = self.stats.births / elapsed_minutes
        self.stats.deaths_per_minute = self.stats.deaths / elapsed_minutes
