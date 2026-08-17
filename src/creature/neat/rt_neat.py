from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from random import Random

from configs.sim_config import PopulationConfig
from src.creature.model import Creature
from src.creature.fitness import CreatureFitness
from src.creature.neat.controller import NeatBrainController


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
    def __init__(
        self,
        brain_controller: NeatBrainController | None,
        rng: Random | None = None,
    ) -> None:
        """Execute init behavior.

Parameters
----------
brain_controller
    Input used by this creature-domain operation.
rng
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.brain_controller = brain_controller
        self.rng = Random(0) if rng is None else rng
        self.stats = RtNeatStats()
        self.eligible_parent_ids: list[int] = []
        self._lifespan_at_death_total = 0.0
        self._lifespan_at_death_count = 0

    def update_stats(
        self,
        creatures: list[Creature],
        fitness_by_creature_id: dict[int, CreatureFitness],
        population_config: PopulationConfig,
        elapsed_time: float = 0.0,
    ) -> None:
        """Execute update stats behavior.

Parameters
----------
creatures
    Input used by this creature-domain operation.
fitness_by_creature_id
    Input used by this creature-domain operation.
population_config
    Input used by this creature-domain operation.
elapsed_time
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep update stats behavior explicit in its owning subsystem.
        live_scores: list[tuple[int, float]] = []
        eligible_scores: list[tuple[int, float]] = []
        live_fitnesses: list[CreatureFitness] = []

        for creature in creatures:
            fitness = fitness_by_creature_id.get(creature.creature_id)
            if fitness is not None:
                score = fitness.score(creature)
                live_scores.append((creature.creature_id, score))
                live_fitnesses.append(fitness)
                if self.is_reproduction_eligible(creature, fitness, population_config):
                    eligible_scores.append((creature.creature_id, score))

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
            creature_id for creature_id, _ in eligible_scores
        ]

        self.stats.best_fitness = best_score
        self.stats.average_fitness = average_score
        self.stats.worst_fitness = worst_score
        self.stats.best_creature_id = best_id
        self.stats.evaluated_count = len(live_scores)
        self.stats.eligible_parent_count = len(eligible_scores)
        self.stats.best_eligible_parent_id = (
            max(eligible_scores, key=lambda item: (item[1], -item[0]))[0]
            if eligible_scores
            else None
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
        """Execute record normal replacement behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep record normal replacement behavior explicit in its owning subsystem.
        self.stats.births += 1
        self.stats.normal_replacements += 1

    def record_extinction_replacements(self, count: int) -> None:
        """Execute record extinction replacements behavior.

Parameters
----------
count
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep record extinction replacements behavior explicit in its owning subsystem.
        replacement_count = max(0, count)
        self.stats.births += replacement_count
        self.stats.extinction_replacements += replacement_count

    def record_death(self, fitness: CreatureFitness | None) -> None:
        """Execute record death behavior.

Parameters
----------
fitness
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep record death behavior explicit in its owning subsystem.
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
        """Execute is reproduction eligible behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
fitness
    Input used by this creature-domain operation.
population_config
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep is reproduction eligible behavior explicit in its owning subsystem.
        if fitness.age_seconds < population_config.min_reproduction_age:
            return False
        if creature.energy < population_config.reproduction_energy_threshold:
            return False
        return (
            fitness.seconds_since_reproduction()
            >= population_config.reproduction_cooldown
        )

    def select_parent(
        self,
        eligible_pool: list[Creature],
        k1: int = 3,
        k2: int = 2,
    ) -> Creature | None:
        """Select one eligible parent by fitness, then network parsimony.
        
        Parameters
        ----------
        eligible_pool
            Input used by this creature-domain operation.
        k1
            Input used by this creature-domain operation.
        k2
            Input used by this creature-domain operation.
        Returns
        -------
        Creature | None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep select parent behavior explicit in its owning subsystem.
        if not eligible_pool:
            return None
        if type(k1) is not int or k1 <= 0:
            raise ValueError("k1 must be a positive integer.")
        if type(k2) is not int or k2 <= 0:
            raise ValueError("k2 must be a positive integer.")
        if k2 > k1:
            raise ValueError("k2 must not exceed k1.")

        if len(eligible_pool) <= k1:
            return min(
                eligible_pool,
                key=lambda creature: (
                    -self._selection_energy(creature),
                    self.network_complexity(creature),
                    creature.creature_id,
                ),
            )

        sampled = self.rng.sample(eligible_pool, k1)
        fitness_finalists = sorted(
            sampled,
            key=lambda creature: (
                -self._selection_energy(creature),
                creature.creature_id,
            ),
        )[:k2]
        return min(
            fitness_finalists,
            key=lambda creature: (
                self.network_complexity(creature),
                -self._selection_energy(creature),
                creature.creature_id,
            ),
        )

    def network_size(self, creature: Creature) -> tuple[int, int]:
        """Return node and enabled-connection counts for selection telemetry.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
tuple[int, int]
    Result produced by this creature-domain operation."""
        # Keep network size behavior explicit in its owning subsystem.
        if self.brain_controller is None:
            return 0, 0
        brain_for = getattr(self.brain_controller, "brain_for", None)
        if not callable(brain_for):
            return 0, 0
        brain = brain_for(creature.creature_id)
        genome = None if brain is None else getattr(brain, "genome", None)
        if genome is None:
            return 0, 0
        nodes = getattr(genome, "nodes", {}) or {}
        connections = getattr(genome, "connections", {}) or {}
        enabled_connections = sum(
            1
            for connection in connections.values()
            if bool(getattr(connection, "enabled", True))
        )
        return len(nodes), enabled_connections

    def network_complexity(self, creature: Creature) -> float:
        """Return parsimony complexity; missing genomes rank behind valid ones.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep network complexity behavior explicit in its owning subsystem.
        if self.brain_controller is None:
            return inf
        brain_for = getattr(self.brain_controller, "brain_for", None)
        if not callable(brain_for):
            return inf
        brain = brain_for(creature.creature_id)
        if brain is None or getattr(brain, "genome", None) is None:
            return inf
        nodes, enabled_connections = self.network_size(creature)
        return float(nodes + enabled_connections)

    @staticmethod
    def _selection_energy(creature: Creature) -> float:
        """Execute selection energy behavior.

Parameters
----------
creature
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep selection energy behavior explicit in its owning subsystem.
        try:
            gathered = float(creature.total_energy_gathered)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return 0.0
        return max(0.0, gathered) if isfinite(gathered) else 0.0

    def _update_brain_size_stats(self, creatures: list[Creature]) -> None:
        """Execute update brain size stats behavior.

Parameters
----------
creatures
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep update brain size stats behavior explicit in its owning subsystem.
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
        """Execute update event rates behavior.

Parameters
----------
elapsed_time
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep update event rates behavior explicit in its owning subsystem.
        elapsed_minutes = max(0.0, elapsed_time) / 60.0
        if elapsed_minutes <= 0.0:
            self.stats.births_per_minute = 0.0
            self.stats.deaths_per_minute = 0.0
            return

        self.stats.births_per_minute = self.stats.births / elapsed_minutes
        self.stats.deaths_per_minute = self.stats.deaths / elapsed_minutes
