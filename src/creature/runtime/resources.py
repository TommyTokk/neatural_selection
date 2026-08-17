"""Ownership of metabolism, fitness, carrying, and resource caches."""

from __future__ import annotations

from src.creature.fitness import CreatureFitness
from src.creature.metabolism import Metabolism


class CreatureResourceService:
    """Own creature resource systems and per-creature resource state."""

    def __init__(self, metabolism: Metabolism) -> None:
        """Initialize resource state around a configured metabolism system.

Parameters
----------
metabolism
    Configured metabolism calculator.

Returns
-------
None
    Empty resource registries are created."""
        # Keep init behavior explicit in its owning subsystem.
        # Fitness and carrying state now share explicit resource ownership.
        self.metabolism = metabolism
        self.fitness: dict[int, CreatureFitness] = {}
        self.held_food_by_creature_id: dict[int, int] = {}
        self.carrier_by_food_id: dict[int, int] = {}
        self.mouth_exposures: object | None = None
        self.chronometers: dict[int, float] = {}
        self.last_digestion_processing_costs_per_second: dict[int, float] = {}

    def initialize(self, creature_id: int) -> None:
        """Install default resource state for a newly live creature.

Parameters
----------
creature_id
    Stable new creature identity.

Returns
-------
None
    A fresh fitness record is cached."""
        # Keep initialize behavior explicit in its owning subsystem.
        # Every live creature receives exactly one current fitness ledger.
        self.fitness[creature_id] = CreatureFitness()

    def discard(self, creature_id: int) -> CreatureFitness | None:
        """Discard resource state and return the final fitness record.

Parameters
----------
creature_id
    Stable identity leaving the live population.

Returns
-------
CreatureFitness | None
    Removed fitness record when one existed."""
        # Keep discard behavior explicit in its owning subsystem.
        # Carry mappings are cleared symmetrically before fitness is returned.
        food_id = self.held_food_by_creature_id.pop(creature_id, None)
        if food_id is not None:
            self.carrier_by_food_id.pop(food_id, None)
        return self.fitness.pop(creature_id, None)
