from __future__ import annotations

from pathlib import Path

import copy

import neat

from src.action import Action
from src.neat_brain import NeatBrain
from src.vision import SensorSnapshot


class NeatBrainController:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path),
        )
        self.population = neat.Population(self.config)
        self.brains: dict[int, NeatBrain] = {}

    def assign_initial_brains(self, creature_ids: list[int]) -> None:
        # Get the list of genome IDs from the input genomes
        genomes = list(self.population.population.items())

        if len(genomes) < len(creature_ids):
            raise ValueError(
                f"Not enough genomes in the population to assign to all creatures. "
                f"Genomes: {len(genomes)}, Creatures: {len(creature_ids)}"
            )
        # Assign brains to creatures based on the genome IDs
        for creature_id, (genome_id, genome) in zip(creature_ids, genomes):
            self.brains[creature_id] = NeatBrain.from_genome(
                genome_id,
                genome,
                self.config,
            )

    def decide(self, creature_id: int, snapshot: SensorSnapshot) -> Action:
        # Get the brain for the given creature ID
        brain = self.brains.get(creature_id)

        # If no brain is assigned to this creature, return a default action.
        if brain is None:
            return Action(accelerate=0.0, rotate=0.0, herding=0.0)

        return brain.decide(snapshot)

    def remove_brain(self, creature_id: int) -> None:
        self.brains.pop(creature_id, None)
    
    def create_child_brain(self, parent_creature_id: int, child_creature_id:int) -> bool:
        # Retrieve the parent brain
        parent_brain = self.brains.get(parent_creature_id)
        if parent_brain is None: # No parent brain found, cannot create child brain
            return False
        
        # Create a child genome by mutating the parent's genome
        child_genome = copy.deepcopy(parent_brain.genome)
        # Assign a new unique genome ID to the child genome and reset its fitness
        child_genome.key = self._next_genome_id()
        # Reset the fitness of the child genome to None (or 0.0) before mutation
        child_genome.fitness = None
        # Mutate the child genome using the NEAT configuration
        child_genome.mutate(self.config.genome_config)

        self.brains[child_creature_id] = NeatBrain.from_genome(
            child_genome.key,
            child_genome,
            self.config,
        )
        return True
    
    def _next_genome_id(self) -> int:
        genome_ids = [
            brain.genome_id
            for brain in self.brains.values()
        ]
        population_ids = list(self.population.population.keys())
        return max([0, *genome_ids, *population_ids]) + 1
        



        
