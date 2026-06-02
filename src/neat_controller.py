from __future__ import annotations

from pathlib import Path
from typing import Any

import copy

import neat

from src.action import ACTION_OUTPUT_COUNT, Action
from src.neat_brain import NeatBrain
from src.vision import SENSOR_INPUT_COUNT, SensorSnapshot

FALLBACK_ACTION = Action(
    accelerate=0.0,
    rotate=0.0,
    want_reproduce=0.0,
    want_eat=0.0,
    reset_chronometer=0.0,
    want_grab=0.0,
    want_release=0.0,
)


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
        self._validate_network_contract()
        self.population = neat.Population(self.config)
        self.brains: dict[int, NeatBrain] = {}

    def assign_initial_brains(self, creature_ids: list[int]) -> None:
        self._validate_creature_ids(creature_ids)
        genomes = self._initial_genomes()

        if len(genomes) < len(creature_ids):
            raise ValueError(
                f"Not enough genomes in the population to assign to all creatures. "
                f"Genomes: {len(genomes)}, Creatures: {len(creature_ids)}"
            )

        for creature_id, (genome_id, genome) in zip(creature_ids, genomes):
            self.brains[creature_id] = NeatBrain.from_genome(
                genome_id,
                genome,
                self.config,
            )

    def decide(self, creature_id: int, snapshot: SensorSnapshot) -> Action:
        brain = self.brains.get(creature_id)
        if brain is None:
            return self.fallback_action()

        return brain.decide(snapshot)

    def fallback_action(self) -> Action:
        return Action(
            accelerate=FALLBACK_ACTION.accelerate,
            rotate=FALLBACK_ACTION.rotate,
            want_reproduce=FALLBACK_ACTION.want_reproduce,
            want_eat=FALLBACK_ACTION.want_eat,
            reset_chronometer=FALLBACK_ACTION.reset_chronometer,
            want_grab=FALLBACK_ACTION.want_grab,
            want_release=FALLBACK_ACTION.want_release,
        )

    def remove_brain(self, creature_id: int) -> None:
        self.brains.pop(creature_id, None)

    def archive_brain(self, creature_id: int, fitness_score: float) -> bool:
        brain = self.brains.get(creature_id)
        if brain is None:
            return False

        brain.genome.fitness = fitness_score
        self.population.population[brain.genome_id] = brain.genome
        return True

    def genome_id_for(self, creature_id: int) -> int | None:
        brain = self.brains.get(creature_id)
        if brain is None:
            return None
        return brain.genome_id

    def brain_for(self, creature_id: int) -> NeatBrain | None:
        return self.brains.get(creature_id)

    def create_child_brain(self, parent_creature_id: int, child_creature_id: int) -> bool:
        parent_brain = self.brains.get(parent_creature_id)
        if parent_brain is None:
            return False

        return self.create_mutated_brain_from_genome(
            parent_brain.genome,
            child_creature_id,
        )

    def create_mutated_brain_from_genome(
        self,
        parent_genome: Any,
        creature_id: int,
    ) -> bool:
        child_genome = copy.deepcopy(parent_genome)
        child_genome.key = self._next_genome_id()
        child_genome.fitness = None
        child_genome.mutate(self.config.genome_config)
        self.population.population[child_genome.key] = child_genome

        self.brains[creature_id] = NeatBrain.from_genome(
            child_genome.key,
            child_genome,
            self.config,
        )
        return True

    def best_genomes(self, count: int) -> list[Any]:
        scored_genomes = [
            genome
            for genome in self.population.population.values()
            if genome.fitness is not None
        ]
        return sorted(
            scored_genomes,
            key=lambda genome: genome.fitness,
            reverse=True,
        )[:count]
    
    def _next_genome_id(self) -> int:
        genome_ids = [
            brain.genome_id
            for brain in self.brains.values()
        ]
        population_ids = list(self.population.population.keys())
        return max([0, *genome_ids, *population_ids]) + 1

    def _initial_genomes(self) -> list[tuple[int, Any]]:
        return list(self.population.population.items())

    def _validate_creature_ids(self, creature_ids: list[int]) -> None:
        if len(set(creature_ids)) != len(creature_ids):
            raise ValueError("Cannot assign NEAT brains to duplicate creature ids.")

    def _validate_network_contract(self) -> None:
        genome_config = self.config.genome_config
        input_count = len(genome_config.input_keys)
        output_count = len(genome_config.output_keys)

        if input_count != SENSOR_INPUT_COUNT:
            raise ValueError(
                f"NEAT config input count mismatch. "
                f"Config: {input_count}, code: {SENSOR_INPUT_COUNT}"
            )

        if output_count != ACTION_OUTPUT_COUNT:
            raise ValueError(
                f"NEAT config output count mismatch. "
                f"Config: {output_count}, code: {ACTION_OUTPUT_COUNT}"
            )
