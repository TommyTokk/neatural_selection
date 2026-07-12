from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

import copy

import neat

from configs.sim_config import TraitConfig, VisionConfig
from src.action import ACTION_OUTPUT_COUNT, Action
from src.creature import Creature, PhysicalTraits, VisionTraits
from src.neat_brain import NeatBrain
from src.speciation import (
    NeatChangeSummary,
    SpeciesDistanceBreakdown,
    SpeciesTraitSnapshot,
    NeuralShift,
    extract_neural_shifts,
)
from src.vision import SENSOR_INPUT_COUNT, SensorSnapshot

FALLBACK_ACTION = Action(
    accelerate=0.0,
    rotate=0.0,
    want_reproduce=0.0,
    want_eat=0.0,
    reset_chronometer=0.0,
    want_grab=0.0,
    want_release=0.0,
    want_nurse=0.0,
    flee_panic_intensity=0.0,
    weight_separation=0.0,
    weight_alignment=0.0,
    weight_cohesion=0.0,
)


SpeciesRepresentative = tuple[Any, PhysicalTraits, VisionTraits]


def _normalized_trait_difference(
    first: float,
    second: float,
    minimum: float,
    maximum: float,
) -> float:
    if maximum <= minimum:
        raise ValueError("Phenotypic trait ranges must have a positive width.")
    clamped_first = max(minimum, min(maximum, first))
    clamped_second = max(minimum, min(maximum, second))
    return abs(clamped_first - clamped_second) / (maximum - minimum)


def calculate_phenotypic_distance(
    child_physical_traits: PhysicalTraits,
    child_vision: VisionTraits,
    representative_physical_traits: PhysicalTraits,
    representative_vision: VisionTraits,
    trait_config: TraitConfig,
    vision_config: VisionConfig,
) -> float:
    components = calculate_phenotypic_distance_components(
        child_physical_traits,
        child_vision,
        representative_physical_traits,
        representative_vision,
        trait_config,
        vision_config,
    )
    return (
        components.radius
        + components.vision_range
        + components.vision_angle
        + components.movement_cost_multiplier
    )


def calculate_phenotypic_distance_components(
    child_physical_traits: PhysicalTraits,
    child_vision: VisionTraits,
    representative_physical_traits: PhysicalTraits,
    representative_vision: VisionTraits,
    trait_config: TraitConfig,
    vision_config: VisionConfig,
) -> SpeciesTraitSnapshot:
    return SpeciesTraitSnapshot(
        radius=_normalized_trait_difference(
            child_physical_traits.radius,
            representative_physical_traits.radius,
            trait_config.min_radius,
            trait_config.max_radius,
        ),
        vision_range=_normalized_trait_difference(
            child_vision.range,
            representative_vision.range,
            vision_config.min_range,
            vision_config.max_range,
        ),
        vision_angle=_normalized_trait_difference(
            child_vision.angle,
            representative_vision.angle,
            vision_config.min_angle,
            vision_config.max_angle,
        ),
        movement_cost_multiplier=_normalized_trait_difference(
            child_physical_traits.movement_cost_multiplier,
            representative_physical_traits.movement_cost_multiplier,
            trait_config.min_movement_cost_multiplier,
            trait_config.max_movement_cost_multiplier,
        ),
    )


@dataclass(frozen=True, slots=True)
class SpeciationResult:
    species_id: int
    parent_species_id: int
    is_new_species: bool
    founder_traits: SpeciesTraitSnapshot
    trait_deltas: SpeciesTraitSnapshot
    distances: SpeciesDistanceBreakdown
    neat_changes: NeatChangeSummary | None = None
    neural_shifts: tuple[NeuralShift, ...] = ()


class ContinuousSpeciesManager:
    def __init__(
        self,
        compatibility_threshold: float,
        phenotypic_weight: float = 2.0,
        trait_config: TraitConfig | None = None,
        vision_config: VisionConfig | None = None,
    ) -> None:
        self.compatibility_threshold = compatibility_threshold
        self.phenotypic_weight = phenotypic_weight
        self.trait_config = trait_config or TraitConfig()
        self.vision_config = vision_config or VisionConfig()
        self.representatives: dict[int, SpeciesRepresentative] = {}
        self.next_species_id = 2

    def register_initial_representative(
        self,
        genome: neat.DefaultGenome,
        physical_traits: PhysicalTraits,
        vision: VisionTraits,
    ) -> None:
        self.representatives.setdefault(
            1,
            (
                genome,
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
            ),
        )

    def evaluate_species(
        self,
        child_genome: neat.DefaultGenome,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        parent_species_id: int,
        genome_config: Any,
    ) -> SpeciationResult:
        (
            representative_genome,
            representative_physical_traits,
            representative_vision,
        ) = self.representatives[parent_species_id]
        neat_distance = child_genome.distance(
            representative_genome,
            genome_config,
        )
        phenotype_components = calculate_phenotypic_distance_components(
            child_physical_traits,
            child_vision,
            representative_physical_traits,
            representative_vision,
            self.trait_config,
            self.vision_config,
        )
        phenotypic_distance = (
            phenotype_components.radius
            + phenotype_components.vision_range
            + phenotype_components.vision_angle
            + phenotype_components.movement_cost_multiplier
        )
        weighted_phenotypic_distance = (
            self.phenotypic_weight * phenotypic_distance
        )
        composite_distance = (
            neat_distance + weighted_phenotypic_distance
        )
        trait_deltas = SpeciesTraitSnapshot(
            radius=(
                child_physical_traits.radius
                - representative_physical_traits.radius
            ),
            vision_range=child_vision.range - representative_vision.range,
            vision_angle=child_vision.angle - representative_vision.angle,
            movement_cost_multiplier=(
                child_physical_traits.movement_cost_multiplier
                - representative_physical_traits.movement_cost_multiplier
            ),
        )
        distances = SpeciesDistanceBreakdown(
            neat_distance=neat_distance,
            phenotypic_distance=phenotypic_distance,
            weighted_phenotypic_distance=weighted_phenotypic_distance,
            composite_distance=composite_distance,
            compatibility_threshold=self.compatibility_threshold,
            phenotypic_weight=self.phenotypic_weight,
            radius_component=phenotype_components.radius,
            vision_range_component=phenotype_components.vision_range,
            vision_angle_component=phenotype_components.vision_angle,
            movement_cost_component=(
                phenotype_components.movement_cost_multiplier
            ),
        )
        if composite_distance > self.compatibility_threshold:
            neural_shifts = extract_neural_shifts(
                representative_genome,
                child_genome,
            )
            new_species_id = self.next_species_id
            self.representatives[new_species_id] = (
                child_genome,
                copy.deepcopy(child_physical_traits),
                copy.deepcopy(child_vision),
            )
            self.next_species_id += 1
            return SpeciationResult(
                species_id=new_species_id,
                parent_species_id=parent_species_id,
                is_new_species=True,
                founder_traits=SpeciesTraitSnapshot.from_traits(
                    child_physical_traits,
                    child_vision,
                ),
                trait_deltas=trait_deltas,
                distances=distances,
                neural_shifts=neural_shifts,
            )

        return SpeciationResult(
            species_id=parent_species_id,
            parent_species_id=parent_species_id,
            is_new_species=False,
            founder_traits=SpeciesTraitSnapshot.from_traits(
                child_physical_traits,
                child_vision,
            ),
            trait_deltas=trait_deltas,
            distances=distances,
        )


class NeatBrainController:
    """
    Manages NEAT brains for a population of creatures, including creation,
    mutation, speciation, and decision-making based on sensor snapshots.
    """
    def __init__(
        self,
        config_path: str | Path,
        compatibility_threshold: float = 3.0,
        phenotypic_weight: float = 2.0,
        trait_config: TraitConfig | None = None,
        vision_config: VisionConfig | None = None,
    ) -> None:
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
        self._next_genome_id_value = (
            max(self.population.population, default=0) + 1
        )
        self.brains: dict[int, NeatBrain] = {}
        self.species_manager = ContinuousSpeciesManager(
            compatibility_threshold,
            phenotypic_weight,
            trait_config,
            vision_config,
        )

    def assign_initial_brains(self, creatures: list[Creature]) -> None:
        creature_ids = [creature.creature_id for creature in creatures]
        self._validate_creature_ids(creature_ids)
        genomes = self._initial_genomes()
        if genomes and creatures:
            first_creature = creatures[0]
            self.species_manager.register_initial_representative(
                genomes[0][1],
                first_creature.physical_traits,
                first_creature.vision,
            )

        if len(genomes) < len(creature_ids):
            raise ValueError(
                f"Not enough genomes in the population to assign to all creatures. "
                f"Genomes: {len(genomes)}, Creatures: {len(creature_ids)}"
            )

        for creature, (genome_id, genome) in zip(creatures, genomes):
            self.brains[creature.creature_id] = NeatBrain.from_genome(
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
            want_nurse=FALLBACK_ACTION.want_nurse,
            flee_panic_intensity=FALLBACK_ACTION.flee_panic_intensity,
            weight_separation=FALLBACK_ACTION.weight_separation,
            weight_alignment=FALLBACK_ACTION.weight_alignment,
            weight_cohesion=FALLBACK_ACTION.weight_cohesion,
        )

    def remove_brain(self, creature_id: int) -> None:
        self.brains.pop(creature_id, None)

    def prune_population_archive(self, archive_size: int) -> set[int]:
        live_genome_ids = {
            brain.genome_id
            for brain in self.brains.values()
        }
        scored_dead_genomes = [
            genome
            for genome_id, genome in self.population.population.items()
            if genome_id not in live_genome_ids
            and genome.fitness is not None
        ]
        scored_dead_genomes.sort(
            key=lambda genome: (
                float(genome.fitness),
                int(genome.key),
            ),
            reverse=True,
        )
        retained_genome_ids = live_genome_ids | {
            genome.key
            for genome in scored_dead_genomes[:max(0, archive_size)]
        }
        self.population.population = {
            genome_id: genome
            for genome_id, genome in self.population.population.items()
            if genome_id in retained_genome_ids
        }
        return retained_genome_ids

    def prune_species_representatives(
        self,
        retained_species_ids: set[int],
    ) -> None:
        protected_species_ids = {1, *retained_species_ids}
        self.species_manager.representatives = {
            species_id: representative
            for species_id, representative in (
                self.species_manager.representatives.items()
            )
            if species_id in protected_species_ids
        }

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

    def restore_brain(self, creature_id: int, genome_id: int) -> NeatBrain:
        genome = self.population.population.get(genome_id)
        if genome is None:
            raise ValueError(
                f"Cannot restore creature {creature_id}: "
                f"genome {genome_id} is missing from the population."
            )
        brain = NeatBrain.from_genome(genome_id, genome, self.config)
        self.brains[creature_id] = brain
        return brain

    def migrate_legacy_brain_contract(self) -> None:
        """Bring saved genomes forward to the current brain contract.

        New input keys need no genome nodes and therefore remain unconnected
        until mutation evolves a connection. Missing action outputs do require
        explicit, inert nodes.
        """
        genomes = self._known_genomes()
        seen: set[int] = set()
        for genome in genomes:
            identity = id(genome)
            if identity in seen:
                continue
            seen.add(identity)
            self._ensure_current_output_nodes(genome)

    def _ensure_current_output_nodes(self, genome: Any) -> None:
        genome_config = self.config.genome_config
        nodes = getattr(genome, "nodes", None)
        if nodes is None:
            return

        for output_key in genome_config.output_keys:
            if output_key in nodes:
                continue
            node = genome_config.node_gene_type(output_key)
            node.init_attributes(genome_config)
            if hasattr(node, "bias"):
                node.bias = float(getattr(genome_config, "bias_min_value", -5.0))
            nodes[output_key] = node

    def evolution_allocator_state(self) -> dict[str, int]:
        """Return allocator positions needed to continue mutating after a load."""
        genome_config = self.config.genome_config
        minimum_next_node_id = self._minimum_next_node_id()
        next_node_id = minimum_next_node_id
        node_indexer = getattr(genome_config, "node_indexer", None)
        if node_indexer is not None:
            try:
                reduce_args = node_indexer.__reduce__()[1]
                next_node_id = max(next_node_id, int(reduce_args[0]))
            except (AttributeError, IndexError, TypeError, ValueError):
                pass

        innovation_number = self._minimum_innovation_number()
        innovation_tracker = getattr(genome_config, "innovation_tracker", None)
        if innovation_tracker is not None:
            innovation_number = max(
                innovation_number,
                int(getattr(innovation_tracker, "global_counter", 0)),
            )

        return {
            "next_node_id": next_node_id,
            "innovation_number": innovation_number,
        }

    def restore_evolution_allocators(
        self,
        next_node_id: int | None = None,
        innovation_number: int | None = None,
    ) -> None:
        """Restore or reconstruct NEAT's process-local structural allocators."""
        genome_config = self.config.genome_config
        minimum_next_node_id = self._minimum_next_node_id()
        restored_next_node_id = max(
            minimum_next_node_id,
            minimum_next_node_id if next_node_id is None else int(next_node_id),
        )
        genome_config.node_indexer = count(restored_next_node_id)

        innovation_tracker = getattr(genome_config, "innovation_tracker", None)
        if innovation_tracker is None:
            return

        restored_innovation_number = max(
            self._minimum_innovation_number(),
            int(getattr(innovation_tracker, "global_counter", 0)),
            0 if innovation_number is None else int(innovation_number),
        )
        innovation_tracker.global_counter = restored_innovation_number
        generation_innovations = getattr(
            innovation_tracker,
            "generation_innovations",
            None,
        )
        if generation_innovations is not None:
            generation_innovations.clear()

    def _known_genomes(self) -> list[Any]:
        genomes = list(self.population.population.values())
        genomes.extend(
            representative[0]
            if isinstance(representative, tuple)
            else representative
            for representative in self.species_manager.representatives.values()
        )
        return genomes

    def _minimum_next_node_id(self) -> int:
        genome_config = self.config.genome_config
        node_ids = [
            int(node_id)
            for genome in self._known_genomes()
            for node_id in (getattr(genome, "nodes", {}) or {})
            if int(node_id) >= 0
        ]
        return max(
            [
                int(getattr(genome_config, "num_outputs", 0)) - 1,
                *(int(key) for key in getattr(genome_config, "output_keys", ())),
                *node_ids,
            ],
            default=-1,
        ) + 1

    def _minimum_innovation_number(self) -> int:
        return max(
            (
                int(innovation)
                for genome in self._known_genomes()
                for connection in (
                    getattr(genome, "connections", {}) or {}
                ).values()
                if (innovation := getattr(connection, "innovation", None))
                is not None
            ),
            default=0,
        )

    def create_child_brain(
        self,
        parent_creature_id: int,
        child_creature_id: int,
        parent_species_id: int,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
    ) -> tuple[NeatBrain | None, SpeciationResult | None]:
        parent_brain = self.brains.get(parent_creature_id)
        if parent_brain is None:
            return None, None

        return self.create_mutated_brain_from_genome(
            parent_brain.genome,
            child_creature_id,
            parent_species_id,
            child_physical_traits,
            child_vision,
        )

    def create_mutated_brain_from_genome(
        self,
        parent_genome: Any,
        creature_id: int,
        parent_species_id: int,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
    ) -> tuple[NeatBrain, SpeciationResult]:
        child_genome = copy.deepcopy(parent_genome)
        child_genome.key = self._next_genome_id()
        child_genome.fitness = None
        child_genome.mutate(self.config.genome_config)
        speciation_result = self.species_manager.evaluate_species(
            child_genome,
            child_physical_traits,
            child_vision,
            parent_species_id,
            self.config.genome_config,
        )
        self.population.population[child_genome.key] = child_genome

        child_brain = NeatBrain.from_genome(
            child_genome.key,
            child_genome,
            self.config,
        )
        self.brains[creature_id] = child_brain
        return child_brain, speciation_result

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
        if not hasattr(self, "_next_genome_id_value"):
            representative_ids = [
                getattr(representative[0], "key", 0)
                for representative in self.species_manager.representatives.values()
                if isinstance(representative, tuple)
            ]
            self._next_genome_id_value = max(
                [
                    0,
                    *self.population.population,
                    *(brain.genome_id for brain in self.brains.values()),
                    *representative_ids,
                ]
            ) + 1
        genome_id = self._next_genome_id_value
        self._next_genome_id_value += 1
        return genome_id

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
