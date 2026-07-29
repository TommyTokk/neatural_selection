from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count
from pathlib import Path
import random
from typing import Any

import copy

import neat

from configs.sim_config import TraitConfig, VisionConfig
from src.action import ACTION_OUTPUT_COUNT, Action
from src.creature import Creature, FlockingTraits, PhysicalTraits, VisionTraits
from src.neat_brain import NeatBrain
from src.speciation import (
    NeatChangeSummary,
    SpeciesDistanceBreakdown,
    SpeciesTraitSnapshot,
    NeuralShift,
    extract_neural_shifts,
)
from src.vision import (
    SENSOR_CONTRACT,
    SensorContract,
    SensorSnapshot,
)

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
    herding=0.0,
)


SpeciesRepresentative = tuple[Any, PhysicalTraits, VisionTraits, FlockingTraits]


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
    digestive_trait_component = (
        components.stomach_capacity
        + components.digestion_rate
        + components.digestion_efficiency
    ) / 3.0
    return (
        components.radius
        + components.vision_range
        + components.vision_angle
        + components.movement_cost_multiplier
        + digestive_trait_component
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
        stomach_capacity=_normalized_trait_difference(
            getattr(
                child_physical_traits,
                "stomach_capacity",
                trait_config.default_stomach_capacity,
            ),
            getattr(
                representative_physical_traits,
                "stomach_capacity",
                trait_config.default_stomach_capacity,
            ),
            trait_config.min_stomach_capacity,
            trait_config.max_stomach_capacity,
        ),
        digestion_rate=_normalized_trait_difference(
            getattr(
                child_physical_traits,
                "digestion_rate",
                trait_config.default_digestion_rate,
            ),
            getattr(
                representative_physical_traits,
                "digestion_rate",
                trait_config.default_digestion_rate,
            ),
            trait_config.min_digestion_rate,
            trait_config.max_digestion_rate,
        ),
        digestion_efficiency=_normalized_trait_difference(
            getattr(
                child_physical_traits,
                "digestion_efficiency",
                trait_config.default_digestion_efficiency,
            ),
            getattr(
                representative_physical_traits,
                "digestion_efficiency",
                trait_config.default_digestion_efficiency,
            ),
            trait_config.min_digestion_efficiency,
            trait_config.max_digestion_efficiency,
        ),
    )


def calculate_flocking_trait_distance(
    first: FlockingTraits,
    second: FlockingTraits,
) -> tuple[float, float, float, float]:
    """Return mean and per-gene bounded flocking-trait distances."""
    separation = abs(first.separation_gene - second.separation_gene)
    alignment = abs(first.alignment_gene - second.alignment_gene)
    cohesion = abs(first.cohesion_gene - second.cohesion_gene)
    return (separation + alignment + cohesion) / 3.0, separation, alignment, cohesion


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


@dataclass(frozen=True, slots=True)
class CompositeCompatibilityDistance:
    neat_distance: float
    phenotype_components: SpeciesTraitSnapshot
    phenotypic_distance: float
    weighted_phenotypic_distance: float
    flocking_trait_distance: float
    weighted_flocking_trait_distance: float
    separation_gene_component: float
    alignment_gene_component: float
    cohesion_gene_component: float
    composite_distance: float


class ContinuousSpeciesManager:
    def __init__(
        self,
        compatibility_threshold: float,
        phenotypic_weight: float = 2.0,
        trait_config: TraitConfig | None = None,
        vision_config: VisionConfig | None = None,
        flocking_trait_distance_coefficient: float = 1.0,
    ) -> None:
        self.compatibility_threshold = compatibility_threshold
        self.phenotypic_weight = phenotypic_weight
        self.trait_config = trait_config or TraitConfig()
        self.vision_config = vision_config or VisionConfig()
        self.flocking_trait_distance_coefficient = max(
            0.0,
            float(flocking_trait_distance_coefficient),
        )
        self.representatives: dict[int, SpeciesRepresentative] = {}
        self.next_species_id = 2

    def register_initial_representative(
        self,
        genome: neat.DefaultGenome,
        physical_traits: PhysicalTraits,
        vision: VisionTraits,
        species_id: int = 1,
        flocking_traits: FlockingTraits | None = None,
    ) -> None:
        self.representatives.setdefault(
            species_id,
            (
                genome,
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
                copy.deepcopy(flocking_traits or FlockingTraits()),
            ),
        )

    def evaluate_species(
        self,
        child_genome: neat.DefaultGenome,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        parent_species_id: int,
        genome_config: Any,
        child_flocking_traits: FlockingTraits | None = None,
    ) -> SpeciationResult:
        (
            representative_genome,
            representative_physical_traits,
            representative_vision,
            representative_flocking_traits,
        ) = self.representatives[parent_species_id]
        child_flocking_traits = child_flocking_traits or FlockingTraits()
        compatibility = self.composite_distance(
            child_genome,
            child_physical_traits,
            child_vision,
            child_flocking_traits,
            representative_genome,
            representative_physical_traits,
            representative_vision,
            representative_flocking_traits,
            genome_config,
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
            separation_gene=(
                child_flocking_traits.separation_gene
                - representative_flocking_traits.separation_gene
            ),
            alignment_gene=(
                child_flocking_traits.alignment_gene
                - representative_flocking_traits.alignment_gene
            ),
            cohesion_gene=(
                child_flocking_traits.cohesion_gene
                - representative_flocking_traits.cohesion_gene
            ),
            stomach_capacity=(
                child_physical_traits.stomach_capacity
                - representative_physical_traits.stomach_capacity
            ),
            digestion_rate=(
                child_physical_traits.digestion_rate
                - representative_physical_traits.digestion_rate
            ),
            digestion_efficiency=(
                child_physical_traits.digestion_efficiency
                - representative_physical_traits.digestion_efficiency
            ),
        )
        digestive_trait_component = (
            compatibility.phenotype_components.stomach_capacity
            + compatibility.phenotype_components.digestion_rate
            + compatibility.phenotype_components.digestion_efficiency
        ) / 3.0
        distances = SpeciesDistanceBreakdown(
            neat_distance=compatibility.neat_distance,
            phenotypic_distance=compatibility.phenotypic_distance,
            weighted_phenotypic_distance=(
                compatibility.weighted_phenotypic_distance
            ),
            composite_distance=compatibility.composite_distance,
            compatibility_threshold=self.compatibility_threshold,
            phenotypic_weight=self.phenotypic_weight,
            radius_component=compatibility.phenotype_components.radius,
            vision_range_component=(
                compatibility.phenotype_components.vision_range
            ),
            vision_angle_component=(
                compatibility.phenotype_components.vision_angle
            ),
            movement_cost_component=(
                compatibility.phenotype_components.movement_cost_multiplier
            ),
            flocking_trait_distance=compatibility.flocking_trait_distance,
            weighted_flocking_trait_distance=(
                compatibility.weighted_flocking_trait_distance
            ),
            flocking_trait_distance_coefficient=(
                self.flocking_trait_distance_coefficient
            ),
            separation_gene_component=compatibility.separation_gene_component,
            alignment_gene_component=compatibility.alignment_gene_component,
            cohesion_gene_component=compatibility.cohesion_gene_component,
            stomach_capacity_component=(
                compatibility.phenotype_components.stomach_capacity
            ),
            digestion_rate_component=(
                compatibility.phenotype_components.digestion_rate
            ),
            digestion_efficiency_component=(
                compatibility.phenotype_components.digestion_efficiency
            ),
            digestive_trait_component=digestive_trait_component,
        )
        if compatibility.composite_distance > self.compatibility_threshold:
            neural_shifts = extract_neural_shifts(
                representative_genome,
                child_genome,
            )
            new_species_id = self.next_species_id
            self.representatives[new_species_id] = (
                child_genome,
                copy.deepcopy(child_physical_traits),
                copy.deepcopy(child_vision),
                copy.deepcopy(child_flocking_traits),
            )
            self.next_species_id += 1
            return SpeciationResult(
                species_id=new_species_id,
                parent_species_id=parent_species_id,
                is_new_species=True,
                founder_traits=SpeciesTraitSnapshot.from_traits(
                    child_physical_traits,
                    child_vision,
                    child_flocking_traits,
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
                child_flocking_traits,
            ),
            trait_deltas=trait_deltas,
            distances=distances,
        )

    def composite_distance(
        self,
        first_genome: Any,
        first_physical_traits: PhysicalTraits,
        first_vision: VisionTraits,
        first_flocking_traits: FlockingTraits,
        second_genome: Any,
        second_physical_traits: PhysicalTraits,
        second_vision: VisionTraits,
        second_flocking_traits: FlockingTraits,
        genome_config: Any,
    ) -> CompositeCompatibilityDistance:
        """Return the same composite distance used by live and birth speciation."""
        neat_distance = first_genome.distance(second_genome, genome_config)
        phenotype_components = calculate_phenotypic_distance_components(
            first_physical_traits,
            first_vision,
            second_physical_traits,
            second_vision,
            self.trait_config,
            self.vision_config,
        )
        digestive_trait_component = (
            phenotype_components.stomach_capacity
            + phenotype_components.digestion_rate
            + phenotype_components.digestion_efficiency
        ) / 3.0
        phenotypic_distance = (
            phenotype_components.radius
            + phenotype_components.vision_range
            + phenotype_components.vision_angle
            + phenotype_components.movement_cost_multiplier
            + digestive_trait_component
        )
        weighted_phenotypic_distance = (
            self.phenotypic_weight * phenotypic_distance
        )
        (
            flocking_trait_distance,
            separation_gene_component,
            alignment_gene_component,
            cohesion_gene_component,
        ) = calculate_flocking_trait_distance(
            first_flocking_traits,
            second_flocking_traits,
        )
        weighted_flocking_trait_distance = (
            self.flocking_trait_distance_coefficient
            * flocking_trait_distance
        )
        composite_distance = (
            neat_distance
            + weighted_phenotypic_distance
            + weighted_flocking_trait_distance
        )
        return CompositeCompatibilityDistance(
            neat_distance=neat_distance,
            phenotype_components=phenotype_components,
            phenotypic_distance=phenotypic_distance,
            weighted_phenotypic_distance=weighted_phenotypic_distance,
            flocking_trait_distance=flocking_trait_distance,
            weighted_flocking_trait_distance=weighted_flocking_trait_distance,
            separation_gene_component=separation_gene_component,
            alignment_gene_component=alignment_gene_component,
            cohesion_gene_component=cohesion_gene_component,
            composite_distance=composite_distance,
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
        flocking_trait_distance_coefficient: float = 1.0,
        sensor_contract: SensorContract | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._evolution_rng = random.Random(random_seed)
        self.config_path = Path(config_path)
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path),
        )
        self.sensor_contract = sensor_contract or SENSOR_CONTRACT
        self._validate_network_contract()
        with self._using_evolution_rng():
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
            flocking_trait_distance_coefficient,
        )
        self._pairwise_compatibility_distance_cache: dict[
            tuple[int, int], float
        ] = {}

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
                flocking_traits=getattr(
                    first_creature,
                    "flocking_traits",
                    FlockingTraits(),
                ),
            )

        if len(genomes) < len(creature_ids):
            raise ValueError(
                f"Not enough genomes in the population to assign to all creatures. "
                f"Genomes: {len(genomes)}, Creatures: {len(creature_ids)}"
            )

        for creature, (genome_id, genome) in zip(creatures, genomes):
            self.brains[creature.creature_id] = self._brain_from_genome(
                genome_id,
                genome,
            )

    def reset_for_new_sensing_epoch(
        self,
        creatures: list[Creature],
        root_species_id: int,
    ) -> None:
        """Replace all neural evolution state with current-contract genomes."""
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            str(self.config_path),
        )
        self._validate_network_contract()
        with self._using_evolution_rng():
            self.population = neat.Population(self.config)
        self.brains = {}
        self._pairwise_compatibility_distance_cache.clear()
        self.species_manager.representatives = {}
        self.species_manager.next_species_id = root_species_id + 1
        self._next_genome_id_value = (
            max(self.population.population, default=0) + 1
        )

        genomes = list(self.population.population.items())
        if len(genomes) < len(creatures):
            raise ValueError(
                "Not enough fresh genomes for sensing epoch reset. "
                f"Genomes: {len(genomes)}, creatures: {len(creatures)}"
            )
        if creatures:
            first_genome = genomes[0][1]
            first_creature = creatures[0]
            self.species_manager.register_initial_representative(
                first_genome,
                first_creature.physical_traits,
                first_creature.vision,
                species_id=root_species_id,
                flocking_traits=getattr(
                    first_creature,
                    "flocking_traits",
                    FlockingTraits(),
                ),
            )

        for creature, (genome_id, genome) in zip(creatures, genomes):
            self.brains[creature.creature_id] = self._brain_from_genome(
                genome_id,
                genome,
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
            herding=FALLBACK_ACTION.herding,
        )

    def remove_brain(self, creature_id: int) -> None:
        brain = self.brains.pop(creature_id, None)
        if brain is not None:
            self._discard_cached_genome(brain.genome_id)

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
        pairwise_cache = getattr(self, "_pairwise_compatibility_distance_cache", {})
        self._pairwise_compatibility_distance_cache = {
            pair: distance
            for pair, distance in pairwise_cache.items()
            if pair[0] in retained_genome_ids and pair[1] in retained_genome_ids
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

    def flocking_compatibility(
        self,
        first: Creature,
        second: Creature,
    ) -> float:
        """Return continuous live compatibility using speciation's distance."""
        first_brain = self.brains.get(first.creature_id)
        second_brain = self.brains.get(second.creature_id)
        if first_brain is None or second_brain is None:
            return self._binary_species_compatibility(first, second)

        pair = tuple(sorted((first_brain.genome_id, second_brain.genome_id)))
        distance = self._pairwise_compatibility_distance_cache.get(pair)
        if distance is None:
            distance = self.species_manager.composite_distance(
                first_brain.genome,
                first.physical_traits,
                first.vision,
                getattr(first, "flocking_traits", FlockingTraits()),
                second_brain.genome,
                second.physical_traits,
                second.vision,
                getattr(second, "flocking_traits", FlockingTraits()),
                self.config.genome_config,
            ).composite_distance
            self._pairwise_compatibility_distance_cache[pair] = distance

        threshold = float(self.species_manager.compatibility_threshold)
        if threshold <= 0.0:
            return 1.0 if distance <= 1e-12 else 0.0
        return max(0.0, min(1.0, 1.0 - distance / threshold))

    def _discard_cached_genome(self, genome_id: int) -> None:
        pairwise_cache = getattr(self, "_pairwise_compatibility_distance_cache", {})
        self._pairwise_compatibility_distance_cache = {
            pair: distance
            for pair, distance in pairwise_cache.items()
            if genome_id not in pair
        }

    @staticmethod
    def _binary_species_compatibility(
        first: Creature,
        second: Creature,
    ) -> float:
        first_species = getattr(
            getattr(first, "lineage", None),
            "species_id",
            getattr(first, "species_id", None),
        )
        second_species = getattr(
            getattr(second, "lineage", None),
            "species_id",
            getattr(second, "species_id", None),
        )
        if first_species is None or second_species is None:
            return 0.0
        return 1.0 if first_species == second_species else 0.0

    def restore_brain(self, creature_id: int, genome_id: int) -> NeatBrain:
        genome = self.population.population.get(genome_id)
        if genome is None:
            raise ValueError(
                f"Cannot restore creature {creature_id}: "
                f"genome {genome_id} is missing from the population."
            )
        brain = self._brain_from_genome(genome_id, genome)
        self.brains[creature_id] = brain
        return brain

    def _brain_from_genome(self, genome_id: int, genome: Any) -> NeatBrain:
        """Build a brain already labelled with the selected sensor contract."""
        brain = NeatBrain.from_genome(genome_id, genome, self.config)
        contract = getattr(
            self,
            "sensor_contract",
            SENSOR_CONTRACT,
        )
        brain.last_input_names = contract.input_names
        return brain

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
        child_flocking_traits: FlockingTraits | None = None,
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
            child_flocking_traits,
        )

    def create_mutated_brain_from_genome(
        self,
        parent_genome: Any,
        creature_id: int,
        parent_species_id: int,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        child_flocking_traits: FlockingTraits | None = None,
    ) -> tuple[NeatBrain, SpeciationResult]:
        child_genome = copy.deepcopy(parent_genome)
        child_genome.key = self._next_genome_id()
        child_genome.fitness = None
        with self._using_evolution_rng():
            child_genome.mutate(self.config.genome_config)
        speciation_result = self.species_manager.evaluate_species(
            child_genome,
            child_physical_traits,
            child_vision,
            parent_species_id,
            self.config.genome_config,
            child_flocking_traits,
        )
        self.population.population[child_genome.key] = child_genome

        child_brain = self._brain_from_genome(
            child_genome.key,
            child_genome,
        )
        self.brains[creature_id] = child_brain
        return child_brain, speciation_result

    @contextmanager
    def _using_evolution_rng(self):
        evolution_rng = getattr(self, "_evolution_rng", None)
        if evolution_rng is None:
            yield
            return
        external_state = random.getstate()
        random.setstate(evolution_rng.getstate())
        try:
            yield
        finally:
            evolution_rng.setstate(random.getstate())
            random.setstate(external_state)

    def evolution_random_state(self) -> object | None:
        evolution_rng = getattr(self, "_evolution_rng", None)
        return None if evolution_rng is None else evolution_rng.getstate()

    def restore_evolution_random_state(self, state: object) -> None:
        if not hasattr(self, "_evolution_rng"):
            self._evolution_rng = random.Random()
        self._evolution_rng.setstate(state)

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

        if input_count != self.sensor_contract.input_count:
            raise ValueError(
                f"NEAT config input count mismatch. "
                f"Config: {input_count}, "
                f"code: {self.sensor_contract.input_count}"
            )

        if output_count != ACTION_OUTPUT_COUNT:
            raise ValueError(
                f"NEAT config output count mismatch. "
                f"Config: {output_count}, code: {ACTION_OUTPUT_COUNT}"
            )
