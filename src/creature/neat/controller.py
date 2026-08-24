from __future__ import annotations

from contextlib import contextmanager
from itertools import count
from pathlib import Path
import random
from typing import Any

import copy

import neat

from configs.sim_config import TraitConfig, VisionConfig
from src.creature.action import (
    ACTION_OUTPUT_COUNT,
    Action,
    BrainOutputIndex,
    neutral_action,
)
from src.creature.neat.brain import NeatBrain
from src.creature.speciation import (
    CompositeCompatibilityDistance,
    ContinuousSpeciesManager,
    NeatChangeSummary,
    SpeciationResult,
    SpeciesDistanceBreakdown,
    SpeciesTraitSnapshot,
    NeuralShift,
    calculate_flocking_trait_distance,
    calculate_phenotypic_distance,
    calculate_phenotypic_distance_components,
    extract_neural_shifts,
)
from src.creature.vision import (
    SENSOR_CONTRACT,
    SensorContract,
    SensorSnapshot,
)

FALLBACK_ACTION = neutral_action()


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
        herding_decay_rate: float = 1.0,
    ) -> None:
        """Execute init behavior.

Parameters
----------
config_path
    Input used by this creature-domain operation.
compatibility_threshold
    Input used by this creature-domain operation.
phenotypic_weight
    Input used by this creature-domain operation.
trait_config
    Input used by this creature-domain operation.
vision_config
    Input used by this creature-domain operation.
flocking_trait_distance_coefficient
    Input used by this creature-domain operation.
sensor_contract
    Input used by this creature-domain operation.
random_seed
    Input used by this creature-domain operation.
herding_decay_rate
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
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
        self.herding_decay_rate = herding_decay_rate
        self._validate_network_contract()
        with self._using_evolution_rng():
            self.population = neat.Population(self.config)
        self._next_genome_id_value = (
            max(self.population.population, default=0) + 1
        )
        self._next_brain_revision_value = 1
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

    def assign_initial_brains(
        self,
        creature_ids: list[int],
    ) -> list[tuple[int, Any]]:
        """Assign initial neural genomes to stable creature identities.
        
        Parameters
        ----------
        creature_ids
            Stable identities receiving the initial neural population.
        Returns
        -------
        list[tuple[int, Any]]
            Ordered genome identities and genomes used for assignment.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Validate identities before mutating the live brain registry.
        self._validate_creature_ids(creature_ids)
        genomes = self._initial_genomes()
        if len(genomes) < len(creature_ids):
            raise ValueError(
                f"Not enough genomes in the population to assign to all creatures. "
                f"Genomes: {len(genomes)}, Creatures: {len(creature_ids)}"
            )

        # Network creation is neural-only; speciation is coordinated separately.
        for creature_id, (genome_id, genome) in zip(creature_ids, genomes):
            self._enforce_reproduction_output_contract(genome, founder=True)
            self.brains[creature_id] = self._brain_from_genome(
                genome_id,
                genome,
            )
        return genomes

    def reset_for_new_sensing_epoch(
        self,
        creature_ids: list[int],
        root_species_id: int,
    ) -> list[tuple[int, Any]]:
        """Replace neural state with genomes matching the current sensor contract.
        
        Parameters
        ----------
        creature_ids
            Stable identities receiving replacement neural genomes.
        root_species_id
            Input used by this creature-domain operation.
        Returns
        -------
        list[tuple[int, Any]]
            Ordered genome identities and genomes used for replacement.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep reset for new sensing epoch behavior explicit in its owning subsystem.
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
        if len(genomes) < len(creature_ids):
            raise ValueError(
                "Not enough fresh genomes for sensing epoch reset. "
                f"Genomes: {len(genomes)}, creatures: {len(creature_ids)}"
            )
        # Rebuild networks without accepting any non-neural trait values.
        for creature_id, (genome_id, genome) in zip(creature_ids, genomes):
            self._enforce_reproduction_output_contract(genome, founder=True)
            self.brains[creature_id] = self._brain_from_genome(
                genome_id,
                genome,
            )
        return genomes

    def decide(
        self,
        creature_id: int,
        snapshot: SensorSnapshot,
        *,
        capture_inputs: bool = False,
        decision_dt: float | None = None,
    ) -> Action:
        """Execute decide behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
snapshot
    Input used by this creature-domain operation.
capture_inputs
    Input used by this creature-domain operation.
decision_dt
    Input used by this creature-domain operation.
Returns
-------
Action
    Result produced by this creature-domain operation."""
        # Keep decide behavior explicit in its owning subsystem.
        brain = self.brains.get(creature_id)
        if brain is None:
            return self.fallback_action()

        return brain.decide(
            snapshot,
            capture_inputs=capture_inputs,
            decision_dt=decision_dt,
        )

    def capture_input_snapshot(self, creature_id: int) -> None:
        """Publish the latest private activation inputs for diagnostics.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep capture input snapshot behavior explicit in its owning subsystem.
        brain = self.brains.get(creature_id)
        if brain is not None:
            brain.capture_input_snapshot()

    def decide_with_input_capture(
        self,
        creature_id: int,
        snapshot: SensorSnapshot,
        *,
        decision_dt: float | None = None,
    ) -> Action:
        """Decide while publishing a stable inspector/telemetry input copy.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
snapshot
    Input used by this creature-domain operation.
decision_dt
    Input used by this creature-domain operation.
Returns
-------
Action
    Result produced by this creature-domain operation."""
        # Keep decide with input capture behavior explicit in its owning subsystem.
        return self.decide(
            creature_id,
            snapshot,
            capture_inputs=True,
            decision_dt=decision_dt,
        )

    def fallback_action(self) -> Action:
        """Execute fallback action behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
Action
    Result produced by this creature-domain operation."""
        # Keep fallback action behavior explicit in its owning subsystem.
        return neutral_action()

    def remove_brain(self, creature_id: int) -> None:
        """Execute remove brain behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep remove brain behavior explicit in its owning subsystem.
        brain = self.brains.pop(creature_id, None)
        if brain is not None:
            self._discard_cached_genome(brain.genome_id)

    def prune_population_archive(self, archive_size: int) -> set[int]:
        """Execute prune population archive behavior.

Parameters
----------
archive_size
    Input used by this creature-domain operation.
Returns
-------
set[int]
    Result produced by this creature-domain operation."""
        # Keep prune population archive behavior explicit in its owning subsystem.
        live_genome_ids = {
            brain.genome_id
            for brain in self.brains.values()
        }
        dead_genomes = [
            genome
            for genome_id, genome in self.population.population.items()
            if genome_id not in live_genome_ids
        ]
        retained_dead = (
            dead_genomes
            if len(dead_genomes) <= max(0, archive_size)
            else self._evolution_rng.sample(dead_genomes, max(0, archive_size))
        )
        retained_genome_ids = live_genome_ids | {
            genome.key for genome in retained_dead
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
        """Execute prune species representatives behavior.

Parameters
----------
retained_species_ids
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep prune species representatives behavior explicit in its owning subsystem.
        protected_species_ids = {1, *retained_species_ids}
        self.species_manager.representatives = {
            species_id: representative
            for species_id, representative in (
                self.species_manager.representatives.items()
            )
            if species_id in protected_species_ids
        }

    def archive_brain(self, creature_id: int) -> bool:
        """Execute archive brain behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep archive brain behavior explicit in its owning subsystem.
        brain = self.brains.get(creature_id)
        if brain is None:
            return False

        brain.genome.fitness = None
        self.population.population[brain.genome_id] = brain.genome
        return True

    def genome_id_for(self, creature_id: int) -> int | None:
        """Execute genome id for behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
int | None
    Result produced by this creature-domain operation."""
        # Keep genome id for behavior explicit in its owning subsystem.
        brain = self.brains.get(creature_id)
        if brain is None:
            return None
        return brain.genome_id

    def brain_revision_for(self, creature_id: int) -> int | None:
        """Execute brain revision for behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
int | None
    Result produced by this creature-domain operation."""
        # Keep brain revision for behavior explicit in its owning subsystem.
        brain = self.brains.get(creature_id)
        if brain is None:
            return None
        return brain.brain_revision

    def brain_for(self, creature_id: int) -> NeatBrain | None:
        """Execute brain for behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
NeatBrain | None
    Result produced by this creature-domain operation."""
        # Keep brain for behavior explicit in its owning subsystem.
        return self.brains.get(creature_id)


    def _discard_cached_genome(self, genome_id: int) -> None:
        """Execute discard cached genome behavior.

Parameters
----------
genome_id
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep discard cached genome behavior explicit in its owning subsystem.
        pairwise_cache = getattr(self, "_pairwise_compatibility_distance_cache", {})
        self._pairwise_compatibility_distance_cache = {
            pair: distance
            for pair, distance in pairwise_cache.items()
            if genome_id not in pair
        }


    def restore_brain(self, creature_id: int, genome_id: int) -> NeatBrain:
        """Execute restore brain behavior.
        
        Parameters
        ----------
        creature_id
            Input used by this creature-domain operation.
        genome_id
            Input used by this creature-domain operation.
        Returns
        -------
        NeatBrain
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep restore brain behavior explicit in its owning subsystem.
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
        """Build a brain already labelled with the selected sensor contract.

Parameters
----------
genome_id
    Input used by this creature-domain operation.
genome
    Input used by this creature-domain operation.
Returns
-------
NeatBrain
    Result produced by this creature-domain operation."""
        # Keep brain from genome behavior explicit in its owning subsystem.
        self._enforce_reproduction_output_contract(genome, founder=False)
        brain = NeatBrain.from_genome(genome_id, genome, self.config)
        next_revision = getattr(self, "_next_brain_revision_value", 1)
        brain.brain_revision = next_revision
        self._next_brain_revision_value = next_revision + 1
        brain.herding_decay_rate = getattr(
            self,
            "herding_decay_rate",
            1.0,
        )
        contract = getattr(
            self,
            "sensor_contract",
            SENSOR_CONTRACT,
        )
        brain.last_input_names = contract.input_names
        return brain

    def evolution_allocator_state(self) -> dict[str, object]:
        """Return allocator positions needed to continue mutating after a load.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
dict[str, object]
    Result produced by this creature-domain operation."""
        # Keep evolution allocator state behavior explicit in its owning subsystem.
        genome_config = self.config.genome_config
        minimum_next_node_id = self._minimum_next_node_id()
        next_node_id = minimum_next_node_id
        node_indexer_position = minimum_next_node_id
        node_indexer = getattr(genome_config, "node_indexer", None)
        if node_indexer is not None:
            try:
                # ``itertools.count`` stopped supporting ``__reduce__`` in
                # Python 3.14. Consume its next value and immediately replace
                # it with an equivalent iterator. This preserves the exact
                # allocator position, including IDs no longer represented by
                # a living or archived genome.
                node_indexer_position = int(next(node_indexer))
                genome_config.node_indexer = count(node_indexer_position)
                next_node_id = max(next_node_id, node_indexer_position)
            except (AttributeError, StopIteration, TypeError, ValueError):
                pass

        innovation_number = self._minimum_innovation_number()
        innovation_tracker = getattr(genome_config, "innovation_tracker", None)
        innovation_history: dict[object, int] = {}
        if innovation_tracker is not None:
            innovation_number = max(
                innovation_number,
                int(getattr(innovation_tracker, "global_counter", 0)),
            )
            innovation_history = copy.deepcopy(
                getattr(
                    innovation_tracker,
                    "generation_innovations",
                    {},
                )
            )

        return {
            "next_node_id": next_node_id,
            "node_indexer_position": node_indexer_position,
            "innovation_number": innovation_number,
            "innovation_history": innovation_history,
        }

    def transaction_shadow(self) -> NeatBrainController:
        """Clone mutable evolution state without advancing live allocators.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
NeatBrainController
    Result produced by this creature-domain operation."""
        # Keep transaction shadow behavior explicit in its owning subsystem.
        shadow = copy.copy(self)
        shadow.config = copy.copy(self.config)
        live_genome_config = self.config.genome_config
        shadow_genome_config = copy.copy(live_genome_config)
        shadow.config.genome_config = shadow_genome_config

        allocator_state = self.evolution_allocator_state()
        shadow_genome_config.node_indexer = count(
            allocator_state["next_node_id"]
        )
        live_tracker = getattr(live_genome_config, "innovation_tracker", None)
        if live_tracker is not None:
            shadow_tracker = copy.deepcopy(live_tracker)
            shadow_tracker.global_counter = allocator_state["innovation_number"]
            shadow_genome_config.innovation_tracker = shadow_tracker

        # Child mutation already deep-copies the selected parent genome.  The
        # transaction only adds entries to population and brain mappings, so
        # copying their containers isolates staging without cloning every live
        # genome and brain.
        shadow.population = copy.copy(self.population)
        shadow.population.population = dict(self.population.population)
        shadow.brains = dict(self.brains)

        shadow.species_manager = copy.copy(self.species_manager)
        shadow.species_manager.representatives = {
            species_id: (
                genome,
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
                copy.deepcopy(flocking_traits),
            )
            for species_id, (
                genome,
                physical_traits,
                vision,
                flocking_traits,
            ) in self.species_manager.representatives.items()
        }
        shadow._pairwise_compatibility_distance_cache = dict(
            getattr(self, "_pairwise_compatibility_distance_cache", {})
        )
        shadow._evolution_rng = random.Random()
        evolution_state = self.evolution_random_state()
        if evolution_state is not None:
            shadow._evolution_rng.setstate(evolution_state)
        return shadow

    def restore_evolution_allocators(
        self,
        next_node_id: int | None = None,
        innovation_number: int | None = None,
        innovation_history: dict[object, int] | None = None,
    ) -> None:
        """Restore or reconstruct NEAT's process-local structural allocators.

Parameters
----------
next_node_id
    Input used by this creature-domain operation.
innovation_number
    Input used by this creature-domain operation.
innovation_history
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep restore evolution allocators behavior explicit in its owning subsystem.
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
            if innovation_history is not None:
                generation_innovations.update(
                    copy.deepcopy(innovation_history)
                )

    def _known_genomes(self) -> list[Any]:
        """Execute known genomes behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
list[Any]
    Result produced by this creature-domain operation."""
        # Keep known genomes behavior explicit in its owning subsystem.
        genomes = list(self.population.population.values())
        genomes.extend(
            representative[0]
            if isinstance(representative, tuple)
            else representative
            for representative in self.species_manager.representatives.values()
        )
        return genomes

    def _minimum_next_node_id(self) -> int:
        """Execute minimum next node id behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
int
    Result produced by this creature-domain operation."""
        # Keep minimum next node id behavior explicit in its owning subsystem.
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
        """Execute minimum innovation number behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
int
    Result produced by this creature-domain operation."""
        # Keep minimum innovation number behavior explicit in its owning subsystem.
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

    def create_child_neural_brain(
        self,
        parent_creature_id: int,
        child_creature_id: int,
    ) -> NeatBrain | None:
        """Mutate only the neural genome of a live parent.

Parameters
----------
parent_creature_id
    Live parent identity used to locate its neural genome.
child_creature_id
    Reserved identity for the new brain.

Returns
-------
NeatBrain | None
    Mutated neural brain, or ``None`` when the parent has no brain."""
        # Keep create child neural brain behavior explicit in its owning subsystem.
        # This neural API deliberately receives no non-neural trait values.
        parent_brain = self.brains.get(parent_creature_id)
        if parent_brain is None:
            return None
        return self.create_mutated_neural_brain(
            parent_brain.genome,
            child_creature_id,
        )

    def create_mutated_neural_brain(
        self,
        parent_genome: Any,
        creature_id: int,
    ) -> NeatBrain:
        """Create and register a neural-only mutation of an archived genome.

Parameters
----------
parent_genome
    Neural genome used as the mutation baseline.
creature_id
    Stable identity that will own the new brain.

Returns
-------
NeatBrain
    Registered brain backed by the mutated child genome."""
        # Keep create mutated neural brain behavior explicit in its owning subsystem.
        # Copy before mutation so parent archives and representatives stay immutable.
        child_genome = copy.deepcopy(parent_genome)
        child_genome.key = self._next_genome_id()
        child_genome.fitness = None
        with self._using_evolution_rng():
            child_genome.mutate(self.config.genome_config)
        self._enforce_reproduction_output_contract(child_genome, founder=False)
        self.population.population[child_genome.key] = child_genome
        child_brain = self._brain_from_genome(child_genome.key, child_genome)
        self.brains[creature_id] = child_brain
        return child_brain

    def _enforce_reproduction_output_contract(
        self,
        genome: Any,
        *,
        founder: bool,
    ) -> None:
        """Pin reproduction to sigmoid and make founders quiescent.

        Parameters
        ----------
        genome
            Neural genome whose reproduction output is normalized.
        founder
            Whether the output bias must be reset to the founder default.

        Returns
        -------
        None
            The output node is normalized in place when present.
        """
        # Resolve the declared output key instead of assuming NEAT node IDs.
        genome_config = getattr(self.config, "genome_config", None)
        output_keys = tuple(getattr(genome_config, "output_keys", ()))
        output_index = int(BrainOutputIndex.REPRODUCE)
        if output_index >= len(output_keys):
            return
        node = (getattr(genome, "nodes", {}) or {}).get(
            output_keys[output_index]
        )
        if node is None:
            return
        node.activation = "sigmoid"
        if founder:
            node.bias = -1.0



    @contextmanager
    def _using_evolution_rng(self):
        """Execute using evolution rng behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep using evolution rng behavior explicit in its owning subsystem.
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
        """Execute evolution random state behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
object | None
    Result produced by this creature-domain operation."""
        # Keep evolution random state behavior explicit in its owning subsystem.
        evolution_rng = getattr(self, "_evolution_rng", None)
        return None if evolution_rng is None else evolution_rng.getstate()

    def restore_evolution_random_state(self, state: object) -> None:
        """Execute restore evolution random state behavior.

Parameters
----------
state
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep restore evolution random state behavior explicit in its owning subsystem.
        if not hasattr(self, "_evolution_rng"):
            self._evolution_rng = random.Random()
        self._evolution_rng.setstate(state)

    def archived_genomes(self, count: int) -> list[Any]:
        """Return an unranked deterministic sample of retained genomes.

Parameters
----------
count
    Input used by this creature-domain operation.
Returns
-------
list[Any]
    Result produced by this creature-domain operation."""
        # Sampling never reads genome fitness or species-adjusted values.
        genomes = list(self.population.population.values())
        requested = max(0, int(count))
        if len(genomes) <= requested:
            return genomes
        return self._evolution_rng.sample(genomes, requested)

    def _next_genome_id(self) -> int:
        """Execute next genome id behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
int
    Result produced by this creature-domain operation."""
        # Keep next genome id behavior explicit in its owning subsystem.
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
        """Execute initial genomes behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
list[tuple[int, Any]]
    Result produced by this creature-domain operation."""
        # Keep initial genomes behavior explicit in its owning subsystem.
        return list(self.population.population.items())

    def _validate_creature_ids(self, creature_ids: list[int]) -> None:
        """Execute validate creature ids behavior.
        
        Parameters
        ----------
        creature_ids
            Input used by this creature-domain operation.
        Returns
        -------
        None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep validate creature ids behavior explicit in its owning subsystem.
        if len(set(creature_ids)) != len(creature_ids):
            raise ValueError("Cannot assign NEAT brains to duplicate creature ids.")

    def _validate_network_contract(self) -> None:
        """Execute validate network contract behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        Returns
        -------
        None
            Result produced by this creature-domain operation.
        
        Raises
        ------
        ValueError
            If an input or restored value violates validation rules.
        """
        # Keep validate network contract behavior explicit in its owning subsystem.
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
