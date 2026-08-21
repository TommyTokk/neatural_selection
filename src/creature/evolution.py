"""Coordination of independent genotype, neural, and species evolution."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from random import Random
from typing import Any

from src.creature.genotype import CreatureGenotype, GenotypeManager, LineageInfo
from src.creature.model import Creature
from src.creature.neat.brain import NeatBrain
from src.creature.neat.controller import NeatBrainController
from src.creature.speciation import SpeciationResult


@dataclass(frozen=True, slots=True)
class OffspringPlan:
    """Describe a fully evolved child before physics materialization."""

    child_id: int
    genotype: CreatureGenotype
    lineage: LineageInfo
    brain: NeatBrain
    speciation_result: SpeciationResult


@dataclass(slots=True)
class EvolutionTransaction:
    """Hold isolated evolution state and its shadow simulation RNG."""

    coordinator: CreatureEvolutionCoordinator
    simulation_rng: Random


class CreatureEvolutionCoordinator:
    """Combine genotype mutation, neural mutation, and speciation atomically."""

    def __init__(
        self,
        genotype_manager: GenotypeManager,
        brain_controller: NeatBrainController,
    ) -> None:
        """Initialize evolution subsystem dependencies.

Parameters
----------
genotype_manager
    Independent non-neural mutation service.
brain_controller
    Neural genome and network service.

Returns
-------
None
    Services are retained without duplicating their state."""
        # Keep init behavior explicit in its owning subsystem.
        # The coordinator is the only public object intentionally holding both.
        self.genotype_manager = genotype_manager
        self.brain_controller = brain_controller
        self.species_manager = brain_controller.species_manager

    def assign_initial_brains(self, creatures: list[Creature]) -> None:
        """Assign neural brains, then register the initial genotype representative.

        Parameters
        ----------
        creatures
            Ordered initial live creatures.

        Returns
        -------
        None
            Neural brains and the root representative are registered.

        Raises
        ------
        ValueError
            If identities are invalid or genomes are insufficient.
        """
        # Phase one creates networks using stable identities only.
        assign = self.brain_controller.assign_initial_brains
        if "creature_ids" not in inspect.signature(assign).parameters:
            # Historical test/extension controllers still coordinate both phases.
            assign(creatures)
            return
        genomes = assign([creature.creature_id for creature in creatures])
        if not genomes or not creatures:
            return

        # Phase two attaches the founder's non-neural representative traits.
        founder = creatures[0]
        self.species_manager.register_initial_representative(
            genomes[0][1],
            founder.physical_traits,
            founder.vision,
            flocking_traits=founder.flocking_traits,
        )

    def reset_for_new_sensing_epoch(
        self,
        creatures: list[Creature],
        root_species_id: int,
    ) -> None:
        """Reset neural networks, then register the epoch's genotype founder.

        Parameters
        ----------
        creatures
            Ordered live creatures receiving replacement brains.
        root_species_id
            Species identity retained across the sensing-contract reset.

        Returns
        -------
        None
            Neural state and the root species representative are replaced.

        Raises
        ------
        ValueError
            If replacement genomes cannot cover every creature identity.
        """
        # Keep neural reset isolated from trait-aware representative registration.
        reset = self.brain_controller.reset_for_new_sensing_epoch
        if "creature_ids" not in inspect.signature(reset).parameters:
            # Historical test/extension controllers still coordinate both phases.
            reset(creatures, root_species_id)
            return
        genomes = reset(
            [creature.creature_id for creature in creatures], root_species_id
        )
        self.species_manager = self.brain_controller.species_manager
        if not genomes or not creatures:
            return
        founder = creatures[0]
        self.species_manager.register_initial_representative(
            genomes[0][1],
            founder.physical_traits,
            founder.vision,
            species_id=root_species_id,
            flocking_traits=founder.flocking_traits,
        )

    def flocking_compatibility(
        self,
        first: Creature,
        second: Creature,
    ) -> float:
        """Return live compatibility using neural and genotype distance.

        Parameters
        ----------
        first
            First live creature.
        second
            Second live creature.

        Returns
        -------
        float
            Unit-interval compatibility for social steering.
        """
        # Honor historical extension controllers outside the canonical package.
        legacy_resolver = getattr(
            self.brain_controller,
            "flocking_compatibility",
            None,
        )
        if callable(legacy_resolver):
            return float(legacy_resolver(first, second))

        # A missing brain degrades to the stable historical species relation.
        brains = getattr(self.brain_controller, "brains", None)
        if not isinstance(brains, dict):
            return 0.0
        first_brain = brains.get(first.creature_id)
        second_brain = brains.get(second.creature_id)
        if first_brain is None or second_brain is None:
            return self._binary_species_compatibility(first, second)

        # Cache by immutable genome IDs to avoid composite hot-loop recomputation.
        pair = tuple(sorted((first_brain.genome_id, second_brain.genome_id)))
        cache = self.brain_controller._pairwise_compatibility_distance_cache
        distance = cache.get(pair)
        if distance is None:
            distance = self.species_manager.composite_distance(
                first_brain.genome,
                first.physical_traits,
                first.vision,
                first.flocking_traits,
                second_brain.genome,
                second.physical_traits,
                second.vision,
                second.flocking_traits,
                self.brain_controller.config.genome_config,
            ).composite_distance
            cache[pair] = distance
        threshold = float(self.species_manager.compatibility_threshold)
        if threshold <= 0.0:
            return 1.0 if distance <= 1e-12 else 0.0
        return max(0.0, min(1.0, 1.0 - distance / threshold))

    @staticmethod
    def _binary_species_compatibility(
        first: Creature,
        second: Creature,
    ) -> float:
        """Return one only when both creatures share a known species.

        Parameters
        ----------
        first
            First live creature.
        second
            Second live creature.

        Returns
        -------
        float
            Binary species compatibility.
        """
        # Unknown lineage cannot safely imply social compatibility.
        first_species = getattr(getattr(first, "lineage", None), "species_id", None)
        second_species = getattr(getattr(second, "lineage", None), "species_id", None)
        if first_species is None or second_species is None:
            return 0.0
        return 1.0 if first_species == second_species else 0.0

    def plan_child(
        self,
        parent: Creature,
        child_id: int,
        rng: Random,
    ) -> OffspringPlan | None:
        """Plan one child from a live parent without creating physics state.

Parameters
----------
parent
    Live parent supplying genotype, lineage, and neural identity.
child_id
    Reserved stable identity for the child.
rng
    Simulation random generator used only for non-neural mutation.

Returns
-------
OffspringPlan | None
    Complete plan, or ``None`` when the parent brain is unavailable."""
        # Keep plan child behavior explicit in its owning subsystem.
        # Non-neural mutation is completed independently before neural evolution.
        mutation = self.genotype_manager.mutate(parent.genotype, rng)
        genotype = mutation.genotype
        lineage = LineageInfo(
            parent_id=parent.creature_id,
            generation=parent.lineage.generation + 1,
            species_id=parent.lineage.species_id,
            mutation_delta=mutation.mutation_delta,
        )
        return self.finalize_child(parent, child_id, genotype, lineage, rng)

    def plan_from_genome(
        self,
        parent_genome: Any,
        parent_genotype: CreatureGenotype,
        parent_lineage: LineageInfo,
        parent_creature_id: int,
        child_id: int,
        rng: Random,
    ) -> OffspringPlan:
        """Plan one extinction-recovery child from archived parent state.

Parameters
----------
parent_genome
    Archived parent neural genome.
parent_genotype
    Archived parent non-neural genotype.
parent_lineage
    Archived ancestry and species metadata.
parent_creature_id
    Historical identity of the archived parent.
child_id
    Reserved stable identity for the child.
rng
    Simulation random generator used for non-neural mutation.

Returns
-------
OffspringPlan
    Complete recovery plan."""
        # Keep plan from genome behavior explicit in its owning subsystem.
        # Recovery shares the exact genotype path used by normal reproduction.
        mutation = self.genotype_manager.mutate(parent_genotype, rng)
        genotype = mutation.genotype
        lineage = LineageInfo(
            parent_id=parent_creature_id,
            generation=parent_lineage.generation + 1,
            species_id=parent_lineage.species_id,
            mutation_delta=mutation.mutation_delta,
        )
        return self.finalize_from_genome(
            parent_genome,
            parent_genotype.color,
            child_id,
            genotype,
            lineage,
            rng,
        )

    def finalize_child(
        self,
        parent: Creature,
        child_id: int,
        genotype: CreatureGenotype,
        lineage: LineageInfo,
        rng: Random,
    ) -> OffspringPlan | None:
        """Complete neural evolution and speciation for a mutated live child.

        Parameters
        ----------
        parent
            Live parent supplying neural identity and founder colour.
        child_id
            Reserved stable identity for the child.
        genotype
            Already-mutated non-neural genotype.
        lineage
            Already-mutated ancestry metadata with the parent species.
        rng
            Shadow or live simulation RNG used only for founder colour.

        Returns
        -------
        OffspringPlan | None
            Complete plan, or ``None`` if the live parent has no neural brain.
        """
        # Neural mutation consumes only the controller's dedicated evolution RNG.
        brain = self.brain_controller.create_child_neural_brain(
            parent.creature_id,
            child_id,
        )
        if brain is None:
            return None
        return self._finalize_plan(
            child_id,
            genotype,
            lineage,
            brain,
            parent.color,
            rng,
        )

    def finalize_from_genome(
        self,
        parent_genome: Any,
        parent_color: tuple[int, ...],
        child_id: int,
        genotype: CreatureGenotype,
        lineage: LineageInfo,
        rng: Random,
    ) -> OffspringPlan:
        """Complete recovery evolution from an archived neural genome.

        Parameters
        ----------
        parent_genome
            Archived neural genome used as the mutation baseline.
        parent_color
            Archived founder colour used if a new species is created.
        child_id
            Reserved stable identity for the recovery child.
        genotype
            Already-mutated non-neural genotype.
        lineage
            Recovery lineage containing the archived parent species.
        rng
            Shadow or live simulation RNG used only for founder colour.

        Returns
        -------
        OffspringPlan
            Complete recovery plan.
        """
        # Archived recovery differs only in how its neural parent is located.
        brain = self.brain_controller.create_mutated_neural_brain(
            parent_genome,
            child_id,
        )
        return self._finalize_plan(
            child_id,
            genotype,
            lineage,
            brain,
            parent_color,
            rng,
        )

    def _finalize_plan(
        self,
        child_id: int,
        genotype: CreatureGenotype,
        lineage: LineageInfo,
        brain: NeatBrain,
        parent_color: tuple[int, ...],
        rng: Random,
    ) -> OffspringPlan:
        """Apply composite speciation and final founder-colour replacement.

        Parameters
        ----------
        child_id
            Reserved stable child identity.
        genotype
            Mutated non-neural genotype.
        lineage
            Mutable child lineage still carrying the parent species.
        brain
            Newly registered neural brain.
        parent_color
            Parent RGB or RGBA founder colour.
        rng
            Simulation RNG used only if speciation creates a new species.

        Returns
        -------
        OffspringPlan
            Final genotype, lineage, neural brain, and speciation result.
        """
        # Composite compatibility is the explicit integration boundary.
        speciation = self.species_manager.evaluate_species(
            brain.genome,
            genotype.physical_traits,
            genotype.vision,
            lineage.species_id,
            self.brain_controller.config.genome_config,
            genotype.flocking_traits,
        )
        lineage.species_id = speciation.species_id

        # Founder colour is replaced only after the final species is known.
        if speciation.is_new_species:
            genotype.color = self.genotype_manager.new_species_color(
                parent_color,
                rng,
            )
        return OffspringPlan(child_id, genotype, lineage, brain, speciation)

    def transaction_shadow(self) -> CreatureEvolutionCoordinator:
        """Clone mutable neural/species state for transactional staging.

Parameters
----------
None
    This method receives no external parameters.

Returns
-------
CreatureEvolutionCoordinator
    Coordinator backed by an independent controller shadow."""
        # Keep transaction shadow behavior explicit in its owning subsystem.
        # Genotype configuration is immutable; only controller state needs cloning.
        return CreatureEvolutionCoordinator(
            self.genotype_manager,
            self.brain_controller.transaction_shadow(),
        )

    def begin_transaction(self, simulation_rng: Random) -> EvolutionTransaction:
        """Create isolated neural, species, allocator, and simulation RNG state.

        Parameters
        ----------
        simulation_rng
            Live simulation random generator whose state is staged.

        Returns
        -------
        EvolutionTransaction
            Independent coordinator and RNG suitable for offspring staging.

        Notes
        -----
        Failure may discard this value without advancing any live allocator,
        representative, neural RNG, or simulation RNG state.
        """
        # Clone the simulation RNG separately from neural controller state.
        shadow_rng = Random()
        shadow_rng.setstate(simulation_rng.getstate())
        return EvolutionTransaction(self.transaction_shadow(), shadow_rng)

    def commit_transaction(
        self,
        transaction: EvolutionTransaction,
        simulation_rng: Random,
    ) -> None:
        """Commit a successfully staged evolution transaction atomically.

        Parameters
        ----------
        transaction
            Completed shadow evolution transaction.
        simulation_rng
            Live simulation RNG that adopts the staged position.

        Returns
        -------
        None
            Neural, species, allocator, and RNG state are adopted in place.
        """
        # Adopt controller state before exposing the matching simulation RNG state.
        self.commit_shadow(transaction.coordinator)
        simulation_rng.setstate(transaction.simulation_rng.getstate())

    def prune_archives(
        self,
        archive: dict[int, Any],
        active_species_ids: set[int],
        archive_size: int,
    ) -> dict[int, Any]:
        """Prune aligned neural, genotype, and representative archives.

        Parameters
        ----------
        archive
            Genotype archive keyed by neural genome identity.
        active_species_ids
            Species identities represented by live creatures.
        archive_size
            Maximum number of unranked dead neural genomes to retain.

        Returns
        -------
        dict[int, Any]
            Filtered genotype archive aligned with retained neural genomes.
        """
        # Neural retention is authoritative because genome IDs join the archives.
        retained_genome_ids = self.brain_controller.prune_population_archive(
            archive_size
        )
        filtered = {
            genome_id: value
            for genome_id, value in archive.items()
            if genome_id in retained_genome_ids
        }

        # Representatives survive while either live or retained archives need them.
        retained_species_ids = set(active_species_ids)
        retained_species_ids.update(
            int(value.lineage.species_id) for value in filtered.values()
        )
        self.brain_controller.prune_species_representatives(retained_species_ids)
        self.species_manager = self.brain_controller.species_manager
        return filtered

    def commit_shadow(self, shadow: CreatureEvolutionCoordinator) -> None:
        """Adopt all mutable neural and species state from a staged shadow.

Parameters
----------
shadow
    Successfully staged evolution coordinator.

Returns
-------
None
    Live controller state is replaced in place."""
        # Keep commit shadow behavior explicit in its owning subsystem.
        # Copy the established mutable state set without replacing public aliases.
        for name in (
            "config",
            "population",
            "brains",
            "species_manager",
            "_next_genome_id_value",
            "_next_brain_revision_value",
            "_evolution_rng",
            "_pairwise_compatibility_distance_cache",
        ):
            if hasattr(shadow.brain_controller, name):
                setattr(
                    self.brain_controller,
                    name,
                    getattr(shadow.brain_controller, name),
                )
        self.species_manager = self.brain_controller.species_manager
