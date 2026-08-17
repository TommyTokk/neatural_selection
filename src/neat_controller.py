"""Documented compatibility façade for the creature NEAT controller.

New code imports :class:`src.creature.neat.controller.NeatBrainController`,
whose methods accept neural values only. This legacy subclass preserves the
former creature/trait-aware integration calls for external users.
"""

from __future__ import annotations

from typing import Any

from src.creature.genotype import FlockingTraits, PhysicalTraits, VisionTraits
from src.creature.model import Creature
from src.creature.neat.controller import *  # noqa: F401,F403
from src.creature.neat.controller import NeatBrainController as _NeuralController
from src.creature.speciation import *  # noqa: F401,F403


class NeatBrainController(_NeuralController):
    """Preserve historical creature-aware calls over the neural controller."""

    def assign_initial_brains(self, creatures: list[Creature]) -> None:
        """Assign neural brains and register the initial trait representative.

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
        # Delegate neural creation using IDs, then perform legacy integration.
        genomes = super().assign_initial_brains(
            [creature.creature_id for creature in creatures]
        )
        if genomes and creatures:
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
        """Reset neural state and re-register a creature trait representative.

        Parameters
        ----------
        creatures
            Ordered live creatures receiving replacement brains.
        root_species_id
            Species identity retained for the new sensing epoch.

        Returns
        -------
        None
            Neural brains and the root representative are replaced.

        Raises
        ------
        ValueError
            If replacement genomes cannot cover all creature identities.
        """
        # Complete neural reset before attaching non-neural representative traits.
        genomes = super().reset_for_new_sensing_epoch(
            [creature.creature_id for creature in creatures],
            root_species_id,
        )
        if genomes and creatures:
            founder = creatures[0]
            self.species_manager.register_initial_representative(
                genomes[0][1],
                founder.physical_traits,
                founder.vision,
                species_id=root_species_id,
                flocking_traits=founder.flocking_traits,
            )

    def flocking_compatibility(self, first: Creature, second: Creature) -> float:
        """Return legacy continuous compatibility for two live creatures.

        Parameters
        ----------
        first
            First live creature.
        second
            Second live creature.

        Returns
        -------
        float
            Unit-interval compatibility derived from composite distance.
        """
        # Missing brains fall back to historical binary species membership.
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
                first.flocking_traits,
                second_brain.genome,
                second.physical_traits,
                second.vision,
                second.flocking_traits,
                self.config.genome_config,
            ).composite_distance
            self._pairwise_compatibility_distance_cache[pair] = distance
        threshold = float(self.species_manager.compatibility_threshold)
        if threshold <= 0.0:
            return 1.0 if distance <= 1e-12 else 0.0
        return max(0.0, min(1.0, 1.0 - distance / threshold))

    @staticmethod
    def _binary_species_compatibility(
        first: Creature,
        second: Creature,
    ) -> float:
        """Return binary compatibility from historical species membership.

        Parameters
        ----------
        first
            First live creature.
        second
            Second live creature.

        Returns
        -------
        float
            One for equal known species, otherwise zero.
        """
        # Missing lineage values are incompatible rather than guessed.
        first_species = getattr(getattr(first, "lineage", None), "species_id", None)
        second_species = getattr(getattr(second, "lineage", None), "species_id", None)
        if first_species is None or second_species is None:
            return 0.0
        return 1.0 if first_species == second_species else 0.0

    def create_child_brain(
        self,
        parent_creature_id: int,
        child_creature_id: int,
        parent_species_id: int,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        child_flocking_traits: FlockingTraits | None = None,
    ) -> tuple[Any | None, SpeciationResult | None]:
        """Preserve the historical combined neural/speciation child call.

        Parameters
        ----------
        parent_creature_id
            Live neural parent identity.
        child_creature_id
            Reserved child identity.
        parent_species_id
            Parent species used as the speciation baseline.
        child_physical_traits
            Mutated child physical traits.
        child_vision
            Mutated child visual traits.
        child_flocking_traits
            Mutated child social traits.

        Returns
        -------
        tuple[Any | None, SpeciationResult | None]
            Child brain and speciation result, or two ``None`` values.
        """
        # Neural mutation remains delegated to the trait-free canonical method.
        brain = self.create_child_neural_brain(parent_creature_id, child_creature_id)
        if brain is None:
            return None, None
        result = self.species_manager.evaluate_species(
            brain.genome,
            child_physical_traits,
            child_vision,
            parent_species_id,
            self.config.genome_config,
            child_flocking_traits,
        )
        return brain, result

    def create_mutated_brain_from_genome(
        self,
        parent_genome: Any,
        creature_id: int,
        parent_species_id: int,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        child_flocking_traits: FlockingTraits | None = None,
    ) -> tuple[Any, SpeciationResult]:
        """Preserve archived-genome mutation with integrated speciation.

        Parameters
        ----------
        parent_genome
            Archived neural parent genome.
        creature_id
            Stable recovery child identity.
        parent_species_id
            Archived parent species identity.
        child_physical_traits
            Mutated recovery physical traits.
        child_vision
            Mutated recovery visual traits.
        child_flocking_traits
            Mutated recovery social traits.

        Returns
        -------
        tuple[Any, SpeciationResult]
            Registered child brain and composite speciation result.
        """
        # Neural mutation completes before non-neural composite speciation.
        brain = self.create_mutated_neural_brain(parent_genome, creature_id)
        result = self.species_manager.evaluate_species(
            brain.genome,
            child_physical_traits,
            child_vision,
            parent_species_id,
            self.config.genome_config,
            child_flocking_traits,
        )
        return brain, result
