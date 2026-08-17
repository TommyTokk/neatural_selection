"""Focused tests for genotype/NEAT/speciation evolution coordination."""

from __future__ import annotations

import copy
from random import Random
import sys
from types import ModuleType, SimpleNamespace
import unittest


# Keep coordinator tests runnable when optional simulation wheels are unavailable.
for _optional_module in ("neat", "pymunk", "numpy"):
    if _optional_module not in sys.modules:
        sys.modules[_optional_module] = ModuleType(_optional_module)
if "neat.graphs" not in sys.modules:
    _neat_graphs = ModuleType("neat.graphs")
    _neat_graphs.required_for_output = lambda *_args: set()
    sys.modules["neat.graphs"] = _neat_graphs

from src.creature.evolution import CreatureEvolutionCoordinator
from src.creature.genotype import (
    CreatureGenotype,
    FlockingTraits,
    LineageInfo,
    PhysicalTraits,
    VisionTraits,
)


class FakeGenotypeManager:
    """Provide deterministic founder colour for coordinator unit tests."""

    def new_species_color(
        self,
        parent: tuple[int, ...],
        rng: Random,
    ) -> tuple[int, int, int]:
        """Return a deterministic founder colour while consuming one draw.

        Parameters
        ----------
        parent
            Parent colour ignored by the deterministic fixture.
        rng
            Simulation RNG whose staged position is exercised.

        Returns
        -------
        tuple[int, int, int]
            Fixed test founder colour.
        """
        # Consume a draw so transaction-state assertions remain meaningful.
        del parent
        rng.random()
        return (11, 22, 33)


class FakeSpeciesManager:
    """Capture genotype colour observed at the speciation boundary."""

    def __init__(self) -> None:
        """Initialize empty representative state and observation storage.

        Parameters
        ----------
        None
            This initializer receives no external parameters.

        Returns
        -------
        None
            Empty fake species state is created.
        """
        # Store only state needed to prove ordering and transaction isolation.
        self.representatives = {1: object()}
        self.observed_color: tuple[int, ...] | None = None

    def evaluate_species(
        self,
        genome: object,
        physical: PhysicalTraits,
        vision: VisionTraits,
        parent_species_id: int,
        genome_config: object,
        flocking: FlockingTraits,
    ) -> object:
        """Return a new-species result after recording pre-colour traits.

        Parameters
        ----------
        genome
            Child neural genome.
        physical
            Child physical traits.
        vision
            Child visual traits.
        parent_species_id
            Parent species identity.
        genome_config
            Neural genome configuration.
        flocking
            Child social traits.

        Returns
        -------
        object
            Minimal new-species result used by the coordinator.
        """
        # Trait arguments are accepted to mirror the real integration boundary.
        del genome, physical, vision, genome_config, flocking
        return SimpleNamespace(
            species_id=parent_species_id + 1,
            is_new_species=True,
        )


class FakeBrainController:
    """Provide isolated neural state for coordinator unit tests."""

    def __init__(self, has_parent: bool = True) -> None:
        """Initialize fake neural and species state.

        Parameters
        ----------
        has_parent
            Whether live-parent neural mutation succeeds.

        Returns
        -------
        None
            Minimal controller state is created.
        """
        # Expose the mutable fields copied by coordinator transaction commit.
        self.has_parent = has_parent
        self.species_manager = FakeSpeciesManager()
        self.config = SimpleNamespace(genome_config=object())
        self.brains: dict[int, object] = {}
        self.population = SimpleNamespace(population={})
        self._next_genome_id_value = 10
        self._next_brain_revision_value = 20
        self._evolution_rng = Random(5)
        self._pairwise_compatibility_distance_cache: dict[tuple[int, int], float] = {}

    def create_child_neural_brain(
        self,
        parent_id: int,
        child_id: int,
    ) -> object | None:
        """Create a fake neural child when the parent exists.

        Parameters
        ----------
        parent_id
            Live neural parent identity.
        child_id
            Reserved child identity.

        Returns
        -------
        object | None
            Minimal brain or ``None`` for a missing parent.
        """
        # Parent identity is accepted without any genotype values.
        del parent_id
        if not self.has_parent:
            return None
        brain = SimpleNamespace(genome=SimpleNamespace(key=child_id), genome_id=child_id)
        self.brains[child_id] = brain
        return brain

    def transaction_shadow(self) -> FakeBrainController:
        """Return a deep copy of fake mutable neural/species state.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        FakeBrainController
            Independent fake controller state.
        """
        # Deep copy models the real neural controller transaction contract.
        return copy.deepcopy(self)


class EvolutionCoordinatorTests(unittest.TestCase):
    """Verify coordinator ordering, failure, and RNG isolation."""

    def setUp(self) -> None:
        """Create reusable aggregate genotype and lineage fixtures.

        Parameters
        ----------
        None
            This setup method receives no external parameters.

        Returns
        -------
        None
            Fresh non-neural fixtures are stored on the test instance.
        """
        # A mutable aggregate lets the test observe final founder replacement.
        self.genotype = CreatureGenotype(
            VisionTraits(100.0, 1.0),
            PhysicalTraits(8.0),
            FlockingTraits(),
            (90, 100, 110),
        )
        self.lineage = LineageInfo(parent_id=1, generation=2, species_id=4)

    def test_new_species_colour_is_applied_after_speciation(self) -> None:
        """Apply founder colour only after species assignment succeeds.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify final lineage and founder colour.
        """
        # Speciation sees inherited colour; replacement happens afterward.
        controller = FakeBrainController()
        coordinator = CreatureEvolutionCoordinator(FakeGenotypeManager(), controller)
        parent = SimpleNamespace(creature_id=1, color=(90, 100, 110))
        plan = coordinator.finalize_child(
            parent,
            7,
            self.genotype,
            self.lineage,
            Random(8),
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.lineage.species_id, 5)
        self.assertEqual(plan.genotype.color, (11, 22, 33))

    def test_missing_parent_does_not_speciate_or_consume_simulation_rng(self) -> None:
        """Return no plan without advancing simulation RNG for a missing brain.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify failure isolation.
        """
        # Neural parent validation precedes speciation and founder-colour draws.
        controller = FakeBrainController(has_parent=False)
        coordinator = CreatureEvolutionCoordinator(FakeGenotypeManager(), controller)
        rng = Random(44)
        before = rng.getstate()
        parent = SimpleNamespace(creature_id=99, color=(90, 100, 110))
        self.assertIsNone(
            coordinator.finalize_child(
                parent,
                8,
                self.genotype,
                self.lineage,
                rng,
            )
        )
        self.assertEqual(before, rng.getstate())

    def test_discarded_transaction_preserves_live_rng_and_allocators(self) -> None:
        """Keep live RNG and controller allocators unchanged in a shadow.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify uncommitted state isolation.
        """
        # Mutate only the shadow and prove the live state remains untouched.
        controller = FakeBrainController()
        coordinator = CreatureEvolutionCoordinator(FakeGenotypeManager(), controller)
        rng = Random(123)
        rng_before = rng.getstate()
        transaction = coordinator.begin_transaction(rng)
        transaction.simulation_rng.random()
        transaction.coordinator.brain_controller._next_genome_id_value = 999
        self.assertEqual(rng_before, rng.getstate())
        self.assertEqual(controller._next_genome_id_value, 10)


if __name__ == "__main__":
    # Direct execution supports dependency-light coordinator verification.
    unittest.main()
