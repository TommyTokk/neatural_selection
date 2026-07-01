from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import unittest

try:
    import neat  # noqa: F401
except ModuleNotFoundError:
    sys.modules["neat"] = ModuleType("neat")

try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = SimpleNamespace(
        Body=object,
        Circle=object,
    )

from src.neat_controller import ContinuousSpeciesManager, NeatBrainController


class FakeGenome:
    def __init__(self, distance: float = 0.0, key: int = 1) -> None:
        self.reported_distance = distance
        self.key = key
        self.fitness: float | None = None

    def distance(self, other: object, genome_config: object) -> float:
        del other, genome_config
        return self.reported_distance

    def mutate(self, genome_config: object) -> None:
        del genome_config


class ContinuousSpeciesManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ContinuousSpeciesManager(compatibility_threshold=3.0)
        self.representative = FakeGenome()
        self.manager.register_initial_representative(self.representative)

    def test_distance_below_threshold_keeps_parent_species(self) -> None:
        child = FakeGenome(distance=2.99)

        self.assertEqual(
            self.manager.evaluate_species(child, 1, object()),
            (1, False),
        )
        self.assertEqual(self.manager.representatives, {1: self.representative})

    def test_distance_equal_to_threshold_keeps_parent_species(self) -> None:
        child = FakeGenome(distance=3.0)

        self.assertEqual(
            self.manager.evaluate_species(child, 1, object()),
            (1, False),
        )

    def test_distance_above_threshold_creates_sequential_species(self) -> None:
        first_founder = FakeGenome(distance=3.01)
        second_founder = FakeGenome(distance=4.0)

        self.assertEqual(
            self.manager.evaluate_species(first_founder, 1, object()),
            (2, True),
        )
        self.assertEqual(
            self.manager.evaluate_species(second_founder, 2, object()),
            (3, True),
        )
        self.assertIs(self.manager.representatives[2], first_founder)
        self.assertIs(self.manager.representatives[3], second_founder)
        self.assertEqual(self.manager.next_species_id, 4)

    def test_initial_brain_assignment_registers_first_genome(self) -> None:
        controller = object.__new__(NeatBrainController)
        controller.brains = {}
        controller.config = SimpleNamespace()
        controller.species_manager = ContinuousSpeciesManager(3.0)
        first_genome = FakeGenome()
        second_genome = FakeGenome()
        controller._initial_genomes = lambda: [
            (10, first_genome),
            (11, second_genome),
        ]

        from src import neat_controller as controller_module

        original_from_genome = controller_module.NeatBrain.from_genome
        controller_module.NeatBrain.from_genome = lambda *args: SimpleNamespace()
        try:
            controller.assign_initial_brains([1, 2])
        finally:
            controller_module.NeatBrain.from_genome = original_from_genome

        self.assertIs(controller.species_manager.representatives[1], first_genome)
        self.assertEqual(set(controller.brains), {1, 2})

    def test_child_brain_creation_returns_species_result(self) -> None:
        controller = object.__new__(NeatBrainController)
        genome_config = object()
        controller.config = SimpleNamespace(genome_config=genome_config)
        controller.population = SimpleNamespace(population={1: FakeGenome(key=1)})
        parent_genome = FakeGenome(distance=3.1, key=1)
        controller.brains = {
            10: SimpleNamespace(genome=parent_genome, genome_id=1),
        }
        controller.species_manager = ContinuousSpeciesManager(3.0)
        controller.species_manager.register_initial_representative(parent_genome)

        from src import neat_controller as controller_module

        original_from_genome = controller_module.NeatBrain.from_genome
        controller_module.NeatBrain.from_genome = (
            lambda genome_id, genome, config: SimpleNamespace(
                genome_id=genome_id,
                genome=genome,
                config=config,
            )
        )
        try:
            brain, species_id, is_new_species = controller.create_child_brain(
                10,
                20,
                1,
            )
        finally:
            controller_module.NeatBrain.from_genome = original_from_genome

        self.assertIsNotNone(brain)
        self.assertEqual(species_id, 2)
        self.assertTrue(is_new_species)
        self.assertIs(controller.brains[20], brain)


if __name__ == "__main__":
    unittest.main()
