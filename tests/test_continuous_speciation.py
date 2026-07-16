from __future__ import annotations

from math import pi
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

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

from configs.sim_config import TraitConfig, VisionConfig
from src.creature import FlockingTraits, PhysicalTraits, VisionTraits
from src.neat_controller import (
    ContinuousSpeciesManager,
    NeatBrainController,
    calculate_phenotypic_distance,
    calculate_phenotypic_distance_components,
)
from src.speciation import extract_neural_shifts, summarize_neat_changes


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


class FakeGene:
    def __init__(
        self,
        *,
        weight: float = 0.0,
        enabled: bool = True,
        bias: float = 0.0,
        activation: str = "sigmoid",
        aggregation: str = "sum",
    ) -> None:
        self.weight = weight
        self.enabled = enabled
        self.bias = bias
        self.response = 1.0
        self.activation = activation
        self.aggregation = aggregation


def physical(radius: float = 16.0, movement: float = 1.0) -> PhysicalTraits:
    return PhysicalTraits(
        radius=radius,
        movement_cost_multiplier=movement,
    )


def vision(range_: float = 100.0, angle: float = 1.0) -> VisionTraits:
    return VisionTraits(range=range_, angle=angle)


class PhenotypicDistanceTest(unittest.TestCase):
    def test_radius_and_vision_range_are_normalized_by_configured_ranges(self) -> None:
        trait_config = TraitConfig()
        vision_config = VisionConfig()
        representative_physical = physical(radius=trait_config.min_radius)
        representative_vision = vision(range_=vision_config.min_range)
        radius_delta = 2.0
        vision_range_delta = 15.0

        radius_distance = calculate_phenotypic_distance(
            physical(radius=trait_config.min_radius + radius_delta),
            representative_vision,
            representative_physical,
            representative_vision,
            trait_config,
            vision_config,
        )
        vision_distance = calculate_phenotypic_distance(
            representative_physical,
            vision(range_=vision_config.min_range + vision_range_delta),
            representative_physical,
            representative_vision,
            trait_config,
            vision_config,
        )

        self.assertAlmostEqual(
            radius_distance,
            radius_delta / (trait_config.max_radius - trait_config.min_radius),
        )
        self.assertAlmostEqual(
            vision_distance,
            vision_range_delta / (vision_config.max_range - vision_config.min_range),
        )

    def test_values_are_clamped_and_custom_ranges_are_used(self) -> None:
        trait_config = TraitConfig(
            min_radius=10.0,
            max_radius=20.0,
            min_movement_cost_multiplier=0.5,
            max_movement_cost_multiplier=1.5,
        )
        vision_config = VisionConfig(
            min_range=50.0,
            max_range=100.0,
            min_angle=0.0,
            max_angle=pi,
        )

        distance = calculate_phenotypic_distance(
            physical(radius=25.0, movement=2.0),
            vision(range_=125.0, angle=2.0 * pi),
            physical(radius=5.0, movement=0.0),
            vision(range_=25.0, angle=-pi),
            trait_config,
            vision_config,
        )

        self.assertEqual(distance, 4.0)

        components = calculate_phenotypic_distance_components(
            physical(radius=25.0, movement=2.0),
            vision(range_=125.0, angle=2.0 * pi),
            physical(radius=5.0, movement=0.0),
            vision(range_=25.0, angle=-pi),
            trait_config,
            vision_config,
        )
        self.assertEqual(
            sum(
                (
                    components.radius,
                    components.vision_range,
                    components.vision_angle,
                    components.movement_cost_multiplier,
                )
            ),
            distance,
        )


class ContinuousSpeciesManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.trait_config = TraitConfig()
        self.vision_config = VisionConfig()
        self.manager = ContinuousSpeciesManager(
            compatibility_threshold=3.0,
            phenotypic_weight=2.0,
            trait_config=self.trait_config,
            vision_config=self.vision_config,
        )
        self.representative = FakeGenome()
        self.representative_physical = physical(
            radius=self.trait_config.min_radius,
            movement=self.trait_config.min_movement_cost_multiplier,
        )
        self.representative_vision = vision(
            range_=self.vision_config.min_range,
            angle=self.vision_config.min_angle,
        )
        self.manager.register_initial_representative(
            self.representative,
            self.representative_physical,
            self.representative_vision,
        )

    def evaluate(
        self,
        child: FakeGenome,
        child_physical: PhysicalTraits | None = None,
        child_vision: VisionTraits | None = None,
        parent_species_id: int = 1,
    ) -> object:
        return self.manager.evaluate_species(
            child,
            child_physical or self.representative_physical,
            child_vision or self.representative_vision,
            parent_species_id,
            object(),
        )

    def test_distance_below_threshold_keeps_parent_species(self) -> None:
        result = self.evaluate(FakeGenome(distance=2.99))
        self.assertEqual(result.species_id, 1)
        self.assertFalse(result.is_new_species)
        self.assertEqual(set(self.manager.representatives), {1})

    def test_distance_equal_to_threshold_keeps_parent_species(self) -> None:
        with patch(
            "src.neat_controller.extract_neural_shifts",
            side_effect=AssertionError("diff must remain lazy"),
        ):
            result = self.evaluate(FakeGenome(distance=3.0))
        self.assertEqual(result.species_id, 1)
        self.assertFalse(result.is_new_species)

    def test_brain_only_change_creates_species(self) -> None:
        result = self.evaluate(FakeGenome(distance=3.5))
        self.assertEqual(result.species_id, 2)
        self.assertTrue(result.is_new_species)

    def test_evaluation_reads_runtime_threshold_changes(self) -> None:
        self.manager.compatibility_threshold = 3.1

        result = self.evaluate(FakeGenome(distance=3.05))

        self.assertEqual(result.species_id, 1)
        self.assertFalse(result.is_new_species)
        self.assertEqual(result.distances.compatibility_threshold, 3.1)

    def test_body_only_change_creates_species(self) -> None:
        result = self.evaluate(
            FakeGenome(distance=0.0),
            physical(
                radius=self.trait_config.max_radius,
                movement=self.trait_config.max_movement_cost_multiplier,
            ),
        )

        self.assertEqual(result.species_id, 2)
        self.assertTrue(result.is_new_species)

    def test_composite_change_creates_species(self) -> None:
        result = self.evaluate(
            FakeGenome(distance=1.8),
            physical(radius=19.0, movement=0.75),
        )

        self.assertEqual(result.species_id, 2)
        self.assertTrue(result.is_new_species)
        self.assertAlmostEqual(result.distances.neat_distance, 1.8)
        self.assertAlmostEqual(result.distances.phenotypic_distance, 0.7)
        self.assertAlmostEqual(
            result.distances.weighted_phenotypic_distance,
            1.4,
        )
        self.assertAlmostEqual(result.distances.composite_distance, 3.2)
        self.assertAlmostEqual(result.trait_deltas.radius, 7.0)

    def test_new_species_store_complete_founder_representatives(self) -> None:
        first_founder = FakeGenome(distance=3.01)
        first_physical = physical(radius=14.0)
        first_vision = vision(range_=95.0)
        second_founder = FakeGenome(distance=4.0)

        first_result = self.evaluate(
            first_founder,
            first_physical,
            first_vision,
        )
        second_result = self.evaluate(
            second_founder,
            first_physical,
            first_vision,
            parent_species_id=2,
        )
        self.assertEqual(first_result.species_id, 2)
        self.assertTrue(first_result.is_new_species)
        self.assertEqual(second_result.species_id, 3)
        self.assertTrue(second_result.is_new_species)
        stored_genome, stored_physical, stored_vision, stored_flocking = (
            self.manager.representatives[2]
        )
        self.assertIs(stored_genome, first_founder)
        self.assertEqual(stored_physical, first_physical)
        self.assertEqual(stored_vision, first_vision)
        self.assertIsNot(stored_physical, first_physical)
        self.assertIsNot(stored_vision, first_vision)
        self.assertEqual(stored_flocking, FlockingTraits())
        self.assertEqual(self.manager.next_species_id, 4)

    def test_initial_brain_assignment_uses_first_creature_as_luca(self) -> None:
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
        creatures = [
            SimpleNamespace(
                creature_id=1,
                physical_traits=physical(radius=13.0),
                vision=vision(range_=91.0),
            ),
            SimpleNamespace(
                creature_id=2,
                physical_traits=physical(radius=21.0),
                vision=vision(range_=150.0),
            ),
        ]

        from src import neat_controller as controller_module

        original_from_genome = controller_module.NeatBrain.from_genome
        controller_module.NeatBrain.from_genome = lambda *args: SimpleNamespace()
        try:
            controller.assign_initial_brains(creatures)
        finally:
            controller_module.NeatBrain.from_genome = original_from_genome

        representative = controller.species_manager.representatives[1]
        self.assertIs(representative[0], first_genome)
        self.assertEqual(representative[1], creatures[0].physical_traits)
        self.assertEqual(representative[2], creatures[0].vision)
        self.assertEqual(representative[3], FlockingTraits())
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
        child_physical = physical()
        child_vision = vision()
        controller.species_manager.register_initial_representative(
            parent_genome,
            child_physical,
            child_vision,
        )

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
            brain, result = controller.create_child_brain(
                10,
                20,
                1,
                child_physical,
                child_vision,
            )
        finally:
            controller_module.NeatBrain.from_genome = original_from_genome

        self.assertIsNotNone(brain)
        self.assertIsNotNone(result)
        self.assertEqual(result.species_id, 2)
        self.assertTrue(result.is_new_species)
        self.assertIs(controller.brains[20], brain)

    def test_flocking_gene_distance_contributes_to_species_compatibility(
        self,
    ) -> None:
        self.manager.compatibility_threshold = 0.49
        self.manager.flocking_trait_distance_coefficient = 1.0

        result = self.manager.evaluate_species(
            FakeGenome(distance=0.0),
            self.representative_physical,
            self.representative_vision,
            1,
            object(),
            FlockingTraits(1.0, 1.0, 1.0),
        )

        self.assertTrue(result.is_new_species)
        self.assertEqual(result.distances.flocking_trait_distance, 0.5)
        self.assertEqual(result.distances.composite_distance, 0.5)


class NeatChangeSummaryTest(unittest.TestCase):
    def test_neural_shifts_filter_jitter_and_keep_structural_changes(self) -> None:
        parent = FakeGenome()
        parent.connections = {
            (-1, 0): FakeGene(weight=0.2),
            (-2, 7): FakeGene(weight=0.8),
            (-3, 1): FakeGene(weight=-0.4),
        }
        child = FakeGenome()
        child.connections = {
            (-1, 0): FakeGene(weight=0.7),
            (-2, 7): FakeGene(weight=1.31),
            (-4, 7): FakeGene(weight=-0.9),
        }

        shifts = extract_neural_shifts(parent, child)

        self.assertNotIn((0, -1, "weight", 0.5), shifts)
        self.assertIn((7, -2, "weight", 0.51), shifts)
        self.assertIn((1, -3, "removed", 0.4), shifts)
        self.assertIn((7, -4, "added", -0.9), shifts)

    def test_summarizes_structural_and_parameter_changes(self) -> None:
        parent = FakeGenome()
        parent.nodes = {0: FakeGene(bias=0.1), 1: FakeGene()}
        parent.connections = {
            (-1, 0): FakeGene(weight=0.5),
            (-2, 0): FakeGene(weight=-0.2, enabled=True),
        }
        child = FakeGenome()
        child.nodes = {
            0: FakeGene(bias=0.8, aggregation="max"),
            2: FakeGene(),
        }
        child.connections = {
            (-1, 0): FakeGene(weight=1.0),
            (-2, 0): FakeGene(weight=-0.2, enabled=False),
            (-1, 2): FakeGene(weight=0.3),
        }

        summary = summarize_neat_changes(parent, child)

        self.assertEqual(summary.nodes_added, 1)
        self.assertEqual(summary.nodes_removed, 1)
        self.assertEqual(summary.connections_added, 1)
        self.assertEqual(summary.connections_disabled, 1)
        self.assertEqual(summary.weights_changed, 1)
        self.assertEqual(summary.node_parameters_changed, 2)
        self.assertEqual(
            summary.key_changes,
            (
                "Node 1 removed",
                "Node 2 added",
                "Connection -1->2 added",
                "Connection -2->0 disabled",
                "Weight -1->0 +0.500 -> +1.000",
                "Node 0 aggregation sum -> max",
            ),
        )

    def test_key_changes_are_bounded_and_unchanged_genomes_are_empty(self) -> None:
        genome = FakeGenome()
        genome.nodes = {0: FakeGene()}
        genome.connections = {
            (index, 0): FakeGene(weight=float(index))
            for index in range(10)
        }
        changed = FakeGenome()
        changed.nodes = genome.nodes.copy()
        changed.connections = {
            key: FakeGene(weight=gene.weight + 1.0)
            for key, gene in genome.connections.items()
        }

        self.assertEqual(
            summarize_neat_changes(genome, genome).weights_changed,
            0,
        )
        self.assertEqual(
            len(summarize_neat_changes(genome, changed).key_changes),
            6,
        )

    def test_large_genome_summary_keeps_exact_counts_and_bounded_details(
        self,
    ) -> None:
        parent = FakeGenome()
        parent.nodes = {0: FakeGene()}
        parent.connections = {
            (index, 0): FakeGene(weight=float(index))
            for index in range(10_000)
        }
        child = FakeGenome()
        child.nodes = parent.nodes
        child.connections = {
            key: FakeGene(weight=gene.weight + 1.0)
            for key, gene in parent.connections.items()
        }
        child.connections.update(
            {
                (index, 1): FakeGene(weight=1.0)
                for index in range(500)
            }
        )

        summary = summarize_neat_changes(parent, child)

        self.assertEqual(summary.connections_added, 500)
        self.assertEqual(summary.weights_changed, 10_000)
        self.assertEqual(len(summary.key_changes), 6)
        self.assertTrue(
            all(
                change.startswith("Connection")
                for change in summary.key_changes
            )
        )


if __name__ == "__main__":
    unittest.main()
