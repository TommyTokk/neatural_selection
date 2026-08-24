"""Regression tests for bounded mutation and dynamic species centroids."""

from __future__ import annotations

import math
from collections import Counter
from random import Random
import unittest

from configs.sim_config import FlockingConfig, SimConfig, TraitConfig, VisionConfig
from src.creature.genotype import (
    FlockingTraits,
    GenotypeManager,
    PhysicalTraits,
    VisionTraits,
    mutate_logit_bounded,
)
from src.creature.speciation import ContinuousSpeciesManager


class FakeGenome:
    """Minimal genome whose compatibility distance is one-dimensional."""

    def __init__(self, position: float, key: int = 0) -> None:
        self.position = position
        self.key = key
        self.connections = {}

    def distance(self, other: FakeGenome, _config: object) -> float:
        """Return absolute distance between deterministic test positions."""
        return abs(self.position - other.position)


def physical(value: float) -> PhysicalTraits:
    """Create a compact physical-trait fixture around one scalar value."""
    return PhysicalTraits(
        radius=value,
        movement_cost_multiplier=1.0,
        stomach_capacity=1.6,
        digestion_rate=0.2,
        digestion_efficiency=0.9,
    )


class LogitMutationTest(unittest.TestCase):
    """Verify logit mutation remains strictly inside configured bounds."""

    def test_boundary_mutations_have_no_exact_boundary_accumulation(self) -> None:
        """Mutate each endpoint 100,000 times without producing boundary atoms."""
        for boundary in (0.0, 1.0):
            rng = Random(734 + int(boundary))
            values = [
                mutate_logit_bounded(boundary, 0.0, 1.0, 0.20, rng)
                for _ in range(100_000)
            ]
            self.assertTrue(all(0.0 < value < 1.0 for value in values))
            self.assertGreater(len(set(values)), 1000)
            frequencies = Counter(values)
            self.assertLess(max(frequencies.values()), 10)
            epsilon_value = 1e-6 if boundary == 0.0 else 1.0 - 1e-6
            self.assertTrue(any(value < epsilon_value for value in values))
            self.assertTrue(any(value > epsilon_value for value in values))
            if boundary == 0.0:
                self.assertGreater(sum(values) / len(values), 0.0)
            else:
                self.assertLess(sum(values) / len(values), 1.0)

    def test_locked_span_returns_minimum_without_consuming_rng(self) -> None:
        """Treat numerically locked traits as constants and preserve RNG state."""
        rng = Random(9)
        before = rng.getstate()
        self.assertEqual(
            mutate_logit_bounded(4.0, 2.0, 2.0 + 1e-10, 0.20, rng),
            2.0,
        )
        self.assertEqual(rng.getstate(), before)

    def test_extreme_latent_noise_is_overflow_safe_and_interior(self) -> None:
        """Keep the sigmoid representable for extreme custom Gaussian values."""
        class ExtremeRng:
            def gauss(self, _mean: float, _sigma: float) -> float:
                return -10_000.0

        result = mutate_logit_bounded(0.5, 0.0, 1.0, 0.20, ExtremeRng())
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)


class MutationGateTest(unittest.TestCase):
    """Verify mutually exclusive gates retain deterministic RNG ordering."""

    def test_uniform_replacement_gates_consume_one_roll_and_one_draw_each(self) -> None:
        """Replace all genes and tags without touching the Gaussian stream."""
        config = SimConfig()
        config.flocking.flocking_gene_replace_rate = 1.0
        config.flocking.flocking_gene_mutation_rate = 0.0
        config.flocking.social_tag_replace_rate = 1.0
        config.flocking.social_tag_mutation_rate = 0.0
        manager = GenotypeManager(config, ((1, 2, 3),))

        class ReplacementRng:
            def __init__(self) -> None:
                self.values = iter((0.1, 0.2, 0.3, 0.4, 0.5))
                self.events: list[str] = []

            def random(self) -> float:
                self.events.append("gate")
                return 0.0

            def uniform(self, minimum: float, maximum: float) -> float:
                self.assert_unit_bounds(minimum, maximum)
                self.events.append("replace")
                return next(self.values)

            @staticmethod
            def assert_unit_bounds(minimum: float, maximum: float) -> None:
                if (minimum, maximum) != (0.0, 1.0):
                    raise AssertionError("replacement bounds changed")

            def gauss(self, _mean: float, _sigma: float) -> float:
                raise AssertionError("replacement must not consume Gaussian RNG")

        rng = ReplacementRng()
        child, delta = manager.mutate_flocking_traits(FlockingTraits(), rng)

        self.assertEqual(
            child,
            FlockingTraits(0.1, 0.2, 0.3, 0.4, 0.5),
        )
        self.assertEqual(delta.separation_gene, -0.4)
        self.assertEqual(
            rng.events,
            ["gate", "replace"] * 5,
        )

    def test_seeded_logit_perturbations_and_physical_deltas_are_repeatable(self) -> None:
        """Repeat the same mutation stream and record exact physical deltas."""
        config = SimConfig()
        config.flocking.flocking_gene_replace_rate = 0.0
        config.flocking.flocking_gene_mutation_rate = 1.0
        config.flocking.social_tag_replace_rate = 0.0
        config.flocking.social_tag_mutation_rate = 1.0
        manager = GenotypeManager(config, ((1, 2, 3),))
        parent = FlockingTraits(0.0, 1.0, 0.5, 0.25, 0.75)

        first_child, first_delta = manager.mutate_flocking_traits(
            parent,
            Random(42),
        )
        second_child, second_delta = manager.mutate_flocking_traits(
            parent,
            Random(42),
        )

        self.assertEqual(first_child, second_child)
        self.assertEqual(first_delta, second_delta)
        self.assertTrue(
            all(
                0.0 < value < 1.0
                for value in (
                    first_child.separation_gene,
                    first_child.alignment_gene,
                    first_child.cohesion_gene,
                    first_child.social_tag_x,
                    first_child.social_tag_y,
                )
            )
        )
        self.assertAlmostEqual(
            first_delta.social_tag_x,
            first_child.social_tag_x - parent.social_tag_x,
        )


class DynamicSpeciationTest(unittest.TestCase):
    """Verify global assignment and physical-unit EMA tuple behavior."""

    def manager(self, threshold: float = 2.0) -> ContinuousSpeciesManager:
        """Return a deterministic manager with default trait ranges."""
        return ContinuousSpeciesManager(
            threshold,
            trait_config=TraitConfig(),
            vision_config=VisionConfig(),
        )

    def test_global_assignment_ignores_parent_species(self) -> None:
        """Assign a child to the closest active species, not its parent species."""
        manager = self.manager(0.25)
        for species_id, position, radius in (
            (1, 0.0, 12.0),
            (2, 5.0, 17.0),
            (3, 10.0, 22.0),
        ):
            manager.register_initial_representative(
                FakeGenome(position, species_id),
                physical(radius),
                VisionTraits(150.0, 0.95),
                species_id=species_id,
            )
        manager.next_species_id = 4

        child = FakeGenome(5.0, 20)
        result = manager.evaluate_species(
            child,
            physical(17.0),
            VisionTraits(150.0, 0.95),
            1,
            object(),
            FlockingTraits(),
            {1, 2, 3},
        )

        self.assertEqual(result.species_id, 2)
        self.assertFalse(result.is_new_species)
        self.assertIs(manager.representatives[2][0], child)

    def test_equal_global_distances_choose_lowest_species_id(self) -> None:
        """Resolve exact compatibility ties independently of insertion order."""
        manager = self.manager(1.0)
        for species_id in (2, 1):
            manager.register_initial_representative(
                FakeGenome(0.0, species_id),
                physical(17.0),
                VisionTraits(150.0, 0.95),
                species_id=species_id,
            )
        manager.next_species_id = 3

        result = manager.evaluate_species(
            FakeGenome(0.0, 8),
            physical(17.0),
            VisionTraits(150.0, 0.95),
            2,
            object(),
            FlockingTraits(),
            {1, 2},
        )

        self.assertEqual(result.species_id, 1)
        self.assertEqual(result.parent_species_id, 2)

    def test_distance_normalizes_physical_values_only_when_evaluated(self) -> None:
        """Expose physical radii while evaluating their configured span fraction."""
        manager = self.manager(2.0)
        manager.register_initial_representative(
            FakeGenome(0.0, 1),
            physical(12.0),
            VisionTraits(150.0, 0.95),
            flocking_traits=FlockingTraits(0.5, 0.5, 0.5, 0.0, 0.0),
        )

        result = manager.evaluate_species(
            FakeGenome(0.0, 2),
            physical(17.0),
            VisionTraits(150.0, 0.95),
            1,
            object(),
            FlockingTraits(0.5, 0.5, 0.5, 1.0, 1.0),
            {1},
        )

        self.assertAlmostEqual(result.distances.radius_component, 0.5)
        self.assertAlmostEqual(result.distances.composite_distance, 1.0)
        self.assertAlmostEqual(manager.representatives[1][1].radius, 12.5)

    def test_ema_tuple_stays_in_unnormalized_physical_units(self) -> None:
        """Update all centroid fields directly in their physical domains."""
        manager = self.manager(100.0)
        old_physical = PhysicalTraits(12.0, 0.75, 0.8, 0.05, 0.55)
        old_vision = VisionTraits(100.0, 0.35)
        old_flocking = FlockingTraits(0.0, 0.0, 0.0, 0.1, 0.2)
        manager.register_initial_representative(
            FakeGenome(0.0, 1),
            old_physical,
            old_vision,
            flocking_traits=old_flocking,
        )
        child_genome = FakeGenome(0.0, 2)
        child_physical = PhysicalTraits(22.0, 1.35, 2.6, 0.4, 0.98)
        child_vision = VisionTraits(200.0, math.pi)
        child_flocking = FlockingTraits(1.0, 1.0, 1.0, 0.8, 0.9)

        manager.evaluate_species(
            child_genome,
            child_physical,
            child_vision,
            1,
            object(),
            child_flocking,
            {1},
        )

        representative = manager.representatives[1]
        self.assertEqual(len(representative), 4)
        genome, centroid_physical, centroid_vision, centroid_flocking = representative
        self.assertIs(genome, child_genome)
        self.assertAlmostEqual(centroid_physical.radius, 13.0)
        self.assertAlmostEqual(centroid_physical.movement_cost_multiplier, 0.81)
        self.assertAlmostEqual(centroid_physical.stomach_capacity, 0.98)
        self.assertAlmostEqual(centroid_physical.digestion_rate, 0.085)
        self.assertAlmostEqual(centroid_physical.digestion_efficiency, 0.593)
        self.assertAlmostEqual(centroid_vision.range, 110.0)
        self.assertAlmostEqual(centroid_vision.angle, 0.9 * 0.35 + 0.1 * math.pi)
        self.assertAlmostEqual(centroid_flocking.separation_gene, 0.1)
        self.assertAlmostEqual(centroid_flocking.alignment_gene, 0.1)
        self.assertAlmostEqual(centroid_flocking.cohesion_gene, 0.1)

    def test_recovery_cluster_does_not_radiate_into_many_species(self) -> None:
        """Keep 35 slightly different recovery children in a compact radiation."""
        manager = self.manager(0.5)
        active: set[int] = set()
        new_species = 0
        for index in range(35):
            offset = (index % 5) * 0.01
            result = manager.evaluate_species(
                FakeGenome(offset, index + 1),
                physical(16.0 + offset),
                VisionTraits(150.0 + offset, 0.95),
                1,
                object(),
                FlockingTraits(),
                active,
            )
            active.add(result.species_id)
            new_species += int(result.is_new_species)
        self.assertLessEqual(new_species, math.ceil(35 / 4))


class EvolutionConfigurationTest(unittest.TestCase):
    """Exercise strict construction-time evolution configuration validation."""

    def test_invalid_trait_and_vision_ranges_are_rejected(self) -> None:
        """Reject inverted ranges and defaults outside their physical bounds."""
        with self.assertRaises(ValueError):
            TraitConfig(min_radius=12.0, max_radius=12.0)
        with self.assertRaises(ValueError):
            VisionConfig(default_range=99.0)

    def test_invalid_latent_powers_and_probability_sums_are_rejected(self) -> None:
        """Require positive powers and mutually exclusive mutation gates."""
        with self.assertRaises(ValueError):
            TraitConfig(radius_mutation_sigma_u=0.0)
        with self.assertRaises(ValueError):
            VisionConfig(angle_mutation_sigma_u=float("nan"))
        with self.assertRaises(ValueError):
            FlockingConfig(
                flocking_gene_mutation_rate=0.8,
                flocking_gene_replace_rate=0.3,
            )


if __name__ == "__main__":
    unittest.main()
