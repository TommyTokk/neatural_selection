"""Tests for aggregate non-neural creature genotype behavior."""

from __future__ import annotations

import unittest

import pymunk

from src.creature import (
    Creature,
    CreatureGenotype,
    FlockingTraits,
    PhysicalTraits,
    VisionTraits,
)


class CreatureGenotypeTest(unittest.TestCase):
    """Verify aggregate storage and legacy access remain equivalent."""

    def _physics(self, radius: float = 4.0) -> tuple[pymunk.Body, pymunk.Circle]:
        """Create minimal physics values for entity construction.
        
        Parameters
        ----------
        radius
            Radius of the test collision circle.
        
        Returns
        -------
        tuple[pymunk.Body, pymunk.Circle]
            Detached body and circular shape.
        """
        # Keep the physics test intent explicit.
        # The unit test does not need to insert these values into a space.
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
        return body, pymunk.Circle(body, radius)

    def test_aggregate_genotype_is_the_flat_property_source(self) -> None:
        """Route legacy reads and writes through one aggregate genotype.
        
        Parameters
        ----------
        None
            This test receives no external parameters.
        
        Returns
        -------
        None
            Assertions validate property identity and mutation.
        """
        # Keep the test aggregate genotype is the flat property source test intent explicit.
        # Construct through the new explicit aggregate API.
        body, shape = self._physics()
        genotype = CreatureGenotype(
            VisionTraits(120.0, 1.2),
            PhysicalTraits(4.0),
            FlockingTraits(),
            (10, 20, 30),
        )
        creature = Creature(1, "test", body, shape, 0.5, genotype=genotype)

        self.assertIs(creature.vision, genotype.vision)
        self.assertIs(creature.physical_traits, genotype.physical_traits)
        creature.color = (40, 50, 60)
        self.assertEqual((40, 50, 60), genotype.color)

    def test_legacy_constructor_builds_an_aggregate(self) -> None:
        """Accept existing flat trait constructor arguments unchanged.
        
        Parameters
        ----------
        None
            This test receives no external parameters.
        
        Returns
        -------
        None
            Assertions validate the normalized aggregate.
        """
        # Keep the test legacy constructor builds an aggregate test intent explicit.
        # Legacy callers remain valid while runtime storage uses the aggregate.
        body, shape = self._physics(5.0)
        creature = Creature(
            2,
            "legacy",
            body,
            shape,
            0.5,
            VisionTraits(90.0, 0.9),
            PhysicalTraits(5.0),
            (70, 80, 90),
        )
        self.assertIsInstance(creature.genotype, CreatureGenotype)
        self.assertEqual(5.0, creature.radius)

    def test_mixed_constructor_styles_are_rejected(self) -> None:
        """Reject ambiguous aggregate and flat genotype inputs.
        
        Parameters
        ----------
        None
            This test receives no external parameters.
        
        Returns
        -------
        None
            Assertions validate the explicit constructor failure.
        """
        # Keep the test mixed constructor styles are rejected test intent explicit.
        # Mixing representations could silently create two genotype sources.
        body, shape = self._physics()
        genotype = CreatureGenotype(
            VisionTraits(100.0, 1.0),
            PhysicalTraits(4.0),
            FlockingTraits(),
            (1, 2, 3),
        )
        with self.assertRaises(ValueError):
            Creature(
                3,
                "mixed",
                body,
                shape,
                0.5,
                vision=VisionTraits(80.0, 0.8),
                genotype=genotype,
            )
