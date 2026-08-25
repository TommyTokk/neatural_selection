from __future__ import annotations

from dataclasses import dataclass
import unittest

from configs.sim_config import (
    CommunicationConfig,
    MetabolismConfig,
    PheromoneConfig,
    TraitConfig,
)
from src.creature import PhysicalTraits
from src.metabolism import Metabolism


@dataclass(slots=True)
class FakeCreature:
    radius: float
    speed: float
    energy: float
    physical_traits: PhysicalTraits
    creature_id: int = 1


class FakeVision:
    def __init__(self, cost: float) -> None:
        """Exercise init behavior.
        
        Parameters
        ----------
        cost
            Value supplied to ``cost`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the init test intent explicit.
        self.cost = cost

    def energy_cost_per_second(self, creature: FakeCreature) -> float:
        """Exercise energy cost per second behavior.
        
        Parameters
        ----------
        creature
            Value supplied to ``creature`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the energy cost per second test intent explicit.
        del creature
        return self.cost


class MetabolismTraitCostTest(unittest.TestCase):
    def test_default_and_maximum_digestive_upkeep_are_bounded(self) -> None:
        """Exercise test default and maximum digestive upkeep are bounded behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test default and maximum digestive upkeep are bounded test intent explicit.
        config = MetabolismConfig(
            basic_metabolism_rate=0.0,
            movement_energy_cost_factor=0.0,
        )
        traits = TraitConfig()
        metabolism = Metabolism(
            config,
            FakeVision(cost=0.0),
            traits,
        )
        default = FakeCreature(
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(
                radius=16.0,
                stomach_capacity=traits.default_stomach_capacity,
                digestion_rate=traits.default_digestion_rate,
                digestion_efficiency=traits.default_digestion_efficiency,
            ),
        )
        maximum = FakeCreature(
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(
                radius=16.0,
                stomach_capacity=traits.max_stomach_capacity,
                digestion_rate=traits.max_digestion_rate,
                digestion_efficiency=traits.max_digestion_efficiency,
            ),
        )

        default_cost = metabolism.digestive_upkeep_energy_cost_per_second(
            default
        )
        maximum_cost = metabolism.digestive_upkeep_energy_cost_per_second(
            maximum
        )

        self.assertAlmostEqual(default_cost, 0.004)
        self.assertGreater(maximum_cost, default_cost)
        self.assertLessEqual(maximum_cost, 0.012)

    def test_invalid_digestive_configuration_is_rejected(self) -> None:
        """Exercise test invalid digestive configuration is rejected behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test invalid digestive configuration is rejected test intent explicit.
        with self.assertRaises(ValueError):
            Metabolism(
                MetabolismConfig(),
                FakeVision(cost=0.0),
                TraitConfig(digestive_trait_mutation_rate=1.1),
            )
        with self.assertRaises(ValueError):
            Metabolism(
                MetabolismConfig(
                    max_digestion_processing_fraction=0.6,
                ),
                FakeVision(cost=0.0),
                TraitConfig(),
            )
        with self.assertRaises(ValueError):
            Metabolism(
                MetabolismConfig(
                    digestive_upkeep_at_default_per_second=-0.001,
                ),
                FakeVision(cost=0.0),
                TraitConfig(),
            )

    def test_communication_costs_are_charged_and_reported_as_trait_cost(self) -> None:
        """Exercise test communication costs are charged and reported as trait cost behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test communication costs are charged and reported as trait cost test intent explicit.
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.0,
                digestive_upkeep_at_default_per_second=0.0,
            ),
            FakeVision(cost=0.0),
            TraitConfig(body_metabolism_cost_factor=0.0),
            communication_config=CommunicationConfig(
                acoustic_energy_cost_per_second=0.006,
                pheromone=PheromoneConfig(energy_cost_per_second=0.002),
            ),
        )
        creature = FakeCreature(
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )

        cost = metabolism.energy_cost_breakdown_per_second(
            creature,
            max_speed=100.0,
            communication_intensities=(0.5, 1.0, 0.5, 0.0),
        )

        self.assertAlmostEqual(cost.acoustic, 0.0015)
        self.assertAlmostEqual(cost.pheromone, 0.003)
        self.assertAlmostEqual(cost.total, 0.0045)
        self.assertAlmostEqual(cost.trait, 0.0045)
    def test_large_body_and_movement_multiplier_increase_energy_cost(self) -> None:
        """Exercise test large body and movement multiplier increase energy cost behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test large body and movement multiplier increase energy cost test intent explicit.
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.0,
                movement_energy_cost_factor=0.02,
                digestive_upkeep_at_default_per_second=0.0,
            ),
            FakeVision(cost=0.01),
            TraitConfig(
                max_radius=20.0,
                body_metabolism_cost_factor=0.008,
            ),
        )
        efficient_creature = FakeCreature(
            radius=10.0,
            speed=50.0,
            energy=1.0,
            physical_traits=PhysicalTraits(
                radius=10.0,
                movement_cost_multiplier=1.0,
            ),
        )
        costly_creature = FakeCreature(
            radius=20.0,
            speed=50.0,
            energy=1.0,
            physical_traits=PhysicalTraits(
                radius=20.0,
                movement_cost_multiplier=1.5,
            ),
        )

        efficient_cost = metabolism.energy_cost_breakdown_per_second(
            efficient_creature,
            max_speed=100.0,
        )
        costly_cost = metabolism.energy_cost_breakdown_per_second(
            costly_creature,
            max_speed=100.0,
        )

        self.assertGreater(costly_cost.total, efficient_cost.total)
        self.assertGreater(costly_cost.trait, efficient_cost.trait)

    def test_full_sprint_adds_configured_energy_cost(self) -> None:
        """Exercise test full sprint adds configured energy cost behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test full sprint adds configured energy cost test intent explicit.
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.0,
                movement_energy_cost_factor=0.0,
                sprint_energy_cost_per_second=0.04,
                digestive_upkeep_at_default_per_second=0.0,
            ),
            FakeVision(cost=0.0),
            TraitConfig(body_metabolism_cost_factor=0.0),
        )
        creature = FakeCreature(
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )

        normal = metabolism.energy_cost_breakdown_per_second(
            creature,
            max_speed=100.0,
        )
        sprinting = metabolism.energy_cost_breakdown_per_second(
            creature,
            max_speed=100.0,
            sprint_intensity=1.0,
        )

        self.assertAlmostEqual(normal.sprint, 0.0)
        self.assertAlmostEqual(sprinting.sprint, 0.04)
        self.assertAlmostEqual(sprinting.total - normal.total, 0.04)

    def test_energy_cost_multiplier_scales_complete_energy_drain(self) -> None:
        """Exercise test energy cost multiplier scales complete energy drain behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test energy cost multiplier scales complete energy drain test intent explicit.
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.1,
                movement_energy_cost_factor=0.0,
                sprint_energy_cost_per_second=0.0,
                digestive_upkeep_at_default_per_second=0.0,
            ),
            FakeVision(cost=0.0),
            TraitConfig(body_metabolism_cost_factor=0.0),
        )
        creature = FakeCreature(
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )

        metabolism.consume_energy(
            creature,
            delta_time=2.0,
            max_speed=100.0,
            energy_cost_multiplier=1.5,
        )

        self.assertAlmostEqual(creature.energy, 0.7)

    def test_enabled_connections_increase_neural_upkeep_before_multiplier(
        self,
    ) -> None:
        """Exercise test brain complexity increases basal upkeep before multiplier behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test brain complexity increases basal upkeep before multiplier test intent explicit.
        genomes = {
            1: SimpleGenome(node_count=2, connection_count=3),
            2: SimpleGenome(node_count=20, connection_count=30),
        }
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.0,
                brain_upkeep_per_enabled_connection=0.0005,
                movement_energy_cost_factor=0.0,
                sprint_energy_cost_per_second=0.0,
                digestive_upkeep_at_default_per_second=0.0,
            ),
            FakeVision(cost=0.0),
            TraitConfig(body_metabolism_cost_factor=0.0),
            genome_for_creature_id=genomes.get,
        )
        simple = FakeCreature(
            creature_id=1,
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )
        complex_creature = FakeCreature(
            creature_id=2,
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )

        metabolism.consume_energy(simple, delta_time=1.0, max_speed=100.0)
        metabolism.consume_energy(
            complex_creature,
            delta_time=1.0,
            max_speed=100.0,
            energy_cost_multiplier=2.0,
        )

        self.assertAlmostEqual(simple.energy, 1.0 - 0.0015)
        self.assertAlmostEqual(complex_creature.energy, 1.0 - 0.03)
        self.assertLess(complex_creature.energy, simple.energy)

    def test_infant_brain_tax_holiday_skips_complexity_upkeep(self) -> None:
        """Exercise test infant brain tax holiday skips complexity upkeep behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test infant brain tax holiday skips complexity upkeep test intent explicit.
        genomes = {
            1: SimpleGenome(node_count=0, connection_count=0),
            2: SimpleGenome(node_count=20, connection_count=20),
        }
        metabolism = Metabolism(
            MetabolismConfig(
                basic_metabolism_rate=0.01,
                brain_upkeep_per_enabled_connection=0.001,
                movement_energy_cost_factor=0.0,
                sprint_energy_cost_per_second=0.0,
                digestive_upkeep_at_default_per_second=0.0,
            ),
            FakeVision(cost=0.0),
            TraitConfig(body_metabolism_cost_factor=0.0),
            genome_for_creature_id=genomes.get,
        )
        simple_infant = FakeCreature(
            creature_id=1,
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )
        complex_infant = FakeCreature(
            creature_id=2,
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )
        complex_at_boundary = FakeCreature(
            creature_id=2,
            radius=16.0,
            speed=0.0,
            energy=1.0,
            physical_traits=PhysicalTraits(radius=16.0),
        )

        metabolism.consume_energy(
            simple_infant,
            delta_time=1.0,
            max_speed=100.0,
            age_seconds=2.0,
        )
        metabolism.consume_energy(
            complex_infant,
            delta_time=1.0,
            max_speed=100.0,
            age_seconds=2.0,
        )
        metabolism.consume_energy(
            complex_at_boundary,
            delta_time=1.0,
            max_speed=100.0,
            age_seconds=5.0,
        )

        self.assertAlmostEqual(simple_infant.energy, 0.99)
        self.assertAlmostEqual(complex_infant.energy, simple_infant.energy)
        self.assertAlmostEqual(complex_at_boundary.energy, 0.97)


class SimpleGenome:
    def __init__(self, node_count: int, connection_count: int) -> None:
        """Exercise init behavior.
        
        Parameters
        ----------
        node_count
            Value supplied to ``node_count`` by the test scenario.
        connection_count
            Value supplied to ``connection_count`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the init test intent explicit.
        self.nodes = {node_id: object() for node_id in range(node_count)}
        self.connections = {
            connection_id: object()
            for connection_id in range(connection_count)
        }


if __name__ == "__main__":
    unittest.main()
