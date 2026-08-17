from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from types import ModuleType, SimpleNamespace
import sys
import unittest

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import PopulationConfig
from src.creature import LineageInfo
from src.fitness import CreatureFitness
from src.rt_neat import RtNeatManager


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0
    total_energy_gathered: float = 0.0
    lineage: LineageInfo = field(default_factory=LineageInfo)


class FakeBrainController:
    def __init__(self) -> None:
        """Exercise init behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the init test intent explicit.
        self.brains = {
            1: SimpleNamespace(
                genome=SimpleNamespace(
                    nodes={1: object(), 2: object(), 3: object()},
                    connections={
                        1: SimpleNamespace(enabled=True),
                        2: SimpleNamespace(enabled=False),
                    },
                )
            ),
            2: SimpleNamespace(
                genome=SimpleNamespace(
                    nodes={1: object(), 2: object(), 3: object(), 4: object()},
                    connections={
                        1: SimpleNamespace(enabled=True),
                        2: SimpleNamespace(enabled=True),
                        3: SimpleNamespace(enabled=True),
                    },
                )
            ),
        }

    def brain_for(self, creature_id: int) -> object | None:
        """Exercise brain for behavior.
        
        Parameters
        ----------
        creature_id
            Value supplied to ``creature_id`` by the test scenario.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the brain for test intent explicit.
        return self.brains.get(creature_id)


class RtNeatFitnessRankingTest(unittest.TestCase):
    def test_update_stats_keeps_non_ranked_eligible_parent_snapshot(self) -> None:
        """Exercise test update stats keeps non ranked eligible parent snapshot behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test update stats keeps non ranked eligible parent snapshot test intent explicit.
        manager = RtNeatManager(brain_controller=None)
        population_config = PopulationConfig(
            min_reproduction_age=0.0,
            reproduction_cooldown=0.0,
            reproduction_energy_threshold=0.0,
        )
        manager.update_stats(
            creatures=[
                FakeCreature(creature_id=1, total_energy_gathered=1.0),
                FakeCreature(creature_id=2, total_energy_gathered=3.0),
            ],
            fitness_by_creature_id={
                1: CreatureFitness(),
                2: CreatureFitness(),
            },
            population_config=population_config,
        )

        self.assertEqual(manager.eligible_parent_ids, [1, 2])
        self.assertEqual(manager.stats.best_eligible_parent_id, 2)
        self.assertEqual(manager.stats.best_creature_id, 2)
        self.assertAlmostEqual(manager.stats.best_fitness, 3.0)
        self.assertAlmostEqual(manager.stats.worst_fitness, 1.0)

    def test_update_stats_does_not_share_fitness_across_species(self) -> None:
        """Exercise test update stats does not share fitness across species behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test update stats does not share fitness across species test intent explicit.
        manager = RtNeatManager(brain_controller=None)
        population_config = PopulationConfig(
            min_reproduction_age=0.0,
            reproduction_cooldown=0.0,
            reproduction_energy_threshold=0.0,
        )
        creatures = [
            FakeCreature(
                creature_id=creature_id,
                total_energy_gathered=100.0,
                lineage=LineageInfo(species_id=1),
            )
            for creature_id in range(1, 11)
        ]
        creatures.append(
            FakeCreature(
                creature_id=11,
                total_energy_gathered=30.0,
                lineage=LineageInfo(species_id=2),
            )
        )
        fitness_by_creature_id = {
            creature_id: CreatureFitness() for creature_id in range(1, 12)
        }

        manager.update_stats(
            creatures=creatures,
            fitness_by_creature_id=fitness_by_creature_id,
            population_config=population_config,
        )

        self.assertEqual(len(manager.eligible_parent_ids), 11)
        self.assertEqual(manager.stats.eligible_parent_count, 11)
        self.assertEqual(manager.eligible_parent_ids[0], 1)
        self.assertEqual(manager.stats.best_eligible_parent_id, 1)
        self.assertEqual(manager.stats.best_creature_id, 1)
        self.assertAlmostEqual(manager.stats.best_fitness, 100.0)

    def test_select_parent_returns_none_for_empty_pool(self) -> None:
        """Exercise test select parent returns none for empty pool behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test select parent returns none for empty pool test intent explicit.
        manager = RtNeatManager(brain_controller=None, rng=Random(7))

        self.assertIsNone(manager.select_parent([]))

    def test_noneligible_creatures_are_never_selected(self) -> None:
        """Exercise test noneligible creatures are never selected behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test noneligible creatures are never selected test intent explicit.
        manager = RtNeatManager(brain_controller=None, rng=Random(7))
        config = PopulationConfig(
            min_reproduction_age=20.0,
            reproduction_cooldown=12.0,
            reproduction_energy_threshold=0.8,
        )
        creatures = [
            FakeCreature(creature_id=1, energy=0.79, total_energy_gathered=99.0),
            FakeCreature(creature_id=2, energy=1.0, total_energy_gathered=2.0),
        ]
        fitness = {
            1: CreatureFitness(age_seconds=30.0),
            2: CreatureFitness(age_seconds=30.0),
        }
        eligible = [
            creature
            for creature in creatures
            if manager.is_reproduction_eligible(
                creature,
                fitness[creature.creature_id],
                config,
            )
        ]

        self.assertEqual(manager.select_parent(eligible).creature_id, 2)

    def test_parsimony_round_prefers_smaller_enabled_network(self) -> None:
        """Exercise test parsimony round prefers smaller enabled network behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test parsimony round prefers smaller enabled network test intent explicit.
        controller = FakeBrainController()
        manager = RtNeatManager(brain_controller=controller)
        manager.rng = SimpleNamespace(sample=lambda pool, count: pool[:count])
        creatures = [
            FakeCreature(creature_id=1, total_energy_gathered=10.0),
            FakeCreature(creature_id=2, total_energy_gathered=10.0),
            FakeCreature(creature_id=3, total_energy_gathered=10.0),
            FakeCreature(creature_id=4, total_energy_gathered=10.0),
        ]

        selected = manager.select_parent(creatures, k1=3, k2=2)

        self.assertEqual(manager.network_size(creatures[0]), (3, 1))
        self.assertEqual(selected.creature_id, 1)

    def test_small_pool_breaks_energy_ties_by_complexity(self) -> None:
        """Exercise test small pool breaks energy ties by complexity behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test small pool breaks energy ties by complexity test intent explicit.
        manager = RtNeatManager(
            brain_controller=FakeBrainController(),
            rng=Random(7),
        )
        creatures = [
            FakeCreature(creature_id=1, total_energy_gathered=10.0),
            FakeCreature(creature_id=2, total_energy_gathered=10.0),
        ]

        selected = manager.select_parent(creatures, k1=3, k2=2)

        self.assertEqual(selected.creature_id, 1)

    def test_tournament_sizes_must_be_positive_and_ordered(self) -> None:
        """Exercise test tournament sizes must be positive and ordered behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test tournament sizes must be positive and ordered test intent explicit.
        with self.assertRaises(ValueError):
            PopulationConfig(tournament_k1=0)
        with self.assertRaises(ValueError):
            PopulationConfig(tournament_k1=2, tournament_k2=3)

    def test_update_stats_tracks_trend_metrics(self) -> None:
        """Exercise test update stats tracks trend metrics behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test update stats tracks trend metrics test intent explicit.
        manager = RtNeatManager(brain_controller=FakeBrainController())
        manager.record_normal_replacement()
        manager.record_extinction_replacements(2)
        manager.record_death(CreatureFitness(age_seconds=20.0))
        manager.record_death(CreatureFitness(age_seconds=40.0))

        manager.update_stats(
            creatures=[
                FakeCreature(creature_id=1),
                FakeCreature(creature_id=2),
            ],
            fitness_by_creature_id={
                1: CreatureFitness(age_seconds=10.0, distance_traveled=50.0),
                2: CreatureFitness(age_seconds=20.0, distance_traveled=150.0),
            },
            population_config=PopulationConfig(
                min_reproduction_age=0.0,
                reproduction_cooldown=0.0,
                reproduction_energy_threshold=0.0,
            ),
            elapsed_time=120.0,
        )

        self.assertEqual(manager.stats.births, 3)
        self.assertEqual(manager.stats.normal_replacements, 1)
        self.assertEqual(manager.stats.extinction_replacements, 2)
        self.assertEqual(manager.stats.deaths, 2)
        self.assertAlmostEqual(manager.stats.average_lifespan_at_death, 30.0)
        self.assertAlmostEqual(manager.stats.average_speed, 6.25)
        self.assertAlmostEqual(manager.stats.average_distance_traveled, 100.0)
        self.assertAlmostEqual(manager.stats.average_brain_nodes, 3.5)
        self.assertAlmostEqual(manager.stats.average_brain_enabled_connections, 2.0)
        self.assertAlmostEqual(manager.stats.average_brain_connections, 2.5)
        self.assertAlmostEqual(manager.stats.births_per_minute, 1.5)
        self.assertAlmostEqual(manager.stats.deaths_per_minute, 1.0)


if __name__ == "__main__":
    unittest.main()
