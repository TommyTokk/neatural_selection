from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
import sys
import unittest

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import FitnessConfig, PopulationConfig
from src.creature import LineageInfo
from src.fitness import CreatureFitness
from src.rt_neat import RtNeatManager


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0
    lineage: LineageInfo = field(default_factory=LineageInfo)


class FakeBrainController:
    def __init__(self) -> None:
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
        return self.brains.get(creature_id)


class RtNeatFitnessRankingTest(unittest.TestCase):
    def test_update_stats_ranks_eligible_parents_with_fitness_config(self) -> None:
        manager = RtNeatManager(brain_controller=None)
        population_config = PopulationConfig(
            min_reproduction_age=0.0,
            reproduction_cooldown=0.0,
            reproduction_energy_threshold=0.0,
        )
        fitness_config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=0.0,
            energy_gained_weight=10.0,
            energy_efficiency_weight=0.0,
            movement_effort_penalty=0.0,
            offspring_weight=0.0,
        )

        manager.update_stats(
            creatures=[
                FakeCreature(creature_id=1),
                FakeCreature(creature_id=2),
            ],
            fitness_by_creature_id={
                1: CreatureFitness(energy_gained=1.0),
                2: CreatureFitness(energy_gained=3.0),
            },
            population_config=population_config,
            fitness_config=fitness_config,
        )

        self.assertEqual(manager.eligible_parent_ids, [2, 1])
        self.assertEqual(manager.stats.best_creature_id, 2)
        self.assertAlmostEqual(manager.stats.best_fitness, 30.0)
        self.assertAlmostEqual(manager.stats.worst_fitness, 10.0)

    def test_update_stats_shares_fitness_across_living_species(self) -> None:
        manager = RtNeatManager(brain_controller=None)
        population_config = PopulationConfig(
            min_reproduction_age=0.0,
            reproduction_cooldown=0.0,
            reproduction_energy_threshold=0.0,
        )
        fitness_config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=0.0,
            energy_gained_weight=1.0,
            energy_efficiency_weight=0.0,
            movement_effort_penalty=0.0,
            offspring_weight=0.0,
            matured_offspring_weight=0.0,
            trait_energy_cost_penalty_weight=0.0,
        )
        creatures = [
            FakeCreature(
                creature_id=creature_id,
                lineage=LineageInfo(species_id=1),
            )
            for creature_id in range(1, 11)
        ]
        creatures.append(
            FakeCreature(
                creature_id=11,
                lineage=LineageInfo(species_id=2),
            )
        )
        fitness_by_creature_id = {
            creature_id: CreatureFitness(energy_gained=100.0)
            for creature_id in range(1, 11)
        }
        fitness_by_creature_id[11] = CreatureFitness(energy_gained=30.0)

        manager.update_stats(
            creatures=creatures,
            fitness_by_creature_id=fitness_by_creature_id,
            population_config=population_config,
            fitness_config=fitness_config,
        )

        self.assertEqual(len(manager.eligible_parent_ids), 11)
        self.assertEqual(manager.stats.eligible_parent_count, 11)
        self.assertEqual(manager.eligible_parent_ids[0], 11)
        self.assertEqual(manager.stats.best_eligible_parent_id, 11)
        self.assertEqual(manager.stats.best_creature_id, 1)
        self.assertAlmostEqual(manager.stats.best_fitness, 100.0)

    def test_update_stats_tracks_trend_metrics(self) -> None:
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
            fitness_config=FitnessConfig(),
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
