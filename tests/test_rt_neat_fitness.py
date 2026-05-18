from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
import sys
import unittest

for optional_module in ("neat", "pymunk"):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import FitnessConfig, PopulationConfig
from src.fitness import CreatureFitness
from src.rt_neat import RtNeatManager


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    energy: float = 1.0


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
            food_eaten_weight=10.0,
            energy_gained_weight=0.0,
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
                1: CreatureFitness(food_eaten=1),
                2: CreatureFitness(food_eaten=3),
            },
            population_config=population_config,
            fitness_config=fitness_config,
        )

        self.assertEqual(manager.eligible_parent_ids, [2, 1])
        self.assertEqual(manager.stats.best_creature_id, 2)
        self.assertAlmostEqual(manager.stats.best_fitness, 30.0)
        self.assertAlmostEqual(manager.stats.worst_fitness, 10.0)


if __name__ == "__main__":
    unittest.main()
