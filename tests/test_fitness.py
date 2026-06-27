from __future__ import annotations

import unittest

from configs.sim_config import FitnessConfig
from src.fitness import CreatureFitness


class CreatureFitnessScoreTest(unittest.TestCase):
    def test_food_discovery_reward_is_capped(self) -> None:
        config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=1.0,
            food_discovery_cap=3,
            food_eaten_weight=0.0,
            energy_gained_weight=0.0,
            energy_efficiency_weight=0.0,
            movement_effort_penalty=0.0,
            offspring_weight=0.0,
        )
        fitness = CreatureFitness(food_discovered=10)

        self.assertAlmostEqual(fitness.score(config), 3.0)

    def test_energy_efficiency_uses_minimum_age_for_young_creatures(self) -> None:
        config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=0.0,
            food_eaten_weight=0.0,
            energy_gained_weight=0.0,
            energy_efficiency_weight=100.0,
            efficiency_min_age_seconds=20.0,
            movement_effort_penalty=0.0,
            offspring_weight=0.0,
        )
        fitness = CreatureFitness(age_seconds=5.0, energy_gained=1.0)

        self.assertAlmostEqual(fitness.score(config), 5.0)

    def test_eating_and_energy_gain_reward_foraging(self) -> None:
        config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=0.0,
            food_eaten_weight=8.0,
            energy_gained_weight=80.0,
            energy_efficiency_weight=0.0,
            movement_effort_penalty=0.0,
            offspring_weight=0.0,
        )
        fitness = CreatureFitness(food_eaten=2, energy_gained=0.5)

        self.assertAlmostEqual(fitness.score(config), 56.0)

    def test_partial_food_bite_records_energy_without_counting_food_eaten(self) -> None:
        fitness = CreatureFitness()

        fitness.record_food(0.2, depleted=False)
        fitness.record_food(0.3, depleted=True)

        self.assertEqual(fitness.food_eaten, 1)
        self.assertAlmostEqual(fitness.energy_gained, 0.5)

    def test_offspring_bonus_increases_score(self) -> None:
        config = FitnessConfig(offspring_weight=12.0)
        without_child = CreatureFitness(age_seconds=20.0)
        with_child = CreatureFitness(age_seconds=20.0, offspring_count=1)

        self.assertAlmostEqual(
            with_child.score(config) - without_child.score(config),
            12.0,
        )

    def test_matured_offspring_bonus_increases_score(self) -> None:
        config = FitnessConfig(
            offspring_weight=0.0,
            matured_offspring_weight=30.0,
        )
        without_matured_child = CreatureFitness(age_seconds=20.0)
        with_matured_child = CreatureFitness(
            age_seconds=20.0,
            matured_offspring_ids=[7, 8],
        )

        self.assertAlmostEqual(
            with_matured_child.score(config) - without_matured_child.score(config),
            60.0,
        )

    def test_movement_effort_reduces_score(self) -> None:
        config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=0.0,
            food_eaten_weight=0.0,
            energy_gained_weight=0.0,
            energy_efficiency_weight=0.0,
            movement_effort_penalty=0.5,
            offspring_weight=0.0,
        )
        fitness = CreatureFitness(movement_effort=10.0)

        self.assertAlmostEqual(fitness.score(config), -5.0)

    def test_trait_energy_cost_reduces_score(self) -> None:
        config = FitnessConfig(
            age_weight=0.0,
            food_discovery_weight=0.0,
            food_eaten_weight=0.0,
            energy_gained_weight=0.0,
            energy_efficiency_weight=0.0,
            movement_effort_penalty=0.0,
            offspring_weight=0.0,
            trait_energy_cost_penalty_weight=4.0,
        )
        fitness = CreatureFitness()

        fitness.record_trait_cost(cost_per_second=0.25, delta_time=10.0)

        self.assertAlmostEqual(fitness.trait_energy_cost, 2.5)
        self.assertAlmostEqual(fitness.score(config), -10.0)

    def test_record_tick_tracks_distance_and_average_speed(self) -> None:
        fitness = CreatureFitness()

        fitness.record_tick(delta_time=2.0, speed=12.0, max_speed=20.0)
        fitness.record_tick(delta_time=3.0, speed=8.0, max_speed=20.0)

        self.assertAlmostEqual(fitness.distance_traveled, 48.0)
        self.assertAlmostEqual(fitness.average_speed(), 9.6)


if __name__ == "__main__":
    unittest.main()
