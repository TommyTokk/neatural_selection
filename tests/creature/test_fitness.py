from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.fitness import CreatureFitness


class CreatureFitnessScoreTest(unittest.TestCase):
    def test_score_is_lifetime_energy_plus_lifespan_tiebreaker(self) -> None:
        """Exercise test score is lifetime energy plus lifespan tiebreaker behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test score is lifetime energy plus lifespan tiebreaker test intent explicit.
        creature = SimpleNamespace(total_energy_gathered=3.5)
        fitness = CreatureFitness(age_seconds=20.0)

        self.assertAlmostEqual(fitness.score(creature), 3.52)

    def test_gathered_energy_scores_before_evaluation_age_advances(self) -> None:
        """Exercise test gathered energy scores before evaluation age advances behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test gathered energy scores before evaluation age advances test intent explicit.
        creature = SimpleNamespace(total_energy_gathered=2.0)
        fitness = CreatureFitness(
            age_seconds=30.0,
            evaluation_start_age_seconds=30.0,
        )

        self.assertAlmostEqual(fitness.score(creature), 2.03)

    def test_offspring_and_flocking_diagnostics_do_not_change_score(self) -> None:
        """Exercise test offspring and flocking diagnostics do not change score behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test offspring and flocking diagnostics do not change score test intent explicit.
        creature = SimpleNamespace(total_energy_gathered=1.25)
        baseline = CreatureFitness(age_seconds=10.0)
        diagnostic = CreatureFitness(
            age_seconds=10.0,
            offspring_count=4,
            matured_offspring_ids=[7, 8],
            flocking_benchmark_reward=100.0,
        )

        self.assertEqual(diagnostic.score(creature), baseline.score(creature))

    def test_food_recording_only_counts_depleted_food(self) -> None:
        """Exercise test food recording only counts depleted food behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test food recording only counts depleted food test intent explicit.
        fitness = CreatureFitness()

        fitness.record_food(depleted=False)
        fitness.record_food(depleted=True)

        self.assertEqual(fitness.food_eaten, 1)

    def test_record_tick_tracks_distance_and_average_speed(self) -> None:
        """Exercise test record tick tracks distance and average speed behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test record tick tracks distance and average speed test intent explicit.
        fitness = CreatureFitness()

        fitness.record_tick(delta_time=2.0, speed=12.0)
        fitness.record_tick(delta_time=3.0, speed=8.0)

        self.assertAlmostEqual(fitness.distance_traveled, 48.0)
        self.assertAlmostEqual(fitness.average_speed(), 9.6)

    def test_legacy_state_retains_historical_energy_for_migration(self) -> None:
        """Exercise test legacy state retains historical energy for migration behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test legacy state retains historical energy for migration test intent explicit.
        fitness = CreatureFitness()

        fitness.__setstate__((None, {
            "age_seconds": 12.0,
            "energy_gained": 4.25,
            "movement_effort": 99.0,
        }))

        self.assertEqual(fitness.age_seconds, 12.0)
        self.assertEqual(fitness._legacy_energy_gained, 4.25)


if __name__ == "__main__":
    unittest.main()
