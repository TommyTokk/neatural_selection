from __future__ import annotations

import unittest

from src.fitness import CreatureFitness, CreatureTelemetry


class CreatureTelemetryTest(unittest.TestCase):
    def test_net_energy_balance_tracks_ingestion_minus_spend(self) -> None:
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
        telemetry = CreatureTelemetry(age_seconds=20.0)
        telemetry.record_energy_transaction(ingested=3.5, spent=1.25)

        self.assertAlmostEqual(telemetry.net_energy_balance, 2.25)

    def test_net_metabolic_rate_uses_lifetime_age(self) -> None:
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
        telemetry = CreatureTelemetry(age_seconds=30.0)
        telemetry.record_energy_transaction(ingested=5.0, spent=2.0)

        self.assertAlmostEqual(telemetry.net_metabolic_rate, 0.1)

    def test_telemetry_has_no_scalar_selection_score(self) -> None:
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
        diagnostic = CreatureFitness(
            age_seconds=10.0,
            offspring_count=4,
            matured_offspring_ids=[7, 8],
            flocking_benchmark_reward=100.0,
        )

        self.assertFalse(hasattr(diagnostic, "score"))
        self.assertEqual(diagnostic.lifetime_offspring_count, 4)

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
