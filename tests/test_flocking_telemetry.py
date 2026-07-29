from __future__ import annotations

from types import SimpleNamespace
import unittest

from configs.sim_config import FlockingBenchmarkConfig
from src.fitness import CreatureFitness
from src.flocking import (
    FlockingRuntimeSnapshot,
    FlockingWeights,
    SocialIntent,
    SocialObservation,
)
from src.flocking_telemetry import (
    FlockingTelemetryAggregator,
    GroupSample,
    PersistentGroupTracker,
)
from src.telemetry import TelemetryDatabase
from src.world import World


def creature(creature_id: int, x: float, y: float):
    return SimpleNamespace(
        creature_id=creature_id,
        position=(x, y),
        body=SimpleNamespace(
            velocity=SimpleNamespace(x=10.0, y=0.0)
        ),
    )


class PersistentGroupTrackerTest(unittest.TestCase):
    def test_group_identity_survives_small_membership_change(self) -> None:
        tracker = PersistentGroupTracker(0.5)
        first = [creature(1, 0, 0), creature(2, 10, 0), creature(3, 20, 0)]

        def nearby(item, _range):
            return [other for other in first if other is not item]

        sample = tracker.sample(
            first,
            sim_time=1.0,
            group_range=30.0,
            minimum_compatibility=0.5,
            compatibility=lambda _a, _b: 1.0,
            nearby=nearby,
        )
        group_id = sample.groups[0].group_id
        second = [first[0], first[1], creature(4, 15, 0)]

        def nearby_second(item, _range):
            return [other for other in second if other is not item]

        sample = tracker.sample(
            second,
            sim_time=2.0,
            group_range=30.0,
            minimum_compatibility=0.5,
            compatibility=lambda _a, _b: 1.0,
            nearby=nearby_second,
        )
        self.assertEqual(sample.groups[0].group_id, group_id)

    def test_incompatible_neighbors_do_not_form_group(self) -> None:
        items = [creature(1, 0, 0), creature(2, 5, 0)]
        sample = PersistentGroupTracker().sample(
            items,
            sim_time=1.0,
            group_range=30.0,
            minimum_compatibility=0.5,
            compatibility=lambda _a, _b: 0.25,
            nearby=lambda item, _range: [
                other for other in items if other is not item
            ],
        )
        self.assertEqual(sample.groups, ())

    def test_split_merge_counts_and_tracker_state_round_trip(self) -> None:
        tracker = PersistentGroupTracker()
        joined = [
            creature(1, 0, 0),
            creature(2, 5, 0),
            creature(3, 10, 0),
            creature(4, 15, 0),
        ]

        def local(items):
            return lambda item, search_range: [
                other
                for other in items
                if other is not item
                and abs(other.position[0] - item.position[0]) <= search_range
            ]

        tracker.sample(
            joined,
            sim_time=1.0,
            group_range=20.0,
            minimum_compatibility=0.5,
            compatibility=lambda _a, _b: 1.0,
            nearby=local(joined),
        )
        separated = [
            creature(1, 0, 0),
            creature(2, 5, 0),
            creature(3, 100, 0),
            creature(4, 105, 0),
        ]
        split = tracker.sample(
            separated,
            sim_time=2.0,
            group_range=20.0,
            minimum_compatibility=0.5,
            compatibility=lambda _a, _b: 1.0,
            nearby=local(separated),
        )
        self.assertEqual(split.fragmentation_count, 1)

        restored = PersistentGroupTracker()
        restored.restore(tracker.state_dict())
        merged = restored.sample(
            joined,
            sim_time=3.0,
            group_range=20.0,
            minimum_compatibility=0.5,
            compatibility=lambda _a, _b: 1.0,
            nearby=local(joined),
        )
        self.assertEqual(merged.merger_count, 1)

    def test_aggregator_is_read_only_and_population_normalized(self) -> None:
        metrics = FlockingTelemetryAggregator.aggregate(
            sim_time=2.0,
            population_size=4,
            runtime={},
            groups=PersistentGroupTracker().sample(
                [],
                sim_time=2.0,
                group_range=20.0,
                minimum_compatibility=0.5,
                compatibility=lambda _a, _b: 1.0,
                nearby=lambda _item, _range: [],
            ),
        )
        self.assertEqual(metrics["population_size"], 4)
        self.assertEqual(metrics["seeing_any_percent"], 0.0)

    def test_exposure_and_center_metrics_use_their_exact_contracts(self) -> None:
        runtime = {
            1: FlockingRuntimeSnapshot(
                observation=SocialObservation(
                    visible_creature_count=1,
                    compatible_visible_count=0,
                    personal_space_presence=0.0,
                    social_presence=0.0,
                    center_distance=42.0,
                    mean_neighbor_distance=17.0,
                ),
                intent=SocialIntent(
                    weights=FlockingWeights(engagement=0.25)
                ),
                neural_herding=0.75,
            )
        }
        metrics = FlockingTelemetryAggregator.aggregate(
            sim_time=2.0,
            population_size=1,
            runtime=runtime,
            groups=GroupSample(),
        )
        self.assertEqual(metrics["seeing_any_percent"], 100.0)
        self.assertEqual(metrics["seeing_compatible_percent"], 0.0)
        self.assertEqual(metrics["mean_center_distance"], 42.0)
        self.assertEqual(metrics["mean_neural_herding"], 0.75)


class BenchmarkFitnessTest(unittest.TestCase):
    def test_stationary_group_receives_zero(self) -> None:
        fitness = CreatureFitness()
        observation = SocialObservation(
            social_presence=1.0,
            effective_count=3.0,
            mean_neighbor_distance=60.0,
            mean_heading_error=0.0,
            mean_group_velocity=(0.0, 0.0),
        )
        reward = fitness.record_flocking_benchmark(
            observation,
            1.0,
            FlockingBenchmarkConfig(enabled=True),
        )
        self.assertEqual(reward, 0.0)

    def test_isolated_creature_receives_zero(self) -> None:
        fitness = CreatureFitness()
        reward = fitness.record_flocking_benchmark(
            SocialObservation(),
            1.0,
            FlockingBenchmarkConfig(enabled=True),
        )
        self.assertEqual(reward, 0.0)

    def test_overlapping_pile_is_strongly_penalized(self) -> None:
        ideal = CreatureFitness()
        pile = CreatureFitness()
        config = FlockingBenchmarkConfig(enabled=True, reward_rate=1.0)
        common = {
            "social_presence": 1.0,
            "effective_count": 3.0,
            "mean_heading_error": 0.0,
            "mean_group_velocity": (50.0, 0.0),
        }
        ideal.record_flocking_benchmark(
            SocialObservation(mean_neighbor_distance=60.0, **common),
            1.0,
            config,
        )
        pile.record_flocking_benchmark(
            SocialObservation(mean_neighbor_distance=0.0, **common),
            1.0,
            config,
        )
        self.assertLess(
            pile.flocking_benchmark_reward,
            ideal.flocking_benchmark_reward * 0.02,
        )

    def test_benchmark_is_disabled_by_default(self) -> None:
        self.assertFalse(FlockingBenchmarkConfig().enabled)

    def test_moving_aligned_group_is_positive_and_capped(self) -> None:
        fitness = CreatureFitness()
        config = FlockingBenchmarkConfig(
            enabled=True,
            reward_rate=10.0,
            max_per_evaluation=1.0,
        )
        observation = SocialObservation(
            social_presence=1.0,
            effective_count=3.0,
            mean_neighbor_distance=60.0,
            mean_heading_error=0.0,
            mean_group_velocity=(50.0, 0.0),
        )
        fitness.record_flocking_benchmark(observation, 1.0, config)
        self.assertEqual(fitness.flocking_benchmark_reward, 1.0)

    def test_disabled_reward_is_independent_of_sampling(self) -> None:
        fitness = CreatureFitness()
        observation = SocialObservation(
            social_presence=1.0,
            effective_count=3.0,
            mean_neighbor_distance=60.0,
            mean_group_velocity=(50.0, 0.0),
        )
        for _ in range(20):
            fitness.record_flocking_benchmark(
                observation,
                0.5,
                FlockingBenchmarkConfig(enabled=False),
            )
        self.assertEqual(fitness.flocking_benchmark_reward, 0.0)

    def test_reward_is_independent_of_observational_sampling_interval(self) -> None:
        config = FlockingBenchmarkConfig(enabled=True)
        observation = SocialObservation(
            social_presence=1.0,
            effective_count=3.0,
            mean_neighbor_distance=60.0,
            mean_heading_error=0.1,
            mean_group_velocity=(50.0, 0.0),
        )
        per_fixed_step = CreatureFitness()
        for _ in range(600):
            per_fixed_step.record_flocking_benchmark(
                observation,
                1.0 / 60.0,
                config,
            )
        per_telemetry_second = CreatureFitness()
        for _ in range(10):
            per_telemetry_second.record_flocking_benchmark(
                observation,
                1.0,
                config,
            )
        self.assertAlmostEqual(
            per_fixed_step.flocking_benchmark_reward,
            per_telemetry_second.flocking_benchmark_reward,
            places=12,
        )


class FlockingTelemetryDatabaseTest(unittest.TestCase):
    def test_population_metrics_schema_and_upsert_work_in_memory(self) -> None:
        database = TelemetryDatabase(":memory:")
        try:
            columns = {
                row[1]
                for row in database.connection.execute(
                    "PRAGMA table_info(flocking_population_metrics)"
                )
            }
            self.assertIn("mean_panic", columns)
            self.assertIn("mean_neural_herding", columns)
            self.assertIn("benchmark_reward_contribution", columns)
            database.log_flocking_metrics(
                {
                    "sim_time": 1.0,
                    "population_size": 4,
                    "mean_panic": 0.25,
                    "benchmark_reward_contribution": 0.5,
                }
            )
            row = database.connection.execute(
                "SELECT population_size, mean_panic, "
                "benchmark_reward_contribution "
                "FROM flocking_population_metrics WHERE sim_time = 1.0"
            ).fetchone()
            self.assertEqual(row, (4, 0.25, 0.5))
        finally:
            database.close()

    def test_early_flocking_table_gains_new_columns_non_destructively(
        self,
    ) -> None:
        database = TelemetryDatabase(":memory:")
        try:
            database.connection.execute(
                "ALTER TABLE flocking_population_metrics "
                "RENAME TO old_flocking_population_metrics"
            )
            database.connection.execute(
                "CREATE TABLE flocking_population_metrics "
                "(sim_time REAL PRIMARY KEY, population_size INTEGER)"
            )
            database.connection.execute(
                "INSERT INTO flocking_population_metrics VALUES (1.0, 7)"
            )
            database._ensure_flocking_population_metrics_columns()
            columns = {
                row[1]
                for row in database.connection.execute(
                    "PRAGMA table_info(flocking_population_metrics)"
                )
            }
            self.assertIn("mean_neural_herding", columns)
            self.assertIn("mean_panic", columns)
            self.assertIn("benchmark_reward_contribution", columns)
            self.assertEqual(
                database.connection.execute(
                    "SELECT population_size "
                    "FROM flocking_population_metrics WHERE sim_time = 1.0"
                ).fetchone(),
                (7,),
            )
        finally:
            database.close()

    def test_disabled_telemetry_does_not_sample_groups(self) -> None:
        world = object.__new__(World)
        world.telemetry = None
        world._flocking_group_tracker = SimpleNamespace(
            sample=lambda *args, **kwargs: self.fail(
                "group detection ran while telemetry was disabled"
            )
        )
        world._update_flocking_telemetry(10.0)


if __name__ == "__main__":
    unittest.main()
