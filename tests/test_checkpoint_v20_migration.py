from __future__ import annotations

import gzip
import json
from pathlib import Path
import pickle
import unittest

from configs.sim_config import build_sim_config
from src.persistence import CHECKPOINT_VERSION, PersistenceManager, SimulationPaths
from src.flocking import SocialRuntime
from tests.scheduler_validation import AuthoritativeStateDigest


FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parent / "fixtures" / "checkpoint_v20_migration"
)


def migration_config():
    config = build_sim_config()
    config.random_seed = 11
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    config.behavior.enabled = False
    config.counterfactual_why.enabled = False
    config.population.initial_creatures = 2
    config.population.min_reproduction_age = 10_000.0
    config.population.senescence_age_seconds = 10_000.0
    config.food.initial_food_items = 8
    config.flocking.cohort_spawn.enabled = True
    config.flocking.cohort_spawn.size = 2
    config.flocking.cohort_spawn.radius = 24.0
    config.flocking.long_range.enabled = True
    return config


class CheckpointV20MigrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(
            (FIXTURE_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
        )

    def test_manifest_covers_all_next_decision_phases(self) -> None:
        self.assertEqual(self.manifest["checkpoint_version"], 20)
        self.assertEqual(
            {
                fixture["next_decision_phase"]
                for fixture in self.manifest["fixtures"]
            },
            {0, 1, 2},
        )
        self.assertTrue(
            all(
                fixture["continuation_length"] == 15
                for fixture in self.manifest["fixtures"]
            )
        )

    def test_v20_fixtures_migrate_and_follow_the_recorded_trajectory(self) -> None:
        self.assertEqual(CHECKPOINT_VERSION, 22)
        for metadata in self.manifest["fixtures"]:
            with self.subTest(fixture=metadata["file"]):
                with gzip.open(FIXTURE_DIRECTORY / metadata["file"], "rb") as stream:
                    payload = pickle.load(stream)
                checkpoint = payload["checkpoint"]
                self.assertEqual(checkpoint["version"], 20)
                world = PersistenceManager._restore_world(
                    checkpoint,
                    migration_config(),
                    SimulationPaths(Path(".").resolve()),
                )
                try:
                    runtimes = tuple(world._cached_social_intentions.values())
                    self.assertTrue(
                        all(isinstance(runtime, SocialRuntime) for runtime in runtimes)
                    )
                    self.assertEqual(len({id(runtime) for runtime in runtimes}), len(runtimes))
                    sentinel_id = metadata["sentinel_creature_id"]
                    action_reused = False
                    for expected in payload["trajectory"]:
                        phase = world._simulation_step % 3
                        before = world._last_actions[sentinel_id]
                        world.update(world.fixed_timestep)
                        if sentinel_id % 3 != phase:
                            action_reused = action_reused or (
                                world._last_actions[sentinel_id] is before
                            )
                        actual = AuthoritativeStateDigest.capture(world)
                        difference = expected.compare(actual)
                        self.assertIsNone(
                            difference,
                            None if difference is None else difference.describe(),
                        )
                    self.assertTrue(action_reused)
                finally:
                    world.close()


if __name__ == "__main__":
    unittest.main()
