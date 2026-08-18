from __future__ import annotations

import gzip
import json
from pathlib import Path
import pickle
import unittest

from configs.sim_config import build_sim_config
from src.persistence import (
    CHECKPOINT_VERSION,
    CheckpointContractError,
    PersistenceManager,
    SimulationPaths,
)
from src.flocking import SocialRuntime


FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parent / "fixtures" / "checkpoint_v20_migration"
)


def migration_config():
    config = build_sim_config()
    # The recorded v20 trajectory predates the larger default world and must
    # retain the bounds and food capacity used to generate the fixtures.
    config.environment.world_width = 3200.0
    config.environment.world_height = 2200.0
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
    config.food.max_food_items = 300
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

    def test_v20_schema_six_fixtures_require_and_apply_brain_reset(self) -> None:
        self.assertEqual(CHECKPOINT_VERSION, 24)
        for metadata in self.manifest["fixtures"]:
            with self.subTest(fixture=metadata["file"]):
                with gzip.open(FIXTURE_DIRECTORY / metadata["file"], "rb") as stream:
                    payload = pickle.load(stream)
                checkpoint = payload["checkpoint"]
                self.assertEqual(checkpoint["version"], 20)
                self.assertEqual(checkpoint["brain_contract"]["sensor_schema"], 6)
                self.assertEqual(checkpoint["brain_contract"]["inputs"], 44)
                with self.assertRaises(CheckpointContractError):
                    PersistenceManager._restore_world(
                        checkpoint,
                        migration_config(),
                        SimulationPaths(Path(".").resolve()),
                    )
                world = PersistenceManager._restore_world(
                    checkpoint,
                    migration_config(),
                    SimulationPaths(Path(".").resolve()),
                    allow_brain_contract_reset=True,
                )
                try:
                    self.assertTrue(world.brain_contract_reset_occurred)
                    self.assertEqual(
                        [tuple(creature.position) for creature in world.creatures],
                        [
                            tuple(creature_state["position"])
                            for creature_state in checkpoint["creatures"]
                        ],
                    )
                    self.assertEqual(
                        [creature.energy for creature in world.creatures],
                        [
                            creature_state["energy"]
                            for creature_state in checkpoint["creatures"]
                        ],
                    )
                    runtimes = tuple(world._cached_social_intentions.values())
                    self.assertTrue(
                        all(isinstance(runtime, SocialRuntime) for runtime in runtimes)
                    )
                    self.assertEqual(len({id(runtime) for runtime in runtimes}), len(runtimes))
                    sentinel_id = metadata["sentinel_creature_id"]
                    for _ in payload["trajectory"]:
                        world.update(world.fixed_timestep)
                    self.assertIn(sentinel_id, world._last_actions)
                finally:
                    world.close()


if __name__ == "__main__":
    unittest.main()
