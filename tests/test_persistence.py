from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import os
import pickle
import unittest

from configs.sim_config import build_sim_config
from src.persistence import (
    CHECKPOINT_VERSION,
    CheckpointError,
    PersistenceManager,
    SimulationPaths,
)
from src.telemetry import TelemetryDatabase


class PersistenceManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.config = build_sim_config()
        self.config.persistence.simulation_root_directory = str(
            Path(self.temporary_directory.name) / "saves"
        )
        self.simulation_paths = SimulationPaths.create_new(
            self.config.persistence,
            now=datetime(2026, 7, 1, tzinfo=timezone.utc),
            unique_suffix="test0001",
        )
        self.checkpoint = self.simulation_paths.quick_checkpoint

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_atomic_write_rotates_quick_backup(self) -> None:
        first_state = {"version": CHECKPOINT_VERSION, "value": "first"}
        second_state = {"version": CHECKPOINT_VERSION, "value": "second"}

        PersistenceManager._write_atomic(first_state, self.checkpoint)
        with patch("src.persistence.os.replace", wraps=os.replace) as replace:
            PersistenceManager._write_atomic(second_state, self.checkpoint)

        with self.checkpoint.open("rb") as stream:
            current = pickle.load(stream)
        with Path(f"{self.checkpoint}.bak").open("rb") as stream:
            backup = pickle.load(stream)

        self.assertEqual(current["value"], "second")
        self.assertEqual(backup["value"], "first")
        self.assertFalse(Path(f"{self.checkpoint}.tmp").exists())
        self.assertEqual(
            [call.args for call in replace.call_args_list],
            [
                (self.checkpoint, Path(f"{self.checkpoint}.bak")),
                (Path(f"{self.checkpoint}.tmp"), self.checkpoint),
            ],
        )

    def test_failed_quick_promotion_restores_previous_checkpoint(self) -> None:
        original_state = {"version": CHECKPOINT_VERSION, "value": "safe"}
        PersistenceManager._write_atomic(original_state, self.checkpoint)
        real_replace = os.replace

        def fail_promotion(source: object, destination: object) -> None:
            if Path(source) == Path(f"{self.checkpoint}.tmp"):
                raise OSError("simulated promotion failure")
            real_replace(source, destination)

        with patch("src.persistence.os.replace", side_effect=fail_promotion):
            with self.assertRaises(OSError):
                PersistenceManager._write_atomic(
                    {"version": CHECKPOINT_VERSION, "value": "new"},
                    self.checkpoint,
                )

        with self.checkpoint.open("rb") as stream:
            self.assertEqual(pickle.load(stream), original_state)
        self.assertFalse(Path(f"{self.checkpoint}.tmp").exists())

    def test_load_falls_back_to_valid_backup(self) -> None:
        self.checkpoint.write_bytes(b"not a pickle")
        backup = Path(f"{self.checkpoint}.bak")
        with backup.open("wb") as stream:
            pickle.dump(
                {
                    "version": CHECKPOINT_VERSION,
                    "simulation_id": self.simulation_paths.simulation_id,
                    "marker": "backup",
                },
                stream,
            )

        with patch.object(
            PersistenceManager,
            "_restore_world",
            return_value="restored",
        ) as restore_world:
            restored = PersistenceManager.load_simulation(
                self.config,
                self.simulation_paths.simulation_directory,
            )

        self.assertEqual(restored, "restored")
        self.assertEqual(restore_world.call_args.args[0]["marker"], "backup")
        self.assertEqual(restore_world.call_args.args[2], self.simulation_paths)

    def test_latest_simulation_load_falls_back_to_hourly_archive(self) -> None:
        archive = self.simulation_paths.hourly_target(
            now=datetime(2026, 7, 1, 1, tzinfo=timezone.utc)
        )
        with archive.path.open("wb") as stream:
            pickle.dump(
                {
                    "version": CHECKPOINT_VERSION,
                    "simulation_id": self.simulation_paths.simulation_id,
                    "marker": "hourly",
                },
                stream,
            )
        self.checkpoint.write_bytes(b"newer but corrupt")

        with patch.object(
            PersistenceManager,
            "_restore_world",
            return_value="restored",
        ) as restore_world:
            restored = PersistenceManager.load_simulation(self.config)

        self.assertEqual(restored, "restored")
        self.assertEqual(restore_world.call_args.args[0]["marker"], "hourly")
        self.assertEqual(restore_world.call_args.args[2], self.simulation_paths)

    def test_load_checkpoint_accepts_exact_quick_backup_and_hourly_files(
        self,
    ) -> None:
        checkpoints = (
            self.simulation_paths.quick_checkpoint,
            Path(f"{self.simulation_paths.quick_checkpoint}.bak"),
            self.simulation_paths.hourly_target(
                now=datetime(2026, 7, 1, 1, tzinfo=timezone.utc)
            ).path,
        )

        for index, checkpoint in enumerate(checkpoints):
            with self.subTest(checkpoint=checkpoint):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open("wb") as stream:
                    pickle.dump(
                        {
                            "version": CHECKPOINT_VERSION,
                            "simulation_id": (
                                self.simulation_paths.simulation_id
                            ),
                            "marker": index,
                        },
                        stream,
                    )
                with patch.object(
                    PersistenceManager,
                    "_restore_world",
                    return_value=f"restored-{index}",
                ) as restore_world:
                    restored = PersistenceManager.load_checkpoint(
                        self.config,
                        checkpoint,
                    )

                self.assertEqual(restored, f"restored-{index}")
                self.assertEqual(
                    restore_world.call_args.args[0]["marker"],
                    index,
                )
                self.assertEqual(
                    restore_world.call_args.args[2],
                    self.simulation_paths,
                )

    def test_load_checkpoint_does_not_fallback_from_corrupt_selection(
        self,
    ) -> None:
        self.simulation_paths.quick_checkpoint.write_bytes(b"corrupt")
        backup = Path(f"{self.simulation_paths.quick_checkpoint}.bak")
        with backup.open("wb") as stream:
            pickle.dump(
                {
                    "version": CHECKPOINT_VERSION,
                    "simulation_id": self.simulation_paths.simulation_id,
                },
                stream,
            )

        with patch.object(PersistenceManager, "_restore_world") as restore_world:
            with self.assertRaises(CheckpointError):
                PersistenceManager.load_checkpoint(
                    self.config,
                    self.simulation_paths.quick_checkpoint,
                )

        restore_world.assert_not_called()

    def test_background_save_flushes_all_targets_before_close(self) -> None:
        manager = PersistenceManager()
        state = {"version": CHECKPOINT_VERSION, "marker": "queued"}
        archive = self.simulation_paths.hourly_target(
            now=datetime(2026, 7, 1, 1, tzinfo=timezone.utc)
        )

        with patch.object(manager, "_capture_state", return_value=state):
            manager.save_simulation(
                object(),
                object(),
                (self.simulation_paths.quick_target(), archive),
            )
            manager.close()

        for checkpoint in (self.checkpoint, archive.path):
            with checkpoint.open("rb") as stream:
                self.assertEqual(pickle.load(stream), state)
        self.assertFalse(Path(f"{archive.path}.bak").exists())

    def test_coalescing_retains_pending_hourly_target(self) -> None:
        manager = PersistenceManager()
        writer_started = Event()
        release_writer = Event()
        writes: list[tuple[int, Path]] = []
        archive = self.simulation_paths.hourly_target(
            now=datetime(2026, 7, 1, 1, tzinfo=timezone.utc)
        )

        def write(
            state: dict[str, object],
            checkpoint: Path,
            *,
            rotate_backup: bool,
        ) -> None:
            del rotate_backup
            writes.append((int(state["marker"]), checkpoint))
            if state["marker"] == 1:
                writer_started.set()
                release_writer.wait(timeout=2.0)

        states = [
            {"version": CHECKPOINT_VERSION, "marker": 1},
            {"version": CHECKPOINT_VERSION, "marker": 2},
            {"version": CHECKPOINT_VERSION, "marker": 3},
        ]
        try:
            with (
                patch.object(manager, "_capture_state", side_effect=states),
                patch.object(manager, "_write_atomic", side_effect=write),
            ):
                manager.save_simulation(
                    object(),
                    object(),
                    (self.simulation_paths.quick_target(),),
                )
                self.assertTrue(writer_started.wait(timeout=2.0))
                manager.save_simulation(object(), object(), (archive,))
                manager.save_simulation(
                    object(),
                    object(),
                    (self.simulation_paths.quick_target(),),
                )
                release_writer.set()
                manager.flush()
        finally:
            release_writer.set()
            manager.close()

        self.assertEqual(
            writes,
            [
                (1, self.simulation_paths.quick_checkpoint),
                (3, archive.path),
                (3, self.simulation_paths.quick_checkpoint),
            ],
        )

    def test_capture_excludes_brains_and_records_tier_timers(self) -> None:
        body = SimpleNamespace(
            velocity=SimpleNamespace(x=1.0, y=2.0),
            angular_velocity=0.5,
        )
        creature = SimpleNamespace(
            creature_id=4,
            name="Herbivore 04",
            position=(3.0, 7.0),
            heading=0.25,
            body=body,
            energy=0.8,
            vision=SimpleNamespace(range=110.0, angle=1.0),
            physical_traits=SimpleNamespace(
                radius=16.0,
                movement_cost_multiplier=1.0,
            ),
            color=(1, 2, 3),
            lineage=SimpleNamespace(species_id=2),
        )
        spawner = SimpleNamespace(
            _next_food_id=1,
            _spawn_credit=0.0,
            _burst_credit=0.0,
            _low_food_burst_credit=0.0,
            _pending_burst_items=0,
            _pending_low_food_burst_items=0,
        )
        rt_neat = SimpleNamespace(
            stats=SimpleNamespace(),
            eligible_parent_ids=[],
            _lifespan_at_death_total=0.0,
            _lifespan_at_death_count=0,
        )
        world = SimpleNamespace(
            creatures=[creature],
            foods=[],
            fitness={4: SimpleNamespace(age_seconds=9.0)},
            _chronometers={4: 2.0},
            elapsed_time=9.0,
            rng=__import__("random").Random(7),
            _physics_accumulator=0.0,
            _reproduction_accumulator=0.0,
            time_since_last_quick_save=20.0,
            time_since_last_archive_save=50.0,
            simulation_paths=self.simulation_paths,
            total_biomass_energy=3.0,
            simulation_speed=1.0,
            is_paused=False,
            selected_creature_id=None,
            _previous_biome_here_by_creature_id={},
            _held_food_by_creature_id={},
            _carrier_by_food_id={},
            food_spawner=spawner,
            fitness_archive={},
            _trait_archive_by_genome_id={},
            rt_neat=rt_neat,
        )
        controller = SimpleNamespace(
            genome_id_for=lambda creature_id: 17,
            population=SimpleNamespace(population={17: "genome"}, generation=0),
            species_manager=SimpleNamespace(
                compatibility_threshold=3.0,
                representatives={1: "genome"},
                next_species_id=2,
            ),
            brains={4: object()},
        )

        state = PersistenceManager._capture_state(world, controller)

        self.assertEqual(state["simulation_id"], self.simulation_paths.simulation_id)
        self.assertEqual(state["creatures"][0]["genome_id"], 17)
        self.assertEqual(state["world"]["time_since_last_quick_save"], 20.0)
        self.assertEqual(state["world"]["time_since_last_archive_save"], 50.0)
        self.assertNotIn("brains", state)
        self.assertNotIn("brain", state["creatures"][0])

    def test_real_world_round_trip_reuses_simulation_directory(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 1
        world = World(self.config)
        restored = None
        try:
            original_brain = world.neat_controller.brain_for(
                world.creatures[0].creature_id
            )
            world.foods[0].consume_energy(
                world.foods[0].energy_value * 0.25,
                min_remainder_ratio=0.0,
            )
            saved_food_energy = world.foods[0].energy_value
            world.persistence_manager.save_simulation(
                world,
                world.neat_controller,
                (world.simulation_paths.quick_target(),),
            )
            world.persistence_manager.flush()

            restored = PersistenceManager.load_checkpoint(
                self.config,
                world.simulation_paths.quick_checkpoint,
            )
            restored_brain = restored.neat_controller.brain_for(
                restored.creatures[0].creature_id
            )

            self.assertAlmostEqual(restored.foods[0].energy_value, saved_food_energy)
            self.assertIsNot(restored_brain, original_brain)
            self.assertEqual(restored.live_brain_count(), 1)
            self.assertEqual(restored.simulation_paths, world.simulation_paths)
            self.assertFalse(world.simulation_paths.telemetry_database.exists())
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_new_simulations_have_distinct_isolated_directories(self) -> None:
        second = SimulationPaths.create_new(
            self.config.persistence,
            now=datetime(2026, 7, 1, tzinfo=timezone.utc),
            unique_suffix="test0002",
        )

        self.assertNotEqual(
            self.simulation_paths.simulation_directory,
            second.simulation_directory,
        )
        self.assertEqual(
            self.simulation_paths.telemetry_database.parent,
            self.simulation_paths.simulation_directory,
        )
        self.assertEqual(
            second.telemetry_database.parent,
            second.simulation_directory,
        )

        first_database = TelemetryDatabase(
            self.simulation_paths.telemetry_database
        )
        second_database = TelemetryDatabase(second.telemetry_database)
        try:
            first_database.log_species(1, None, 0.0)
            self.assertEqual(
                first_database.connection.execute(
                    "SELECT COUNT(*) FROM species"
                ).fetchone(),
                (1,),
            )
            self.assertEqual(
                second_database.connection.execute(
                    "SELECT COUNT(*) FROM species"
                ).fetchone(),
                (0,),
            )
        finally:
            first_database.close()
            second_database.close()

    def test_latest_simulation_uses_timestamped_directory_order(self) -> None:
        SimulationPaths.create_new(
            self.config.persistence,
            now=datetime(2026, 7, 1, tzinfo=timezone.utc)
            + timedelta(minutes=1),
            unique_suffix="later001",
        )

        latest = SimulationPaths.latest(self.config.persistence)

        self.assertIn("20260701T000100", latest.simulation_id)

    def test_hourly_targets_are_retained_with_distinct_names(self) -> None:
        first = self.simulation_paths.hourly_target(
            now=datetime(2026, 7, 1, 1, tzinfo=timezone.utc)
        )
        second = self.simulation_paths.hourly_target(
            now=datetime(2026, 7, 1, 2, tzinfo=timezone.utc)
        )
        state = {"version": CHECKPOINT_VERSION}

        PersistenceManager._write_atomic(
            state,
            first.path,
            rotate_backup=first.rotate_backup,
        )
        PersistenceManager._write_atomic(
            state,
            second.path,
            rotate_backup=second.rotate_backup,
        )

        self.assertNotEqual(first.path, second.path)
        self.assertFalse(first.rotate_backup)
        self.assertEqual(
            set(self.simulation_paths.hourly_directory.glob("*.pkl")),
            {first.path, second.path},
        )


if __name__ == "__main__":
    unittest.main()
