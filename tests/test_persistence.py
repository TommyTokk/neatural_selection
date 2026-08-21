from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import gc
import os
import pickle
import unittest
import weakref

from configs.sim_config import build_sim_config
from src.creature import FlockingTraits, PhysicalTraits, VisionTraits
from src.persistence import (
    CHECKPOINT_VERSION,
    CheckpointContractError,
    CheckpointError,
    PersistenceManager,
    SavePriority,
    SimulationPaths,
)
from src.speciation import (
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)
from src.ui.layouts.species_tree import build_species_tree_layout
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

    def test_checkpoint_versions_2_through_19_remain_loadable(self) -> None:
        for version in range(2, 20):
            PersistenceManager._validate_state({"version": version})

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

    def test_manual_save_takes_precedence_and_retains_hourly_target(self) -> None:
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
            {"version": CHECKPOINT_VERSION, "marker": 4},
        ]
        try:
            with (
                patch.object(
                    manager,
                    "_capture_state",
                    side_effect=states,
                ) as capture_state,
                patch.object(manager, "_write_atomic", side_effect=write),
            ):
                manager.save_simulation(
                    object(),
                    object(),
                    (self.simulation_paths.quick_target(),),
                )
                self.assertTrue(writer_started.wait(timeout=2.0))
                self.assertTrue(manager.is_busy)
                manager.save_simulation(object(), object(), (archive,))
                manager.save_simulation(
                    object(),
                    object(),
                    (self.simulation_paths.quick_target(),),
                    priority=SavePriority.MANUAL,
                )
                manager.save_simulation(
                    object(),
                    object(),
                    (self.simulation_paths.quick_target(),),
                    priority=SavePriority.AUTO,
                )
                release_writer.set()
                manager.flush()
                self.assertEqual(capture_state.call_count, 3)
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
        self.assertFalse(manager.is_busy)

    def test_completed_save_releases_snapshot_references(self) -> None:
        class Payload:
            pass

        manager = PersistenceManager()
        payload_references: list[weakref.ReferenceType[Payload]] = []

        def capture_state(*args: object) -> dict[str, object]:
            del args
            payload = Payload()
            payload_references.append(weakref.ref(payload))
            return {"version": CHECKPOINT_VERSION, "payload": payload}

        manager._capture_state = capture_state
        manager._write_atomic = lambda *args, **kwargs: None
        try:
            manager.save_simulation(
                object(),
                object(),
                (self.simulation_paths.quick_target(),),
            )
            manager.flush()
            gc.collect()
            self.assertIsNone(payload_references[0]())
        finally:
            manager.close()

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
            _low_food_burst_credit=0.0,
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
            _held_food_by_creature_id={},
            _carrier_by_food_id={},
            food_spawner=spawner,
            species_history={},
            fitness_archive={},
            _trait_archive_by_genome_id={},
            rt_neat=rt_neat,
        )
        controller = SimpleNamespace(
            genome_id_for=lambda creature_id: 17,
            population=SimpleNamespace(population={17: "genome"}, generation=0),
            species_manager=SimpleNamespace(
                compatibility_threshold=3.0,
                phenotypic_weight=2.0,
                representatives={1: "genome"},
                next_species_id=2,
            ),
            brains={4: object()},
        )

        state = PersistenceManager._capture_state(world, controller)

        self.assertEqual(state["simulation_id"], self.simulation_paths.simulation_id)
        self.assertEqual(state["creatures"][0]["genome_id"], 17)
        self.assertNotIn("biome_fertility_ema", state["creatures"][0])
        self.assertNotIn(
            "biome_fertility_ema_updated_at",
            state["creatures"][0],
        )
        self.assertEqual(state["brain_contract"]["sensor_schema"], 7)
        self.assertEqual(state["brain_contract"]["inputs"], 43)
        self.assertNotIn("previous_biome", state["world"])
        self.assertNotIn("physics_accumulator", state["world"])
        self.assertEqual(state["world"]["simulation_step"], 0)
        self.assertEqual(state["world"]["mouth_exposures"], ())
        self.assertEqual(
            state["world"]["speciation_adjustment_accumulator"],
            0.0,
        )
        self.assertEqual(state["world"]["time_since_last_quick_save"], 20.0)
        self.assertEqual(state["world"]["time_since_last_archive_save"], 50.0)
        self.assertEqual(state["world"]["next_creature_id"], 5)
        self.assertEqual(state["population"]["next_genome_id"], 18)
        self.assertNotIn("brains", state)
        self.assertNotIn("brain", state["creatures"][0])
        self.assertEqual(state["species_manager"]["phenotypic_weight"], 2.0)
        self.assertEqual(state["species_history"], {})
        self.assertEqual(
            set(state["food_spawner"]),
            {
                "next_food_id",
                "spawn_credit",
                "low_food_burst_credit",
                "pending_low_food_burst_items",
            },
        )

    def test_legacy_representative_migration_uses_living_traits(self) -> None:
        genome = SimpleNamespace(key=17)
        physical_traits = PhysicalTraits(radius=18.0)
        vision = VisionTraits(range=120.0, angle=1.2)

        migrated = PersistenceManager._migrate_species_representatives(
            {2: genome},
            [
                {
                    "genome_id": 17,
                    "physical_traits": physical_traits,
                    "vision": vision,
                }
            ],
            {},
        )

        self.assertIs(migrated[2][0], genome)
        self.assertEqual(migrated[2][1], physical_traits)
        self.assertEqual(migrated[2][2], vision)
        self.assertEqual(migrated[2][3], FlockingTraits())
        self.assertIsNot(migrated[2][1], physical_traits)
        self.assertIsNot(migrated[2][2], vision)

    def test_legacy_representative_migration_uses_archived_traits(self) -> None:
        genome = SimpleNamespace(key=17)
        archived = SimpleNamespace(
            physical_traits=PhysicalTraits(radius=18.0),
            vision=VisionTraits(range=120.0, angle=1.2),
        )

        migrated = PersistenceManager._migrate_species_representatives(
            {2: genome},
            [],
            {17: archived},
        )

        self.assertEqual(migrated[2][1], archived.physical_traits)
        self.assertEqual(migrated[2][2], archived.vision)

    def test_legacy_representative_migration_requires_exact_traits(self) -> None:
        with self.assertRaisesRegex(
            CheckpointError,
            "has no living or archived phenotype",
        ):
            PersistenceManager._migrate_species_representatives(
                {2: SimpleNamespace(key=17)},
                [],
                {},
            )

    def test_real_world_round_trip_reuses_simulation_directory(self) -> None:
        from src.behavior_history import (
            BehaviorTermination,
            CompletedBehaviorBoutDraft,
        )
        from src.behavior_observer import BehaviorKind
        from src.communication import AcousticSignal
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 1
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            focal = world.creatures[0]
            world.behavior_history.register_creature(
                focal.creature_id,
                focal.name,
                1.0,
            )
            world.behavior_history.append_draft(
                CompletedBehaviorBoutDraft(
                    creature_id=focal.creature_id,
                    selection_generation=1,
                    behavior=BehaviorKind.RESTING,
                    local_bout_id=1,
                    start_time=1.0,
                    end_time=2.0,
                    duration=1.0,
                    evidence_summary=(),
                    outcome=None,
                    termination=BehaviorTermination.FOCUS_CHANGED,
                )
            )
            world.behavior_history.mark_incomplete(2)
            world._behavior_automatic_cohort = {
                focal.lineage.species_id: (focal.creature_id,),
            }
            original_brain = world.neat_controller.brain_for(
                world.creatures[0].creature_id
            )
            self.assertEqual(original_brain.herding_decay_rate, 0.15)
            original_brain.herding_state = 0.8
            original_brain.last_raw_herding = 0.9
            recurrent_inputs = [0.25] * len(
                original_brain.network.input_nodes
            )
            original_brain.network.activate(recurrent_inputs)
            saved_network_state = original_brain.export_network_state()
            saved_member_color = (77, 88, 199)
            world.creatures[0].color = saved_member_color
            world._physics_accumulator = 0.007
            world._simulation_step = 7
            world._speciation_adjustment_accumulator = 1.75
            world._mouth_exposures.append(
                6,
                world.creatures[0].creature_id,
                world.foods[0].id,
                world.fixed_timestep,
            )
            world.creatures[0].stomach_energy = 0.42
            world.creatures[0].stomach_difficulty_load = 0.47
            world.creatures[0].total_energy_gathered = 2.75
            saved_digestive_traits = (
                world.creatures[0].physical_traits.stomach_capacity,
                world.creatures[0].physical_traits.digestion_rate,
                world.creatures[0].physical_traits.digestion_efficiency,
            )
            world.pheromones.deposit(
                world.creatures[0].position,
                trail_amount=0.4,
                alarm_amount=0.2,
            )
            world.pheromones.accumulator = 0.1
            world.acoustics.replace_signals(
                [
                    AcousticSignal(
                        emitter_id=world.creatures[0].creature_id,
                        position=world.creatures[0].position,
                        strength=0.8,
                        tone=-0.25,
                    )
                ]
            )
            world.foods[0].consume_energy(
                world.foods[0].energy_value * 0.25,
                min_remainder_ratio=0.0,
            )
            saved_food_energy = world.foods[0].energy_value
            saved_food_original_radius = world.foods[0].original_radius
            luca = world.species_history[1]
            second_species = replace(
                luca,
                species_id=2,
                parent_species_id=1,
                founder_creature_id=20,
                founder_genome_id=20,
                emerged_at=7.5,
                founder_color=(210, 40, 90),
            )
            world.species_history[2] = second_species
            world.neat_controller.species_manager.next_species_id = 2
            captured = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            serialized_keys: set[str] = set()

            def collect_keys(value) -> None:
                if isinstance(value, dict):
                    serialized_keys.update(str(key) for key in value)
                    for child in value.values():
                        collect_keys(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        collect_keys(child)

            collect_keys(captured)
            self.assertNotIn("herding_state", serialized_keys)
            self.assertNotIn("last_raw_herding", serialized_keys)
            metadata = captured["communication"]["pheromone_metadata"]
            self.assertEqual(metadata, world.pheromones.state_metadata())
            self.assertEqual(captured["version"], CHECKPOINT_VERSION)
            captured_network_state = captured["creatures"][0][
                "scheduler_continuation"
            ]["brain_network_state"]
            self.assertEqual(captured_network_state, saved_network_state)
            self.assertIsNot(
                captured_network_state["values"][0],
                original_brain.network.values[0],
            )
            self.assertEqual(
                captured["world"]["behavior_history"],
                world.behavior_history.state_dict(),
            )
            self.assertEqual(
                captured["world"]["behavior_automatic_cohort"],
                world._behavior_automatic_cohort,
            )
            history_state = captured["world"]["behavior_history"]
            self.assertEqual(
                set(history_state),
                {
                    "history_incomplete",
                    "history_completions_not_recorded",
                    "creatures_evicted",
                    "bout_finalizations",
                    "why_summaries_finalized",
                    "duplicate_completions_ignored",
                    "creatures",
                },
            )
            self.assertNotIn("active_windows", serialized_keys)
            self.assertNotIn("completion_outbox", serialized_keys)
            self.assertNotIn("raw_probes", serialized_keys)
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
            restored_creature = restored.creatures[0]
            restored_report = restored.behavior_report_for(
                restored_creature.creature_id
            )
            self.assertIsNotNone(restored_report)
            self.assertEqual(len(restored_report.completed_bouts), 1)
            self.assertTrue(restored_report.history_incomplete)
            self.assertEqual(
                restored._behavior_automatic_cohort,
                world._behavior_automatic_cohort,
            )
            self.assertEqual(
                restored._flocking_capture_origin,
                world._flocking_capture_origin,
            )
            self.assertEqual(
                restored._flocking_capture_ordinal,
                world._flocking_capture_ordinal,
            )
            self.assertFalse(restored._flocking_capture_due_this_step)
            self.assertTrue(restored._behavior_cohort_dirty)
            self.assertEqual(
                restored_report.history_completions_not_recorded,
                2,
            )
            self.assertIs(
                restored._creature_by_shape_id[id(restored_creature.shape)],
                restored_creature,
            )

            self.assertEqual(luca.data_quality, "exact")
            self.assertEqual(luca.parent_species_id, None)
            self.assertEqual(luca.distances.composite_distance, 0.0)
            self.assertEqual(restored.species_history, world.species_history)
            restored_layout = build_species_tree_layout(
                restored.species_history
            )
            self.assertEqual(set(restored_layout.positions), {1, 2})
            self.assertEqual(restored_layout.roots, (1,))
            self.assertIn((1, 2), restored_layout.edges)
            self.assertEqual(
                restored.species_history[2].founder_color,
                (210, 40, 90),
            )
            self.assertEqual(
                restored.creatures[0].color,
                saved_member_color,
            )
            self.assertEqual(restored._physics_accumulator, 0.0)
            self.assertEqual(restored._simulation_step, 7)
            self.assertEqual(
                restored._speciation_adjustment_accumulator,
                1.75,
            )
            self.assertEqual(
                restored._mouth_exposures.state(),
                world._mouth_exposures.state(),
            )
            self.assertEqual(
                restored.simulation_lag_metrics.session_requested_seconds,
                0.0,
            )
            self.assertAlmostEqual(restored.creatures[0].stomach_energy, 0.42)
            self.assertAlmostEqual(
                restored.creatures[0].total_energy_gathered,
                2.75,
            )
            self.assertAlmostEqual(
                restored.creatures[0].stomach_difficulty_load,
                0.47,
            )
            self.assertEqual(
                (
                    restored.creatures[0].physical_traits.stomach_capacity,
                    restored.creatures[0].physical_traits.digestion_rate,
                    restored.creatures[0].physical_traits.digestion_efficiency,
                ),
                saved_digestive_traits,
            )
            self.assertAlmostEqual(restored.pheromones.accumulator, 0.1)
            self.assertAlmostEqual(
                float(restored.pheromones.trail.sum()),
                float(world.pheromones.trail.sum()),
            )
            self.assertAlmostEqual(
                float(restored.pheromones.alarm.sum()),
                float(world.pheromones.alarm.sum()),
            )
            self.assertEqual(restored.acoustics.signals, world.acoustics.signals)
            self.assertEqual(
                restored.neat_controller.species_manager.next_species_id,
                3,
            )

            species_manager = restored.neat_controller.species_manager
            representative_genome, physical_traits, vision, flocking_traits = (
                species_manager.representatives[1]
            )
            species_manager.compatibility_threshold = -1.0
            result = species_manager.evaluate_species(
                copy.deepcopy(representative_genome),
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
                1,
                restored.neat_controller.config.genome_config,
                copy.deepcopy(flocking_traits),
            )
            founder = restored.creatures[0]
            founder.lineage.species_id = result.species_id
            founder.color = restored.genotype_manager.new_species_color(
                founder.color,
                restored.rng,
            )
            restored._record_new_species(founder, result)
            post_load_layout = build_species_tree_layout(
                restored.species_history
            )
            self.assertEqual(result.species_id, 3)
            self.assertIn((1, 3), post_load_layout.edges)
            self.assertAlmostEqual(restored.foods[0].energy_value, saved_food_energy)
            self.assertAlmostEqual(
                restored.foods[0].original_radius,
                saved_food_original_radius,
            )
            self.assertIsNot(restored_brain, original_brain)
            self.assertEqual(
                restored_brain.export_network_state(),
                saved_network_state,
            )
            self.assertEqual(
                restored_brain.network.activate(recurrent_inputs),
                original_brain.network.activate(recurrent_inputs),
            )
            self.assertEqual(
                restored_brain.export_network_state(),
                original_brain.export_network_state(),
            )
            self.assertEqual(restored_brain.herding_decay_rate, 0.15)
            self.assertEqual(restored_brain.herding_state, 0.8)
            self.assertEqual(restored_brain.last_raw_herding, 0.0)
            self.assertEqual(restored.live_brain_count(), 1)
            self.assertEqual(restored.simulation_paths, world.simulation_paths)
            self.assertFalse(world.simulation_paths.telemetry_database.exists())
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_legacy_checkpoint_ignores_obsolete_biome_memory(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            creature_id = world.creatures[0].creature_id
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            state["version"] = 10
            state["brain_contract"]["sensor_schema"] = 1
            state["creatures"][0]["biome_fertility_ema"] = 0.37
            state["creatures"][0]["biome_fertility_ema_updated_at"] = 4.25
            state["world"]["previous_biome"] = {creature_id: 0.42}

            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
                allow_brain_contract_reset=True,
            )

            self.assertFalse(
                hasattr(restored.creatures[0], "biome_fertility_ema")
            )
            self.assertFalse(
                hasattr(
                    restored.creatures[0],
                    "biome_fertility_ema_updated_at",
                )
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_legacy_checkpoint_starts_with_clean_recurrent_state(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            brain = world.neat_controller.brain_for(
                world.creatures[0].creature_id
            )
            brain.network.activate(
                [0.5] * len(brain.network.input_nodes)
            )
            self.assertEqual(brain.network.active, 1)
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            state["version"] = 25
            state["creatures"][0]["scheduler_continuation"].pop(
                "brain_network_state"
            )

            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
            )
            restored_brain = restored.neat_controller.brain_for(
                restored.creatures[0].creature_id
            )
            restored_state = restored_brain.export_network_state()
            self.assertEqual(restored_state["active"], 0)
            self.assertTrue(
                all(
                    value == 0.0
                    for buffer in restored_state["values"]
                    for value in buffer.values()
                )
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_live_food_configuration_round_trips_with_spawner_state(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        original_max_food = self.config.food.max_food_items
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            world.set_live_food_config_value("forest_spawn_weight", 4.25)
            world.set_live_food_config_value("bushes_spawn_weight", 0.75)
            world.set_live_food_config_value("prairie_spawn_weight", 0.10)
            world.set_live_food_config_value("max_food_items", 777)
            world.set_live_food_config_value(
                "low_food_pressure_threshold",
                0.4,
            )
            world.set_live_food_config_value("critical_food_ratio", 0.2)
            world.set_live_food_config_value("low_food_burst_items", 125)
            world.set_live_food_config_value("low_food_burst_interval", 1.25)
            world.food_spawner._low_food_burst_credit = 0.6
            world.food_spawner._pending_low_food_burst_items = 9

            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
            )

            self.assertEqual(
                restored.live_food_config.to_primitive(),
                world.live_food_config.to_primitive(),
            )
            self.assertEqual(restored.food_spawner.config.max_food_items, 777)
            self.assertEqual(
                restored.food_spawner._low_food_burst_credit,
                0.6,
            )
            self.assertEqual(
                restored.food_spawner._pending_low_food_burst_items,
                9,
            )
            self.assertEqual(self.config.food.max_food_items, original_max_food)
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_checkpoint_without_live_food_configuration_uses_launch_values(
        self,
    ) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            state["world"].pop("live_food_config")
            self.config.food.max_food_items = 912
            self.config.biome.forest_spawn_weight = 3.75

            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
            )

            self.assertEqual(restored.live_food_config.max_food_items, 912)
            self.assertEqual(
                restored.live_food_config.forest_spawn_weight,
                3.75,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_version_14_checkpoint_migrates_digestive_state(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            state["version"] = 14
            legacy_physical = SimpleNamespace(
                radius=18.0,
                movement_cost_multiplier=1.1,
            )
            state["creatures"][0]["physical_traits"] = legacy_physical
            state["creatures"][0]["stomach_energy"] = 2.0
            state["creatures"][0].pop("stomach_difficulty_load")
            for species_id, representative in list(
                state["species_manager"]["representatives"].items()
            ):
                genome, _, vision, flocking = representative
                state["species_manager"]["representatives"][species_id] = (
                    genome,
                    legacy_physical,
                    vision,
                    flocking,
                )

            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
            )

            creature = restored.creatures[0]
            self.assertEqual(creature.physical_traits.stomach_capacity, 2.0)
            self.assertEqual(creature.physical_traits.digestion_rate, 0.2)
            self.assertEqual(
                creature.physical_traits.digestion_efficiency,
                0.9,
            )
            self.assertEqual(creature.stomach_energy, 2.0)
            self.assertEqual(creature.stomach_difficulty_load, 2.0)
            representative_traits = (
                restored.neat_controller.species_manager.representatives[1][1]
            )
            self.assertAlmostEqual(
                representative_traits.stomach_capacity,
                1.8,
            )
            self.assertEqual(representative_traits.digestion_rate, 0.2)
            self.assertEqual(
                representative_traits.digestion_efficiency,
                0.9,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_schema_2_checkpoint_starts_one_fresh_sensing_epoch(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 2
        self.config.food.initial_food_items = 1
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        round_tripped = None
        try:
            parent, infant = world.creatures
            infant.lineage.parent_id = parent.creature_id
            parent.age_seconds = 28.0
            infant.age_seconds = 7.0
            world.fitness[parent.creature_id].age_seconds = 28.0
            world.fitness[infant.creature_id].age_seconds = 7.0
            historical_root = replace(
                world.species_history[1],
                species_id=5,
                founder_creature_id=99,
                founder_genome_id=99,
            )
            world.species_history[5] = historical_root
            saved_positions = [creature.position for creature in world.creatures]
            saved_energies = [creature.energy for creature in world.creatures]

            state = PersistenceManager._capture_state(
                world,
                world.neat_controller,
            )
            old_genomes = state["population"]["genomes"]
            state["version"] = 10
            state["brain_contract"]["sensor_schema"] = 2
            state["brain_contract"]["inputs"] = 37
            for creature_state in state["creatures"]:
                creature_state.pop("total_energy_gathered", None)
                creature_state.pop("age_seconds", None)
                creature_state.pop("last_birth_time", None)
                creature_state.pop("lifetime_offspring_count", None)
            state["creatures"][0]["fitness"].__setstate__((None, {
                "age_seconds": 28.0,
                "energy_gained": 3.5,
            }))
            state["fitness_archive"] = {
                99: copy.deepcopy(state["creatures"][0]["fitness"])
            }
            state["rt_neat"]["eligible_parent_ids"] = [parent.creature_id]
            state["rt_neat"]["stats"].births = 12

            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
                allow_brain_contract_reset=True,
            )

            self.assertEqual(
                [creature.position for creature in restored.creatures],
                saved_positions,
            )
            self.assertEqual(
                [creature.energy for creature in restored.creatures],
                saved_energies,
            )
            self.assertEqual(
                restored.creatures[1].lineage.parent_id,
                parent.creature_id,
            )
            self.assertEqual(set(restored.species_history), {1, 5, 6})
            self.assertEqual(
                {creature.lineage.species_id for creature in restored.creatures},
                {6},
            )
            self.assertEqual(
                set(restored.neat_controller.species_manager.representatives),
                {6},
            )
            for creature, age, gathered in zip(
                restored.creatures,
                (28.0, 7.0),
                (3.5, 0.0),
            ):
                fitness = restored.fitness[creature.creature_id]
                self.assertEqual(fitness.age_seconds, age)
                self.assertEqual(fitness.evaluation_start_age_seconds, age)
                self.assertEqual(creature.total_energy_gathered, gathered)
                brain = restored.neat_controller.brain_for(creature.creature_id)
                self.assertIsNot(brain.genome, old_genomes[brain.genome_id])
            self.assertEqual(restored.fitness_archive, {})
            self.assertEqual(restored.rt_neat.eligible_parent_ids, [])
            self.assertEqual(restored.rt_neat.stats.births, 0)

            current_state = PersistenceManager._capture_state(
                restored,
                restored.neat_controller,
            )
            self.assertEqual(current_state["version"], CHECKPOINT_VERSION)
            self.assertEqual(current_state["brain_contract"]["sensor_schema"], 7)
            self.assertEqual(current_state["brain_contract"]["inputs"], 43)
            self.assertEqual(current_state["brain_contract"]["outputs"], 15)
            self.assertEqual(current_state["brain_contract"]["action_schema"], 2)
            round_tripped = PersistenceManager._restore_world(
                current_state,
                self.config,
                self.simulation_paths,
            )

            self.assertEqual(set(round_tripped.species_history), {1, 5, 6})
            self.assertEqual(
                set(round_tripped.neat_controller.species_manager.representatives),
                {6},
            )
            self.assertFalse(
                hasattr(round_tripped.creatures[0], "biome_fertility_ema")
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()
            if round_tripped is not None:
                round_tripped.close()

    def test_version_11_output_schema_migrates_genes_and_resets_brains(self) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            state = PersistenceManager._capture_state(world, world.neat_controller)
            original_brain = world.neat_controller.brain_for(1)
            state["version"] = 11
            state["brain_contract"]["outputs"] = 16
            state["brain_contract"].pop("action_schema")
            state["creatures"][0].pop("flocking_traits")
            state["creatures"][0]["lineage"].parent_id = 77
            state["creatures"][0]["lineage"].generation = 4
            state["creatures"][0]["lineage"].mutation_delta.radius = 1.25

            with self.assertRaises(CheckpointContractError):
                PersistenceManager._restore_world(
                    copy.deepcopy(state),
                    self.config,
                    self.simulation_paths,
                )
            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
                allow_brain_contract_reset=True,
            )

            self.assertEqual(
                restored.creatures[0].flocking_traits,
                FlockingTraits(0.5, 0.5, 0.5),
            )
            self.assertEqual(restored.creatures[0].lineage.parent_id, 77)
            self.assertEqual(restored.creatures[0].lineage.generation, 4)
            self.assertEqual(
                restored.creatures[0].lineage.mutation_delta.radius,
                1.25,
            )
            restored_brain = restored.neat_controller.brain_for(1)
            self.assertIsNot(restored_brain.genome, original_brain.genome)
            self.assertEqual(
                len(restored.neat_controller.config.genome_config.output_keys),
                15,
            )
        finally:
            world.close()
            if restored is not None:
                restored.close()

    def test_legacy_checkpoint_reconstructs_neat_allocators_before_mutation(
        self,
    ) -> None:
        from src.world import World

        self.config.persistence.enable_telemetry = False
        self.config.population.initial_creatures = 1
        self.config.food.initial_food_items = 0
        world = World(self.config, simulation_paths=self.simulation_paths)
        restored = None
        try:
            controller = world.neat_controller
            creature = world.creatures[0]
            brain = controller.brain_for(creature.creature_id)
            self.assertIsNotNone(brain)
            genome = brain.genome
            genome_config = controller.config.genome_config
            evolved_node_id = 1_000
            genome.nodes[evolved_node_id] = genome.create_node(
                genome_config,
                evolved_node_id,
            )
            max_saved_innovation = 5_000
            next(iter(genome.connections.values())).innovation = (
                max_saved_innovation
            )

            state = PersistenceManager._capture_state(world, controller)
            self.assertGreater(
                state["population"]["next_node_id"],
                evolved_node_id,
            )
            self.assertGreaterEqual(
                state["population"]["innovation_number"],
                max_saved_innovation,
            )
            state["population"].pop("next_node_id")
            state["population"].pop("innovation_number")

            restored = PersistenceManager._restore_world(
                state,
                self.config,
                self.simulation_paths,
            )
            restored_genome = restored.neat_controller.brain_for(
                creature.creature_id
            ).genome
            restored_config = restored.neat_controller.config.genome_config

            restored_genome.mutate_add_node(restored_config)

            self.assertIn(evolved_node_id + 1, restored_genome.nodes)
            self.assertGreater(
                restored_config.innovation_tracker.global_counter,
                max_saved_innovation,
            )
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


class SpeciesHistoryReconstructionTest(unittest.TestCase):
    def test_checkpoint_versions_two_through_seven_are_supported(self) -> None:
        for version in (2, 3, 4, 5, 6, 7):
            PersistenceManager._validate_state({"version": version})
        with self.assertRaises(ValueError):
            PersistenceManager._validate_state({"version": 1})

    def _record(
        self,
        species_id: int,
        parent_species_id: int | None,
        emerged_at: float,
    ) -> SpeciesRecord:
        zero = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
        return SpeciesRecord(
            species_id=species_id,
            parent_species_id=parent_species_id,
            founder_creature_id=species_id * 10,
            founder_genome_id=species_id,
            emerged_at=emerged_at,
            founder_color=(species_id * 10, 20, 30),
            data_quality="exact",
            founder_traits=SpeciesTraitSnapshot(12.0, 90.0, 0.35, 1.0),
            trait_deltas=zero,
            distances=SpeciesDistanceBreakdown(
                neat_distance=0.0,
                phenotypic_distance=0.0,
                weighted_phenotypic_distance=0.0,
                composite_distance=0.0,
                compatibility_threshold=3.0,
                phenotypic_weight=2.0,
                radius_component=0.0,
                vision_range_component=0.0,
                vision_angle_component=0.0,
                movement_cost_component=0.0,
            ),
        )

    def test_version_21_legacy_neural_shift_is_normalized_without_guessing_weights(self) -> None:
        PersistenceManager._validate_state({"version": 21})
        legacy = replace(
            self._record(2, 1, 4.5),
            neural_shifts=((8, -17, "weight", 0.6),),  # type: ignore[arg-type]
        )

        normalized = PersistenceManager._normalize_species_record(legacy)

        shift = normalized.neural_shifts[0]
        self.assertEqual((shift.source_node_id, shift.target_node_id), (-17, 8))
        self.assertEqual(shift.change_type, "changed")
        self.assertEqual(shift.weight_delta, 0.6)
        self.assertFalse(shift.weights_complete)

    def test_incomplete_legacy_shift_is_reconstructed_from_representatives(self) -> None:
        parent_genome = SimpleNamespace(
            nodes={},
            connections={
                (-17, 8): SimpleNamespace(enabled=True, weight=-1.2),
            },
        )
        child_genome = SimpleNamespace(
            nodes={},
            connections={
                (-17, 8): SimpleNamespace(enabled=True, weight=-0.6),
            },
        )
        legacy = replace(
            self._record(2, 1, 4.5),
            neural_shifts=((8, -17, "weight", 0.6),),  # type: ignore[arg-type]
        )
        controller = SimpleNamespace(
            species_manager=SimpleNamespace(
                representatives={
                    1: (parent_genome, None, None, None),
                    2: (child_genome, None, None, None),
                }
            )
        )

        enriched = PersistenceManager._enrich_species_neat_changes(
            {1: self._record(1, None, 0.0), 2: legacy},
            controller,
        )

        shift = enriched[2].neural_shifts[0]
        self.assertTrue(shift.weights_complete)
        self.assertEqual((shift.parent_weight, shift.child_weight), (-1.2, -0.6))
        self.assertAlmostEqual(shift.weight_delta, 0.6)

    def _reconstruction_inputs(
        self,
        lineage: list[tuple[int, int | None, float | None]],
    ) -> tuple[object, object]:
        config = build_sim_config()
        root_radius = config.trait.min_radius
        child_radius = root_radius + 0.5 * (
            config.trait.max_radius - config.trait.min_radius
        )
        root_vision_range = config.vision.min_range
        child_vision_range = root_vision_range + 0.2 * (
            config.vision.max_range - config.vision.min_range
        )
        root_genome = SimpleNamespace(key=1)
        child_genome = SimpleNamespace(
            key=2,
            distance=lambda other, genome_config: 2.0,
        )
        manager = SimpleNamespace(
            representatives={
                1: (
                    root_genome,
                    PhysicalTraits(radius=root_radius),
                    VisionTraits(
                        range=root_vision_range,
                        angle=config.vision.min_angle,
                    ),
                ),
                2: (
                    child_genome,
                    PhysicalTraits(radius=child_radius),
                    VisionTraits(
                        range=child_vision_range,
                        angle=config.vision.min_angle,
                    ),
                ),
            },
            compatibility_threshold=3.0,
            phenotypic_weight=2.0,
            trait_config=config.trait,
            vision_config=config.vision,
        )
        controller = SimpleNamespace(
            species_manager=manager,
            config=SimpleNamespace(genome_config=object()),
            genome_id_for=lambda creature_id: 2,
        )
        world = SimpleNamespace(
            telemetry=SimpleNamespace(load_species_lineage=lambda: lineage),
            creatures=[SimpleNamespace(creature_id=9, color=(10, 20, 30))],
            _trait_archive_by_genome_id={},
        )
        return world, controller

    def test_reconstructs_metrics_from_topology_and_representatives(self) -> None:
        world, controller = self._reconstruction_inputs(
            [(1, None, 0.0), (2, 1, 12.5)]
        )

        records = PersistenceManager._reconstruct_species_history(
            world,
            controller,
        )

        self.assertEqual(records[2].parent_species_id, 1)
        self.assertEqual(records[2].emerged_at, 12.5)
        self.assertEqual(records[2].data_quality, "reconstructed")
        self.assertAlmostEqual(records[2].distances.neat_distance, 2.0)
        self.assertAlmostEqual(records[2].distances.phenotypic_distance, 0.7)
        self.assertAlmostEqual(records[2].distances.composite_distance, 3.4)

    def test_missing_topology_creates_partial_records(self) -> None:
        world, controller = self._reconstruction_inputs([])
        world.telemetry = None

        records = PersistenceManager._reconstruct_species_history(
            world,
            controller,
        )

        self.assertEqual(records[1].data_quality, "reconstructed")
        self.assertEqual(records[2].data_quality, "partial")
        self.assertIsNone(records[2].parent_species_id)
        self.assertIsNone(records[2].distances.composite_distance)
        layout = build_species_tree_layout(records)
        self.assertEqual(set(layout.positions), {1, 2})
        self.assertEqual(layout.roots, (1, 2))

    def test_complete_checkpoint_history_is_authoritative(self) -> None:
        world, controller = self._reconstruction_inputs([])
        saved = {1: self._record(1, None, 0.0), 2: self._record(2, 1, 4.5)}
        world.elapsed_time = 8.0
        world.creatures[0].lineage = SimpleNamespace(species_id=2)
        world.telemetry = SimpleNamespace(
            load_species_records=lambda **kwargs: self.fail(
                "complete checkpoint should not consult telemetry"
            ),
        )

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            saved,
        )

        self.assertEqual(restored, saved)
        self.assertIsNot(restored, saved)

    def test_complete_legacy_history_recovers_missing_founder_color(self) -> None:
        world, controller = self._reconstruction_inputs([])
        world.elapsed_time = 8.0
        world.creatures[0].lineage = SimpleNamespace(species_id=2)
        saved = {
            1: self._record(1, None, 0.0),
            2: replace(
                self._record(2, 1, 4.5),
                founder_color=None,
            ),
        }

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            saved,
        )

        self.assertEqual(restored[2].founder_color, (10, 20, 30))

    def test_saved_founder_color_is_not_replaced_by_member_color(self) -> None:
        world, controller = self._reconstruction_inputs([])
        world.elapsed_time = 8.0
        world.creatures[0].lineage = SimpleNamespace(species_id=2)
        saved_color = (220, 30, 160)
        saved = {
            1: self._record(1, None, 0.0),
            2: replace(
                self._record(2, 1, 4.5),
                founder_color=saved_color,
            ),
        }

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            saved,
        )

        self.assertEqual(restored[2].founder_color, saved_color)

    def test_legacy_slotted_species_record_is_normalized_before_copying(
        self,
    ) -> None:
        world, controller = self._reconstruction_inputs([])
        world.elapsed_time = 8.0
        world.creatures[0].lineage = SimpleNamespace(species_id=1)
        current = self._record(1, None, 0.0)
        legacy = object.__new__(SpeciesRecord)
        legacy.__setstate__(
            [
                current.species_id,
                current.parent_species_id,
                current.founder_creature_id,
                current.founder_genome_id,
                current.emerged_at,
                current.founder_color,
                current.data_quality,
                current.founder_traits,
                current.trait_deltas,
                current.distances,
            ]
        )

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            {1: legacy},
        )

        self.assertIsNone(restored[1].neat_changes)

    def test_incomplete_history_uses_only_telemetry_at_checkpoint_time(
        self,
    ) -> None:
        world, controller = self._reconstruction_inputs(
            [(1, None, 0.0), (2, 1, 4.5), (3, 2, 12.0)]
        )
        current = self._record(2, 1, 4.5)
        future = self._record(3, 2, 12.0)
        world.elapsed_time = 8.0
        world.creatures[0].lineage = SimpleNamespace(species_id=2)
        world.telemetry = SimpleNamespace(
            load_species_lineage=lambda: [
                (1, None, 0.0),
                (2, 1, 4.5),
                (3, 2, 12.0),
            ],
            load_species_records=lambda *, up_to_time: (
                {2: current} if up_to_time == 8.0 else {2: current, 3: future}
            ),
        )

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            {1: self._record(1, None, 0.0)},
        )

        self.assertEqual(restored[2], current)
        self.assertNotIn(3, restored)

    def test_missing_extinct_species_is_recovered_from_telemetry(self) -> None:
        world, controller = self._reconstruction_inputs([(1, None, 0.0)])
        controller.species_manager.representatives = {
            1: controller.species_manager.representatives[1]
        }
        controller.species_manager.next_species_id = 3
        world.elapsed_time = 8.0
        world.creatures[0].lineage = SimpleNamespace(species_id=1)
        extinct = self._record(2, 1, 4.5)
        world.telemetry = SimpleNamespace(
            load_species_lineage=lambda: [(1, None, 0.0), (2, 1, 4.5)],
            load_species_records=lambda *, up_to_time: {2: extinct},
        )

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            {1: self._record(1, None, 0.0)},
        )

        self.assertEqual(restored[2], extinct)

    def test_next_species_id_is_above_history_representatives_and_living(
        self,
    ) -> None:
        manager = SimpleNamespace(
            representatives={1: object(), 4: object()},
            next_species_id=2,
        )
        controller = SimpleNamespace(species_manager=manager)
        creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=6)),
        ]

        PersistenceManager._reconcile_next_species_id(
            controller,
            {1: object(), 8: object()},
            creatures,
        )

        self.assertEqual(manager.next_species_id, 9)

    def test_missing_living_species_gets_partial_history_record(self) -> None:
        config = build_sim_config()
        manager = SimpleNamespace(
            representatives={},
            compatibility_threshold=3.0,
            phenotypic_weight=2.0,
            trait_config=config.trait,
            vision_config=config.vision,
        )
        controller = SimpleNamespace(
            species_manager=manager,
            config=SimpleNamespace(genome_config=object()),
            genome_id_for=lambda creature_id: 70,
        )
        creature = SimpleNamespace(
            creature_id=7,
            color=(70, 80, 90),
            lineage=SimpleNamespace(species_id=7),
            physical_traits=PhysicalTraits(radius=18.0),
            vision=VisionTraits(range=120.0, angle=1.2),
        )
        world = SimpleNamespace(
            elapsed_time=20.0,
            telemetry=None,
            creatures=[creature],
            _trait_archive_by_genome_id={},
        )

        restored = PersistenceManager._restore_species_history(
            world,
            controller,
            {},
        )

        self.assertEqual(restored[7].data_quality, "partial")
        self.assertEqual(restored[7].founder_creature_id, 7)
        self.assertEqual(restored[7].founder_genome_id, 70)
        self.assertEqual(restored[7].founder_color, (70, 80, 90))


if __name__ == "__main__":
    unittest.main()
