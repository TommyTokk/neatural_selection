from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from configs.sim_config import build_sim_config
from src.behavior_history import BehaviorTermination
from src.neat_controller import SpeciationResult
from src.persistence import CheckpointTarget, SavePriority
from src.speciation import SpeciesDistanceBreakdown, SpeciesTraitSnapshot
from src.world import World


class _Telemetry:
    def __init__(self) -> None:
        self.metrics: list[tuple[object, ...]] = []
        self.births: list[tuple[object, ...]] = []
        self.species: list[tuple[object, ...]] = []
        self.species_records: list[object] = []
        self.deaths: list[tuple[object, ...]] = []

    def log_metrics(self, *values: object) -> None:
        self.metrics.append(values)

    def log_creature_birth(self, *values: object) -> None:
        self.births.append(values)

    def log_species(self, *values: object) -> None:
        self.species.append(values)

    def log_species_record(self, record: object) -> None:
        self.species_records.append(record)

    def log_creature_death(self, *values: object) -> None:
        self.deaths.append(values)


class _Persistence:
    def __init__(self) -> None:
        self.saves: list[tuple[object, ...]] = []
        self.priorities: list[SavePriority] = []
        self.is_busy = False
        self.error: Exception | None = None

    def save_simulation(
        self,
        *values: object,
        priority: SavePriority = SavePriority.AUTO,
    ) -> None:
        if self.error is not None:
            raise self.error
        self.saves.append(values)
        self.priorities.append(priority)


class WorldPersistenceTimerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.world = object.__new__(World)
        self.world.config = build_sim_config()
        self.world.config.persistence.quick_save_interval_seconds = 10.0
        self.world.config.persistence.archive_save_interval_seconds = 30.0
        self.world.time_since_last_quick_save = 0.0
        self.world.time_since_last_archive_save = 0.0
        self.world.simulation_paths = SimpleNamespace(
            quick_target=lambda: CheckpointTarget(
                Path("checkpoint.pkl"),
                rotate_backup=True,
            ),
            hourly_target=lambda: CheckpointTarget(
                Path("hourly/checkpoint_1.pkl"),
                rotate_backup=False,
            ),
        )
        self.world.elapsed_time = 25.0
        self.world.creatures = [object(), object()]
        self.world.foods = [object()]
        self.world.rt_neat = SimpleNamespace(
            stats=SimpleNamespace(best_fitness=4.5)
        )
        self.world.neat_controller = object()
        self.world.telemetry = _Telemetry()
        self.world.persistence_manager = _Persistence()

    def test_default_intervals_are_two_and_sixty_minutes(self) -> None:
        config = build_sim_config()

        self.assertEqual(
            config.persistence.quick_save_interval_seconds,
            120.0,
        )
        self.assertEqual(
            config.persistence.archive_save_interval_seconds,
            3600.0,
        )

    def test_exact_interval_triggers_once(self) -> None:
        self.world._update_persistence_timer(4.0)
        self.world._update_persistence_timer(6.0)
        self.world._update_persistence_timer(1.0)

        self.assertEqual(len(self.world.persistence_manager.saves), 1)
        self.assertEqual(
            self.world.persistence_manager.priorities,
            [SavePriority.AUTO],
        )
        self.assertEqual(len(self.world.telemetry.metrics), 1)
        self.assertEqual(self.world.time_since_last_quick_save, 1.0)
        self.assertEqual(self.world.time_since_last_archive_save, 11.0)
        targets = self.world.persistence_manager.saves[0][2]
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].path, Path("checkpoint.pkl"))

    def test_timer_uses_unscaled_delta_time(self) -> None:
        self.world.simulation_speed = 5.0

        self.world._update_persistence_timer(2.0)

        self.assertEqual(self.world.time_since_last_quick_save, 2.0)
        self.assertEqual(self.world.time_since_last_archive_save, 2.0)
        self.assertEqual(self.world.persistence_manager.saves, [])

    def test_hourly_boundary_captures_once_for_both_tiers(self) -> None:
        self.world._update_persistence_timer(30.0)

        self.assertEqual(len(self.world.persistence_manager.saves), 1)
        self.assertEqual(len(self.world.telemetry.metrics), 1)
        targets = self.world.persistence_manager.saves[0][2]
        self.assertEqual(
            {target.path for target in targets},
            {
                Path("checkpoint.pkl"),
                Path("hourly/checkpoint_1.pkl"),
            },
        )
        self.assertEqual(self.world.time_since_last_quick_save, 0.0)
        self.assertEqual(self.world.time_since_last_archive_save, 0.0)

    def test_manual_save_uses_quick_target_and_resets_only_quick_timer(
        self,
    ) -> None:
        self.world.time_since_last_quick_save = 8.0
        self.world.time_since_last_archive_save = 21.0

        self.world.save_now()

        self.assertEqual(self.world.time_since_last_quick_save, 0.0)
        self.assertEqual(self.world.time_since_last_archive_save, 21.0)
        self.assertEqual(len(self.world.persistence_manager.saves), 1)
        targets = self.world.persistence_manager.saves[0][2]
        self.assertEqual(
            [target.path for target in targets],
            [Path("checkpoint.pkl")],
        )
        self.assertEqual(
            self.world.persistence_manager.priorities,
            [SavePriority.MANUAL],
        )
        self.assertEqual(len(self.world.telemetry.metrics), 1)

    def test_manual_save_restores_quick_timer_when_capture_fails(self) -> None:
        self.world.time_since_last_quick_save = 8.0
        self.world.time_since_last_archive_save = 21.0
        self.world.persistence_manager.error = RuntimeError("capture failed")

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            self.world.save_now()

        self.assertEqual(self.world.time_since_last_quick_save, 8.0)
        self.assertEqual(self.world.time_since_last_archive_save, 21.0)
        self.assertEqual(self.world.telemetry.metrics, [])

    def test_save_in_progress_reflects_persistence_manager(self) -> None:
        self.assertFalse(self.world.save_in_progress)

        self.world.persistence_manager.is_busy = True

        self.assertTrue(self.world.save_in_progress)

    def test_birth_and_species_helpers_forward_phylogeny(self) -> None:
        creature = SimpleNamespace(
            creature_id=9,
            lineage=SimpleNamespace(species_id=3),
            vision=SimpleNamespace(range=125.0, angle=1.0),
            physical_traits=SimpleNamespace(
                radius=17.0,
                movement_cost_multiplier=1.0,
            ),
            radius=17.0,
            color=(10, 20, 30),
        )
        zero = SpeciesTraitSnapshot(0.0, 0.0, 0.0, 0.0)
        result = SpeciationResult(
            species_id=3,
            parent_species_id=1,
            is_new_species=True,
            founder_traits=SpeciesTraitSnapshot(17.0, 125.0, 1.0, 1.0),
            trait_deltas=zero,
            distances=SpeciesDistanceBreakdown(
                neat_distance=3.1,
                phenotypic_distance=0.0,
                weighted_phenotypic_distance=0.0,
                composite_distance=3.1,
                compatibility_threshold=3.0,
                phenotypic_weight=2.0,
                radius_component=0.0,
                vision_range_component=0.0,
                vision_angle_component=0.0,
                movement_cost_component=0.0,
            ),
        )
        self.world.neat_controller = SimpleNamespace(
            genome_id_for=lambda creature_id: 30
        )
        self.world.species_history = {}

        self.world._record_new_species(creature, result)
        self.world._log_creature_birth(creature)

        self.assertEqual(len(self.world.telemetry.species_records), 1)
        self.assertEqual(self.world.species_history[3].parent_species_id, 1)
        self.assertEqual(
            self.world.telemetry.births,
            [(9, 3, 25.0, 125.0, 17.0)],
        )

    def test_luca_record_uses_first_creature_and_zero_distances(self) -> None:
        founder = SimpleNamespace(
            creature_id=1,
            color=(10, 20, 30),
            physical_traits=SimpleNamespace(
                radius=16.0,
                movement_cost_multiplier=1.0,
            ),
            vision=SimpleNamespace(range=100.0, angle=1.0),
        )
        self.world.creatures = [founder]
        self.world.species_history = {}
        self.world.elapsed_time = 0.0
        self.world.neat_controller = SimpleNamespace(
            genome_id_for=lambda creature_id: 11,
            species_manager=SimpleNamespace(
                compatibility_threshold=3.0,
                phenotypic_weight=2.0,
            ),
        )

        self.world._initialize_luca_record()

        record = self.world.species_history[1]
        self.assertEqual(record.founder_creature_id, 1)
        self.assertEqual(record.founder_genome_id, 11)
        self.assertEqual(record.founder_color, (10, 20, 30))
        self.assertEqual(record.data_quality, "exact")
        self.assertEqual(record.distances.composite_distance, 0.0)

    def test_remove_creature_logs_death_reason(self) -> None:
        creature = SimpleNamespace(
            creature_id=7,
            name="Seven",
            body=object(),
            shape=object(),
        )
        self.world._archive_creature_traits = lambda value: None
        self.world._release_food_for = lambda value: None
        self.world.fitness = {}
        self.world.fitness_archive = {}
        self.world.neat_controller = SimpleNamespace()
        self.world.creatures = []
        self.world.rt_neat = SimpleNamespace(record_death=lambda fitness: None)
        self.world.selected_creature_id = 7
        self.world.behavior_observer = Mock()
        self.world.behavior_history = Mock()
        self.world._chronometers = {}
        self.world._remove_creature(creature, death_reason="manual")

        self.assertEqual(self.world.telemetry.deaths, [(7, 25.0, "manual")])
        self.world.behavior_observer.finalize_focus.assert_called_once_with(
            BehaviorTermination.CREATURE_DIED
        )
        self.world.behavior_history.mark_deceased.assert_called_once_with(
            7,
            25.0,
        )


if __name__ == "__main__":
    unittest.main()
