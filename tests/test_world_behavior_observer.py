from __future__ import annotations

from types import SimpleNamespace
import unittest

from configs.sim_config import build_sim_config
from src.behavior_history import (
    BehaviorTermination,
    CreatureBehaviorHistoryStore,
)
from src.behavior_observer import BehaviorSnapshot
from src.world import World


class _Observer:
    def __init__(self) -> None:
        self.submitted = []
        self.focuses = []
        self.poll_count = 0
        self.latest_snapshot = None
        self.finalized = []
        self.subjects = []
        self.latest_snapshots = {}

    def set_focus(self, creature_id, generation) -> None:
        self.focuses.append((creature_id, generation))
        self.latest_snapshot = None

    def submit(self, observation) -> bool:
        self.submitted.append(observation)
        return True

    def submit_batch(self, observations) -> bool:
        self.submitted.extend(observations)
        return True

    def set_subjects(self, subjects) -> None:
        self.subjects.append(tuple(subjects))

    def poll(self):
        self.poll_count += 1
        return self.latest_snapshot

    def finalize_focus(self, termination) -> bool:
        self.finalized.append(termination)
        return True


def world_shell() -> World:
    world = object.__new__(World)
    world.config = build_sim_config()
    world.elapsed_time = 0.0
    world.selected_creature_id = 4
    world._behavior_selection_generation = 2
    world._behavior_next_sample_time = 0.0
    world._behavior_food_consumption_count = 3
    world._behavior_food_consumed_energy_total = 0.25
    world._held_food_by_creature_id = {}
    world.behavior_observer = _Observer()
    velocity = SimpleNamespace(x=20.0, y=0.0)
    creature = SimpleNamespace(
        creature_id=4,
        position=(10.0, 20.0),
        heading=0.0,
        speed=20.0,
        body=SimpleNamespace(
            velocity=velocity,
            angular_velocity=0.4,
        ),
    )
    world.creatures = [creature]
    world._last_sensor_snapshots = {
        4: SimpleNamespace(
            food=SimpleNamespace(
                nearest_id=12,
                visible=1.0,
                surface_distance=80.0,
                relative_angle=0.2,
            ),
            flock=SimpleNamespace(
                flockmate_count=2.0,
                center_distance=70.0,
                cohesion_absolute_angle=0.1,
                actual_average_flockmate_velocity=(18.0, 1.0),
                visible_personal_space_count=0,
            ),
            pheromones=SimpleNamespace(
                alarm_here=0.3,
                alarm_forward_left=0.2,
                alarm_forward_right=0.22,
            ),
        )
    }
    return world


class WorldBehaviorObserverTest(unittest.TestCase):
    def test_sampling_uses_cached_raw_state_at_simulation_rate(self) -> None:
        world = world_shell()

        world._sample_selected_behavior()
        world.elapsed_time = 0.05
        world._sample_selected_behavior()
        world.elapsed_time = 0.10
        world._sample_selected_behavior()

        self.assertEqual(len(world.behavior_observer.submitted), 2)
        sample = world.behavior_observer.submitted[-1]
        self.assertEqual(sample.nearest_food_id, 12)
        self.assertEqual(sample.food_distance, 80.0)
        self.assertEqual(sample.food_relative_angle, 0.2)
        self.assertEqual(sample.food_consumption_count, 3)
        self.assertEqual(sample.selection_generation, 2)

    def test_selection_change_resets_event_totals_and_result(self) -> None:
        world = world_shell()
        world.behavior_observer.latest_snapshot = object()

        world._reset_behavior_focus(4)

        self.assertEqual(world._behavior_selection_generation, 3)
        self.assertEqual(world._behavior_food_consumption_count, 0)
        self.assertEqual(world._behavior_food_consumed_energy_total, 0.0)
        self.assertEqual(world.behavior_observer.focuses, [(4, 3)])
        self.assertIsNone(world.behavior_observer.latest_snapshot)

    def test_deselection_forces_old_focus_before_reset(self) -> None:
        world = world_shell()

        world.select_creature_by_id(None)

        self.assertEqual(
            world.behavior_observer.finalized,
            [BehaviorTermination.MODE_SWITCHED],
        )
        self.assertEqual(world.behavior_observer.focuses, [(None, 3)])

    def test_selection_starts_at_next_simulation_time_boundary(self) -> None:
        world = world_shell()
        world.elapsed_time = 0.03
        world._reset_behavior_focus(4)

        world._sample_selected_behavior()
        world.elapsed_time = 0.099
        world._sample_selected_behavior()
        self.assertEqual(world.behavior_observer.submitted, [])

        world.elapsed_time = 0.10
        world._sample_selected_behavior()
        self.assertEqual(len(world.behavior_observer.submitted), 1)

    def test_only_focal_consumption_updates_monotonic_event_totals(self) -> None:
        world = world_shell()

        world._record_behavior_food_consumption(
            SimpleNamespace(creature_id=8, energy_swallowed=1.0)
        )
        world._record_behavior_food_consumption(
            SimpleNamespace(creature_id=4, energy_swallowed=0.1)
        )

        self.assertEqual(world._behavior_food_consumption_count, 4)
        self.assertAlmostEqual(
            world._behavior_food_consumed_energy_total,
            0.35,
        )

    def test_selected_snapshot_rejects_stale_generation(self) -> None:
        world = world_shell()
        world.behavior_observer.latest_snapshot = BehaviorSnapshot(
            creature_id=4,
            selection_generation=1,
            simulation_time=0.0,
            behaviors=(),
            observations_processed=1,
            produced_monotonic=0.0,
        )

        self.assertIsNone(world.selected_behavior_snapshot)

    def test_paused_update_still_polls_worker(self) -> None:
        world = object.__new__(World)
        world.behavior_observer = _Observer()
        world.fps = 0.0
        world.is_paused = True
        world._refresh_stats = lambda: None

        world.update(0.0)

        self.assertEqual(world.behavior_observer.poll_count, 1)

    def test_automatic_cohort_is_stable_and_bounded_per_species(self) -> None:
        world = object.__new__(World)
        world.config = build_sim_config()
        world.selected_creature_id = None
        world.elapsed_time = 0.0
        world.behavior_observer = _Observer()
        world.behavior_history = CreatureBehaviorHistoryStore(
            max_completed_bouts_per_creature=4,
            max_remembered_creatures=4,
            minimum_stable_bouts=3,
        )
        world._behavior_automatic_cohort = {}
        world._behavior_active_subjects = {}
        world._behavior_subject_generation_counter = 0
        world._behavior_consumption_totals = {}

        def creature(creature_id, species_id):
            return SimpleNamespace(
                creature_id=creature_id,
                name=f"Creature {creature_id}",
                lineage=SimpleNamespace(species_id=species_id),
            )

        world.creatures = [
            *(creature(index, 1) for index in range(1, 6)),
            creature(6, 2),
            creature(7, 2),
        ]
        world._sync_automatic_behavior_cohort()
        initial = dict(world._behavior_automatic_cohort)

        self.assertEqual(len(initial[1]), 3)
        self.assertEqual(len(initial[2]), 2)
        world.creatures.append(creature(8, 1))
        world._sync_automatic_behavior_cohort()
        self.assertEqual(world._behavior_automatic_cohort[1], initial[1])

        removed = initial[1][0]
        world.creatures = [
            item for item in world.creatures if item.creature_id != removed
        ]
        world._sync_automatic_behavior_cohort()
        self.assertEqual(len(world._behavior_automatic_cohort[1]), 3)
        self.assertNotIn(removed, world._behavior_automatic_cohort[1])

    def test_worker_skip_diagnostics_accumulate_into_permanent_history(
        self,
    ) -> None:
        world = object.__new__(World)
        world.behavior_history = CreatureBehaviorHistoryStore(
            max_completed_bouts_per_creature=4,
            max_remembered_creatures=2,
            minimum_stable_bouts=3,
        )
        diagnostics = SimpleNamespace(
            history_completions_not_recorded=2
        )
        world.behavior_observer = SimpleNamespace(
            drain_completed_bouts=lambda: (),
            diagnostics=diagnostics,
        )
        world._behavior_history_worker_skipped_seen = 0

        world._drain_completed_behavior_bouts()
        diagnostics.history_completions_not_recorded = 5
        world._drain_completed_behavior_bouts()

        history = world.behavior_history.diagnostics
        self.assertTrue(history.history_incomplete)
        self.assertEqual(history.history_completions_not_recorded, 5)


if __name__ == "__main__":
    unittest.main()
