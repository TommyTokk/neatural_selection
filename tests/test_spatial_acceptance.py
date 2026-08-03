from __future__ import annotations

import unittest

from benchmarks.benchmark_multirate_scheduler import (
    collect_churn_counters,
    collect_counters,
)


class StaticSpatialCounterAcceptanceTest(unittest.TestCase):
    def test_sixty_step_static_population_counts_and_warm_allocations(self) -> None:
        counters = collect_counters(warmup_steps=30, measured_steps=60)
        self.assertEqual(counters["physics_steps"], 60)
        self.assertEqual(counters["spatial_rebuilds"], 60)
        self.assertEqual(counters["scheduled_shared_queries"], 1_100)
        self.assertEqual(counters["unscheduled_collision_queries"], 2_200)
        self.assertEqual(counters["spatial_queries"], 3_300)
        self.assertEqual(counters["motion_applications"], 3_300)
        self.assertEqual(counters["migrated_pymunk_point_queries"], 0)
        self.assertEqual(counters["child_full_scans"], 0)
        self.assertEqual(counters["child_result_lists"], 0)
        self.assertEqual(counters["candidate_wrappers"], 0)
        self.assertEqual(
            counters["ordinary_unselected_diagnostic_snapshots"], 0
        )
        self.assertEqual(counters["spatial_buffer_growth"], 0)
        self.assertEqual(counters["visible_index_growth"], 0)
        self.assertEqual(counters["occlusion_buffer_growth"], 0)
        self.assertEqual(counters["visible_food_id_growth"], 0)
        self.assertEqual(counters["vision_result_growth"], 0)
        self.assertEqual(counters["new_motion_commands"], 0)

    def test_churn_counts_follow_recorded_living_ids_and_phase_membership(self) -> None:
        counters = collect_churn_counters()
        self.assertEqual(counters["initial_population"], 55)
        self.assertEqual(counters["final_population"], 55)
        self.assertGreater(counters["removed_ids"], 0)
        self.assertEqual(counters["removed_ids"], counters["spawned_ids"])
        self.assertEqual(counters["reused_ids"], 0)
        self.assertEqual(
            counters["actual_rebuilds"],
            counters["expected_rebuilds"],
        )
        self.assertEqual(
            counters["actual_scheduled_queries"],
            counters["expected_scheduled_queries"],
        )
        self.assertEqual(
            counters["actual_collision_queries"],
            counters["expected_collision_queries"],
        )


if __name__ == "__main__":
    unittest.main()
