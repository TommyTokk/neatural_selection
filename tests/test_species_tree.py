from __future__ import annotations

from dataclasses import dataclass
import unittest

from src.species_tree import (
    SpeciesTreeLayout,
    TreeLayoutManager,
    build_species_tree_layout,
    route_species_tree_edges,
    species_tree_line_width,
)


@dataclass(frozen=True)
class _Record:
    species_id: int
    parent_species_id: int | None
    emerged_at: float | None = None


class SpeciesTreeLayoutTest(unittest.TestCase):
    def test_tree_depths_edges_and_distinct_living_lanes(self) -> None:
        records = {
            1: _Record(1, None, 0.0),
            2: _Record(2, 1, 10.0),
            3: _Record(3, 1, 20.0),
            4: _Record(4, 2, 30.0),
        }

        layout = build_species_tree_layout(records)

        self.assertEqual(layout.roots, (1,))
        self.assertEqual(layout.depths, {1: 0, 2: 1, 4: 2, 3: 1})
        self.assertEqual(layout.edges, ((1, 2), (1, 3), (2, 4)))
        self.assertLess(layout.positions[1][1], layout.positions[2][1])
        self.assertLess(layout.positions[2][1], layout.positions[4][1])
        self.assertNotEqual(layout.positions[1][0], layout.positions[2][0])
        self.assertGreater(layout.positions[3][0], layout.positions[1][0])
        self.assertEqual(
            layout.positions[4][1] - layout.positions[2][1],
            20.0 * 2.0,
        )

    def test_living_siblings_balance_around_their_parent(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 0.0),
                2: _Record(2, 1, 10.0),
                3: _Record(3, 1, 20.0),
                4: _Record(4, 1, 30.0),
                5: _Record(5, 1, 40.0),
            }
        )

        self.assertEqual(layout.lanes[1], 0)
        self.assertEqual(
            [layout.lanes[species_id] for species_id in (2, 3, 4, 5)],
            [-1, 1, -2, 2],
        )
        self.assertEqual(layout.content_left, -2 * 92.0 - 48.0)
        self.assertEqual(layout.leaf_count, 5)

    def test_independent_living_roots_are_centered_around_origin(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 0.0),
                2: _Record(2, None, 10.0),
                3: _Record(3, None, 20.0),
            }
        )

        self.assertEqual(layout.lanes, {1: 0, 2: -1, 3: 1})

    def test_child_reuses_parent_lane_after_parent_has_ended(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 0.0),
                2: _Record(2, 1, 15.0),
            },
            species_end_times={1: 10.0, 2: float("inf")},
        )

        self.assertEqual(layout.lanes[1], layout.lanes[2])

    def test_lane_reuses_only_after_species_has_ended(self) -> None:
        records = {
            1: _Record(1, None, 0.0),
            2: _Record(2, None, 15.0),
            3: _Record(3, None, 20.0),
        }

        layout = build_species_tree_layout(
            records,
            species_end_times={1: 10.0, 2: 20.0, 3: 30.0},
        )

        self.assertEqual(layout.lanes[1], layout.lanes[2])
        self.assertNotEqual(layout.lanes[2], layout.lanes[3])

    def test_living_species_reserves_lane(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 0.0),
                2: _Record(2, None, 15.0),
            },
            species_end_times={1: float("inf"), 2: 20.0},
        )

        self.assertNotEqual(layout.lanes[1], layout.lanes[2])

    def test_new_species_can_reuse_lane_freed_in_same_sync(self) -> None:
        records = {1: _Record(1, None, 0.0)}
        manager = TreeLayoutManager()
        manager.sync(
            records,
            timeline_end=5.0,
            species_end_times={1: float("inf")},
        )
        records[2] = _Record(2, None, 15.0)

        layout = manager.sync(
            records,
            timeline_end=15.0,
            species_end_times={1: 10.0, 2: float("inf")},
        )

        self.assertEqual(layout.lanes[1], layout.lanes[2])

    def test_revived_species_rebuilds_overlapping_lanes(self) -> None:
        records = {
            1: _Record(1, None, 0.0),
            2: _Record(2, None, 15.0),
        }
        manager = TreeLayoutManager()
        compact = manager.sync(
            records,
            species_end_times={1: 10.0, 2: float("inf")},
        )
        self.assertEqual(compact.lanes[1], compact.lanes[2])

        rebuilt = manager.sync(
            records,
            species_end_times={1: float("inf"), 2: float("inf")},
        )

        self.assertNotEqual(rebuilt.lanes[1], rebuilt.lanes[2])

    def test_descendant_counts_and_spindle_widths(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 0.0),
                2: _Record(2, 1, 10.0),
                3: _Record(3, 1, 20.0),
                4: _Record(4, 2, 30.0),
            }
        )

        self.assertEqual(
            layout.descendant_counts,
            {1: 3, 2: 1, 3: 0, 4: 0},
        )
        self.assertEqual(species_tree_line_width(0), 1.0)
        self.assertGreater(species_tree_line_width(50), 1.0)
        self.assertEqual(species_tree_line_width(1_000_000), 10.0)

    def test_branch_connector_meets_parent_at_child_emergence(self) -> None:
        manager = TreeLayoutManager()
        layout = manager.sync(
            {
                1: _Record(1, None, 0.0),
                2: _Record(2, 1, 15.0),
            },
            species_end_times={1: float("inf"), 2: float("inf")},
        )

        route = manager.routes[(1, 2)]
        self.assertEqual(route[0][0], layout.positions[1][0])
        self.assertEqual(route[0][1], layout.positions[2][1])
        self.assertEqual(route[-1], layout.positions[2])

    def test_layout_is_deterministic_for_different_mapping_order(self) -> None:
        forward = {
            1: _Record(1, None),
            2: _Record(2, 1),
            3: _Record(3, 1),
        }
        reverse = dict(reversed(tuple(forward.items())))

        self.assertEqual(
            build_species_tree_layout(forward),
            build_species_tree_layout(reverse),
        )

    def test_missing_parent_becomes_an_additional_root(self) -> None:
        records = {
            1: _Record(1, None),
            4: _Record(4, 99),
            5: _Record(5, 4),
        }

        layout = build_species_tree_layout(records)

        self.assertEqual(layout.roots, (1, 4))
        self.assertIn((4, 5), layout.edges)
        self.assertEqual(layout.depths[5], 1)

    def test_cycle_is_broken_without_losing_nodes(self) -> None:
        records = {
            2: _Record(2, 3),
            3: _Record(3, 2),
            4: _Record(4, 3),
        }

        layout = build_species_tree_layout(records)

        self.assertEqual(layout.roots, (2,))
        self.assertEqual(set(layout.positions), {2, 3, 4})
        self.assertEqual(layout.edges, ((2, 3), (3, 4)))

    def test_empty_history_has_zero_sized_layout(self) -> None:
        layout = build_species_tree_layout({}, timeline_end=25.0)

        self.assertEqual(layout.positions, {})
        self.assertEqual(layout.content_width, 0.0)
        self.assertEqual(layout.content_height, 0.0)
        self.assertEqual(layout.timeline_end, 25.0)

    def test_missing_and_invalid_times_use_one_visual_generation_gap(self) -> None:
        records = {
            1: _Record(1, None, 5.0),
            2: _Record(2, 1, None),
            3: _Record(3, 2, float("nan")),
            4: _Record(4, 3, -1.0),
        }

        layout = build_species_tree_layout(records)

        self.assertEqual(layout.effective_times[1], 5.0)
        self.assertEqual(layout.effective_times[2], 37.0)
        self.assertEqual(layout.effective_times[3], 69.0)
        self.assertEqual(layout.effective_times[4], 101.0)
        self.assertEqual(layout.timeline_end, 5.0)

    def test_timeline_end_extends_content_beyond_latest_event(self) -> None:
        layout = build_species_tree_layout(
            {1: _Record(1, None, 10.0)},
            timeline_end=90.0,
        )

        self.assertEqual(layout.timeline_start, 0.0)
        self.assertEqual(layout.timeline_end, 90.0)
        self.assertEqual(layout.content_height, 96.0 + 90.0 * 2.0)

    def test_child_recorded_before_parent_never_draws_upward(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 30.0),
                2: _Record(2, 1, 10.0),
            }
        )

        self.assertEqual(layout.effective_times[1], 30.0)
        self.assertEqual(layout.effective_times[2], 30.0)
        self.assertGreaterEqual(layout.positions[2][1], layout.positions[1][1])

    def test_edge_routing_avoids_an_unrelated_node(self) -> None:
        layout = SpeciesTreeLayout(
            positions={1: (50.0, 20.0), 2: (50.0, 100.0), 3: (50.0, 180.0)},
            edges=((1, 3),),
            depths={1: 0, 2: 0, 3: 1},
            effective_times={1: 0.0, 2: 40.0, 3: 80.0},
            roots=(1, 2),
            content_width=100.0,
            content_height=220.0,
            leaf_count=2,
            timeline_start=0.0,
            timeline_end=80.0,
        )

        route = route_species_tree_edges(
            layout,
            {1: 10.0, 2: 10.0, 3: 10.0},
        )[(1, 3)]

        self.assertAlmostEqual(
            (route[0][0] - 50.0) ** 2 + (route[0][1] - 20.0) ** 2,
            100.0,
        )
        self.assertAlmostEqual(
            (route[-1][0] - 50.0) ** 2 + (route[-1][1] - 180.0) ** 2,
            100.0,
        )
        self.assertTrue(
            all(
                not self._segment_crosses_box(
                    start, end, 34.0, 66.0, 84.0, 116.0
                )
                for start, end in zip(route, route[1:])
            )
        )

    def test_edge_routing_is_deterministic_with_equal_timestamps(self) -> None:
        layout = build_species_tree_layout(
            {
                1: _Record(1, None, 10.0),
                2: _Record(2, 1, 10.0),
                3: _Record(3, 1, 10.0),
            }
        )
        radii = {1: 12.0, 2: 12.0, 3: 12.0}

        self.assertEqual(
            route_species_tree_edges(layout, radii),
            route_species_tree_edges(layout, radii),
        )

    def test_incremental_sync_preserves_existing_coordinates(self) -> None:
        records = {
            1: _Record(1, None, 0.0),
            2: _Record(2, 1, 10.0),
        }
        manager = TreeLayoutManager()
        first = manager.sync(records)
        original_positions = dict(first.positions)
        placements = manager.placement_count

        reopened = manager.sync(records)
        self.assertEqual(manager.placement_count, placements)
        self.assertEqual(reopened.positions, original_positions)

        records[3] = _Record(3, 1, 20.0)
        updated = manager.sync(records)
        self.assertEqual(manager.placement_count, placements + 1)
        self.assertEqual(
            {key: updated.positions[key] for key in original_positions},
            original_positions,
        )

    def test_two_hour_species_uses_bucket_four(self) -> None:
        manager = TreeLayoutManager(bucket_seconds=1800.0)
        manager.sync({1: _Record(1, None, 7200.0)})

        self.assertEqual(manager.bucket_for_time(7200.0), 4)
        self.assertEqual(manager.bucket_summaries()[0].bucket_id, 4)

    def test_large_viewport_query_returns_only_intersecting_bucket(self) -> None:
        records = {
            species_id: _Record(
                species_id,
                None if species_id == 1 else species_id - 1,
                (species_id - 1) * (36000.0 / 4999.0),
            )
            for species_id in range(1, 5001)
        }
        manager = TreeLayoutManager()
        layout = manager.sync(records, timeline_end=36000.0)
        hour_nine_y = 9.0 * 3600.0 * manager.time_scale + manager.padding

        visible = manager.viewport_slice(
            left=layout.content_left,
            right=layout.content_left + layout.content_width,
            top=hour_nine_y,
            bottom=hour_nine_y + 900.0,
        )

        self.assertGreater(len(visible.node_ids), 0)
        self.assertLess(len(visible.node_ids), 500)
        self.assertTrue(
            all(
                17
                <= manager.bucket_for_time(layout.effective_times[species_id])
                <= 19
                for species_id in visible.node_ids
            )
        )

    @staticmethod
    def _segment_crosses_box(
        start: tuple[float, float],
        end: tuple[float, float],
        left: float,
        right: float,
        top: float,
        bottom: float,
    ) -> bool:
        if start[0] == end[0]:
            low, high = sorted((start[1], end[1]))
            return left < start[0] < right and low < bottom and high > top
        low, high = sorted((start[0], end[0]))
        return top < start[1] < bottom and low < right and high > left


if __name__ == "__main__":
    unittest.main()
