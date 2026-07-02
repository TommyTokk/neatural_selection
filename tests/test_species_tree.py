from __future__ import annotations

from dataclasses import dataclass
import unittest

from src.species_tree import build_species_tree_layout


@dataclass(frozen=True)
class _Record:
    species_id: int
    parent_species_id: int | None
    emerged_at: float | None = None


class SpeciesTreeLayoutTest(unittest.TestCase):
    def test_tree_depths_edges_and_parent_centering(self) -> None:
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
        self.assertEqual(
            layout.positions[1][0],
            (layout.positions[2][0] + layout.positions[3][0]) * 0.5,
        )
        self.assertEqual(
            layout.positions[4][1] - layout.positions[2][1],
            20.0 * 2.0,
        )

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


if __name__ == "__main__":
    unittest.main()
