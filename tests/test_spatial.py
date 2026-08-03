from __future__ import annotations

from types import SimpleNamespace
import unittest

from src.spatial import BroadPhaseGeometry, CandidateBuffer, CreatureSpatialIndex


def creature(creature_id: int, x: float, y: float, radius: float, parent=None):
    return SimpleNamespace(
        creature_id=creature_id,
        body=SimpleNamespace(position=SimpleNamespace(x=x, y=y)),
        shape=SimpleNamespace(radius=radius),
        lineage=SimpleNamespace(parent_id=parent),
    )


class BroadPhaseGeometryTest(unittest.TestCase):
    def test_formulas_include_only_the_specified_radius_padding(self) -> None:
        geometry = BroadPhaseGeometry.calculate(
            observer_radius=12.0,
            maximum_target_radius=22.0,
            collision_margin=8.0,
            vision_range=100.0,
            flock_range=150.0,
            long_range=400.0,
            long_range_enabled=True,
        )
        self.assertEqual(geometry.collision, 42.0)
        self.assertEqual(geometry.detailed_vision, 126.2)
        self.assertEqual(geometry.flocking, 150.0)
        self.assertEqual(geometry.long_range, 400.0)
        self.assertEqual(geometry.scheduled, 400.0)

    def test_disabled_long_range_does_not_expand_scheduled_query(self) -> None:
        geometry = BroadPhaseGeometry.calculate(
            observer_radius=20.0,
            maximum_target_radius=30.0,
            collision_margin=5.0,
            vision_range=80.0,
            flock_range=70.0,
            long_range=1_000.0,
            long_range_enabled=False,
        )
        self.assertEqual(geometry.long_range, 0.0)
        self.assertEqual(geometry.scheduled, 117.0)

    def test_minimum_radii_and_exact_vision_origin_padding(self) -> None:
        geometry = BroadPhaseGeometry.calculate(
            observer_radius=0.0,
            maximum_target_radius=0.0,
            collision_margin=0.0,
            vision_range=25.0,
            flock_range=0.0,
        )
        self.assertEqual(geometry.collision, 0.0)
        self.assertEqual(geometry.detailed_vision, 25.0)
        self.assertEqual(geometry.scheduled, 25.0)

    def test_large_target_padding_does_not_leak_into_flock_ranges(self) -> None:
        geometry = BroadPhaseGeometry.calculate(
            observer_radius=40.0,
            maximum_target_radius=120.0,
            collision_margin=9.0,
            vision_range=200.0,
            flock_range=150.0,
            long_range=400.0,
            long_range_enabled=True,
        )
        self.assertEqual(geometry.collision, 169.0)
        self.assertEqual(geometry.detailed_vision, 334.0)
        self.assertEqual(geometry.flocking, 150.0)
        self.assertEqual(geometry.long_range, 400.0)


class CreatureSpatialIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.first = creature(1, 0.0, 0.0, 10.0)
        self.second = creature(2, 64.0, 0.0, 22.0, parent=1)
        self.registry = {1: self.first, 2: self.second}
        self.index = CreatureSpatialIndex(
            cell_size=64.0,
            living_registry=self.registry,
        )

    def test_query_traverses_scan_edge_cells_and_is_a_broad_superset(self) -> None:
        self.index.rebuild([self.first, self.second])
        output = CandidateBuffer(self.index)
        self.index.query_into(0.0, 0.0, 64.0, output)
        self.assertEqual(
            {candidate.creature_id for candidate in output},
            {1, 2},
        )

    def test_exact_zero_envelope_and_negative_cell_edge_are_inclusive(self) -> None:
        edge = creature(3, -64.0, -64.0, 3.0)
        self.registry[3] = edge
        self.index.rebuild([self.first, edge])
        at_origin = CandidateBuffer(self.index)
        self.index.query_into(0.0, 0.0, 0.0, at_origin)
        self.assertEqual([item.creature_id for item in at_origin], [1])
        at_edge = CandidateBuffer(self.index)
        self.index.query_into(-64.0, -64.0, 0.0, at_edge)
        self.assertEqual([item.creature_id for item in at_edge], [3])

    def test_generation_and_registry_invalidate_stale_slots(self) -> None:
        self.index.rebuild([self.first, self.second])
        output = CandidateBuffer(self.index)
        self.index.query_into(0.0, 0.0, 128.0, output)
        del self.registry[2]
        self.assertEqual(
            [candidate.creature_id for candidate in output],
            [1],
        )
        self.registry[2] = self.second
        self.index.rebuild([self.second, self.first])
        with self.assertRaises(RuntimeError):
            len(tuple(output))

    def test_failed_rebuild_publishes_no_partial_generation(self) -> None:
        self.index.rebuild([self.first, self.second])
        generation = self.index.generation
        duplicate = creature(1, 10.0, 10.0, 5.0)
        self.registry[1] = duplicate
        with self.assertRaises(ValueError):
            self.index.rebuild([duplicate, self.second, duplicate])
        self.assertFalse(self.index.valid)
        self.assertEqual(self.index.generation, generation)
        with self.assertRaises(RuntimeError):
            self.index.query_into(0.0, 0.0, 10.0, CandidateBuffer())

    def test_family_view_preserves_world_order_without_result_list(self) -> None:
        third = creature(3, 20.0, 0.0, 8.0, parent=1)
        self.registry[3] = third
        self.index.rebuild([self.first, third, self.second])
        view = self.index.family_view(1, lambda _child: True)
        self.assertEqual(
            [child.creature_id for child in view],
            [3, 2],
        )

    def test_only_previously_active_cells_are_reset_and_buffers_are_reused(self) -> None:
        self.index.rebuild([self.first, self.second])
        growth = self.index.counters.cell_buffer_growth
        active = len(self.index._active_cells)
        self.index.rebuild([self.first, self.second])
        self.assertEqual(self.index.counters.cell_resets, active)
        self.assertEqual(self.index.counters.cell_buffer_growth, growth)

    def test_newborn_is_registered_now_but_indexed_on_the_next_rebuild(self) -> None:
        self.index.rebuild([self.first])
        child = creature(3, 1.0, 0.0, 4.0, parent=1)
        self.registry[3] = child

        current = CandidateBuffer(self.index)
        self.index.query_into(0.0, 0.0, 10.0, current)
        self.assertEqual([item.creature_id for item in current], [1])
        self.assertEqual(list(self.index.family_view(1, lambda _item: True)), [])
        self.assertIsNone(self.index.values_for(child))

        self.index.rebuild([self.first, child])
        following = CandidateBuffer(self.index)
        self.index.query_into(0.0, 0.0, 10.0, following)
        self.assertEqual(
            [item.creature_id for item in following],
            [1, 3],
        )
        self.assertEqual(
            [item.creature_id for item in self.index.family_view(
                1, lambda _item: True
            )],
            [3],
        )


if __name__ == "__main__":
    unittest.main()
