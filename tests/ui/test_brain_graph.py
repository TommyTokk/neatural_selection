from __future__ import annotations

from dataclasses import dataclass
import unittest

from src.ui.layouts.brain_graph import (
    BrainEdgeKind,
    BrainNodeKind,
    build_brain_graph_layout,
    highlighted_path_through_node,
)


@dataclass(slots=True)
class FakeBounds:
    left: float = 0.0
    right: float = 600.0
    bottom: float = 0.0
    top: float = 300.0

    @property
    def width(self) -> float:
        return self.right - self.left


@dataclass(slots=True)
class FakeConnection:
    key: tuple[int, int]
    weight: float = 1.0
    enabled: bool = True


@dataclass(slots=True)
class FakeGenome:
    nodes: dict[int, object]
    connections: dict[tuple[int, int], FakeConnection]


def genome_with_connections(
    hidden_keys: list[int],
    connections: list[tuple[int, int]],
) -> FakeGenome:
    output_keys = [0]
    nodes = {key: object() for key in [*output_keys, *hidden_keys]}
    return FakeGenome(
        nodes=nodes,
        connections={
            key: FakeConnection(key)
            for key in connections
        },
    )


def layout_for(genome: FakeGenome):
    return build_brain_graph_layout(
        genome,
        input_keys=[-1],
        output_keys=[0],
        bounds=FakeBounds(),
        input_labels=["sensor"],
        output_labels=["action"],
    )


class BrainGraphLayoutTest(unittest.TestCase):
    def test_input_column_preserves_configured_top_to_bottom_order(self) -> None:
        input_keys = list(range(-1, -28, -1))
        layout = build_brain_graph_layout(
            genome_with_connections([], []),
            input_keys=input_keys,
            output_keys=[0],
            bounds=FakeBounds(),
            input_labels=[str(key) for key in input_keys],
            output_labels=["action"],
        )

        visual_order = sorted(
            input_keys,
            key=lambda key: layout.positions[key][1],
            reverse=True,
        )

        self.assertEqual(visual_order, input_keys)
        self.assertEqual(layout.nodes[-1].label, "-1")
        self.assertEqual(layout.nodes[-27].label, "-27")

    def test_direct_input_to_output_graph_uses_outer_columns(self) -> None:
        layout = layout_for(genome_with_connections([], [(-1, 0)]))

        self.assertEqual(layout.nodes[-1].kind, BrainNodeKind.INPUT)
        self.assertEqual(layout.nodes[0].kind, BrainNodeKind.OUTPUT)
        self.assertLess(layout.positions[-1][0], layout.positions[0][0])
        self.assertEqual(layout.edges[0].kind, BrainEdgeKind.FORWARD)

    def test_hidden_chain_gets_multiple_depth_columns(self) -> None:
        layout = layout_for(
            genome_with_connections(
                [1, 2],
                [
                    (-1, 1),
                    (1, 2),
                    (2, 0),
                ],
            )
        )

        self.assertEqual(layout.nodes[1].depth, 1)
        self.assertEqual(layout.nodes[2].depth, 2)
        self.assertLess(layout.positions[1][0], layout.positions[2][0])

    def test_backward_edge_is_classified_as_recurrent(self) -> None:
        layout = layout_for(
            genome_with_connections(
                [1, 2],
                [
                    (-1, 1),
                    (1, 2),
                    (2, 1),
                    (2, 0),
                ],
            )
        )

        edge_kinds = {
            (edge.source, edge.target): edge.kind
            for edge in layout.edges
        }
        self.assertEqual(edge_kinds[(2, 1)], BrainEdgeKind.RECURRENT)

    def test_self_loop_is_classified_separately(self) -> None:
        layout = layout_for(
            genome_with_connections(
                [1],
                [
                    (-1, 1),
                    (1, 1),
                    (1, 0),
                ],
            )
        )

        edge_kinds = {
            (edge.source, edge.target): edge.kind
            for edge in layout.edges
        }
        self.assertEqual(edge_kinds[(1, 1)], BrainEdgeKind.SELF_LOOP)

    def test_disabled_connection_keeps_disabled_state(self) -> None:
        genome = genome_with_connections([1], [(-1, 1), (1, 0)])
        genome.connections[(1, 0)].enabled = False

        layout = layout_for(genome)

        disabled_edges = [
            edge for edge in layout.edges
            if edge.source == 1 and edge.target == 0
        ]
        self.assertEqual(len(disabled_edges), 1)
        self.assertFalse(disabled_edges[0].enabled)

    def test_disconnected_hidden_node_is_placed_without_crashing(self) -> None:
        layout = layout_for(genome_with_connections([1], [(-1, 0)]))

        self.assertEqual(layout.nodes[1].kind, BrainNodeKind.HIDDEN)
        self.assertIn(1, layout.positions)
        self.assertEqual(layout.nodes[1].depth, 1)

    def test_highlighted_path_includes_full_upstream_and_downstream_route(self) -> None:
        genome = genome_with_connections(
            [1, 2, 3],
            [(-1, 1), (1, 2), (2, 0), (-1, 3), (3, 0)],
        )

        highlight = highlighted_path_through_node(layout_for(genome), 1)

        self.assertEqual(highlight.nodes, {-1, 1, 2, 0})
        self.assertEqual(highlight.edges, {(-1, 1), (1, 2), (2, 0)})
        self.assertEqual(highlight.direct_edges, {(-1, 1), (1, 2)})
        self.assertEqual(highlight.upstream_edges, {(-1, 1)})
        self.assertEqual(highlight.downstream_edges, {(1, 2), (2, 0)})

    def test_highlighted_path_excludes_disabled_connections(self) -> None:
        genome = genome_with_connections([1, 2], [(-1, 1), (1, 0), (1, 2), (2, 0)])
        genome.connections[(1, 2)].enabled = False

        highlight = highlighted_path_through_node(layout_for(genome), 1)

        self.assertEqual(highlight.nodes, {-1, 1, 0})
        self.assertEqual(highlight.edges, {(-1, 1), (1, 0)})
        self.assertEqual(
            highlight.direct_edges,
            {(-1, 1), (1, 0), (1, 2)},
        )
        self.assertEqual(highlight.upstream_edges, {(-1, 1)})
        self.assertEqual(highlight.downstream_edges, {(1, 0)})

    def test_highlighted_path_handles_input_output_recurrence_and_self_loop(self) -> None:
        genome = genome_with_connections(
            [1, 2],
            [(-1, 1), (1, 2), (2, 1), (1, 1), (2, 0)],
        )
        layout = layout_for(genome)

        input_highlight = highlighted_path_through_node(layout, -1)
        output_highlight = highlighted_path_through_node(layout, 0)
        hidden_highlight = highlighted_path_through_node(layout, 1)

        expected_edges = {(-1, 1), (1, 2), (2, 1), (1, 1), (2, 0)}
        self.assertEqual(input_highlight.edges, expected_edges)
        self.assertEqual(output_highlight.edges, expected_edges)
        self.assertEqual(
            hidden_highlight.direct_edges,
            {(-1, 1), (1, 2), (2, 1), (1, 1)},
        )
        self.assertEqual(
            hidden_highlight.upstream_edges,
            {(-1, 1), (1, 2), (2, 1), (1, 1)},
        )
        self.assertEqual(
            hidden_highlight.downstream_edges,
            {(1, 2), (2, 1), (1, 1), (2, 0)},
        )

    def test_highlighted_path_for_isolated_or_unknown_node(self) -> None:
        layout = layout_for(genome_with_connections([1], [(-1, 0)]))

        self.assertEqual(
            highlighted_path_through_node(layout, 1).nodes,
            {1},
        )
        self.assertEqual(
            highlighted_path_through_node(layout, 999).nodes,
            set(),
        )
        unknown = highlighted_path_through_node(layout, 999)
        self.assertEqual(unknown.direct_edges, set())
        self.assertEqual(unknown.upstream_edges, set())
        self.assertEqual(unknown.downstream_edges, set())


if __name__ == "__main__":
    unittest.main()
