from __future__ import annotations

from dataclasses import dataclass
import unittest

from src.brain_graph import BrainEdgeKind, BrainNodeKind, build_brain_graph_layout


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


if __name__ == "__main__":
    unittest.main()
