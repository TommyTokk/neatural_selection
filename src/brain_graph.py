from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class BrainNodeKind(Enum):
    INPUT = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"


class BrainEdgeKind(Enum):
    FORWARD = "forward"
    RECURRENT = "recurrent"
    SELF_LOOP = "self_loop"


class LayoutBounds(Protocol):
    left: float
    right: float
    bottom: float
    top: float
    width: float


@dataclass(frozen=True, slots=True)
class BrainGraphNode:
    key: int
    kind: BrainNodeKind
    label: str
    depth: int


@dataclass(frozen=True, slots=True)
class BrainGraphEdge:
    source: int
    target: int
    weight: float
    enabled: bool
    kind: BrainEdgeKind


@dataclass(frozen=True, slots=True)
class BrainGraphLayout:
    nodes: dict[int, BrainGraphNode]
    edges: list[BrainGraphEdge]
    positions: dict[int, tuple[float, float]]
    max_depth: int


@dataclass(frozen=True, slots=True)
class BrainGraphHighlight:
    nodes: frozenset[int]
    edges: frozenset[tuple[int, int]]


def build_brain_graph_layout(
    genome: Any,
    input_keys: list[int],
    output_keys: list[int],
    bounds: LayoutBounds,
    input_labels: list[str],
    output_labels: list[str],
) -> BrainGraphLayout:
    hidden_keys = sorted(key for key in genome.nodes if key not in output_keys)
    depths = _compute_depths(genome, input_keys, output_keys, hidden_keys)
    hidden_depths = [depths.get(key, 1) for key in hidden_keys]
    output_depth = max([1, *hidden_depths]) + 1

    nodes: dict[int, BrainGraphNode] = {}
    for index, key in enumerate(input_keys):
        nodes[key] = BrainGraphNode(
            key=key,
            kind=BrainNodeKind.INPUT,
            label=input_labels[index] if index < len(input_labels) else str(key),
            depth=0,
        )

    for key in hidden_keys:
        nodes[key] = BrainGraphNode(
            key=key,
            kind=BrainNodeKind.HIDDEN,
            label=str(key),
            depth=depths.get(key, 1),
        )

    for index, key in enumerate(output_keys):
        nodes[key] = BrainGraphNode(
            key=key,
            kind=BrainNodeKind.OUTPUT,
            label=output_labels[index] if index < len(output_labels) else str(key),
            depth=output_depth,
        )

    edges = [
        _build_edge(connection, nodes)
        for connection in genome.connections.values()
        if connection.key[0] in nodes and connection.key[1] in nodes
    ]
    max_depth = max((node.depth for node in nodes.values()), default=0)
    positions = _layout_positions(nodes, bounds, max_depth)
    return BrainGraphLayout(
        nodes=nodes,
        edges=edges,
        positions=positions,
        max_depth=max_depth,
    )


def highlighted_path_through_node(
    layout: BrainGraphLayout,
    node_key: int,
) -> BrainGraphHighlight:
    """Return the complete enabled signal route through ``node_key``."""
    if node_key not in layout.nodes:
        return BrainGraphHighlight(frozenset(), frozenset())

    incoming: dict[int, list[BrainGraphEdge]] = {}
    outgoing: dict[int, list[BrainGraphEdge]] = {}
    for edge in layout.edges:
        if not edge.enabled:
            continue
        incoming.setdefault(edge.target, []).append(edge)
        outgoing.setdefault(edge.source, []).append(edge)

    highlighted_nodes = {node_key}
    highlighted_edges: set[tuple[int, int]] = set()

    def traverse(
        start: int,
        adjacency: dict[int, list[BrainGraphEdge]],
        *,
        upstream: bool,
    ) -> None:
        pending = [start]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for edge in adjacency.get(current, ()):
                highlighted_edges.add((edge.source, edge.target))
                highlighted_nodes.update((edge.source, edge.target))
                neighbor = edge.source if upstream else edge.target
                if neighbor not in visited:
                    pending.append(neighbor)

    traverse(node_key, incoming, upstream=True)
    traverse(node_key, outgoing, upstream=False)
    return BrainGraphHighlight(
        nodes=frozenset(highlighted_nodes),
        edges=frozenset(highlighted_edges),
    )


def _compute_depths(
    genome: Any,
    input_keys: list[int],
    output_keys: list[int],
    hidden_keys: list[int],
) -> dict[int, int]:
    input_set = set(input_keys)
    output_set = set(output_keys)
    hidden_set = set(hidden_keys)
    forward_edges = [
        connection.key
        for connection in genome.connections.values()
        if connection.enabled
        and connection.key[0] != connection.key[1]
        and connection.key[1] not in input_set
    ]
    recurrent_edges = _cycle_edges(forward_edges)
    usable_edges = [
        (source, target)
        for source, target in forward_edges
        if (source, target) not in recurrent_edges and target not in output_set
    ]

    depths = {key: 0 for key in input_keys}
    for key in hidden_keys:
        depths.setdefault(key, 1)

    for _ in range(max(1, len(hidden_keys) + len(output_keys) + 1)):
        changed = False
        for source, target in usable_edges:
            if target not in hidden_set:
                continue
            source_depth = depths.get(source)
            if source_depth is None:
                continue
            next_depth = max(1, source_depth + 1)
            if next_depth > depths.get(target, 1):
                depths[target] = next_depth
                changed = True
        if not changed:
            break

    return {key: max(1, depths.get(key, 1)) for key in hidden_keys}


def _build_edge(
    connection: Any,
    nodes: dict[int, BrainGraphNode],
) -> BrainGraphEdge:
    source, target = connection.key
    if source == target:
        kind = BrainEdgeKind.SELF_LOOP
    elif nodes[source].depth >= nodes[target].depth:
        kind = BrainEdgeKind.RECURRENT
    else:
        kind = BrainEdgeKind.FORWARD

    return BrainGraphEdge(
        source=source,
        target=target,
        weight=float(connection.weight),
        enabled=bool(connection.enabled),
        kind=kind,
    )


def _layout_positions(
    nodes: dict[int, BrainGraphNode],
    bounds: LayoutBounds,
    max_depth: int,
) -> dict[int, tuple[float, float]]:
    grouped: dict[int, list[BrainGraphNode]] = {}
    for node in nodes.values():
        grouped.setdefault(node.depth, []).append(node)

    positions: dict[int, tuple[float, float]] = {}
    left = bounds.left + 32.0
    right = bounds.right - 32.0
    horizontal_step = (right - left) / max(1, max_depth)

    for depth, depth_nodes in grouped.items():
        # Nodes are inserted in their configured input/output order when the
        # graph is built. Preserve that sequence here: numerically sorting
        # negative NEAT input keys would reverse -1 .. -N on screen.
        ordered_nodes = depth_nodes
        x = left + horizontal_step * depth
        vertical_padding = 28.0
        bottom = bounds.bottom + vertical_padding
        top = bounds.top - vertical_padding
        if len(ordered_nodes) == 1:
            positions[ordered_nodes[0].key] = (x, (bottom + top) * 0.5)
            continue

        step = (top - bottom) / max(1, len(ordered_nodes) - 1)
        for index, node in enumerate(ordered_nodes):
            positions[node.key] = (x, top - index * step)

    return positions


def _cycle_edges(edges: list[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacency: dict[int, list[int]] = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)

    return {
        (source, target)
        for source, target in edges
        if _has_path(adjacency, target, source)
    }


def _has_path(adjacency: dict[int, list[int]], start: int, goal: int) -> bool:
    pending = [start]
    visited: set[int] = set()
    while pending:
        node = pending.pop()
        if node == goal:
            return True
        if node in visited:
            continue
        visited.add(node)
        pending.extend(adjacency.get(node, []))
    return False
