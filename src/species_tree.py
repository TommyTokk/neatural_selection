from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol


class SpeciesRecordLike(Protocol):
    species_id: int
    parent_species_id: int | None
    emerged_at: float | None


@dataclass(frozen=True, slots=True)
class SpeciesTreeLayout:
    positions: dict[int, tuple[float, float]]
    edges: tuple[tuple[int, int], ...]
    depths: dict[int, int]
    effective_times: dict[int, float]
    roots: tuple[int, ...]
    content_width: float
    content_height: float
    leaf_count: int
    timeline_start: float
    timeline_end: float


SpeciesTreeRoute = tuple[tuple[float, float], ...]


def build_species_tree_layout(
    records: Mapping[int, SpeciesRecordLike],
    *,
    horizontal_gap: float = 92.0,
    time_scale: float = 2.0,
    minimum_generation_gap: float = 64.0,
    padding: float = 48.0,
    timeline_end: float | None = None,
) -> SpeciesTreeLayout:
    """Lay out a possibly imperfect species ancestry graph as a forest.

    Invalid, self-referencing, missing, and cyclic parent relationships are
    detached into roots so every supplied species remains visible.
    Coordinates are measured from the content's top-left corner.
    """
    species_ids = tuple(sorted(int(species_id) for species_id in records))
    if not species_ids:
        end = _valid_time(timeline_end) or 0.0
        return SpeciesTreeLayout(
            {}, (), {}, {}, (), 0.0, 0.0, 0, 0.0, end
        )

    known_ids = set(species_ids)
    parents: dict[int, int | None] = {}
    for species_id in species_ids:
        candidate = records[species_id].parent_species_id
        parents[species_id] = (
            int(candidate)
            if candidate is not None
            and int(candidate) in known_ids
            and int(candidate) != species_id
            else None
        )

    _break_parent_cycles(parents)

    children: dict[int, list[int]] = {species_id: [] for species_id in species_ids}
    for species_id in species_ids:
        parent_id = parents[species_id]
        if parent_id is not None:
            children[parent_id].append(species_id)
    for child_ids in children.values():
        child_ids.sort()

    roots = tuple(
        species_id for species_id in species_ids if parents[species_id] is None
    )
    depths: dict[int, int] = {}
    effective_times: dict[int, float] = {}
    positions: dict[int, tuple[float, float]] = {}
    next_leaf_column = 0
    safe_time_scale = max(0.0001, float(time_scale))
    fallback_time_gap = max(0.0, minimum_generation_gap) / safe_time_scale

    def place(species_id: int, depth: int, parent_time: float | None) -> float:
        nonlocal next_leaf_column
        depths[species_id] = depth
        recorded_time = _valid_time(getattr(records[species_id], "emerged_at", None))
        if recorded_time is None:
            effective_time = (
                0.0 if parent_time is None else parent_time + fallback_time_gap
            )
        else:
            effective_time = max(recorded_time, parent_time or 0.0)
        effective_times[species_id] = effective_time

        child_ids = children[species_id]
        if child_ids:
            child_columns = [
                place(child_id, depth + 1, effective_time)
                for child_id in child_ids
            ]
            column = (child_columns[0] + child_columns[-1]) * 0.5
        else:
            column = float(next_leaf_column)
            next_leaf_column += 1
        positions[species_id] = (
            padding + column * horizontal_gap,
            padding + effective_time * safe_time_scale,
        )
        return column

    for root_id in roots:
        place(root_id, 0, None)

    leaf_count = max(1, next_leaf_column)
    recorded_times = [
        time
        for record in records.values()
        if (time := _valid_time(getattr(record, "emerged_at", None))) is not None
    ]
    requested_end = _valid_time(timeline_end)
    timeline_candidates = [0.0, *recorded_times]
    if requested_end is not None:
        timeline_candidates.append(requested_end)
    real_timeline_end = max(timeline_candidates)
    display_end = max(real_timeline_end, *effective_times.values())
    edges = tuple(
        (parent_id, species_id)
        for species_id in species_ids
        if (parent_id := parents[species_id]) is not None
    )
    return SpeciesTreeLayout(
        positions=positions,
        edges=edges,
        depths=depths,
        effective_times=effective_times,
        roots=roots,
        content_width=padding * 2.0 + (leaf_count - 1) * horizontal_gap,
        content_height=padding * 2.0 + display_end * safe_time_scale,
        leaf_count=leaf_count,
        timeline_start=0.0,
        timeline_end=real_timeline_end,
    )


def route_species_tree_edges(
    layout: SpeciesTreeLayout,
    node_radii: Mapping[int, float],
    *,
    clearance: float = 6.0,
) -> dict[tuple[int, int], SpeciesTreeRoute]:
    """Build deterministic orthogonal routes that avoid unrelated nodes."""
    safe_clearance = max(0.0, float(clearance))
    obstacles = {
        species_id: _Obstacle.around(
            position,
            max(0.0, float(node_radii.get(species_id, 0.0)))
            + safe_clearance,
        )
        for species_id, position in layout.positions.items()
    }
    if not obstacles:
        return {}

    outer_left = min(obstacle.left for obstacle in obstacles.values()) - 16.0
    outer_right = max(obstacle.right for obstacle in obstacles.values()) + 16.0
    outer_top = min(obstacle.top for obstacle in obstacles.values()) - 16.0
    outer_bottom = max(obstacle.bottom for obstacle in obstacles.values()) + 16.0
    routes: dict[tuple[int, int], SpeciesTreeRoute] = {}

    for edge in layout.edges:
        parent_id, child_id = edge
        start = layout.positions[parent_id]
        end = layout.positions[child_id]
        start_radius = max(0.0, float(node_radii.get(parent_id, 0.0)))
        end_radius = max(0.0, float(node_radii.get(child_id, 0.0)))
        blocked = tuple(
            obstacle
            for species_id, obstacle in obstacles.items()
            if species_id not in edge
        )
        candidates = _route_candidates(
            start,
            end,
            start_radius,
            end_radius,
            safe_clearance,
            obstacles,
            outer_left,
            outer_right,
            outer_top,
            outer_bottom,
        )
        valid = [
            candidate
            for candidate in candidates
            if not _route_hits_obstacle(candidate, blocked)
        ]
        routes[edge] = min(
            valid or candidates,
            key=lambda route: (
                _route_length(route) + max(0, len(route) - 2) * 4.0,
                route,
            ),
        )
    return routes


@dataclass(frozen=True, slots=True)
class _Obstacle:
    left: float
    right: float
    top: float
    bottom: float

    @classmethod
    def around(
        cls,
        position: tuple[float, float],
        radius: float,
    ) -> _Obstacle:
        return cls(
            position[0] - radius,
            position[0] + radius,
            position[1] - radius,
            position[1] + radius,
        )


def _route_candidates(
    start: tuple[float, float],
    end: tuple[float, float],
    start_radius: float,
    end_radius: float,
    clearance: float,
    obstacles: Mapping[int, _Obstacle],
    outer_left: float,
    outer_right: float,
    outer_top: float,
    outer_bottom: float,
) -> list[SpeciesTreeRoute]:
    start_x, start_y = start
    end_x, end_y = end
    candidates: list[SpeciesTreeRoute] = []

    if end_y >= start_y:
        start_anchor = (start_x, start_y + start_radius)
        start_clear = (start_x, start_y + start_radius + clearance)
        end_clear = (end_x, end_y - end_radius - clearance)
        end_anchor = (end_x, end_y - end_radius)
    else:
        start_anchor = (start_x, start_y - start_radius)
        start_clear = (start_x, start_y - start_radius - clearance)
        end_clear = (end_x, end_y + end_radius + clearance)
        end_anchor = (end_x, end_y + end_radius)

    midpoint_y = (start_clear[1] + end_clear[1]) * 0.5
    channel_ys = {midpoint_y, start_clear[1], end_clear[1]}
    for obstacle in obstacles.values():
        channel_ys.update((obstacle.top, obstacle.bottom))
    for channel_y in sorted(
        channel_ys,
        key=lambda value: (abs(value - midpoint_y), value),
    ):
        candidates.append(
            _compact_route(
                (
                    start_anchor,
                    start_clear,
                    (start_x, channel_y),
                    (end_x, channel_y),
                    end_clear,
                    end_anchor,
                )
            )
        )

    for side_x, direction in ((outer_left, -1.0), (outer_right, 1.0)):
        parent_anchor = (start_x + direction * start_radius, start_y)
        parent_clear = (
            start_x + direction * (start_radius + clearance),
            start_y,
        )
        child_clear = (
            end_x + direction * (end_radius + clearance),
            end_y,
        )
        child_anchor = (end_x + direction * end_radius, end_y)
        candidates.append(
            _compact_route(
                (
                    parent_anchor,
                    parent_clear,
                    (side_x, start_y),
                    (side_x, end_y),
                    child_clear,
                    child_anchor,
                )
            )
        )

    for side_y, direction in ((outer_top, -1.0), (outer_bottom, 1.0)):
        parent_anchor = (start_x, start_y + direction * start_radius)
        parent_clear = (
            start_x,
            start_y + direction * (start_radius + clearance),
        )
        child_clear = (
            end_x,
            end_y + direction * (end_radius + clearance),
        )
        child_anchor = (end_x, end_y + direction * end_radius)
        candidates.append(
            _compact_route(
                (
                    parent_anchor,
                    parent_clear,
                    (start_x, side_y),
                    (end_x, side_y),
                    child_clear,
                    child_anchor,
                )
            )
        )
    return candidates


def _compact_route(points: tuple[tuple[float, float], ...]) -> SpeciesTreeRoute:
    compact: list[tuple[float, float]] = []
    for point in points:
        if compact and point == compact[-1]:
            continue
        if (
            len(compact) >= 2
            and (
                compact[-2][0] == compact[-1][0] == point[0]
                or compact[-2][1] == compact[-1][1] == point[1]
            )
        ):
            compact[-1] = point
        else:
            compact.append(point)
    return tuple(compact)


def _route_hits_obstacle(
    route: SpeciesTreeRoute,
    obstacles: tuple[_Obstacle, ...],
) -> bool:
    return any(
        _segment_hits_obstacle(start, end, obstacle)
        for start, end in zip(route, route[1:])
        for obstacle in obstacles
    )


def _segment_hits_obstacle(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacle: _Obstacle,
) -> bool:
    if start[0] == end[0]:
        x = start[0]
        low, high = sorted((start[1], end[1]))
        return (
            obstacle.left < x < obstacle.right
            and low < obstacle.bottom
            and high > obstacle.top
        )
    if start[1] == end[1]:
        y = start[1]
        low, high = sorted((start[0], end[0]))
        return (
            obstacle.top < y < obstacle.bottom
            and low < obstacle.right
            and high > obstacle.left
        )
    return True


def _route_length(route: SpeciesTreeRoute) -> float:
    return sum(
        abs(end[0] - start[0]) + abs(end[1] - start[1])
        for start, end in zip(route, route[1:])
    )


def _valid_time(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _break_parent_cycles(parents: dict[int, int | None]) -> None:
    visited: set[int] = set()
    for start_id in sorted(parents):
        if start_id in visited:
            continue

        path: list[int] = []
        path_indexes: dict[int, int] = {}
        current: int | None = start_id
        while current is not None and current not in visited:
            if current in path_indexes:
                cycle = path[path_indexes[current] :]
                parents[min(cycle)] = None
                break
            path_indexes[current] = len(path)
            path.append(current)
            current = parents[current]
        visited.update(path)
