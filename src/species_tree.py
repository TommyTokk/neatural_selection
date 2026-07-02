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
