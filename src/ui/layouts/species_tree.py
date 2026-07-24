from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, log
from typing import Mapping, Protocol


class SpeciesRecordLike(Protocol):
    """Provide SpeciesRecordLike UI behavior."""
    species_id: int
    parent_species_id: int | None
    emerged_at: float | None


@dataclass(frozen=True, slots=True)
class SpeciesTreeLayout:
    """Provide SpeciesTreeLayout UI behavior."""
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
    lanes: dict[int, int] = field(default_factory=dict)
    end_times: dict[int, float] = field(default_factory=dict)
    descendant_counts: dict[int, int] = field(default_factory=dict)
    content_left: float = 0.0


SpeciesTreeRoute = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TreeViewportSlice:
    """Provide TreeViewportSlice UI behavior."""
    node_ids: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    routes: dict[tuple[int, int], SpeciesTreeRoute]
    lifeline_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TimeBucketSummary:
    """Provide TimeBucketSummary UI behavior."""
    bucket_id: int
    node_count: int
    start_time: float
    end_time: float


class TreeLayoutManager:
    """Incremental Romerogram layout with compact lanes and time buckets."""

    def __init__(
        self,
        *,
        horizontal_gap: float = 92.0,
        time_scale: float = 2.0,
        minimum_generation_gap: float = 64.0,
        padding: float = 48.0,
        bucket_seconds: float = 1800.0,
    ) -> None:
        """Initialize the component.

        Parameters
        ----------
        horizontal_gap
            Value used by the operation.
        time_scale
            Value used by the operation.
        minimum_generation_gap
            Value used by the operation.
        padding
            Value used by the operation.
        bucket_seconds
            Value used by the operation.
        """
        self.horizontal_gap = max(1.0, float(horizontal_gap))
        self.time_scale = max(0.0001, float(time_scale))
        self.minimum_generation_gap = max(
            0.0,
            float(minimum_generation_gap),
        )
        self.padding = max(0.0, float(padding))
        self.bucket_seconds = max(1.0, float(bucket_seconds))
        self.placement_count = 0
        self._records_identity: int | None = None
        self._record_count = 0
        self._max_species_id: int | None = None
        self._positions: dict[int, tuple[float, float]] = {}
        self._depths: dict[int, int] = {}
        self._effective_times: dict[int, float] = {}
        self._parents: dict[int, int | None] = {}
        self._child_counts: dict[int, int] = {}
        self._lanes: dict[int, int] = {}
        self._lane_end_times: dict[int, float] = {}
        self._end_times: dict[int, float] = {}
        self._descendant_counts: dict[int, int] = {}
        self._roots: list[int] = []
        self._edges: list[tuple[int, int]] = []
        self._routes: dict[tuple[int, int], SpeciesTreeRoute] = {}
        self._node_buckets: dict[int, list[int]] = {}
        self._edge_buckets: dict[int, list[tuple[int, int]]] = {}
        self._lifeline_buckets: dict[int, list[int]] = {}
        self._lifeline_bucket_ends: dict[int, int] = {}
        self._timeline_end = 0.0
        self._display_end = 0.0
        self._latest_species_id: int | None = None
        self._edges_tuple: tuple[tuple[int, int], ...] = ()
        self._roots_tuple: tuple[int, ...] = ()
        self._bucket_ids: tuple[int, ...] = ()
        self._buckets_dirty = False

    @property
    def latest_species_id(self) -> int | None:
        """Return latest species id.

        Returns
        -------
        int | None
            Computed result.
        """
        return self._latest_species_id

    @property
    def parents(self) -> Mapping[int, int | None]:
        """Return parents.

        Returns
        -------
        Mapping[int, int | None]
            Computed result.
        """
        return self._parents

    @property
    def routes(self) -> Mapping[tuple[int, int], SpeciesTreeRoute]:
        """Return routes.

        Returns
        -------
        Mapping[tuple[int, int], SpeciesTreeRoute]
            Computed result.
        """
        return self._routes

    def sync(
        self,
        records: Mapping[int, SpeciesRecordLike],
        *,
        timeline_end: float | None = None,
        species_end_times: Mapping[int, float] | None = None,
    ) -> SpeciesTreeLayout:
        """Return sync.

        Parameters
        ----------
        records
            Species history data to inspect.
        timeline_end
            Value used by the operation.
        species_end_times
            Value used by the operation.

        Returns
        -------
        SpeciesTreeLayout
            Computed result.
        """
        requested_ends = {
            int(species_id): _normalize_end_time(end_time)
            for species_id, end_time in (species_end_times or {}).items()
        }
        records_identity = id(records)
        record_count = len(records)
        revived_species = any(
            isfinite(previous_end)
            and not isfinite(requested_ends.get(species_id, float("inf")))
            for species_id, previous_end in self._end_times.items()
            if species_id in records
        )
        must_rebuild = (
            self._records_identity is not None
            and (
                records_identity != self._records_identity
                or record_count < self._record_count
                or revived_species
            )
        )
        if must_rebuild:
            self._reset()

        if self._records_identity is None:
            self._records_identity = records_identity
            new_ids = sorted(int(species_id) for species_id in records)
        elif record_count == self._record_count:
            new_ids = []
        else:
            added_count = record_count - self._record_count
            sequential_ids = (
                []
                if self._max_species_id is None
                else list(
                    range(
                        self._max_species_id + 1,
                        self._max_species_id + added_count + 1,
                    )
                )
            )
            if sequential_ids and all(
                species_id in records for species_id in sequential_ids
            ):
                new_ids = sequential_ids
            else:
                new_ids = sorted(
                    int(species_id)
                    for species_id in records
                    if int(species_id) not in self._positions
                )

        for species_id in self._positions:
            requested_end = requested_ends.get(species_id, float("inf"))
            previous_end = self._end_times[species_id]
            if previous_end == requested_end:
                continue
            if isfinite(previous_end) and requested_end != previous_end:
                continue
            normalized_end = max(
                self._effective_times[species_id],
                requested_end,
            )
            self._end_times[species_id] = normalized_end
            lane = self._lanes[species_id]
            if self._lane_end_times.get(lane) == previous_end:
                self._lane_end_times[lane] = normalized_end

        for species_id in new_ids:
            self._place(
                species_id,
                records[species_id],
                requested_ends.get(species_id, float("inf")),
            )

        self._record_count = record_count
        if self._positions:
            self._max_species_id = max(
                self._max_species_id or min(self._positions),
                max(self._positions),
            )
        requested_end = _valid_time(timeline_end)
        latest_time = max(self._effective_times.values(), default=0.0)
        self._display_end = max(self._display_end, latest_time)
        self._timeline_end = max(
            self._timeline_end,
            0.0 if requested_end is None else requested_end,
        )
        if new_ids or must_rebuild:
            self._recompute_descendant_counts()
        self._index_lifelines_through(self._timeline_end)
        return self._layout()

    def viewport_slice(
        self,
        *,
        left: float,
        right: float,
        top: float,
        bottom: float,
        node_padding: float = 24.0,
    ) -> TreeViewportSlice:
        """Return viewport slice.

        Parameters
        ----------
        left
            Logical screen coordinate.
        right
            Logical screen coordinate.
        top
            Logical screen coordinate.
        bottom
            Logical screen coordinate.
        node_padding
            Value used by the operation.

        Returns
        -------
        TreeViewportSlice
            Computed result.
        """
        if not self._positions:
            return TreeViewportSlice((), (), {})
        low_x, high_x = sorted((float(left), float(right)))
        low_y, high_y = sorted((float(top), float(bottom)))
        padding = max(0.0, float(node_padding))
        start_time = max(0.0, (low_y - self.padding - padding) / self.time_scale)
        end_time = max(0.0, (high_y - self.padding + padding) / self.time_scale)
        start_bucket = self.bucket_for_time(start_time)
        end_bucket = self.bucket_for_time(end_time)
        bucket_ids = tuple(
            bucket_id
            for bucket_id in self._sorted_bucket_ids()
            if start_bucket <= bucket_id <= end_bucket
        )

        node_ids: list[int] = []
        for bucket_id in bucket_ids:
            for species_id in self._node_buckets.get(bucket_id, ()):
                x, y = self._positions[species_id]
                if (
                    low_x - padding <= x <= high_x + padding
                    and low_y - padding <= y <= high_y + padding
                ):
                    node_ids.append(species_id)

        candidate_edges = {
            edge
            for bucket_id in bucket_ids
            for edge in self._edge_buckets.get(bucket_id, ())
        }
        edges = tuple(
            edge
            for edge in sorted(candidate_edges)
            if _route_intersects_bounds(
                self._routes[edge],
                low_x - padding,
                high_x + padding,
                low_y - padding,
                high_y + padding,
            )
        )
        candidate_lifelines = {
            species_id
            for bucket_id in bucket_ids
            for species_id in self._lifeline_buckets.get(bucket_id, ())
        }
        lifeline_ids = tuple(
            species_id
            for species_id in sorted(candidate_lifelines)
            if (
                low_x - padding
                <= self._positions[species_id][0]
                <= high_x + padding
                and self._positions[species_id][1] <= high_y + padding
                and (
                    self.padding
                    + min(self._end_times[species_id], self._timeline_end)
                    * self.time_scale
                    >= low_y - padding
                )
            )
        )
        return TreeViewportSlice(
            node_ids=tuple(node_ids),
            edges=edges,
            routes={edge: self._routes[edge] for edge in edges},
            lifeline_ids=lifeline_ids,
        )

    def bucket_for_time(self, time_value: float) -> int:
        """Return bucket for time.

        Parameters
        ----------
        time_value
            Value used by the operation.

        Returns
        -------
        int
            Computed result.
        """
        return int(max(0.0, float(time_value)) // self.bucket_seconds)

    def bucket_summaries(self) -> tuple[TimeBucketSummary, ...]:
        """Return bucket summaries.

        Returns
        -------
        tuple[TimeBucketSummary, ...]
            Computed collection.
        """
        return tuple(
            TimeBucketSummary(
                bucket_id=bucket_id,
                node_count=len(self._node_buckets[bucket_id]),
                start_time=bucket_id * self.bucket_seconds,
                end_time=(bucket_id + 1) * self.bucket_seconds,
            )
            for bucket_id in sorted(self._node_buckets)
        )

    def _place(
        self,
        species_id: int,
        record: SpeciesRecordLike,
        end_time: float,
    ) -> None:
        """Return place.

        Parameters
        ----------
        species_id
            Value used by the operation.
        record
            Species history data to inspect.
        end_time
            Value used by the operation.
        """
        candidate = getattr(record, "parent_species_id", None)
        parent_id = (
            int(candidate)
            if candidate is not None
            and int(candidate) in self._positions
            and int(candidate) != species_id
            else None
        )
        recorded_time = _valid_time(getattr(record, "emerged_at", None))
        if recorded_time is not None:
            self._timeline_end = max(self._timeline_end, recorded_time)
        if parent_id is None:
            depth = 0
            effective_time = 0.0 if recorded_time is None else recorded_time
            self._roots.append(species_id)
            self._roots_tuple = tuple(self._roots)
            child_ordinal = 0
        else:
            parent_time = self._effective_times[parent_id]
            fallback_gap = self.minimum_generation_gap / self.time_scale
            effective_time = (
                parent_time + fallback_gap
                if recorded_time is None
                else max(parent_time, recorded_time)
            )
            child_count = self._child_counts.get(parent_id, 0)
            self._child_counts[parent_id] = child_count + 1
            child_ordinal = child_count
            depth = self._depths[parent_id] + 1

        normalized_end = max(effective_time, end_time)
        lane = self._allocate_lane(
            effective_time,
            normalized_end,
            parent_lane=(
                None if parent_id is None else self._lanes[parent_id]
            ),
            child_ordinal=child_ordinal,
        )

        position = (
            lane * self.horizontal_gap,
            self.padding + effective_time * self.time_scale,
        )
        self._positions[species_id] = position
        self._depths[species_id] = depth
        self._effective_times[species_id] = effective_time
        self._parents[species_id] = parent_id
        self._child_counts.setdefault(species_id, 0)
        self._lanes[species_id] = lane
        self._end_times[species_id] = normalized_end
        bucket_id = self.bucket_for_time(effective_time)
        self._node_buckets.setdefault(bucket_id, []).append(species_id)
        self._buckets_dirty = True
        self.placement_count += 1

        if parent_id is not None:
            edge = (parent_id, species_id)
            self._edges.append(edge)
            self._edges_tuple = tuple(self._edges)
            route = _incremental_edge_route(
                (
                    self._positions[parent_id][0],
                    position[1],
                ),
                position,
            )
            self._routes[edge] = route
            self._edge_buckets.setdefault(bucket_id, []).append(edge)
            self._buckets_dirty = True

        latest = self._latest_species_id
        if latest is None or (effective_time, species_id) >= (
            self._effective_times[latest],
            latest,
        ):
            self._latest_species_id = species_id

    def _allocate_lane(
        self,
        emergence_time: float,
        end_time: float,
        *,
        parent_lane: int | None,
        child_ordinal: int,
    ) -> int:
        """Return allocate lane.

        Parameters
        ----------
        emergence_time
            Value used by the operation.
        end_time
            Value used by the operation.
        parent_lane
            Value used by the operation.
        child_ordinal
            Value used by the operation.

        Returns
        -------
        int
            Computed result.
        """
        def available(candidate: int) -> bool:
            """Return available.

            Parameters
            ----------
            candidate
                Value used by the operation.

            Returns
            -------
            bool
                Whether the operation succeeded or consumed the input.
            """
            return self._lane_end_times.get(candidate, float("-inf")) < emergence_time

        if parent_lane is None:
            candidate_index = 0
            while True:
                if candidate_index == 0:
                    lane = 0
                else:
                    distance = (candidate_index + 1) // 2
                    lane = -distance if candidate_index % 2 else distance
                if available(lane):
                    break
                candidate_index += 1
        else:
            if available(parent_lane):
                self._lane_end_times[parent_lane] = end_time
                return parent_lane
            prefer_left = child_ordinal % 2 == 0
            span = (
                max(self._lane_end_times) - min(self._lane_end_times) + 2
                if self._lane_end_times
                else 2
            )
            lane = parent_lane - 1 if prefer_left else parent_lane + 1
            for distance in range(1, span + 2):
                candidates = (
                    (parent_lane - distance, parent_lane + distance)
                    if prefer_left
                    else (parent_lane + distance, parent_lane - distance)
                )
                available_lane = next(
                    (candidate for candidate in candidates if available(candidate)),
                    None,
                )
                if available_lane is not None:
                    lane = available_lane
                    break
        self._lane_end_times[lane] = end_time
        return lane

    def _recompute_descendant_counts(self) -> None:
        """Return recompute descendant counts.
        """
        counts = {species_id: 0 for species_id in self._positions}
        for species_id in reversed(tuple(self._positions)):
            parent_id = self._parents.get(species_id)
            if parent_id is not None:
                counts[parent_id] += counts[species_id] + 1
        self._descendant_counts = counts

    def _index_lifelines_through(self, timeline_end: float) -> None:
        """Return index lifelines through.

        Parameters
        ----------
        timeline_end
            Value used by the operation.
        """
        for species_id, emergence_time in self._effective_times.items():
            end_time = min(self._end_times[species_id], timeline_end)
            end_bucket = self.bucket_for_time(max(emergence_time, end_time))
            indexed_end = self._lifeline_bucket_ends.get(species_id, -1)
            start_bucket = max(
                self.bucket_for_time(emergence_time),
                indexed_end + 1,
            )
            for bucket_id in range(start_bucket, end_bucket + 1):
                self._lifeline_buckets.setdefault(bucket_id, []).append(
                    species_id
                )
                self._buckets_dirty = True
            self._lifeline_bucket_ends[species_id] = max(
                indexed_end,
                end_bucket,
            )

    def _layout(self) -> SpeciesTreeLayout:
        """Return layout.

        Returns
        -------
        SpeciesTreeLayout
            Computed result.
        """
        latest_time = max(self._timeline_end, self._display_end, 0.0)
        minimum_lane = min(self._lane_end_times, default=0)
        maximum_lane = max(self._lane_end_times, default=-1)
        content_left = (
            0.0
            if not self._positions
            else minimum_lane * self.horizontal_gap - self.padding
        )
        return SpeciesTreeLayout(
            positions=self._positions,
            edges=self._edges_tuple,
            depths=self._depths,
            effective_times=self._effective_times,
            roots=self._roots_tuple,
            content_width=(
                0.0
                if not self._positions
                else self.padding * 2.0
                + max(0, maximum_lane - minimum_lane) * self.horizontal_gap
            ),
            content_height=(
                0.0
                if not self._positions
                else self.padding * 2.0 + latest_time * self.time_scale
            ),
            leaf_count=(
                0
                if not self._positions
                else maximum_lane - minimum_lane + 1
            ),
            timeline_start=0.0,
            timeline_end=self._timeline_end,
            lanes=self._lanes,
            end_times=self._end_times,
            descendant_counts=self._descendant_counts,
            content_left=content_left,
        )

    def _sorted_bucket_ids(self) -> tuple[int, ...]:
        """Return sorted bucket ids.

        Returns
        -------
        tuple[int, ...]
            Computed collection.
        """
        if self._buckets_dirty:
            self._bucket_ids = tuple(
                sorted(
                    set(self._node_buckets)
                    | set(self._edge_buckets)
                    | set(self._lifeline_buckets)
                )
            )
            self._buckets_dirty = False
        return self._bucket_ids

    def _reset(self) -> None:
        """Return reset.
        """
        configuration = (
            self.horizontal_gap,
            self.time_scale,
            self.minimum_generation_gap,
            self.padding,
            self.bucket_seconds,
        )
        placement_count = self.placement_count
        self.__init__(
            horizontal_gap=configuration[0],
            time_scale=configuration[1],
            minimum_generation_gap=configuration[2],
            padding=configuration[3],
            bucket_seconds=configuration[4],
        )
        self.placement_count = placement_count


def build_species_tree_layout(
    records: Mapping[int, SpeciesRecordLike],
    *,
    horizontal_gap: float = 92.0,
    time_scale: float = 2.0,
    minimum_generation_gap: float = 64.0,
    padding: float = 48.0,
    timeline_end: float | None = None,
    species_end_times: Mapping[int, float] | None = None,
) -> SpeciesTreeLayout:
    """Build species tree layout.

    Parameters
    ----------
    records
        Species history data to inspect.
    horizontal_gap
        Value used by the operation.
    time_scale
        Value used by the operation.
    minimum_generation_gap
        Value used by the operation.
    padding
        Value used by the operation.
    timeline_end
        Value used by the operation.
    species_end_times
        Value used by the operation.

    Returns
    -------
    SpeciesTreeLayout
        Computed result.
    """
    manager = TreeLayoutManager(
        horizontal_gap=horizontal_gap,
        time_scale=time_scale,
        minimum_generation_gap=minimum_generation_gap,
        padding=padding,
    )
    return manager.sync(
        records,
        timeline_end=timeline_end,
        species_end_times=species_end_times,
    )


def species_tree_line_width(total_descendants: int) -> float:
    """Return the spindle width for a species' transitive clade size."""
    descendants = max(0, int(total_descendants))
    return max(1.0, min(10.0, log(1.0 + descendants) * 2.0))


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
    """Provide Obstacle UI behavior."""
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
        """Return around.

        Parameters
        ----------
        position
            Value used by the operation.
        radius
            Requested logical size.

        Returns
        -------
        _Obstacle
            Computed result.
        """
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
    """Route candidates.

    Parameters
    ----------
    start
        Value used by the operation.
    end
        Value used by the operation.
    start_radius
        Value used by the operation.
    end_radius
        Value used by the operation.
    clearance
        Value used by the operation.
    obstacles
        Value used by the operation.
    outer_left
        Value used by the operation.
    outer_right
        Value used by the operation.
    outer_top
        Value used by the operation.
    outer_bottom
        Value used by the operation.

    Returns
    -------
    list[SpeciesTreeRoute]
        Computed collection.
    """
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
    """Return compact route.

    Parameters
    ----------
    points
        Value used by the operation.

    Returns
    -------
    SpeciesTreeRoute
        Computed result.
    """
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


def _incremental_edge_route(
    start: tuple[float, float],
    end: tuple[float, float],
) -> SpeciesTreeRoute:
    """Return incremental edge route.

    Parameters
    ----------
    start
        Value used by the operation.
    end
        Value used by the operation.

    Returns
    -------
    SpeciesTreeRoute
        Computed result.
    """
    if start[0] == end[0] or start[1] == end[1]:
        return (start, end)
    midpoint_y = (start[1] + end[1]) * 0.5
    return _compact_route(
        (
            start,
            (start[0], midpoint_y),
            (end[0], midpoint_y),
            end,
        )
    )


def _route_intersects_bounds(
    route: SpeciesTreeRoute,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> bool:
    """Route intersects bounds.

    Parameters
    ----------
    route
        Value used by the operation.
    left
        Logical screen coordinate.
    right
        Logical screen coordinate.
    top
        Logical screen coordinate.
    bottom
        Logical screen coordinate.

    Returns
    -------
    bool
        Whether the operation succeeded or consumed the input.
    """
    low_y, high_y = sorted((top, bottom))
    return any(
        max(min(start[0], end[0]), left)
        <= min(max(start[0], end[0]), right)
        and max(min(start[1], end[1]), low_y)
        <= min(max(start[1], end[1]), high_y)
        for start, end in zip(route, route[1:])
    )


def _route_hits_obstacle(
    route: SpeciesTreeRoute,
    obstacles: tuple[_Obstacle, ...],
) -> bool:
    """Route hits obstacle.

    Parameters
    ----------
    route
        Value used by the operation.
    obstacles
        Value used by the operation.

    Returns
    -------
    bool
        Whether the operation succeeded or consumed the input.
    """
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
    """Return segment hits obstacle.

    Parameters
    ----------
    start
        Value used by the operation.
    end
        Value used by the operation.
    obstacle
        Value used by the operation.

    Returns
    -------
    bool
        Whether the operation succeeded or consumed the input.
    """
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
    """Route length.

    Parameters
    ----------
    route
        Value used by the operation.

    Returns
    -------
    float
        Computed result.
    """
    return sum(
        abs(end[0] - start[0]) + abs(end[1] - start[1])
        for start, end in zip(route, route[1:])
    )


def _valid_time(value: object) -> float | None:
    """Return whether valid time.

    Parameters
    ----------
    value
        Value used by the operation.

    Returns
    -------
    float | None
        Computed result.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(parsed) or parsed < 0.0:
        return None
    return parsed


def _normalize_end_time(value: object) -> float:
    """Return normalize end time.

    Parameters
    ----------
    value
        Value used by the operation.

    Returns
    -------
    float
        Computed result.
    """
    if value is None:
        return float("inf")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if parsed < 0.0 or parsed != parsed:
        return float("inf")
    return parsed


def _break_parent_cycles(parents: dict[int, int | None]) -> None:
    """Return break parent cycles.

    Parameters
    ----------
    parents
        Value used by the operation.
    """
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
