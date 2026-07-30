from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from src.flocking import FlockingRuntimeSnapshot


@dataclass(frozen=True, slots=True)
class TrackedGroup:
    group_id: int
    members: frozenset[int]
    created_at: float
    centroid: tuple[float, float]
    mean_velocity: tuple[float, float]
    displacement: float = 0.0


@dataclass(frozen=True, slots=True)
class GroupSample:
    groups: tuple[TrackedGroup, ...] = ()
    group_by_creature: dict[int, tuple[int, int]] = field(default_factory=dict)
    fragmentation_count: int = 0
    merger_count: int = 0


class PersistentGroupTracker:
    def __init__(self, overlap_threshold: float = 0.5) -> None:
        self.overlap_threshold = max(0.0, min(1.0, overlap_threshold))
        self.next_group_id = 1
        self.previous: dict[int, TrackedGroup] = {}

    def sample(
        self,
        creatures: list[object],
        *,
        sim_time: float,
        group_range: float,
        minimum_compatibility: float,
        compatibility,
        nearby,
    ) -> GroupSample:
        by_id = {creature.creature_id: creature for creature in creatures}
        parent = {creature_id: creature_id for creature_id in by_id}

        def find(value: int) -> int:
            root = value
            while parent[root] != root:
                root = parent[root]
            while parent[value] != value:
                next_value = parent[value]
                parent[value] = root
                value = next_value
            return root

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(
                    first_root, second_root
                )

        seen_pairs: set[tuple[int, int]] = set()
        for creature in creatures:
            for neighbor in nearby(creature, group_range):
                pair = tuple(
                    sorted((creature.creature_id, neighbor.creature_id))
                )
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                dx = creature.position[0] - neighbor.position[0]
                dy = creature.position[1] - neighbor.position[1]
                if hypot(dx, dy) > group_range:
                    continue
                if compatibility(creature, neighbor) < minimum_compatibility:
                    continue
                union(*pair)

        components: dict[int, set[int]] = {}
        for creature_id in by_id:
            components.setdefault(find(creature_id), set()).add(creature_id)
        current_members = [
            frozenset(members)
            for members in components.values()
            if len(members) >= 2
        ]

        overlaps: list[tuple[float, int, int]] = []
        previous_items = list(self.previous.items())
        for current_index, members in enumerate(current_members):
            for previous_id, previous_group in previous_items:
                union_size = len(members | previous_group.members)
                score = (
                    0.0
                    if union_size == 0
                    else len(members & previous_group.members) / union_size
                )
                if score >= self.overlap_threshold:
                    overlaps.append((-score, previous_id, current_index))
        overlaps.sort()
        matched_previous: set[int] = set()
        matched_current: set[int] = set()
        assigned: dict[int, int] = {}
        for _negative_score, previous_id, current_index in overlaps:
            if (
                previous_id in matched_previous
                or current_index in matched_current
            ):
                continue
            matched_previous.add(previous_id)
            matched_current.add(current_index)
            assigned[current_index] = previous_id

        fragmentation = sum(
            1
            for group in self.previous.values()
            if sum(bool(group.members & members) for members in current_members)
            > 1
        )
        merger = sum(
            1
            for members in current_members
            if sum(bool(group.members & members) for group in self.previous.values())
            > 1
        )

        groups: list[TrackedGroup] = []
        group_by_creature: dict[int, tuple[int, int]] = {}
        for index, members in enumerate(current_members):
            group_id = assigned.get(index)
            previous_group = (
                None if group_id is None else self.previous[group_id]
            )
            if group_id is None:
                group_id = self.next_group_id
                self.next_group_id += 1
            members_creatures = [by_id[member] for member in members]
            centroid = (
                sum(item.position[0] for item in members_creatures)
                / len(members_creatures),
                sum(item.position[1] for item in members_creatures)
                / len(members_creatures),
            )
            mean_velocity = (
                sum(item.body.velocity.x for item in members_creatures)
                / len(members_creatures),
                sum(item.body.velocity.y for item in members_creatures)
                / len(members_creatures),
            )
            displacement = (
                0.0
                if previous_group is None
                else hypot(
                    centroid[0] - previous_group.centroid[0],
                    centroid[1] - previous_group.centroid[1],
                )
            )
            group = TrackedGroup(
                group_id=group_id,
                members=members,
                created_at=(
                    sim_time
                    if previous_group is None
                    else previous_group.created_at
                ),
                centroid=centroid,
                mean_velocity=mean_velocity,
                displacement=displacement,
            )
            groups.append(group)
            for member in members:
                group_by_creature[member] = (group_id, len(members))
        self.previous = {group.group_id: group for group in groups}
        return GroupSample(
            groups=tuple(groups),
            group_by_creature=group_by_creature,
            fragmentation_count=fragmentation,
            merger_count=merger,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "overlap_threshold": self.overlap_threshold,
            "next_group_id": self.next_group_id,
            "previous": self.previous,
        }

    def restore(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self.overlap_threshold = float(
            state.get("overlap_threshold", self.overlap_threshold)
        )
        self.next_group_id = int(state.get("next_group_id", 1))
        self.previous = dict(state.get("previous", {}))


class FlockingTelemetryAggregator:
    @staticmethod
    def aggregate(
        *,
        sim_time: float,
        population_size: int,
        runtime: dict[int, FlockingRuntimeSnapshot],
        groups: GroupSample,
    ) -> dict[str, float | int]:
        snapshots = list(runtime.values())
        count = max(1, population_size)

        def mean(values) -> float:
            values = list(values)
            return 0.0 if not values else sum(values) / len(values)

        effective_counts = [
            item.observation.effective_count for item in snapshots
        ]
        group_members = sum(
            len(group.members) for group in groups.groups if len(group.members) >= 3
        )
        return {
            "sim_time": sim_time,
            "population_size": population_size,
            "seeing_any_percent": 100.0
            * sum(
                item.observation.visible_creature_count > 0
                for item in snapshots
            )
            / count,
            "seeing_compatible_percent": 100.0
            * sum(
                item.observation.compatible_visible_count > 0
                for item in snapshots
            )
            / count,
            "effective_count_ge_2_percent": 100.0
            * sum(value >= 2.0 for value in effective_counts)
            / count,
            "mean_effective_count": mean(effective_counts),
            "max_effective_count": max(effective_counts, default=0.0),
            "mean_engagement": mean(
                item.intent.weights.engagement for item in snapshots
            ),
            "mean_neural_herding": mean(
                item.raw_neural_herding for item in snapshots
            ),
            "mean_effective_herding": mean(
                item.effective_herding for item in snapshots
            ),
            "mean_panic": mean(item.panic for item in snapshots),
            "mean_panic_attenuation": mean(
                item.intent.weights.panic_attenuation for item in snapshots
            ),
            "mean_separation_weight": mean(
                item.intent.weights.separation for item in snapshots
            ),
            "mean_alignment_weight": mean(
                item.intent.weights.alignment for item in snapshots
            ),
            "mean_cohesion_weight": mean(
                item.intent.weights.cohesion for item in snapshots
            ),
            "mean_requested_social_force": mean(
                hypot(*item.requested_social_contribution)
                for item in snapshots
            ),
            "mean_accepted_social_force": mean(
                hypot(*item.accepted_counterfactual_delta)
                for item in snapshots
            ),
            "mean_social_blend": mean(
                item.social_influence for item in snapshots
            ),
            "mean_alignment_error": mean(
                item.observation.mean_heading_error for item in snapshots
            ),
            "mean_center_distance": mean(
                item.observation.center_distance for item in snapshots
            ),
            "in_groups_ge_3_percent": 100.0 * group_members / count,
            "largest_group_size": max(
                (len(group.members) for group in groups.groups),
                default=0,
            ),
            "mean_group_lifetime": mean(
                sim_time - group.created_at for group in groups.groups
            ),
            "fragmentation_count": groups.fragmentation_count,
            "merger_count": groups.merger_count,
        }
