from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Iterator, Sequence


@dataclass(slots=True)
class SpatialIndexCounters:
    """Allocation and workload counters kept outside timed benchmark samples."""

    rebuilds: int = 0
    failed_rebuilds: int = 0
    queries: int = 0
    candidates: int = 0
    cell_visits: int = 0
    cell_resets: int = 0
    cell_buffer_growth: int = 0
    candidate_buffer_growth: int = 0
    family_buffer_growth: int = 0
    maximum_candidates: int = 0
    maximum_active_cells: int = 0
    maximum_cell_occupancy: int = 0
    stable_sorts: int = 0
    invalid_slots_skipped: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


class CandidateBuffer(Sequence[object]):
    """Retained active-prefix storage for spatial slot numbers.

    The buffer deliberately exposes creatures instead of wrapper objects.  Slot
    metadata remains owned by ``CreatureSpatialIndex`` and is validated on each
    access against both the rebuild generation and the living registry.
    """

    __slots__ = ("index", "slots", "count", "generation")

    def __init__(self, index: CreatureSpatialIndex | None = None) -> None:
        self.index = index
        self.slots: list[int] = []
        self.count = 0
        self.generation = 0

    def reset(self, index: CreatureSpatialIndex, generation: int) -> None:
        self.index = index
        self.count = 0
        self.generation = generation

    def invalidate(self) -> None:
        self.count = 0
        self.generation = 0
        self.index = None

    def append_slot(self, slot: int) -> None:
        if self.count == len(self.slots):
            growth = max(16, len(self.slots) or 16)
            self.slots.extend([0] * growth)
            if self.index is not None:
                self.index.counters.candidate_buffer_growth += 1
        self.slots[self.count] = slot
        self.count += 1

    def sort_by_stable_id(self) -> None:
        index = self._valid_index()
        if self.count < 2:
            return
        # Insertion sort changes only the active prefix and creates neither a
        # replacement list nor per-candidate key wrappers.
        for cursor in range(1, self.count):
            slot = self.slots[cursor]
            stable_id = index.stable_ids[slot]
            insertion = cursor
            while (
                insertion > 0
                and index.stable_ids[self.slots[insertion - 1]] > stable_id
            ):
                self.slots[insertion] = self.slots[insertion - 1]
                insertion -= 1
            self.slots[insertion] = slot
        index.counters.stable_sorts += 1

    def _valid_index(self) -> CreatureSpatialIndex:
        index = self.index
        if index is None or self.generation != index.generation:
            raise RuntimeError("Spatial candidate buffer is stale.")
        return index

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, item):
        if isinstance(item, slice):
            # Public/debug callers may request a materialized slice.  The
            # simulation hot path only indexes and iterates the active prefix.
            return [self[index] for index in range(*item.indices(self.count))]
        position = int(item)
        if position < 0:
            position += self.count
        if position < 0 or position >= self.count:
            raise IndexError(position)
        index = self._valid_index()
        creature = index.creature_for_slot(self.slots[position])
        if creature is None:
            raise RuntimeError("Spatial candidate slot is no longer live.")
        return creature

    def __iter__(self) -> Iterator[object]:
        index = self._valid_index()
        generation = index.generation
        creatures = index.creatures
        stable_ids = index.stable_ids
        slot_generations = index.slot_generations
        registry = index.living_registry
        for position in range(self.count):
            slot = self.slots[position]
            creature = creatures[slot]
            stable_id = stable_ids[slot]
            if (
                slot_generations[slot] == generation
                and creature is not None
                and getattr(creature, "creature_id", None) == stable_id
                and registry.get(stable_id) is creature
            ):
                yield creature
            else:
                index.counters.invalid_slots_skipped += 1


class FamilyView(Sequence[object]):
    """Non-allocating view of the current-generation children of a parent."""

    __slots__ = ("index", "slots", "count", "generation", "infant_predicate")

    def __init__(
        self,
        index: CreatureSpatialIndex,
        slots: list[int],
        count: int,
        infant_predicate,
    ) -> None:
        self.index = index
        self.slots = slots
        self.count = count
        self.generation = index.generation
        self.infant_predicate = infant_predicate

    def __iter__(self) -> Iterator[object]:
        if self.generation != self.index.generation:
            return
        for position in range(self.count):
            creature = self.index.creature_for_slot(self.slots[position])
            if creature is not None and self.infant_predicate(creature):
                yield creature

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return list(self)[item]
        position = int(item)
        if position < 0:
            position += len(self)
        for index, creature in enumerate(self):
            if index == position:
                return creature
        raise IndexError(position)


@dataclass(frozen=True, slots=True)
class BroadPhaseGeometry:
    collision: float
    detailed_vision: float
    flocking: float
    long_range: float
    scheduled: float

    @classmethod
    def calculate(
        cls,
        *,
        observer_radius: float,
        maximum_target_radius: float,
        collision_margin: float,
        vision_range: float,
        flock_range: float,
        long_range: float = 0.0,
        long_range_enabled: bool = False,
    ) -> BroadPhaseGeometry:
        observer = max(0.0, float(observer_radius))
        target = max(0.0, float(maximum_target_radius))
        collision = observer + max(0.0, collision_margin) + target
        detailed_vision = 0.35 * observer + max(0.0, vision_range) + target
        flocking = max(0.0, flock_range)
        enabled_long_range = max(0.0, long_range) if long_range_enabled else 0.0
        return cls(
            collision=collision,
            detailed_vision=detailed_vision,
            flocking=flocking,
            long_range=enabled_long_range,
            scheduled=max(
                collision,
                detailed_vision,
                flocking,
                enabled_long_range,
            ),
        )


class CreatureSpatialIndex:
    """Retained deterministic uniform grid over authoritative circle centres."""

    def __init__(
        self,
        *,
        cell_size: float = 64.0,
        living_registry: dict[int, object] | None = None,
    ) -> None:
        if cell_size <= 0.0:
            raise ValueError("cell_size must be positive.")
        self.cell_size = float(cell_size)
        self.living_registry = (
            living_registry if living_registry is not None else {}
        )
        self.generation = 0
        self.valid = False
        self.maximum_radius = 0.0
        self.creatures: list[object | None] = []
        self.stable_ids: list[int] = []
        self.centres_x: list[float] = []
        self.centres_y: list[float] = []
        self.radii: list[float] = []
        self.slot_generations: list[int] = []
        self._slot_by_stable_id: dict[int, int] = {}
        self._cells: dict[tuple[int, int], list[int]] = {}
        self._cell_counts: dict[tuple[int, int], int] = {}
        self._cell_pool: list[list[int]] = []
        self._active_cells: list[tuple[int, int]] = []
        self._family_slots: dict[int, list[int]] = {}
        self._family_counts: dict[int, int] = {}
        self._family_pool: list[list[int]] = []
        self.counters = SpatialIndexCounters()

    @staticmethod
    def _circle_values(creature: object) -> tuple[float, float, float]:
        body = getattr(creature, "body", None)
        position = (
            getattr(body, "position")
            if body is not None
            else getattr(creature, "position", (0.0, 0.0))
        )
        x = position.x if hasattr(position, "x") else position[0]
        y = position.y if hasattr(position, "y") else position[1]
        shape = getattr(creature, "shape", None)
        radius = getattr(shape, "radius", getattr(creature, "radius", 0.0))
        return float(x), float(y), float(radius)

    def _ensure_slot_capacity(self, capacity: int) -> None:
        if capacity <= len(self.creatures):
            return
        growth = capacity - len(self.creatures)
        self.creatures.extend([None] * growth)
        self.stable_ids.extend([0] * growth)
        self.centres_x.extend([0.0] * growth)
        self.centres_y.extend([0.0] * growth)
        self.radii.extend([0.0] * growth)
        self.slot_generations.extend([0] * growth)

    def _claim_cell(self, key: tuple[int, int]) -> list[int]:
        cell = self._cells.get(key)
        if cell is not None:
            return cell
        cell = self._cell_pool.pop() if self._cell_pool else []
        if not cell:
            self.counters.cell_buffer_growth += 1
        self._cells[key] = cell
        self._cell_counts[key] = 0
        self._active_cells.append(key)
        return cell

    def _append_cell_slot(self, key: tuple[int, int], slot: int) -> None:
        cell = self._claim_cell(key)
        count = self._cell_counts[key]
        if count == len(cell):
            cell.extend([0] * max(8, len(cell) or 8))
            self.counters.cell_buffer_growth += 1
        cell[count] = slot
        count += 1
        self._cell_counts[key] = count
        self.counters.maximum_cell_occupancy = max(
            self.counters.maximum_cell_occupancy,
            count,
        )

    def _append_family_slot(self, parent_id: int, slot: int) -> None:
        family = self._family_slots.get(parent_id)
        if family is None:
            family = self._family_pool.pop() if self._family_pool else []
            self._family_slots[parent_id] = family
            self._family_counts[parent_id] = 0
        count = self._family_counts[parent_id]
        if count == len(family):
            family.extend([0] * max(4, len(family) or 4))
            self.counters.family_buffer_growth += 1
        family[count] = slot
        self._family_counts[parent_id] = count + 1

    def _reset_active_storage(self) -> None:
        self.counters.cell_resets += len(self._active_cells)
        for key in self._active_cells:
            cell = self._cells.pop(key)
            self._cell_pool.append(cell)
        self._active_cells.clear()
        self._cell_counts.clear()
        for family in self._family_slots.values():
            self._family_pool.append(family)
        self._family_slots.clear()
        self._family_counts.clear()

    def rebuild(self, creatures: Sequence[object]) -> None:
        """Publish a new generation only after a complete successful rebuild."""
        next_generation = self.generation + 1
        self.valid = False
        self.maximum_radius = 0.0
        self._reset_active_storage()
        self._slot_by_stable_id.clear()
        try:
            self._ensure_slot_capacity(len(creatures))
            seen: set[int] = set()
            for slot, creature in enumerate(creatures):
                stable_id = getattr(creature, "creature_id", None)
                if type(stable_id) is not int:
                    raise TypeError("Every indexed creature needs an integer ID.")
                if stable_id in seen:
                    raise ValueError(f"Duplicate live creature ID {stable_id}.")
                seen.add(stable_id)
                if self.living_registry.get(stable_id) is not creature:
                    raise ValueError(
                        f"Creature {stable_id} is absent from the living registry."
                    )
                x, y, radius = self._circle_values(creature)
                self.creatures[slot] = creature
                self.stable_ids[slot] = stable_id
                self.centres_x[slot] = x
                self.centres_y[slot] = y
                self.radii[slot] = radius
                self.maximum_radius = max(self.maximum_radius, radius)
                self.slot_generations[slot] = next_generation
                self._slot_by_stable_id[stable_id] = slot
                key = (floor(x / self.cell_size), floor(y / self.cell_size))
                self._append_cell_slot(key, slot)
                lineage = getattr(creature, "lineage", None)
                parent_id = getattr(lineage, "parent_id", None)
                if type(parent_id) is int:
                    self._append_family_slot(parent_id, slot)
            for slot in range(len(creatures), len(self.creatures)):
                self.creatures[slot] = None
                self.slot_generations[slot] = 0
            self._active_cells.sort()
        except BaseException:
            self.counters.failed_rebuilds += 1
            self._reset_active_storage()
            self.valid = False
            raise
        self.generation = next_generation
        self.valid = True
        self.counters.rebuilds += 1
        self.counters.maximum_active_cells = max(
            self.counters.maximum_active_cells,
            len(self._active_cells),
        )

    def query_into(
        self,
        x: float,
        y: float,
        centre_radius: float,
        output: CandidateBuffer,
    ) -> CandidateBuffer:
        if not self.valid:
            raise RuntimeError("Spatial index has no complete rebuild.")
        output.reset(self, self.generation)
        radius = max(0.0, float(centre_radius))
        minimum_x = floor((float(x) - radius) / self.cell_size)
        maximum_x = floor((float(x) + radius) / self.cell_size)
        minimum_y = floor((float(y) - radius) / self.cell_size)
        maximum_y = floor((float(y) + radius) / self.cell_size)
        rectangle_cell_count = (
            (maximum_x - minimum_x + 1)
            * (maximum_y - minimum_y + 1)
        )
        if len(self._active_cells) < rectangle_cell_count:
            for key in self._active_cells:
                if not (
                    minimum_x <= key[0] <= maximum_x
                    and minimum_y <= key[1] <= maximum_y
                ):
                    continue
                self.counters.cell_visits += 1
                cell = self._cells[key]
                for position in range(self._cell_counts[key]):
                    output.append_slot(cell[position])
        else:
            for cell_y in range(minimum_y, maximum_y + 1):
                for cell_x in range(minimum_x, maximum_x + 1):
                    key = (cell_x, cell_y)
                    self.counters.cell_visits += 1
                    cell = self._cells.get(key)
                    if cell is None:
                        continue
                    for position in range(self._cell_counts[key]):
                        output.append_slot(cell[position])
        self.counters.queries += 1
        self.counters.candidates += output.count
        self.counters.maximum_candidates = max(
            self.counters.maximum_candidates,
            output.count,
        )
        return output

    def creature_for_slot(self, slot: int) -> object | None:
        if (
            not self.valid
            or slot < 0
            or slot >= len(self.creatures)
            or self.slot_generations[slot] != self.generation
        ):
            self.counters.invalid_slots_skipped += 1
            return None
        creature = self.creatures[slot]
        stable_id = self.stable_ids[slot]
        if (
            creature is None
            or getattr(creature, "creature_id", None) != stable_id
            or self.living_registry.get(stable_id) is not creature
        ):
            self.counters.invalid_slots_skipped += 1
            return None
        return creature

    def values_for(self, creature: object) -> tuple[float, float, float] | None:
        stable_id = getattr(creature, "creature_id", None)
        if type(stable_id) is not int or not self.valid:
            return None
        # The living registry check prevents a stale entity with a recycled
        # object address from observing another creature's retained position.
        if self.living_registry.get(stable_id) is not creature:
            return None
        slot = self._slot_by_stable_id.get(stable_id)
        if (
            slot is not None
            and self.creature_for_slot(slot) is creature
        ):
            return (
                self.centres_x[slot],
                self.centres_y[slot],
                self.radii[slot],
            )
        return None

    def family_view(self, parent_id: int, infant_predicate) -> FamilyView:
        slots = self._family_slots.get(parent_id, ())
        count = self._family_counts.get(parent_id, 0)
        return FamilyView(self, slots, count, infant_predicate)
