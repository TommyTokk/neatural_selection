from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys
import unittest

if "arcade" not in sys.modules:
    arcade = ModuleType("arcade")

    @dataclass(slots=True)
    class FakeRect:
        left: float
        bottom: float
        width: float
        height: float

        @property
        def right(self) -> float:
            return self.left + self.width

        @property
        def top(self) -> float:
            return self.bottom + self.height

        @property
        def center_x(self) -> float:
            return self.left + self.width / 2.0

        @property
        def center_y(self) -> float:
            return self.bottom + self.height / 2.0

    def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
        return FakeRect(left, bottom, width, height)

    arcade.Rect = FakeRect
    arcade.LBWH = fake_lbwh
    sys.modules["arcade"] = arcade

for optional_module in ("neat", "pymunk"):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

from src.action import Action
from src.world import World


@dataclass(slots=True)
class FakePoint:
    x: float
    y: float


@dataclass(slots=True)
class FakeBody:
    position: FakePoint | tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    angular_velocity: float = 0.0


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    body: FakeBody
    radius: float = 10.0
    heading: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        position = self.body.position
        if isinstance(position, tuple):
            return position
        return position.x, position.y


@dataclass(slots=True)
class FakeFood:
    id: int
    body: FakeBody
    radius: float = 4.0
    overlaps_mouth: bool = True
    shape: object = object()

    @property
    def position(self) -> tuple[float, float]:
        position = self.body.position
        if isinstance(position, tuple):
            return position
        return position.x, position.y


class FakeMetabolism:
    def __init__(self) -> None:
        self.report = SimpleNamespace(
            food_consumptions=[],
            touched_foods=[],
            depleted_foods=[],
            dead_creatures=[],
        )
        self.seen_food_items: list[FakeFood] = []

    def mouth_position(self, creature: FakeCreature) -> tuple[float, float]:
        return creature.position[0] + creature.radius, creature.position[1]

    def food_overlaps_mouth(self, creature: FakeCreature, food: FakeFood) -> bool:
        del creature
        return food.overlaps_mouth

    def update(
        self,
        creatures: list[FakeCreature],
        food_items: list[FakeFood],
        *args: object,
        **kwargs: object,
    ) -> object:
        del creatures, args, kwargs
        self.seen_food_items = food_items
        return self.report


def action(
    *,
    want_grab: float = 0.0,
    want_release: float = 0.0,
    want_eat: float = 0.0,
) -> Action:
    return Action(
        accelerate=0.0,
        rotate=0.0,
        want_reproduce=0.0,
        want_eat=want_eat,
        reset_chronometer=0.0,
        want_grab=want_grab,
        want_release=want_release,
    )


class WorldCarryTest(unittest.TestCase):
    def make_world(
        self,
        creatures: list[FakeCreature],
        foods: list[FakeFood],
    ) -> tuple[World, FakeMetabolism, list[object]]:
        world = object.__new__(World)
        metabolism = FakeMetabolism()
        removed: list[object] = []
        world.creatures = creatures
        world.foods = foods
        world.metabolism = metabolism
        world._held_food_by_creature_id = {}
        world._carrier_by_food_id = {}
        world._last_actions = {}
        world._food_grid_dirty = False
        world.space = SimpleNamespace(
            reindex_shape=lambda shape: None,
            remove=lambda *items: removed.extend(items),
        )
        world.fitness = {}
        world.selected_creature_id = None
        world.MAX_SPEED = 100.0
        world._eatable_foods_for = lambda creature: foods
        return world, metabolism, removed

    def test_sensor_snapshot_reports_current_carry_state(self) -> None:
        creature = SimpleNamespace(
            creature_id=1,
            position=(0.0, 0.0),
            heading=0.0,
            vision=SimpleNamespace(range=100.0),
        )
        world = object.__new__(World)
        world.config = SimpleNamespace(
            food=SimpleNamespace(max_food_radius=10.0),
            trait=SimpleNamespace(max_radius=20.0),
            population=SimpleNamespace(min_reproduction_age=10.0),
            environment=SimpleNamespace(world_width=100.0, world_height=100.0),
            biome_sensor=SimpleNamespace(
                forward_distance=48.0,
                side_offset=24.0,
            ),
        )
        world.MAX_SPEED = 100.0
        world.creatures = [creature]
        world.fitness = {}
        world._chronometers = {}
        world._held_food_by_creature_id = {}
        world._nearby_foods_for = lambda creature, distance: []
        world._nearby_creatures_for = lambda creature, distance: []
        world.vision = SimpleNamespace(
            sense=lambda *args, **kwargs: SimpleNamespace(
                is_grabbing=kwargs["is_grabbing"],
            ),
        )

        self.assertEqual(world.sensor_snapshot_for(creature).is_grabbing, 0.0)

        world._held_food_by_creature_id[creature.creature_id] = 7

        self.assertEqual(world.sensor_snapshot_for(creature).is_grabbing, 1.0)

    def test_grab_assigns_nearest_mouth_overlapping_food(self) -> None:
        creature = FakeCreature(1, FakeBody(FakePoint(0.0, 0.0)))
        far_food = FakeFood(1, FakeBody(FakePoint(30.0, 0.0)), overlaps_mouth=True)
        near_food = FakeFood(2, FakeBody(FakePoint(11.0, 0.0)), overlaps_mouth=True)
        world, _, _ = self.make_world(creatures=[creature], foods=[far_food, near_food])

        world._apply_carry_intent(creature, action(want_grab=1.0))

        self.assertEqual(world._held_food_by_creature_id, {1: 2})
        self.assertEqual(world._carrier_by_food_id, {2: 1})

    def test_grab_ignores_food_that_is_not_mouth_overlapping(self) -> None:
        creature = FakeCreature(1, FakeBody(FakePoint(0.0, 0.0)))
        food = FakeFood(1, FakeBody(FakePoint(11.0, 0.0)), overlaps_mouth=False)
        world, _, _ = self.make_world(creatures=[creature], foods=[food])

        world._apply_carry_intent(creature, action(want_grab=1.0))

        self.assertEqual(world._held_food_by_creature_id, {})
        self.assertEqual(world._carrier_by_food_id, {})

    def test_release_wins_when_grab_and_release_are_both_high(self) -> None:
        creature = FakeCreature(1, FakeBody(FakePoint(0.0, 0.0)))
        food = FakeFood(1, FakeBody(FakePoint(11.0, 0.0)))
        world, _, _ = self.make_world(creatures=[creature], foods=[food])
        world._held_food_by_creature_id = {1: 1}
        world._carrier_by_food_id = {1: 1}

        world._apply_carry_intent(
            creature,
            action(want_grab=1.0, want_release=1.0),
        )

        self.assertEqual(world._held_food_by_creature_id, {})
        self.assertEqual(world._carrier_by_food_id, {})

    def test_carried_food_follows_carrier_front_offset(self) -> None:
        creature = FakeCreature(
            1,
            FakeBody(FakePoint(5.0, 3.0), velocity=(7.0, 2.0), angular_velocity=0.4),
            radius=10.0,
        )
        food = FakeFood(1, FakeBody(FakePoint(0.0, 0.0)), radius=4.0)
        world, _, _ = self.make_world(creatures=[creature], foods=[food])
        world._held_food_by_creature_id = {1: 1}
        world._carrier_by_food_id = {1: 1}

        world._sync_carried_foods()

        self.assertEqual(food.position, (17.0, 3.0))
        self.assertEqual(food.body.velocity, creature.body.velocity)
        self.assertEqual(food.body.angular_velocity, creature.body.angular_velocity)
        self.assertFalse(world._food_grid_dirty)
        self.assertEqual(world._food_grid_cells_by_id[food.id], (0, 0))
        self.assertEqual(world._food_grid[(0, 0)], [food])

    def test_food_grid_reindex_moves_food_between_cells_without_dirty_rebuild(self) -> None:
        world = object.__new__(World)
        food = FakeFood(1, FakeBody(FakePoint(3.0, 3.0)))
        world.foods = [food]
        world._food_grid = {}
        world._food_grid_cells_by_id = {}
        world._food_grid_dirty = False
        world._food_grid_cell_size = 10.0

        world._index_food(food)
        food.body.position = FakePoint(25.0, 3.0)
        world._reindex_food(food)

        self.assertFalse(world._food_grid_dirty)
        self.assertNotIn((0, 0), world._food_grid)
        self.assertEqual(world._food_grid_cells_by_id[food.id], (2, 0))
        self.assertEqual(
            world._foods_in_world_bounds(20.0, 0.0, 30.0, 10.0),
            [food],
        )

    def test_depleted_carried_food_clears_ownership_before_removal(self) -> None:
        creature = FakeCreature(1, FakeBody(FakePoint(0.0, 0.0)))
        food = FakeFood(1, FakeBody(FakePoint(11.0, 0.0)))
        world, metabolism, removed = self.make_world(creatures=[creature], foods=[food])
        world._held_food_by_creature_id = {1: 1}
        world._carrier_by_food_id = {1: 1}
        metabolism.report.depleted_foods = [food]

        world._update_metabolism(1.0)

        self.assertEqual(world._held_food_by_creature_id, {})
        self.assertEqual(world._carrier_by_food_id, {})
        self.assertEqual(world.foods, [])
        self.assertEqual(removed, [food.body, food.shape])

    def test_other_creature_can_eat_but_cannot_grab_carried_food(self) -> None:
        carrier = FakeCreature(1, FakeBody(FakePoint(0.0, 0.0)))
        other = FakeCreature(2, FakeBody(FakePoint(0.0, 0.0)))
        food = FakeFood(1, FakeBody(FakePoint(11.0, 0.0)))
        world, metabolism, _ = self.make_world(
            creatures=[carrier, other],
            foods=[food],
        )
        world._held_food_by_creature_id = {1: 1}
        world._carrier_by_food_id = {1: 1}

        world._apply_carry_intent(other, action(want_grab=1.0))
        world._update_metabolism(1.0)

        self.assertEqual(world._held_food_by_creature_id, {1: 1})
        self.assertEqual(world._carrier_by_food_id, {1: 1})
        self.assertEqual(metabolism.seen_food_items, [food])


if __name__ == "__main__":
    unittest.main()
