from __future__ import annotations

from dataclasses import dataclass, field
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
    arcade.draw_line = lambda *args, **kwargs: None
    arcade.draw_lrbt_rectangle_filled = lambda *args, **kwargs: None
    arcade.draw_circle_filled = lambda *args, **kwargs: None
    arcade.draw_circle_outline = lambda *args, **kwargs: None
    sys.modules["arcade"] = arcade
else:
    arcade = sys.modules["arcade"]

if not hasattr(arcade, "draw_line"):
    arcade.draw_line = lambda *args, **kwargs: None
if not hasattr(arcade, "draw_lrbt_rectangle_filled"):
    arcade.draw_lrbt_rectangle_filled = lambda *args, **kwargs: None
if not hasattr(arcade, "draw_circle_filled"):
    arcade.draw_circle_filled = lambda *args, **kwargs: None
if not hasattr(arcade, "draw_circle_outline"):
    arcade.draw_circle_outline = lambda *args, **kwargs: None

for optional_module in ("neat",):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

try:
    import pymunk  # noqa: F401
except ModuleNotFoundError:
    sys.modules["pymunk"] = ModuleType("pymunk")

pymunk = sys.modules["pymunk"]
if not hasattr(pymunk, "Space"):
    pymunk.Space = lambda: SimpleNamespace(
        gravity=(0.0, 0.0),
        damping=0.0,
        iterations=0,
        add=lambda *args: None,
        remove=lambda *args: None,
    )
if not hasattr(pymunk, "Shape"):
    pymunk.Shape = object

from configs.sim_config import build_sim_config
from src.layout import build_screen_layout
from src.rendering import EnvironmentRenderer
import src.world as world_module
from src.world import World


@dataclass(slots=True)
class FakeFood:
    position: tuple[float, float]
    radius: float = 8.0
    energy_value: float = 0.1


@dataclass(slots=True)
class FakeCreature:
    creature_id: int
    position: tuple[float, float]
    radius: float = 16.0
    heading: float = 0.0
    energy: float = 1.0
    color: tuple[int, int, int] = (86, 156, 214)
    body: object = field(default_factory=object)
    shape: object = field(default_factory=object)


class CapturingFoodSpawner:
    captured_bounds: tuple[float, float, float, float] | None = None

    def __init__(self, config: object, rng: object) -> None:
        del config, rng

    def create_initial_foods(
        self,
        bounds: tuple[float, float, float, float],
    ) -> list[object]:
        CapturingFoodSpawner.captured_bounds = bounds
        return []

    def creature_pressure_factor(self, creature_count: int) -> float:
        del creature_count
        return 1.0


class FakeNeatBrainController:
    def __init__(self, config_path: str) -> None:
        del config_path

    def assign_initial_brains(self, creature_ids: list[int]) -> None:
        del creature_ids


class WorldCameraTest(unittest.TestCase):
    def make_world_shell(self) -> World:
        config = build_sim_config()
        world = object.__new__(World)
        world.config = config
        world.layout = build_screen_layout(
            config.display.width,
            config.display.height,
            config.layout,
        )
        world.environment_zoom = config.zoom.default
        world.environment_pan_x = 0.0
        world.environment_pan_y = 0.0
        world.foods = []
        world.creatures = []
        world._food_grid = {}
        world._food_grid_dirty = True
        world._food_grid_cell_size = 100.0
        return world

    def test_fixed_world_bounds_do_not_change_after_resize(self) -> None:
        world = self.make_world_shell()
        before = world.environment_world_bounds

        world.resize(2200, 1400)

        self.assertEqual(world.environment_world_bounds, before)

    def test_screen_environment_transform_round_trips_after_resize(self) -> None:
        world = self.make_world_shell()
        world.environment_zoom = 1.5
        world.environment_pan_x = -120.0
        world.environment_pan_y = 75.0

        for width, height in ((1440, 900), (2200, 1400)):
            world.resize(width, height)
            screen = world.environment_to_screen(320.0, -140.0)
            model = world.screen_to_environment(*screen)

            self.assertAlmostEqual(model[0], 320.0)
            self.assertAlmostEqual(model[1], -140.0)

    def test_visible_world_bounds_reflect_pan_and_zoom(self) -> None:
        world = self.make_world_shell()
        world.environment_zoom = 2.0
        world.environment_pan_x = -100.0
        world.environment_pan_y = 80.0

        left, bottom, right, top = world.visible_world_bounds()
        screen_left = world.screen_to_environment(world.layout.environment.left, 0.0)[0]
        screen_right = world.screen_to_environment(world.layout.environment.right, 0.0)[0]

        self.assertAlmostEqual(left, screen_left)
        self.assertAlmostEqual(right, screen_right)
        self.assertLess(bottom, top)

    def test_zoom_out_can_reveal_whole_fixed_world(self) -> None:
        world = self.make_world_shell()
        world.environment_pan_x = 200.0
        world.environment_pan_y = -100.0

        for _ in range(20):
            world.adjust_environment_zoom(-1)

        self.assertEqual(world.environment_zoom, 0.3)
        self.assertEqual(
            world.visible_world_bounds(),
            world.environment_world_bounds,
        )
        self.assertEqual(world.environment_pan_x, 0.0)
        self.assertEqual(world.environment_pan_y, 0.0)

    def test_initial_food_spawns_use_fixed_world_bounds(self) -> None:
        config = build_sim_config()
        config.population.initial_creatures = 0
        config.food.initial_food_items = 0
        original_rebuild_boundaries = World._rebuild_boundaries
        original_food_spawner = world_module.FoodSpawner
        original_neat_controller = world_module.NeatBrainController

        CapturingFoodSpawner.captured_bounds = None
        World._rebuild_boundaries = lambda self: None
        world_module.FoodSpawner = CapturingFoodSpawner
        world_module.NeatBrainController = FakeNeatBrainController

        try:
            World(config)
        finally:
            World._rebuild_boundaries = original_rebuild_boundaries
            world_module.FoodSpawner = original_food_spawner
            world_module.NeatBrainController = original_neat_controller

        self.assertEqual(
            CapturingFoodSpawner.captured_bounds,
            (-1600.0, -1100.0, 1600.0, 1100.0),
        )

    def test_viewport_food_query_excludes_distant_food(self) -> None:
        world = self.make_world_shell()
        visible_food = FakeFood(position=(0.0, 0.0))
        distant_food = FakeFood(position=(1500.0, 1000.0))
        world.foods = [visible_food, distant_food]

        self.assertEqual(world.visible_foods_for_viewport(), [visible_food])

    def test_viewport_creature_query_excludes_distant_creatures(self) -> None:
        world = self.make_world_shell()
        visible_creature = FakeCreature(creature_id=1, position=(0.0, 0.0))
        distant_creature = FakeCreature(creature_id=2, position=(1500.0, 1000.0))
        world.creatures = [visible_creature, distant_creature]

        self.assertEqual(world.visible_creatures_for_viewport(), [visible_creature])

    def test_renderer_does_not_draw_offscreen_food_candidates(self) -> None:
        world = self.make_world_shell()
        renderer = EnvironmentRenderer(world.config)
        foods = [
            FakeFood(position=(0.0, 0.0)),
            FakeFood(position=(1500.0, 1000.0)),
        ]
        draw_calls: list[tuple[float, float, float]] = []
        original_draw_circle_filled = arcade.draw_circle_filled
        original_draw_circle_outline = arcade.draw_circle_outline
        arcade.draw_circle_filled = lambda x, y, radius, color: draw_calls.append(
            (x, y, radius)
        )
        arcade.draw_circle_outline = lambda *args, **kwargs: None

        try:
            renderer._draw_food(foods, world.layout.environment, world)
        finally:
            arcade.draw_circle_filled = original_draw_circle_filled
            arcade.draw_circle_outline = original_draw_circle_outline

        self.assertEqual(len(draw_calls), 1)


if __name__ == "__main__":
    unittest.main()
