from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import importlib
import sys
import unittest


if "pyglet" not in sys.modules:
    pyglet = ModuleType("pyglet")
    pyglet.options = {}
    sys.modules["pyglet"] = pyglet
else:
    pyglet = sys.modules["pyglet"]
    if not hasattr(pyglet, "options"):
        pyglet.options = {}

if "arcade" not in sys.modules:
    arcade = ModuleType("arcade")
    sys.modules["arcade"] = arcade
else:
    arcade = sys.modules["arcade"]


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


class FakeView:
    def __init__(self) -> None:
        self.window = None
        self.clear_count = 0

    def clear(self) -> None:
        self.clear_count += 1

    def on_mouse_press(
        self,
        x: int,
        y: int,
        button: int,
        modifiers: int,
    ) -> bool | None:
        del x, y, button, modifiers
        return None

    def on_mouse_release(
        self,
        x: int,
        y: int,
        button: int,
        modifiers: int,
    ) -> bool | None:
        del x, y, button, modifiers
        return None

    def on_mouse_motion(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> bool | None:
        del x, y, dx, dy
        return None

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        del symbol, modifiers
        return None


class FakeText:
    def __init__(
        self,
        text: str,
        x: float,
        y: float,
        color: object,
        size: float,
        **kwargs: object,
    ) -> None:
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.font_size = size
        self.bold = kwargs.get("bold", False)
        self.width = kwargs.get("width")
        self.multiline = kwargs.get("multiline", False)
        self.align = kwargs.get("align", "left")
        self.anchor_x = kwargs.get("anchor_x", "left")
        self.anchor_y = kwargs.get("anchor_y", "baseline")
        self.rotation = kwargs.get("rotation", 0.0)

    def draw(self) -> None:
        return None


class FakeWindow:
    def __init__(self, width: int = 1440, height: int = 900) -> None:
        self.width = width
        self.height = height
        self.background_color = None
        self.shown_views: list[object] = []

    def get_size(self) -> tuple[int, int]:
        return self.width, self.height

    def show_view(self, view: object) -> None:
        self.shown_views.append(view)
        if hasattr(view, "window"):
            view.window = self


def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
    return FakeRect(left, bottom, width, height)


arcade.Rect = FakeRect
arcade.LBWH = fake_lbwh
arcade.View = FakeView
arcade.Text = FakeText
arcade.MOUSE_BUTTON_LEFT = 1
arcade.key = SimpleNamespace(ENTER=10, SPACE=32, N=78)
arcade.draw_lrbt_rectangle_filled = lambda *args, **kwargs: None
arcade.draw_circle_filled = lambda *args, **kwargs: None
arcade.draw_polygon_filled = lambda *args, **kwargs: None
arcade.draw_texture_rectangle = lambda *args, **kwargs: None

for optional_module in ("neat",):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

if "pymunk" not in sys.modules:
    pymunk = ModuleType("pymunk")
    sys.modules["pymunk"] = pymunk
else:
    pymunk = sys.modules["pymunk"]
if not hasattr(pymunk, "Space"):
    pymunk.Space = lambda: SimpleNamespace()
if not hasattr(pymunk, "Shape"):
    pymunk.Shape = object


from configs.sim_config import build_sim_config
from src.menu import StartMenuView


class StartMenuViewTest(unittest.TestCase):
    def make_view(self, width: int = 1440, height: int = 900) -> StartMenuView:
        view = StartMenuView(build_sim_config(), lambda: "simulation")
        view.window = FakeWindow(width, height)
        return view

    def test_layout_centers_text_stack_and_card_row(self) -> None:
        view = self.make_view()

        layout = view.layout()
        stack_center_y = (layout.left_card.bottom + layout.title_y) / 2.0

        self.assertEqual(layout.window.center_x, 720.0)
        self.assertEqual(layout.left_card.center_y, layout.right_card.center_y)
        self.assertAlmostEqual(
            (layout.left_card.left + layout.right_card.right) / 2.0,
            layout.window.center_x,
        )
        self.assertAlmostEqual(stack_center_y, layout.window.center_y, delta=30.0)
        self.assertEqual(layout.left_card.top, layout.cards_top)
        self.assertGreater(layout.title_y, layout.subtitle_y)
        self.assertGreater(layout.subtitle_y, layout.cards_top)

    def test_menu_uses_dark_stage_and_darker_disabled_card(self) -> None:
        view = self.make_view()

        self.assertLess(StartMenuView.BACKGROUND[0], 16)
        self.assertLess(StartMenuView.BACKGROUND[1], 48)
        self.assertLess(StartMenuView.BACKGROUND[2], 64)
        self.assertLess(sum(StartMenuView.CARD_FILL_DISABLED), sum(StartMenuView.CARD_FILL))
        self.assertNotEqual(StartMenuView.CARD_BORDER_DISABLED, StartMenuView.CARD_BORDER)
        self.assertGreater(sum(StartMenuView.TEXT_PRIMARY), sum(StartMenuView.TEXT_DISABLED))

    def test_card_content_blocks_have_padding_and_do_not_overlap(self) -> None:
        for width, height in ((1512, 982), (1440, 900), (800, 600)):
            with self.subTest(width=width, height=height):
                view = self.make_view(width, height)
                layout = view.layout()

                for card in (layout.left_card, layout.right_card):
                    content = view._card_content_layout(card)
                    badge_title_gap = content.badge.bottom - content.title_block.top
                    title_body_gap = content.title_block.bottom - content.body_block.top

                    self.assertLessEqual(content.badge.top, card.top)
                    self.assertGreaterEqual(content.badge.left, card.left)
                    self.assertLessEqual(content.badge.right, card.right)
                    self.assertGreater(badge_title_gap, 12.0)
                    self.assertGreater(title_body_gap, 10.0)
                    self.assertGreater(
                        content.title_block.height,
                        StartMenuView.CARD_TITLE_SIZE * 1.8,
                    )
                    self.assertGreaterEqual(content.badge.top, card.top - 44.0)
                    self.assertGreaterEqual(content.title_block.left, card.left)
                    self.assertLessEqual(content.title_block.right, card.right)
                    self.assertGreaterEqual(content.body_block.left, card.left)
                    self.assertLessEqual(content.body_block.right, card.right)
                    self.assertGreaterEqual(content.body_block.bottom, card.bottom + 8.0)

    def test_ribbon_label_sits_inside_red_triangle(self) -> None:
        view = self.make_view()
        layout = view.layout()

        ribbon = view._ribbon_layout(layout.right_card)
        left_top, right_top, right_bottom = ribbon.points
        width = right_top[0] - left_top[0]
        height = right_top[1] - right_bottom[1]
        dx = (ribbon.label_x - left_top[0]) / width
        dy = (right_top[1] - ribbon.label_y) / height

        self.assertGreaterEqual(dx, 0.0)
        self.assertLessEqual(dx, 1.0)
        self.assertGreaterEqual(dy, 0.0)
        self.assertLessEqual(dy, 1.0)
        self.assertLessEqual(dy, dx)
        self.assertLessEqual(ribbon.label_size, 7.0)

    def test_hover_only_tracks_icon_badges(self) -> None:
        view = self.make_view()
        layout = view.layout()
        start_badge = view._card_content_layout(layout.left_card).badge

        view.on_mouse_motion(
            int(layout.left_card.center_x),
            int(layout.left_card.bottom + 20),
            0,
            0,
        )
        self.assertIsNone(view._hovered_button)

        view.on_mouse_motion(
            int(start_badge.center_x),
            int(start_badge.center_y),
            0,
            0,
        )
        self.assertEqual(view._hovered_button, "start")

    def test_start_badge_click_shows_simulation_on_release(self) -> None:
        view = self.make_view()
        layout = view.layout()
        start_badge = view._card_content_layout(layout.left_card).badge

        pressed = view.on_mouse_press(
            int(start_badge.center_x),
            int(start_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )
        self.assertTrue(pressed)
        self.assertEqual(view._pressed_button, "start")
        self.assertEqual(view.window.shown_views, [])

        released = view.on_mouse_release(
            int(start_badge.center_x),
            int(start_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )

        self.assertTrue(released)
        self.assertIsNone(view._pressed_button)
        self.assertEqual(view.window.shown_views, ["simulation"])

    def test_load_badge_click_does_not_show_view(self) -> None:
        view = self.make_view()
        layout = view.layout()
        load_badge = view._card_content_layout(layout.right_card).badge

        pressed = view.on_mouse_press(
            int(load_badge.center_x),
            int(load_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )
        released = view.on_mouse_release(
            int(load_badge.center_x),
            int(load_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )

        self.assertTrue(pressed)
        self.assertTrue(released)
        self.assertEqual(view.window.shown_views, [])

    def test_hover_and_press_change_badge_visual_bounds(self) -> None:
        view = self.make_view()
        layout = view.layout()
        base_badge = view._card_content_layout(layout.left_card).badge

        view._button_animation["start"] = 1.0
        hovered = view._button_visual_bounds(base_badge, "start")
        self.assertGreater(hovered.width, base_badge.width)
        self.assertGreater(hovered.center_y, base_badge.center_y)

        view._pressed_button = "start"
        pressed = view._button_visual_bounds(base_badge, "start")
        self.assertLess(pressed.width, hovered.width)

    def test_start_keyboard_shortcuts_show_simulation_view(self) -> None:
        for key in (arcade.key.ENTER, arcade.key.SPACE, arcade.key.N):
            with self.subTest(key=key):
                view = self.make_view()

                handled = view.on_key_press(key, 0)

                self.assertTrue(handled)
                self.assertEqual(view.window.shown_views, ["simulation"])

    def test_menu_draw_loads_play_and_save_icons(self) -> None:
        calls: list[str] = []
        previous_load_texture = getattr(arcade, "load_texture", None)

        def fake_load_texture(path: str) -> object:
            calls.append(path)
            return "texture"

        arcade.load_texture = fake_load_texture
        try:
            self.make_view().on_draw()
        finally:
            if previous_load_texture is None:
                delattr(arcade, "load_texture")
            else:
                arcade.load_texture = previous_load_texture

        self.assertTrue(any(path.endswith("assets/play.png") for path in calls))
        self.assertTrue(any(path.endswith("assets/save.png") for path in calls))


class CreateAndRunMenuTest(unittest.TestCase):
    def test_create_and_run_shows_start_menu_initially(self) -> None:
        created_windows: list[FakeWindow] = []
        previous_window = getattr(arcade, "Window", None)
        previous_run = getattr(arcade, "run", None)

        def fake_window(*args: object, **kwargs: object) -> FakeWindow:
            del args, kwargs
            window = FakeWindow()
            created_windows.append(window)
            return window

        arcade.Window = fake_window
        arcade.run = lambda: None
        try:
            app = importlib.import_module("src.app")
            app.log_graphics_context = lambda: None
            app.create_and_run(build_sim_config())
        finally:
            if previous_window is None:
                delattr(arcade, "Window")
            else:
                arcade.Window = previous_window
            if previous_run is None:
                delattr(arcade, "run")
            else:
                arcade.run = previous_run

        self.assertEqual(len(created_windows), 1)
        self.assertEqual(len(created_windows[0].shown_views), 1)
        self.assertIsInstance(created_windows[0].shown_views[0], StartMenuView)


if __name__ == "__main__":
    unittest.main()
