from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
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

    def on_key_release(self, symbol: int, modifiers: int) -> bool | None:
        del symbol, modifiers
        return None

    def on_mouse_scroll(
        self,
        x: int,
        y: int,
        scroll_x: int,
        scroll_y: int,
    ) -> bool | None:
        del x, y, scroll_x, scroll_y
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
arcade.key = SimpleNamespace(
    ENTER=10,
    SPACE=32,
    N=78,
    LCOMMAND=65517,
    RCOMMAND=65518,
)
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
import src.menu as menu_module
from src.menu import StartMenuView


class _FakeNSString:
    def __init__(self, value: str) -> None:
        self.value = value
        self.ptr = value


class _FakeArray(list[object]):
    def addObject_(self, value: object) -> None:
        self.append(value)


class _FakeURLs:
    def __init__(self, selected_path: str | None) -> None:
        self.selected_path = selected_path

    def count(self) -> int:
        return 0 if self.selected_path is None else 1

    def objectAtIndex_(self, index: int) -> object:
        self.assert_valid_index(index)
        selected_path = self.selected_path
        assert selected_path is not None
        return SimpleNamespace(
            path=lambda: _FakeNSString(selected_path),
        )

    def assert_valid_index(self, index: int) -> None:
        if index != 0 or self.selected_path is None:
            raise IndexError(index)


class _FakeOpenPanel:
    def __init__(self, selected_path: str | None, response: int = 1) -> None:
        self.selected_path = selected_path
        self.response = response
        self.can_choose_files = None
        self.can_choose_directories = None
        self.allows_multiple_selection = None
        self.resolves_aliases = None
        self.title = None
        self.directory_url = None
        self.allowed_types = None

    def setCanChooseFiles_(self, value: bool) -> None:
        self.can_choose_files = value

    def setCanChooseDirectories_(self, value: bool) -> None:
        self.can_choose_directories = value

    def setAllowsMultipleSelection_(self, value: bool) -> None:
        self.allows_multiple_selection = value

    def setResolvesAliases_(self, value: bool) -> None:
        self.resolves_aliases = value

    def setTitle_(self, value: object) -> None:
        self.title = value

    def setDirectoryURL_(self, value: object) -> None:
        self.directory_url = value

    def setAllowedFileTypes_(self, value: object) -> None:
        self.allowed_types = value

    def runModal(self) -> int:
        return self.response

    def URLs(self) -> _FakeURLs:
        return _FakeURLs(self.selected_path)


def _fake_cocoa_api(
    panel: _FakeOpenPanel,
) -> tuple[object, object, object]:
    class FakeNSOpenPanel:
        @staticmethod
        def openPanel() -> _FakeOpenPanel:
            return panel

    class FakeNSURL:
        @staticmethod
        def fileURLWithPath_(value: _FakeNSString) -> tuple[str, str]:
            return "url", value.value

    class FakeNSMutableArray:
        @staticmethod
        def array() -> _FakeArray:
            return _FakeArray()

    classes = {
        "NSOpenPanel": FakeNSOpenPanel,
        "NSURL": FakeNSURL,
        "NSMutableArray": FakeNSMutableArray,
    }
    return (
        lambda name: classes[name],
        _FakeNSString,
        lambda pointer: pointer,
    )


class CheckpointFilePickerTest(unittest.TestCase):
    def test_macos_picker_configures_panel_and_returns_checkpoint(self) -> None:
        selected = "/tmp/simulation/hourly/checkpoint_1.pkl"
        panel = _FakeOpenPanel(selected)

        with patch.object(
            menu_module,
            "_cocoa_api",
            return_value=_fake_cocoa_api(panel),
        ):
            checkpoint = menu_module._select_checkpoint_file_macos(Path("."))

        self.assertEqual(checkpoint, Path(selected))
        self.assertTrue(panel.can_choose_files)
        self.assertFalse(panel.can_choose_directories)
        self.assertFalse(panel.allows_multiple_selection)
        self.assertTrue(panel.resolves_aliases)
        self.assertEqual(
            panel.directory_url,
            ("url", str(Path(".").resolve())),
        )
        self.assertEqual(
            [item.value for item in panel.allowed_types],
            ["pkl", "bak"],
        )

    def test_macos_picker_returns_none_when_cancelled(self) -> None:
        panel = _FakeOpenPanel(None, response=0)

        with patch.object(
            menu_module,
            "_cocoa_api",
            return_value=_fake_cocoa_api(panel),
        ):
            checkpoint = menu_module._select_checkpoint_file_macos(Path("."))

        self.assertIsNone(checkpoint)

    def test_macos_picker_rejects_unrelated_backup_file(self) -> None:
        panel = _FakeOpenPanel("/tmp/notes.bak")

        with patch.object(
            menu_module,
            "_cocoa_api",
            return_value=_fake_cocoa_api(panel),
        ):
            with self.assertRaisesRegex(ValueError, "pkl"):
                menu_module._select_checkpoint_file_macos(Path("."))

    def test_darwin_dispatch_never_imports_tkinter(self) -> None:
        panel = _FakeOpenPanel("/tmp/checkpoint.pkl")
        real_import = __import__

        def guarded_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("tkinter"):
                self.fail("Tkinter must not be imported on macOS.")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(menu_module.sys, "platform", "darwin"),
            patch.object(
                menu_module,
                "_cocoa_api",
                return_value=_fake_cocoa_api(panel),
            ),
            patch("builtins.__import__", side_effect=guarded_import),
        ):
            checkpoint = menu_module.select_checkpoint_file(Path("."))

        self.assertEqual(checkpoint, Path("/tmp/checkpoint.pkl"))


class StartMenuViewTest(unittest.TestCase):
    def make_view(
        self,
        width: int = 1440,
        height: int = 900,
        *,
        file_picker: object | None = None,
        load_view_factory: object | None = None,
    ) -> StartMenuView:
        picker = file_picker or (lambda initial_directory: None)
        loader = load_view_factory or (lambda checkpoint: "loaded")
        view = StartMenuView(
            build_sim_config(),
            lambda: "simulation",
            loader,
            file_picker=picker,
        )
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

    def test_load_card_is_drawn_enabled_without_coming_soon_ribbon(self) -> None:
        view = self.make_view()
        cards: list[dict[str, object]] = []
        original_draw_card = view._draw_card

        def capture_card(*args: object, **kwargs: object) -> None:
            del args
            cards.append(kwargs)

        view._draw_card = capture_card
        try:
            view.on_draw()
        finally:
            view._draw_card = original_draw_card

        load_card = next(card for card in cards if card["key"] == "load")
        self.assertFalse(load_card["disabled"])

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

    def test_load_badge_click_opens_picker_and_shows_loaded_view(self) -> None:
        selected = Path("saves/simulation_1/checkpoint.pkl")
        picker_directories: list[Path] = []
        loaded_paths: list[Path] = []
        resize_calls: list[tuple[int, int]] = []
        loaded_view = SimpleNamespace(
            on_resize=lambda width, height: resize_calls.append((width, height))
        )

        def picker(initial_directory: Path) -> Path:
            picker_directories.append(initial_directory)
            return selected

        def loader(checkpoint: Path) -> object:
            loaded_paths.append(checkpoint)
            return loaded_view

        view = self.make_view(
            file_picker=picker,
            load_view_factory=loader,
        )
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
        self.assertEqual(picker_directories, [Path("saves")])
        self.assertEqual(loaded_paths, [selected])
        self.assertEqual(view.window.shown_views, [loaded_view])
        self.assertEqual(resize_calls, [(1440, 900)])

    def test_load_picker_cancel_keeps_menu_visible(self) -> None:
        view = self.make_view(file_picker=lambda initial_directory: None)
        layout = view.layout()
        load_badge = view._card_content_layout(layout.right_card).badge

        view.on_mouse_press(
            int(load_badge.center_x),
            int(load_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )
        view.on_mouse_release(
            int(load_badge.center_x),
            int(load_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )

        self.assertEqual(view.window.shown_views, [])
        self.assertIsNone(view._load_error)

    def test_load_error_stays_on_menu_and_is_drawn_in_card(self) -> None:
        def fail(checkpoint: Path) -> object:
            raise ValueError(f"bad checkpoint {checkpoint}")

        view = self.make_view(
            file_picker=lambda initial_directory: Path("broken.pkl"),
            load_view_factory=fail,
        )
        layout = view.layout()
        load_badge = view._card_content_layout(layout.right_card).badge
        view.on_mouse_press(
            int(load_badge.center_x),
            int(load_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )
        view.on_mouse_release(
            int(load_badge.center_x),
            int(load_badge.center_y),
            arcade.MOUSE_BUTTON_LEFT,
            0,
        )

        self.assertEqual(view.window.shown_views, [])
        self.assertIn("bad checkpoint", view._load_error)

        cards: list[dict[str, object]] = []
        original_draw_card = view._draw_card
        view._draw_card = lambda *args, **kwargs: cards.append(kwargs)
        try:
            view.on_draw()
        finally:
            view._draw_card = original_draw_card
        load_card = next(card for card in cards if card["key"] == "load")
        self.assertIn("bad checkpoint", load_card["description"])
        self.assertEqual(load_card["description_color"], view.ERROR_TEXT)

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
    def _game_view_with_ui(self, ui_renderer: object) -> object:
        app = importlib.import_module("src.app")
        config = build_sim_config()
        restored_world = SimpleNamespace(config=config)
        original_environment_renderer = app.EnvironmentRenderer
        original_ui_renderer = app.UiRenderer
        app.EnvironmentRenderer = lambda config: "environment-renderer"
        app.UiRenderer = lambda config: ui_renderer
        try:
            return app.NeatGameView(world=restored_world)
        finally:
            app.EnvironmentRenderer = original_environment_renderer
            app.UiRenderer = original_ui_renderer

    def test_game_view_tracks_left_and_right_command_independently(self) -> None:
        ui = SimpleNamespace(
            handle_key_press=lambda world, symbol, modifiers: True,
        )
        view = self._game_view_with_ui(ui)

        view.on_key_press(arcade.key.LCOMMAND, 0)
        view.on_key_press(arcade.key.RCOMMAND, 0)
        self.assertEqual(
            view._command_keys_down,
            {arcade.key.LCOMMAND, arcade.key.RCOMMAND},
        )
        view.on_key_release(arcade.key.LCOMMAND, 0)
        self.assertEqual(view._command_keys_down, {arcade.key.RCOMMAND})
        view.on_key_release(arcade.key.RCOMMAND, 0)
        self.assertEqual(view._command_keys_down, set())

    def test_game_view_forwards_command_state_with_scroll(self) -> None:
        calls: list[tuple[object, ...]] = []
        ui = SimpleNamespace(
            handle_mouse_scroll=lambda *args: calls.append(args) or True,
        )
        view = self._game_view_with_ui(ui)
        view._command_keys_down.add(arcade.key.LCOMMAND)

        view.on_mouse_scroll(120, 240, 0, 1)

        self.assertEqual(calls, [(120, 240, 1, 0, True)])

    def test_game_view_uses_supplied_world_without_creating_fresh_world(
        self,
    ) -> None:
        app = importlib.import_module("src.app")
        config = build_sim_config()
        restored_world = SimpleNamespace(config=config)
        original_world = app.World
        original_environment_renderer = app.EnvironmentRenderer
        original_ui_renderer = app.UiRenderer
        app.World = lambda config: self.fail("Fresh World must not be created")
        app.EnvironmentRenderer = lambda config: "environment-renderer"
        app.UiRenderer = lambda config: "ui-renderer"
        try:
            view = app.NeatGameView(world=restored_world)
        finally:
            app.World = original_world
            app.EnvironmentRenderer = original_environment_renderer
            app.UiRenderer = original_ui_renderer

        self.assertIs(view.world, restored_world)
        self.assertIs(view.config, config)

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
