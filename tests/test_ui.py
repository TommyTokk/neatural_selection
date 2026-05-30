from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest

try:
    import arcade
except ModuleNotFoundError:
    arcade = ModuleType("arcade")
    sys.modules["arcade"] = arcade

if not hasattr(arcade, "LBWH"):

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
            return self.left + self.width / 2

        @property
        def center_y(self) -> float:
            return self.bottom + self.height / 2

    def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
        return FakeRect(left, bottom, width, height)

    arcade.LBWH = fake_lbwh
    arcade.Rect = FakeRect

if not hasattr(arcade, "Text"):

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

        def draw(self) -> None:
            return None

    arcade.Text = FakeText

for draw_name in (
    "draw_lrbt_rectangle_filled",
    "draw_circle_filled",
    "draw_circle_outline",
    "draw_line",
    "draw_polygon_filled",
    "draw_texture_rectangle",
):
    if not hasattr(arcade, draw_name):
        setattr(arcade, draw_name, lambda *args, **kwargs: None)

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import build_sim_config
from src.layout import build_screen_layout
from src.ui import UiRenderer


class UiRendererBrainWindowScrollTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())
        self.renderer._brain_window_open = True
        self.renderer._brain_window_bounds = arcade.LBWH(100, 100, 400, 300)
        self.renderer._control_hitboxes["brain_window_graph"] = arcade.LBWH(
            120,
            150,
            360,
            180,
        )

    def test_scroll_over_graph_is_consumed_without_zooming_brain_graph(self) -> None:
        handled = self.renderer.handle_mouse_scroll(200, 200, 1)

        self.assertTrue(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_scroll_over_non_graph_brain_window_is_consumed(self) -> None:
        handled = self.renderer.handle_mouse_scroll(200, 380, 1)

        self.assertTrue(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_scroll_outside_ui_is_not_consumed(self) -> None:
        handled = self.renderer.handle_mouse_scroll(20, 20, 1)

        self.assertFalse(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_drag_inside_graph_is_consumed_without_moving_graph(self) -> None:
        world = SimpleNamespace()
        graph_bounds = self.renderer._control_hitboxes["brain_window_graph"]
        graph_position = (graph_bounds.center_x + 40, graph_bounds.center_y - 25)
        before = self.renderer._brain_graph_screen_position(
            graph_position,
            graph_bounds,
        )

        pressed = self.renderer.handle_mouse_press(world, 200, 200)
        dragged = self.renderer.handle_mouse_drag(world, 260, 240)
        after = self.renderer._brain_graph_screen_position(
            graph_position,
            graph_bounds,
        )

        self.assertTrue(pressed)
        self.assertFalse(dragged)
        self.assertEqual(after, before)

    def test_closed_brain_window_bounds_do_not_consume_scroll(self) -> None:
        self.renderer._brain_window_open = False
        self.renderer._control_hitboxes.clear()

        handled = self.renderer.handle_mouse_scroll(200, 380, 1)

        self.assertFalse(handled)

    def test_default_brain_window_uses_moderate_smaller_size(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(
                environment=arcade.LBWH(340, 20, 1000, 700),
                window=arcade.LBWH(0, 0, 1440, 900),
            )
        )

        self.renderer._brain_window_bounds = None
        self.renderer._ensure_brain_window_bounds(world)

        bounds = self.renderer._brain_window_bounds
        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertEqual(bounds.width, 620)
        self.assertEqual(bounds.height, 406)


class FloatingSimulationUiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())

    def make_world_shell(self) -> SimpleNamespace:
        return SimpleNamespace(
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            )
        )

    def make_inspector_world(
        self,
        *,
        energy: float = 0.4,
        max_energy: float = 2.0,
        vision_range: float = 160.0,
        panel_height: float = 414.0,
        selected: object | None = None,
    ) -> SimpleNamespace:
        creature = selected or SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=energy,
            speed=170.0,
            heading=1.25,
            vision=SimpleNamespace(range=vision_range, angle=2.17),
        )
        return SimpleNamespace(
            selected_creature=creature,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
            config=SimpleNamespace(
                metabolism=SimpleNamespace(max_energy=max_energy),
                fitness=SimpleNamespace(),
            ),
            sensor_snapshot_for=lambda selected_creature: SimpleNamespace(
                food=SimpleNamespace(visible=2.0, density=0.25),
                creatures=SimpleNamespace(visible=1.0, density=0.1),
            ),
            fitness_for=lambda selected_creature: None,
            neat_controller=SimpleNamespace(
                genome_id_for=lambda creature_id: creature_id,
            ),
            vision=SimpleNamespace(
                energy_cost_per_second=lambda selected_creature: 0.01,
            ),
        )

    def test_layout_uses_full_window_environment_and_slim_rail(self) -> None:
        config = build_sim_config()

        layout = build_screen_layout(1440, 900, config.layout)
        expected_rail_height = (
            UiRenderer.ICON_BUTTON_SIZE * 3
            + UiRenderer.ICON_BUTTON_GAP * 2
            + UiRenderer.RAIL_VERTICAL_PADDING
        )

        self.assertEqual(layout.environment.left, 0)
        self.assertEqual(layout.environment.bottom, 0)
        self.assertEqual(layout.environment.width, 1440)
        self.assertEqual(layout.environment.height, 900)
        self.assertEqual(layout.left_sidebar.width, config.layout.left_panel_width)
        self.assertLess(layout.left_sidebar.width, 120)
        self.assertEqual(layout.left_sidebar.height, expected_rail_height)
        self.assertEqual(
            layout.left_sidebar.center_y,
            layout.window.center_y,
        )

    def test_left_rail_stays_inside_small_window_padding(self) -> None:
        config = build_sim_config()

        layout = build_screen_layout(360, 300, config.layout)

        self.assertGreaterEqual(layout.left_sidebar.left, config.layout.outer_padding)
        self.assertGreaterEqual(layout.left_sidebar.bottom, config.layout.outer_padding)
        self.assertLessEqual(
            layout.left_sidebar.top,
            layout.window.top - config.layout.outer_padding,
        )

    def test_icon_helper_uses_loaded_asset_texture_path(self) -> None:
        calls = []
        previous_load_texture = getattr(arcade, "load_texture", None)
        previous_draw_texture_rectangle = getattr(
            arcade,
            "draw_texture_rectangle",
            None,
        )

        def fake_load_texture(path: str) -> object:
            calls.append(("load", path))
            return "texture"

        def fake_draw_texture_rectangle(*args: object) -> None:
            calls.append(("draw", args))

        arcade.load_texture = fake_load_texture
        arcade.draw_texture_rectangle = fake_draw_texture_rectangle
        try:
            self.renderer._draw_icon(
                arcade.LBWH(10, 20, 30, 40),
                "search",
                "test_search",
            )
        finally:
            if previous_load_texture is None:
                delattr(arcade, "load_texture")
            else:
                arcade.load_texture = previous_load_texture
            if previous_draw_texture_rectangle is None:
                delattr(arcade, "draw_texture_rectangle")
            else:
                arcade.draw_texture_rectangle = previous_draw_texture_rectangle

        self.assertEqual(calls[0][0], "load")
        self.assertTrue(str(calls[0][1]).endswith("assets/search.png"))
        self.assertEqual(calls[1], ("draw", (25.0, 40.0, 30, 40, "texture")))

    def test_panel_default_bounds_are_created_once_and_preserved(self) -> None:
        world = self.make_world_shell()

        first = self.renderer._stats_panel_bounds(world)
        self.renderer._panel_bounds["stats"] = arcade.LBWH(
            first.left - 80,
            first.bottom - 40,
            first.width,
            first.height,
        )
        second = self.renderer._stats_panel_bounds(world)

        self.assertNotEqual(second.left, first.left)
        self.assertNotEqual(second.bottom, first.bottom)

    def test_panel_icon_buttons_toggle_independently(self) -> None:
        self.renderer._control_hitboxes["panel_toggle_inspector"] = arcade.LBWH(0, 0, 20, 20)
        self.renderer._control_hitboxes["panel_toggle_stats"] = arcade.LBWH(30, 0, 20, 20)
        world = SimpleNamespace()

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(self.renderer.handle_mouse_press(world, 40, 10))

        self.assertTrue(self.renderer._panel_open["inspector"])
        self.assertTrue(self.renderer._panel_open["stats"])
        self.assertFalse(self.renderer._panel_open["settings"])

    def test_floating_panel_hitbox_consumes_click(self) -> None:
        self.renderer._control_hitboxes["stats_panel"] = arcade.LBWH(100, 100, 200, 120)

        handled = self.renderer.handle_mouse_press(SimpleNamespace(), 150, 150)

        self.assertTrue(handled)

    def test_dragging_inspector_updates_panel_bounds(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 300, 240)
        self.renderer._control_hitboxes["inspector_drag"] = arcade.LBWH(100, 300, 300, 40)

        pressed = self.renderer.handle_mouse_press(world, 130, 320)
        dragged = self.renderer.handle_mouse_drag(world, 190, 360)

        self.assertTrue(pressed)
        self.assertTrue(dragged)
        self.assertEqual(self.renderer._panel_bounds["inspector"].left, 160)
        self.assertEqual(self.renderer._panel_bounds["inspector"].bottom, 140)

    def test_dragging_stats_updates_panel_bounds(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["stats"] = arcade.LBWH(500, 500, 300, 200)
        self.renderer._control_hitboxes["stats_drag"] = arcade.LBWH(500, 660, 300, 40)

        self.assertTrue(self.renderer.handle_mouse_press(world, 520, 680))
        self.assertTrue(self.renderer.handle_mouse_drag(world, 470, 640))

        self.assertEqual(self.renderer._panel_bounds["stats"].left, 450)
        self.assertEqual(self.renderer._panel_bounds["stats"].bottom, 460)

    def test_dragging_settings_updates_panel_bounds(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["settings"] = arcade.LBWH(420, 40, 500, 148)
        self.renderer._control_hitboxes["settings_drag"] = arcade.LBWH(420, 144, 500, 44)

        self.assertTrue(self.renderer.handle_mouse_press(world, 500, 166))
        self.assertTrue(self.renderer.handle_mouse_drag(world, 540, 206))

        self.assertEqual(self.renderer._panel_bounds["settings"].left, 460)
        self.assertEqual(self.renderer._panel_bounds["settings"].bottom, 80)

    def test_panel_controls_do_not_start_dragging(self) -> None:
        world = SimpleNamespace(toggle_pause=lambda: None)
        self.renderer._panel_bounds["settings"] = arcade.LBWH(420, 40, 500, 148)
        self.renderer._control_hitboxes["settings_drag"] = arcade.LBWH(420, 144, 500, 44)
        self.renderer._control_hitboxes["pause"] = arcade.LBWH(480, 140, 50, 50)

        handled = self.renderer.handle_mouse_press(world, 500, 166)

        self.assertTrue(handled)
        self.assertIsNone(self.renderer._active_panel_drag)

    def test_panel_close_does_not_start_dragging(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_open["stats"] = True
        self.renderer._panel_bounds["stats"] = arcade.LBWH(500, 500, 300, 200)
        self.renderer._control_hitboxes["stats_drag"] = arcade.LBWH(500, 660, 300, 40)
        self.renderer._control_hitboxes["stats_close"] = arcade.LBWH(760, 660, 28, 28)

        handled = self.renderer.handle_mouse_press(world, 770, 670)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._panel_open["stats"])
        self.assertIsNone(self.renderer._active_panel_drag)

    def test_dragged_panel_clamps_to_window(self) -> None:
        world = self.make_world_shell()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 300, 240)
        self.renderer._control_hitboxes["inspector_drag"] = arcade.LBWH(100, 300, 300, 40)

        self.assertTrue(self.renderer.handle_mouse_press(world, 130, 320))
        self.assertTrue(self.renderer.handle_mouse_drag(world, -500, 2000))

        bounds = self.renderer._panel_bounds["inspector"]
        self.assertEqual(bounds.left, self.renderer.config.layout.outer_padding)
        self.assertEqual(
            bounds.bottom,
            world.layout.window.height
            - self.renderer.config.layout.outer_padding
            - bounds.height,
        )

    def test_inspector_energy_ratio_uses_creature_energy_not_vision(self) -> None:
        world = self.make_inspector_world(energy=0.4, max_energy=2.0, vision_range=999.0)

        self.assertEqual(self.renderer._inspector_energy_ratio(world), 0.2)

    def test_inspector_energy_ratio_clamps(self) -> None:
        high = self.make_inspector_world(energy=3.0, max_energy=2.0)
        low = self.make_inspector_world(energy=-0.25, max_energy=2.0)

        self.assertEqual(self.renderer._inspector_energy_ratio(high), 1.0)
        self.assertEqual(self.renderer._inspector_energy_ratio(low), 0.0)

    def test_inspector_draw_registers_scroll_region_and_action_hitboxes(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)

        self.renderer._draw_inspector_panel(world)

        self.assertIn("inspector", self.renderer._scroll_regions)
        self.assertGreater(self.renderer._scroll_limits["inspector"], 0)
        self.renderer._scroll_offsets["inspector"] = self.renderer._scroll_limits["inspector"]
        self.renderer._draw_inspector_panel(world)

        self.assertIn("open_brain_window", self.renderer._control_hitboxes)
        self.assertIn("kill_selected_creature", self.renderer._control_hitboxes)

    def test_inspector_scroll_region_uses_inner_card_viewport(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)

        self.renderer._draw_inspector_panel(world)

        body = self.renderer._control_hitboxes["inspector_body"]
        scroll_region = self.renderer._scroll_regions["inspector"]
        self.assertGreater(scroll_region.left, body.left)
        self.assertGreater(scroll_region.bottom, body.bottom)
        self.assertLess(scroll_region.width, body.width)
        self.assertLess(scroll_region.height, body.height)

    def test_inspector_content_uses_clip_for_inner_viewport(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)
        clips = []
        original_ui_clip = self.renderer._ui_clip

        @contextmanager
        def capture_clip(bounds: object):
            clips.append(bounds)
            yield

        self.renderer._ui_clip = capture_clip
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            self.renderer._ui_clip = original_ui_clip

        self.assertEqual(clips, [self.renderer._scroll_regions["inspector"]])

    def test_inspector_top_scroll_does_not_leave_action_hitboxes(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)

        self.renderer._draw_inspector_panel(world)

        self.assertNotIn("open_brain_window", self.renderer._control_hitboxes)
        self.assertNotIn("kill_selected_creature", self.renderer._control_hitboxes)

    def test_inspector_progress_bar_uses_energy_ratio(self) -> None:
        world = self.make_inspector_world(energy=0.5, max_energy=2.0, vision_range=160.0)
        ratios = []
        original_draw_progress_bar = self.renderer._draw_progress_bar

        def capture_progress_bar(bounds: object, ratio: float, **kwargs: object) -> None:
            del bounds, kwargs
            ratios.append(ratio)

        self.renderer._draw_progress_bar = capture_progress_bar
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            self.renderer._draw_progress_bar = original_draw_progress_bar

        self.assertEqual(ratios, [0.25])

    def test_progress_bar_zero_ratio_skips_fill(self) -> None:
        fills = []
        original_fill = self.renderer._draw_rounded_rect_fill

        def capture_fill(bounds: object, color: object, radius: float) -> None:
            del color, radius
            fills.append(bounds)

        self.renderer._draw_rounded_rect_fill = capture_fill
        try:
            self.renderer._draw_progress_bar(arcade.LBWH(10, 20, 100, 8), 0.0)
        finally:
            self.renderer._draw_rounded_rect_fill = original_fill

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].width, 100)

    def test_progress_bar_tiny_ratio_does_not_raise(self) -> None:
        self.renderer._draw_progress_bar(arcade.LBWH(10, 20, 100, 8), 1e-9)
        self.renderer._draw_progress_bar(arcade.LBWH(10, 20, 100, 8), 1e-5)

    def test_rounded_rect_fill_tolerates_zero_and_tiny_widths(self) -> None:
        self.renderer._draw_rounded_rect_fill(arcade.LBWH(10, 20, 0, 8), (1, 2, 3), 4)
        self.renderer._draw_rounded_rect_fill(
            arcade.LBWH(10, 20, 0.0000000001, 8),
            (1, 2, 3),
            4,
        )

    def test_inspector_zero_energy_render_does_not_raise(self) -> None:
        world = self.make_inspector_world(energy=0.0, max_energy=1.0)

        self.renderer._draw_inspector_panel(world)

    def test_inspector_empty_state_draws_without_selection(self) -> None:
        labels = []
        original_draw_text = self.renderer._draw_text
        world = self.make_inspector_world(selected=None)
        world.selected_creature = None

        def capture_text(key: str, text: str, *args: object, **kwargs: object) -> None:
            del key, args, kwargs
            labels.append(text)

        self.renderer._draw_text = capture_text
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            self.renderer._draw_text = original_draw_text

        self.assertIn("No creature selected", labels)

    def test_action_button_icon_and_text_have_padding_and_centered_group(self) -> None:
        icon_bounds = []
        texts = []
        original_draw_icon = self.renderer._draw_icon
        original_draw_text = self.renderer._draw_text

        def capture_icon(bounds: object, icon_name: str, key: str) -> None:
            del icon_name, key
            icon_bounds.append(bounds)

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del key, args, kwargs
            texts.append((text, x, y))

        self.renderer._draw_icon = capture_icon
        self.renderer._draw_text = capture_text
        try:
            button = arcade.LBWH(100, 200, 150, 36)
            self.renderer._draw_action_button(
                button,
                "Open Brain",
                "brain",
                "open_brain_window",
                fill_color=(1, 2, 3),
                text_color=(4, 5, 6),
            )
        finally:
            self.renderer._draw_icon = original_draw_icon
            self.renderer._draw_text = original_draw_text

        icon = icon_bounds[0]
        text, text_x, text_y = texts[-1]
        text_width = len(text) * 7.0
        group_left = icon.left
        group_right = text_x + text_width
        self.assertGreaterEqual(text_x - icon.right, 12.0)
        self.assertAlmostEqual((group_left + group_right) / 2.0, button.center_x)
        self.assertEqual(text_y, button.center_y)

    def test_open_brain_button_requires_selected_creature(self) -> None:
        self.renderer._control_hitboxes["open_brain_window"] = arcade.LBWH(0, 0, 40, 20)
        world = SimpleNamespace(selected_creature=None)

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._brain_window_open)

    def test_open_brain_button_opens_when_creature_selected(self) -> None:
        self.renderer._control_hitboxes["open_brain_window"] = arcade.LBWH(0, 0, 40, 20)
        world = SimpleNamespace(
            selected_creature=object(),
            layout=SimpleNamespace(
                environment=arcade.LBWH(0, 0, 1440, 900),
                window=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertTrue(self.renderer._brain_window_open)

    def test_kill_button_calls_world_and_closes_brain_window(self) -> None:
        calls = []
        self.renderer._brain_window_open = True
        self.renderer._control_hitboxes["kill_selected_creature"] = arcade.LBWH(0, 0, 40, 20)
        world = SimpleNamespace(
            kill_selected_creature=lambda: calls.append("kill") or True
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertEqual(calls, ["kill"])
        self.assertFalse(self.renderer._brain_window_open)

    def test_speed_buttons_preserve_world_speed_api(self) -> None:
        calls = []
        self.renderer._control_hitboxes["speed_down"] = arcade.LBWH(0, 0, 20, 20)
        self.renderer._control_hitboxes["speed_up"] = arcade.LBWH(30, 0, 20, 20)
        self.renderer._control_hitboxes["reset_speed"] = arcade.LBWH(60, 0, 20, 20)
        world = SimpleNamespace(
            decrease_simulation_speed=lambda: calls.append("down"),
            increase_simulation_speed=lambda: calls.append("up"),
            reset_simulation_speed=lambda: calls.append("reset"),
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(self.renderer.handle_mouse_press(world, 40, 10))
        self.assertTrue(self.renderer.handle_mouse_press(world, 70, 10))

        self.assertEqual(calls, ["down", "up", "reset"])

    def test_speed_min_button_sets_minimum_speed(self) -> None:
        calls = []
        self.renderer._control_hitboxes["speed_min"] = arcade.LBWH(0, 0, 20, 20)
        world = SimpleNamespace(
            MIN_SIMULATION_SPEED=0.25,
            set_simulation_speed=lambda speed: calls.append(speed),
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))

        self.assertEqual(calls, [0.25])

    def test_speed_max_button_sets_maximum_speed(self) -> None:
        calls = []
        self.renderer._control_hitboxes["speed_max"] = arcade.LBWH(0, 0, 20, 20)
        world = SimpleNamespace(
            MAX_SIMULATION_SPEED=5.0,
            set_simulation_speed=lambda speed: calls.append(speed),
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))

        self.assertEqual(calls, [5.0])

    def test_settings_panel_draws_all_speed_controls_inside_panel(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        panel = self.renderer._control_hitboxes["settings_panel"]
        for key in (
            "speed_slider",
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        ):
            bounds = self.renderer._control_hitboxes[key]
            self.assertGreaterEqual(bounds.left, panel.left)
            self.assertLessEqual(bounds.right, panel.right)
        self.assertNotIn("reset_speed", self.renderer._control_hitboxes)

    def test_settings_speed_control_order_places_pause_between_arrows(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        ordered_keys = (
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        )
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            self.assertLess(
                self.renderer._control_hitboxes[left_key].left,
                self.renderer._control_hitboxes[right_key].left,
            )

    def test_settings_speed_controls_are_compact_and_padded(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        self.renderer._draw_settings_panel(world)

        slider = self.renderer._control_hitboxes["speed_slider"]
        self.assertGreater(slider.width, 72.0)

        ordered_keys = (
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        )
        previous = slider
        for key in ordered_keys:
            bounds = self.renderer._control_hitboxes[key]
            if key == "pause":
                self.assertGreater(bounds.width, 32.0)
                self.assertGreater(bounds.height, 32.0)
            else:
                self.assertLess(bounds.width, 32.0)
                self.assertLess(bounds.height, 32.0)
            self.assertGreaterEqual(bounds.left - previous.right, 8.0)
            previous = bounds

    def test_settings_speed_controls_do_not_overlap_at_minimum_panel_width(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 520, 720),
                environment=arcade.LBWH(0, 0, 520, 720),
            ),
        )

        self.renderer._draw_settings_panel(world)

        ordered_keys = (
            "speed_slider",
            "speed_min",
            "speed_down",
            "pause",
            "speed_up",
            "speed_max",
        )
        for left_key, right_key in zip(ordered_keys, ordered_keys[1:]):
            left_bounds = self.renderer._control_hitboxes[left_key]
            right_bounds = self.renderer._control_hitboxes[right_key]
            self.assertGreaterEqual(right_bounds.left - left_bounds.right, 8.0)

    def test_settings_panel_removes_stale_reset_speed_hitbox(self) -> None:
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )
        self.renderer._control_hitboxes["reset_speed"] = arcade.LBWH(0, 0, 20, 20)

        self.renderer._draw_settings_panel(world)

        self.assertNotIn("reset_speed", self.renderer._control_hitboxes)

    def test_pause_button_text_is_drawn_at_button_center(self) -> None:
        texts = []
        original_draw_text = self.renderer._draw_text

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del text, args, kwargs
            texts.append((key, x, y))

        self.renderer._draw_text = capture_text
        try:
            pause_button = arcade.LBWH(100, 200, 50, 50)
            self.renderer._draw_icon_text_button(
                pause_button,
                "||",
                "pause",
                fill_color=(1, 2, 3),
                size=19,
            )
        finally:
            self.renderer._draw_text = original_draw_text

        _, x, y = texts[-1]
        self.assertEqual(x, pause_button.center_x)
        self.assertEqual(y, pause_button.center_y)

    def test_pause_state_bars_are_centered_in_button(self) -> None:
        fills = []
        original_fill = self.renderer._draw_rounded_rect_fill

        def capture_fill(bounds: object, color: object, radius: float) -> None:
            del color, radius
            fills.append(bounds)

        self.renderer._draw_rounded_rect_fill = capture_fill
        try:
            button = arcade.LBWH(100, 200, 50, 50)
            self.renderer._draw_play_pause_button(button, is_paused=False)
        finally:
            self.renderer._draw_rounded_rect_fill = original_fill

        bars = fills[-2:]
        center = (bars[0].center_x + bars[1].center_x) / 2.0
        self.assertEqual(center, button.center_x)
        self.assertEqual(bars[0].center_y, button.center_y)
        self.assertEqual(bars[1].center_y, button.center_y)
        self.assertLess(bars[0].height, 22)

    def test_play_state_icon_is_centered_in_button(self) -> None:
        polygons = []
        previous_draw_polygon = getattr(arcade, "draw_polygon_filled", None)

        def capture_polygon(points: object, color: object) -> None:
            del color
            polygons.append(points)

        arcade.draw_polygon_filled = capture_polygon
        try:
            button = arcade.LBWH(100, 200, 50, 50)
            self.renderer._draw_play_pause_button(button, is_paused=True)
        finally:
            if previous_draw_polygon is None:
                delattr(arcade, "draw_polygon_filled")
            else:
                arcade.draw_polygon_filled = previous_draw_polygon

        points = polygons[0]
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        self.assertAlmostEqual(center_x, button.center_x - 2 / 3)
        self.assertEqual(center_y, button.center_y)

    def test_keycap_label_has_padding_after_keycap(self) -> None:
        texts = []
        original_draw_text = self.renderer._draw_text

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del text, y, args, kwargs
            texts.append((key, x))

        self.renderer._draw_text = capture_text
        try:
            self.renderer._draw_keycap("settings_hint_test", "SPACE", "PLAY/PAUSE", 100, 50)
        finally:
            self.renderer._draw_text = original_draw_text

        key_x = next(x for key, x in texts if key == "settings_hint_test_key")
        label_x = next(x for key, x in texts if key == "settings_hint_test_label")
        self.assertEqual(label_x - key_x, 33)

    def test_keyboard_hints_have_space_after_play_pause_label(self) -> None:
        texts = []
        original_draw_text = self.renderer._draw_text
        world = SimpleNamespace(
            is_paused=False,
            simulation_speed=1.0,
            MIN_SIMULATION_SPEED=0.25,
            MAX_SIMULATION_SPEED=5.0,
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, 1440, 900),
                environment=arcade.LBWH(0, 0, 1440, 900),
            ),
        )

        def capture_text(
            key: str,
            text: str,
            x: float,
            y: float,
            *args: object,
            **kwargs: object,
        ) -> None:
            del y, args, kwargs
            texts.append((key, text, x))

        self.renderer._draw_text = capture_text
        try:
            self.renderer._draw_settings_panel(world)
        finally:
            self.renderer._draw_text = original_draw_text

        play_pause = next(
            item for item in texts if item[0] == "settings_hint_0_label"
        )
        a_key = next(item for item in texts if item[0] == "settings_hint_1_key")
        play_pause_right = play_pause[2] + len(play_pause[1]) * 7.0
        self.assertGreaterEqual(a_key[2] - play_pause_right, 20.0)


if __name__ == "__main__":
    unittest.main()
