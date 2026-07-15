from __future__ import annotations

from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, replace
from time import perf_counter
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, Mock, patch

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
    setattr(arcade, draw_name, lambda *args, **kwargs: None)

for optional_module in ("neat", "pymunk"):
    try:
        __import__(optional_module)
    except ModuleNotFoundError:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import build_sim_config
from src.analysis import generate_inspector_report
from src.creature import LineageInfo, PhysicalTraits, TraitMutationDelta
from src.layout import build_screen_layout
from src.speciation import (
    NeatChangeSummary,
    SpeciesDistanceBreakdown,
    SpeciesRecord,
    SpeciesTraitSnapshot,
)
from src.ui import UiRenderer
from src.vision import SENSOR_INPUT_NAMES


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
                stomach_fullness=0.6,
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
            UiRenderer.ICON_BUTTON_SIZE * 4
            + UiRenderer.ICON_BUTTON_GAP * 3
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

    def test_left_rail_registers_map_submenu_button(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            environment_map_mode="none",
        )

        self.renderer._draw_icon_rail(world)

        self.assertIn("open_map_submenu", self.renderer._control_hitboxes)

    def test_left_rail_registers_save_button_with_save_sim_icon(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=False,
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn("save_simulation", self.renderer._control_hitboxes)
        self.assertIn(("save_simulation", "save_sim", False), calls)
        self.assertTrue(self.renderer._icon_path("save_sim").is_file())

    def test_save_button_active_state_follows_world(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=True,
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn(("save_simulation", "save_sim", True), calls)

    def test_save_button_calls_world_save_now(self) -> None:
        calls: list[str] = []
        world = SimpleNamespace(save_now=lambda: calls.append("save"))
        self.renderer._control_hitboxes["save_simulation"] = arcade.LBWH(
            0,
            0,
            20,
            20,
        )

        handled = self.renderer.handle_mouse_press(world, 10, 10)

        self.assertTrue(handled)
        self.assertEqual(calls, ["save"])

    def test_six_left_rail_buttons_fit_without_overlap(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(800, 600, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=False,
        )

        self.renderer._draw_icon_rail(world)

        keys = (
            "panel_toggle_inspector",
            "panel_toggle_stats",
            "panel_toggle_settings",
            "open_map_submenu",
            "save_simulation",
            "open_species_tree",
        )
        buttons = [self.renderer._control_hitboxes[key] for key in keys]
        rail = world.layout.left_sidebar
        for button in buttons:
            self.assertGreaterEqual(button.bottom, rail.bottom)
            self.assertLessEqual(button.top, rail.top)
        for upper, lower in zip(buttons, buttons[1:]):
            self.assertGreaterEqual(upper.bottom, lower.top)

    def test_left_rail_uses_supplied_speciation_icon(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            show_biome_background=False,
            save_in_progress=False,
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn(("open_species_tree", "species", False), calls)
        self.assertTrue(self.renderer._icon_path("species").is_file())

    def test_globe_button_opens_and_closes_map_submenu(self) -> None:
        world = SimpleNamespace()
        self.renderer._control_hitboxes["open_map_submenu"] = arcade.LBWH(
            0,
            0,
            20,
            20,
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(self.renderer._map_submenu_open)
        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertFalse(self.renderer._map_submenu_open)

    def test_globe_button_active_state_follows_selected_map(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            environment_map_mode="pheromones",
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )

        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button

        self.assertIn(("open_map_submenu", "globe", True), calls)

    def test_globe_button_is_active_while_map_submenu_is_open(self) -> None:
        world = SimpleNamespace(
            layout=build_screen_layout(1440, 900, build_sim_config().layout),
            environment_map_mode="none",
        )
        calls: list[tuple[str, str, bool]] = []
        original_draw_icon_button = self.renderer._draw_icon_button
        original_draw_map_submenu = self.renderer._draw_map_submenu
        self.renderer._map_submenu_open = True
        self.renderer._draw_icon_button = (
            lambda bounds, icon_name, key, active: calls.append(
                (key, icon_name, active)
            )
        )
        self.renderer._draw_map_submenu = lambda active_world, anchor: None
        try:
            self.renderer._draw_icon_rail(world)
        finally:
            self.renderer._draw_icon_button = original_draw_icon_button
            self.renderer._draw_map_submenu = original_draw_map_submenu

        self.assertIn(("open_map_submenu", "globe", True), calls)

    def test_map_submenu_uses_supplied_icons_and_stays_inside_window(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(window=arcade.LBWH(0, 0, 240, 180)),
            environment_map_mode="none",
        )
        icons: list[tuple[str, str]] = []
        original_draw_icon = self.renderer._draw_icon
        self.renderer._draw_icon = (
            lambda bounds, icon_name, key: icons.append((key, icon_name))
        )
        try:
            self.renderer._draw_map_submenu(
                world,
                arcade.LBWH(8, 8, 58, 58),
            )
        finally:
            self.renderer._draw_icon = original_draw_icon

        card = self.renderer._control_hitboxes["map_submenu"]
        self.assertGreaterEqual(card.left, world.layout.window.left)
        self.assertLessEqual(card.right, world.layout.window.right)
        self.assertGreaterEqual(card.bottom, world.layout.window.bottom)
        self.assertLessEqual(card.top, world.layout.window.top)
        self.assertIn(("map_layer_biome", "biome_map"), icons)
        self.assertIn(("map_layer_pheromones", "pheromone_map"), icons)
        self.assertTrue(self.renderer._icon_path("biome_map").is_file())
        self.assertTrue(self.renderer._icon_path("pheromone_map").is_file())

    def test_map_submenu_highlights_only_active_layer(self) -> None:
        world = SimpleNamespace(
            layout=SimpleNamespace(window=arcade.LBWH(0, 0, 800, 600)),
            environment_map_mode="biome",
        )
        fills: list[tuple[object, object]] = []
        original_draw_rounded_rect = self.renderer._draw_rounded_rect
        original_draw_icon = self.renderer._draw_icon
        original_draw_text = self.renderer._draw_text
        self.renderer._draw_rounded_rect = (
            lambda bounds, fill, border, radius, width: fills.append(
                (bounds, fill)
            )
        )
        self.renderer._draw_icon = lambda *args, **kwargs: None
        self.renderer._draw_text = lambda *args, **kwargs: None
        try:
            self.renderer._draw_map_submenu(
                world,
                arcade.LBWH(20, 250, 58, 58),
            )
        finally:
            self.renderer._draw_rounded_rect = original_draw_rounded_rect
            self.renderer._draw_icon = original_draw_icon
            self.renderer._draw_text = original_draw_text

        biome_row = self.renderer._control_hitboxes["map_layer_biome"]
        pheromone_row = self.renderer._control_hitboxes["map_layer_pheromones"]
        fill_by_id = {id(bounds): fill for bounds, fill in fills}
        self.assertEqual(
            fill_by_id[id(biome_row)],
            self.renderer.theme.accent_soft,
        )
        self.assertEqual(
            fill_by_id[id(pheromone_row)],
            self.renderer.theme.panel_background_alt,
        )

    def test_map_selection_is_exclusive_toggles_off_and_closes_menu(self) -> None:
        world = SimpleNamespace(environment_map_mode="none")

        def select_environment_map(mode: str) -> None:
            world.environment_map_mode = (
                "none" if world.environment_map_mode == mode else mode
            )

        world.select_environment_map = select_environment_map
        self.renderer._map_submenu_open = True
        self.renderer._control_hitboxes["map_submenu"] = arcade.LBWH(
            0, 0, 80, 80
        )
        self.renderer._control_hitboxes["map_layer_biome"] = arcade.LBWH(
            0, 40, 80, 40
        )
        self.renderer._control_hitboxes["map_layer_pheromones"] = arcade.LBWH(
            0, 0, 80, 40
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 20, 60))
        self.assertEqual(world.environment_map_mode, "biome")
        self.assertFalse(self.renderer._map_submenu_open)

        self.renderer._map_submenu_open = True
        self.assertTrue(self.renderer.handle_mouse_press(world, 20, 20))
        self.assertEqual(world.environment_map_mode, "pheromones")

        self.renderer._map_submenu_open = True
        self.assertTrue(self.renderer.handle_mouse_press(world, 20, 20))
        self.assertEqual(world.environment_map_mode, "none")

    def test_outside_submenu_click_closes_it_and_reaches_control(self) -> None:
        self.renderer._map_submenu_open = True
        self.renderer._control_hitboxes["map_submenu"] = arcade.LBWH(
            100, 100, 100, 100
        )
        self.renderer._control_hitboxes["panel_toggle_stats"] = arcade.LBWH(
            0, 0, 20, 20
        )

        handled = self.renderer.handle_mouse_press(SimpleNamespace(), 10, 10)

        self.assertTrue(handled)
        self.assertFalse(self.renderer._map_submenu_open)
        self.assertTrue(self.renderer._panel_open["stats"])

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

    def test_inspector_page_marker_follows_inner_card_edge(self) -> None:
        world = self.make_inspector_world()
        self.renderer._panel_bounds["inspector"] = arcade.LBWH(100, 100, 368, 330)
        marker_rectangles = []
        marker_circles = []
        original_rectangle = arcade.draw_lrbt_rectangle_filled
        original_circle = arcade.draw_circle_filled

        def capture_rectangle(
            left: float,
            right: float,
            bottom: float,
            top: float,
            color: object,
        ) -> None:
            if color == self.renderer.theme.accent:
                marker_rectangles.append((left, right, bottom, top))

        def capture_circle(
            x: float,
            y: float,
            radius: float,
            color: object,
        ) -> None:
            if color == self.renderer.theme.accent:
                marker_circles.append((x, y, radius))

        arcade.draw_lrbt_rectangle_filled = capture_rectangle
        arcade.draw_circle_filled = capture_circle
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            arcade.draw_lrbt_rectangle_filled = original_rectangle
            arcade.draw_circle_filled = original_circle

        body = self.renderer._control_hitboxes["inspector_body"]
        marker_left = body.left + 8
        marker_bottom = body.bottom + 10
        marker_width = 7.0
        marker_height = body.height - 18
        marker_right = marker_left + marker_width
        marker_top = marker_bottom + marker_height
        radius = marker_width / 2.0

        self.assertIn(
            (marker_left + radius, marker_right, marker_bottom, marker_top),
            marker_rectangles,
        )
        self.assertIn(
            (marker_left, marker_right, marker_bottom + radius, marker_top - radius),
            marker_rectangles,
        )
        self.assertEqual(
            marker_circles,
            [
                (marker_left + radius, marker_bottom + radius, radius),
                (marker_left + radius, marker_top - radius, radius),
            ],
        )

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

    def test_inspector_progress_bars_use_energy_and_stomach_ratios(self) -> None:
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

        self.assertEqual(ratios, [0.25, 0.6])

    def test_creature_inspector_marker_matches_selected_creature(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=10.0,
            heading=0.0,
            color=(9, 9, 9),
            vision=SimpleNamespace(range=120.0, angle=1.0),
            lineage=LineageInfo(species_id=7),
        )
        world = self.make_inspector_world(selected=selected)
        world.species_history = {
            7: SimpleNamespace(founder_color=(210, 40, 90))
        }

        with (
            patch("src.ui.arcade.draw_circle_filled") as filled,
            patch("src.ui.arcade.draw_circle_outline") as outlined,
        ):
            self.renderer._draw_inspector_panel(world)

        filled.assert_any_call(
            ANY,
            ANY,
            8.0,
            (9, 9, 9),
        )
        outlined.assert_any_call(
            ANY,
            ANY,
            8.0,
            self.renderer.theme.selected_outline,
            2.5,
        )
        self.assertEqual(
            self.renderer._text_cache["inspector_species"].text,
            "Species #7",
        )

    def test_creature_inspector_marker_uses_theme_fallback(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=10.0,
            heading=0.0,
            vision=SimpleNamespace(range=120.0, angle=1.0),
            lineage=LineageInfo(species_id=7),
        )
        world = self.make_inspector_world(selected=selected)
        world.species_history = {
            7: SimpleNamespace(founder_color=None)
        }

        with patch("src.ui.arcade.draw_circle_filled") as filled:
            self.renderer._draw_inspector_panel(world)

        filled.assert_any_call(
            ANY,
            ANY,
            8.0,
            self.renderer.theme.herbivore_fill,
        )

    def test_creature_inspector_marker_falls_back_to_species_color(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=10.0,
            heading=0.0,
            vision=SimpleNamespace(range=120.0, angle=1.0),
            lineage=LineageInfo(species_id=7),
        )
        world = self.make_inspector_world(selected=selected)
        world.species_history = {
            7: SimpleNamespace(founder_color=(210, 40, 90))
        }

        with patch("src.ui.arcade.draw_circle_filled") as filled:
            self.renderer._draw_inspector_panel(world)

        filled.assert_any_call(
            ANY,
            ANY,
            8.0,
            (210, 40, 90),
        )

    def test_inspector_draws_trait_and_lineage_rows(self) -> None:
        selected = SimpleNamespace(
            creature_id=938,
            name="Herbivore 938",
            energy=0.5,
            speed=170.0,
            heading=1.25,
            vision=SimpleNamespace(range=120.0, angle=1.7),
            physical_traits=PhysicalTraits(
                radius=18.0,
                movement_cost_multiplier=1.12,
            ),
            lineage=LineageInfo(
                parent_id=12,
                generation=3,
                mutation_delta=TraitMutationDelta(
                    vision_range=2.0,
                    vision_angle=-0.03,
                    radius=1.0,
                    movement_cost_multiplier=0.04,
                ),
            ),
        )
        world = self.make_inspector_world(selected=selected)
        rows: dict[str, tuple[str, str]] = {}
        original_metric_row = self.renderer._draw_metric_row_in_viewport

        def capture_metric_row(
            viewport: object,
            key: str,
            label: str,
            value: str,
            *args: object,
        ) -> None:
            del viewport, args
            rows[key] = (label, value)

        self.renderer._draw_metric_row_in_viewport = capture_metric_row
        try:
            self.renderer._draw_inspector_panel(world)
        finally:
            self.renderer._draw_metric_row_in_viewport = original_metric_row

        self.assertEqual(rows["inspector_body"], ("Body", "18.0px / 1.12x move"))
        self.assertEqual(rows["inspector_lineage"], ("Lineage", "Parent 12 / Gen 3"))
        self.assertEqual(
            rows["inspector_mutations"],
            ("Mutations", "R +1.0, V +2.0/-0.03, M +0.04"),
        )

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

    def test_selected_fitness_label_uses_live_creature_fitness(self) -> None:
        selected = SimpleNamespace(creature_id=938)
        live_fitness = SimpleNamespace(score=lambda fitness_config: 7.25)
        world = SimpleNamespace(
            config=SimpleNamespace(fitness=object()),
            fitness_for=lambda creature: live_fitness
            if creature is selected
            else None,
        )

        self.assertEqual(
            self.renderer._selected_fitness_label(world, selected),
            "7.25",
        )


class SpeciesTreeWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = UiRenderer(build_sim_config())

    def make_record(
        self,
        species_id: int,
        parent_species_id: int | None,
        *,
        exact: bool = True,
        emerged_at: float | None = None,
    ) -> SpeciesRecord:
        traits = (
            SpeciesTraitSnapshot(16.0 + species_id, 100.0, 0.9, 1.0)
            if exact
            else None
        )
        deltas = (
            SpeciesTraitSnapshot(1.0, 2.0, 0.1, 0.02)
            if exact
            else None
        )
        return SpeciesRecord(
            species_id=species_id,
            parent_species_id=parent_species_id,
            founder_creature_id=species_id * 10 if exact else None,
            founder_genome_id=species_id * 100 if exact else None,
            emerged_at=(
                (float(species_id) * 60.0 if emerged_at is None else emerged_at)
                if exact
                else None
            ),
            founder_color=(100, 120, 140) if exact else None,
            data_quality="exact" if exact else "reconstructed",
            founder_traits=traits,
            trait_deltas=deltas,
            distances=SpeciesDistanceBreakdown(
                neat_distance=2.0 if exact else None,
                phenotypic_distance=0.6 if exact else None,
                weighted_phenotypic_distance=1.2 if exact else None,
                composite_distance=3.2 if exact else None,
                compatibility_threshold=3.0 if exact else None,
                phenotypic_weight=2.0 if exact else None,
                radius_component=0.2 if exact else None,
                vision_range_component=0.2 if exact else None,
                vision_angle_component=0.1 if exact else None,
                movement_cost_component=0.1 if exact else None,
            ),
        )

    def make_world(
        self,
        records: dict[int, SpeciesRecord] | None = None,
        *,
        paused: bool = False,
        width: float = 1440.0,
        height: float = 900.0,
        representatives: dict[int, object] | None = None,
    ) -> SimpleNamespace:
        active_records = records or {}
        emerged_times = [
            float(record.emerged_at)
            for record in active_records.values()
            if record.emerged_at is not None
        ]
        return SimpleNamespace(
            is_paused=paused,
            species_history=active_records,
            elapsed_time=max([0.0, *emerged_times]),
            config=self.renderer.config,
            telemetry=None,
            neat_controller=SimpleNamespace(
                config=SimpleNamespace(
                    genome_config=SimpleNamespace(
                        input_keys=tuple(range(-1, -28, -1)),
                        output_keys=tuple(range(12)),
                    )
                ),
                species_manager=SimpleNamespace(
                    representatives=representatives or {}
                ),
            ),
            layout=SimpleNamespace(
                window=arcade.LBWH(0, 0, width, height),
                environment=arcade.LBWH(0, 0, width, height),
            ),
        )

    def test_open_and_close_restore_running_state(self) -> None:
        world = self.make_world()
        self.renderer._control_hitboxes["open_species_tree"] = arcade.LBWH(
            0, 0, 30, 30
        )

        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertTrue(world.is_paused)
        self.assertTrue(self.renderer.species_tree_open)
        self.renderer._control_hitboxes["species_tree_close"] = arcade.LBWH(
            40, 40, 30, 30
        )
        self.assertTrue(self.renderer.handle_mouse_press(world, 50, 50))
        self.assertFalse(world.is_paused)
        self.assertFalse(self.renderer.species_tree_open)

    def test_close_preserves_preexisting_pause(self) -> None:
        world = self.make_world(paused=True)

        self.renderer.open_species_tree(world)
        self.renderer.close_species_tree(world)

        self.assertTrue(world.is_paused)

    def test_modal_captures_keyboard_and_underlying_controls(self) -> None:
        calls = []
        world = self.make_world()
        world.toggle_pause = lambda: calls.append("pause")
        self.renderer.open_species_tree(world)
        self.renderer._control_hitboxes["pause"] = arcade.LBWH(0, 0, 40, 40)

        self.assertTrue(self.renderer.handle_key_press(world, 1, 0))
        self.assertTrue(self.renderer.handle_mouse_press(world, 10, 10))
        self.assertEqual(calls, [])
        self.assertTrue(world.is_paused)

    def test_window_uses_full_inset_bounds_and_registers_nodes(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)

        self.renderer._draw_species_tree_window(world)

        bounds = self.renderer._control_hitboxes["species_tree_window"]
        self.assertEqual((bounds.left, bounds.bottom), (20, 20))
        self.assertEqual((bounds.width, bounds.height), (1400, 860))
        self.assertEqual(set(self.renderer._species_tree_node_bounds), {1, 2})

    def test_runtime_species_lifecycle_tracks_extinction_and_revival(self) -> None:
        world = self.make_world(
            {1: self.make_record(1, None, emerged_at=0.0)}
        )
        world.elapsed_time = 5.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=1))
        ]

        living = self.renderer._sync_species_tree_layout(world)
        self.assertEqual(living.end_times[1], float("inf"))

        world.elapsed_time = 12.0
        world.creatures = []
        extinct = self.renderer._sync_species_tree_layout(world)
        self.assertEqual(extinct.end_times[1], 12.0)

        world.elapsed_time = 15.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=1))
        ]
        revived = self.renderer._sync_species_tree_layout(world)
        self.assertEqual(revived.end_times[1], float("inf"))

    def test_loaded_species_tree_uses_telemetry_extinction_times(self) -> None:
        world = self.make_world(
            {
                1: self.make_record(1, None, emerged_at=0.0),
                2: self.make_record(2, 1, emerged_at=15.0),
            }
        )
        world.elapsed_time = 20.0
        world.creatures = [
            SimpleNamespace(lineage=SimpleNamespace(species_id=2))
        ]
        world.telemetry = SimpleNamespace(
            load_species_end_times=lambda *, up_to_time: {
                1: 10.0,
                2: float("inf"),
            }
        )

        layout = self.renderer._sync_species_tree_layout(world)

        self.assertEqual(layout.end_times[1], 10.0)
        self.assertEqual(layout.end_times[2], float("inf"))
        self.assertEqual(layout.lanes[1], layout.lanes[2])

    def test_species_endpoint_markers_distinguish_extant_and_extinct(self) -> None:
        with patch("src.ui.arcade.draw_circle_filled") as filled:
            self.renderer._draw_species_tree_extant_marker(
                (10.0, 20.0),
                (100, 120, 140),
            )

        self.assertEqual(filled.call_count, 3)
        self.assertEqual(filled.call_args.args[-1], (100, 120, 140, 255))

        with patch("src.ui.arcade.draw_line") as line:
            self.renderer._draw_species_tree_extinct_marker((10.0, 20.0))

        self.assertEqual(line.call_count, 2)
        self.assertTrue(
            all(
                call.args[-2] == self.renderer.theme.text_muted
                for call in line.call_args_list
            )
        )

    def test_hover_finds_node_and_legacy_tooltip_marks_unavailable_data(self) -> None:
        record = self.make_record(1, None, exact=False)
        world = self.make_world({1: record})
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[1]

        self.assertTrue(
            self.renderer.handle_mouse_motion(
                world, node.center_x, node.center_y
            )
        )
        self.assertEqual(self.renderer._species_tree_hovered_id, 1)
        self.assertIn(
            "Unavailable",
            " ".join(self.renderer._species_tree_tooltip_lines(record)),
        )

    def test_click_selects_node_and_highlights_its_ancestry(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
            3: self.make_record(3, 2),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[3]

        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()
        nodes, edges = self.renderer._species_tree_highlighted_path(
            self.renderer._species_tree_last_layout
        )

        self.assertEqual(self.renderer._species_tree_selected_id, 3)
        self.assertEqual(nodes, {1, 2, 3})
        self.assertEqual(edges, {(1, 2), (2, 3)})

    def test_dragging_from_node_pans_without_changing_selection(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_selected_id = 1
        node = self.renderer._species_tree_node_bounds[2]

        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_drag(
            world,
            node.center_x + 8.0,
            node.center_y + 8.0,
        )
        self.renderer.handle_mouse_release()

        self.assertEqual(self.renderer._species_tree_selected_id, 1)

    def test_inspector_report_is_generated_only_after_click(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records)
        self.renderer.open_species_tree(world)

        with patch(
            "src.ui.generate_inspector_report",
            wraps=generate_inspector_report,
        ) as generate:
            self.renderer._draw_species_tree_window(world)
            node = self.renderer._species_tree_node_bounds[2]
            self.renderer.handle_mouse_motion(
                world,
                node.center_x,
                node.center_y,
            )
            self.renderer._draw_species_tree_window(world)
            self.assertEqual(generate.call_count, 0)

            self.renderer.handle_mouse_press(
                world,
                node.center_x,
                node.center_y,
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(generate.call_count, 1)
            self.assertEqual(
                self.renderer._species_tree_report_species_id,
                2,
            )

    def test_radar_texture_is_lazy_cached_replaced_and_cleared(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        genome = SimpleNamespace(
            nodes={
                key: SimpleNamespace(bias=0.0)
                for key in range(12)
            },
            connections={},
        )
        representative = (genome, object(), object())
        world = self.make_world(
            records,
            representatives={1: representative, 2: representative},
        )
        self.renderer.open_species_tree(world)
        textures = [object(), object()]
        futures: list[Future[object]] = [Future(), Future()]
        executor = SimpleNamespace(
            submit=Mock(side_effect=futures),
        )
        self.renderer._species_tree_radar_executor = executor

        with (
            patch(
                "src.ui.arcade.Texture",
                side_effect=textures,
                create=True,
            ) as texture,
            patch("src.ui.arcade.draw_texture_rect", create=True),
        ):
            self.renderer._draw_species_tree_window(world)
            self.assertEqual(executor.submit.call_count, 0)

            child_node = self.renderer._species_tree_node_bounds[2]
            self.renderer.handle_mouse_press(
                world, child_node.center_x, child_node.center_y
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(executor.submit.call_count, 1)
            self.assertEqual(texture.call_count, 0)
            self.assertIsNone(self.renderer._species_tree_radar_texture)
            self.assertEqual(
                self.renderer._text_cache["species_tree_radar_status"].text,
                "Loading behavioral profile...",
            )

            futures[0].set_result(object())
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(texture.call_count, 1)
            self.assertIs(self.renderer._species_tree_radar_texture, textures[0])

            parent_node = self.renderer._species_tree_node_bounds[1]
            self.renderer.handle_mouse_press(
                world, parent_node.center_x, parent_node.center_y
            )
            self.renderer.handle_mouse_release()
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(executor.submit.call_count, 2)
            self.assertEqual(texture.call_count, 1)
            self.assertIsNone(self.renderer._species_tree_radar_texture)

            futures[1].set_result(object())
            self.renderer._draw_species_tree_window(world)

            self.assertEqual(texture.call_count, 2)
            self.assertIs(self.renderer._species_tree_radar_texture, textures[1])

        self.renderer.close_species_tree(world)
        self.assertIsNone(self.renderer._species_tree_radar_texture)
        self.assertIsNone(self.renderer._species_tree_radar_species_id)
        self.renderer.open_species_tree(world)
        self.assertIsNone(self.renderer._species_tree_radar_texture)

    def test_radar_draw_bounds_are_above_text_and_inside_viewport(self) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 320.0, 500.0)
        self.renderer._species_tree_radar_texture = object()
        self.renderer._species_tree_radar_species_id = 2

        with patch("src.ui.arcade.draw_texture_rect", create=True) as draw:
            text_viewport = self.renderer._draw_species_radar_chart(viewport)

        draw.assert_called_once()
        texture, chart_bounds = draw.call_args.args
        self.assertIs(texture, self.renderer._species_tree_radar_texture)
        self.assertGreaterEqual(chart_bounds.left, viewport.left)
        self.assertLessEqual(chart_bounds.right, viewport.right)
        self.assertGreaterEqual(chart_bounds.bottom, viewport.bottom)
        self.assertLessEqual(chart_bounds.top, viewport.top)
        self.assertLess(text_viewport.top, chart_bounds.bottom)

    def test_stale_radar_result_is_not_converted_to_texture(self) -> None:
        old_future: Future[object] = Future()
        old_future.set_result(object())
        new_future: Future[object] = Future()
        self.renderer._species_tree_open = True
        self.renderer._species_tree_selected_id = 2
        self.renderer._species_tree_radar_species_id = 1
        self.renderer._species_tree_radar_future = old_future

        with patch("src.ui.arcade.Texture", create=True) as texture:
            self.renderer._consume_species_radar_result()

        texture.assert_not_called()
        self.assertIsNone(self.renderer._species_tree_radar_texture)

        self.renderer._species_tree_radar_species_id = 2
        self.renderer._species_tree_radar_future = new_future
        new_future.set_result(object())
        with patch(
            "src.ui.arcade.Texture",
            return_value="texture",
            create=True,
        ) as texture:
            self.renderer._consume_species_radar_result()

        texture.assert_called_once()
        self.assertEqual(self.renderer._species_tree_radar_texture, "texture")

    def test_radar_render_failure_uses_unavailable_placeholder(self) -> None:
        future: Future[object] = Future()
        future.set_exception(RuntimeError("render failed"))
        self.renderer._species_tree_open = True
        self.renderer._species_tree_selected_id = 2
        self.renderer._species_tree_radar_species_id = 2
        self.renderer._species_tree_radar_future = future

        self.renderer._consume_species_radar_result()
        viewport = arcade.LBWH(100.0, 100.0, 320.0, 500.0)
        text_viewport = self.renderer._draw_species_radar_chart(viewport)

        self.assertEqual(self.renderer._species_tree_radar_error, "render_failed")
        self.assertEqual(
            self.renderer._text_cache["species_tree_radar_status"].text,
            "Behavioral profile unavailable",
        )
        self.assertLess(text_viewport.height, viewport.height)

    def test_close_cancels_radar_work_and_shuts_down_executor(self) -> None:
        future: Future[object] = Future()
        executor = Mock()
        self.renderer._species_tree_radar_future = future
        self.renderer._species_tree_radar_executor = executor
        self.renderer._species_tree_radar_species_id = 2

        self.renderer.close()

        self.assertTrue(future.cancelled())
        self.assertIsNone(self.renderer._species_tree_radar_future)
        self.assertIsNone(self.renderer._species_tree_radar_executor)
        executor.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )

    def test_species_inspector_header_marker_matches_selected_node(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(2, 1)
        report = generate_inspector_report(
            record,
            parent,
            None,
            self.renderer.config,
            range(12),
        )
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 600.0)

        with (
            patch("src.ui.arcade.draw_circle_filled") as filled,
            patch("src.ui.arcade.draw_circle_outline") as outlined,
        ):
            self.renderer._draw_species_inspector(
                bounds,
                report,
                record,
            )

        filled.assert_any_call(
            bounds.left + 27.0,
            bounds.top - 23.0,
            11.0,
            record.founder_color,
        )
        outlined.assert_any_call(
            bounds.left + 27.0,
            bounds.top - 23.0,
            11.0,
            self.renderer.theme.selected_outline,
            3.0,
        )
        self.assertEqual(
            self.renderer._text_cache[
                "species_tree_inspector_title"
            ].text,
            "SPECIES 2 INSPECTOR",
        )
        inspector_lines = self.renderer._species_inspector_lines(report)
        self.assertIn("Anatomy & Morphology", inspector_lines)
        self.assertIn("Radius: 18.00 px", inspector_lines)
        self.assertIn("Vision: 100.00 px / 0.900 rad", inspector_lines)
        self.assertIn("Metabolic Profile", inspector_lines)
        self.assertIn("Neuro-Integration Hubs", inspector_lines)
        self.assertIn("Behavioral Ethogram", inspector_lines)
        self.assertTrue(
            any(
                line.startswith("Basal metabolic BMR:")
                for line in inspector_lines
            )
        )
        self.assertTrue(
            any(
                line.startswith("Parent BMR/active:")
                for line in inspector_lines
            )
        )

    def test_species_inspector_marker_uses_legacy_color_fallback(self) -> None:
        record = replace(
            self.make_record(7, None),
            founder_color=None,
        )
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 600.0)

        with patch("src.ui.arcade.draw_circle_filled") as filled:
            self.renderer._draw_species_inspector(bounds, None, record)

        self.assertEqual(
            filled.call_args.args[3],
            self.renderer.theme.herbivore_fill,
        )

    def test_species_inspector_marker_stays_fixed_while_scrolling(self) -> None:
        record = self.make_record(2, 1)
        bounds = arcade.LBWH(100.0, 100.0, 340.0, 220.0)
        marker_positions: list[tuple[float, float]] = []

        with patch("src.ui.arcade.draw_circle_filled") as filled:
            self.renderer._scroll_offsets["species_tree_inspector"] = 0.0
            self.renderer._draw_species_inspector(bounds, None, record)
            marker_positions.append(filled.call_args.args[:2])
            self.renderer._scroll_offsets["species_tree_inspector"] = 100.0
            self.renderer._draw_species_inspector(bounds, None, record)
            marker_positions.append(filled.call_args.args[:2])

        self.assertEqual(marker_positions[0], marker_positions[1])

    def test_species_inspector_defaults_wider_than_legacy_cap(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()

        self.renderer._draw_species_tree_window(world)

        content = self.renderer._control_hitboxes["species_tree_body"]
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]
        inspector_left = resize.left + 5.0
        inspector_width = content.right - inspector_left
        self.assertGreater(inspector_width, 380.0)

    def test_species_inspector_resize_clamps_to_two_thirds_width(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()
        self.renderer._draw_species_tree_window(world)
        content = self.renderer._control_hitboxes["species_tree_body"]
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]

        self.assertTrue(
            self.renderer.handle_mouse_press(
                world,
                resize.center_x,
                resize.center_y,
            )
        )
        self.assertTrue(
            self.renderer.handle_mouse_drag(
                world,
                content.left,
                resize.center_y,
            )
        )
        self.renderer.handle_mouse_release()
        self.renderer._draw_species_tree_window(world)

        self.assertAlmostEqual(
            self.renderer._species_tree_inspector_width or 0.0,
            content.width * 2.0 / 3.0,
        )

    def test_species_inspector_resize_drag_does_not_select_or_pan(self) -> None:
        records = {
            1: self.make_record(1, None),
            2: self.make_record(2, 1),
        }
        world = self.make_world(records, width=1440, height=900)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        node = self.renderer._species_tree_node_bounds[2]
        self.renderer.handle_mouse_press(world, node.center_x, node.center_y)
        self.renderer.handle_mouse_release()
        self.renderer._draw_species_tree_window(world)
        resize = self.renderer._control_hitboxes["species_tree_inspector_resize"]
        selected = self.renderer._species_tree_selected_id
        horizontal_offset = self.renderer._species_tree_horizontal_offset
        vertical_offset = self.renderer._species_tree_vertical_offset

        self.renderer.handle_mouse_press(world, resize.center_x, resize.center_y)
        self.renderer.handle_mouse_drag(
            world,
            resize.center_x - 80.0,
            resize.center_y,
        )
        self.renderer.handle_mouse_release()

        self.assertEqual(self.renderer._species_tree_selected_id, selected)
        self.assertEqual(
            self.renderer._species_tree_horizontal_offset,
            horizontal_offset,
        )
        self.assertEqual(
            self.renderer._species_tree_vertical_offset,
            vertical_offset,
        )
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_species_inspector_wraps_long_lines_without_ellipses(self) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 145.0, 180.0)
        lines = [
            "Behavioral Ethogram",
            (
                "🟢 [Load Carriage State (Carrying Object)] now actively "
                "triggers/sensitizes [Threat Avoidance Reflexes]"
            ),
        ]

        self.renderer._draw_scrollable_lines_in_bounds(
            "species_tree_inspector",
            viewport,
            lines,
            line_spacing=19.0,
            first_line_color=self.renderer.theme.text_primary,
            body_color=self.renderer.theme.text_muted,
            first_line_bold=True,
            wrap_lines=True,
            draw_ethogram_markers=True,
        )

        rendered = [
            text.text
            for key, text in self.renderer._text_cache.items()
            if key.startswith("species_tree_inspector_line_")
        ]
        self.assertGreater(len(rendered), len(lines))
        self.assertFalse(any("..." in line for line in rendered))
        self.assertFalse(any(line.startswith("🟢") for line in rendered))

    def test_species_inspector_ethogram_markers_use_bright_custom_colors(self) -> None:
        viewport = arcade.LBWH(100.0, 100.0, 520.0, 120.0)
        circles: list[tuple[object, ...]] = []
        original_circle = arcade.draw_circle_filled
        arcade.draw_circle_filled = lambda *args: circles.append(args)
        try:
            self.renderer._draw_scrollable_lines_in_bounds(
                "species_tree_inspector",
                viewport,
                ["🟢 [Sense] now actively triggers/sensitizes [Behavior]"],
                line_spacing=19.0,
                first_line_color=self.renderer.theme.text_primary,
                body_color=self.renderer.theme.text_muted,
                wrap_lines=True,
                draw_ethogram_markers=True,
            )
        finally:
            arcade.draw_circle_filled = original_circle

        self.assertIn(
            (viewport.left + 8.0, viewport.top - 8.0, 6.0, (0, 210, 72)),
            circles,
        )
        self.assertEqual(
            self.renderer._text_cache["species_tree_inspector_line_0"].text,
            "[Sense] now actively triggers/sensitizes [Behavior]",
        )

    def test_tooltip_is_compact_and_uses_parent_percentages(self) -> None:
        parent = self.make_record(1, None)
        record = self.make_record(2, 1)

        tooltip = self.renderer._species_tree_tooltip_lines(record, parent)

        self.assertLessEqual(len(tooltip), 5)
        self.assertIn("Species ID: 2", tooltip)
        self.assertTrue(any("% Radius" in line for line in tooltip))
        self.assertNotIn("NEAT CHANGES FROM PARENT", " ".join(tooltip))

    def test_species_tree_builds_neat_labels_from_active_config(self) -> None:
        world = SimpleNamespace(
            neat_controller=SimpleNamespace(
                config=SimpleNamespace(
                    genome_config=SimpleNamespace(
                        input_keys=list(range(-1, -38, -1)),
                        output_keys=list(range(16)),
                    )
                )
            )
        )

        labels = self.renderer._species_tree_neat_node_labels(world)

        self.assertEqual(
            [labels[key] for key in range(-1, -38, -1)],
            list(SENSOR_INPUT_NAMES),
        )
        self.assertEqual(labels[-11], "food_proximity")
        self.assertEqual(labels[-27], "stomach_fullness")
        self.assertEqual(labels[-28], "sound_strength")
        self.assertEqual(labels[-37], "alarm_pheromone_forward_right")
        self.assertEqual(labels[0], "accelerate")
        self.assertEqual(labels[3], "want_eat")
        self.assertEqual(labels[12], "emit_sound")

    def test_species_tree_reuses_and_invalidates_neat_label_cache(self) -> None:
        genome_config = SimpleNamespace(
            input_keys=[-1],
            output_keys=[0],
        )
        world = SimpleNamespace(
            neat_controller=SimpleNamespace(
                config=SimpleNamespace(genome_config=genome_config)
            )
        )

        first = self.renderer._species_tree_neat_node_labels(world)
        second = self.renderer._species_tree_neat_node_labels(world)
        genome_config.output_keys = [10]
        changed = self.renderer._species_tree_neat_node_labels(world)

        self.assertIs(first, second)
        self.assertIsNot(first, changed)
        self.assertEqual(changed[10], "accelerate")

    def test_species_tree_missing_config_reuses_empty_neat_labels(self) -> None:
        world = SimpleNamespace()

        first = self.renderer._species_tree_neat_node_labels(world)
        second = self.renderer._species_tree_neat_node_labels(world)

        self.assertIs(first, second)
        self.assertEqual(first, {})

    def test_species_tree_formats_named_neat_nodes_in_every_change_kind(
        self,
    ) -> None:
        labels = {-11: "food_proximity", 0: "accelerate", 3: "want_eat"}
        cases = {
            "Node 0 added": "Node accelerate added",
            "Node 0 removed": "Node accelerate removed",
            "Node 0 bias +0.100 -> +0.200": (
                "Node accelerate bias +0.100 -> +0.200"
            ),
            "Node 0 activation sigmoid -> tanh": (
                "Node accelerate activation sigmoid -> tanh"
            ),
            "Connection -11->3 added": (
                "Connection food_proximity -> want_eat added"
            ),
            "Connection -11->3 removed": (
                "Connection food_proximity -> want_eat removed"
            ),
            "Connection -11->3 enabled": (
                "Connection food_proximity -> want_eat enabled"
            ),
            "Connection -11->3 disabled": (
                "Connection food_proximity -> want_eat disabled"
            ),
            "Weight -11->3 +0.100 -> +0.900": (
                "Weight food_proximity -> want_eat +0.100 -> +0.900"
            ),
        }

        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(
                    self.renderer._format_species_tree_neat_change(
                        original,
                        labels,
                    ),
                    expected,
                )

    def test_species_tree_neat_names_keep_hidden_ids_and_custom_text(self) -> None:
        labels = {0: "accelerate"}

        self.assertEqual(
            self.renderer._format_species_tree_neat_change(
                "Connection 42->0 added",
                labels,
            ),
            "Connection 42 -> accelerate added",
        )
        self.assertEqual(
            self.renderer._format_species_tree_neat_change(
                "A legacy custom description",
                labels,
            ),
            "A legacy custom description",
        )

    def test_wheel_scrolls_vertically_and_clamps(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]

        self.assertGreater(self.renderer._species_tree_vertical_limit, 0.0)
        self.assertTrue(
            self.renderer.handle_mouse_scroll(
                canvas.center_x, canvas.center_y, -3
            )
        )
        self.assertGreater(self.renderer._species_tree_vertical_offset, 0.0)
        self.renderer.handle_mouse_scroll(
            canvas.center_x, canvas.center_y, 1000
        )
        self.assertEqual(self.renderer._species_tree_vertical_offset, 0.0)

    def test_many_branches_expose_horizontal_overflow(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 14):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=120.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)

        self.assertGreater(self.renderer._species_tree_horizontal_limit, 0.0)
        self.assertIn(
            "species_tree_horizontal_thumb",
            self.renderer._control_hitboxes,
        )

    def test_horizontal_scrollbar_drag_and_resize_clamp_offset(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=120.0,
            )
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        thumb = self.renderer._control_hitboxes["species_tree_horizontal_thumb"]
        track = self.renderer._control_hitboxes["species_tree_horizontal_track"]

        self.assertTrue(
            self.renderer.handle_mouse_press(
                world, thumb.center_x, thumb.center_y
            )
        )
        self.assertTrue(
            self.renderer.handle_mouse_drag(
                world, track.right - 1.0, track.center_y
            )
        )
        self.assertGreater(self.renderer._species_tree_horizontal_offset, 0.0)
        self.renderer.handle_mouse_release()

        world.layout.window = arcade.LBWH(0, 0, 2400, 900)
        self.renderer._draw_species_tree_window(world)
        self.assertEqual(self.renderer._species_tree_horizontal_limit, 0.0)
        self.assertEqual(self.renderer._species_tree_horizontal_offset, 0.0)

    def test_trackpad_horizontal_axis_scrolls_tree(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 10):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=120.0,
            )
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]

        self.renderer.handle_mouse_scroll(
            canvas.center_x,
            canvas.center_y,
            0.0,
            -2.0,
        )

        self.assertGreater(self.renderer._species_tree_horizontal_offset, 0.0)

    def test_opening_focuses_latest_chunk_without_fit(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 16):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)

        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertEqual(self.renderer._species_tree_zoom, 1.0)
        self.assertEqual(
            self.renderer._species_tree_vertical_offset,
            self.renderer._species_tree_vertical_offset_max,
        )
        self.assertLess(
            len(self.renderer._species_tree_node_bounds),
            len(records),
        )

    def test_opening_culls_offscreen_branched_nodes(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 18):
            records[species_id] = self.make_record(species_id, 1)
        world = self.make_world(records, width=720, height=420)

        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        self.assertLess(
            len(self.renderer._species_tree_node_bounds),
            len(records),
        )
        self.assertEqual(
            set(self.renderer._species_tree_node_bounds),
            set(self.renderer._species_tree_visible_slice.node_ids),
        )

    def test_zoom_controls_switch_to_manual_and_fit_again(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 10):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        initial_zoom = self.renderer._species_tree_zoom
        for key in (
            "species_tree_zoom_out",
            "species_tree_zoom_label",
            "species_tree_zoom_in",
            "species_tree_zoom_fit",
        ):
            self.assertIn(key, self.renderer._control_hitboxes)
        self.assertEqual(
            self.renderer._text_cache["species_tree_zoom_percentage"].text,
            f"{initial_zoom * 100.0:.0f}%",
        )
        self.assertTrue(self.renderer._icon_path("zoom_in").is_file())
        self.assertTrue(self.renderer._icon_path("zoom_out").is_file())
        self.assertIn("zoom_in", self.renderer._texture_cache)
        self.assertIn("zoom_out", self.renderer._texture_cache)
        self.assertNotIn(
            "button_species_tree_zoom_in",
            self.renderer._text_cache,
        )
        self.assertNotIn(
            "button_species_tree_zoom_out",
            self.renderer._text_cache,
        )
        plus = self.renderer._control_hitboxes["species_tree_zoom_in"]

        self.renderer.handle_mouse_press(world, plus.center_x, plus.center_y)

        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertGreater(self.renderer._species_tree_zoom, initial_zoom)
        self.assertGreater(self.renderer._species_tree_vertical_limit, 0.0)
        fit = self.renderer._control_hitboxes["species_tree_zoom_fit"]
        self.renderer.handle_mouse_press(world, fit.center_x, fit.center_y)
        self.assertTrue(self.renderer._species_tree_fit_mode)
        self.assertLess(self.renderer._species_tree_zoom, initial_zoom)
        self.assertEqual(self.renderer._species_tree_horizontal_offset, 0.0)
        self.assertEqual(self.renderer._species_tree_vertical_offset, 0.0)

    def test_manual_zoom_preserves_content_at_canvas_center(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        layout = self.renderer._species_tree_last_layout
        canvas = self.renderer._species_tree_last_canvas
        assert layout is not None
        assert canvas is not None

        old_zoom = self.renderer._species_tree_zoom
        old_inset_x, old_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, old_zoom
        )
        old_content_center = (
            (
                canvas.center_x
                - canvas.left
                - old_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / old_zoom,
            (
                canvas.top
                - old_inset_y
                + self.renderer._species_tree_vertical_offset
                - canvas.center_y
            )
            / old_zoom,
        )

        self.renderer._adjust_species_tree_zoom(
            self.renderer.SPECIES_TREE_ZOOM_FACTOR
        )

        new_zoom = self.renderer._species_tree_zoom
        new_inset_x, new_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, new_zoom
        )
        new_content_center = (
            (
                canvas.center_x
                - canvas.left
                - new_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / new_zoom,
            (
                canvas.top
                - new_inset_y
                + self.renderer._species_tree_vertical_offset
                - canvas.center_y
            )
            / new_zoom,
        )
        self.assertAlmostEqual(new_content_center[0], old_content_center[0])
        self.assertAlmostEqual(new_content_center[1], old_content_center[1])

    def test_manual_zoom_limits_are_ten_to_two_hundred_percent(self) -> None:
        world = self.make_world({1: self.make_record(1, None)})
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)

        for _ in range(30):
            self.renderer._adjust_species_tree_zoom(
                self.renderer.SPECIES_TREE_ZOOM_FACTOR
            )
        self.assertEqual(
            self.renderer._species_tree_zoom,
            self.renderer.SPECIES_TREE_MAX_ZOOM,
        )
        for _ in range(60):
            self.renderer._adjust_species_tree_zoom(
                1.0 / self.renderer.SPECIES_TREE_ZOOM_FACTOR
            )
        self.assertEqual(
            self.renderer._species_tree_zoom,
            self.renderer.SPECIES_TREE_MIN_ZOOM,
        )

    def test_fit_recalculates_after_resize_and_history_change(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 10):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._activate_species_tree_fit()
        self.renderer._draw_species_tree_window(world)
        small_window_zoom = self.renderer._species_tree_zoom

        world.layout.window = arcade.LBWH(0, 0, 1600, 900)
        self.renderer._draw_species_tree_window(world)
        self.assertGreater(self.renderer._species_tree_zoom, small_window_zoom)

        for species_id in range(10, 18):
            records[species_id] = self.make_record(species_id, species_id - 1)
        self.renderer._draw_species_tree_window(world)
        self.assertLess(self.renderer._species_tree_zoom, 1.0)

    def test_command_wheel_zoom_preserves_content_beneath_cursor(self) -> None:
        records = {1: self.make_record(1, None)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(species_id, species_id - 1)
        world = self.make_world(records, width=700, height=500)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        layout = self.renderer._species_tree_last_layout
        canvas = self.renderer._species_tree_last_canvas
        assert layout is not None
        assert canvas is not None
        cursor_x = canvas.left + canvas.width * 0.72
        cursor_y = canvas.bottom + canvas.height * 0.38

        old_zoom = self.renderer._species_tree_zoom
        old_inset_x, old_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, old_zoom
        )
        old_content_point = (
            (
                cursor_x
                - canvas.left
                - old_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / old_zoom,
            (
                canvas.top
                - old_inset_y
                + self.renderer._species_tree_vertical_offset
                - cursor_y
            )
            / old_zoom,
        )

        handled = self.renderer.handle_mouse_scroll(
            cursor_x,
            cursor_y,
            1.0,
            command_down=True,
        )

        self.assertTrue(handled)
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertGreater(self.renderer._species_tree_zoom, old_zoom)
        new_zoom = self.renderer._species_tree_zoom
        new_inset_x, new_inset_y = self.renderer._species_tree_content_insets(
            layout, canvas, new_zoom
        )
        new_content_point = (
            (
                cursor_x
                - canvas.left
                - new_inset_x
                + self.renderer._species_tree_horizontal_offset
            )
            / new_zoom,
            (
                canvas.top
                - new_inset_y
                + self.renderer._species_tree_vertical_offset
                - cursor_y
            )
            / new_zoom,
        )
        self.assertAlmostEqual(new_content_point[0], old_content_point[0])
        self.assertAlmostEqual(new_content_point[1], old_content_point[1])

        self.renderer.handle_mouse_scroll(
            cursor_x,
            cursor_y,
            -1.0,
            command_down=True,
        )
        self.assertAlmostEqual(self.renderer._species_tree_zoom, old_zoom)
        self.assertFalse(self.renderer._species_tree_fit_mode)

    def test_command_wheel_outside_canvas_is_consumed_without_zooming(self) -> None:
        world = self.make_world({1: self.make_record(1, None)})
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        old_zoom = self.renderer._species_tree_zoom

        handled = self.renderer.handle_mouse_scroll(
            canvas.center_x,
            canvas.top + 12.0,
            1.0,
            command_down=True,
        )

        self.assertTrue(handled)
        self.assertEqual(self.renderer._species_tree_zoom, old_zoom)
        self.assertFalse(self.renderer._species_tree_fit_mode)

    def test_timeline_formats_ticks_and_registers_only_timed_events(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=3661.0),
            3: self.make_record(3, 2, exact=False),
        }
        world = self.make_world(records)
        world.elapsed_time = 4000.0
        self.renderer.open_species_tree(world)

        self.renderer._draw_species_tree_window(world)

        self.assertEqual(
            self.renderer._format_species_tree_time(61.0),
            "01:01",
        )
        self.assertEqual(
            self.renderer._format_species_tree_time(3661.0),
            "1:01:01",
        )
        self.assertEqual(
            set(self.renderer._species_tree_timeline_bucket_bounds),
            {0, 2},
        )
        self.assertIn("species_tree_timeline", self.renderer._control_hitboxes)

    def test_timeline_bucket_jumps_to_its_time_range(self) -> None:
        records = {
            1: self.make_record(1, None, emerged_at=0.0),
            2: self.make_record(2, 1, emerged_at=300.0),
            3: self.make_record(3, 1, emerged_at=600.0),
        }
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        marker = self.renderer._species_tree_timeline_bucket_bounds[0]

        handled = self.renderer.handle_mouse_press(
            world,
            marker.center_x,
            marker.center_y,
        )

        self.assertTrue(handled)
        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertEqual(self.renderer._species_tree_zoom, 1.0)
        self.assertGreaterEqual(self.renderer._species_tree_vertical_offset, 0.0)

    def test_timeline_ruler_jump_preserves_horizontal_offset(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=600.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.0
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_horizontal_offset = 30.0
        timeline = self.renderer._control_hitboxes["species_tree_timeline"]

        self.renderer.handle_mouse_press(
            world,
            timeline.left + 5.0,
            timeline.bottom + 13.0,
        )

        self.assertAlmostEqual(
            self.renderer._species_tree_horizontal_offset,
            30.0,
        )
        self.assertGreater(self.renderer._species_tree_vertical_offset, 0.0)

    def test_canvas_drag_preserves_fitted_zoom_and_pans(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=600.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]
        self.assertFalse(self.renderer._species_tree_fit_mode)
        fitted_zoom = self.renderer._species_tree_zoom
        initial_horizontal_offset = self.renderer._species_tree_horizontal_offset

        self.renderer.handle_mouse_press(
            world,
            canvas.center_x,
            canvas.center_y,
        )
        self.renderer.handle_mouse_drag(
            world,
            canvas.center_x + 24.0,
            canvas.center_y + 18.0,
        )

        self.assertFalse(self.renderer._species_tree_fit_mode)
        self.assertEqual(self.renderer._species_tree_zoom, fitted_zoom)
        self.assertTrue(self.renderer._species_tree_canvas_drag_started)
        self.assertNotEqual(
            self.renderer._species_tree_horizontal_offset,
            initial_horizontal_offset,
        )
        self.renderer.handle_mouse_release()
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_canvas_drag_preserves_manual_zoom_and_pans_both_axes(self) -> None:
        records = {1: self.make_record(1, None, emerged_at=0.0)}
        for species_id in range(2, 12):
            records[species_id] = self.make_record(
                species_id,
                1,
                emerged_at=600.0,
            )
        world = self.make_world(records, width=700, height=420)
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        self.renderer._species_tree_fit_mode = False
        self.renderer._species_tree_zoom = 1.25
        self.renderer._draw_species_tree_window(world)
        canvas = self.renderer._control_hitboxes["species_tree_canvas"]

        self.renderer.handle_mouse_press(
            world,
            canvas.center_x,
            canvas.center_y,
        )
        self.renderer.handle_mouse_drag(
            world,
            canvas.center_x - 20.0,
            canvas.center_y + 15.0,
        )

        self.assertEqual(self.renderer._species_tree_zoom, 1.25)
        self.assertGreater(self.renderer._species_tree_horizontal_offset, 0.0)
        self.assertGreater(self.renderer._species_tree_vertical_offset, 0.0)
        self.renderer.handle_mouse_release()
        self.assertFalse(self.renderer._species_tree_canvas_drag)

    def test_cached_five_thousand_node_modal_opens_under_fifty_ms(self) -> None:
        records = {
            species_id: self.make_record(
                species_id,
                None if species_id == 1 else species_id - 1,
                emerged_at=(species_id - 1) * (36000.0 / 4999.0),
            )
            for species_id in range(1, 5001)
        }
        world = self.make_world(records, width=1440, height=900)
        world.elapsed_time = 36000.0
        self.renderer._sync_species_tree_layout(world)
        placements = self.renderer._species_tree_layout_manager.placement_count

        started = perf_counter()
        self.renderer.open_species_tree(world)
        self.renderer._draw_species_tree_window(world)
        elapsed = perf_counter() - started

        self.assertLess(elapsed, 0.05)
        self.assertEqual(
            self.renderer._species_tree_layout_manager.placement_count,
            placements,
        )
        self.assertLess(len(self.renderer._species_tree_node_bounds), 500)
        self.assertLess(
            len(self.renderer._species_tree_visible_slice.routes),
            500,
        )
        self.assertLessEqual(
            len(self.renderer._species_tree_timeline_bucket_bounds),
            21,
        )


if __name__ == "__main__":
    unittest.main()
