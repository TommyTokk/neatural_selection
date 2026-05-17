from __future__ import annotations

from dataclasses import dataclass
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest

try:
    import arcade
except ModuleNotFoundError:
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
            return self.left + self.width / 2

        @property
        def center_y(self) -> float:
            return self.bottom + self.height / 2

    def fake_lbwh(left: float, bottom: float, width: float, height: float) -> FakeRect:
        return FakeRect(left, bottom, width, height)

    arcade.LBWH = fake_lbwh
    arcade.Rect = FakeRect
    sys.modules["arcade"] = arcade

for optional_module in ("neat", "pymunk"):
    if optional_module not in sys.modules:
        sys.modules[optional_module] = ModuleType(optional_module)

from configs.sim_config import build_sim_config
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

    def test_scroll_over_graph_zooms_brain_graph(self) -> None:
        handled = self.renderer.handle_mouse_scroll(200, 200, 1)

        self.assertTrue(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.1)

    def test_scroll_over_non_graph_brain_window_is_consumed(self) -> None:
        handled = self.renderer.handle_mouse_scroll(200, 380, 1)

        self.assertTrue(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

    def test_scroll_outside_ui_is_not_consumed(self) -> None:
        handled = self.renderer.handle_mouse_scroll(20, 20, 1)

        self.assertFalse(handled)
        self.assertAlmostEqual(self.renderer._brain_graph_zoom, 1.0)

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


if __name__ == "__main__":
    unittest.main()
