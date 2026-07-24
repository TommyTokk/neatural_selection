from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import arcade

from src.ui.common.drawing import ArcadePainter
from src.ui.common.interaction import UiInteractionState, rect_contains


class ArcadePainterTest(unittest.TestCase):
    """Verify the shared drawing implementation used by all UI views."""

    def test_draw_text_reuses_cached_text_object(self) -> None:
        """Update an existing text object rather than allocating every frame."""
        painter = ArcadePainter()

        with patch("src.ui.common.drawing.arcade.Text") as text_type:
            cached = SimpleNamespace(draw=Mock())
            text_type.return_value = cached
            painter.draw_text("fps", "30", 10, 20, (255, 255, 255), 12)
            painter.draw_text("fps", "60", 30, 40, (1, 2, 3), 14, bold=True)

        text_type.assert_called_once()
        self.assertEqual(cached.text, "60")
        self.assertEqual(cached.x, 30)
        self.assertEqual(cached.y, 40)
        self.assertEqual(cached.font_size, 14)
        self.assertTrue(cached.bold)
        self.assertEqual(cached.draw.call_count, 2)

    def test_texture_draw_falls_back_to_rect_api(self) -> None:
        """Use the newer rectangle API when the legacy call is incompatible."""
        painter = ArcadePainter()
        bounds = arcade.LBWH(2, 3, 10, 12)
        texture = object()

        with (
            patch(
                "src.ui.common.drawing.arcade.draw_texture_rectangle",
                side_effect=TypeError,
                create=True,
            ),
            patch(
                "src.ui.common.drawing.arcade.draw_texture_rect",
                create=True,
            ) as draw_rect,
        ):
            self.assertTrue(painter.draw_icon_texture(bounds, texture))

        draw_rect.assert_called_once_with(texture, bounds)

    def test_interaction_state_clears_only_frame_geometry(self) -> None:
        """Preserve scroll position while clearing transient frame regions."""
        state = UiInteractionState()
        bounds = arcade.LBWH(0, 0, 20, 20)
        state.hitboxes["button"] = bounds
        state.scroll_regions["list"] = bounds
        state.scroll_limits["list"] = 100
        state.scroll_offsets["list"] = 25

        state.begin_frame()

        self.assertEqual(state.hitboxes, {})
        self.assertEqual(state.scroll_regions, {})
        self.assertEqual(state.scroll_limits, {})
        self.assertEqual(state.scroll_offsets, {"list": 25})
        self.assertTrue(rect_contains(bounds, 10, 10))
        self.assertFalse(rect_contains(bounds, 25, 10))
