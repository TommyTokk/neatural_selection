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

    def test_draw_text_does_not_reassign_unchanged_layout_properties(
        self,
    ) -> None:
        """Avoid invalidating Arcade's text layout on every frame."""

        class CountingText:
            def __init__(
                self,
                text: str,
                x: float,
                y: float,
                color: object,
                size: float,
                **kwargs: object,
            ) -> None:
                object.__setattr__(self, "_count_changes", False)
                self.text = text
                self.x = x
                self.y = y
                normalized_color = tuple(color)  # type: ignore[arg-type]
                self.color = (
                    (*normalized_color, 255)
                    if len(normalized_color) == 3
                    else normalized_color
                )
                self.font_size = size
                self.bold = kwargs.get("bold", False)
                self.width = kwargs.get("width")
                self.multiline = kwargs.get("multiline", False)
                self.align = kwargs.get("align", "left")
                self.anchor_x = kwargs.get("anchor_x", "left")
                self.anchor_y = kwargs.get("anchor_y", "baseline")
                self.rotation = kwargs.get("rotation", 0.0)
                self.assignment_count = 0
                object.__setattr__(self, "_count_changes", True)

            def __setattr__(self, name: str, value: object) -> None:
                if getattr(self, "_count_changes", False):
                    object.__setattr__(
                        self,
                        "assignment_count",
                        self.assignment_count + 1,
                    )
                object.__setattr__(self, name, value)

            def draw(self) -> None:
                return None

        painter = ArcadePainter()
        with patch(
            "src.ui.common.drawing.arcade.Text",
            CountingText,
        ):
            painter.draw_text(
                "stable",
                "Stable",
                10,
                20,
                (1, 2, 3),
                12,
                width=100,
                multiline=True,
            )
            cached = painter.text_cache["stable"]
            painter.draw_text(
                "stable",
                "Stable",
                10,
                20,
                (1, 2, 3),
                12,
                width=100,
                multiline=True,
            )

        self.assertEqual(cached.assignment_count, 0)

    def test_wrap_text_uses_glyph_widths_and_caches_layout(self) -> None:
        """Use rendered widths instead of average character counts."""

        class MeasuredText:
            text_updates = 0

            def __init__(
                self,
                text: str,
                *_args: object,
                **_kwargs: object,
            ) -> None:
                self._text = text

            @property
            def text(self) -> str:
                return self._text

            @text.setter
            def text(self, value: str) -> None:
                type(self).text_updates += 1
                self._text = value

            @property
            def content_width(self) -> float:
                return sum(
                    12.0 if character == "W" else 4.0
                    for character in self._text
                )

        painter = ArcadePainter()
        with patch(
            "src.ui.common.drawing.arcade.Text",
            MeasuredText,
        ):
            first = painter.wrap_text("WWWWWW", 25.0, 12.0)
            updates_after_first_layout = MeasuredText.text_updates
            second = painter.wrap_text("WWWWWW", 25.0, 12.0)

        self.assertEqual(first, ("WW", "WW", "WW"))
        self.assertEqual(second, first)
        self.assertEqual(
            MeasuredText.text_updates,
            updates_after_first_layout,
        )

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
