"""Small reusable immediate-mode UI widgets."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from collections.abc import Sequence

import arcade

from src.ui.common.drawing import ArcadePainter, Color
from src.ui.common.interaction import UiInteractionState, rect_contains


@dataclass(frozen=True, slots=True)
class IconButton:
    """Describe and draw a reusable icon button."""

    key: str
    icon_name: str
    bounds: arcade.Rect
    active: bool = False

    def draw(
        self,
        painter: ArcadePainter,
        interactions: UiInteractionState,
        *,
        fill_color: Color,
        active_color: Color,
        border_color: Color,
        radius: float = 8.0,
    ) -> None:
        """Draw the button and register its hitbox.

        Parameters
        ----------
        painter
            Shared Arcade drawing service.
        interactions
            Current interaction registry.
        fill_color, active_color, border_color
            Button palette.
        radius
            Corner radius.
        """
        interactions.register_hitbox(self.key, self.bounds)
        painter.draw_rounded_rect(
            self.bounds,
            active_color if self.active else fill_color,
            border_color,
            radius,
            1.0,
        )


@dataclass(frozen=True, slots=True)
class PanelFrame:
    """Describe a titled floating-panel frame."""

    key: str
    title: str
    bounds: arcade.Rect

    def content_bounds(self, header_height: float = 42.0) -> arcade.Rect:
        """Return the panel body rectangle.

        Parameters
        ----------
        header_height
            Height reserved for the title and drag controls.

        Returns
        -------
        arcade.Rect
            Panel body bounds.
        """
        return arcade.LBWH(
            self.bounds.left,
            self.bounds.bottom,
            self.bounds.width,
            max(0.0, self.bounds.height - header_height),
        )


@dataclass(frozen=True, slots=True)
class ScrollableText:
    """Describe scrollable text content and its line spacing."""

    key: str
    bounds: arcade.Rect
    lines: tuple[str, ...]
    line_spacing: float

    def scroll_limit(self) -> float:
        """Return the maximum vertical scroll offset.

        Returns
        -------
        float
            Nonnegative content overflow.
        """
        return max(0.0, len(self.lines) * self.line_spacing - self.bounds.height)


class CommonUiComponent:
    """Provide CommonUiComponent UI behavior."""
    def _draw_panel(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        """Draw panel.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        fill_color
            Arcade-compatible color.
        """
        self._draw_rounded_rect(
            bounds,
            fill_color or self.theme.panel_background,
            self.theme.panel_border,
            self.config.layout.panel_radius,
            2,
        )
    @contextmanager
    def _ui_clip(self, bounds: arcade.Rect):
        """Clip drawing to UI bounds while preserving prior GL state.

        Parameters
        ----------
        bounds
            Logical-coordinate clipping rectangle.

        Yields
        ------
        None
            Control while clipping is active.
        """
        try:
            arcade.get_window()
        except (AttributeError, RuntimeError):
            yield
            return
        with self._painter.clip(bounds):
            yield
    def _framebuffer_scale(self) -> tuple[float, float]:
        """Return logical-to-framebuffer scale factors.

        Returns
        -------
        tuple[float, float]
            Horizontal and vertical framebuffer scales.
        """
        return self._painter.framebuffer_scale()
    def _draw_floating_panel(
        self,
        bounds: arcade.Rect,
        title: str,
        key: str,
        *,
        icon_name: str | None = None,
        show_close: bool = True,
        body_top_padding: float = 24.0,
        panel_fill: arcade.Color | tuple[int, ...] | None = None,
        title_icon_size: float = 20.0,
    ) -> arcade.Rect:
        """Draw floating panel.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        title
            Text displayed by the UI.
        key
            Stable identifier used by the UI.
        icon_name
            Stable identifier used by the UI.
        show_close
            Whether the corresponding behavior is enabled.
        body_top_padding
            Value used by the operation.
        panel_fill
            Value used by the operation.
        title_icon_size
            Value used by the operation.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        frame = PanelFrame(key=key, title=title, bounds=bounds)
        self._draw_rounded_rect(
            frame.bounds,
            panel_fill or self.theme.panel_background_alt,
            self.theme.panel_border,
            14,
            1.5,
        )
        self._control_hitboxes[f"{frame.key}_panel"] = frame.bounds

        header_height = 58.0 if title else 0.0
        if header_height > 0.0:
            header = arcade.LBWH(
                bounds.left + 1.5,
                bounds.top - header_height,
                bounds.width - 3.0,
                header_height - 1.5,
            )
            self._draw_rounded_rect_fill(
                header,
                self.theme.panel_background,
                max(0.0, 14.0 - 1.5),
            )
            title_x = bounds.left + 28.0
            if icon_name is not None:
                icon_bounds = arcade.LBWH(
                    bounds.left + 26.0,
                    bounds.top - 28.0 - title_icon_size / 2.0,
                    title_icon_size,
                    title_icon_size,
                )
                self._draw_icon(icon_bounds, icon_name, f"{key}_title_icon")
                title_x = icon_bounds.right + 12.0
            self._draw_text(
                f"{key}_panel_title",
                title,
                title_x,
                bounds.top - 33.0,
                self.theme.text_primary,
                19,
                bold=True,
                anchor_y="center",
            )
            self._control_hitboxes[f"{key}_drag"] = header
        else:
            self._control_hitboxes[f"{key}_drag"] = arcade.LBWH(
                bounds.left,
                bounds.top - 44.0,
                bounds.width,
                44.0,
            )

        if show_close:
            close_bounds = arcade.LBWH(bounds.right - 48, bounds.top - 42, 28, 28)
            self._control_hitboxes[f"{key}_close"] = close_bounds
            self._draw_panel_close_button(close_bounds, key)

        body_top = bounds.top - header_height - body_top_padding
        body = arcade.LBWH(
            bounds.left + 22.0,
            bounds.bottom + 18.0,
            max(0.0, bounds.width - 44.0),
            max(0.0, body_top - bounds.bottom - 18.0),
        )
        self._control_hitboxes[f"{key}_body"] = body
        return body
    def _draw_icon_button(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
        *,
        active: bool,
    ) -> None:
        """Draw icon button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        icon_name
            Stable identifier used by the UI.
        key
            Stable identifier used by the UI.
        active
            Whether the corresponding behavior is enabled.
        """
        button = IconButton(
            key=key,
            icon_name=icon_name,
            bounds=bounds,
            active=active,
        )
        button.draw(
            self._painter,
            self._interaction,
            fill_color=self.theme.panel_background_alt,
            active_color=self.theme.accent_soft,
            border_color=(
                self.theme.accent_soft
                if active
                else self.theme.panel_background_alt
            ),
        )
        icon_size = 26.0
        self._draw_icon(
            arcade.LBWH(
                bounds.center_x - icon_size / 2.0,
                bounds.center_y - icon_size / 2.0,
                icon_size,
                icon_size,
            ),
            icon_name,
            key,
        )
    def _draw_panel_close_button(self, bounds: arcade.Rect, key: str) -> None:
        """Draw panel close button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.
        """
        self._draw_icon(bounds, "kill", f"{key}_close_icon")
    def _draw_icon(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
    ) -> None:
        """Draw icon.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        icon_name
            Stable identifier used by the UI.
        key
            Stable identifier used by the UI.
        """
        texture = self._icon_texture(icon_name)
        if texture is not None and self._draw_icon_texture(bounds, texture):
            return
        if self._draw_icon_sprite(bounds, icon_name, texture):
            return
        fallback = {
            "search": "?",
            "analytics": "#",
            "tune": "=",
            "brain": "@",
            "kill": "x",
            "globe": "O",
        }.get(icon_name, "*")
        self._draw_text(
            f"icon_fallback_{key}",
            fallback,
            bounds.center_x,
            bounds.center_y,
            self.theme.text_primary,
            max(11.0, bounds.height * 0.58),
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
    def _draw_icon_texture(self, bounds: arcade.Rect, texture: object) -> bool:
        """Draw an icon through supported Arcade texture APIs.

        Parameters
        ----------
        bounds
            Destination rectangle.
        texture
            Arcade texture-like object.

        Returns
        -------
        bool
            Whether a compatible draw call succeeded.
        """
        return self._painter.draw_icon_texture(bounds, texture)
    def _draw_icon_sprite(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        texture: object | None,
    ) -> bool:
        """Draw an icon through the sprite compatibility path.

        Parameters
        ----------
        bounds
            Destination rectangle.
        icon_name
            Icon asset name.
        texture
            Previously loaded texture, if available.

        Returns
        -------
        bool
            Whether the sprite rendered successfully.
        """
        return self._painter.draw_icon_sprite(bounds, icon_name, texture)
    def _icon_sprite(self, icon_name: str, texture: object | None) -> object | None:
        """Return a cached sprite for an icon.

        Parameters
        ----------
        icon_name
            Icon asset name.
        texture
            Previously loaded texture, if available.

        Returns
        -------
        object or None
            Cached sprite or ``None`` when unavailable.
        """
        return self._painter.icon_sprite(icon_name, texture)
    def _icon_texture(self, icon_name: str) -> object | None:
        """Return a cached texture for an icon.

        Parameters
        ----------
        icon_name
            Icon asset name.

        Returns
        -------
        object or None
            Cached texture or ``None`` when unavailable.
        """
        return self._painter.icon_texture(icon_name)
    def _icon_path(self, icon_name: str) -> Path:
        """Return the absolute asset path for an icon.

        Parameters
        ----------
        icon_name
            Icon asset name.

        Returns
        -------
        pathlib.Path
            Absolute PNG path.
        """
        return self._painter.icon_path(icon_name)
    def _draw_status_chip(self, bounds: arcade.Rect, label: str) -> None:
        """Draw status chip.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        label
            Text displayed by the UI.
        """
        self._draw_rounded_rect(bounds, (188, 237, 220), (188, 237, 220), 999, 1)
        self._draw_text(
            f"status_chip_{label}",
            label,
            bounds.center_x,
            bounds.center_y,
            self.theme.accent,
            9,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
    def _rect_intersects(self, first: arcade.Rect, second: arcade.Rect) -> bool:
        """Return rect intersects.

        Parameters
        ----------
        first
            Value used by the operation.
        second
            Value used by the operation.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        return not (
            first.right < second.left
            or first.left > second.right
            or first.top < second.bottom
            or first.bottom > second.top
        )
    def _draw_metric_row(
        self,
        key: str,
        label: str,
        value: str,
        x: float,
        y: float,
        width: float,
        *,
        value_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> float:
        """Draw a responsive, height-aware metric row shared by UI cards.

        Parameters
        ----------
        key
            Stable identifier used by the UI.
        label
            Text displayed by the UI.
        value
            Value used by the operation.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        width
            Requested logical size.
        value_color
            Value used by the operation.

        Returns
        -------
        float
            Vertical space consumed by the responsive row.
        """
        (
            label_lines,
            value_lines,
            label_bounds,
            value_bounds,
            row_height,
        ) = self._metric_row_layout(label, value, x, y, width)
        self._draw_text(
            f"{key}_label",
            "\n".join(label_lines),
            label_bounds.left,
            label_bounds.top,
            self.theme.text_muted,
            10,
            width=label_bounds.width,
            multiline=True,
            align="left",
            anchor_y="top",
        )
        self._draw_text(
            f"{key}_value",
            "\n".join(value_lines),
            value_bounds.left,
            value_bounds.top,
            value_color or self.theme.text_primary,
            12,
            width=value_bounds.width,
            multiline=True,
            align=("left" if value_bounds.left == x else "right"),
            anchor_y="top",
        )
        return row_height

    def _metric_row_layout(
        self,
        label: str,
        value: str,
        x: float,
        y: float,
        width: float,
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        arcade.Rect,
        arcade.Rect,
        float,
    ]:
        """Measure a responsive card row and wrap both text columns."""
        available_width = max(1.0, float(width))
        gap = 16.0
        if available_width < 180.0:
            label_width = available_width
            value_width = available_width
            label_lines = tuple(
                self._wrap_line(label, label_width, font_size=10.0)
            )
            value_lines = tuple(
                self._wrap_line(value, value_width, font_size=12.0)
            )
            label_height = max(1, len(label_lines)) * 13.0
            value_top = y - label_height - 3.0
            value_height = max(1, len(value_lines)) * 15.0
            row_height = label_height + 3.0 + value_height + 7.0
            return (
                label_lines,
                value_lines,
                arcade.LBWH(
                    x,
                    y - label_height,
                    label_width,
                    label_height,
                ),
                arcade.LBWH(
                    x,
                    value_top - value_height,
                    value_width,
                    value_height,
                ),
                row_height,
            )

        label_width = max(1.0, (available_width - gap) * 0.44)
        value_left = round(x + label_width + gap)
        value_width = max(1.0, x + available_width - value_left)
        label_lines = tuple(
            self._wrap_line(label, label_width, font_size=10.0)
        )
        value_lines = tuple(
            self._wrap_line(value, value_width, font_size=12.0)
        )
        text_height = max(
            max(1, len(label_lines)) * 13.0,
            max(1, len(value_lines)) * 15.0,
        )
        row_height = max(25.0, text_height + 7.0)
        return (
            label_lines,
            value_lines,
            arcade.LBWH(
                x,
                y - text_height,
                label_width,
                text_height,
            ),
            arcade.LBWH(
                value_left,
                y - text_height,
                value_width,
                text_height,
            ),
            row_height,
        )
    def _draw_progress_bar(
        self,
        bounds: arcade.Rect,
        ratio: float,
        *,
        fill_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        """Draw progress bar.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        ratio
            Value used by the operation.
        fill_color
            Arcade-compatible color.
        """
        epsilon = 1e-6
        ratio = max(0.0, min(1.0, ratio))
        radius = bounds.height / 2.0
        self._draw_rounded_rect_fill(bounds, (222, 224, 255), radius)
        fill_width = bounds.width * ratio
        if fill_width <= epsilon:
            return
        fill = arcade.LBWH(bounds.left, bounds.bottom, fill_width, bounds.height)
        fill_radius = min(radius, fill.width / 2.0, fill.height / 2.0)
        self._draw_rounded_rect_fill(fill, fill_color or self.theme.accent, fill_radius)
    def _draw_action_button(
        self,
        bounds: arcade.Rect,
        label: str,
        icon_name: str,
        key: str,
        *,
        fill_color: arcade.Color | tuple[int, ...],
        text_color: arcade.Color | tuple[int, ...],
    ) -> None:
        """Draw action button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        label
            Text displayed by the UI.
        icon_name
            Stable identifier used by the UI.
        key
            Stable identifier used by the UI.
        fill_color
            Arcade-compatible color.
        text_color
            Value used by the operation.
        """
        self._draw_rounded_rect(bounds, fill_color, fill_color, 7, 1)
        icon_size = 20.0
        icon_text_gap = 12.0
        fitted_label = self._fit_line(
            label, bounds.width - icon_size - icon_text_gap - 18
        )
        text_width = min(
            max(0.0, bounds.width - icon_size - icon_text_gap - 18),
            len(fitted_label) * 7.0,
        )
        group_width = icon_size + icon_text_gap + text_width
        group_left = bounds.center_x - group_width / 2.0
        icon_bounds = arcade.LBWH(
            group_left,
            bounds.center_y - icon_size / 2.0,
            icon_size,
            icon_size,
        )
        self._draw_icon(icon_bounds, icon_name, f"{key}_icon")
        self._draw_text(
            f"action_button_{key}",
            fitted_label,
            icon_bounds.right + icon_text_gap,
            bounds.center_y,
            text_color,
            12,
            bold=True,
            anchor_x="left",
            anchor_y="center",
        )
    def _draw_icon_text_button(
        self,
        bounds: arcade.Rect,
        label: str,
        key: str,
        *,
        fill_color: arcade.Color | tuple[int, ...] | None,
        size: float,
        y_offset: float = 0.0,
    ) -> None:
        """Draw icon text button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        label
            Text displayed by the UI.
        key
            Stable identifier used by the UI.
        fill_color
            Arcade-compatible color.
        size
            Requested logical size.
        y_offset
            Value used by the operation.
        """
        if fill_color is not None:
            self._draw_rounded_rect(bounds, fill_color, fill_color, 8, 1)
        self._draw_text(
            f"icon_text_button_{key}",
            label,
            bounds.center_x,
            bounds.center_y + y_offset,
            self.theme.accent,
            size,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
    def _draw_card(self, bounds: arcade.Rect, title: str) -> None:
        """Draw card.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        title
            Text displayed by the UI.
        """
        self._draw_rounded_rect(
            bounds,
            self.theme.card_background,
            self.theme.panel_border,
            self.config.layout.card_radius,
            2,
        )
        self._draw_text(
            f"card_title_{title}",
            title,
            bounds.left + 16,
            bounds.top - 24,
            self.theme.text_primary,
            14,
            bold=True,
        )
    def _draw_button(self, bounds: arcade.Rect, label: str, key: str) -> None:
        """Draw button.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        label
            Text displayed by the UI.
        key
            Stable identifier used by the UI.
        """
        self._draw_rounded_rect(
            bounds,
            self.theme.panel_background,
            self.theme.panel_border,
            8,
            1.5,
        )
        self._draw_text(
            f"button_{key}",
            self._fit_line(label, bounds.width - 8),
            bounds.center_x,
            bounds.center_y,
            self.theme.text_primary,
            14,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
    def _contains_hitbox(self, key: str, x: float, y: float) -> bool:
        """Return whether a registered hitbox contains a point.

        Parameters
        ----------
        key
            Registered hitbox key.
        x, y
            Pointer coordinates.

        Returns
        -------
        bool
            Whether the pointer lies inside the hitbox.
        """
        return self._interaction.contains(key, x, y)
    def _contains_bounds(self, bounds: arcade.Rect, x: float, y: float) -> bool:
        """Return whether bounds contain a point.

        Parameters
        ----------
        bounds
            Rectangle to inspect.
        x, y
            Pointer coordinates.

        Returns
        -------
        bool
            Whether the point lies inside the rectangle.
        """
        return rect_contains(bounds, x, y)
    def _draw_text(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        color: arcade.Color | tuple[int, ...],
        size: float,
        *,
        bold: bool = False,
        width: float | None = None,
        multiline: bool = False,
        align: str = "left",
        anchor_x: str = "left",
        anchor_y: str = "baseline",
    ) -> None:
        """Draw cached UI text.

        Parameters
        ----------
        key
            Stable text cache key.
        text
            Text to display.
        x, y
            Text anchor coordinates.
        color
            Arcade-compatible text color.
        size
            Font size.
        bold, width, multiline, align, anchor_x, anchor_y
            Arcade text presentation options.
        """
        self._painter.draw_text(
            key,
            text,
            x,
            y,
            color,
            size,
            bold=bold,
            width=width,
            multiline=multiline,
            align=align,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
    def _draw_scrollable_lines(
        self,
        key: str,
        card_bounds: arcade.Rect,
        lines: Sequence[str],
        *,
        line_spacing: float,
        first_line_color: arcade.Color | tuple[int, ...],
        body_color: arcade.Color | tuple[int, ...],
        first_line_bold: bool = False,
    ) -> None:
        """Draw scrollable lines.

        Parameters
        ----------
        key
            Stable identifier used by the UI.
        card_bounds
            Value used by the operation.
        lines
            Value used by the operation.
        line_spacing
            Requested logical size.
        first_line_color
            Value used by the operation.
        body_color
            Value used by the operation.
        first_line_bold
            Value used by the operation.
        """
        content = self._card_content_bounds(card_bounds)
        self._draw_scrollable_lines_in_bounds(
            key,
            content,
            lines,
            line_spacing=line_spacing,
            first_line_color=first_line_color,
            body_color=body_color,
            first_line_bold=first_line_bold,
        )
    def _draw_scrollable_lines_in_bounds(
        self,
        key: str,
        content: arcade.Rect,
        lines: Sequence[str],
        *,
        line_spacing: float,
        first_line_color: arcade.Color | tuple[int, ...],
        body_color: arcade.Color | tuple[int, ...],
        first_line_bold: bool = False,
        wrap_lines: bool = False,
        draw_ethogram_markers: bool = False,
    ) -> None:
        """Draw scrollable lines in bounds.

        Parameters
        ----------
        key
            Stable identifier used by the UI.
        content
            Value used by the operation.
        lines
            Value used by the operation.
        line_spacing
            Requested logical size.
        first_line_color
            Value used by the operation.
        body_color
            Value used by the operation.
        first_line_bold
            Value used by the operation.
        wrap_lines
            Whether logical lines should wrap within the available width.
        draw_ethogram_markers
            Value used by the operation.
        """
        visual_lines = (
            self._wrapped_scrollable_lines(
                lines,
                content.width - 12.0,
                draw_ethogram_markers=draw_ethogram_markers,
                first_line_bold=first_line_bold,
            )
            if wrap_lines
            else [
                (
                    line,
                    line_index == 0,
                    None,
                    0.0,
                )
                for line_index, line in enumerate(lines)
            ]
        )
        scroll_limit = max(
            0.0,
            len(visual_lines) * line_spacing - content.height,
        )
        scroll_offset = max(
            0.0,
            min(scroll_limit, self._scroll_offsets.get(key, 0.0)),
        )
        self._scroll_offsets[key] = scroll_offset
        self._scroll_limits[key] = scroll_limit
        self._scroll_regions[key] = content

        first_visible = max(
            0,
            ceil((scroll_offset - 12.0) / line_spacing),
        )
        visible_stop = min(
            len(visual_lines),
            floor(
                (content.height - 12.0 + scroll_offset) / line_spacing
            )
            + 1,
        )
        for line_index in range(first_visible, visible_stop):
            line, is_first_line, marker_color, x = visual_lines[line_index]
            y = content.top - 12 - line_index * line_spacing + scroll_offset
            if marker_color is not None:
                arcade.draw_circle_filled(
                    content.left + 8.0,
                    y + 4.0,
                    6.0,
                    marker_color,
                )
            self._draw_text(
                f"{key}_line_{line_index}",
                (
                    line
                    if wrap_lines
                    else self._fit_line(
                        line,
                        content.width - (12 if scroll_limit > 0 else 0),
                    )
                ),
                content.left + x,
                y,
                first_line_color if is_first_line else body_color,
                12,
                bold=first_line_bold and is_first_line,
            )

        if scroll_limit > 0.0:
            self._draw_scrollbar(content, scroll_offset, scroll_limit)
    def _wrapped_scrollable_lines(
        self,
        lines: Sequence[str],
        width: float,
        *,
        draw_ethogram_markers: bool,
        first_line_bold: bool = False,
    ) -> tuple[
        tuple[
            str,
            bool,
            tuple[int, int, int] | None,
            float,
        ],
        ...,
    ]:
        """Return wrapped scrollable lines.

        Parameters
        ----------
        lines
            Value used by the operation.
        width
            Requested logical size.
        draw_ethogram_markers
            Value used by the operation.
        first_line_bold
            Whether the first logical line uses the bold text metrics.

        Returns
        -------
        tuple[tuple[str, bool, tuple[int, int, int] | None, float], ...]
            Computed collection.
        """
        logical_lines = lines if isinstance(lines, tuple) else tuple(lines)
        cache_key = (
            logical_lines,
            round(float(width), 2),
            bool(draw_ethogram_markers),
            bool(first_line_bold),
        )
        cached_lines = self._painter.wrapped_line_block_cache.get(cache_key)
        if cached_lines is not None:
            return cached_lines

        visual_lines: list[
            tuple[str, bool, tuple[int, int, int] | None, float]
        ] = []
        base_x = 0.0
        for logical_index, raw_line in enumerate(logical_lines):
            marker_color: tuple[int, int, int] | None = None
            line = raw_line
            marker_indent = 0.0
            if draw_ethogram_markers and line:
                marker_color = self._ethogram_marker_color(line[0])
                if marker_color is not None:
                    line = line[1:].lstrip()
                    marker_indent = 20.0

            available_width = max(24.0, width - marker_indent)
            wrapped = self._wrap_line(
                line,
                available_width,
                bold=first_line_bold and logical_index == 0,
            )
            if not wrapped:
                wrapped = [""]
            for wrapped_index, wrapped_line in enumerate(wrapped):
                visual_lines.append(
                    (
                        wrapped_line,
                        logical_index == 0,
                        marker_color if wrapped_index == 0 else None,
                        base_x + marker_indent,
                    )
                )
        wrapped_block = tuple(visual_lines)
        if len(self._painter.wrapped_line_block_cache) >= 16:
            self._painter.wrapped_line_block_cache.clear()
        self._painter.wrapped_line_block_cache[cache_key] = wrapped_block
        return wrapped_block
    @staticmethod
    def _ethogram_marker_color(
        marker: str,
    ) -> tuple[int, int, int] | None:
        """Return ethogram marker color.

        Parameters
        ----------
        marker
            Value used by the operation.

        Returns
        -------
        tuple[int, int, int] | None
            Computed collection.
        """
        return {
            "🟢": (0, 210, 72),
            "🔴": (255, 55, 65),
            "⚪": (150, 160, 170),
        }.get(marker)
    def _wrap_line(
        self,
        text: str,
        width: float,
        *,
        font_size: float = 12.0,
        bold: bool = False,
    ) -> list[str]:
        """Wrap line.

        Parameters
        ----------
        text
            Text displayed by the UI.
        width
            Requested logical size.
        font_size
            Font size used to measure the visual line length.
        bold
            Whether to use the bold font metrics.

        Returns
        -------
        list[str]
            Computed collection.
        """
        return list(
            self._painter.wrap_text(
                text,
                width,
                font_size,
                bold=bold,
            )
        )
    def _card_content_bounds(self, bounds: arcade.Rect) -> arcade.Rect:
        """Return card content bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        bottom = bounds.bottom + 12
        top = bounds.top - 42
        return arcade.LBWH(
            bounds.left + 16,
            bottom,
            max(0.0, bounds.width - 32),
            max(0.0, top - bottom),
        )
    def _draw_scrollbar(
        self, bounds: arcade.Rect, scroll_offset: float, scroll_limit: float
    ) -> None:
        """Draw scrollbar.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        scroll_offset
            Value used by the operation.
        scroll_limit
            Value used by the operation.
        """
        track_width = 3
        track_left = bounds.right - track_width
        arcade.draw_lrbt_rectangle_filled(
            track_left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.theme.panel_border,
        )
        visible_ratio = bounds.height / (bounds.height + scroll_limit)
        thumb_height = max(18.0, bounds.height * visible_ratio)
        travel = max(0.0, bounds.height - thumb_height)
        thumb_top = bounds.top - travel * (scroll_offset / scroll_limit)
        arcade.draw_lrbt_rectangle_filled(
            track_left,
            bounds.right,
            thumb_top - thumb_height,
            thumb_top,
            self.theme.accent,
        )
    def _fit_line(self, text: str, width: float) -> str:
        """Fit line.

        Parameters
        ----------
        text
            Text displayed by the UI.
        width
            Requested logical size.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        max_chars = max(4, int(width / 7.0))
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3]}..."
    def _draw_rounded_rect(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...],
        border_color: arcade.Color | tuple[int, ...],
        radius: float,
        border_width: float,
    ) -> None:
        """Draw a rounded rectangle with a border.

        Parameters
        ----------
        bounds
            Outer rectangle.
        fill_color, border_color
            Arcade-compatible colors.
        radius
            Outer corner radius.
        border_width
            Border thickness.
        """
        self._painter.draw_rounded_rect(
            bounds,
            fill_color,
            border_color,
            radius,
            border_width,
        )
    def _draw_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        """Draw a filled rounded rectangle.

        Parameters
        ----------
        bounds
            Rectangle to fill.
        color
            Arcade-compatible fill color.
        radius
            Corner radius.
        """
        self._painter.draw_rounded_rect_fill(bounds, color, radius)
