from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import arcade

from configs.sim_config import SimConfig


@dataclass(slots=True)
class StartMenuLayout:
    window: arcade.Rect
    title_y: float
    subtitle_y: float
    cards_top: float
    left_card: arcade.Rect
    right_card: arcade.Rect


@dataclass(slots=True)
class CardContentLayout:
    badge: arcade.Rect
    icon: arcade.Rect
    title_block: arcade.Rect
    body_block: arcade.Rect


@dataclass(slots=True)
class RibbonLayout:
    points: tuple[tuple[float, float], ...]
    label_x: float
    label_y: float
    label_size: float
    rotation: float


class StartMenuView(arcade.View):
    BACKGROUND = (2, 30, 48)
    CARD_FILL = (43, 48, 70)
    CARD_FILL_DISABLED = (21, 39, 58)
    CARD_BORDER = (64, 72, 96)
    CARD_BORDER_DISABLED = (34, 55, 76)
    TEXT_PRIMARY = (240, 239, 255)
    TEXT_MUTED = (198, 203, 220)
    TEXT_DISABLED = (139, 146, 166)
    ACCENT = (66, 97, 125)
    ACCENT_SOFT = (167, 199, 231)
    SECONDARY_SOFT = (188, 237, 220)
    DISABLED_BADGE = (96, 128, 137)
    RIBBON_FILL = (186, 26, 26)
    RIBBON_TEXT = (255, 255, 255)

    CARD_GAP = 48.0
    CARD_WIDTH = 472.0
    CARD_HEIGHT = 272.0
    ICON_BADGE_SIZE = 80.0
    CARD_INSET_X = 48.0
    CARD_TOP_PADDING = 36.0
    BADGE_TITLE_GAP = 22.0
    TITLE_BLOCK_HEIGHT = 62.0
    TITLE_BODY_GAP = 14.0
    BODY_BLOCK_HEIGHT = 44.0
    CARD_TITLE_SIZE = 22.0
    CARD_BODY_SIZE = 14.0
    BUTTON_HOVER_SCALE = 1.07
    BUTTON_PRESS_SCALE = 0.92
    BUTTON_HOVER_LIFT = 4.0
    BUTTON_PRESS_NUDGE = 2.0

    def __init__(
        self,
        config: SimConfig,
        start_view_factory: Callable[[], arcade.View],
    ) -> None:
        super().__init__()
        self.config = config
        self.start_view_factory = start_view_factory
        self._text_cache: dict[str, arcade.Text] = {}
        self._texture_cache: dict[str, object | None] = {}
        self._sprite_cache: dict[str, object | None] = {}
        self._hovered_button: str | None = None
        self._pressed_button: str | None = None
        self._button_animation: dict[str, float] = {"start": 0.0, "load": 0.0}

    def on_show_view(self) -> None:
        if self.window is not None:
            self.window.background_color = self.BACKGROUND

    def on_draw(self) -> None:
        self.clear()
        layout = self.layout()
        self._draw_background(layout.window)
        self._draw_heading(layout)
        self._draw_card(
            layout.left_card,
            key="start",
            icon_name="play",
            title="Start a new simulation",
            description="Initialize a fresh canvas and watch emergence unfold.",
            disabled=False,
        )
        self._draw_card(
            layout.right_card,
            key="load",
            icon_name="save",
            title="Load an already existing simulation",
            description="Restore a previous system state from your archives.",
            disabled=True,
        )
        self._draw_coming_soon_ribbon(layout.right_card)

    def on_mouse_press(
        self,
        x: int,
        y: int,
        button: int,
        modifiers: int,
    ) -> bool | None:
        if button == arcade.MOUSE_BUTTON_LEFT:
            pressed_button = self._button_key_at(x, y, self.layout())
            if pressed_button is not None:
                self._pressed_button = pressed_button
                self._hovered_button = pressed_button
                return True
        return super().on_mouse_press(x, y, button, modifiers)

    def on_mouse_release(
        self,
        x: int,
        y: int,
        button: int,
        modifiers: int,
    ) -> bool | None:
        if button == arcade.MOUSE_BUTTON_LEFT and self._pressed_button is not None:
            pressed_button = self._pressed_button
            released_button = self._button_key_at(x, y, self.layout())
            self._pressed_button = None
            self._hovered_button = released_button
            if pressed_button == "start" and released_button == "start":
                self._start_simulation()
            return True
        return super().on_mouse_release(x, y, button, modifiers)

    def on_mouse_motion(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> bool | None:
        self._hovered_button = self._button_key_at(x, y, self.layout())
        return super().on_mouse_motion(x, y, dx, dy)

    def on_update(self, delta_time: float) -> None:
        step = min(1.0, max(0.0, delta_time * 12.0))
        for key, current in self._button_animation.items():
            target = 1.0 if key == self._hovered_button or key == self._pressed_button else 0.0
            self._button_animation[key] = current + (target - current) * step

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        start_keys = {
            getattr(arcade.key, "ENTER", -1),
            getattr(arcade.key, "RETURN", -4),
            getattr(arcade.key, "SPACE", -2),
            getattr(arcade.key, "N", -3),
        }
        if symbol in start_keys:
            self._start_simulation()
            return True
        return super().on_key_press(symbol, modifiers)

    def layout(self) -> StartMenuLayout:
        width, height = self._window_size()
        window = arcade.LBWH(0, 0, width, height)
        title_size = self._title_size(width)
        title_height = title_size * 1.18
        title_subtitle_gap = 20.0 if height < 620 else 30.0
        subtitle_height = 70.0 if height < 620 else 76.0
        subtitle_card_gap = 60.0 if height < 620 else 96.0

        card_width = min(
            self.CARD_WIDTH,
            max(300.0, (width - 96.0 - self.CARD_GAP) / 2.0),
        )
        card_height_ratio = 0.48 if height < 620 else 0.32
        card_height = min(self.CARD_HEIGHT, max(232.0, height * card_height_ratio))
        total_cards_width = card_width * 2.0 + self.CARD_GAP
        left = width * 0.5 - total_cards_width * 0.5
        stack_height = (
            title_height
            + title_subtitle_gap
            + subtitle_height
            + subtitle_card_gap
            + card_height
        )
        min_bottom = 12.0 if height < 620 else 36.0
        bottom = max(min_bottom, (height - stack_height) / 2.0)
        cards_top = bottom + card_height
        subtitle_y = cards_top + subtitle_card_gap + subtitle_height / 2.0
        title_y = (
            subtitle_y
            + subtitle_height / 2.0
            + title_subtitle_gap
            + title_height / 2.0
        )

        if title_y > height - 42.0:
            shift = title_y - (height - 42.0)
            bottom = max(4.0, bottom - shift)
            cards_top = bottom + card_height
            subtitle_y = cards_top + subtitle_card_gap + subtitle_height / 2.0
            title_y = height - 42.0

        left_card = arcade.LBWH(left, bottom, card_width, card_height)
        right_card = arcade.LBWH(
            left_card.right + self.CARD_GAP,
            bottom,
            card_width,
            card_height,
        )
        return StartMenuLayout(
            window=window,
            title_y=title_y,
            subtitle_y=subtitle_y,
            cards_top=cards_top,
            left_card=left_card,
            right_card=right_card,
        )

    def _title_size(self, width: float) -> float:
        return min(62.0, max(42.0, width / 20.5))

    def _window_size(self) -> tuple[int, int]:
        if self.window is None:
            return self.config.display.width, self.config.display.height

        get_size = getattr(self.window, "get_size", None)
        if callable(get_size):
            width, height = get_size()
            return int(width), int(height)

        width = getattr(self.window, "width", self.config.display.width)
        height = getattr(self.window, "height", self.config.display.height)
        return int(width), int(height)

    def _start_simulation(self) -> None:
        if self.window is None:
            return
        view = self.start_view_factory()
        self.window.show_view(view)
        on_resize = getattr(view, "on_resize", None)
        if callable(on_resize):
            width, height = self._window_size()
            on_resize(width, height)

    def _draw_background(self, bounds: arcade.Rect) -> None:
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.BACKGROUND,
        )

    def _draw_heading(self, layout: StartMenuLayout) -> None:
        title_size = self._title_size(layout.window.width)
        self._draw_text(
            "menu_title",
            "Neat Game Of Life",
            layout.window.center_x,
            layout.title_y,
            self.TEXT_PRIMARY,
            title_size,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )
        self._draw_text(
            "menu_subtitle",
            (
                "Explore the beauty of emergent systems and witness the evolution "
                "of life in a serene digital laboratory, where complex patterns "
                "arise from simple rules."
            ),
            layout.window.center_x,
            layout.subtitle_y,
            self.TEXT_MUTED,
            16,
            width=min(650.0, max(320.0, layout.window.width - 96.0)),
            multiline=True,
            align="center",
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_card(
        self,
        bounds: arcade.Rect,
        *,
        key: str,
        icon_name: str,
        title: str,
        description: str,
        disabled: bool,
    ) -> None:
        fill = self.CARD_FILL_DISABLED if disabled else self.CARD_FILL
        border = self.CARD_BORDER_DISABLED if disabled else self.CARD_BORDER
        title_color = self.TEXT_DISABLED if disabled else self.TEXT_PRIMARY
        body_color = self.TEXT_DISABLED if disabled else self.TEXT_MUTED
        content = self._card_content_layout(bounds)
        badge_fill, badge_border = self._button_colors(disabled, key)
        visual_badge = self._button_visual_bounds(content.badge, key)
        visual_icon = self._button_icon_bounds(content.icon, visual_badge, key)

        self._draw_rounded_rect(bounds, fill, border, 16.0, 1.5)
        self._draw_rounded_rect(visual_badge, badge_fill, badge_border, 999.0, 1.0)
        self._draw_icon(
            visual_icon,
            icon_name,
            key,
            disabled=disabled,
        )
        self._draw_text(
            f"{key}_title",
            title,
            content.title_block.center_x,
            content.title_block.center_y,
            title_color,
            self.CARD_TITLE_SIZE,
            bold=True,
            width=content.title_block.width,
            multiline=True,
            align="center",
            anchor_x="center",
            anchor_y="center",
        )
        self._draw_text(
            f"{key}_description",
            description,
            content.body_block.center_x,
            content.body_block.center_y,
            body_color,
            self.CARD_BODY_SIZE,
            width=content.body_block.width,
            multiline=True,
            align="center",
            anchor_x="center",
            anchor_y="center",
        )

    def _card_content_layout(self, bounds: arcade.Rect) -> CardContentLayout:
        badge_size = min(
            self.ICON_BADGE_SIZE,
            max(64.0, bounds.height * 0.30),
        )
        top_padding = min(
            self.CARD_TOP_PADDING,
            max(28.0, bounds.height * 0.13),
        )
        badge_title_gap = min(
            self.BADGE_TITLE_GAP,
            max(16.0, bounds.height * 0.08),
        )
        title_block_height = min(
            self.TITLE_BLOCK_HEIGHT,
            max(54.0, bounds.height * 0.23),
        )
        title_body_gap = min(
            self.TITLE_BODY_GAP,
            max(12.0, bounds.height * 0.05),
        )
        body_block_height = min(
            self.BODY_BLOCK_HEIGHT,
            max(38.0, bounds.height * 0.17),
        )
        inset_x = min(
            self.CARD_INSET_X,
            max(34.0, bounds.width * 0.11),
        )
        text_width = max(210.0, bounds.width - inset_x * 2.0)

        badge = arcade.LBWH(
            bounds.center_x - badge_size / 2.0,
            bounds.top - top_padding - badge_size,
            badge_size,
            badge_size,
        )
        title_top = badge.bottom - badge_title_gap
        title_block = arcade.LBWH(
            bounds.center_x - text_width / 2.0,
            title_top - title_block_height,
            text_width,
            title_block_height,
        )
        body_top = title_block.bottom - title_body_gap
        body_block = arcade.LBWH(
            bounds.center_x - text_width / 2.0,
            body_top - body_block_height,
            text_width,
            body_block_height,
        )

        icon_size = min(34.0, max(28.0, badge_size * 0.42))
        icon = arcade.LBWH(
            badge.center_x - icon_size / 2.0,
            badge.center_y - icon_size / 2.0,
            icon_size,
            icon_size,
        )
        return CardContentLayout(
            badge=badge,
            icon=icon,
            title_block=title_block,
            body_block=body_block,
        )

    def _button_key_at(
        self,
        x: float,
        y: float,
        layout: StartMenuLayout,
    ) -> str | None:
        if self._contains(self._card_content_layout(layout.left_card).badge, x, y):
            return "start"
        if self._contains(self._card_content_layout(layout.right_card).badge, x, y):
            return "load"
        return None

    def _button_visual_bounds(self, bounds: arcade.Rect, key: str) -> arcade.Rect:
        progress = self._button_animation.get(key, 0.0)
        is_pressed = self._pressed_button == key
        scale = 1.0 + (self.BUTTON_HOVER_SCALE - 1.0) * progress
        lift = self.BUTTON_HOVER_LIFT * progress
        if is_pressed:
            scale *= self.BUTTON_PRESS_SCALE
            lift -= self.BUTTON_PRESS_NUDGE
        return self._scaled_bounds(bounds, scale, offset_y=lift)

    def _button_icon_bounds(
        self,
        icon: arcade.Rect,
        visual_badge: arcade.Rect,
        key: str,
    ) -> arcade.Rect:
        progress = self._button_animation.get(key, 0.0)
        is_pressed = self._pressed_button == key
        scale = 1.0 + 0.05 * progress
        if is_pressed:
            scale *= 0.94
        width = icon.width * scale
        height = icon.height * scale
        center_y = visual_badge.center_y - (
            self.BUTTON_PRESS_NUDGE if is_pressed else 0.0
        )
        return arcade.LBWH(
            visual_badge.center_x - width / 2.0,
            center_y - height / 2.0,
            width,
            height,
        )

    def _button_colors(
        self,
        disabled: bool,
        key: str,
    ) -> tuple[arcade.Color | tuple[int, ...], arcade.Color | tuple[int, ...]]:
        progress = self._button_animation.get(key, 0.0)
        is_pressed = self._pressed_button == key
        fill = self.DISABLED_BADGE if disabled else self.SECONDARY_SOFT
        border = fill
        hover_target = (119, 153, 162) if disabled else (207, 255, 239)
        fill = self._mix_color(fill, hover_target, 0.36 * progress)
        border = self._mix_color(border, hover_target, 0.54 * progress)
        if is_pressed:
            press_target = (72, 100, 108) if disabled else self.ACCENT_SOFT
            fill = self._mix_color(fill, press_target, 0.45)
            border = self._mix_color(border, press_target, 0.35)
        return fill, border

    def _scaled_bounds(
        self,
        bounds: arcade.Rect,
        scale: float,
        *,
        offset_y: float = 0.0,
    ) -> arcade.Rect:
        width = bounds.width * scale
        height = bounds.height * scale
        return arcade.LBWH(
            bounds.center_x - width / 2.0,
            bounds.center_y - height / 2.0 + offset_y,
            width,
            height,
        )

    def _mix_color(
        self,
        start: arcade.Color | tuple[int, ...],
        end: arcade.Color | tuple[int, ...],
        amount: float,
    ) -> tuple[int, ...]:
        amount = max(0.0, min(1.0, amount))
        channels = min(len(start), len(end))
        return tuple(
            round(
                float(start[index])
                + (float(end[index]) - float(start[index])) * amount
            )
            for index in range(channels)
        )

    def _draw_coming_soon_ribbon(self, bounds: arcade.Rect) -> None:
        ribbon = self._ribbon_layout(bounds)
        arcade.draw_polygon_filled(ribbon.points, self.RIBBON_FILL)
        self._draw_text(
            "load_ribbon",
            "COMING SOON",
            ribbon.label_x,
            ribbon.label_y,
            self.RIBBON_TEXT,
            ribbon.label_size,
            bold=True,
            anchor_x="center",
            anchor_y="center",
            rotation=ribbon.rotation,
        )

    def _ribbon_layout(self, bounds: arcade.Rect) -> RibbonLayout:
        width = min(112.0, max(92.0, bounds.width * 0.24))
        height = min(96.0, max(78.0, bounds.height * 0.34))
        points = (
            (bounds.right - width, bounds.top),
            (bounds.right, bounds.top),
            (bounds.right, bounds.top - height),
        )
        return RibbonLayout(
            points=points,
            label_x=bounds.right - width * 0.38,
            label_y=bounds.top - height * 0.40,
            label_size=7.0,
            rotation=-45.0,
        )

    def _draw_icon(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
        *,
        disabled: bool,
    ) -> None:
        texture = self._icon_texture(icon_name)
        if texture is not None and self._draw_icon_texture(bounds, texture):
            return
        if self._draw_icon_sprite(bounds, icon_name, texture):
            return
        fallback = ">" if icon_name == "play" else "#"
        self._draw_text(
            f"{key}_icon_fallback",
            fallback,
            bounds.center_x,
            bounds.center_y,
            self.TEXT_DISABLED if disabled else self.ACCENT,
            24,
            bold=True,
            anchor_x="center",
            anchor_y="center",
        )

    def _draw_icon_texture(self, bounds: arcade.Rect, texture: object) -> bool:
        draw_texture_rectangle = getattr(arcade, "draw_texture_rectangle", None)
        if draw_texture_rectangle is not None:
            try:
                draw_texture_rectangle(
                    bounds.center_x,
                    bounds.center_y,
                    bounds.width,
                    bounds.height,
                    texture,
                )
                return True
            except TypeError:
                pass

        draw_texture_rect = getattr(arcade, "draw_texture_rect", None)
        if draw_texture_rect is not None:
            try:
                draw_texture_rect(texture, bounds)
                return True
            except TypeError:
                pass
        return False

    def _draw_icon_sprite(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        texture: object | None,
    ) -> bool:
        sprite = self._icon_sprite(icon_name, texture)
        if sprite is None:
            return False
        try:
            sprite.center_x = bounds.center_x
            sprite.center_y = bounds.center_y
            sprite.width = bounds.width
            sprite.height = bounds.height
            sprite.draw()
        except (AttributeError, TypeError):
            return False
        return True

    def _icon_sprite(self, icon_name: str, texture: object | None) -> object | None:
        if icon_name in self._sprite_cache:
            return self._sprite_cache[icon_name]
        sprite_cls = getattr(arcade, "Sprite", None)
        if sprite_cls is None:
            self._sprite_cache[icon_name] = None
            return None

        sprite = None
        if texture is not None:
            try:
                sprite = sprite_cls(texture=texture)
            except TypeError:
                sprite = None
        if sprite is None:
            try:
                sprite = sprite_cls(str(self._icon_path(icon_name)))
            except (TypeError, FileNotFoundError):
                sprite = None

        self._sprite_cache[icon_name] = sprite
        return sprite

    def _icon_texture(self, icon_name: str) -> object | None:
        if icon_name in self._texture_cache:
            return self._texture_cache[icon_name]
        load_texture = getattr(arcade, "load_texture", None)
        if load_texture is None:
            self._texture_cache[icon_name] = None
            return None
        try:
            texture = load_texture(str(self._icon_path(icon_name)))
        except Exception:
            texture = None
        self._texture_cache[icon_name] = texture
        return texture

    def _icon_path(self, icon_name: str) -> Path:
        return Path(__file__).resolve().parents[1] / "assets" / f"{icon_name}.png"

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
        rotation: float = 0.0,
    ) -> None:
        rx = round(x)
        ry = round(y)
        cached = self._text_cache.get(key)
        if cached is None:
            text_kwargs = {
                "font_name": ("Hanken Grotesk", "Manrope", "JetBrains Mono", "Arial"),
                "bold": bold,
                "width": width,
                "align": align,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "multiline": multiline,
                "rotation": rotation,
            }
            try:
                cached = arcade.Text(text, rx, ry, color, size, **text_kwargs)
            except TypeError:
                text_kwargs.pop("rotation")
                cached = arcade.Text(text, rx, ry, color, size, **text_kwargs)
            self._text_cache[key] = cached
        else:
            cached.text = text
            cached.x = rx
            cached.y = ry
            cached.color = color
            cached.font_size = size
            cached.bold = bold
            cached.width = width
            cached.multiline = multiline
            cached.align = align
            cached.anchor_x = anchor_x
            cached.anchor_y = anchor_y
            if hasattr(cached, "rotation"):
                cached.rotation = rotation
        cached.draw()

    def _draw_rounded_rect(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...],
        border_color: arcade.Color | tuple[int, ...],
        radius: float,
        border_width: float,
    ) -> None:
        self._draw_rounded_rect_fill(bounds, border_color, radius)
        inner = arcade.LBWH(
            bounds.left + border_width,
            bounds.bottom + border_width,
            max(0.0, bounds.width - border_width * 2.0),
            max(0.0, bounds.height - border_width * 2.0),
        )
        self._draw_rounded_rect_fill(inner, fill_color, max(0.0, radius - border_width))

    def _draw_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        if bounds.width <= 0 or bounds.height <= 0:
            return
        radius = min(radius, bounds.width / 2.0, bounds.height / 2.0)
        horizontal_left = bounds.left + radius
        horizontal_right = bounds.right - radius
        vertical_bottom = bounds.bottom + radius
        vertical_top = bounds.top - radius
        if horizontal_left <= horizontal_right:
            arcade.draw_lrbt_rectangle_filled(
                horizontal_left,
                horizontal_right,
                bounds.bottom,
                bounds.top,
                color,
            )
        if vertical_bottom <= vertical_top:
            arcade.draw_lrbt_rectangle_filled(
                bounds.left,
                bounds.right,
                vertical_bottom,
                vertical_top,
                color,
            )
        if radius <= 0:
            return
        arcade.draw_circle_filled(
            bounds.left + radius,
            bounds.bottom + radius,
            radius,
            color,
        )
        arcade.draw_circle_filled(
            bounds.right - radius,
            bounds.bottom + radius,
            radius,
            color,
        )
        arcade.draw_circle_filled(
            bounds.left + radius,
            bounds.top - radius,
            radius,
            color,
        )
        arcade.draw_circle_filled(
            bounds.right - radius,
            bounds.top - radius,
            radius,
            color,
        )

    def _contains(self, bounds: arcade.Rect, x: float, y: float) -> bool:
        return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
