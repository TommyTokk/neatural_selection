"""Reusable Arcade drawing and asset-cache primitives."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import arcade

Color = tuple[int, ...]
DEFAULT_FONTS = ("Hanken Grotesk", "Manrope", "JetBrains Mono", "Arial")
ASSET_DIRECTORY = Path(__file__).resolve().parents[3] / "assets"


class ArcadePainter:
    """Draw cached text, icons, shapes, and clipped UI content."""

    def __init__(self, asset_directory: Path = ASSET_DIRECTORY) -> None:
        """Initialize reusable rendering caches.

        Parameters
        ----------
        asset_directory
            Directory containing icon PNG files.
        """
        self.asset_directory = asset_directory
        self.text_cache: dict[str, arcade.Text] = {}
        self.texture_cache: dict[str, object | None] = {}
        self.sprite_cache: dict[str, object | None] = {}

    def draw_text(
        self,
        key: str,
        text: str,
        x: float,
        y: float,
        color: Color,
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
        """Draw text while reusing the Arcade text object.

        Parameters
        ----------
        key
            Stable cache identifier.
        text
            Text to display.
        x, y
            Text anchor coordinates.
        color
            Arcade-compatible text color.
        size
            Font size in points.
        bold, width, multiline, align, anchor_x, anchor_y, rotation
            Arcade text presentation options.
        """
        rx = round(x)
        ry = round(y)
        cached = self.text_cache.get(key)
        if cached is None:
            kwargs = {
                "font_name": DEFAULT_FONTS,
                "bold": bold,
                "width": width,
                "align": align,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "multiline": multiline,
                "rotation": rotation,
            }
            try:
                cached = arcade.Text(text, rx, ry, color, size, **kwargs)
            except TypeError:
                # Older Arcade versions do not accept rotation at construction.
                kwargs.pop("rotation")
                cached = arcade.Text(text, rx, ry, color, size, **kwargs)
            self.text_cache[key] = cached
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

    def icon_path(self, icon_name: str) -> Path:
        """Return the configured file path for an icon.

        Parameters
        ----------
        icon_name
            Icon name without the PNG extension.

        Returns
        -------
        pathlib.Path
            Absolute icon path.
        """
        return self.asset_directory / f"{icon_name}.png"

    def icon_texture(self, icon_name: str) -> object | None:
        """Load and cache an icon texture when Arcade supports it.

        Parameters
        ----------
        icon_name
            Icon name without the PNG extension.

        Returns
        -------
        object or None
            Cached texture, or ``None`` when loading fails.
        """
        if icon_name in self.texture_cache:
            return self.texture_cache[icon_name]
        load_texture = getattr(arcade, "load_texture", None)
        if load_texture is None:
            self.texture_cache[icon_name] = None
            return None
        try:
            texture = load_texture(str(self.icon_path(icon_name)))
        except Exception:
            texture = None
        self.texture_cache[icon_name] = texture
        return texture

    def icon_sprite(
        self,
        icon_name: str,
        texture: object | None,
    ) -> object | None:
        """Create and cache an icon sprite as a texture fallback.

        Parameters
        ----------
        icon_name
            Icon name without the PNG extension.
        texture
            Previously loaded texture, if available.

        Returns
        -------
        object or None
            Cached sprite, or ``None`` when sprite construction fails.
        """
        if icon_name in self.sprite_cache:
            return self.sprite_cache[icon_name]
        sprite_cls = getattr(arcade, "Sprite", None)
        if sprite_cls is None:
            self.sprite_cache[icon_name] = None
            return None
        sprite = None
        if texture is not None:
            try:
                sprite = sprite_cls(texture=texture)
            except TypeError:
                sprite = None
        if sprite is None:
            try:
                sprite = sprite_cls(str(self.icon_path(icon_name)))
            except (TypeError, FileNotFoundError):
                sprite = None
        self.sprite_cache[icon_name] = sprite
        return sprite

    def draw_icon_texture(self, bounds: arcade.Rect, texture: object) -> bool:
        """Draw a texture across supported Arcade texture APIs.

        Parameters
        ----------
        bounds
            Destination rectangle.
        texture
            Arcade texture-like object.

        Returns
        -------
        bool
            ``True`` when a supported draw call succeeds.
        """
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

    def draw_icon_sprite(
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
            Icon name without the PNG extension.
        texture
            Texture to use when creating the sprite.

        Returns
        -------
        bool
            ``True`` when the sprite draws successfully.
        """
        sprite = self.icon_sprite(icon_name, texture)
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

    def draw_rounded_rect(
        self,
        bounds: arcade.Rect,
        fill_color: Color,
        border_color: Color,
        radius: float,
        border_width: float,
    ) -> None:
        """Draw a rounded rectangle with an inset border.

        Parameters
        ----------
        bounds
            Outer rectangle.
        fill_color, border_color
            Arcade-compatible fill and border colors.
        radius
            Outer corner radius.
        border_width
            Border thickness.
        """
        self.draw_rounded_rect_fill(bounds, border_color, radius)
        inner = arcade.LBWH(
            bounds.left + border_width,
            bounds.bottom + border_width,
            max(0.0, bounds.width - border_width * 2.0),
            max(0.0, bounds.height - border_width * 2.0),
        )
        self.draw_rounded_rect_fill(
            inner,
            fill_color,
            max(0.0, radius - border_width),
        )

    def draw_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: Color,
        radius: float,
    ) -> None:
        """Draw a filled rounded rectangle from rectangles and circles.

        Parameters
        ----------
        bounds
            Rectangle to fill.
        color
            Arcade-compatible fill color.
        radius
            Requested corner radius.
        """
        if bounds.width <= 0 or bounds.height <= 0:
            return
        radius = min(radius, bounds.width / 2.0, bounds.height / 2.0)
        if radius <= 0:
            arcade.draw_lrbt_rectangle_filled(
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
                color,
            )
            return
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
        for center_x, center_y in (
            (bounds.left + radius, bounds.bottom + radius),
            (bounds.right - radius, bounds.bottom + radius),
            (bounds.left + radius, bounds.top - radius),
            (bounds.right - radius, bounds.top - radius),
        ):
            arcade.draw_circle_filled(center_x, center_y, radius, color)

    def framebuffer_scale(self) -> tuple[float, float]:
        """Return logical-to-framebuffer scale factors.

        Returns
        -------
        tuple[float, float]
            Horizontal and vertical framebuffer scales.
        """
        try:
            window = arcade.get_window()
            window_width, window_height = window.get_size()
            framebuffer_width, framebuffer_height = window.get_framebuffer_size()
        except (AttributeError, RuntimeError):
            return 1.0, 1.0
        if window_width <= 0 or window_height <= 0:
            return 1.0, 1.0
        return (
            framebuffer_width / window_width,
            framebuffer_height / window_height,
        )

    @contextmanager
    def clip(
        self,
        bounds: arcade.Rect,
        *,
        inset: float = 0.0,
    ) -> Iterator[None]:
        """Clip drawing to a rectangle and restore prior OpenGL state.

        Parameters
        ----------
        bounds
            Logical-coordinate clipping rectangle.
        inset
            Optional inset applied on all sides.

        Yields
        ------
        None
            Control while the scissor rectangle is active.
        """
        try:
            from pyglet import gl
        except ImportError:
            yield
            return
        clipped = arcade.LBWH(
            bounds.left + inset,
            bounds.bottom + inset,
            max(0.0, bounds.width - inset * 2.0),
            max(0.0, bounds.height - inset * 2.0),
        )
        scale_x, scale_y = self.framebuffer_scale()
        box = (
            round(clipped.left * scale_x),
            round(clipped.bottom * scale_y),
            round(clipped.width * scale_x),
            round(clipped.height * scale_y),
        )
        previous_box = (gl.GLint * 4)()
        was_enabled = bool(gl.glIsEnabled(gl.GL_SCISSOR_TEST))
        gl.glGetIntegerv(gl.GL_SCISSOR_BOX, previous_box)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(*box)
        try:
            yield
        finally:
            # Clipping is nested throughout the UI, so restore—not reset—the
            # previous box and enable state.
            gl.glScissor(*previous_box)
            if not was_enabled:
                gl.glDisable(gl.GL_SCISSOR_TEST)


def mix_color(start: Color, end: Color, amount: float) -> tuple[int, ...]:
    """Blend matching channels from two colors.

    Parameters
    ----------
    start, end
        Colors to interpolate.
    amount
        Blend amount clamped to ``[0, 1]``.

    Returns
    -------
    tuple[int, ...]
        Blended color channels.
    """
    normalized = max(0.0, min(1.0, amount))
    return tuple(
        round(float(start[index]) + (float(end[index]) - float(start[index])) * normalized)
        for index in range(min(len(start), len(end)))
    )
