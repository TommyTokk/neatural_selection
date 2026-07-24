from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import arcade

from configs.sim_config import SimConfig
from src.ui.common.drawing import ArcadePainter, mix_color
from src.ui.common.interaction import rect_contains


def select_checkpoint_file(initial_directory: Path) -> Path | None:
    """Select checkpoint file.

    Parameters
    ----------
    initial_directory
        Value used by the operation.

    Returns
    -------
    Path | None
        Computed result.
    """
    if sys.platform == "darwin":
        return _select_checkpoint_file_macos(initial_directory)
    return _select_checkpoint_file_tk(initial_directory)


def _select_checkpoint_file_macos(initial_directory: Path) -> Path | None:
    """Select checkpoint file macos.

    Parameters
    ----------
    initial_directory
        Value used by the operation.

    Returns
    -------
    Path | None
        Computed result.
    """
    ObjCClass, get_NSString, cfstring_to_string = _cocoa_api()
    panel = ObjCClass("NSOpenPanel").openPanel()
    panel.setCanChooseFiles_(True)
    panel.setCanChooseDirectories_(False)
    panel.setAllowsMultipleSelection_(False)
    panel.setResolvesAliases_(True)
    panel.setTitle_(get_NSString("Load a saved simulation"))

    if initial_directory.is_dir():
        directory_url = ObjCClass("NSURL").fileURLWithPath_(
            get_NSString(str(initial_directory.resolve()))
        )
        panel.setDirectoryURL_(directory_url)

    allowed_types = ObjCClass("NSMutableArray").array()
    allowed_types.addObject_(get_NSString("pkl"))
    allowed_types.addObject_(get_NSString("bak"))
    panel.setAllowedFileTypes_(allowed_types)

    if panel.runModal() != 1:
        return None
    urls = panel.URLs()
    if urls.count() == 0:
        return None

    selected_url = urls.objectAtIndex_(0)
    selected_path = selected_url.path()
    path_text = cfstring_to_string(selected_path.ptr)
    if path_text is None:
        raise ValueError("The selected file path could not be decoded.")

    checkpoint = Path(path_text)
    if not _is_checkpoint_file(checkpoint):
        raise ValueError("Select a .pkl or .pkl.bak checkpoint file.")
    return checkpoint


def _cocoa_api() -> tuple[object, object, object]:
    """Return cocoa api.

    Returns
    -------
    tuple[object, object, object]
        Computed collection.
    """
    from pyglet.libs.darwin.cocoapy import (
        ObjCClass,
        cfstring_to_string,
        get_NSString,
    )

    return ObjCClass, get_NSString, cfstring_to_string


def _select_checkpoint_file_tk(initial_directory: Path) -> Path | None:
    """Select checkpoint file tk.

    Parameters
    ----------
    initial_directory
        Value used by the operation.

    Returns
    -------
    Path | None
        Computed result.
    """
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        root.update_idletasks()
        selected = filedialog.askopenfilename(
            parent=root,
            title="Load a saved simulation",
            initialdir=str(initial_directory),
            filetypes=(
                ("NEAT checkpoints", "*.pkl *.pkl.bak"),
                ("All files", "*.*"),
            ),
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _is_checkpoint_file(path: Path) -> bool:
    """Return whether is checkpoint file.

    Parameters
    ----------
    path
        Value used by the operation.

    Returns
    -------
    bool
        Whether the operation succeeded or consumed the input.
    """
    name = path.name.lower()
    return name.endswith(".pkl") or name.endswith(".pkl.bak")


@dataclass(slots=True)
class StartMenuLayout:
    """Provide StartMenuLayout UI behavior."""
    window: arcade.Rect
    title_y: float
    subtitle_y: float
    cards_top: float
    left_card: arcade.Rect
    right_card: arcade.Rect


@dataclass(slots=True)
class CardContentLayout:
    """Provide CardContentLayout UI behavior."""
    badge: arcade.Rect
    icon: arcade.Rect
    title_block: arcade.Rect
    body_block: arcade.Rect


class StartMenuView(arcade.View):
    """Provide StartMenuView UI behavior."""
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
    ERROR_TEXT = (255, 178, 178)

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
        load_view_factory: Callable[[Path], arcade.View],
        *,
        file_picker: Callable[[Path], Path | None] = select_checkpoint_file,
    ) -> None:
        """Initialize the component.

        Parameters
        ----------
        config
            Simulation configuration.
        start_view_factory
            Value used by the operation.
        load_view_factory
            Value used by the operation.
        file_picker
            Value used by the operation.
        """
        super().__init__()
        self.config = config
        self.start_view_factory = start_view_factory
        self.load_view_factory = load_view_factory
        self.file_picker = file_picker
        self._painter = ArcadePainter()
        self._text_cache = self._painter.text_cache
        self._texture_cache = self._painter.texture_cache
        self._sprite_cache = self._painter.sprite_cache
        self._hovered_button: str | None = None
        self._pressed_button: str | None = None
        self._button_animation: dict[str, float] = {"start": 0.0, "load": 0.0}
        self._load_error: str | None = None

    def on_show_view(self) -> None:
        """Return on show view.
        """
        if self.window is not None:
            self.window.background_color = self.BACKGROUND

    def on_draw(self) -> None:
        """Return on draw.
        """
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
            description=(
                self._load_error
                or "Restore a previous system state from your archives."
            ),
            disabled=False,
            description_color=(
                self.ERROR_TEXT if self._load_error is not None else None
            ),
        )

    def on_mouse_press(
        self,
        x: int,
        y: int,
        button: int,
        modifiers: int,
    ) -> bool | None:
        """Return on mouse press.

        Parameters
        ----------
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        button
            Arcade input value.
        modifiers
            Arcade input value.

        Returns
        -------
        bool | None
            Computed result.
        """
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
        """Return on mouse release.

        Parameters
        ----------
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        button
            Arcade input value.
        modifiers
            Arcade input value.

        Returns
        -------
        bool | None
            Computed result.
        """
        if button == arcade.MOUSE_BUTTON_LEFT and self._pressed_button is not None:
            pressed_button = self._pressed_button
            released_button = self._button_key_at(x, y, self.layout())
            self._pressed_button = None
            self._hovered_button = released_button
            if pressed_button == "start" and released_button == "start":
                self._start_simulation()
            elif pressed_button == "load" and released_button == "load":
                self._load_simulation()
            return True
        return super().on_mouse_release(x, y, button, modifiers)

    def on_mouse_motion(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> bool | None:
        """Return on mouse motion.

        Parameters
        ----------
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        dx
            Logical screen coordinate.
        dy
            Logical screen coordinate.

        Returns
        -------
        bool | None
            Computed result.
        """
        self._hovered_button = self._button_key_at(x, y, self.layout())
        return super().on_mouse_motion(x, y, dx, dy)

    def on_update(self, delta_time: float) -> None:
        """Return on update.

        Parameters
        ----------
        delta_time
            Value used by the operation.
        """
        step = min(1.0, max(0.0, delta_time * 12.0))
        for key, current in self._button_animation.items():
            target = (
                1.0
                if key == self._hovered_button or key == self._pressed_button
                else 0.0
            )
            self._button_animation[key] = current + (target - current) * step

    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        """Return on key press.

        Parameters
        ----------
        symbol
            Arcade input value.
        modifiers
            Arcade input value.

        Returns
        -------
        bool | None
            Computed result.
        """
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
        """Return layout.

        Returns
        -------
        StartMenuLayout
            Computed result.
        """
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
        """Return title size.

        Parameters
        ----------
        width
            Requested logical size.

        Returns
        -------
        float
            Computed result.
        """
        return min(62.0, max(42.0, width / 20.5))

    def _window_size(self) -> tuple[int, int]:
        """Return window size.

        Returns
        -------
        tuple[int, int]
            Computed collection.
        """
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
        """Return start simulation.
        """
        if self.window is None:
            return
        view = self.start_view_factory()
        self._show_view(view)

    def _load_simulation(self) -> None:
        """Load simulation.
        """
        if self.window is None:
            return
        try:
            selected = self.file_picker(
                Path(self.config.persistence.simulation_root_directory)
            )
            if selected is None:
                return
            view = self.load_view_factory(selected)
        except Exception as error:
            detail = " ".join(str(error).split())
            if len(detail) > 120:
                detail = f"{detail[:117]}..."
            self._load_error = f"Unable to load checkpoint: {detail}"
            return

        self._load_error = None
        self._show_view(view)

    def _show_view(self, view: arcade.View) -> None:
        """Return show view.

        Parameters
        ----------
        view
            Value used by the operation.
        """
        if self.window is None:
            return
        self.window.show_view(view)
        on_resize = getattr(view, "on_resize", None)
        if callable(on_resize):
            width, height = self._window_size()
            on_resize(width, height)

    def _draw_background(self, bounds: arcade.Rect) -> None:
        """Draw background.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        """
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
            self.BACKGROUND,
        )

    def _draw_heading(self, layout: StartMenuLayout) -> None:
        """Draw heading.

        Parameters
        ----------
        layout
            Value used by the operation.
        """
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
        description_color: arcade.Color | tuple[int, ...] | None = None,
    ) -> None:
        """Draw card.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.
        icon_name
            Stable identifier used by the UI.
        title
            Text displayed by the UI.
        description
            Value used by the operation.
        disabled
            Whether the corresponding behavior is enabled.
        description_color
            Value used by the operation.
        """
        fill = self.CARD_FILL_DISABLED if disabled else self.CARD_FILL
        border = self.CARD_BORDER_DISABLED if disabled else self.CARD_BORDER
        title_color = self.TEXT_DISABLED if disabled else self.TEXT_PRIMARY
        body_color = (
            description_color
            if description_color is not None
            else self.TEXT_DISABLED if disabled else self.TEXT_MUTED
        )
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
        """Return card content layout.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        CardContentLayout
            Computed result.
        """
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
        """Return button key at.

        Parameters
        ----------
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        layout
            Value used by the operation.

        Returns
        -------
        str | None
            Computed result.
        """
        if self._contains(self._card_content_layout(layout.left_card).badge, x, y):
            return "start"
        if self._contains(self._card_content_layout(layout.right_card).badge, x, y):
            return "load"
        return None

    def _button_visual_bounds(self, bounds: arcade.Rect, key: str) -> arcade.Rect:
        """Return button visual bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        key
            Stable identifier used by the UI.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
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
        """Return button icon bounds.

        Parameters
        ----------
        icon
            Value used by the operation.
        visual_badge
            Value used by the operation.
        key
            Stable identifier used by the UI.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
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
        """Return button colors.

        Parameters
        ----------
        disabled
            Whether the corresponding behavior is enabled.
        key
            Stable identifier used by the UI.

        Returns
        -------
        tuple[arcade.Color | tuple[int, ...], arcade.Color | tuple[int, ...]]
            Computed collection.
        """
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
        """Return scaled bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        scale
            Value used by the operation.
        offset_y
            Value used by the operation.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
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
        """Blend two colors for hover and pressed states.

        Parameters
        ----------
        start, end
            Colors to interpolate.
        amount
            Blend amount clamped to ``[0, 1]``.

        Returns
        -------
        tuple[int, ...]
            Blended channels.
        """
        return mix_color(start, end, amount)

    def _draw_icon(
        self,
        bounds: arcade.Rect,
        icon_name: str,
        key: str,
        *,
        disabled: bool,
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
        disabled
            Whether the corresponding behavior is enabled.
        """
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
        """Draw cached menu text.

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
        bold, width, multiline, align, anchor_x, anchor_y, rotation
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
            rotation=rotation,
        )

    def _draw_rounded_rect(
        self,
        bounds: arcade.Rect,
        fill_color: arcade.Color | tuple[int, ...],
        border_color: arcade.Color | tuple[int, ...],
        radius: float,
        border_width: float,
    ) -> None:
        """Draw a rounded card with a border.

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
        """Draw a filled rounded card.

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

    def _contains(self, bounds: arcade.Rect, x: float, y: float) -> bool:
        """Return whether a menu rectangle contains a point.

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
