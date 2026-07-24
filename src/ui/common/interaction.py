"""Shared interaction state for immediate-mode UI components."""

from __future__ import annotations

from dataclasses import dataclass, field

import arcade


@dataclass(slots=True)
class UiInteractionState:
    """Store transient hit regions and persistent scroll positions."""

    hitboxes: dict[str, arcade.Rect] = field(default_factory=dict)
    scroll_regions: dict[str, arcade.Rect] = field(default_factory=dict)
    scroll_offsets: dict[str, float] = field(default_factory=dict)
    scroll_limits: dict[str, float] = field(default_factory=dict)

    def begin_frame(self) -> None:
        """Clear geometry that components will register again this frame."""
        self.hitboxes.clear()
        self.scroll_regions.clear()
        self.scroll_limits.clear()

    def register_hitbox(self, key: str, bounds: arcade.Rect) -> arcade.Rect:
        """Register and return an interactive rectangle.

        Parameters
        ----------
        key
            Stable interaction identifier.
        bounds
            Rectangle receiving pointer input.

        Returns
        -------
        arcade.Rect
            The supplied bounds for convenient inline use.
        """
        self.hitboxes[key] = bounds
        return bounds

    def contains(self, key: str, x: float, y: float) -> bool:
        """Return whether a registered region contains a point.

        Parameters
        ----------
        key
            Registered interaction identifier.
        x, y
            Pointer coordinates.

        Returns
        -------
        bool
            ``True`` when the point lies within the region.
        """
        bounds = self.hitboxes.get(key)
        return bounds is not None and rect_contains(bounds, x, y)

    def register_scroll_region(
        self,
        key: str,
        bounds: arcade.Rect,
        limit: float,
    ) -> float:
        """Register a scroll viewport and clamp its persistent offset.

        Parameters
        ----------
        key
            Stable scroll-region identifier.
        bounds
            Visible scroll viewport.
        limit
            Maximum permitted offset.

        Returns
        -------
        float
            Clamped current offset.
        """
        normalized_limit = max(0.0, float(limit))
        offset = max(
            0.0,
            min(normalized_limit, self.scroll_offsets.get(key, 0.0)),
        )
        self.scroll_regions[key] = bounds
        self.scroll_limits[key] = normalized_limit
        self.scroll_offsets[key] = offset
        return offset


def rect_contains(bounds: arcade.Rect, x: float, y: float) -> bool:
    """Return whether a rectangle contains a point.

    Parameters
    ----------
    bounds
        Rectangle to inspect.
    x, y
        Point coordinates.

    Returns
    -------
    bool
        ``True`` for points on or inside the rectangle.
    """
    return bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top


def rects_overlap(first: arcade.Rect, second: arcade.Rect) -> bool:
    """Return whether two rectangles overlap.

    Parameters
    ----------
    first, second
        Rectangles to compare.

    Returns
    -------
    bool
        ``True`` when their areas intersect.
    """
    return not (
        first.right < second.left
        or second.right < first.left
        or first.top < second.bottom
        or second.top < first.bottom
    )
