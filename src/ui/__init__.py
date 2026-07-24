"""User-interface package for the simulation presentation layer.

The exports are resolved lazily so importing the screen-layout module from
``World`` does not import the renderer back into the world model.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "EnvironmentRenderer",
    "ScreenLayout",
    "StartMenuView",
    "UiRenderer",
    "build_screen_layout",
)

_EXPORTS = {
    "EnvironmentRenderer": ("src.ui.renderers.environment", "EnvironmentRenderer"),
    "ScreenLayout": ("src.ui.layouts.screen", "ScreenLayout"),
    "StartMenuView": ("src.ui.views.start_menu", "StartMenuView"),
    "UiRenderer": ("src.ui.renderer", "UiRenderer"),
    "build_screen_layout": ("src.ui.layouts.screen", "build_screen_layout"),
}


def __getattr__(name: str) -> Any:
    """Load a public UI symbol without eagerly importing renderer modules.

    Parameters
    ----------
    name
        Name requested from the package.

    Returns
    -------
    Any
        Exported class or function.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the public UI API.
    """
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    return getattr(import_module(module_name), attribute_name)


def __dir__() -> list[str]:
    """Return package attributes including lazily exported UI symbols.

    Returns
    -------
    list[str]
        Sorted package attribute names.
    """
    return sorted(set(globals()) | set(__all__))
