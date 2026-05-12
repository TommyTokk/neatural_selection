from __future__ import annotations

import ctypes

import pyglet

SOFTWARE_RENDERER_MARKERS = (
    "llvmpipe",
    "softpipe",
    "software",
    "swiftshader",
)

_GRAPHICS_CONFIGURED = False


def configure_graphics() -> None:
    global _GRAPHICS_CONFIGURED
    if _GRAPHICS_CONFIGURED:
        return

    # Keep window/framebuffer in 1:1 logical rendering mode to avoid stretched text.
    pyglet.options["dpi_scaling"] = "real"
    _GRAPHICS_CONFIGURED = True


def log_graphics_context() -> None:
    from pyglet import gl

    vendor = _gl_string(gl.GL_VENDOR)
    renderer = _gl_string(gl.GL_RENDERER)
    version = _gl_string(gl.GL_VERSION)
    print(
        "OpenGL context: "
        f"vendor={vendor}, renderer={renderer}, version={version}"
    )

    if _is_software_renderer(renderer):
        raise RuntimeError(
            "OpenGL is using a software renderer instead of the GPU: "
            f"{renderer}. Check your graphics drivers or launch environment."
        )


def _gl_string(name: int) -> str:
    from pyglet import gl

    raw_value = gl.glGetString(name)
    if raw_value is None:
        return "unknown"

    if isinstance(raw_value, bytes):
        value = raw_value
    else:
        value = ctypes.cast(raw_value, ctypes.c_char_p).value

    if value is None:
        return "unknown"

    return value.decode("utf-8", errors="replace")


def _is_software_renderer(renderer: str) -> bool:
    normalized = renderer.lower()
    return any(marker in normalized for marker in SOFTWARE_RENDERER_MARKERS)
