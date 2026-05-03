import pyglet

# Keep window/framebuffer in 1:1 logical rendering mode to avoid stretched text.
pyglet.options["dpi_scaling"] = "real"

from src.app import create_and_run


if __name__ == "__main__":
    create_and_run()
