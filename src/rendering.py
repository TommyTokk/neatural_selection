from __future__ import annotations

from contextlib import contextmanager
from math import ceil, floor, cos, sin
from src.world import World
import arcade

from configs.sim_config import SimConfig
from src.creature import Creature
from src.food import Food


class EnvironmentRenderer:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.theme = config.theme
        self._text_cache: dict[str, arcade.Text] = {}

    def draw(self, world: World) -> None:
        bounds = world.layout.environment
        self._draw_panel(bounds)
        pan_x = world.environment_pan_x
        pan_y = world.environment_pan_y

        with self._environment_clip(bounds):
            self._draw_grid(bounds, world.environment_zoom, pan_x, pan_y)
            self._draw_food(world.visible_foods_for_viewport(), bounds, world)
            self._draw_creatures(
                world.visible_creatures_for_viewport(),
                bounds,
                world,
                world.selected_creature_id,
            )
            self._draw_selected_overlay(world, bounds)

        self._draw_environment_header(bounds, world)

    @contextmanager
    def _environment_clip(self, bounds: arcade.Rect):
        try:
            from pyglet import gl
        except ImportError:
            yield
            return

        clip_bounds = self._content_clip_bounds(bounds)
        x, y, width, height = self._scissor_box_for_bounds(clip_bounds)
        previous_box = (gl.GLint * 4)()
        was_enabled = bool(gl.glIsEnabled(gl.GL_SCISSOR_TEST))
        gl.glGetIntegerv(gl.GL_SCISSOR_BOX, previous_box)

        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(x, y, width, height)
        try:
            yield
        finally:
            gl.glScissor(
                previous_box[0],
                previous_box[1],
                previous_box[2],
                previous_box[3],
            )
            if not was_enabled:
                gl.glDisable(gl.GL_SCISSOR_TEST)

    def _content_clip_bounds(self, bounds: arcade.Rect) -> arcade.Rect:
        border_width = 2.0
        return arcade.LBWH(
            bounds.left + border_width,
            bounds.bottom + border_width,
            max(0.0, bounds.width - border_width * 2),
            max(0.0, bounds.height - border_width * 2),
        )

    def _scissor_box_for_bounds(self, bounds: arcade.Rect) -> tuple[int, int, int, int]:
        scale_x, scale_y = self._framebuffer_scale()
        return (
            round(bounds.left * scale_x),
            round(bounds.bottom * scale_y),
            round(bounds.width * scale_x),
            round(bounds.height * scale_y),
        )

    def _framebuffer_scale(self) -> tuple[float, float]:
        try:
            window = arcade.get_window()
            window_width, window_height = window.get_size()
            framebuffer_width, framebuffer_height = window.get_framebuffer_size()
        except (AttributeError, RuntimeError):
            return 1.0, 1.0

        if window_width <= 0 or window_height <= 0:
            return 1.0, 1.0

        return framebuffer_width / window_width, framebuffer_height / window_height

    def _draw_panel(self, bounds: arcade.Rect) -> None:
        self._draw_rounded_rect(
            bounds,
            self.theme.environment_background,
            self.theme.environment_border,
            self.config.layout.environment_radius,
            2,
        )

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
            max(0, bounds.width - border_width * 2),
            max(0, bounds.height - border_width * 2),
        )
        self._draw_rounded_rect_fill(inner, fill_color, max(0, radius - border_width))

    def _draw_rounded_rect_fill(
        self,
        bounds: arcade.Rect,
        color: arcade.Color | tuple[int, ...],
        radius: float,
    ) -> None:
        radius = min(radius, bounds.width / 2, bounds.height / 2)
        if radius <= 0:
            arcade.draw_lrbt_rectangle_filled(
                bounds.left,
                bounds.right,
                bounds.bottom,
                bounds.top,
                color,
            )
            return
        arcade.draw_lrbt_rectangle_filled(
            bounds.left + radius,
            bounds.right - radius,
            bounds.bottom,
            bounds.top,
            color,
        )
        arcade.draw_lrbt_rectangle_filled(
            bounds.left,
            bounds.right,
            bounds.bottom + radius,
            bounds.top - radius,
            color,
        )
        arcade.draw_circle_filled(
            bounds.left + radius, bounds.bottom + radius, radius, color
        )
        arcade.draw_circle_filled(
            bounds.right - radius, bounds.bottom + radius, radius, color
        )
        arcade.draw_circle_filled(
            bounds.left + radius, bounds.top - radius, radius, color
        )
        arcade.draw_circle_filled(
            bounds.right - radius, bounds.top - radius, radius, color
        )

    def _draw_grid(self, bounds: arcade.Rect, zoom: float, pan_x: float, pan_y: float) -> None:
        step = max(18.0, 48.0 * zoom)
        center_x = (bounds.left + bounds.right) * 0.5 + pan_x
        center_y = (bounds.bottom + bounds.top) * 0.5 + pan_y
        # Draw only the grid lines that can be visible inside the environment.
        start_x = center_x + floor((bounds.left - center_x) / step) * step
        if start_x < bounds.left:
            start_x += step
        end_x = center_x + ceil((bounds.right - center_x) / step) * step
        x = start_x
        while x <= end_x:
            arcade.draw_line(x, bounds.bottom, x, bounds.top, self.theme.environment_grid, 1)
            x += step

        start_y = center_y + floor((bounds.bottom - center_y) / step) * step
        if start_y < bounds.bottom:
            start_y += step
        end_y = center_y + ceil((bounds.top - center_y) / step) * step
        y = start_y
        while y <= end_y:
            arcade.draw_line(bounds.left, y, bounds.right, y, self.theme.environment_grid, 1)
            y += step

    def _draw_environment_header(self, bounds: arcade.Rect, world: World) -> None:
        self._draw_text(
            "env_fps",
            f"FPS: {world.fps:0.0f}",
            bounds.left + 16,
            bounds.top - 34,
            self.theme.environment_text,
            18,
            bold=True,
        )

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
    ) -> None:
        rx = round(x)
        ry = round(y)
        cached = self._text_cache.get(key)
        if cached is None:
            cached = arcade.Text(text, rx, ry, color, size, bold=bold)
            self._text_cache[key] = cached
        else:
            cached.text = text
            cached.x = rx
            cached.y = ry
            cached.color = color
            cached.font_size = size
            cached.bold = bold
        cached.draw()

    def _draw_food(
        self,
        foods: list[Food],
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        zoom = world.environment_zoom
        draw_outlines = len(foods) <= 250 or zoom >= 1.25
        for food in foods:
            pos_x, pos_y = food.position
            draw_x, draw_y = world.environment_to_screen(pos_x, pos_y)
            radius = max(2.0, food.radius * zoom)
            if not self._circle_intersects_visible_bounds(bounds, draw_x, draw_y, radius):
                continue
            arcade.draw_circle_filled(draw_x, draw_y, radius, self.theme.food_fill)
            if not draw_outlines:
                continue
            arcade.draw_circle_outline(
                draw_x,
                draw_y,
                radius,
                self.theme.environment_border,
                1,
            )

    def _draw_creatures(
        self,
        creatures: list[Creature],
        bounds: arcade.Rect,
        world: World,
        selected_creature_id: int | None,
    ) -> None:
        zoom = world.environment_zoom
        for creature in creatures:
            model_x, model_y = creature.position
            draw_x, draw_y = world.environment_to_screen(model_x, model_y)
            radius = max(3.0, creature.radius * zoom)
            if not self._circle_intersects_visible_bounds(bounds, draw_x, draw_y, radius):
                continue
            arcade.draw_circle_filled(draw_x, draw_y, radius, creature.color)
            outline_color = (
                self.theme.selected_outline
                if selected_creature_id == creature.creature_id
                else self.theme.herbivore_outline
            )
            arcade.draw_circle_outline(
                draw_x,
                draw_y,
                radius,
                outline_color,
                2,
            )

            heading_x = draw_x + cos(creature.heading) * radius
            heading_y = draw_y + sin(creature.heading) * radius
            arcade.draw_circle_filled(
                heading_x,
                heading_y,
                max(2.4, radius * 0.18),
                self.theme.herbivore_outline,
            )

            left_eye_model, right_eye_model = self._creature_eye_positions(creature)
            left_eye_x, left_eye_y = world.environment_to_screen(*left_eye_model)
            right_eye_x, right_eye_y = world.environment_to_screen(*right_eye_model)
            eye_radius = max(2.0, 2.3 * zoom)
            arcade.draw_circle_filled(left_eye_x, left_eye_y, eye_radius, (247, 247, 241))
            arcade.draw_circle_filled(right_eye_x, right_eye_y, eye_radius, (247, 247, 241))

            if selected_creature_id == creature.creature_id:
                self._draw_energy_bar(creature, bounds, world)

    def _draw_energy_bar(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        zoom = world.environment_zoom
        center_x, center_y = creature.position
        draw_x, draw_y = world.environment_to_screen(center_x, center_y)
        radius = creature.radius * zoom

        width = max(20.0, radius * 2.1)
        height = max(3.0, 5.0 * zoom)
        left = draw_x - width / 2
        bottom = draw_y + radius + (8.0 * zoom)
        ratio = max(0.0, min(1.0, creature.energy))
        if not self._rect_fits_visible_bounds(
            bounds, left, bottom, left + width, bottom + height
        ):
            return

        arcade.draw_lrbt_rectangle_filled(
            left,
            left + width,
            bottom,
            bottom + height,
            (25, 30, 36, 180),
        )
        arcade.draw_lrbt_rectangle_filled(
            left,
            left + width * ratio,
            bottom,
            bottom + height,
            self.theme.accent_soft,
        )

    def _draw_selected_overlay(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        selected = world.selected_creature
        if selected is None:
            return

        if world.debug_vision_enabled:
            self._draw_vision_cone(selected, bounds, world)
            self._draw_visible_food_highlights(
                world.visible_foods_for(selected),
                bounds,
                world,
            )
            self._draw_visible_creature_highlights(
                world.visible_creatures_for(selected),
                bounds,
                world,
            )

    def _draw_vision_cone(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        zoom = world.environment_zoom
        left_eye_model, right_eye_model = self._creature_eye_positions(creature)
        left_eye = world.environment_to_screen(*left_eye_model)
        right_eye = world.environment_to_screen(*right_eye_model)
        eye_origin = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        eye_cone_points = self._vision_cone_points(creature, eye_origin, zoom)
        blind_zone_points = self._mouth_blind_zone_points(creature, world)

        arcade.draw_polygon_filled(eye_cone_points, (111, 220, 128, 48))
        arcade.draw_polygon_filled(blind_zone_points, (224, 74, 74, 76))
        arcade.draw_polygon_outline(eye_cone_points, (111, 220, 128, 132), 1)
        arcade.draw_polygon_outline(blind_zone_points, (224, 74, 74, 142), 2)

    def _vision_cone_points(
        self,
        creature: Creature,
        origin: tuple[float, float],
        zoom: float,
    ) -> list[tuple[float, float]]:
        heading = creature.heading
        cone_radius = creature.vision.range * zoom
        cone_angle = creature.vision.angle
        steps = 18

        points: list[tuple[float, float]] = [origin]
        for index in range(steps + 1):
            factor = index / steps
            angle = heading - cone_angle / 2 + cone_angle * factor
            points.append(
                (
                    origin[0] + cos(angle) * cone_radius,
                    origin[1] + sin(angle) * cone_radius,
                )
            )

        return points

    def _mouth_blind_zone_points(
        self,
        creature: Creature,
        world: World,
    ) -> list[tuple[float, float]]:
        center_x, center_y = creature.position
        heading = creature.heading
        zoom = world.environment_zoom

        mouth_x = center_x + cos(heading) * creature.radius
        mouth_y = center_y + sin(heading) * creature.radius
        origin_x, origin_y = world.environment_to_screen(mouth_x, mouth_y)

        forward_x = cos(heading) * zoom
        forward_y = sin(heading) * zoom
        lateral_x = -sin(heading) * zoom
        lateral_y = cos(heading) * zoom

        rear_length = max(4.0, creature.radius * 0.45)
        half_width = max(2.0, creature.radius * 0.35)
        half_width += max(1.0, creature.radius * 0.2)

        return [
            (
                origin_x + lateral_x * half_width,
                origin_y + lateral_y * half_width,
            ),
            (
                origin_x - lateral_x * half_width,
                origin_y - lateral_y * half_width,
            ),
            (
                origin_x - forward_x * rear_length - lateral_x * half_width,
                origin_y - forward_y * rear_length - lateral_y * half_width,
            ),
            (
                origin_x - forward_x * rear_length + lateral_x * half_width,
                origin_y - forward_y * rear_length + lateral_y * half_width,
            ),
        ]

    def _draw_visible_food_highlights(
        self,
        foods: list[Food],
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        zoom = world.environment_zoom
        for food in foods:
            pos_x, pos_y = food.position
            draw_x, draw_y = world.environment_to_screen(pos_x, pos_y)
            radius = max(4.0, (food.radius + 4.0) * zoom)
            if not self._circle_fits_visible_bounds(bounds, draw_x, draw_y, radius):
                continue

            arcade.draw_circle_outline(draw_x, draw_y, radius, (246, 232, 116), 3)
            arcade.draw_circle_outline(draw_x, draw_y, radius + 3.0, (246, 232, 116, 110), 2)

    def _draw_visible_creature_highlights(
        self,
        creatures: list[Creature],
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        zoom = world.environment_zoom
        for creature in creatures:
            pos_x, pos_y = creature.position
            draw_x, draw_y = world.environment_to_screen(pos_x, pos_y)
            radius = max(6.0, (creature.radius + 5.0) * zoom)
            if not self._circle_fits_visible_bounds(bounds, draw_x, draw_y, radius):
                continue

            arcade.draw_circle_outline(draw_x, draw_y, radius, (116, 202, 246), 3)
            arcade.draw_circle_outline(draw_x, draw_y, radius + 3.0, (116, 202, 246, 105), 2)

    def _creature_eye_positions(
        self, creature: Creature
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        center_x, center_y = creature.position
        heading = creature.heading
        radius = creature.radius
        front_x = center_x + cos(heading) * radius * 0.35
        front_y = center_y + sin(heading) * radius * 0.35
        side_x = cos(heading + 1.5708) * radius * 0.34
        side_y = sin(heading + 1.5708) * radius * 0.34
        return (
            (front_x + side_x, front_y + side_y),
            (front_x - side_x, front_y - side_y),
        )

    def _circle_fits_visible_bounds(
        self, bounds: arcade.Rect, x: float, y: float, radius: float
    ) -> bool:
        return (
            x - radius >= bounds.left
            and x + radius <= bounds.right
            and y - radius >= bounds.bottom
            and y + radius <= bounds.top
        )

    def _circle_intersects_visible_bounds(
        self, bounds: arcade.Rect, x: float, y: float, radius: float
    ) -> bool:
        return (
            x + radius >= bounds.left
            and x - radius <= bounds.right
            and y + radius >= bounds.bottom
            and y - radius <= bounds.top
        )

    def _rect_fits_visible_bounds(
        self,
        bounds: arcade.Rect,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> bool:
        return (
            left >= bounds.left
            and right <= bounds.right
            and bottom >= bounds.bottom
            and top <= bounds.top
        )
