from __future__ import annotations

from math import ceil, dist, floor, cos, sin
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
        self._draw_grid(bounds, world.environment_zoom, pan_x, pan_y)
        self._draw_food(world.foods, bounds, world.environment_zoom, pan_x, pan_y)
        self._draw_creatures(
            world.creatures,
            bounds,
            world.elapsed_time,
            world.environment_zoom,
            pan_x,
            pan_y,
            world.selected_creature_id,
        )
        self._draw_selected_overlay(world, bounds, world.environment_zoom, pan_x, pan_y)
        self._draw_environment_header(bounds, world)

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
            "env_title",
            world.stats.environment_name,
            bounds.left + 16,
            bounds.top - 30,
            self.theme.environment_text,
            16,
            bold=True,
        )
        self._draw_text(
            "env_subtitle",
            "Central simulation viewport",
            bounds.left + 16,
            bounds.top - 52,
            self.theme.environment_text_muted,
            11,
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

    def _zoom_point(
        self,
        bounds: arcade.Rect,
        point_x: float,
        point_y: float,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> tuple[float, float]:
        center_x = bounds.center_x
        center_y = bounds.center_y
        return (
            center_x + (point_x - center_x) * zoom + pan_x,
            center_y + (point_y - center_y) * zoom + pan_y,
        )

    def _zoom_points(
        self,
        bounds: arcade.Rect,
        points: list[tuple[float, float]],
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> list[tuple[float, float]]:
        return [
            self._zoom_point(bounds, point_x, point_y, zoom, pan_x, pan_y)
            for point_x, point_y in points
        ]

    def _draw_food(
        self,
        foods: list[Food],
        bounds: arcade.Rect,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> None:
        for food in foods:
            pos_x, pos_y = food.get_screen_position(bounds)
            draw_x, draw_y = self._zoom_point(bounds, pos_x, pos_y, zoom, pan_x, pan_y)
            radius = max(2.0, food.radius * zoom)
            arcade.draw_circle_filled(draw_x, draw_y, radius, self.theme.food_fill)
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
        elapsed_time: float,
        zoom: float,
        pan_x: float,
        pan_y: float,
        selected_creature_id: int | None,
    ) -> None:
        for creature in creatures:
            model_points = creature.triangle_points(bounds, elapsed_time)
            draw_points = self._zoom_points(bounds, model_points, zoom, pan_x, pan_y)
            arcade.draw_polygon_filled(draw_points, creature.color)
            arcade.draw_polygon_outline(draw_points, self.theme.herbivore_outline, 2)

            left_eye_model, right_eye_model = creature.eye_positions(bounds, elapsed_time)
            left_eye_x, left_eye_y = self._zoom_point(
                bounds, left_eye_model[0], left_eye_model[1], zoom, pan_x, pan_y
            )
            right_eye_x, right_eye_y = self._zoom_point(
                bounds, right_eye_model[0], right_eye_model[1], zoom, pan_x, pan_y
            )
            eye_radius = max(2.0, 2.3 * zoom)
            arcade.draw_circle_filled(left_eye_x, left_eye_y, eye_radius, (247, 247, 241))
            arcade.draw_circle_filled(right_eye_x, right_eye_y, eye_radius, (247, 247, 241))

            if selected_creature_id == creature.creature_id:
                self._draw_energy_bar(creature, bounds, elapsed_time, zoom, pan_x, pan_y)

    def _draw_energy_bar(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        elapsed_time: float,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> None:
        center_x, center_y = creature.get_screen_position(bounds, elapsed_time)
        draw_x, draw_y = self._zoom_point(
            bounds, center_x, center_y, zoom, pan_x, pan_y
        )
        radius = creature.get_radius(bounds) * zoom

        width = max(20.0, radius * 2.1)
        height = max(3.0, 5.0 * zoom)
        left = draw_x - width / 2
        bottom = draw_y + radius + (8.0 * zoom)
        ratio = max(0.0, min(1.0, creature.energy))

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
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> None:
        selected = world.selected_creature
        if selected is None:
            return

        center_x, center_y = selected.get_screen_position(bounds, world.elapsed_time)
        draw_x, draw_y = self._zoom_point(
            bounds, center_x, center_y, zoom, pan_x, pan_y
        )
        radius = selected.get_radius(bounds) * zoom

        arcade.draw_circle_outline(
            draw_x,
            draw_y,
            radius + 8.0,
            self.theme.selected_outline,
            2.5,
        )

        if world.debug_vision_enabled:
            self._draw_vision_cone(
                selected, bounds, world.elapsed_time, zoom, pan_x, pan_y
            )

    def _draw_vision_cone(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        elapsed_time: float,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> None:
        left_eye_model, right_eye_model = creature.eye_positions(bounds, elapsed_time)
        left_eye = self._zoom_point(
            bounds, left_eye_model[0], left_eye_model[1], zoom, pan_x, pan_y
        )
        right_eye = self._zoom_point(
            bounds, right_eye_model[0], right_eye_model[1], zoom, pan_x, pan_y
        )
        draw_center_x = (left_eye[0] + right_eye[0]) / 2
        draw_center_y = (left_eye[1] + right_eye[1]) / 2
        heading = creature.get_heading(elapsed_time)
        eye_distance = dist(left_eye, right_eye)
        cone_radius = max(eye_distance * 8.5, 74.0 * zoom)
        cone_angle = 0.95
        steps = 18

        points: list[tuple[float, float]] = [(draw_center_x, draw_center_y)]
        for index in range(steps + 1):
            factor = index / steps
            angle = heading - cone_angle / 2 + cone_angle * factor
            points.append(
                (
                    draw_center_x + cos(angle) * cone_radius,
                    draw_center_y + sin(angle) * cone_radius,
                )
            )

        arcade.draw_polygon_filled(points, self.theme.vision_fill)
        arcade.draw_polygon_outline(points, self.theme.accent, 1)
