from __future__ import annotations

from contextlib import contextmanager
from math import ceil, cos, degrees, floor, hypot, sin

import arcade
import numpy as np

from configs.sim_config import SimConfig
from src.creature import Creature
from src.food import Food
from src.ui.common.drawing import ArcadePainter
from src.world import World


class EnvironmentRenderer:
    """Provide EnvironmentRenderer UI behavior."""
    FOOD_SPRITE_TEXTURE_DIAMETER = 64
    FOOD_OUTLINE_ENABLE_COUNT = 225
    FOOD_OUTLINE_DISABLE_COUNT = 275
    CREATURE_BASE_TEXTURE_RADIUS = 100
    CREATURE_BASE_TEXTURE_DIAMETER = CREATURE_BASE_TEXTURE_RADIUS * 2

    def __init__(self, config: SimConfig) -> None:
        """Initialize the component.

        Parameters
        ----------
        config
            Simulation configuration.
        """
        self.config = config
        self.theme = config.theme
        self._painter = ArcadePainter()
        self._text_cache = self._painter.text_cache
        self._biome_texture: object | None = None
        self._biome_texture_key: int | None = None
        self._pheromone_texture: object | None = None
        self._pheromone_texture_key: tuple[int, int] | None = None
        self._food_sprite_list: object | None = None
        self._food_sprite_list_keys: set[int] = set()
        self._food_sprite_cache: dict[int, object] = {}
        self._food_sprite_texture: object | None = None
        self._food_batch_disabled = False
        self._food_outlines_enabled: bool | None = None
        self._creature_sprite_list: object | None = None
        self._creature_detail_sprite_list: object | None = None
        self._creature_sprite_cache: dict[int, object] = {}
        self._creature_detail_sprite_cache: dict[int, object] = {}
        self._creature_sprite_texture: object | None = None
        self._creature_detail_texture: object | None = None
        self._creature_batch_disabled = False

    def draw(self, world: World) -> None:
        """Return draw.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        bounds = world.layout.environment
        self._draw_panel(bounds)
        pan_x = world.environment_pan_x
        pan_y = world.environment_pan_y

        with self._environment_clip(bounds):
            map_mode = self._environment_map_mode(world)
            if map_mode == "biome":
                self._draw_biomes(bounds, world)
            elif map_mode == "pheromones":
                self._draw_pheromones(bounds, world)
            self._draw_grid(bounds, world.environment_zoom, pan_x, pan_y)
            self._draw_food(world.visible_foods_for_viewport(), bounds, world)
            self._draw_creatures(
                world.visible_creatures_for_viewport(),
                bounds,
                world,
                world.selected_creature_id,
            )
            self._draw_selected_overlay(world, bounds)
            self._draw_selected_creature_status(world, bounds)

        self._draw_environment_header(bounds, world)

    @staticmethod
    def _environment_map_mode(world: World) -> str:
        """Return environment map mode.

        Parameters
        ----------
        world
            Simulation world providing current state.

        Returns
        -------
        str
            Formatted or resolved value.
        """
        mode = getattr(world, "environment_map_mode", None)
        if mode in {"none", "biome", "pheromones"}:
            return mode
        return (
            "biome"
            if getattr(world, "show_biome_background", False)
            else "none"
        )

    @contextmanager
    def _environment_clip(self, bounds: arcade.Rect):
        """Clip world drawing inside the environment border.

        Parameters
        ----------
        bounds
            Environment panel bounds.

        Yields
        ------
        None
            Control while clipping is active.
        """
        with self._painter.clip(bounds, inset=2.0):
            yield

    def _content_clip_bounds(self, bounds: arcade.Rect) -> arcade.Rect:
        """Return content clip bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        arcade.Rect
            Computed UI rectangle.
        """
        border_width = 2.0
        return arcade.LBWH(
            bounds.left + border_width,
            bounds.bottom + border_width,
            max(0.0, bounds.width - border_width * 2),
            max(0.0, bounds.height - border_width * 2),
        )

    def _scissor_box_for_bounds(self, bounds: arcade.Rect) -> tuple[int, int, int, int]:
        """Return scissor box for bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.

        Returns
        -------
        tuple[int, int, int, int]
            Computed collection.
        """
        scale_x, scale_y = self._framebuffer_scale()
        return (
            round(bounds.left * scale_x),
            round(bounds.bottom * scale_y),
            round(bounds.width * scale_x),
            round(bounds.height * scale_y),
        )

    def _framebuffer_scale(self) -> tuple[float, float]:
        """Return logical-to-framebuffer scale factors.

        Returns
        -------
        tuple[float, float]
            Horizontal and vertical framebuffer scales.
        """
        return self._painter.framebuffer_scale()

    def _draw_panel(self, bounds: arcade.Rect) -> None:
        """Draw panel.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        """
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
        """Draw the environment panel with a rounded border.

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
        """Draw a filled rounded environment rectangle.

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

    def _draw_grid(self, bounds: arcade.Rect, zoom: float, pan_x: float, pan_y: float) -> None:
        """Draw grid.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        zoom
            Value used by the operation.
        pan_x
            Value used by the operation.
        pan_y
            Value used by the operation.
        """
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

    def _draw_biomes(self, bounds: arcade.Rect, world: World) -> None:
        """Draw biomes.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        biome_map = getattr(world, "biome_map", None)
        if biome_map is None:
            return

        self._draw_world_map_texture(
            bounds,
            world,
            self._texture_for_biome_map(biome_map),
        )

    def _draw_pheromones(self, bounds: arcade.Rect, world: World) -> None:
        """Draw pheromones.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        pheromones = getattr(world, "pheromones", None)
        if pheromones is None:
            return
        self._draw_world_map_texture(
            bounds,
            world,
            self._texture_for_pheromones(pheromones),
        )

    def _draw_world_map_texture(
        self,
        bounds: arcade.Rect,
        world: World,
        texture: object | None,
    ) -> None:
        """Draw world map texture.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        texture
            Value used by the operation.
        """
        draw_texture_rect = getattr(arcade, "draw_texture_rect", None)
        if texture is None or draw_texture_rect is None:
            return

        world_left, world_bottom, world_right, world_top = world.environment_world_bounds
        screen_left, screen_bottom = world.environment_to_screen(world_left, world_bottom)
        screen_right, screen_top = world.environment_to_screen(world_right, world_top)
        rect = arcade.LBWH(
            min(screen_left, screen_right),
            min(screen_bottom, screen_top),
            abs(screen_right - screen_left),
            abs(screen_top - screen_bottom),
        )
        if not self._rect_intersects_visible_bounds(
            bounds,
            rect.left,
            rect.bottom,
            rect.right,
            rect.top,
        ):
            return

        try:
            draw_texture_rect(texture, rect, pixelated=False)
        except TypeError:
            draw_texture_rect(texture, rect)

    def _texture_for_biome_map(self, biome_map: object) -> object | None:
        """Return texture for biome map.

        Parameters
        ----------
        biome_map
            Value used by the operation.

        Returns
        -------
        object | None
            Computed result.
        """
        render_rgba = getattr(biome_map, "render_rgba", None)
        texture_key = id(render_rgba)
        if self._biome_texture is not None and self._biome_texture_key == texture_key:
            return self._biome_texture

        texture_factory = getattr(arcade, "Texture", None)
        if texture_factory is None or render_rgba is None:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        image = Image.fromarray(render_rgba[::-1])
        try:
            texture = texture_factory(image, hash=f"biomes-{texture_key}")
        except TypeError:
            texture = texture_factory(image)

        self._biome_texture = texture
        self._biome_texture_key = texture_key
        return texture

    @staticmethod
    def _pheromone_rgba(pheromones: object) -> np.ndarray:
        """Return pheromone rgba.

        Parameters
        ----------
        pheromones
            Value used by the operation.

        Returns
        -------
        np.ndarray
            Computed result.
        """
        field = np.asarray(getattr(pheromones, "field"), dtype=np.float32)
        if field.ndim != 3 or field.shape[2] != 3:
            raise ValueError("Pheromone field must have width-major shape (W, H, 3).")
        maximum = max(0.0, float(getattr(pheromones.config, "max_concentration", 1.0)))
        image_field = np.swapaxes(field, 0, 1)
        output = np.zeros((*image_field.shape[:2], 4), dtype=np.uint8)
        if maximum <= 0.0:
            return output
        rgb = (np.clip(image_field / maximum, 0.0, 1.0) * 255.0).astype(np.uint8)
        output[..., :3] = rgb
        output[..., 3] = np.max(rgb, axis=2)
        return output

    def _texture_for_pheromones(self, pheromones: object) -> object | None:
        """Return texture for pheromones.

        Parameters
        ----------
        pheromones
            Value used by the operation.

        Returns
        -------
        object | None
            Computed result.
        """
        revision = int(getattr(pheromones, "update_count", 0))
        texture_key = (id(pheromones), revision)
        if (
            self._pheromone_texture is not None
            and self._pheromone_texture_key == texture_key
        ):
            return self._pheromone_texture

        texture_factory = getattr(arcade, "Texture", None)
        if texture_factory is None:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None

        image = Image.fromarray(self._pheromone_rgba(pheromones)[::-1])
        try:
            texture = texture_factory(
                image,
                hash=f"pheromones-{texture_key[0]}-{texture_key[1]}",
            )
        except TypeError:
            texture = texture_factory(image)
        self._pheromone_texture = texture
        self._pheromone_texture_key = texture_key
        return texture

    def _draw_environment_header(self, bounds: arcade.Rect, world: World) -> None:
        """Draw environment header.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
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
        """Draw cached environment text.

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
        bold
            Whether to render bold text.
        """
        self._painter.draw_text(key, text, x, y, color, size, bold=bold)

    def _draw_food(
        self,
        foods: list[Food],
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw food.

        Parameters
        ----------
        foods
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        zoom = world.environment_zoom
        draw_outlines = self._should_draw_food_outlines(len(foods), zoom)
        visible_foods: list[tuple[Food, float, float, float]] = []
        for food in foods:
            pos_x, pos_y = food.position
            draw_x, draw_y = world.environment_to_screen(pos_x, pos_y)
            radius = max(2.0, food.radius * zoom)
            if not self._circle_intersects_visible_bounds(bounds, draw_x, draw_y, radius):
                continue
            visible_foods.append((food, draw_x, draw_y, radius))

        if not self._draw_food_sprite_batch(visible_foods):
            for _food, draw_x, draw_y, radius in visible_foods:
                arcade.draw_circle_filled(draw_x, draw_y, radius, self.theme.food_fill)

        for _food, draw_x, draw_y, radius in visible_foods:
            if not draw_outlines:
                continue
            arcade.draw_circle_outline(
                draw_x,
                draw_y,
                radius,
                self.theme.environment_border,
                1,
            )

    def _draw_food_sprite_batch(
        self,
        visible_foods: list[tuple[Food, float, float, float]],
    ) -> bool:
        """Draw food sprite batch.

        Parameters
        ----------
        visible_foods
            Value used by the operation.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        if self._food_batch_disabled:
            return False

        if not self._has_active_window():
            return False

        sprite_list = self._food_sprite_list
        if sprite_list is None:
            sprite_list_cls = getattr(arcade, "SpriteList", None)
            if sprite_list_cls is None:
                self._food_batch_disabled = True
                return False
            try:
                sprite_list = sprite_list_cls(
                    use_spatial_hash=False,
                    capacity=max(100, len(visible_foods)),
                )
            except Exception:
                self._food_batch_disabled = True
                return False
            self._food_sprite_list = sprite_list
            self._food_sprite_list_keys.clear()

        try:
            visible_keys = {
                self._food_sprite_key(food)
                for food, _draw_x, _draw_y, _radius in visible_foods
            }
            for food_key in self._food_sprite_list_keys - visible_keys:
                sprite = self._food_sprite_cache.get(food_key)
                if sprite is not None:
                    sprite_list.remove(sprite)
            self._food_sprite_list_keys.intersection_update(visible_keys)

            for food, draw_x, draw_y, radius in visible_foods:
                food_key = self._food_sprite_key(food)
                sprite = self._food_sprite_cache.get(food_key)
                if sprite is None:
                    sprite = self._create_food_sprite()
                    self._food_sprite_cache[food_key] = sprite

                sprite.center_x = draw_x
                sprite.center_y = draw_y
                sprite.width = radius * 2
                sprite.height = radius * 2
                if food_key not in self._food_sprite_list_keys:
                    sprite_list.append(sprite)
                    self._food_sprite_list_keys.add(food_key)

            self._prune_food_sprite_cache(visible_keys)
            if visible_foods:
                sprite_list.draw()
        except Exception:
            self._food_batch_disabled = True
            return False

        return True

    def _should_draw_food_outlines(self, food_count: int, zoom: float) -> bool:
        """Return whether should draw food outlines.

        Parameters
        ----------
        food_count
            Value used by the operation.
        zoom
            Value used by the operation.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        if self._food_outlines_enabled is None:
            self._food_outlines_enabled = food_count <= 250
        elif (
            self._food_outlines_enabled
            and food_count >= self.FOOD_OUTLINE_DISABLE_COUNT
        ):
            self._food_outlines_enabled = False
        elif (
            not self._food_outlines_enabled
            and food_count <= self.FOOD_OUTLINE_ENABLE_COUNT
        ):
            self._food_outlines_enabled = True
        return zoom >= 1.25 or self._food_outlines_enabled

    def _has_active_window(self) -> bool:
        """Return whether has active window.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        try:
            return arcade.get_window() is not None
        except (AttributeError, RuntimeError):
            return False

    def _create_food_sprite(self) -> object:
        """Create food sprite.

        Returns
        -------
        object
            Computed result.
        """
        sprite_cls = getattr(arcade, "Sprite", None)
        if sprite_cls is None:
            raise RuntimeError("Arcade Sprite is unavailable.")
        return sprite_cls(self._food_circle_texture())

    def _food_circle_texture(self) -> object:
        """Return food circle texture.

        Returns
        -------
        object
            Computed result.
        """
        if self._food_sprite_texture is not None:
            return self._food_sprite_texture

        make_circle_texture = getattr(arcade, "make_circle_texture", None)
        if make_circle_texture is None:
            raise RuntimeError("Arcade circle texture factory is unavailable.")

        self._food_sprite_texture = make_circle_texture(
            self.FOOD_SPRITE_TEXTURE_DIAMETER,
            self._rgba_color(self.theme.food_fill),
            name=f"food-fill-{self.theme.food_fill}",
        )
        return self._food_sprite_texture

    def _rgba_color(
        self,
        color: arcade.Color | tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        """Return rgba color.

        Parameters
        ----------
        color
            Arcade-compatible color.

        Returns
        -------
        tuple[int, int, int, int]
            Computed collection.
        """
        channels = tuple(color)
        if len(channels) >= 4:
            return (
                int(channels[0]),
                int(channels[1]),
                int(channels[2]),
                int(channels[3]),
            )
        return (int(channels[0]), int(channels[1]), int(channels[2]), 255)

    def _food_sprite_key(self, food: Food) -> int:
        """Return food sprite key.

        Parameters
        ----------
        food
            Value used by the operation.

        Returns
        -------
        int
            Computed result.
        """
        return getattr(food, "id", id(food))

    def _prune_food_sprite_cache(self, visible_keys: set[int]) -> None:
        """Remove stale food sprite cache.

        Parameters
        ----------
        visible_keys
            Value used by the operation.
        """
        if len(self._food_sprite_cache) <= max(2000, len(visible_keys) * 3):
            return
        self._food_sprite_cache = {
            key: sprite
            for key, sprite in self._food_sprite_cache.items()
            if key in visible_keys
        }

    def _draw_creatures(
        self,
        creatures: list[Creature],
        bounds: arcade.Rect,
        world: World,
        selected_creature_id: int | None,
    ) -> None:
        """Draw creatures.

        Parameters
        ----------
        creatures
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        selected_creature_id
            Value used by the operation.
        """
        zoom = world.environment_zoom
        visible_creatures: list[tuple[Creature, float, float, float]] = []
        for creature in creatures:
            model_x, model_y = creature.position
            draw_x = bounds.center_x + model_x * zoom + world.environment_pan_x
            draw_y = bounds.center_y + model_y * zoom + world.environment_pan_y
            radius = max(3.0, creature.radius * zoom)
            if not self._circle_intersects_visible_bounds(bounds, draw_x, draw_y, radius):
                continue
            visible_creatures.append((creature, draw_x, draw_y, radius))

        if not self._draw_creature_sprite_batch(visible_creatures, zoom, world.creatures):
            self._draw_creatures_immediate(visible_creatures, world, zoom)
        del selected_creature_id

    def _draw_creatures_immediate(
        self,
        visible_creatures: list[tuple[Creature, float, float, float]],
        world: World,
        zoom: float,
    ) -> None:
        """Draw creatures immediate.

        Parameters
        ----------
        visible_creatures
            Value used by the operation.
        world
            Simulation world providing current state.
        zoom
            Value used by the operation.
        """
        for creature, draw_x, draw_y, radius in visible_creatures:
            arcade.draw_circle_filled(draw_x, draw_y, radius, creature.color)
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

    def _draw_creature_sprite_batch(
        self,
        visible_creatures: list[tuple[Creature, float, float, float]],
        zoom: float,
        active_creatures: list[Creature],
    ) -> bool:
        """Draw creature sprite batch.

        Parameters
        ----------
        visible_creatures
            Value used by the operation.
        zoom
            Value used by the operation.
        active_creatures
            Value used by the operation.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        if self._creature_batch_disabled or not visible_creatures:
            return False

        if not self._has_active_window():
            return False

        sprite_list = self._creature_sprite_list
        detail_sprite_list = self._creature_detail_sprite_list
        if sprite_list is None:
            sprite_list_cls = getattr(arcade, "SpriteList", None)
            if sprite_list_cls is None:
                self._creature_batch_disabled = True
                return False
            try:
                sprite_list = sprite_list_cls(
                    use_spatial_hash=False,
                    capacity=max(100, len(visible_creatures)),
                )
                detail_sprite_list = sprite_list_cls(
                    use_spatial_hash=False,
                    capacity=max(100, len(visible_creatures)),
                )
            except Exception:
                self._creature_batch_disabled = True
                return False
            self._creature_sprite_list = sprite_list
            self._creature_detail_sprite_list = detail_sprite_list
        elif detail_sprite_list is None:
            sprite_list_cls = getattr(arcade, "SpriteList", None)
            if sprite_list_cls is None:
                self._creature_batch_disabled = True
                return False
            try:
                detail_sprite_list = sprite_list_cls(
                    use_spatial_hash=False,
                    capacity=max(100, len(visible_creatures)),
                )
            except Exception:
                self._creature_batch_disabled = True
                return False
            self._creature_detail_sprite_list = detail_sprite_list

        try:
            sprite_list.clear()
            detail_sprite_list.clear()
            for creature, draw_x, draw_y, _radius in visible_creatures:
                creature_key = self._creature_sprite_key(creature)
                sprite = self._creature_sprite_cache.get(creature_key)
                if sprite is None:
                    sprite = self._create_creature_sprite()
                    self._creature_sprite_cache[creature_key] = sprite
                    try:
                        creature.render_sprite = sprite
                    except AttributeError:
                        pass
                detail_sprite = self._creature_detail_sprite_cache.get(creature_key)
                if detail_sprite is None:
                    detail_sprite = self._create_creature_detail_sprite()
                    self._creature_detail_sprite_cache[creature_key] = detail_sprite

                sprite.center_x = draw_x
                sprite.center_y = draw_y
                sprite.scale = (
                    creature.radius * zoom / self.CREATURE_BASE_TEXTURE_RADIUS
                )
                sprite.angle = self._creature_sprite_angle(creature)
                sprite.color = creature.color
                sprite_list.append(sprite)

                detail_sprite.center_x = draw_x
                detail_sprite.center_y = draw_y
                detail_sprite.scale = sprite.scale
                detail_sprite.angle = sprite.angle
                detail_sprite.color = (255, 255, 255, 255)
                detail_sprite_list.append(detail_sprite)

            active_keys = (
                {
                    self._creature_sprite_key(creature)
                    for creature in active_creatures
                }
            )
            self._prune_creature_sprite_cache(active_keys)
            sprite_list.draw()
            detail_sprite_list.draw()
        except Exception:
            self._creature_batch_disabled = True
            return False

        return True

    def _create_creature_sprite(self) -> object:
        """Create creature sprite.

        Returns
        -------
        object
            Computed result.
        """
        sprite_cls = getattr(arcade, "Sprite", None)
        if sprite_cls is None:
            raise RuntimeError("Arcade Sprite is unavailable.")
        return sprite_cls(self._creature_base_texture())

    def _create_creature_detail_sprite(self) -> object:
        """Create creature detail sprite.

        Returns
        -------
        object
            Computed result.
        """
        sprite_cls = getattr(arcade, "Sprite", None)
        if sprite_cls is None:
            raise RuntimeError("Arcade Sprite is unavailable.")
        return sprite_cls(self._creature_detail_base_texture())

    def _creature_base_texture(self) -> object:
        """Return creature base texture.

        Returns
        -------
        object
            Computed result.
        """
        if self._creature_sprite_texture is not None:
            return self._creature_sprite_texture

        make_circle_texture = getattr(arcade, "make_circle_texture", None)
        if make_circle_texture is None:
            raise RuntimeError("Arcade circle texture factory is unavailable.")

        texture = make_circle_texture(
            self.CREATURE_BASE_TEXTURE_DIAMETER,
            (255, 255, 255, 255),
            name="creature-base-body",
        )
        self._creature_sprite_texture = texture
        return self._creature_sprite_texture

    def _creature_detail_base_texture(self) -> object:
        """Return creature detail base texture.

        Returns
        -------
        object
            Computed result.
        """
        if self._creature_detail_texture is not None:
            return self._creature_detail_texture

        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise RuntimeError("Pillow is required for creature detail texture.")

        try:
            image = Image.new(
                "RGBA",
                (
                    self.CREATURE_BASE_TEXTURE_DIAMETER,
                    self.CREATURE_BASE_TEXTURE_DIAMETER,
                ),
                (0, 0, 0, 0),
            )
            draw = ImageDraw.Draw(image)
            center = self.CREATURE_BASE_TEXTURE_RADIUS
            marker_radius = 16
            draw.ellipse(
                (
                    center - marker_radius,
                    center + 76 - marker_radius,
                    center + marker_radius,
                    center + 76 + marker_radius,
                ),
                fill=(76, 76, 76, 255),
            )
            eye_radius = 12
            for eye_x in (center - 34, center + 34):
                draw.ellipse(
                    (
                        eye_x - eye_radius,
                        center + 42 - eye_radius,
                        eye_x + eye_radius,
                        center + 42 + eye_radius,
                    ),
                    fill=(255, 255, 255, 255),
                )
            texture_factory = getattr(arcade, "Texture", None)
            if texture_factory is None:
                raise RuntimeError("Arcade Texture is unavailable.")
            try:
                self._creature_detail_texture = texture_factory(
                    image,
                    hash="creature-face-details",
                )
            except TypeError:
                self._creature_detail_texture = texture_factory(image)
            return self._creature_detail_texture
        except Exception as exc:
            raise RuntimeError("Could not create creature detail texture.") from exc

    def _creature_sprite_angle(self, creature: Creature) -> float:
        """Return creature sprite angle.

        Parameters
        ----------
        creature
            Value used by the operation.

        Returns
        -------
        float
            Computed result.
        """
        return (270.0 - degrees(creature.heading)) % 360.0

    def _creature_sprite_key(self, creature: Creature) -> int:
        """Return creature sprite key.

        Parameters
        ----------
        creature
            Value used by the operation.

        Returns
        -------
        int
            Computed result.
        """
        return getattr(creature, "creature_id", id(creature))

    def _prune_creature_sprite_cache(self, active_keys: set[int]) -> None:
        """Remove stale creature sprite cache.

        Parameters
        ----------
        active_keys
            Value used by the operation.
        """
        self._creature_sprite_cache = {
            key: sprite
            for key, sprite in self._creature_sprite_cache.items()
            if key in active_keys
        }
        self._creature_detail_sprite_cache = {
            key: sprite
            for key, sprite in self._creature_detail_sprite_cache.items()
            if key in active_keys
        }

    def _draw_selected_creature_status(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        """Draw selected creature status.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.
        """
        creature = world.selected_creature
        if creature is None:
            return

        zoom = world.environment_zoom
        draw_x, draw_y = world.environment_to_screen(*creature.position)
        radius = max(3.0, creature.radius * zoom)
        if not self._circle_intersects_visible_bounds(
            bounds,
            draw_x,
            draw_y,
            radius,
        ):
            return

        arcade.draw_circle_outline(
            draw_x,
            draw_y,
            radius,
            self.theme.selected_outline,
            2,
        )
        self._draw_metabolism_bars(creature, bounds, world)

    def _draw_metabolism_bars(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw metabolism bars.

        Parameters
        ----------
        creature
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        zoom = world.environment_zoom
        center_x, center_y = creature.position
        draw_x, draw_y = world.environment_to_screen(center_x, center_y)
        radius = creature.radius * zoom

        width = max(20.0, radius * 2.1)
        height = max(3.0, 5.0 * zoom)
        gap = max(1.0, 2.0 * zoom)
        left = draw_x - width / 2
        energy_bottom = draw_y + radius + (8.0 * zoom)
        life_bottom = energy_bottom + height + gap
        stomach_bottom = energy_bottom - height - gap
        max_energy = max(0.0001, self.config.metabolism.max_energy)
        energy_ratio = max(0.0, min(1.0, creature.energy / max_energy))
        max_life = max(
            0.0001,
            float(getattr(self.config.metabolism, "max_life", 1.0)),
        )
        life_ratio = max(
            0.0,
            min(1.0, float(getattr(creature, "life", max_life)) / max_life),
        )
        stomach_capacity = max(
            0.0,
            float(
                getattr(
                    getattr(creature, "physical_traits", None),
                    "stomach_capacity",
                    creature.radius
                    * self.config.metabolism.stomach_capacity_per_radius,
                )
            ),
        )
        stomach_ratio = (
            0.0
            if stomach_capacity <= 0.0
            else max(
                0.0,
                min(
                    1.0,
                    getattr(creature, "stomach_energy", 0.0) / stomach_capacity,
                ),
            )
        )
        if not self._rect_fits_visible_bounds(
            bounds,
            left,
            stomach_bottom,
            left + width,
            life_bottom + height,
        ):
            return

        for bottom, ratio, color in (
            (life_bottom, life_ratio, (226, 74, 74, 230)),
            (energy_bottom, energy_ratio, (70, 140, 235, 230)),
            (stomach_bottom, stomach_ratio, (236, 153, 45, 230)),
        ):
            arcade.draw_lrbt_rectangle_filled(
                left,
                left + width,
                bottom,
                bottom + height,
                (25, 30, 36, 180),
            )
            if ratio > 0.0:
                arcade.draw_lrbt_rectangle_filled(
                    left,
                    left + width * ratio,
                    bottom,
                    bottom + height,
                    color,
                )

    def _draw_selected_overlay(
        self,
        world: World,
        bounds: arcade.Rect,
    ) -> None:
        """Draw selected overlay.

        Parameters
        ----------
        world
            Simulation world providing current state.
        bounds
            Rectangle defining the relevant UI area.
        """
        selected = world.selected_creature
        if selected is None:
            return

        if world.debug_vision_enabled:
            self._draw_vision_cone(selected, bounds, world)
            self._draw_flock_perception_radius(selected, bounds, world)
            self._draw_biome_sensor_markers(selected, bounds, world)
            self._draw_acoustic_debug(selected, bounds, world)
            self._draw_pheromone_debug(selected, bounds, world)
            self._draw_flock_steering_debug(selected, bounds, world)
            self._draw_flocking_velocity_debug(selected, bounds, world)
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

    def _draw_flock_steering_debug(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw flock steering debug.

        Parameters
        ----------
        creature
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        del bounds
        debug_by_creature = getattr(world, "_last_flock_steering_debug", {})
        debug = debug_by_creature.get(creature.creature_id)
        if debug is None:
            return

        force_x, force_y = debug.accepted_counterfactual_delta
        magnitude = hypot(force_x, force_y)
        max_force = float(debug.max_force)
        if magnitude <= 1e-9 or max_force <= 0.0:
            return

        unit_x = force_x / magnitude
        unit_y = force_y / magnitude
        strength = max(0.0, min(1.0, magnitude / max_force))
        center_x, center_y = world.environment_to_screen(*creature.position)
        start_offset = creature.radius * world.environment_zoom + 4.0
        start_x = center_x + unit_x * start_offset
        start_y = center_y + unit_y * start_offset
        shaft_length = 52.0 * strength
        end_x = start_x + unit_x * shaft_length
        end_y = start_y + unit_y * shaft_length
        color = (255, 170, 70, int(100 + 155 * strength))
        line_width = 1.5 + 1.5 * strength

        arcade.draw_line(start_x, start_y, end_x, end_y, color, line_width)
        arrowhead_length = min(8.0, shaft_length * 0.4)
        arrowhead_half_width = min(4.5, shaft_length * 0.225)
        base_x = end_x - unit_x * arrowhead_length
        base_y = end_y - unit_y * arrowhead_length
        perpendicular_x = -unit_y * arrowhead_half_width
        perpendicular_y = unit_x * arrowhead_half_width
        arcade.draw_line(
            end_x,
            end_y,
            base_x + perpendicular_x,
            base_y + perpendicular_y,
            color,
            line_width,
        )
        arcade.draw_line(
            end_x,
            end_y,
            base_x - perpendicular_x,
            base_y - perpendicular_y,
            color,
            line_width,
        )

    def _draw_flocking_velocity_debug(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw cached neural, social, blended, and avoidance vectors."""
        del bounds
        runtime = getattr(world, "_last_flocking_runtime", {}).get(
            creature.creature_id
        )
        if runtime is None:
            return
        vectors = (
            (runtime.neural_desired_velocity, (65, 125, 220, 210)),
            (runtime.intent.desired_velocity, (145, 85, 210, 210)),
            (runtime.blended_desired_velocity, (45, 165, 105, 220)),
            (runtime.mandatory_avoidance, (205, 60, 70, 230)),
        )
        center_x, center_y = world.environment_to_screen(*creature.position)
        start_offset = creature.radius * world.environment_zoom + 4.0
        scale = 56.0 / max(1.0, world.MAX_SPEED)
        for vector, color in vectors:
            magnitude = hypot(*vector)
            if magnitude <= 1e-9:
                continue
            unit_x = vector[0] / magnitude
            unit_y = vector[1] / magnitude
            start_x = center_x + unit_x * start_offset
            start_y = center_y + unit_y * start_offset
            length = min(64.0, max(8.0, magnitude * scale))
            arcade.draw_line(
                start_x,
                start_y,
                start_x + unit_x * length,
                start_y + unit_y * length,
                color,
                2.0,
            )

    def _draw_acoustic_debug(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw acoustic debug.

        Parameters
        ----------
        creature
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        snapshots = getattr(world, "_last_sensor_snapshots", {})
        snapshot = snapshots.get(creature.creature_id)
        acoustic = None if snapshot is None else getattr(snapshot, "acoustic", None)
        debug_by_creature = getattr(world, "_last_acoustic_debug", {})
        debug = debug_by_creature.get(creature.creature_id)
        source_position = None if debug is None else debug.source_position
        strength = 0.0 if acoustic is None else float(acoustic.strength)
        if source_position is None or strength <= 0.0:
            return
        start_x, start_y = world.environment_to_screen(*creature.position)
        end_x, end_y = world.environment_to_screen(*source_position)
        alpha = int(80 + 175 * max(0.0, min(1.0, strength)))
        color = (125, 205, 255, alpha)
        arcade.draw_line(start_x, start_y, end_x, end_y, color, 1.0 + 2.0 * strength)
        arcade.draw_circle_outline(end_x, end_y, 7.0, color, 2)
        self._draw_text(
            f"sound_strength_{creature.creature_id}",
            f"SND {strength:.2f}",
            start_x + 12,
            start_y + 12,
            color,
            10,
            bold=True,
        )

    def _draw_pheromone_debug(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw pheromone debug.

        Parameters
        ----------
        creature
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        snapshots = getattr(world, "_last_sensor_snapshots", {})
        snapshot = snapshots.get(creature.creature_id)
        pheromones = None if snapshot is None else getattr(snapshot, "pheromones", None)
        positions_for = getattr(world, "pheromone_sensor_positions_for", None)
        if pheromones is None or positions_for is None:
            return
        rgb_values = (
            pheromones.local,
            pheromones.forward_left,
            pheromones.forward_right,
        )
        radius = max(4.0, 5.0 * world.environment_zoom)
        for position, rgb in zip(
            positions_for(creature),
            rgb_values,
        ):
            draw_x, draw_y = world.environment_to_screen(*position)
            if not self._circle_intersects_visible_bounds(bounds, draw_x, draw_y, radius):
                continue
            color = tuple(int(255 * max(0.0, min(1.0, value))) for value in rgb)
            alpha = max(45, max(color))
            arcade.draw_circle_outline(
                draw_x,
                draw_y,
                radius + 2.0,
                (*color, alpha),
                2,
            )
            arcade.draw_circle_filled(
                draw_x,
                draw_y,
                radius,
                (*color, alpha),
            )

    def _draw_vision_cone(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw vision cone.

        Parameters
        ----------
        creature
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        zoom = world.environment_zoom
        left_eye_model, right_eye_model = self._creature_eye_positions(creature)
        left_eye = world.environment_to_screen(*left_eye_model)
        right_eye = world.environment_to_screen(*right_eye_model)
        eye_origin = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )

        eye_cone_points = self._vision_cone_points(creature, eye_origin, zoom)

        arcade.draw_polygon_filled(eye_cone_points, self.theme.vision_fill)
        arcade.draw_polygon_outline(eye_cone_points, (111, 220, 128, 132), 1)

    def _draw_flock_perception_radius(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw the selected creature's omnidirectional Boid radius."""
        del bounds
        radius = (
            self.config.flocking.perception_radius
            * world.environment_zoom
        )
        if radius <= 0.0:
            return
        center_x, center_y = world.environment_to_screen(*creature.position)
        arcade.draw_circle_outline(
            center_x,
            center_y,
            radius,
            self.theme.flock_perception_outline,
            2,
        )

    def _draw_biome_sensor_markers(
        self,
        creature: Creature,
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw biome sensor markers.

        Parameters
        ----------
        creature
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
        sensor_positions_for = getattr(world, "biome_sensor_positions_for", None)
        if sensor_positions_for is None:
            return

        colors = (
            (255, 255, 255, 210),
            (110, 220, 180, 210),
            (110, 180, 255, 210),
        )
        zoom = world.environment_zoom
        radius = max(3.0, 4.5 * zoom)
        for position, color in zip(sensor_positions_for(creature), colors):
            draw_x, draw_y = world.environment_to_screen(*position)
            if not self._circle_intersects_visible_bounds(bounds, draw_x, draw_y, radius):
                continue
            arcade.draw_circle_filled(draw_x, draw_y, radius, color)
            arcade.draw_circle_outline(draw_x, draw_y, radius + 2.0, color, 1)

    def _vision_cone_points(
        self,
        creature: Creature,
        origin: tuple[float, float],
        zoom: float,
        start_factor: float = 0.0,
        end_factor: float = 1.0,
    ) -> list[tuple[float, float]]:
        """Return vision cone points.

        Parameters
        ----------
        creature
            Value used by the operation.
        origin
            Value used by the operation.
        zoom
            Value used by the operation.
        start_factor
            Value used by the operation.
        end_factor
            Value used by the operation.

        Returns
        -------
        list[tuple[float, float]]
            Computed collection.
        """
        heading = creature.heading
        cone_radius = creature.vision.range * zoom
        cone_angle = creature.vision.angle
        start_factor = max(0.0, min(1.0, start_factor))
        end_factor = max(start_factor, min(1.0, end_factor))
        steps = max(3, ceil(18 * (end_factor - start_factor)))

        points: list[tuple[float, float]] = [origin]
        for index in range(steps + 1):
            factor = index / steps
            cone_factor = start_factor + (end_factor - start_factor) * factor
            angle = heading - cone_angle / 2 + cone_angle * cone_factor
            points.append(
                (
                    origin[0] + cos(angle) * cone_radius,
                    origin[1] + sin(angle) * cone_radius,
                )
            )

        return points

    def _draw_visible_food_highlights(
        self,
        foods: list[Food],
        bounds: arcade.Rect,
        world: World,
    ) -> None:
        """Draw visible food highlights.

        Parameters
        ----------
        foods
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
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
        """Draw visible creature highlights.

        Parameters
        ----------
        creatures
            Value used by the operation.
        bounds
            Rectangle defining the relevant UI area.
        world
            Simulation world providing current state.
        """
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
        """Return creature eye positions.

        Parameters
        ----------
        creature
            Value used by the operation.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            Computed collection.
        """
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
        """Return circle fits visible bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        radius
            Requested logical size.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        return (
            x - radius >= bounds.left
            and x + radius <= bounds.right
            and y - radius >= bounds.bottom
            and y + radius <= bounds.top
        )

    def _circle_intersects_visible_bounds(
        self, bounds: arcade.Rect, x: float, y: float, radius: float
    ) -> bool:
        """Return circle intersects visible bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        radius
            Requested logical size.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
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
        """Return rect fits visible bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        left
            Logical screen coordinate.
        bottom
            Logical screen coordinate.
        right
            Logical screen coordinate.
        top
            Logical screen coordinate.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        return (
            left >= bounds.left
            and right <= bounds.right
            and bottom >= bounds.bottom
            and top <= bounds.top
        )

    def _rect_intersects_visible_bounds(
        self,
        bounds: arcade.Rect,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> bool:
        """Return rect intersects visible bounds.

        Parameters
        ----------
        bounds
            Rectangle defining the relevant UI area.
        left
            Logical screen coordinate.
        bottom
            Logical screen coordinate.
        right
            Logical screen coordinate.
        top
            Logical screen coordinate.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        return (
            right >= bounds.left
            and left <= bounds.right
            and top >= bounds.bottom
            and bottom <= bounds.top
        )
