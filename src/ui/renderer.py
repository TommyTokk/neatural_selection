from __future__ import annotations

import arcade

from configs.sim_config import SimConfig
from src.ui.components.state import (
    BrainWindowState,
    PanelState,
    SpeciesTreeState,
)
from src.ui.common.drawing import ArcadePainter
from src.ui.common.interaction import UiInteractionState
from src.ui.layouts.species_tree import TreeLayoutManager
from src.world import World
from src.ui.components.species_tree.window import SpeciesTreeWindowComponent
from src.ui.components.navigation import NavigationComponent
from src.ui.components.panels.manager import PanelManagerComponent
from src.ui.components.panels.inspector import InspectorPanelComponent
from src.ui.components.panels.stats import StatsPanelComponent
from src.ui.components.panels.settings import SettingsPanelComponent
from src.ui.components.brain.window import BrainWindowComponent
from src.ui.components.brain.graph import BrainGraphComponent
from src.ui.components.brain.inspector import BrainInspectorComponent
from src.ui.components.species_tree.inspector import SpeciesTreeInspectorComponent
from src.ui.components.species_tree.canvas import SpeciesTreeCanvasComponent
from src.ui.common.widgets import CommonUiComponent

class UiRenderer(
    SpeciesTreeWindowComponent,
    NavigationComponent,
    PanelManagerComponent,
    InspectorPanelComponent,
    StatsPanelComponent,
    SettingsPanelComponent,
    BrainWindowComponent,
    BrainGraphComponent,
    BrainInspectorComponent,
    SpeciesTreeInspectorComponent,
    SpeciesTreeCanvasComponent,
    CommonUiComponent,
):
    """Coordinate presentation components and route simulation UI input."""
    ICON_BUTTON_SIZE = 58.0
    ICON_BUTTON_GAP = 20.0
    RAIL_VERTICAL_PADDING = 32.0
    PANEL_KEYS = ("inspector", "stats", "settings")
    SPECIES_TREE_MIN_ZOOM = 0.1
    SPECIES_TREE_MAX_ZOOM = 2.0
    SPECIES_TREE_ZOOM_FACTOR = 1.2
    SPECIES_TREE_TIME_SCALE = 2.0
    SPECIES_TREE_CONTENT_PADDING = 48.0
    SPECIES_TREE_TIMELINE_WIDTH = 118.0
    SPECIES_TREE_TIMELINE_GAP = 12.0
    _STATE_FIELDS = {
        "_active_slider": ("_panel_state", "active_slider"),
        "_panel_open": ("_panel_state", "open_panels"),
        "_map_submenu_open": ("_panel_state", "map_submenu_open"),
        "_panel_bounds": ("_panel_state", "bounds"),
        "_active_panel_drag": ("_panel_state", "active_drag"),
        "_panel_drag_offset": ("_panel_state", "drag_offset"),
        "_brain_window_open": ("_brain_state", "open"),
        "_brain_window_bounds": ("_brain_state", "bounds"),
        "_brain_graph_zoom": ("_brain_state", "graph_zoom"),
        "_brain_node_bounds": ("_brain_state", "node_bounds"),
        "_brain_selected_node_key": ("_brain_state", "selected_node_key"),
        "_brain_node_inspector_open": ("_brain_state", "node_inspector_open"),
        "_brain_selection_identity": ("_brain_state", "selection_identity"),
        "_species_tree_open": ("_species_tree_state", "open"),
        "_species_tree_previous_pause": ("_species_tree_state", "previous_pause"),
        "_species_tree_mouse": ("_species_tree_state", "mouse"),
        "_species_tree_hovered_id": ("_species_tree_state", "hovered_id"),
        "_species_tree_selected_id": ("_species_tree_state", "selected_id"),
        "_species_tree_pending_selection_id": (
            "_species_tree_state",
            "pending_selection_id",
        ),
        "_species_tree_report": ("_species_tree_state", "report"),
        "_species_tree_report_species_id": (
            "_species_tree_state",
            "report_species_id",
        ),
        "_species_tree_radar_texture": ("_species_tree_state", "radar_texture"),
        "_species_tree_radar_species_id": (
            "_species_tree_state",
            "radar_species_id",
        ),
        "_species_tree_radar_future": ("_species_tree_state", "radar_future"),
        "_species_tree_radar_executor": (
            "_species_tree_state",
            "radar_executor",
        ),
        "_species_tree_radar_error": ("_species_tree_state", "radar_error"),
        "_species_tree_horizontal_offset": (
            "_species_tree_state",
            "horizontal_offset",
        ),
        "_species_tree_vertical_offset": (
            "_species_tree_state",
            "vertical_offset",
        ),
        "_species_tree_horizontal_limit": (
            "_species_tree_state",
            "horizontal_limit",
        ),
        "_species_tree_vertical_limit": (
            "_species_tree_state",
            "vertical_limit",
        ),
        "_species_tree_horizontal_offset_min": (
            "_species_tree_state",
            "horizontal_offset_min",
        ),
        "_species_tree_horizontal_offset_max": (
            "_species_tree_state",
            "horizontal_offset_max",
        ),
        "_species_tree_vertical_offset_min": (
            "_species_tree_state",
            "vertical_offset_min",
        ),
        "_species_tree_vertical_offset_max": (
            "_species_tree_state",
            "vertical_offset_max",
        ),
        "_species_tree_scroll_drag": ("_species_tree_state", "scroll_drag"),
        "_species_tree_scroll_drag_offset": (
            "_species_tree_state",
            "scroll_drag_offset",
        ),
        "_species_tree_canvas_drag": ("_species_tree_state", "canvas_drag"),
        "_species_tree_canvas_drag_started": (
            "_species_tree_state",
            "canvas_drag_started",
        ),
        "_species_tree_canvas_drag_last": (
            "_species_tree_state",
            "canvas_drag_last",
        ),
        "_species_tree_inspector_width": (
            "_species_tree_state",
            "inspector_width",
        ),
        "_species_tree_inspector_resize_drag": (
            "_species_tree_state",
            "inspector_resize_drag",
        ),
        "_species_tree_timeline_bucket_bounds": (
            "_species_tree_state",
            "timeline_bucket_bounds",
        ),
        "_species_tree_node_bounds": ("_species_tree_state", "node_bounds"),
        "_species_tree_zoom": ("_species_tree_state", "zoom"),
        "_species_tree_fit_mode": ("_species_tree_state", "fit_mode"),
        "_species_tree_fit_requested": ("_species_tree_state", "fit_requested"),
        "_species_tree_last_layout": ("_species_tree_state", "last_layout"),
        "_species_tree_last_canvas": ("_species_tree_state", "last_canvas"),
        "_species_tree_layout_manager": (
            "_species_tree_state",
            "layout_manager",
        ),
        "_species_tree_extinction_times": (
            "_species_tree_state",
            "extinction_times",
        ),
        "_species_tree_cached_layout": ("_species_tree_state", "cached_layout"),
        "_species_tree_visible_slice": ("_species_tree_state", "visible_slice"),
        "_species_tree_focus_latest_pending": (
            "_species_tree_state",
            "focus_latest_pending",
        ),
        "_species_tree_highlight_cache_id": (
            "_species_tree_state",
            "highlight_cache_id",
        ),
        "_species_tree_highlight_nodes": (
            "_species_tree_state",
            "highlight_nodes",
        ),
        "_species_tree_highlight_edges": (
            "_species_tree_state",
            "highlight_edges",
        ),
        "_species_tree_neat_label_signature": (
            "_species_tree_state",
            "neat_label_signature",
        ),
        "_species_tree_neat_labels": ("_species_tree_state", "neat_labels"),
        "_species_tree_sync_signature": (
            "_species_tree_state",
            "sync_signature",
        ),
    }

    def __getattr__(self, name: str) -> object:
        """Read a legacy renderer attribute from its component state.

        Parameters
        ----------
        name
            Attribute requested by renderer code or existing integrations.

        Returns
        -------
        object
            Value owned by the corresponding component state.

        Raises
        ------
        AttributeError
            If the name is not a mapped component-state field.
        """
        target = self._STATE_FIELDS.get(name)
        if target is None:
            raise AttributeError(name)
        state_name, field_name = target
        return getattr(object.__getattribute__(self, state_name), field_name)

    def __setattr__(self, name: str, value: object) -> None:
        """Write legacy renderer attributes into component state.

        Parameters
        ----------
        name
            Attribute being assigned.
        value
            New attribute value.
        """
        target = self._STATE_FIELDS.get(name)
        if target is not None:
            state_name, field_name = target
            state = self.__dict__.get(state_name)
            if state is not None:
                setattr(state, field_name, value)
                return
        object.__setattr__(self, name, value)

    def __init__(self, config: SimConfig) -> None:
        """Initialize the component.

        Parameters
        ----------
        config
            Simulation configuration.
        """
        self._panel_state = PanelState()
        self._brain_state = BrainWindowState()
        self._species_tree_state = SpeciesTreeState(
            TreeLayoutManager(
                horizontal_gap=92.0,
                time_scale=self.SPECIES_TREE_TIME_SCALE,
                padding=self.SPECIES_TREE_CONTENT_PADDING,
            )
        )
        self.config = config
        self.theme = config.theme
        self._painter = ArcadePainter()
        self._interaction = UiInteractionState()
        # Preserve the established diagnostic/test attributes while assigning
        # ownership to the shared drawing and interaction services.
        self._text_cache = self._painter.text_cache
        self._texture_cache = self._painter.texture_cache
        self._sprite_cache = self._painter.sprite_cache
        self._control_hitboxes = self._interaction.hitboxes
        self._scroll_regions = self._interaction.scroll_regions
        self._scroll_offsets = self._interaction.scroll_offsets
        self._scroll_limits = self._interaction.scroll_limits

    def draw(self, world: World) -> None:
        """Draw every simulation UI layer.

        Parameters
        ----------
        world
            Simulation world providing current state.
        """
        self._interaction.begin_frame()
        self._brain_node_bounds.clear()
        # Components draw back-to-front so modal windows remain the topmost
        # visual and interaction layer.
        self._draw_icon_rail(world)
        self._draw_floating_panels(world)
        self._draw_brain_window(world)
        self._draw_species_tree_window(world)


    def close(self) -> None:
        """Release asynchronous UI resources owned by this renderer."""
        self._clear_species_radar_state()
        executor = self._species_tree_radar_executor
        self._species_tree_radar_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def handle_key_press(self, world: World, symbol: int, modifiers: int) -> bool:
        """Handle key press.

        Parameters
        ----------
        world
            Simulation world providing current state.
        symbol
            Arcade input value.
        modifiers
            Arcade input value.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        del world, symbol, modifiers
        return self._species_tree_open

    def handle_mouse_motion(
        self,
        world: World,
        x: float,
        y: float,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> bool:
        """Handle mouse motion.

        Parameters
        ----------
        world
            Simulation world providing current state.
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
        bool
            Whether the operation succeeded or consumed the input.
        """
        del world, dx, dy
        self._species_tree_mouse = (x, y)
        if not self._species_tree_open:
            self._species_tree_hovered_id = None
            return False
        self._species_tree_hovered_id = self._species_tree_node_at(x, y)
        return True


    def handle_mouse_press(self, world: World, x: float, y: float) -> bool:
        """Handle mouse press.

        Parameters
        ----------
        world
            Simulation world providing current state.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        # Dispatch from the highest z-order component downward. An open modal
        # consumes every press, including clicks outside its visible controls.
        if self._species_tree_open:
            if self._contains_hitbox("species_tree_close", x, y):
                self.close_species_tree(world)
                return True
            if self._contains_hitbox("species_tree_zoom_out", x, y):
                self._adjust_species_tree_zoom(
                    1.0 / self.SPECIES_TREE_ZOOM_FACTOR
                )
                return True
            if self._contains_hitbox("species_tree_zoom_in", x, y):
                self._adjust_species_tree_zoom(self.SPECIES_TREE_ZOOM_FACTOR)
                return True
            if self._contains_hitbox("species_tree_zoom_fit", x, y):
                self._activate_species_tree_fit()
                return True
            if self._contains_hitbox("species_tree_parent_button", x, y):
                records = getattr(world, "species_history", {})
                selected = records.get(self._species_tree_selected_id)
                parent_id = (
                    None
                    if selected is None
                    else selected.parent_species_id
                )
                if parent_id is not None and parent_id in records:
                    self._select_species_tree_species(parent_id, focus=True)
                return True
            if self._contains_hitbox("species_tree_inspector_resize", x, y):
                self._species_tree_inspector_resize_drag = True
                return True
            if self._contains_hitbox("species_tree_horizontal_thumb", x, y):
                thumb = self._control_hitboxes["species_tree_horizontal_thumb"]
                self._species_tree_scroll_drag = "horizontal"
                self._species_tree_scroll_drag_offset = x - thumb.left
                return True
            if self._contains_hitbox("species_tree_vertical_thumb", x, y):
                thumb = self._control_hitboxes["species_tree_vertical_thumb"]
                self._species_tree_scroll_drag = "vertical"
                self._species_tree_scroll_drag_offset = thumb.top - y
                return True
            if self._contains_hitbox("species_tree_horizontal_track", x, y):
                self._set_species_tree_scroll_from_pointer("horizontal", x, y)
                return True
            if self._contains_hitbox("species_tree_vertical_track", x, y):
                self._set_species_tree_scroll_from_pointer("vertical", x, y)
                return True
            for bucket_id in sorted(
                self._species_tree_timeline_bucket_bounds,
                reverse=True,
            ):
                marker = self._species_tree_timeline_bucket_bounds[bucket_id]
                if self._contains_bounds(marker, x, y):
                    bucket_seconds = (
                        self._species_tree_layout_manager.bucket_seconds
                    )
                    self._jump_species_tree_to_time(
                        (bucket_id + 0.5) * bucket_seconds
                    )
                    return True
            if self._contains_hitbox("species_tree_timeline", x, y):
                timeline = self._control_hitboxes["species_tree_timeline"]
                self._jump_species_tree_from_timeline(timeline, y)
                return True
            if self._contains_hitbox("species_tree_canvas", x, y):
                self._species_tree_pending_selection_id = (
                    self._species_tree_node_at(x, y)
                )
                self._species_tree_canvas_drag = True
                self._species_tree_canvas_drag_started = False
                self._species_tree_canvas_drag_last = (x, y)
                return True
            return True

        # The brain window sits above floating panels and the navigation rail.
        if (
            self._brain_window_open
            and self._brain_window_bounds is not None
            and self._contains_bounds(self._brain_window_bounds, x, y)
        ):
            if self._contains_hitbox("brain_window_close", x, y):
                self._close_brain_window()
                return True
            if self._contains_hitbox("brain_node_inspector_toggle", x, y):
                self._brain_node_inspector_open = not self._brain_node_inspector_open
                return True
            node_key = self._brain_node_at(x, y)
            if node_key is not None:
                self._brain_selected_node_key = node_key
                self._scroll_offsets["brain_node_inspector"] = 0.0
                return True
            if self._contains_hitbox("brain_window_graph", x, y):
                self._brain_selected_node_key = None
                self._scroll_offsets["brain_node_inspector"] = 0.0
                return True
            return True

        if self._contains_hitbox("open_map_submenu", x, y):
            self._map_submenu_open = not self._map_submenu_open
            return True
        if self._map_submenu_open:
            if self._contains_hitbox("map_layer_biome", x, y):
                world.select_environment_map("biome")
                self._map_submenu_open = False
                return True
            if self._contains_hitbox("map_layer_pheromones", x, y):
                world.select_environment_map("pheromones")
                self._map_submenu_open = False
                return True
            if self._contains_hitbox("map_submenu", x, y):
                return True
            self._map_submenu_open = False

        for panel_name in self.PANEL_KEYS:
            if self._contains_hitbox(f"{panel_name}_close", x, y):
                self._panel_open[panel_name] = False
                self._active_panel_drag = None
                return True
        if self._contains_hitbox("panel_toggle_inspector", x, y):
            self._panel_open["inspector"] = not self._panel_open["inspector"]
            return True
        if self._contains_hitbox("panel_toggle_stats", x, y):
            self._panel_open["stats"] = not self._panel_open["stats"]
            return True
        if self._contains_hitbox("panel_toggle_settings", x, y):
            self._panel_open["settings"] = not self._panel_open["settings"]
            return True
        if self._contains_hitbox("save_simulation", x, y):
            world.save_now()
            return True
        if self._contains_hitbox("open_species_tree", x, y):
            self.open_species_tree(world)
            return True
        if self._contains_hitbox("open_brain_window", x, y):
            if world.selected_creature is not None:
                self._brain_window_open = True
                self._ensure_brain_window_bounds(world)
            return True
        if self._contains_hitbox("kill_selected_creature", x, y):
            if world.kill_selected_creature():
                self._close_brain_window()
            return True
        if self._contains_hitbox("pause", x, y):
            world.toggle_pause()
            return True
        if self._contains_hitbox("reset_speed", x, y):
            world.reset_simulation_speed()
            return True
        if self._contains_hitbox("speed_down", x, y):
            world.decrease_simulation_speed()
            return True
        if self._contains_hitbox("speed_up", x, y):
            world.increase_simulation_speed()
            return True
        if self._contains_hitbox("speed_min", x, y):
            world.set_simulation_speed(world.MIN_SIMULATION_SPEED)
            return True
        if self._contains_hitbox("speed_max", x, y):
            world.set_simulation_speed(world.MAX_SIMULATION_SPEED)
            return True
        if self._contains_hitbox("speed_slider", x, y):
            self._active_slider = True
            self._set_speed_from_slider(world, x)
            return True
        for panel_name in self.PANEL_KEYS:
            if self._contains_hitbox(f"{panel_name}_drag", x, y):
                bounds = self._panel_bounds.get(panel_name)
                if bounds is not None:
                    self._active_panel_drag = panel_name
                    self._panel_drag_offset = (x - bounds.left, y - bounds.bottom)
                    return True
        for key in (
            "icon_rail",
            "inspector_panel",
            "inspector_body",
            "stats_panel",
            "stats_body",
            "settings_panel",
            "settings_body",
        ):
            if self._contains_hitbox(key, x, y):
                return True
        return False

    def handle_mouse_drag(self, world: World, x: float, y: float) -> bool:
        """Handle mouse drag.

        Parameters
        ----------
        world
            Simulation world providing current state.
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        # Active modal gestures retain capture until mouse release.
        if self._species_tree_open:
            if self._species_tree_inspector_resize_drag:
                content = self._control_hitboxes.get("species_tree_body")
                if content is not None:
                    min_width, max_width = (
                        self._species_tree_inspector_width_limits(content)
                    )
                    requested_width = content.right - x
                    self._species_tree_inspector_width = max(
                        min_width,
                        min(max_width, requested_width),
                    )
                return True
            if self._species_tree_scroll_drag is not None:
                self._set_species_tree_scroll_from_pointer(
                    self._species_tree_scroll_drag,
                    x,
                    y,
                    preserve_grab_offset=True,
                )
            elif self._species_tree_canvas_drag:
                previous_x, previous_y = self._species_tree_canvas_drag_last
                if not self._species_tree_canvas_drag_started:
                    self._species_tree_fit_mode = False
                    self._species_tree_fit_requested = False
                    self._species_tree_canvas_drag_started = True
                    self._species_tree_pending_selection_id = None
                self._species_tree_horizontal_offset -= x - previous_x
                self._species_tree_vertical_offset += y - previous_y
                self._species_tree_canvas_drag_last = (x, y)
                self._clamp_species_tree_offsets()
            return True
        if self._active_panel_drag is not None:
            bounds = self._panel_bounds.get(self._active_panel_drag)
            if bounds is None:
                return False
            offset_x, offset_y = self._panel_drag_offset
            self._panel_bounds[self._active_panel_drag] = self._clamp_panel_bounds(
                world,
                arcade.LBWH(
                    x - offset_x,
                    y - offset_y,
                    bounds.width,
                    bounds.height,
                ),
            )
            return True
        if not self._active_slider:
            return False
        self._set_speed_from_slider(world, x)
        return True

    def handle_mouse_release(self) -> None:
        """Handle mouse release.
        """
        self._active_slider = False
        self._active_panel_drag = None
        self._species_tree_scroll_drag = None
        self._species_tree_inspector_resize_drag = False
        if (
            self._species_tree_canvas_drag
            and not self._species_tree_canvas_drag_started
            and self._species_tree_pending_selection_id is not None
        ):
            self._select_species_tree_species(
                self._species_tree_pending_selection_id,
            )
        self._species_tree_pending_selection_id = None
        self._species_tree_canvas_drag = False
        self._species_tree_canvas_drag_started = False

    def handle_mouse_scroll(
        self,
        x: float,
        y: float,
        scroll_y: float,
        scroll_x: float = 0.0,
        command_down: bool = False,
    ) -> bool:
        """Handle mouse scroll.

        Parameters
        ----------
        x
            Logical screen coordinate.
        y
            Logical screen coordinate.
        scroll_y
            Value used by the operation.
        scroll_x
            Value used by the operation.
        command_down
            Whether the corresponding behavior is enabled.

        Returns
        -------
        bool
            Whether the operation succeeded or consumed the input.
        """
        # Scroll follows the same modal-first routing as pointer presses.
        if self._species_tree_open:
            inspector_region = self._scroll_regions.get(
                "species_tree_inspector"
            )
            if (
                inspector_region is not None
                and self._contains_bounds(inspector_region, x, y)
            ):
                limit = self._scroll_limits.get(
                    "species_tree_inspector",
                    0.0,
                )
                current = self._scroll_offsets.get(
                    "species_tree_inspector",
                    0.0,
                )
                self._scroll_offsets["species_tree_inspector"] = max(
                    0.0,
                    min(limit, current - scroll_y * 24.0),
                )
                return True
            if self._contains_hitbox("species_tree_canvas", x, y):
                if command_down and scroll_y != 0.0:
                    zoom_steps = max(-8.0, min(8.0, scroll_y))
                    self._adjust_species_tree_zoom(
                        self.SPECIES_TREE_ZOOM_FACTOR**zoom_steps,
                        anchor=(x, y),
                    )
                else:
                    if (
                        scroll_x != 0.0
                        and self._species_tree_horizontal_limit > 0.0
                    ):
                        self._species_tree_horizontal_offset -= scroll_x * 36.0
                    if (
                        scroll_y != 0.0
                        and self._species_tree_vertical_limit > 0.0
                    ):
                        self._species_tree_vertical_offset -= scroll_y * 36.0
                    self._clamp_species_tree_offsets()
            return True
        if (
            self._brain_window_open
            and self._brain_window_bounds is not None
            and self._contains_bounds(self._brain_window_bounds, x, y)
        ):
            inspector_region = self._scroll_regions.get("brain_node_inspector")
            if (
                inspector_region is not None
                and self._contains_bounds(inspector_region, x, y)
            ):
                limit = self._scroll_limits.get("brain_node_inspector", 0.0)
                current = self._scroll_offsets.get("brain_node_inspector", 0.0)
                self._scroll_offsets["brain_node_inspector"] = max(
                    0.0,
                    min(limit, current - scroll_y * 24.0),
                )
            return True

        for key, bounds in self._scroll_regions.items():
            if not (
                bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top
            ):
                continue

            limit = self._scroll_limits.get(key, 0.0)
            current = self._scroll_offsets.get(key, 0.0)
            self._scroll_offsets[key] = max(
                0.0,
                min(limit, current - scroll_y * 24.0),
            )
            return True

        return False
