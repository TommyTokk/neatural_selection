"""Mutable state and small view models owned by UI components."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import arcade

from src.analysis import InspectorReport
from src.ui.layouts.brain_graph import (
    BrainGraphHighlight,
    BrainGraphLayout,
)
from src.ui.layouts.species_tree import (
    SpeciesTreeLayout,
    TreeLayoutManager,
    TreeViewportSlice,
)


@dataclass(frozen=True, slots=True)
class BrainConnectionRowView:
    """Describe one connection rendered by the brain node inspector."""

    endpoint_label: str
    endpoint_key: int
    source_key: int
    target_key: int
    weight: float | None
    enabled: bool
    relation: str | None = None


@dataclass(frozen=True, slots=True)
class BrainNodeInspectorView:
    """Cache structured node details and connection rows for one selection."""

    details: tuple[tuple[str, str], ...]
    incoming_rows: tuple[BrainConnectionRowView, ...]
    outgoing_rows: tuple[BrainConnectionRowView, ...]
    route_rows: tuple[BrainConnectionRowView, ...]


@dataclass(slots=True)
class PanelState:
    """Track floating-panel, navigation, and speed-control interaction."""

    active_slider: str | None = None
    settings_expanded: bool = False
    open_panels: dict[str, bool] = field(
        default_factory=lambda: {
            "inspector": False,
            "stats": False,
            "settings": False,
        }
    )
    map_submenu_open: bool = False
    bounds: dict[str, arcade.Rect] = field(default_factory=dict)
    active_drag: str | None = None
    drag_offset: tuple[float, float] = (0.0, 0.0)
    inspector_content_height: float = 0.0
    radar_executor: ThreadPoolExecutor | None = None
    creature_radar_texture: arcade.Texture | None = None
    creature_radar_identity: tuple[int, int | None] | None = None
    creature_radar_future: Future[object] | None = None
    creature_radar_error: str | None = None


@dataclass(slots=True)
class BrainWindowState:
    """Track selected-brain window geometry and selection."""

    open: bool = False
    bounds: arcade.Rect | None = None
    graph_zoom: float = 1.0
    node_bounds: dict[int, arcade.Rect] = field(default_factory=dict)
    selected_node_key: int | None = None
    node_inspector_open: bool = True
    inspector_page: str = "node"
    connection_direction: str = "both"
    connection_filter: str = "all"
    connection_sort_descending: bool = True
    behavior_scroll_offset: float = 0.0
    why_scroll_offset: float = 0.0
    expanded_behavior: str | None = None
    expanded_why_behavior: str | None = None
    selection_identity: tuple[int, int] | None = None
    layout_cache_key: tuple[object, ...] | None = None
    layout: BrainGraphLayout | None = None
    highlight_layout: BrainGraphLayout | None = None
    highlight_node_key: int | None = None
    highlight: BrainGraphHighlight | None = None
    inspector_brain: object | None = None
    inspector_layout: BrainGraphLayout | None = None
    inspector_node_key: int | None = None
    inspector_lines: tuple[str, ...] = ()
    inspector_view: BrainNodeInspectorView | None = None


@dataclass(slots=True)
class BehaviorReportState:
    """Track the separate completed-behaviour report modal."""

    open: bool = False
    bounds: arcade.Rect | None = None
    creature_id: int | None = None
    species_id: int | None = None
    species_selected: bool = False
    historical_expanded: bool = False
    help_open: bool = False
    page: str = "timeline"
    selected_bout_id: int | None = None
    why_behavior: str | None = None


@dataclass(slots=True)
class SpeciesTreeState:
    """Track species-tree modal, viewport, selection, and async analysis."""

    layout_manager: TreeLayoutManager
    open: bool = False
    previous_pause: bool | None = None
    mouse: tuple[float, float] = (0.0, 0.0)
    hovered_id: int | None = None
    selected_id: int | None = None
    pending_selection_id: int | None = None
    report: InspectorReport | None = None
    report_species_id: int | None = None
    brain_changes_view: BrainChangesView | None = None
    neuro_integration_view: NeuroIntegrationView | None = None
    radar_texture: arcade.Texture | None = None
    radar_species_id: int | None = None
    radar_future: Future[object] | None = None
    radar_error: str | None = None
    horizontal_offset: float = 0.0
    vertical_offset: float = 0.0
    horizontal_limit: float = 0.0
    vertical_limit: float = 0.0
    horizontal_offset_min: float = 0.0
    horizontal_offset_max: float = 0.0
    vertical_offset_min: float = 0.0
    vertical_offset_max: float = 0.0
    scroll_drag: str | None = None
    scroll_drag_offset: float = 0.0
    canvas_drag: bool = False
    canvas_drag_started: bool = False
    canvas_drag_last: tuple[float, float] = (0.0, 0.0)
    inspector_width: float | None = None
    inspector_resize_drag: bool = False
    timeline_bucket_bounds: dict[int, arcade.Rect] = field(default_factory=dict)
    node_bounds: dict[int, arcade.Rect] = field(default_factory=dict)
    zoom: float = 1.0
    fit_mode: bool = True
    fit_requested: bool = True
    last_layout: SpeciesTreeLayout | None = None
    last_canvas: arcade.Rect | None = None
    extinction_times: dict[int, float] = field(default_factory=dict)
    cached_layout: SpeciesTreeLayout | None = None
    visible_slice: TreeViewportSlice = field(
        default_factory=lambda: TreeViewportSlice((), (), {})
    )
    focus_latest_pending: bool = False
    highlight_cache_id: int | None = None
    highlight_nodes: set[int] = field(default_factory=set)
    highlight_edges: set[tuple[int, int]] = field(default_factory=set)
    neat_label_signature: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    neat_labels: dict[int, str] = field(default_factory=dict)
    sync_signature: tuple[int, int, float, frozenset[int]] | None = None


@dataclass(frozen=True, slots=True)
class SpeciesInspectorRow:
    """Represent one formatted species-inspector row."""

    label: str | None
    value: str
    tone: str = "default"
    marker_color: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class SpeciesInspectorSection:
    """Group related species-inspector rows under a heading."""

    title: str
    rows: tuple[SpeciesInspectorRow, ...]
    kind: str = "rows"


@dataclass(frozen=True, slots=True)
class ConnectionChangeRowView:
    """Preformatted factual neural-connection transition for card renderers."""

    source_node_id: int
    target_node_id: int
    change_type: str
    badge_label: str
    endpoint_primary: str
    endpoint_technical: str
    endpoint_meta: str
    classification: str
    transition: str
    delta: str | None
    child_sign: str
    movement: str | None
    weights_complete: bool


@dataclass(frozen=True, slots=True)
class BrainChangeGroupView:
    """One source input and its deterministically ordered output changes."""

    source_node_id: int
    source_primary: str
    source_technical: str
    source_meta: str
    connection_count: int
    rows: tuple[ConnectionChangeRowView, ...]


@dataclass(frozen=True, slots=True)
class BrainChangesView:
    """Cached summary and source cards for direct parent brain changes."""

    parent_species_id: int | None
    total_count: int
    added_count: int
    changed_count: int
    removed_count: int
    groups: tuple[BrainChangeGroupView, ...]


@dataclass(frozen=True, slots=True)
class NeuroIntegrationHubView:
    """Cached incoming and outgoing connection rows for one hidden hub."""

    hub_id: int
    title: str
    technical: str
    incoming_count: int
    outgoing_count: int
    incoming_rows: tuple[ConnectionChangeRowView, ...]
    outgoing_rows: tuple[ConnectionChangeRowView, ...]


@dataclass(frozen=True, slots=True)
class NeuroIntegrationView:
    """Cached summary and hub cards for parent-relative hidden-node changes."""

    parent_species_id: int | None
    hub_count: int
    incoming_count: int
    outgoing_count: int
    hubs: tuple[NeuroIntegrationHubView, ...]


@dataclass(frozen=True, slots=True)
class SpeciesTreeLabel:
    """Represent a positioned label on the species-tree canvas."""

    species_id: int
    text: str
    bounds: arcade.Rect
    emphasized: bool
