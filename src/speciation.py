from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

from src.creature import PhysicalTraits, VisionTraits


@dataclass(frozen=True, slots=True)
class SpeciesTraitSnapshot:
    radius: float
    vision_range: float
    vision_angle: float
    movement_cost_multiplier: float

    @classmethod
    def from_traits(
        cls,
        physical_traits: PhysicalTraits,
        vision: VisionTraits,
    ) -> SpeciesTraitSnapshot:
        return cls(
            radius=physical_traits.radius,
            vision_range=vision.range,
            vision_angle=vision.angle,
            movement_cost_multiplier=physical_traits.movement_cost_multiplier,
        )


@dataclass(frozen=True, slots=True)
class SpeciesDistanceBreakdown:
    neat_distance: float | None
    phenotypic_distance: float | None
    weighted_phenotypic_distance: float | None
    composite_distance: float | None
    compatibility_threshold: float | None
    phenotypic_weight: float | None
    radius_component: float | None
    vision_range_component: float | None
    vision_angle_component: float | None
    movement_cost_component: float | None


@dataclass(frozen=True, slots=True)
class NeatChangeSummary:
    nodes_added: int
    nodes_removed: int
    connections_added: int
    connections_removed: int
    connections_enabled: int
    connections_disabled: int
    weights_changed: int
    node_parameters_changed: int
    key_changes: tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> NeatChangeSummary:
        return cls(0, 0, 0, 0, 0, 0, 0, 0)


def summarize_neat_changes(
    parent_genome: Any,
    child_genome: Any,
    *,
    max_key_changes: int = 6,
) -> NeatChangeSummary:
    """Return a compact, deterministic diff between two NEAT genomes."""
    parent_nodes = getattr(parent_genome, "nodes", {}) or {}
    child_nodes = getattr(child_genome, "nodes", {}) or {}
    parent_connections = getattr(parent_genome, "connections", {}) or {}
    child_connections = getattr(child_genome, "connections", {}) or {}
    detail_limit = max(0, int(max_key_changes))

    node_structure: list[tuple[object, str]] = []
    connection_structure: list[tuple[object, str]] = []
    connection_states: list[tuple[object, str]] = []
    weight_details: list[tuple[object, str]] = []
    node_details: list[tuple[object, str]] = []

    nodes_added = 0
    for key in child_nodes:
        if key in parent_nodes:
            continue
        nodes_added += 1
        description = f"Node {key} added"
        _retain_bounded(
            node_structure,
            (description, description),
            detail_limit,
        )

    nodes_removed = 0
    for key in parent_nodes:
        if key in child_nodes:
            continue
        nodes_removed += 1
        description = f"Node {key} removed"
        _retain_bounded(
            node_structure,
            (description, description),
            detail_limit,
        )

    connections_added = 0
    for key in child_connections:
        if key in parent_connections:
            continue
        connections_added += 1
        description = f"Connection {_format_gene_key(key)} added"
        _retain_bounded(
            connection_structure,
            (description, description),
            detail_limit,
        )

    connections_removed = 0
    for key in parent_connections:
        if key in child_connections:
            continue
        connections_removed += 1
        description = f"Connection {_format_gene_key(key)} removed"
        _retain_bounded(
            connection_structure,
            (description, description),
            detail_limit,
        )

    connections_enabled = 0
    connections_disabled = 0
    weights_changed = 0
    for key, parent_gene in parent_connections.items():
        child_gene = child_connections.get(key)
        if child_gene is None:
            continue
        parent_enabled = bool(getattr(parent_gene, "enabled", True))
        child_enabled = bool(getattr(child_gene, "enabled", True))
        if not parent_enabled and child_enabled:
            connections_enabled += 1
            description = f"Connection {_format_gene_key(key)} enabled"
            _retain_bounded(
                connection_states,
                (description, description),
                detail_limit,
            )
        elif parent_enabled and not child_enabled:
            connections_disabled += 1
            description = f"Connection {_format_gene_key(key)} disabled"
            _retain_bounded(
                connection_states,
                (description, description),
                detail_limit,
            )

        parent_weight = _finite_float(getattr(parent_gene, "weight", None))
        child_weight = _finite_float(getattr(child_gene, "weight", None))
        if (
            parent_weight is not None
            and child_weight is not None
            and not isclose(parent_weight, child_weight)
        ):
            weights_changed += 1
            magnitude = abs(child_weight - parent_weight)
            description = (
                f"Weight {_format_gene_key(key)} "
                f"{parent_weight:+.3f} -> {child_weight:+.3f}"
            )
            _retain_bounded(
                weight_details,
                ((-magnitude, description), description),
                detail_limit,
            )

    node_parameter_count = 0
    for key, parent_gene in parent_nodes.items():
        child_gene = child_nodes.get(key)
        if child_gene is None:
            continue
        for attribute in ("bias", "response", "activation", "aggregation"):
            before = getattr(parent_gene, attribute, None)
            after = getattr(child_gene, attribute, None)
            if _values_equal(before, after):
                continue
            node_parameter_count += 1
            before_number = _finite_float(before)
            after_number = _finite_float(after)
            magnitude = (
                abs(after_number - before_number)
                if before_number is not None and after_number is not None
                else float("inf")
            )
            description = (
                f"Node {key} {attribute} {_format_change_value(before)}"
                f" -> {_format_change_value(after)}"
            )
            _retain_bounded(
                node_details,
                ((-magnitude, description), description),
                detail_limit,
            )

    key_changes = tuple(
        description
        for candidates in (
            node_structure,
            connection_structure,
            connection_states,
            weight_details,
            node_details,
        )
        for _, description in candidates
    )[:detail_limit]

    return NeatChangeSummary(
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        connections_added=connections_added,
        connections_removed=connections_removed,
        connections_enabled=connections_enabled,
        connections_disabled=connections_disabled,
        weights_changed=weights_changed,
        node_parameters_changed=node_parameter_count,
        key_changes=key_changes,
    )


@dataclass(frozen=True, slots=True)
class SpeciesRecord:
    species_id: int
    parent_species_id: int | None
    founder_creature_id: int | None
    founder_genome_id: int | None
    emerged_at: float | None
    founder_color: tuple[int, int, int] | None
    data_quality: str
    founder_traits: SpeciesTraitSnapshot | None
    trait_deltas: SpeciesTraitSnapshot | None
    distances: SpeciesDistanceBreakdown
    neat_changes: NeatChangeSummary | None = None


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _retain_bounded(
    candidates: list[tuple[object, str]],
    candidate: tuple[object, str],
    limit: int,
) -> None:
    if limit <= 0:
        return
    candidates.append(candidate)
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > limit:
        candidates.pop()


def _values_equal(first: object, second: object) -> bool:
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is not None and second_number is not None:
        return isclose(first_number, second_number)
    return first == second


def _format_gene_key(key: object) -> str:
    if isinstance(key, tuple) and len(key) == 2:
        return f"{key[0]}->{key[1]}"
    return str(key)


def _format_change_value(value: object) -> str:
    number = _finite_float(value)
    return f"{number:+.3f}" if number is not None else str(value)
