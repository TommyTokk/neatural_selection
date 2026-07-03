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
    parent_nodes = dict(getattr(parent_genome, "nodes", {}) or {})
    child_nodes = dict(getattr(child_genome, "nodes", {}) or {})
    parent_connections = dict(
        getattr(parent_genome, "connections", {}) or {}
    )
    child_connections = dict(
        getattr(child_genome, "connections", {}) or {}
    )

    added_nodes = sorted(set(child_nodes) - set(parent_nodes), key=repr)
    removed_nodes = sorted(set(parent_nodes) - set(child_nodes), key=repr)
    added_connections = sorted(
        set(child_connections) - set(parent_connections),
        key=repr,
    )
    removed_connections = sorted(
        set(parent_connections) - set(child_connections),
        key=repr,
    )

    enabled: list[Any] = []
    disabled: list[Any] = []
    weight_changes: list[tuple[float, Any, float, float]] = []
    for key in sorted(
        set(parent_connections) & set(child_connections),
        key=repr,
    ):
        parent_gene = parent_connections[key]
        child_gene = child_connections[key]
        parent_enabled = bool(getattr(parent_gene, "enabled", True))
        child_enabled = bool(getattr(child_gene, "enabled", True))
        if not parent_enabled and child_enabled:
            enabled.append(key)
        elif parent_enabled and not child_enabled:
            disabled.append(key)

        parent_weight = _finite_float(getattr(parent_gene, "weight", None))
        child_weight = _finite_float(getattr(child_gene, "weight", None))
        if (
            parent_weight is not None
            and child_weight is not None
            and not isclose(parent_weight, child_weight)
        ):
            weight_changes.append(
                (
                    abs(child_weight - parent_weight),
                    key,
                    parent_weight,
                    child_weight,
                )
            )

    node_changes: list[tuple[float, Any, str, object, object]] = []
    node_parameter_count = 0
    for key in sorted(set(parent_nodes) & set(child_nodes), key=repr):
        parent_gene = parent_nodes[key]
        child_gene = child_nodes[key]
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
            node_changes.append((magnitude, key, attribute, before, after))

    highlights: list[tuple[int, float, str]] = []
    for key in added_nodes:
        highlights.append((0, 0.0, f"Node {key} added"))
    for key in removed_nodes:
        highlights.append((0, 0.0, f"Node {key} removed"))
    for key in added_connections:
        highlights.append((1, 0.0, f"Connection {_format_gene_key(key)} added"))
    for key in removed_connections:
        highlights.append((1, 0.0, f"Connection {_format_gene_key(key)} removed"))
    for key in enabled:
        highlights.append((2, 0.0, f"Connection {_format_gene_key(key)} enabled"))
    for key in disabled:
        highlights.append((2, 0.0, f"Connection {_format_gene_key(key)} disabled"))
    for magnitude, key, before, after in weight_changes:
        highlights.append(
            (
                3,
                -magnitude,
                f"Weight {_format_gene_key(key)} {before:+.3f} -> {after:+.3f}",
            )
        )
    for magnitude, key, attribute, before, after in node_changes:
        highlights.append(
            (
                4,
                -magnitude,
                f"Node {key} {attribute} {_format_change_value(before)}"
                f" -> {_format_change_value(after)}",
            )
        )
    highlights.sort(key=lambda item: (item[0], item[1], item[2]))

    return NeatChangeSummary(
        nodes_added=len(added_nodes),
        nodes_removed=len(removed_nodes),
        connections_added=len(added_connections),
        connections_removed=len(removed_connections),
        connections_enabled=len(enabled),
        connections_disabled=len(disabled),
        weights_changed=len(weight_changes),
        node_parameters_changed=node_parameter_count,
        key_changes=tuple(
            item[2] for item in highlights[: max(0, int(max_key_changes))]
        ),
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
