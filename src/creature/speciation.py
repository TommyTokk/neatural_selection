from __future__ import annotations

import copy
from dataclasses import dataclass
from math import isclose
from typing import Any, Literal

import neat

from configs.sim_config import TraitConfig, VisionConfig
from src.creature.genotype import FlockingTraits, PhysicalTraits, VisionTraits

NeuralShiftType = Literal["added", "changed", "removed"]
SpeciesRepresentative = tuple[Any, PhysicalTraits, VisionTraits, FlockingTraits]


@dataclass(frozen=True, slots=True)
class NeuralShift:
    """One enabled neural-connection transition relative to a parent genome."""

    source_node_id: int
    target_node_id: int
    change_type: NeuralShiftType
    parent_weight: float | None
    child_weight: float | None
    weight_delta: float | None = None

    def __post_init__(self) -> None:
        """Execute post init behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep post init behavior explicit in its owning subsystem.
        if (
            self.change_type == "changed"
            and self.parent_weight is not None
            and self.child_weight is not None
            and self.weight_delta is None
        ):
            object.__setattr__(
                self,
                "weight_delta",
                self.child_weight - self.parent_weight,
            )

    @property
    def weights_complete(self) -> bool:
        """Return whether the transition contains every expected endpoint.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
bool
    Result produced by this creature-domain operation."""
        # Keep weights complete behavior explicit in its owning subsystem.
        if self.change_type == "added":
            return self.parent_weight is None and self.child_weight is not None
        if self.change_type == "removed":
            return self.parent_weight is not None and self.child_weight is None
        return self.parent_weight is not None and self.child_weight is not None


def normalize_neural_shift(value: object) -> NeuralShift | None:
    """Normalize current records and legacy ``(target, source, type, delta)`` rows.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
NeuralShift | None
    Result produced by this creature-domain operation."""
    # Keep normalize neural shift behavior explicit in its owning subsystem.
    if isinstance(value, NeuralShift):
        return value
    if isinstance(value, dict):
        try:
            raw_type = str(value["change_type"])
            change_type = "changed" if raw_type == "weight" else raw_type
            if change_type not in {"added", "changed", "removed"}:
                return None
            parent = _finite_float(value.get("parent_weight"))
            child = _finite_float(value.get("child_weight"))
            delta = _finite_float(value.get("weight_delta"))
            return NeuralShift(
                source_node_id=int(value["source_node_id"]),
                target_node_id=int(value["target_node_id"]),
                change_type=change_type,  # type: ignore[arg-type]
                parent_weight=parent,
                child_weight=child,
                weight_delta=delta,
            )
        except (KeyError, TypeError, ValueError):
            return None
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        target, source, raw_type, raw_delta = value
        legacy_type = str(raw_type)
        delta = float(raw_delta)
        if legacy_type == "added":
            return NeuralShift(int(source), int(target), "added", None, delta)
        if legacy_type == "removed":
            return NeuralShift(int(source), int(target), "removed", -delta, None)
        if legacy_type == "weight":
            return NeuralShift(
                int(source),
                int(target),
                "changed",
                None,
                None,
                delta,
            )
    except (TypeError, ValueError):
        return None
    return None


def normalize_neural_shifts(values: object) -> tuple[NeuralShift, ...]:
    """Return all valid current or legacy neural transitions in source order.

Parameters
----------
values
    Input used by this creature-domain operation.
Returns
-------
tuple[NeuralShift, ...]
    Result produced by this creature-domain operation."""
    # Keep normalize neural shifts behavior explicit in its owning subsystem.
    try:
        candidates = tuple(values)  # type: ignore[arg-type]
    except TypeError:
        return ()
    normalized = tuple(normalize_neural_shift(value) for value in candidates)
    return tuple(value for value in normalized if value is not None)


@dataclass(frozen=True, slots=True)
class SpeciesTraitSnapshot:
    radius: float
    vision_range: float
    vision_angle: float
    movement_cost_multiplier: float
    separation_gene: float = 0.0
    alignment_gene: float = 0.0
    cohesion_gene: float = 0.0
    stomach_capacity: float = 0.0
    digestion_rate: float = 0.0
    digestion_efficiency: float = 0.0

    @classmethod
    def from_traits(
        cls,
        physical_traits: PhysicalTraits,
        vision: VisionTraits,
        flocking_traits: FlockingTraits | None = None,
    ) -> SpeciesTraitSnapshot:
        """Execute from traits behavior.

Parameters
----------
physical_traits
    Input used by this creature-domain operation.
vision
    Input used by this creature-domain operation.
flocking_traits
    Input used by this creature-domain operation.
Returns
-------
SpeciesTraitSnapshot
    Result produced by this creature-domain operation."""
        # Keep from traits behavior explicit in its owning subsystem.
        flocking = flocking_traits or FlockingTraits()
        return cls(
            radius=physical_traits.radius,
            vision_range=vision.range,
            vision_angle=vision.angle,
            movement_cost_multiplier=physical_traits.movement_cost_multiplier,
            separation_gene=flocking.separation_gene,
            alignment_gene=flocking.alignment_gene,
            cohesion_gene=flocking.cohesion_gene,
            stomach_capacity=getattr(
                physical_traits,
                "stomach_capacity",
                1.6,
            ),
            digestion_rate=getattr(
                physical_traits,
                "digestion_rate",
                0.2,
            ),
            digestion_efficiency=getattr(
                physical_traits,
                "digestion_efficiency",
                0.9,
            ),
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
    flocking_trait_distance: float | None = None
    weighted_flocking_trait_distance: float | None = None
    flocking_trait_distance_coefficient: float | None = None
    separation_gene_component: float | None = None
    alignment_gene_component: float | None = None
    cohesion_gene_component: float | None = None
    stomach_capacity_component: float | None = None
    digestion_rate_component: float | None = None
    digestion_efficiency_component: float | None = None
    digestive_trait_component: float | None = None


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
        """Execute empty behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
NeatChangeSummary
    Result produced by this creature-domain operation."""
        # Keep empty behavior explicit in its owning subsystem.
        return cls(0, 0, 0, 0, 0, 0, 0, 0)


def summarize_neat_changes(
    parent_genome: Any,
    child_genome: Any,
    *,
    max_key_changes: int = 6,
) -> NeatChangeSummary:
    """Return a compact, deterministic diff between two NEAT genomes.

Parameters
----------
parent_genome
    Input used by this creature-domain operation.
child_genome
    Input used by this creature-domain operation.
max_key_changes
    Input used by this creature-domain operation.
Returns
-------
NeatChangeSummary
    Result produced by this creature-domain operation."""
    # Keep summarize neat changes behavior explicit in its owning subsystem.
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
    emergence_food_ratio: float | None = None
    emergence_pop_ratio: float | None = None
    neural_shifts: tuple[NeuralShift, ...] = ()


def extract_neural_shifts(
    parent_genome: Any,
    child_genome: Any,
    *,
    weight_threshold: float = 0.5,
) -> tuple[NeuralShift, ...]:
    """Return deterministic material enabled-connection transitions.

Parameters
----------
parent_genome
    Input used by this creature-domain operation.
child_genome
    Input used by this creature-domain operation.
weight_threshold
    Input used by this creature-domain operation.
Returns
-------
tuple[NeuralShift, ...]
    Result produced by this creature-domain operation."""
    # Keep extract neural shifts behavior explicit in its owning subsystem.
    parent_connections = getattr(parent_genome, "connections", {}) or {}
    child_connections = getattr(child_genome, "connections", {}) or {}
    shifts: list[NeuralShift] = []

    for key in sorted(
        set(parent_connections) | set(child_connections),
        key=str,
    ):
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        try:
            source_node_id = int(key[0])
            target_node_id = int(key[1])
        except (TypeError, ValueError):
            continue

        parent_gene = parent_connections.get(key)
        child_gene = child_connections.get(key)
        parent_enabled = parent_gene is not None and bool(
            getattr(parent_gene, "enabled", True)
        )
        child_enabled = child_gene is not None and bool(
            getattr(child_gene, "enabled", True)
        )
        parent_weight = (
            _finite_float(getattr(parent_gene, "weight", None))
            if parent_gene is not None
            else None
        )
        child_weight = (
            _finite_float(getattr(child_gene, "weight", None))
            if child_gene is not None
            else None
        )

        if not parent_enabled and child_enabled:
            if child_weight is not None:
                shifts.append(
                    NeuralShift(
                        source_node_id,
                        target_node_id,
                        "added",
                        None,
                        child_weight,
                    )
                )
            continue
        if parent_enabled and not child_enabled:
            if parent_weight is not None:
                shifts.append(
                    NeuralShift(
                        source_node_id,
                        target_node_id,
                        "removed",
                        parent_weight,
                        None,
                    )
                )
            continue
        if (
            parent_enabled
            and child_enabled
            and parent_weight is not None
            and child_weight is not None
        ):
            delta = child_weight - parent_weight
            if abs(delta) > weight_threshold:
                shifts.append(
                    NeuralShift(
                        source_node_id,
                        target_node_id,
                        "changed",
                        parent_weight,
                        child_weight,
                        delta,
                    )
                )

    return tuple(shifts)


def _finite_float(value: object) -> float | None:
    """Execute finite float behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
float | None
    Result produced by this creature-domain operation."""
    # Keep finite float behavior explicit in its owning subsystem.
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
    """Execute retain bounded behavior.

Parameters
----------
candidates
    Input used by this creature-domain operation.
candidate
    Input used by this creature-domain operation.
limit
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
    # Keep retain bounded behavior explicit in its owning subsystem.
    if limit <= 0:
        return
    candidates.append(candidate)
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > limit:
        candidates.pop()


def _values_equal(first: object, second: object) -> bool:
    """Execute values equal behavior.

Parameters
----------
first
    Input used by this creature-domain operation.
second
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
    # Keep values equal behavior explicit in its owning subsystem.
    first_number = _finite_float(first)
    second_number = _finite_float(second)
    if first_number is not None and second_number is not None:
        return isclose(first_number, second_number)
    return first == second


def _format_gene_key(key: object) -> str:
    """Execute format gene key behavior.

Parameters
----------
key
    Input used by this creature-domain operation.
Returns
-------
str
    Result produced by this creature-domain operation."""
    # Keep format gene key behavior explicit in its owning subsystem.
    if isinstance(key, tuple) and len(key) == 2:
        return f"{key[0]}->{key[1]}"
    return str(key)


def _format_change_value(value: object) -> str:
    """Execute format change value behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
str
    Result produced by this creature-domain operation."""
    # Keep format change value behavior explicit in its owning subsystem.
    number = _finite_float(value)
    return f"{number:+.3f}" if number is not None else str(value)

def _normalized_trait_difference(
    first: float,
    second: float,
    minimum: float,
    maximum: float,
) -> float:
    """Execute normalized trait difference behavior.
    
    Parameters
    ----------
    first
        Input used by this creature-domain operation.
    second
        Input used by this creature-domain operation.
    minimum
        Input used by this creature-domain operation.
    maximum
        Input used by this creature-domain operation.
    Returns
    -------
    float
        Result produced by this creature-domain operation.
    
    Raises
    ------
    ValueError
        If an input or restored value violates validation rules.
    """
    # Keep normalized trait difference behavior explicit in its owning subsystem.
    if maximum <= minimum:
        raise ValueError("Phenotypic trait ranges must have a positive width.")
    clamped_first = max(minimum, min(maximum, first))
    clamped_second = max(minimum, min(maximum, second))
    return abs(clamped_first - clamped_second) / (maximum - minimum)


def calculate_phenotypic_distance(
    child_physical_traits: PhysicalTraits,
    child_vision: VisionTraits,
    representative_physical_traits: PhysicalTraits,
    representative_vision: VisionTraits,
    trait_config: TraitConfig,
    vision_config: VisionConfig,
) -> float:
    """Execute calculate phenotypic distance behavior.

Parameters
----------
child_physical_traits
    Input used by this creature-domain operation.
child_vision
    Input used by this creature-domain operation.
representative_physical_traits
    Input used by this creature-domain operation.
representative_vision
    Input used by this creature-domain operation.
trait_config
    Input used by this creature-domain operation.
vision_config
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
    # Keep calculate phenotypic distance behavior explicit in its owning subsystem.
    components = calculate_phenotypic_distance_components(
        child_physical_traits,
        child_vision,
        representative_physical_traits,
        representative_vision,
        trait_config,
        vision_config,
    )
    digestive_trait_component = (
        components.stomach_capacity
        + components.digestion_rate
        + components.digestion_efficiency
    ) / 3.0
    return (
        components.radius
        + components.vision_range
        + components.vision_angle
        + components.movement_cost_multiplier
        + digestive_trait_component
    )


def calculate_phenotypic_distance_components(
    child_physical_traits: PhysicalTraits,
    child_vision: VisionTraits,
    representative_physical_traits: PhysicalTraits,
    representative_vision: VisionTraits,
    trait_config: TraitConfig,
    vision_config: VisionConfig,
) -> SpeciesTraitSnapshot:
    """Execute calculate phenotypic distance components behavior.

Parameters
----------
child_physical_traits
    Input used by this creature-domain operation.
child_vision
    Input used by this creature-domain operation.
representative_physical_traits
    Input used by this creature-domain operation.
representative_vision
    Input used by this creature-domain operation.
trait_config
    Input used by this creature-domain operation.
vision_config
    Input used by this creature-domain operation.
Returns
-------
SpeciesTraitSnapshot
    Result produced by this creature-domain operation."""
    # Keep calculate phenotypic distance components behavior explicit in its owning subsystem.
    return SpeciesTraitSnapshot(
        radius=_normalized_trait_difference(
            child_physical_traits.radius,
            representative_physical_traits.radius,
            trait_config.min_radius,
            trait_config.max_radius,
        ),
        vision_range=_normalized_trait_difference(
            child_vision.range,
            representative_vision.range,
            vision_config.min_range,
            vision_config.max_range,
        ),
        vision_angle=_normalized_trait_difference(
            child_vision.angle,
            representative_vision.angle,
            vision_config.min_angle,
            vision_config.max_angle,
        ),
        movement_cost_multiplier=_normalized_trait_difference(
            child_physical_traits.movement_cost_multiplier,
            representative_physical_traits.movement_cost_multiplier,
            trait_config.min_movement_cost_multiplier,
            trait_config.max_movement_cost_multiplier,
        ),
        stomach_capacity=_normalized_trait_difference(
            getattr(
                child_physical_traits,
                "stomach_capacity",
                trait_config.default_stomach_capacity,
            ),
            getattr(
                representative_physical_traits,
                "stomach_capacity",
                trait_config.default_stomach_capacity,
            ),
            trait_config.min_stomach_capacity,
            trait_config.max_stomach_capacity,
        ),
        digestion_rate=_normalized_trait_difference(
            getattr(
                child_physical_traits,
                "digestion_rate",
                trait_config.default_digestion_rate,
            ),
            getattr(
                representative_physical_traits,
                "digestion_rate",
                trait_config.default_digestion_rate,
            ),
            trait_config.min_digestion_rate,
            trait_config.max_digestion_rate,
        ),
        digestion_efficiency=_normalized_trait_difference(
            getattr(
                child_physical_traits,
                "digestion_efficiency",
                trait_config.default_digestion_efficiency,
            ),
            getattr(
                representative_physical_traits,
                "digestion_efficiency",
                trait_config.default_digestion_efficiency,
            ),
            trait_config.min_digestion_efficiency,
            trait_config.max_digestion_efficiency,
        ),
    )


def calculate_flocking_trait_distance(
    first: FlockingTraits,
    second: FlockingTraits,
) -> tuple[float, float, float, float]:
    """Return mean and per-gene bounded flocking-trait distances.

Parameters
----------
first
    Input used by this creature-domain operation.
second
    Input used by this creature-domain operation.
Returns
-------
tuple[float, float, float, float]
    Result produced by this creature-domain operation."""
    # Keep calculate flocking trait distance behavior explicit in its owning subsystem.
    separation = abs(first.separation_gene - second.separation_gene)
    alignment = abs(first.alignment_gene - second.alignment_gene)
    cohesion = abs(first.cohesion_gene - second.cohesion_gene)
    return (separation + alignment + cohesion) / 3.0, separation, alignment, cohesion


@dataclass(frozen=True, slots=True)
class SpeciationResult:
    species_id: int
    parent_species_id: int
    is_new_species: bool
    founder_traits: SpeciesTraitSnapshot
    trait_deltas: SpeciesTraitSnapshot
    distances: SpeciesDistanceBreakdown
    neat_changes: NeatChangeSummary | None = None
    neural_shifts: tuple[NeuralShift, ...] = ()


@dataclass(frozen=True, slots=True)
class CompositeCompatibilityDistance:
    neat_distance: float
    phenotype_components: SpeciesTraitSnapshot
    phenotypic_distance: float
    weighted_phenotypic_distance: float
    flocking_trait_distance: float
    weighted_flocking_trait_distance: float
    separation_gene_component: float
    alignment_gene_component: float
    cohesion_gene_component: float
    composite_distance: float


class ContinuousSpeciesManager:
    def __init__(
        self,
        compatibility_threshold: float,
        phenotypic_weight: float = 2.0,
        trait_config: TraitConfig | None = None,
        vision_config: VisionConfig | None = None,
        flocking_trait_distance_coefficient: float = 1.0,
    ) -> None:
        """Execute init behavior.

Parameters
----------
compatibility_threshold
    Input used by this creature-domain operation.
phenotypic_weight
    Input used by this creature-domain operation.
trait_config
    Input used by this creature-domain operation.
vision_config
    Input used by this creature-domain operation.
flocking_trait_distance_coefficient
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.compatibility_threshold = compatibility_threshold
        self.phenotypic_weight = phenotypic_weight
        self.trait_config = trait_config or TraitConfig()
        self.vision_config = vision_config or VisionConfig()
        self.flocking_trait_distance_coefficient = max(
            0.0,
            float(flocking_trait_distance_coefficient),
        )
        self.representatives: dict[int, SpeciesRepresentative] = {}
        self.next_species_id = 2

    def register_initial_representative(
        self,
        genome: neat.DefaultGenome,
        physical_traits: PhysicalTraits,
        vision: VisionTraits,
        species_id: int = 1,
        flocking_traits: FlockingTraits | None = None,
    ) -> None:
        """Execute register initial representative behavior.

Parameters
----------
genome
    Input used by this creature-domain operation.
physical_traits
    Input used by this creature-domain operation.
vision
    Input used by this creature-domain operation.
species_id
    Input used by this creature-domain operation.
flocking_traits
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep register initial representative behavior explicit in its owning subsystem.
        self.representatives.setdefault(
            species_id,
            (
                genome,
                copy.deepcopy(physical_traits),
                copy.deepcopy(vision),
                copy.deepcopy(flocking_traits or FlockingTraits()),
            ),
        )

    def evaluate_species(
        self,
        child_genome: neat.DefaultGenome,
        child_physical_traits: PhysicalTraits,
        child_vision: VisionTraits,
        parent_species_id: int,
        genome_config: Any,
        child_flocking_traits: FlockingTraits | None = None,
    ) -> SpeciationResult:
        """Execute evaluate species behavior.

Parameters
----------
child_genome
    Input used by this creature-domain operation.
child_physical_traits
    Input used by this creature-domain operation.
child_vision
    Input used by this creature-domain operation.
parent_species_id
    Input used by this creature-domain operation.
genome_config
    Input used by this creature-domain operation.
child_flocking_traits
    Input used by this creature-domain operation.
Returns
-------
SpeciationResult
    Result produced by this creature-domain operation."""
        # Keep evaluate species behavior explicit in its owning subsystem.
        (
            representative_genome,
            representative_physical_traits,
            representative_vision,
            representative_flocking_traits,
        ) = self.representatives[parent_species_id]
        child_flocking_traits = child_flocking_traits or FlockingTraits()
        compatibility = self.composite_distance(
            child_genome,
            child_physical_traits,
            child_vision,
            child_flocking_traits,
            representative_genome,
            representative_physical_traits,
            representative_vision,
            representative_flocking_traits,
            genome_config,
        )
        trait_deltas = SpeciesTraitSnapshot(
            radius=(
                child_physical_traits.radius
                - representative_physical_traits.radius
            ),
            vision_range=child_vision.range - representative_vision.range,
            vision_angle=child_vision.angle - representative_vision.angle,
            movement_cost_multiplier=(
                child_physical_traits.movement_cost_multiplier
                - representative_physical_traits.movement_cost_multiplier
            ),
            separation_gene=(
                child_flocking_traits.separation_gene
                - representative_flocking_traits.separation_gene
            ),
            alignment_gene=(
                child_flocking_traits.alignment_gene
                - representative_flocking_traits.alignment_gene
            ),
            cohesion_gene=(
                child_flocking_traits.cohesion_gene
                - representative_flocking_traits.cohesion_gene
            ),
            stomach_capacity=(
                child_physical_traits.stomach_capacity
                - representative_physical_traits.stomach_capacity
            ),
            digestion_rate=(
                child_physical_traits.digestion_rate
                - representative_physical_traits.digestion_rate
            ),
            digestion_efficiency=(
                child_physical_traits.digestion_efficiency
                - representative_physical_traits.digestion_efficiency
            ),
        )
        digestive_trait_component = (
            compatibility.phenotype_components.stomach_capacity
            + compatibility.phenotype_components.digestion_rate
            + compatibility.phenotype_components.digestion_efficiency
        ) / 3.0
        distances = SpeciesDistanceBreakdown(
            neat_distance=compatibility.neat_distance,
            phenotypic_distance=compatibility.phenotypic_distance,
            weighted_phenotypic_distance=(
                compatibility.weighted_phenotypic_distance
            ),
            composite_distance=compatibility.composite_distance,
            compatibility_threshold=self.compatibility_threshold,
            phenotypic_weight=self.phenotypic_weight,
            radius_component=compatibility.phenotype_components.radius,
            vision_range_component=(
                compatibility.phenotype_components.vision_range
            ),
            vision_angle_component=(
                compatibility.phenotype_components.vision_angle
            ),
            movement_cost_component=(
                compatibility.phenotype_components.movement_cost_multiplier
            ),
            flocking_trait_distance=compatibility.flocking_trait_distance,
            weighted_flocking_trait_distance=(
                compatibility.weighted_flocking_trait_distance
            ),
            flocking_trait_distance_coefficient=(
                self.flocking_trait_distance_coefficient
            ),
            separation_gene_component=compatibility.separation_gene_component,
            alignment_gene_component=compatibility.alignment_gene_component,
            cohesion_gene_component=compatibility.cohesion_gene_component,
            stomach_capacity_component=(
                compatibility.phenotype_components.stomach_capacity
            ),
            digestion_rate_component=(
                compatibility.phenotype_components.digestion_rate
            ),
            digestion_efficiency_component=(
                compatibility.phenotype_components.digestion_efficiency
            ),
            digestive_trait_component=digestive_trait_component,
        )
        if compatibility.composite_distance > self.compatibility_threshold:
            neural_shifts = extract_neural_shifts(
                representative_genome,
                child_genome,
            )
            new_species_id = self.next_species_id
            self.representatives[new_species_id] = (
                child_genome,
                copy.deepcopy(child_physical_traits),
                copy.deepcopy(child_vision),
                copy.deepcopy(child_flocking_traits),
            )
            self.next_species_id += 1
            return SpeciationResult(
                species_id=new_species_id,
                parent_species_id=parent_species_id,
                is_new_species=True,
                founder_traits=SpeciesTraitSnapshot.from_traits(
                    child_physical_traits,
                    child_vision,
                    child_flocking_traits,
                ),
                trait_deltas=trait_deltas,
                distances=distances,
                neural_shifts=neural_shifts,
            )

        return SpeciationResult(
            species_id=parent_species_id,
            parent_species_id=parent_species_id,
            is_new_species=False,
            founder_traits=SpeciesTraitSnapshot.from_traits(
                child_physical_traits,
                child_vision,
                child_flocking_traits,
            ),
            trait_deltas=trait_deltas,
            distances=distances,
        )

    def composite_distance(
        self,
        first_genome: Any,
        first_physical_traits: PhysicalTraits,
        first_vision: VisionTraits,
        first_flocking_traits: FlockingTraits,
        second_genome: Any,
        second_physical_traits: PhysicalTraits,
        second_vision: VisionTraits,
        second_flocking_traits: FlockingTraits,
        genome_config: Any,
    ) -> CompositeCompatibilityDistance:
        """Return the same composite distance used by live and birth speciation.

Parameters
----------
first_genome
    Input used by this creature-domain operation.
first_physical_traits
    Input used by this creature-domain operation.
first_vision
    Input used by this creature-domain operation.
first_flocking_traits
    Input used by this creature-domain operation.
second_genome
    Input used by this creature-domain operation.
second_physical_traits
    Input used by this creature-domain operation.
second_vision
    Input used by this creature-domain operation.
second_flocking_traits
    Input used by this creature-domain operation.
genome_config
    Input used by this creature-domain operation.
Returns
-------
CompositeCompatibilityDistance
    Result produced by this creature-domain operation."""
        # Keep composite distance behavior explicit in its owning subsystem.
        neat_distance = first_genome.distance(second_genome, genome_config)
        phenotype_components = calculate_phenotypic_distance_components(
            first_physical_traits,
            first_vision,
            second_physical_traits,
            second_vision,
            self.trait_config,
            self.vision_config,
        )
        digestive_trait_component = (
            phenotype_components.stomach_capacity
            + phenotype_components.digestion_rate
            + phenotype_components.digestion_efficiency
        ) / 3.0
        phenotypic_distance = (
            phenotype_components.radius
            + phenotype_components.vision_range
            + phenotype_components.vision_angle
            + phenotype_components.movement_cost_multiplier
            + digestive_trait_component
        )
        weighted_phenotypic_distance = (
            self.phenotypic_weight * phenotypic_distance
        )
        (
            flocking_trait_distance,
            separation_gene_component,
            alignment_gene_component,
            cohesion_gene_component,
        ) = calculate_flocking_trait_distance(
            first_flocking_traits,
            second_flocking_traits,
        )
        weighted_flocking_trait_distance = (
            self.flocking_trait_distance_coefficient
            * flocking_trait_distance
        )
        composite_distance = (
            neat_distance
            + weighted_phenotypic_distance
            + weighted_flocking_trait_distance
        )
        return CompositeCompatibilityDistance(
            neat_distance=neat_distance,
            phenotype_components=phenotype_components,
            phenotypic_distance=phenotypic_distance,
            weighted_phenotypic_distance=weighted_phenotypic_distance,
            flocking_trait_distance=flocking_trait_distance,
            weighted_flocking_trait_distance=weighted_flocking_trait_distance,
            separation_gene_component=separation_gene_component,
            alignment_gene_component=alignment_gene_component,
            cohesion_gene_component=cohesion_gene_component,
            composite_distance=composite_distance,
        )
