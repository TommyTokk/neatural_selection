from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Iterable

from configs.sim_config import SimConfig
from src.action import ACTION_OUTPUT_NAMES
from src.speciation import NeuralShift, SpeciesRecord, SpeciesTraitSnapshot


@dataclass(frozen=True, slots=True)
class MorphologyInsight:
    trait: str
    percent_change: float
    description: str


@dataclass(frozen=True, slots=True)
class MetabolicProfile:
    parent_idle_cost: float | None
    child_idle_cost: float | None
    idle_percent_change: float | None
    parent_active_cost: float | None
    child_active_cost: float | None
    active_percent_change: float | None


@dataclass(frozen=True, slots=True)
class CognitiveShift:
    source_node_id: int
    shift_type: str
    weight_delta: float


@dataclass(frozen=True, slots=True)
class BehavioralShiftGroup:
    action: str
    excitatory: tuple[CognitiveShift, ...]
    inhibitory: tuple[CognitiveShift, ...]


@dataclass(frozen=True, slots=True)
class LegacyProfile:
    descendant_count: int | None
    average_lifespan: float | None


@dataclass(frozen=True, slots=True)
class InspectorReport:
    species_id: int
    parent_species_id: int | None
    species_traits: SpeciesTraitSnapshot | None
    morphology: tuple[MorphologyInsight, ...]
    metabolism: MetabolicProfile
    behavioral_shifts: tuple[BehavioralShiftGroup, ...]
    food_scarcity: float | None
    population_density: float | None
    legacy: LegacyProfile


_TRAIT_LABELS = {
    "radius": ("Larger Body", "Smaller Body"),
    "vision_range": ("Longer Vision", "Shorter Vision"),
    "vision_angle": ("Wider Vision", "Narrower Vision"),
    "movement_cost_multiplier": (
        "Higher Movement Cost",
        "Lower Movement Cost",
    ),
}

_ACTION_LABELS = {
    "accelerate": "Movement (Accelerate)",
    "rotate": "Movement (Turn)",
    "want_reproduce": "Reproduction",
    "want_eat": "Eating",
    "reset_chronometer": "Chronometer Reset",
    "want_grab": "Grabbing",
    "want_release": "Releasing",
    "want_nurse": "Nursing",
    "flee_panic_intensity": "Fleeing",
    "weight_separation": "Flocking (Separation)",
    "weight_alignment": "Flocking (Alignment)",
    "weight_cohesion": "Flocking (Cohesion)",
}

_HIDDEN_ACTION = "Sensory Processing (Hidden)"


def generate_inspector_report(
    species_record: SpeciesRecord,
    parent_record: SpeciesRecord | None,
    db_connection: sqlite3.Connection | None,
    sim_config: SimConfig,
    output_keys: Iterable[int],
) -> InspectorReport:
    """Build a semantic report on demand for one selected species."""
    morphology = profile_morphology(species_record, parent_record)
    metabolism = profile_metabolism(species_record, parent_record, sim_config)
    behavioral_shifts = profile_cognition(
        species_record.neural_shifts,
        output_keys,
    )
    legacy = query_species_legacy(db_connection, species_record.species_id)
    food_ratio = _bounded_ratio(species_record.emergence_food_ratio)
    return InspectorReport(
        species_id=species_record.species_id,
        parent_species_id=species_record.parent_species_id,
        species_traits=species_record.founder_traits,
        morphology=morphology,
        metabolism=metabolism,
        behavioral_shifts=behavioral_shifts,
        food_scarcity=None if food_ratio is None else 1.0 - food_ratio,
        population_density=_bounded_ratio(
            species_record.emergence_pop_ratio
        ),
        legacy=legacy,
    )


def profile_morphology(
    species_record: SpeciesRecord,
    parent_record: SpeciesRecord | None,
) -> tuple[MorphologyInsight, ...]:
    child = species_record.founder_traits
    parent = None if parent_record is None else parent_record.founder_traits
    if child is None or parent is None:
        return ()

    insights: list[MorphologyInsight] = []
    for trait, descriptions in _TRAIT_LABELS.items():
        child_value = float(getattr(child, trait))
        parent_value = float(getattr(parent, trait))
        if abs(parent_value) <= 1e-12:
            continue
        percent_change = (child_value - parent_value) / abs(parent_value) * 100.0
        if abs(percent_change) <= 1e-9:
            continue
        insights.append(
            MorphologyInsight(
                trait=trait,
                percent_change=percent_change,
                description=descriptions[0 if percent_change > 0.0 else 1],
            )
        )
    return tuple(insights)


def profile_metabolism(
    species_record: SpeciesRecord,
    parent_record: SpeciesRecord | None,
    sim_config: SimConfig,
) -> MetabolicProfile:
    child = species_record.founder_traits
    parent = None if parent_record is None else parent_record.founder_traits
    if child is None:
        return MetabolicProfile(None, None, None, None, None, None)

    child_idle, child_active = _metabolic_costs(child, sim_config)
    if parent is None:
        return MetabolicProfile(
            parent_idle_cost=None,
            child_idle_cost=child_idle,
            idle_percent_change=None,
            parent_active_cost=None,
            child_active_cost=child_active,
            active_percent_change=None,
        )

    parent_idle, parent_active = _metabolic_costs(parent, sim_config)
    return MetabolicProfile(
        parent_idle_cost=parent_idle,
        child_idle_cost=child_idle,
        idle_percent_change=_percent_difference(child_idle, parent_idle),
        parent_active_cost=parent_active,
        child_active_cost=child_active,
        active_percent_change=_percent_difference(child_active, parent_active),
    )


def profile_cognition(
    neural_shifts: Iterable[NeuralShift],
    output_keys: Iterable[int],
) -> tuple[BehavioralShiftGroup, ...]:
    action_by_target = {
        int(key): _ACTION_LABELS.get(name, name.replace("_", " ").title())
        for key, name in zip(output_keys, ACTION_OUTPUT_NAMES)
    }
    grouped: dict[str, tuple[list[CognitiveShift], list[CognitiveShift]]] = {}
    for target, source, shift_type, delta in neural_shifts:
        action = action_by_target.get(int(target), _HIDDEN_ACTION)
        excitatory, inhibitory = grouped.setdefault(action, ([], []))
        shift = CognitiveShift(
            source_node_id=int(source),
            shift_type=str(shift_type),
            weight_delta=float(delta),
        )
        (excitatory if delta > 0.0 else inhibitory).append(shift)

    return tuple(
        BehavioralShiftGroup(
            action=action,
            excitatory=tuple(values[0]),
            inhibitory=tuple(values[1]),
        )
        for action, values in grouped.items()
    )


def query_species_legacy(
    db_connection: sqlite3.Connection | None,
    species_id: int,
) -> LegacyProfile:
    if db_connection is None:
        return LegacyProfile(None, None)
    try:
        row = db_connection.execute(
            """
            WITH RECURSIVE descendants(species_id) AS (
                SELECT species_id
                FROM species
                WHERE parent_species_id = ?
                UNION
                SELECT species.species_id
                FROM species
                JOIN descendants
                  ON species.parent_species_id = descendants.species_id
            )
            SELECT
                (SELECT COUNT(*) FROM descendants),
                (
                    SELECT AVG(death_time - birth_time)
                    FROM creatures
                    WHERE species_id = ? AND death_time IS NOT NULL
                )
            """,
            (int(species_id), int(species_id)),
        ).fetchone()
    except (sqlite3.Error, AttributeError):
        return LegacyProfile(None, None)
    if row is None:
        return LegacyProfile(0, None)
    return LegacyProfile(
        descendant_count=int(row[0]),
        average_lifespan=None if row[1] is None else float(row[1]),
    )


def _metabolic_costs(
    traits: SpeciesTraitSnapshot,
    sim_config: SimConfig,
) -> tuple[float, float]:
    metabolism = sim_config.metabolism
    trait = sim_config.trait
    vision = sim_config.vision
    max_radius = max(float(trait.max_radius), 0.0001)
    max_range = max(float(vision.max_range), 0.0001)
    max_angle = max(float(vision.max_angle), 0.0001)
    vision_area_ratio = (
        float(traits.vision_angle)
        / max_angle
        * (float(traits.vision_range) / max_range) ** 2
    )
    vision_cost = (
        float(vision.base_energy_cost)
        + float(vision.area_energy_cost_factor) * vision_area_ratio
    )
    body_cost = (
        float(trait.body_metabolism_cost_factor)
        * (max(0.0, float(traits.radius)) / max_radius) ** 2
    )
    idle = float(metabolism.basic_metabolism_rate) + vision_cost + body_cost
    movement = (
        float(metabolism.movement_energy_cost_factor)
        * max(0.0, float(traits.movement_cost_multiplier))
    )
    return idle, idle + movement


def _percent_difference(value: float, baseline: float) -> float | None:
    if abs(baseline) <= 1e-12:
        return None
    return (value - baseline) / abs(baseline) * 100.0


def _bounded_ratio(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or abs(parsed) == float("inf"):
        return None
    return max(0.0, min(1.0, parsed))
