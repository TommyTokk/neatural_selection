from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import exp, isfinite, pi
import sqlite3
from typing import Any, Iterable, Sequence

from PIL import Image

from configs.sim_config import SimConfig
from src.action import ACTION_OUTPUT_NAMES
from src.speciation import NeuralShift, SpeciesRecord, SpeciesTraitSnapshot
from src.vision import SENSOR_INPUT_NAMES


BEHAVIOR_RADAR_LABELS = (
    "Motility",
    "Voracity",
    "Sociability",
    "Nurturing",
    "Fecundity",
    "Vigilance",
)


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
class EthogramReflex:
    behavior: str
    sense: str
    description: str
    target_node_id: int
    source_node_id: int
    shift_type: str
    weight_delta: float


@dataclass(frozen=True, slots=True)
class NeuroIntegrationHub:
    hub_id: int
    sensory_integrations: tuple[str, ...]
    behavioral_modulations: tuple[str, ...]


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
    neuro_integration_hubs: tuple[NeuroIntegrationHub, ...]
    behavioral_ethogram: tuple[EthogramReflex, ...]
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
    "accelerate": "Kinetic / Locomotion Reflexes",
    "rotate": "Kinetic / Locomotion Reflexes",
    "want_reproduce": "Reproductive / Fecundity Reflexes",
    "want_eat": "Metabolic Ingestion Reflexes",
    "reset_chronometer": "Chronometer Reset",
    "want_grab": "Foraging / Object Manipulation Reflexes",
    "want_release": "Foraging / Object Manipulation Reflexes",
    "want_nurse": "Parental Care / Nurturing Reflexes",
    "flee_panic_intensity": "Threat Avoidance Reflexes",
    "weight_separation": "Cohort Spacing Reflexes",
    "weight_alignment": "Cohort Alignment Reflexes",
    "weight_cohesion": "Cohort Cohesion Reflexes",
    "emit_sound": "Acoustic Communication Reflexes",
    "sound_tone": "Acoustic Communication Reflexes",
    "emit_trail_pheromone": "Trail Pheromone Reflexes",
    "emit_alarm_pheromone": "Alarm Pheromone Reflexes",
}

_HIDDEN_ACTION = "Sensory Processing (Hidden)"

_SENSORY_DESCRIPTIONS = {
    "constant": "Endogenous Baseline Drive",
    "hungriness": "Nutritional Deficit (Hunger)",
    "maturity": "Developmental Maturity",
    "energy_percent": "Metabolic Reserve (Energy)",
    "speed": "Self-Velocity",
    "creature_count": "Visual Field: Local Crowd Density",
    "food_count": "Visual Foraging Field: Local Resource Density",
    "clock_tik_tok": "Biological Pacemaker (Alternating Drive)",
    "clock_chronometer": "Dynamic Interval Timer (Chronometer)",
    "clock_time_alive": "Absolute Longevity Sensation (Age)",
    "food_proximity": "Nearest Food Distance (Sight/Olfaction)",
    "food_angle": "Nearest Food Direction (Sight/Olfactory)",
    "creature_proximity": "Nearest Neighbor Distance (Social Sight)",
    "creature_angle": "Nearest Neighbor Direction (Social Sight)",
    "wall_proximity": "Boundary Proximity (Wall Tactile)",
    "wall_angle": "Boundary Direction (Wall Tactile)",
    "is_grabbing": "Load Carriage State (Carrying Object)",
    "biome_fertility_here": "Local Soil Quality (Fertility Here)",
    "biome_fertility_forward_left": "Left-Forward Soil Nutrient Gradient",
    "biome_fertility_forward_right": "Right-Forward Soil Nutrient Gradient",
    "biome_fertility_delta": "Temporal Nutrient Flux (Fertility Delta)",
    "own_infant_proximity": "Nearest Offspring Distance",
    "own_infant_angle": "Nearest Offspring Direction",
    "flock_center_proximity": "Cohort Cohesion Distance",
    "flock_center_angle": "Cohort Cohesion Direction",
    "flock_average_relative_heading": (
        "Cohort Alignment Delta (Average Heading)"
    ),
    "stomach_fullness": "Stomach Fullness (Satiety)",
    "sound_strength": "Acoustic Signal Strength",
    "sound_dir_sin": "Acoustic Relative Direction (Sine)",
    "sound_dir_cos": "Acoustic Relative Direction (Cosine)",
    "sound_tone": "Acoustic Signal Tone",
    "trail_pheromone_here": "Trail Pheromone Concentration (Here)",
    "trail_pheromone_forward_left": "Trail Pheromone Concentration (Forward Left)",
    "trail_pheromone_forward_right": "Trail Pheromone Concentration (Forward Right)",
    "alarm_pheromone_here": "Alarm Pheromone Concentration (Here)",
    "alarm_pheromone_forward_left": "Alarm Pheromone Concentration (Forward Left)",
    "alarm_pheromone_forward_right": "Alarm Pheromone Concentration (Forward Right)",
}

if set(_SENSORY_DESCRIPTIONS) != set(SENSOR_INPUT_NAMES):
    raise RuntimeError("Sensory descriptions must match SensorSnapshot.as_inputs().")

_SENSORY_LEXICON = tuple(
    _SENSORY_DESCRIPTIONS[name] for name in SENSOR_INPUT_NAMES
)


def calculate_behavior_scores(
    genome: Any,
    output_keys: Iterable[int],
) -> tuple[float, ...]:
    """Return six normalized behavioral tendencies for a NEAT genome."""
    action_keys = {
        name: int(key)
        for key, name in zip(output_keys, ACTION_OUTPUT_NAMES)
    }
    nodes = getattr(genome, "nodes", {}) or {}
    connections = getattr(genome, "connections", {}) or {}

    incoming_weights: dict[int, float] = {}
    for connection_key, connection in connections.items():
        if not bool(getattr(connection, "enabled", True)):
            continue
        key = getattr(connection, "key", connection_key)
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        try:
            target = int(key[1])
            weight = float(getattr(connection, "weight", 0.0))
        except (TypeError, ValueError):
            continue
        if isfinite(weight):
            incoming_weights[target] = incoming_weights.get(target, 0.0) + weight

    def output_score(action: str) -> float:
        key = action_keys.get(action)
        if key is None or key not in nodes:
            return 0.5
        try:
            bias = float(getattr(nodes[key], "bias", 0.0))
        except (TypeError, ValueError):
            return 0.5
        drive = bias + incoming_weights.get(key, 0.0)
        if drive != drive:
            return 0.5
        return _stable_sigmoid(drive)

    motility = output_score("accelerate")
    voracity = (
        output_score("want_eat") + output_score("want_grab")
    ) / 2.0
    sociability = (
        output_score("weight_alignment")
        + output_score("weight_cohesion")
        + 1.0
        - output_score("weight_separation")
    ) / 3.0
    nurturing = output_score("want_nurse")
    fecundity = output_score("want_reproduce")
    vigilance = output_score("flee_panic_intensity")
    return (
        motility,
        voracity,
        sociability,
        nurturing,
        fecundity,
        vigilance,
    )


def generate_radar_chart_image(
    child_scores: Sequence[float],
    parent_scores: Sequence[float] | None,
    labels: Sequence[str],
) -> Image.Image:
    """Render a transparent radar chart and return a detached RGBA image."""
    if len(child_scores) != len(labels):
        raise ValueError("child_scores and labels must have the same length")
    if len(labels) < 3:
        raise ValueError("a radar chart requires at least three axes")
    if parent_scores is not None and len(parent_scores) != len(labels):
        raise ValueError("parent_scores and labels must have the same length")

    # These imports are deliberately lazy: normal simulation frames do not pay
    # Matplotlib's import and initialization cost.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", palette="muted")
    figure = None
    try:
        figure, axis = plt.subplots(
            figsize=(4, 4),
            subplot_kw={"polar": True},
            facecolor="none",
        )
        figure.patch.set_alpha(0.0)
        axis.set_facecolor("none")

        count = len(labels)
        angles = [index * 2.0 * pi / count for index in range(count)]
        closed_angles = [*angles, angles[0]]
        child_values = [_bounded_score(value) for value in child_scores]
        closed_child = [*child_values, child_values[0]]

        if parent_scores is not None:
            parent_values = [_bounded_score(value) for value in parent_scores]
            closed_parent = [*parent_values, parent_values[0]]
            axis.plot(
                closed_angles,
                closed_parent,
                color="#374151",
                linewidth=2.0,
                linestyle="--",
                label="Parent",
            )
            axis.fill(
                closed_angles,
                closed_parent,
                color="#6b7280",
                alpha=0.10,
            )

        child_fill = "#4c78a8"
        child_border = "#173f5f"
        axis.plot(
            closed_angles,
            closed_child,
            color=child_border,
            linewidth=2.6,
            label="Selected species",
        )
        axis.fill(closed_angles, closed_child, color=child_fill, alpha=0.30)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(angles, labels=labels)
        axis.set_yticks((0.25, 0.5, 0.75, 1.0))
        axis.set_yticklabels(("0.25", "0.50", "0.75", "1.0"))
        axis.set_rlabel_position(90)
        axis.tick_params(axis="x", colors="#161a32", labelsize=9, pad=8)
        axis.tick_params(axis="y", colors="#42474d", labelsize=7)
        axis.grid(color="#4b5563", alpha=0.48, linewidth=0.9)
        axis.spines["polar"].set_color("#374151")
        axis.spines["polar"].set_linewidth(1.4)
        axis.spines["polar"].set_alpha(0.85)
        figure.subplots_adjust(left=0.17, right=0.83, bottom=0.17, top=0.83)

        with BytesIO() as buffer:
            figure.savefig(
                buffer,
                format="png",
                transparent=True,
                dpi=110,
            )
            buffer.seek(0)
            with Image.open(buffer) as image:
                return image.convert("RGBA").copy()
    finally:
        if figure is not None:
            plt.close(figure)


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    exponential = exp(value)
    return exponential / (1.0 + exponential)


def _bounded_score(value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not isfinite(numeric):
        return 0.5
    return max(0.0, min(1.0, numeric))


def generate_inspector_report(
    species_record: SpeciesRecord,
    parent_record: SpeciesRecord | None,
    db_connection: sqlite3.Connection | None,
    sim_config: SimConfig,
    output_keys: Iterable[int],
    input_keys: Iterable[int] | None = None,
) -> InspectorReport:
    """Build a semantic report on demand for one selected species."""
    morphology = profile_morphology(species_record, parent_record)
    metabolism = profile_metabolism(species_record, parent_record, sim_config)
    behavioral_shifts = profile_cognition(
        species_record.neural_shifts,
        output_keys,
    )
    neuro_integration_hubs, behavioral_ethogram = profile_neuroethology(
        species_record.neural_shifts,
        output_keys,
        input_keys,
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
        neuro_integration_hubs=neuro_integration_hubs,
        behavioral_ethogram=behavioral_ethogram,
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


def profile_neuroethology(
    neural_shifts: Iterable[NeuralShift],
    output_keys: Iterable[int],
    input_keys: Iterable[int] | None = None,
) -> tuple[tuple[NeuroIntegrationHub, ...], tuple[EthogramReflex, ...]]:
    output_key_sequence = tuple(int(key) for key in output_keys)
    input_key_sequence = _ordered_input_keys(input_keys)
    action_by_target = _action_labels_by_output_key(output_key_sequence)
    sense_by_source = _sense_labels_by_input_key(input_key_sequence)
    input_rank = {key: index for index, key in enumerate(input_key_sequence)}
    output_rank = {key: index for index, key in enumerate(output_key_sequence)}
    integrations: dict[int, list[tuple[int, str]]] = {}
    modulations: dict[int, list[tuple[int, str]]] = {}
    reflexes: list[EthogramReflex] = []

    for target, source, shift_type, delta in neural_shifts:
        target_id = int(target)
        source_id = int(source)
        normalized_type = str(shift_type)
        weight_delta = float(delta)
        sense = sense_by_source.get(source_id)
        behavior = action_by_target.get(target_id)

        if sense is not None and behavior is not None:
            description = _reflex_description(
                sense,
                behavior,
                normalized_type,
                weight_delta,
            )
            if description is not None:
                reflexes.append(
                    EthogramReflex(
                        behavior=behavior,
                        sense=sense,
                        description=description,
                        target_node_id=target_id,
                        source_node_id=source_id,
                        shift_type=normalized_type,
                        weight_delta=weight_delta,
                    )
                )
            continue

        if sense is not None and target_id >= 0:
            integrations.setdefault(target_id, []).append(
                (
                    input_rank[source_id],
                    f"Integration Hub {target_id} is now integrating "
                    f"[{sense}] into its internal state "
                    f"(Delta: {weight_delta:+.2f})",
                )
            )
            continue

        if source_id >= 0 and behavior is not None:
            modulations.setdefault(source_id, []).append(
                (
                    output_rank[target_id],
                    f"{behavior} is now modulated by abstract concepts from "
                    f"[Integration Hub {source_id}] "
                    f"(Delta: {weight_delta:+.2f})",
                )
            )

    hub_ids = sorted(set(integrations) | set(modulations))
    hubs = tuple(
        NeuroIntegrationHub(
            hub_id=hub_id,
            sensory_integrations=tuple(
                description
                for _, description in sorted(integrations.get(hub_id, ()))
            ),
            behavioral_modulations=tuple(
                description
                for _, description in sorted(modulations.get(hub_id, ()))
            ),
        )
        for hub_id in hub_ids
    )
    reflexes.sort(
        key=lambda reflex: (
            input_rank.get(reflex.source_node_id, len(input_rank)),
            output_rank.get(reflex.target_node_id, len(output_rank)),
        )
    )
    return hubs, tuple(reflexes)


def profile_cognition(
    neural_shifts: Iterable[NeuralShift],
    output_keys: Iterable[int],
) -> tuple[BehavioralShiftGroup, ...]:
    action_by_target = _action_labels_by_output_key(output_keys)
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


def _action_labels_by_output_key(
    output_keys: Iterable[int],
) -> dict[int, str]:
    return {
        int(key): _ACTION_LABELS.get(name, name.replace("_", " ").title())
        for key, name in zip(output_keys, ACTION_OUTPUT_NAMES)
    }


def _sense_labels_by_input_key(
    input_keys: Iterable[int] | None,
) -> dict[int, str]:
    keys = _ordered_input_keys(input_keys)
    return {
        key: _SENSORY_LEXICON[index]
        for index, key in enumerate(keys[: len(_SENSORY_LEXICON)])
    }


def _ordered_input_keys(input_keys: Iterable[int] | None) -> tuple[int, ...]:
    if input_keys is None:
        return tuple(-(index + 1) for index in range(len(SENSOR_INPUT_NAMES)))
    return tuple(int(key) for key in input_keys)


def _reflex_description(
    sense: str,
    behavior: str,
    shift_type: str,
    weight_delta: float,
) -> str | None:
    if shift_type == "removed":
        return (
            f"⚪ Lost the instinct to trigger [{behavior}] "
            f"in response to [{sense}]"
        )
    if weight_delta > 0.0:
        return (
            f"🟢 [{sense}] now actively triggers/sensitizes "
            f"[{behavior}]"
        )
    if weight_delta < 0.0:
        return (
            f"🔴 [{sense}] now actively suppresses/brakes "
            f"[{behavior}]"
        )
    return None


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
