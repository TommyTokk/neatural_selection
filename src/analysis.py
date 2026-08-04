from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import exp, isfinite, pi
import sqlite3
from typing import Any, Iterable, Sequence

from PIL import Image

from configs.sim_config import SimConfig
from src.action import ACTION_OUTPUT_NAMES
from src.speciation import (
    NeuralShift,
    SpeciesRecord,
    SpeciesTraitSnapshot,
    normalize_neural_shifts,
)
from src.vision import SENSOR_INPUT_NAMES


BEHAVIOR_RADAR_LABELS = (
    "Motility",
    "Voracity",
    "Sociability",
    "Nurturing",
    "Fecundity",
    "Vigilance",
)
RADAR_AXIS_LABEL_SIZE = 12.5
RADAR_VALUE_LABEL_SIZE = 9.0


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
class NodeSemanticLabel:
    primary: str
    technical: str


@dataclass(frozen=True, slots=True)
class ConnectionClassification:
    key: str
    label: str
    child_sign: str
    movement: str | None


@dataclass(frozen=True, slots=True)
class NeuroIntegrationHub:
    hub_id: int
    incoming_sensor_changes: tuple[NeuralShift, ...]
    outgoing_action_changes: tuple[NeuralShift, ...]


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
    neuro_integration_hubs: tuple[NeuroIntegrationHub, ...]
    direct_brain_changes: tuple[NeuralShift, ...]
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
    "stomach_capacity": ("Larger Stomach", "Smaller Stomach"),
    "digestion_rate": ("Faster Digestion", "Slower Digestion"),
    "digestion_efficiency": (
        "More Efficient Digestion",
        "Less Efficient Digestion",
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
    "herding": "Same-Species Herding Reflexes",
    "emit_sound": "Acoustic Communication Reflexes",
    "sound_tone": "Acoustic Communication Reflexes",
    "emit_trail_pheromone": "Trail Pheromone Reflexes",
    "emit_alarm_pheromone": "Alarm Pheromone Reflexes",
    "rest": "Rest / Recovery Reflexes",
}

_ACTION_PRIMARY_LABELS = {
    "accelerate": "Accelerate",
    "rotate": "Turn",
    "want_reproduce": "Reproduce",
    "want_eat": "Eat / consume",
    "reset_chronometer": "Reset internal timer",
    "want_grab": "Pick up food",
    "want_release": "Release carried food",
    "want_nurse": "Care for offspring",
    "flee_panic_intensity": "Panic intensity",
    "herding": "Herding",
    "emit_sound": "Communicate acoustically",
    "sound_tone": "Acoustic tone",
    "emit_trail_pheromone": "Emit trail pheromone",
    "emit_alarm_pheromone": "Emit alarm pheromone",
    "rest": "Rest / recover",
}

_SENSORY_PRIMARY_LABELS = {
    "constant": "Baseline drive",
    "feeding_drive": "Hunger / feeding drive",
    "reproductive_readiness": "Reproductive readiness",
    "energy_percent": "Energy reserve",
    "speed": "Current speed",
    "creature_count": "Nearby creature density",
    "food_count": "Nearby food density",
    "clock_tik_tok": "Alternating body clock",
    "clock_chronometer": "Internal timer",
    "clock_time_alive": "Age",
    "food_proximity": "Nearest food distance",
    "food_angle": "Nearest food direction",
    "creature_proximity": "Nearest creature distance",
    "creature_angle": "Nearest creature direction",
    "wall_proximity": "Wall distance",
    "wall_angle": "Wall direction",
    "is_grabbing": "Carrying something",
    "biome_fertility_here": "Local fertility",
    "biome_fertility_left_gradient": "Fertility to the left",
    "biome_fertility_right_gradient": "Fertility to the right",
    "biome_fertility_trend": "Fertility trend",
    "own_infant_proximity": "Nearest offspring distance",
    "own_infant_angle": "Nearest offspring direction",
    "flock_center_proximity": "Flock-centre distance",
    "flock_center_angle": "Flock-centre direction",
    "flock_average_relative_heading": "Flock heading difference",
    "flockmate_count": "Compatible flockmate count",
    "flock_presence": "Compatible flock presence",
    "flock_effective_count": "Effective flockmate count",
    "flock_center_forward": "Flock centre ahead / behind",
    "flock_center_right": "Flock centre left / right",
    "flock_relative_velocity_forward": "Flock forward velocity difference",
    "flock_relative_velocity_right": "Flock sideways velocity difference",
    "long_range_social_intensity": "Long-range social signal strength",
    "long_range_social_direction_forward": "Long-range social direction ahead",
    "long_range_social_direction_right": "Long-range social direction right",
    "stomach_fullness": "Stomach fullness",
    "sound_strength": "Sound strength",
    "sound_dir_sin": "Sound direction (sine)",
    "sound_dir_cos": "Sound direction (cosine)",
    "sound_tone": "Sound tone",
    "trail_pheromone_here": "Trail pheromone here",
    "trail_pheromone_forward_left": "Trail pheromone ahead-left",
    "trail_pheromone_forward_right": "Trail pheromone ahead-right",
    "alarm_pheromone_here": "Alarm pheromone here",
    "alarm_pheromone_forward_left": "Alarm pheromone ahead-left",
    "alarm_pheromone_forward_right": "Alarm pheromone ahead-right",
    "life_normalized": "Remaining life reserve",
}

_SENSORY_DESCRIPTIONS = {
    "constant": "Endogenous Baseline Drive",
    "feeding_drive": "Feeding Drive (Satiety-Modulated)",
    "reproductive_readiness": "Reproductive Readiness",
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
    "biome_fertility_left_gradient": "Left-Forward Fertility Advantage",
    "biome_fertility_right_gradient": "Right-Forward Fertility Advantage",
    "biome_fertility_trend": "Temporal Fertility Trend",
    "own_infant_proximity": "Nearest Offspring Distance",
    "own_infant_angle": "Nearest Offspring Direction",
    "flock_center_proximity": "Cohort Cohesion Distance",
    "flock_center_angle": "Cohort Cohesion Direction",
    "flock_average_relative_heading": (
        "Cohort Alignment Delta (Average Heading)"
    ),
    "flockmate_count": "Effective Compatible Flockmate Count",
    "flock_presence": "Compatible Flock Presence",
    "flock_effective_count": "Target-Scaled Compatible Flockmate Count",
    "flock_center_forward": "Flock Centre Forward Offset",
    "flock_center_right": "Flock Centre Right Offset",
    "flock_relative_velocity_forward": "Flock Relative Forward Velocity",
    "flock_relative_velocity_right": "Flock Relative Right Velocity",
    "long_range_social_intensity": "Long-Range Social Intensity",
    "long_range_social_direction_forward": "Long-Range Social Forward Direction",
    "long_range_social_direction_right": "Long-Range Social Right Direction",
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
    "life_normalized": "Remaining Life Reserve",
}

if not set(SENSOR_INPUT_NAMES).issubset(_SENSORY_DESCRIPTIONS):
    raise RuntimeError("Sensory descriptions must match SensorSnapshot.as_inputs().")
if not set(SENSOR_INPUT_NAMES).issubset(_SENSORY_PRIMARY_LABELS):
    raise RuntimeError("Plain sensory labels must match SensorSnapshot.as_inputs().")
if not set(ACTION_OUTPUT_NAMES).issubset(_ACTION_PRIMARY_LABELS):
    raise RuntimeError("Plain action labels must match the action output contract.")


def sensory_node_label(name: str) -> NodeSemanticLabel:
    """Return centralized plain and technical labels for one sensor name."""
    fallback = name.replace("_", " ").title()
    return NodeSemanticLabel(
        _SENSORY_PRIMARY_LABELS.get(name, fallback),
        _SENSORY_DESCRIPTIONS.get(name, fallback),
    )


def action_node_label(name: str) -> NodeSemanticLabel:
    """Return centralized plain and technical labels for one action name."""
    fallback = name.replace("_", " ").title()
    return NodeSemanticLabel(
        _ACTION_PRIMARY_LABELS.get(name, fallback),
        _ACTION_LABELS.get(name, fallback),
    )


def classify_connection_transition(
    parent_weight: float | None,
    child_weight: float | None,
) -> ConnectionClassification:
    """Classify one factual weight transition without claiming behavior causation."""
    if parent_weight is None and child_weight is None:
        return ConnectionClassification(
            "unavailable",
            "Historical weights unavailable",
            "Unknown",
            None,
        )
    if parent_weight is None:
        assert child_weight is not None
        if child_weight > 0.0:
            return ConnectionClassification(
                "positive_added", "Positive influence added", "Positive", None
            )
        if child_weight < 0.0:
            return ConnectionClassification(
                "negative_added", "Negative influence added", "Negative", None
            )
        return ConnectionClassification(
            "neutral_added", "Zero-weight connection added", "Neutral", None
        )
    if child_weight is None:
        if parent_weight > 0.0:
            return ConnectionClassification(
                "positive_removed", "Positive influence removed", "No connection", None
            )
        if parent_weight < 0.0:
            return ConnectionClassification(
                "negative_removed", "Negative influence removed", "No connection", None
            )
        return ConnectionClassification(
            "neutral_removed", "Zero-weight connection removed", "No connection", None
        )

    movement = (
        "Increased"
        if child_weight > parent_weight
        else "Decreased" if child_weight < parent_weight else "Unchanged"
    )
    child_sign = (
        "Positive"
        if child_weight > 0.0
        else "Negative" if child_weight < 0.0 else "Neutral"
    )
    if parent_weight == child_weight:
        return ConnectionClassification(
            "unchanged", "Influence unchanged", child_sign, movement
        )
    if parent_weight > 0.0 and child_weight < 0.0:
        return ConnectionClassification(
            "positive_to_negative",
            "Influence changed from positive to negative",
            child_sign,
            movement,
        )
    if parent_weight < 0.0 and child_weight > 0.0:
        return ConnectionClassification(
            "negative_to_positive",
            "Influence changed from negative to positive",
            child_sign,
            movement,
        )
    if parent_weight > 0.0 and child_weight == 0.0:
        label = "Positive influence weakened to zero"
        key = "positive_to_neutral"
    elif parent_weight < 0.0 and child_weight == 0.0:
        label = "Negative influence weakened to zero"
        key = "negative_to_neutral"
    elif parent_weight == 0.0 and child_weight > 0.0:
        label = "Influence changed from zero to positive"
        key = "neutral_to_positive"
    elif parent_weight == 0.0 and child_weight < 0.0:
        label = "Influence changed from zero to negative"
        key = "neutral_to_negative"
    elif parent_weight > 0.0:
        strengthened = child_weight > parent_weight
        label = f"Positive influence {'strengthened' if strengthened else 'weakened'}"
        key = "positive_strengthened" if strengthened else "positive_weakened"
    else:
        strengthened = abs(child_weight) > abs(parent_weight)
        label = f"Negative influence {'strengthened' if strengthened else 'weakened'}"
        key = "negative_strengthened" if strengthened else "negative_weakened"
    return ConnectionClassification(key, label, child_sign, movement)


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
    sociability = output_score("herding")
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
            figsize=(4.5, 4.5),
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
        axis.tick_params(
            axis="x",
            colors="#161a32",
            labelsize=RADAR_AXIS_LABEL_SIZE,
            pad=10,
        )
        for axis_label in axis.get_xticklabels():
            axis_label.set_fontweight("semibold")
        axis.tick_params(
            axis="y",
            colors="#42474d",
            labelsize=RADAR_VALUE_LABEL_SIZE,
        )
        axis.grid(color="#4b5563", alpha=0.48, linewidth=0.9)
        axis.spines["polar"].set_color("#374151")
        axis.spines["polar"].set_linewidth(1.4)
        axis.spines["polar"].set_alpha(0.85)
        figure.subplots_adjust(left=0.20, right=0.80, bottom=0.20, top=0.80)

        with BytesIO() as buffer:
            figure.savefig(
                buffer,
                format="png",
                transparent=True,
                dpi=140,
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
    neuro_integration_hubs, direct_brain_changes = profile_neuroethology(
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
        neuro_integration_hubs=neuro_integration_hubs,
        direct_brain_changes=direct_brain_changes,
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
) -> tuple[tuple[NeuroIntegrationHub, ...], tuple[NeuralShift, ...]]:
    output_key_sequence = tuple(int(key) for key in output_keys)
    input_key_sequence = _ordered_input_keys(input_keys)
    output_key_set = set(output_key_sequence)
    input_key_set = set(input_key_sequence)
    input_rank = {key: index for index, key in enumerate(input_key_sequence)}
    output_rank = {key: index for index, key in enumerate(output_key_sequence)}
    integrations: dict[int, list[tuple[int, NeuralShift]]] = {}
    modulations: dict[int, list[tuple[int, NeuralShift]]] = {}
    direct_changes: list[NeuralShift] = []

    for shift in normalize_neural_shifts(neural_shifts):
        target_id = shift.target_node_id
        source_id = shift.source_node_id
        is_sensor = source_id in input_key_set
        is_action = target_id in output_key_set

        if is_sensor and is_action:
            direct_changes.append(shift)
            continue

        if is_sensor and target_id >= 0:
            integrations.setdefault(target_id, []).append(
                (input_rank[source_id], shift)
            )
            continue

        if source_id >= 0 and is_action:
            modulations.setdefault(source_id, []).append(
                (output_rank[target_id], shift)
            )

    hub_ids = sorted(set(integrations) | set(modulations))
    hubs = tuple(
        NeuroIntegrationHub(
            hub_id=hub_id,
            incoming_sensor_changes=tuple(
                shift for _, shift in sorted(integrations.get(hub_id, ()))
            ),
            outgoing_action_changes=tuple(
                shift for _, shift in sorted(modulations.get(hub_id, ()))
            ),
        )
        for hub_id in hub_ids
    )
    direct_changes.sort(
        key=lambda shift: (
            input_rank.get(shift.source_node_id, len(input_rank)),
            output_rank.get(shift.target_node_id, len(output_rank)),
        )
    )
    return hubs, tuple(direct_changes)


def _ordered_input_keys(input_keys: Iterable[int] | None) -> tuple[int, ...]:
    if input_keys is None:
        return tuple(-(index + 1) for index in range(len(SENSOR_INPUT_NAMES)))
    return tuple(int(key) for key in input_keys)


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
    capacity_ratio = max(0.0, float(traits.stomach_capacity)) / max(
        float(trait.default_stomach_capacity),
        0.0001,
    )
    rate_ratio = max(0.0, float(traits.digestion_rate)) / max(
        float(trait.default_digestion_rate),
        0.0001,
    )
    efficiency_ratio = max(
        0.0,
        float(traits.digestion_efficiency),
    ) / max(float(trait.default_digestion_efficiency), 0.0001)
    digestive_upkeep = min(
        float(metabolism.max_digestive_upkeep_per_second),
        float(metabolism.digestive_upkeep_at_default_per_second)
        * (
            float(metabolism.digestive_capacity_upkeep_weight)
            * capacity_ratio**2
            + float(metabolism.digestive_rate_upkeep_weight) * rate_ratio**2
            + float(metabolism.digestive_efficiency_upkeep_weight)
            * efficiency_ratio**2
        ),
    )
    idle = (
        float(metabolism.basic_metabolism_rate)
        + vision_cost
        + body_cost
        + digestive_upkeep
    )
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
