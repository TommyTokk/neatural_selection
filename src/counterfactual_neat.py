"""Local counterfactual explanations for one focal recurrent NEAT brain.

This module is deliberately independent from Arcade and mutable world objects.
It probes the real evolved network with semantically coherent input
substitutions and never feeds explanatory results back into the simulation.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from enum import Enum
from math import cos, isfinite, pi
from statistics import median
from time import monotonic
from typing import Any

from src.creature.action import ACTION_OUTPUT_NAMES, BrainOutputIndex
from src.behavior_observer import BehaviorKind, BoutStatus
from src.behavior_history import (
    BoundedMetricAccumulator,
    CompletedOutputEffectSummary,
    CompletedSemanticEffect,
    CompletedWhyExplanation,
    EffectDirectionCounts,
)
from src.creature.communication import AcousticObservation, PheromoneSnapshot
from src.creature.neat.brain import NeatBrain
from src.creature.vision import (
    BiomeSensorSnapshot,
    FlockSensorSnapshot,
    SENSOR_CONTRACT,
    SENSOR_INPUT_NAMES,
    VisionTargetSnapshot,
)


MINIMAL_INFLUENCE_THRESHOLD = 0.10
WEAK_INFLUENCE_THRESHOLD = 0.30
MODERATE_INFLUENCE_THRESHOLD = 0.60
DIRECTION_NOISE_EPSILON = 0.05


class SemanticIntervention(str, Enum):
    VISIBLE_FOOD_CUES = "visible_food_cues"
    RESOURCE_GRADIENT_CUES = "resource_gradient_cues"
    SATIATED_STATE = "satiated_state"
    SOCIAL_CUES = "social_cues"
    OFFSPRING_CUES = "offspring_cues"
    ACOUSTIC_CUES = "acoustic_cues"
    RED_PHEROMONE_CUES = "red_pheromone_cues"
    GREEN_PHEROMONE_CUES = "green_pheromone_cues"
    BLUE_PHEROMONE_CUES = "blue_pheromone_cues"
    WALL_CUES = "wall_cues"


class InfluenceLabel(str, Enum):
    MINIMAL = "minimal"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class EffectDirection(str, Enum):
    SUPPORTIVE = "supportive"
    SUPPRESSIVE = "suppressive"
    REVERSING = "reversing"
    MIXED = "mixed"
    MINIMAL = "minimal"


@dataclass(frozen=True, slots=True)
class ProbeBehavior:
    behavior: BehaviorKind
    status: BoutStatus
    bout_id: int
    duration_seconds: float
    target_id: int | None = None


@dataclass(frozen=True, slots=True)
class CounterfactualProbeInput:
    creature_id: int
    selection_generation: int
    brain_revision: int
    simulation_time: float
    sensor_schema_version: int
    behaviors: tuple[ProbeBehavior, ...]
    actual_inputs: tuple[float, ...]
    actual_outputs: tuple[float, ...]
    submitted_monotonic: float
    target_visible: bool = False
    food_target_id: int | None = None
    food_relative_angle: float | None = None
    group_visible: bool = False
    group_relative_angle: float | None = None
    network_state: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PureNeatEvaluator:
    """Picklable functional snapshot of a focal recurrent network."""

    network: Any
    output_activations: tuple[str, ...]
    network_state: dict[str, Any] | None = None

    @classmethod
    def from_brain(cls, brain: NeatBrain) -> PureNeatEvaluator:
        state = brain.captured_activation_network_state()
        if state is None:
            state = brain.export_network_state()
        return cls(
            NeatBrain._clone_network(brain.network, state),
            tuple(brain.output_activations),
            state,
        )

    def evaluate(
        self,
        inputs: tuple[float, ...],
        network_state: dict[str, Any] | None = None,
    ) -> tuple[float, ...]:
        """Evaluate from a state-matched buffer snapshot.

        The evaluator owns its compiled network, so restoring its two buffers
        before activation is sufficient to keep interventions independent. It
        avoids constructing and initializing another RecurrentNetwork for every
        semantic intervention.
        """
        state = self.network_state if network_state is None else network_state
        if state is not None:
            NeatBrain._restore_network_state(self.network, state)
        raw_outputs = self.network.activate(inputs)
        return NeatBrain.normalize_outputs_pure(
            raw_outputs,
            self.output_activations,
        )


@dataclass(frozen=True, slots=True)
class FocalBrainUpdate:
    creature_id: int | None
    selection_generation: int
    brain_revision: int | None
    evaluator_payload: bytes | None


@dataclass(frozen=True, slots=True)
class OutputEffect:
    output_name: str
    actual: float
    counterfactual: float
    delta: float
    influence_score: float
    direction: EffectDirection
    secondary_context: bool = False
    actual_target_alignment: float | None = None
    counterfactual_target_alignment: float | None = None


@dataclass(frozen=True, slots=True)
class SemanticEffectSnapshot:
    intervention: SemanticIntervention
    influence_score: float
    influence_label: InfluenceLabel
    effect_direction: EffectDirection
    output_effects: tuple[OutputEffect, ...]
    sample_count: int


@dataclass(frozen=True, slots=True)
class WhySnapshot:
    creature_id: int
    selection_generation: int
    brain_revision: int
    simulation_time: float
    behavior: BehaviorKind
    status: BoutStatus
    bout_id: int
    behavior_duration: float
    effects: tuple[SemanticEffectSnapshot, ...]
    produced_monotonic: float
    target_id: int | None = None


@dataclass(frozen=True, slots=True)
class WhyBatchResult:
    creature_id: int
    selection_generation: int
    brain_revision: int
    simulation_time: float
    snapshots: tuple[WhySnapshot, ...]
    evaluations_performed: int
    probes_superseded: int
    result_drops: int
    produced_monotonic: float


@dataclass(frozen=True, slots=True)
class CounterfactualWorkerError:
    creature_id: int | None
    selection_generation: int | None
    brain_revision: int | None
    message: str


@dataclass(frozen=True, slots=True)
class CounterfactualDiagnostics:
    probe_requests: int = 0
    probe_requests_dropped: int = 0
    probes_superseded: int = 0
    evaluations_performed: int = 0
    result_drops: int = 0
    result_latency_ms: float | None = None
    latest_result_age_ms: float | None = None
    evaluations_per_second: float = 0.0
    probe_queue_size: int | None = None
    worker_health: str = "idle"
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class BehaviorExplanationSpec:
    scored_outputs: tuple[str, ...]
    displayed_outputs: tuple[str, ...]
    reversal_critical_outputs: tuple[str, ...]
    interventions: tuple[SemanticIntervention, ...]


_COLOR_MASKS = (
    SemanticIntervention.RED_PHEROMONE_CUES,
    SemanticIntervention.GREEN_PHEROMONE_CUES,
    SemanticIntervention.BLUE_PHEROMONE_CUES,
)


BEHAVIOR_EXPLANATION_SPECS: dict[BehaviorKind, BehaviorExplanationSpec] = {
    BehaviorKind.FOOD_ORIENTATION: BehaviorExplanationSpec(
        scored_outputs=("rotate",),
        displayed_outputs=("rotate", "accelerate"),
        reversal_critical_outputs=("rotate",),
        interventions=(
            SemanticIntervention.VISIBLE_FOOD_CUES,
            SemanticIntervention.RESOURCE_GRADIENT_CUES,
            SemanticIntervention.SATIATED_STATE,
            SemanticIntervention.SOCIAL_CUES,
            SemanticIntervention.WALL_CUES,
            *_COLOR_MASKS,
        ),
    ),
    BehaviorKind.FOOD_APPROACH: BehaviorExplanationSpec(
        scored_outputs=("accelerate", "rotate"),
        displayed_outputs=("accelerate", "rotate"),
        reversal_critical_outputs=("accelerate", "rotate"),
        interventions=(
            SemanticIntervention.VISIBLE_FOOD_CUES,
            SemanticIntervention.RESOURCE_GRADIENT_CUES,
            SemanticIntervention.SATIATED_STATE,
            SemanticIntervention.SOCIAL_CUES,
            SemanticIntervention.WALL_CUES,
            *_COLOR_MASKS,
        ),
    ),
    BehaviorKind.FEEDING: BehaviorExplanationSpec(
        scored_outputs=("want_eat",),
        displayed_outputs=("want_eat",),
        reversal_critical_outputs=(),
        interventions=(
            SemanticIntervention.VISIBLE_FOOD_CUES,
            SemanticIntervention.SATIATED_STATE,
            *_COLOR_MASKS,
        ),
    ),
    BehaviorKind.COHESION: BehaviorExplanationSpec(
        scored_outputs=("herding", "rotate", "accelerate"),
        displayed_outputs=("herding", "rotate", "accelerate"),
        reversal_critical_outputs=("rotate", "accelerate"),
        interventions=(
            SemanticIntervention.SOCIAL_CUES,
            SemanticIntervention.SATIATED_STATE,
            SemanticIntervention.VISIBLE_FOOD_CUES,
            SemanticIntervention.RESOURCE_GRADIENT_CUES,
            SemanticIntervention.WALL_CUES,
            *_COLOR_MASKS,
        ),
    ),
    BehaviorKind.PHEROMONE_GRADIENT_RESPONSE: BehaviorExplanationSpec(
        scored_outputs=(
            "flee_panic_intensity",
            "accelerate",
            "rotate",
        ),
        displayed_outputs=(
            "flee_panic_intensity",
            "accelerate",
            "rotate",
        ),
        reversal_critical_outputs=("accelerate", "rotate"),
        interventions=(
            *_COLOR_MASKS,
            SemanticIntervention.SATIATED_STATE,
            SemanticIntervention.SOCIAL_CUES,
            SemanticIntervention.WALL_CUES,
        ),
    ),
}


def _satiated_values() -> dict[str, float]:
    energy_percent = 1.0
    stomach_fullness = 1.0
    feeding_drive = (
        max(0.0, 1.0 - energy_percent)
        * max(0.0, 1.0 - stomach_fullness)
    )
    return {
        "feeding_drive": feeding_drive,
        "energy_percent": energy_percent,
        "stomach_fullness": stomach_fullness,
    }


_NEUTRAL_TARGET = VisionTargetSnapshot(0.0, 0.0, 0.0, 0.0, 0)
_EMPTY_PROXIMITY = _NEUTRAL_TARGET.proximity
_EMPTY_ANGLE = _NEUTRAL_TARGET.angle
_NEUTRAL_BIOME = BiomeSensorSnapshot()
_NEUTRAL_FLOCK = FlockSensorSnapshot()
_NEUTRAL_ACOUSTIC = AcousticObservation()
_NEUTRAL_PHEROMONES = PheromoneSnapshot()

INTERVENTION_REPLACEMENTS: dict[
    SemanticIntervention,
    dict[str, float],
] = {
    SemanticIntervention.VISIBLE_FOOD_CUES: {
        "food_count": min(float(_NEUTRAL_TARGET.count) / 10.0, 1.0),
        "food_proximity": _EMPTY_PROXIMITY,
        "food_angle": _EMPTY_ANGLE,
    },
    SemanticIntervention.RESOURCE_GRADIENT_CUES: {
        "local_richness": _NEUTRAL_BIOME.local_richness,
        "lateral_gradient": _NEUTRAL_BIOME.lateral_gradient,
        "forward_gradient": _NEUTRAL_BIOME.forward_gradient,
    },
    SemanticIntervention.SATIATED_STATE: _satiated_values(),
    SemanticIntervention.SOCIAL_CUES: {
        "creature_count": min(float(_NEUTRAL_TARGET.count) / 5.0, 1.0),
        "creature_proximity": _EMPTY_PROXIMITY,
        "creature_angle": _EMPTY_ANGLE,
        "flock_presence": _NEUTRAL_FLOCK.social_presence,
        "flock_effective_count": max(
            0.0,
            min(1.0, float(_NEUTRAL_FLOCK.flockmate_count) / 4.0),
        ),
        "flock_center_forward": _NEUTRAL_FLOCK.center_forward,
        "flock_center_right": _NEUTRAL_FLOCK.center_right,
        "flock_relative_velocity_forward": (
            _NEUTRAL_FLOCK.relative_velocity_forward
        ),
        "flock_relative_velocity_right": (
            _NEUTRAL_FLOCK.relative_velocity_right
        ),
        "long_range_social_intensity": _NEUTRAL_FLOCK.long_range.intensity,
        "long_range_social_direction_forward": (
            _NEUTRAL_FLOCK.long_range.direction_forward
        ),
        "long_range_social_direction_right": (
            _NEUTRAL_FLOCK.long_range.direction_right
        ),
    },
    SemanticIntervention.OFFSPRING_CUES: {
        "own_infant_proximity": _EMPTY_PROXIMITY,
        "own_infant_angle": _EMPTY_ANGLE,
    },
    SemanticIntervention.ACOUSTIC_CUES: {
        "sound_strength": _NEUTRAL_ACOUSTIC.strength,
        "sound_dir_sin": _NEUTRAL_ACOUSTIC.direction_sin,
        "sound_dir_cos": _NEUTRAL_ACOUSTIC.direction_cos,
        "sound_tone": _NEUTRAL_ACOUSTIC.tone,
    },
    SemanticIntervention.RED_PHEROMONE_CUES: {
        "pheromone_local_red": 0.0,
        "pheromone_lateral_red": 0.0,
        "pheromone_forward_red": 0.0,
    },
    SemanticIntervention.GREEN_PHEROMONE_CUES: {
        "pheromone_local_green": 0.0,
        "pheromone_lateral_green": 0.0,
        "pheromone_forward_green": 0.0,
    },
    SemanticIntervention.BLUE_PHEROMONE_CUES: {
        "pheromone_local_blue": 0.0,
        "pheromone_lateral_blue": 0.0,
        "pheromone_forward_blue": 0.0,
    },
    SemanticIntervention.WALL_CUES: {
        "wall_proximity": _EMPTY_PROXIMITY,
        "wall_angle": _EMPTY_ANGLE,
    },
}

_SENSOR_INDEX = {
    name: index for index, name in enumerate(SENSOR_INPUT_NAMES)
}
for _intervention, _replacements in INTERVENTION_REPLACEMENTS.items():
    unknown = set(_replacements) - set(_SENSOR_INDEX)
    if unknown:
        raise RuntimeError(
            f"{_intervention.value} references unknown sensors: {unknown}"
        )


def apply_intervention(
    intervention: SemanticIntervention,
    actual_inputs: tuple[float, ...],
) -> tuple[float, ...]:
    """Return one coherent counterfactual without changing the factual tuple."""
    if len(actual_inputs) != len(SENSOR_INPUT_NAMES):
        raise ValueError(
            f"Expected {len(SENSOR_INPUT_NAMES)} inputs, "
            f"received {len(actual_inputs)}."
        )
    counterfactual = list(actual_inputs)
    for name, replacement in INTERVENTION_REPLACEMENTS[intervention].items():
        counterfactual[_SENSOR_INDEX[name]] = float(replacement)
    return tuple(counterfactual)


def influence_label(score: float) -> InfluenceLabel:
    if score < MINIMAL_INFLUENCE_THRESHOLD:
        return InfluenceLabel.MINIMAL
    if score < WEAK_INFLUENCE_THRESHOLD:
        return InfluenceLabel.WEAK
    if score < MODERATE_INFLUENCE_THRESHOLD:
        return InfluenceLabel.MODERATE
    return InfluenceLabel.STRONG


_SIGNED_OUTPUTS = {
    ACTION_OUTPUT_NAMES[BrainOutputIndex.ACCELERATE],
    ACTION_OUTPUT_NAMES[BrainOutputIndex.ROTATE],
    ACTION_OUTPUT_NAMES[BrainOutputIndex.ACOUSTIC_TONE],
}
_OUTPUT_INDEX = {
    name: index for index, name in enumerate(ACTION_OUTPUT_NAMES)
}
_FOOD_TARGET_BEHAVIORS = {
    BehaviorKind.FOOD_ORIENTATION,
    BehaviorKind.FOOD_APPROACH,
}


def _project_output(name: str, value: float) -> float:
    finite = float(value) if isfinite(float(value)) else 0.0
    if name in _SIGNED_OUTPUTS:
        return max(-1.0, min(1.0, finite))
    return max(0.0, min(1.0, finite))


def _output_span(name: str) -> float:
    return 2.0 if name in _SIGNED_OUTPUTS else 1.0


def output_effect(
    name: str,
    actual_value: float,
    counterfactual_value: float,
    *,
    secondary_context: bool = False,
) -> OutputEffect:
    actual = _project_output(name, actual_value)
    counterfactual = _project_output(name, counterfactual_value)
    delta = actual - counterfactual
    score = max(0.0, min(1.0, abs(delta) / _output_span(name)))
    if score < MINIMAL_INFLUENCE_THRESHOLD:
        direction = EffectDirection.MINIMAL
    elif (
        name in _SIGNED_OUTPUTS
        and actual * counterfactual < 0.0
        and abs(actual) >= DIRECTION_NOISE_EPSILON
        and abs(counterfactual) >= DIRECTION_NOISE_EPSILON
    ):
        direction = EffectDirection.REVERSING
    elif abs(counterfactual) < abs(actual):
        direction = EffectDirection.SUPPORTIVE
    elif abs(counterfactual) > abs(actual):
        direction = EffectDirection.SUPPRESSIVE
    else:
        direction = EffectDirection.MINIMAL
    return OutputEffect(
        output_name=name,
        actual=actual,
        counterfactual=counterfactual,
        delta=delta,
        influence_score=score,
        direction=direction,
        secondary_context=secondary_context,
    )


def steering_toward_target(
    rotate_output: float,
    target_relative_angle: float,
    center_dead_zone: float,
) -> float:
    """Return signed steering quality relative to one factual target.

    Positive relative angles and positive rotation are both
    counter-clockwise in the simulation. Outside the centered dead zone,
    positive results therefore turn toward the target and negative results
    turn away. Inside the dead zone, avoiding an unnecessary turn is the
    best stabilizing response.
    """
    angle = float(target_relative_angle)
    dead_zone = float(center_dead_zone)
    if not isfinite(angle):
        raise ValueError("Target-relative angle must be finite.")
    if not isfinite(dead_zone) or dead_zone < 0.0:
        raise ValueError(
            "Target center dead zone must be finite and non-negative."
        )
    rotate = _project_output("rotate", rotate_output)
    if abs(angle) <= dead_zone:
        return -abs(rotate)
    return rotate if angle > 0.0 else -rotate


def _target_relative_rotate_effect(
    actual_value: float,
    counterfactual_value: float,
    *,
    target_relative_angle: float,
    center_dead_zone: float,
    secondary_context: bool,
) -> OutputEffect:
    """Return a raw rotate delta with target-relative direction semantics."""
    return _aligned_output_effect(
        "rotate",
        actual_value,
        counterfactual_value,
        actual_alignment=steering_toward_target(
            actual_value,
            target_relative_angle,
            center_dead_zone,
        ),
        counterfactual_alignment=steering_toward_target(
            counterfactual_value,
            target_relative_angle,
            center_dead_zone,
        ),
        secondary_context=secondary_context,
    )


def _target_relative_acceleration_effect(
    actual_value: float,
    counterfactual_value: float,
    *,
    target_relative_angle: float,
    secondary_context: bool,
) -> OutputEffect:
    """Return acceleration influence aligned with one factual target."""
    actual = _project_output("accelerate", actual_value)
    counterfactual = _project_output("accelerate", counterfactual_value)
    forward_alignment = cos(float(target_relative_angle))
    if abs(forward_alignment) < DIRECTION_NOISE_EPSILON:
        forward_alignment = 0.0
    return _aligned_output_effect(
        "accelerate",
        actual,
        counterfactual,
        actual_alignment=actual * forward_alignment,
        counterfactual_alignment=counterfactual * forward_alignment,
        secondary_context=secondary_context,
    )


def _gradient_response_movement_effect(
    name: str,
    actual_value: float,
    counterfactual_value: float,
    *,
    secondary_context: bool,
) -> OutputEffect:
    """Interpret a generic realized gradient response."""
    actual = _project_output(name, actual_value)
    counterfactual = _project_output(name, counterfactual_value)
    if name == "accelerate":
        actual_alignment = actual
        counterfactual_alignment = counterfactual
    else:
        actual_alignment = -abs(actual)
        counterfactual_alignment = -abs(counterfactual)
    return _aligned_output_effect(
        name,
        actual,
        counterfactual,
        actual_alignment=actual_alignment,
        counterfactual_alignment=counterfactual_alignment,
        secondary_context=secondary_context,
    )


def _aligned_output_effect(
    name: str,
    actual_value: float,
    counterfactual_value: float,
    *,
    actual_alignment: float,
    counterfactual_alignment: float,
    secondary_context: bool,
) -> OutputEffect:
    """Return one output change with explicit behavior-relative alignment."""
    actual = _project_output(name, actual_value)
    counterfactual = _project_output(name, counterfactual_value)
    delta = actual - counterfactual
    score = max(0.0, min(1.0, abs(delta) / _output_span(name)))
    direction = _target_alignment_direction(
        score,
        actual_alignment,
        counterfactual_alignment,
    )
    return OutputEffect(
        output_name=name,
        actual=actual,
        counterfactual=counterfactual,
        delta=delta,
        influence_score=score,
        direction=direction,
        secondary_context=secondary_context,
        actual_target_alignment=actual_alignment,
        counterfactual_target_alignment=counterfactual_alignment,
    )


def _target_alignment_direction(
    influence_score: float,
    actual_alignment: float,
    counterfactual_alignment: float,
) -> EffectDirection:
    """Classify one effect from factual behavior-relative alignment."""
    if influence_score < MINIMAL_INFLUENCE_THRESHOLD:
        return EffectDirection.MINIMAL
    materially_reverses = (
        actual_alignment >= DIRECTION_NOISE_EPSILON
        and counterfactual_alignment <= -DIRECTION_NOISE_EPSILON
    ) or (
        actual_alignment <= -DIRECTION_NOISE_EPSILON
        and counterfactual_alignment >= DIRECTION_NOISE_EPSILON
    )
    if materially_reverses:
        return EffectDirection.REVERSING
    if actual_alignment > counterfactual_alignment:
        return EffectDirection.SUPPORTIVE
    if actual_alignment < counterfactual_alignment:
        return EffectDirection.SUPPRESSIVE
    return EffectDirection.MINIMAL


def _aggregate_direction(
    score: float,
    effects: tuple[OutputEffect, ...],
    spec: BehaviorExplanationSpec,
) -> EffectDirection:
    if score < MINIMAL_INFLUENCE_THRESHOLD:
        return EffectDirection.MINIMAL
    by_name = {effect.output_name: effect for effect in effects}
    if any(
        by_name[name].direction is EffectDirection.REVERSING
        for name in spec.reversal_critical_outputs
        if name in by_name
    ):
        return EffectDirection.REVERSING
    directions = {
        by_name[name].direction
        for name in spec.scored_outputs
        if (
            name in by_name
            and by_name[name].direction is not EffectDirection.MINIMAL
        )
    }
    if (
        EffectDirection.SUPPORTIVE in directions
        and EffectDirection.SUPPRESSIVE in directions
    ):
        return EffectDirection.MIXED
    if EffectDirection.SUPPORTIVE in directions:
        return EffectDirection.SUPPORTIVE
    if EffectDirection.SUPPRESSIVE in directions:
        return EffectDirection.SUPPRESSIVE
    if EffectDirection.REVERSING in directions:
        return EffectDirection.REVERSING
    return EffectDirection.MINIMAL


def _dominant_direction(
    score: float,
    directions: Counter[EffectDirection],
) -> EffectDirection:
    """Return the modal non-minimal direction, with deterministic ties."""
    if score < MINIMAL_INFLUENCE_THRESHOLD:
        return EffectDirection.MINIMAL
    non_minimal = {
        direction: count
        for direction, count in directions.items()
        if direction is not EffectDirection.MINIMAL and count > 0
    }
    if not non_minimal:
        return EffectDirection.MINIMAL
    highest = max(non_minimal.values())
    leaders = [
        direction
        for direction, count in non_minimal.items()
        if count == highest
    ]
    return leaders[0] if len(leaders) == 1 else EffectDirection.MIXED


def semantic_effect(
    behavior: BehaviorKind,
    intervention: SemanticIntervention,
    actual_outputs: tuple[float, ...],
    counterfactual_outputs: tuple[float, ...],
    *,
    target_visible: bool = False,
    food_relative_angle: float | None = None,
    group_visible: bool = False,
    group_relative_angle: float | None = None,
    target_center_dead_zone: float = 0.05,
) -> SemanticEffectSnapshot:
    spec = BEHAVIOR_EXPLANATION_SPECS[behavior]
    food_target_relative = behavior in _FOOD_TARGET_BEHAVIORS
    if food_target_relative and (
        not target_visible
        or food_relative_angle is None
        or not isfinite(float(food_relative_angle))
    ):
        raise ValueError(
            "Food-oriented counterfactual effects require a visible factual "
            "food target with a finite relative angle."
        )
    group_target_relative = behavior is BehaviorKind.COHESION
    if group_target_relative and (
        not group_visible
        or group_relative_angle is None
        or not isfinite(float(group_relative_angle))
    ):
        raise ValueError(
            "Cohesion counterfactual effects require a visible factual "
            "flock center with a finite relative angle."
        )

    def build_output_effect(name: str) -> OutputEffect:
        actual = actual_outputs[_OUTPUT_INDEX[name]]
        counterfactual = counterfactual_outputs[_OUTPUT_INDEX[name]]
        secondary = name not in spec.scored_outputs
        if food_target_relative and name == "rotate":
            return _target_relative_rotate_effect(
                actual,
                counterfactual,
                target_relative_angle=float(food_relative_angle),
                center_dead_zone=target_center_dead_zone,
                secondary_context=secondary,
            )
        if food_target_relative and name == "accelerate":
            return _target_relative_acceleration_effect(
                actual,
                counterfactual,
                target_relative_angle=float(food_relative_angle),
                secondary_context=secondary,
            )
        if group_target_relative and name == "rotate":
            return _target_relative_rotate_effect(
                actual,
                counterfactual,
                target_relative_angle=float(group_relative_angle),
                center_dead_zone=target_center_dead_zone,
                secondary_context=secondary,
            )
        if group_target_relative and name == "accelerate":
            return _target_relative_acceleration_effect(
                actual,
                counterfactual,
                target_relative_angle=float(group_relative_angle),
                secondary_context=secondary,
            )
        return output_effect(
            name,
            actual,
            counterfactual,
            secondary_context=secondary,
        )

    effects = tuple(
        build_output_effect(name)
        for name in spec.displayed_outputs
    )
    scored = [
        effect.influence_score
        for effect in effects
        if effect.output_name in spec.scored_outputs
    ]
    score = (
        0.0
        if not scored
        else max(0.0, min(1.0, sum(scored) / len(scored)))
    )
    return SemanticEffectSnapshot(
        intervention=intervention,
        influence_score=score,
        influence_label=influence_label(score),
        effect_direction=_aggregate_direction(score, effects, spec),
        output_effects=effects,
        sample_count=1,
    )


class CounterfactualProbeJob:
    """Cooperative probe that performs at most one activation per advance."""

    def __init__(
        self,
        probe: CounterfactualProbeInput,
        evaluator: PureNeatEvaluator,
    ) -> None:
        self.probe = probe
        self.evaluator = evaluator
        interventions = {
            intervention
            for behavior in probe.behaviors
            for intervention in BEHAVIOR_EXPLANATION_SPECS[
                behavior.behavior
            ].interventions
        }
        self.interventions = tuple(
            intervention
            for intervention in SemanticIntervention
            if intervention in interventions
        )
        self._next_index = 0
        self.outputs: dict[SemanticIntervention, tuple[float, ...]] = {}

    @property
    def complete(self) -> bool:
        return self._next_index >= len(self.interventions)

    def advance(self) -> bool:
        if self.complete:
            return True
        intervention = self.interventions[self._next_index]
        inputs = apply_intervention(intervention, self.probe.actual_inputs)
        self.outputs[intervention] = self.evaluator.evaluate(
            inputs,
            self.probe.network_state,
        )
        self._next_index += 1
        return self.complete


class _CompletedOutputAccumulator:
    def __init__(self, output_name: str, capacity: int) -> None:
        self.output_name = output_name
        self.actual = BoundedMetricAccumulator(capacity)
        self.counterfactual = BoundedMetricAccumulator(capacity)
        self.delta = BoundedMetricAccumulator(capacity)

    def add(self, effect: OutputEffect) -> None:
        self.actual.add(effect.actual)
        self.counterfactual.add(effect.counterfactual)
        self.delta.add(effect.delta)

    def completed_summary(self) -> CompletedOutputEffectSummary:
        delta_median, delta_p25, delta_p75, estimated = (
            self.delta.summary_values()
        )
        return CompletedOutputEffectSummary(
            output_name=self.output_name,
            sample_count=self.actual.total_count,
            median_factual=self.actual.summary_values()[0],
            median_counterfactual=self.counterfactual.summary_values()[0],
            median_delta=delta_median,
            delta_p25=delta_p25,
            delta_p75=delta_p75,
            quantiles_estimated=estimated,
        )


class _CompletedSemanticAccumulator:
    def __init__(
        self,
        behavior: BehaviorKind,
        intervention: SemanticIntervention,
        capacity: int,
    ) -> None:
        self.behavior = behavior
        self.intervention = intervention
        self.influence = BoundedMetricAccumulator(capacity)
        self.directions: Counter[EffectDirection] = Counter()
        self.outputs: dict[str, _CompletedOutputAccumulator] = {}

    def add(self, sample: SemanticEffectSnapshot) -> None:
        self.influence.add(sample.influence_score)
        self.directions[sample.effect_direction] += 1
        for effect in sample.output_effects:
            accumulator = self.outputs.get(effect.output_name)
            if accumulator is None:
                accumulator = _CompletedOutputAccumulator(
                    effect.output_name,
                    self.influence.capacity,
                )
                self.outputs[effect.output_name] = accumulator
            accumulator.add(effect)

    def finalize(self) -> CompletedSemanticEffect:
        score, p25, p75, estimated = self.influence.summary_values()
        return CompletedSemanticEffect(
            intervention=self.intervention,
            sample_count=self.influence.total_count,
            median_influence=score,
            p25=p25,
            p75=p75,
            influence_label=influence_label(score),
            effect_direction=_dominant_direction(score, self.directions),
            direction_counts=EffectDirectionCounts(
                supportive=self.directions[EffectDirection.SUPPORTIVE],
                suppressive=self.directions[EffectDirection.SUPPRESSIVE],
                reversing=self.directions[EffectDirection.REVERSING],
                mixed=self.directions[EffectDirection.MIXED],
                minimal=self.directions[EffectDirection.MINIMAL],
            ),
            output_summaries=tuple(
                output.completed_summary()
                for output in self.outputs.values()
            ),
            quantiles_estimated=estimated,
        )


class CounterfactualBoutAggregator:
    """Bounded median aggregation, isolated by behavior and bout identity."""

    def __init__(
        self,
        history_capacity: int,
        target_center_dead_zone: float = 0.05,
    ) -> None:
        self.history_capacity = history_capacity
        self.target_center_dead_zone = target_center_dead_zone
        self._history: dict[
            tuple[
                int,
                int,
                int,
                BehaviorKind,
                int,
                int | None,
                SemanticIntervention,
            ],
            deque[SemanticEffectSnapshot],
        ] = {}
        self._completion_history: dict[
            tuple[
                int,
                int,
                BehaviorKind,
                int,
                SemanticIntervention,
            ],
            _CompletedSemanticAccumulator,
        ] = {}

    def reset(self) -> None:
        self._history.clear()
        self._completion_history.clear()

    def complete_job(
        self,
        job: CounterfactualProbeJob,
    ) -> tuple[WhySnapshot, ...]:
        probe = job.probe
        active_prefixes = {
            (
                probe.creature_id,
                probe.selection_generation,
                probe.brain_revision,
                observed.behavior,
                observed.bout_id,
                observed.target_id,
            )
            for observed in probe.behaviors
        }
        active_completion_prefixes = {
            (
                probe.creature_id,
                probe.selection_generation,
                observed.behavior,
                observed.bout_id,
            )
            for observed in probe.behaviors
        }
        self._history = {
            key: history
            for key, history in self._history.items()
            if key[:6] in active_prefixes
        }
        self._completion_history = {
            key: history
            for key, history in self._completion_history.items()
            if key[:4] in active_completion_prefixes
        }
        snapshots: list[WhySnapshot] = []
        produced = monotonic()
        for observed in probe.behaviors:
            spec = BEHAVIOR_EXPLANATION_SPECS[observed.behavior]
            aggregated: list[SemanticEffectSnapshot] = []
            for intervention in spec.interventions:
                sample = semantic_effect(
                    observed.behavior,
                    intervention,
                    probe.actual_outputs,
                    job.outputs[intervention],
                    target_visible=probe.target_visible,
                    food_relative_angle=probe.food_relative_angle,
                    group_visible=probe.group_visible,
                    group_relative_angle=probe.group_relative_angle,
                    target_center_dead_zone=self.target_center_dead_zone,
                )
                key = (
                    probe.creature_id,
                    probe.selection_generation,
                    probe.brain_revision,
                    observed.behavior,
                    observed.bout_id,
                    observed.target_id,
                    intervention,
                )
                history = self._history.setdefault(
                    key,
                    deque(maxlen=self.history_capacity),
                )
                history.append(sample)
                completion_key = (
                    probe.creature_id,
                    probe.selection_generation,
                    observed.behavior,
                    observed.bout_id,
                    intervention,
                )
                completion = self._completion_history.get(completion_key)
                if completion is None:
                    completion = _CompletedSemanticAccumulator(
                        observed.behavior,
                        intervention,
                        max(4, self.history_capacity),
                    )
                    self._completion_history[completion_key] = completion
                completion.add(sample)
                aggregated.append(
                    self._aggregate_samples(observed.behavior, history)
                )
            aggregated.sort(
                key=lambda effect: (
                    -effect.influence_score,
                    list(SemanticIntervention).index(effect.intervention),
                )
            )
            snapshots.append(
                WhySnapshot(
                    creature_id=probe.creature_id,
                    selection_generation=probe.selection_generation,
                    brain_revision=probe.brain_revision,
                    simulation_time=probe.simulation_time,
                    behavior=observed.behavior,
                    status=observed.status,
                    bout_id=observed.bout_id,
                    behavior_duration=observed.duration_seconds,
                    effects=tuple(aggregated),
                    produced_monotonic=produced,
                    target_id=observed.target_id,
                )
            )
        return tuple(snapshots)

    def finalize_completed_bout(
        self,
        creature_id: int,
        selection_generation: int,
        behavior: BehaviorKind,
        bout_id: int,
    ) -> CompletedWhyExplanation | None:
        effects: list[CompletedSemanticEffect] = []
        for intervention in SemanticIntervention:
            key = (
                creature_id,
                selection_generation,
                behavior,
                bout_id,
                intervention,
            )
            accumulator = self._completion_history.pop(key, None)
            if accumulator is not None:
                effects.append(accumulator.finalize())
        self._history = {
            key: history
            for key, history in self._history.items()
            if not (
                key[0] == creature_id
                and key[1] == selection_generation
                and key[3] is behavior
                and key[4] == bout_id
            )
        }
        if not effects:
            return None
        effects.sort(
            key=lambda effect: (
                -effect.median_influence,
                list(SemanticIntervention).index(effect.intervention),
            )
        )
        return CompletedWhyExplanation(
            behavior=behavior,
            bout_id=bout_id,
            effects=tuple(effects),
        )

    @staticmethod
    def _aggregate_samples(
        _behavior: BehaviorKind,
        samples: deque[SemanticEffectSnapshot],
    ) -> SemanticEffectSnapshot:
        median_score = float(
            median(sample.influence_score for sample in samples)
        )
        representative = samples[-1]
        closest_distance = abs(
            representative.influence_score - median_score
        )
        for sample in reversed(tuple(samples)[:-1]):
            distance = abs(sample.influence_score - median_score)
            if distance < closest_distance - 1e-12:
                representative = sample
                closest_distance = distance
        return replace(representative, sample_count=len(samples))


def mapped_probe_behaviors(
    behaviors: tuple[Any, ...],
) -> tuple[ProbeBehavior, ...]:
    """Convert current observer states to the compact WHY behavior contract."""
    return tuple(
        ProbeBehavior(
            behavior=state.behavior,
            status=state.status,
            bout_id=int(getattr(state, "bout_id", 0)),
            duration_seconds=float(state.duration_seconds),
            target_id=getattr(state, "target_id", None),
        )
        for state in behaviors
        if state.behavior in BEHAVIOR_EXPLANATION_SPECS
    )


def validate_probe(probe: CounterfactualProbeInput) -> None:
    if probe.sensor_schema_version != SENSOR_CONTRACT.schema_version:
        raise ValueError(
            "Counterfactual probe sensor schema does not match the worker."
        )
    if len(probe.actual_inputs) != len(SENSOR_INPUT_NAMES):
        raise ValueError("Counterfactual probe has an invalid input vector.")
    if len(probe.actual_outputs) != len(ACTION_OUTPUT_NAMES):
        raise ValueError("Counterfactual probe has an invalid output vector.")
    if not probe.behaviors:
        raise ValueError("Counterfactual probe has no mapped behaviors.")
    if type(probe.target_visible) is not bool:
        raise ValueError(
            "Counterfactual probe target visibility must be boolean."
        )
    if type(probe.group_visible) is not bool:
        raise ValueError(
            "Counterfactual probe group visibility must be boolean."
        )
    if (
        probe.food_target_id is not None
        and (
            type(probe.food_target_id) is not int
            or probe.food_target_id < 0
        )
    ):
        raise ValueError("Counterfactual probe has an invalid food target ID.")
    if probe.food_relative_angle is not None and (
        not isfinite(float(probe.food_relative_angle))
        or abs(float(probe.food_relative_angle)) > pi
    ):
        raise ValueError(
            "Counterfactual probe food-relative angle must be finite and "
            "within [-pi, pi]."
        )
    if probe.group_relative_angle is not None and (
        not isfinite(float(probe.group_relative_angle))
        or abs(float(probe.group_relative_angle)) > pi
    ):
        raise ValueError(
            "Counterfactual probe group-relative angle must be finite and "
            "within [-pi, pi]."
        )
    for observed in probe.behaviors:
        if observed.behavior not in _FOOD_TARGET_BEHAVIORS:
            continue
        if (
            not probe.target_visible
            or probe.food_target_id is None
            or probe.food_relative_angle is None
            or type(observed.target_id) is not int
            or observed.target_id != probe.food_target_id
        ):
            raise ValueError(
                "Food-oriented probe behavior lacks matching factual target "
                "context."
            )
    if any(
        observed.behavior is BehaviorKind.COHESION
        for observed in probe.behaviors
    ) and (
        not probe.group_visible or probe.group_relative_angle is None
    ):
        raise ValueError(
            "Cohesion probe behavior lacks factual flock-center context."
        )
