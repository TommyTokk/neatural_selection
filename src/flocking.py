from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot

from configs.sim_config import (
    FlockingConfig,
    SocialCompatibilityConfig,
    SocialCompatibilityMode,
)

Vector = tuple[float, float]
ZERO_VECTOR: Vector = (0.0, 0.0)


class SocialCompatibilityResolver:
    def __init__(
        self,
        config: SocialCompatibilityConfig,
        legacy_resolver,
    ) -> None:
        self.config = config
        self.legacy_resolver = legacy_resolver
        self._social_tag_cache: dict[tuple[int, int], float] = {}

    def compatibility(self, first, second) -> float:
        mode = SocialCompatibilityMode(self.config.mode)
        if mode is SocialCompatibilityMode.LEGACY:
            return _clamp(float(self.legacy_resolver(first, second)))
        if mode is SocialCompatibilityMode.SPECIES:
            first_species = getattr(
                getattr(first, "lineage", None), "species_id", None
            )
            second_species = getattr(
                getattr(second, "lineage", None), "species_id", None
            )
            return (
                1.0
                if first_species is not None
                and first_species == second_species
                else 0.0
            )
        pair = tuple(sorted((first.creature_id, second.creature_id)))
        cached = self._social_tag_cache.get(pair)
        if cached is not None:
            return cached
        first_traits = first.flocking_traits
        second_traits = second.flocking_traits
        dx = first_traits.social_tag_x - second_traits.social_tag_x
        dy = first_traits.social_tag_y - second_traits.social_tag_y
        sigma = max(1e-12, float(self.config.social_tag_sigma))
        value = exp(-((dx * dx + dy * dy) / (2.0 * sigma * sigma)))
        self._social_tag_cache[pair] = value
        return value

    def discard_creature(self, creature_id: int) -> None:
        self._social_tag_cache = {
            pair: value
            for pair, value in self._social_tag_cache.items()
            if creature_id not in pair
        }


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _limit(vector: Vector, maximum: float) -> Vector:
    maximum = max(0.0, float(maximum))
    magnitude = hypot(*vector)
    if magnitude <= maximum or magnitude <= 1e-12:
        return vector
    scale = maximum / magnitude
    return vector[0] * scale, vector[1] * scale


@dataclass(frozen=True, slots=True)
class LongRangeSocialObservation:
    intensity: float = 0.0
    direction_forward: float = 0.0
    direction_right: float = 0.0


@dataclass(frozen=True, slots=True)
class SocialObservation:
    present: bool = False
    visible_creature_count: int = 0
    compatible_visible_count: int = 0
    personal_space_presence: float = 0.0
    social_presence: float = 0.0
    effective_count: float = 0.0
    center_forward: float = 0.0
    center_right: float = 0.0
    relative_velocity_forward: float = 0.0
    relative_velocity_right: float = 0.0
    separation_forward: float = 0.0
    separation_right: float = 0.0
    mean_proximity: float = 0.0
    center_distance: float = 0.0
    mean_neighbor_distance: float = 0.0
    mean_heading_error: float = 0.0
    mean_group_velocity: Vector = ZERO_VECTOR
    long_range: LongRangeSocialObservation = field(
        default_factory=LongRangeSocialObservation
    )


@dataclass(frozen=True, slots=True)
class FlockingWeights:
    separation: float = 0.0
    alignment: float = 0.0
    cohesion: float = 0.0
    engagement: float = 0.0
    panic_attenuation: float = 1.0


@dataclass(frozen=True, slots=True)
class SocialIntent:
    desired_velocity: Vector = ZERO_VECTOR
    requested_force: Vector = ZERO_VECTOR
    confidence: float = 0.0
    weights: FlockingWeights = field(default_factory=FlockingWeights)
    separation_velocity: Vector = ZERO_VECTOR
    alignment_velocity: Vector = ZERO_VECTOR
    cohesion_velocity: Vector = ZERO_VECTOR


@dataclass(frozen=True, slots=True)
class FlockingRuntimeSnapshot:
    observation: SocialObservation = field(default_factory=SocialObservation)
    intent: SocialIntent = field(default_factory=SocialIntent)
    neural_desired_velocity: Vector = ZERO_VECTOR
    blended_desired_velocity: Vector = ZERO_VECTOR
    mandatory_avoidance: Vector = ZERO_VECTOR
    requested_social_contribution: Vector = ZERO_VECTOR
    accepted_social_contribution: Vector = ZERO_VECTOR
    social_influence: float = 0.0
    neural_herding: float = 0.0
    panic: float = 0.0
    local_group_id: int | None = None
    local_group_size: int = 0


def calculate_flocking_weights(
    *,
    herding: float,
    panic: float,
    separation_gene: float,
    alignment_gene: float,
    cohesion_gene: float,
    personal_space_presence: float = 1.0,
    social_presence: float = 1.0,
    minimum_social_engagement: float = 0.25,
    panic_suppression_strength: float = 0.5,
) -> FlockingWeights:
    herding = _clamp(herding)
    panic = _clamp(panic)
    separation_gene = _clamp(separation_gene)
    alignment_gene = _clamp(alignment_gene)
    cohesion_gene = _clamp(cohesion_gene)
    personal_space_presence = _clamp(personal_space_presence)
    social_presence = _clamp(social_presence)
    minimum_social_engagement = _clamp(minimum_social_engagement)
    panic_suppression_strength = _clamp(panic_suppression_strength)

    engagement = social_presence * (
        minimum_social_engagement
        + (1.0 - minimum_social_engagement) * herding
    )
    panic_attenuation = 1.0 - panic_suppression_strength * panic
    return FlockingWeights(
        separation=personal_space_presence * separation_gene,
        alignment=engagement * alignment_gene * panic_attenuation,
        cohesion=engagement * cohesion_gene * panic_attenuation,
        engagement=engagement,
        panic_attenuation=panic_attenuation,
    )


def calculate_social_intent(
    *,
    current_velocity: Vector,
    separation_velocity: Vector,
    alignment_velocity: Vector,
    cohesion_velocity: Vector,
    weights: FlockingWeights,
    effective_count: float,
    target_group_size: int,
    max_speed: float,
) -> SocialIntent:
    weighted = (
        separation_velocity[0] * weights.separation
        + alignment_velocity[0] * weights.alignment
        + cohesion_velocity[0] * weights.cohesion,
        separation_velocity[1] * weights.separation
        + alignment_velocity[1] * weights.alignment
        + cohesion_velocity[1] * weights.cohesion,
    )
    weight_sum = weights.separation + weights.alignment + weights.cohesion
    desired = (
        current_velocity
        if weight_sum <= 1e-12
        else (weighted[0] / weight_sum, weighted[1] / weight_sum)
    )
    desired = _limit(desired, max_speed)
    group_scale = _clamp(
        max(0.0, effective_count) / max(1, int(target_group_size))
    )
    confidence = _clamp(
        max(weights.separation, weights.alignment, weights.cohesion)
        * (0.5 + 0.5 * group_scale)
    )
    return SocialIntent(
        desired_velocity=desired,
        requested_force=(
            desired[0] - current_velocity[0],
            desired[1] - current_velocity[1],
        ),
        confidence=confidence,
        weights=weights,
        separation_velocity=separation_velocity,
        alignment_velocity=alignment_velocity,
        cohesion_velocity=cohesion_velocity,
    )


def blend_desired_velocity(
    neural_desired_velocity: Vector,
    social_desired_velocity: Vector,
    social_influence: float,
) -> Vector:
    influence = _clamp(social_influence)
    if influence <= 0.0:
        return neural_desired_velocity
    return (
        neural_desired_velocity[0] * (1.0 - influence)
        + social_desired_velocity[0] * influence,
        neural_desired_velocity[1] * (1.0 - influence)
        + social_desired_velocity[1] * influence,
    )


def configured_social_influence(
    config: FlockingConfig,
    intent: SocialIntent,
) -> float:
    movement_engagement = max(
        intent.weights.engagement,
        intent.weights.separation,
    )
    return _clamp(
        config.max_social_influence
        * movement_engagement
        * intent.confidence,
        0.0,
        config.max_social_influence,
    )


def remove_opposing_component(vector: Vector, avoidance: Vector) -> Vector:
    avoidance_magnitude_squared = (
        avoidance[0] * avoidance[0] + avoidance[1] * avoidance[1]
    )
    if avoidance_magnitude_squared <= 1e-12:
        return vector
    projection_scale = (
        vector[0] * avoidance[0] + vector[1] * avoidance[1]
    ) / avoidance_magnitude_squared
    if projection_scale >= 0.0:
        return vector
    return (
        vector[0] - avoidance[0] * projection_scale,
        vector[1] - avoidance[1] * projection_scale,
    )


def allocate_force_budget(
    requested: Vector,
    remaining_budget: float,
) -> tuple[Vector, float]:
    accepted = _limit(requested, remaining_budget)
    return accepted, max(0.0, remaining_budget - hypot(*accepted))


def accepted_counterfactual_contribution(
    *,
    blended_request: Vector,
    neural_request: Vector,
    mandatory_avoidance: Vector,
    remaining_budget: float,
) -> tuple[Vector, Vector, Vector]:
    constrained_blended = remove_opposing_component(
        blended_request, mandatory_avoidance
    )
    constrained_neural = remove_opposing_component(
        neural_request, mandatory_avoidance
    )
    accepted_blended, _ = allocate_force_budget(
        constrained_blended, remaining_budget
    )
    accepted_neural, _ = allocate_force_budget(
        constrained_neural, remaining_budget
    )
    contribution = (
        accepted_blended[0] - accepted_neural[0],
        accepted_blended[1] - accepted_neural[1],
    )
    return accepted_blended, accepted_neural, contribution
