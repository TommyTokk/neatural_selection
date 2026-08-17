from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot

from configs.sim_config import (
    FlockingConfig,
    SocialCompatibilityConfig,
    SocialCompatibilityMode,
)
from src.creature.common import clamp as _clamp

Vector = tuple[float, float]
ZERO_VECTOR: Vector = (0.0, 0.0)


class SocialCompatibilityResolver:
    def __init__(
        self,
        config: SocialCompatibilityConfig,
        legacy_resolver,
    ) -> None:
        """Execute init behavior.

Parameters
----------
config
    Input used by this creature-domain operation.
legacy_resolver
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep init behavior explicit in its owning subsystem.
        self.config = config
        self.legacy_resolver = legacy_resolver
        self._social_tag_cache: dict[tuple[int, int], float] = {}

    def compatibility(self, first, second) -> float:
        """Execute compatibility behavior.

Parameters
----------
first
    Input used by this creature-domain operation.
second
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep compatibility behavior explicit in its owning subsystem.
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
        """Execute discard creature behavior.

Parameters
----------
creature_id
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep discard creature behavior explicit in its owning subsystem.
        self._social_tag_cache = {
            pair: value
            for pair, value in self._social_tag_cache.items()
            if creature_id not in pair
        }


def _limit(vector: Vector, maximum: float) -> Vector:
    """Execute limit behavior.

Parameters
----------
vector
    Input used by this creature-domain operation.
maximum
    Input used by this creature-domain operation.
Returns
-------
Vector
    Result produced by this creature-domain operation."""
    # Keep limit behavior explicit in its owning subsystem.
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
    accepted_counterfactual_delta: Vector = ZERO_VECTOR
    social_influence: float = 0.0
    raw_neural_herding: float = 0.0
    effective_herding: float = 0.0
    panic: float = 0.0
    local_group_id: int | None = None
    local_group_size: int = 0


@dataclass(slots=True)
class SocialRuntime:
    """Flat mutable continuation state; public dataclasses are snapshots only."""

    observation_present: bool = False
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
    mean_group_velocity_x: float = 0.0
    mean_group_velocity_y: float = 0.0
    long_range_intensity: float = 0.0
    long_range_direction_forward: float = 0.0
    long_range_direction_right: float = 0.0
    desired_velocity_x: float = 0.0
    desired_velocity_y: float = 0.0
    requested_force_x: float = 0.0
    requested_force_y: float = 0.0
    confidence: float = 0.0
    weight_separation: float = 0.0
    weight_alignment: float = 0.0
    weight_cohesion: float = 0.0
    weight_engagement: float = 0.0
    weight_panic_attenuation: float = 1.0
    separation_velocity_x: float = 0.0
    separation_velocity_y: float = 0.0
    alignment_velocity_x: float = 0.0
    alignment_velocity_y: float = 0.0
    cohesion_velocity_x: float = 0.0
    cohesion_velocity_y: float = 0.0
    influence: float = 0.0
    requested_contribution_x: float = 0.0
    requested_contribution_y: float = 0.0

    @property
    def observation(self) -> SocialObservation:
        """Execute observation behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
SocialObservation
    Result produced by this creature-domain operation."""
        # Keep observation behavior explicit in its owning subsystem.
        return SocialObservation(
            present=self.observation_present,
            visible_creature_count=self.visible_creature_count,
            compatible_visible_count=self.compatible_visible_count,
            personal_space_presence=self.personal_space_presence,
            social_presence=self.social_presence,
            effective_count=self.effective_count,
            center_forward=self.center_forward,
            center_right=self.center_right,
            relative_velocity_forward=self.relative_velocity_forward,
            relative_velocity_right=self.relative_velocity_right,
            separation_forward=self.separation_forward,
            separation_right=self.separation_right,
            mean_proximity=self.mean_proximity,
            center_distance=self.center_distance,
            mean_neighbor_distance=self.mean_neighbor_distance,
            mean_heading_error=self.mean_heading_error,
            mean_group_velocity=(
                self.mean_group_velocity_x,
                self.mean_group_velocity_y,
            ),
            long_range=LongRangeSocialObservation(
                intensity=self.long_range_intensity,
                direction_forward=self.long_range_direction_forward,
                direction_right=self.long_range_direction_right,
            ),
        )

    @property
    def intent(self) -> SocialIntent:
        """Execute intent behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
SocialIntent
    Result produced by this creature-domain operation."""
        # Keep intent behavior explicit in its owning subsystem.
        return SocialIntent(
            desired_velocity=(self.desired_velocity_x, self.desired_velocity_y),
            requested_force=(self.requested_force_x, self.requested_force_y),
            confidence=self.confidence,
            weights=FlockingWeights(
                separation=self.weight_separation,
                alignment=self.weight_alignment,
                cohesion=self.weight_cohesion,
                engagement=self.weight_engagement,
                panic_attenuation=self.weight_panic_attenuation,
            ),
            separation_velocity=(
                self.separation_velocity_x,
                self.separation_velocity_y,
            ),
            alignment_velocity=(
                self.alignment_velocity_x,
                self.alignment_velocity_y,
            ),
            cohesion_velocity=(
                self.cohesion_velocity_x,
                self.cohesion_velocity_y,
            ),
        )

    @property
    def requested_contribution(self) -> Vector:
        """Execute requested contribution behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
Vector
    Result produced by this creature-domain operation."""
        # Keep requested contribution behavior explicit in its owning subsystem.
        return self.requested_contribution_x, self.requested_contribution_y

    def update(
        self,
        intent: SocialIntent,
        observation: SocialObservation,
        influence: float,
        requested_contribution: Vector,
    ) -> None:
        """Execute update behavior.

Parameters
----------
intent
    Input used by this creature-domain operation.
observation
    Input used by this creature-domain operation.
influence
    Input used by this creature-domain operation.
requested_contribution
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep update behavior explicit in its owning subsystem.
        weights = intent.weights
        long_range = observation.long_range
        self.observation_present = bool(observation.present)
        self.visible_creature_count = int(observation.visible_creature_count)
        self.compatible_visible_count = int(
            observation.compatible_visible_count
        )
        self.personal_space_presence = float(
            observation.personal_space_presence
        )
        self.social_presence = float(observation.social_presence)
        self.effective_count = float(observation.effective_count)
        self.center_forward = float(observation.center_forward)
        self.center_right = float(observation.center_right)
        self.relative_velocity_forward = float(
            observation.relative_velocity_forward
        )
        self.relative_velocity_right = float(
            observation.relative_velocity_right
        )
        self.separation_forward = float(observation.separation_forward)
        self.separation_right = float(observation.separation_right)
        self.mean_proximity = float(observation.mean_proximity)
        self.center_distance = float(observation.center_distance)
        self.mean_neighbor_distance = float(observation.mean_neighbor_distance)
        self.mean_heading_error = float(observation.mean_heading_error)
        self.mean_group_velocity_x = float(observation.mean_group_velocity[0])
        self.mean_group_velocity_y = float(observation.mean_group_velocity[1])
        self.long_range_intensity = float(long_range.intensity)
        self.long_range_direction_forward = float(long_range.direction_forward)
        self.long_range_direction_right = float(long_range.direction_right)
        self.desired_velocity_x = float(intent.desired_velocity[0])
        self.desired_velocity_y = float(intent.desired_velocity[1])
        self.requested_force_x = float(intent.requested_force[0])
        self.requested_force_y = float(intent.requested_force[1])
        self.confidence = float(intent.confidence)
        self.weight_separation = float(weights.separation)
        self.weight_alignment = float(weights.alignment)
        self.weight_cohesion = float(weights.cohesion)
        self.weight_engagement = float(weights.engagement)
        self.weight_panic_attenuation = float(weights.panic_attenuation)
        self.separation_velocity_x = float(intent.separation_velocity[0])
        self.separation_velocity_y = float(intent.separation_velocity[1])
        self.alignment_velocity_x = float(intent.alignment_velocity[0])
        self.alignment_velocity_y = float(intent.alignment_velocity[1])
        self.cohesion_velocity_x = float(intent.cohesion_velocity[0])
        self.cohesion_velocity_y = float(intent.cohesion_velocity[1])
        self.influence = float(influence)
        self.requested_contribution_x = float(requested_contribution[0])
        self.requested_contribution_y = float(requested_contribution[1])

    def __iter__(self):
        """Execute iter behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep iter behavior explicit in its owning subsystem.
        yield self.intent
        yield self.observation
        yield self.influence
        yield self.requested_contribution

    def __getitem__(self, index: int):
        """Execute getitem behavior.

Parameters
----------
index
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep getitem behavior explicit in its owning subsystem.
        return tuple(self)[index]

    @classmethod
    def from_legacy(cls, value: object) -> SocialRuntime:
        """Execute from legacy behavior.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
SocialRuntime
    Result produced by this creature-domain operation."""
        # Keep from legacy behavior explicit in its owning subsystem.
        if isinstance(value, cls):
            runtime = cls()
            runtime.update(
                value.intent,
                value.observation,
                value.influence,
                value.requested_contribution,
            )
            return runtime
        if isinstance(value, (tuple, list)) and len(value) == 4:
            intent, observation, influence, contribution = value
            if isinstance(intent, SocialIntent) and isinstance(
                observation, SocialObservation
            ):
                runtime = cls()
                runtime.update(intent, observation, influence, contribution)
                return runtime
        return cls()

    def to_primitive(self) -> dict[str, object]:
        """Execute to primitive behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
dict[str, object]
    Result produced by this creature-domain operation."""
        # Keep to primitive behavior explicit in its owning subsystem.
        return {
            "observation": {
                "present": self.observation_present,
                "visible_creature_count": self.visible_creature_count,
                "compatible_visible_count": self.compatible_visible_count,
                "personal_space_presence": self.personal_space_presence,
                "social_presence": self.social_presence,
                "effective_count": self.effective_count,
                "center_forward": self.center_forward,
                "center_right": self.center_right,
                "relative_velocity_forward": self.relative_velocity_forward,
                "relative_velocity_right": self.relative_velocity_right,
                "separation_forward": self.separation_forward,
                "separation_right": self.separation_right,
                "mean_proximity": self.mean_proximity,
                "center_distance": self.center_distance,
                "mean_neighbor_distance": self.mean_neighbor_distance,
                "mean_heading_error": self.mean_heading_error,
                "mean_group_velocity": (
                    self.mean_group_velocity_x,
                    self.mean_group_velocity_y,
                ),
                "long_range": {
                    "intensity": self.long_range_intensity,
                    "direction_forward": self.long_range_direction_forward,
                    "direction_right": self.long_range_direction_right,
                },
            },
            "intent": {
                "desired_velocity": (
                    self.desired_velocity_x,
                    self.desired_velocity_y,
                ),
                "requested_force": (
                    self.requested_force_x,
                    self.requested_force_y,
                ),
                "confidence": self.confidence,
                "weights": {
                    "separation": self.weight_separation,
                    "alignment": self.weight_alignment,
                    "cohesion": self.weight_cohesion,
                    "engagement": self.weight_engagement,
                    "panic_attenuation": self.weight_panic_attenuation,
                },
                "separation_velocity": (
                    self.separation_velocity_x,
                    self.separation_velocity_y,
                ),
                "alignment_velocity": (
                    self.alignment_velocity_x,
                    self.alignment_velocity_y,
                ),
                "cohesion_velocity": (
                    self.cohesion_velocity_x,
                    self.cohesion_velocity_y,
                ),
            },
            "influence": self.influence,
            "requested_contribution": (
                self.requested_contribution_x,
                self.requested_contribution_y,
            ),
        }

    @classmethod
    def from_primitive(cls, state: object) -> SocialRuntime:
        """Execute from primitive behavior.

Parameters
----------
state
    Input used by this creature-domain operation.
Returns
-------
SocialRuntime
    Result produced by this creature-domain operation."""
        # Keep from primitive behavior explicit in its owning subsystem.
        if not isinstance(state, dict):
            return cls.from_legacy(state)
        observation_state = state.get("observation", {})
        intent_state = state.get("intent", {})
        if not isinstance(observation_state, dict) or not isinstance(
            intent_state, dict
        ):
            return cls()
        long_state = observation_state.get("long_range", {})
        weight_state = intent_state.get("weights", {})
        long_state = long_state if isinstance(long_state, dict) else {}
        weight_state = weight_state if isinstance(weight_state, dict) else {}
        vector = lambda value: (
            float(value[0]),
            float(value[1]),
        ) if isinstance(value, (tuple, list)) and len(value) == 2 else ZERO_VECTOR
        observation = SocialObservation(
            present=bool(observation_state.get("present", False)),
            visible_creature_count=int(
                observation_state.get("visible_creature_count", 0)
            ),
            compatible_visible_count=int(
                observation_state.get("compatible_visible_count", 0)
            ),
            personal_space_presence=float(
                observation_state.get("personal_space_presence", 0.0)
            ),
            social_presence=float(observation_state.get("social_presence", 0.0)),
            effective_count=float(observation_state.get("effective_count", 0.0)),
            center_forward=float(observation_state.get("center_forward", 0.0)),
            center_right=float(observation_state.get("center_right", 0.0)),
            relative_velocity_forward=float(
                observation_state.get("relative_velocity_forward", 0.0)
            ),
            relative_velocity_right=float(
                observation_state.get("relative_velocity_right", 0.0)
            ),
            separation_forward=float(
                observation_state.get("separation_forward", 0.0)
            ),
            separation_right=float(
                observation_state.get("separation_right", 0.0)
            ),
            mean_proximity=float(observation_state.get("mean_proximity", 0.0)),
            center_distance=float(observation_state.get("center_distance", 0.0)),
            mean_neighbor_distance=float(
                observation_state.get("mean_neighbor_distance", 0.0)
            ),
            mean_heading_error=float(
                observation_state.get("mean_heading_error", 0.0)
            ),
            mean_group_velocity=vector(
                observation_state.get("mean_group_velocity", ZERO_VECTOR)
            ),
            long_range=LongRangeSocialObservation(
                intensity=float(long_state.get("intensity", 0.0)),
                direction_forward=float(long_state.get("direction_forward", 0.0)),
                direction_right=float(long_state.get("direction_right", 0.0)),
            ),
        )
        intent = SocialIntent(
            desired_velocity=vector(
                intent_state.get("desired_velocity", ZERO_VECTOR)
            ),
            requested_force=vector(intent_state.get("requested_force", ZERO_VECTOR)),
            confidence=float(intent_state.get("confidence", 0.0)),
            weights=FlockingWeights(
                separation=float(weight_state.get("separation", 0.0)),
                alignment=float(weight_state.get("alignment", 0.0)),
                cohesion=float(weight_state.get("cohesion", 0.0)),
                engagement=float(weight_state.get("engagement", 0.0)),
                panic_attenuation=float(
                    weight_state.get("panic_attenuation", 1.0)
                ),
            ),
            separation_velocity=vector(
                intent_state.get("separation_velocity", ZERO_VECTOR)
            ),
            alignment_velocity=vector(
                intent_state.get("alignment_velocity", ZERO_VECTOR)
            ),
            cohesion_velocity=vector(
                intent_state.get("cohesion_velocity", ZERO_VECTOR)
            ),
        )
        runtime = cls()
        runtime.update(
            intent,
            observation,
            float(state.get("influence", 0.0)),
            vector(state.get("requested_contribution", ZERO_VECTOR)),
        )
        return runtime


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
    """Execute calculate flocking weights behavior.

Parameters
----------
herding
    Input used by this creature-domain operation.
panic
    Input used by this creature-domain operation.
separation_gene
    Input used by this creature-domain operation.
alignment_gene
    Input used by this creature-domain operation.
cohesion_gene
    Input used by this creature-domain operation.
personal_space_presence
    Input used by this creature-domain operation.
social_presence
    Input used by this creature-domain operation.
minimum_social_engagement
    Input used by this creature-domain operation.
panic_suppression_strength
    Input used by this creature-domain operation.
Returns
-------
FlockingWeights
    Result produced by this creature-domain operation."""
    # Keep calculate flocking weights behavior explicit in its owning subsystem.
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
    """Execute calculate social intent behavior.

Parameters
----------
current_velocity
    Input used by this creature-domain operation.
separation_velocity
    Input used by this creature-domain operation.
alignment_velocity
    Input used by this creature-domain operation.
cohesion_velocity
    Input used by this creature-domain operation.
weights
    Input used by this creature-domain operation.
effective_count
    Input used by this creature-domain operation.
target_group_size
    Input used by this creature-domain operation.
max_speed
    Input used by this creature-domain operation.
Returns
-------
SocialIntent
    Result produced by this creature-domain operation."""
    # Keep calculate social intent behavior explicit in its owning subsystem.
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
    """Execute blend desired velocity behavior.

Parameters
----------
neural_desired_velocity
    Input used by this creature-domain operation.
social_desired_velocity
    Input used by this creature-domain operation.
social_influence
    Input used by this creature-domain operation.
Returns
-------
Vector
    Result produced by this creature-domain operation."""
    # Keep blend desired velocity behavior explicit in its owning subsystem.
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
    """Execute configured social influence behavior.

Parameters
----------
config
    Input used by this creature-domain operation.
intent
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
    # Keep configured social influence behavior explicit in its owning subsystem.
    return _clamp(
        config.max_social_influence * intent.confidence,
        0.0,
        config.max_social_influence,
    )


def remove_opposing_component(vector: Vector, avoidance: Vector) -> Vector:
    """Execute remove opposing component behavior.

Parameters
----------
vector
    Input used by this creature-domain operation.
avoidance
    Input used by this creature-domain operation.
Returns
-------
Vector
    Result produced by this creature-domain operation."""
    # Keep remove opposing component behavior explicit in its owning subsystem.
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
    """Execute allocate force budget behavior.

Parameters
----------
requested
    Input used by this creature-domain operation.
remaining_budget
    Input used by this creature-domain operation.
Returns
-------
tuple[Vector, float]
    Result produced by this creature-domain operation."""
    # Keep allocate force budget behavior explicit in its owning subsystem.
    accepted = _limit(requested, remaining_budget)
    return accepted, max(0.0, remaining_budget - hypot(*accepted))


def accepted_counterfactual_contribution(
    *,
    blended_request: Vector,
    neural_request: Vector,
    mandatory_avoidance: Vector,
    remaining_budget: float,
) -> tuple[Vector, Vector, Vector]:
    """Execute accepted counterfactual contribution behavior.

Parameters
----------
blended_request
    Input used by this creature-domain operation.
neural_request
    Input used by this creature-domain operation.
mandatory_avoidance
    Input used by this creature-domain operation.
remaining_budget
    Input used by this creature-domain operation.
Returns
-------
tuple[Vector, Vector, Vector]
    Result produced by this creature-domain operation."""
    # Keep accepted counterfactual contribution behavior explicit in its owning subsystem.
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
