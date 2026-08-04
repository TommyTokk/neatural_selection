from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

import pymunk


Color = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(slots=True)
class VisionTraits:
    range: float
    angle: float


@dataclass(slots=True)
class PhysicalTraits:
    radius: float
    movement_cost_multiplier: float = 1.0
    stomach_capacity: float = 1.6
    digestion_rate: float = 0.2
    digestion_efficiency: float = 0.9


@dataclass(frozen=True, slots=True)
class FlockingTraits:
    separation_gene: float = 0.5
    alignment_gene: float = 0.5
    cohesion_gene: float = 0.5
    social_tag_x: float = 0.5
    social_tag_y: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "separation_gene",
            "alignment_gene",
            "cohesion_gene",
            "social_tag_x",
            "social_tag_y",
        ):
            object.__setattr__(
                self,
                name,
                max(0.0, min(1.0, float(getattr(self, name)))),
            )


@dataclass(slots=True)
class TraitMutationDelta:
    vision_range: float = 0.0
    vision_angle: float = 0.0
    radius: float = 0.0
    movement_cost_multiplier: float = 0.0
    stomach_capacity: float = 0.0
    digestion_rate: float = 0.0
    digestion_efficiency: float = 0.0
    separation_gene: float = 0.0
    alignment_gene: float = 0.0
    cohesion_gene: float = 0.0
    social_tag_x: float = 0.0
    social_tag_y: float = 0.0


@dataclass(slots=True)
class LineageInfo:
    parent_id: int | None = None
    generation: int = 0
    species_id: int = 1
    mutation_delta: TraitMutationDelta = field(default_factory=TraitMutationDelta)


@dataclass(slots=True)
class ActivityDiagnostics:
    voluntary_motor_effort: float = 0.0
    normalized_speed: float = 0.0
    turn: float = 0.0
    communication: float = 0.0
    reproduction: float = 0.0
    nursing: float = 0.0
    weighted_total: float = 0.0


@dataclass(slots=True)
class LedgerDiagnostics:
    activity: ActivityDiagnostics = field(default_factory=ActivityDiagnostics)
    effective_efficiency: float = 0.0
    stomach_consumed: float = 0.0
    gross_energy: float = 0.0
    processing_cost: float = 0.0
    net_energy: float = 0.0
    rest_energy_recovered: float = 0.0
    healing_energy_spent: float = 0.0
    life_healed: float = 0.0
    total_energy_demand: float = 0.0
    powered_movement_energy_demand: float = 0.0
    unmet_energy_demand: float = 0.0
    unmet_other_energy_demand: float = 0.0
    unmet_powered_movement_demand: float = 0.0
    movement_life_penalty_multiplier: float = 1.0
    movement_life_damage: float = 0.0
    life_damage_from_deficit: float = 0.0
    direct_life_damage: float = 0.0
    final_energy: float = 0.0
    final_life: float = 0.0
    transaction_status: str = "not_evaluated"


@dataclass(slots=True)
class Creature:
    creature_id: int
    name: str
    body: pymunk.Body
    shape: pymunk.Circle
    energy: float
    vision: VisionTraits
    physical_traits: PhysicalTraits
    color: Color
    flocking_traits: FlockingTraits = field(default_factory=FlockingTraits)
    stomach_energy: float = 0.0
    stomach_difficulty_load: float = 0.0
    lineage: LineageInfo = field(default_factory=LineageInfo)
    render_sprite: object | None = None
    last_action: object | None = None
    smoothed_rotation: float = 0.0
    smoothed_acceleration: float = 0.0
    biome_fertility_ema: float = 0.0
    biome_fertility_ema_updated_at: float = 0.0
    life: float = 1.0
    rest_intent: float = 0.0
    smoothed_rest: float = 0.0
    effective_rest: float = 0.0
    activity: float = 0.0
    pending_direct_life_damage: float = 0.0
    effective_voluntary_motor_effort: float = 0.0
    ledger_diagnostics: LedgerDiagnostics = field(
        default_factory=LedgerDiagnostics
    )

    @property
    def position(self) -> tuple[float, float]:
        return self.body.position.x, self.body.position.y

    @property
    def heading(self) -> float:
        return self.body.angle

    @property
    def speed(self) -> float:
        return self.body.velocity.length

    def contains_point(self, x: float, y: float) -> bool:
        return dist(self.position, (x, y)) <= self.radius + 4.0

    @property
    def radius(self) -> float:
        return self.physical_traits.radius
