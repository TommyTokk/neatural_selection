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


@dataclass(frozen=True, slots=True)
class FlockingTraits:
    separation_gene: float = 0.5
    alignment_gene: float = 0.5
    cohesion_gene: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "separation_gene",
            "alignment_gene",
            "cohesion_gene",
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
    separation_gene: float = 0.0
    alignment_gene: float = 0.0
    cohesion_gene: float = 0.0


@dataclass(slots=True)
class LineageInfo:
    parent_id: int | None = None
    generation: int = 0
    species_id: int = 1
    mutation_delta: TraitMutationDelta = field(default_factory=TraitMutationDelta)


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
    lineage: LineageInfo = field(default_factory=LineageInfo)
    render_sprite: object | None = None
    last_action: object | None = None
    smoothed_rotation: float = 0.0
    smoothed_acceleration: float = 0.0
    biome_fertility_ema: float = 0.0
    biome_fertility_ema_updated_at: float = 0.0

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
