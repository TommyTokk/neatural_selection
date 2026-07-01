from __future__ import annotations

from dataclasses import dataclass

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
