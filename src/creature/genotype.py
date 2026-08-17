"""Non-neural hereditary state and mutation services for creatures."""

from __future__ import annotations

from colorsys import hsv_to_rgb, rgb_to_hsv
import copy
from dataclasses import dataclass, field
from random import Random

from configs.sim_config import SimConfig, SocialCompatibilityMode
from src.creature.common import clamp

Color = tuple[int, int, int] | tuple[int, int, int, int]


@dataclass(slots=True)
class VisionTraits:
    """Store inherited visual range and field-of-view values."""

    range: float
    angle: float


@dataclass(slots=True)
class PhysicalTraits:
    """Store inherited body, movement, and digestive values."""

    radius: float
    movement_cost_multiplier: float = 1.0
    stomach_capacity: float = 1.6
    digestion_rate: float = 0.2
    digestion_efficiency: float = 0.9


@dataclass(frozen=True, slots=True)
class FlockingTraits:
    """Store bounded inherited social steering and identity values."""

    separation_gene: float = 0.5
    alignment_gene: float = 0.5
    cohesion_gene: float = 0.5
    social_tag_x: float = 0.5
    social_tag_y: float = 0.5

    def __post_init__(self) -> None:
        """Normalize every flocking value to the supported unit interval.

Parameters
----------
None
    This initializer receives no external parameters.

Returns
-------
None
    The frozen instance is normalized in place."""
        # Keep post init behavior explicit in its owning subsystem.
        # Normalize once so downstream hot loops can trust the value range.
        for name in (
            "separation_gene",
            "alignment_gene",
            "cohesion_gene",
            "social_tag_x",
            "social_tag_y",
        ):
            object.__setattr__(self, name, clamp(getattr(self, name), 0.0, 1.0))


@dataclass(slots=True)
class TraitMutationDelta:
    """Record the effective inherited change for every mutable trait."""

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
    """Describe ancestry and species membership independently of genotype."""

    parent_id: int | None = None
    generation: int = 0
    species_id: int = 1
    mutation_delta: TraitMutationDelta = field(default_factory=TraitMutationDelta)

    def snapshot(self) -> LineageInfo:
        """Return an independent lineage value for persistence or archives.

Parameters
----------
None
    This method receives no external parameters.

Returns
-------
LineageInfo
    Deep copy detached from live lineage state."""
        # Keep snapshot behavior explicit in its owning subsystem.
        # Deep copying automatically includes future mutation-delta fields.
        return copy.deepcopy(self)


@dataclass(slots=True)
class CreatureGenotype:
    """Aggregate every inherited non-neural creature characteristic."""

    vision: VisionTraits
    physical_traits: PhysicalTraits
    flocking_traits: FlockingTraits
    color: Color

    def snapshot(self) -> CreatureGenotype:
        """Return an independent genotype for staging or archival use.

Parameters
----------
None
    This method receives no external parameters.

Returns
-------
CreatureGenotype
    Deep copy detached from live mutable traits."""
        # Keep snapshot behavior explicit in its owning subsystem.
        # Centralize copying instead of repeating every trait field at call sites.
        return copy.deepcopy(self)


@dataclass(frozen=True, slots=True)
class GenotypeMutationResult:
    """Pair a mutated genotype with its effective bounded trait deltas."""

    genotype: CreatureGenotype
    mutation_delta: TraitMutationDelta


class GenotypeManager:
    """Create, mutate, colour, and snapshot non-neural creature genotypes."""

    def __init__(self, config: SimConfig, color_palette: tuple[Color, ...]) -> None:
        """Initialize genotype operations from simulation configuration.

Parameters
----------
config
    Complete simulation configuration.
color_palette
    Stable palette used for initial creatures.

Returns
-------
None
    Configuration references are retained.

Raises
------
ValueError
    If ``color_palette`` is empty."""
        # Keep init behavior explicit in its owning subsystem.
        # Palette selection uses modulo and therefore requires at least one entry.
        if not color_palette:
            raise ValueError("Creature color palette must not be empty.")
        self.config = config
        self.color_palette = color_palette

    def initial_color(self, index: int) -> Color:
        """Select a deterministic initial colour from the palette.

Parameters
----------
index
    Zero-based creature palette index.

Returns
-------
Color
    Palette colour selected with wraparound."""
        # Keep initial color behavior explicit in its owning subsystem.
        # Wraparound retains existing cohort and population colouring.
        return self.color_palette[index % len(self.color_palette)]

    def initial_physical_traits(self, rng: Random) -> PhysicalTraits:
        """Sample bounded initial physical traits in historical RNG order.

Parameters
----------
rng
    Authoritative simulation random generator.

Returns
-------
PhysicalTraits
    Newly sampled physical traits."""
        # Keep initial physical traits behavior explicit in its owning subsystem.
        # Keep this sequence stable because later world randomness depends on it.
        trait = self.config.trait
        values = (
            clamp(trait.default_radius + rng.gauss(0.0, trait.initial_radius_jitter), trait.min_radius, trait.max_radius),
            clamp(trait.default_movement_cost_multiplier + rng.gauss(0.0, trait.initial_movement_cost_jitter), trait.min_movement_cost_multiplier, trait.max_movement_cost_multiplier),
            clamp(trait.default_stomach_capacity + rng.gauss(0.0, trait.initial_stomach_capacity_jitter), trait.min_stomach_capacity, trait.max_stomach_capacity),
            clamp(trait.default_digestion_rate + rng.gauss(0.0, trait.initial_digestion_rate_jitter), trait.min_digestion_rate, trait.max_digestion_rate),
            clamp(trait.default_digestion_efficiency + rng.gauss(0.0, trait.initial_digestion_efficiency_jitter), trait.min_digestion_efficiency, trait.max_digestion_efficiency),
        )
        return PhysicalTraits(*values)

    def initial_flocking_traits(self, rng: Random) -> FlockingTraits:
        """Sample bounded initial flocking genes in historical RNG order.

Parameters
----------
rng
    Authoritative simulation random generator.

Returns
-------
FlockingTraits
    Newly sampled social traits."""
        # Keep initial flocking traits behavior explicit in its owning subsystem.
        # Disabled social tags deliberately consume no random values.
        trait = self.config.trait
        tagged = self.config.flocking.compatibility.mode is SocialCompatibilityMode.SOCIAL_TAG
        return FlockingTraits(
            clamp(rng.gauss(trait.default_separation_gene, trait.initial_flocking_gene_stdev), 0.0, 1.0),
            clamp(rng.gauss(trait.default_alignment_gene, trait.initial_flocking_gene_stdev), 0.0, 1.0),
            clamp(rng.gauss(trait.default_cohesion_gene, trait.initial_flocking_gene_stdev), 0.0, 1.0),
            clamp(rng.gauss(trait.default_social_tag_x, trait.initial_social_tag_stdev), 0.0, 1.0) if tagged else trait.default_social_tag_x,
            clamp(rng.gauss(trait.default_social_tag_y, trait.initial_social_tag_stdev), 0.0, 1.0) if tagged else trait.default_social_tag_y,
        )

    def initial_vision(self, rng: Random) -> VisionTraits:
        """Sample initial visual traits in historical RNG order.

Parameters
----------
rng
    Authoritative simulation random generator.

Returns
-------
VisionTraits
    Newly sampled visual traits."""
        # Keep initial vision behavior explicit in its owning subsystem.
        # Range is sampled before angle to retain deterministic continuation.
        vision = self.config.vision
        return VisionTraits(
            rng.uniform(vision.min_range, vision.max_range),
            rng.uniform(vision.min_angle, vision.max_angle),
        )

    def mutate_vision(self, parent: VisionTraits, rng: Random) -> tuple[VisionTraits, TraitMutationDelta]:
        """Mutate visual traits and report effective bounded changes.

Parameters
----------
parent
    Parent visual traits.
rng
    Authoritative simulation random generator.

Returns
-------
tuple[VisionTraits, TraitMutationDelta]
    Child traits and their effective delta."""
        # Keep mutate vision behavior explicit in its owning subsystem.
        # Fixed powers preserve the established visual mutation distribution.
        child = VisionTraits(
            clamp(parent.range + rng.gauss(0.0, 8.0), self.config.vision.min_range, self.config.vision.max_range),
            clamp(parent.angle + rng.gauss(0.0, 0.08), self.config.vision.min_angle, self.config.vision.max_angle),
        )
        return child, TraitMutationDelta(
            vision_range=child.range - parent.range,
            vision_angle=child.angle - parent.angle,
        )

    def mutate_physical_traits(
        self,
        parent: PhysicalTraits,
        rng: Random,
    ) -> tuple[PhysicalTraits, TraitMutationDelta]:
        """Mutate physical and digestive traits within configured bounds.

Parameters
----------
parent
    Parent physical traits.
rng
    Authoritative simulation random generator.

Returns
-------
tuple[PhysicalTraits, TraitMutationDelta]
    Child traits and their effective bounded delta."""
        # Keep mutate physical traits behavior explicit in its owning subsystem.
        # Body mutations always precede the conditional digestive mutations.
        trait = self.config.trait
        radius = clamp(
            parent.radius + rng.gauss(0.0, trait.radius_mutation_stddev),
            trait.min_radius,
            trait.max_radius,
        )
        movement = clamp(
            parent.movement_cost_multiplier
            + rng.gauss(0.0, trait.movement_cost_mutation_stddev),
            trait.min_movement_cost_multiplier,
            trait.max_movement_cost_multiplier,
        )

        def mutate_digestive(
            value: float,
            standard_deviation: float,
            minimum: float,
            maximum: float,
        ) -> float:
            """Apply the configured probability gate to one digestive trait.

Parameters
----------
value
    Parent digestive value.
standard_deviation
    Gaussian mutation standard deviation.
minimum
    Inclusive lower bound.
maximum
    Inclusive upper bound.

Returns
-------
float
    Bounded inherited or mutated value."""
            # Keep mutate digestive behavior explicit in its owning subsystem.
            # Every digestive trait consumes exactly one mutation-gate roll.
            if rng.random() >= trait.digestive_trait_mutation_rate:
                return clamp(value, minimum, maximum)
            return clamp(
                value + rng.gauss(0.0, standard_deviation),
                minimum,
                maximum,
            )

        parent_capacity = getattr(
            parent,
            "stomach_capacity",
            trait.default_stomach_capacity,
        )
        parent_rate = getattr(parent, "digestion_rate", trait.default_digestion_rate)
        parent_efficiency = getattr(
            parent,
            "digestion_efficiency",
            trait.default_digestion_efficiency,
        )
        capacity = mutate_digestive(
            parent_capacity,
            trait.stomach_capacity_mutation_stddev,
            trait.min_stomach_capacity,
            trait.max_stomach_capacity,
        )
        rate = mutate_digestive(
            parent_rate,
            trait.digestion_rate_mutation_stddev,
            trait.min_digestion_rate,
            trait.max_digestion_rate,
        )
        efficiency = mutate_digestive(
            parent_efficiency,
            trait.digestion_efficiency_mutation_stddev,
            trait.min_digestion_efficiency,
            trait.max_digestion_efficiency,
        )
        child = PhysicalTraits(radius, movement, capacity, rate, efficiency)
        return child, TraitMutationDelta(
            radius=radius - parent.radius,
            movement_cost_multiplier=movement - parent.movement_cost_multiplier,
            stomach_capacity=capacity - parent_capacity,
            digestion_rate=rate - parent_rate,
            digestion_efficiency=efficiency - parent_efficiency,
        )

    def mutate_flocking_traits(
        self,
        parent: FlockingTraits,
        rng: Random,
    ) -> tuple[FlockingTraits, TraitMutationDelta]:
        """Mutate flocking genes and optional social tags.

Parameters
----------
parent
    Parent flocking traits.
rng
    Authoritative simulation random generator.

Returns
-------
tuple[FlockingTraits, TraitMutationDelta]
    Child traits and their effective bounded delta."""
        # Keep mutate flocking traits behavior explicit in its owning subsystem.
        # All steering genes share one canonical replace-or-perturb operation.
        trait = self.config.trait

        def mutate_gene(value: float) -> float:
            """Mutate one unit-interval steering gene.

Parameters
----------
value
    Parent gene value.

Returns
-------
float
    Bounded inherited or mutated value."""
            # Keep mutate gene behavior explicit in its owning subsystem.
            # One roll selects mutually exclusive replacement and perturbation.
            roll = rng.random()
            if roll < trait.flocking_gene_replace_rate:
                return rng.uniform(0.0, 1.0)
            if roll < trait.flocking_gene_replace_rate + trait.flocking_gene_mutation_rate:
                value += rng.gauss(0.0, trait.flocking_gene_mutation_power)
            return clamp(value, 0.0, 1.0)

        def mutate_tag(value: float) -> float:
            """Mutate one social tag only when tag mode is enabled.

Parameters
----------
value
    Parent tag value.

Returns
-------
float
    Bounded inherited or mutated tag."""
            # Keep mutate tag behavior explicit in its owning subsystem.
            # Disabled tags consume no RNG, preserving later mutation draws.
            if self.config.flocking.compatibility.mode is not SocialCompatibilityMode.SOCIAL_TAG:
                return value
            roll = rng.random()
            if roll < trait.social_tag_replace_rate:
                return rng.uniform(0.0, 1.0)
            if roll < trait.social_tag_replace_rate + trait.social_tag_mutation_rate:
                value += rng.gauss(0.0, trait.social_tag_mutation_power)
            return clamp(value, 0.0, 1.0)

        child = FlockingTraits(
            mutate_gene(parent.separation_gene),
            mutate_gene(parent.alignment_gene),
            mutate_gene(parent.cohesion_gene),
            mutate_tag(parent.social_tag_x),
            mutate_tag(parent.social_tag_y),
        )
        return child, TraitMutationDelta(
            separation_gene=child.separation_gene - parent.separation_gene,
            alignment_gene=child.alignment_gene - parent.alignment_gene,
            cohesion_gene=child.cohesion_gene - parent.cohesion_gene,
            social_tag_x=child.social_tag_x - parent.social_tag_x,
            social_tag_y=child.social_tag_y - parent.social_tag_y,
        )

    def mutate(
        self,
        parent: CreatureGenotype,
        rng: Random,
    ) -> GenotypeMutationResult:
        """Mutate all non-neural inherited characteristics of one parent.

Parameters
----------
parent
    Parent aggregate genotype.
rng
    Authoritative simulation random generator.

Returns
-------
GenotypeMutationResult
    Child genotype and combined effective mutation delta."""
        # Keep mutate behavior explicit in its owning subsystem.
        # Preserve the historical vision, physical, flocking, then colour order.
        vision, vision_delta = self.mutate_vision(parent.vision, rng)
        physical, physical_delta = self.mutate_physical_traits(
            parent.physical_traits,
            rng,
        )
        flocking, flocking_delta = self.mutate_flocking_traits(
            parent.flocking_traits,
            rng,
        )
        delta = TraitMutationDelta(
            vision_range=vision_delta.vision_range,
            vision_angle=vision_delta.vision_angle,
            radius=physical_delta.radius,
            movement_cost_multiplier=physical_delta.movement_cost_multiplier,
            stomach_capacity=physical_delta.stomach_capacity,
            digestion_rate=physical_delta.digestion_rate,
            digestion_efficiency=physical_delta.digestion_efficiency,
            separation_gene=flocking_delta.separation_gene,
            alignment_gene=flocking_delta.alignment_gene,
            cohesion_gene=flocking_delta.cohesion_gene,
            social_tag_x=flocking_delta.social_tag_x,
            social_tag_y=flocking_delta.social_tag_y,
        )
        return GenotypeMutationResult(
            CreatureGenotype(
                vision,
                physical,
                flocking,
                self.mutate_color(parent.color, rng),
            ),
            delta,
        )

    def mutate_color(self, parent: Color, rng: Random) -> Color:
        """Apply a small inherited HSV colour mutation.

Parameters
----------
parent
    Parent RGB or RGBA colour.
rng
    Authoritative simulation random generator.

Returns
-------
Color
    Mutated RGB colour that avoids the food-colour region."""
        # Keep mutate color behavior explicit in its owning subsystem.
        # Preserve the historical uniform draw sequence and HSV bounds exactly.
        red, green, blue = parent[:3]
        hue, saturation, value = rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
        hue = (hue + rng.uniform(-0.035, 0.035)) % 1.0
        saturation = clamp(saturation + rng.uniform(-0.06, 0.06), 0.48, 0.82)
        value = clamp(value + rng.uniform(-0.05, 0.05), 0.62, 0.92)
        candidate = hsv_to_rgb(hue, saturation, value)
        if self.is_food_like_color(candidate):
            hue = (hue + 0.22) % 1.0
        candidate = hsv_to_rgb(hue, saturation, value)
        return tuple(int(channel * 255) for channel in candidate)

    def new_species_color(self, parent: Color, rng: Random) -> Color:
        """Generate a bright colour separated from the parent species.

Parameters
----------
parent
    Parent RGB or RGBA colour.
rng
    Authoritative simulation random generator.

Returns
-------
Color
    Distinct RGB founder colour."""
        # Keep new species color behavior explicit in its owning subsystem.
        # Attempt the original randomized search before deterministic fallbacks.
        red, green, blue = parent[:3]
        parent_hue, _, _ = rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
        for _ in range(32):
            hue = (parent_hue + rng.uniform(0.18, 0.82)) % 1.0
            candidate = hsv_to_rgb(
                hue,
                rng.uniform(0.7, 1.0),
                rng.uniform(0.8, 1.0),
            )
            color = tuple(int(channel * 255) for channel in candidate)
            if not self.is_food_like_color(tuple(channel / 255.0 for channel in color)):
                return color
        for offset in (0.5, 1.0 / 3.0, 2.0 / 3.0):
            candidate = hsv_to_rgb((parent_hue + offset) % 1.0, 0.85, 0.9)
            color = tuple(int(channel * 255) for channel in candidate)
            if not self.is_food_like_color(tuple(channel / 255.0 for channel in color)):
                return color
        candidate = hsv_to_rgb((parent_hue + 0.5) % 1.0, 1.0, 1.0)
        return tuple(int(channel * 255) for channel in candidate)

    def is_food_like_color(self, color: tuple[float, float, float]) -> bool:
        """Return whether a normalized RGB colour resembles food.

Parameters
----------
color
    Normalized RGB colour.

Returns
-------
bool
    Whether green dominates enough to resemble food."""
        # Keep is food like color behavior explicit in its owning subsystem.
        # Compare against configured food colour using the historical RGB radius.
        food_red, food_green, food_blue = self.config.theme.food_fill[:3]
        red, green, blue = (channel * 255.0 for channel in color)
        distance_squared = (
            (red - food_red) ** 2
            + (green - food_green) ** 2
            + (blue - food_blue) ** 2
        )
        return distance_squared < 70.0**2


# Preserve version-23 pickle globals while implementation lives in this module.
for _legacy_type in (
    VisionTraits,
    PhysicalTraits,
    FlockingTraits,
    TraitMutationDelta,
    LineageInfo,
):
    _legacy_type.__module__ = "src.creature"
