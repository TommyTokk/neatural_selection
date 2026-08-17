"""Construction of live creature physics entities."""

from __future__ import annotations

from random import Random

import pymunk

from configs.sim_config import SimConfig
from src.collision import BOUNDARY_CATEGORY, CREATURE_CATEGORY, FOOD_CATEGORY
from src.creature.genotype import (
    Color,
    CreatureGenotype,
    FlockingTraits,
    GenotypeManager,
    LineageInfo,
    PhysicalTraits,
    VisionTraits,
)
from src.creature.model import Creature


class CreatureFactory:
    """Build creature bodies and entities without registering world identity."""

    def __init__(
        self,
        config: SimConfig,
        space: pymunk.Space,
        genotype_manager: GenotypeManager,
    ) -> None:
        """Initialize construction dependencies.

Parameters
----------
config
    Simulation configuration controlling initial runtime values.
space
    Physics space that will own created bodies and shapes.
genotype_manager
    Non-neural genotype creation service.

Returns
-------
None
    Dependencies are retained for subsequent construction."""
        # Keep init behavior explicit in its owning subsystem.
        # Construction owns physics insertion but never world registration.
        self.config = config
        self.space = space
        self.genotype_manager = genotype_manager

    def create(
        self,
        creature_id: int,
        world_bounds: tuple[float, float, float, float],
        rng: Random,
        *,
        position: tuple[float, float] | None = None,
        heading: float | None = None,
        energy: float | None = None,
        life: float | None = None,
        genotype: CreatureGenotype | None = None,
        color: Color | None = None,
        vision: VisionTraits | None = None,
        physical_traits: PhysicalTraits | None = None,
        flocking_traits: FlockingTraits | None = None,
        lineage: LineageInfo | None = None,
    ) -> Creature:
        """Create and add one creature body and shape to the physics space.

Parameters
----------
creature_id
    Stable creature identity.
world_bounds
    Environment ``(left, bottom, right, top)`` bounds.
rng
    Authoritative simulation random generator.
position
    Optional explicit body position.
heading
    Optional explicit heading in radians.
energy
    Optional initial energy override.
life
    Optional initial life override.
genotype
    Optional complete non-neural genotype.
color
    Legacy colour override when ``genotype`` is absent.
vision
    Legacy visual-trait override when ``genotype`` is absent.
physical_traits
    Legacy physical-trait override when ``genotype`` is absent.
flocking_traits
    Legacy flocking-trait override when ``genotype`` is absent.
lineage
    Optional ancestry metadata.

Returns
-------
Creature
    Constructed live creature already inserted into the physics space.

Raises
------
ValueError
    If aggregate and legacy genotype inputs are mixed."""
        # Keep create behavior explicit in its owning subsystem.
        # Genotype and legacy overrides are intentionally mutually exclusive.
        legacy = (color, vision, physical_traits, flocking_traits)
        if genotype is not None and any(value is not None for value in legacy):
            raise ValueError("Pass either genotype or legacy trait fields, not both.")

        # Preserve historical RNG ordering for default physical and social traits.
        if genotype is None:
            physical_traits = physical_traits or self.genotype_manager.initial_physical_traits(rng)
            flocking_traits = flocking_traits or self.genotype_manager.initial_flocking_traits(rng)
            radius = physical_traits.radius
        else:
            radius = genotype.physical_traits.radius

        left, bottom, right, top = world_bounds
        margin = radius + 10.0
        body = pymunk.Body(1.0, pymunk.moment_for_circle(1.0, 0.0, radius))
        if position is None:
            body.position = (
                rng.uniform(left + margin, right - margin),
                rng.uniform(bottom + margin, top - margin),
            )
        else:
            body.position = position
        body.angle = rng.uniform(0.0, 6.283185307179586) if heading is None else heading
        body.velocity = (0.0, 0.0)

        # Collision settings match the previous in-World construction exactly.
        shape = pymunk.Circle(body, radius)
        shape.filter = pymunk.ShapeFilter(
            categories=CREATURE_CATEGORY,
            mask=CREATURE_CATEGORY | FOOD_CATEGORY | BOUNDARY_CATEGORY,
        )
        shape.elasticity = 0.15
        shape.friction = 0.0
        self.space.add(body, shape)

        # Vision was historically sampled after body placement and heading.
        if genotype is None:
            vision = vision or self.genotype_manager.initial_vision(rng)
            genotype = CreatureGenotype(
                vision=vision,
                physical_traits=physical_traits,
                flocking_traits=flocking_traits,
                color=(
                    color
                    if color is not None
                    else self.genotype_manager.initial_color(creature_id - 1)
                ),
            )

        return Creature(
            creature_id=creature_id,
            name=f"Herbivore {creature_id:02d}",
            body=body,
            shape=shape,
            energy=rng.uniform(0.55, 0.95) if energy is None else energy,
            life=(
                self.config.metabolism.max_life
                * self.config.metabolism.initial_life_fraction
                if life is None
                else max(0.0, min(self.config.metabolism.max_life, life))
            ),
            genotype=genotype,
            lineage=lineage,
        )
