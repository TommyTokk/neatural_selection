"""Core live creature entity and runtime diagnostic records."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist

import pymunk

from src.creature.genotype import (
    Color,
    CreatureGenotype,
    FlockingTraits,
    LineageInfo,
    PhysicalTraits,
    VisionTraits,
)


@dataclass(slots=True)
class ActivityDiagnostics:
    """Expose normalized activity terms used by resource accounting."""

    voluntary_motor_effort: float = 0.0
    normalized_speed: float = 0.0
    turn: float = 0.0
    communication: float = 0.0
    reproduction: float = 0.0
    nursing: float = 0.0
    weighted_total: float = 0.0


@dataclass(slots=True)
class LedgerDiagnostics:
    """Expose the selected creature's latest resource transaction."""

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


@dataclass(slots=True, init=False)
class Creature:
    """Represent one live physical creature with an explicit genotype."""

    creature_id: int
    name: str
    body: pymunk.Body
    shape: pymunk.Circle
    energy: float
    genotype: CreatureGenotype
    stomach_energy: float
    stomach_difficulty_load: float
    total_energy_gathered: float
    lineage: LineageInfo
    render_sprite: object | None
    last_action: object | None
    smoothed_rotation: float
    smoothed_acceleration: float
    life: float
    rest_intent: float
    smoothed_rest: float
    effective_rest: float
    activity: float
    pending_direct_life_damage: float
    effective_voluntary_motor_effort: float
    ledger_diagnostics: LedgerDiagnostics

    def __init__(
        self,
        creature_id: int,
        name: str,
        body: pymunk.Body,
        shape: pymunk.Circle,
        energy: float,
        vision: VisionTraits | None = None,
        physical_traits: PhysicalTraits | None = None,
        color: Color | None = None,
        flocking_traits: FlockingTraits | None = None,
        *,
        genotype: CreatureGenotype | None = None,
        stomach_energy: float = 0.0,
        stomach_difficulty_load: float = 0.0,
        total_energy_gathered: float = 0.0,
        lineage: LineageInfo | None = None,
        render_sprite: object | None = None,
        last_action: object | None = None,
        smoothed_rotation: float = 0.0,
        smoothed_acceleration: float = 0.0,
        life: float = 1.0,
        rest_intent: float = 0.0,
        smoothed_rest: float = 0.0,
        effective_rest: float = 0.0,
        activity: float = 0.0,
        pending_direct_life_damage: float = 0.0,
        effective_voluntary_motor_effort: float = 0.0,
        ledger_diagnostics: LedgerDiagnostics | None = None,
    ) -> None:
        """Initialize from an aggregate genotype or legacy flat traits.

Parameters
----------
creature_id
    Stable simulation identity.
name
    Human-readable creature label.
body
    Pymunk rigid body.
shape
    Pymunk circular collision shape.
energy
    Initial usable energy.
vision
    Legacy visual traits when ``genotype`` is absent.
physical_traits
    Legacy physical traits when ``genotype`` is absent.
color
    Legacy colour when ``genotype`` is absent.
flocking_traits
    Legacy flocking traits when ``genotype`` is absent.
genotype
    Complete non-neural genotype.
stomach_energy
    Initial stored digestive energy.
stomach_difficulty_load
    Initial weighted digestive load.
total_energy_gathered
    Lifetime gathered energy used by fitness.
lineage
    Ancestry and species metadata.
render_sprite
    Optional renderer-owned sprite cache.
last_action
    Most recently executed action.
smoothed_rotation
    Smoothed turning command.
smoothed_acceleration
    Smoothed acceleration command.
life
    Initial life reserve.
rest_intent
    Raw resting intent.
smoothed_rest
    Smoothed resting intent.
effective_rest
    Rest applied after activity.
activity
    Aggregate recent activity.
pending_direct_life_damage
    Damage queued for fixed-step commit.
effective_voluntary_motor_effort
    Physically powered movement effort.
ledger_diagnostics
    Optional existing resource diagnostics.

Returns
-------
None
    The creature is initialized in place.

Raises
------
ValueError
    If aggregate and legacy genotype inputs are mixed or incomplete."""
        # Keep init behavior explicit in its owning subsystem.
        # Enforce one genotype source while preserving the historical constructor.
        legacy = (vision, physical_traits, color, flocking_traits)
        if genotype is not None and any(value is not None for value in legacy):
            raise ValueError("Pass either genotype or legacy trait fields, not both.")
        if genotype is None:
            if vision is None or physical_traits is None or color is None:
                raise ValueError(
                    "vision, physical_traits, and color are required without genotype."
                )
            genotype = CreatureGenotype(
                vision,
                physical_traits,
                flocking_traits or FlockingTraits(),
                color,
            )

        # Runtime values remain independent from inherited genotype state.
        self.creature_id = creature_id
        self.name = name
        self.body = body
        self.shape = shape
        self.energy = energy
        self.genotype = genotype
        self.stomach_energy = stomach_energy
        self.stomach_difficulty_load = stomach_difficulty_load
        self.total_energy_gathered = total_energy_gathered
        self.lineage = lineage or LineageInfo()
        self.render_sprite = render_sprite
        self.last_action = last_action
        self.smoothed_rotation = smoothed_rotation
        self.smoothed_acceleration = smoothed_acceleration
        self.life = life
        self.rest_intent = rest_intent
        self.smoothed_rest = smoothed_rest
        self.effective_rest = effective_rest
        self.activity = activity
        self.pending_direct_life_damage = pending_direct_life_damage
        self.effective_voluntary_motor_effort = effective_voluntary_motor_effort
        self.ledger_diagnostics = ledger_diagnostics or LedgerDiagnostics()

    @property
    def vision(self) -> VisionTraits:
        """Return visual traits from the aggregate genotype.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
VisionTraits
    Live inherited visual traits."""
        # Keep vision behavior explicit in its owning subsystem.
        # Preserve the established flat access path.
        return self.genotype.vision

    @vision.setter
    def vision(self, value: VisionTraits) -> None:
        """Replace visual traits in the aggregate genotype.

Parameters
----------
value
    Replacement visual traits.

Returns
-------
None
    The genotype is updated in place."""
        # Keep vision behavior explicit in its owning subsystem.
        # Route compatibility writes to the aggregate source of truth.
        self.genotype.vision = value

    @property
    def physical_traits(self) -> PhysicalTraits:
        """Return physical traits from the aggregate genotype.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
PhysicalTraits
    Live inherited physical traits."""
        # Keep physical traits behavior explicit in its owning subsystem.
        # Preserve the established hot-loop access path.
        return self.genotype.physical_traits

    @physical_traits.setter
    def physical_traits(self, value: PhysicalTraits) -> None:
        """Replace physical traits in the aggregate genotype.

Parameters
----------
value
    Replacement physical traits.

Returns
-------
None
    The genotype is updated in place."""
        # Keep physical traits behavior explicit in its owning subsystem.
        # Route compatibility writes to the aggregate source of truth.
        self.genotype.physical_traits = value

    @property
    def flocking_traits(self) -> FlockingTraits:
        """Return flocking traits from the aggregate genotype.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
FlockingTraits
    Live inherited social traits."""
        # Keep flocking traits behavior explicit in its owning subsystem.
        # Preserve the established social-system access path.
        return self.genotype.flocking_traits

    @flocking_traits.setter
    def flocking_traits(self, value: FlockingTraits) -> None:
        """Replace flocking traits in the aggregate genotype.

Parameters
----------
value
    Replacement flocking traits.

Returns
-------
None
    The genotype is updated in place."""
        # Keep flocking traits behavior explicit in its owning subsystem.
        # Route compatibility writes to the aggregate source of truth.
        self.genotype.flocking_traits = value

    @property
    def color(self) -> Color:
        """Return colour from the aggregate genotype.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
Color
    Live inherited RGB or RGBA colour."""
        # Keep color behavior explicit in its owning subsystem.
        # Colour stays writable for new-species founder assignment.
        return self.genotype.color

    @color.setter
    def color(self, value: Color) -> None:
        """Replace colour in the aggregate genotype.

Parameters
----------
value
    Replacement RGB or RGBA colour.

Returns
-------
None
    The genotype is updated in place."""
        # Keep color behavior explicit in its owning subsystem.
        # Route compatibility writes to the aggregate source of truth.
        self.genotype.color = value

    @property
    def position(self) -> tuple[float, float]:
        """Return the body's current Cartesian position.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
tuple[float, float]
    Current ``(x, y)`` coordinates."""
        # Keep position behavior explicit in its owning subsystem.
        # Convert Pymunk's vector into a stable primitive tuple.
        return self.body.position.x, self.body.position.y

    @property
    def heading(self) -> float:
        """Return the body's heading in radians.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
float
    Current body angle."""
        # Keep heading behavior explicit in its owning subsystem.
        # Physics owns the authoritative heading.
        return self.body.angle

    @property
    def speed(self) -> float:
        """Return the magnitude of current velocity.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
float
    Current linear speed."""
        # Keep speed behavior explicit in its owning subsystem.
        # Pymunk computes magnitude without a tuple allocation.
        return self.body.velocity.length

    def contains_point(self, x: float, y: float) -> bool:
        """Return whether a point intersects the selectable body area.

Parameters
----------
x
    Point x-coordinate.
y
    Point y-coordinate.

Returns
-------
bool
    Whether the point is inside the padded radius."""
        # Keep contains point behavior explicit in its owning subsystem.
        # Preserve the four-pixel selection affordance.
        return dist(self.position, (x, y)) <= self.radius + 4.0

    @property
    def radius(self) -> float:
        """Return body radius from the aggregate genotype.

Parameters
----------
None
    This property receives no external parameters.

Returns
-------
float
    Inherited physical radius."""
        # Keep radius behavior explicit in its owning subsystem.
        # Keep the high-frequency convenience alias used by physics and rendering.
        return self.physical_traits.radius


# Keep historical pickle globals stable despite the package implementation split.
for _legacy_type in (ActivityDiagnostics, LedgerDiagnostics, Creature):
    _legacy_type.__module__ = "src.creature"
