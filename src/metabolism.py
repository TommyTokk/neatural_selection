from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from math import cos, isfinite, sin

from src.food import Food
from src.creature import Creature
from configs.sim_config import (
    CommunicationConfig,
    FoodConfig,
    MetabolismConfig,
    TraitConfig,
)
from src.vision import VisionSystem


ENERGY_EPSILON = 1e-9


def is_energy_depleted(value: float) -> bool:
    """Return whether energy is unavailable under the shared tolerance."""
    try:
        energy = float(value)
    except (TypeError, ValueError, OverflowError):
        return True
    return not isfinite(energy) or energy <= ENERGY_EPSILON


def movement_life_penalty_multiplier(
    life: float,
    max_life: float,
    maximum_multiplier: float,
) -> float:
    """Return the quadratic multiplier for life-powered movement."""
    maximum = max(1.0, float(maximum_multiplier))
    if max_life <= 0.0:
        return maximum
    life_ratio = min(1.0, max(0.0, float(life) / float(max_life)))
    return 1.0 + (maximum - 1.0) * (1.0 - life_ratio) ** 2


@dataclass(slots=True)
class FoodConsumption:
    creature_id: int
    food: Food
    energy_swallowed: float
    depleted: bool


@dataclass(slots=True)
class MetabolismReport:
    depleted_foods: list[Food] = field(default_factory=list)
    touched_foods: list[Food] = field(default_factory=list)
    food_consumptions: list[FoodConsumption] = field(default_factory=list)
    digested_energy_gained: dict[int, float] = field(default_factory=dict)
    digestion_processing_costs: dict[int, float] = field(default_factory=dict)
    dead_creatures: list[Creature] = field(default_factory=list)


@dataclass(slots=True)
class EnergyCostBreakdown:
    base: float
    movement: float
    vision: float
    body: float
    trait: float
    sprint: float = 0.0
    acoustic: float = 0.0
    pheromone: float = 0.0
    digestive_upkeep: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.base
            + self.movement
            + self.sprint
            + self.vision
            + self.body
            + self.acoustic
            + self.pheromone
            + self.digestive_upkeep
        )


@dataclass(frozen=True, slots=True)
class DigestionResult:
    stomach_consumed: float
    gross_energy: float
    processing_cost: float
    net_energy: float

    @property
    def raw_digested(self) -> float:
        """Compatibility alias for the consumed stomach quantity."""
        return self.stomach_consumed

    @property
    def net_energy_gained(self) -> float:
        """Compatibility alias for net digestive energy."""
        return self.net_energy


@dataclass(frozen=True, slots=True)
class ResourceCandidate:
    digestion: DigestionResult
    total_energy_demand: float
    final_stomach_energy: float
    final_stomach_difficulty_load: float
    available_energy: float
    remaining_energy: float
    unmet_energy_demand: float
    life_damage_from_deficit: float
    direct_life_damage: float
    final_energy: float
    final_life: float
    rest_energy_recovered: float = 0.0
    healing_energy_spent: float = 0.0
    life_healed: float = 0.0
    powered_movement_energy_demand: float = 0.0
    unmet_other_energy_demand: float = 0.0
    unmet_powered_movement_demand: float = 0.0
    movement_life_penalty_multiplier: float = 1.0
    movement_life_damage: float = 0.0

    @property
    def survives(self) -> bool:
        return self.final_life > 0.0


def calculate_digestion(
    *,
    stomach_contents: float,
    digestion_rate: float,
    delta_time: float,
    trait_efficiency: float,
    rest_digestion_efficiency_bonus: float,
    effective_rest: float,
    processing_fraction: float,
    total_energy_demand: float,
    max_energy: float,
    starting_energy: float,
    numerical_tolerance: float = 1e-12,
) -> DigestionResult:
    """Purely resolve stomach use and gross-to-net digestive conversion."""

    values = (
        stomach_contents,
        digestion_rate,
        delta_time,
        trait_efficiency,
        rest_digestion_efficiency_bonus,
        effective_rest,
        processing_fraction,
        total_energy_demand,
        max_energy,
        starting_energy,
    )
    try:
        parsed = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError):
        return DigestionResult(0.0, 0.0, 0.0, 0.0)
    if not all(isfinite(value) for value in parsed):
        return DigestionResult(0.0, 0.0, 0.0, 0.0)
    (
        stomach_contents,
        digestion_rate,
        delta_time,
        trait_efficiency,
        rest_digestion_efficiency_bonus,
        effective_rest,
        processing_fraction,
        total_energy_demand,
        max_energy,
        starting_energy,
    ) = parsed
    if not 0.0 <= processing_fraction <= 1.0:
        raise ValueError("processing_fraction must be within [0, 1].")

    maximum_stomach_process = min(
        max(0.0, stomach_contents),
        max(0.0, digestion_rate) * max(0.0, delta_time),
    )
    effective_efficiency = min(
        1.0,
        max(
            0.0,
            trait_efficiency
            + max(0.0, rest_digestion_efficiency_bonus)
            * min(1.0, max(0.0, effective_rest)),
        ),
    )
    net_per_stomach = effective_efficiency * (1.0 - processing_fraction)
    usable_net_limit = (
        max(0.0, total_energy_demand)
        + max(0.0, max_energy)
        - min(max(0.0, max_energy), max(0.0, starting_energy))
    )
    tolerance = max(0.0, numerical_tolerance)
    if (
        net_per_stomach <= tolerance
        or usable_net_limit <= tolerance
        or maximum_stomach_process <= tolerance
    ):
        stomach_consumed = 0.0
    else:
        stomach_consumed = min(
            maximum_stomach_process,
            usable_net_limit / net_per_stomach,
        )
    gross_energy = stomach_consumed * effective_efficiency
    processing_cost = gross_energy * processing_fraction
    net_energy = gross_energy - processing_cost
    return DigestionResult(
        stomach_consumed=stomach_consumed,
        gross_energy=gross_energy,
        processing_cost=processing_cost,
        net_energy=net_energy,
    )


@dataclass(frozen=True, slots=True)
class ActivityResult:
    voluntary_motor_effort: float
    normalized_speed: float
    turn: float
    communication: float
    reproduction: float
    nursing: float
    total: float


def calculate_weighted_activity(
    *,
    voluntary_motor_effort: float,
    normalized_speed: float,
    turn_command: float,
    normalized_angular_speed: float,
    communication_cost: float,
    reproduction_selected: bool,
    nursing_transfer: float,
) -> ActivityResult:
    """Return the bounded weighted activity used to inhibit effective rest."""

    def bounded(value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return 0.0 if not isfinite(numeric) else min(1.0, max(0.0, numeric))

    motor = bounded(voluntary_motor_effort)
    speed = bounded(normalized_speed)
    turn = 0.5 * (
        bounded(abs(turn_command)) + bounded(abs(normalized_angular_speed))
    )
    communication = bounded(communication_cost)
    reproduction = 1.0 if reproduction_selected else 0.0
    nursing = 1.0 if nursing_transfer > 0.0 else 0.0
    total = min(
        1.0,
        0.40 * motor
        + 0.10 * speed
        + 0.15 * turn
        + 0.15 * communication
        + 0.10 * reproduction
        + 0.10 * nursing,
    )
    if reproduction > 0.0 or nursing > 0.0:
        total = 1.0
    return ActivityResult(
        voluntary_motor_effort=motor,
        normalized_speed=speed,
        turn=turn,
        communication=communication,
        reproduction=reproduction,
        nursing=nursing,
        total=total,
    )


class Metabolism:
    def __init__(
        self,
        config: MetabolismConfig,
        vision: VisionSystem,
        trait_config: TraitConfig | None = None,
        genome_for_creature_id: Callable[[int], object | None] | None = None,
        communication_config: CommunicationConfig | None = None,
        food_config: FoodConfig | None = None,
    ) -> None:
        self.config = config
        self.vision = vision
        self.trait_config = trait_config or TraitConfig()
        self.genome_for_creature_id = genome_for_creature_id
        self.communication_config = communication_config or CommunicationConfig()
        self.food_config = food_config or FoodConfig()
        self._validate_config()

    def update(
        self,
        creatures: list[Creature],
        food_items: list[Food],
        delta_time: float,
        max_speed: float,
        nearby_foods_for: Callable[[Creature], Sequence[Food]] | None = None,
        can_eat: Callable[[Creature], bool] | None = None,
        sprint_intensities: dict[int, float] | None = None,
        energy_cost_multipliers: dict[int, float] | None = None,
        creature_age_seconds: dict[int, float] | None = None,
        communication_intensities: dict[int, tuple[float, float, float]] | None = None,
    ) -> MetabolismReport:
        depleted_foods: list[Food] = []
        touched_foods: list[Food] = []
        food_consumptions: list[FoodConsumption] = []
        dead_creatures: list[Creature] = []
        digested_energy_gained: dict[int, float] = {}
        digestion_processing_costs: dict[int, float] = {}

        for creature in creatures:
            sprint_intensity = (
                0.0
                if sprint_intensities is None
                else sprint_intensities.get(creature.creature_id, 0.0)
            )
            upkeep = (
                self.energy_cost_breakdown_per_second(
                    creature,
                    max_speed,
                    sprint_intensity=sprint_intensity,
                    age_seconds=(
                        None
                        if creature_age_seconds is None
                        else creature_age_seconds.get(creature.creature_id)
                    ),
                    communication_intensities=(
                        (0.0, 0.0, 0.0)
                        if communication_intensities is None
                        else communication_intensities.get(
                            creature.creature_id,
                            (0.0, 0.0, 0.0),
                        )
                    ),
                ).total
                * max(0.0, delta_time)
                * max(
                    0.0,
                    (
                        1.0
                        if energy_cost_multipliers is None
                        else energy_cost_multipliers.get(
                            creature.creature_id,
                            1.0,
                        )
                    ),
                )
            )
            candidate = self.evaluate_candidate(
                creature,
                delta_time,
                total_energy_demand=upkeep,
                effective_rest=getattr(creature, "effective_rest", 0.0),
            )
            self.commit_candidate(creature, candidate)
            digestion = candidate.digestion
            if digestion.net_energy_gained > 0.0:
                digested_energy_gained[creature.creature_id] = (
                    digestion.net_energy_gained
                )
            if digestion.processing_cost > 0.0:
                digestion_processing_costs[creature.creature_id] = (
                    digestion.processing_cost
                )

            if can_eat is not None and not can_eat(creature):
                if self.is_dead(creature):
                    dead_creatures.append(creature)
                continue

            # Calculate the eatable food only after passing the mechanical gate.
            candidate_foods = (
                food_items if nearby_foods_for is None else nearby_foods_for(creature)
            )
            food = self.find_eatable_food(creature, candidate_foods, touched_foods)

            if food is not None:
                consumption = self.eat(creature, food, delta_time)
                if consumption.energy_swallowed > 0.0 or consumption.depleted:
                    touched_foods.append(food)
                    if consumption.depleted:
                        depleted_foods.append(food)
                    food_consumptions.append(
                        FoodConsumption(
                            creature_id=creature.creature_id,
                            food=food,
                            energy_swallowed=consumption.energy_swallowed,
                            depleted=consumption.depleted,
                        )
                    )

            if self.is_dead(creature):
                dead_creatures.append(creature)

        return MetabolismReport(
            depleted_foods=depleted_foods,
            touched_foods=touched_foods,
            food_consumptions=food_consumptions,
            digested_energy_gained=digested_energy_gained,
            digestion_processing_costs=digestion_processing_costs,
            dead_creatures=dead_creatures,
        )

    def digest(self, creature: Creature, delta_time: float) -> float:
        return self._digest_with_ledger(
            creature,
            delta_time,
            upkeep_cost=0.0,
        ).net_energy_gained

    def _digest_with_ledger(
        self,
        creature: Creature,
        delta_time: float,
        *,
        upkeep_cost: float,
    ) -> DigestionResult:
        candidate = self.evaluate_candidate(
            creature,
            delta_time,
            total_energy_demand=max(0.0, upkeep_cost),
            effective_rest=getattr(creature, "effective_rest", 0.0),
        )
        self.commit_candidate(creature, candidate)
        return candidate.digestion

    def evaluate_candidate(
        self,
        creature: Creature,
        delta_time: float,
        *,
        total_energy_demand: float,
        powered_movement_energy_demand: float = 0.0,
        effective_rest: float = 0.0,
    ) -> ResourceCandidate:
        """Evaluate one complete resource ledger without mutating the creature."""
        stomach_energy = max(0.0, getattr(creature, "stomach_energy", 0.0))
        stomach_load = self.normalized_stomach_difficulty_load(
            creature,
            stomach_energy,
        )
        traits = getattr(creature, "physical_traits", None)
        digestion_rate = max(
            0.0,
            float(
                getattr(
                    traits,
                    "digestion_rate",
                    self.config.digestion_rate_per_second,
                )
            ),
        )
        digestion_efficiency = float(
            getattr(
                traits,
                "digestion_efficiency",
                self.config.digestion_efficiency,
            )
        )
        difficulty = (
            1.0
            if stomach_energy <= 0.0
            else stomach_load / stomach_energy
        )
        processing_fraction = self.digestion_processing_fraction(
            digestion_rate,
            difficulty,
        )
        demand = max(0.0, float(total_energy_demand))
        powered_movement_demand = min(
            demand,
            max(0.0, float(powered_movement_energy_demand)),
        )
        other_demand = demand - powered_movement_demand
        starting_energy = min(
            self.config.max_energy,
            max(0.0, creature.energy),
        )
        digestion = calculate_digestion(
            stomach_contents=stomach_energy,
            digestion_rate=digestion_rate,
            delta_time=delta_time,
            trait_efficiency=digestion_efficiency,
            rest_digestion_efficiency_bonus=(
                self.config.rest_digestion_efficiency_bonus
            ),
            effective_rest=effective_rest,
            processing_fraction=processing_fraction,
            total_energy_demand=demand,
            max_energy=self.config.max_energy,
            starting_energy=starting_energy,
        )
        remaining_stomach = max(
            0.0,
            stomach_energy - digestion.stomach_consumed,
        )
        remaining_load = max(
            0.0,
            stomach_load - digestion.stomach_consumed * difficulty,
        )
        if remaining_stomach <= 1e-12:
            remaining_stomach = 0.0
            remaining_load = 0.0
        energy_before_rest_recovery = min(
            self.config.max_energy,
            starting_energy + digestion.net_energy,
        )
        rest_strength = min(1.0, max(0.0, float(effective_rest)))
        recovery_limit = min(
            self.config.max_energy,
            max(0.0, self.config.starvation_energy_threshold),
        )
        rest_energy_recovered = min(
            max(0.0, self.config.rest_energy_recovery_per_second)
            * rest_strength
            * max(0.0, float(delta_time)),
            max(0.0, recovery_limit - energy_before_rest_recovery),
        )
        available_energy = energy_before_rest_recovery + rest_energy_recovered
        energy_after_other = max(0.0, available_energy - other_demand)
        unmet_other_demand = max(0.0, other_demand - available_energy)
        paid_powered_movement = min(
            powered_movement_demand,
            energy_after_other,
        )
        unmet_powered_movement = max(
            0.0,
            powered_movement_demand - paid_powered_movement,
        )
        remaining_energy = max(
            0.0,
            energy_after_other - paid_powered_movement,
        )
        unmet_energy_demand = (
            unmet_other_demand + unmet_powered_movement
        )
        life_damage_rate = max(
            0.0,
            self.config.life_damage_per_energy_deficit,
        )
        ordinary_life_damage = unmet_other_demand * life_damage_rate
        direct_life_damage = max(
            0.0,
            float(getattr(creature, "pending_direct_life_damage", 0.0)),
        )
        starting_life = min(
            self.config.max_life,
            max(
                0.0,
                float(
                    getattr(
                        creature,
                        "life",
                        self.config.max_life
                        * self.config.initial_life_fraction,
                    )
                ),
            ),
        )
        life_before_movement = max(
            0.0,
            starting_life - direct_life_damage - ordinary_life_damage,
        )
        movement_multiplier = movement_life_penalty_multiplier(
            life_before_movement,
            self.config.max_life,
            self.config.movement_life_penalty_max_multiplier,
        )
        movement_life_damage = (
            unmet_powered_movement
            * life_damage_rate
            * movement_multiplier
        )
        life_damage_from_deficit = (
            ordinary_life_damage + movement_life_damage
        )
        damaged_life = min(
            self.config.max_life,
            max(
                0.0,
                starting_life
                - life_damage_from_deficit
                - direct_life_damage,
            ),
        )
        life_healed = 0.0
        healing_energy_spent = 0.0
        if damaged_life > 0.0 and damaged_life < self.config.max_life:
            healing_cost_per_life = max(
                ENERGY_EPSILON,
                self.config.rest_healing_energy_cost_per_life,
            )
            life_healed = min(
                self.config.max_life - damaged_life,
                max(0.0, self.config.rest_healing_rate_per_second)
                * rest_strength
                * max(0.0, float(delta_time)),
                remaining_energy / healing_cost_per_life,
            )
            healing_energy_spent = life_healed * healing_cost_per_life
            remaining_energy = max(0.0, remaining_energy - healing_energy_spent)
        final_energy = min(self.config.max_energy, remaining_energy)
        final_life = min(self.config.max_life, damaged_life + life_healed)
        return ResourceCandidate(
            digestion=digestion,
            total_energy_demand=demand,
            final_stomach_energy=remaining_stomach,
            final_stomach_difficulty_load=remaining_load,
            available_energy=available_energy,
            remaining_energy=remaining_energy,
            unmet_energy_demand=unmet_energy_demand,
            life_damage_from_deficit=life_damage_from_deficit,
            direct_life_damage=direct_life_damage,
            final_energy=final_energy,
            final_life=final_life,
            rest_energy_recovered=rest_energy_recovered,
            healing_energy_spent=healing_energy_spent,
            life_healed=life_healed,
            powered_movement_energy_demand=powered_movement_demand,
            unmet_other_energy_demand=unmet_other_demand,
            unmet_powered_movement_demand=unmet_powered_movement,
            movement_life_penalty_multiplier=movement_multiplier,
            movement_life_damage=movement_life_damage,
        )

    def commit_candidate(
        self,
        creature: Creature,
        candidate: ResourceCandidate,
        *,
        transaction_status: str = "baseline_committed",
        record_diagnostics: bool = True,
    ) -> None:
        """Commit a previously evaluated candidate exactly once."""
        creature.stomach_energy = candidate.final_stomach_energy
        creature.stomach_difficulty_load = (
            candidate.final_stomach_difficulty_load
        )
        creature.energy = candidate.final_energy
        try:
            creature.life = candidate.final_life
        except AttributeError:
            pass
        try:
            creature.pending_direct_life_damage = 0.0
        except AttributeError:
            pass
        if not record_diagnostics:
            return
        diagnostics = getattr(creature, "ledger_diagnostics", None)
        if diagnostics is None:
            return
        diagnostics.effective_efficiency = (
            0.0
            if candidate.digestion.stomach_consumed <= 0.0
            else candidate.digestion.gross_energy
            / candidate.digestion.stomach_consumed
        )
        diagnostics.stomach_consumed = candidate.digestion.stomach_consumed
        diagnostics.gross_energy = candidate.digestion.gross_energy
        diagnostics.processing_cost = candidate.digestion.processing_cost
        diagnostics.net_energy = candidate.digestion.net_energy
        diagnostics.rest_energy_recovered = candidate.rest_energy_recovered
        diagnostics.healing_energy_spent = candidate.healing_energy_spent
        diagnostics.life_healed = candidate.life_healed
        diagnostics.total_energy_demand = candidate.total_energy_demand
        diagnostics.powered_movement_energy_demand = (
            candidate.powered_movement_energy_demand
        )
        diagnostics.unmet_energy_demand = candidate.unmet_energy_demand
        diagnostics.unmet_other_energy_demand = (
            candidate.unmet_other_energy_demand
        )
        diagnostics.unmet_powered_movement_demand = (
            candidate.unmet_powered_movement_demand
        )
        diagnostics.movement_life_penalty_multiplier = (
            candidate.movement_life_penalty_multiplier
        )
        diagnostics.movement_life_damage = candidate.movement_life_damage
        diagnostics.life_damage_from_deficit = (
            candidate.life_damage_from_deficit
        )
        diagnostics.direct_life_damage = candidate.direct_life_damage
        diagnostics.final_energy = candidate.final_energy
        diagnostics.final_life = candidate.final_life
        diagnostics.transaction_status = transaction_status

    def consume_energy(
        self,
        creature: Creature,
        delta_time: float,
        max_speed: float,
        sprint_intensity: float = 0.0,
        energy_cost_multiplier: float = 1.0,
        age_seconds: float | None = None,
        communication_intensities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        energy_cost = (
            self.energy_cost_breakdown_per_second(
                creature,
                max_speed,
                sprint_intensity=sprint_intensity,
                age_seconds=age_seconds,
                communication_intensities=communication_intensities,
            ).total
            * delta_time
            * energy_cost_multiplier
        )

        # Update the creature's energy
        creature.energy = max(0.0, creature.energy - energy_cost)

    def energy_cost_breakdown_per_second(
        self,
        creature: Creature,
        max_speed: float,
        sprint_intensity: float = 0.0,
        age_seconds: float | None = None,
        communication_intensities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> EnergyCostBreakdown:
        speed_ratio: float = 0.0
        if max_speed > 0:
            speed_ratio = min(max(creature.speed, 0.0) / max_speed, 1.0)

        base_movement = self.config.movement_energy_cost_factor * speed_ratio
        movement_multiplier = max(
            0.0,
            creature.physical_traits.movement_cost_multiplier,
        )
        movement = base_movement * movement_multiplier
        sprint = getattr(self.config, "sprint_energy_cost_per_second", 0.04) * min(
            max(sprint_intensity, 0.0),
            1.0,
        )
        vision = self.vision.energy_cost_per_second(creature)
        body = self.body_energy_cost_per_second(creature)
        digestive_upkeep = self.digestive_upkeep_energy_cost_per_second(
            creature
        )
        sound, trail, alarm = communication_intensities
        acoustic = (
            max(0.0, self.communication_config.acoustic_energy_cost_per_second)
            * min(max(sound, 0.0), 1.0) ** 2
        )
        pheromone = (
            max(0.0, self.communication_config.pheromone_energy_cost_per_second)
            * (
                min(max(trail, 0.0), 1.0)
                + min(max(alarm, 0.0), 1.0)
            )
        )
        trait = (
            vision
            + body
            + max(0.0, movement - base_movement)
            + acoustic
            + pheromone
            + digestive_upkeep
        )

        return EnergyCostBreakdown(
            base=(
                self.config.basic_metabolism_rate
                + self.brain_upkeep_energy_cost_per_second(
                    creature,
                    age_seconds=age_seconds,
                )
            ),
            movement=movement,
            sprint=sprint,
            vision=vision,
            body=body,
            trait=trait,
            acoustic=acoustic,
            pheromone=pheromone,
            digestive_upkeep=digestive_upkeep,
        )

    def trait_energy_cost_per_second(
        self,
        creature: Creature,
        max_speed: float,
        communication_intensities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> float:
        return self.energy_cost_breakdown_per_second(
            creature,
            max_speed,
            communication_intensities=communication_intensities,
        ).trait

    def body_energy_cost_per_second(self, creature: Creature) -> float:
        max_radius = max(self.trait_config.max_radius, 0.0001)
        radius_ratio = max(0.0, creature.radius) / max_radius
        return self.trait_config.body_metabolism_cost_factor * radius_ratio**2

    def digestive_upkeep_energy_cost_per_second(
        self,
        creature: Creature,
    ) -> float:
        traits = getattr(creature, "physical_traits", None)
        capacity = max(
            0.0,
            float(
                getattr(
                    traits,
                    "stomach_capacity",
                    self.trait_config.default_stomach_capacity,
                )
            ),
        )
        rate = max(
            0.0,
            float(
                getattr(
                    traits,
                    "digestion_rate",
                    self.trait_config.default_digestion_rate,
                )
            ),
        )
        efficiency = max(
            0.0,
            float(
                getattr(
                    traits,
                    "digestion_efficiency",
                    self.trait_config.default_digestion_efficiency,
                )
            ),
        )
        capacity_ratio = capacity / max(
            self.trait_config.default_stomach_capacity,
            1e-12,
        )
        rate_ratio = rate / max(
            self.trait_config.default_digestion_rate,
            1e-12,
        )
        efficiency_ratio = efficiency / max(
            self.trait_config.default_digestion_efficiency,
            1e-12,
        )
        weighted_scale = (
            self.config.digestive_capacity_upkeep_weight * capacity_ratio**2
            + self.config.digestive_rate_upkeep_weight * rate_ratio**2
            + self.config.digestive_efficiency_upkeep_weight
            * efficiency_ratio**2
        )
        return min(
            self.config.max_digestive_upkeep_per_second,
            self.config.digestive_upkeep_at_default_per_second
            * weighted_scale,
        )

    def digestion_processing_fraction(
        self,
        digestion_rate: float,
        difficulty: float,
    ) -> float:
        rate_ratio = max(0.0, digestion_rate) / max(
            self.trait_config.max_digestion_rate,
            1e-12,
        )
        fraction = (
            self.config.digestion_processing_base_fraction
            * max(0.0, difficulty)
            * (1.0 + self.config.digestion_rate_cost_factor * rate_ratio**2)
        )
        return min(
            max(0.0, self.config.max_digestion_processing_fraction),
            max(0.0, fraction),
        )

    def _validate_config(self) -> None:
        trait = self.trait_config
        metabolism = self.config

        def require_finite(name: str, value: float) -> float:
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"{name} must be finite.")
            return numeric

        ranges = (
            (
                "stomach_capacity",
                trait.min_stomach_capacity,
                trait.default_stomach_capacity,
                trait.max_stomach_capacity,
            ),
            (
                "digestion_rate",
                trait.min_digestion_rate,
                trait.default_digestion_rate,
                trait.max_digestion_rate,
            ),
            (
                "digestion_efficiency",
                trait.min_digestion_efficiency,
                trait.default_digestion_efficiency,
                trait.max_digestion_efficiency,
            ),
        )
        for name, minimum, default, maximum in ranges:
            minimum = require_finite(f"min_{name}", minimum)
            default = require_finite(f"default_{name}", default)
            maximum = require_finite(f"max_{name}", maximum)
            if minimum <= 0.0 or not minimum <= default <= maximum:
                raise ValueError(
                    f"{name} must have 0 < minimum <= default <= maximum."
                )
            if maximum <= minimum:
                raise ValueError(f"{name} range must have positive width.")
        if trait.max_digestion_efficiency > 1.0:
            raise ValueError("digestion_efficiency cannot exceed 1.0.")
        mutation_rate = require_finite(
            "digestive_trait_mutation_rate",
            trait.digestive_trait_mutation_rate,
        )
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(
                "digestive_trait_mutation_rate must be between 0 and 1."
            )
        nonnegative = {
            "stomach_capacity_mutation_stddev": (
                trait.stomach_capacity_mutation_stddev
            ),
            "digestion_rate_mutation_stddev": (
                trait.digestion_rate_mutation_stddev
            ),
            "digestion_efficiency_mutation_stddev": (
                trait.digestion_efficiency_mutation_stddev
            ),
            "initial_stomach_capacity_jitter": (
                trait.initial_stomach_capacity_jitter
            ),
            "initial_digestion_rate_jitter": (
                trait.initial_digestion_rate_jitter
            ),
            "initial_digestion_efficiency_jitter": (
                trait.initial_digestion_efficiency_jitter
            ),
            "digestive_upkeep_at_default_per_second": (
                metabolism.digestive_upkeep_at_default_per_second
            ),
            "max_digestive_upkeep_per_second": (
                metabolism.max_digestive_upkeep_per_second
            ),
            "digestion_processing_base_fraction": (
                metabolism.digestion_processing_base_fraction
            ),
            "digestion_rate_cost_factor": (
                metabolism.digestion_rate_cost_factor
            ),
            "min_food_difficulty_multiplier": (
                metabolism.min_food_difficulty_multiplier
            ),
            "max_food_difficulty_multiplier": (
                metabolism.max_food_difficulty_multiplier
            ),
            "life_damage_per_energy_deficit": (
                metabolism.life_damage_per_energy_deficit
            ),
            "movement_life_penalty_max_multiplier": (
                metabolism.movement_life_penalty_max_multiplier
            ),
            "rest_digestion_efficiency_bonus": (
                metabolism.rest_digestion_efficiency_bonus
            ),
            "rest_energy_recovery_per_second": (
                metabolism.rest_energy_recovery_per_second
            ),
            "rest_healing_rate_per_second": (
                metabolism.rest_healing_rate_per_second
            ),
        }
        for name, value in nonnegative.items():
            if require_finite(name, value) < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if metabolism.movement_life_penalty_max_multiplier < 1.0:
            raise ValueError(
                "movement_life_penalty_max_multiplier must be at least 1.0."
            )
        weights = (
            metabolism.digestive_capacity_upkeep_weight,
            metabolism.digestive_rate_upkeep_weight,
            metabolism.digestive_efficiency_upkeep_weight,
        )
        if any(
            require_finite("digestive upkeep weight", weight) < 0.0
            for weight in weights
        ) or sum(weights) <= 0.0:
            raise ValueError(
                "Digestive upkeep weights must be nonnegative and sum above zero."
            )
        if (
            metabolism.max_food_difficulty_multiplier
            < metabolism.min_food_difficulty_multiplier
        ):
            raise ValueError(
                "Food difficulty maximum must be at least its minimum."
            )
        min_food_radius = require_finite(
            "min_food_radius",
            self.food_config.min_food_radius,
        )
        max_food_radius = require_finite(
            "max_food_radius",
            self.food_config.max_food_radius,
        )
        if min_food_radius <= 0.0 or max_food_radius <= min_food_radius:
            raise ValueError(
                "Food radii must have 0 < minimum < maximum."
            )
        max_processing = require_finite(
            "max_digestion_processing_fraction",
            metabolism.max_digestion_processing_fraction,
        )
        if not 0.0 <= max_processing <= 0.5:
            raise ValueError(
                "max_digestion_processing_fraction must be in [0, 0.5]."
            )
        base_processing = require_finite(
            "digestion_processing_base_fraction",
            metabolism.digestion_processing_base_fraction,
        )
        if base_processing > max_processing:
            raise ValueError(
                "digestion_processing_base_fraction cannot exceed "
                "max_digestion_processing_fraction."
            )
        max_life = require_finite("max_life", metabolism.max_life)
        if max_life <= 0.0:
            raise ValueError("max_life must be positive.")
        max_energy = require_finite("max_energy", metabolism.max_energy)
        if max_energy <= 0.0:
            raise ValueError("max_energy must be positive.")
        starvation_threshold = require_finite(
            "starvation_energy_threshold",
            metabolism.starvation_energy_threshold,
        )
        if not 0.0 <= starvation_threshold <= max_energy:
            raise ValueError(
                "starvation_energy_threshold must be within [0, max_energy]."
            )
        healing_energy_cost = require_finite(
            "rest_healing_energy_cost_per_life",
            metabolism.rest_healing_energy_cost_per_life,
        )
        if healing_energy_cost <= 0.0:
            raise ValueError(
                "rest_healing_energy_cost_per_life must be positive."
            )
        initial_life_fraction = require_finite(
            "initial_life_fraction",
            metabolism.initial_life_fraction,
        )
        if not 0.0 <= initial_life_fraction <= 1.0:
            raise ValueError("initial_life_fraction must be within [0, 1].")
        if metabolism.rest_digestion_efficiency_bonus > 1.0:
            raise ValueError(
                "rest_digestion_efficiency_bonus must be within [0, 1]."
            )

    def brain_upkeep_energy_cost_per_second(
        self,
        creature: Creature,
        age_seconds: float | None = None,
    ) -> float:
        if age_seconds is not None and age_seconds < 5.0:
            return 0.0

        if self.genome_for_creature_id is None:
            return 0.0

        genome = self.genome_for_creature_id(creature.creature_id)
        if genome is None:
            return 0.0

        nodes = getattr(genome, "nodes", {}) or {}
        connections = getattr(genome, "connections", {}) or {}
        return (
            len(nodes) * self.config.brain_upkeep_per_node
            + len(connections) * self.config.brain_upkeep_per_connection
        )

    def eat(
        self,
        creature: Creature,
        food: Food,
        delta_time: float,
    ) -> FoodConsumption:
        stomach_capacity = self.stomach_capacity(creature)
        stomach_energy = max(0.0, creature.stomach_energy)
        stomach_space = max(0.0, stomach_capacity - stomach_energy)
        if stomach_space <= 0.0:
            return FoodConsumption(
                creature_id=creature.creature_id,
                food=food,
                energy_swallowed=0.0,
                depleted=False,
            )

        bite_limit = (
            max(0.0, self.config.max_bite_size_per_second)
            * max(0.0, delta_time)
        )
        food_energy = max(0.0, food.energy_value)
        bite = min(stomach_space, bite_limit, food_energy)
        tolerance = max(0.0, self.config.micro_food_remainder_ratio)
        if (
            food_energy <= stomach_space
            and food_energy <= bite_limit * (1.0 + tolerance)
        ):
            bite = food_energy
        result = food.consume_energy(
            bite,
            0.0,
        )
        swallowed = min(stomach_space, max(0.0, result.energy_removed))
        creature.stomach_energy = min(
            stomach_capacity,
            stomach_energy + swallowed,
        )
        difficulty = self.food_difficulty_multiplier(food)
        previous_load = self.normalized_stomach_difficulty_load(
            creature,
            stomach_energy,
        )
        creature.stomach_difficulty_load = (
            previous_load + swallowed * difficulty
        )
        depleted = result.depleted
        return FoodConsumption(
            creature_id=creature.creature_id,
            food=food,
            energy_swallowed=swallowed,
            depleted=depleted,
        )

    def stomach_capacity(self, creature: Creature) -> float:
        traits = getattr(creature, "physical_traits", None)
        inherited = getattr(traits, "stomach_capacity", None)
        if inherited is not None:
            return max(0.0, float(inherited))
        return max(
            0.0,
            creature.radius * self.config.stomach_capacity_per_radius,
        )

    def food_difficulty_multiplier(self, food: Food) -> float:
        minimum_radius = float(self.food_config.min_food_radius)
        maximum_radius = float(self.food_config.max_food_radius)
        original_radius = max(
            0.0,
            float(getattr(food, "original_radius", food.radius)),
        )
        if maximum_radius <= minimum_radius:
            normalized_radius = 0.5
        else:
            normalized_radius = min(
                max(
                    (original_radius - minimum_radius)
                    / (maximum_radius - minimum_radius),
                    0.0,
                ),
                1.0,
            )
        minimum_difficulty = max(
            0.0,
            self.config.min_food_difficulty_multiplier,
        )
        maximum_difficulty = max(
            minimum_difficulty,
            self.config.max_food_difficulty_multiplier,
        )
        return minimum_difficulty + normalized_radius * (
            maximum_difficulty - minimum_difficulty
        )

    def normalized_stomach_difficulty_load(
        self,
        creature: Creature,
        stomach_energy: float | None = None,
    ) -> float:
        energy = (
            max(0.0, getattr(creature, "stomach_energy", 0.0))
            if stomach_energy is None
            else max(0.0, stomach_energy)
        )
        if energy <= 0.0:
            return 0.0
        load = max(
            0.0,
            getattr(creature, "stomach_difficulty_load", 0.0),
        )
        if load <= 0.0:
            return energy
        minimum_load = (
            energy * max(0.0, self.config.min_food_difficulty_multiplier)
        )
        maximum_load = energy * max(
            self.config.min_food_difficulty_multiplier,
            self.config.max_food_difficulty_multiplier,
        )
        return min(max(load, minimum_load), maximum_load)

    def find_eatable_food(
        self,
        creature: Creature,
        food_items: Sequence[Food],
        ignored_foods: list[Food],
    ) -> Food | None:
        ignored_food_ids = {food.id for food in ignored_foods}

        for food in food_items:
            # Check if ignored food
            if food.id in ignored_food_ids:
                continue

            if self.food_overlaps_mouth(creature, food):
                return food

        return None

    def mouth_position(self, creature: Creature) -> tuple[float, float]:
        creature_x, creature_y = creature.position
        return (
            creature_x + cos(creature.heading) * creature.radius,
            creature_y + sin(creature.heading) * creature.radius,
        )

    def food_overlaps_mouth(self, creature: Creature, food: Food) -> bool:
        creature_x, creature_y = creature.position
        food_x, food_y = food.position
        dx = food_x - creature_x
        dy = food_y - creature_y

        contact_slop = max(1.0, min(3.0, self.config.eating_distance * 0.25))
        contact_range = creature.radius + food.radius + contact_slop
        if dx * dx + dy * dy > contact_range * contact_range:
            return False

        forward_x = cos(creature.heading)
        forward_y = sin(creature.heading)
        forward_distance = dx * forward_x + dy * forward_y
        if forward_distance < creature.radius - contact_slop:
            return False

        lateral_x = -forward_y
        lateral_y = forward_x
        lateral_distance = abs(dx * lateral_x + dy * lateral_y)
        mouth_half_width = max(2.0, creature.radius * 0.35)
        return lateral_distance <= food.radius + mouth_half_width

    def is_starving(self, creature: Creature) -> bool:
        return creature.energy < self.config.starvation_energy_threshold

    def is_dead(self, creature: Creature) -> bool:
        return float(
            getattr(
                creature,
                "life",
                self.config.max_life * self.config.initial_life_fraction,
            )
        ) <= 0.0
