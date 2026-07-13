from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from math import cos, sin

from src.food import Food
from src.creature import Creature
from configs.sim_config import CommunicationConfig, MetabolismConfig, TraitConfig
from src.vision import VisionSystem


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
        )


class Metabolism:
    def __init__(
        self,
        config: MetabolismConfig,
        vision: VisionSystem,
        trait_config: TraitConfig | None = None,
        genome_for_creature_id: Callable[[int], object | None] | None = None,
        communication_config: CommunicationConfig | None = None,
    ) -> None:
        self.config = config
        self.vision = vision
        self.trait_config = trait_config or TraitConfig()
        self.genome_for_creature_id = genome_for_creature_id
        self.communication_config = communication_config or CommunicationConfig()

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

        for creature in creatures:
            energy_gained = self.digest(creature, delta_time)
            if energy_gained > 0.0:
                digested_energy_gained[creature.creature_id] = energy_gained

            # Consume the energy from the creatures
            sprint_intensity = (
                0.0
                if sprint_intensities is None
                else sprint_intensities.get(creature.creature_id, 0.0)
            )
            self.consume_energy(
                creature,
                delta_time,
                max_speed,
                sprint_intensity=sprint_intensity,
                energy_cost_multiplier=(
                    1.0
                    if energy_cost_multipliers is None
                    else energy_cost_multipliers.get(creature.creature_id, 1.0)
                ),
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
            )

            # Calculate the eatble food
            candidate_foods = (
                food_items if nearby_foods_for is None else nearby_foods_for(creature)
            )

            if can_eat is not None and not can_eat(creature):
                if self.is_dead(creature):
                    dead_creatures.append(creature)
                continue
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
            dead_creatures=dead_creatures,
        )

    def digest(self, creature: Creature, delta_time: float) -> float:
        stomach_energy = max(0.0, getattr(creature, "stomach_energy", 0.0))
        to_digest = min(
            stomach_energy,
            max(0.0, self.config.digestion_rate_per_second) * max(0.0, delta_time),
        )
        if to_digest <= 0.0:
            creature.stomach_energy = stomach_energy
            return 0.0

        creature.stomach_energy = stomach_energy - to_digest
        previous_energy = creature.energy
        net_energy = to_digest * max(0.0, self.config.digestion_efficiency)
        creature.energy = min(
            self.config.max_energy,
            creature.energy + net_energy,
        )
        return max(0.0, creature.energy - previous_energy)

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
        stomach_capacity = max(
            0.0,
            creature.radius * self.config.stomach_capacity_per_radius,
        )
        stomach_energy = max(0.0, creature.stomach_energy)
        stomach_space = max(0.0, stomach_capacity - stomach_energy)
        bite_limit = (
            max(0.0, self.config.max_bite_size_per_second)
            * max(0.0, delta_time)
        )
        bite = min(stomach_space, bite_limit, max(0.0, food.energy_value))
        result = food.consume_energy(
            bite,
            0.0,
        )
        creature.stomach_energy = min(
            stomach_capacity,
            stomach_energy + result.energy_removed,
        )
        depleted = result.depleted or (
            result.energy_removed > 0.0 and food.energy_value <= 0.01
        )
        if depleted:
            food.energy_value = 0.0
        return FoodConsumption(
            creature_id=creature.creature_id,
            food=food,
            energy_swallowed=result.energy_removed,
            depleted=depleted,
        )

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
        return creature.energy <= 0
