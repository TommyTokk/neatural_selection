from __future__ import annotations

from dataclasses import dataclass, field

from math import dist

from src.food import Food
from src.creature import Creature
from configs.sim_config import MetabolismConfig
from src.vision import VisionSystem


@dataclass(slots=True)
class MetabolismReport:
    eaten_foods: list[Food] = field(default_factory=list)
    dead_creatures: list[Creature] = field(default_factory=list)


class Metabolism:
    def __init__(self, config: MetabolismConfig, vision: VisionSystem) -> None:
        self.config = config
        self.vision = vision

    def update(self, creatures: list[Creature], food_items: list[Food], delta_time:float, max_speed:float) -> MetabolismReport:

        eaten_foods: list[Food] = []
        dead_creatures: list[Creature] = []
        
        for creature in creatures:
            # Consume the energy from the creatures
            self.consume_energy(creature, delta_time, max_speed)

            # Calculate the eatble food
            food = self.find_eatable_food(creature, food_items, eaten_foods)

            if food is not None:
                self.eat(creature, food)
                eaten_foods.append(food)

            if self.is_dead(creature):
                dead_creatures.append(creature)

        return MetabolismReport(eaten_foods=eaten_foods, dead_creatures=dead_creatures)
    
    def consume_energy(self, creature: Creature, delta_time: float, max_speed: float) -> None:

        speed_ratio: float = 0

        if max_speed > 0:
            speed_ratio = min(creature.speed / max_speed, 1)

        # Define the energy cost
        energy_cost = (
            self.config.basic_metabolism_rate
            + self.config.movement_energy_cost_factor * speed_ratio
            + self.vision.energy_cost_per_second(creature)
        ) * delta_time

        # Update the creature's energy
        creature.energy = max(0.0, creature.energy - energy_cost)

    def eat(self, creature: Creature, food: Food) -> None:
        creature.energy = min(
            self.config.max_energy,
            creature.energy + food.energy_value,
        )

    def find_eatable_food(self, creature: Creature, food_items: list[Food], ignored_foods: list[Food]) -> Food | None:

        for food in food_items:
            
            # Check if ignored food
            if food in ignored_foods:
                continue

            # Calculate the eating radius 
            eating_range = creature.radius + food.radius + self.config.eating_distance

            # Check if the food is within the eating range
            if dist(creature.position, food.position) <= eating_range:
                return food
            
        return None
    
    def is_starving(self, creature: Creature) -> bool:
        return creature.energy < self.config.starvation_energy_threshold
    
    def is_dead(self, creature: Creature) -> bool:
        return creature.energy <= 0
    
    def can_reproduce(self, creature: Creature) -> bool:
        return creature.energy >= self.config.reproduction_energy_threshold
    
    def spend_reproduction_energy(self, creature: Creature) -> None:
        creature.energy = max(
            0.0,
            creature.energy - self.config.reproduction_energy_cost,
        )


        

