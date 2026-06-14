from dataclasses import dataclass
from random import Random

from configs.sim_config import ActionConfig
from src.action import Action
from src.vision import SensorSnapshot


@dataclass(slots=True)
class ScavengeState:
    turn: float
    steps_remaining: int


class BaselineFoodController:
    def __init__(self, config: ActionConfig, rng: Random | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else Random(17)
        self._scavenge_states: dict[int, ScavengeState] = {}

    def decide(self, snapshot: SensorSnapshot, creature_id: int | None = None) -> Action:
        state_id = 0 if creature_id is None else creature_id

        # Get the visible foods
        visible_foods = snapshot.food.visible

        if snapshot.boundary.pressure > 0.0:
            self._scavenge_states.pop(state_id, None)
            pressure = snapshot.boundary.pressure
            turn = snapshot.boundary.turn
            turn_strength = (
                self.config.boundary_avoidance_min_turn
                + pressure * self.config.boundary_avoidance_turn
            )
            rotate = turn * turn_strength
            if (
                pressure >= self.config.boundary_escape_pressure
                and abs(turn) >= self.config.boundary_escape_turn_threshold
            ):
                accelerate = self.config.boundary_escape_acceleration
            else:
                accelerate = self.config.boundary_avoidance_acceleration
        elif not visible_foods:
            # No food, scavenge by covering ground with intermittent scanning arcs.
            rotate = self._scavenge_turn(state_id)
            accelerate = self.config.search_acceleration
        else:
            self._scavenge_states.pop(state_id, None)
            # Food is visible, move toward the strongest sector.
            left = snapshot.food.proximity_left
            center = snapshot.food.proximity_center
            right = snapshot.food.proximity_right
            if center >= left and center >= right:
                rotate = 0.0
                accelerate = 1.0
            elif left >= right:
                turn_strength = max(0.0, left - center)
                rotate = -turn_strength * self.config.food_turn_factor
                accelerate = max(
                    self.config.min_food_acceleration,
                    1.0 - turn_strength * 0.3,
                )
            else:
                turn_strength = max(0.0, right - center)
                rotate = turn_strength * self.config.food_turn_factor
                accelerate = max(
                    self.config.min_food_acceleration,
                    1.0 - turn_strength * 0.3,
                )

        # Eval conservation of energy
        if snapshot.energy < self.config.low_energy_threshold:
            accelerate *= self.config.low_energy_acceleration_factor

        return Action(
            accelerate=accelerate,
            rotate=rotate,
            want_reproduce=1.0,
            want_eat=1.0,
            reset_chronometer=0.0,
            want_grab=0.0,
            want_release=0.0,
        ).clamped()

    def _scavenge_turn(self, creature_id: int) -> float:
        state = self._scavenge_states.get(creature_id)
        if state is None or state.steps_remaining <= 0:
            state = self._new_scavenge_state()
            self._scavenge_states[creature_id] = state

        state.steps_remaining -= 1
        return state.turn

    def _new_scavenge_state(self) -> ScavengeState:
        minimum = max(1, self.config.search_turn_interval_min)
        maximum = max(minimum, self.config.search_turn_interval_max)
        steps_remaining = self.rng.randint(minimum, maximum)

        straight_probability = max(
            0.0,
            min(1.0, self.config.search_straight_probability),
        )
        if self.rng.random() < straight_probability:
            turn = self.rng.uniform(
                -self.config.search_turn_jitter,
                self.config.search_turn_jitter,
            )
        else:
            direction = -1.0 if self.rng.random() < 0.5 else 1.0
            turn = direction * max(
                0.0,
                self.config.search_turn
                + self.rng.uniform(
                    -self.config.search_turn_jitter,
                    self.config.search_turn_jitter,
                ),
            )

        return ScavengeState(turn=turn, steps_remaining=steps_remaining)
