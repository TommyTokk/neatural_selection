from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import cos, sin

class BrainOutputIndex(IntEnum):
    ACCELERATE = 0
    ROTATE = 1
    REPRODUCE = 2
    EAT = 3
    RESET_CHRONOMETER = 4
    GRAB_FOOD = 5
    RELEASE_FOOD = 6
    NURSE = 7
    PANIC = 8
    HERDING = 9
    ACOUSTIC_EMISSION = 10
    ACOUSTIC_TONE = 11
    TRAIL_PHEROMONE = 12
    ALARM_PHEROMONE = 13
    REST = 14


ACTION_SCHEMA_VERSION = 2
_ACTION_OUTPUT_NAME_BY_INDEX = {
    BrainOutputIndex.ACCELERATE: "accelerate",
    BrainOutputIndex.ROTATE: "rotate",
    BrainOutputIndex.REPRODUCE: "want_reproduce",
    BrainOutputIndex.EAT: "want_eat",
    BrainOutputIndex.RESET_CHRONOMETER: "reset_chronometer",
    BrainOutputIndex.GRAB_FOOD: "want_grab",
    BrainOutputIndex.RELEASE_FOOD: "want_release",
    BrainOutputIndex.NURSE: "want_nurse",
    BrainOutputIndex.PANIC: "flee_panic_intensity",
    BrainOutputIndex.HERDING: "herding",
    BrainOutputIndex.ACOUSTIC_EMISSION: "emit_sound",
    BrainOutputIndex.ACOUSTIC_TONE: "sound_tone",
    BrainOutputIndex.TRAIL_PHEROMONE: "emit_trail_pheromone",
    BrainOutputIndex.ALARM_PHEROMONE: "emit_alarm_pheromone",
    BrainOutputIndex.REST: "rest",
}
ACTION_OUTPUT_NAMES = tuple(
    _ACTION_OUTPUT_NAME_BY_INDEX[output] for output in BrainOutputIndex
)
ACTION_OUTPUT_COUNT = len(ACTION_OUTPUT_NAMES)
INTENT_THRESHOLD = 0.1


def is_active_intent(value: float) -> bool:
    """Return whether a positive action intent is strong enough to activate."""
    return value > INTENT_THRESHOLD


@dataclass(slots=True)
class Action:
    accelerate: float
    rotate: float
    want_reproduce: float
    want_eat: float
    reset_chronometer: float
    want_grab: float
    want_release: float
    want_nurse: float = 0.0
    flee_panic_intensity: float = 0.0
    herding: float = 0.0
    emit_sound: float = 0.0
    sound_tone: float = 0.0
    emit_trail_pheromone: float = 0.0
    emit_alarm_pheromone: float = 0.0
    rest: float = 0.0

    def clamped(self) -> Action:
        return Action(
            accelerate=max(-1.0, min(1.0, self.accelerate)),
            rotate=max(-1.0, min(1.0, self.rotate)),
            want_reproduce=max(0.0, min(1.0, self.want_reproduce)),
            want_eat=max(0.0, min(1.0, self.want_eat)),
            reset_chronometer=max(0.0, min(1.0, self.reset_chronometer)),
            want_grab=max(0.0, min(1.0, self.want_grab)),
            want_release=max(0.0, min(1.0, self.want_release)),
            want_nurse=max(0.0, min(1.0, self.want_nurse)),
            flee_panic_intensity=max(
                0.0,
                min(1.0, self.flee_panic_intensity),
            ),
            herding=max(0.0, min(1.0, self.herding)),
            emit_sound=max(0.0, min(1.0, self.emit_sound)),
            sound_tone=max(-1.0, min(1.0, self.sound_tone)),
            emit_trail_pheromone=max(
                0.0,
                min(1.0, self.emit_trail_pheromone),
            ),
            emit_alarm_pheromone=max(
                0.0,
                min(1.0, self.emit_alarm_pheromone),
            ),
            rest=max(0.0, min(1.0, self.rest)),
        )


def acceleration_force_vector(
    accelerate: float,
    heading: float,
    max_forward_force: float,
    max_backward_force: float,
) -> tuple[float, float]:
    if accelerate >= 0.0:
        magnitude = max_forward_force * accelerate
    else:
        magnitude = max_backward_force * accelerate

    return (
        cos(heading) * magnitude,
        sin(heading) * magnitude,
    )
