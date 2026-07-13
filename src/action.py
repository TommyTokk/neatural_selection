from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

ACTION_OUTPUT_COUNT = 16
ACTION_OUTPUT_NAMES = (
    "accelerate",
    "rotate",
    "want_reproduce",
    "want_eat",
    "reset_chronometer",
    "want_grab",
    "want_release",
    "want_nurse",
    "flee_panic_intensity",
    "weight_separation",
    "weight_alignment",
    "weight_cohesion",
    "emit_sound",
    "sound_tone",
    "emit_trail_pheromone",
    "emit_alarm_pheromone",
)
NEUTRAL_NETWORK_OUTPUT = 0.5


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
    weight_separation: float = 0.0
    weight_alignment: float = 0.0
    weight_cohesion: float = 0.0
    emit_sound: float = 0.0
    sound_tone: float = 0.0
    emit_trail_pheromone: float = 0.0
    emit_alarm_pheromone: float = 0.0

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
            weight_separation=max(0.0, min(1.0, self.weight_separation)),
            weight_alignment=max(0.0, min(1.0, self.weight_alignment)),
            weight_cohesion=max(0.0, min(1.0, self.weight_cohesion)),
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
        )


def signed_output(value: float) -> float:
    return (value - NEUTRAL_NETWORK_OUTPUT) * 2.0


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
