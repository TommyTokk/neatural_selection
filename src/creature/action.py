from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import cos, sin
from typing import Mapping

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
    EMIT_RED = 12
    EMIT_GREEN = 13
    EMIT_BLUE = 14
    REST = 15


ACTION_SCHEMA_VERSION = 3
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
    BrainOutputIndex.EMIT_RED: "emit_red",
    BrainOutputIndex.EMIT_GREEN: "emit_green",
    BrainOutputIndex.EMIT_BLUE: "emit_blue",
    BrainOutputIndex.REST: "rest",
}
ACTION_OUTPUT_NAMES = tuple(
    _ACTION_OUTPUT_NAME_BY_INDEX[output] for output in BrainOutputIndex
)
ACTION_OUTPUT_COUNT = len(ACTION_OUTPUT_NAMES)
INTENT_THRESHOLD = 0.1


def is_active_intent(value: float) -> bool:
    """Return whether a positive action intent is strong enough to activate.

Parameters
----------
value
    Input used by this creature-domain operation.
Returns
-------
bool
    Result produced by this creature-domain operation."""
    # Keep is active intent behavior explicit in its owning subsystem.
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
    emit_red: float = 0.0
    emit_green: float = 0.0
    emit_blue: float = 0.0
    rest: float = 0.0

    def __setstate__(self, state: object) -> None:
        """Restore known action fields while tolerating obsolete checkpoint slots.

        Parameters
        ----------
        state
            Pickle state produced by current or historical action schemas.

        Returns
        -------
        None
            Known fields are restored and unavailable current fields are zeroed.
        """
        # Contract-incompatible brains are reset after loading, so unknown slots
        # need only be ignored safely while the checkpoint is deserialized.
        values: Mapping[str, object] = {}
        if isinstance(state, Mapping):
            values = state
        elif (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[1], Mapping)
        ):
            values = state[1]
        for name in ACTION_OUTPUT_NAMES:
            object.__setattr__(self, name, float(values.get(name, 0.0)))

    def clamped(self) -> Action:
        """Execute clamped behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
Action
    Result produced by this creature-domain operation."""
        # Keep clamped behavior explicit in its owning subsystem.
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
            emit_red=max(0.0, min(1.0, self.emit_red)),
            emit_green=max(0.0, min(1.0, self.emit_green)),
            emit_blue=max(0.0, min(1.0, self.emit_blue)),
            rest=max(0.0, min(1.0, self.rest)),
        )


def neutral_action() -> Action:
    """Create a fresh action with every neural intent disabled.

    Parameters
    ----------
    None
        This callable receives no external parameters.

    Returns
    -------
    Action
        Independent mutable action containing only zero-valued intents.
    """
    # Return a fresh value so one creature cannot mutate another's neutral state.
    return Action(
        accelerate=0.0,
        rotate=0.0,
        want_reproduce=0.0,
        want_eat=0.0,
        reset_chronometer=0.0,
        want_grab=0.0,
        want_release=0.0,
    )


def acceleration_force_vector(
    accelerate: float,
    heading: float,
    max_forward_force: float,
    max_backward_force: float,
) -> tuple[float, float]:
    """Execute acceleration force vector behavior.

Parameters
----------
accelerate
    Input used by this creature-domain operation.
heading
    Input used by this creature-domain operation.
max_forward_force
    Input used by this creature-domain operation.
max_backward_force
    Input used by this creature-domain operation.
Returns
-------
tuple[float, float]
    Result produced by this creature-domain operation."""
    # Keep acceleration force vector behavior explicit in its owning subsystem.
    if accelerate >= 0.0:
        magnitude = max_forward_force * accelerate
    else:
        magnitude = max_backward_force * accelerate

    return (
        cos(heading) * magnitude,
        sin(heading) * magnitude,
    )
