"""Small behavior-preserving helpers shared by creature subsystems."""

from __future__ import annotations


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value to an inclusive interval.

Parameters
----------
value
    Value to constrain.
minimum
    Inclusive lower bound.
maximum
    Inclusive upper bound.

Returns
-------
float
    Constrained floating-point value."""
    # Keep clamp behavior explicit in its owning subsystem.
    # Retain the historical max/min ordering used by creature calculations.
    return max(minimum, min(maximum, float(value)))
