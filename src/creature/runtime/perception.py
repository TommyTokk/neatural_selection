"""Ownership of transient sensing results and reusable perception state."""

from __future__ import annotations

from typing import Any

from src.creature.vision import SensorSnapshot, VisionSystem


class CreaturePerceptionService:
    """Own the vision system and per-creature sensor snapshot cache."""

    def __init__(self, vision: VisionSystem) -> None:
        """Initialize perception state around an existing vision system.

Parameters
----------
vision
    Configured geometric sensing system.

Returns
-------
None
    Empty transient sensing state is created."""
        # Keep init behavior explicit in its owning subsystem.
        # Vision remains a compatibility alias while snapshots gain one owner.
        self.vision = vision
        self.spatial_index: Any | None = None
        self.candidate_buffer: Any | None = None
        self.last_snapshots: dict[int, SensorSnapshot] = {}
        self.last_acoustic_debug: dict[int, Any] = {}
        self.debug_sensor_positions: dict[int, tuple[float, float]] = {}

    def discard(self, creature_id: int) -> None:
        """Discard cached perception state for one creature.

Parameters
----------
creature_id
    Stable identity leaving the live population.

Returns
-------
None
    Associated perception caches are cleared."""
        # Keep discard behavior explicit in its owning subsystem.
        # Cache cleanup is idempotent because removal can follow partial births.
        self.last_snapshots.pop(creature_id, None)
        self.last_acoustic_debug.pop(creature_id, None)
        self.debug_sensor_positions.pop(creature_id, None)
