"""Ownership of transient neural actions and motion commands."""

from __future__ import annotations

from typing import Any

from src.creature.action import Action


class CreatureActionService:
    """Own raw/effective action and motion-command caches."""

    def __init__(self) -> None:
        """Initialize empty per-creature action state.

Parameters
----------
None
    This initializer receives no external parameters.

Returns
-------
None
    Empty action caches are created."""
        # Keep init behavior explicit in its owning subsystem.
        # Raw and effective actions remain separate for resource accounting.
        self.raw: dict[int, Action] = {}
        self.effective: dict[int, Action] = {}
        self.motion_commands: dict[int, Any] = {}

    def initialize(self, creature_id: int, action: Action, motion: Any) -> None:
        """Install deterministic neutral state for a newly live creature.

Parameters
----------
creature_id
    Stable new creature identity.
action
    Neutral raw and effective action.
motion
    Neutral motion command.

Returns
-------
None
    All action caches receive initial entries."""
        # Keep initialize behavior explicit in its owning subsystem.
        # Use the same immutable action object until a decision replaces it.
        self.raw[creature_id] = action
        self.effective[creature_id] = action
        self.motion_commands[creature_id] = motion

    def discard(self, creature_id: int) -> None:
        """Discard cached action state for one creature.

Parameters
----------
creature_id
    Stable identity leaving the live population.

Returns
-------
None
    Associated action caches are cleared."""
        # Keep discard behavior explicit in its owning subsystem.
        # Idempotent cleanup supports rollback after partially staged births.
        self.raw.pop(creature_id, None)
        self.effective.pop(creature_id, None)
        self.motion_commands.pop(creature_id, None)
