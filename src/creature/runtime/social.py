"""Ownership of transient flocking and communication runtime state."""

from __future__ import annotations

from typing import Any

from src.creature.flocking import SocialCompatibilityResolver, SocialRuntime


class CreatureSocialService:
    """Own compatibility and per-creature social runtime caches."""

    def __init__(self, compatibility: SocialCompatibilityResolver) -> None:
        """Initialize social state around a compatibility resolver.

Parameters
----------
compatibility
    Configured live compatibility resolver.

Returns
-------
None
    Empty social caches are created."""
        # Keep init behavior explicit in its owning subsystem.
        # Compatibility and its cache share lifecycle ownership here.
        self.compatibility = compatibility
        self.intentions: dict[int, SocialRuntime] = {}
        self.last_runtime: dict[int, Any] = {}
        self.last_debug: dict[int, Any] = {}
        self.communication_positions: Any | None = None
        self.communication_color_amounts: Any | None = None

    def initialize(self, creature_id: int) -> None:
        """Install neutral social state for one creature.

Parameters
----------
creature_id
    Stable new creature identity.

Returns
-------
None
    A neutral social runtime is cached."""
        # Keep initialize behavior explicit in its owning subsystem.
        # Neutral state prevents missing-cache branches in the hot loop.
        self.intentions[creature_id] = SocialRuntime()

    def discard(self, creature_id: int) -> None:
        """Discard social state and compatibility entries for one creature.

Parameters
----------
creature_id
    Stable identity leaving the live population.

Returns
-------
None
    Associated social caches are cleared."""
        # Keep discard behavior explicit in its owning subsystem.
        # Clear the resolver last so no stale pair survives local cache cleanup.
        self.intentions.pop(creature_id, None)
        self.last_runtime.pop(creature_id, None)
        self.last_debug.pop(creature_id, None)
        self.compatibility.discard_creature(creature_id)
