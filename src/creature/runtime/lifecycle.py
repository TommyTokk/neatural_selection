"""Live creature identity registration and lifecycle cache coordination."""

from __future__ import annotations

from collections.abc import Callable

from src.creature.model import Creature
from src.creature.runtime.context import creature_id


class CreatureLifecycleService:
    """Own live identity registries and coordinated cache cleanup."""

    def __init__(self) -> None:
        """Initialize empty identity registries and cleanup listeners.

Parameters
----------
None
    This initializer receives no external parameters.

Returns
-------
None
    Empty lifecycle state is created."""
        # Keep init behavior explicit in its owning subsystem.
        # Registries live here while World exposes compatibility aliases.
        self.living: dict[int, Creature] = {}
        self.issued_ids: set[int] = set()
        self.next_id_value = 1
        self._discard_callbacks: list[Callable[[int], None]] = []

    def synchronize_allocator(self, next_id: int) -> None:
        """Advance the stable identity allocator to at least ``next_id``.

        Parameters
        ----------
        next_id
            First identity that may be allocated after synchronization.

        Returns
        -------
        None
            Allocator state is advanced but never rewound.

        Raises
        ------
        ValueError
            If ``next_id`` is not a positive integer.
        """
        # Never permit allocator rewinds that could reuse a historical identity.
        if type(next_id) is not int or next_id < 1:
            raise ValueError("next creature ID must be a positive integer.")
        self.next_id_value = max(self.next_id_value, next_id)

    def allocate_id(self) -> int:
        """Reserve and return the next stable creature identity.

        Parameters
        ----------
        None
            This method receives no external parameters.

        Returns
        -------
        int
            Newly reserved identity that has never been issued before.
        """
        # Skip issued values defensively when restoring legacy allocator state.
        while self.next_id_value in self.issued_ids:
            self.next_id_value += 1
        identity = self.next_id_value
        self.next_id_value += 1
        return identity

    def add_discard_callback(self, callback: Callable[[int], None]) -> None:
        """Register one cache cleanup callback.

Parameters
----------
callback
    Function accepting a removed creature identity.

Returns
-------
None
    The callback is retained in registration order."""
        # Keep add discard callback behavior explicit in its owning subsystem.
        # Stable registration order makes cleanup deterministic and debuggable.
        self._discard_callbacks.append(callback)

    def register(self, creature: Creature) -> None:
        """Register a live creature and reject identity reuse.

Parameters
----------
creature
    Creature entering the live population.

Returns
-------
None
    Registry state is updated.

Raises
------
ValueError
    If another live creature or historical creature owns the identity."""
        # Keep register behavior explicit in its owning subsystem.
        # Validate against both current and session-historical identities.
        identity = creature_id(creature)
        existing = self.living.get(identity)
        if existing is not None and existing is not creature:
            raise ValueError(f"Duplicate live creature ID {identity}.")
        if existing is None and identity in self.issued_ids:
            raise ValueError(f"Creature ID {identity} was already issued in this session.")
        self.living[identity] = creature
        self.issued_ids.add(identity)
        self.synchronize_allocator(identity + 1)

    def unregister(self, creature: Creature) -> None:
        """Remove one live identity and clear subsystem caches.

Parameters
----------
creature
    Creature leaving the live population.

Returns
-------
None
    Live and transient state is discarded."""
        # Keep unregister behavior explicit in its owning subsystem.
        # Only the exact registered object may release an identity.
        identity = creature_id(creature)
        if self.living.get(identity) is creature:
            del self.living[identity]
        for callback in self._discard_callbacks:
            callback(identity)
