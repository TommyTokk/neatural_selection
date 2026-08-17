"""Narrow structural contracts used by creature runtime services."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class CreatureRuntimeContext(Protocol):
    """Describe shared world-owned state without importing ``World``."""

    config: Any
    creatures: list[Any]
    foods: list[Any]
    space: Any
    rng: Any


class LifecycleContext(CreatureRuntimeContext, Protocol):
    """Describe world callbacks needed for physical birth and removal."""

    _next_creature_id_value: int
    _living_creatures: dict[int, Any]
    _issued_creature_ids: set[int]


class PerceptionContext(CreatureRuntimeContext, Protocol):
    """Describe environment data required by creature perception."""

    biome_map: Any
    pheromones: Any
    acoustics: Any
    environment_world_bounds: tuple[float, float, float, float]


class ActionContext(CreatureRuntimeContext, Protocol):
    """Describe physics callbacks required to execute neural actions."""

    fixed_timestep: float
    _simulation_step: int
    _scheduler_validation_failure_point: Callable[[str], None]


class SocialContext(CreatureRuntimeContext, Protocol):
    """Describe communication systems required by social behavior."""

    acoustics: Any
    pheromones: Any
    elapsed_time: float


class ResourceContext(CreatureRuntimeContext, Protocol):
    """Describe state required for creature resource transactions."""

    elapsed_time: float
    fitness: dict[int, Any]
    fitness_archive: dict[int, Any]


def creature_id(value: object) -> int:
    """Return and validate a stable integer creature identity.

Parameters
----------
value
    Creature-like object exposing ``creature_id``.

Returns
-------
int
    Stable integer identity.

Raises
------
TypeError
    If the identity is not a plain integer."""
    # Keep creature id behavior explicit in its owning subsystem.
    # Central validation prevents lifecycle and cache services from diverging.
    identity = getattr(value, "creature_id", None)
    if type(identity) is not int:
        raise TypeError("creature_id must be a stable integer.")
    return identity
