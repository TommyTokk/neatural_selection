"""Lazy public façade for the creature domain package.

Historical checkpoints resolve several classes through ``src.creature``. The
lazy export table preserves those names without importing physics, NEAT, and
runtime modules while the package itself is being initialized.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "ActivityDiagnostics": ("src.creature.model", "ActivityDiagnostics"),
    "Color": ("src.creature.genotype", "Color"),
    "Creature": ("src.creature.model", "Creature"),
    "CreatureActionService": (
        "src.creature.runtime.actions",
        "CreatureActionService",
    ),
    "CreatureEvolutionCoordinator": (
        "src.creature.evolution",
        "CreatureEvolutionCoordinator",
    ),
    "EvolutionTransaction": (
        "src.creature.evolution",
        "EvolutionTransaction",
    ),
    "CreatureFactory": ("src.creature.factory", "CreatureFactory"),
    "CreatureGenotype": ("src.creature.genotype", "CreatureGenotype"),
    "CreatureLifecycleService": (
        "src.creature.runtime.lifecycle",
        "CreatureLifecycleService",
    ),
    "CreaturePerceptionService": (
        "src.creature.runtime.perception",
        "CreaturePerceptionService",
    ),
    "CreatureResourceService": (
        "src.creature.runtime.resources",
        "CreatureResourceService",
    ),
    "CreatureSocialService": (
        "src.creature.runtime.social",
        "CreatureSocialService",
    ),
    "FlockingTraits": ("src.creature.genotype", "FlockingTraits"),
    "GenotypeManager": ("src.creature.genotype", "GenotypeManager"),
    "GenotypeMutationResult": (
        "src.creature.genotype",
        "GenotypeMutationResult",
    ),
    "LedgerDiagnostics": ("src.creature.model", "LedgerDiagnostics"),
    "LineageInfo": ("src.creature.genotype", "LineageInfo"),
    "OffspringPlan": ("src.creature.evolution", "OffspringPlan"),
    "PhysicalTraits": ("src.creature.genotype", "PhysicalTraits"),
    "TraitMutationDelta": ("src.creature.genotype", "TraitMutationDelta"),
    "VisionTraits": ("src.creature.genotype", "VisionTraits"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load and cache one public creature-domain symbol on first access.

    Parameters
    ----------
    name
        Public attribute requested from :mod:`src.creature`.

    Returns
    -------
    Any
        Exported class or type alias resolved from its owning module.

    Raises
    ------
    AttributeError
        If ``name`` is not part of the supported public façade.

    Notes
    -----
    Caching the resolved object preserves normal module attribute semantics and
    avoids repeating import lookups after the first access.
    """
    # Reject unknown names before importing so typos have normal module errors.
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Import only the owning module, then cache the compatibility symbol.
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return discoverable package attributes including lazy exports.

    Parameters
    ----------
    None
        This callable receives no external parameters.

    Returns
    -------
    list[str]
        Sorted names available through normal or lazy module lookup.
    """
    # Merge globals with the export table for IDE and REPL discovery.
    return sorted({*globals(), *_EXPORTS})
