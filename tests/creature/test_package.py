"""Tests for canonical and historical creature imports."""

from __future__ import annotations

import unittest
from pathlib import Path
import subprocess
import sys


class CreaturePackageTest(unittest.TestCase):
    """Verify package exports and compatibility façades resolve consistently."""

    def test_legacy_and_canonical_types_are_identical(self) -> None:
        """Resolve historical imports to canonical package classes.
        
        Parameters
        ----------
        None
            This test receives no external parameters.
        
        Returns
        -------
        None
            Assertions validate compatibility identity.
        """
        # Keep the test legacy and canonical types are identical test intent explicit.
        # Pickle compatibility depends on module lookup returning these exact types.
        from src.creature import Creature, PhysicalTraits
        from src.creature.genotype import PhysicalTraits as CanonicalPhysicalTraits
        from src.creature.model import Creature as CanonicalCreature

        self.assertIs(Creature, CanonicalCreature)
        self.assertIs(PhysicalTraits, CanonicalPhysicalTraits)

    def test_package_import_does_not_eagerly_load_domain_modules(self) -> None:
        """Import the public façade without loading physics or NEAT modules.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            A clean interpreter verifies lazy package state.
        """
        # A subprocess prevents earlier compatibility imports from polluting state.
        repository_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import src.creature; "
                    "assert 'src.creature.model' not in sys.modules; "
                    "assert 'src.creature.neat.controller' not in sys.modules"
                ),
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_root_module_facades_resolve_domain_symbols(self) -> None:
        """Resolve old root modules to their relocated implementations.
        
        Parameters
        ----------
        None
            This test receives no external parameters.
        
        Returns
        -------
        None
            Assertions validate façade symbol identity.
        """
        # Keep the test root module facades resolve domain symbols test intent explicit.
        # Existing scripts may continue importing the historical module paths.
        from src.action import Action as LegacyAction
        from src.creature.action import Action
        from src.creature.vision import VisionSystem
        from src.vision import VisionSystem as LegacyVisionSystem

        self.assertIs(LegacyAction, Action)
        self.assertIs(LegacyVisionSystem, VisionSystem)
