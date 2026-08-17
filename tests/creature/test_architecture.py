"""Dependency-boundary and import-compatibility checks for creature modules."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _imported_modules(path: Path) -> set[str]:
    """Collect absolute modules named by imports in one source file.

    Parameters
    ----------
    path
        Python source file whose imports are inspected.

    Returns
    -------
    set[str]
        Absolute import module names found in the syntax tree.
    """
    # AST inspection avoids importing optional simulation dependencies.
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class CreatureArchitectureTests(unittest.TestCase):
    """Protect one-way package boundaries introduced by the refactor."""

    def test_runtime_services_do_not_import_world(self) -> None:
        """Ensure composed runtime services depend only on narrow contexts.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Runtime imports are checked through assertions.
        """
        # A World import would recreate the cycle that service composition removes.
        runtime_root = REPOSITORY_ROOT / "src" / "creature" / "runtime"
        for path in runtime_root.glob("*.py"):
            self.assertNotIn("src.world", _imported_modules(path), str(path))

    def test_genotype_does_not_import_neat(self) -> None:
        """Ensure non-neural inheritance remains independent of NEAT.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Genotype imports are checked through assertions.
        """
        # Genotype mutation must be usable without loading neural dependencies.
        path = REPOSITORY_ROOT / "src" / "creature" / "genotype.py"
        imports = _imported_modules(path)
        self.assertFalse(any(name == "neat" or ".neat" in name for name in imports))

    def test_neat_package_does_not_import_genotype(self) -> None:
        """Ensure every NEAT implementation module has no genotype dependency.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Neural-core imports are checked through assertions.
        """
        # Composite compatibility belongs to the coordinator/species integration.
        neat_root = REPOSITORY_ROOT / "src" / "creature" / "neat"
        for path in neat_root.glob("*.py"):
            self.assertNotIn("src.creature.genotype", _imported_modules(path), str(path))

    def test_neural_controller_signatures_are_trait_free(self) -> None:
        """Ensure canonical controller parameters expose no creature traits.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Controller signatures are checked through assertions.
        """
        # Compatibility trait adapters live only in the documented root façade.
        path = REPOSITORY_ROOT / "src" / "creature" / "neat" / "controller.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden = {"Creature", "CreatureGenotype", "PhysicalTraits", "VisionTraits", "FlockingTraits"}
        controller = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NeatBrainController"
        )
        for method in controller.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for argument in (
                *method.args.posonlyargs,
                *method.args.args,
                *method.args.kwonlyargs,
            ):
                annotation = "" if argument.annotation is None else ast.unparse(argument.annotation)
                self.assertFalse(forbidden & set(annotation.replace("[", " ").replace("]", " ").replace("|", " ").split()))


if __name__ == "__main__":
    # Direct execution supports dependency-light architecture verification.
    unittest.main()
