"""Static documentation and explainability checks for creature-domain code."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import tokenize
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CREATURE_ROOT = REPOSITORY_ROOT / "src" / "creature"
CREATURE_TEST_ROOT = REPOSITORY_ROOT / "tests" / "creature"

# Keep this list explicit so integration changes outside the package are audited.
INTEGRATION_MANIFEST: dict[str, set[str]] = {
    "src/world.py": {
        "ArchivedCreatureTraits.genotype",
        "ArchivedCreatureTraits.from_creature",
        "World.__init__",
        "World.rebind_creature_services",
        "World._spawn_creature",
        "World._next_creature_id",
        "World._neutral_action",
        "World._ensure_communication_buffer_capacity",
        "World._mutated_child_traits",
        "World._mutated_child_traits_from_parent_values",
        "World._energy_demands_for",
        "World._update_metabolism_legacy_adapter",
        "World._complete_metabolism_update",
        "World._stage_final_reproductions",
        "World._commit_staged_reproductions",
        "World._prune_historical_archives",
        "World._archive_creature_traits",
        "World._recover_extinct_population",
        "World._prepare_reproduction_requests",
        "World._register_living_creature",
        "World._unregister_living_creature",
    },
    "src/persistence.py": {"PersistenceManager._restore_world"},
    "src/neat_controller.py": {
        "NeatBrainController.assign_initial_brains",
        "NeatBrainController.reset_for_new_sensing_epoch",
        "NeatBrainController.flocking_compatibility",
        "NeatBrainController._binary_species_compatibility",
        "NeatBrainController.create_child_brain",
        "NeatBrainController.create_mutated_brain_from_genome",
    },
}


def _callables(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Index functions and methods by their dotted lexical names.

    Parameters
    ----------
    tree
        Parsed Python syntax tree.

    Returns
    -------
    dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
        Mapping from dotted callable names to syntax nodes.
    """
    # Track class and function nesting so manifest names are stable and readable.
    indexed: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    class Visitor(ast.NodeVisitor):
        """Collect lexical callable names from one syntax tree."""

        def __init__(self) -> None:
            """Initialize an empty lexical-name stack.

            Parameters
            ----------
            None
                This initializer receives no external parameters.

            Returns
            -------
            None
                The visitor is ready to collect callables.
            """
            # A local stack avoids attaching traversal state to AST nodes.
            self.stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Visit a class while including its name in child callables.

            Parameters
            ----------
            node
                Class definition currently being traversed.

            Returns
            -------
            None
                Descendant callables are added to ``indexed``.
            """
            # Push and pop symmetrically so sibling names cannot leak together.
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Index a synchronous callable and inspect nested helpers.

            Parameters
            ----------
            node
                Function definition currently being traversed.

            Returns
            -------
            None
                The callable and nested definitions are indexed.
            """
            # Record before recursion so nested helpers receive the parent path.
            qualified = ".".join((*self.stack, node.name))
            indexed[qualified] = node
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Index an asynchronous callable using the synchronous path logic.

            Parameters
            ----------
            node
                Asynchronous function definition currently being traversed.

            Returns
            -------
            None
                The callable and nested definitions are indexed.
            """
            # Async definitions have identical naming and documentation rules.
            qualified = ".".join((*self.stack, node.name))
            indexed[qualified] = node
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    # Traverse once and return the completed lexical index.
    Visitor().visit(tree)
    return indexed


def _external_parameters(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Return signature parameter names excluding implicit receivers.

    Parameters
    ----------
    node
        Callable syntax node whose signature is inspected.

    Returns
    -------
    list[str]
        Positional, keyword-only, variadic, and mapping parameter names.
    """
    # Preserve signature order before appending variadic parameters.
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    names = [argument.arg for argument in arguments if argument.arg not in {"self", "cls"}]
    if node.args.vararg is not None:
        names.append(node.args.vararg.arg)
    if node.args.kwarg is not None:
        names.append(node.args.kwarg.arg)
    return names


def _comment_lines(source: str) -> set[int]:
    """Return source lines containing Python comment tokens.

    Parameters
    ----------
    source
        Python source text to tokenize.

    Returns
    -------
    set[int]
        One-based line numbers containing implementation comments.
    """
    # Token inspection ignores hash characters embedded inside string literals.
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return {token.start[0] for token in tokens if token.type == tokenize.COMMENT}


def _directly_raises(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a callable directly owns a raise statement.

    Parameters
    ----------
    node
        Callable syntax node whose validation behavior is inspected.

    Returns
    -------
    bool
        Whether a raise exists outside nested callables.
    """
    # A small visitor excludes failures intentionally owned by nested helpers.
    found = False

    class Visitor(ast.NodeVisitor):
        """Find direct raise statements without entering child callables."""

        def visit_Raise(self, child: ast.Raise) -> None:
            """Record a directly owned raise statement.

            Parameters
            ----------
            child
                Raise statement encountered during traversal.

            Returns
            -------
            None
                The enclosing ``found`` flag is updated.
            """
            # Only presence matters; exception names remain human documentation.
            nonlocal found
            found = True

        def visit_FunctionDef(self, child: ast.FunctionDef) -> None:
            """Enter only the root callable supplied to the helper.

            Parameters
            ----------
            child
                Function definition encountered during traversal.

            Returns
            -------
            None
                Root statements are visited while nested functions are skipped.
            """
            # Identity distinguishes the root node from lexically nested helpers.
            if child is node:
                self.generic_visit(child)

        def visit_AsyncFunctionDef(self, child: ast.AsyncFunctionDef) -> None:
            """Enter only the root asynchronous callable.

            Parameters
            ----------
            child
                Asynchronous function encountered during traversal.

            Returns
            -------
            None
                Root statements are visited while nested functions are skipped.
            """
            # Async ownership follows the same lexical rule as sync ownership.
            if child is node:
                self.generic_visit(child)

        def visit_Lambda(self, child: ast.Lambda) -> None:
            """Skip lambda bodies when assigning raise ownership.

            Parameters
            ----------
            child
                Lambda expression encountered during traversal.

            Returns
            -------
            None
                No descendant expressions are visited.
            """
            # Lambdas cannot contain raise statements and need no recursion.
            del child

    # Visit once and expose the ownership result to the documentation assertion.
    Visitor().visit(node)
    return found


class CreatureDocumentationTests(unittest.TestCase):
    """Enforce callable documentation across the refactored creature domain."""

    def _assert_callable_contract(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        comments: set[int],
    ) -> None:
        """Assert one callable has signature-aware docs and a code comment.

        Parameters
        ----------
        path
            Source path used in assertion messages.
        node
            Callable syntax node under inspection.
        comments
            One-based comment-token line numbers for the source file.

        Returns
        -------
        None
            Assertions report any documentation contract violation.
        """
        # Validate required NumPy/Pandas-style sections before parameter detail.
        document = ast.get_docstring(node, clean=True) or ""
        self.assertIn("Parameters\n----------", document, f"{path}:{node.lineno}")
        self.assertIn("Returns\n-------", document, f"{path}:{node.lineno}")
        for parameter in _external_parameters(node):
            self.assertIn(f"\n{parameter}\n", f"\n{document}\n", f"{path}:{node.lineno}")
        if not _external_parameters(node):
            self.assertIn("\nNone\n", f"\n{document}\n", f"{path}:{node.lineno}")
        if _directly_raises(node):
            self.assertIn("Raises\n------", document, f"{path}:{node.lineno}")

        # Require a genuine comment token within the callable's lexical extent.
        end_line = node.end_lineno or node.lineno
        self.assertTrue(
            any(node.lineno <= line <= end_line for line in comments),
            f"{path}:{node.lineno} has no implementation comment",
        )

    def test_creature_package_callables_are_documented(self) -> None:
        """Check every callable below ``src/creature``.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Every discovered callable is checked through assertions.
        """
        # Sort paths to keep the first failure deterministic across platforms.
        for path in sorted(CREATURE_ROOT.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            comments = _comment_lines(source)
            for node in _callables(ast.parse(source)).values():
                self._assert_callable_contract(path, node, comments)

    def test_manifested_integration_callables_are_documented(self) -> None:
        """Check explicitly touched callables outside the creature package.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Every manifested integration callable is checked through assertions.
        """
        # The manifest prevents broad World scans from coupling unrelated systems.
        for relative_path, expected_names in INTEGRATION_MANIFEST.items():
            path = REPOSITORY_ROOT / relative_path
            source = path.read_text(encoding="utf-8")
            indexed = _callables(ast.parse(source))
            comments = _comment_lines(source)
            self.assertTrue(expected_names <= indexed.keys())
            for name in sorted(expected_names):
                self._assert_callable_contract(path, indexed[name], comments)

    def test_moved_creature_tests_are_documented(self) -> None:
        """Check callables in the focused creature test package.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Every moved test callable is checked through assertions.
        """
        # Tests follow the production documentation rule to preserve intent.
        for path in sorted(CREATURE_TEST_ROOT.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            comments = _comment_lines(source)
            for node in _callables(ast.parse(source)).values():
                self._assert_callable_contract(path, node, comments)


if __name__ == "__main__":
    # Direct execution remains useful in minimal environments without pytest.
    unittest.main()
