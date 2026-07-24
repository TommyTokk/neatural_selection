from __future__ import annotations

import ast
from pathlib import Path
import unittest


class UiDocumentationTest(unittest.TestCase):
    """Enforce the UI package's documented-callable convention."""

    def test_every_ui_function_and_method_has_a_docstring(self) -> None:
        """Require a docstring on every function, method, and nested helper."""
        ui_root = Path(__file__).resolve().parents[2] / "src" / "ui"
        missing: list[str] = []

        for path in sorted(ui_root.rglob("*.py")):
            module = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(module):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if ast.get_docstring(node) is None:
                    relative = path.relative_to(ui_root)
                    missing.append(f"{relative}:{node.lineno} {node.name}")

        self.assertEqual([], missing)
