from __future__ import annotations

import unittest


class UiPackageExportTest(unittest.TestCase):
    """Verify lazy package exports remain stable for application callers."""

    def test_public_exports_resolve_without_import_cycle(self) -> None:
        """Load the public façade, views, renderer, and layout helpers."""
        from src.ui import (
            EnvironmentRenderer,
            ScreenLayout,
            StartMenuView,
            UiRenderer,
            build_screen_layout,
        )

        self.assertEqual(UiRenderer.__name__, "UiRenderer")
        self.assertEqual(EnvironmentRenderer.__name__, "EnvironmentRenderer")
        self.assertEqual(StartMenuView.__name__, "StartMenuView")
        self.assertEqual(ScreenLayout.__name__, "ScreenLayout")
        self.assertTrue(callable(build_screen_layout))
