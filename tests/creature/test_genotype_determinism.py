"""Determinism checks for genotype colour extraction from ``World``."""

from __future__ import annotations

from colorsys import hsv_to_rgb, rgb_to_hsv
from random import Random
from types import SimpleNamespace
import unittest

from src.creature.genotype import GenotypeManager


def _food_like(config: object, color: tuple[float, float, float]) -> bool:
    """Evaluate the version-23 food-colour exclusion predicate.

    Parameters
    ----------
    config
        Configuration exposing ``theme.food_fill``.
    color
        Normalized RGB colour.

    Returns
    -------
    bool
        Whether the colour lies within the historical food-colour radius.
    """
    # Keep the reference independent from the production helper under test.
    food_red, food_green, food_blue = config.theme.food_fill[:3]
    red, green, blue = (channel * 255.0 for channel in color)
    return (
        (red - food_red) ** 2
        + (green - food_green) ** 2
        + (blue - food_blue) ** 2
    ) < 70.0**2


def _legacy_mutated_color(
    config: object,
    parent: tuple[int, int, int],
    rng: Random,
) -> tuple[int, int, int]:
    """Reproduce the pre-refactor descendant-colour algorithm.

    Parameters
    ----------
    config
        Configuration used for food-colour avoidance.
    parent
        Parent RGB colour.
    rng
        Reference random generator.

    Returns
    -------
    tuple[int, int, int]
        Historical mutated RGB colour.
    """
    # Draw in the exact order used by version-23 continuation.
    red, green, blue = parent
    hue, saturation, value = rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    hue = (hue + rng.uniform(-0.035, 0.035)) % 1.0
    saturation = max(0.48, min(0.82, saturation + rng.uniform(-0.06, 0.06)))
    value = max(0.62, min(0.92, value + rng.uniform(-0.05, 0.05)))
    if _food_like(config, hsv_to_rgb(hue, saturation, value)):
        hue = (hue + 0.22) % 1.0
    return tuple(int(channel * 255) for channel in hsv_to_rgb(hue, saturation, value))


def _legacy_new_species_color(
    config: object,
    parent: tuple[int, int, int],
    rng: Random,
) -> tuple[int, int, int]:
    """Reproduce the pre-refactor founder-colour search.

    Parameters
    ----------
    config
        Configuration used for food-colour avoidance.
    parent
        Parent RGB colour.
    rng
        Reference random generator.

    Returns
    -------
    tuple[int, int, int]
        Historical new-species RGB colour.
    """
    # Preserve attempt count and fallback order because both affect RNG state.
    red, green, blue = parent
    parent_hue, _, _ = rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    for _ in range(32):
        candidate = hsv_to_rgb(
            (parent_hue + rng.uniform(0.18, 0.82)) % 1.0,
            rng.uniform(0.7, 1.0),
            rng.uniform(0.8, 1.0),
        )
        color = tuple(int(channel * 255) for channel in candidate)
        if not _food_like(config, tuple(channel / 255.0 for channel in color)):
            return color
    for shift in (0.5, 1.0 / 3.0, 2.0 / 3.0):
        candidate = hsv_to_rgb((parent_hue + shift) % 1.0, 0.85, 0.9)
        color = tuple(int(channel * 255) for channel in candidate)
        if not _food_like(config, tuple(channel / 255.0 for channel in color)):
            return color
    return tuple(
        int(channel * 255)
        for channel in hsv_to_rgb((parent_hue + 0.5) % 1.0, 1.0, 1.0)
    )


class GenotypeColorDeterminismTests(unittest.TestCase):
    """Verify extracted colour methods preserve output and RNG position."""

    def setUp(self) -> None:
        """Create a dependency-light genotype manager fixture.

        Parameters
        ----------
        None
            This setup method receives no external parameters.

        Returns
        -------
        None
            A manager and theme configuration are stored on the test instance.
        """
        # Colour operations only require the configured food-fill colour.
        self.config = SimpleNamespace(theme=SimpleNamespace(food_fill=(72, 170, 82)))
        self.manager = GenotypeManager(self.config, ((120, 90, 210),))

    def test_descendant_colour_preserves_rng_consumption(self) -> None:
        """Compare descendant colour and final RNG state to version 23.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify output and RNG state identity.
        """
        # Independent generators reveal both visible and continuation drift.
        actual_rng = Random(78123)
        expected_rng = Random(78123)
        parent = (94, 181, 103)
        self.assertEqual(
            _legacy_mutated_color(self.config, parent, expected_rng),
            self.manager.mutate_color(parent, actual_rng),
        )
        self.assertEqual(expected_rng.getstate(), actual_rng.getstate())

    def test_founder_colour_preserves_rng_consumption(self) -> None:
        """Compare founder colour and final RNG state to version 23.

        Parameters
        ----------
        None
            This test receives no external parameters.

        Returns
        -------
        None
            Assertions verify output and RNG state identity.
        """
        # The loop test protects attempt count as well as individual ranges.
        actual_rng = Random(91234)
        expected_rng = Random(91234)
        parent = (120, 90, 210)
        self.assertEqual(
            _legacy_new_species_color(self.config, parent, expected_rng),
            self.manager.new_species_color(parent, actual_rng),
        )
        self.assertEqual(expected_rng.getstate(), actual_rng.getstate())


if __name__ == "__main__":
    # Direct execution keeps this regression test usable without pytest.
    unittest.main()
