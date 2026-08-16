from __future__ import annotations

from dataclasses import replace
from math import isfinite

import pytest

from configs.sim_config import BiomeConfig
from src.biome import Biome, BiomeGenerationHandler
from src.vision import BiomeSensorSnapshot


WORLD_BOUNDS = (-1600.0, -1100.0, 1600.0, 1100.0)


def test_prairie_baseline_greater_than_zero() -> None:
    biome_map = BiomeGenerationHandler(BiomeConfig()).generate(WORLD_BOUNDS)
    prairie_density = float(
        biome_map._expected_density_grid[biome_map.biome_ids == int(Biome.PRAIRIE)][0]
    )

    assert prairie_density > 0.0
    assert prairie_density == pytest.approx(
        0.10 + 0.90 * (0.25 / 2.75)
    )


def test_homogeneous_field_zero_gradients() -> None:
    snapshot = BiomeSensorSnapshot.from_probe_samples(0.4, 0.4, 0.4)

    assert snapshot.local_richness == 0.4
    assert snapshot.lateral_gradient == 0.0
    assert snapshot.forward_gradient == 0.0


def test_left_richer_positive_lateral() -> None:
    snapshot = BiomeSensorSnapshot.from_probe_samples(0.4, 0.6, 0.3)

    assert snapshot.lateral_gradient > 0.0


def test_right_richer_negative_lateral() -> None:
    snapshot = BiomeSensorSnapshot.from_probe_samples(0.4, 0.3, 0.6)

    assert snapshot.lateral_gradient < 0.0


def test_scale_invariance() -> None:
    low_density = BiomeSensorSnapshot.from_probe_samples(0.1, 0.12, 0.1)
    high_density = BiomeSensorSnapshot.from_probe_samples(0.5, 0.6, 0.5)

    assert low_density.lateral_gradient == pytest.approx(
        high_density.lateral_gradient,
        abs=0.002,
    )
    assert low_density.forward_gradient == pytest.approx(
        high_density.forward_gradient,
        abs=0.002,
    )


def test_gradient_clamping() -> None:
    zero_local_spike = BiomeSensorSnapshot.from_probe_samples(0.0, 1.0, 0.0)
    zero_local_right = BiomeSensorSnapshot.from_probe_samples(0.0, 0.0, 1.0)
    extreme_drop = BiomeSensorSnapshot.from_probe_samples(1.0, 0.0, 0.0)

    assert zero_local_spike.lateral_gradient == 1.0
    assert zero_local_spike.forward_gradient == 1.0
    assert zero_local_right.lateral_gradient == -1.0
    for snapshot in (zero_local_spike, zero_local_right, extreme_drop):
        for value in (
            snapshot.local_richness,
            snapshot.lateral_gradient,
            snapshot.forward_gradient,
        ):
            assert isfinite(value)
        assert -1.0 <= snapshot.lateral_gradient <= 1.0
        assert -1.0 <= snapshot.forward_gradient <= 1.0


def test_uniform_probability_propagates_and_rebuilds_density_cache() -> None:
    config = BiomeConfig(uniform_spawn_chance=0.25)
    biome_map = BiomeGenerationHandler(config).generate(WORLD_BOUNDS)
    prairie_mask = biome_map.biome_ids == int(Biome.PRAIRIE)
    initial_density = float(biome_map._expected_density_grid[prairie_mask][0])

    assert biome_map.uniform_spawn_chance == 0.25
    assert initial_density == pytest.approx(0.25 + 0.75 * (0.25 / 2.75))

    rebuilt = replace(biome_map, uniform_spawn_chance=0.5)
    rebuilt_density = float(rebuilt._expected_density_grid[prairie_mask][0])

    assert rebuilt_density == pytest.approx(0.5 + 0.5 * (0.25 / 2.75))
    assert rebuilt_density > initial_density
