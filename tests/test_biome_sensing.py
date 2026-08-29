from __future__ import annotations

from dataclasses import replace
from math import cos, isfinite, pi, sin

import numpy as np
import pytest

from configs.sim_config import BiomeConfig, BiomeSensorConfig
from src.biome import Biome, BiomeGenerationHandler, BiomeMap
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forward_distance", -1.0),
        ("forward_distance", float("inf")),
        ("side_offset", float("nan")),
        ("field_smoothing_sigma", -0.01),
        ("field_smoothing_sigma", float("inf")),
        ("gradient_contrast_gain", 0.0),
        ("gradient_contrast_gain", -1.0),
        ("gradient_contrast_gain", float("nan")),
    ],
)
def test_biome_sensor_config_rejects_invalid_values(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        BiomeSensorConfig(**{field: value})


def test_left_richer_positive_lateral() -> None:
    snapshot = BiomeSensorSnapshot.from_probe_samples(0.4, 0.6, 0.3)

    assert snapshot.lateral_gradient > 0.0


def test_right_richer_negative_lateral() -> None:
    snapshot = BiomeSensorSnapshot.from_probe_samples(0.4, 0.3, 0.6)

    assert snapshot.lateral_gradient < 0.0


def test_lateral_gradient_is_antisymmetric() -> None:
    left_richer = BiomeSensorSnapshot.from_probe_samples(0.4, 0.7, 0.2)
    right_richer = BiomeSensorSnapshot.from_probe_samples(0.4, 0.2, 0.7)

    assert left_richer.lateral_gradient == pytest.approx(
        -right_richer.lateral_gradient,
        abs=1e-15,
    )


def test_forward_gradient_reverses_when_here_and_ahead_are_exchanged() -> None:
    outward = BiomeSensorSnapshot.from_probe_samples(0.2, 0.6, 0.6)
    reversed_samples = BiomeSensorSnapshot.from_probe_samples(0.6, 0.2, 0.2)

    assert outward.forward_gradient == pytest.approx(
        -reversed_samples.forward_gradient,
        abs=1e-15,
    )


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


def test_extreme_gradients_remain_finite_and_strictly_bounded() -> None:
    zero_local_spike = BiomeSensorSnapshot.from_probe_samples(0.0, 1.0, 0.0)
    zero_local_right = BiomeSensorSnapshot.from_probe_samples(0.0, 0.0, 1.0)
    extreme_drop = BiomeSensorSnapshot.from_probe_samples(1.0, 0.0, 0.0)

    assert 0.99 < zero_local_spike.lateral_gradient < 1.0
    assert 0.99 < zero_local_spike.forward_gradient < 1.0
    assert -1.0 < zero_local_right.lateral_gradient < -0.99
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


def _step_biome_map(*, sigma: float) -> BiomeMap:
    biome_ids = np.array(
        [[Biome.PRAIRIE] * 20 + [Biome.FOREST] * 20],
        dtype=np.uint8,
    )
    return BiomeMap(
        biome_ids=biome_ids,
        render_rgba=np.zeros((1, 40, 4), dtype=np.uint8),
        world_bounds=(0.0, 0.0, 40.0, 1.0),
        area_shares={
            Biome.PRAIRIE: 0.5,
            Biome.BUSHES: 0.0,
            Biome.FOREST: 0.5,
        },
        spawn_weights={
            Biome.PRAIRIE: 0.0,
            Biome.BUSHES: 0.5,
            Biome.FOREST: 1.0,
        },
        uniform_spawn_chance=0.0,
        max_spawn_attempts=4,
        field_smoothing_sigma=sigma,
    )


def test_gaussian_smoothing_preserves_homogeneous_fields() -> None:
    homogeneous = replace(
        _step_biome_map(sigma=2.0),
        biome_ids=np.full((1, 40), int(Biome.FOREST), dtype=np.uint8),
    )

    assert np.array_equal(
        homogeneous._sensor_richness_grid,
        homogeneous._expected_density_grid,
    )


def test_gaussian_smoothing_spreads_step_only_over_finite_support() -> None:
    biome_map = _step_biome_map(sigma=2.0)
    sensed = biome_map._sensor_richness_grid[0]

    assert sensed[10] == 0.0
    assert 0.0 < sensed[19] < 0.5
    assert 0.5 < sensed[20] < 1.0
    assert sensed[29] == 1.0


def test_smoothed_field_is_deterministic_and_does_not_change_raw_density() -> None:
    raw = _step_biome_map(sigma=0.0)
    first = _step_biome_map(sigma=2.0)
    second = _step_biome_map(sigma=2.0)

    assert np.array_equal(first._sensor_richness_grid, second._sensor_richness_grid)
    assert np.array_equal(first._expected_density_grid, raw._expected_density_grid)
    assert first.expected_food_density_at(19.75, 0.5) == raw.expected_food_density_at(
        19.75,
        0.5,
    )
    assert first.sensed_food_richness_at(19.75, 0.5) != (
        first.expected_food_density_at(19.75, 0.5)
    )
    assert first._sensor_richness_grid.dtype == np.float64
    assert not first._sensor_richness_grid.flags.writeable


def test_spawn_weight_rebuild_refreshes_both_caches_without_map_artifacts() -> None:
    original = _step_biome_map(sigma=2.0)
    rebuilt = replace(
        original,
        spawn_weights={
            Biome.PRAIRIE: 1.0,
            Biome.BUSHES: 0.5,
            Biome.FOREST: 0.0,
        },
    )

    assert rebuilt.biome_ids is original.biome_ids
    assert rebuilt.render_rgba is original.render_rgba
    assert not np.array_equal(
        rebuilt._expected_density_grid,
        original._expected_density_grid,
    )
    assert not np.array_equal(
        rebuilt._sensor_richness_grid,
        original._sensor_richness_grid,
    )


def test_default_map_gradients_are_distributed_without_widespread_saturation() -> None:
    sensor_config = BiomeSensorConfig()
    biome_map = BiomeGenerationHandler(
        BiomeConfig(),
        sensor_config=sensor_config,
    ).generate(WORLD_BOUNDS)
    left, bottom, right, top = WORLD_BOUNDS
    lateral: list[float] = []
    forward: list[float] = []

    for x in np.linspace(left, right, 24, endpoint=False) + (right - left) / 48:
        for y in np.linspace(bottom, top, 18, endpoint=False) + (top - bottom) / 36:
            for heading in np.linspace(0.0, 2.0 * pi, 8, endpoint=False):
                hx, hy = cos(heading), sin(heading)
                lx, ly = -hy, hx
                f = sensor_config.forward_distance
                s = sensor_config.side_offset
                here = biome_map.sensed_food_richness_at(x, y)
                left_sample = biome_map.sensed_food_richness_at(
                    x + hx * f + lx * s,
                    y + hy * f + ly * s,
                )
                right_sample = biome_map.sensed_food_richness_at(
                    x + hx * f - lx * s,
                    y + hy * f - ly * s,
                )
                snapshot = BiomeSensorSnapshot.from_probe_samples(
                    here,
                    left_sample,
                    right_sample,
                    contrast_gain=sensor_config.gradient_contrast_gain,
                )
                lateral.append(snapshot.lateral_gradient)
                forward.append(snapshot.forward_gradient)

    for readings in (lateral, forward):
        magnitudes = np.abs(readings)
        assert np.mean(magnitudes < 0.01) < 0.25
        assert np.mean(magnitudes > 0.99) < 0.10
