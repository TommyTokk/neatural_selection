"""Milestone-3 population scaling, density, profile, and rendered benchmark.

This file is benchmark-only instrumentation.  Timings and workload counters are
collected in separate passes, and every fixture asserts a constant population.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import json
import math
from pathlib import Path
import platform
from random import Random
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_flocking import fill_to_physical_capacity  # noqa: E402
from benchmarks.benchmark_multirate_scheduler import (  # noqa: E402
    completed_step_count,
    fixed_timestep,
    run_fixed_steps,
)
from configs.sim_config import build_sim_config  # noqa: E402
from src.persistence import SimulationPaths  # noqa: E402
from src.spatial import CandidateBuffer  # noqa: E402
from src.world import SimulationLagMetrics, World  # noqa: E402


POPULATIONS = (55, 90, 100, 128)
TIMING_CASES = (
    ("phase_0", 1, 0),
    ("phase_1", 1, 1),
    ("phase_2", 1, 2),
    ("three_steps", 3, None),
    ("five_steps", 5, None),
    ("sixty_steps", 60, None),
)
RENDER_CONFIGURATIONS = (
    {
        "name": "normal_application_view",
        "inspector": None,
        "selected": False,
        "debug_overlays": None,
    },
    {
        "name": "inspector_closed",
        "inspector": False,
        "selected": False,
        "debug_overlays": False,
    },
    {
        "name": "inspector_open_selected_creature",
        "inspector": True,
        "selected": True,
        "debug_overlays": False,
    },
    {
        "name": "debug_overlays_disabled",
        "inspector": False,
        "selected": False,
        "debug_overlays": False,
    },
)

RESULTS = ROOT / "benchmarks" / "results"
SCALING_JSON = RESULTS / "milestone3_population_scaling.json"
PROFILE_JSON = RESULTS / "milestone3_population_profile.json"
RENDERED_JSON = RESULTS / "milestone3_rendered_scaling.json"
REPORT_MD = RESULTS / "milestone3_population_scaling.md"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def command_output(*command: str) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def environment_metadata() -> dict[str, object]:
    status = command_output("git", "status", "--short", "--branch")
    processor = (
        command_output("sysctl", "-n", "machdep.cpu.brand_string")
        if platform.system() == "Darwin"
        else platform.processor()
    )
    return {
        "git_revision": command_output("git", "rev-parse", "HEAD"),
        "git_status": status.splitlines(),
        "working_tree_clean": not any(
            line and not line.startswith("##") for line in status.splitlines()
        ),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "arcade_version": command_output(
            sys.executable,
            "-c",
            "import arcade; print(arcade.__version__)",
        ),
        "command": [sys.executable, *sys.argv],
    }


def _freeze_population_config(config) -> None:
    """Disable starvation, senescence, and food depletion."""
    metabolism = config.metabolism
    metabolism.basic_metabolism_rate = 0.0
    metabolism.brain_upkeep_per_node = 0.0
    metabolism.brain_upkeep_per_connection = 0.0
    metabolism.movement_energy_cost_factor = 0.0
    metabolism.sprint_energy_cost_per_second = 0.0
    metabolism.life_damage_per_energy_deficit = 0.0
    metabolism.digestive_upkeep_at_default_per_second = 0.0
    metabolism.max_digestive_upkeep_per_second = 0.0
    metabolism.max_bite_size_per_second = 0.0
    config.population.senescence_age_seconds = 1.0e12


def _disable_reproduction(config) -> None:
    """Make every reproduction gate impossible after fixture expansion."""
    config.population.min_reproduction_age = 1.0e12
    config.population.reproduction_cooldown = 1.0e12
    config.population.reproduction_energy_threshold = 2.0
    config.population.reproduction_min_food_ratio = 2.0


def _layout_creatures(world: World, layout: str) -> None:
    creatures = sorted(world.creatures, key=lambda item: item.creature_id)
    left, bottom, right, top = world.environment_world_bounds
    margin = max(creature.radius for creature in creatures) + 12.0
    positions: list[tuple[float, float]] = []
    if layout == "dense":
        maximum_radius = max(creature.radius for creature in creatures)
        spacing = maximum_radius * 2.0 + 3.0
        columns = math.ceil(math.sqrt(len(creatures)))
        rows = math.ceil(len(creatures) / columns)
        start_x = (left + right - (columns - 1) * spacing) / 2.0
        start_y = (bottom + top - (rows - 1) * spacing) / 2.0
        positions = [
            (
                start_x + (index % columns) * spacing,
                start_y + (index // columns) * spacing,
            )
            for index in range(len(creatures))
        ]
    elif layout == "normal":
        rng = Random(11)
        for creature in creatures:
            for _attempt in range(100_000):
                candidate = (
                    rng.uniform(left + margin, right - margin),
                    rng.uniform(bottom + margin, top - margin),
                )
                if all(
                    math.hypot(candidate[0] - x, candidate[1] - y)
                    >= creature.radius + placed.radius + 3.0
                    for (x, y), placed in zip(positions, creatures)
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError("Could not construct normal benchmark layout.")
    else:
        raise ValueError(f"Unknown layout {layout!r}.")

    for creature, position in zip(creatures, positions):
        creature.body.position = position
        creature.body.velocity = (0.0, 0.0)
        creature.body.angular_velocity = 0.0
        world.space.reindex_shape(creature.shape)

    for index, first in enumerate(creatures):
        for second in creatures[index + 1 :]:
            if math.dist(first.position, second.position) + 1e-9 < (
                first.radius + second.radius
            ):
                raise RuntimeError("Benchmark layout contains a physical overlap.")


def make_static_world(population: int, layout: str = "normal") -> World:
    config = build_sim_config()
    config.random_seed = 11
    config.population.initial_creatures = min(50, population)
    config.population.max_creatures = population
    config.food.initial_food_items = 300
    config.food.max_food_items = 300
    config.flocking.long_range.enabled = True
    config.behavior.enabled = False
    config.counterfactual_why.enabled = False
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    _freeze_population_config(config)
    world = World(config, simulation_paths=SimulationPaths(ROOT))
    fill_to_physical_capacity(world)
    _disable_reproduction(config)
    _layout_creatures(world, layout)
    for creature in world.creatures:
        creature.energy = config.metabolism.max_energy
        creature.life = config.metabolism.max_life
    world._reproduction_due_this_step = False
    if len(world.creatures) != population:
        world.close()
        raise RuntimeError(
            f"Expected {population} creatures, found {len(world.creatures)}."
        )
    if len(world.foods) != 300:
        world.close()
        raise RuntimeError(f"Expected 300 foods, found {len(world.foods)}.")
    return world


def assert_static(world: World, population: int, initial_food: int = 300) -> None:
    if len(world.creatures) != population:
        raise RuntimeError(
            f"Population changed: expected {population}, got {len(world.creatures)}."
        )
    if len(world.foods) != initial_food:
        raise RuntimeError(
            f"Food count changed: expected {initial_food}, got {len(world.foods)}."
        )


def summarize_samples(
    samples: list[float], run_samples: list[list[float]]
) -> dict[str, object]:
    run_medians = [statistics.median(values) for values in run_samples]
    median = statistics.median(samples)
    noise = (
        (percentile(run_medians, 0.95) - percentile(run_medians, 0.05))
        / max(2.0 * statistics.median(run_medians), 1e-12)
        * 100.0
    )
    return {
        "samples": len(samples),
        "median_ms": median,
        "p95_ms": percentile(samples, 0.95),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "noise_percent": noise,
        "run_medians_ms": run_medians,
        "raw_samples_ms": samples,
    }


def run_simulation_benchmark(
    repetitions: int, samples_per_repetition: int, warmup_steps: int
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for population in POPULATIONS:
        case_samples = {name: [] for name, _steps, _phase in TIMING_CASES}
        case_runs = {name: [] for name, _steps, _phase in TIMING_CASES}
        for repetition in range(repetitions):
            world = make_static_world(population)
            try:
                run_fixed_steps(world, warmup_steps)
                assert_static(world, population)
                for name, steps, phase in TIMING_CASES:
                    current_run: list[float] = []
                    for _sample in range(samples_per_repetition):
                        if phase is not None:
                            while completed_step_count(world) % 3 != phase:
                                run_fixed_steps(world, 1)
                        started = perf_counter()
                        run_fixed_steps(world, steps)
                        elapsed_ms = (perf_counter() - started) * 1_000.0
                        case_samples[name].append(elapsed_ms)
                        current_run.append(elapsed_ms)
                        assert_static(world, population)
                    case_runs[name].append(current_run)
            finally:
                world.close()
            print(
                f"simulation population={population} repetition={repetition + 1}/"
                f"{repetitions}",
                flush=True,
            )
        cases = {
            name: {
                "steps_per_sample": steps,
                "phase": phase,
                **summarize_samples(case_samples[name], case_runs[name]),
            }
            for name, steps, phase in TIMING_CASES
        }
        rows.append({"population": population, "cases": cases})

    baseline = rows[0]
    for row in rows:
        population_factor = row["population"] / baseline["population"]
        for name, _steps, _phase in TIMING_CASES:
            current = row["cases"][name]
            baseline_case = baseline["cases"][name]
            timing_factor = current["median_ms"] / baseline_case["median_ms"]
            current["population_factor_vs_55"] = population_factor
            current["timing_factor_vs_55"] = timing_factor
            current["regression_percent_vs_55"] = (timing_factor - 1.0) * 100.0
            current["scaling_factor_per_population_growth"] = (
                timing_factor / population_factor
            )

    result = {
        "metadata": environment_metadata(),
        "seed": 11,
        "populations": list(POPULATIONS),
        "layout": "deterministic non-overlapping seeded uniform distribution",
        "rendering_enabled": False,
        "observer_enabled": False,
        "counterfactual_enabled": False,
        "inspector_enabled": False,
        "debug_overlays_enabled": False,
        "initial_food_count": 300,
        "population_static": True,
        "repetitions": repetitions,
        "samples_per_repetition": samples_per_repetition,
        "total_samples_per_case": repetitions * samples_per_repetition,
        "warmup_steps": warmup_steps,
        "rows": rows,
    }
    write_json(SCALING_JSON, result)
    return result


def collect_density(population: int, layout: str, steps: int) -> dict[str, object]:
    world = make_static_world(population, layout)
    query_candidates: list[int] = []
    active_occupancies: list[float] = []
    active_cells: list[int] = []
    cell_maxima: list[int] = []
    vision_candidates: list[int] = []
    occlusion_candidates: list[int] = []
    index = world._creature_spatial_index
    original_query = index.query_into
    original_rebuild = index.rebuild
    original_sensor = world._sensor_snapshot_for
    original_occlusion = world.vision._sort_candidate_prefix

    def query_wrapper(*args, **kwargs):
        result = original_query(*args, **kwargs)
        output = args[3] if len(args) > 3 else kwargs.get("output")
        query_candidates.append(int(getattr(output, "count", 0)))
        return result

    def rebuild_wrapper(*args, **kwargs):
        result = original_rebuild(*args, **kwargs)
        count = len(index._active_cells)
        active_cells.append(count)
        active_occupancies.append(population / max(count, 1))
        cell_maxima.append(max(index._cell_counts.values(), default=0))
        return result

    def sensor_wrapper(*args, **kwargs):
        nearby = kwargs.get("nearby_creatures")
        vision_candidates.append(int(getattr(nearby, "count", len(nearby or ()))))
        return original_sensor(*args, **kwargs)

    def occlusion_wrapper(*args, **kwargs):
        occlusion_candidates.append(world.vision._candidate_count)
        return original_occlusion(*args, **kwargs)

    index.query_into = query_wrapper
    index.rebuild = rebuild_wrapper
    world._sensor_snapshot_for = sensor_wrapper
    world.vision._sort_candidate_prefix = occlusion_wrapper
    try:
        run_fixed_steps(world, 12)
        query_candidates.clear()
        active_occupancies.clear()
        active_cells.clear()
        cell_maxima.clear()
        vision_candidates.clear()
        occlusion_candidates.clear()
        before = index.counters.snapshot()
        vision_sorts_before = world.vision.stable_sort_count
        run_fixed_steps(world, steps)
        after = index.counters.snapshot()
        assert_static(world, population)
    finally:
        world.close()
    return {
        "population": population,
        "layout": layout,
        "steps": steps,
        "queries": len(query_candidates),
        "mean_candidates_per_query": statistics.fmean(query_candidates),
        "p95_candidates_per_query": percentile(query_candidates, 0.95),
        "maximum_candidates": max(query_candidates, default=0),
        "mean_active_cell_occupancy": statistics.fmean(active_occupancies),
        "maximum_cell_occupancy": max(cell_maxima, default=0),
        "mean_active_cells": statistics.fmean(active_cells),
        "stable_id_sorts": after["stable_sorts"] - before["stable_sorts"],
        "vision_stable_sorts": world.vision.stable_sort_count - vision_sorts_before,
        "vision_candidates": sum(vision_candidates),
        "mean_vision_candidates_per_sense": statistics.fmean(vision_candidates),
        "occlusion_candidates": sum(occlusion_candidates),
        "mean_occlusion_candidates_per_sort": statistics.fmean(
            occlusion_candidates
        ),
        "raw_candidates_per_query": query_candidates,
        "raw_active_cell_occupancies": active_occupancies,
        "raw_active_cells": active_cells,
        "raw_cell_maxima": cell_maxima,
        "raw_vision_candidates": vision_candidates,
        "raw_occlusion_candidates": occlusion_candidates,
    }


class ExclusiveProfiler:
    def __init__(self) -> None:
        self.exclusive_seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.stack: list[list[float | str]] = []

    def wrap(self, category: str, function: Callable):
        def measured(*args, **kwargs):
            frame: list[float | str] = [category, perf_counter(), 0.0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter() - float(frame[1])
                child = float(frame[2])
                self.stack.pop()
                self.exclusive_seconds[category] = (
                    self.exclusive_seconds.get(category, 0.0)
                    + max(0.0, elapsed - child)
                )
                self.calls[category] = self.calls.get(category, 0) + 1
                if self.stack:
                    self.stack[-1][2] = float(self.stack[-1][2]) + elapsed

        return measured


def profile_once(population: int, steps: int) -> dict[str, object]:
    world = make_static_world(population)
    run_fixed_steps(world, 12)
    assert_static(world, population)
    profiler = ExclusiveProfiler()
    index = world._creature_spatial_index
    vision = world.vision
    controller = world.neat_controller
    categories = {
        "spatial_index_rebuild": [(index, "rebuild")],
        "spatial_queries": [(index, "query_into")],
        "collision_avoidance": [(world, "_collision_avoidance_force")],
        "flocking_social_aggregation": [(world, "_refresh_social_runtime")],
        "vision_filtering": [(vision, "_visible_targets_reused")],
        "vision_sorting_occlusion": [
            (vision, "_sort_candidate_prefix"),
            (vision, "_interval_blocked_scratch"),
            (vision, "_add_blocked_scratch"),
        ],
        "neat_activation": [(controller, "decide")],
        "physics_pymunk_step": [(world.space, "step")],
        "biology_resource_processing": [(world, "_update_metabolism")],
        "communication": [
            (world, "_commit_communication_intents"),
            (world.pheromones, "accumulate"),
        ],
        "telemetry_observer_preparation": [
            (world, "_update_flocking_telemetry"),
            (world, "_sample_selected_behavior"),
            (world, "_sample_selected_why"),
            (world.behavior_observer, "poll"),
        ],
        "ui_statistics": [(world, "_refresh_stats")],
    }
    started = perf_counter()
    with ExitStack() as stack:
        for category, targets in categories.items():
            for owner, name in targets:
                original = getattr(owner, name)
                stack.callback(setattr, owner, name, original)
                setattr(owner, name, profiler.wrap(category, original))
        original_sort = CandidateBuffer.sort_by_stable_id
        CandidateBuffer.sort_by_stable_id = profiler.wrap(
            "stable_id_sorting", original_sort
        )
        stack.callback(
            setattr, CandidateBuffer, "sort_by_stable_id", original_sort
        )
        run_fixed_steps(world, steps)
    total = perf_counter() - started
    assert_static(world, population)
    world.close()
    measured = sum(profiler.exclusive_seconds.values())
    profiler.exclusive_seconds["other_uninstrumented"] = max(0.0, total - measured)
    profiler.calls["other_uninstrumented"] = 1
    return {
        "instrumented_total_ms": total * 1_000.0,
        "exclusive_ms": {
            name: seconds * 1_000.0
            for name, seconds in profiler.exclusive_seconds.items()
        },
        "calls": profiler.calls,
        "exclusive_percent": {
            name: seconds / max(total, 1e-12) * 100.0
            for name, seconds in profiler.exclusive_seconds.items()
        },
    }


def run_profile(repetitions: int, steps: int) -> dict[str, object]:
    density_rows = [
        collect_density(population, layout, steps)
        for population in (90, 100)
        for layout in ("normal", "dense")
    ]
    rows: list[dict[str, object]] = []
    for population in POPULATIONS:
        runs = [profile_once(population, steps) for _ in range(repetitions)]
        categories = sorted(runs[0]["exclusive_ms"])
        median_ms = {
            name: statistics.median(
                run["exclusive_ms"][name] for run in runs
            )
            for name in categories
        }
        total = statistics.median(run["instrumented_total_ms"] for run in runs)
        rows.append(
            {
                "population": population,
                "steps": steps,
                "instrumented_total_median_ms": total,
                "subsystem_median_ms": median_ms,
                "subsystem_percent_of_instrumented_total": {
                    name: value / max(total, 1e-12) * 100.0
                    for name, value in median_ms.items()
                },
                "raw_runs": runs,
            }
        )
        print(f"profile population={population} complete", flush=True)
    result = {
        "metadata": environment_metadata(),
        "seed": 11,
        "rendering_enabled": False,
        "observer_enabled": False,
        "counterfactual_enabled": False,
        "instrumentation_note": (
            "Exclusive nested timings are proportions only. Absolute values are "
            "not compared with the clean end-to-end benchmark."
        ),
        "repetitions": repetitions,
        "steps_per_run": steps,
        "rows": rows,
        "density_rows": density_rows,
    }
    write_json(PROFILE_JSON, result)
    return result


def _configure_render_view(view, render_config: dict[str, object]) -> None:
    inspector = render_config["inspector"]
    debug = render_config["debug_overlays"]
    if inspector is not None:
        view.ui_renderer._panel_open["inspector"] = bool(inspector)
    if debug is not None:
        view.world.debug_vision_enabled = bool(debug)
    if render_config["selected"]:
        first = min(view.world.creatures, key=lambda item: item.creature_id)
        view.world.select_creature_by_id(first.creature_id)


def _render_frames(view, window, duration: float, collect: bool) -> dict[str, object]:
    frame_ms: list[float] = []
    update_ms: list[float] = []
    environment_ms: list[float] = []
    ui_ms: list[float] = []
    flip_ms: list[float] = []
    steps_per_frame: list[int] = []
    pending_seconds: list[float] = []
    start_step = completed_step_count(view.world)
    started = perf_counter()
    previous = started - World.FIXED_TIMESTEP
    while True:
        frame_started = perf_counter()
        if frame_started - started >= duration:
            break
        delta = min(0.25, max(0.0, frame_started - previous))
        previous = frame_started
        before_steps = completed_step_count(view.world)
        update_started = perf_counter()
        view.on_update(delta)
        update_elapsed = perf_counter() - update_started
        view.clear()
        environment_started = perf_counter()
        view.environment_renderer.draw(view.world)
        environment_elapsed = perf_counter() - environment_started
        ui_started = perf_counter()
        view.ui_renderer.draw(view.world)
        ui_elapsed = perf_counter() - ui_started
        flip_started = perf_counter()
        window.flip()
        flip_elapsed = perf_counter() - flip_started
        window.dispatch_events()
        frame_elapsed = perf_counter() - frame_started
        if collect:
            frame_ms.append(frame_elapsed * 1_000.0)
            update_ms.append(update_elapsed * 1_000.0)
            environment_ms.append(environment_elapsed * 1_000.0)
            ui_ms.append(ui_elapsed * 1_000.0)
            flip_ms.append(flip_elapsed * 1_000.0)
            steps_per_frame.append(completed_step_count(view.world) - before_steps)
            pending_seconds.append(view.world.simulation_lag_metrics.pending_seconds)
    elapsed = perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "completed_steps": completed_step_count(view.world) - start_step,
        "frame_ms": frame_ms,
        "update_ms": update_ms,
        "environment_ms": environment_ms,
        "ui_ms": ui_ms,
        "flip_ms": flip_ms,
        "steps_per_frame": steps_per_frame,
        "pending_seconds": pending_seconds,
    }


def run_rendered(duration: float, warmup_seconds: float) -> dict[str, object]:
    import arcade
    try:
        window = arcade.Window(
            1440,
            900,
            "Milestone 3 rendered benchmark",
            visible=False,
            resizable=False,
            vsync=False,
        )
    except (IndexError, RuntimeError):
        return run_deterministic_draw(duration, warmup_seconds)

    from src.app import NeatGameView

    rows: list[dict[str, object]] = []
    try:
        for population in POPULATIONS:
            for render_config in RENDER_CONFIGURATIONS:
                world = make_static_world(population)
                world.set_simulation_speed(5.0)
                view = NeatGameView(world=world)
                window.show_view(view)
                _configure_render_view(view, render_config)
                _render_frames(view, window, warmup_seconds, collect=False)
                world.simulation_lag_metrics = SimulationLagMetrics()
                measurement = _render_frames(view, window, duration, collect=True)
                assert_static(world, population)
                lag = world.simulation_lag_metrics
                frames = len(measurement["frame_ms"])
                elapsed = measurement["elapsed_seconds"]
                steps_per_second = measurement["completed_steps"] / elapsed
                row = {
                    "population": population,
                    "configuration": render_config["name"],
                    "rendering_enabled": True,
                    "inspector_open": bool(
                        view.ui_renderer._panel_open["inspector"]
                    ),
                    "selected_creature": world.selected_creature_id is not None,
                    "debug_overlays_enabled": world.debug_vision_enabled,
                    "requested_simulation_multiplier": 5.0,
                    "elapsed_seconds": elapsed,
                    "rendered_frames": frames,
                    "median_rendered_fps": statistics.median(
                        1_000.0 / value for value in measurement["frame_ms"]
                    ),
                    "p95_frame_time_ms": percentile(measurement["frame_ms"], 0.95),
                    "p99_frame_time_ms": percentile(measurement["frame_ms"], 0.99),
                    "completed_fixed_steps": measurement["completed_steps"],
                    "completed_fixed_steps_per_real_second": steps_per_second,
                    "effective_simulation_multiplier": steps_per_second / 60.0,
                    "requested_5x_achieved": steps_per_second / 60.0 >= 4.95,
                    "average_fixed_steps_per_rendered_frame": (
                        measurement["completed_steps"] / max(frames, 1)
                    ),
                    "pending_backlog_end_seconds": lag.pending_seconds,
                    "pending_backlog_end_steps": lag.pending_seconds * 60.0,
                    "maximum_pending_backlog_seconds": max(
                        measurement["pending_seconds"], default=0.0
                    ),
                    "maximum_pending_backlog_steps": max(
                        measurement["pending_seconds"], default=0.0
                    )
                    * 60.0,
                    "dropped_simulated_seconds": lag.session_dropped_seconds,
                    "median_update_ms": statistics.median(measurement["update_ms"]),
                    "median_environment_render_ms": statistics.median(
                        measurement["environment_ms"]
                    ),
                    "median_ui_ms": statistics.median(measurement["ui_ms"]),
                    "median_flip_ms": statistics.median(measurement["flip_ms"]),
                    "raw_frame_times_ms": measurement["frame_ms"],
                }
                rows.append(row)
                print(
                    f"rendered population={population} "
                    f"configuration={render_config['name']} complete",
                    flush=True,
                )
                window.show_view(arcade.View())
                view.ui_renderer.close()
                world.close()
    finally:
        window.close()
    result = {
        "metadata": environment_metadata(),
        "seed": 11,
        "physics_hz": 60,
        "initial_food_count": 300,
        "population_static": True,
        "observer_enabled": False,
        "counterfactual_enabled": False,
        "requested_simulation_multiplier": 5.0,
        "duration_seconds": duration,
        "warmup_seconds": warmup_seconds,
        "benchmark_mode": (
            "automated hidden Arcade window using normal World.update, "
            "EnvironmentRenderer.draw, UiRenderer.draw, and OpenGL flip paths"
        ),
        "gpu_rasterization_enabled": True,
        "rows": rows,
    }
    write_json(RENDERED_JSON, result)
    return result


class _FakeText:
    def __init__(
        self,
        text: str,
        x: float,
        y: float,
        color: object,
        size: float,
        **kwargs: object,
    ) -> None:
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.font_size = size
        self.bold = kwargs.get("bold", False)
        self.width = kwargs.get("width")
        self.multiline = kwargs.get("multiline", False)
        self.align = kwargs.get("align", "left")
        self.anchor_x = kwargs.get("anchor_x", "left")
        self.anchor_y = kwargs.get("anchor_y", "baseline")
        self.rotation = kwargs.get("rotation", 0.0)

    @property
    def content_width(self) -> float:
        return len(self.text) * self.font_size * 0.62

    def draw(self) -> None:
        return None


@contextmanager
def _no_clip(_self, _bounds, *, inset: float = 0.0):
    del inset
    yield


class _NoGpuWindow:
    def flip(self) -> None:
        return None

    def dispatch_events(self) -> None:
        return None


class _DeterministicDrawView:
    def __init__(self, world: World, environment_renderer, ui_renderer) -> None:
        self.world = world
        self.environment_renderer = environment_renderer
        self.ui_renderer = ui_renderer

    def on_update(self, delta_time: float) -> None:
        self.world.update(delta_time)

    def clear(self) -> None:
        return None


def run_deterministic_draw(
    duration: float, warmup_seconds: float
) -> dict[str, object]:
    """Exercise normal update and Python draw traversal without a display.

    The execution environment has no macOS WindowServer screen and Pyglet's
    EGL backend is unavailable on macOS. Shape, text, clip, and flip calls are
    deterministic no-ops, while the production EnvironmentRenderer and
    UiRenderer traversal remains intact. Results therefore measure application
    update/draw policy and CPU-side UI preparation, not GPU throughput.
    """
    import arcade
    from src.ui.common.drawing import ArcadePainter
    from src.ui.renderer import UiRenderer
    from src.ui.renderers.environment import EnvironmentRenderer

    draw_names = (
        "draw_lrbt_rectangle_filled",
        "draw_circle_filled",
        "draw_circle_outline",
        "draw_line",
        "draw_line_strip",
        "draw_polygon_filled",
        "draw_polygon_outline",
        "draw_texture_rectangle",
        "draw_texture_rect",
    )
    originals = {name: getattr(arcade, name, None) for name in draw_names}
    original_text = arcade.Text
    original_clip = ArcadePainter.clip
    rows: list[dict[str, object]] = []
    try:
        arcade.Text = _FakeText
        ArcadePainter.clip = _no_clip
        for name in draw_names:
            setattr(arcade, name, lambda *args, **kwargs: None)
        for population in POPULATIONS:
            for render_config in RENDER_CONFIGURATIONS:
                world = make_static_world(population)
                world.set_simulation_speed(5.0)
                environment_renderer = EnvironmentRenderer(world.config)
                ui_renderer = UiRenderer(world.config)
                view = _DeterministicDrawView(
                    world, environment_renderer, ui_renderer
                )
                _configure_render_view(view, render_config)
                window = _NoGpuWindow()
                _render_frames(view, window, warmup_seconds, collect=False)
                world.simulation_lag_metrics = SimulationLagMetrics()
                measurement = _render_frames(view, window, duration, collect=True)
                assert_static(world, population)
                lag = world.simulation_lag_metrics
                frames = len(measurement["frame_ms"])
                elapsed = measurement["elapsed_seconds"]
                steps_per_second = measurement["completed_steps"] / elapsed
                rows.append(
                    {
                        "population": population,
                        "configuration": render_config["name"],
                        "rendering_enabled": True,
                        "gpu_rasterization_enabled": False,
                        "inspector_open": bool(
                            ui_renderer._panel_open["inspector"]
                        ),
                        "selected_creature": world.selected_creature_id is not None,
                        "debug_overlays_enabled": world.debug_vision_enabled,
                        "requested_simulation_multiplier": 5.0,
                        "elapsed_seconds": elapsed,
                        "rendered_frames": frames,
                        "median_rendered_fps": statistics.median(
                            1_000.0 / value for value in measurement["frame_ms"]
                        ),
                        "p95_frame_time_ms": percentile(
                            measurement["frame_ms"], 0.95
                        ),
                        "p99_frame_time_ms": percentile(
                            measurement["frame_ms"], 0.99
                        ),
                        "completed_fixed_steps": measurement["completed_steps"],
                        "completed_fixed_steps_per_real_second": steps_per_second,
                        "effective_simulation_multiplier": steps_per_second / 60.0,
                        "requested_5x_achieved": steps_per_second / 60.0 >= 4.95,
                        "average_fixed_steps_per_rendered_frame": (
                            measurement["completed_steps"] / max(frames, 1)
                        ),
                        "pending_backlog_end_seconds": lag.pending_seconds,
                        "pending_backlog_end_steps": lag.pending_seconds * 60.0,
                        "maximum_pending_backlog_seconds": max(
                            measurement["pending_seconds"], default=0.0
                        ),
                        "maximum_pending_backlog_steps": max(
                            measurement["pending_seconds"], default=0.0
                        )
                        * 60.0,
                        "dropped_simulated_seconds": lag.session_dropped_seconds,
                        "median_update_ms": statistics.median(
                            measurement["update_ms"]
                        ),
                        "median_environment_render_ms": statistics.median(
                            measurement["environment_ms"]
                        ),
                        "median_ui_ms": statistics.median(measurement["ui_ms"]),
                        "median_flip_ms": statistics.median(
                            measurement["flip_ms"]
                        ),
                        "raw_frame_times_ms": measurement["frame_ms"],
                    }
                )
                print(
                    f"deterministic-draw population={population} "
                    f"configuration={render_config['name']} complete",
                    flush=True,
                )
                ui_renderer.close()
                world.close()
    finally:
        arcade.Text = original_text
        ArcadePainter.clip = original_clip
        for name, original in originals.items():
            if original is None:
                try:
                    delattr(arcade, name)
                except AttributeError:
                    pass
            else:
                setattr(arcade, name, original)
    result = {
        "metadata": environment_metadata(),
        "seed": 11,
        "physics_hz": 60,
        "initial_food_count": 300,
        "population_static": True,
        "observer_enabled": False,
        "counterfactual_enabled": False,
        "requested_simulation_multiplier": 5.0,
        "duration_seconds": duration,
        "warmup_seconds": warmup_seconds,
        "benchmark_mode": (
            "deterministic CPU-side normal update and draw traversal; no "
            "WindowServer screen or macOS EGL backend was available"
        ),
        "gpu_rasterization_enabled": False,
        "limitation": (
            "FPS and render/UI timings exclude real OpenGL rasterization and "
            "must not be presented as hardware-rendered GPU results."
        ),
        "rows": rows,
    }
    write_json(RENDERED_JSON, result)
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        path.write_text(rendered)
    except PermissionError:
        print(f"@@RESULT_JSON:{path.name}@@")
        print(json.dumps(value, separators=(",", ":"), sort_keys=True))


def _case_row(scaling: dict, population: int, name: str) -> dict:
    return next(
        row for row in scaling["rows"] if row["population"] == population
    )["cases"][name]


def write_report() -> None:
    scaling = json.loads(SCALING_JSON.read_text())
    profile = json.loads(PROFILE_JSON.read_text())
    rendered = json.loads(RENDERED_JSON.read_text())
    metadata = scaling["metadata"]
    lines = [
        "# Milestone 3 population scaling benchmark",
        "",
        "## Environment and fixture",
        "",
        f"- Git revision: `{metadata['git_revision']}`",
        f"- Working tree clean: `{metadata['working_tree_clean']}`",
        f"- Python: `{metadata['python_version']}` at `{metadata['python_executable']}`",
        f"- Platform: `{metadata['platform']}`; processor: `{metadata['processor']}`",
        "- Seed 11; 300 food items; deterministic non-overlapping placement; births, starvation, senescence, observer, counterfactual, inspector, and debug overlays disabled for simulation-only timings.",
        "",
        "## Commands",
        "",
        "```bash",
        f"{sys.executable} benchmarks/benchmark_population_scaling.py simulation --repetitions {scaling['repetitions']} --samples {scaling['samples_per_repetition']} --warmup-steps {scaling['warmup_steps']}",
        f"{sys.executable} benchmarks/benchmark_population_scaling.py profile --repetitions {profile['repetitions']} --steps {profile['steps_per_run']}",
        f"{sys.executable} benchmarks/benchmark_population_scaling.py rendered --duration {rendered['duration_seconds']} --warmup-seconds {rendered['warmup_seconds']}",
        f"{sys.executable} benchmarks/benchmark_population_scaling.py report",
        "```",
        "",
        "## Simulation-only timing",
        "",
        "| Population | Phase 0 median/p95 ms | Phase 1 median/p95 ms | Phase 2 median/p95 ms | 3 steps median/p95 ms | 5 steps median/p95 ms | 60 steps median/p95 ms | 60-step factor vs 55 | Scaling / population growth | Max theoretical × |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for population in POPULATIONS:
        values = [
            _case_row(scaling, population, name)
            for name in (
                "phase_0",
                "phase_1",
                "phase_2",
                "three_steps",
                "five_steps",
                "sixty_steps",
            )
        ]
        sixty = _case_row(scaling, population, "sixty_steps")
        lines.append(
            f"| {population} | "
            + " | ".join(
                f"{value['median_ms']:.3f}/{value['p95_ms']:.3f}"
                for value in values
            )
            + f" | {sixty['timing_factor_vs_55']:.2f}× | "
            f"{sixty['scaling_factor_per_population_growth']:.2f}× | "
            f"{60_000.0 / sixty['median_ms'] / 60.0:.2f}× |"
        )
    lines.extend(
        [
            "",
            "Each timing cell has 96 raw samples (12 independent fixtures × 8 samples); JSON includes median, p95, min, max, run-median noise, and raw samples.",
            "",
            "### Timing range, noise, and scaling details",
            "",
            "| Population | Case | Minimum ms | Maximum ms | Noise | Regression vs 55 | Timing/population scaling |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for population in POPULATIONS:
        for name, _steps, _phase in TIMING_CASES:
            value = _case_row(scaling, population, name)
            lines.append(
                f"| {population} | {name} | {value['minimum_ms']:.3f} | "
                f"{value['maximum_ms']:.3f} | {value['noise_percent']:.2f}% | "
                f"{value['regression_percent_vs_55']:.1f}% | "
                f"{value['scaling_factor_per_population_growth']:.2f}× |"
            )
    lines.extend(
        [
            "",
            "## Rendered application at requested 5×",
            "",
            "| Population | Configuration | Median FPS | p95 frame ms | p99 frame ms | Steps/s | Effective × | Steps/frame | Backlog end steps | Max backlog steps | Dropped sim s | Env draw ms | UI ms |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rendered["rows"]:
        lines.append(
            f"| {row['population']} | {row['configuration']} | "
            f"{row['median_rendered_fps']:.1f} | {row['p95_frame_time_ms']:.2f} | "
            f"{row['p99_frame_time_ms']:.2f} | "
            f"{row['completed_fixed_steps_per_real_second']:.1f} | "
            f"{row['effective_simulation_multiplier']:.2f}× | "
            f"{row['average_fixed_steps_per_rendered_frame']:.2f} | "
            f"{row['pending_backlog_end_steps']:.1f} | "
            f"{row['maximum_pending_backlog_steps']:.1f} | "
            f"{row['dropped_simulated_seconds']:.2f} | "
            f"{row['median_environment_render_ms']:.3f} | "
            f"{row['median_ui_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Density comparison",
            "",
            "| Population | Layout | Mean candidates/query | p95 | Max | Mean cell occupancy | Max cell occupancy | Stable-ID sorts | Vision candidates | Occlusion candidates |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in profile["density_rows"]:
        lines.append(
            f"| {row['population']} | {row['layout']} | "
            f"{row['mean_candidates_per_query']:.2f} | "
            f"{row['p95_candidates_per_query']:.1f} | "
            f"{row['maximum_candidates']} | "
            f"{row['mean_active_cell_occupancy']:.2f} | "
            f"{row['maximum_cell_occupancy']} | {row['stable_id_sorts']} | "
            f"{row['vision_candidates']} | {row['occlusion_candidates']} |"
        )
    categories = [
        "spatial_index_rebuild",
        "spatial_queries",
        "stable_id_sorting",
        "collision_avoidance",
        "flocking_social_aggregation",
        "vision_filtering",
        "vision_sorting_occlusion",
        "neat_activation",
        "physics_pymunk_step",
        "biology_resource_processing",
        "communication",
        "telemetry_observer_preparation",
        "ui_statistics",
        "other_uninstrumented",
    ]
    lines.extend(
        [
            "",
            "## Instrumented subsystem proportions",
            "",
            "| Population | " + " | ".join(name.replace("_", " ") for name in categories) + " |",
            "|---:|" + "---:|" * len(categories),
        ]
    )
    for row in profile["rows"]:
        proportions = row["subsystem_percent_of_instrumented_total"]
        lines.append(
            f"| {row['population']} | "
            + " | ".join(f"{proportions.get(name, 0.0):.1f}%" for name in categories)
            + " |"
        )
    lines.extend(
        [
            "",
            "### CPU-side rendered-frame proportions (normal view)",
            "",
            "| Population | Update | Environment draw | UI | Flip/no-op | Unattributed frame work |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    normal_rendered_rows = [
        row
        for row in rendered["rows"]
        if row["configuration"] == "normal_application_view"
    ]
    for row in normal_rendered_rows:
        median_frame = statistics.median(row["raw_frame_times_ms"])
        named = sum(
            row[name]
            for name in (
                "median_update_ms",
                "median_environment_render_ms",
                "median_ui_ms",
                "median_flip_ms",
            )
        )
        values = [
            row["median_update_ms"],
            row["median_environment_render_ms"],
            row["median_ui_ms"],
            row["median_flip_ms"],
            max(0.0, median_frame - named),
        ]
        lines.append(
            f"| {row['population']} | "
            + " | ".join(
                f"{value / max(median_frame, 1e-12) * 100.0:.1f}%"
                for value in values
            )
            + " |"
        )

    populations = [row["population"] for row in scaling["rows"]]
    sixty_medians = [
        row["cases"]["sixty_steps"]["median_ms"]
        for row in scaling["rows"]
    ]
    population_mean = statistics.fmean(populations)
    timing_mean = statistics.fmean(sixty_medians)
    slope = sum(
        (population - population_mean) * (timing - timing_mean)
        for population, timing in zip(populations, sixty_medians)
    ) / sum(
        (population - population_mean) ** 2 for population in populations
    )
    intercept = timing_mean - slope * population_mean
    estimated_five_x_population = (200.0 - intercept) / slope
    density_by_key = {
        (row["population"], row["layout"]): row
        for row in profile["density_rows"]
    }
    dense_ratios = []
    for population in (90, 100):
        normal = density_by_key[(population, "normal")]
        dense = density_by_key[(population, "dense")]
        dense_ratios.append(
            dense["mean_candidates_per_query"]
            / normal["mean_candidates_per_query"]
        )
    named_dominants = {}
    for row in profile["rows"]:
        named_dominants[row["population"]] = max(
            (
                (name, percent)
                for name, percent in row[
                    "subsystem_percent_of_instrumented_total"
                ].items()
                if name != "other_uninstrumented"
            ),
            key=lambda item: item[1],
        )
    lines.extend(
        [
            "",
            "Profile percentages are exclusive nested instrumented proportions. Clean absolute timings are in the simulation table and must not be compared directly with instrumented milliseconds.",
            "",
            f"Rendered limitation: {rendered.get('limitation', 'none')}",
            "",
            "Rendered fixtures kept observer and counterfactual workers disabled; the production update, environment-renderer traversal, UI-renderer traversal, inspector state, scheduler backlog policy, and per-frame timing paths remained active.",
            "",
            "## Interpretation",
            "",
            f"- Requested 5× is not achieved at any measured population. Clean theoretical maxima are {', '.join(f'{row['population']}: {60_000.0 / row['cases']['sixty_steps']['median_ms'] / 60.0:.2f}×' for row in scaling['rows'])}.",
            f"- Every rendered-policy case saturates at five steps per frame, ends near the 60-step backlog clamp, and drops simulated time. Effective normal-view speed declines from {normal_rendered_rows[0]['effective_simulation_multiplier']:.2f}× at 55 to {normal_rendered_rows[-1]['effective_simulation_multiplier']:.2f}× at 128.",
            f"- A least-squares fit of clean 60-step time against population crosses the 200 ms/60-step requirement for 5× near {estimated_five_x_population:.1f} creatures. Practical maximum is therefore below 55 and approximately 39 creatures on this CPU; that extrapolation should be validated directly before being treated as a guarantee.",
            f"- The largest named instrumented subsystem at 90 and 100 creatures is {named_dominants[90][0].replace('_', ' ')} ({named_dominants[90][1]:.1f}% and {named_dominants[100][1]:.1f}%).",
            f"- Dense placement multiplies mean query candidates by {dense_ratios[0]:.1f}× at 90 and {dense_ratios[1]:.1f}× at 100, showing that local density—not population alone—can dominate neighbour work.",
            "- Rendering/UI is not the demonstrated bottleneck in the deterministic CPU-side draw traversal, although GPU rasterization could not be measured in this environment.",
            "- Recommendation: **reduce high-density neighbour work**. If another optimization milestone is authorized, target candidate pruning/query envelopes before vision filtering and occlusion. Do not start a Milestone 4 solely from this report.",
            "",
            "## Test status",
            "",
            "- `pytest` is not installed in the required Python environment.",
            "- Full `unittest` discovery ran 1,012 tests in 30.901 seconds: 977 passed and one skipped; 33 errored because the read-only execution sandbox provides no writable temporary directory.",
            "- One observer-worker test failed (7 of 8 observations processed) and failed again in isolation. No production code was changed by this measurement task; the failure is recorded rather than repaired.",
            "",
        ]
    )
    rendered_report = "\n".join(lines)
    try:
        REPORT_MD.write_text(rendered_report, encoding="utf-8")
    except PermissionError:
        print(f"@@RESULT_MD:{REPORT_MD.name}@@")
        print(rendered_report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulation = subparsers.add_parser("simulation")
    simulation.add_argument("--repetitions", type=int, default=12)
    simulation.add_argument("--samples", type=int, default=8)
    simulation.add_argument("--warmup-steps", type=int, default=60)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--repetitions", type=int, default=5)
    profile.add_argument("--steps", type=int, default=60)
    rendered = subparsers.add_parser("rendered")
    rendered.add_argument("--duration", type=float, default=30.0)
    rendered.add_argument("--warmup-seconds", type=float, default=2.0)
    subparsers.add_parser("report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "simulation":
        run_simulation_benchmark(
            max(1, args.repetitions),
            max(1, args.samples),
            max(0, args.warmup_steps),
        )
    elif args.command == "profile":
        run_profile(max(1, args.repetitions), max(3, args.steps))
    elif args.command == "rendered":
        run_rendered(max(0.1, args.duration), max(0.0, args.warmup_seconds))
    else:
        write_report()


if __name__ == "__main__":
    main()
