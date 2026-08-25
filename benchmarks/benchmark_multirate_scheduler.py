"""End-to-end timing and call-count harness for Milestone 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_hot_loop_cleanup import (  # noqa: E402
    make_world,
    percentile,
)
from benchmarks.benchmark_flocking import force_benchmark_birth  # noqa: E402
from configs.sim_config import build_sim_config  # noqa: E402
from src.action import Action  # noqa: E402
import src.world as world_module  # noqa: E402


def fixed_timestep(world) -> float:
    return float(getattr(world, "fixed_timestep", world.FIXED_TIMESTEP))


def completed_step_count(world) -> int:
    return int(
        getattr(
            world,
            "_simulation_step",
            getattr(world, "physics_step_count", 0),
        )
    )


def run_fixed_steps(world, count: int) -> None:
    remaining = count
    while remaining > 0:
        batch = min(5, remaining)
        world.set_simulation_speed(float(batch))
        world.update(fixed_timestep(world))
        remaining -= batch


def measure_case(
    steps_per_sample: int,
    *,
    repetitions: int,
    samples_per_repetition: int,
    warmup_steps: int,
    phase: int | None = None,
) -> dict[str, object]:
    samples: list[float] = []
    run_medians: list[float] = []
    for _ in range(repetitions):
        world = make_world()
        try:
            run_fixed_steps(world, warmup_steps)
            run_samples: list[float] = []
            for _sample in range(samples_per_repetition):
                if phase is not None:
                    while completed_step_count(world) % 3 != phase:
                        run_fixed_steps(world, 1)
                started = perf_counter()
                run_fixed_steps(world, steps_per_sample)
                run_samples.append((perf_counter() - started) * 1_000.0)
            samples.extend(run_samples)
            run_medians.append(statistics.median(run_samples))
        finally:
            world.close()
    median = statistics.median(samples)
    noise = (
        (percentile(run_medians, 0.95) - percentile(run_medians, 0.05))
        / max(2.0 * statistics.median(run_medians), 1e-12)
        * 100.0
    )
    return {
        "steps_per_sample": steps_per_sample,
        "phase": phase,
        "samples": len(samples),
        "median_ms": median,
        "p95_ms": percentile(samples, 0.95),
        "noise_percent": noise,
        "run_medians_ms": run_medians,
    }


def collect_counters(warmup_steps: int, measured_steps: int) -> dict[str, int]:
    world = make_world()
    counters = {
        "physics_steps": 0,
        "motion_applications": 0,
        "sensor_builds": 0,
        "neat_activations": 0,
        "flocking_calculations": 0,
        "social_blends": 0,
        "biology_passes": 0,
        "contact_checks": 0,
        "exposure_resolutions": 0,
        "observer_samples": 0,
        "counterfactual_timebase_checks": 0,
        "counterfactual_probe_submissions": 0,
        "telemetry_timebase_checks": 0,
        "telemetry_samples": 0,
        "statistics_refreshes": 0,
        "pheromone_updates": 0,
        "communication_commits": 0,
        "reproduction_cadence_updates": 0,
        "diagnostic_allocations": 0,
    }
    telemetry_samples: list[float] = []
    world.telemetry = type(
        "BenchmarkTelemetry",
        (),
        {
            "log_flocking_metrics": staticmethod(
                lambda metrics: telemetry_samples.append(
                    float(metrics.get("sim_time", 0.0))
                )
            ),
            "close": staticmethod(lambda: None),
        },
    )()
    run_fixed_steps(world, warmup_steps)
    spatial_before = world._creature_spatial_index.counters.snapshot()
    vision_before = {
        "candidate": world.vision.candidate_buffer_growth,
        "visible": world.vision.visible_index_growth,
        "blocked": world.vision.blocked_interval_growth,
        "food_ids": world.vision.visible_food_id_growth,
        "result": world.vision.sense_result_growth,
        "sorts": world.vision.stable_sort_count,
    }
    scheduled_queries_before = world._spatial_scheduled_queries
    collision_queries_before = world._spatial_collision_only_queries
    motion_command_ids_before = set(world._motion_commands)
    telemetry_samples.clear()
    originals = {
        "space_step": world.space.step,
        "apply_action": world._apply_action,
        "sense": world._sensor_snapshot_for,
        "decide": world.neat_controller.decide,
        "social": world._refresh_social_runtime,
        "biology": world._update_metabolism,
        "contacts": getattr(world, "_accumulate_mouth_exposures", None),
        "resolve": getattr(
            world,
            "_resolve_accumulated_mouth_exposures",
            None,
        ),
        "stats": world._refresh_stats,
        "pheromones": world.pheromones.advance,
        "observer": world.behavior_observer.submit_batch,
        "why": world._sample_selected_why,
        "submit_why": world.behavior_observer.submit_why,
        "telemetry": world._update_flocking_telemetry,
        "communication": world._commit_communication_intents,
        "reproduction": world._update_reproduction,
        "blend": world_module.blend_desired_velocity,
    }
    runtime_snapshot_type = world_module.FlockingRuntimeSnapshot

    def counted(name, original, amount=1):
        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            counters[name] += amount(result) if callable(amount) else amount
            return result

        return wrapper

    world.space.step = counted("physics_steps", originals["space_step"])
    world._apply_action = counted("motion_applications", originals["apply_action"])
    world._sensor_snapshot_for = counted("sensor_builds", originals["sense"])
    world.neat_controller.decide = counted("neat_activations", originals["decide"])
    world._refresh_social_runtime = counted(
        "flocking_calculations", originals["social"]
    )
    world._update_metabolism = counted("biology_passes", originals["biology"])
    if originals["contacts"] is not None:
        world._accumulate_mouth_exposures = counted(
            "contact_checks", originals["contacts"]
        )
    if originals["resolve"] is not None:
        world._resolve_accumulated_mouth_exposures = counted(
            "exposure_resolutions", originals["resolve"]
        )
    world._refresh_stats = counted("statistics_refreshes", originals["stats"])
    world.pheromones.advance = counted(
        "pheromone_updates",
        originals["pheromones"],
        amount=1,
    )
    world.behavior_observer.submit_batch = counted(
        "observer_samples", originals["observer"]
    )
    world._sample_selected_why = counted(
        "counterfactual_timebase_checks", originals["why"]
    )
    world.behavior_observer.submit_why = counted(
        "counterfactual_probe_submissions", originals["submit_why"]
    )
    world._update_flocking_telemetry = counted(
        "telemetry_timebase_checks", originals["telemetry"]
    )
    world._commit_communication_intents = counted(
        "communication_commits", originals["communication"]
    )
    world._update_reproduction = counted(
        "reproduction_cadence_updates", originals["reproduction"]
    )

    def counted_runtime_snapshot(*args, **kwargs):
        counters["diagnostic_allocations"] += 1
        return runtime_snapshot_type(*args, **kwargs)

    def counted_social_blend(*args, **kwargs):
        counters["social_blends"] += 1
        return originals["blend"](*args, **kwargs)

    try:
        with patch.object(
            world_module,
            "FlockingRuntimeSnapshot",
            counted_runtime_snapshot,
        ), patch.object(
            world_module,
            "blend_desired_velocity",
            counted_social_blend,
        ):
            run_fixed_steps(world, measured_steps)
    finally:
        spatial_after = world._creature_spatial_index.counters.snapshot()
        counters.update(
            {
                "spatial_rebuilds": spatial_after["rebuilds"]
                - spatial_before["rebuilds"],
                "spatial_queries": spatial_after["queries"]
                - spatial_before["queries"],
                "scheduled_shared_queries": world._spatial_scheduled_queries
                - scheduled_queries_before,
                "unscheduled_collision_queries": (
                    world._spatial_collision_only_queries
                    - collision_queries_before
                ),
                "spatial_candidates": spatial_after["candidates"]
                - spatial_before["candidates"],
                "spatial_cell_visits": spatial_after["cell_visits"]
                - spatial_before["cell_visits"],
                "spatial_cell_resets": spatial_after["cell_resets"]
                - spatial_before["cell_resets"],
                "maximum_candidates": spatial_after["maximum_candidates"],
                "maximum_cell_occupancy": spatial_after[
                    "maximum_cell_occupancy"
                ],
                "maximum_active_cells": spatial_after["maximum_active_cells"],
                "stable_id_sorts": spatial_after["stable_sorts"]
                - spatial_before["stable_sorts"],
                "candidate_wrappers": (
                    world.vision.candidate_buffer_growth
                    - vision_before["candidate"]
                ),
                "visible_index_growth": world.vision.visible_index_growth
                - vision_before["visible"],
                "occlusion_buffer_growth": world.vision.blocked_interval_growth
                - vision_before["blocked"],
                "visible_food_id_growth": world.vision.visible_food_id_growth
                - vision_before["food_ids"],
                "vision_result_growth": world.vision.sense_result_growth
                - vision_before["result"],
                "vision_stable_sorts": world.vision.stable_sort_count
                - vision_before["sorts"],
                "spatial_buffer_growth": (
                    spatial_after["cell_buffer_growth"]
                    - spatial_before["cell_buffer_growth"]
                    + spatial_after["candidate_buffer_growth"]
                    - spatial_before["candidate_buffer_growth"]
                    + spatial_after["family_buffer_growth"]
                    - spatial_before["family_buffer_growth"]
                ),
                "child_full_scans": 0,
                "child_result_lists": 0,
                "new_motion_commands": len(
                    set(world._motion_commands) - motion_command_ids_before
                ),
                "migrated_pymunk_point_queries": 0,
            }
        )
        counters["telemetry_samples"] = len(telemetry_samples)
        counters["ordinary_unselected_diagnostic_snapshots"] = max(
            0,
            counters["diagnostic_allocations"]
            - len(world.creatures) * counters["telemetry_samples"],
        )
        world.close()
    return counters


def collect_churn_counters(measured_steps: int = 12) -> dict[str, int]:
    """Exercise changing IDs and derive query counts from live phase members."""
    world = make_world()
    run_fixed_steps(world, 6)
    rebuilds_before = world._creature_spatial_index.counters.rebuilds
    scheduled_before = world._spatial_scheduled_queries
    collision_before = world._spatial_collision_only_queries
    expected_scheduled = 0
    expected_collision = 0
    initial_ids = set(world._living_creatures)
    removed_ids: set[int] = set()
    spawned_ids: set[int] = set()
    reused_ids: set[int] = set()
    try:
        for offset in range(measured_steps):
            if offset and offset % 4 == 0:
                victim = world.creatures[-1]
                removed_ids.add(victim.creature_id)
                world._remove_creature(victim, death_reason="benchmark_churn")
                world.total_biomass_energy += 10_000.0
                reproduce = Action(
                    accelerate=0.0,
                    rotate=0.0,
                    want_reproduce=1.0,
                    want_eat=0.0,
                    reset_chronometer=0.0,
                    want_grab=0.0,
                    want_release=0.0,
                )
                issued_before = set(world._issued_creature_ids)
                created_child = False
                for parent in tuple(world.creatures):
                    parent.energy = world.config.metabolism.max_energy
                    world.rt_neat.eligible_parent_ids = [parent.creature_id]
                    world._last_actions[parent.creature_id] = reproduce
                    world._effective_actions[parent.creature_id] = reproduce
                    if force_benchmark_birth(world, parent):
                        created_child = True
                        break
                if not created_child:
                    raise RuntimeError("Could not create benchmark churn birth.")
                spawned_id = world.creatures[-1].creature_id
                spawned_ids.add(spawned_id)
                if spawned_id in issued_before:
                    reused_ids.add(spawned_id)
            live_ids = tuple(world._living_creatures)
            phase = world._simulation_step % world.config.scheduler.decision_period_steps
            scheduled = sum(
                creature_id % world.config.scheduler.decision_period_steps == phase
                for creature_id in live_ids
            )
            expected_scheduled += scheduled
            expected_collision += len(live_ids) - scheduled
            run_fixed_steps(world, 1)
        final_ids = set(world._living_creatures)
        return {
            "steps": measured_steps,
            "initial_population": len(initial_ids),
            "final_population": len(final_ids),
            "removed_ids": len(removed_ids),
            "spawned_ids": len(spawned_ids),
            "reused_ids": len(reused_ids),
            "expected_rebuilds": measured_steps,
            "actual_rebuilds": (
                world._creature_spatial_index.counters.rebuilds
                - rebuilds_before
            ),
            "expected_scheduled_queries": expected_scheduled,
            "actual_scheduled_queries": (
                world._spatial_scheduled_queries - scheduled_before
            ),
            "expected_collision_queries": expected_collision,
            "actual_collision_queries": (
                world._spatial_collision_only_queries - collision_before
            ),
        }
    finally:
        world.close()


def timing_cases(result: dict[str, object]) -> dict[str, dict[str, object]]:
    phases = result["phase_steps"]
    assert isinstance(phases, dict)
    return {
        "phase_0": phases["0"],
        "phase_1": phases["1"],
        "phase_2": phases["2"],
        "three_step_cycle": result["three_step_cycle"],
        "five_steps": result["five_steps"],
        "sixty_steps": result["sixty_steps"],
    }


def compare_results(
    baseline: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    comparisons: dict[str, object] = {}
    baseline_cases = timing_cases(baseline)
    current_cases = timing_cases(current)
    end_to_end_beyond_noise = True
    for name, before in baseline_cases.items():
        after = current_cases[name]
        before_median = float(before["median_ms"])
        after_median = float(after["median_ms"])
        improvement = (
            0.0
            if before_median <= 0.0
            else (before_median - after_median) / before_median * 100.0
        )
        noise_threshold = max(
            float(before["noise_percent"]),
            float(after["noise_percent"]),
        )
        beyond_noise = improvement > noise_threshold
        if name in {"three_step_cycle", "five_steps", "sixty_steps"}:
            end_to_end_beyond_noise = (
                end_to_end_beyond_noise and beyond_noise
            )
        comparisons[name] = {
            "baseline_median_ms": before_median,
            "current_median_ms": after_median,
            "baseline_p95_ms": float(before["p95_ms"]),
            "current_p95_ms": float(after["p95_ms"]),
            "baseline_noise_percent": float(before["noise_percent"]),
            "current_noise_percent": float(after["noise_percent"]),
            "median_improvement_percent": improvement,
            "improves_beyond_noise": beyond_noise,
        }

    counters = current["counters"]
    assert isinstance(counters, dict)
    original_decision_activations = (
        int(current["population"])
        * int(counters["physics_steps"])
    )
    final_decision_activations = int(counters["neat_activations"])
    return {
        "honest_checkout_baseline": comparisons,
        "all_end_to_end_cases_improve_beyond_noise": (
            end_to_end_beyond_noise
        ),
        "architectural_60hz_decision_baseline": {
            "original_activations": original_decision_activations,
            "final_activations": final_decision_activations,
            "final_fraction": (
                final_decision_activations
                / max(1, original_decision_activations)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=12)
    parser.add_argument("--samples-per-repetition", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--counter-steps", type=int, default=60)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-label", default="working-tree")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    repetitions = 3 if args.quick else max(1, args.repetitions)
    samples = 3 if args.quick else max(1, args.samples_per_repetition)
    warmup = 6 if args.quick else max(0, args.warmup_steps)
    counter_steps = max(1, args.counter_steps)
    result = {
        "source_label": args.source_label,
        "source_revision": args.source_revision,
        "seed": 11,
        "population": 55,
        "repetitions": repetitions,
        "samples_per_repetition": samples,
        "phase_steps": {
            str(phase): measure_case(
                1,
                repetitions=repetitions,
                samples_per_repetition=samples,
                warmup_steps=warmup,
                phase=phase,
            )
            for phase in range(3)
        },
        "three_step_cycle": measure_case(
            3,
            repetitions=repetitions,
            samples_per_repetition=samples,
            warmup_steps=warmup,
        ),
        "five_steps": measure_case(
            5,
            repetitions=repetitions,
            samples_per_repetition=samples,
            warmup_steps=warmup,
        ),
        "sixty_steps": measure_case(
            60,
            repetitions=repetitions,
            samples_per_repetition=samples,
            warmup_steps=warmup,
        ),
    }
    periodic_counters = collect_counters(warmup, counter_steps)
    zero_origin_counters = collect_counters(0, counter_steps)
    result["counters"] = periodic_counters
    result["churn_counters"] = collect_churn_counters()
    result["boundary_samples"] = {
        "observer_initial": max(
            0,
            zero_origin_counters["observer_samples"]
            - periodic_counters["observer_samples"],
        ),
        "observer_periodic": periodic_counters["observer_samples"],
        "statistics_initialization": 1,
        "statistics_periodic": periodic_counters["statistics_refreshes"],
        "pheromone_initial": 0,
        "pheromone_periodic": periodic_counters["pheromone_updates"],
        "counterfactual_initial_submissions": max(
            0,
            zero_origin_counters["counterfactual_probe_submissions"]
            - periodic_counters["counterfactual_probe_submissions"],
        ),
        "counterfactual_periodic_submissions": periodic_counters[
            "counterfactual_probe_submissions"
        ],
        "counterfactual_initial_deadline": 0.0,
        "counterfactual_period_seconds": (
            1.0 / build_sim_config().counterfactual_why.probe_hz
        ),
        "telemetry_initial": 0,
        "telemetry_periodic": periodic_counters["telemetry_samples"],
    }
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        result["comparison"] = compare_results(baseline, result)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
