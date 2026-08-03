"""Export permanent v20 migration fixtures from the exact clean baseline.

Run this script with the baseline detached worktree as its first argument and
the fixture directory as its second.  It intentionally imports every project
module from that worktree, never from the checkout containing this script.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import importlib.metadata
import json
from math import cos, pi, sin
from pathlib import Path
import pickle
import subprocess
import sys


BASELINE_REVISION = "75924f39c213b6f2e765c7ea7ce8553477360f73"
CAPTURE_STEPS = (12, 13, 14)
CONTINUATION_STEPS = 15


def _git(worktree: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(worktree), *arguments),
        text=True,
    ).strip()


def _verify_worktree(worktree: Path) -> None:
    revision = _git(worktree, "rev-parse", "HEAD")
    if revision != BASELINE_REVISION:
        raise RuntimeError(
            f"Expected baseline {BASELINE_REVISION}, found {revision}."
        )
    if _git(worktree, "status", "--porcelain"):
        raise RuntimeError("Baseline fixture worktree must be clean.")
    sys.path.insert(0, str(worktree))
    if Path(sys.path[0]).resolve() != worktree.resolve():
        raise RuntimeError("Baseline worktree is not first on sys.path.")


def _non_neutral_action(action: object) -> bool:
    return any(
        abs(float(getattr(action, name, 0.0))) > 1e-12
        for name in (
            "accelerate",
            "rotate",
            "herding",
            "rest",
            "emit_sound",
            "emit_trail_pheromone",
            "emit_alarm_pheromone",
        )
    )


def _choose_sentinel(world, next_phase: int):
    for creature in world.creatures:
        action = world._last_actions.get(creature.creature_id)
        social = world._cached_social_intentions.get(creature.creature_id)
        brain = world.neat_controller.brain_for(creature.creature_id)
        if (
            creature.creature_id % 3 != next_phase
            and action is not None
            and _non_neutral_action(action)
            and social is not None
            and any(abs(float(value)) > 1e-12 for value in (
                social[1].effective_count,
                social[2],
                *social[3],
                getattr(brain, "herding_state", 0.0),
                creature.smoothed_acceleration,
                creature.smoothed_rotation,
                creature.smoothed_rest,
            ))
        ):
            return creature
    raise RuntimeError("No sentinel demonstrates non-neutral continuation state.")


def _assert_capture_contract(world, sentinel) -> None:
    social_intent, observation, influence, contribution = (
        world._cached_social_intentions[sentinel.creature_id]
    )
    weights = social_intent.weights
    required = (
        observation.effective_count,
        weights.separation,
        weights.alignment,
        weights.cohesion,
        social_intent.confidence,
        influence,
        *contribution,
        world._flocking_benchmark_quality_by_creature_id.get(
            sentinel.creature_id, 0.0
        ),
    )
    if not any(abs(float(value)) > 1e-12 for value in required):
        raise RuntimeError("Sentinel social continuation is neutral.")
    if not world.acoustics.signals:
        raise RuntimeError("Capture lacks active cached acoustic state.")
    trail = getattr(world.pheromones, "trail", None)
    alarm = getattr(world.pheromones, "alarm", None)
    if trail is None or alarm is None or not (trail.any() or alarm.any()):
        raise RuntimeError("Capture lacks active pheromone state.")
    signatures = {
        pickle.dumps(
            (
                world._last_actions[creature.creature_id],
                world._cached_social_intentions[creature.creature_id],
                world.neat_controller.brain_for(creature.creature_id).herding_state,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        for creature in world.creatures
    }
    if len(signatures) != len(world.creatures):
        raise RuntimeError("Creature continuation states are not distinct.")


def _world():
    from configs.sim_config import build_sim_config
    from src.persistence import SimulationPaths
    from src.world import World

    config = build_sim_config()
    config.random_seed = 11
    config.persistence.enable_telemetry = False
    config.persistence.quick_save_interval_seconds = 0.0
    config.persistence.archive_save_interval_seconds = 0.0
    config.behavior.enabled = False
    config.counterfactual_why.enabled = False
    config.population.initial_creatures = 2
    config.population.min_reproduction_age = 10_000.0
    config.population.senescence_age_seconds = 10_000.0
    config.food.initial_food_items = 8
    config.flocking.cohort_spawn.enabled = True
    config.flocking.cohort_spawn.size = 2
    config.flocking.cohort_spawn.radius = 24.0
    config.flocking.long_range.enabled = True
    world = World(
        config,
        simulation_paths=SimulationPaths(Path(".").resolve()),
    )
    # Keep both creatures inside flock range while placing their circle
    # centres beyond the largest collision envelope.  Migration fixtures then
    # exercise social reductions without baking the retired Pymunk query order
    # into collision-force floating-point sums.
    for index, creature in enumerate(world.creatures):
        angle = 2.0 * pi * index / len(world.creatures)
        creature.body.position = (40.0 * cos(angle), 40.0 * sin(angle))
        creature.body.velocity = (0.0, 0.0)
        world.space.reindex_shape(creature.shape)
    return world


def _run_to(world, completed_step: int) -> None:
    while world._simulation_step < completed_step:
        world.update(world.fixed_timestep)


def _export_one(output: Path, capture_step: int) -> dict[str, object]:
    from src.persistence import PersistenceManager
    from tests.scheduler_validation import AuthoritativeStateDigest

    source = _world()
    restored = None
    try:
        _run_to(source, capture_step)
        next_phase = capture_step % 3
        sentinel = _choose_sentinel(source, next_phase)
        _assert_capture_contract(source, sentinel)
        checkpoint = PersistenceManager._capture_state(
            source, source.neat_controller
        )
        if checkpoint.get("version") != 20:
            raise RuntimeError("Baseline exporter did not produce checkpoint v20.")
        restored = PersistenceManager._restore_world(
            copy.deepcopy(checkpoint),
            source.config,
            source.simulation_paths,
        )
        sentinel_action = restored._last_actions[sentinel.creature_id]
        trajectory = []
        action_reused = False
        for _ in range(CONTINUATION_STEPS):
            before_phase = restored._simulation_step % 3
            before_action = restored._last_actions[sentinel.creature_id]
            restored.update(restored.fixed_timestep)
            if sentinel.creature_id % 3 != before_phase:
                action_reused = action_reused or (
                    restored._last_actions[sentinel.creature_id]
                    is before_action
                )
            trajectory.append(AuthoritativeStateDigest.capture(restored))
        if not action_reused or sentinel_action is None:
            raise RuntimeError("Continuous cached action reuse was not demonstrated.")
        payload = {
            "checkpoint": checkpoint,
            "trajectory": trajectory,
        }
        with gzip.open(output, "wb", compresslevel=9) as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "file": output.name,
            "captured_step": capture_step,
            "next_decision_phase": next_phase,
            "sentinel_creature_id": sentinel.creature_id,
            "continuation_length": CONTINUATION_STEPS,
            "crossed_biology_steps": [
                step
                for step in range(capture_step + 1, capture_step + 16)
                if step % 3 == 0
            ],
            "crossed_statistics_steps": [
                step
                for step in range(capture_step + 1, capture_step + 16)
                if step % 12 == 0
            ],
        }
    finally:
        if restored is not None:
            restored.close()
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_worktree", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    _verify_worktree(arguments.baseline_worktree)
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    fixtures = [
        _export_one(
            arguments.output_directory / f"phase_{step % 3}_step_{step}.pkl.gz",
            step,
        )
        for step in CAPTURE_STEPS
    ]
    dependencies = {}
    for package in ("numpy", "pymunk", "neat-python"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    manifest = {
        "baseline_revision": BASELINE_REVISION,
        "checkpoint_version": 20,
        "seed": 11,
        "configuration": {
            "creatures": 2,
            "foods": 8,
            "cohort_radius": 24.0,
            "fixture_ring_radius": 40.0,
            "reproduction_and_senescence_disabled": True,
        },
        "dependencies": dependencies,
        "fixtures": fixtures,
    }
    (arguments.output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
