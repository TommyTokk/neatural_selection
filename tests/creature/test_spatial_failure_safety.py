from __future__ import annotations

from pathlib import Path
import unittest

from configs.sim_config import build_sim_config
from src.persistence import SimulationPaths
from src.world import World


class SpatialFailureSafetyTest(unittest.TestCase):
    def make_world(self) -> World:
        """Exercise make world behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the make world test intent explicit.
        config = build_sim_config()
        config.persistence.enable_telemetry = False
        config.persistence.quick_save_interval_seconds = 0.0
        config.persistence.archive_save_interval_seconds = 0.0
        config.behavior.enabled = False
        config.counterfactual_why.enabled = False
        config.population.initial_creatures = 6
        config.food.initial_food_items = 0
        config.flocking.cohort_spawn.enabled = True
        config.flocking.cohort_spawn.size = 6
        config.flocking.cohort_spawn.radius = 1.0
        return World(
            config,
            simulation_paths=SimulationPaths(Path(".").resolve()),
        )

    def test_hot_path_failures_clear_leases_and_retry_cleanly(self) -> None:
        """Exercise test hot path failures clear leases and retry cleanly behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test hot path failures clear leases and retry cleanly test intent explicit.
        for failure_point in (
            "grid.rebuild",
            "grid.query",
            "flocking.accumulation",
            "vision.filtering",
            "vision.occlusion",
            "collision.evaluation",
        ):
            with self.subTest(failure_point=failure_point):
                world = self.make_world()
                hits = []

                def inject(point: str) -> None:
                    """Exercise inject behavior.
                    
                    Parameters
                    ----------
                    point
                        Value supplied to ``point`` by the test scenario.
                    
                    Returns
                    -------
                    None
                        The test completes through assertions.
                    
                    Raises
                    ------
                    RuntimeError
                        If runtime state violates the callable invariant.
                    """
                    # Keep the inject test intent explicit.
                    if point == failure_point:
                        hits.append(point)
                        raise RuntimeError(failure_point)

                try:
                    world._scheduler_validation_failure_injector = inject
                    with self.assertRaisesRegex(RuntimeError, failure_point):
                        world.update(world.fixed_timestep)
                    self.assertEqual(hits, [failure_point])
                    self.assertEqual(world._simulation_step, 0)
                    self.assertFalse(world._candidate_buffer_leased)
                    self.assertEqual(world._candidate_buffer.count, 0)
                    self.assertEqual(world.vision._candidate_count, 0)
                    self.assertEqual(world.vision._visible_count, 0)
                    self.assertEqual(world.vision._blocked_count, 0)
                    self.assertEqual(world.vision._visible_food_id_count, 0)
                    del world._scheduler_validation_failure_injector
                    world.update(0.0)
                    self.assertEqual(world._simulation_step, 1)
                    self.assertFalse(world._candidate_buffer_leased)
                finally:
                    world.close()


if __name__ == "__main__":
    unittest.main()
