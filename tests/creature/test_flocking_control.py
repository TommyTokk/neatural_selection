from __future__ import annotations

import math
import unittest

from configs.sim_config import (
    FlockingConfig,
    FlockingTelemetryConfig,
    SocialCompatibilityConfig,
)
from src.flocking import (
    FlockingWeights,
    SocialIntent,
    accepted_counterfactual_contribution,
    blend_desired_velocity,
    calculate_flocking_weights,
    calculate_social_intent,
    configured_social_influence,
)


class FlockingWeightsTest(unittest.TestCase):
    def test_personal_space_is_not_socially_gated(self) -> None:
        """Exercise test personal space is not socially gated behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test personal space is not socially gated test intent explicit.
        weights = calculate_flocking_weights(
            herding=0.0,
            panic=1.0,
            separation_gene=0.8,
            alignment_gene=0.6,
            cohesion_gene=0.4,
            personal_space_presence=1.0,
            social_presence=0.0,
        )
        self.assertEqual(weights.separation, 0.8)
        self.assertEqual(weights.alignment, 0.0)
        self.assertEqual(weights.cohesion, 0.0)

    def test_full_panic_retains_configured_fraction(self) -> None:
        """Exercise test full panic retains configured fraction behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test full panic retains configured fraction test intent explicit.
        weights = calculate_flocking_weights(
            herding=1.0,
            panic=1.0,
            separation_gene=1.0,
            alignment_gene=1.0,
            cohesion_gene=1.0,
            personal_space_presence=1.0,
            social_presence=1.0,
            panic_suppression_strength=0.5,
        )
        self.assertEqual(weights.panic_attenuation, 0.5)
        self.assertEqual(weights.alignment, 0.5)
        self.assertEqual(weights.cohesion, 0.5)

    def test_absence_disables_every_social_component(self) -> None:
        """Exercise test absence disables every social component behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test absence disables every social component test intent explicit.
        weights = calculate_flocking_weights(
            herding=1.0,
            panic=0.0,
            separation_gene=1.0,
            alignment_gene=1.0,
            cohesion_gene=1.0,
            personal_space_presence=0.0,
            social_presence=0.0,
        )
        self.assertEqual(weights.separation, 0.0)
        self.assertEqual(weights.alignment, 0.0)
        self.assertEqual(weights.cohesion, 0.0)

    def test_weights_are_bounded_when_inputs_are_out_of_range(self) -> None:
        """Exercise test weights are bounded when inputs are out of range behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test weights are bounded when inputs are out of range test intent explicit.
        weights = calculate_flocking_weights(
            herding=4.0,
            panic=-3.0,
            separation_gene=8.0,
            alignment_gene=-2.0,
            cohesion_gene=9.0,
            personal_space_presence=7.0,
            social_presence=5.0,
            minimum_social_engagement=3.0,
            panic_suppression_strength=-1.0,
        )
        for value in (
            weights.separation,
            weights.alignment,
            weights.cohesion,
            weights.engagement,
            weights.panic_attenuation,
        ):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class FlockingConfigurationTest(unittest.TestCase):
    def test_defaults_have_no_flocking_mode_switch(self) -> None:
        """Exercise test defaults have no flocking mode switch behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test defaults have no flocking mode switch test intent explicit.
        config = FlockingConfig()
        self.assertFalse(hasattr(config, "mode"))
        self.assertEqual(config.perception_radius, 150.0)
        self.assertEqual(config.herding_decay_rate, 0.15)
        self.assertFalse(config.long_range.enabled)
        self.assertFalse(config.cohort_spawn.enabled)
        self.assertTrue(config.benchmark.enabled)

    def test_string_compatibility_mode_is_normalized(self) -> None:
        """Exercise test string compatibility mode is normalized behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test string compatibility mode is normalized test intent explicit.
        config = FlockingConfig(
            compatibility=SocialCompatibilityConfig(mode="social_tag"),
        )
        self.assertEqual(config.compatibility.mode.value, "social_tag")

    def test_invalid_fractions_ranges_and_group_sizes_fail(self) -> None:
        """Exercise test invalid fractions ranges and group sizes fail behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test invalid fractions ranges and group sizes fail test intent explicit.
        invalid_factories = (
            lambda: FlockingConfig(max_social_influence=1.01),
            lambda: FlockingConfig(minimum_social_engagement=math.nan),
            lambda: FlockingConfig(herding_decay_rate=0.0),
            lambda: FlockingConfig(herding_decay_rate=-0.1),
            lambda: FlockingConfig(herding_decay_rate=math.nan),
            lambda: FlockingConfig(herding_decay_rate=math.inf),
            lambda: FlockingConfig(herding_decay_rate=1.01),
            lambda: FlockingConfig(perception_radius=0.0),
            lambda: FlockingConfig(preferred_personal_space=0.0),
            lambda: FlockingConfig(target_group_size=1),
            lambda: FlockingConfig(
                telemetry=FlockingTelemetryConfig(interval_seconds=0.1)
            ),
            lambda: FlockingConfig(
                compatibility=SocialCompatibilityConfig(
                    social_tag_sigma=0.0
                )
            ),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()

    def test_validate_rechecks_mutated_values(self) -> None:
        """Exercise test validate rechecks mutated values behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test validate rechecks mutated values test intent explicit.
        config = FlockingConfig()
        config.herding_decay_rate = math.inf
        with self.assertRaises(ValueError):
            config.validate()

    def test_legacy_herding_decay_rate_is_valid(self) -> None:
        """Exercise test legacy herding decay rate is valid behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test legacy herding decay rate is valid test intent explicit.
        config = FlockingConfig(herding_decay_rate=1.0)
        self.assertEqual(config.herding_decay_rate, 1.0)


class DesiredVelocityBlendingTest(unittest.TestCase):
    def test_zero_influence_returns_same_object_value(self) -> None:
        """Exercise test zero influence returns same object value behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test zero influence returns same object value test intent explicit.
        neural = (12.5, -4.0)
        self.assertEqual(
            blend_desired_velocity(neural, (100.0, 100.0), 0.0),
            neural,
        )

    def test_configured_influence_is_bounded(self) -> None:
        """Exercise test configured influence is bounded behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test configured influence is bounded test intent explicit.
        config = FlockingConfig(max_social_influence=0.35)
        intent = SocialIntent(
            confidence=1.0,
            weights=FlockingWeights(
                separation=1.0,
                alignment=1.0,
                cohesion=1.0,
                engagement=1.0,
            ),
        )
        self.assertEqual(configured_social_influence(config, intent), 0.35)

    def test_zero_confidence_disables_configured_influence(self) -> None:
        """Exercise test zero confidence disables configured influence behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test zero confidence disables configured influence test intent explicit.
        config = FlockingConfig(max_social_influence=0.35)
        intent = SocialIntent(
            confidence=0.0,
            weights=FlockingWeights(engagement=1.0),
        )

        self.assertEqual(configured_social_influence(config, intent), 0.0)

    def test_engagement_is_not_applied_twice(self) -> None:
        """Exercise test engagement is not applied twice behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test engagement is not applied twice test intent explicit.
        config = FlockingConfig(max_social_influence=0.35)
        low_engagement = SocialIntent(
            confidence=0.4,
            weights=FlockingWeights(engagement=0.1),
        )
        full_engagement = SocialIntent(
            confidence=0.4,
            weights=FlockingWeights(engagement=1.0),
        )

        expected = 0.35 * 0.4
        self.assertAlmostEqual(
            configured_social_influence(config, low_engagement),
            expected,
        )
        self.assertAlmostEqual(
            configured_social_influence(config, full_engagement),
            expected,
        )

    def test_default_one_neighbor_social_influence_is_functional(self) -> None:
        """Exercise test default one neighbor social influence is functional behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test default one neighbor social influence is functional test intent explicit.
        config = FlockingConfig()
        weights = calculate_flocking_weights(
            herding=0.5,
            panic=0.0,
            separation_gene=0.5,
            alignment_gene=0.5,
            cohesion_gene=0.5,
            personal_space_presence=0.0,
            social_presence=1.0,
            minimum_social_engagement=config.minimum_social_engagement,
            panic_suppression_strength=config.panic_suppression_strength,
        )
        intent = calculate_social_intent(
            current_velocity=(0.0, 0.0),
            separation_velocity=(0.0, 0.0),
            alignment_velocity=(10.0, 0.0),
            cohesion_velocity=(0.0, 10.0),
            weights=weights,
            effective_count=1.0,
            target_group_size=config.target_group_size,
            max_speed=170.0,
        )

        self.assertAlmostEqual(
            configured_social_influence(config, intent),
            0.068359375,
        )
        self.assertGreater(
            configured_social_influence(config, intent),
            0.05,
        )

    def test_full_configured_influence_produces_expected_blend(self) -> None:
        """Exercise test full configured influence produces expected blend behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test full configured influence produces expected blend test intent explicit.
        self.assertEqual(
            blend_desired_velocity((10.0, 0.0), (0.0, 20.0), 0.35),
            (6.5, 7.0),
        )

    def test_counterfactual_uses_same_avoidance_and_budget(self) -> None:
        """Exercise test counterfactual uses same avoidance and budget behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test counterfactual uses same avoidance and budget test intent explicit.
        blended, neural, contribution = accepted_counterfactual_contribution(
            blended_request=(-5.0, 20.0),
            neural_request=(-10.0, 0.0),
            mandatory_avoidance=(10.0, 0.0),
            remaining_budget=25.0,
        )
        self.assertGreaterEqual(blended[0], 0.0)
        self.assertGreaterEqual(neural[0], 0.0)
        self.assertEqual(
            contribution,
            (blended[0] - neural[0], blended[1] - neural[1]),
        )

    def test_redesigned_allocation_retains_neural_counterfactual(self) -> None:
        """Exercise test redesigned allocation retains neural counterfactual behavior.
        
        Parameters
        ----------
        None
            This callable receives no external parameters.
        
        Returns
        -------
        None
            The test completes through assertions.
        """
        # Keep the test redesigned allocation retains neural counterfactual test intent explicit.
        blended, neural, contribution = accepted_counterfactual_contribution(
            blended_request=(60.0, 20.0),
            neural_request=(80.0, 0.0),
            mandatory_avoidance=(0.0, 10.0),
            remaining_budget=100.0,
        )
        self.assertEqual(neural, (80.0, 0.0))
        self.assertNotEqual(blended, contribution)
        self.assertEqual(
            blended,
            (neural[0] + contribution[0], neural[1] + contribution[1]),
        )


if __name__ == "__main__":
    unittest.main()
