from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, hypot, isfinite, pi

from configs.sim_config import FlockingBenchmarkConfig
from src.creature.flocking import SocialObservation


def flocking_benchmark_quality(
    observation: SocialObservation,
    config: FlockingBenchmarkConfig,
) -> float:
    """Calculate the instantaneous benchmark quality without retaining a snapshot.

Parameters
----------
observation
    Input used by this creature-domain operation.
config
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
    # Keep flocking benchmark quality behavior explicit in its owning subsystem.
    if not config.enabled or observation.effective_count <= 0.0:
        return 0.0
    group_presence = min(
        max(
            observation.effective_count
            / max(1, config.target_group_size - 1),
            0.0,
        ),
        1.0,
    )
    alignment_quality = min(
        max(1.0 - observation.mean_heading_error / pi, 0.0),
        1.0,
    )
    spacing_quality = exp(
        -(
            (observation.mean_neighbor_distance - config.target_spacing)
            / config.spacing_tolerance
        )
        ** 2
    )
    movement_quality = min(
        hypot(*observation.mean_group_velocity) / config.reference_speed,
        1.0,
    )
    return (
        group_presence
        * alignment_quality
        * spacing_quality
        * movement_quality
    )


@dataclass(slots=True)
class CreatureTelemetry:
    """Passive lifetime diagnostics with no role in reproduction selection."""

    age_seconds: float = 0.0
    food_eaten: int = 0
    distance_traveled: float = 0.0
    last_reproduction_age: float = -1_000_000.0
    offspring_count: int = 0
    matured_offspring_ids: list[int] = field(default_factory=list)
    evaluation_start_age_seconds: float = 0.0
    flocking_benchmark_reward: float = 0.0
    lifetime_energy_ingested: float = 0.0
    lifetime_energy_spent: float = 0.0
    _legacy_energy_gained: float = field(
        default=0.0,
        repr=False,
        compare=False,
    )

    def record_tick(self, delta_time: float, speed: float) -> None:
        """Execute record tick behavior.

Parameters
----------
delta_time
    Input used by this creature-domain operation.
speed
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep record tick behavior explicit in its owning subsystem.
        self.age_seconds += delta_time
        self.distance_traveled += max(0.0, speed) * max(0.0, delta_time)

    def record_food(self, depleted: bool = True) -> None:
        """Execute record food behavior.

Parameters
----------
depleted
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep record food behavior explicit in its owning subsystem.
        if depleted:
            self.food_eaten += 1

    def record_reproduction(self) -> None:
        """Execute record reproduction behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep record reproduction behavior explicit in its owning subsystem.
        self.last_reproduction_age = self.age_seconds
        self.offspring_count += 1

    def record_energy_transaction(
        self,
        *,
        ingested: float = 0.0,
        spent: float = 0.0,
    ) -> None:
        """Record realized energy flows without producing a selection score.

        Parameters
        ----------
        ingested
            Usable energy credited during the committed transaction.
        spent
            Energy expended during the committed transaction.

        Returns
        -------
        None
            The passive lifetime counters are updated in place.
        """
        # Sanitize each external quantity before adding it to the lifetime ledger.
        for name, value in (("ingested", ingested), ("spent", spent)):
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                numeric = 0.0
            if not isfinite(numeric):
                numeric = 0.0
            if name == "ingested":
                self.lifetime_energy_ingested += max(0.0, numeric)
            else:
                self.lifetime_energy_spent += max(0.0, numeric)

    @property
    def net_energy_balance(self) -> float:
        """Return lifetime ingestion minus realized expenditure.

        Parameters
        ----------
        None
            This property receives no external parameters.

        Returns
        -------
        float
            Signed lifetime usable-energy balance.
        """
        # Keep the diagnostic derived so it cannot drift from its counters.
        return self.lifetime_energy_ingested - self.lifetime_energy_spent

    @property
    def net_metabolic_rate(self) -> float:
        """Return net energy balance per second of life.

        Parameters
        ----------
        None
            This property receives no external parameters.

        Returns
        -------
        float
            Signed net balance normalized by at least one second.
        """
        # The one-second floor keeps newborn diagnostics finite.
        return self.net_energy_balance / max(self.age_seconds, 1.0)

    @property
    def lifetime_offspring_count(self) -> int:
        """Expose the passive offspring total with physiological naming.

        Parameters
        ----------
        None
            This property receives no external parameters.

        Returns
        -------
        int
            Number of successfully committed offspring.
        """
        # Preserve the older serialized field while exposing current terminology.
        return self.offspring_count

    def seconds_since_reproduction(self) -> float:
        """Execute seconds since reproduction behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep seconds since reproduction behavior explicit in its owning subsystem.
        return self.age_seconds - self.last_reproduction_age

    def record_flocking_benchmark(
        self,
        observation: SocialObservation,
        delta_time: float,
        config: FlockingBenchmarkConfig,
    ) -> float:
        """Execute record flocking benchmark behavior.

Parameters
----------
observation
    Input used by this creature-domain operation.
delta_time
    Input used by this creature-domain operation.
config
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep record flocking benchmark behavior explicit in its owning subsystem.
        quality = flocking_benchmark_quality(observation, config)
        return self.record_flocking_benchmark_quality(
            quality,
            delta_time,
            config,
        )

    def record_flocking_benchmark_quality(
        self,
        quality: float,
        delta_time: float,
        config: FlockingBenchmarkConfig,
    ) -> float:
        """Record a precomputed scalar quality without a diagnostic object.

Parameters
----------
quality
    Input used by this creature-domain operation.
delta_time
    Input used by this creature-domain operation.
config
    Input used by this creature-domain operation.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep record flocking benchmark quality behavior explicit in its owning subsystem.
        if not config.enabled or quality <= 0.0:
            return 0.0
        before = self.flocking_benchmark_reward
        self.flocking_benchmark_reward = min(
            config.max_per_evaluation,
            before
            + quality
            * max(0.0, delta_time)
            * config.reward_rate,
        )
        return self.flocking_benchmark_reward - before

    def average_speed(self) -> float:
        """Execute average speed behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep average speed behavior explicit in its owning subsystem.
        evaluation_age = self.evaluation_age_seconds()
        if evaluation_age <= 0.0:
            return 0.0
        return self.distance_traveled / evaluation_age

    def evaluation_age_seconds(self) -> float:
        """Execute evaluation age seconds behavior.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
float
    Result produced by this creature-domain operation."""
        # Keep evaluation age seconds behavior explicit in its owning subsystem.
        return max(0.0, self.age_seconds - self.evaluation_start_age_seconds)

    def __getstate__(self) -> dict[str, object]:
        """Exclude one-shot legacy migration data from new checkpoints.

Parameters
----------
None
    This callable receives no external parameters.
Returns
-------
dict[str, object]
    Result produced by this creature-domain operation."""
        # Keep getstate behavior explicit in its owning subsystem.
        return {
            "age_seconds": self.age_seconds,
            "food_eaten": self.food_eaten,
            "distance_traveled": self.distance_traveled,
            "last_reproduction_age": self.last_reproduction_age,
            "offspring_count": self.offspring_count,
            "matured_offspring_ids": self.matured_offspring_ids,
            "evaluation_start_age_seconds": self.evaluation_start_age_seconds,
            "flocking_benchmark_reward": self.flocking_benchmark_reward,
            "lifetime_energy_ingested": self.lifetime_energy_ingested,
            "lifetime_energy_spent": self.lifetime_energy_spent,
        }

    def __setstate__(self, state: object) -> None:
        """Load current state and retain old energy totals for checkpoint migration.

Parameters
----------
state
    Input used by this creature-domain operation.
Returns
-------
None
    Result produced by this creature-domain operation."""
        # Keep setstate behavior explicit in its owning subsystem.
        values: dict[str, object]
        if (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[1], dict)
        ):
            values = state[1]
        elif isinstance(state, dict):
            values = state
        else:
            values = {}

        self.age_seconds = float(values.get("age_seconds", 0.0))
        self.food_eaten = int(values.get("food_eaten", 0))
        self.distance_traveled = float(values.get("distance_traveled", 0.0))
        self.last_reproduction_age = float(
            values.get("last_reproduction_age", -1_000_000.0)
        )
        self.offspring_count = int(values.get("offspring_count", 0))
        self.matured_offspring_ids = list(
            values.get("matured_offspring_ids", [])
        )
        self.evaluation_start_age_seconds = float(
            values.get("evaluation_start_age_seconds", 0.0)
        )
        self.flocking_benchmark_reward = float(
            values.get("flocking_benchmark_reward", 0.0)
        )
        self.lifetime_energy_ingested = float(
            values.get(
                "lifetime_energy_ingested",
                values.get("energy_gained", 0.0),
            )
        )
        self.lifetime_energy_spent = float(
            values.get("lifetime_energy_spent", 0.0)
        )
        try:
            legacy_energy = float(values.get("energy_gained", 0.0))
        except (TypeError, ValueError, OverflowError):
            legacy_energy = 0.0
        self._legacy_energy_gained = (
            max(0.0, legacy_energy) if isfinite(legacy_energy) else 0.0
        )


# Older imports and pickles resolve this name, but the object is telemetry only.
CreatureFitness = CreatureTelemetry
