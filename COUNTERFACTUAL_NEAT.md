# Counterfactual NEAT WHY

The WHY inspector is a local mechanistic explanation of the selected
creature's current feed-forward NEAT decision. It asks how the same evolved
network's action requests change under a biologically meaningful sensory or
internal-state counterfactual. It does not claim complete biological causation
and never affects behavior classification, actions, physics, fitness, or
evolution.

## Data flow

`World` reuses the factual input/output vectors from one completed live brain
decision. At 5 Hz, and only while a mapped emerging or active behavior exists,
it submits a compact immutable probe. The worker caches a serialized
`PureNeatEvaluator` once per selected creature and `brain_revision`.

The existing behavior-observer process owns separate queues with this strict
priority:

1. drain and process pending temporal behavior observations;
2. apply the newest focal-brain control update;
3. advance at most one counterfactual network activation;
4. return immediately to behavior observations.

WHY probes are latest-wins and cooperatively resumable. New factual state may
supersede partial explanatory work. Temporal ethogram freshness always takes
priority over WHY completeness.

## Semantic interventions

- `VISIBLE_FOOD_CUES`: no visible food count, proximity, or direction.
- `RESOURCE_GRADIENT_CUES`: neutral local fertility/smell and spatial/temporal
  gradients.
- `SATIATED_STATE`: the live biological semantics for full energy and stomach,
  which derive zero feeding drive. It asks, “What would the same brain request
  if this creature were satiated?”
- `SOCIAL_CUES`: no generic creature target, compatible flock, or long-range
  social signal.
- `OFFSPRING_CUES`: no own infant target.
- `ACOUSTIC_CUES`: the default absent acoustic observation.
- `TRAIL_PHEROMONE_CUES` and `ALARM_PHEROMONE_CUES`: baseline channel
  concentrations.
- `WALL_CUES`: the live absent wall-target encoding.

Indices are resolved from `SENSOR_INPUT_NAMES`. Direction values are replaced
only with their associated presence/strength values, and every unrelated input
is preserved.

## Scoring and interpretation

The worker stores factual, counterfactual, and signed delta values for the
complete action vector. UI influence is the mean absolute normalized change
over behavior-scored outputs. Signed output span is two; one-sided intent span
is one.

- below `0.10`: minimal;
- below `0.30`: weak;
- below `0.60`: moderate;
- otherwise: strong.

Each output is supportive, suppressive, reversing, or minimal. A semantic
effect may be `MIXED` when relevant outputs disagree. Material reversal of a
behavior-critical signed output takes precedence. Scores are not probabilities
and are never normalized to sum to 100.

Food orientation scores rotation alone; acceleration is secondary context.
Food approach scores acceleration and rotation. Resting has no direct neural
WHY in v1 because it is defined from realized locomotion and has no dedicated
NEAT rest output.

Per-behavior histories are keyed by creature, selection generation, brain
revision, behavior, bout ID, and directional target ID. The latest 64 samples
are aggregated with medians. Results are rejected after focus, brain, bout, or
food-target changes.

### Target-relative food steering

For `FOOD_ORIENTATION` and `FOOD_APPROACH`, raw rotate influence remains
`abs(actual - counterfactual) / 2`, but its direction is evaluated relative to
the factual food heading. Positive food angle and positive rotate are both
counter-clockwise. Outside the configured centered-target dead zone, alignment
is `sign(food_relative_angle) * rotate`; positive values steer toward the food
and negative values steer away.

Within the default `0.05` radian dead zone, alignment is `-abs(rotate)`, so a
near-zero request is correctly treated as stabilizing relative to a large
unnecessary turn. Material target-alignment sign crossings are `REVERSING`;
otherwise better factual alignment is `SUPPORTIVE` and better counterfactual
alignment is `SUPPRESSIVE`.

The food visibility, target ID, and relative angle come from the cached sensor
snapshot that produced the factual NEAT inputs and outputs. This immutable
reference is not altered by a food-removal counterfactual. Food-oriented WHY
is deferred when that factual context is missing or no longer matches the
observed bout; other simultaneous explanations continue normally.

## Inspector layout

The WHY page mirrors the BEHAVIOURS page: every `BehaviorKind` always occupies
one fixed collapsed card in enum order. New probes update text and bars inside
those cards without inserting, removing, or reordering them. This prevents
active bouts and ranked interventions from moving the rest of the inspector.

Clicking a card expands only that behavior. Its expanded height is derived
from the behavior's static explanation mapping rather than the current result,
so waiting, calculating, and completed states keep identical geometry.
Intervention detail cards also remain in semantic mapping order. Inactive
details show placeholders until the relevant bout has a completed probe.

The persistent “How the values are calculated” section documents the
factual/counterfactual comparison, natural-span normalization, mean across
scored outputs, label thresholds, direction meanings, and the limitation that
local influences are neither causal percentages nor values that sum to 100%.

## Validation

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_counterfactual_neat \
  tests.test_world_counterfactual \
  tests.test_behavior_observer \
  tests.test_world_behavior_observer \
  tests.ui.test_renderer
```

Benchmark with:

```sh
python benchmarks/benchmark_counterfactual_neat.py
```

Manual validation should compare visible-food-driven and gradient-driven
approaches, inspect satiation-sensitive genomes, observe supportive,
suppressive, reversing, and mixed effects, switch focus rapidly, and verify
that actual simulation decisions remain unchanged.
