# Milestone 2 deterministic multi-rate scheduler audit

## Result

Milestone 2 is implemented with a 60 Hz authoritative fixed step, staggered
20 Hz decisions, 20 Hz biology, 5 Hz aggregate statistics, the existing 10 Hz
observer deadline, and the existing 4 Hz pheromone diffusion schedule.

The complete repository `unittest` suite passes: 980 tests in 8.751 seconds.
The benchmark used seed 11, 55 creatures, 12 repetitions, and 8 samples per
repetition, giving 96 samples for each timing case. The honest baseline is Git
revision `2e0471c8ae481fbd767c46ecdd0c7dc04055f796`.

## Verified biology order

The previous relative order was:

1. survival fitness and age;
2. flocking benchmark fitness;
3. chronometers and reproduction cadence;
4. reproduction and nursing resource preparation;
5. digestion, upkeep, life/energy evaluation, and metabolism commit;
6. nursing and reproduction commit;
7. metabolic death processing;
8. mouth eating and stomach filling;
9. food, digestion, and trait-cost fitness bookkeeping.

The fixed-step order is now:

1. advance the fixed-step clock and run the historical pre-intention
   speciation hook;
2. calculate a fresh intention for the scheduled creature cohort;
3. apply cached continuous actions and safety corrections to all living
   creatures;
4. commit continuous communication using the fixed-step duration;
5. step Pymunk and synchronize motion, carried food, and boundaries;
6. detect and append mouth-contact exposure;
7. commit direct damage and remove immediate deaths;
8. at completed-step boundaries 3, 6, and so on, advance survival/age,
   flocking fitness, chronometers, and reproduction cadence, then resolve
   accumulated eating before resource candidates, digestion, and metabolism;
9. commit nursing and reproduction, process metabolic deaths, and perform
   dependent fitness bookkeeping;
10. run pheromone accumulation, food spawning, telemetry, observer, and
    counterfactual timebases;
11. refresh aggregate statistics when due;
12. increment `_simulation_step` after the fixed step returns successfully.

Eating deliberately moved before resource-candidate evaluation. Without that
change, the 20 Hz digestion pass would use stale stomach state. A contact on
the biology-boundary physics step can consequently participate in that
boundary's batched digestion, whereas the old 60 Hz pipeline made an end-step
bite available on the following step.

## Mouth exposure and deaths

Exposure is replayed chronologically by physics step, stable food ID, then
stable creature ID. Primitive parallel arrays and reusable active-count/order
storage avoid per-contact record allocation. Each creature and food can win at
most one successful claim per physics step. Failed claims do not block a later
valid claimant; missing creatures, missing food, and depleted food are skipped.

Exposure remains authoritative until the entire biology operation succeeds.
Resolver and downstream failures restore stomach, food, and carry state while
retaining the active records for retry. Checkpoint version 19 persists the
active portion of the buffer.

Direct-damage deaths are removed after contacts and before biology; their
records are rejected during validation. Metabolic/starvation deaths are
removed after biological commits and before pheromones, spawning, telemetry,
observer, or counterfactual consumers. Centralized removal checks prevent a
creature from being removed twice.

## Decisions, smoothing, and action semantics

`Creature.creature_id` is a persisted monotonic integer. The phase is
`creature_id % decision_period_steps`, and a cohort executes when the current
zero-based step modulo the period equals that phase. Selection, population
order, removal, birth, and loading cannot change it.

Decision-level work remains at 20 Hz: sensing, NEAT activation, raw-output
herding filtering, biome-memory adaptation, social/flocking intent, and target
blending. `decision_dt` is the full three-step duration. Herding uses an
elapsed-time conversion referenced to its historical 30 Hz response.

Physics-level work remains at 60 Hz: rest response, acceleration and rotation
smoothing, turn response/damping, angular retention, steering, collision
avoidance, and speed limits. Historical 60 Hz coefficients are converted by
elapsed time. Equal-duration step-response and turn-trajectory tests cover
60 Hz and 120 Hz application.

Current outputs are classified as follows:

- continuous: acceleration, rotation, panic/sprint, herding, rest, acoustic
  intensity/tone, and trail/alarm pheromone intensity;
- level-triggered and cadence-gated: eating, nursing, and reproduction;
- fresh-decision edges: chronometer reset, grab, and release.

Reproduction eligibility consumes its due flag once per biology cadence.
Grab, release, and reset execute only with a fresh decision. Eating is sampled
through physics-rate contact exposure, while nursing is integrated once using
`biology_dt`.

Pheromone emission is `deposit_rate * intensity * fixed_dt`. Acoustic output
replaces continuous emitter state on every physics step instead of appending
semantic events, and disappears when the next cached level is inactive.
Communication energy uses per-second acoustic/pheromone rates multiplied by
the complete biology duration; fitness trait-cost bookkeeping uses the same
quantity.

A selected creature without a scheduled decision displays neutral/waiting
state. It does not sense, activate NEAT, publish input caches, update flocking,
replay an edge, communicate, affect fitness, or advance scheduler state.

## Persistence and render backlog

`_simulation_step` is the sole mutable completed-step count;
`physics_step_count` is a read-only property. Current checkpoints persist the
step, unresolved exposure, observer/counterfactual deadlines and focus
generation, and the pre-existing authoritative world state. Runtime action,
sensor, social, and motion-command caches restart neutral.

Frame accumulator debt, requested/completed/dropped session totals, and
effective-speed metrics are not persisted because they describe a renderer
session rather than simulated history. Loading starts these values at zero.
Legacy checkpoints derive the step from completed simulated time, ignore any
legacy accumulator, and begin with no exposure.

The scheduler admits requested scaled time up to 60 fixed steps of backlog,
executes at most five per rendered frame, and reports overflow as dropped time.

## Exact boundary semantics over 60 completed steps

- physics, motion, contact detection, food-spawner time, speciation time, and
  communication: 60 applications;
- decisions: phase 0 can execute before the first physics step; phases 1 and 2
  first execute on steps 1 and 2; every creature executes 20 decisions;
- biology: no time-zero update; 20 passes on completions 3, 6, ..., 60;
- statistics: one initialization refresh, then five periodic refreshes on
  completions 12, 24, ..., 60;
- observer: one zero-origin sample and ten periodic samples through 1 second;
- counterfactual: zero-origin deadline and 0.2-second periodic deadlines;
  submission remains conditional on an eligible selected behavior;
- pheromone diffusion: no initial update; updates at 0.25, 0.5, 0.75, and
  1.0 seconds;
- flocking telemetry: no initial periodic capture; the default first capture
  occurs at 1.0 second;
- reproduction cadence: advanced by 20 biology calls and becomes due once at
  the one-second boundary.

## Performance

| Case | Baseline median / p95 / noise | Final median / p95 / noise | Median improvement |
|---|---:|---:|---:|
| Phase 0 | 8.232 / 9.131 ms / 3.086% | 4.979 / 5.259 ms / 2.816% | 39.5% |
| Phase 1 | 8.174 / 9.002 ms / 4.191% | 5.035 / 5.460 ms / 3.954% | 38.4% |
| Phase 2 | 8.338 / 8.985 ms / 2.116% | 5.908 / 6.813 ms / 2.796% | 29.1% |
| Three-step cycle | 23.694 / 25.175 ms / 2.683% | 15.788 / 16.975 ms / 1.400% | 33.4% |
| Five-step frame | 39.438 / 42.323 ms / 2.906% | 26.853 / 27.703 ms / 1.248% | 31.9% |
| Sixty steps | 478.811 / 494.306 ms / 1.334% | 312.546 / 317.488 ms / 0.628% | 34.7% |

All complete-cycle, five-step-frame, and 60-step improvements exceed measured
noise. The final scheduler performs 1,100 decision activations for the measured
population and interval, exactly one third of the original architectural 60 Hz
decision baseline's 3,300 activations.

Raw data is in `milestone2_baseline.json` and `milestone2_final.json`.

## Remaining risk and recommendation

The completed-step counter and mouth-exposure transaction are failure-safe,
but an arbitrary exception elsewhere in a fixed step is not a general rollback
transaction for every world mutation. It is surfaced and does not advance the
authoritative step.

The recommended next milestone is a long-horizon deterministic replay and
failure-injection soak with asynchronous checkpoint capture, heavy population
churn, and observer backpressure.
