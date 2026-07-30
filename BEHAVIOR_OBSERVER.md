# Real-Time Temporal Ethogram

The temporal ethogram describes the realized world and action history of the
currently selected creature. It does not infer neural intent, modify the NEAT
graph, feed results back into the simulation, or use the genome-derived
ethogram in `src/analysis.py`.

## Runtime pipeline

After each completed fixed simulation step, `World` may produce one immutable
observation at the next 10 Hz simulation-time boundary. It uses the latest
cached vision and flocking snapshots, so observation creation does not rerun
vision or rescan the population.

Observations enter a bounded queue without blocking the simulation. If the
queue is full, the oldest item is discarded and the newest is retried once.
A lazily started, spawn-safe worker owns exactly one temporal deque and one set
of bout states. Changing the selected creature or selection generation clears
that state completely. Results use the same latest-wins queue policy.

`World.update` drains worker results even while the simulation is paused and
retains only a result whose creature ID and selection generation match the
current focus. Pausing stops simulation-time sampling.

## Behaviours

The initial high-evidence rules are:

- `FOOD_ORIENTATION`: the same food target persists for at least three
  samples, absolute angle error falls by 0.15 rad/s with at least two-thirds
  improving steps, and realized angular velocity turns toward it at
  0.10 rad/s or more.
- `FOOD_APPROACH`: the same visible target closes at 8 px/s or more, at least
  two-thirds of distance steps close, and realized velocity points toward it
  with cosine alignment of at least 0.35.
- `FEEDING`: the focal creature's cumulative explicit consumption count rises
  and cumulative swallowed energy rises. Intent, proximity, and stomach state
  cannot establish feeding.
- `RESTING`: current realized speed is at most 2 px/s and at least 80% of
  samples in the focal segment are at or below that threshold.

The secondary rules are:

- `COHESION`: a compatible group is present for at least 60% of the segment
  and the creature remains outside collision-separation range for at least
  80%; it must then either close on the group center at 5 px/s while moving
  toward it, or align its realized velocity with the group by at least 0.75
  while separation remains stable.
- `ALARM_RETREAT`: local alarm is at least 0.10, forward samples are at least
  0.02 lower, exposure falls by at least 0.03/s with two-thirds consistency,
  and realized forward locomotion is at least 10 px/s. This label establishes
  retreat down an alarm-pheromone gradient only. A broader threat-response
  label is deferred until the simulation exposes a reliable threat source.

Rule-derived Evidence values are clamped to `[0, 1]`. They expose how strongly
the observations satisfy a rule and are not calibrated probabilities.

## Bouts and UI

Ordinary bouts first appear as `EMERGING`, become `ACTIVE` after 0.5 simulation
seconds, and tolerate 0.3 seconds without matching evidence before ending.
Feeding activates immediately and remains visible for 0.75 simulation seconds
after the last consumption event.

The Brain window keeps the neural graph visible and adds a `NODE | BEHAVIOURS`
selector to its right panel. The behaviour page identifies the source as
world/action history and keeps one fixed card visible for every behaviour in
the enum's stable order. Inactive cards remain dim; emerging and active cards
brighten from their existing rule-derived Evidence score and show their status,
duration, numeric Evidence, and Evidence bar. Clicking a card expands it in
place with a concise explanation of the realized conditions that activate that
behaviour; only one card expands at a time. Static explanation text and cached
wrapping keep this presentation independent from the live behaviour
calculation. A fixed-height observer-status slot remains above the larger card
stack, so cards keep the same coordinates when bouts appear or disappear. It
does not add decay, smoothing, or a second behaviour calculation. Observer
diagnostics are visible only with vision debugging enabled. Selecting a graph
node returns the panel to `NODE`; reopening the window also defaults to `NODE`.

## Configuration, lifecycle, and validation

Defaults live in `BehaviorObserverConfig` under `SimConfig.behavior`. The
worker starts only after the first non-null selection. On shutdown it receives
a stop signal and gets up to two seconds to join before a termination fallback;
queue feeder threads are then closed. Temporal history and process state are
not checkpointed.

Run deterministic coverage with:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  tests.test_behavior_observer \
  tests.test_world_behavior_observer \
  tests.ui.test_renderer
```

Run the standalone throughput and latency probe with:

```sh
PYTHONDONTWRITEBYTECODE=1 python benchmarks/benchmark_behavior_observer.py
```

For manual validation, switch focus between living creatures, pause and resume,
observe food orientation preceding approach, verify that feeding appears only
after consumption, and inspect sustained resting, cohesion, alarm retreat,
Evidence presentation, delayed/error states, and clean application exit.
