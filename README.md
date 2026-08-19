# Creature Modelling and Functioning in an Evolving Artificial-Life System

## Abstract

This project models a population of autonomous herbivorous creatures whose morphology, metabolism, perception, social tendencies, and neural controllers can change through continuous reproduction and mutation. A creature is not represented by a single state vector alone: the implementation separates inherited genotype, transient physiological state, rigid-body dynamics, neural decision state, ancestry, and observational records. Behaviour emerges from the repeated coupling of perception, feed-forward NEAT activation, action execution, resource accounting, and selection. No explicit objective such as “move toward food” is programmed into the controller; instead, creatures that gather more environmental energy are preferentially selected as parents, while the energetic costs of sensing, motion, neural complexity, digestion, communication, and reproduction constrain viable strategies.

This report describes the current computational model, its principal mathematical formulations, and the mechanisms used to analyse emergent behaviour. Environmental quantities are considered only where they enter a creature's sensors, motion, resource balance, communication, reproduction, or survival.

_Primary implementation: [creature domain package](src/creature/) · [simulation integration](src/world.py) · [simulation parameters](configs/sim_config.py)_

## Table of contents

1. [Modelling scope and execution cycle](#1-modelling-scope-and-execution-cycle)
2. [Creature state and genotype](#2-creature-state-and-genotype)
3. [Perception and sensor formulation](#3-perception-and-sensor-formulation)
4. [Neural decision model and actions](#4-neural-decision-model-and-actions)
5. [Locomotion and social dynamics](#5-locomotion-and-social-dynamics)
6. [Communication](#6-communication)
7. [Feeding, metabolism, rest, and vitality](#7-feeding-metabolism-rest-and-vitality)
8. [Reproduction, development, and evolution](#8-reproduction-development-and-evolution)
9. [Fitness and speciation](#9-fitness-and-speciation)
10. [Behavioural analysis and explainability](#10-behavioural-analysis-and-explainability)
11. [Concluding interpretation](#11-concluding-interpretation)

## Notation

The operator

$$
\operatorname{clip}_{[a,b]}(x)=\min(b,\max(a,x))
$$

is used throughout. Bold lower-case symbols denote two-dimensional vectors; $\lVert\mathbf{x}\rVert$ is the Euclidean norm; $\Delta t$ is simulated elapsed time; and normalized intensities lie in $[0,1]$ unless stated otherwise. Configuration values quoted below are the current defaults rather than universal constants of the model.

_Implementation: [common numerical helpers](src/creature/common.py) · [configuration](configs/sim_config.py)_

---

## 1. Modelling scope and execution cycle

### 1.1 Artificial-life formulation

Each creature is an embodied, autonomous agent. Its body occupies continuous two-dimensional space; its sensors transform local physical and biological conditions into a fixed neural input vector; its feed-forward neural genome maps that vector to action intensities; and the resulting motion and biological transactions alter both the creature and its surroundings. Reproduction is asexual: a selected parent supplies the child's inherited traits and neural genome, after which both are independently mutated. The population therefore implements a continuous evolutionary process rather than a sequence of isolated generations.

_Implementation: [creature entity](src/creature/model.py) · [evolution coordinator](src/creature/evolution.py) · [real-time NEAT manager](src/creature/neat/rt_neat.py)_

The model distinguishes four kinds of state:

1. **Hereditary state:** vision, morphology, digestive traits, flocking genes, social tags, colour, and the NEAT genome.
2. **Physical and physiological state:** position, velocity, heading, energy, life, stomach contents, age, and carried food.
3. **Controller state:** the instantiated neural network, smoothed action values, internal chronometers, and cached social intention.
4. **Historical state:** lineage, fitness measurements, completed behavioural bouts, species records, and archived genomes.

This separation prevents temporary conditions—such as infant movement penalties or senescence—from accidentally changing inherited traits.

_Implementation: [genotype](src/creature/genotype.py) · [live state](src/creature/model.py) · [runtime services](src/creature/runtime/) · [historical fitness](src/creature/fitness.py)_

### 1.2 Deterministic multi-rate cycle

The simulation uses a fixed physical frequency of $60\,\mathrm{Hz}$. Neural decisions and biological updates occur every three physical steps, hence at $20\,\mathrm{Hz}$, while aggregate statistics are refreshed every twelve steps, hence at $5\,\mathrm{Hz}$. A fixed-step accumulator limits work per external update without changing the duration of an accepted simulation step. This is important biologically: forces, bite exposures, pheromone evolution, ageing, digestion, reproduction, and energy expenditure are calculated from simulated time rather than from the rate at which external frames arrive.

| Process | Default frequency | Default period |
|---|---:|---:|
| Rigid-body integration | $60\,\mathrm{Hz}$ | $1/60\,\mathrm{s}$ |
| Neural decisions | $20\,\mathrm{Hz}$ | $3$ physical steps |
| Biological/resource update | $20\,\mathrm{Hz}$ | $3$ physical steps |
| Population statistics | $5\,\mathrm{Hz}$ | $12$ physical steps |

_Implementation: [scheduler configuration](configs/sim_config.py) · [fixed-step execution](src/world.py) · [scheduler validation](tests/test_scheduler_validation.py)_

At a conceptual level, one cycle is

$$
\text{sense}\rightarrow\text{decide}\rightarrow\text{apply intent}
\rightarrow\text{integrate physics}\rightarrow\text{resolve biology}.
$$

Perception and decisions are staggered deterministically across creature identities, but cached actions remain active between decisions. Mouth contacts are accumulated at the physical rate and resolved chronologically before the next biological ledger. Consequently, short-lived contacts are not lost merely because biology runs at a lower frequency.

_Implementation: [decision and action phases](src/world.py) · [perception service](src/creature/runtime/perception.py) · [mouth-exposure transaction](src/world.py)_

---

## 2. Creature state and genotype

### 2.1 Embodied state

A live creature owns a unit-mass Pymunk rigid body and a circular collision shape. The body's moment of inertia is that of a solid circle with inherited radius $r$. Its authoritative position, heading, linear velocity, and angular velocity are held by the physics body. The convenience quantities used by other subsystems are

$$
\mathbf{p}=(x,y),\qquad \theta=\text{body angle},\qquad
v=\lVert\mathbf{v}\rVert.
$$

The collision shape has elasticity $0.15$ and zero friction; controlled anisotropic drag is applied explicitly instead of relying on surface friction. A newly created founder receives a random heading and an energy reserve sampled uniformly from $[0.55,0.95]$.

_Implementation: [creature model](src/creature/model.py) · [creature factory](src/creature/factory.py) · [collision categories](src/collision.py)_

The principal live physiological variables are usable energy $E$, life reserve $L$, stomach energy $S$, weighted stomach difficulty load $D_S$, lifetime gathered energy, resting intent, recent activity, and any direct damage awaiting the next resource transaction. Energy and life are distinct: exhaustion first creates unmet energetic demand, which is then converted into damage; a creature dies when life reaches zero, not merely when usable energy is temporarily empty.

_Implementation: [live physiological fields and ledger diagnostics](src/creature/model.py) · [resource ledger](src/creature/metabolism.py)_

### 2.2 Aggregate genotype

The non-neural genotype is the tuple

$$
G=(G_V,G_P,G_F,C),
$$

where $G_V$ contains vision traits, $G_P$ contains physical and digestive traits, $G_F$ contains social traits, and $C$ is inherited colour. The neural genome is managed separately because it follows NEAT's graph mutation and innovation bookkeeping. Both parts are coordinated atomically when an offspring is planned.

_Implementation: [genotype records](src/creature/genotype.py) · [neural brain](src/creature/neat/brain.py) · [evolution transaction](src/creature/evolution.py)_

| Trait | Meaning | Initial/default domain |
|---|---|---:|
| Vision range $R$ | Maximum visual reach | $[100,200]$ |
| Vision angle $\Phi$ | Total field of view | $[0.35,\pi]$ rad |
| Radius $r$ | Body and mouth scale | $[12,22]$ |
| Movement multiplier $m$ | Inherited locomotion cost factor | $[0.75,1.35]$ |
| Stomach capacity $K$ | Maximum stored food energy | $[0.8,2.6]$ |
| Digestion rate $q$ | Maximum stomach processing per second | $[0.05,0.40]$ |
| Digestion efficiency $\eta$ | Gross conversion fraction | $[0.55,0.98]$ |
| Separation, alignment, cohesion | Social steering genes | each in $[0,1]$ |
| Social tag $(t_x,t_y)$ | Heritable compatibility coordinates | each in $[0,1]$ |

Founder physical and flocking values are Gaussian perturbations around configured defaults, clipped to their admissible intervals; founder vision is uniformly sampled over its allowed ranges. All flocking genes and tags are normalized when constructed, so downstream social calculations can assume a valid unit interval.

_Implementation: [genotype initialization and bounds](src/creature/genotype.py) · [trait and vision defaults](configs/sim_config.py)_

### 2.3 Lineage and diagnostics

Every creature has a stable integer identity and a `LineageInfo` record containing its parent identity, generation, species identity, and the effective bounded change of every mutable non-neural trait. The recorded mutation is the difference after clipping, not the unbounded random proposal; it therefore describes the phenotype actually inherited by the child.

_Implementation: [lineage and mutation deltas](src/creature/genotype.py) · [offspring planning](src/creature/evolution.py)_

Resource diagnostics retain the latest complete transaction: digestive conversion, rest recovery, healing, total demand, movement-powered demand, deficits, direct damage, and final energy and life. These records do not determine biology; they expose the already resolved ledger for scientific inspection.

_Implementation: [diagnostic records](src/creature/model.py) · [ledger commit](src/creature/metabolism.py)_

---

## 3. Perception and sensor formulation

### 3.1 Sensor contract

The current sensing schema is version 7 and contains exactly 43 ordered inputs. This ordering is a formal interface: it must match the `num_inputs = 43` declaration used to construct every NEAT network. A schema change therefore represents a change in the meaning of evolved genomes, not a cosmetic renaming.

_Implementation: [sensor contract and serialization](src/creature/vision.py) · [NEAT input declaration](configs/neat_herbivore.ini) · [controller contract validation](src/creature/neat/controller.py)_

| Indices | Functional group | Ordered signals |
|---:|---|---|
| 1–4 | Endogenous resources | constant, feeding drive, reproductive readiness, energy percentage |
| 5–7 | Motion and local counts | speed, creature count, food count |
| 8–10 | Internal time | alternating clock, resettable chronometer, normalized age |
| 11–16 | Nearest visible targets | food proximity/angle, creature proximity/angle, wall proximity/angle |
| 17–20 | Manipulation and resource field | carrying state, local richness, lateral richness gradient, forward richness gradient |
| 21–22 | Offspring | own-infant proximity and angle |
| 23–31 | Compatible group | presence, effective count, centre in body coordinates, relative velocity, long-range intensity and direction |
| 32 | Digestion | stomach fullness |
| 33–36 | Acoustic communication | strength, direction sine, direction cosine, tone |
| 37–42 | Chemical communication | trail and alarm concentrations at here, forward-left, and forward-right probes |
| 43 | Vitality | normalized remaining life |

_Implementation: [input names and writer](src/creature/vision.py) · [sensor construction](src/world.py)_

### 3.2 Endogenous normalization

Let normalized energy and stomach fullness be

$$
e=\operatorname{clip}_{[0,1]}(E/E_{\max}),\qquad
s=\operatorname{clip}_{[0,1]}(S/K).
$$

The feeding drive is not simple hunger. It is suppressed when the stomach is full:

$$
h=(1-e)(1-s).
$$

Thus a low-energy creature with no stomach space cannot obtain a strong feeding-drive input until digestion creates capacity. Visible creature and food counts are normalized as $\min(n_c/5,1)$ and $\min(n_f/10,1)$. The compatible flock count is normalized relative to the configured target group size, currently four.

_Implementation: [sensor input formulation](src/creature/vision.py) · [stomach fullness](src/creature/vision.py) · [flocking defaults](configs/sim_config.py)_

Reproductive readiness is computed from biological eligibility rather than from the raw neural wish to reproduce. Energy, minimum age, cooldown, available population capacity, and resource conditions remain authoritative external gates. Internal time is represented through an alternating signal, a resettable chronometer, and normalized age, allowing networks to evolve periodic or interval-dependent behaviour without recurrent connections.

_Implementation: [sensor snapshot assembly](src/world.py) · [reproductive eligibility](src/creature/neat/rt_neat.py) · [chronometer update](src/world.py)_

### 3.3 Visual geometry

The visual field is a circular sector parameterized by inherited range $R$ and angle $\Phi$. Candidates are compared by surface distance and relative bearing, so the physical sizes of observer and target influence visibility. Relative angles are normalized to the signed interval $[-1,1]$: zero denotes the forward axis, negative and positive values denote opposite sides of the field, and the magnitudes approach one near the field boundaries. Proximity increases from zero for absent or distant targets toward one as the target approaches.

_Implementation: [visibility candidates and angle normalization](src/creature/vision.py) · [vision traits](src/creature/genotype.py)_

Visible targets are sorted deterministically and nearer creatures can occlude angular intervals occupied by more distant food or creatures. Wall sensing uses ray–segment intersections against environmental boundaries. Mouth contact is handled separately from ordinary visibility so a food item physically touching the mouth can be processed even in close geometric edge cases.

_Implementation: [occlusion and blocked intervals](src/creature/vision.py) · [wall rays](src/creature/vision.py) · [mouth-contact candidate](src/creature/vision.py)_

Larger sensory fields are metabolically costly. With maximum configured range $R_{\max}$ and angle $\Phi_{\max}$, the per-second vision cost is

$$
C_{\mathrm{vision}}
=C_0+C_A\left(\frac{\Phi}{\Phi_{\max}}\right)
\left(\frac{R}{R_{\max}}\right)^2,
$$

where $C_0=0.002$ and $C_A=0.018$ by default. The quadratic range term approximates the growth of sensed sector area and creates an evolutionary trade-off between information and upkeep.

_Implementation: [vision energy model](src/creature/vision.py) · [vision configuration](configs/sim_config.py)_

### 3.4 Resource, family, social, and communication senses

Three body-relative resource probes produce local expected food richness and two gradients: left versus right, and ahead versus the current position. These inputs provide a continuous environmental cue even when no individual food item is visible. Family sensing is restricted to the creature's own immature offspring and supplies nearest-infant proximity and angle, supporting the evolution of nursing behaviour.

_Implementation: [biome sensor snapshot](src/creature/vision.py) · [biome probes](src/world.py) · [family index and infant sensing](src/world.py)_

Compatible neighbours are aggregated into effective count, group-centre displacement, and relative group velocity in the focal creature's body frame. A separate long-range observation can represent compatible social mass beyond the immediate group when enabled. Compatibility is itself inherited or evolutionary: the default mode compares two-dimensional social tags, while alternative modes can use species or legacy relations.

_Implementation: [flock sensor construction](src/creature/vision.py) · [social compatibility resolver](src/creature/flocking.py) · [compatibility configuration](configs/sim_config.py)_

Acoustic sensing reports the strongest audible non-self signal after distance attenuation, expressed as strength, body-relative direction sine/cosine, and tone. Chemical sensing samples two independent scalar fields—trail and alarm pheromones—at the body centre and two forward lateral probes. The neural network therefore observes local chemical gradients without receiving privileged emitter identities.

_Implementation: [acoustic and pheromone observations](src/creature/communication.py) · [pheromone sensor positions](src/world.py)_

---

## 4. Neural decision model and actions

### 4.1 Feed-forward NEAT controller

Each creature owns a NEAT genome and a feed-forward network instantiated from it. Founder genomes begin without hidden nodes and with a sparse random sample—currently $15\%$—of direct input-to-output connections. Evolution may add or delete nodes and connections, change connection weights, toggle connections, and occasionally change activation or aggregation functions. Feed-forward topology guarantees that a decision depends on the current sensor vector and explicit controller state, not on an unbounded recurrent activation history.

_Implementation: [brain representation](src/creature/neat/brain.py) · [brain controller](src/creature/neat/controller.py) · [NEAT genome parameters](configs/neat_herbivore.ini)_

The network computes

$$
\mathbf{z}=N_{G_N}(\mathbf{x}),
$$

where $\mathbf{x}\in\mathbb{R}^{43}$ is the ordered sensor vector and $G_N$ is the neural genome. Output centering is activation-aware. For example, a sigmoid result is transformed by $y=2\operatorname{clip}_{[0,1]}(z)-1$, while `tanh` and clamped outputs are clipped directly to $[-1,1]`; unsupported finite activation outputs pass through $\tanh$. Invalid or missing outputs become zero. This centering makes a neutral sigmoid value of $0.5$ correspond to zero action evidence.

_Implementation: [network evaluation and normalization](src/creature/neat/brain.py) · [initial activation choices](configs/neat_herbivore.ini)_

If a brain is missing or a decision cannot be obtained, the controller returns a neutral action with every intent disabled. Sensor and output counts are validated when the controller is built, preventing a network with an incompatible contract from silently operating.

_Implementation: [controller fallback](src/creature/neat/controller.py) · [neutral action](src/creature/action.py)_

### 4.2 Action contract

The action schema contains exactly 15 ordered outputs:

| Index | Action | Range | Interpretation |
|---:|---|---:|---|
| 1 | `accelerate` | $[-1,1]$ | forward/backward thrust |
| 2 | `rotate` | $[-1,1]$ | signed turn command |
| 3 | `want_reproduce` | $[0,1]$ | reproductive intent |
| 4 | `want_eat` | $[0,1]$ | ingestion intent |
| 5 | `reset_chronometer` | $[0,1]$ | reset internal interval timer |
| 6 | `want_grab` | $[0,1]$ | pick up nearby food |
| 7 | `want_release` | $[0,1]$ | release carried food |
| 8 | `want_nurse` | $[0,1]$ | transfer energy to own infant |
| 9 | `flee_panic_intensity` | $[0,1]$ | sprint/panic intensity |
| 10 | `herding` | $[0,1]$ | social engagement drive |
| 11 | `emit_sound` | $[0,1]$ | acoustic emission strength |
| 12 | `sound_tone` | $[-1,1]$ | signed acoustic tone |
| 13 | `emit_trail_pheromone` | $[0,1]$ | trail deposition intensity |
| 14 | `emit_alarm_pheromone` | $[0,1]$ | alarm deposition intensity |
| 15 | `rest` | $[0,1]$ | continuous resting intent |

Separation, alignment, and cohesion are not actions: they are inherited traits that modulate social steering. Positive discrete intents become active only when their value exceeds $0.1$. This threshold prevents arbitrarily small neural noise from triggering eating, reproduction, carrying, nursing, or chronometer resets.

_Implementation: [action ordering, ranges, and threshold](src/creature/action.py) · [NEAT output declaration](configs/neat_herbivore.ini)_

Herding has additional elapsed-time smoothing. If the reference update coefficient is $\alpha$ at $30\,\mathrm{Hz}$, its continuous-equivalent response rate is

$$
\lambda=-\frac{\ln(1-\alpha)}{1/30},\qquad
\alpha_{\Delta t}=1-e^{-\lambda\Delta t},
$$

and the state is updated by linear interpolation with $\alpha_{\Delta t}$. Motion commands and rest are also smoothed before physical execution, reducing discontinuities while preserving approximately the same response under different fixed cadences.

_Implementation: [herding filter](src/creature/neat/brain.py) · [action and rest smoothing](src/world.py)_

---

## 5. Locomotion and social dynamics

### 5.1 Propulsion and turning

For heading $\theta$ and signed acceleration $a$, the direct neural force is

$$
\mathbf{F}_N=
\begin{cases}
aF_{+}(\cos\theta,\sin\theta), & a\ge 0,\\
aF_{-}(\cos\theta,\sin\theta), & a<0,
\end{cases}
$$

with default forward and backward maxima $F_+=125$ and $F_-=70$. Panic $p$ increases force and speed limits through the multiplier $1+0.5p$. The command is smoothed at the physical rate before application.

_Implementation: [force-vector formulation](src/creature/action.py) · [action application](src/world.py) · [motion parameters](configs/sim_config.py)_

Turning is controlled as a target angular velocity rather than an accumulated torque. For normalized turn command $u$, the target is $\omega^*=u\omega_{\max}$ and the current velocity approaches it by

$$
\omega' = \omega + \rho(\omega^*-\omega),
$$

where the response coefficient $\rho$ is adjusted for elapsed physical time. A dead zone removes very small commands, a different damping response is used when the target is zero, and both linear and angular velocities are clamped to their current limits.

_Implementation: [turn control and motion limits](src/world.py) · [turn parameters](configs/sim_config.py)_

### 5.2 Planar drag and rest inhibition

Velocity is decomposed into forward and lateral body axes. At each step the components are multiplied by cadence-adjusted retentions, currently $0.992$ and $0.72$ per reference physical step. This strong lateral damping produces top-down locomotion with directional inertia. Rest adds exponential braking

$$
b_{\mathrm{rest}}=\exp(-k_b\,r_s\,\Delta t),
$$

where $r_s$ is smoothed rest and $k_b=2.5$. Voluntary force is additionally scaled by $1-r_s^\gamma$ with default exponent $\gamma=2$, and neural turning is reduced by $1-0.5r_s$.

_Implementation: [planar drag and rest braking](src/world.py) · [rest parameters](configs/sim_config.py)_

### 5.3 Collision avoidance and force priority

Collision avoidance is mandatory and receives first access to a finite force budget. Any lower-priority force component that opposes accepted avoidance is projected out. If $\mathbf{a}$ is the avoidance vector and $\mathbf{v}\cdot\mathbf{a}<0$, the opposing component is removed as

$$
\mathbf{v}'=\mathbf{v}
-\mathbf{a}\frac{\mathbf{v}\cdot\mathbf{a}}{\lVert\mathbf{a}\rVert^2}.
$$

Requested forces are then magnitude-limited to the remaining budget. This ordering guarantees that social or neural steering cannot spend force already required for immediate collision avoidance.

_Implementation: [collision avoidance](src/world.py) · [opposition removal and force allocation](src/creature/flocking.py)_

### 5.4 Flocking formulation

For herding drive $h$, panic $p$, compatible-social presence $s$, personal-space presence $q$, minimum engagement $e_0$, panic suppression $\kappa$, and inherited genes $(g_s,g_a,g_c)$, the effective flocking weights are

$$
e=s\left[e_0+(1-e_0)h\right],\qquad
\pi=1-\kappa p,
$$

$$
w_s=qg_s,\qquad w_a=eg_a\pi,\qquad w_c=eg_c\pi.
$$

The defaults are $e_0=0.25$ and $\kappa=0.5$. Separation remains responsive to immediate personal-space violation, whereas panic attenuates alignment and cohesion.

_Implementation: [flocking weights](src/creature/flocking.py) · [flocking parameters](configs/sim_config.py)_

Let $\mathbf{v}_s$, $\mathbf{v}_a$, and $\mathbf{v}_c$ be the desired separation, alignment, and cohesion velocities. The social target is

$$
\mathbf{v}_{S}=
\frac{w_s\mathbf{v}_s+w_a\mathbf{v}_a+w_c\mathbf{v}_c}
{w_s+w_a+w_c},
$$

bounded by maximum speed. Confidence combines the strongest active weight with compatible group size. Neural and social desired velocities are blended with influence $\beta$, where $0\le\beta\le0.35$ by default:

$$
\mathbf{v}_{B}=(1-\beta)\mathbf{v}_{N}+\beta\mathbf{v}_{S}.
$$

_Implementation: [social intention and blending](src/creature/flocking.py) · [world social-runtime integration](src/world.py)_

Compatibility can be derived from social-tag distance, species membership, or the same composite evolutionary distance used by speciation. In the composite case, compatibility decreases linearly from one at zero distance to zero at the current species threshold. Pairwise distances are cached by immutable neural genome identities.

_Implementation: [compatibility resolver](src/creature/flocking.py) · [composite live compatibility](src/creature/evolution.py)_

---

## 6. Communication

### 6.1 Acoustic channel

A creature may emit a signal with strength $s_e\in[0,1]$ and tone $\tau\in[-1,1]$. Emissions below $0.05$ are omitted. For source–receiver distance $d$ within acoustic range $R_a=480$, perceived strength is

$$
s_h=s_e\left(1-\frac{d}{R_a}\right)^2.
$$

Signals below the hearing threshold $0.05$ are ignored. Among audible candidates, the strongest attenuated signal is selected, with emitter identity used only as a deterministic tie-breaker. The receiver obtains no emitter identity; it receives strength, tone, and body-relative direction represented by sine and cosine.

_Implementation: [acoustic sensing](src/creature/communication.py) · [communication defaults](configs/sim_config.py) · [signal commitment](src/world.py)_

Acoustic energetic cost is quadratic in emitted strength:

$$
C_{\mathrm{sound}}=c_a s_e^2,
$$

with $c_a=0.006$ energy units per second. The quadratic term makes strong long-range signalling disproportionately costly.

_Implementation: [communication energy demand](src/creature/metabolism.py) · [communication configuration](configs/sim_config.py)_

### 6.2 Pheromone fields

Trail and alarm pheromones are separate floating-point grids over the physical domain. A deposit is distributed bilinearly among the four nearest grid cells and clipped to maximum concentration one. With deposit intensity $u$, configured rate $r_p=0.75$, and elapsed time $\Delta t$, the committed amount is proportional to $u r_p\Delta t$.

_Implementation: [pheromone deposition](src/creature/communication.py) · [batched communication intents](src/world.py) · [pheromone defaults](configs/sim_config.py)_

Each field follows a finite-difference approximation of diffusion with evaporation,

$$
\frac{\partial P}{\partial t}=D\nabla^2P-\lambda P,
$$

where the defaults are diffusion coefficient $D=390$ world-distance squared per simulated second and evaporation rate $\lambda=0.08$. Stable substeps are selected from grid geometry and $D$. The default boundary condition is reflective; wrap and absorbing modes are also defined. Field evolution is accumulated and normally processed every $0.25\,\mathrm{s}$, with a bounded catch-up count after large time increments.

_Implementation: [pheromone solver](src/creature/communication.py) · [boundary modes and numerical parameters](configs/sim_config.py)_

Pheromone sampling is bilinear and returns concentrations at three body-relative locations. Trail and alarm emission costs are linear in intensity:

$$
C_{\mathrm{pheromone}}=c_p(u_{\mathrm{trail}}+u_{\mathrm{alarm}}),
$$

where $c_p=0.002$ per second. These costs enter the same resource ledger as locomotion and neural upkeep.

_Implementation: [pheromone sensing](src/creature/communication.py) · [communication energy accounting](src/creature/metabolism.py)_

---

## 7. Feeding, metabolism, rest, and vitality

### 7.1 Ingestion and stomach state

Eating requires active intent and geometric overlap between a food item and the creature's forward mouth position. At a physical contact, the maximum bite is

$$
B=\min(K-S,\;b_{\max}\Delta t,\;E_f),
$$

where $K$ is inherited stomach capacity, $S$ is current stomach energy, $b_{\max}=0.5$ is the default bite rate, and $E_f$ is remaining food energy. A small remainder tolerance allows a nearly finished item to be consumed atomically. Bite claims are ordered by physical step and stable identities, and per-food bite capacity prevents unlimited simultaneous consumption.

_Implementation: [mouth geometry and eating](src/creature/metabolism.py) · [food energy removal](src/food.py) · [chronological exposure resolution](src/world.py)_

Larger original food radii map linearly to a difficulty multiplier in $[0.75,1.25]$. The stomach stores both swallowed energy and its difficulty-weighted load. This preserves the composition of mixed bites without retaining every swallowed item. A carried item is kept ahead of the creature and cannot be eaten as though it were an unrelated nearby object.

_Implementation: [food difficulty and stomach load](src/creature/metabolism.py) · [food carrying](src/world.py)_

### 7.2 Digestion

During a biological interval, the maximum processable stomach amount is

$$
S_{\max}=\min(S,q\Delta t),
$$

where $q$ is the inherited digestion rate. Effective conversion efficiency is increased by effective rest $r_e$:

$$
\eta_e=\operatorname{clip}_{[0,1]}
(\eta+b_r r_e),
$$

with inherited efficiency $\eta$ and default rest bonus $b_r=0.10$. Processing difficulty and rapid digestion impose the fraction

$$
f_p=\min\left(f_{\max},
f_0d\left[1+k_q\left(\frac{q}{q_{\max}}\right)^2\right]\right),
$$

where $d$ is mean stomach difficulty, $f_0=0.08$, $k_q=1.5$, and $f_{\max}=0.5$.

_Implementation: [digestion and processing fraction](src/creature/metabolism.py) · [metabolic parameters](configs/sim_config.py)_

For stomach amount $S_c$ actually consumed,

$$
E_{\mathrm{gross}}=S_c\eta_e,\qquad
C_{\mathrm{process}}=E_{\mathrm{gross}}f_p,\qquad
E_{\mathrm{net}}=E_{\mathrm{gross}}-C_{\mathrm{process}}.
$$

$S_c$ is reduced when neither current demand nor remaining energy capacity can use the resulting net energy. Digestion therefore does not destroy stomach contents merely because the usable energy reserve is already full.

_Implementation: [pure digestion calculation](src/creature/metabolism.py) · [digestion tests](tests/creature/test_metabolism_traits.py)_

### 7.3 Energy demand

The per-second demand is the sum

$$
C=C_{\mathrm{base}}+C_{\mathrm{brain}}+C_{\mathrm{move}}
+C_{\mathrm{sprint}}+C_{\mathrm{vision}}+C_{\mathrm{body}}
+C_{\mathrm{sound}}+C_{\mathrm{pheromone}}+C_{\mathrm{digestive}}.
$$

Default basal demand is $0.01$. Movement demand is $0.02(v/v_{\max})m$, where $m$ is the inherited movement-cost multiplier; panic adds up to $0.04$ per second. Body cost is $0.006(r/r_{\max})^2$. After five seconds of age, neural upkeep adds $0.0003$ per node and $0.0001$ per connection per second.

_Implementation: [energy cost breakdown](src/creature/metabolism.py) · [body and brain upkeep](src/creature/metabolism.py) · [metabolism and trait defaults](configs/sim_config.py)_

Digestive upkeep penalizes capacity, rate, and efficiency relative to their defaults:

$$
C_{\mathrm{digestive}}=
\min\left(C_{\max},C_d\left[
0.40\left(\frac{K}{K_0}\right)^2+
0.35\left(\frac{q}{q_0}\right)^2+
0.25\left(\frac{\eta}{\eta_0}\right)^2
\right]\right),
$$

with $C_d=0.004$ and $C_{\max}=0.012$. Hence no digestive trait is unconditionally advantageous.

_Implementation: [digestive upkeep](src/creature/metabolism.py) · [digestive defaults](configs/sim_config.py)_

### 7.4 Activity-gated rest

Resting effectiveness depends on realized activity, not merely on the neural rest output. Motor effort, normalized speed, turning, communication, reproduction, and nursing produce

$$
A=\min\left(1,
0.40a_m+0.10a_v+0.15a_\omega+0.15a_c
+0.10a_r+0.10a_n\right).
$$

Reproduction or nursing forces $A=1$. Effective rest is

$$
r_e=r_s(1-A),
$$

where $r_s$ is smoothed neural rest intent. A creature cannot obtain full digestive, recovery, or healing benefits while simultaneously performing costly behaviour.

_Implementation: [activity formulation](src/creature/metabolism.py) · [effective-rest commit](src/world.py)_

Rest can replenish usable energy only up to the starvation threshold, currently $0.3$, at a maximum rate $0.04r_e$ per second. It can also heal life at up to $0.01r_e$ life units per second, provided the creature can pay one unit of energy per life unit restored. Healing never revives a creature whose life has already reached zero.

_Implementation: [rest recovery and healing](src/creature/metabolism.py) · [rest defaults](configs/sim_config.py)_

### 7.5 Deficit damage, senescence, and death

Available energy first pays non-movement demand and then powered voluntary movement. Unmet ordinary demand produces life damage at $0.25$ life units per missing energy unit. Unmet movement demand is more dangerous at low life. Its multiplier is

$$
M_L=1+(M_{\max}-1)\left(1-\frac{L}{L_{\max}}\right)^2,
$$

with $M_{\max}=4$. The quadratic increase prevents indefinite high-speed motion powered only by life when energy is depleted.

_Implementation: [resource candidate and movement penalty](src/creature/metabolism.py) · [resource-ledger tests](tests/test_life_rest_transactions.py)_

After the default senescence age of $200$ seconds, all per-interval energy costs are multiplied by

$$
M_{\mathrm{age}}=1+0.05(t-200).
$$

Direct collision or other queued damage and deficit damage are committed in the same candidate transaction. A creature survives exactly when its candidate final life is positive; otherwise it is removed and its neural and trait state can be archived for evolutionary recovery.

_Implementation: [senescence factor and death processing](src/world.py) · [survival predicate](src/creature/metabolism.py) · [population defaults](configs/sim_config.py)_

---

## 8. Reproduction, development, and evolution

### 8.1 Eligibility and parent selection

A creature becomes eligible only if it is at least $20$ seconds old, has energy of at least $0.8$, and has completed the $12$-second reproduction cooldown. It must also express reproductive intent above $0.1$. Population capacity and environmental biomass gates are checked independently, so a strong neural output cannot bypass ecological or physiological constraints.

_Implementation: [eligibility](src/creature/neat/rt_neat.py) · [intent and resource gates](src/world.py) · [population defaults](configs/sim_config.py)_

Parent selection uses a two-stage tournament. From $k_1=3$ sampled eligible creatures, the $k_2=2$ largest lifetime gathered-energy values become finalists; the least complex neural network among them wins, with energy and stable identity as deterministic tie-breakers. If the pool is no larger than $k_1$, the same energy–parsimony ordering is applied to the entire pool. Neural complexity is the number of nodes plus enabled connections.

_Implementation: [tournament and parsimony](src/creature/neat/rt_neat.py) · [selection configuration](configs/sim_config.py)_

The parent's reproduction demand depends on network size:

$$
C_R=\min\left(0.75,
0.35+0.008N+0.002C\right),
$$

where $N$ and $C$ are the numbers of neural nodes and connections. The parent must survive the same-step combination of upkeep, reproduction, and any nursing transfer.

_Implementation: [dynamic reproduction cost](src/world.py) · [population parameters](configs/sim_config.py)_

### 8.2 Atomic birth transaction

Reproduction is staged rather than immediately mutating the live population. The system first evaluates baseline resource candidates, ranks reproduction requests, resolves nursing and reproduction costs, and rejects any action that would make its donor fail the survival constraint. Genotype mutation, neural mutation, species assignment, random-number state, and child construction are prepared in a shadow transaction. They become observable only if the complete batch succeeds.

_Implementation: [resource transaction resolution](src/world.py) · [evolution shadow transaction](src/creature/evolution.py) · [transaction tests](tests/test_world_reproduction.py)_

The child receives energy $0.15$, its parent's heading, a position offset safely from the parent and boundaries, generation $g_p+1$, and the parent identity. A newly detected species receives the assigned species identity before registration. The parent's cooldown and offspring count are updated only after successful materialization.

_Implementation: [offspring commit and placement](src/world.py) · [lineage planning](src/creature/evolution.py)_

### 8.3 Non-neural mutation

Vision mutates additively by Gaussian noise with standard deviations $8$ for range and $0.08$ radians for angle. Radius and movement-cost multiplier always receive Gaussian proposals using configured standard deviations. Each digestive trait independently mutates with probability $0.15$; otherwise it is inherited unchanged. Every result is clipped to its biological interval.

_Implementation: [vision and physical mutation](src/creature/genotype.py) · [mutation parameters](configs/sim_config.py)_

Each flocking gene has probability $0.005$ of replacement by a uniform unit value and probability $0.05$ of Gaussian perturbation with standard deviation $0.05$. Social-tag coordinates follow the same default probabilities when tag compatibility is enabled. Colour undergoes a small HSV mutation and is kept away from the configured food-colour neighbourhood; colour has no direct energetic or behavioural effect.

_Implementation: [flocking, tag, and colour mutation](src/creature/genotype.py) · [trait configuration](configs/sim_config.py)_

### 8.4 Neural mutation and real-time evolution

The child's neural genome is a mutated copy of its parent's genome. Current NEAT parameters include connection addition probability $0.5$, connection deletion $0.05$, node addition $0.1$, node deletion $0.05$, weight mutation $0.8$, and weight replacement $0.02$. Innovation and allocator state are retained across births and persistence boundaries so structurally homologous changes remain comparable.

_Implementation: [neural child creation](src/creature/neat/controller.py) · [NEAT mutation parameters](configs/neat_herbivore.ini) · [evolution coordinator](src/creature/evolution.py)_

### 8.5 Infancy, nursing, and extinction recovery

A creature is an infant until age $12$ seconds. Infant movement cost is multiplied by three as a runtime penalty without modifying the inherited movement gene. A parent can nurse only its own nearby infant; the default transfer rate is $0.05$ energy units per second, and accepted transfers enter the donor's same-step resource transaction. Crossing the maturity boundary is recorded in the parent's fitness history.

_Implementation: [infancy and nursing](src/world.py) · [family lineage](src/creature/genotype.py) · [population configuration](configs/sim_config.py)_

If no creatures remain, the model draws up to five parent genomes from the retained elite archive and creates a configured recovery population, currently up to 35 creatures. Archived non-neural traits are mutated through the same path used by ordinary births, and recovered neural genomes are mutated and re-evaluated for speciation. Extinction recovery therefore preserves evolutionary memory without cloning an unchanged population.

_Implementation: [extinction recovery](src/world.py) · [aligned archive pruning](src/creature/evolution.py) · [elite neural archive](src/creature/neat/controller.py)_

---

## 9. Fitness and speciation

### 9.1 Implicit fitness

The selection score is

$$
F=\max(0,E_{\mathrm{gathered}})+0.001\max(0,t_{\mathrm{age}}).
$$

Lifetime gathered energy is increased by net digestive energy, so selection rewards energy successfully extracted from the world rather than merely food contacts, current energy, distance, or action intensity. Age is only a small deterministic tie-breaker. Parent tournament selection uses gathered energy directly before neural parsimony.

_Implementation: [fitness score and lifetime measures](src/creature/fitness.py) · [digestion ledger commit](src/creature/metabolism.py) · [parent selection](src/creature/neat/rt_neat.py)_

Additional measurements—age, food items depleted, distance travelled, average speed, offspring count, mature offspring, flocking quality, births, deaths, and neural size—support analysis but do not replace the implicit gathered-energy criterion.

_Implementation: [creature fitness records](src/creature/fitness.py) · [real-time population statistics](src/creature/neat/rt_neat.py)_

An optional flocking benchmark measures group presence, heading alignment, spacing, and movement:

$$
Q=Q_gQ_aQ_sQ_m,
$$

$$
Q_s=\exp\left[-\left(\frac{d-d^*}{\sigma_d}\right)^2\right].
$$

The current defaults use target group size four, target spacing $60$, tolerance $30$, and reference speed $50$. Its bounded accumulated reward is diagnostic and does not enter the main selection score.

_Implementation: [flocking benchmark](src/creature/fitness.py) · [benchmark configuration](configs/sim_config.py)_

### 9.2 Phenotypic distance

For any bounded trait $x\in[x_{\min},x_{\max}]$, the normalized difference between child $c$ and representative $r$ is

$$
\delta_x=
\frac{|\operatorname{clip}(x_c)-\operatorname{clip}(x_r)|}
{x_{\max}-x_{\min}}.
$$

The phenotypic distance is

$$
D_P=\delta_r+\delta_R+\delta_\Phi+\delta_m
+\frac{\delta_K+\delta_q+\delta_\eta}{3}.
$$

Digestive traits are averaged into one component so that adding three closely related digestive dimensions does not triple their total weight.

_Implementation: [phenotypic components and distance](src/creature/speciation.py) · [trait bounds](configs/sim_config.py)_

The inherited flocking-trait distance is

$$
D_F=\frac{|g_{s,c}-g_{s,r}|+|g_{a,c}-g_{a,r}|+|g_{c,c}-g_{c,r}|}{3}.
$$

Social tags affect live compatibility in tag mode but are not part of this speciation term.

_Implementation: [flocking-trait distance](src/creature/speciation.py) · [live social-tag compatibility](src/creature/flocking.py)_

### 9.3 Composite species criterion

Let $D_N$ be neat-python's genomic compatibility distance. The complete distance is

$$
D=D_N+w_PD_P+w_FD_F,
$$

with current defaults $w_P=2$ and $w_F=1$. A child remains in its parent's species if $D\le T$; if $D>T$, it becomes the representative and founder of a new species. The default initial threshold is $T=3.5$.

_Implementation: [composite compatibility and species evaluation](src/creature/speciation.py) · [speciation defaults](configs/sim_config.py)_

Every five seconds, the threshold is adjusted toward a target of five active species. It decreases by $0.05$ per elapsed adjustment interval when too few species exist and increases by the same amount when too many exist, remaining in $[2,7]$. This feedback stabilizes taxonomic resolution as the population evolves.

_Implementation: [adaptive threshold](src/world.py) · [threshold configuration](configs/sim_config.py)_

When a new species is formed, the system records founder traits, raw trait changes, every normalized distance component, the compatibility threshold, and enabled neural connection additions, removals, or weight changes. These records permit later reconstruction of why the offspring crossed the species boundary.

_Implementation: [speciation result and neural shifts](src/creature/speciation.py) · [species telemetry](src/telemetry.py)_

---

## 10. Behavioural analysis and explainability

### 10.1 Behaviour from realized evidence

The behavioural analyser deliberately does not infer behaviour from named NEAT outputs. It receives primitive observations of realized state and trajectory, then applies temporal rules to six categories:

- orientation toward food;
- approach toward food;
- feeding;
- resting;
- group cohesion;
- retreat from alarm pheromone.

This distinction is methodologically important. A positive `want_eat` output is an intention; feeding is recognized only from an actual consumption event. Likewise, turning output alone is not evidence of food orientation unless the creature's realized direction changes consistently toward a visible food target.

_Implementation: [behaviour observation contract and analyser](src/behavior_observer.py) · [world observation sampling](src/world.py)_

The default analyser samples at $10\,\mathrm{Hz}$ over a $2.5$-second sliding window. Candidate evidence must persist for $0.5$ seconds before an ordinary behaviour becomes active; a $0.3$-second grace period prevents momentary evidence loss from splitting one bout. Feeding is event-driven and remains associated with its event for $0.75$ seconds.

_Implementation: [temporal bout state machine](src/behavior_observer.py) · [observer parameters](configs/sim_config.py)_

### 10.2 Evidence and completed bouts

Every active rule emits named evidence with a value, unit, pass/fail result, and explanatory label. A completed bout stores start and end times, duration, evidence summaries, termination reason, and an outcome appropriate to the behaviour, such as food consumed, target lost, approach started, alarm exposure reduced, or interruption.

_Implementation: [behaviour evidence and outcomes](src/behavior_observer.py) · [completed-bout records](src/behavior_history.py)_

Evidence summaries retain sample count, pass count, first and last values, median, quartiles, and total. Sampling memory is bounded. Once capacity is exceeded, deterministic compaction increases the retained stride; consequently long bouts retain exact counts and totals while quantiles are explicitly marked as estimated.

_Implementation: [bounded metric and evidence accumulators](src/behavior_history.py) · [history capacity configuration](configs/sim_config.py)_

Completed bouts are aggregated into per-creature lifetime summaries and species-normalized reports. Species reports include observed creature count, observation time, total bouts, median duration, and bouts per creature-hour, preventing raw bout count from being mistaken for a rate when observation effort differs.

_Implementation: [creature and species history reports](src/behavior_history.py) · [history integration](src/world.py)_

### 10.3 Counterfactual NEAT explanations

Counterfactual analysis evaluates an immutable copy of the focal network. Starting from the factual sensor vector $\mathbf{x}$, one semantic group of inputs is replaced by neutral values to obtain $\mathbf{x}^{(-I)}$. The same network is evaluated without mutating live state:

$$
\mathbf{y}=N(\mathbf{x}),\qquad
\mathbf{y}^{(-I)}=N(\mathbf{x}^{(-I)}),\qquad
\Delta\mathbf{y}=\mathbf{y}-\mathbf{y}^{(-I)}.
$$

Available interventions remove visible-food cues, resource gradients, hunger through a satiated state, social cues, offspring cues, acoustic cues, trail pheromone, alarm pheromone, or temporal cues. Because all other inputs and the genome remain fixed, the difference estimates local neural dependence on that semantic information group; it is not a claim of causal effect on the complete future world trajectory.

_Implementation: [semantic interventions and pure evaluator](src/counterfactual_neat.py) · [pure brain evaluation](src/creature/neat/brain.py)_

For an output with admissible span $s_y$, influence is normalized as

$$
I_y=\operatorname{clip}_{[0,1]}
\left(\frac{|y-y^{(-I)}|}{s_y}\right).
$$

Behaviour-level influence aggregates only outputs relevant to that behaviour. Direction is interpreted relative to the factual target: an output can be supportive, suppressive, reversing, mixed, or minimal. Influence labels use thresholds $0.10$, $0.30$, and $0.60$ for minimal, weak, moderate, and strong effects.

_Implementation: [output and semantic effects](src/counterfactual_neat.py) · [bout-level counterfactual aggregation](src/counterfactual_neat.py)_

Counterfactual samples belonging to a completed bout are summarized by medians, quartiles, direction counts, and per-output factual/counterfactual values. This joins mechanistic neural sensitivity with independently observed behaviour while keeping the two evidence sources conceptually separate.

_Implementation: [counterfactual bout aggregator](src/counterfactual_neat.py) · [completed explanation records](src/behavior_history.py)_

### 10.4 Species-level scientific profiles

Species analysis compares founder morphology with the parent species, reporting percentage change as

$$
100\frac{x_c-x_p}{|x_p|}.
$$

It also estimates idle and active metabolic costs from founder traits, separates direct sensor-to-action neural changes from hidden-node integration changes, and queries descendant count and average lifespan. These profiles interpret recorded model state; they do not modify creatures or selection.

_Implementation: [morphology, metabolism, and neuroethology profiles](src/analysis.py) · [species records](src/creature/speciation.py)_

Creature and species telemetry stores births, deaths, parent-selection attempts, resource and population metrics, flocking measurements, founder traits, compatibility components, and neural shifts in a relational database. Persistent flock groups are matched over time by membership overlap, enabling group duration and stability measurements without adding a “group identity” sensor to creatures.

_Implementation: [telemetry database](src/telemetry.py) · [persistent flock tracking](src/flocking_telemetry.py)_

---

## 11. Concluding interpretation

The model couples evolution across several levels. Inherited radius, vision, digestion, movement efficiency, and social genes determine both opportunities and energetic liabilities. The NEAT genome transforms a high-dimensional but local sensory contract into continuous intentions. Physical integration, collision avoidance, resource availability, and transaction rules determine which intentions become realized behaviour. Gathered energy then influences parent selection, while network parsimony discourages unnecessary controller complexity. Speciation combines neural and phenotypic change, so neither morphology nor neural topology alone defines evolutionary divergence.

_Implementation synthesis: [genotype](src/creature/genotype.py) · [brain](src/creature/neat/brain.py) · [metabolism](src/creature/metabolism.py) · [evolution and speciation](src/creature/evolution.py)_

Several limitations follow from the formulation. The world is planar; bodies are circular and have equal mass; reproduction is asexual; sensory channels are engineered summaries rather than raw physical receptor arrays; the neural controller is feed-forward; compatibility and species thresholds are computational constructs; and the fitness proxy privileges lifetime gathered energy. Behavioural rules and counterfactual probes improve interpretability but remain operational definitions, not proof of subjective intention. These simplifications are deliberate: they create a tractable experimental system in which morphology, physiology, neural structure, social interaction, and evolutionary history can be measured under one deterministic simulation contract.

_Sources and validation: [creature architecture tests](tests/creature/test_architecture.py) · [genotype determinism tests](tests/creature/test_genotype_determinism.py) · [scheduler validation](tests/test_scheduler_validation.py) · [behaviour observer tests](tests/test_behavior_observer.py)_
