# Project "MMA-Machine" — Mathematical MMA Evolution Engine
### Architecture Blueprint v3.2

> **Changelog from v3.0:** Dropped MJX entirely (CPU MuJoCo + AsyncVectorEnv is the architecture). Actions are now PD joint-angle targets, not raw torques. PPO terminology corrected (rollout buffer, not replay buffer); anti-exploit redesigned as rollout masking. Opponent lives inside `env.step()`. Observations made egocentric; dimension math corrected (~382). Policy ELO and morphology ELO separated. Virtual-grip constraints specified for clinch. Impact reward made impulse-based; terminal reward rescaled. Morphology model precompilation cache added. Episode-length curriculum + judge decision added. MoCap demoted to hand-authored keyframes for Phase I. Step 0 vertical slice added. Grappling honestly framed as a research extension beyond published SOTA.

> **Changelog from v3.1:** Interface frozen at full final size from day one (30-dim action head incl. masked grips; ~390-dim observation incl. 120-dim action history). Reward schedule renamed WARMUP/DEVELOPMENT/FULL to kill the curriculum-phase name collision; all λ changes are ramps, never steps. Octagon introduction pinned to Phase III. Grip welds velocity-gated against constraint-snap impulses. PD-gain tuning and a ROCm smoke test added to Step 0 exit criteria. Added: phase-conditional early termination, truncation-vs-termination bootstrapping, judge anti-point-fighting term, knockdown/takedown disambiguation, snapshot morphology-compatibility tags, ELO anchoring.

---

## ⚙️ Finalized Tech Stack

| Component | Decision | Rationale |
|---|---|---|
| **Physics Engine** | MuJoCo 3.x (CPU), Python `mujoco` package | Free, Python-native, best-in-class contact/articulation physics. CPU MuJoCo is the deliberate choice: JAX/ROCm support for the RX 9060 XT is unreliable, MJX's worst case is exactly our workload (contact-rich two-humanoid scenes), and MuJoCo Warp is CUDA-only. The 9950X's 32 cores are the throughput engine. |
| **Parallelization** | Gymnasium `AsyncVectorEnv`, 32 workers | One env process per core. Each worker runs a full fight episode including frozen-opponent inference. `torch.set_num_threads(1)` in every worker to prevent thread thrashing. |
| **RL Algorithm** | Custom PPO (PyTorch, ROCm) | On-policy, clipped objective, stable for continuous control. Full control over reward shaping, asymmetric critic, and self-play logic. **PPO uses a rollout buffer (collected, trained on, discarded) — there is no replay buffer.** |
| **Low-Level Control** | PD position controllers | Policy outputs target joint angles; per-joint PD controllers convert to torques at the physics rate. The DeepMimic-standard approach: faster learning, smoother motion, biomechanical torque limits enforced via actuator `forcerange`. |
| **Self-Play System** | Custom League Play (opponent-in-env) | The learning agent is "the agent"; the opponent is a frozen league snapshot running inside `env.step()`. Described in Step 8. |
| **Fighter Population** | Shared Morphology Policy | A single network conditioned on a per-fighter morphology vector. One brain, 1,000 builds. Staged rollout (Step 10). |
| **Experiment Tracking** | Weights & Biases (offline mode) | Reward curves, policy entropy, KO rates, ELO progression. |
| **Phase I Reference Motion** | Hand-authored keyframes + CMU locomotion | Keyframed stance/jab/cross/teep/level-change references. CMU MoCap for locomotion only. MediaPipe-on-UFC-footage extraction is a **research extension**, not on the critical path (monocular pose under constant two-body occlusion + retargeting is a subproject). |
| **Language** | Python 3.12+ | — |

---

## 🖥️ Local vs. Web: Decision

**Training: 100% local.** AsyncVectorEnv saturates the 9950X's 32 cores with parallel physics; the RX 9060 XT (PyTorch ROCm) handles PPO gradient updates — the small policy network makes this trivial. The GPU is never the bottleneck; CPU physics throughput is, and that's what the hardware is built for. 60GB of RAM holds the policy archive, precompiled morphology models, and rollout buffers with no pressure.

**Portfolio visualization: hybrid.** MuJoCo's native renderer is the live training monitor. For the portfolio, episodes are serialized to JSON and played back in a Three.js viewer hosted on GitHub Pages — zero server cost, zero physics in the browser, a shareable URL for any fight at any training epoch.

---

## 📋 Full Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│            MuJoCo 3 Physics Engine (CPU, 480Hz substeps)          │
│                   ╔═══ THE OCTAGON ═══╗                           │
│  ┌──────────────┐ ║  8-sided fence    ║ ┌──────────────────────┐  │
│  │ Learning     │ ║  high-friction    ║ │ League Opponent      │  │
│  │ Fighter      │◄╬── wall contact ──╬►│ (frozen snapshot,    │  │
│  │ [28 joints]  │ ║  cage-pin geoms   ║ │  runs inside env)    │  │
│  │ Morph: mᵢ    │ ╚═══════════════════╝ │ Morph: mⱼ            │  │
│  │ Stamina ·    │                       │ + its own archived   │  │
│  │ Damage state │                       │   obs-norm stats     │  │
│  └──────┬───────┘                       └──────────────────────┘  │
└─────────║────────────────────────────────────────────────────────┘
          ║ ~390-dim egocentric observation
          ▼
┌──────────────────────────────────────────────────────────────────┐
│              Shared Morphology-Conditioned Policy                 │
│   Morphology mᵢ ─► [Embedding] ─┐                                 │
│   Observation ──► [LayerNorm] ──┴─► trunk MLP                     │
│   Actor head:  30-dim — 28 PD targets + 2 grips (masked ≤ III)    │
│   Critic head: V(s) — receives privileged state (training only)   │
└───────────┬──────────────────────────────────────┬───────────────┘
            ▼                                      ▼
     League Play Loop                      W&B Logger · Episode
  (Policy-ELO matchmaking ·                Serializer · Morphology
   snapshot archive on disk ·              Registry · Eval Harness
   workers load opponents lazily)
```

---

## 🛠️ Step-by-Step Execution Plan

---

### Step 0: Vertical Slice *(build this first — everything else bolts on)*

Deliberately boring, end-to-end proof of the pipeline:

- **One** fixed morphology. Flat circular arena (no Octagon yet).
- Striking only, vs. a stationary scripted dummy.
- PD-target actions, plain PPO, AsyncVectorEnv, native MuJoCo viewer.
- Reward: stability + simple impact.

**Exit criteria:**
1. **ROCm smoke test passed.** A real PyTorch training loop runs on the RX 9060 XT before anything else is built. (Fallback if ROCm misbehaves: the policy MLP is small enough to train on CPU — ugly but viable.)
2. **PD gains tuned to stable standing.** Per-joint kp/kd tuning is real work, not a given — too stiff is jerky, too soft lags targets. A short sweep on the balance task happens here, before any combat training.
3. Agent stands for 10s, then learns to strike the dummy.
4. **Throughput benchmark recorded.** The v3.0 estimate (1.1M frames/hr) assumed real-time simulation and is likely pessimistic by ~10×; realistic CPU MuJoCo throughput for this scene is plausibly 10–60M frames/hr across 32 workers. The measured number recalibrates the entire training timeline.
5. Worker→learner data flow, checkpointing, and W&B logging all proven.

---

### Step 1: Environment Setup & Rigging

**Humanoid Assembly:** MuJoCo `humanoid.xml` as the base skeleton, extended to the 28-DOF joint map below with MMA proportions. Two rigs per scene; morphology parameters applied at model-build time (see Step 10 for the precompilation cache).

**Physics Rigging:**
- `hinge`/`ball` joints with anatomical range limits enforced as **soft limits**: a stiff restoring torque beyond the anatomical range rather than a hard stop. (Hard stops make submissions undefinable — see Step 6.)
- Human mass distribution (torso ≈ 50%, head ≈ 8%, thighs ≈ 10% each, etc.).
- **Position actuators with per-joint PD gains; `forcerange` = ±T_max** enforces biomechanical torque ceilings.
- Dedicated collision geoms for strike targets: head, jaw, temple, liver, solar plexus, thighs/calves.
- Grip geoms on hands/wrists for clinch contact detection (constraint activation, Step 6).
- Self-collision enabled (limbs cannot pass through the torso).

**Timing:** physics at **480Hz** (`dt = 1/480`), policy at **30Hz** → integer frame-skip of 16. (500Hz/30Hz is non-integer; don't ship a fractional frame-skip.)

**The Octagon Arena (`arena_octagon.xml`):** 8 rigid wall panels, 9.1m diameter, 45° between adjacent panels.

```xml
<geom name="wall_N" type="box" size="1.8 0.05 1.2"
      pos="0 4.55 1.2" friction="1.2 0.005 0.0001"
      contype="1" conaffinity="1"/>
<!-- repeated for NE, E, SE, S, SW, W, NW -->
```

- **Wall friction 1.2** — cage grinding, press mechanics, wall-walking.
- **Cage-pin detection** — `CAGE_CONTACT` flag when a fighter's back contacts a panel.
- **Wall-off thrust vectors** — wall contact normals exposed in observations.
- **Canvas friction 0.8** — footwork traction, distinct from walls.

**Joint Map (28 controllable joints per fighter):**
```
Head/Neck:  neck_pan, neck_tilt                          (2)
Shoulders:  L/R shoulder_x/y/z                           (6)
Elbows:     L/R elbow_flexion                            (2)
Wrists:     L/R wrist_x/y                                (4)
Spine:      spine_lower, spine_mid, spine_upper          (3)
Hips:       L/R hip_x/y/z                                (6)
Knees:      L/R knee_flexion                             (2)
Ankles:     L/R ankle_x/y                                (4)
                                                 Total: 28 DOF
```

**Spawn randomization:** initial positions, orientations, and separation distance randomized per episode, or the policy overfits to one starting geometry.

---

### Step 2: Designing the Senses (Observation Space)

**All spatial quantities are egocentric** — expressed in the fighter's heading-local frame (yaw removed). No global x/y coordinates anywhere: global frames waste the Octagon's 8-fold symmetry and generalize poorly. ~**382 dimensions**, six modules:

#### Module A — Morphology Specs (12)
| Signal | Dims | Purpose |
|---|---|---|
| Limb segment lengths (arms, legs, torso) | 5 | Reach and leverage awareness |
| Segment mass ratios (normalized) | 4 | Inertia/momentum |
| Per-joint T_max scale (key joints) | 3 | Strength ceiling awareness |

#### Module B — Self-Proprioception (111)
| Signal | Dims | Purpose |
|---|---|---|
| Joint angles | 28 | Body configuration |
| Joint angular velocities | 28 | Movement rate |
| Actuator torque load | 28 | Muscle stress |
| Root height (z only) | 1 | Posture vs. ground |
| Root tilt (projected gravity vector) | 3 | Orientation, heading-invariant |
| Root linear velocity (local frame) | 3 | Translational momentum |
| Root angular velocity (local frame) | 3 | Rotational state *(missing in v3.0)* |
| CoM offset from support midpoint | 3 | Balance anchor |
| CoM velocity (local frame) | 3 | Balance trajectory |
| Foot contact sensors | 4 | Heel/toe contact |
| Stamina (normalized) | 1 | Fatigue state |
| Damage state (6 zones) | 6 | Structural damage |

#### Module C — Exteroception / Opponent (125)
| Signal | Dims | Purpose |
|---|---|---|
| Opponent body positions relative to self (28 bodies × 3, self frame) | 84 | Spatial threat map |
| Opponent joint angular velocities | **28** | Incoming-strike prediction *(v3.0 said 84 — a hinge velocity is scalar)* |
| Opponent heading relative to self | 3 | Facing/angle awareness |
| Opponent CoM relative to self | 3 | Takedown range |
| Opponent CoM velocity (relative) | 3 | Closing speed |
| Opponent foot contacts | 4 | Off-balance detection |

#### Module D — Octagon Awareness (16)
| Signal | Dims |
|---|---|
| Distance to each of 8 wall panels | 8 |
| Cage-center direction (local frame) | 3 |
| Cage contact flag (self / opponent) | 2 |
| Wall normal at nearest panel (local frame) | 3 |

#### Module E — Action History (120)
Last 4 frames × 30 actions (28 PD targets + 2 grips; grip slots zero-filled through Phase III). Enables combo chaining; prevents jerky one-off actions.

#### Module F — Combat Phase State (6)
| Signal | Dims |
|---|---|
| Phase one-hot: Standing / Clinch / Transition / Ground | 4 |
| Round time remaining (normalized) | 1 |
| Opponent stamina (**true value** — justified as visible fatigue; the v3.0 "learned estimate" had no ground-truth estimator and was circular) | 1 |

**Total: ~390** (finalized in code).

**Interface freeze (hard rule):** the observation layout and the 30-dim action head are allocated at their full, final size from day one. Features belonging to later phases — grip slots, grappling/Octagon state — are zero-filled until their phase activates. Reason: league snapshots from epoch 40M must still load and run at epoch 400M; if the interface ever changes shape, every earlier snapshot becomes unusable and league play loses its history. Retrofitting tensor shapes mid-training is not survivable.

**Observation normalization:** running mean/std (VecNormalize-style). **The normalizer statistics are part of every policy snapshot** — an archived opponent must normalize with its own training-era stats or it plays silently brain-damaged and corrupts league ELO. This is the classic self-play bug; it is a hard rule here.

**Asymmetric actor-critic:** the critic is training-only, so it additionally receives privileged state (opponent true stamina/damage, exact contact forces). Better value estimates for free; the actor never sees it.

---

### Step 3: Designing the Actions (Output Space)

The network outputs a **30-dim action vector**: 28 target joint angles (Tanh-squashed to anatomical ranges) + 2 grip activations (sigmoid, for the Step 6 virtual-grip system). **Grip outputs are masked to zero through Phase III** — the head is built at final size from day one (see interface freeze, Step 2), because retrofitting an output layer mid-training invalidates every league snapshot. Per-joint PD controllers convert angle targets to torques at 480Hz, clamped by `forcerange` (±T_max).

```
Output: [30,] float32 — 28 × target_angle[i] ∈ anatomical range[i] · 2 × grip ∈ [0,1]
Torque:  τ[i] = clamp( kp·(target − θ) − kd·θ̇ ,  ±T_max[i] )
```

Why PD targets, not raw torques: this is the DeepMimic-standard interface. It learns dramatically faster for high-DOF humanoids, produces smooth motion (the v3.0 jerk limiter becomes nearly redundant), and stamina still works — exertion is metered on the *resulting PD torques*, not the action.
**Network Architecture:**
```
Morphology (12) ─► Linear(12→64) + ELU ──────────────┐
Observation (~378) ─► LayerNorm ─► Linear(→512)+ELU ─┴► concat (576)
        ─► Linear(576→256)+ELU ─► Linear(256→256)+ELU
              ├─► Actor head: Linear(256→30) [28 Tanh→joints · 2 sigmoid grips]
              └─► Critic head: Linear(256→1)   (+ privileged inputs)
```

---

### Step 4: Constructing the Reward Function

#### Reward scale discipline *(new, load-bearing)*

Dense shaping components are normalized so a typical episode's total shaped return is **O(1)**. The terminal finish reward is **±10**, so winning always dominates point-farming. Without this, a ±1.0 terminal signal drowns under dense returns of ±50 and agents optimize for points, not victory.

#### Core Reward Matrix

| ID | Component | Criteria | Weight λ | Purpose |
|---|---|---|---|---|
| **Rᵢ** | Imitation | Pose similarity vs. keyframe references (joint-angle MSE, velocity cosine) | High early, annealed to 0 | Bootstraps stance, footwork, strike mechanics |
| **Rᶜ** | Combat Impact | **Impulse-based**: reward only at contact *onset*, gated by relative velocity at impact. Anatomical multipliers (head ×2.0, liver ×1.5, body ×1.0). *Sustained force earns nothing — v3.0's F×V was farmable by leaning/pushing.* | Medium | Offensive striking |
| **Rᵈ** | Defensive Mitigation | Penalty ∝ impact impulse received; bonus for measurable evasion | Medium | Slipping, blocking, guard |
| **Rₛ** | Stability | Penalize uncontrolled-limb KE, CoM outside support polygon, excessive jerk. Reward stance maintenance | Medium | Anti-flopping |
| **Rₑ** | Stamina Economy | Penalty ∝ Σ\|τ_applied\| (PD output torques) | Low-Med | Gas-tank management |
| **Rᵣ** | Range Control | Reward optimal striking-range band per weapon; penalize strikes from dead range | Medium | Distance management |
| **Rₖ** | Finish | **Terminal ±10** for KO/TKO/tap (vs. being finished) | Dominant | Win condition |
| **Rᵍ** | Grappling Position | Δ(position score) per Step 6 hierarchy | High (ground) | Ground strategy |
| **Rₒ** | Combo Bonus | Multiplier on Rᶜ when 2+ distinct strikes land within 0.8s | Low | Combinations |
| **Rₗ** | Leg Kick Utility | Strike multiplier for thigh/calf impacts, scaled by stamina damage caused | Low-Med | Kick diversity |

*Removed from v3.0: scripted "guard dropping frequency increases" at low stamina — the policy controls the joints; behaviors cannot be scripted into it. Stamina reduces T_max only, and guard drops emerge because tired arms can't stay up.*

#### Reward Weighting Schedule

Named distinctly from the curriculum phases — "Phase III" must never be ambiguous between two numbering systems:

```
WARMUP      (0–30M):    λᵢ=1.0  λᶜ=0.3  λₛ=0.5  grappling=0
DEVELOPMENT (30–100M):  λᵢ=0.2  λᶜ=1.0  λᵈ=0.8  λᵣ=0.5
FULL        (100M+):    λᵢ=0.0  all active  λᵍ=1.0
```

All λ transitions are **continuous ramps (~5M steps wide), never step changes**. A step change instantly shifts the critic's value targets, spikes advantage estimates, and can destabilize PPO at every boundary.

---

### Step 5: The Training Pipeline

#### Curriculum Phases

| Phase | Name | Arena | Description | Episode length | Gate to next phase |
|---|---|---|---|---|---|
| **0** | Balance Bootcamp | Flat | Single agent, stability reward only. | 15s | Stands 10s under small random pushes |
| **I** | Imitation | Flat | Keyframe imitation: stance, jab, cross, teep, level change. CMU for locomotion only. | 15–30s | Pose-tracking error below threshold on all references |
| **II** | Dummy Striking | Flat | Stationary ragdoll; Rᶜ introduced. | 30s | Clean impulses landed consistently (eval harness) |
| **III** | Self-Play Striking | **Octagon** | Standing only; League Play activates. Knockdown = TKO terminal. **Portfolio-complete milestone.** | 60–90s | Technical-Era behaviors on eval battery; sustained ELO growth |
| **IV** | Grappling Integration | Octagon | Clinch/ground systems live; grip actions unmasked. **Research extension — see Step 6 honesty note.** | 90–180s | Takedowns + positional control vs. eval battery |
| **V** | Full MMA League | Octagon | Everything active; morphology rollout per Step 10. | 1 × 5-min round | — |

**Octagon timing:** Phases 0–II run on the flat arena — agents must learn to fight before they learn cage geometry; walls during the exploitative era would just train fence-spam, not spatial tactics. The Octagon (and Module D observations, zero-filled until then) activates at Phase III.

**Phase-conditional early termination:** in Phases 0–II a fall (torso below height threshold for ~1s) ends the episode immediately — DeepMimic-style early termination is one of the strongest known accelerators of imitation learning, because rollouts stop filling with useless on-the-floor frames. In Phase III a knockdown is a TKO terminal. From Phase IV the ground is a valid fighting state and fall-termination is disabled.

**Episode-length curriculum** *(new)*: stamina decay and damage accumulation only matter over minutes, but 9,000-step episodes are a brutal credit-assignment horizon early on. Lengths scale with the curriculum as above.

**Judge decision** *(new)*: when time expires with no finish, the winner is decided by damage-differential score (with a draw band). Every fight produces a result of 1 / 0.5 / 0 for ELO. Two subtleties: **(a) decision gaming** — a damage-only judge teaches "land one clean strike, then run for three minutes," which is authentically MMA but degenerate; a small activity/ring-control term in the judge score prices it out. **(b) Truncation ≠ termination** — a judge-decided timeout delivers terminal reward and is a true termination for PPO, but any episode cut purely for length (early-phase time caps) is a *truncation* and must bootstrap V(s_final) rather than treat it as zero. Conflating the two is a classic silent PPO bug.

#### Parallelization Architecture

```
Main Process
 ├── AsyncVectorEnv (32 workers, torch threads=1 each)
 │     └── each worker: MuJoCo env + frozen opponent policy
 │         (snapshot weights + obs-norm stats loaded from disk,
 │          assignment delivered via reset options, cached locally)
 ├── PPO Rollout Buffer  ← on-policy; collected, trained on, discarded
 ├── PPO Update (RX 9060 XT, PyTorch ROCm)
 ├── Anti-Exploit Masking ← flagged episodes masked out of the
 │                          current rollout before the update
 │                          (v3.0's "quarantine from replay buffer"
 │                           was incompatible with PPO's data flow)
 └── W&B Logger · Episode Serializer · Eval Harness
```

**Eval harness** *(new)*: a fixed battery of scripted/frozen reference opponents with fixed seeds, run at every checkpoint. Training ELO is relative and drifts; this is the absolute yardstick that catches regressions.

**Throughput:** measured in Step 0, not assumed. Working range: 10–60M frames/hour.

---

### Step 6: Grappling & Ground Game Sub-System

> **Honesty note:** emergent takedowns and wrestling-style pressure have been demonstrated in the literature (Bansal et al. 2017, sumo humanoids). **Emergent positional ground game — guard, mount, transitions, submissions — has not been convincingly demonstrated by any major lab.** This phase is a genuine research extension. The portfolio does not depend on it; the striking system does not wait for it.

#### Combat Phase State Machine
```
 ┌────►│ STANDING │◄────┐
 │      clinch init      │ scramble / stand-up
 │           ▼           │
 │      │ CLINCH  │      │
 │      takedown ▼       │
 └──────│ GROUND  │──────┘
```

#### Virtual Grip System *(new — replaces friction-only gripping)*
Friction alone cannot produce gripping with mitten hands; this fails in every physics-RL setting that has tried it. Instead:
- When a grip geom contacts a valid target **and** the grip action for that hand is active, a MuJoCo equality (weld) constraint is created between hand and target body.
- **Velocity gate:** the weld may only activate when relative hand-to-target velocity is below ~0.5 m/s; above that, the grip action no-ops for the frame. Welding bodies with high relative velocity forces the solver to fire a huge corrective impulse — a physics spike agents would learn to exploit as a slam cannon. The gate is also the realistic semantics: you can't grab something flying past you.
- Releasing the grip action (or a break-force threshold) removes the constraint.
- Grip maintenance drains stamina (grip strength cost).

#### Positional Hierarchy (Ground Phase)
```
 1.0  Back Mount (hooks in)      -0.25 Full Guard (bottom)
 0.85 Mount                      -0.40 Half Guard (bottom)
 0.70 Side Control               -0.55 Side Control (bottom)
 0.55 Knee on Belly              -0.85 Mount (bottom)
 0.40 Half Guard (top)           -1.0  Back Mount (bottom)
 0.25 Full Guard (top)
 0.0  Neutral / Scramble
```
Rᵍ per timestep = Δ(position score).

#### Takedown Detection
Registered when fighter A's hip geom contacts the ground while fighter B's hip geom is within proximity and above A's — **and** B had grip/clinch contact with A within the previous ~1s. A strike-caused knockdown satisfies the same geometry; the grip-history check routes it to the knockdown reward path instead, keeping takedown stats and Rᵍ honest.

#### Submission System *(soft-limit based)*
Anatomical joint limits are **soft** (stiff restoring torque past the limit), not hard stops — a hard stop makes hyperextension undefinable. Submission stress = **integrated torque past the soft limit**:
- Armbar: accumulated elbow hyperextension stress
- Heel hook: knee rotational torque past limit under load
- Rear naked choke: sustained neck-geom contact pressure from arm geoms > 2s

Stress applies progressive T_max degradation to the defender (pain compliance). Held past threshold → tap-out terminal, Rₖ to the submitter.

---

### Step 7: Stamina & Damage Model

#### Stamina Pool — S ∈ [0, 1]
```
dS/dt = α_recovery − β_exertion
  α_recovery = ~0.01/s passive regen
  β_exertion = Σ |τ_applied[i]| × joint_cost[i]   (PD output torques)
```

| Stamina | Effect |
|---|---|
| S > 0.6 | Full performance |
| 0.4–0.6 | T_max → 85% |
| 0.2–0.4 | T_max → 65% |
| ≤ 0.2 | T_max → 40%, survival mode |

All effects flow **through T_max only** — fatigue behaviors (dropped guard, slow footwork) emerge from physics, never from scripts.

#### Damage State — six zones, permanent within an episode
| Zone | Trigger | Critical effect |
|---|---|---|
| Head | High-impulse head contacts | T_max penalty, neck/shoulders |
| Body | Liver/solar plexus impulses | Short full-body torque stall |
| L/R Leg | Accumulated leg kicks | T_max penalty hip/knee, limp |
| Neck | Choke accumulation | Observation noise injection |
| Arms | Submission stress | T_max penalty, affected elbow |

---

### Step 8: League Play & ELO Matchmaking

Naive latest-vs-latest self-play collapses to one dominant counter-strategy. League play (AlphaStar-style) fixes this.

#### The Archive
Every N steps the policy is snapshotted to disk: **weights + observation-normalizer statistics, always together** (a snapshot restored without its own normalizer stats plays corrupted and inflates ELO). Small MLP → ~5MB each; archive cost is trivial. Each snapshot is also tagged with the **morphology distribution it was trained under**: a Stage-1 snapshot has only ever piloted the standardized body, and the matchmaker must never assign it an extreme morphology it can't drive — those matches are free wins that corrupt league ELO.

#### Opponent Sampling (per worker, per episode)
| Pool | Probability | Purpose |
|---|---|---|
| Current policy | 35% | Core improvement loop |
| High-ELO historical | 40% | Robustness vs. best strategies |
| Random historical | 25% | Anti-forgetting |

Workers receive an opponent assignment in reset options, lazily load the snapshot from disk, and cache it.

#### Two ELO systems, deliberately separate *(v3.0 conflated these)*
1. **Policy ELO** — per archived snapshot; drives league matchmaking; updated from training match results. The primary training-progress metric.
2. **Morphology ELO** — per fighter body, **per policy generation**: computed by periodic frozen-policy tournaments across sampled fighter pairs. A lifetime morphology ELO is meaningless because the shared policy underneath it keeps improving; per-generation leaderboards instead show *which bodies rise as the meta evolves* — a flagship portfolio chart.

#### ELO Update
```python
K = 32
expected_A = 1 / (1 + 10 ** ((elo_B - elo_A) / 400))
elo_A += K * (result - expected_A)   # result: 1 / 0.5 / 0 (judge decides non-finishes)
```

**Anchoring:** a closed self-play league's ratings inflate over time. Two scripted reference opponents (the stationary dummy; a simple aggressive bot) are pinned at fixed ratings and included in periodic calibration matches — anchoring the scale so ELO is comparable across the whole run, and across reruns.

---

### Step 9: Portfolio Visualization Layer

**Live monitor:** MuJoCo native renderer + HUD overlay — stamina bars, damage-zone skeleton, ELO, combat phase, per-component reward breakdown.

**Web replay viewer:** notable episodes serialized to JSON (joint angles, positions, phase, stamina per frame; playback downsampled to 30Hz, gzipped). Three.js viewer on GitHub Pages: timeline scrubbing, epoch switching (watch the AI evolve), ELO display, slow motion. Zero server cost, zero in-browser physics.

---

### Step 10: The 1,000-Fighter Morphology System

One policy conditioned on a morphology vector; the network learns how anatomy changes optimal motor strategy.

#### Morphology Vector
```python
morphology = {
    "height":           [1.55, 2.05],   # m
    "mass":             [57, 125],      # kg
    "arm_reach":        [0.65, 0.90],   # m
    "leg_length":       [0.80, 1.10],   # m
    "torso_mass_ratio": [0.42, 0.58],
    "shoulder_width":   [0.40, 0.55],   # m
    "t_max_scale":      [0.7, 1.3],
}
```

#### Precompiled Model Cache *(new — load-bearing)*
Morphology values modify the MuJoCo XML, and `mj_compile` costs ~100ms+. Rebuilding XML per episode would burn a large share of wall-clock on compilation. Instead: all 1,000 fighter models are generated and compiled **once**, saved as binary `.mjb` files; workers preload their assigned subset and swap precompiled models at reset.

#### Registry
`morphology_registry.json`: 1,000 fighters via stratified sampling; per-generation morphology ELO (see Step 8), win/loss records, archetype tags.

#### Emergent Archetypes (hypotheses to validate, not promises)
| Archetype | Build | Expected emergent tactics |
|---|---|---|
| Stocky Powerhouse | 5'7", 85kg, short reach, high T_max | Level changes, hooks, cage-press wrestling |
| Lanky Sniper | 6'2", 80kg, long reach | Jab/teep range control, long-lever submissions |
| Heavy Pressure | 5'10", 115kg, very high T_max | Forward pressure, clinch grinding, attrition |
| Athletic Freak | 6'0", 84kg, long reach, very high T_max | Most versatile; likely ELO apex |

#### Staged Rollout *(unchanged — and the Step 0 slice extends the same philosophy downward)*
| Stage | Gate | Then |
|---|---|---|
| 1 | Single standardized fighter; full combat + league stable; Technical Era reached | Validate base system |
| 2 | 3–5 hand-picked morphologies | Confirm embedding learns meaningful representations |
| 3 | Full 1,000-fighter registry | Cross-morphology league live |

**Phase I imitation caveat stands:** keyframe references are authored for the standardized body; imitation weight is annealed before extreme morphologies are introduced, so physics — not average-body references — shapes their styles.

---

## 📈 Expected Training Eras *(striking eras are the commitment; grappling eras are the research extension)*

| Frames | Era | Behavior |
|---|---|---|
| 0–10M | Primordial | Gravity wins. Flailing, collapse. Locomotion emerges. |
| 10M–50M | Exploitative | Bull-rush, spin-to-win, leap-attack spam. |
| 50M–150M | Technical Striking | Counters, guard, footwork, cage awareness, early archetype divergence. **Portfolio-complete milestone.** |
| 150M–300M | Grappling Integration *(extension)* | Takedowns, cage-pinning, wall-off escapes. |
| 300M–500M | Strategic *(extension)* | Morphology-driven styles crystallize; pacing, leg-kick attrition. |
| 500M+ | Meta *(extension)* | Counter-styles; ELO stratification across the league. |

---

## 🛟 Guardrails

| Guardrail | Implementation |
|---|---|
| Torque ceiling | Actuator `forcerange` = ±T_max per joint |
| Anatomical limits | **Soft limits** (restoring torque past range); doubles as submission-stress sensor |
| Self-intersection | MuJoCo self-collision geoms |
| Stun accumulator | Impulse above threshold → temporary T_max debuff (10–30 frames) |
| Smoothness | PD control provides it structurally; small jerk penalty in Rₛ as backstop |
| Takedown velocity cap | Slam velocity clamped to biological range |
| Anti-exploit monitor | Episodes with total impulse > 5σ above rolling mean are **masked out of the current rollout** before the PPO update |
| Energy conservation | Output energy per second above human-muscle ceiling → action clipped pre-physics |

---

## 🗂️ Repository Structure

```
mma-machine/
├── envs/
│   ├── mma_env.py              # Gymnasium env; frozen opponent runs inside step()
│   ├── physics/
│   │   ├── humanoid_base.xml
│   │   ├── humanoid_builder.py # Morphology-parameterized XML generator
│   │   ├── model_cache.py      # Precompiled .mjb cache (1,000 fighters)
│   │   └── arena_octagon.xml
│   ├── pd_control.py           # PD target→torque layer, T_max clamping
│   ├── grappling.py            # Phase state machine, virtual grip welds
│   ├── stamina.py              # Stamina + damage model
│   └── octagon.py              # Cage-pin detection, wall normals
├── agents/
│   ├── ppo.py                  # Custom PPO (rollout buffer, episode masking)
│   ├── actor_critic.py         # Morphology-conditioned net, privileged critic
│   ├── normalizer.py           # Running obs stats — snapshotted with weights
│   └── league.py               # Archive, opponent sampling, policy ELO
├── morphology/
│   ├── registry.py             # Population generator (stratified sampling)
│   ├── morphology_registry.json
│   ├── tournaments.py          # Frozen-policy morphology-ELO tournaments
│   └── archetypes.py
├── training/
│   ├── train.py
│   ├── curriculum.py           # Phase gates, reward schedule, episode lengths
│   ├── reward_matrix.py
│   └── eval_harness.py         # Fixed scripted/frozen opponents, fixed seeds
├── reference_motion/
│   ├── keyframes/              # Hand-authored stance/jab/cross/teep/level-change
│   └── process_cmu.py          # Locomotion only
├── visualization/
│   ├── monitor.py              # Live viewer + HUD
│   └── serialize_episode.py
├── web_viewer/                 # GitHub Pages portfolio viewer
│   ├── index.html
│   ├── viewer.js
│   ├── leaderboard.js          # Per-generation morphology ELO boards
│   └── episodes/
├── configs/
│   └── training_config.yaml
└── README.md
```

---

*Blueprint v3.2 — planning is complete; the remaining unknowns are empirical. Step 0 is the next action: one morphology, flat arena, scripted dummy, PD targets, plain PPO, tuned PD gains, a ROCm pass, and a measured throughput number.*
