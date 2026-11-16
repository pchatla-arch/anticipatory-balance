# Independent review of *Fleet Robust Execution*, design draft 0.2

Reviewed: `docs/adaptive_execution/paper.md` at `d44883d`, `foresight/adaptive/actuator_env.py`,
`scripts/eval_adaptive_gate.py`. Date: November 15, 2026. Review only: no paid jobs, no
history changes. CPU checks: `python docs/adaptive_execution/reviews/check_effort_alignment.py`.

## Overall readiness

**Stage A (known-gain gate) is ready to run.** The interface identity holds and the
evaluator's failure accounting is sound (items V1–V5). **Stage B is not ready.** Two
verified measurement defects (R1, R2) would corrupt the very identification result it is
designed to test. One design gap (R3) leaves the central fault-versus-confound claim
untestable in the primary telemetry condition. All three fixes are small. I accept the
draft's corrections to the four-page specification, which I wrote: contact freezing,
1 kHz, the total-target slew bound, the "safe fallback" wording, the automatic theme
switch, and "validated component".

## Verified correct (no change needed)

| ID | Item | Evidence |
|---|---|---|
| V1 | Actor history is nominal: `last_act` restored after the parent step; `last_last_act` inherits the restored value on the next step; the observation uses `last_act` only | `actuator_env.py` `step`; Playground `joystick.py` L402–403, L512 |
| V2 | Healthy identity: with `g = 1` the correction is exactly 0 and the parent receives the original action, unclipped | `bounded_gain_correction`, `has_correction` branch |
| V3 | Fault scales `gainprm[0]` and `biasprm[1]` together; damping is joint `dof_damping` (`biasprm[2] = 0` asserted) | constructor assertions |
| V4 | No projectile in the gate (`launch=False`), no command resampling within 400 steps, no domain randomization at evaluation | `eval_adaptive_gate.py` L242–289 |
| V5 | Fixed-horizon loss charges 4 to every step after termination; non-finite lanes recorded separately and fed safe reset inputs | `one_step` |
| V6 | Proposition algebra: [g(1+c)−1]² − (g−1)² = gc[gc + 2(g−1)], nonpositive for 0 ≤ c ≤ 2(1/g − 1) | paper, "A conditional pointwise result" |

## Findings

| ID | Severity | Location | Verified / hypothesis | Mechanism | Minimal correction | Smallest resolving check |
|---|---|---|---|---|---|---|
| R1 | **Blocking for Stage B** | Paper, "Scalar estimator"; any estimator reading `data.actuator_force` with the returned `data.qpos` | **Verified** (CPU) | After a MuJoCo step, `actuator_force` was computed from the joint position *before* the final integration substep. Pairing it with the returned position mis-states φ = r − q. On a healthy robot (g = 1) the implied gain error is median 0.33%, max 7.2% per joint in a single 2 ms substep, growing with joint speed. That is a third of the 0.85 fault's size, appearing as a fake fault. | Telemetry packet = (q, effort) from the same physics evaluation: record q before the last substep, or recompute effort at the returned state with one `mjx.forward`. State this in the telemetry contract. | Healthy-robot RLS run with exact effort: ĝ within 1e-4 of 1 when aligned, biased when not (`check_effort_alignment.py` prints both) |
| R2 | **Major** | Paper, "Constraint interaction" and "Saturation rejection"; evaluator `max_force` diagnostic | **Verified** (CPU) | `actuator_force` is computed before joint-level actuator-force limits (`jnt_actfrclimited`; leg limits 88/139/50 N·m). Under a saturating command, 8 of 12 leg joints read 166–225 N·m while receiving 88 or 139. An "exact effort" channel built on `actuator_force` therefore never sees saturation, so the estimator reads a perfect gain while the joint is torque-limited. | Define effort as applied joint torque (`qfrc_actuator` at the actuator DoF); set the saturation flag when it reaches the joint limit. Rename the evaluator field to `max_commanded_actuator_force` or switch it to applied torque. | Same script, saturating command: applied ≠ commanded on 8 of 12 leg joints |
| R3 | **Major** | Paper, Stage B nuisance set (friction, external loading, contact) | Hypothesis, from the model | With synchronized applied-torque telemetry, τ = g·Kp·(r − q) holds whatever the payload, push or contact state. These confounds change the motion, not that relation, so effort-based false adaptation is near zero by construction and the headline control cannot fail. The confounds that *do* mimic a gain change are on the measurement side: joint friction or damping inside a torque-sensor reading, sensor scale and bias, and saturation. Payload, push and contact are the right confounds only for the motion-based condition. | Make the confound set per telemetry condition. **Effort-based**: friction-loss change, sensor scale/bias, saturation. **Motion-based**: payload, push, contact. State the research question for each. | One pilot per condition on a healthy robot: false-adaptation rate is a non-trivial number in at least one cell |
| R4 | Major | Paper, "No direct effort" condition | Hypothesis | The draft leaves the motion-history baseline unspecified. Without effort, gain must come from inverse dynamics, which needs contact forces the contract withholds. As written, the arm can only report "unavailable". | Make the no-effort arm the **learned history estimator only** (Stage D), framed as "what learning recovers when the physical model cannot be run". Drop it from the Stage B gate. | — |
| R5 | Minor | Paper, fleet/instance model | **Verified** (model) | Leg proportional gains are 75 (hip, knee), 20 (ankle pitch) and **2 (ankle roll)**. An ankle-roll gain of 2 makes τ tiny for normal errors, so gain on that joint is weakly identifiable and also matters little. | Report identifiability per joint group; exclude ankle roll from the gain-identification claim or state its excitation separately | Excitation statistic I per joint on healthy walking |
| R6 | Minor | Paper header; README | — | The evidence boundary is stated well. The build depends on Pandoc plus Liberation fonts; fine, already documented. | none | — |
| R7 | Minor | Stage A gate thresholds | Hypothesis | "Least severe passing gain" can pick 0.85 with a small excess cost, so a half-recovery ratio rests on a small denominator. | Require a minimum absolute degradation (e.g. excess post-fault loss and failure difference both above stated floors) before computing R_feasible | Read off the screen's excess-cost column before selection |

## Corrected statements

- **Effort–state pairing (R1).** φᵢ = r^sent − q(tᵢ), where q(tᵢ) is the joint position at
  which the effort sample was computed. MuJoCo's post-step `actuator_force` corresponds to
  the pre-integration position of the last substep.
- **Effort definition (R2).** Applied effort is τ^app = clip(g·Kp·φ, −τ_max, τ_max) at the
  actuator's DoF. The regression y = τ^app/Kp = g·φ is valid only for unsaturated samples;
  a sample with |τ^app| = τ_max is rejected and raises a headroom flag.
- **What effort-based confounds can be (R3).** If the effort channel is a joint-torque
  sensor after the gearbox, it reads τ^app − τ_fric, so a friction change biases ĝ by
  −τ_fric/(Kp·φ). A friction change is the effort-side analogue of a gain change, and the
  right negative control.

## Strongest supportable contribution statement (conditional on Stages B–D)

> A position-target execution adapter for a frozen G1 PPO walker that applies an estimated
> gain inverse only when an identification-validity screen supports it: excitation,
> innovation consistency, measurement age, and joint-torque headroom. Evaluated against
> an uncapped estimated inverse, a learned history estimator and actuator-randomized PPO
> on held-out simulated instances, it reports recovery, false adaptation under
> measurement-side confounds (friction, sensor scale and bias) and per-step runtime.

## Implementation order (architecture unchanged)

1. Run Stage A as built. Optionally rename the `max_force` field (R2); everything else is
   unaffected.
2. Telemetry module: a synchronized (q, applied effort) packet (R1, R2), declared noise,
   bias and delay, saturation flag. Unit test: healthy ĝ = 1 within 1e-4 with exact
   aligned effort.
3. RLS estimator plus excitation and innovation statistics, replayed offline on Stage A
   traces before any new GPU run.
4. Stage B with per-condition confound sets (R3) and per-joint identifiability (R5).
5. Confidence screen and the uncapped-inverse comparison (H3).
6. Learned history estimator as the no-effort arm and the learned comparison (R4).
7. Actuator-randomized PPO; then the work cycle, C++ replay parity and timing.
