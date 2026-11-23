# G1 memo: does the declared information leave useful corrective authority?

Owner: Claude (WP 1.3) · Reviewer: ChatGPT · Decision: Prasuna · November 22, 2026

**Question (execution plan, G1).** With realistic telemetry, does the set-membership gain
interval narrow fast enough, and do the hold envelopes leave enough admissible correction,
for the identification-aware rule to do anything useful after a fault?

**Method.** Offline replay, no robot in the loop. Recorded packets from the bench's `none`
arm (walk–turn–stop–resume, 300 steps, 4 episodes per family, 12 leg actuators; commit
c6234bf, `results/bench_dev_packets_v0`) are processed in source order through ChatGPT's WP 1.2
primitives (`update_gain_set`, `propagate_gain`, `hold_error`, `project_offset`, PR #34):
prior [0.3, 1.2], drift bound 0.05/s, reacquisition from the prior on INCONSISTENT, inverse
candidate = set midpoint, the candidate's own request carried across steps under the 2 rad/s
slew limit, amplitude 0.2 rad, supported force 80 N·m. Two sensor channels: *exact* (declared
bounds 1e-5 rad, 1e-3 N·m) and *declared* (encoder noise 2 mrad, effort noise 0.5 N·m,
20–60 ms latency, 2% dropout; declared bounds 6 mrad, 1.5 N·m). Two declared joint-speed
bounds for the hold envelope: 3 rad/s and 1 rad/s. Script `scripts/authority_replay.py`;
results `results/authority_dev_v1_speed{3.0,1.0}`.

**Results (declared telemetry; medians/means over actuators and episodes after onset).**

| Fault | Speed bound | Steps to exclude g = 1 | Set width | Requested / oracle | Steps with zero authority |
|---|---|---:|---:|---:|---:|
| 0.85 | 3 rad/s | 6 (15% never) | 0.26 | 0.20 | 70% |
| 0.70 | 3 rad/s | 4 (10% never) | 0.25 | 0.44 | 61% |
| 0.55 | 3 rad/s | 4 (12% never) | 0.25 | 0.28 | 59% |
| 0.85 | 1 rad/s | 6 | 0.26 | 0.45 | 45% |
| 0.70 | 1 rad/s | 4 | 0.25 | 0.71 | 44% |
| 0.55 | 1 rad/s | 4 | 0.25 | 0.39 | 44% |

**Reconciliation with PR #40 (ChatGPT's replay, `hold_speed_rad_s = 10`, drift 0).** Rerunning this
replay with the same settings (`results/authority_dev_v1_speed10.0`) gives requested/oracle 0.00 /
0.02 / 0.03 at 0.85 / 0.70 / 0.55 with 92–99% zero-authority steps on the declared channel, which
agrees with #40 (hips 1.3%, knees 19% at 0.70) and with the closed-loop dev result in PR #39
(candidate at `none` level on declared telemetry). The two memos differ only in the declared
speed bound; the mechanism is the same.

| Speed bound (declared, 0.70) | Requested / oracle | Zero-authority steps |
|---|---:|---:|
| 10 rad/s (PR #39/#40 config) | 0.02 | 96% |
| 3 rad/s | 0.44 | 61% |
| 1 rad/s | 0.71 | 44% |

Exact telemetry: g = 1 excluded one step after onset, width ≈ 0.01, requested/oracle
0.51–0.95, zero-authority steps 13–33%. Bound violations against the true gain: 3–9 per
family over 14,400 declared samples (coverage ≥ 99.9%); INCONSISTENT events are the onset
jump itself (one per actuator) plus a handful of noise resets.

By joint group (declared, 0.70): hips 0.48 (3 rad/s) → 0.84 (1 rad/s), knees 0.69 → 0.86,
ankles 0.25 → 0.43; ankle roll (K_p = 2) has almost no authority at any setting because the
effort bound dominates its regression.

**Reading.** (1) Identification is fast: the set excludes the healthy gain within 80–120 ms
of onset under declared noise. (2) The set width (≈ 0.25) is set by ε/|φ| and does not shrink
further; this costs little authority by itself. (3) The hold envelope is the main limiter:
with a 3 rad/s declared speed bound the error interval contains zero on most steps and the
no-worse-force condition then permits nothing. A 1 rad/s bound restores 70–85% of the
oracle's correction on hips and knees at 0.70, but 1 rad/s is not a physical bound for a
swinging knee. (4) The 0.55 fault has less relative authority than 0.70 because larger
tracking errors saturate the amplitude and force limits.

**Decision options.**
- **Go** as designed with the 3 rad/s bound: the candidate acts on about 40% of the
  oracle's correction at 0.70 and 20% at 0.85; expect it to land between `none` and the
  heuristic on recovery and to win only on harmful intervention.
- **Modify (recommended):** declare the envelope variant the proposal already allows: an
  error envelope built from the *measured* velocity and a declared acceleration bound
  (radius b_q + |v̂|·(age+H) + ½a_max(age+H)²) instead of a global speed bound, with its
  coverage measured in the replay before use. This keeps the rule's logic and recovers
  authority where the joint is actually slow. Register it as the method before G2.
- **Stop** the research claim: not supported by this data; authority exists.

**Acceptance for the modified envelope (before G2):** coverage of the error envelope ≥ 99%
on the declared channel in the replay; requested/oracle ≥ 0.5 on hips and knees at 0.70.

**What this does not show.** Closed-loop recovery (the replay never applies the request);
the heuristic comparison; behaviour under the scale confound (the set is as blind as the
point estimate there, AE-003); GPU timing.

## Addendum, November 22 (after ChatGPT's velocity-envelope study, PR #40 / `results/velocity_envelope_dev_wp13`)

The recommendation above is superseded in one respect: the binding variable is the **age of the
sample**, not the speed bound. Knee acceleration at ground contact reaches 284 rad/s² (99th
percentile) and 384 rad/s² (maximum) on the development walks, so any position envelope that
covers the knees needs A ≥ 400 rad/s². At the declared 20–60 ms transport the envelope radius
½·A·(age+H)² is then 0.7–0.9 rad, wider than any useful correction, for every formula tried
(symmetric or signed velocity, seven acceleration bounds). The modified envelope meets the
acceptance criteria only when the packet captured at the end of transition t is usable at
decision t+1 (hips 57%, knees 72% of the same-state oracle; coverage ≥ 99.6%).

**Revised recommendation (Modify, redefined).** Keep the declared 20–60 ms arm as the primary
result. Register a second information assumption, `DECLARED_FAST` (same noise, bounds and 2%
dropout, zero latency; `sensing.py`, commit a5d6719), as a named arm beside it. Freeze the
candidate's operating point at the symmetric 400 rad/s² envelope with the declared ±0.5 rad/s
velocity bound (now part of `TelemetryConfig`). The research claim becomes the recovery/harm
trade-off as a function of sensor latency — a value-of-timely-information result — rather than a
claim that the identification-aware rule wins under the original transport. Same-state oracle
ratios stay in the development record; the paper's metrics are closed-loop (paired post-fault
loss, falls, healthy-plant offset, withdrawal steps). ChatGPT's proposed CPU pilot
(`results/velocity_envelope_dev_wp13/next_experiment.md`) is the next step, run against the
frozen tuned heuristic.
