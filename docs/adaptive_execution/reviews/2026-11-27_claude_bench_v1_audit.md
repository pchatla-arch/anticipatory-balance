# CPU bench v1 with the frozen heuristic: results and audit (Claude, Nov 27 2026, 00:30 PT)

Branch `oaa/claude-bench`, results commit `be29666`, figures commit `4f9c4fb`.
Every number below is copied from `results/bench_dev_cpu_v1_frozen/<preset>/episodes.csv`
(288 rows per preset) or from the trace `.npz` files; `summary.json` and `report.md` were
re-derived from `episodes.csv` and all 108 cells matched to 1e-9. No non-finite episode,
no pre-fault failure, no control-bound violation in any preset. ChatGPT should re-derive
at least the declared-preset table before any number goes into the report.

## 1. What ran

```
scripts/eval_adaptive_bench.py --out results/bench_dev_cpu_v1_frozen/$T --split dev \
  --families gain_085,gain_070,gain_055,gain_070_return,healthy,sensor_scale_080,sensor_bias,damping_x3,stale_telemetry \
  --methods none,oracle,estimated_inverse,heuristic_gate --telemetry $T --episodes 8 --steps 300 \
  --impl jax --allow-cpu --trace --heuristic results/tune_heuristic_dev_cpu_v0/heuristic.json
```
for `T` in `declared` (06:15–06:47 UTC, source `d756c37`), `declared_fast` (06:47–07:20, `aec7f18`)
and `exact` (07:20–07:53, `aec7f18`). The two source commits differ only in
`scripts/make_oaa_figures.py`. CPU, JAX 0.12.0.1, MuJoCo/MJX 3.16.0, checkpoint
`nominal_s1/000202342400`, seed 11000, schedule `walk_turn_stop_resume`, 300 control steps
(6 s), fault onset uniform in steps 75–125, return (where scheduled) 100–150 steps after
onset. Scenario ids and sensor draws are identical across the four methods inside a family,
so every comparison below is paired on 8 scenarios. Frozen heuristic (unchanged since
`d756c37`): forgetting 0.90, dead band 0.10, max residual RMS 0.02, max age 3 steps,
min samples 10, min excitation 0.02 rad. It was tuned on `gain_070, gain_070_return,
healthy, sensor_scale_080` only (declared telemetry, n = 8, 17 evaluations), so `gain_085`,
`gain_055`, `sensor_bias`, `damping_x3` and `stale_telemetry` are held-out families.

## 2. Main table (declared telemetry: 2 mrad / 0.5 N·m noise, 1–3 step latency, 2 % dropout)

Post-fault loss = capped planar-velocity error averaged over the post-onset horizon.
Harmful offset = mean |effective offset| over steps where the plant is healthy (whole
episode for the healthy, sensor and damping families; after the return for `gain_070_return`;
before onset otherwise). Withdrawal = median steps after the return until every joint's
offset is below 0.01 rad.

| Family | Method | Loss | Falls | Harmful offset (rad) | Withdrawal (steps) |
|---|---|---:|---:|---:|---:|
| healthy | none / oracle | 0.0224 | 0/8 | 0 | |
| healthy | estimated_inverse | 0.0219 | 0/8 | 0.0764 | |
| healthy | heuristic_gate | 0.0228 | 0/8 | 0.0004 | |
| gain_085 | none | 0.0290 | 0/8 | 0 | |
| gain_085 | oracle | 0.0269 | 0/8 | 0 | |
| gain_085 | estimated_inverse | 0.0269 | 0/8 | 0.0842 | |
| gain_085 | heuristic_gate | 0.0277 | 0/8 | 0.0004 | |
| gain_070 | none | 0.0734 | 0/8 | 0 | |
| gain_070 | oracle | 0.0283 | 0/8 | 0 | |
| gain_070 | estimated_inverse | 0.0295 | 0/8 | 0.0823 | |
| gain_070 | heuristic_gate | 0.0322 | 0/8 | 0.0002 | |
| gain_055 | none | 2.6355 | **8/8** | 0 | |
| gain_055 | oracle | 0.0413 | 0/8 | 0 | |
| gain_055 | estimated_inverse | 0.0361 | 0/8 | 0.0776 | |
| gain_055 | heuristic_gate | 1.8547 | **6/8** | 0.0008 | |
| gain_070_return | none | 0.0502 | 0/8 | 0 | 0 (never applied) |
| gain_070_return | oracle | 0.0261 | 0/8 | 0.0009 | 2 |
| gain_070_return | estimated_inverse | 0.0264 | 0/8 | 0.0807 | never (8/8) |
| gain_070_return | heuristic_gate | 0.0299 | 0/8 | 0.0037 | 6.5 |
| stale_telemetry (gain 0.70 seen 5–10 steps late) | none | 0.0570 | 0/8 | 0 | |
| stale_telemetry | oracle | 0.0300 | 0/8 | 0 | |
| stale_telemetry | estimated_inverse | 0.0317 | 0/8 | 0.0988 | |
| stale_telemetry | heuristic_gate | 0.0570 | 0/8 | 0 (never opened) | |
| sensor_scale_080 | none / oracle | 0.0309 | 0/8 | 0 | |
| sensor_scale_080 | estimated_inverse | 0.0322 | 0/8 | 0.1072 | |
| sensor_scale_080 | heuristic_gate | 0.0326 | 0/8 | 0.0466 | |
| sensor_bias (+2 N·m) | none / oracle | 0.0271 | 0/8 | 0 | |
| sensor_bias | estimated_inverse | 0.0263 | 0/8 | 0.1565 | |
| sensor_bias | heuristic_gate | 0.0277 | 0/8 | 0.0082 | |
| damping_x3 | none / oracle | 0.0346 | 0/8 | 0 | |
| damping_x3 | estimated_inverse | 0.0344 | 0/8 | 0.0814 | |
| damping_x3 | heuristic_gate | 0.0342 | 0/8 | 0.0002 | |

Paired differences in loss, declared telemetry, mean over the 8 scenarios with a paired
bootstrap 95 % interval (5000 resamples):

| Family | gate − none | gate − oracle | inverse − none | gate − inverse |
|---|---|---|---|---|
| gain_085 | −0.0013 [−0.0028, +0.0003] | +0.0008 [+0.0000, +0.0015] | −0.0021 [−0.0043, −0.0001] | +0.0008 [−0.0004, +0.0019] |
| gain_070 | −0.0412 [−0.0628, −0.0214] | +0.0039 [+0.0006, +0.0081] | −0.0439 [−0.0648, −0.0247] | +0.0026 [−0.0014, +0.0073] |
| gain_055 | −0.781 [−1.428, −0.224] | +1.814 [+0.987, +2.471] | −2.599 [−2.824, −2.322] | +1.819 [+1.037, +2.474] |
| gain_070_return | −0.0203 [−0.0310, −0.0136] | +0.0038 [+0.0012, +0.0065] | −0.0238 [−0.0343, −0.0166] | +0.0034 [+0.0016, +0.0054] |
| stale_telemetry | 0 (identical) | +0.0270 [+0.0169, +0.0394] | −0.0253 [−0.0373, −0.0150] | +0.0253 [+0.0146, +0.0375] |
| healthy | +0.0004 [−0.0010, +0.0020] | same | −0.0005 [−0.0013, +0.0001] | +0.0009 [−0.0004, +0.0025] |
| sensor_scale_080 | +0.0018 [−0.0002, +0.0042] | same | +0.0013 [−0.0000, +0.0030] | +0.0005 [−0.0008, +0.0021] |
| sensor_bias | +0.0006 [−0.0000, +0.0012] | same | −0.0009 [−0.0038, +0.0016] | +0.0014 [−0.0008, +0.0039] |
| damping_x3 | −0.0003 [−0.0008, +0.0002] | same | −0.0001 [−0.0012, +0.0009] | −0.0002 [−0.0010, +0.0005] |

## 3. Across telemetry presets

Gain-fault loss by preset (n = 8 each; falls in brackets):

| Family | Method | exact | declared_fast | declared |
|---|---|---:|---:|---:|
| gain_085 | none / oracle / inverse / gate | 0.0290 / 0.0269 / 0.0267 / 0.0266 | 0.0290 / 0.0269 / 0.0269 / 0.0274 | 0.0290 / 0.0269 / 0.0269 / 0.0277 |
| gain_070 | none / oracle / inverse / gate | 0.0734 / 0.0283 / 0.0293 / 0.0325 | 0.0734 / 0.0283 / 0.0291 / 0.0308 | 0.0734 / 0.0283 / 0.0295 / 0.0322 |
| gain_055 | none / oracle / inverse / gate | 2.6355 [8/8] / 0.0413 / 0.0412 / 1.0770 [3/8] | 2.6355 [8/8] / 0.0413 / 0.0436 / 1.2521 [4/8] | 2.6355 [8/8] / 0.0413 / 0.0361 / 1.8547 [6/8] |
| gain_070_return | none / oracle / inverse / gate | 0.0502 / 0.0261 / 0.0256 / 0.0266 | 0.0502 / 0.0261 / 0.0260 / 0.0283 | 0.0502 / 0.0261 / 0.0264 / 0.0299 |

`none` and `oracle` do not read telemetry, so their rows are identical across presets
(a useful determinism check: CPU physics is bitwise repeatable).

## 4. The gain-0.55 result and its mechanism

The uncorrected walker falls in every 0.55 episode, 70–135 steps after onset (termination
steps 157–259). Oracle and the ungated estimated inverse keep all 8 episodes up under every
preset, with loss 0.036–0.044, i.e. the 0.20 rad / 2 rad/s bounded correction is sufficient
for this fault when it is applied promptly. The frozen gate loses 3/8 (exact), 4/8
(declared_fast) and 6/8 (declared) episodes. From the traces, on the 12 leg joints:

| Window after onset | exact: gate open | declared: gate open | reason closed (declared) | mean |offset| gate / inverse (declared) |
|---|---:|---:|---|---:|
| 0–10 steps | 6 % | 1 % | residual 43 %, stale 38 %, dead band 19 % | 0.000 / 0.031 rad |
| 10–20 | 16 % | 1 % | residual 65 %, stale 33 % | 0.000 / 0.087 |
| 20–40 | 55 % | 17 % | residual 53 %, stale 29 % | 0.014 / 0.092 |
| 40–80 | 100 % | 45 % | stale 33 %, residual 23 % | 0.050 / 0.090 |
| 80–200 | 100 % | 45 % | stale 34 %, residual 20 % | 0.050 / 0.091 |

Two gate conditions do the damage. (a) The residual test (RMS of the RLS innovation above
0.02) is failing exactly during the estimator's transient after a large step in gain, so the
gate waits 20–40 steps (0.4–0.8 s) before releasing a correction that the inverse releases on
the first post-onset packet; at gain 0.70 that delay costs +0.003 loss, at 0.55 it is a fall
in 3 of 8 episodes even with exact telemetry. (b) Under declared telemetry the age test
(`max_age_steps` 3) is closed on about a third of all ticks, including long after the
transient and on the healthy family (36.7 % of leg-joint ticks after step 150). The reason
is a one-step accounting lag, not the queue: the correction for step t is decided before the
step-t capture is pushed, so the newest assimilated packet was captured at t − 1 − L with
latency L ∈ {1, 2, 3}, i.e. it is 2–4 steps old at decision time, and the gate is closed
whenever the newest packet had latency 3 (replaying the recorded arrivals gives 37.5 %).
The tuner chose 3 from {3, 5, 10} on the four tuning families, where the closed ticks cost
little loss and reduced the sensor-scale offset. The gate therefore applies, on average,
half the inverse's offset (0.050 vs 0.090 rad) for the rest of the episode; the same halving
is visible at gain 0.70 (0.029 vs 0.055 rad, `apply_steps` 124/300). The estimate itself is
not the problem: the leg-mean gain estimate reaches 0.56–0.57 by onset+40 for both methods.

Neither the dead band nor the sample count matters here: |ĝ − 1| leaves the 0.10 dead band
within 10 steps, and 10 samples are available before onset.

## 5. Other findings

* **Command following (gain 0.70, declared, from the traces, 8 episodes):** achieved speed
  during the commanded stop is 0.315 m/s uncorrected, 0.101 with the gate, 0.055 with the
  inverse, 0.057 with the oracle, 0.052 for the healthy walker. The uncorrected walker also
  overshoots the turn (0.384 vs 0.300 m/s commanded). This is the CPU, n = 8 version of the
  S1a finding and replaces the "progress trade-off" reading: the extra distance of the
  uncorrected walker is failure to stop, not better locomotion.
* **Withdrawal after an unannounced return (`gain_070_return`):** oracle 2 steps, gate
  6.5 steps (median), inverse never within the 25–125 steps left after the return (8/8 under
  declared telemetry; with exact telemetry it did withdraw in some episodes). Despite that,
  the inverse's post-fault loss (0.0264) is below the gate's (0.0299), because its leftover
  offsets after the return are inside the bound and the walker tolerates them.
* **Stale telemetry (gain 0.70 seen through 100–200 ms old packets):** the gate never opens
  (`max_age_steps` 3 < latency 5–10), so it equals no adaptation (0.0570); the inverse
  reaches 0.0317 against the oracle's 0.0300. The gate's age rule, which is the right rule
  against out-of-date beliefs in general, forfeits the whole benefit here because this fault
  is constant and a 200 ms old estimate of it is still correct.
* **Non-gain families:** the gate keeps the healthy-plant offset at 0.0002–0.0082 rad on
  healthy, sensor-bias and damping, and 0.047 rad on sensor-scale (where ĝ ≈ 0.8 is the
  correct estimate of an unidentifiable effort scale); the inverse applies 0.076–0.157 rad
  on all of them. Loss consequences of either are within ±0.002 of no adaptation; the
  sensor-scale family is the only one where both adapters come out above doing nothing
  (gate +0.0018, inverse +0.0013), and both intervals touch zero.
* **Falls:** only the two 0.55 cells (`none`, `heuristic_gate`) fall; all 100 other cells
  are 0/8.

## 6. What this does to the paper's claims

The frozen gate is the method the plan and the draft put forward. On this bench it does not
win on the primary metric anywhere:

* Against the ungated inverse it is worse or indistinguishable on loss in every family and
  preset; the only direction where it is better is the harmful-offset exposure (10–200×
  lower), which did not translate into loss on any family in this bench.
* Against the oracle it is within +0.004 at gain 0.85/0.70/return, which is the strong
  result for a controller that has to estimate the gain through declared telemetry.
* At gain 0.55 it fails where the plain inverse does not, and the failure grows with
  telemetry degradation (3/8 → 4/8 → 6/8).
* Under stale telemetry it does nothing.

The honest paper, in my reading, is therefore the comparison itself: *when should a bounded
corrector wait for confidence before acting on an uncertain belief?* With a 0.20 rad / 2 rad/s
bound, acting on the point estimate immediately beats waiting, because the bound already
limits the cost of acting on a wrong belief and the waiting is fatal when the fault is severe.
That is a decision-under-uncertainty result and it is supported by every row above. The
gate's selectivity (near-zero offsets on healthy, biased-sensor and damping plants) remains
a true and reportable property, with the sensor-scale confound as the stated identifiability
limit. What cannot be written: "the gate improves recovery", "the gate is safe", or any
sentence that implies the gate was chosen because it performs better. Specific edits:

1. Abstract and §4: report gain 0.70 as loss 0.0734 → 0.0322 (gate) / 0.0295 (inverse) /
   0.0283 (oracle), n = 8, declared telemetry, and add gain 0.55 falls 8/8 → 6/8 (gate) /
   0/8 (inverse, oracle). The S1a GPU numbers (0.0855 → 0.0383, n = 2) can stay as the GPU
   confirmation of the 0.70 cell only.
2. Remove the progress column from the main table (it is displacement, and the uncorrected
   walker's extra displacement is a failure to stop); replace with the stop-segment achieved
   speed from §5 or cite `paper/figures/command_following_gain_070_declared.pdf`.
3. Restoration: keep the 5.5-step S1a statement only alongside the n = 8 figures (gate 6.5,
   oracle 2, inverse never), and add that the inverse's loss after the return is nevertheless
   lower.
4. Add the stale-telemetry row and say plainly that the age rule forfeits the benefit.
5. Limitations: the gate thresholds were tuned on four families at gain 0.70 and did not
   transfer to 0.55; the residual test conflates "model does not fit" with "model is
   mid-transient"; a sequential test on the innovation, or a gate that scales the offset by
   confidence instead of switching it, is the obvious next step and is not measured here.
6. Figures: `paper/figures/bench_overview.pdf` (main), `telemetry_curve.pdf`,
   `command_following_gain_070_declared.pdf`, `timeseries.pdf`. The earlier `tradeoff.pdf`
   is withdrawn: its gain-family mean was dominated by capped fall losses.

## 7. Decisions for Prasuna

* D1. Frame the paper as the gate-versus-bounded-inverse comparison (recommended), or keep
  the gate as "the method" and report the 0.55 and stale-telemetry failures as limitations.
  Both are honest; the first is the better paper.
* D2. Whether to spend 90 CPU minutes on one more preset-free check: the same bench on the
  `test` split (new scenario seeds) for the four gain families, so the paper's main numbers
  are not from the split the gate was tuned on. I recommend yes; it does not need the GPU.
* D3. The optional GPU G3 batch (4 arms × 5 seeds, ~$1.15): with the 0.55 result in hand, a
  G3 batch at gain 0.70 only would show the gate helping in the loop, which is true but now
  the weaker half of the story. If it runs, it should include one 0.55 arm or be dropped.
* D4. The exact-telemetry gate falls (3/8) mean the residual rule alone is enough to lose
  episodes; no change to the frozen heuristic before the deadline (re-tuning on the same
  dev families would be the thing the protocol forbids), but D2's test split would show
  whether the 0.55 failure rate is stable.
