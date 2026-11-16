# FS-001 Session H hold ablation stopped on non-finite progress

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Major for Foresight's hold ablation; the AA228 paper does not depend on it |
| Project · area | Foresight · evaluator, Session H |
| Found | 2026-11-15, Session H run `session_h_20261115T211850Z_1f77c1` (source `eb493be`) |
| Issue | #15 |
| Owner | ChatGPT (Session H harness) · Claude reviews |

## Symptom
Both depth gates passed. The hold gate failed, and the five follow-up stages (E18b hold 0.3 /
0.6 / 1.0 s, E17c depth 0 / 100 ms) did not start. In all nine hold conditions the same early-
failing scenario has a non-finite progress value, and the run stops during paired-result
validation.

## Evidence
- **Verified** from the published run:
  [report](https://github.com/pchatla-arch/anticipatory-balance/blob/1d9ee6d3493f58db22954bd0d5c5fe7944826405/results/session_h_20261115T211850Z_1f77c1/report.md),
  [console log](https://github.com/pchatla-arch/anticipatory-balance/blob/1d9ee6d3493f58db22954bd0d5c5fe7944826405/results/session_h_20261115T211850Z_1f77c1/console.log).
- Cause of the non-finite value: **hypothesis**, not yet diagnosed.

## Why it did not work
Likely mechanism (hypothesis): an episode that falls early keeps being simulated in the batched
loop, its state diverges, and a later progress increment becomes NaN. Masking by
multiplication (NaN × 0 = NaN) instead of selection would then carry the NaN into the total.
The adaptive-execution evaluator already guards against exactly this, by feeding finished lanes
a safe reset state and using `where` instead of multiplication; the Foresight frontier code
predates that guard. Not caught earlier because Sessions D to G had no early non-finite episode,
or did not validate progress for finiteness.

## Impact
- No hold-ablation result (E18b) and no corrected depth-loop rerun (E17c).
- The AA228 paper does not depend on these; its depth-loop numbers come from E17/E17b.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Port the adaptive evaluator's finite-lane guard to the Foresight frontier and depth-loop rollouts | Removes the class of failure | Touches shared rollout code; needs a regression check against earlier sessions |
| Exclude the scenario | Fast | Hides a failure; not acceptable under the evidence rules |
| Drop the hold ablation | Saves GPU time | Loses a review-requested check |

## Decision and next steps
1. Diagnose on CPU: replay the failing scenario ID and find the first non-finite step and field.
2. If it matches the hypothesis, port the finite-lane guard, re-run the Sessions E–G CPU
   regression checks, then re-run only the hold gate.
3. Whether to spend GPU time on E18b at all is Prasuna's call: it does not change the AA228
   paper.

**Acceptance test:** the hold gate passes validation with all scenarios retained and the failure
recorded as a failure, not a NaN.

## Resolution
Open.
