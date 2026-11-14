# E6b response-model audit for Session F

## Decision

E6b supports an exploratory model of episode failure under its fixed oracle switching protocol, conditional on momentum, speed, and nominal switch lead in the observed 0.1–1.0 s range. It does not validate an online Session F policy, late interventions at lead 0 or 0.05, or transfer to detection-timeout or forecast-deadline release. Use all scheduled projectile scenarios when estimating total action benefit; contact-only selection drops an action-dependent consequence of bracing.

## Evidence provenance and support

Pinned repository revision: `58445c74a02172a94d2e2db36e2b1a8449849c9e`. Tree inspection verified `results/E6b_oracle_v2/episodes.csv` blob `820e3dfd0b079c53bd1d8678280bc10bdaab95d5` (2,621,657 bytes) and `summary.json` blob `e50c3a0716ad9949c823c5da9351afba56583535`.

The CSV contains 16,384 rows, representing 512 unique scenarios, each repeated over eight leads, two modes (BRACE and zero-command), and projectile/no-projectile conditions. Each of two batches has 256 unique scenarios. The generator tiles scenario masses, speeds, nominal hit times, and reset keys over leads and replays a CONTINUE reference trajectory across conditions. The tuple `(batch, mass, speed, t_hit_nominal)` uniquely reconstructs all 512 groups in this CSV, with exactly 32 rows each. Future outputs should include an explicit scenario ID and reset seed. The launch script uses two 256-scenario batches and default seed 100; the first row's sampled mass, speed, and hit time reproduce from NumPy RandomState(100).

Observed support: mass 4.0143–15.987 kg, speed 3.0022–7.992 m/s, momentum 12.659–123.06 kg m/s, nominal hit time 1.5141–3.499 s. Scheduled leads are -1 (CONTINUE), 0.1, 0.2, 0.3, 0.5, 0.75, 1, and 9. Lead 9 is an episode-start intervention, not a genuine nine-second lead; exclude it from a continuous lead regression. There is no lead 0 or 0.05. All scenarios have `is_miss=0`; no-launch controls are absent projectiles, not physical near misses. The generator's environment defaults to a 2–4 m range, whereas the current frontier evaluator explicitly uses 3–5 m. E6b's summary records checkpoints, hold, shared-trajectory flag, and counts, but lacks a complete environment/configuration manifest.

| Nominal lead, s | Scheduled n | Failures | Contacts | Scheduled failure rate | Contact-conditioned failure rate |
|---|---:|---:|---:|---:|---:|
| CONTINUE | 512 | 359 | 511 | 70.12% | 70.25% |
| 0.1 | 512 | 325 | 510 | 63.48% | 63.73% |
| 0.2 | 512 | 306 | 507 | 59.77% | 60.36% |
| 0.3 | 512 | 253 | 503 | 49.41% | 50.30% |
| 0.5 | 512 | 211 | 503 | 41.21% | 41.95% |
| 0.75 | 512 | 97 | 412 | 18.95% | 23.54% |
| 1.0 | 512 | 49 | 205 | 9.57% | 23.90% |
| Episode start | 512 | 11 | 69 | 2.15% | 15.94% |

These rows use the BRACE mode family. The two nominally equivalent CONTINUE mode copies disagree on 4/512 failure outcomes; use one fixed baseline and investigate repeatability before interpreting very small effects. All no-launch rows have zero observed failures. Two non-falling contact cases at lead 1 are censored under the CSV's 2.5 s post-contact window, plus four at episode-start lead 9. The exploratory fit below targets failure within the recorded six-second episode, not guaranteed complete 2.5 s post-contact survival.

## Selection, timing, and release limitations

`fit_response_model.py` selects `launch=1`, `contact=1`, BRACE-mode rows at CONTINUE and lead 0.5. That produces 511 and 503 rows, matching the existing JSON. At lead 1, only 205/512 still contact. Conditioning on observed contact changes the scenario population after treatment and excludes avoidance created by earlier switching. A total-benefit model should retain all scheduled scenarios. A separate contact-conditioned robustness analysis is legitimate only with that narrower interpretation; a policy using it must also model action-dependent contact probabilities.

The schedule is based on an absolute nominal arrival measured from episode start: switch when episode time reaches nominal arrival minus lead, release at nominal arrival plus 0.5 s. Online belief TTC is time remaining at the decision point. These variables cannot be substituted without explicit conversion. The contact event also occurs earlier or later than nominal arrival, and the intervention can alter it. At nominal lead 0.1, the median reconstructed time from switch to the logged contact-step start is 0.04 s. Physical contact falls within that 20 ms physics step. In 20/510 contacting cases the contact step precedes reconstructed onset; 57/510 are at or before it. This is an approximate diagnostic because the old CSV omits `brace_onset_step` and stores nominal times to five significant figures. Never label it exact physical lead.

Do not train on actual post-treatment contact time, pre-contact posture, pre-contact projectile velocity, or readiness as if they were available online before choosing the action. They are outcomes or downstream measurements. A deployable model needs input definitions tied to the same decision-time observations the controller receives.

E6b's ground-truth nominal-arrival-plus-0.5 s release differs from Session E's detection-timeout retention and the current estimated-arrival deadline, which also imposes minimum commitment and a duration cap. A response surface estimated from E6b is specific to its release rule. Performance under deadline error or stale detections needs new matched data or an explicit duration/release-conditioned response model.

## Exploratory CPU-only grouped prediction check

The reproducible script `audit_response.py` reads `e6b_response_subset.csv` and writes `exploratory_grouped_fit.json`. It uses no simulator or GPU. Select BRACE mode, launch=1, all scheduled scenarios at leads 0.1–1.0. Fit on batch 0 and evaluate batch 1, then reverse. All rows for each scenario stay in one fold: six BRACE leads plus a separately analyzed CONTINUE row. There are 3,072 BRACE observations over only 512 independent scenarios. Standardization is trained inside each fold. Logistic fits use a small fixed 1e-4 L2 penalty on standardized non-intercept coefficients. No held-out policy evaluation data is used.

Run `python audit_response.py` from this directory with Python, NumPy, and SciPy available; `--output PATH` optionally selects another JSON destination. Before fitting, the script verifies the subset SHA-256 against `source_manifest.json`, positive finite momentum and speed, binary outcomes, exactly 512 scenario IDs appearing once at each of the seven retained leads, and isolation of 256 scenarios in each batch. The JSON records the runtime versions and validation results. The manifest records the exact extraction/serialization procedure and source provenance: both fetched source payloads independently reproduced the Git blob SHA-1 recorded in the pinned commit tree. Runtime reproduction is offline and verifies the committed subset; it does not redownload the source blobs.

| BRACE predictors | Out-of-fold Brier | Out-of-fold log loss |
|---|---:|---:|
| log J | 0.176126 | 0.523354 |
| log J, log speed | 0.169637 | 0.507032 |
| log J, lead, lead², log J × lead | 0.124299 | 0.389174 |
| Above plus log speed and log speed × lead | 0.113999 | 0.362951 |

Both fold directions favor adding lead and speed. Separately, CONTINUE's 512 observations give Brier/log loss 0.093516/0.302089 for log J, versus 0.089016/0.283041 after adding log speed. This supports using lead and speed as candidate predictors in this data-generating regime. It is exploratory evidence; it does not establish a better policy, calibrated extrapolation, or sensitivity to release error. Pooling across leads without including lead deliberately gives an information-poor reference model. The lead polynomial is one compact candidate, not a selected final architecture. After observing these results, E6b is development data, not a clean confirmatory test set.

## Required Session F data and evaluation contract

1. Freeze the actual release protocol before response data collection. Record entry, nominal/estimated/actual contact timing, release and reason, hold duration, censor status, and full configuration/checkpoint/source hashes. If testing release-error robustness, vary or log that error explicitly.
2. Add nominal leads 0 and 0.05, plus late/negative opportunities if the controller may act after nominal arrival. Collect enough independent scenarios around the transition where useful preparation becomes ineffective; the current grid cannot resolve it.
3. Pair all candidate interventions and CONTINUE on fixed scenario/reset IDs and shared world trajectories. Keep every scenario, including contact avoidance, falls before launch, and misses. Add physical near misses appropriate to the deployment geometry.
4. Match E6b and frontier sampling geometry or measure the shift explicitly. Save launch azimuth, distance, height, velocity, and robot state available at decision time; the existing CSV cannot stratify these missing inputs.
5. Split by scenario/reset seed before model or threshold selection. Use independent development, calibration/policy-tuning, and final-policy-test scenario sets. If training/checkpoint generalization is claimed, hold those out separately. Bootstrap or form uncertainty estimates over scenarios, never over the 3,072 repeated rows as independent draws.
6. Give each contact sufficient observation horizon or use a prespecified censor-aware endpoint. Compare the frozen new policy with the legacy and fixed-threshold baselines on identical held-out scenarios, detector draws, and release rules. Only then report failure/preparation tradeoffs.

Source URLs: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E6b_oracle_v2/summary.json ; https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/scripts/eval_oracle_brace.py ; https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/scripts/fit_response_model.py ; https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/foresight/eval/rollout.py ; https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/scripts/run_session_b.sh .
