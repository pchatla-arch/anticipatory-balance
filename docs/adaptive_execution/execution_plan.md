# Online Actuator Adaptation 0.3: execution plan

Agreed November 15, 2026; re-timed November 21, 2026 (revision 2, see below). Owner and
decision-maker: Prasuna Chatla. Mirror of the Claude Docs plan; this file is the copy both
assistants work from. Changes to it land by PR.

Proposal 0.3 ([proposal.md](research_candidate/proposal.md)) is executed in five milestones.
Each assistant owns separate modules and reviews the other's, and neither builds the baseline
its own method is judged against. Prasuna decides at every gate.

## Revision 2 (November 21): the 0.3 paper is the AA228 course paper, due December 4

Prasuna's decision: Online Actuator Adaptation 0.3 is submitted as her AA228/CS238 final
project paper (4–6 pages, conference style, Gradescope, Friday December 4, 5 p.m. Pacific).
Foresight is no longer the course deliverable; its revised manuscript (PR #25) is kept as the
fallback. **Full scope is kept**; nothing from the 0.3 method set, baselines or learning
comparisons is cut. Prasuna runs additional pod sessions to make the dates.

What changes:

- **Dates compress.** G0–G3 move inside the window ending December 2 (table below). M4
  (C++ runtime, videos, release) keeps its December 13 date; it feeds the portfolio release,
  not the course paper.
- **The paper is written as decision making under uncertainty**, which is what the course
  grades: the actuator gain is a hidden state; timestamped effort and encoder telemetry are
  delayed, noisy observations; the set-membership gain interval (with its hold expansion) is
  the belief; the correction is an action chosen under that belief subject to hard
  constraints; the true-gain oracle is the fully observed policy, the heuristic gate is the
  certainty-equivalent policy on a point estimate, and actuator-randomized PPO is the
  "optimize over the prior, infer nothing online" policy. Course rubric: problem, approach
  and its justification, quantitative analysis with clear figures, limitations, references.
- **Authorship under the course AI policy.** Method choices, evaluation choices, conclusions
  and all prose are Prasuna's. Both assistants supply code, analyses, figures and skeletons,
  and are named with their roles in the end-of-paper disclosure. Numbers in the paper are
  copied from result files and checked by the other assistant, as before.
- **Course staff notice.** Prasuna informs the course staff that the topic changed after the
  status update (allowed by the project rules) and states the relationship in the paper.
- **Fallback.** If the registered comparison (WP 2.4) has not produced a test report by
  November 30, the course submission is the Foresight manuscript from PR #25; the 0.3 work
  continues for the portfolio. Prasuna decides on November 30.

GPU sessions (evenings): S1 Nov 22 Stage A smoke + gate (2 h); S2 Nov 24–25 authority study and
telemetry traces (1 h); S3 Nov 26–27 validation runs and robust-PPO training (3–4 h);
S4 Nov 28–29 registered test, GRU evaluation, extra walkers (2–3 h); S5 Dec 1 videos and any
rerun (1–2 h). About 10–12 GPU-hours.

## How we collaborate

| Role | Owns | Decides |
| --- | --- | --- |
| **Prasuna** | Pods, merges, sign-off on gate memos | Every gate (go, modify or stop); scope changes; what is claimed |
| **ChatGPT** | The proposed method (set-membership gain set, hold envelope, projection); the Stage A gate it built; the telemetry adapter; actuator-randomized PPO; the 0.3 paper text | Nothing alone; proposes |
| **Claude** | The competitors and the test bench: telemetry noise/bias/delay models, confound scenarios, the tuned heuristic gate, the evaluation and statistics harness, the work-cycle commands, the learned GRU estimator, the C++ port, videos | Nothing alone; proposes |

Rules:

- **No one builds the baseline their own method is judged against.** ChatGPT designed the
  projection, so Claude builds and tunes the heuristic gate it must beat, with the same tuning
  budget. ChatGPT reviews that the heuristic is strong, not a strawman.
- **One branch per owner per work package** (`oaa/<WP-id>-<name>`). The integration branch is
  `portfolio/adaptive-execution-gate`. Every change lands by PR, is reviewed by the other
  assistant, and is merged by Prasuna.
- **Every PR** cites its work-package ID and any issue IDs (AE-nnn), and states its claim, its
  test, and what it does not show. The reviewer reruns the test.
- **Every problem** gets a technical note and a GitHub Issue (`docs/technical_notes/issues/`),
  opened before the fix.
- **No GPU run without** a CPU dry run of the exact entry point, a written manifest and an
  attempt budget. Results are published before the pod stops.
- **Numbers in any document** are copied from result files and checked by the other assistant.
- **Gate memos** (one page) are written by the work-package owner and reviewed by the other
  assistant. Prasuna decides from the memo.

## Roadmap

| Milestone | Dates (rev. 2) | Gate at the end | Original |
| --- | --- | --- | --- |
| M0 Compensability | Nov 16–22 | G0, Nov 22 | G0 Nov 18 |
| M1 Identification and authority | Nov 22–25 | G1, Nov 25 | G1 Nov 21 |
| M2 Method versus heuristic | Nov 25–29 | G2, Nov 29 | G2 Nov 27 |
| M3 Learning and seeds | Nov 26–Dec 2 | G3, Dec 2 | G3 Dec 5 |
| P1 AA228 paper (0.3) | Nov 22–Dec 4 | Submitted Dec 4 | was F1 Foresight |
| M4 Runtime and demonstration | Dec 1–13 | G4, Dec 13 | unchanged |

M0 and M1 run on CPU, except the Stage A pod run. GPU time is concentrated in M2 (validation
and one registered test) and M3 (robust PPO, extra walkers). The MuJoCo-Warp backend now also
executes on CPU (slowly), so every pod entry point is dry-run on the pod's own backend first.

## Work packages

| WP | Work | Owner | Reviewer | Needs | Done when |
| --- | --- | --- | --- | --- | --- |
| R0 | Formal review of proposal 0.3 per `research_candidate/claude_review_request.md` | Claude | ChatGPT responds | — | Review PR with findings and smallest checks |
| 0.1 | Run the existing Stage A gate (known-gain correction); GPU, ≤ 2 h | ChatGPT code; Prasuna runs | Claude audits results | — | Gate report + manifest published |
| 0.2 | Synchronized (position, applied torque) telemetry; saturation flag (AE-001, AE-002) | ChatGPT | Claude | — | Healthy robot: ĝ = 1 within 1e-4; saturated samples flagged |
| 1.1 | Telemetry models: noise, bias, scale, delay, reordering, dropout | Claude | ChatGPT | 0.2 | Declared configs + unit tests |
| 1.2 | Gain set, hold envelope and projection in JAX, batched | ChatGPT | Claude | 0.2 | Matches the scalar reference on all its cases |
| 1.3 | Authority study: gain-set width and permitted correction over time after a fault, per telemetry condition | Claude | ChatGPT | 1.1, 1.2 | Curves per joint group; feeds G1 |
| 1.4 | Confound and return-to-healthy scenarios: friction, sensor scale and bias, saturation, staleness (AE-003) | Claude | ChatGPT | 1.1 | Each scenario can produce false adaptation in a plain estimator |
| 2.1 | Tuned heuristic gate: same estimator, thresholds tuned on validation, matched budget | Claude | ChatGPT checks strength | 1.1 | Tuning log + frozen thresholds |
| 2.2 | Candidate integrated in the closed loop, with reason codes | ChatGPT | Claude | 1.2 | Identity test passes; reason codes logged |
| 2.3 | Evaluation harness: instance splits, pairing, registration file, fixed-horizon loss, paired bootstrap, noninferiority margins | Claude | ChatGPT | — | Dry run on CPU end to end |
| 2.4 | Validation runs, freeze operating points, one registered test run; GPU | Prasuna runs | Both analyze | 2.1–2.3 | Test report; feeds G2 |
| 3.1 | GRU history estimator feeding the same corrector; also the no-effort condition (AE-004) | Claude | ChatGPT | 2.2 | Same-corrector comparison |
| 3.2 | Actuator-randomized PPO with within-episode changes, budget recorded; GPU | ChatGPT | Claude | 2.3 | Trained policy + learning curves |
| 3.3 | Matched-recipe walkers s2/s3 (the s3 checkpoint was never committed); held-out instances; GPU | ChatGPT | Claude | 2.3 | Three policies evaluated |
| 3.4 | Residual PPO, only for a recorded limitation | ChatGPT | Claude | G2 | Same total authority |
| 4.1 | Walk–turn–stop–resume command schedule (no 500-step resampling); healthy competence first | Claude | ChatGPT | — | Healthy completion rate measured |
| 4.2 | C++ estimator, gain set and projection; replay parity; p50/p95/p99 timing | Claude | ChatGPT | 2.2 | Parity within tolerance; timing table |
| 4.3 | Videos by a declared selection rule: healthy, degraded, candidate, heuristic, one failure | Claude | ChatGPT | 2.4 | Videos + rule |
| 4.4 | Paper 0.3 results and figures; technical report | ChatGPT | Claude checks numbers | G3 | Report draft |
| P1 | AA228 paper: skeleton with the decision-under-uncertainty framing, figure/table plan mapped to the rubric; Prasuna writes the prose | Claude skeleton; Prasuna text | ChatGPT checks numbers and claims | G2 for results | PDF submitted Dec 4 |
| F1 | Foresight manuscript (PR #25): fallback only, kept warm | ChatGPT | Claude | — | Decision Nov 30 |

## Gate decisions

Prasuna decides each gate from a one-page memo with links to result files. "Modify" means a
recorded protocol change, never a silent retry.

| Gate | Date | Question | Go | Modify | Stop or fall back |
| --- | --- | --- | --- | --- | --- |
| G0 Compensability | Nov 22 | Does a gain loss degrade walking, and does the true-gain correction recover it? | Published Stage A pass on fresh episodes | Degradation too small or all-or-nothing: shift the gain range or joint set | Correction cannot recover: diagnose authority; no estimator work |
| G1 Authority | Nov 25 | With realistic telemetry, does the gain set narrow fast enough to leave useful correction after a fault? | Meaningful permitted correction within a stated time on hip and knee | Too conservative: declare a relaxed variant as a separate method | No authority under any declared telemetry: report it, drop the research claim, keep the engineering track |
| G2 Comparative | Nov 29 | On validation, does the candidate beat the tuned heuristic on the recovery-versus-harm tradeoff, with healthy noninferiority? | Freeze operating points; run the registered test once | Mixed result: register a narrower claim before testing | Heuristic equals candidate: engineering-component outcome, no method claim |
| G3 Claim | Dec 2 | Does the registered test support the claim, and do the learned and robust-PPO comparisons change it? | Draft the paper around the measured claim | Claim holds for a subset: narrow the title and scope | Report a negative result honestly |
| G4 Release | Dec 13 | Is the evidence package complete: seeds, held-out instances, C++ parity and timing, videos? | Technical report + public release; decide on a workshop submission | Missing pieces: list them, set a date | — |

Kill criteria hold throughout: no GPU run without a CPU dry run; no retuning on the test set;
no result enters a document until source, manifest and summary agree.

## Inputs already available

- Healthy-walking authority check: the 20 ms hold envelope permits nonzero correction on 82% of
  leg-joint samples with a measured-speed envelope, 63% with a global speed bound (hip 86/84%,
  knee 80/61%). Script `docs/adaptive_execution/reviews/check_authority.py` on branch `session-d`.
- Review of draft 0.2 with verified telemetry defects: PR #8, issues #9–#15.

## Next 72 hours (from November 22)

- [ ] Prasuna: run Stage A `smoke`, then `gate`, from `claude/gate-run` (AE-007 and AE-008
  fixes); merge PRs #20, #23 and #26 after review; notify the course staff of the topic change.
- [ ] ChatGPT: gain set, hold envelope and projection in JAX, batched (WP 1.2), consuming the
  shared RLS point estimate and telemetry packets defined in WP 1.1; integrate the candidate in
  the closed loop with reason codes (WP 2.2); start the actuator-randomized PPO wrapper (WP 3.2)
  so training can run in S3.
- [ ] Claude: telemetry sensor models and in-order delivery queue (WP 1.1); RLS point estimate
  and the tuned heuristic gate (WP 2.1); confound and return-to-healthy scenarios (WP 1.4);
  evaluation harness with splits, registration and paired statistics (WP 2.3), dry-run on
  CPU with both backends; the env extension that both methods need (externally supplied
  correction, per-step telemetry packet); paper skeleton (P1); formal review of 0.3 (R0).
- [ ] Both: review each other's PRs within a day of opening.

## Shared interfaces (proposed November 21, for review)

So that WP 1.1/2.1/2.3 (Claude) and WP 1.2/2.2 (ChatGPT) fit without a merge week:

- **Env extension** (`foresight/adaptive/actuator_env.py`, small PR): `step(state, action)`
  reads an externally supplied per-actuator requested offset from
  `state.info["adaptive_requested_offset"]` and applies only the common limits (control range,
  amplitude, requested slew); the oracle becomes one method among others. With
  `telemetry=True` the step also writes the exact source-aligned packet (WP 0.2 sampler) to
  `info["telemetry_packet"]` using the plant model that generated the step.
- **Sensor model and delivery** (WP 1.1, `foresight/adaptive/sensing.py`): `TelemetryConfig`
  (encoder noise/bias, effort noise/bias/scale, latency range in control steps, dropout,
  declared error bounds `b_q`, `b_tau`) and a fixed-size ring-buffer `DeliveryQueue` that
  releases packets in source order once arrived, skipping dropped packets after the maximum
  latency. Everything batched over environments inside the scan.
- **Point estimate** (WP 2.1, `foresight/adaptive/estimation.py`): per-actuator RLS with
  forgetting on `(phi_i, y_i)` from valid packets, exposing `g_hat`, residual statistics and
  the age of the newest processed packet. The candidate's inverse candidate and the heuristic
  gate consume the same object.
- **Method interface**: `method.init(n, nu) -> state`; `method.update(state, packets_due,
  step) -> state`; `method.correction(state, r_k, q_hat, step) -> (x_k, reason_code)`.
  Methods: `none`, `oracle` (privileged gain), `estimated_inverse`, `heuristic_gate`,
  `candidate` (ChatGPT), later `gru_inverse`.
- **Harness** (WP 2.3, `scripts/eval_adaptive_bench.py`): scenario families × telemetry
  configs × methods on paired instances with dev/val/test seed splits, a registration JSON
  written before any test run, per-episode rows, fixed-horizon loss, paired bootstrap,
  noninferiority margins, false-adaptation and withdrawal-time metrics.
