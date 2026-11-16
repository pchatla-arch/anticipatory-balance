# Online Actuator Adaptation 0.3: execution plan

Agreed November 15, 2026. Owner and decision-maker: Prasuna Chatla. Mirror of the Claude Docs
plan; this file is the copy both assistants work from. Changes to it land by PR.

Proposal 0.3 ([proposal.md](research_candidate/proposal.md)) is executed in five milestones
over four weeks. Each assistant owns separate modules and reviews the other's, and neither
builds the baseline its own method is judged against. Prasuna decides at every gate.

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

| Milestone | Dates | Gate at the end |
| --- | --- | --- |
| M0 Compensability | Nov 16–18 | G0, Nov 18 |
| M1 Identification and authority | Nov 17–21 | G1, Nov 21 |
| M2 Method versus heuristic | Nov 21–27 | G2, Nov 27 |
| M3 Learning and seeds | Nov 24–Dec 5 | G3, Dec 5 |
| M4 Runtime and demonstration | Nov 24–Dec 13 | G4, Dec 13 |
| F1 Foresight paper (parallel) | Nov 16–Dec 4 | Submitted Dec 4 |

M0 and M1 run on CPU, except the Stage A pod run. GPU time is concentrated in M2 (validation
and one registered test) and M3 (robust PPO, extra walkers).

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
| F1 | Foresight AA228 paper: review corrections and Session G numbers | Claude | ChatGPT audits | — | Submitted Dec 4 |

## Gate decisions

Prasuna decides each gate from a one-page memo with links to result files. "Modify" means a
recorded protocol change, never a silent retry.

| Gate | Date | Question | Go | Modify | Stop or fall back |
| --- | --- | --- | --- | --- | --- |
| G0 Compensability | Nov 18 | Does a gain loss degrade walking, and does the true-gain correction recover it? | Published Stage A pass on fresh episodes | Degradation too small or all-or-nothing: shift the gain range or joint set | Correction cannot recover: diagnose authority; no estimator work |
| G1 Authority | Nov 21 | With realistic telemetry, does the gain set narrow fast enough to leave useful correction after a fault? | Meaningful permitted correction within a stated time on hip and knee | Too conservative: declare a relaxed variant as a separate method | No authority under any declared telemetry: report it, drop the research claim, keep the engineering track |
| G2 Comparative | Nov 27 | On validation, does the candidate beat the tuned heuristic on the recovery-versus-harm tradeoff, with healthy noninferiority? | Freeze operating points; run the registered test once | Mixed result: register a narrower claim before testing | Heuristic equals candidate: engineering-component outcome, no method claim |
| G3 Claim | Dec 5 | Does the registered test support the claim, and do the learned and robust-PPO comparisons change it? | Draft the paper around the measured claim | Claim holds for a subset: narrow the title and scope | Report a negative result honestly |
| G4 Release | Dec 13 | Is the evidence package complete: seeds, held-out instances, C++ parity and timing, videos? | Technical report + public release; decide on a workshop submission | Missing pieces: list them, set a date | — |

Kill criteria hold throughout: no GPU run without a CPU dry run; no retuning on the test set;
no result enters a document until source, manifest and summary agree.

## Inputs already available

- Healthy-walking authority check: the 20 ms hold envelope permits nonzero correction on 82% of
  leg-joint samples with a measured-speed envelope, 63% with a global speed bound (hip 86/84%,
  knee 80/61%). Script `docs/adaptive_execution/reviews/check_authority.py` on branch `session-d`.
- Review of draft 0.2 with verified telemetry defects: PR #8, issues #9–#15.

## Next 72 hours

- [ ] Prasuna: confirm 0.3 is selected; run the Stage A gate (WP 0.1) with `start_adaptive_gate.sh`.
- [ ] ChatGPT: respond to PR #8 (R1–R7); fix synchronized telemetry (WP 0.2, AE-001/002); start
  the JAX gain set and projection (WP 1.2); amend the 0.3 confound list per AE-003.
- [ ] Claude: formal review of 0.3 (WP R0); telemetry models (WP 1.1) and confound scenarios
  (WP 1.4), CPU-tested; evaluation harness skeleton with instance splits and the registration
  file (WP 2.3); Foresight paper corrections (F1), first pass.
- [ ] Both: review each other's PRs within a day of opening.
