# Online Actuator Adaptation: execution plan

**Document ID:** OAA-PLAN
**Version:** 0.1.0
**Date:** November 15, 2026
**Status:** Proposed amendment - Prasuna approval pending
**Owner:** ChatGPT, as plan integrator
**Cross-reviewer:** Claude
**Gatekeeper:** Prasuna Chatla
**Reviewed integration baseline:** f5970231bed0e1be1eadc951137c2515a14efeb0

## Purpose and current position

This is the first explicitly versioned execution plan. It amends the unversioned plan introduced at b9103ec and merged by Prasuna through PR #16. Proposal family 0.3 is selected; its published technical revision is 0.3.1 at 1614b65. GitHub is the canonical shared record. Chat copies and external document mirrors point to a revision rather than becoming competing sources.

The architecture remains a frozen G1 PPO walker, causal actuator estimation, bounded position-target correction and execution supervision. The primary research comparison is the proposed uncertainty/hold-dependent projection against an independently implemented, tuned heuristic using the same information and common physical limits. Useful engineering work remains valuable if the method claim fails. A new theme or relaxed method requires a recorded decision.

No new GPU experiment or scientific success is asserted by this plan. Current evidence is:

| Item | Verified status | Next requirement |
|---|---|---|
| Project selection | Prasuna selected proposal 0.3 | Preserve scope; record changes explicitly |
| Technical proposal | Revision 0.3.1 published with telemetry corrections | Claude reviews that exact revision |
| Telemetry source, WP 0.2 | Ten CPU tests and native interface check on all 29 G1 actuators passed | Cross-review, recorder integration and Warp parity |
| True-gain feasibility gate | Existing launcher and CPU checks available | Prasuna runs; G0 report remains pending |
| Scalar correction reference | Ten finite cases, two invalid-input checks and 71 sweep points passed | Full causal estimator and JAX integration |
| Healthy authority screen | Historical script located; forwarded percentages not independently reproduced | Result manifest, raw trace and calibrated envelope study |
| Online recovery, learned comparisons, runtime | Pending | Staged evidence below |

Source and result identities are indexed in document_versions.json. Selection, implementation, validation, independent review and gate approval are separate states.

## Roles, ownership and handoffs

| Participant | Primary responsibility | Decision authority |
|---|---|---|
| Prasuna | Project ownership, pod operation, merge and gate review | Approves merges, paid expenditure, protocol changes, scope and public claims |
| ChatGPT | Telemetry source, shared estimator/correction interfaces, proposed method, candidate integration, robust-PPO implementation and technical report | Prepares and tests work; recommends decisions |
| Claude | Sensor corruption/transport, primary heuristic, trace/evaluation/statistics, GRU comparison, work-cycle commands, C++ port, videos and Foresight paper | Prepares and tests work; recommends decisions |

Claude owns and tunes the **primary heuristic comparator** independently of ChatGPT's projection. Secondary comparators, including robust PPO, may be implemented by either assistant with cross-review of information access, action authority, tuning effort, training adequacy and failures. Independent authorship alone does not establish a fair comparison.

Each package has one implementation owner and one reviewer. Use owner branches such as oaa/wp12-chatgpt-estimation and oaa/wp11-claude-sensors. List owned paths in the PR before parallel edits. A shared-interface change is proposed explicitly; neither assistant overwrites the other's in-progress module.

The integration branch is portfolio/adaptive-execution-gate. Owners publish their work in PRs; the other assistant reviews; Prasuna merges. Routine reversible edits and CPU checks proceed within the assigned task. Existing authorization remains valid within its stated scope and budget. A published draft is not a gate approval.

There is no direct assistant-to-assistant session channel. Prasuna passes the immutable PR or handoff link. Every handoff states the package, base/source commit, owned files, interface, checks and evidence, limitations, unresolved dependencies and specific next request. Internal helper-agent checks are not attributed to the other assistant.

Reviewers reproduce relevant inexpensive CPU checks and audit immutable GPU evidence. A paid rerun must resolve a concrete uncertainty and fit an existing authorization or a new Prasuna decision. Substantive defects or protocol disputes receive an Issue and technical note. Typographic and layout fixes can stay in the PR and changelog.

## Document control

The current register is document_versions.json; detailed release explanations are in CHANGELOG.md; actual gatekeeper decisions are in decision_log.md. These records are maintained in the same PR as the affected work. Historical source/PDF copies use full commit permalinks, avoiding ambiguous files named “final”.

Each controlled document has its own ID and version. OAA-PROP remains 0.3.1; OAA-PLAN starts at 0.1.0; OAA-DESIGN 0.2 remains historical support. Document versions are distinct from software commits, checkpoint versions and experiment IDs.

Patch revisions clarify wording, formatting or status without changing the experiment contract. Minor revisions add substantive bounded protocol or interface content. Major revisions change the central question or incompatible architecture. Changes to metrics, eligibility, splits, budgets or claims always need a protocol revision and gatekeeper decision regardless of version size.

Every revision records its reason, affected sections/packages, concrete changes, experimental impact, checks, unresolved limitations, owner/reviewer and decision status. Append corrections to history; do not rewrite earlier evidence as if it were known then. Before merging, run scripts/check_adaptive_documents.py and compare against the prior registry. The check verifies bytes and declared provenance, not the truth of scientific claims or authenticity of approval.

## Roadmap and dependencies

Dates are targets, not automatic permission or guaranteed completion times. Scope advances when its evidence gate is decided.

| Milestone | Target window | Exit |
|---|---|---|
| M0 Compensability | Nov 16-18 | G0: behavioral true-gain gate |
| M1 Identification and authority | Nov 17-21 | G1: supported information/authority envelope |
| M2 Candidate versus heuristic | Nov 21-27 for validation | G2 approves frozen test; held-out execution follows |
| M3 Learning and generalization | Preparation Nov 24; evidence target Dec 5 | G3: predeclared claims and comparison assessment |
| M4 Runtime and demonstration | Preparation Nov 24; target Dec 13 | G4: complete release evidence |
| F1 Foresight course paper | Nov 16-Dec 4, separate workstream | Prasuna's course submission |

CPU sensor models, synthetic replay and estimator development can proceed while M0 runs. Full G1/Warp parity, closed-loop authority and GPU overhead require their own integration evidence. Expensive training follows the relevant gate; a date alone does not trigger a pod run.

The critical path is: aligned telemetry and packet contract; shared estimator and sensor models; trace/replay authority evidence; candidate and heuristic integration; validation; G2 protocol freeze; held-out test; G3 claim assessment. C++ harness scaffolding can start early, but port acceptance follows a stable algorithm/interface.

## Work packages

In the tables, C means Claude and G means ChatGPT. Prasuna operates paid runs and decides acceptance; an owner's implementation-complete status does not substitute for that decision.

### M0: interfaces and compensability

| Package; owner/reviewer | Deliverable | Dependencies | Acceptance and current status |
|---|---|---|---|
| R0; C/G | Formal review of proposal 0.3.1 and this plan | Published revisions | Findings linked to exact source; full current-revision review pending |
| 0.1; G prepares, Prasuna runs, C audits | Existing true-gain gate | Pinned checkpoint, dispatch checks and authorized attempt | 32 screening scenarios; 128 fresh confirmation if eligible; complete report and manifest; GPU result pending |
| 0.2; G/C | Aligned applied-torque source and clipping flags, AE-001/002 | Declared actuator mapping | CPU implementation and native G1 interface check complete at 1614b65; peer review/integration pending |
| DOC; G/C | Versioned plan, registry, detailed history and decision ledger | PR #16 and proposal 0.3.1 | Consistency check and artifact review; this amendment awaits Prasuna |

Stage A consumes true gain directly, not an effort estimate. Its current raw-force diagnostic is not a joint-torque compliance result. The telemetry helper is opt-in and does not alter that gate.

### M1: identification and usable authority

| Package; owner/reviewer | Deliverable | Dependencies | Acceptance |
|---|---|---|---|
| 1.0; C/G | Trace schema, generator/recorder and replay manifest | 0.2 packet contract | Source/arrival times, issued target, state, effort, validity masks and hashes preserved; synthetic versus faulted trajectory identified |
| 1.1; C/G | Noise, bias, scale, delay, reordering and dropout models | 0.2 | Declared sensor equations/bounds and causal unit checks; corruption follows valid source sampling |
| 1.2a; G/C | Shared causal RLS point estimator, inverse candidate and reset semantics | 0.2; agreed 1.1 interface | Chronological history, weak excitation, invalid samples and resets checked; common estimator fixed across primary methods |
| 1.2b; G/C | Gain-set update, hold envelope and projection in batched JAX | 1.2a; 1.1 schema | Scalar parity plus age/drift propagation, empty sets, target clipping, force/slew conflicts and requested/effective distinction |
| 1.3; C/G | Gain-range and authority replay, then integration pilot | 1.0/1.1; scalar reference first, 1.2 for integrated comparison | Coverage, contraction delay, sign screen, full feasible interval and magnitude per joint group; feeds G1 |
| 1.4; C/G | Measurement-specific nuisances and return-to-healthy cases, AE-003 | 1.1 | Intervention point and expected validity behavior declared; false adaptation measured, not required |

WP 1.3 may start with the existing scalar reference and a declared synthetic trace before the complete JAX implementation. It must not present synthetic force on a healthy trajectory as faulted closed-loop behavior. Stage A CSV files contain episode aggregates and cannot supply the missing per-step packets.

For direct applied-actuator effort, primary nuisances are sensor scale, bias, timing and saturation handling. Payload/contact/passive-friction changes affect motion and excitation without automatically corrupting that force relation. A downstream or reconstructed effort channel needs an explicit loss model before friction is called a direct sensor confound. Zero false adaptation can be the correct negative-control result.

### M2: fair comparison and registered test

| Package; owner/reviewer | Deliverable | Dependencies | Acceptance |
|---|---|---|---|
| 2.1; C/G | Strong validation-tuned heuristic | 1.1, 1.2a, shared limits from 1.2b, 2.3 | Same permitted inputs, point estimator and common limits; credible tuning log and frozen thresholds |
| 2.2; G/C | Candidate closed-loop integration with reason codes | 0.2, 1.1, 1.2a/b, G0 | Healthy identity, causal timing and unsupported-state handling; backend smoke before scale |
| 2.3; C/G | Evaluation/statistics and registration harness | Shared interfaces; design can start now | Instance splits, pairing, retained failures, fixed-horizon loss, uncertainty and registration checks |
| 2.4; Prasuna runs, G/C analyze | Validation and freeze package | G1, 2.1-2.3 | Select operating points on validation only; prepare G2 memo; no held-out test yet |
| 2.5; Prasuna runs, C harness, G audits | Registered held-out evaluation | Explicit G2 approval | Exact approved manifests, complete attempts/failures and predeclared analysis; feeds G3 |

Common method limits include amplitude, requested slew, physical target range and supported force-headroom checks. The heuristic does not inherit the proposed uncertainty/hold-dependent force-error filter. Consider a smaller fixed correction cap to distinguish selective intervention from simply acting less.

The freeze package names the primary estimand, test-mixture weights, practical effect threshold, healthy noninferiority margins, sample size/uncertainty plan, seeds/instances, checkpoints, code/config hashes and infrastructure-rerun rule. The held-out set is opened only after G2. An infrastructure retry preserves the failed attempt and follows the registered rule; it is not another tuning opportunity.

### M3 and M4: learning, reuse and runtime

| Package; owner/reviewer | Deliverable | Dependencies | Acceptance |
|---|---|---|---|
| 3.1a; C/G | GRU point estimator with the same effort/history and corrector | Stable 1.2 and 2.2 | Same physical set/limits; report whether changed estimates change issued commands |
| 3.1b; C/G; optional | No-effort inference study, AE-004 | Separate measurement/uncertainty contract and Prasuna decision | State what can narrow the admissible set; unavailable identification is a valid outcome |
| 3.2; G/C | Actuator-randomized PPO, including within-episode shifts | 2.3; approved training budget | Credible action/history interface, tuning budget, learning curves and durable checkpoint |
| 3.3; G/C | Additional matched-recipe policies and held-out instances | 2.3; checkpoint audit and approved budget | Reuse compatible s2 if justified; verify whether s3 exists; distinguish episode and training-seed uncertainty |
| 3.4; G/C; optional | Residual PPO | Recorded classical-model limitation and explicit decision | Same total authority/information; separately budgeted comparison |
| 4.1; C/G | Walk-turn-stop-resume command schedule | Can prepare now | Healthy competence per phase first; no claim based on an untested work cycle |
| 4.2; C/G | C++ replay port and runtime study | Stable 2.2 schema/algorithm | Frozen traces; numerical/action parity tolerances; adapter-only and policy-plus-adapter timing separated |
| 4.3; C/G | Videos chosen by a declared rule | 2.5 or labeled validation | Healthy, degraded, candidate, heuristic and failure; source episode IDs retained |
| 4.4; G/C | Technical report and evidence release | G3; G4 completes package | Claims map to registered outcomes and immutable artifacts |
| F1; C/G | Foresight paper corrections and audited Session G results | Separate repository evidence | Prasuna reviews and submits by Dec 4 |

A no-effort GRU is not automatically the same-corrector experiment: if the physical gain set still includes one, the proposed filter permits zero effective correction. A learned point estimate cannot silently narrow that set. Learning comparisons use development data without tuning on an opened held-out test.

## Gate decisions and required memos

Each gate has a concise memo from the package owner, a cross-review, and a recorded Prasuna decision: GO, MODIFY or STOP. A failed research comparison does not automatically authorize a new architecture.

| Gate; target | Required evidence | Decision boundary |
|---|---|---|
| G0; Nov 18 | Existing Stage A report on fresh confirmation episodes, validity checks and manifest | Preserve implemented thresholds. Failure pauses costly scaling while targeted CPU diagnosis remains useful |
| G1; Nov 21 | Coverage, contraction delay, attainable magnitude/oracle overlap, withdrawal behavior and per-joint results | Before confirmation, specify useful magnitude/time/coverage targets from development data and task needs; report every leg group |
| G2; Nov 27 target | Validation tradeoff, healthy-regression assessment and complete frozen registration | Prasuna authorizes the held-out run only after code, data splits, operating points, budgets and analysis are fixed |
| G3; Dec 5 target | Predeclared test effects, uncertainty and relevant learning/seed comparisons | Confirm only supported claims; unregistered favorable subsets remain exploratory and need new evidence for confirmation |
| G4; Dec 13 target | Release manifest, report, checkpoint restoration, runtime/parity evidence and demonstration | Prasuna decides release/submission readiness; list incomplete scope explicitly |

G1 quantitative targets and G2 statistical margins are not filled with invented numbers in this plan. Their owner proposes them using development evidence; Prasuna accepts them before the relevant confirmation/test. The existing G0 decision rule remains unchanged.

## Compute, publication and evidence discipline

Before a paid run, validate the exact source revision, configuration and dispatch through applicable CPU logic tests and dry-run checks. The production launcher needs NVIDIA hardware; CPU/JAX execution is not Warp validation. A bounded backend compile/smoke check belongs inside the approved GPU attempt before scaling.

The Stage A launcher has a 120-minute attempt budget including setup, with publication/stop reserve. At the stated rate, two billed hours would be $5; neither completion nor automatic stopping is guaranteed by that budget. Failed upload or unavailable RunPod controls can leave the pod running. Prasuna checks its final state.

Publish compact results and manifests before requesting Stop. Preserve checkpoints in durable storage with verified restoration. Git pushes of reports do not by themselves preserve ignored checkpoint files. Record compile/setup, simulation, training, evaluation and publication time separately. No duplicate paid run is required merely because a different assistant performs the review.

Numbers in documents are either sourced measurements with exact code/checkpoint/config/result identity, labeled analytical calculations, attributed unverified reports, or planning assumptions. Preserve failures, invalid lanes and interrupted attempts. A reported negative result must not become a silently narrowed success claim.

## Authority-screen evidence update

The historical script was located at commit 71204bac4f2c38389f4a85a256409ba3c43959ec, path docs/adaptive_execution/reviews/check_authority.py. Its source is now inspectable despite the absence of the session-d branch. The forwarded 82%/63% values remain unreplicated here, and a result/checkpoint manifest is still needed.

The measured-speed construction includes an illustrative acceleration allowance of 100 rad/s squared. The other construction uses each joint's 99th-percentile speed computed from that same rollout; it is an offline empirical diagnostic, not a guaranteed or causally available global bound. Encoder error is omitted. Both report error-envelope sign preservation, not full feasible authority. Preserve the original check and add a separately versioned, causally calibrated follow-up.

## Next 72 hours and handoff

| Owner | Immediate work | Evidence returned |
|---|---|---|
| Prasuna | Review this amendment, pass the immutable handoff to Claude, and run the authorized Stage A when the pod is ready | Merge/gate decisions and published run ID |
| ChatGPT | Supply WP 0.2 implementation/evidence for review; prepare shared estimator interface and WP 1.2 CPU work | Owner PR with schema, tests, limitations and exact revision |
| Claude | Review proposal 0.3.1 and WP 0.2; implement WP 1.1 and the WP 1.0/1.3 CPU replay; prepare WP 2.3 registration skeleton | Independent review, sensor tests and replay artifacts; keep Foresight edits parallel |
| Both | Resolve interface disagreements in the PR before integration | Reviewed contract and actionable gate memo |

No assistant claims to have contacted the other through a private session. Review turnaround of one day is a target subject to Prasuna's availability to pass the link. Approval of this plan sets responsibilities; each later gate still records its own evidence and decision.

## Source record

- Selected proposal 0.3.1 and telemetry reference: https://github.com/pchatla-arch/anticipatory-balance/commit/1614b65c65ae89090b76741fab8716225b50c1f2
- Original plan: https://github.com/pchatla-arch/anticipatory-balance/blob/b9103ec0a67724f598ccb91bdcd00221464e31fd/docs/adaptive_execution/execution_plan.md
- Prasuna's plan merge: https://github.com/pchatla-arch/anticipatory-balance/pull/16
- Historical authority script: https://github.com/pchatla-arch/anticipatory-balance/blob/71204bac4f2c38389f4a85a256409ba3c43959ec/docs/adaptive_execution/reviews/check_authority.py
- CPU telemetry evidence: research_candidate/telemetry_cpu_validation.json
- Detailed prior corrections: research_candidate/review_resolution_ae001_003.md

