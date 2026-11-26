# ADR-006 — Language-conditioned navigation and actuator-aware execution

**Date:** 2026-11-26  
**Status:** Architectural selection and first CPU interface tests complete; actual VLM inference, rendered scene grounding, and G1 end-to-end physics are **not** yet run.  
**Scope:** Stanford AA228/CS238 course submission due Dec 4, plus a separable portfolio research continuation.

## Core scientific question (unchanged)
Can a frozen humanoid locomotion policy retain task competence when actuator gain unexpectedly changes, using delayed/noisy effort telemetry and an uncertainty-limited execution correction?

**Added auxiliary system question:** Can a robot accept a language goal, perceive an obstacle, plan a physically safe detour, and preserve this navigation objective despite actuator mismatch and stale perception? This is not a claim of novel end-to-end VLA learning.

## Decision: hierarchical, not end-to-end, VLA

Architecture:
```
Language instruction + RGB/depth frame
    |  (high-level, low-rate, may be a VLM after validation)
    v
Grounded target ID -> allowlisted schema/checks -> fresh perceived landmark XY
    |
    v
Rendered head-depth -> metric obstacle blobs -> timestamped static obstacle
    |
    v
Fail-closed detour/replanning -> (v_forward, yaw_rate) at 50Hz
    |
    v
Frozen G1 PPO -> nominal joint targets
    |
    v
RLS + gain-set belief + corrected feasible interval -> bounded actuation
    |
    v
G1 MuJoCo plant / contacts + delayed causal telemetry
```

Only a high-level target ID can be proposed by a future VLM. It may **not** output servo targets, policy actions, physical gain labels, unrestricted waypoints or direct motor commands. The deterministic baseline `language_nav.py` parses simple goal instructions and resolves IDs from *recent perceived landmarks*; it is **not a VLM or learned VLA**. A later VLM must output `{"operation":"navigate","target_id":"station","confidence":0.9}` using the same checked schema. The parser/schema and safety gate are explicitly different from a learned action policy.

The interface rejects unknown/ambiguous/stale landmarks, low-confidence predictions, unsupported model actions, and untrusted arbitrary key-value fields. Scene coordinate origin and goal landmarks are accessible through a disclosed simulator-localization assumption; no obstacle geom coordinate, gain or onset labels enter runtime decisions. Depth detection and the waypoint follower remain independent of the language interface.

## Alternatives assessed and why

| Choice | Decision | Reason |
|---|---|---|
| Language instructions encoded as direct joint torques | Reject | No bounded latency, action-space mismatch with frozen G1, no preservation of verified adapter |
| OpenVLA 7B directly producing G1 actions | Defer | Published system primarily trained for manipulation, requires target-embodiment adaptation, higher compute / integration risk |
| SmolVLA 450M predicting G1 navigation action chunks | **Follow-on** | More approachable model size and asynchronous action-chunk interface, but still needs G1 camera/state/action datasets and evidence of generalization |
| Generic remote VLM choosing semantic target IDs | Optional scoped add-on | Only after rendered images and detected landmark registry exist, with logged model name/latency/outputs |
| Deterministic constrained language parser -> target ID | **Select as registered baseline** | Runs without extra model or GPU, isolates physical adaptation, records safe failure modes, makes comparative study possible |
| Depth-based obstacle detector + bounded detour | **Keep primary navigator** | Obstacle coordinates inferred from rendered depth; direct ground-truth geom rejected for primary perception experiment |
| Replacing locomotion PPO with a learned VLA | Reject for Dec 4 | Makes actuator study hard to interpret and risks untestable performance claims |

Primary references: https://huggingface.co/docs/lerobot/main/smolvla ; https://huggingface.co/blog/smolvla ; https://openvla.github.io/ . Published research model metrics are **not** project results.

## Multi-rate contract

- Language task decision: event-triggered / only when goal changes, and logged with model or grammar name and latency.
- Depth: target 10 Hz; record source_step, release_step, range error (evaluation only), measured latency.
- Waypoint and frozen PPO: nominal 50 Hz / 20 ms.
- MuJoCo physics: nominal 500 Hz (10 substeps per control action).
- Adapter receives *causal* encoder/effort packets; gain, onset, return schedule excluded from estimator inputs.
- Replan only when a new, valid obstacle estimate is available. Stop forward motion on stale camera frames or invalid geometry.
- All methods share the **same high-level planner and depth sensor** in each paired evaluation.

These rates are *design targets*, not yet end-to-end measured deadlines. Record p50/p95/p99 for perception, planning, adapter, and combined control latency; report missed 20 ms control deadlines rather than implying real time from GPU throughput alone.

## Registered study matrix (before new task observations)

**A. Perception and language grounding (CPU/GPU dev)**: simple goal command -> allowlisted object ID; unknown goal; ambiguous landmark; stale landmark; empty scene; one cylinder; false blob; camera drop. Report command acceptance/rejection, detection recall/false alarms, pose/range error and stale-stop counts. Do not count deterministic parser as VLM.

**B. Integrated G1 navigation (development only)**: clear path vs contact-enabled cylinder, healthy vs gain 0.70 with recovery, with the same navigator under 3 actuator methods: correction-off, frozen heuristic, velocity-informed set-membership candidate. **72 runs = 2 scenes × 2 gain schedules × 3 methods × 6 paired scenario seeds, max 1,200 ticks each.** Freeze geometry, fault timing, sensor and model before physics. Report success, collisions, falls, completion time, path length, tracking error, healthy intervention, withdrawal count/median and uncertainty intervals over paired scenario seeds. Do not claim there are 72 *independent* scenes; pairs share the same seeds. Preserve all failed cases.

**C. Causal ablation**: same physical seeds with or without high-level goal parsing; same route under exact vs delayed depth; correction off vs adapted. Only run extra arms after separately registering them and accounting for budget. Results from earlier 256-episode CPU development benchmark are **not** results of this new integrated task.

**VLM model evaluation (conditional)**: compare constrained grammar against an actual selected VLM on a fixed instruction/image set. Give the VLM only rendered image(s), a bounded perceived-landmark registry, and text instruction; schema-check every response. Report grounding accuracy, ambiguity/stop decisions, model latency and fallback frequency. Do not describe the study as VLA until a learned action model genuinely outputs evaluated robot actions.

## Exit criteria for Dec 4

1. Real contact-enabled obstacle observed from G1 head depth. Detector and robot-localized target feed the planner; no privileged obstacle coordinates online.
2. Healthy frozen PPO completes route with measured collision/fall labels on selected, preregistered seeds.
3. Paired degraded/restore trajectories compared with correction-off under the same navigation decisions and seed map; report failures even when no improvement.
4. At least one H.264 clip rendered from actual G1 MuJoCo states, timestamped and indexed to source run record, showing healthy and degraded/adapter outcomes, including honest failure if applicable.
5. Paper clearly separates confirmed CPU baseline (256 episodes), 17 tuning evaluations, any completed protected WP2.2 pilot, and the new navigation study; no held-out or hardware claims without evidence.
6. Independent raw CSV/trace-to-figure audit, source/checkpoint hashes, simulator version and GPU billing/stop receipt.
7. If perception or GPU integration fails, scope the claim back to actuator adaptation and show the documented failure; do not invent visual navigation success.

## Decision/change-control log

- Nov 15: Original OAA task was frozen G1 walking with actuator adaptation. Visual navigation was a later extension.
- Nov 21–22: G1 review exposed excessive conservatism of speed-only hold envelope; chose velocity-aware correction and declared-fast transport study.
- Nov 25: Restricted final course paper to development split due to missing held-out scientific verification; frozen heuristic available.
- Nov 26: Published 256-episode CPU baseline, 17-point tuning provenance, and scoped obstacle demo under PR #50/#51.
- Nov 26: Chose head-depth detection and fail-closed inflated-circle waypoint detour (ADR-005); no ground-truth navigation shortcut.
- Nov 26: Prasuna requested VLA reconsideration for Stanford/portfolio. Chose **strict task-language interface + depth navigator + frozen PPO + uncertainty-aware adapter**, with optional VLM as a gated layer and post-submission SmolVLA action-model extension. No assertion of trained VLA on G1.
- Nov 26: User confirmed project direction and publishable evidence requirement. Initial strict language goal interface has 15/15 local CPU unit tests in Python 3.13.5; G1 physics and actual VLM inference remain pending.

Future changes require a date, empirical failure or motivating observation, changed hypothesis, budget impact and a new development registration. Do not change test thresholds after viewing outcomes.
