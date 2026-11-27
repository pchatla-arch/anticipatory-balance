# Evidence-backed results and next experiment decision log
**Online Actuator Adaptation for Reliable Humanoid Locomotion — Unitree G1**  
**Evidence cutoff:** 2026-11-27 UTC, from the frozen S0/S1a runs  
**Purpose:** a checkable empirical section for the course paper and an explicit plan–run–audit–revise loop. Not a manuscript acceptance claim.

## Research question and scientific boundary

Can a frozen PPO humanoid walking controller use delayed, noisy, estimated actuator feedback to compensate for hidden reductions in position-servo gain, while (a) avoiding unnecessary correction when actuators are healthy, (b) withdrawing correction when gains return to normal, and (c) preserving actual route progress and stability?

A calibrated simulator, correct G1 model/checkpoint, and paired scenario seeds are necessary but insufficient to establish hardware transfer. The oracle baseline knows the injected gain schedule; practical controllers do not. The tests here do not demonstrate a learned VLA, physical robot, collision-avoidance navigation, or statistically robust generalization.

## Experimental records

| Evidence | Source | Matrix | Outcome |
|---|---|---|---|
| S0 GPU smoke | `foresight/experiment-results/results/g1_course_dev_20261127T024308Z_104a34` | gain 0.70 and healthy × none, oracle, heuristic × 2 paired seeds; 160 steps | 12/12 rows, CUDA, audit passed |
| S1a GPU long-horizon | `foresight/experiment-results/results/g1_course_dev_20261127T032240Z_5fc7dd` | four families × none, oracle, estimated inverse, heuristic × 2 paired seeds; 400 steps at 50 Hz | 32/32 rows, 16 cells, audit passed; RunPod provider EXITED |
| G2a contact-model gate | `foresight/experiment-results/results/g1_g2a_model_contact_20261127/physical_contact.json` | contact-enabled static cylinder in actual G1 model; evaluator-only forced overlap | Actual model compiled, 4/4 contact pairs triggered in overlap, 0 contacts initially |
| G2b head-depth navigation | `scripts/eval_g1_obstacle_nominal.py` | real G1 head-camera, conservative detour, actual contact monitoring and video | **In progress / not yet verified** as of this evidence cutoff |

G2a is a model contact gate, not a walking collision-avoidance result. Head-depth detection primitives and route planner have CPU tests but were not integrated with physical contacts in S0/S1a.

## Independent descriptive contrasts from S1a raw CSV

For fixed paired scenario seeds, the gain-0.70 fault yields:

| Method | Mean post-fault squared velocity loss ↓ | Mean measured forward progress (m) ↑ | Falls |
|---|---:|---:|---:|
| No correction | 0.085517 | 3.621946 | 0/2 |
| Oracle inverse | 0.028719 | 2.220856 | 0/2 |
| Estimated inverse | 0.031378 | 2.228384 | 0/2 |
| Heuristic gate | 0.038328 | 2.440928 | 0/2 |

The heuristic lowers velocity-error loss by **55.18%** but lowers forward progress by **32.61%** against no correction; the privileged oracle shows the same direction of trade-off. It is incorrect to claim that the learned/frozen policy completes more useful motion based on error alone.

For gain return, the heuristic's observed withdrawal times are **6 and 5** 50-Hz ticks: **2/2 withdrew**, median 110 ms. Estimated inverse: **0/2 withdrew**, despite slightly lower velocity loss. Healthy nuisance torque-scale 0.8: no correction loss **0.026801**; heuristic loss **0.031044** and mean harmful offset **0.043697 rad**, indicating a real false-adaptation failure. Estimated inverse has **0.107939 rad** mean harmful offset in that nuisance family. Healthy without the nuisance: heuristic harmful offset only **0.000321 rad**, illustrating that nominal healthy tests alone are not a sufficient safety evaluation.

No simulated falls were observed in the 32 episodes, but n=2/cell and development split do not justify robustness, safety, statistically meaningful bootstrap confidence intervals or hardware deployment claims. The reported progress is pelvis-heading-projected displacement after the gain-onset window, not task completion or map-based route efficiency.

## Why the result is scientifically informative despite the negative components

The proposed uncertainty-gated correction layer is a constrained residual target adjustment on top of a frozen locomotion policy:

\[
a_t = a_t^{\mathrm{PPO}} + \Pi_{\mathcal U}(\alpha_t \Delta a_t),\qquad
0\le\alpha_t\le1.
\]

A gain estimate can reduce motor-model mismatch while simultaneously changing gait velocity, which means a tracking metric and a task metric can disagree. A confounded effort telemetry channel can produce an apparently plausible gain estimate even when the physical gain is unchanged. These are causal-identification and multi-objective-control limitations—not reasons to hide the experiment.

The next paper comparison should report **both** conditional planar-velocity loss and actual forward progress at matched command schedules, and state the precise restoration denominator. A new confidence gate is an experimental hypothesis, not something established in this dataset.

## Rapid next experiment loop

| Gate | Executable experiment | Observable pass/fail | On failure |
|---|---|---|---|
| G2a (passed) | Actual G1 static cylinder compiles and makes contact when evaluator forces overlap | Model, head camera, 4 contact-pair hits, no initial contact | Repair model/asset loader; do not present geometry unit tests as physics |
| G2b (current) | Frozen G1 navigates with rendered head metric depth; calibrate ground pixels and detect the true cylinder without reading geom position online | Head depth frame, source timestamps, detection confidence, no privileged fallback, real movement and bounded commands | Save log, identify first traceback, patch exact interface, validate on CPU/JAX, rerun same *development* gate on existing pod |
| G2c | Actual G1 follows a conservative detour with contact-enabled cylinder | Goal reached, no collision, no fall, real trajectory, diagnostics and simulator video | Report collision/stall, simplify route transparently, preserve failures |
| G3 | **Only after G2c**: same route/perception for correction-off vs heuristic under healthy/gain 0.70+return | Paired goal completion and distance progress, tracking error, contact and false interventions | Do not expand to the registered 72-row matrix; publish negative result |
| Paper | Quantitative S0/S1a tables, executed G2 evidence, limits, algorithm and AI-use disclosure | Numbers reconcile to raw data, source SHA, frozen checkpoint and CI logs | Omit unsupported G2/video outcomes; submit a rigorous actuator-adaptation study |

**Stop rule:** do not buy another GPU simply to rerun Python import, GL package or observation-shape failures. Test those in CI/container first. The actual scientific G3 comparison must not begin while G2b/G2c fails.

## Practical completion boundary for November 27

A defensible near-term artifact is **a fully auditable paper on online actuator adaptation with verified 44 total S0+S1a development episode records, honest task-progress and telemetry-confound findings, plus an authentic simulator clip if G2c succeeds**. The 72-episode pre-registered obstacle study and a new trained VLA are not prerequisites to this narrower claim and must not be represented as complete. A real G1 obstacle pilot may still fail; preserve and report that result.

## Reproduction and provenance

Pinned S1a source `89b6d0dd31020f00032330c76684c727b1f697a0`, frozen G1 checkpoint SHA-256 `14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de`, and frozen heuristic configuration `d756c37d0564837a74132c30f8e366320a6bb1a5`. Data and logs are on the protected dedicated GitHub results branch; the course report should cite exact commit and filenames. Do not pool S0 and S1a into a 44-episode inferential sample because they use different horizons and overlapping development scenario families.
