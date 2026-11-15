# Controllers and frozen components in Foresight

Technical notes · 15 November 2026

Discussion and follow-up: [issue #7](https://github.com/pchatla-arch/anticipatory-balance/issues/7).

**Scope:** the implementation reviewed at [commit `67abf97`](https://github.com/pchatla-arch/anticipatory-balance/tree/67abf97ed6b88bd51241b45b927d6356cdfa1cac). This note distinguishes implemented controllers, previously trained policies, the Session H evaluation protocol, and proposed extensions. It does not report a completed full Session H run or establish that the value-of-information rule outperforms simpler rules.

## 1. Two levels of control

Foresight asks when an approaching object justifies interrupting locomotion to prepare for impact. The implementation separates **how the robot moves** from **when it changes behavior**.

At the lower level, a PPO actor maps robot observations and a velocity command to 29 joint actions. Playground scales those actions around the default pose to produce joint-position targets, then steps the simulator. These are not direct torque commands; existing MuJoCo position actuators and joint dynamics implement the physical response. See the pinned [Playground action mapping][playground-action].

At the upper level, a controller requests CONTINUE or BRACE. CONTINUE selects the nominal walking actor; BRACE selects a separately trained bracing actor and supplies a zero velocity command. The upper controller selects an existing behavior rather than specifying every joint action. See [policy selection][rollout-switch].

In compact notation: walking applies `pi_walk(observation, command)`, the stillness diagnostic applies `pi_walk(observation, 0)`, and bracing applies `pi_brace(observation, 0)`. The rollout computes both actor outputs and selects the action to apply; switching does not imply that only one network is evaluated.

The low-level skills are learned by PPO, while the current belief-based decision layer uses explicit models, costs, rules, and short lookahead calculations. A learned switching policy would be an additional experiment.

## 2. What has actually been trained

The [registry][registry] exposes three training configurations: nominal, brace, and aware. Different seeds and revisions produce these published bundles:

| Published bundle | Role | Used by Session H? |
| --- | --- | --- |
| `nominal_s1` | Seed-1 nominal walking policy | Yes |
| `nominal_s2` | Independently trained nominal walker | No |
| `brace_s1` | Earlier brace policy without a walker handover | No |
| `brace2_s1` | Seed-1 brace policy with walker-state handovers | Yes |
| `brace2_s2` | Second brace policy, using the seed-2 walker for handovers | No |
| `aware_s1` | Object-aware PPO actor used as a separate baseline | No |

Each bundle contains a checkpoint at step `000202342400`. These are previously trained parameters, not policies trained during Session H. The nominal and brace networks have feed-forward hidden layers of 512, 256, and 128 units, 103 actor inputs, and 29 outputs. The aware actor has 108 inputs. See the [nominal configuration][nominal-network], [aware configuration][aware-network], and [Session H plan][h-plan].

### Standing is not a third learned skill

The zero-command comparison uses **the same nominal walking weights** with the three velocity-command observation entries replaced by zero. In the oracle evaluator, `mode=zerocmd` passes the nominal actor in the second policy slot. This tests asking the walker to stop, without another trained policy. Sources: [oracle comparison][oracle-modes] and [command substitution][rollout-switch].

BRACE also receives zero command, but has different weights learned under a different reward. Its objective favors a base/pelvis height of 0.65 m, foot separation of 0.35 m, and both feet contacting the ground. Walking gait-phase and airtime rewards are disabled, as are default-pose terms favoring the usual taller, narrower posture. Action-rate and torque penalties also change. These are **training targets**, not guarantees that the posture is reached at every switch. The height target is not a measured whole-body CoM height. See [brace configuration][brace-config].

Thus asking the walker to stop and selecting BRACE can produce different transient and impact responses despite both supplying zero command. Neither command guarantees instant stillness. A policy effective near its training reset distribution may also fail when selected during a walking gait.

### What BRACE v2 changes

With a handover checkpoint configured, a frozen nominal walker controls the first randomly selected 0–2 seconds of each training episode. The brace actor then takes over from the resulting walking state. The environment overrides the proposed brace action during the walker-controlled interval and sets that interval's reward to zero. See [handover setup][handover-setup] and [execution and reward handling][handover-step].

The saved `brace_s1` configuration has an empty handover checkpoint; `brace2_s1` identifies the nominal walker. Both saved configurations already specify 0.65 m height and 0.35 m separation targets. The supported distinction between these particular bundles is inclusion of walker-state handovers, rather than evidence of a different height target. See the [v1 configuration][brace1-config] and [v2 configuration][brace2-config].

## 3. “Frozen” means fixed parameters, not fixed behavior

A frozen policy still observes a changing robot and produces different actions every control step. The loaded inference parameters and stored observation-normalization state remain fixed during evaluation. No PPO optimizer updates them during a rollout. The loader restores parameters and constructs deterministic inference by default. See [policy restoration][rollout-policy].

The robot state, contacts, actions, delivered detections, belief mean and covariance, collision estimates, selected mode, commitment counter, and release state still evolve online. These are observations and controller state, not network training. Likewise, a frozen response model keeps its fitted coefficients while its predicted fall probability changes with current momentum and brace lead.

Preserve the existing FP32 policy reference for precision experiments. A lower-precision actor is a separately derived candidate: changing its representation is a controller intervention even without optimizer updates. The [numerical-precision note](numerical_precision.md) separates that proposed study from the completed host-side reporting correction.

## 4. The implemented high-level choices

The frontier evaluator supports eight method names:

| Method | Implemented decision |
| --- | --- |
| `reactive` | Never select BRACE; keep the supplied walking actor active, normally the nominal policy. |
| `always` | Start BRACE when belief hit probability exceeds a configured threshold. This does not mean bracing from reset. |
| `ttc_threshold` | Start when the point-estimate trajectory predicts a hit and time to contact is below a threshold. |
| `belief_commit` | Apply the earlier expected-benefit rule, requiring estimated time to form the brace. |
| `belief_lead` | Compare expected costs using lead-conditioned brace effectiveness, without the earlier formation-time gate. |
| `belief_wait` | Compare bracing now with waiting until another delivery time, predicting the belief without incorporating another observation. |
| `belief_voi` | Compare bracing now with waiting, including hypothetical next-observation and no-detection branches. |
| `oracle` | Follow a privileged schedule based on nominal arrival and the scheduled-hit label. |

Sources: [supported methods][frontier-methods], [request rules][request-rules], and [oracle schedule][oracle-schedule].

The blind-wait comparison matters because a benefit from acting later does not itself establish a benefit from obtaining information. `belief_wait` and `belief_voi` separate those mechanisms within the implemented approximation.

The lookahead considers one future delivery and uses 16 hypothetical observations for its observation branch. Its no-detection probability comes from the detector's configured miss probability; future field-of-view loss is not forecast. This is a tractable approximation, not an exact solution of the full humanoid POMDP. See [lookahead implementation][lookahead].

There is also a timing approximation: belief prediction uses the phase-dependent time to the next delivery, while the rule's waiting cost and contact-during-wait check use the configured full period. At 10 Hz these can differ between deliveries. This should remain visible when interpreting the one-step calculation. See [request rules][request-rules].

### Models and proposed controllers are separate

QMDP, POMCP, a belief-PPO gate, and a recurrent PPO gate are not implemented alternatives in this frontier. Calling the present one-step rule “QMDP” or claiming a completed explicit-versus-recurrent-belief comparison would overstate the code.

The capture-point/LIP module is an analytical diagnostic used by the overlay analysis. It does not select deployed joint actions or implement an online LIP/MPC controller. A physical model supporting an explanation is not automatically another controller baseline. See the [capture-point module][lip-model] and its [analysis use][lip-use].

## 5. What the observation and response models know

Despite its filename, `particle_filter.py` implements a Gaussian constant-velocity Kalman filter for position and velocity, combined with sampled masses and Monte Carlo decision statistics. Capture timestamps allow delayed detections to update the current state estimate. The straight-flight model suits the constructed pre-impact projectile motion; it is not a complete post-contact dynamics model. See [belief implementation][belief].

Mass is sampled from a uniform prior, normally 4–16 kg, when the track initializes. Position observations update position and velocity but do not identify mass. This belief path does not infer a size-conditioned mass distribution from depth; that proposal would require additional observation assumptions and calibration.

The reduced response model predicts fall probability from incoming momentum and action. BRACE optionally includes a lead-time term capped at 0.5 seconds. It is fitted on **contact-conditioned** E6b oracle episodes, and its coefficients stay frozen in Session H. It summarizes particular skills under a particular experimental history, rather than a universal physical impact law. See [response fitting][response-fit] and [evaluation][response-rule].

The object-aware PPO baseline offers a separate route to anticipation. Its actor receives five extra values: detection validity, relative object position in three coordinates, and detection age. These come from the parametric noisy/delayed detector. Its training critic additionally receives true pelvis-relative object position and world-frame object velocity. It is a feed-forward object-aware actor, **not a pixel-input actor or recurrent belief estimator**. See [aware observations][aware-obs] and [Session E training][aware-training].

The evaluator labels its standalone aware-actor condition `reactive` to indicate that no BRACE switch is applied. That label does not remove the actor's object observations or imply that it cannot learn anticipatory behavior.

The rendered-depth experiment instead uses a geometric detector to convert frames into detections for the explicit belief and switching controller. Depth therefore affects selection between frozen skills. Neither experiment establishes a hardware or real-camera result. See [depth evaluator][depth-eval].

## 6. Timing: formation, commitment, cost, and release

The control interval is 0.02 seconds: **50 Hz**. Session H requests a detection every five steps: **10 Hz**. Observation latency is separate from that cadence; five additional steps represent 100 ms of simulated delay. GPU rendering time is not automatically inserted as observation delay. Sources: [evaluation flags][timing-flags] and [depth protocol][depth-timing].

The brace-related timing settings have different purposes:

| Quantity | Setting or interpretation |
| --- | --- |
| Formation estimate, `tau_brace` | 0.22 s in the earlier `belief_commit` start condition. Neither minimum commitment nor a universal readiness guarantee. |
| Minimum commitment, `commit_steps` | 15 control steps, or 0.3 s. A selected brace remains active through this minimum interval. |
| Cost-model duration, `hold` | 0.5 s added after expected contact when estimating preparation cost. |
| Execution release, `release_after` | Time after estimated closest approach before release; Session H compares 0.3, 0.6, and 1.0 s. |
| Staleness, `stale_steps` | Release eligibility after strictly more than 50 steps without a valid detection: at least 1.02 s on the 20 ms grid. |

The shared wrapper also releases after more than 50 steps without a valid detection. Minimum commitment takes priority. Release uses the Kalman mean's closest-approach time, which can become negative after passing; it does not use a mean restricted to future-hit samples. See [parameters and release][release].

Consequently, `release_after` is not total time braced: onset changes that duration, and staleness can release before a longer post-pass delay has elapsed. The fixed oracle bypasses this shared wrapper and uses its own schedule.

Brace lead is the time between starting preparation and the relevant arrival/contact reference. A threshold or cost rule influences lead, but does not make it identical to any duration in the table.

Changing `release_after` does not refit the response model or change its cost `hold`. Session H deliberately keeps those fixed. The experiment measures execution sensitivity of the deployed controller, not the optimum under a newly calibrated hold-dependent model. See the [protocol caution][hold-caution].

## 7. The fixed oracle is a reference, not an upper bound

The default oracle begins bracing 0.5 s before **nominal** arrival and ends 0.5 s afterward for scheduled-hit scenarios. It does not optimize timing separately for every episode, guarantee actual contact, or search all possible low-level actions. Its schedule also differs from adaptive belief-based release. See [oracle flags][timing-flags] and [schedule][oracle-schedule].

A practical controller can therefore outperform that particular oracle schedule without exceeding an optimal perfect-information controller. Report it as a fixed privileged-timing reference. A stronger upper-bound claim requires an optimization argument or a sufficiently broad privileged-policy search.

## 8. What Session H holds constant and varies

Session H fixes the seed-1 nominal and BRACE-v2 checkpoints, fitted response coefficients, current belief implementation, and execution code within the run. There is no low-level training stage. Its operational gates check whether experiments run and produce acceptable evidence; they are not learned neural gates or superiority tests. See the [plan][h-plan].

The full protocol compares:

- Parametric versus rendered-depth detections at 0 and 100 ms added latency, using threshold, lead, blind-wait, and observation-lookahead rules.
- Execution release at 0.3, 0.6, and 1.0 s, using reactive, TTC-threshold, and lead-aware belief conditions.

Depth uses threshold 0.3 s and preparation cost 0.05 for each belief rule. Hold sweeps use thresholds 0.3/0.4 s and preparation costs 0.05/0.1. The plan does not run `always`, the legacy `belief_commit`, the oracle, or the aware actor.

Small gates and larger follow-ups use disjoint seed groups. Follow-ups deliberately repeat Session G scenario seeds for paired comparisons; they are not untouched test scenarios. Holding skills fixed supports attribution to the evaluated intervention, while making uncertainty statements conditional on that pair.

## 9. Why freeze, and when retraining becomes necessary

Freezing avoids repeated PPO training while changing decision logic. More importantly, it supports a clear scientific question: given these motor behaviors, what does improved sensing or timing accomplish? Jointly retraining the actor after every rule change would make improvements harder to attribute.

The choices carry different benefits and limitations. The following are design tradeoffs inferred from the implementations, not a claim that one choice dominates experimentally:

| Choice | Why use it? | What can limit it? |
| --- | --- | --- |
| Continue walking | Preserves commanded progress and the walker's learned reactive recovery, including any stepping it performs. | Strong impacts may exceed that recovery capability; this mode makes no explicit preparation decision. |
| Zero-command nominal policy | Requires no new training and isolates the effect of asking the walker to stop. | Zero desired velocity does not train a wide, low impact posture or guarantee a good stopping transient. |
| Separate BRACE skill | Gives PPO a posture and impact-survival objective, with walking-state handovers in v2. | Costs progress, introduces transitions, and remains limited by its training distribution and learned recovery. |
| TTC threshold | Provides a cheap, interpretable timing rule without fitting a fall-response model. | Does not explicitly trade severity, uncertainty, and preparation cost; its threshold still needs evaluation. |
| Lead-aware belief rule | Combines hit probability, uncertain momentum, preparation cost, and estimated benefit of the available lead. | Depends on the mass prior and fitted response model; a mismatched model can favor the wrong action. |
| One-step observation lookahead | Can represent the value of another detection while accounting for lost preparation time. | Requires additional observation and response assumptions; the current timing approximation and field-of-view omission limit its interpretation. |
| Learned gate, proposed | Could learn context-dependent switching while keeping motor skills fixed. | Adds training, reward choices, and generalization questions; it does not automatically provide calibrated uncertainty or outperform a tuned rule. |

The [Session G review](../SESSION_G_REVIEW.md) reports the observed failure/preparation tradeoff and no demonstrated VOI advantage. That evidence supports comparing methods at matched operating points rather than selecting the most elaborate rule by name.

The upper layer cannot recover capability that the skills lack. A bad switch transition, ineffective brace, or inadequate recovery behavior may dominate the outcome. A model fitted to one pair may mispredict another. Appropriate next steps are controlled retraining or an additional recovery skill, followed by transition checks, response-model recalibration, and comparisons preserving the old policies.

Historical training differences require care. The [registry comment][registry] records that `nominal_s1`, `brace_s1`, and `brace2_s1` used a 1–8 kg randomizer even when brace configuration declared 4–16 kg. The later registry selects ranges per skill, and [Session G][seed2-training] trains the second pair through that path. Pair differences can therefore reflect training-distribution changes as well as seeds. “Variation across independently trained policy pairs” is safer than a pure seed-variance claim unless matched provenance establishes otherwise. See also the [Session G review](../SESSION_G_REVIEW.md).

Future learned-gate, recurrent-policy, and quantization studies should preserve this reference and change one component at a time. Their value requires evidence on falls, progress, preparation cost, and timing across paired scenarios, with uncertainty and failures reported. Additional complexity is not itself an improvement.

## Discussion

Use [issue #7](https://github.com/pchatla-arch/anticipatory-balance/issues/7) for controller questions and proposed comparisons. Useful decisions include whether a weak result is limited by brace capability, model error, or timing; whether to train matched policy pairs; and what evidence would justify a learned gate. Precision-specific questions belong with the [numerical-precision note](numerical_precision.md).

[playground-action]: https://github.com/google-deepmind/mujoco_playground/blob/9de993826d1a486f18aec4e8d3179e93ff64ed41/rss2025/mujoco_playground/_src/locomotion/g1/joystick.py#L363-L365
[registry]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/skills/register.py#L16-L36
[rollout-policy]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/eval/rollout.py#L51-L76
[rollout-switch]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/eval/rollout.py#L158-L210
[nominal-network]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/checkpoints/nominal_s1/000202342400/ppo_network_config.json#L1
[aware-network]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/checkpoints/aware_s1/000202342400/ppo_network_config.json#L1
[oracle-modes]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_oracle_brace.py#L125-L140
[brace-config]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/envs/g1_impact/impact_joystick.py#L80-L119
[handover-setup]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/envs/g1_impact/impact_joystick.py#L183-L213
[handover-step]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/envs/g1_impact/impact_joystick.py#L345-L404
[brace1-config]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/checkpoints/brace_s1/env_config.json#L1-L20
[brace2-config]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/checkpoints/brace2_s1/env_config.json#L1-L20
[frontier-methods]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_decision_frontier.py#L161-L200
[request-rules]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/decision/rules.py#L104-L147
[oracle-schedule]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/eval/rollout.py#L171-L179
[lookahead]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/decision/controller.py#L29-L56
[lip-model]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/lip/capture_point.py#L1-L25
[lip-use]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/lip_overlay.py#L1-L55
[belief]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/belief/particle_filter.py#L1-L114
[response-fit]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/fit_response_model.py#L32-L55
[response-rule]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/decision/rules.py#L34-L54
[aware-obs]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/envs/g1_impact/impact_joystick.py#L283-L325
[aware-training]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/run_session_e.sh#L80-L99
[depth-eval]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_depth_closed_loop.py#L1-L13
[timing-flags]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_decision_frontier.py#L47-L77
[depth-timing]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_depth_closed_loop.py#L182-L195
[release]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/decision/rules.py#L57-L173
[hold-caution]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_decision_frontier.py#L215-L224
[h-plan]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/session_h_plan.py#L1-L51
[seed2-training]: https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/run_session_g.sh#L151-L160
