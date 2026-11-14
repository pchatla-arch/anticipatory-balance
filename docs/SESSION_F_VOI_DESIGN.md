# Session F: minimal one-observation lookahead

## Conclusion

Proceed first with a CPU decision-model prototype, called **one-observation belief-space lookahead**. It does not fix delayed information or establish that waiting is useful. Lead conditioning may improve late BRACE decisions independently. Production deployment and performance claims remain blocked by the release diagnostic, mismatched timing labels, and response-model support described below.

## 1. Exact decision semantics

Fix an absolute evaluation horizon H, a decision trigger shared by all comparisons, one threat, and a frozen nominal/brace checkpoint pair. At the root after processing the current delivery, choose among:

- C: CONTINUE through H, no later intervention.
- B: execute a completely specified bounded BRACE macro and return to nominal afterward.
- W: CONTINUE until the next scheduled delivery d, then choose only C or B, even when that frame contains no detection. Consume the single wait token.

This is a two-stage policy, not unrestricted receding-horizon replanning. If repeatedly giving W another fresh one-step horizon, the controller can postpone forever. The root may not wait when d >= H. A one-step CONTINUE action with later opportunities is a different policy and must not be described by terminal C values.

Represent BRACE's release deadline/duration as part of its action. For a clean binary experiment fix the future B macro ex ante across information counterfactuals. Alternatively optimize over the same finite menu of bounded macros in both cases. If a posterior changes B's latched deadline, an apparent information gain can partly be a different brace duration; this is a real additional adaptation but must be explicit.

## 2. Delivery and capture times

The repository's `delivery_due` attaches phase to delivery. Let n be the already-processed control tick, L latency ticks, and p detector period:

    d = min{k > n : delivery_due(k,p) and k >= L}
    c = d - L
    delta = (d-n) dt

Do not use delta=(period+latency)dt. Example p=2.5, L=7, n=8: d=10, c=3, delta=.04 seconds. The pending frame already existed five ticks ago. Predict its captured position and visibility conditional on delivered history; do not expose the detector's privileged true-state buffers to the planner. Use the robot's own observed pose/heading history at c; for future c use the specified CONTINUE prediction.

An outcome is Y=empty or the actual delivered measurement, with capture timestamp and any provided noise scale. For a latent trajectory x, a simple detector gives

    P(empty|x) = 1 - launched(c) visible(x_c,robot_c) (1-miss_prob).

The positive-detection likelihood includes the complementary gate and range-dependent noise. Visibility/range apply at capture, not delivery. An empty scheduled frame and a tick with no scheduled frame are distinct. Invalid observations currently do not update the deployed filter: either deliberately retain this approximation in every compared arm or implement and calibrate negative evidence in every arm. Changing only the lookahead arm changes both filtering and planning.

`PF.update` uses z=p_d-v_d*(d-c)dt. That relation is exact for constant velocity; the configured process velocity noise is nonzero, so an exact stochastic historical likelihood needs a joint past/current state or smoothing distribution. A zero-process-noise constant-velocity CPU prototype is defensible if explicitly scoped to that model.

## 3. Whole-path value and pure information value

Let x include hidden trajectory, mass, any response-relevant latent variables, and event/survival status. Let b be the belief after current observations, and let u_a(x;s,H) be remaining utility of the specified terminal macro, for example

    u_a = -c_F P(failure by H | x,a,s)
          -c_P E[brace time in [s,H] | x,a]
          -c_T E[tracking loss in [s,H] | x,a].

Fit probabilities/costs for those same macros and horizon. Average over the joint distribution of hit, momentum, and lead; do not use only mean lead or mean momentum in nonlinear response functions. Misses can still incur baseline falls and BRACE cost. Physical contact time and entry into the .45m decision radius are different labels.

At time s, Q_a(s,b)=E_b[u_a(x;s,H)]. For waiting, let R_prefix include actual running utility and any absorbing failure penalty before d, and S mean alive at d. Then

    Q_W = E[R_prefix] + P(S) E_{Y|S}[ max_a E[u_a(x;d,H) | S,Y] ].

No future action is selected on failed trajectories. Future response values are conditional on survival and exclude already charged failure. If instead the model supplies an entire-path outcome for WAIT-then-a, do not add the prefix failure loss again. A Bernoulli eventual-fall model alone does not specify when failure occurs, so it cannot justify unrestricted pre-delivery survival branching.

Define the blind delayed alternative at the same physical d:

    Q_W0 = E[R_prefix] + P(S) max_a E[u_a(x;d,H) | S].
    VOI(d) = Q_W - Q_W0.

These share the same waiting trajectory, response lead, macro menu, and time horizon. In an exact coherent model, ignoring an observation remains feasible, so VOI >= 0. Waiting itself can still be worse than immediate BRACE:

    Q_W - max(Q_C(now),Q_B(now))
      = VOI(d) + Q_W0 - max(Q_C(now),Q_B(now)).

Reduced readiness from delay and changed brace duration already enter the response/cost model. An extra heuristic latency penalty counts them twice unless it denotes a separate real cost. Conversely, saved preparation time from waiting is a timing benefit, not VOI.

## 4. Honest Monte Carlo

For each outer sample, independently draw a latent trajectory from b, simulate CONTINUE to d, and draw Y using that trajectory's captured state. Sample root/inner trajectories from b independently; condition/reweight the *whole bank* by survival and the likelihood of this Y, then average each action's utility and choose one action. Never choose a best action for each hidden trajectory and average those maxima; that leaks perfect state knowledge. Never pair each root particle with its own simulated observation and evaluate only that particle.

Use separate draws or cross-fitting for selecting a branch action and evaluating that selected action, since maximizing noisy estimated means creates optimistic values. Common random numbers across candidate actions can still reduce variance. Preserve the mass mixture: position-only measurements cannot identify two masses on the same kinematic trajectory. Record MC uncertainty and fall back to the lead-conditioned myopic choice when a claimed wait advantage is unresolved; do not hide negative estimated VOI by clipping before diagnostics.

## 5. Response fit and blockers

The current response model is a separate sigmoid in log momentum for CONTINUE and one fixed BRACE lead, fitted only to actual-contact rows. The oracle data schedule BRACE around nominal arrival with an oracle hold. The deployed deadline protocol instead uses a forecast, minimum hold, maximum duration, and single commitment. These are different intervention policies. Selecting actual contacts separately by action also conditions on an action-affected outcome.

For F, pair fixed exogenous scenarios/reset keys across CONTINUE and multiple start-time/macro-duration interventions. Train and hold out by scenario identity, not rows. Use unconditional scenario outcomes or explicitly model action-specific contact and no-contact outcomes. Include no-launch/miss controls for brace cost and baseline failure. Log both absolute switch time and physical first-contact time (possibly absent), nominal arrival, decision-radius entry, duration, failure time, and horizon censoring. Do not silently label all these as lead. A model indexed only by (J,lead) must acknowledge omitted gait/body state, direction, height, duration, and checkpoint dependence.

The .45m future-entry statistic becomes infinity after entering the radius even though physical contact can still be ahead. It is not a signed contact clock. Without a calibrated event definition and evidence for pre-observation failure timing, keep the prototype analytical/offline; do not claim contact-optimal timing or recovered latency performance.

## 6. Minimal ablations and CPU gates

Use the same release macro, perception/update semantics, decision trigger, evaluation horizon, and paired test scenarios:

1. Historical fixed-lead myopic values.
2. Lead-conditioned myopic values: isolates response-model improvement.
3. Lead-conditioned blind delayed option W0: isolates delayed timing/cost scheduling.
4. Lead-conditioned observation-branch option W: W versus W0 isolates information.
5. Uninformative/shuffled future measurement and perfect future measurement: negative control and diagnostic ceiling under the same delay.

Keep default versus narrow-FOV priors, mass mixtures, and chosen delay/cadence fixed within each comparison. Report unconditional fall, preparation/tracking cost, false preparations, wait choice, actual lead at switching, release timing, and paired uncertainty; contact-conditioned fall is diagnostic only.

The accompanying standard-library test file has 12 passing analytic checks: fractional delivery phase, pending old capture, startup timestamp validity, finite horizon, zero VOI for uninformative data, noisy-versus-perfect information, explicit missing frames, informative absence, average-before-max, positive VOI but harmful delay, absorbing pre-delivery failure, and prefix cost once. Additional integration gates need mass-posterior invariance, history/visibility timing, no oracle buffer access, actual filter parity, one wait token, and end-to-end release/contact traces.

These tests validate the formulation, not a production implementation, fitted response quality, or the reported release run.
