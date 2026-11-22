# Stage A repeatability calibration · protocol for approval

**Decision requested.** Approve Option B in AE-009 for one independent calibration of the pinned RunPod Warp backend. The calibration estimates how much two executions of the same healthy walker can differ. G0 and the registered comparative test have separate decisions. The most immediate G1 experiment is the velocity-informed envelope coverage replay on CPU.

## Why a new run is necessary for Option B

The earlier GPU smoke used eight exposed scenarios. The largest same-scenario healthy-repeat differences were 0.0004041083 (m/s)² in fixed-horizon loss, 0.02347875 m in forward progress, and 3.024658 N·m in peak raw actuator force. Healthy versus identity had a 3.025326 N·m difference. Those runs demonstrate a numerical noise source. The small exposed sample cannot establish a fresh-gate tolerance. The unchanged Stage A launcher would either block the full pinned-backend gate or retain the original identity criterion.

## Prospective design

Freeze these three healthy–healthy maxima as **normalization constants** before a new run. For each of 64 fresh scenario batches (seeds 940101 through 940164), replay the same 32 initial states and commands twice with healthy gain and correction off. Use the frozen nominal-s1 model, 400 control ticks, 3.16.0 MuJoCo/MJX, Warp 1.18.0, JAX 0.12.0.1, and the RTX 6000 Ada GPU. Both repeats use identical noise streams and the exact Stage A rollout. The 64 batch pairs generate 4,096 rows. No Stage A screening or confirmation seeds are used.

For each batch, compute the maximum over all 32 lanes and all three quantities of the absolute physical difference divided by the corresponding frozen normalization constant. The maximum score across the 64 batches gives a common multiplier and three reportable physical envelopes in their original units. This is one simultaneous score across the batch, so it accounts for correlation across joints/metrics and lanes. Under independent, identically distributed batch scores and stable hardware/runtime behavior, the sample maximum is a one-sided, distribution-free upper tolerance bound covering at least 95% of the population of future healthy-repeat batches with nominal 96.25% confidence (1−0.95⁶⁴). Scheduling correlations across batches and later platform changes can invalidate that extrapolation. The guarantee concerns one future batch drawn from that population. It does not supply simultaneous error control across multiple future gate phases or excuse a changed implementation.

## Acceptance and limits

Retain all rows, including falls. Any nonfinite measurement or nonzero healthy correction invalidates the calibration. Report all terminal-step and fall discrepancies. A calibrated loss envelope at or above the existing absolute degradation criterion of 0.03 (m/s)² cannot resolve the intended effect and requires a protocol decision. Publish the entire envelope and all seeds before looking at fresh identity or recovery trajectories. A finite numerical envelope is an empirical property of this simulator/backend. The maximum raw actuator-force metric is a pre-joint-limit diagnostic and supplies no torque-safety certification.

After cross-review, Prasuna can decide whether the measured envelope is acceptable. A separate held-out Stage A run would retain zero requested correction on healthy/identity, exact same-state action-forwarding tests, finite episodes, equality of discrete termination/fall outcomes, all original degradation/recovery/healthy criteria and unchanged 32-screen/128-confirm sample counts. Replace only the mixed-unit aggregate continuous-identity check with separate physical envelopes evaluated on pairs distinct from the calibration. Maintain independent untouched confirmation seeds; the earlier eight diagnostic screen scenarios are exposed. The healthy-repeat probe must also fit its independently frozen envelope. A too-wide envelope, discrete discrepancies or a repeat outside the frozen envelope requires a recorded modification or inconclusive result. Candidate-method comparisons and G2 use different data.

## Run budget and evidence

One calibration attempt, ceiling 45 wall-clock minutes from the start of pod setup, including 10 minutes reserved for result publication, verification and shutdown. The previous pod diagnostic used 200.9 seconds including setup and evaluation; its warmed **8-lane** rollouts took about 1.8 seconds per arm. Runtime at 32 lanes is unmeasured on this GPU. Preserve complete batches on timeout, publish partial evidence marked incomplete, and stop the pod; no tolerance is fitted to partial coverage and no retry is authorized by this manifest.

The launcher profile `calibrate-identity` uses the existing verified publication and RunPod stop workflow. Its 45-minute ceiling is checked in the public launcher, inner shell and worker. The worker reserves ten minutes for publication and stop before launching the child. The result summary must be labeled inconclusive with all 64 batches and 4,096 rows; a partial attempt is preserved and cannot establish an envelope. A platform, credential or shutdown failure can leave a pod running and should be monitored by the operator, as in the prior diagnostic. Source-pinned launcher use after review and explicit spend approval:

```bash
# Set ref to the approved full Git commit from the PR.
curl -fsSL "https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/$ref/scripts/start_adaptive_gate.sh" -o /tmp/oaa-calibration.sh
FORESIGHT_CODE_REF="$ref" ADAPTIVE_MAX_MINUTES=45 FORESIGHT_AUTO_STOP=1 bash /tmp/oaa-calibration.sh calibrate-identity
```

The calculation runs with JAX float32 and the same matmul precision and auto XLA deterministic-ops setting as the previous pinned GPU diagnostic; the Warp physics replay remains nondeterministic. The runner validates the GPU model and driver, six physics/training runtime versions, checkpoint SHA-256 and the unchanged gate, actuator, environment, rollout and checkpoint paths. The lock for the full setup is recorded by the standard preflight. The actual CPU wiring check uses the same calculating entry with `--cpu-dry-run`, one 32-lane batch, eight control ticks and onset at zero so post-onset loss is exercised. The prior 72-episode, full-horizon CPU smoke exercised the original rollout.

The checked CPU result had 64 episode rows, zero terminal disagreements and zero differences in every recorded paired field; the calibration loop took 24.53 seconds after model, policy and compiler setup began. These eight-tick checks do not quantify a CUDA envelope. [Dry-run manifest, summary, report and paired rows](../../../results/stage_a_identity_calibration_cpu_20261122)

[GPU smoke and archived result files](https://github.com/pchatla-arch/anticipatory-balance/tree/b77606fa1e25671f7f79877262ba5eb2a17dcec2/results/adaptive_gate_20261122T082123Z_1bfe6c) · [Identity protocol options](https://github.com/pchatla-arch/anticipatory-balance/issues/32) · [Pinned evaluator](https://github.com/pchatla-arch/anticipatory-balance/pull/29)
