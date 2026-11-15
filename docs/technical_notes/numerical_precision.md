# Numerical precision in Foresight

Technical notes · 15 November 2026

Discussion and follow-up: [issue #6](https://github.com/pchatla-arch/anticipatory-balance/issues/6).

Related reference: [controllers, frozen skills, and bracing decisions](controllers_and_freezing.md).

**Scope:** the implementation reviewed at [commit `67abf97`](https://github.com/pchatla-arch/anticipatory-balance/tree/67abf97ed6b88bd51241b45b927d6356cdfa1cac). These notes explain the Session H reporting incident and propose a separate investigation of lower-precision inference. They do not report a completed quantization experiment or a completed full Session H run.

## 1. What the Session H failure actually meant

The first uploaded Session H run completed its small depth rollout, then stopped when the evidence validator compared a reported mean with the saved episode values. The two calculations differed in their accumulation precision:

| Calculation | Mean progress |
| --- | ---: |
| FP32 reduction of reactive-policy progress | 1.408173680305481 m |
| FP64 reduction of the same saved samples | 1.4081736113876104 m |
| Absolute difference | 0.00000006891787052154541 m |

The [validator](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/validate_session_h.py) uses `math.isclose` with relative tolerance $10^{-8}$ and absolute tolerance $10^{-9}$. Its acceptance rule is

$$
|a-b|\leq\max\left(10^{-8}\max(|a|,|b|),10^{-9}\right).
$$

At a mean near 1.408 m, the allowable difference is approximately $1.41\times10^{-8}$ m. The observed difference exceeded that reporting tolerance. This was a consistency check between two computations of the same evidence; it did not establish a physical accuracy requirement for the humanoid.

NumPy normally reduces a floating-point array using its input precision unless an accumulator dtype is supplied. Converting an already computed FP32 mean to Python `float` preserves the rounded result. The [correction](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_depth_closed_loop.py) therefore specifies the accumulator:

```python
progress_mean = float(np.mean(progress_samples, dtype=np.float64))
```

This changes the host-side summary calculation. The checkpoint, policy actions, simulated trajectories, and saved episode samples remain unchanged. Promoting those samples prevents additional reduction rounding; it cannot recover precision lost when the original samples were produced. See the [NumPy mean documentation](https://numpy.org/doc/stable/reference/generated/numpy.mean.html).

The recorded run was `session_h_20261115T071536Z_b46b85`, produced by source `450c834ffa17c6cffd00b4303eeb194e2a3c6242`. Its [original uploaded evidence](https://github.com/pchatla-arch/anticipatory-balance/tree/6f60c8ed2ff546f59a35f64f7c5b9c771724b6ec/results/session_h_20261115T071536Z_b46b85) remains a failed, partial run. Recomputing only pooled progress made all nine conditions pass the unchanged validator locally. The [diagnostic record](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/analysis/session_h_review/depth_progress_precision_fix.json) documents that reanalysis. It does not establish results for the remaining GPU stages.

## 2. Separate three kinds of accuracy

**Evidence consistency** asks whether summaries faithfully reproduce recorded episodes. Counts, scenario identities, configuration hashes, and numeric reductions belong here. A narrowly defined reduction convention makes a strict audit useful.

**Numerical fidelity** asks how computation changes when representations or kernels change. Examples include differences in action vectors, covariance matrices, and estimated collision probabilities.

**Control performance** asks whether those changes matter to behavior: falls, missed braces, unnecessary preparation, recovery, progress, and decision timing.

A policy can tolerate numerically different outputs and retain its performance. Conversely, a small average numerical error can conceal a consequential error near one decision boundary. Therefore, the $10^{-8}$ summary tolerance should not become an acceptance threshold for a quantized policy. That experiment needs behavioral criteria and uncertainty estimates of its own.

## 3. “16-bit computation” is several separate choices

Consider a network layer $y=Wx+b$. Its implementation can choose different representations for:

1. Stored weights and activations.
2. Multiplication operands and products.
3. The accumulator that sums products.
4. Bias, normalization, nonlinearities, and the stored output.

For example, FP16 operands can participate in a matrix operation with FP32 accumulation. That does not make the whole layer equivalent to FP32: the operands were already rounded. Similarly, an FP32 output cannot undo earlier clipping of an INT8 activation.

JAX normally disables 64-bit array creation unless X64 is enabled. Its matrix-operation precision and preferred result type are separate controls. Backend lowering must be checked rather than inferred from a checkpoint filename or an array's storage dtype. The reporting fix uses host NumPy and does not require enabling X64 across the GPU simulation. See [JAX default dtypes](https://docs.jax.dev/en/latest/101/default_dtypes.html) and [`jax.lax.dot_general`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dot_general.html).

| Format | Representation | Potential benefit | Main consideration |
| --- | --- | --- | --- |
| FP32 | 32-bit floating point | Established baseline with useful range and resolution | More tensor storage than narrower formats |
| FP16 | 16-bit floating point; 5 exponent and 10 fraction bits | Half FP32 raw storage; accelerated supported kernels | Limited range; overflow and small-value underflow |
| BF16 | 16-bit floating point; 8 exponent and 7 fraction bits | FP32-like exponent range with half the raw storage | Coarser resolution than FP16 around a normal value |
| INT8 | 8-bit integer plus scale, and sometimes zero point | Roughly one-quarter FP32 raw tensor storage | Rounding, clipping, calibration, and backend support |
| FP8 | Several 8-bit floating-point formats | Compact tensors and supported accelerated kernels | Format-specific range, scaling, and accumulation choices |
| FP64 | 64-bit floating point | Accurate aggregation and selected numerical references | Additional storage and potentially expensive device arithmetic |

These are tensor-storage comparisons. Scale metadata, unchanged FP32 layers, workspace, and simulator state reduce the overall memory saving. Supported operations and hardware determine execution speed. NVIDIA distinguishes rounding and clipping errors and recommends selecting precision by operation sensitivity; see [accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html).

Near 1, the spacing above 1 is $2^{-10}\approx0.000977$ for FP16 and $2^{-7}=0.0078125$ for BF16. Thus BF16's wider range does not give it finer local resolution. Relative spacing is also different from a bound on the error of an entire network.

## 4. Why small actor errors can matter in closed loop

A lower-precision actor produces an action perturbation $\eta_t$ relative to its reference. Within a locally differentiable dynamics regime, a first-order model is

$$
e_{t+1}\approx(A_t+B_tK_t)e_t+B_t\eta_t,
$$

where $e_t$ is the state difference, $A_t$ and $B_t$ describe local dynamics, and $K_t$ describes the policy's response to changed state. The next error depends on both the new action perturbation and previous state differences.

For illustration, if the local closed-loop operator has norm at most $\rho<1$, $\|B_t\|\leq b$, and $\|\eta_t\|\leq\epsilon$, then

$$
\|e_t\|\lesssim\rho^t\|e_0\|+
\frac{b\epsilon(1-\rho^t)}{1-\rho}.
$$

This conditional model explains how a stable controller can attenuate repeated small perturbations. It is not a stability certificate for Foresight. Foot contact changes, impacts, saturation, and skill switches can invalidate the local approximation. Paired physical rollouts remain necessary.

A useful comparison measures typical action error, tail error, saturation, and closed-loop outcomes. Demanding identical long trajectories is generally inappropriate when even small action differences change subsequent contacts.

## 5. Decision margins and timing in the current implementation

A generic decision margin can be written as

$$
M(b)=V_{\mathrm{brace}}(b)-V_{\mathrm{continue}}(b).
$$

Here $b$ denotes the belief, and the values are a conceptual representation of the particular rule's benefit-versus-cost comparison. If numerical approximation changes this margin by $\delta M$, a sufficient condition for its sign to remain unchanged is

$$
|\delta M|<|M(b)|.
$$

This identifies a useful diagnostic: how often does quantization change the decision when the reference margin is small? It does not assume that every changed decision causes a fall.

The implementation's rates must be distinguished:

| Quantity | Current Session H setting |
| --- | ---: |
| Robot control interval | 20 ms |
| Controller/belief/rule invocation | Every control step |
| Perception period | 5 control steps = 100 ms |
| Minimum BRACE commitment | 15 control steps = 300 ms |

The [controller](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/decision/controller.py) predicts and evaluates its rule between detections. Shifting brace onset by one control step changes its timing by 20 ms; waiting for the next observation follows a 100 ms cadence. Calling both rates “10 Hz decision-making” would misdescribe this code. The commitment and release behavior are defined in [the decision rules](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/decision/rules.py).

The belief code also contains covariance regularization such as

```python
L = jp.linalg.cholesky(cov + 1e-9 * jp.eye(6))
```

If this addition were actually performed wholly in FP16, $10^{-9}$ would round to zero: it is below FP16's smallest positive subnormal, approximately $5.96\times10^{-8}$. The intended regularizer would disappear. Actual promotion and backend support must be inspected; this is a reason to preserve belief arithmetic in FP32 rather than blindly cast the pipeline. See the [belief implementation](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/foresight/belief/particle_filter.py).

## 6. An illustrative INT8 action bound

Suppose a final normalized action lies in $[-1,1]$ and uses symmetric integer codes from $-127$ to $127$:

$$
\hat a=s\,\operatorname{clip}\left(\operatorname{round}(a/s),-127,127\right),
\qquad s=\frac{1}{127}.
$$

With round-to-nearest and no clipping, $|\hat a-a|\leq s/2=1/254$. If the target mapping is $q_{\mathrm{target}}=q_{\mathrm{default}}+0.5a$ radians, then

$$
|\Delta q_{\mathrm{target}}|\leq\frac{0.5}{254}
\approx0.00197\ \mathrm{rad}\approx0.113^\circ.
$$

This is a bound for quantizing the **final action alone**, under the stated range and scaling. It does not bound errors introduced by quantized network weights, hidden activations, clipping, or subsequent closed-loop dynamics. Keeping the action output in FP32 avoids imposing this particular final quantizer.

Post-training quantization estimates suitable scales from calibration data. A small scale preserves resolution but clips more outliers; a large scale covers outliers with coarser steps. Per-channel scales can help when channels have different ranges. Calibration should include walking, transitions, impacts, recovery, and observation disturbances. See [NVIDIA's quantized-type documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html).

## 7. Depth precision is a separate design choice

The present rendered-depth path uses a geometric detector. A future learned depth detector would require its own precision benchmark. A depth-storage experiment would change the current geometric detector's inputs and therefore needs its own observation-model assessment. See the [depth evaluator](https://github.com/pchatla-arch/anticipatory-balance/blob/67abf97ed6b88bd51241b45b927d6356cdfa1cac/scripts/eval_depth_closed_loop.py).

For a hypothetical uniform encoding spanning 0–10 m, with both endpoints represented:

| Encoding | Distance between adjacent codes |
| --- | ---: |
| UINT8 | $10/255\approx0.0392$ m = 3.92 cm |
| UINT16 | $10/65535\approx0.000153$ m = 0.153 mm |

These are encoding intervals, not sensor accuracy claims. Floating-point depth, integer depth, and an INT8 actor have different error mechanisms. Depth rounding can perturb estimated motion, especially when velocities are inferred from successive positions. Calibration should account for that changed observation distribution.

## 8. Training and inference need different treatment

Inference uses a fixed network. PPO training adds gradients, optimizer state, returns, advantages, probability ratios, and repeated updates. Lower precision can affect training even when a fixed actor's forward pass looks acceptable.

Mixed-precision training commonly combines lower-precision operations with higher-precision accumulation or parameter updates; FP16 training may require loss scaling to preserve small gradients. BF16's wider range reduces some range problems, while its coarser significand remains relevant. The original [Mixed Precision Training paper](https://arxiv.org/abs/1710.03740) describes the motivations and mechanisms. [NVIDIA's mixed-precision guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) provides implementation considerations.

A deployment experiment should initially quantize a copy of the frozen actor. Retraining, quantization-aware training, and changing the simulator at the same time would introduce additional explanations for any observed difference.

## 9. Measure latency at the correct level

Faster matrix operations help only the fraction of runtime they accelerate. If actor inference is 10% of a measured pipeline and becomes four times faster, an idealized Amdahl calculation gives

$$
S=\frac{1}{(1-0.10)+0.10/4}\approx1.081.
$$

That is about 8% end-to-end speedup, before conversion or synchronization overhead. This is an example, not a Foresight measurement. Profile actor inference, rendering, belief updates, physics, and host/device transfers before choosing an optimization target.

Distinguish single-environment deployment latency from large-batch simulation throughput. Also distinguish wall-clock execution time from the evaluator's configured simulated observation delay. Accelerating inference does not automatically reduce `extra_latency` or move a simulated detection's delivery timestamp. A timing experiment must explicitly model that relationship and measure it on the intended hardware.

## 10. Proposed precision study

The following is a proposed sequence, not work already executed:

| Stage | Change | Evidence required |
| --- | --- | --- |
| Reference | Preserve frozen FP32 checkpoints and current physics | Reproducible paired outcomes and component timings |
| Forward-pass screening | Derive FP16/BF16 actor copies | Finite outputs, action-error distribution, saturation, compiler settings |
| Closed-loop comparison | Replace only actor inference | Falls, progress, false preparation, onset/release timing, tail cases |
| INT8 candidate | Calibrate supported weights/activations | Separate calibration data; clipping and scale diagnostics |
| Held-out evaluation | Freeze conversion choices | Paired scenarios outside calibration, latency/noise conditions, uncertainty intervals |
| Optional training study | Investigate QAT or mixed-precision PPO separately | Training stability, seed variation, and retained baseline evidence |

Predeclare acceptable behavioral degradation. Compare paired outcomes and report uncertainty; a small average action error alone is insufficient. Preserve original checkpoints, conversion configuration, calibration identifiers, precision settings, software versions, and hardware information with each run.

For the first study, retain this allocation:

| Component | Recommended starting precision |
| --- | --- |
| Physics and contact simulation | Existing MuJoCo/Warp configuration |
| Belief covariance, geometry, probabilities, decision rules | FP32 |
| Step counts and logical events | Integer and Boolean types |
| Frozen actor | FP32 reference; separately evaluated mixed-precision candidates |
| Saved episode values | Original recorded precision |
| Aggregates and statistical analysis | FP64 host accumulation |

FP64 reporting improves consistency of calculations over recorded evidence. It does not make the sensor, simulated world, learned policy, or statistical estimate physically accurate to fifteen decimal places. Lower-precision deployment is worthwhile when measured system benefits justify measured behavioral changes; the Session H rounding incident itself establishes neither benefit nor harm from quantizing a controller.
