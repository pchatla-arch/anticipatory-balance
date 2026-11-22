# Foresight release rule and rendered-depth evidence

## The decision applied at each interval

In Session G’s sensor-driven policies, the high-level controller selects one frozen skill for the next 20 ms. Let $w_k$ denote the current policy’s request to **start** BRACE, $b_{k-1}$ the previous applied mode, and $n_{k-1}$ the number of consecutive preceding BRACE intervals. Let $d_k$ be the step of the most recent valid detection **delivery**, including any valid delivery at step $k$. While the filter is initialized, its estimated signed time until closest horizontal approach is

$$
s_k=\frac{(x_{\mathrm{pelvis}}-\hat p)_{xy}^{\top}
                 (\hat v-v_{\mathrm{cmd}})_{xy}}
                {\|(\hat v-v_{\mathrm{cmd}})_{xy}\|^2+10^{-6}} .
$$

The object position and velocity are the Kalman mean in world coordinates. The robot velocity proxy is the current commanded forward speed along the horizontally projected pelvis heading, with zero vertical component. Before filter initialization $s_k=+\infty$. A positive value forecasts closest approach in the future. The start rules use predicted entry into a contact cylinder, a different event. The applied mode satisfies

$$
\begin{aligned}
P_k &= \operatorname{finite}(s_k)\;\land\;s_k<-0.30\ \mathrm{s},\\
S_k &= k-d_k>50,\\
b_k &= (\neg b_{k-1}\land w_k)\;\lor\;
 [b_{k-1}\land(n_{k-1}<15\;\lor\;(\neg P_k\land\neg S_k))] .
\end{aligned}
$$

The start interval counts as one. A start at step $a$ commits BRACE on $a,\ldots,a+14$, and the first possible WALK interval is $a+15$. With a valid delivery at $d$ and no further observations, staleness first allows release at $d+51$ (1.02 s). At exactly −0.30 s the passage comparison is false. A fresh delivery replaces $d$ before this check. After release, a new start request can engage BRACE on the next step. Passage uses the tracker's current prediction, which need not agree with actual contact time. [Release and memory update](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/decision/rules.py#L150-L173) · [Closest approach and contact entry](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/belief/particle_filter.py#L126-L160) · [Commanded velocity and applied skill](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/eval/rollout.py#L203-L232)

Three different time parameters appear in the experiments.

| Parameter | Value for main comparison | Role |
| --- | --- | --- |
| Minimum commitment | 0.30 s | Sets the shortest uninterrupted BRACE segment in a continuing episode. A termination or end of horizon may censor the duration. |
| Passage margin | 0.30 s | Sets the strict closest-approach test above. Estimated time to passage is recomputed every control step. |
| Hold in the belief cost | 0.50 s | Enters the estimated cost of starting now through $c_{\rm prep}(t+0.50)$. It sets no release deadline. |

The privileged fixed-lead reference follows a separate schedule on intended hits. BRACE is active for control times in the inclusive interval from the nominal arrival minus the lead to the nominal arrival plus the oracle hold. The standard reference uses 0.50 s on either side. It never braces on an intended near-miss. Calling that particular reference an oracle describes its information access. It does not optimize all possible switching schedules. [Cost calculation](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/decision/rules.py#L89-L111) · [Oracle interval](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/eval/rollout.py#L171-L181)

## Which historical results share the rule

| Archive | Release implemented in that version |
| --- | --- |
| C/D including the first cadence sweep | After commitment, a withdrawn start request releases BRACE. Staleness can also release it. The passage test used a future-hit entry statistic that could never cross its negative threshold. [Original switch logic](https://github.com/pchatla-arch/anticipatory-balance/blob/d53bbf84d7e5f13b3401822e66797c8280939f2c/foresight/decision/rules.py#L88-L104) |
| E/F including E15, E16 and E17 | Default continuation ignores withdrawal of the start request. Its negative-entry-time passage test remains unreachable, so the effective exit after commitment is staleness. E13's `release_old` ablation explicitly restores release on request withdrawal. [Session E code](https://github.com/pchatla-arch/anticipatory-balance/blob/1faf09d298fbfc17cb3010976253a3c04087cf71/foresight/decision/rules.py#L94-L110) · [Session F code](https://github.com/pchatla-arch/anticipatory-balance/blob/ef795ab8371d5761382715ce6290f62067e97f45/foresight/decision/rules.py#L136-L153) |
| G including E17b, E18, E18h and E19 | Uses the signed closest-approach/age rule above for the threshold, belief, blind-wait and observation-lookahead gates. For zero-delay TTC at 0.3 s in E18, 1,657 of 1,660 ever-braced episodes recorded passage as their final release reason and three recorded staleness. This is an episode count of final recorded reasons. [Session G evidence](../SESSION_G_REVIEW.md) |
| Prospective E20 | The protocol fixes the same 0.30 s margin and 15-step commitment for both rules. Its metric counts BRACE intervals through termination inclusively. Historical G counted requested intervals across the fixed horizon, potentially extending past termination. [Published E20 protocol](https://github.com/pchatla-arch/anticipatory-balance/blob/3b91140c606d474f66042e0b6b6bdc0859087cb2/scripts/run_foresight_matched_duration.py#L16-L57) |

In E17b both sensor arms call the same external-detection controller update, which invokes the same switching wrapper as E18's internal parametric-detector update. Differences in observations can still produce different release times and applied durations. [E17b controller selection](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/scripts/eval_depth_closed_loop.py#L130-L139) · [The two calls to the common wrapper](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/decision/controller.py#L58-L83)

These implementation changes and the distinct scenario/observation settings prevent attributing an E-to-G performance change solely to hold duration. Three CPU regression tests call the production JAX wrapper and belief implementation to check minimum commitment, strict passage and staleness boundaries, treatment of a new delivery, re-entry, and signed closest approach. Run `python -m unittest discover -s tests -p test_foresight_release_contract.py` in the project environment. [Regression tests](../../tests/test_foresight_release_contract.py)

## What the depth stages establish

| Stage | Evidence and interpretation |
| --- | --- |
| E8, detector calibration | 12,800 rendered frames. Of 3,653 with an eligible visible sphere centre, 3,451 produced a detection (94.5%). Among 9,147 other frames, 458 produced a detection (5.0%). Eligibility uses simulator geometry. The latter frames can include a partially visible or parked sphere. These counts describe one rendered scene and detector. [Calibration record](../../results/E8_depth_detector/summary.json) |
| E17, developmental closed loop | 512 scenarios per condition at 0 and 100 ms. The detector used an arithmetic mean and suppressed rendered detections when the simulator's projectile-launch flag was false. Sensor timing and switching also differ from E17b. [Development implementation](https://github.com/pchatla-arch/anticipatory-balance/blob/ef795ab8371d5761382715ce6290f62067e97f45/scripts/eval_depth_closed_loop.py#L145-L170) |
| E17b, supporting simulation experiment | Four sensor-driven rules at both added delays, each using two batches of 256 paired scenarios (367 intended hits, 145 intended near-misses). The reference walker was also evaluated. A 96 × 96 simulated head camera supplies frames at 10 Hz, back-projected with exact camera and robot pose. The detector applies world-height and pelvis-range filters, takes coordinate-wise medians, and shifts by the known 0.15 m sphere radius. A valid measurement is queued with its capture time. Perception drives switching online. [Rendered detector](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/scripts/eval_depth_closed_loop.py#L60-L73) · [Delivery and switching](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/scripts/eval_depth_closed_loop.py#L152-L185) |
| E17c, out-of-range parking | On the later H attempt, the 0- and 100-ms execution checks each passed for 16 scenarios per condition. The independent hold check failed and the run stopped before both full depth stages and the three full hold stages. These small checks establish execution only. [H run status](https://github.com/pchatla-arch/anticipatory-balance/blob/df8d5ee99036b243d62653d1c025ea2898ade16f/results/session_h_20261115T211850Z_1f77c1/report.md) |

E17b's parametric comparator samples a noisy true position, gates on the simulator's capture-time launch flag and a horizontal cone around the pelvis heading, and adds a 5.5% random dropout. The image path can detect a parked object, uses the head-camera frustum and actual visible pixels, and has geometry-dependent misses and centre error. The recorded prelaunch positive rate was approximately 21% per capture frame. A positive frame can contain a correctly detected parked sphere even though the evaluation has labeled the prelaunch time a false alarm for an incoming hazard. The summaries lack pooled frame counts, pixel labels and per-detection source attribution. They cannot establish what fraction of positives the parked object caused. Differences across these arms include all of those changes, alongside any resulting change in how often the controller braces. Render wall time measures computation separately from configured simulated delay. [Parametric configuration](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/scripts/eval_depth_closed_loop.py#L101-L106) · [Parametric validity](https://github.com/pchatla-arch/anticipatory-balance/blob/65d753f9484ed2b480697f62640376f9bcd8e6d1/foresight/perception/detections.py#L63-L79) · [Recorded limitations](../SESSION_G_REVIEW.md)

For the fixed 0.30 s TTC rule, the following counts are pooled from both batches. The mean preparation column uses G's recorded duration.

| Added delay | Source | Failures / 367 intended hits | Contact / 367 intended hits | Failures / 145 intended misses | Prepared / 145 intended misses | Mean preparation (s) | Mean progress (m) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 ms | Parametric | 113 | 338 | 0 | 33 | 0.585 | 1.442 |
| 0 ms | Rendered depth | 131 | 339 | 0 | 24 | 0.484 | 1.433 |
| 100 ms | Parametric | 124 | 339 | 0 | 74 | 0.622 | 1.417 |
| 100 ms | Rendered depth | 135 | 339 | 1 | 48 | 0.495 | 1.432 |

The primary controller budget comparison uses parametric detections. E17b supports a separate simulated depth-to-action demonstration with known geometry, scenario-specific false detections and unmatched durations. A depth-mediated improvement in failure, a clean estimate of the effect of position noise, and performance with a physical camera remain open questions. [Zero-delay arrays](../../results/E17b_depth_lat0/episodes.npz) · [100-ms arrays](../../results/E17b_depth_lat100/episodes.npz)
