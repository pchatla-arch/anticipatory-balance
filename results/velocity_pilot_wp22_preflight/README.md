# WP 2.2 CPU development pilot preflight

Prasuna approved the paired CPU development protocol on Nov 22. The scientific run is reserved for Claude’s final development-tuned heuristic. The preliminary plan records the pending comparator explicitly. [G1 decision](https://github.com/pchatla-arch/anticipatory-balance/pull/40#issuecomment-7087805929) · [Operator claim](https://github.com/pchatla-arch/anticipatory-balance/pull/46#issuecomment-7087861171)

The source for this check is `aa51278507b4354cf6cb3a164ffb44b3f9642aa4`. One healthy episode on the separate wiring seed exercised the velocity-envelope controller for eight CPU control ticks. The episode remained finite and recorded zero falls. The independent trace audit reproduced the mean available-source age of 50.00 ms for hips and knees. The healthy episode includes zero scheduled gain restorations. These checks cover simulator and reporting behavior. The 40-cell recovery and intervention comparisons require the scientific run. [Raw episode](wiring/episodes.csv) · [Bench report](wiring/report.md) · [Independent reduction](wiring_audit.json) · [Preliminary plan](preliminary_plan/pilot_manifest.json)

The prepared runtime used JAX 0.12.0.1, MuJoCo/MJX 3.16.0, MuJoCo Playground `9de993826d1a486f18aec4e8d3179e93ff64ed41` and Menagerie `1b86ece576591213e2b666ebf59508454200ca97`. The simulator, trained checkpoint and telemetry configuration are recorded in the run files. The audit script reads the archived traces, manifests and episode rows, and requires Python and NumPy.

```sh
python verify.py wiring --wiring --expected-source aa51278507b4354cf6cb3a164ffb44b3f9642aa4
```
