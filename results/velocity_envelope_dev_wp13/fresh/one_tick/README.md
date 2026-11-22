# One-tick transport development follow-up

The one-tick experiment uses the same two newly simulated walks and the same source measurements as the declared transport and immediate-delivery cases. It changes only the transport to a fixed 20 ms delay. Dropout can leave an older measurement in the controller cache. The relevant age below is the latest accepted source at decision time.

## First-second coverage and hold width

The primary symmetric formula uses an assumed acceleration limit of 400 rad/s² and ideal velocity readout. Each joint decision is checked at its start and at all ten 2 ms physics samples in its 20 ms hold.

| Transport | Mean measured age (ms) | Hips passing | Knees passing | Hip mean width (rad) | Knee mean width (rad) |
|---|---:|---:|---:|---:|---:|
| declared | 40.20 | 612/612 | 204/204 | 1.613 | 1.751 |
| one tick | 20.20 | 612/612 | 204/204 | 0.696 | 0.767 |
| immediate | 0.20 | 612/612 | 204/204 | 0.195 | 0.231 |

With bounded ±0.5 rad/s velocity readout error, 816/816 sampled hip and knee decisions passed at 400 rad/s² over the first second.

With bounded ±0.5 rad/s velocity readout error, 2,888/2,888 sampled hip and knee decisions passed at 400 rad/s² over the full observed post-onset period.

Every sampled position, command, loss and termination trace matched both prior physical executions. Source-aligned position, velocity, torque, validity and target matched all common released packets. The timing variant supplies a development diagnostic; its correction efficacy requires a closed-loop test, and any deployment must establish the assumed sensor timing and measurement errors.

[Complete coverage grid](coverage.csv) · [Manifest with packet verification](capture_manifest.json) · [Frozen protocol](protocol.md) · [Original substep audit](../README.md)

Rerun with the same environment as the original substep audit:

```bash
python "$AUDIT/one_tick/capture_one_tick.py" "$PROJECT" "$AUDIT" /tmp/fresh-oaa-one-tick
cmp "$AUDIT/one_tick/coverage.csv" /tmp/fresh-oaa-one-tick/coverage.csv
```
