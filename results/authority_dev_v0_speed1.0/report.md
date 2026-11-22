# Authority study (offline replay of recorded packets)

Traces: `results/bench_dev_packets_v0`; drift bound 0.05/s, speed bound 1.0 rad/s, prior [0.3,1.2], available force 80.0 N.m, hold 20 ms. Inverse candidate = set midpoint.

| Family | Telemetry | Steps to exclude g=1 (median) | Never excluded | Mean set width | Permitted/oracle | Zero-authority steps | Inconsistent | Bound violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gain_055 | declared | 4.0 | 0.12 | 0.252 | 0.11 | 0.32 | 41 | 3 |
| gain_055 | exact | 1.0 | 0.00 | 0.014 | 0.18 | 0.07 | 48 | 0 |
| gain_070 | declared | 4.0 | 0.10 | 0.246 | 0.24 | 0.35 | 44 | 9 |
| gain_070 | exact | 1.0 | 0.00 | 0.009 | 0.41 | 0.10 | 48 | 0 |
| gain_085 | declared | 6.0 | 0.15 | 0.261 | 0.37 | 0.45 | 28 | 5 |
| gain_085 | exact | 1.0 | 0.00 | 0.010 | 1.00 | 0.11 | 48 | 0 |
