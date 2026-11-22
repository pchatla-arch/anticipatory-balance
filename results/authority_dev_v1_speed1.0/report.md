# Authority study (offline replay of recorded packets)

Traces: `results/bench_dev_packets_v0`; drift bound 0.05/s, speed bound 1.0 rad/s, prior [0.3,1.2], available force 80.0 N.m, hold 20 ms. Inverse candidate = set midpoint.

| Family | Telemetry | Steps to exclude g=1 (median) | Never excluded | Mean set width | Permitted/oracle | Requested/oracle | Zero-authority steps | Inconsistent | Bound violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gain_055 | declared | 4.0 | 0.12 | 0.252 | 0.33 | 0.39 | 0.44 | 41 | 3 |
| gain_055 | exact | 1.0 | 0.00 | 0.014 | 0.53 | 0.54 | 0.21 | 48 | 0 |
| gain_070 | declared | 4.0 | 0.10 | 0.246 | 0.66 | 0.71 | 0.44 | 44 | 9 |
| gain_070 | exact | 1.0 | 0.00 | 0.009 | 1.02 | 0.89 | 0.18 | 48 | 0 |
| gain_085 | declared | 6.0 | 0.15 | 0.261 | 0.46 | 0.45 | 0.45 | 28 | 5 |
| gain_085 | exact | 1.0 | 0.00 | 0.010 | 1.42 | 0.95 | 0.13 | 48 | 0 |
