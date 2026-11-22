# Authority study (offline replay of recorded packets)

Traces: `results/bench_dev_packets_v0`; drift bound 0.0/s, speed bound 10.0 rad/s, prior [0.3,1.2], available force 80.0 N.m, hold 20 ms. Inverse candidate = set midpoint.

| Family | Telemetry | Steps to exclude g=1 (median) | Never excluded | Mean set width | Permitted/oracle | Requested/oracle | Zero-authority steps | Inconsistent | Bound violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gain_055 | declared | 4.5 | 0.12 | 0.222 | 0.03 | 0.03 | 0.92 | 42 | 3 |
| gain_055 | exact | 1.0 | 0.00 | 0.011 | 0.24 | 0.30 | 0.58 | 48 | 0 |
| gain_070 | declared | 4.0 | 0.10 | 0.202 | 0.01 | 0.02 | 0.96 | 44 | 9 |
| gain_070 | exact | 1.0 | 0.00 | 0.005 | 0.44 | 0.52 | 0.62 | 48 | 0 |
| gain_085 | declared | 6.0 | 0.15 | 0.216 | 0.00 | 0.00 | 0.99 | 42 | 5 |
| gain_085 | exact | 1.0 | 0.00 | 0.006 | 0.38 | 0.37 | 0.71 | 48 | 0 |
