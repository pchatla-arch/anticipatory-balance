# Fresh G1 substep and packet-timing audit

The same frozen walker took two fresh development walks after a 30% bilateral leg gain loss. Changing packet transport reduced the mean age of the position and velocity available at the next decision from 40.20 to 0.20 ms during the first second after onset. The experiment preserved sensor noise, drops, controls and physical trajectories. At 400 rad/s², the resulting reduction in hold width was large enough to make transport timing a concrete follow-up.

## What the simulator observed

The native simulator integrated ten 2 ms physics steps per 20 ms policy decision. The observer recorded all ten actual joint positions plus the pre-decision position, across 300 control ticks per walk. An independent execution of the original rollout matched all 28 shared trace fields bitwise, including q, controls, observations used for estimation, loss, and termination flags. Each recorded last substep matched the original joint position at that boundary. Both walks completed; the fault began at ticks 124 and 115. A separately executed immediate-delivery arm reproduced every sampled physical trajectory and all scored commands. The 585 source packets released in both arms had identical measurement values, validity flags and source timestamps. Immediate delivery released three additional packets at the end of the horizon.

## Sampled coverage versus interval width

For each of 612 observed hip and 204 knee decisions in the first second, coverage tests the time-specific position interval at the decision and all ten subsequent physics samples. The table uses the memo's symmetric speed-magnitude envelope and ideal velocity readout.

| Transport | Acceleration assumption (rad/s²) | Hips passing | Knees passing | Mean hip width (rad) | Mean knee width (rad) |
|---|---:|---:|---:|---:|---:|
| Declared | 100 | 612/612 | 193/204 | 0.455 | 0.593 |
| Immediate | 100 | 610/612 | 189/204 | 0.071 | 0.107 |
| Declared | 400 | 612/612 | 204/204 | 1.613 | 1.751 |
| Immediate | 400 | 612/612 | 204/204 | 0.195 | 0.231 |

Two of the 15 immediate-delivery knee misses at 100 rad/s² occurred only at interior physics samples. Both of their control endpoints passed. Under the additional simulated velocity-readout error of up to 0.5 rad/s, the 400 rad/s² first-second coverage was unchanged and mean hip/knee widths became 1.684/1.816 rad with declared transport and 0.218/0.252 rad with immediate delivery. The velocity error has an explicit source-indexed uniform distribution; it serves as a sensitivity assumption.

Over the complete observed post-fault period, the 400 rad/s² symmetric envelope passed every sampled decision with declared transport. Under immediate delivery six of 2,166 hip and three of 722 knee decisions failed the time-specific test. All nine were observed immediately after a delivered reading whose encoder error exceeded the declared 6 mrad. They persisted at 800 and 1600 rad/s² because the acceleration term vanishes at zero age. The full hold hull contained the sampled positions for these decisions. Per-time bounds and the enclosing hold interval answer different questions.

These are two uncorrected walks from one seed, one walker and one gain condition. They establish sampled coverage and uncertainty width under two transport conditions. Broader claims require more seeds and changes in plant and sensor conditions. Causal immediate delivery assumes the completed source packet can be read by the next decision. That interface requires measured latency in any deployment.

[Protocol](protocol.md) · [Complete results](coverage.csv) · [Capture manifest](capture_manifest.json) · [Runtime and checkpoint](runtime.json) · [Source-packet pairing](packet_pairing.json)

## Reproduction

Run inside the CPU environment specified in `runtime.json`. Set `PROJECT` to the repository source at `b814702456011974e59d9afe2568096208794b1e`, `PLAYGROUND` to the pinned Playground `rss2025` checkout containing its compatible Menagerie assets, and `AUDIT` to this directory.

```bash
export PYTHONPATH="$PROJECT:$PLAYGROUND"
export JAX_PLATFORMS=cpu JAX_DEFAULT_MATMUL_PRECISION=highest XLA_PYTHON_CLIENT_PREALLOCATE=false
python "$AUDIT/capture_substeps.py" "$PROJECT" /tmp/fresh-oaa-audit
python "$AUDIT/score_substeps.py" /tmp/fresh-oaa-audit
cmp "$AUDIT/coverage.csv" /tmp/fresh-oaa-audit/coverage.csv
```

To replay coverage from the saved archives, copy the two NPZ files and their capture manifest into a scratch directory and run `score_substeps.py` on it. The scorer checks the hashes in the manifest before reading the data.
