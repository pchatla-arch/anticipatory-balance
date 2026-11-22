# Development coverage of the velocity-informed hold envelope

Claude’s proposed symmetric envelope covered 99.92% of hip decisions and 95.10% of knee decisions at gain 0.70, under declared telemetry, with an assumed acceleration limit of 100 rad/s². At 400 rad/s² it covered every decision in this first-second cell, but its mean position intervals were wider than the frozen speed-only envelope. This tradeoff determines how promising a setting is for a future controller. Feasible correction is scored separately.

## First second at gain 0.70

Mean interval width spans the full hold. A joint decision counts as covered if its true position lies within the time-specific bound at **both** the decision and the hold end. There were 2,448 observed hip and 816 knee joint decisions across eight walks. All rows use the same observations.

| Envelope | A (rad/s²) | Hip coverage | Knee coverage | Hip width (rad) | Knee width (rad) |
|---|---:|---:|---:|---:|---:|
| Existing speed bound (10 rad/s) | — | 100.00% | 100.00% | 1.269 | 1.269 |
| Symmetric speed magnitude | 25 | 96.98% | 79.41% | 0.190 | 0.314 |
| Symmetric speed magnitude | 50 | 99.39% | 87.99% | 0.294 | 0.419 |
| Symmetric speed magnitude | 100 | 99.92% | 95.10% | 0.504 | 0.629 |
| Symmetric speed magnitude | 200 | 100.00% | 99.26% | 0.923 | 1.048 |
| Symmetric speed magnitude | 400 | 100.00% | 100.00% | 1.761 | 1.886 |
| Symmetric speed magnitude | 800 | 100.00% | 100.00% | 3.437 | 3.562 |
| Symmetric speed magnitude | 1600 | 100.00% | 100.00% | 6.789 | 6.914 |

The signed-velocity alternative shifts the center along the observed velocity. At 100 rad/s² its coverage was 99.39% for hips and 88.85% for knees. At 200 rad/s² its coverage was 100% and 97.55%, with eight knee joint decisions outside the entire hold interval. All acceleration settings and both formulas are in the CSV.

## All observed post-onset decisions

Each channel contributed 32,512 hip and knee joint decisions across three fault gains. The symmetric formula at 400 rad/s² missed three decisions with exact telemetry and one with declared telemetry. The three exact-channel misses were outside the entire hold interval, while the declared-channel miss was outside its time-specific endpoint bound only. Both candidate formulas at 800 and 1600 rad/s² covered all observed endpoints across the six cells. That distinction matters when wider intervals reduce correction.

The existing 10 rad/s formula also exceeded its assumed motion envelope during severe knee faults. At gain 0.55, eight of 1,660 live knee joint decisions had a pointwise violation in each telemetry condition. Eight exact and six declared observations were outside even the full hold hull. All occurred after the first second.

## Motion observed at the control rate

At gain 0.70 during the first second, the 99th percentile of the absolute change in reported joint velocity divided by 20 ms was 92.1 rad/s² for hips and 283.8 rad/s² for knees. The largest observed values were 174.9 and 384.4 rad/s². These are magnitudes of interval-average acceleration, which can hide short peaks at contact. The per-joint CSV includes the count over each proposed limit.

The sensor model passes simulator joint velocity through unchanged in both channels. Ground-truth positions were used solely for scoring. The packet age comes from the latest accepted per-joint source timestamp. Only development seeds, one nominal walker and observed post-onset decisions contribute. Samples within an episode are correlated. The traces contain two control-boundary positions per hold, so empirical endpoint coverage cannot certify a continuous position or acceleration bound.

[Frozen protocol](protocol.md) · [All coverage strata](coverage.csv) · [Measured acceleration by axis](acceleration.csv) · [Scoring program](score_coverage.py)

Run the following from the repository root, with NumPy installed. Replace `experiment` with the directory containing this report.

```sh
python experiment/score_coverage.py results/authority_replay_dev_wp13 /tmp/velocity-coverage
```

The scoring program checks all six archived trace hashes, confirms zero applied correction, matches cached inputs to their original source-indexed packets, and checks that adjacent live transitions share the same position. An independent scalar calculation reproduced 36 gain/channel/group/acceleration cells for the symmetric formula.
