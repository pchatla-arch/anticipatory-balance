# Session H experiment report

Status: **partial**. Run `session_h_20261115T071536Z_b46b85`; source `450c834ffa17c6cffd00b4303eeb194e2a3c6242`.

## Stage completion

| Stage | Status | Evidence |
| --- | --- | --- |
| depth_gate_l0 | failed | incomplete or unaudited |
| depth_gate_l5 | not_started | incomplete or unaudited |
| hold_gate | not_started | incomplete or unaudited |
| E18b_hold_0p3 | not_started | incomplete or unaudited |
| E18b_hold_0p6 | not_started | incomplete or unaudited |
| E18b_hold_1p0 | not_started | incomplete or unaudited |
| E17c_depth_lat0 | not_started | incomplete or unaudited |
| E17c_depth_lat100 | not_started | incomplete or unaudited |

## Primary outcomes

Preparation time below excludes intervals after termination. The raw legacy duration is retained in the stage files.

| Stage | Source | Method | Parameter | Delay (steps) | Hit failures | Contact on hits | Braced time (s) | Progress (m) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |

## Paired hold effects

Effects are longer hold minus 0.3 s on the same scenario IDs. Negative failure difference favors the longer hold.

| Hold stage | Method | Parameter | Delay | Failure difference | Approximate 95% interval | Added brace time (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: |

## Interpretation limits

- These runs test execution hold and the parked-object correction while keeping the Session G controller and fitted response fixed.
- A longer execution hold is a response-model transfer condition; its cost/response fit is not recalibrated. The study does not identify an optimal controller.
- Prelaunch false detections are counted per capture frame. False preparation is counted per near-miss episode. They are different quantities.
- The scene is simulated; radius and robot/camera pose assumptions remain. Render wall time is separate from the configured simulated delay.
- Gate data and partial runs are operational diagnostics. A failed gate does not mean a method failed scientifically.
- Scenario uncertainty for one skill pair does not measure population variation across training seeds.
