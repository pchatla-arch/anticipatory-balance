# Two paired G1 encounters

[Watch the 12-second replay](paired-encounters.mp4) · [Still frame at 3.04 s](paired-encounters.jpg) · [Episode record](example_record.json) · [Joint-position and controller traces](example_traces.npz)

A time-to-contact (TTC) threshold and a lead-conditioned belief rule control the same frozen walking and bracing skills. Each sees the same scheduled encounter, launch, reset and detector random keys. Their physical trajectories and subsequent detections can diverge after a controller changes its action. The two scenarios, one scheduled hit and one scheduled near-miss, were specified before the run with seed 73132. Both clips last six seconds in simulation time.

| Encounter | Controller | First registered contact | Applied brace through termination or horizon | Outcome |
|---|---|---:|---:|---|
| Scheduled hit | TTC threshold, 0.30 s | 3.14 s | 0.62 s | Fall at 3.80 s |
| Scheduled hit | Lead-conditioned belief, preparation parameter 0.05 | 3.12 s | 0 s | Fall at 3.50 s |
| Scheduled near-miss | TTC threshold | None | 0 s | Upright through 6 s |
| Scheduled near-miss | Lead-conditioned belief | None | 0 s | Upright through 6 s |

The hit illustrates that starting a brace can still end in a fall. The near-miss shows both policies continuing to walk. Population-level comparisons use the separate [operating-conditions analysis](../../../analysis/operating_conditions/README.md).

## What the camera shows

The film uses a fixed external MuJoCo observer camera for both panels of each encounter. The controllers received synthetic detections from a parametric model with a 120° field of view, 100 ms update period and 100 ms latency. The observer view renders the recorded 43-coordinate pose with the G1 meshes and the original scene's orange projectile. Off-screen launches and passes are possible. The encoder compresses color, while physics and event labels come directly from recorded arrays.

WALK and BRACE name the skill applied at that control step. FALL is the simulator's first recorded fall termination. Each panel holds its exact terminal pose independently for the remaining video time. The clock shows elapsed encounter time. The contact readout retains the first contact time, and brace duration counts applied 20 ms commands through the terminal step. The 50 Hz recorder writes the state *after* physics advances. Row 0 is 0.02 s, and zero-based row 189 is 3.80 s. The 25 fps video samples every second recorded state, clamping to the actual terminal row.

## Reproduce

From the project root in the course environment, render the existing traces:

```bash
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_foresight_example.py
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 docs/foresight/replay/paired-encounters.mp4
```

This needs the project's G1 scene, the rss2025 Playground checkout and its G1 Menagerie assets, native MuJoCo with an EGL renderer, NumPy, Pillow, FFmpeg with libx264 and the DejaVu Sans / DejaVu Sans Mono fonts. Use `--font-directory` for another local installation of those same fonts. The rendered version uses native MuJoCo 3.16.0. The trajectories were recorded with MJX/JAX on CPU. The clip has 300 frames at 25 fps, 1060 × 492 pixels, and no audio.

To rerun the original paired rollouts with the two checkpoints and fitted response model, write a fresh evidence directory:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/record_foresight_example.py --out /tmp/foresight-example
MUJOCO_GL=egl PYTHONPATH=. python scripts/render_foresight_example.py --directory /tmp/foresight-example
```

The recorder uses `nominal_s1/000202342400` and `brace2_s1/000202342400` under `checkpoints/`, `results/response_model.json`, and the rollout helper's optional joint-position trace. It fixes projectile position and velocity from a shared reference run. Each controller then sees the frozen scenario. The original trace archive has SHA-256 `3b160a4bec7cc4015c3de8374fbdff3590d715a290a8394f63b400a2dab702ac`; the episode record has SHA-256 `639f4604b42c6ff3d9d846d9a5737acd834b919ee6f8a265edb1a0c0f090ca24`.
