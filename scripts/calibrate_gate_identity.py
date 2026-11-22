"""AE-008 prospective healthy-repeat calibration. The arm labels are interchangeable.

Invocations use the Stage A evaluator's 32-lane Warp rollout. The CPU dry-run
has 8 ticks and one pair of healthy arms; it exercises setup, resets, metrics
and serialization and never sets an acceptance threshold. The calibration
experiment uses 400 ticks and disjoint seeds. Generated files are retained
after every complete batch. No identity, fault or confirmation arm is run.
"""

import argparse
import csv
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

import eval_adaptive_gate as gate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = range(940101, 940165)
BATCH = 32
SOURCE = "fdf5754be8e17071b324c0f49fd8ffaec0eeac15"
FROZEN = {
    "postfault_loss": 0.00040410831570625305,
    "progress_m": 0.02347874641418457,
    "max_abs_actuator_force_nm": 3.024658203125,
}
CALIBRATION_METRICS = {
    "postfault_loss": "postfault_loss",
    "progress_m": "progress",
    "max_abs_actuator_force_nm": "max_force",
}
RETAIN = (
    "term_step",
    "first_nonfinite_step",
    "max_correction",
    "max_slew",
    "bound_sum",
)


def write(path, value):
    gate.write_json(path, value)


def make_config(dry):
    return dict(
        profile="cpu_dry_run" if dry else "gpu_calibration",
        backend="warp",
        seeds=list(SEEDS[:1] if dry else SEEDS),
        batch=BATCH,
        steps=8 if dry else gate.HORIZON,
        gpu_budget_minutes=45,
        publication_stop_reserve_minutes=10,
        scales=FROZEN,
        source=SOURCE,
        gate_decision="none",
    )


def validate_records(arms, dry):
    for values in arms:
        for k in (*CALIBRATION_METRICS.values(), *RETAIN):
            if values[k].shape != (BATCH,) or not np.isfinite(values[k]).all():
                raise ValueError(f"Invalid measured batch {k}")
        if (values["first_nonfinite_step"] != -1).any():
            raise ValueError("Nonfinite physics or policy")
        if np.any(values["max_correction"] != 0) or np.any(values["max_slew"] != 0):
            raise ValueError("Healthy arm issued a nonzero correction")
    score = max(
        float(np.max(np.abs(arms[0][k] - arms[1][k]))) / FROZEN[name]
        for name, k in CALIBRATION_METRICS.items()
    )
    return {
        "score": score,
        "terminal_disagreements": int(
            np.sum(arms[0]["term_step"] != arms[1]["term_step"])
        ),
        "exposure": "eight-tick CPU wiring only" if dry else "full horizon",
    }


def run(args):
    dry = args.cpu_dry_run
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    if dry:
        os.environ["JAX_PLATFORMS"] = "cpu"
    gate.configure_determinism(not dry)
    import jax
    import jax.numpy as jp
    from foresight.adaptive.actuator_env import AdaptiveJoystick
    from foresight.envs.g1_impact import default_config
    from foresight.eval import rollout

    if not all(d.platform == ("cpu" if dry else "gpu") for d in jax.devices()):
        raise SystemExit("Unexpected compute backend")
    if jax.config.x64_enabled:
        raise ValueError(
            "Stage A identity calibration requires the published float32 JAX settings"
        )
    config = make_config(dry)
    versions = {
        n: md.version(n)
        for n in ("mujoco", "mujoco-mjx", "warp-lang", "jax", "jaxlib", "brax")
    }
    if versions != {
        "mujoco": "3.16.0",
        "mujoco-mjx": "3.16.0",
        "warp-lang": "1.18.0",
        "jax": "0.12.0.1",
        "jaxlib": "0.12.0.1",
        "brax": "0.14.2",
    }:
        raise ValueError("Stage A lock mismatch")
    # Permit a reviewed descendant as long as all relevant Stage A files remain identical.
    watched = [
        "scripts/eval_adaptive_gate.py",
        "foresight/adaptive/actuator_env.py",
        "foresight/eval",
        "foresight/envs",
        "checkpoints/nominal_s1",
    ]
    subprocess.run(
        ["git", "diff", "--quiet", SOURCE, "--", *watched], cwd=ROOT, check=True
    )
    checkpoint = rollout.latest_checkpoint(ROOT / "checkpoints" / "nominal_s1")
    ck = gate.checkpoint_manifest(checkpoint)
    if (
        ck["tree_sha256"]
        != "14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de"
    ):
        raise ValueError("Checkpoint mismatch")
    cfg = default_config()
    cfg.impact.resample_on_autoreset = False
    cfg.impact.miss_fraction = 0.0
    cfg.impact.reset_perturb = 1.0
    cfg.push_config.enable = False
    cfg.episode_length = gate.HORIZON
    env = AdaptiveJoystick(config=cfg, config_overrides={"impl": "warp"})
    if (
        not np.isclose(env.dt, gate.DT)
        or env.action_size != 29
        or sum(env.leg_actuator_mask) != 12
    ):
        raise ValueError("Unexpected G1 interface")
    env._mjx_model, physics = gate.physics_execution(
        env.mjx_model, "warp", "auto", "smoke", not dry
    )
    if physics["physics_mode"] != "default_warp_diagnostic":
        raise ValueError("Calibration is for the pinned default Warp backend")
    args.out.mkdir(parents=True, exist_ok=True)
    names = (
        "manifest.json",
        "episodes.csv",
        "batches.json",
        "report.md",
        "summary.json",
    )
    if any((args.out / name).exists() for name in names):
        raise FileExistsError("Existing calibration evidence must be preserved")
    if not dry:
        hardware = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=15,
        ).strip()
        if hardware != "NVIDIA RTX 6000 Ada Generation, 595.112.07":
            raise ValueError("Unexpected calibration GPU/driver: " + hardware)
        config["gpu_hardware"] = hardware
    config.update(
        versions=versions,
        devices=[str(d) for d in jax.devices()],
        physics=physics,
        checkpoint=ck["tree_sha256"],
        python=platform.python_version(),
        nominal_s1=True,
        jax_enable_x64=bool(jax.config.x64_enabled),
        matmul_precision=jax.config.jax_default_matmul_precision,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write(args.out / "manifest.json", config)
    model = rollout.load_policy(checkpoint, deterministic=True)
    evaluate = gate.build_rollout(env, model, steps=config["steps"])
    reset = jax.jit(jax.vmap(env.reset))
    results = []
    started = time.monotonic()
    with (args.out / "episodes.csv").open("x", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "lane",
                "arm",
                "speed",
                "onset",
                *CALIBRATION_METRICS,
                *RETAIN,
            ],
        )
        writer.writeheader()
        for seed in config["seeds"]:
            scene = gate.scenarios(seed, BATCH)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(
                jp.arange(BATCH, dtype=jp.int32)
            )
            initial = reset(keys)
            speed = jp.asarray(scene["speed"])
            command = jp.stack(
                [speed, jp.zeros_like(speed), jp.zeros_like(speed)], axis=-1
            )
            initial = rollout.set_scenario(env, initial, launch=False, command=command)
            info = dict(initial.info)
            info.update(
                adaptive_gain=jp.ones((BATCH, env.action_size), jp.float32),
                adaptive_oracle=jp.zeros(BATCH, bool),
                adaptive_onset_step=jp.asarray(scene["onset"]),
            )
            # An engineering dry run places onset at zero to exercise post-onset loss
            # and avoid the gate denominator (steps - onset) becoming negative.
            if dry:
                info["adaptive_onset_step"] = jp.zeros(BATCH, jp.int32)
            initial = initial.replace(
                obs=rollout._with_command(initial.obs, command), info=info
            )
            paired = []
            for arm in ("healthy_a", "healthy_b"):
                raw = jax.device_get(evaluate(initial))
                vals = {
                    name: np.asarray(raw[name], np.float64)
                    for name in (*CALIBRATION_METRICS.values(), *RETAIN)
                }
                for i in range(BATCH):
                    row = dict(
                        seed=seed,
                        lane=i,
                        arm=arm,
                        speed=float(speed[i]),
                        onset=int(info["adaptive_onset_step"][i]),
                    )
                    row.update(
                        {
                            name: float(vals[key][i])
                            for name, key in CALIBRATION_METRICS.items()
                        }
                    )
                    row.update({key: float(vals[key][i]) for key in RETAIN})
                    writer.writerow(row)
                f.flush()
                paired.append(vals)
            results.append(dict(seed=seed, **validate_records(paired, dry)))
            write(args.out / "batches.json", results)
            print(f"Completed batch {len(results)}/{len(config['seeds'])}", flush=True)
    score = max(x["score"] for x in results)
    summary = dict(
        profile=config["profile"],
        batches=len(results),
        rows=2 * BATCH * len(results),
        terminal_disagreements=sum(x["terminal_disagreements"] for x in results),
        max_score=score,
        elapsed_seconds=time.monotonic() - started,
        scientific_gate_status="inconclusive",
    )
    if not dry:
        summary["upper_envelope"] = {
            name: score * scale for name, scale in FROZEN.items()
        }
        summary["nominal_confidence_at_95pct_content"] = 1 - 0.95 ** len(results)
        summary["resolves_registered_loss_gap"] = (
            score * FROZEN["postfault_loss"] < 0.03
        )
    write(args.out / "summary.json", summary)
    report = [
        "# Healthy-repeat identity calibration",
        "",
        f"Profile {summary['profile']}. Completed {len(results)} paired batches ({summary['rows']} rows).",
        f"Terminal-step disagreements: {summary['terminal_disagreements']}. G0 remains undecided.",
    ]
    if not dry:
        report += [
            "",
            "Prospective physical envelopes based on the maximum normalized batch discrepancy:",
            "",
        ]
        report.extend(
            f"- {name}: {upper:.9g}"
            for name, upper in summary["upper_envelope"].items()
        )
        if not summary["resolves_registered_loss_gap"]:
            report.append(
                "The loss envelope is as wide as the registered loss-gap threshold or wider."
            )
    (args.out / "report.md").write_text("\n".join(report) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    profile = parser.add_mutually_exclusive_group(required=True)
    profile.add_argument("--cpu-dry-run", action="store_true")
    profile.add_argument("--gpu-calibration", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
