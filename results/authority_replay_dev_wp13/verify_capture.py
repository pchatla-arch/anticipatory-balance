"""Verify paired development packet captures against repeatable channel draws."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
from foresight.adaptive import bench, sensing

root = Path(__file__).resolve().parent
run = root / "dev"
manifest_path = root / "capture_manifest.json"
manifest = json.loads(manifest_path.read_text())
cfg = json.loads((run / "config.json").read_text())
csv_rows = list(csv.DictReader((run / "episodes.csv").open()))
names = cfg["candidate_model"]["joint_names"]
legs = ["hip" in k or "knee" in k or "ankle" in k for k in names]
env = SimpleNamespace(actuator_joint_names=names, leg_actuator_mask=legs)
steps = cfg["steps"]
batch = cfg["episodes"]
assert (steps, batch) == (300, 8)
assert cfg["devices"] == ["cpu:0"]
assert cfg["source_commit"] == manifest["observer_commit"]
assert len(csv_rows) == 6 * batch
stats = {}
dummy = {"telemetry_joint_position_rad": jp.zeros(1),
         "telemetry_joint_velocity_rad_s": jp.zeros(1),
         "telemetry_applied_joint_torque_nm": jp.zeros(1),
         "telemetry_held_target_rad": jp.zeros(1),
         "telemetry_regression_valid": jp.ones(1, dtype=bool)}
physical_fields = ("command", "alive", "loss", "velocity", "gain_true_active",
                   "authority_decision_q_truth", "authority_reference",
                   "authority_source_q_truth", "authority_source_torque_truth")
for fname, seed in manifest["scenario_families"].items():
    scenarios = bench.draw_scenarios(bench.FAMILIES[fname], seed, batch, env)
    baseline = None
    for channel, sensor_cfg in [("exact", sensing.EXACT), ("declared", sensing.DECLARED)]:
        label = f"{fname}_{channel}_none"
        rows = [row for row in csv_rows if row["family"] == fname and row["telemetry"] == channel]
        assert [row["scenario_id"] for row in rows] == scenarios["ids"]
        assert all(int(row["onset_step"]) == t for row, t in zip(rows, scenarios["onset"]))
        assert all(np.isclose(float(row["speed_cmd_mps"]), v) for row, v in zip(rows, scenarios["speed"]))
        assert all(row["method"] == "none" and row["split"] == "dev" for row in rows)
        p = run / f"trace_{label}.npz"
        with np.load(p) as packet_file:
            tr = dict(packet_file)
        for field in ["requested", "effective", "authority_previous_request"]:
            assert np.count_nonzero(tr[field]) == 0, (label, field)
        if baseline is None:
            baseline = {k: tr[k] for k in physical_fields}
        else:
            assert all(np.array_equal(tr[k], baseline[k]) for k in physical_fields), label
        mask = tr["authority_release_mask"]
        source_ticks = tr["authority_packet_source_step"]
        arrival_ticks = tr["authority_packet_arrival_step"]
        assert mask.shape == (steps, batch, bench.RELEASE_PER_STEP)
        sensor_keys = jax.vmap(jax.random.PRNGKey)(scenarios["sensor_key"])
        def draw_step(t):
            def arrival(key):
                folded = jax.random.fold_in(key, t)
                return sensing.measure(dummy, t, sensor_cfg, folded).arrival_step
            return jax.vmap(arrival)(sensor_keys)
        arrivals = np.asarray(jax.jit(jax.vmap(draw_step))(jp.arange(steps, dtype=jp.int32)))
        totals = dict(generated=steps * batch, sampled_dropped=int(np.count_nonzero(arrivals == int(sensing.DROPPED))),
                      released=int(mask.sum()), released_at_live_tick=int((mask & tr["alive"][:, :, None]).sum()),
                      lost_by_timeout=0, pending_at_horizon=0, undelivered_non_dropped_at_horizon=0)
        for b in range(batch):
            cursor = 0
            for t in range(steps):
                for slot in range(bench.RELEASE_PER_STEP):
                    expected_release = cursor <= t and arrivals[cursor, b] <= t
                    expected_lost = (cursor <= t - sensor_cfg.latency_max - 1 and not expected_release)
                    assert mask[t, b, slot] == expected_release, (label, t, b, slot, cursor)
                    if expected_release:
                        assert source_ticks[t, b, slot] == cursor
                        assert arrival_ticks[t, b, slot] == arrivals[cursor, b]
                    cursor += int(expected_release or expected_lost)
                    totals["lost_by_timeout"] += int(expected_lost)
            totals["pending_at_horizon"] += steps - cursor
            totals["undelivered_non_dropped_at_horizon"] += int(np.count_nonzero(arrivals[cursor:, b] != int(sensing.DROPPED)))
        assert totals["generated"] == totals["released"] + totals["lost_by_timeout"] + totals["pending_at_horizon"]
        assert totals["generated"] == totals["released"] + totals["sampled_dropped"] + totals["undelivered_non_dropped_at_horizon"]
        assert all(int(row["released_packets"]) == 0 and int(row["lost_packets"]) == 0 for row in rows)
        stats[label] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), raw_packet_shape=list(tr["authority_packet_valid"].shape),
                            episodes=len(rows), finite=all(row["nonfinite"] == "False" for row in rows),
                            exposed_live_ticks=int((tr["alive"] & tr["gain_true_active"]).sum()),
                            zero_requested_and_effective=True, packet_counts=totals)
        assert stats[label]["finite"], label
manifest["published_identical_tree_commit"] = "e51d2d7bdbc2e1dccd377c4d752c8b944af72477"
manifest["elapsed_seconds"] = json.loads((run / "summary.json").read_text())["elapsed_seconds"]
manifest["capture_verification"] = stats
manifest["pairing_verification"] = "Within each gain all eight scenario IDs, commands, onsets, speed and every recorded physical field match between channels. All requested and effective offsets are identically zero."
manifest["channel_verification"] = "Independent seeded S.measure draws plus FIFO timeout replay match every release mask, delivered source index, and arrival index. Counts include all 300 scheduled ticks per lane; live releases and fault-exposed live ticks are tabulated separately."
manifest["limitations"] = ["The none method's CSV released/lost counters remain zero by design. Use the observer mask and this replay for packet counts.",
                           "The observer stores combined finite/unsaturated regression validity. Separate joint-torque, actuator-force, and control-limit cause flags are unavailable in these traces.",
                           "This development-only open-loop replay captures one frozen walker on JAX CPU. Report authority at surviving exposures separately from numbers at or after termination."]
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(stats, indent=2))
