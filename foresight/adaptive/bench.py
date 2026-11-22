# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Closed-loop bench for actuator-adaptation methods (WP 1.4, 2.3, 4.1).

One scan runs a batch of paired episodes: a frozen walker follows a command schedule
(walk, turn, stop, resume) while a hidden actuator change follows the scenario's gain
schedule. Every control step the method receives the packets released by the sensor
model, requests an offset, and the environment applies the common correction limits.

Decision-under-uncertainty reading of the loop: the gain is the hidden state, the
released packets are the observations, the method's internal estimate (point estimate,
or ChatGPT's set-membership interval) is its belief, and the requested offset is the
action taken under that belief. The oracle acts on the true state; `none` never acts.

Metrics per episode separate useful recovery from harmful intervention:
  postfault_loss        capped planar-velocity error over the post-onset horizon (the
                        gate's definition; failure charged 4 (m/s)^2 for the rest)
  harmful_offset_rad    mean |effective offset| over steps where the plant is healthy
                        (before onset, after return), the false-adaptation measure
  withdrawal_steps      steps after an unannounced return to healthy until the
                        effective offset falls below 0.01 rad on every actuator
  gain_abs_error        mean |g_hat - g_true| over leg actuators after onset
                        (evaluation-only use of the true gain)
Methods never see adaptive_gain, onset or return steps.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp
import numpy as np

from foresight.adaptive import estimation as E
from foresight.adaptive import sensing as S

DT = 0.02
LOSS_CAP = 4.0
WITHDRAWN_RAD = 0.01
RELEASE_PER_STEP = 3

# ----------------------------------------------------------------------------- scenarios

class ScenarioFamily(NamedTuple):
    """Hidden plant schedule plus the sensor channel used by every method in the family."""

    name: str
    gain: float = 1.0                 # multiplicative leg stiffness after onset
    joint_group: str = "legs"         # legs | hips | knees | left_leg
    onset_range: tuple = (75, 125)    # control steps, inclusive-exclusive
    return_after: tuple = None        # (min, max) steps after onset; None = never
    damping_scale: float = 1.0        # passive-loss confound on the same schedule
    telemetry: S.TelemetryConfig = S.EXACT
    description: str = ""


FAMILIES = {
    "healthy": ScenarioFamily("healthy", description="no plant change; measures false adaptation at rest"),
    "gain_085": ScenarioFamily("gain_085", gain=0.85),
    "gain_070": ScenarioFamily("gain_070", gain=0.70),
    "gain_055": ScenarioFamily("gain_055", gain=0.55),
    "gain_070_return": ScenarioFamily("gain_070_return", gain=0.70, return_after=(100, 150),
                                      description="unannounced return to healthy gain"),
    "gain_070_hips": ScenarioFamily("gain_070_hips", gain=0.70, joint_group="hips"),
    "gain_070_left": ScenarioFamily("gain_070_left", gain=0.70, joint_group="left_leg",
                                    description="asymmetric fault"),
    "sensor_scale_080": ScenarioFamily("sensor_scale_080", telemetry=S.TelemetryConfig(torque_scale=0.8),
                                       description="healthy motor, effort sensor reads 80%: identifiability confound"),
    "sensor_bias": ScenarioFamily("sensor_bias", telemetry=S.TelemetryConfig(torque_bias_nm=2.0),
                                  description="healthy motor, constant effort bias"),
    "damping_x3": ScenarioFamily("damping_x3", damping_scale=3.0,
                                 description="healthy actuators, tripled passive joint losses after onset"),
    "stale_telemetry": ScenarioFamily("stale_telemetry", gain=0.70,
                                      telemetry=S.TelemetryConfig(latency_min=5, latency_max=10),
                                      description="gain fault seen through 100-200 ms old packets"),
}


def joint_group_mask(env, group):
    names = env.actuator_joint_names
    legs = np.asarray(env.leg_actuator_mask, bool)
    if group == "legs":
        return legs
    if group == "hips":
        return np.asarray([n.find("hip") >= 0 for n in names]) & legs
    if group == "knees":
        return np.asarray([n.find("knee") >= 0 for n in names]) & legs
    if group == "left_leg":
        return np.asarray([n.startswith("left_") for n in names]) & legs
    raise ValueError(f"unknown joint group {group}")


def draw_scenarios(family: ScenarioFamily, seed, count, env):
    """Deterministic per-episode schedule from the split seed; same across methods."""
    rng = np.random.default_rng(seed)
    onset = rng.integers(family.onset_range[0], family.onset_range[1], size=count, dtype=np.int32)
    if family.return_after is None:
        ret = np.full(count, 2**30, np.int32)
    else:
        ret = onset + rng.integers(family.return_after[0], family.return_after[1], size=count, dtype=np.int32)
    mask = joint_group_mask(env, family.joint_group)
    gains = np.broadcast_to(np.where(mask, family.gain, 1.0).astype(np.float32), (count, len(mask))).copy()
    return {
        "ids": [f"{family.name}_seed{seed}_env{i:05d}" for i in range(count)],
        "onset": onset, "return": ret, "gain": gains,
        "damping_scale": np.full(count, family.damping_scale, np.float32),
        "speed": rng.choice(np.array([0.4, 0.6], dtype=np.float32), size=count),
        "sensor_key": rng.integers(0, 2**31 - 1, size=count, dtype=np.int64).astype(np.int32),
    }


# ----------------------------------------------------------------------- command schedule

class CommandSchedule(NamedTuple):
    """Walk, turn, stop, resume (WP 4.1): segment start steps and (vx, vy, yaw) per segment."""

    starts: tuple = (0, 100, 200, 275)
    commands: tuple = ((1.0, 0.0, 0.0), (0.6, 0.0, 0.5), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # vx entries are multipliers of the per-episode speed; vy and yaw are absolute.


WALK_ONLY = CommandSchedule(starts=(0,), commands=((1.0, 0.0, 0.0),))
WALK_TURN_STOP_RESUME = CommandSchedule()


def command_at(schedule: CommandSchedule, step, speed):
    starts = jp.asarray(schedule.starts, jp.int32)
    cmds = jp.asarray(schedule.commands, jp.float32)
    idx = jp.sum((step >= starts).astype(jp.int32)) - 1
    c = cmds[idx]
    return jp.stack([c[0] * speed, c[1], c[2]])


# ------------------------------------------------------------------------------- methods

class MethodSpec(NamedTuple):
    name: str
    external: bool          # method supplies adaptive_requested_offset
    oracle: bool            # env applies the privileged true-gain inverse
    rls: E.RLSConfig = E.RLSConfig()
    heuristic: E.HeuristicConfig = E.HeuristicConfig()
    candidate: object = None   # bound method or None (WP 2.2)


METHODS = {
    "none": MethodSpec("none", external=False, oracle=False),
    "oracle": MethodSpec("oracle", external=False, oracle=True),
    "estimated_inverse": MethodSpec("estimated_inverse", external=True, oracle=False),
    "heuristic_gate": MethodSpec("heuristic_gate", external=True, oracle=False),
    # Test-only: the privileged inverse sent through the external path. It must reproduce
    # the in-env oracle bitwise; it is the equivalence check of the shared limits.
    "oracle_external": MethodSpec("oracle_external", external=True, oracle=False),
}


def load_candidate():
    """ChatGPT's identification-aware method, if its module is present."""
    try:
        from foresight.adaptive import candidate  # noqa: F401
    except ImportError:
        return None
    return candidate


class MethodState(NamedTuple):
    rls: E.RLSState
    queue: S.DeliveryQueue
    candidate: object
    reason: jax.Array        # (n, nu) int32 last reason code (heuristic/candidate)


def select_batched_rows(mask, new, old):
    """Per-environment select that leaves unbatched (shared) leaves alone (AE-007)."""
    n = mask.shape[0]

    def pick(x, y):
        if jp.ndim(x) == 0 or jp.shape(x)[0] != n:
            return x
        return jp.where(mask.reshape((n,) + (1,) * (jp.ndim(x) - 1)), x, y)

    return jax.tree_util.tree_map(pick, new, old)


def finite_rows(value):
    return jp.all(jp.isfinite(value), axis=tuple(range(1, value.ndim)))


# -------------------------------------------------------------------------------- rollout

def build_bench_rollout(env, policy, method: MethodSpec, telemetry: S.TelemetryConfig,
                        schedule: CommandSchedule, steps, trace=False, trace_authority_packets=False):
    """Return execute(initial, scenario) -> per-episode metrics for one method and sensor model.

    With trace=True the result also carries per-step arrays (step axis first): g_hat,
    requested and effective offsets, reason codes, per-step loss and the alive mask, for
    estimator-convergence and correction-timeline figures.
    """
    if trace_authority_packets and not trace:
        raise ValueError("authority packet capture requires trace=True")
    from foresight.eval import rollout as R

    nu = env.action_size
    kp = jp.asarray(env.mj_model.actuator_gainprm[:, 0], jp.float32)
    infer = jax.vmap(policy)
    advance = jax.vmap(env.step)
    velocity = jax.vmap(lambda data: env.get_local_linvel(data, "pelvis")[:2])
    pelvis = jax.vmap(env._pelvis_pos)
    slots = S.queue_slots(telemetry, RELEASE_PER_STEP)
    leg = jp.asarray(env.leg_actuator_mask, bool)
    default_pose = env._default_pose
    action_scale = env._config.action_scale
    ctrl_range = env._adaptive_ctrl_range
    ctrl_limited = env._adaptive_ctrl_limited
    uses_estimator = method.external and method.name != "oracle_external"

    def method_init(n):
        cand = method.candidate.init(n, nu) if method.candidate is not None else None
        return MethodState(
            rls=jax.vmap(lambda _: E.rls_init(nu, method.rls))(jp.arange(n)),
            queue=jax.vmap(lambda _: S.queue_init(nu, slots))(jp.arange(n)),
            candidate=cand,
            reason=jp.zeros((n, nu), jp.int32),
        )

    def method_update(mstate, info, t, sensor_keys):
        """Push this step's capture through the sensor model, release due packets, assimilate."""
        exact = {k: info[k] for k in info if k.startswith("telemetry_")}
        keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(sensor_keys)
        packets = jax.vmap(lambda ex, k: S.measure(ex, t, telemetry, k))(exact, keys)
        queue = jax.vmap(S.queue_push)(mstate.queue, packets)
        queue, due, mask = jax.vmap(lambda q: S.queue_release(q, t, telemetry.latency_max, RELEASE_PER_STEP))(queue)
        rls = mstate.rls
        cand = mstate.candidate
        for j in range(RELEASE_PER_STEP):
            pkt_j = jax.tree_util.tree_map(lambda x: x[:, j], due)
            rls = jax.vmap(lambda s, p, m: E.rls_update(s, p, m, kp, method.rls))(rls, pkt_j, mask[:, j])
            if method.candidate is not None:
                cand = method.candidate.update(cand, pkt_j, mask[:, j])
        return mstate._replace(rls=rls, queue=queue, candidate=cand), packets, due, mask

    def method_correction(mstate, reference, t, simulated):
        if method.name == "oracle_external":
            info = simulated.info
            active = (t >= info["adaptive_onset_step"]) & (t < info["adaptive_return_step"])
            gain = jp.where(active[:, None], info["adaptive_gain"], 1.0)
            q = jax.vmap(env.measured_joint_positions)(simulated.data)
            return ((1.0 / gain - 1.0) * (reference - q)).astype(jp.float32), mstate.reason
        if method.name == "estimated_inverse":
            x = jax.vmap(lambda s, r: E.estimated_inverse(s, r, t, DT))(mstate.rls, reference)
            return x, mstate.reason
        if method.name == "heuristic_gate":
            x, reason = jax.vmap(lambda s, r: E.heuristic_gate(s, r, t, DT, method.heuristic))(mstate.rls, reference)
            return x, reason
        if method.candidate is not None:
            return method.candidate.correction(mstate.candidate, mstate.rls, reference, t,
                                               simulated.info["adaptive_prev_delta"])
        return jp.zeros_like(reference), mstate.reason

    @jax.jit
    def execute(initial, scenario):
        n = initial.done.shape[0]
        onset = scenario["onset"]
        ret = scenario["return"]
        speed = scenario["speed"]
        gain_true = scenario["gain"]
        sensor_keys = jax.vmap(jax.random.PRNGKey)(scenario["sensor_key"])
        info0 = dict(initial.info)
        info0["adaptive_gain"] = gain_true
        info0["adaptive_onset_step"] = onset
        info0["adaptive_return_step"] = ret
        info0["adaptive_damping_scale"] = scenario["damping_scale"]
        info0["adaptive_oracle"] = jp.full(n, method.oracle, bool)
        info0["adaptive_external"] = jp.full(n, method.external, bool)
        initial = initial.replace(info=info0)
        metrics = {
            "alive": jp.ones(n, bool), "term_step": jp.full(n, -1, jp.int32),
            "first_nonfinite_step": jp.full(n, -1, jp.int32), "pre_fault_failed": jp.zeros(n, bool),
            "loss_sum": jp.zeros(n), "progress": jp.zeros(n), "exposed": jp.zeros(n, jp.int32),
            "bound_sum": jp.zeros(n), "control_clip_sum": jp.zeros(n),
            "max_correction": jp.zeros(n), "max_slew": jp.zeros(n), "max_force": jp.zeros(n),
            "healthy_steps": jp.zeros(n, jp.int32), "observed_steps": jp.zeros(n, jp.int32), "harmful_sum": jp.zeros(n),
            "applied_sum": jp.zeros(n), "withdrawal_step": jp.full(n, -1, jp.int32),
            "gain_err_sum": jp.zeros(n), "gain_err_steps": jp.zeros(n, jp.int32),
            "reason_apply": jp.zeros(n, jp.int32), "lost_packets": jp.zeros(n, jp.int32),
            "released_packets": jp.zeros(n, jp.int32),
        }

        def one_step(carry, t):
            state, mstate, acc = carry
            active = acc["alive"]
            simulated = select_batched_rows(active, state, initial)
            cmd = jax.vmap(lambda sp: command_at(schedule, t, sp))(speed)
            info = dict(simulated.info)
            info["command_nominal"] = cmd
            obs = R._with_command(simulated.obs, cmd)
            keys = jax.vmap(lambda key: jax.random.fold_in(key, t))(info["rng"])
            action, _ = infer(obs, keys)
            action_finite = finite_rows(action)
            safe_action = jp.where(action_finite[:, None], action, jp.zeros_like(action))
            nominal = default_pose + safe_action * action_scale
            reference = jp.where(ctrl_limited, jp.clip(nominal, ctrl_range[:, 0], ctrl_range[:, 1]), nominal)
            if method.external:
                requested, reason = method_correction(mstate, reference, t, simulated)
                requested = jp.where(jp.isfinite(requested), requested, 0.0)
                info["adaptive_requested_offset"] = requested.astype(jp.float32)
                mstate = mstate._replace(reason=reason)
            simulated = simulated.replace(info=info, obs=obs)
            next_state = advance(simulated, safe_action)
            if trace_authority_packets:
                # Pre-command RLS and issued interface state. Truth channels are
                # observer-only and never passed to any practical control method.
                decision_rls = mstate.rls
            if uses_estimator or method.candidate is not None or trace:
                mstate, measured, due, release_mask = method_update(mstate, next_state.info, t, sensor_keys)
            finite = (finite_rows(next_state.data.qpos) & finite_rows(next_state.data.qvel)
                      & finite_rows(next_state.data.xpos) & finite_rows(next_state.data.sensordata)
                      & finite_rows(next_state.obs["state"]) & action_finite)
            invalid = active & ~finite
            terminal = active & ((next_state.done > 0) | invalid)
            observed = active & ~invalid
            first_term = terminal & (acc["term_step"] < 0)
            after_onset = t >= onset
            # The plant is healthy outside the scheduled window, and at every step when the
            # family changes no actuator at all (healthy, sensor and damping confounds keep
            # every gain at 1): harmful intervention must then count the whole episode (AE-011).
            no_actuator_change = jp.all(gain_true == 1.0, axis=-1)
            plant_healthy = (t < onset) | (t >= ret) | no_actuator_change
            exposed = observed & after_onset
            error = velocity(next_state.data) - cmd[:, :2]
            loss = jp.minimum(jp.sum(jp.square(error), axis=-1), LOSS_CAP)
            loss = jp.where(observed & ~terminal, loss, LOSS_CAP)
            displacement = pelvis(next_state.data)[:, :2] - pelvis(state.data)[:, :2]
            forward = state.data.xmat[:, env._pelvis_body].reshape(n, 3, 3)[:, :2, 0]
            forward = forward / jp.maximum(jp.linalg.norm(forward, axis=-1, keepdims=True), 1e-6)
            progress = jp.where(exposed, jp.sum(displacement * forward, axis=-1), 0.0)
            delta = next_state.info["adaptive_step_delta"]
            effective = next_state.info["adaptive_effective_delta"]
            prev_delta = simulated.info["adaptive_prev_delta"]
            correction = jp.max(jp.abs(delta), axis=-1)
            slew = jp.max(jp.abs(delta - prev_delta), axis=-1) / DT
            force = jp.max(jp.abs(next_state.data.actuator_force), axis=-1)
            eff_mag = jp.max(jp.abs(effective), axis=-1)
            withdrawn = (t >= ret) & (eff_mag < WITHDRAWN_RAD) & (acc["withdrawal_step"] < 0) & observed
            g_hat = mstate.rls.g_hat if uses_estimator else jp.ones_like(gain_true)
            gain_err = jp.sum(jp.abs(g_hat - gain_true) * leg[None, :], axis=-1) / jp.sum(leg)
            apply_now = jp.sum((mstate.reason == E.REASON_APPLY).astype(jp.int32) * leg[None, :], axis=-1) > 0
            out = {
                "alive": active & ~terminal,
                "term_step": jp.where(first_term, t, acc["term_step"]),
                "first_nonfinite_step": jp.where(invalid & (acc["first_nonfinite_step"] < 0), t, acc["first_nonfinite_step"]),
                "pre_fault_failed": acc["pre_fault_failed"] | (first_term & ~after_onset),
                "loss_sum": acc["loss_sum"] + jp.where(after_onset, loss, 0.0),
                "progress": acc["progress"] + progress,
                "exposed": acc["exposed"] + exposed.astype(jp.int32),
                "bound_sum": acc["bound_sum"] + jp.where(exposed, next_state.info["adaptive_bound_hit"], 0.0),
                "control_clip_sum": acc["control_clip_sum"] + jp.where(exposed, next_state.info["adaptive_control_clip_fraction"], 0.0),
                "max_correction": jp.maximum(acc["max_correction"], jp.where(observed, correction, 0.0)),
                "max_slew": jp.maximum(acc["max_slew"], jp.where(observed, slew, 0.0)),
                "max_force": jp.maximum(acc["max_force"], jp.where(observed, force, 0.0)),
                "healthy_steps": acc["healthy_steps"] + (observed & plant_healthy).astype(jp.int32),
                "observed_steps": acc["observed_steps"] + observed.astype(jp.int32),
                "harmful_sum": acc["harmful_sum"] + jp.where(observed & plant_healthy, eff_mag, 0.0),
                "applied_sum": acc["applied_sum"] + jp.where(observed, eff_mag, 0.0),
                "withdrawal_step": jp.where(withdrawn, t, acc["withdrawal_step"]),
                "gain_err_sum": acc["gain_err_sum"] + jp.where(exposed & ~plant_healthy, gain_err, 0.0),
                "gain_err_steps": acc["gain_err_steps"] + (exposed & ~plant_healthy).astype(jp.int32),
                "reason_apply": acc["reason_apply"] + (observed & apply_now).astype(jp.int32),
                "lost_packets": mstate.queue.lost if uses_estimator else acc["lost_packets"],
                "released_packets": mstate.queue.released if uses_estimator else acc["released_packets"],
            }
            state = select_batched_rows(observed, next_state, state)
            ys = None
            if trace:
                ys = {"g_hat": g_hat, "requested": next_state.info["adaptive_step_delta"],
                      "effective": effective, "reason": mstate.reason, "loss": loss,
                      "alive": observed, "gain_true_active": ~plant_healthy,
                      "command": cmd, "velocity": velocity(next_state.data),
                      # The measured packet captured at the end of this step (WP 1.3 replay input).
                      "pkt_arrival": measured.arrival_step, "pkt_target": measured.target_rad,
                      "pkt_position": measured.position_rad, "pkt_velocity": measured.velocity_rad_s,
                      "pkt_torque": measured.torque_nm, "pkt_valid": measured.valid,
                      "reference": reference, "q_true": jax.vmap(env.measured_joint_positions)(simulated.data)}
            if trace_authority_packets:
                if "pilot_substep_q" in next_state.info:
                    ys["pilot_substep_q"] = next_state.info["pilot_substep_q"]
                ys.update(authority_reference=reference,
                          authority_previous_request=prev_delta,
                          authority_decision_q_truth=jax.vmap(env.measured_joint_positions)(simulated.data),
                          authority_source_q_truth=next_state.info["telemetry_joint_position_rad"],
                          authority_source_torque_truth=next_state.info["telemetry_applied_joint_torque_nm"],
                          authority_release_mask=release_mask)
                for field in ("g_hat", "last_source_step", "last_position", "last_velocity"):
                    ys["authority_rls_" + field] = getattr(decision_rls, field)
                for field in due._fields:
                    ys["authority_packet_" + field] = getattr(due, field)
            return (state, mstate, out), ys

        mstate0 = method_init(n)
        (_, _, result), traces = jax.lax.scan(one_step, (initial, mstate0, metrics), jp.arange(steps))
        if trace:
            result["trace"] = traces
        result["postfault_loss"] = result["loss_sum"] / jp.maximum(steps - onset, 1)
        result["harmful_offset_rad"] = result["harmful_sum"] / jp.maximum(result["healthy_steps"], 1)
        result["gain_abs_error"] = result["gain_err_sum"] / jp.maximum(result["gain_err_steps"], 1)
        result["withdrawal_steps"] = jp.where(result["withdrawal_step"] >= 0, result["withdrawal_step"] - ret, -1)
        return result

    return execute


def episode_rows(result, scenario, family, method, telemetry_name, split, phase="eval"):
    n = len(scenario["ids"])
    rows = []
    for i in range(n):
        term = int(result["term_step"][i])
        rows.append({
            "split": split, "phase": phase, "family": family.name, "method": method,
            "telemetry": telemetry_name, "scenario_id": scenario["ids"][i],
            "gain": float(family.gain), "onset_step": int(scenario["onset"][i]),
            "return_step": int(scenario["return"][i]) if scenario["return"][i] < 2**29 else -1,
            "speed_cmd_mps": float(scenario["speed"][i]),
            "fell": term >= 0, "nonfinite": int(result["first_nonfinite_step"][i]) >= 0,
            "termination_step": term, "pre_fault_failed": bool(result["pre_fault_failed"][i]),
            "postfault_loss": float(result["postfault_loss"][i]),
            "progress_m": float(result["progress"][i]),
            "correction_bound_fraction": float(result["bound_sum"][i]) / max(int(result["exposed"][i]), 1),
            "control_clip_fraction": float(result["control_clip_sum"][i]) / max(int(result["exposed"][i]), 1),
            "max_correction_rad": float(result["max_correction"][i]),
            "max_slew_rad_s": float(result["max_slew"][i]),
            "max_abs_actuator_force_nm": float(result["max_force"][i]),
            "harmful_offset_rad": float(result["harmful_offset_rad"][i]),
            "mean_applied_offset_rad": float(result["applied_sum"][i]) / max(int(result["observed_steps"][i]), 1),
            "withdrawal_steps": int(result["withdrawal_steps"][i]),
            "gain_abs_error": float(result["gain_abs_error"][i]),
            "apply_steps": int(result["reason_apply"][i]),
            "lost_packets": int(result["lost_packets"][i]),
            "released_packets": int(result["released_packets"][i]),
        })
    return rows


__all__ = ["ScenarioFamily", "FAMILIES", "draw_scenarios", "CommandSchedule", "WALK_ONLY",
           "WALK_TURN_STOP_RESUME", "command_at", "MethodSpec", "METHODS", "load_candidate",
           "build_bench_rollout", "episode_rows", "DT", "LOSS_CAP"]
