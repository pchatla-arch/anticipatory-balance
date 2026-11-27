"""Causal single-environment adapter for the G1 depth-navigation development gate.

Shares the same source-indexed DECLARED sensor model, RLS estimator and frozen
heuristic with the existing paired actuator bench; never reads true gain,
fault onset/return, or the true obstacle pose to make a correction decision.

A packet produced by MuJoCo at the end of step t is queued with source_step=t
and can only be assimilated after its random delivery latency. Correction for
step t uses only packets released before the action for step t is applied.
"""
from __future__ import annotations

import jax
import jax.numpy as jp

from foresight.adaptive import estimation as E
from foresight.adaptive import sensing as S

RELEASE_PER_STEP = 3


def frozen_heuristic_configs(record):
    """Read a registered heuristic record without silently changing defaults."""
    if not isinstance(record, dict):
        raise ValueError("frozen heuristic must be a JSON dictionary")
    allowed = {"rls", "heuristic"}
    if not set(record).issubset(allowed):
        raise ValueError("unexpected fields in frozen heuristic")
    return E.RLSConfig(**record.get("rls", {})), E.HeuristicConfig(**record.get("heuristic", {}))


class SingleCausalAdapter:
    def __init__(self, env, frozen_record, seed, telemetry=S.DECLARED):
        self.env = env
        self.telemetry = telemetry
        self.seed = int(seed)
        self.rls_config, self.heuristic_config = frozen_heuristic_configs(frozen_record)
        nu = int(env.action_size)
        self.rls = E.rls_init(nu, self.rls_config)
        self.queue = S.queue_init(nu, S.queue_slots(self.telemetry, RELEASE_PER_STEP))
        kp = jp.asarray(env.mj_model.actuator_gainprm[:, 0], jp.float32)
        if kp.shape != (nu,):
            raise ValueError("actuator gains inconsistent with model")
        self.kp = kp
        rls_cfg = self.rls_config
        h_cfg = self.heuristic_config
        tele = self.telemetry

        @jax.jit
        def update(rls, queue, exact, now_step, key):
            packet = S.measure(exact, now_step, tele, key)
            queue = S.queue_push(queue, packet)
            queue, due, mask = S.queue_release(
                queue, now_step, tele.latency_max, RELEASE_PER_STEP)
            for j in range(RELEASE_PER_STEP):
                one = jax.tree_util.tree_map(lambda x: x[j], due)
                rls = E.rls_update(rls, one, mask[j], kp, rls_cfg)
            return rls, queue

        @jax.jit
        def choose(rls, reference, now_step):
            return E.heuristic_gate(rls, reference, now_step, env.dt, h_cfg)

        self._update = update
        self._choose = choose
        self.last_reason = None

    def requested_offset(self, nominal_policy_action, step):
        """Generate a bounded *request* using no injected physics/scene truth."""
        env = self.env
        nominal = env._default_pose + nominal_policy_action * env._config.action_scale
        reference = jp.where(env._adaptive_ctrl_limited,
                             jp.clip(nominal, env._adaptive_ctrl_range[:, 0],
                                     env._adaptive_ctrl_range[:, 1]),
                             nominal)
        offset, reason = self._choose(self.rls, reference, jp.int32(step))
        self.last_reason = reason
        return offset

    def assimilate_captured(self, resulting_info, step):
        """Observe the current step only AFTER physics completes."""
        exact = {k: v for k, v in resulting_info.items() if k.startswith("telemetry_")}
        expected = {
            "telemetry_joint_position_rad",
            "telemetry_joint_velocity_rad_s",
            "telemetry_held_target_rad",
            "telemetry_applied_joint_torque_nm",
            "telemetry_regression_valid",
        }
        if not expected.issubset(exact):
            raise ValueError("missing source-aligned telemetry fields")
        rng = jax.random.fold_in(jax.random.PRNGKey(self.seed), jp.uint32(step))
        self.rls, self.queue = self._update(
            self.rls, self.queue, exact, jp.int32(step), rng)

    def evidence(self):
        """Report estimator status without exposing injected gain labels."""
        return {
            "valid_sample_joints": int(jp.sum(self.rls.n_valid > 0)),
            "valid_samples_total": int(jp.sum(self.rls.n_valid)),
            "gain_estimate_leg_mean": float(jp.mean(self.rls.g_hat[jp.asarray(
                self.env.leg_actuator_mask, dtype=bool)])),
            "lost_packets": int(self.queue.lost),
            "released_packets": int(self.queue.released),
        }
