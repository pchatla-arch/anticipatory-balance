"""Sensor model, in-order delivery and the RLS/heuristic baselines, without physics (WP 1.1, 2.1)."""
import unittest

import jax
import jax.numpy as jp
import numpy as np

from foresight.adaptive import estimation as E
from foresight.adaptive import sensing as S

NU = 3
KP = 75.0


def exact_packet(step, g=1.0, phi=0.1, valid=True):
    phi = jp.full((NU,), phi, jp.float32)
    z = jp.full((NU,), 0.3, jp.float32)
    return {
        "telemetry_source_time_s": jp.float32(step * 0.02),
        "telemetry_held_target_rad": z,
        "telemetry_joint_position_rad": z - phi,
        "telemetry_joint_velocity_rad_s": jp.zeros((NU,), jp.float32),
        "telemetry_applied_joint_torque_nm": (g * KP * phi).astype(jp.float32),
        "telemetry_regression_valid": jp.full((NU,), valid, bool),
    }


class SensorModelTests(unittest.TestCase):
    def test_exact_config_is_identity_and_keys_are_repeatable(self):
        key = jax.random.PRNGKey(3)
        p = S.measure(exact_packet(7, g=0.7), 7, S.EXACT, key)
        q = S.measure(exact_packet(7, g=0.7), 7, S.EXACT, key)
        np.testing.assert_array_equal(p.position_rad, exact_packet(7)["telemetry_joint_position_rad"])
        np.testing.assert_allclose(p.torque_nm, 0.7 * KP * 0.1, rtol=1e-6)
        self.assertEqual(int(p.arrival_step), 7)
        for a, b in zip(p, q):
            np.testing.assert_array_equal(a, b)

    def test_scale_bias_noise_latency_and_dropout(self):
        cfg = S.TelemetryConfig(position_noise_rad=1e-3, torque_noise_nm=0.2, position_bias_rad=0.01,
                                torque_bias_nm=0.5, torque_scale=0.8, latency_min=1, latency_max=3,
                                dropout=0.5, bound_position_rad=4e-3, bound_torque_nm=0.8).validate()
        keys = jax.random.split(jax.random.PRNGKey(0), 400)
        packets = jax.vmap(lambda k: S.measure(exact_packet(10, g=1.0), 10, cfg, k))(keys)
        delivered = packets.arrival_step < S.DROPPED
        self.assertTrue(0.3 < float(delivered.mean()) < 0.7)
        lat = packets.arrival_step[delivered] - 10
        self.assertTrue(bool(jp.all((lat >= 1) & (lat <= 3))))
        self.assertAlmostEqual(float(packets.position_rad.mean()), 0.2 + 0.01, delta=2e-4)
        self.assertAlmostEqual(float(packets.torque_nm.mean()), 0.8 * KP * 0.1 + 0.5, delta=0.05)
        eps = S.declared_bounds(cfg, KP, g_max=1.5)
        self.assertAlmostEqual(float(eps), 0.8 / KP + 1.5 * 4e-3)
        with self.assertRaises(ValueError):
            S.TelemetryConfig(latency_min=2, latency_max=1).validate()
        self.assertEqual(S.config_from_dict(S.summarize(cfg)), cfg)


class DeliveryQueueTests(unittest.TestCase):
    def run_queue(self, latencies, drops=(), steps=12, count=2):
        max_lat = max(latencies.values()) if latencies else 0
        cfg = S.TelemetryConfig(latency_max=max_lat)
        queue = S.queue_init(NU, S.queue_slots(cfg, count))
        released = []
        for now in range(steps):
            if now in latencies:
                pkt = S.measure(exact_packet(now, g=0.7), now, S.EXACT, jax.random.PRNGKey(now))
                arrival = S.DROPPED if now in drops else now + latencies[now]
                pkt = pkt._replace(arrival_step=jp.int32(arrival))
                queue = S.queue_push(queue, pkt)
            queue, packets, mask = S.queue_release(queue, jp.int32(now), max_lat, count)
            for j in range(count):
                if bool(mask[j]):
                    released.append((now, int(packets.source_step[j]), float(packets.torque_nm[j][0])))
        return queue, released

    def test_in_order_release_with_reordering(self):
        # Capture every step 0..7; latencies make step 3 arrive after step 4.
        lat = {0: 0, 1: 1, 2: 1, 3: 3, 4: 1, 5: 0, 6: 2, 7: 0}
        queue, released = self.run_queue(lat, steps=12)
        sources = [s for _, s, _ in released]
        self.assertEqual(sources, list(range(8)), "source order, no gaps, nothing twice")
        for now, s, _ in released:
            self.assertGreaterEqual(now, s + lat[s], "never released before arrival")
        self.assertEqual(int(queue.released), 8)
        self.assertTrue(all(abs(t - 0.7 * KP * 0.1) < 1e-4 for _, _, t in released))

    def test_dropped_packet_is_declared_lost_after_the_maximum_latency(self):
        # A capture every step (as in the closed loop); the step-1 packet is dropped.
        lat = {0: 0, 1: 2, 2: 2, 3: 1, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0}
        queue, released = self.run_queue(lat, drops=(1,), steps=12)
        sources = [s for _, s, _ in released]
        self.assertEqual(sources, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.assertEqual(int(queue.lost), 1)
        first_after = next(now for now, s, _ in released if s == 2)
        self.assertEqual(first_after, 1 + 2 + 1, "cursor waits max_latency + 1 steps for the lost packet")

    def test_release_rate_limit_drains_backlog(self):
        lat = {i: 0 for i in range(6)}
        lat[0] = 4  # step 0 arrives late; steps 1..5 queue behind it
        _, released = self.run_queue(lat, steps=12, count=2)
        self.assertEqual([s for _, s, _ in released], list(range(6)))
        per_step = {}
        for now, s, _ in released:
            per_step.setdefault(now, []).append(s)
        self.assertTrue(all(len(v) <= 2 for v in per_step.values()))


class EstimationTests(unittest.TestCase):
    def feed(self, g_true, n=40, cfg=E.RLSConfig(), noise=0.0, scale=1.0, valid=True, start_state=None):
        state = start_state or E.rls_init(NU, cfg)
        key = jax.random.PRNGKey(1)
        for i in range(n):
            key, k = jax.random.split(key)
            phi = 0.05 + 0.1 * (i % 3)
            exact = exact_packet(i, g=g_true, phi=phi, valid=valid)
            tcfg = S.TelemetryConfig(torque_noise_nm=noise, torque_scale=scale)
            pkt = S.measure(exact, i, tcfg, k)
            state = E.rls_update(state, pkt, jp.array(True), KP, cfg)
        return state

    def test_rls_recovers_the_gain_and_ignores_invalid_samples(self):
        state = self.feed(0.7)
        np.testing.assert_allclose(state.g_hat, 0.7, atol=1e-3)
        self.assertEqual(int(state.n_valid[0]), 40)
        invalid = self.feed(0.7, valid=False)
        np.testing.assert_allclose(invalid.g_hat, 1.0)
        self.assertEqual(int(invalid.n_valid[0]), 0)

    def test_forgetting_tracks_a_gain_change_and_residual_flags_wrong_models(self):
        tracking = E.RLSConfig(forgetting=0.95)
        state = self.feed(1.0, n=30, cfg=tracking)
        state = self.feed(0.6, n=60, cfg=tracking, start_state=state)
        np.testing.assert_allclose(state.g_hat, 0.6, atol=3e-2)
        # An effort-sensor scale of 0.7 on a healthy motor is indistinguishable from a 0.7 gain
        # for a point estimator: the residual stays small while the estimate is wrong (AE-003).
        confounded = self.feed(1.0, n=40, scale=0.7)
        np.testing.assert_allclose(confounded.g_hat, 0.7, atol=1e-3)
        self.assertLess(float(jp.sqrt(confounded.residual_sq).max()), 1e-3)
        noisy = self.feed(0.7, n=40, noise=2.0)
        self.assertGreater(float(jp.sqrt(noisy.residual_sq).mean()), 0.005)

    def test_heuristic_gate_reasons_and_estimated_inverse(self):
        cfg = E.HeuristicConfig(min_samples=5, max_residual_rms=0.05, min_excitation_rad=0.01,
                                dead_band=0.05, max_age_steps=5)
        reference = jp.full((NU,), 0.3, jp.float32)
        fresh = E.rls_init(NU, E.RLSConfig())
        x, reason = E.heuristic_gate(fresh, reference, jp.int32(0), 0.02, cfg)
        self.assertTrue(bool(jp.all(reason == E.REASON_FEW_SAMPLES)))
        np.testing.assert_array_equal(x, 0.0)
        state = self.feed(0.7, n=40)
        now = state.last_source_step[0] + 1
        x, reason = E.heuristic_gate(state, reference, now, 0.02, cfg)
        self.assertTrue(bool(jp.all(reason == E.REASON_APPLY)))
        q_hat, _ = E.causal_position(state, now, 0.02)
        np.testing.assert_allclose(x, (1 / float(state.g_hat[0]) - 1) * (0.3 - q_hat), rtol=1e-5)
        np.testing.assert_allclose(E.estimated_inverse(state, reference, now, 0.02), x, rtol=1e-6)
        _, reason = E.heuristic_gate(state, reference, state.last_source_step[0] + 9, 0.02, cfg)
        self.assertTrue(bool(jp.all(reason == E.REASON_STALE)))
        healthy = self.feed(1.0, n=40)
        _, reason = E.heuristic_gate(healthy, reference, healthy.last_source_step[0] + 1, 0.02, cfg)
        self.assertTrue(bool(jp.all(reason == E.REASON_DEAD_BAND)))


if __name__ == "__main__":
    unittest.main()
