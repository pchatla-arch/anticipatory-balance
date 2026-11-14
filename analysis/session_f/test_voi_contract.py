"""Exact finite-state CPU checks for a proposed Session F design; no production imports."""
from fractions import Fraction
import unittest


def next_delivery(now, period, latency, horizon):
    p = Fraction(str(period))
    for d in range(now + 1, horizon):
        due = (d * p.denominator) // p.numerator > ((d - 1) * p.denominator) // p.numerator
        if due and d >= latency:
            return d, d - latency
    return None


def blind_value(weights, utilities):
    """Actions are columns; average over hidden states before choosing one action."""
    return max(sum(w * u[a] for w, u in zip(weights, utilities))
               for a in range(len(utilities[0])))


def observed_value(weights, likelihood, utilities):
    """Exact sum_y max_a sum_x b(x) P(y|x) U(x,a); includes empty outcome."""
    return sum(blind_value([w * row[y] for w, row in zip(weights, likelihood)], utilities)
               for y in range(len(likelihood[0])))


def wait_value(weights, alive, prefix, likelihood, utilities):
    """Prefix contains incurred reward and absorbing failure loss, each exactly once."""
    return sum(w * r for w, r in zip(weights, prefix)) + observed_value(
        [w * s for w, s in zip(weights, alive)], likelihood, utilities)


class VoiContractTests(unittest.TestCase):
    def setUp(self):
        self.b = [.5, .5]  # benign, dangerous
        self.u = [[0., -.2], [-1., -.2]]  # terminal CONTINUE, bounded BRACE

    def test_fractional_delivery_phase(self):
        self.assertEqual(next_delivery(8, 2.5, 7, 100), (10, 3))
        self.assertEqual(next_delivery(10, 2.5, 7, 100), (13, 6))

    def test_pending_capture_is_not_future_capture(self):
        d, c = next_delivery(11, 5, 7, 100)
        self.assertEqual((d, c), (15, 8))
        self.assertLess(c, 11)
        self.assertEqual(d - 11, 4)  # not period + latency = 12

    def test_startup_respects_capture_validity(self):
        self.assertEqual(next_delivery(0, 2.5, 7, 100), (8, 1))

    def test_finite_horizon_removes_wait(self):
        self.assertIsNone(next_delivery(8, 2.5, 7, 10))

    def test_uninformative_observation_has_zero_voi(self):
        likelihood = [[.3, .7], [.3, .7]]
        self.assertAlmostEqual(observed_value(self.b, likelihood, self.u), blind_value(self.b, self.u))

    def test_noisy_obs_lies_between_no_info_and_perfect_info(self):
        v0 = blind_value(self.b, self.u)
        vn = observed_value(self.b, [[.9, .1], [.1, .9]], self.u)
        vp = observed_value(self.b, [[1., 0.], [0., 1.]], self.u)
        self.assertAlmostEqual((v0, vn, vp)[0], -.2)
        self.assertAlmostEqual(vn, -.15)
        self.assertAlmostEqual(vp, -.1)
        self.assertLess(v0, vn)
        self.assertLess(vn, vp)

    def test_missing_frame_is_an_explicit_branch(self):
        # Half the frames are independently absent; otherwise observation is perfect.
        v = observed_value(self.b, [[.5, .5, 0.], [.5, 0., .5]], self.u)
        self.assertAlmostEqual(v, -.15)

    def test_informative_absence_uses_its_likelihood(self):
        # An empty frame is more likely in the benign state.
        self.assertAlmostEqual(observed_value([.9, .1], [[1., 0.], [.5, .5]], self.u), -.06)

    def test_maximizing_each_hidden_state_leaks_oracle_information(self):
        legitimate = observed_value(self.b, [[1.], [1.]], self.u)
        leaked = sum(w * max(u) for w, u in zip(self.b, self.u))
        self.assertAlmostEqual(legitimate, -.2)
        self.assertAlmostEqual(leaked, -.1)

    def test_positive_voi_can_still_make_waiting_worse(self):
        delayed = [[0., -.2], [-1., -.8]]
        info = observed_value(self.b, [[1., 0.], [0., 1.]], delayed)
        blind = blind_value(self.b, delayed)
        self.assertGreater(info - blind, 0.)
        self.assertLess(info, blind_value(self.b, self.u))

    def test_failure_before_delivery_has_no_recoursal_action(self):
        v = wait_value(self.b, [1., 0.], [0., -1.], [[1., 0.], [0., 1.]], self.u)
        self.assertAlmostEqual(v, -.5)

    def test_prefix_cost_counted_once(self):
        v = wait_value(self.b, [1., 1.], [-.05, -.05], [[1., 0.], [0., 1.]], self.u)
        self.assertAlmostEqual(v, -.15)


if __name__ == '__main__':
    unittest.main()
