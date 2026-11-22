"""Focused WP 2.2 execution, not a locomotion or authority study."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from foresight.adaptive import candidate as C, estimation as E, sensing as S


class CandidateTests(unittest.TestCase):
    def setUp(self):
        layout = SimpleNamespace(ctrl_range=np.array([[-.5,.5]]),
                                 ctrl_limited=np.ones(1,bool),control_clipping_enabled=True,
                                 actuator_force_limited=np.array([True]),actuator_force_range=np.array([[-10.,12.]]),
                                 joint_force_limited=np.array([True]),joint_force_range=np.array([[-9.,8.]]))
        env = SimpleNamespace(_telemetry_layout=layout,action_size=1,
                              mj_model=SimpleNamespace(actuator_gainprm=np.array([[75.]])),
                              max_correction_rad=.2,max_correction_rate_rad_s=2.,dt=.02)
        cfg = C.Config(gain_drift_per_s=0.,hold_speed_rad_s=.1,force_margin_nm=0.,
                       roundoff_effort_bound_nm=1e-5,reacquire_packets=2,regularizer=0.)
        self.c=C.Candidate.bind(env,S.EXACT,E.RLSConfig(),cfg)
        self.s=self.c.init(1,1)
        self.r=jax.vmap(lambda _: E.rls_init(1,E.RLSConfig()))(jnp.arange(1))

    def packet(self,t,g,released=True,valid=True):
        p=S.MeasuredPacket(jnp.array([t]),jnp.array([t]),jnp.array([[.1]]),jnp.zeros((1,1)),
                           jnp.zeros((1,1)),jnp.array([[g*7.5]]),jnp.array([[valid]]))
        mask = jnp.array([released])
        self.r=jax.vmap(lambda s,pk,m:E.rls_update(s,pk,m,self.c.kp,E.RLSConfig()))(self.r,p,mask)
        self.s=self.c.update(self.s,p,mask)

    def command(self,t,previous=0.,ref=.1):
        return jax.jit(self.c.correction)(self.s,self.r,jnp.array([[ref]]),t,jnp.array([[previous]]))

    def test_identity_and_physical_limits(self):
        self.assertEqual(float(self.c.force_ceiling[0]),8.)
        for t in range(3):
            self.packet(t,1.)
        req, reason=self.command(3)
        self.assertEqual(float(req[0,0]),0.)
        self.assertEqual(int(reason[0,0]),C.Reason.NO_EFFECTIVE_CHANGE)
        req, reason=self.command(3,ref=.2)
        self.assertEqual(int(reason[0,0]),C.Reason.UNSUPPORTED_FORCE)
        self.assertEqual(float(req[0,0]),0.)

    def test_reacquire_after_change_and_slew_limited_withdrawal(self):
        self.packet(0,.7)
        self.assertEqual(int(self.command(1)[1][0,0]),C.Reason.REACQUIRING)
        self.packet(1,.7)
        req,reason=self.command(2)
        self.assertGreater(float(req[0,0]),0.)
        self.assertLessEqual(float(req[0,0]),.040001)
        self.assertEqual(int(reason[0,0]),C.Reason.APPLY)
        self.packet(2,1.)
        req,reason=self.command(3,previous=.10)
        self.assertAlmostEqual(float(req[0,0]),.06,places=6)
        self.assertEqual(int(reason[0,0]),C.Reason.MODEL_MISMATCH)
        self.packet(3,1.)
        self.assertEqual(int(self.command(4)[1][0,0]),C.Reason.REACQUIRING)
        self.packet(4,1.)
        # A healthy interval requires zero, but a previous 0.10 request cannot
        # reach zero in one hold. Numerically valid withdrawal is uncertified.
        req,reason=self.command(5,previous=.1)
        self.assertEqual(int(reason[0,0]),C.Reason.CONSTRAINT_CONFLICT)
        self.assertAlmostEqual(float(req[0,0]),.06,places=6)

    def test_padding_and_saturation_preserve_evidence_and_clock(self):
        self.packet(5,.7)
        self.packet(6,.7)
        before = self.s.gains
        self.assertLessEqual(float(before.low[0,0]), .7)
        self.assertGreaterEqual(float(before.high[0,0]), .7)
        self.packet(100,1.,released=False)
        np.testing.assert_array_equal(self.s.gains, before)
        self.assertEqual(int(self.s.last_source[0,0]),6)
        self.packet(7,1.,valid=False)
        self.assertEqual(int(self.s.last_source[0,0]),7)
        np.testing.assert_array_equal(self.s.gains, before)
        self.assertEqual(int(self.s.informative_count[0,0]),2)
        np.testing.assert_allclose(self.c.config.hold_speed_rad_s,.1)


if __name__ == "__main__":
    unittest.main()
