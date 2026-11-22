"""CPU development observer for within-hold coverage in the velocity pilot."""
import jax
import jax.numpy as jp
from mujoco import mjx

from foresight.adaptive.actuator_env import AdaptiveJoystick


class ObservedJoystick(AdaptiveJoystick):
    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info["pilot_substep_q"] = jp.zeros((self.n_substeps, self.action_size), jp.float32)
        return state.replace(info=info)

    def observe_substeps(self, data, target):
        """Retain Playground's order of control assignment and MJX integration."""
        def advance(current, _):
            after = mjx.step(self.mjx_model, current.replace(ctrl=target))
            return after, self.measured_joint_positions(after)

        return jax.lax.scan(advance, data, None, length=self.n_substeps)
