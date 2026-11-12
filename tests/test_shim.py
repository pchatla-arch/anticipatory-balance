"""Verify the parent-source shim rewrites exactly the expected slices (runs without a GPU)."""
import pytest

mp = pytest.importorskip("mujoco_playground")
from foresight.envs.g1_impact import parent_shim as ps  # noqa: E402


def test_shim_builds_and_slices_are_bounded():
  import inspect
  cls = ps.JoystickRobotSliced
  src = inspect.getsource(cls.reset) + inspect.getsource(cls._get_obs) + inspect.getsource(cls._get_reward)
  assert "qpos[7:]" not in src and "qvel[6:]" not in src
  assert f"qpos[7:{ps.NQ_ROBOT}]" in src and f"qvel[6:{ps.NV_ROBOT}]" in src
