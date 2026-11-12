# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Build a variant of Playground's G1 Joystick class whose qpos/qvel slices stop at the robot.

Playground's env assumes the robot owns every coordinate after the floating base
(qpos[7:], qvel[6:], a (29,) joint vector). Appending a free projectile body adds 7 qpos
and 6 qvel at the end, so those slices must become qpos[7:NQ_ROBOT], qvel[6:NV_ROBOT].
Rather than copy ~500 lines, we rewrite the parent's source text for exactly those
patterns and exec it into a new class. The substitutions are listed explicitly and
unit-tested; nothing in the installed Playground package is modified.
"""
from __future__ import annotations

import inspect
import textwrap

from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

NQ_ROBOT = 7 + 29   # base (7) + 29 actuated joints
NV_ROBOT = 6 + 29

SUBSTITUTIONS = [
    ("qpos[7:]", f"qpos[7:{NQ_ROBOT}]"),
    ("qvel[6:]", f"qvel[6:{NV_ROBOT}]"),
    # reset(): joint randomization uses an explicit (29,) sample and .at[7:]; make the
    # .at slice stop at the robot too.
    (".at[7:].set(", f".at[7:{NQ_ROBOT}].set("),
    # _post_init(): joint limits for "all joints after the base" must exclude the
    # projectile's free joint (joint index 30).
    ("jnt_range[1:]", "jnt_range[1:30]"),
]


def make_robot_sliced_joystick() -> type:
  src = textwrap.dedent(inspect.getsource(g1_joystick.Joystick))
  for old, new in SUBSTITUTIONS:
    if old not in src:
      raise RuntimeError(f"shim pattern {old!r} not found; Playground source changed")
    src = src.replace(old, new)
  src = src.replace("class Joystick(", "class JoystickRobotSliced(", 1)
  ns = dict(vars(g1_joystick))        # same globals the original class body used
  exec(compile(src, "<g1_joystick_shim>", "exec"), ns)
  return ns["JoystickRobotSliced"]


JoystickRobotSliced = make_robot_sliced_joystick()
