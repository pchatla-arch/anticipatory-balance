# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Command updates for the inspected rss2025 G1 observation contract.

The actor's state begins with local velocity (3), gyro (3), gravity (3),
and command (3). The privileged observation begins with that same state.
Updating only this slice preserves the sampled sensor noise and its RNG state.
This helper accepts NumPy or JAX arrays; the caller supplies concatenate.
"""
from collections.abc import Mapping


def replace_command_observation(obs: Mapping, command, *, concatenate):
  """Return observations with the effective command, without mutating inputs.

  Supported contract: rss2025 G1 state/privileged_state observations, one
  history frame. command must already have the same leading dimensions.
  An upstream observation-layout change requires revisiting this adapter.
  """
  if "state" not in obs:
    raise ValueError("G1 command adapter requires a 'state' observation")
  state = obs["state"]
  if state.ndim < 1 or state.shape[-1] < 12:
    raise ValueError("G1 state is too short for its command field")
  expected = state.shape[:-1] + (3,)
  if command.shape != expected:
    raise ValueError(f"command shape {command.shape} does not match {expected}")
  result = dict(obs)
  for key in ("state", "privileged_state"):
    if key not in obs:
      continue
    value = obs[key]
    if value.shape[:-1] != state.shape[:-1] or value.shape[-1] < state.shape[-1]:
      raise ValueError(f"{key} does not contain the expected G1 state prefix")
    cmd = command.astype(value.dtype)
    result[key] = concatenate((value[..., :9], cmd, value[..., 12:]), axis=-1)
  return result
