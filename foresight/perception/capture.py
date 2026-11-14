# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Capture-time validity for delayed observations; works with Python or JAX."""


def captured_launch_valid(launch_history, capture_step):
  """Read launch status at the captured frame, rejecting uninitialized history.

  The caller writes current launch status alongside the current object pose before
  reading this ring buffer, and validates latency < buffer length. In particular,
  the object's current launch status must not make an older parked frame valid.
  """
  return (capture_step >= 0) & launch_history[capture_step % len(launch_history)]
