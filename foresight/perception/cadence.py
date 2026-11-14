# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Control-grid observation timing, usable with Python integers or JAX arrays."""
from fractions import Fraction
import math
import operator


def _period_ratio(period):
  """Return a positive rational number of control steps per observation."""
  if not math.isfinite(period) or period < 1:
    raise ValueError("detector period must be finite and >= 1 control step")
  ratio = Fraction(str(period))
  return ratio.numerator, ratio.denominator


def delivery_due(control_step, period):
  """Deliver at the first control tick at or after each scheduled sample.

  Integer periods retain delivery at steps 0, period, 2*period, ... .
  A 2.5-step period alternates 3- and 2-step gaps (0, 3, 5, 8, ...),
  providing exactly 20 observations per second on a 50 Hz control grid.
  The phase is attached to delivery, preserving the existing delay convention.
  """
  numerator, denominator = _period_ratio(period)
  return ((control_step * denominator) // numerator >
          ((control_step - 1) * denominator) // numerator)


def validate_detector_timing(period, latency, buffer_len):
  """Reject an impossible cadence or a delay that aliases the history buffer."""
  _period_ratio(period)
  try:
    delay = operator.index(latency)
    length = operator.index(buffer_len)
  except TypeError as exc:
    raise ValueError("detector latency and buffer_len must be integers") from exc
  if delay < 0:
    raise ValueError("detector latency must be nonnegative")
  if length < 1:
    raise ValueError("detector buffer_len must be positive")
  if delay >= length:
    raise ValueError("detector buffer_len must be >= latency + 1")
