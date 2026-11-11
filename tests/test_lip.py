import numpy as np
from foresight.lip.capture_point import (SupportPolygon, capture_point, j_max, omega,
                                         predicted_fall)


def test_omega_and_cp():
  assert abs(omega(0.7) - np.sqrt(9.81 / 0.7)) < 1e-9
  cp = capture_point(np.zeros(2), np.array([0.5, 0.0]), 0.7)
  assert cp[0] > 0 and abs(cp[1]) < 1e-12


def test_lower_com_and_wider_stance_raise_jmax():
  nominal = SupportPolygon.from_feet([0, 0.1], [0, -0.1])
  wide = SupportPolygon.from_feet([0, 0.2], [0, -0.2])
  cp = np.zeros(2)
  push = np.array([0.0, 1.0])  # lateral push
  j_tall = j_max(35.0, 0.75, nominal, cp, push)
  j_low = j_max(35.0, 0.55, nominal, cp, push)
  j_wide = j_max(35.0, 0.75, wide, cp, push)
  assert j_low > j_tall and j_wide > j_tall


def test_predicted_fall_monotone():
  poly = SupportPolygon.from_feet([0, 0.1], [0, -0.1])
  small = predicted_fall(35.0, 0.7, poly, np.zeros(2), np.zeros(2), 5.0, np.array([0, 1.0]))
  big = predicted_fall(35.0, 0.7, poly, np.zeros(2), np.zeros(2), 200.0, np.array([0, 1.0]))
  assert (not small) and big
