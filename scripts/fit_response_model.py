# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Fit the reduced response model P(fall | J, action) = sigmoid(a + b log J) per action from
oracle episodes (contact-conditioned): CONTINUE = lead -1 rows, BRACE = rows at the given
lead with the projectile on. Writes results/response_model.json for the belief rule.

  python scripts/fit_response_model.py --episodes results/E6b_oracle_v2/episodes.csv \
      --brace_lead 0.5 --out results/response_model.json
"""
import csv
import json

import numpy as np
from absl import app, flags

_EP = flags.DEFINE_string("episodes", "results/E6b_oracle_v2/episodes.csv", "E6 episodes")
_LEAD = flags.DEFINE_float("brace_lead", 0.5, "lead whose BRACE rows define P(fall | BRACE)")
_OUT = flags.DEFINE_string("out", "results/response_model.json", "output json")


def fit_logistic(x, y, iters=500, lr=0.5):
  """Newton's method on (a, b) for y ~ sigmoid(a + b x)."""
  X = np.stack([np.ones_like(x), x], 1)
  w = np.zeros(2)
  for _ in range(iters):
    p = 1 / (1 + np.exp(-X @ w))
    g = X.T @ (p - y) / len(y)
    H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + 1e-6 * np.eye(2)
    w -= np.linalg.solve(H, g)
  return w


def main(_):
  rows = list(csv.DictReader(open(_EP.value)))
  def pick(lead, brace):
    r = [q for q in rows if q["launch"] == "1" and q["contact"] == "1" and float(q["lead"]) == lead
         and (q["mode_brace"] == "1") == brace]
    J = np.array([float(q["momentum"]) for q in r]); y = np.array([int(q["fell"]) for q in r], float)
    return J, y
  Jc, yc = pick(-1.0, True); Jb, yb = pick(_LEAD.value, True)
  wc = fit_logistic(np.log(Jc), yc); wb = fit_logistic(np.log(Jb), yb)
  out = {"a_cont": float(wc[0]), "b_cont": float(wc[1]), "a_brace": float(wb[0]), "b_brace": float(wb[1]),
         "n_cont": int(len(yc)), "n_brace": int(len(yb)), "brace_lead": _LEAD.value,
         "source": _EP.value}
  sig = lambda z: 1 / (1 + np.exp(-z))
  for J in (16, 24, 32, 48, 64, 96):
    pc = sig(wc[0] + wc[1] * np.log(J)); pb = sig(wb[0] + wb[1] * np.log(J))
    print(f"J={J:3d} N·s  P(fall|CONT)={pc:.2f}  P(fall|BRACE)={pb:.2f}  dP={pc-pb:+.2f}")
  json.dump(out, open(_OUT.value, "w"), indent=2)
  print(json.dumps(out, indent=2))


if __name__ == "__main__":
  app.run(main)
