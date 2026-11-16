#!/usr/bin/env python3
"""CPU-only scalar checks for the proposed projection; no robot results.

This is an analytical reference for a research proposal, not a pod experiment
or a replacement for the existing known-gain evaluator. Python stdlib only.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def intersect(a, b):
    out = (max(a[0], b[0]), min(a[1], b[1]))
    return out if out[0] <= out[1] else None


def clip(value, bounds):
    return min(bounds[1], max(bounds[0], value))


@dataclass(frozen=True)
class Case:
    name: str
    gains: tuple[float, float]
    errors: tuple[float, float]
    estimate: float = 0.7
    reference: float = 0.1
    target_range: tuple[float, float] = (-0.5, 0.5)
    kp: float = 75.0
    available_force: float = 25.0
    max_offset: float = 0.20
    slew_rate: float = 2.0
    hold: float = 0.02
    previous: float = 0.0
    regularizer: float = 0.0
    current_error: float | None = None


def project(case):
    """Compute exact interval constraints and the scalar quadratic minimizer."""
    c = case
    numeric = []
    for value in asdict(c).values():
        if isinstance(value, (tuple,list)):
            numeric.extend(value)
        elif isinstance(value, (int,float)):
            numeric.append(value)
    if not all(math.isfinite(v) for v in numeric):
        return {"status": "INVALID_INPUT"}
    assert 0 < c.gains[0] <= c.gains[1]
    assert c.errors[0] <= c.errors[1]
    assert c.target_range[0] <= c.reference <= c.target_range[1]
    assert c.kp > 0 and c.available_force > 0 and c.hold > 0
    assert c.slew_rate >= 0 and c.max_offset >= 0 and c.regularizer >= 0
    # The comparison assumes unsaturated nominal and faulty uncorrected forces.
    for gain in c.gains:
        for error in c.errors:
            if max(abs(c.kp * error), abs(gain * c.kp * error)) > c.available_force:
                return {"status": "UNSUPPORTED_SATURATION"}

    physical = (c.target_range[0] - c.reference, c.target_range[1] - c.reference)
    feasible = physical
    for gain in c.gains:
        for error in c.errors:
            # Delta force squared / (Kp^2*g) = g*d^2+2*(g-1)*e*d.
            root = -2.0 * (gain - 1.0) * error / gain
            feasible = intersect(feasible, (min(0.0, root), max(0.0, root)))
            if feasible is None:
                return {"status": "NO_EFFECTIVE_OFFSET"}
            force_band = (-c.available_force / (gain*c.kp) - error,
                          c.available_force / (gain*c.kp) - error)
            feasible = intersect(feasible, force_band)
            if feasible is None:
                return {"status": "NO_EFFECTIVE_OFFSET"}

    # Exact preimage through d(x)=clip(reference+x,target_range)-reference.
    preimage = (-math.inf if feasible[0] <= physical[0] else feasible[0],
                math.inf if feasible[1] >= physical[1] else feasible[1])
    request = intersect(preimage, (-c.max_offset, c.max_offset))
    if request is not None:
        request = intersect(request, (c.previous-c.slew_rate*c.hold,
                                      c.previous+c.slew_rate*c.hold))
    if request is None:
        # This is a reason code, not a certified physical fallback.
        return {"status": "CONSTRAINT_CONFLICT", "effective_interval": feasible}

    point_gain = clip(c.estimate, c.gains)
    # Midpoint is only an illustrative default. Robot integration must provide
    # the separately timestamped causal current-error estimate.
    current_error = sum(c.errors)/2.0 if c.current_error is None else c.current_error
    candidate = (1.0 / point_gain - 1.0) * current_error
    center = (candidate + c.regularizer*c.previous) / (1.0+c.regularizer)
    requested = clip(center, request)
    effective = clip(c.reference + requested, c.target_range) - c.reference
    return {"status": "FEASIBLE", "candidate": candidate,
            "effective_interval": feasible, "requested_interval": request,
            "requested": requested, "effective": effective}


def check_case(case, samples=1001):
    out = project(case)
    if out["status"] != "FEASIBLE":
        return out
    d, x = out["effective"], out["requested"]
    assert abs(x) <= case.max_offset + 1e-12
    assert abs(x-case.previous) <= case.slew_rate*case.hold + 1e-12
    worst_delta = -math.inf
    # Deterministic interior points complement the endpoint derivation.
    rng = random.Random(97)
    points = [(g,e) for g in case.gains for e in case.errors]
    points += [(rng.uniform(*case.gains), rng.uniform(*case.errors)) for _ in range(samples)]
    for g, e in points:
        baseline = case.kp*(g-1)*e
        corrected = case.kp*((g-1)*e+g*d)
        delta = corrected**2 - baseline**2
        worst_delta = max(worst_delta, delta)
        assert delta <= 1e-9, (case.name, g, e, delta)
        assert abs(g*case.kp*(e+d)) <= case.available_force + 1e-9
    out["sampled_max_squared_force_error_change"] = worst_delta
    out["interior_and_corner_points_checked"] = len(points)
    return out


def main():
    cases = [
        Case("informative_gain_interval", (0.65,0.75), (0.08,0.12)),
        Case("wider_interval", (0.60,0.95), (0.08,0.12)),
        Case("healthy_gain_remains_plausible", (0.60,1.02), (0.08,0.12)),
        Case("target_error_crosses_zero", (0.65,0.75), (-0.01,0.01)),
        Case("limited_force_headroom", (0.60,0.80), (0.08,0.12), available_force=9.0),
        Case("slew_conflicts_with_withdrawal", (0.60,1.02), (0.08,0.12), previous=0.08),
        Case("nominal_saturation_invalidates_comparison", (0.65,0.75), (0.08,0.12), available_force=5.0),
        Case("clipped_target_preimage", (0.65,0.75), (0.08,0.12), target_range=(-0.5,0.12)),
        Case("negative_error", (0.65,0.75), (-0.12,-0.08)),
        Case("gain_above_one", (1.05,1.15), (0.08,0.12), estimate=1.10),
    ]
    records = []
    for case in cases:
        records.append({"case": asdict(case), "result": check_case(case)})
    by_name = {r["case"]["name"]: r["result"] for r in records}
    exceptional = {"slew_conflicts_with_withdrawal": "CONSTRAINT_CONFLICT",
                   "nominal_saturation_invalidates_comparison": "UNSUPPORTED_SATURATION"}
    for case in cases:
        assert by_name[case.name]["status"] == exceptional.get(case.name, "FEASIBLE")
    for name in ("healthy_gain_remains_plausible", "target_error_crosses_zero"):
        assert abs(by_name[name]["effective"]) < 1e-12
    assert by_name["slew_conflicts_with_withdrawal"]["status"] == "CONSTRAINT_CONFLICT"
    assert by_name["nominal_saturation_invalidates_comparison"]["status"] == "UNSUPPORTED_SATURATION"
    assert by_name["wider_interval"]["effective"] < by_name["informative_gain_interval"]["effective"]
    for invalid_case in (replace(cases[0], estimate=math.nan),
                         replace(cases[0], current_error=math.inf)):
        assert project(invalid_case)["status"] == "INVALID_INPUT"
    # Grid for a labeled analytical figure; all parameters are illustrative.
    sweep = []
    for i in range(71):
        upper = 0.70 + 0.005*i
        c = Case("interval_width_sweep", (0.60,upper), (0.08,0.12))
        result = check_case(c, samples=101)
        sweep.append({"upper_gain": upper, "candidate": result["candidate"],
                      "requested": result["requested"], "effective": result["effective"]})
    payload = {"scope": "CPU scalar algebra checks and illustrative calculations only; no robot performance measurements",
               "method": "interval projection with same-state unsaturated proportional-force comparison",
               "case_count": len(records), "invalid_input_checks": 2,
               "cases": records, "sweep": sweep}
    path = ROOT / "analytical_check.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"cases_checked": len(records), "invalid_input_checks": 2, "sweep_points_checked": len(sweep),
                      "result": "PASS", "output": str(path)}))


if __name__ == "__main__":
    main()
