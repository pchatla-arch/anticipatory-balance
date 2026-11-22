"""Combine the fresh CPU correction and substep-coverage measurements."""
import argparse
import csv
from pathlib import Path

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fresh",type=Path)
    d=p.parse_args().fresh
    coverage=read(d/"coverage.csv")+read(d/"one_tick/coverage.csv")
    authority=read(d/"fresh_authority.csv")+read(d/"one_tick/fresh_authority.csv")
    keys={ (r["channel"],r["joint_group"],r["window"],r["method"],int(r["acceleration_rad_s2"]),float(r["velocity_error_rad_s"])):r for r in coverage }
    rows=[]
    for r in authority:
        method={"speed_magnitude":"velocity_magnitude","signed_velocity":"velocity_signed"}[r["envelope"]]
        k=(r["transport"],r["group"],r["window"],method,int(r["acceleration_rad_s2"]),float(r["velocity_error_rad_s"]))
        c=keys[k]
        assert int(c["n_joint_decisions"])==int(r["observed_joint_ticks"])
        rows.append({**r,**{n:c[n] for n in ("mean_source_age_ms","all_sampled_points_coverage","joint_decisions_failed","interior_only_failures","mean_hold_width_rad")}})
    with (d/"joined.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

if __name__=="__main__":
    main()
