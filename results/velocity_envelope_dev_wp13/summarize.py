"""Join the two independently produced tables for the G1 engineering review."""
import csv
from pathlib import Path
import argparse
import json

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory",type=Path)
    d=ap.parse_args().directory
    methods={"speed10":"speed_10","speed_magnitude":"velocity_magnitude","signed_velocity":"velocity_signed"}
    cov={(r["family"],r["telemetry"],r["joint_group"],r["window"],r["envelope"],r["accel_rad_s2"] or "0"):r
         for r in read(d/"coverage.csv") if r["age_ticks"]=="all"}
    rows=[]
    for r in read(d/"authority.csv"):
        k=(r["family"],r["channel"],r["group"],r["window"],methods[r["envelope"]],r["acceleration_rad_s2"])
        c=cov[k]
        n=int(c["scored_joint_steps"])
        rows.append({**r,"coverage_eligible":n,"coverage_missing_source":c["missing_source_all_ages"],
                     "two_endpoint_coverage":float(c["both_pointwise_coverage"]),
                     "hold_interval_endpoint_coverage":1-int(c["outside_hold_hull"])/n})
    with (d/"joined.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    qualifying={}
    for w in ("first_second","all_post_onset"):
        for key in ("two_endpoint_coverage","hold_interval_endpoint_coverage"):
            for env in ("speed_magnitude","signed_velocity"):
                cells=[x for x in rows if (x["family"],x["channel"],x["window"],x["envelope"])==("gain_070","declared",w,env)]
                passed=[]
                for a in sorted({int(x["acceleration_rad_s2"]) for x in cells}):
                    pair=[r for r in cells if int(r["acceleration_rad_s2"])==a]
                    assert len(pair)==2 and {r["group"] for r in pair}=={"hips","knees"}
                    if all(r[key]>=.99 and float(r["applied_over_oracle"])>=.5 for r in pair):
                        passed.append(a)
                qualifying[f"{w}/{key}/{env}"]=passed
    (d/"qualifying_bounds.json").write_text(json.dumps(qualifying,indent=2)+"\n")

if __name__=="__main__":
    main()
