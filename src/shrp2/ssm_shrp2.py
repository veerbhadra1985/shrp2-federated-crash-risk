"""
Classical surrogate safety measures on SHRP2 for every subject-vehicle frame, computed by the SAME project
code as in development (src/baselines/ssm.py: compute()), as registered (pre-registration Section 4:
"TTC and DRAC (subject vehicle and its in-lane leader) and TTC2D ... with the 2 m/s moving-pair filter").

Per trip, the subject vehicle and its radar objects (after the frozen object rules of src/shrp2/objects.py)
form one scene; ssm.compute() is run on the scene and the subject's rows are kept: ttc_min, drac_max,
ttc2d_min, ttc2d_mov (inf / 0 when no leader or no conflict, as in development). Masked frames (NaN
kinematics) are left out before the computation, i.e. treated as missing, as registered.
Output : data/processed/shrp2/ssm_shrp2.parquet (scene_id, time_s, agent_id = 'ego', four measures) - the
         input of the LightGBM features (src/shrp2/features_shrp2.py) and of the H1 classical baseline
Usage  : python src/shrp2/ssm_shrp2.py --object-ref asgiven --object-size car
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import objects as OB
from baselines import ssm as SSM

NEED = ["dataset", "scene_id", "time_s", "agent_id", "x", "y", "vx", "vy", "heading", "length", "width"]


def scene_frame(g, ref, size):
    """One trip -> the common-format rows ssm.compute() needs, with the object rules applied."""
    g = g[g[["x", "y", "vx", "vy", "heading"]].notna().all(axis=1)].copy()
    ego = g[g["is_ego"]].set_index("timestamp_ms")
    ob = g[~g["is_ego"]]
    if len(ob):
        ex = ego["x"].reindex(ob["timestamp_ms"]).to_numpy(); ey = ego["y"].reindex(ob["timestamp_ms"]).to_numpy()
        x, y, L, W = OB.apply(ob["x"].to_numpy(), ob["y"].to_numpy(), ob["heading"].to_numpy(), ex, ey, ref, size)
        g.loc[ob.index, "x"], g.loc[ob.index, "y"] = x, y
        g.loc[ob.index, "length"], g.loc[ob.index, "width"] = L, W
    g["time_s"] = g["timestamp_ms"] / 1000.0          # exact grid clock (ssm groups frames by time_s)
    return g[NEED]


def main(traj_dir, ref, size, out=None):
    OB.check(ref, size)
    t0 = time.time(); traj_dir = Path(traj_dir); parts = []
    for p in sorted(traj_dir.glob("traj_*.parquet")):
        df = pd.read_parquet(p)
        for eid, g in df.groupby("event_id", sort=True):
            r = SSM.compute(scene_frame(g, ref, size))
            parts.append(r[r["agent_id"] == "ego"])
        print(f"  {p.name} done | {time.time() - t0:.0f}s", flush=True)
    R = pd.concat(parts, ignore_index=True)
    out = Path(out or traj_dir / "ssm_shrp2.parquet"); R.to_parquet(out, index=False)
    print(f"subject frames: {len(R)} | TTC<1.5 s {(R.ttc_min < 1.5).mean():.4%} | TTC2D<1.5 s {(R.ttc2d_min < 1.5).mean():.4%} "
          f"| TTC2D moving<1.5 s {(R.ttc2d_mov < 1.5).mean():.4%} | DRAC>3.35 {(R.drac_max > 3.35).mean():.4%} | {time.time() - t0:.0f}s")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/processed/shrp2")
    ap.add_argument("--object-ref", required=True, choices=OB.REFS)
    ap.add_argument("--object-size", required=True, choices=OB.SIZES)
    a = ap.parse_args(); main(a.traj_dir, a.object_ref, a.object_size)
