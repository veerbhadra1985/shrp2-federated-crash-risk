"""
Registered LightGBM features on SHRP2 samples (pre-registration Section 4: "current TTC, DRAC and TTC2D,
subject speed and acceleration, number of objects, nearest gap"), computed by the SAME function as in the
rehearsal (src/rehearsal/lgbm_baseline.py: build_features), pointed at the SHRP2 SSM file. Infinite TTC
("no leader / no conflict") becomes missing, as there.
Input : data/processed/shrp2/samples/<split>_<k>.npz, data/processed/shrp2/ssm_shrp2.parquet
Output: data/processed/shrp2/features_<split>.parquet (event_id, frame, eight features, event_bin, obs_bins)
Usage : python src/shrp2/features_shrp2.py   [--p1: P1 inference samples, every frame, -> features_p1_<split>.parquet]
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from rehearsal import lgbm_baseline as LG


def main(traj_dir="data/processed/shrp2", samples="samples", tag=""):
    traj_dir = Path(traj_dir)
    LG.SSM = traj_dir / "ssm_shrp2.parquet"                 # same function, SHRP2 input
    out = {}
    for s in ("train", "val", "test"):
        parts = []
        for p in sorted((traj_dir / samples).glob(f"{s}_*.npz")):
            d = dict(np.load(p, allow_pickle=True))
            keep = (d["current"] == 0) if samples == "samples" else np.ones(len(d["frame"]), bool)   # P1: all frames
            X = LG.build_features(d, keep).reset_index(drop=True)
            X.insert(0, "frame", d["frame"][keep]); X.insert(0, "event_id", d["event_id"][keep])
            X["event_bin"], X["obs_bins"] = d["event_bin"][keep], d["obs_bins"][keep]
            parts.append(X)
        if parts:
            out[s] = pd.concat(parts, ignore_index=True)
            out[s].to_parquet(traj_dir / f"features{tag}_{s}.parquet", index=False)
            print(f"{s}: {len(out[s])} rows")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--traj-dir", default="data/processed/shrp2")
    ap.add_argument("--p1", action="store_true", help="features for the P1 inference samples (samples_p1 -> features_p1_*)")
    a = ap.parse_args(); main(a.traj_dir, "samples_p1" if a.p1 else "samples", "_p1" if a.p1 else "")
