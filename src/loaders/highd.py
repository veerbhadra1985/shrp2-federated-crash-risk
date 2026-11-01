"""
highD loader -> the project's common trajectory format, resampled to 10 Hz.

highD conventions (official format description, LevelX "highD Dataset Format"):
  x, y          upper-left corner of the vehicle's bounding box [m]; image coordinates
                (origin top-left, y pointing DOWN)
  width/height  bounding-box size: width = vehicle LENGTH, height = vehicle WIDTH
  drivingDirection 1 = left (upper lanes, negative x-velocity), 2 = right (lower lanes)
  frameRate     25 Hz (per recordingMeta)
Conversion: centre = (x + width/2, y + height/2); y, yVelocity, yAcceleration negated so the
y-axis points UP (as in INTERACTION); heading = atan2(vy, vx); linear interpolation to 10 Hz.
Output: one parquet per recording in data/processed/highd/ (avoids one huge file in memory).
Usage : python src/loaders/highd.py            (add --max-recordings 2 for a quick test)
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw/highD/highD-dataset-v1.0/data")
OUT = Path("data/processed/highd")
HZ = 10
COLS = ["dataset", "scene_id", "time_s", "agent_id", "is_ego", "x", "y", "vx", "vy",
        "ax", "ay", "heading", "length", "width", "agent_type", "lane_id", "driving_direction"]


def convert_recording(rec, raw=RAW, hz=HZ):
    meta = pd.read_csv(raw / f"{rec:02d}_recordingMeta.csv").iloc[0]
    tmeta = pd.read_csv(raw / f"{rec:02d}_tracksMeta.csv")
    tr = pd.read_csv(raw / f"{rec:02d}_tracks.csv",
                     usecols=["frame", "id", "x", "y", "width", "height", "xVelocity", "yVelocity",
                              "xAcceleration", "yAcceleration", "laneId"])
    fr = float(meta["frameRate"])
    tr = tr.sort_values(["id", "frame"]).reset_index(drop=True)
    t = (tr["frame"].to_numpy() - 1) / fr
    cx = tr["x"].to_numpy() + tr["width"].to_numpy() / 2
    cy = -(tr["y"].to_numpy() + tr["height"].to_numpy() / 2)          # flip: y up
    vx, vy = tr["xVelocity"].to_numpy(), -tr["yVelocity"].to_numpy()
    ax, ay = tr["xAcceleration"].to_numpy(), -tr["yAcceleration"].to_numpy()
    ids = tr["id"].to_numpy()
    cut = np.flatnonzero(np.diff(ids)) + 1
    info = tmeta.set_index("id")
    out = []
    for seg in np.split(np.arange(len(ids)), cut):
        if len(seg) < 2:
            continue
        ts = t[seg]
        grid = np.arange(np.ceil(ts[0] * hz - 1e-9), np.floor(ts[-1] * hz + 1e-9) + 1) / hz
        if len(grid) == 0:
            continue
        f = lambda a: np.interp(grid, ts, a[seg])
        gx, gy, gvx, gvy = f(cx), f(cy), f(vx) + 0.0, f(vy) + 0.0   # +0.0 removes negative zeros
        lane = tr["laneId"].to_numpy()[seg][np.clip(np.searchsorted(ts, grid), 0, len(seg) - 1)]
        vid = ids[seg[0]]; m = info.loc[vid]
        out.append(pd.DataFrame({
            "dataset": "highD", "scene_id": f"highD_L{int(meta['locationId'])}/{rec:02d}",
            "time_s": np.round(grid, 1) + 0.0, "agent_id": str(vid), "is_ego": False,
            "x": gx, "y": gy, "vx": gvx, "vy": gvy, "ax": f(ax), "ay": f(ay),
            "heading": np.arctan2(gvy, gvx), "length": float(m["width"]), "width": float(m["height"]),
            "agent_type": str(m["class"]).lower(), "lane_id": lane.astype(int),
            "driving_direction": int(m["drivingDirection"])}))
    return pd.concat(out, ignore_index=True)[COLS], int(meta["numVehicles"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-recordings", type=int, default=None)
    a = ap.parse_args()
    recs = sorted(int(p.name[:2]) for p in RAW.glob("*_tracks.csv"))
    if a.max_recordings:
        recs = recs[: a.max_recordings]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); tot_rows = tot_veh = 0; mism = []; speeds = []; classes = {}; locs = set()
    for r in recs:
        df, n_meta = convert_recording(r)
        df.to_parquet(OUT / f"highd_{r:02d}.parquet", index=False)
        nveh = df["agent_id"].nunique(); tot_rows += len(df); tot_veh += nveh
        if nveh != n_meta:
            mism.append((r, nveh, n_meta))
        speeds.append(np.hypot(df.vx, df.vy).to_numpy()[::50])
        for k, v in df.groupby("agent_id")["agent_type"].first().value_counts().items():
            classes[k] = classes.get(k, 0) + v
        locs.add(df.scene_id.iloc[0].split("/")[0])
        assert not df[["x", "y", "vx", "vy", "heading"]].isna().any().any(), f"NaN in recording {r}"
        print(f"  [{r:02d}] {nveh:5d} vehicles | {len(df):8,d} rows | {time.time()-t0:5.0f}s", flush=True)
    sp = np.concatenate(speeds)
    print(f"\nrecordings {len(recs)} | locations {len(locs)} ({', '.join(sorted(locs))})")
    print(f"vehicles {tot_veh:,} | rows at {HZ} Hz {tot_rows:,} | classes {classes}")
    print(f"speed (sample) median {np.median(sp):.1f} m/s | 99th pct {np.percentile(sp, 99):.1f} m/s | max {sp.max():.1f} m/s")
    print("vehicle counts match recordingMeta:", "yes" if not mism else f"NO {mism}")
    print(f"saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()