"""
Step 20a: target statistics from highD for calibrating the SUMO motorway scenario.

Per recording and driving direction (same definitions as src/sim/highway.py summarise):
  flow_veh_h            vehicles per hour in this direction
  speed_mean/p15/p50/p85  over all vehicle-frames (km/h); also for cars and trucks
  truck_share           share of vehicles that are trucks
  time_gap_median_s, time_gap_below_1s_share
                        bumper-to-bumper gap / follower speed (follower faster than 5 m/s)
  lane_changes_per_veh_km
  ttc_below_3s_share, ttc_below_1_5s_share  share of follower-frames with TTC below the threshold
  car/truck_share_right_lane, _left_lane  share of car (truck) frames in the rightmost / leftmost lane
Grouping: lanes per direction (from recordingMeta lane markings) and traffic state
  (congested if median speed < 60 km/h, else free-flowing).
Usage : python src/sim/highd_targets.py
Input : data/processed/highd/*.parquet (from src/loaders/highd.py), data/raw/highD/.../XX_recordingMeta.csv
Output: results/highd_targets_by_direction.csv, results/highd_targets_summary.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed/highd")
RAW = Path("data/raw/highD/highD-dataset-v1.0/data")
CONGESTED_KMH = 60.0


def follower_pairs(df):
    """Vectorised leader-follower pairs within (time, direction, lane).
    Progress coordinate s = x for direction 2 (moving +x), s = -x for direction 1."""
    d = df.copy()
    sign = np.where(d["driving_direction"].to_numpy() == 2, 1.0, -1.0)
    d["s"] = d["x"].to_numpy() * sign
    d["v"] = d["vx"].to_numpy() * sign                       # speed along travel direction
    d = d.sort_values(["time_s", "driving_direction", "lane_id", "s"]).reset_index(drop=True)
    same = ((d["time_s"].values[1:] == d["time_s"].values[:-1]) &
            (d["driving_direction"].values[1:] == d["driving_direction"].values[:-1]) &
            (d["lane_id"].values[1:] == d["lane_id"].values[:-1]))
    f = np.flatnonzero(same); l = f + 1                       # follower i, leader i+1
    gap = d["s"].values[l] - d["s"].values[f] - (d["length"].values[l] + d["length"].values[f]) / 2
    vf, vl = d["v"].values[f], d["v"].values[l]
    return pd.DataFrame({"driving_direction": d["driving_direction"].values[f], "gap": gap, "vf": vf, "vl": vl})


def stats_for(rec_df, meta, direction):
    d = rec_df[rec_df["driving_direction"] == direction]
    if d.empty:
        return None
    marks = str(meta["upperLaneMarkings" if direction == 1 else "lowerLaneMarkings"]).split(";")
    duration_h = float(meta["duration"]) / 3600
    veh = d.groupby("agent_id").agg(t=("agent_type", "first"), n=("time_s", "size"))
    sp = np.hypot(d["vx"], d["vy"]).to_numpy() * 3.6
    is_truck = (d["agent_type"] == "truck").to_numpy()
    lane_chg = (d.sort_values(["agent_id", "time_s"]).groupby("agent_id")["lane_id"].apply(lambda s: int((s.diff().fillna(0) != 0).sum()))).sum()
    dist_km = (np.hypot(d["vx"], d["vy"]).sum() * 0.1) / 1000
    lo, hi = d["lane_id"].min(), d["lane_id"].max()
    pos = (hi - d["lane_id"]) if direction == 2 else (d["lane_id"] - lo)      # 0 = rightmost lane
    nl = len(marks) - 1
    lane_sh = {}
    for kind in ("car", "truck"):
        k = d["agent_type"] == kind
        lane_sh[f"{kind}_share_right_lane"] = float((pos[k] == 0).mean()) if k.any() else np.nan
        lane_sh[f"{kind}_share_left_lane"] = float((pos[k] == nl - 1).mean()) if k.any() else np.nan
    p = follower_pairs(d)
    p = p[(p["gap"] > 0) & (p["vf"] > 5)]
    tg = (p["gap"] / p["vf"]).to_numpy()
    closing = p["vf"] > p["vl"]
    ttc = (p["gap"][closing] / (p["vf"][closing] - p["vl"][closing])).to_numpy()
    return dict(recording=d["scene_id"].iloc[0], location=d["scene_id"].iloc[0].split("/")[0],
                direction=direction, lanes=len(marks) - 1, speed_limit_kmh=float(meta["speedLimit"]) * 3.6 if float(meta["speedLimit"]) > 0 else -1,
                vehicles=len(veh), flow_veh_h=round(len(veh) / duration_h), truck_share=round((veh.t == "truck").mean(), 3),
                speed_mean_kmh=round(sp.mean(), 1), speed_p15_kmh=round(np.percentile(sp, 15), 1),
                speed_p50_kmh=round(np.percentile(sp, 50), 1), speed_p85_kmh=round(np.percentile(sp, 85), 1),
                car_speed_mean_kmh=round(sp[~is_truck].mean(), 1) if (~is_truck).any() else np.nan,
                truck_speed_mean_kmh=round(sp[is_truck].mean(), 1) if is_truck.any() else np.nan,
                time_gap_median_s=round(float(np.median(tg)), 2) if len(tg) else np.nan,
                time_gap_below_1s_share=round(float((tg < 1).mean()), 3) if len(tg) else np.nan,
                lane_changes_per_veh_km=round(lane_chg / dist_km, 3) if dist_km > 0 else np.nan,
                ttc_below_3s_share=float((ttc < 3).sum() / max(len(p), 1)),
                ttc_below_1_5s_share=float((ttc < 1.5).sum() / max(len(p), 1)),
                follower_frames=len(p), **lane_sh)


def main():
    rows = []
    files = sorted(PROC.glob("highd_*.parquet"))
    for k, fpath in enumerate(files, 1):
        rec = int(fpath.stem.split("_")[1])
        meta = pd.read_csv(RAW / f"{rec:02d}_recordingMeta.csv").iloc[0]
        df = pd.read_parquet(fpath)
        for direction in (1, 2):
            r = stats_for(df, meta, direction)
            if r: rows.append(r)
        print(f"  [{k}/{len(files)}] {fpath.stem}", flush=True)
    t = pd.DataFrame(rows)
    t["state"] = np.where(t["speed_p50_kmh"] < CONGESTED_KMH, "congested", "free-flowing")
    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/highd_targets_by_direction.csv", index=False)
    g = t.groupby(["lanes", "state"])
    s = g.agg(directions=("vehicles", "size"), vehicles=("vehicles", "sum"),
              flow_veh_h_per_lane=("flow_veh_h", "median"), truck_share=("truck_share", "median"),
              speed_mean_kmh=("speed_mean_kmh", "median"), speed_p15_kmh=("speed_p15_kmh", "median"),
              speed_p85_kmh=("speed_p85_kmh", "median"), time_gap_median_s=("time_gap_median_s", "median"),
              time_gap_below_1s_share=("time_gap_below_1s_share", "median"),
              lane_changes_per_veh_km=("lane_changes_per_veh_km", "median"),
              ttc_below_3s_share=("ttc_below_3s_share", "median"),
              ttc_below_1_5s_share=("ttc_below_1_5s_share", "median"),
              car_share_left_lane=("car_share_left_lane", "median"),
              truck_share_right_lane=("truck_share_right_lane", "median")).reset_index()
    s["flow_veh_h_per_lane"] = (s["flow_veh_h_per_lane"] / s["lanes"]).round()
    s.to_csv("results/highd_targets_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("\nTargets by lanes per direction and traffic state (medians across recording-directions):")
    print(s.to_string(index=False))
    print("\nSaved results/highd_targets_by_direction.csv and results/highd_targets_summary.csv")


if __name__ == "__main__":
    main()