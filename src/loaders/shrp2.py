"""
SHRP2 bird's-eye dataset -> the project's common trajectory format (same columns as src/loaders/highd.py),
plus a trips table. Runs ONLY on the author's encrypted laptop; outputs stay in data/ (outside Git).

Conventions (src/shrp2/shrp2_format.py, from the dataset authors' documentation):
  subject vehicle  x_ekf, y_ekf = centroid [m]; psi_ekf = heading from the x-axis [rad]; v_ekf speed along the
                   heading [m/s]; acc_ekf acceleration along the heading [m/s^2]  -> vx = v cos psi, vy = v sin psi
  radar objects    x_ekf, y_ekf = the radar-detected reference point AS GIVEN (documented as the front bumper;
                   the authors' code places it relative to the subject's front bumper and does not move it to
                   the object's centre); psi_ekf, v_ekf as above; size not documented -> NaN
  time             'time' in the files is the 0.1-s grid [s]; 'timestamp' is the raw ms stamp, or -1 where the
                   authors interpolated. Clock: time_s = round(time * 1000) / 1000; raw stamp kept as raw_timestamp.
                   Duplicate grid times within a trip (possible when two raw samples round to the same 0.1 s)
                   keep the first row; the number dropped is in trips.parquet
Registered rules applied (pre-registration Section 2; --no-mask turns them off for sensitivity analysis 8):
  * speeds > 60 m/s, |acceleration| > 15 m/s^2 outside +-1 s of a crash's impact time, and object position
    jumps > 10 m are MASKED (kinematics set to NaN, row kept, masked = True), never deleted;
  * a trip with > 10 % implausible subject-vehicle frames is excluded (listed in trips.parquet, no rows written);
  * event trips enter only if they pass the registered GSSM test-set filters (duration_enough and conflict
    != 'none') and have an impact time (C26); all reconstructed safe baseline trips enter.
  * outcome type = the event category mapped as in the GSSM code (C25, shrp2_format.OUTCOME_BY_CATEGORY);
    the +-1 s crash window of the acceleration rule applies to trips of outcome type crash.
NOT decided here (audit-dependent choices, to be frozen and logged by the author before any model is trained):
  object reference point (as given vs shifted to a centre), object size (NaN vs a fixed default), and the
  domain-factor handling of trips without environment data (baselines). The loader keeps the raw facts.
Outputs data/processed/shrp2/traj_<category>[_<chunk>].parquet  and  data/processed/shrp2/trips.parquet
Usage : python src/loaders/shrp2.py --root <folder with ReconstructedTrajectories and SafetyCriticalTestSet>
        [--out data/processed/shrp2] [--no-mask]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shrp2"))
import shrp2_format as F
from audit_counts import find_dirs, category_files, read_h5, implausible_ego, object_flags, binary_factor, clock_ms, dedup_objects, outcome_of

COLS = ["dataset", "scene_id", "time_s", "agent_id", "is_ego", "x", "y", "vx", "vy",
        "ax", "ay", "heading", "length", "width", "agent_type", "lane_id", "driving_direction"]
EXTRA = ["event_id", "timestamp_ms", "raw_timestamp", "masked"]   # timestamp_ms = grid clock; raw_timestamp -1 = interpolated
KIN = ["x", "y", "vx", "vy", "ax", "ay", "heading"]


def load_event_tables(st):
    em = pd.concat([pd.read_csv(p) for p in sorted(st.rglob("event_meta.csv"))], ignore_index=True) if st else pd.DataFrame()
    envf = sorted(st.rglob("environment.csv")) if st else []
    env = pd.concat([pd.read_csv(p) for p in envf], ignore_index=True) if envf else pd.DataFrame(columns=F.ENV_COLS)
    if len(em) and em["event_id"].duplicated().any():
        raise SystemExit("event IDs repeated in event_meta.csv - run the audit and send its summary")
    env = env.drop_duplicates("event_id", keep="first")
    if len(em):
        em["sev"] = em[["severity_first", "severity_second"]].max(axis=1)
        em["included"] = em["duration_enough"].astype(str).str.lower().isin(["true", "1"]) & (em["conflict"].astype(str) != "none")
    return em, env


def convert_file(ego, sur, meta, em, mask=True):
    """One pair of ego/surrounding frames (one category file). Returns (trajectory rows, trip rows)."""
    out, trips = [], []
    ego = ego.sort_values(["event_id", "time"], kind="mergesort")
    sur_by = dict(tuple(sur.groupby("event_id"))) if len(sur) else {}
    for eid, g0 in ego.groupby("event_id", sort=False):
        fm0 = clock_ms(g0["time"]); keep = np.r_[True, np.diff(fm0) != 0]   # duplicate grid times: keep first
        g = g0[keep]
        cat = g["category"].iat[0]; is_base = cat == F.BASELINE_CATEGORY
        inc = bool(eid in em.index and em.at[eid, "included"])
        te = int(em.at[eid, "impact_timestamp"]) if (inc and pd.notna(em.at[eid, "impact_timestamp"])) else None
        otype = outcome_of(cat)                                     # C25: event category, as the GSSM code
        if otype == "UNMAPPED":
            raise SystemExit(f"event category {cat!r} has no outcome type (C25) - run the audit and send its summary")
        in_study = is_base or (inc and te is not None)              # C26: an event without impact time is not a unit
        ts = fm0[keep]
        sp, ac = implausible_ego(ts, g["v_ekf"].to_numpy(), g["acc_ekf"].to_numpy(), te, otype == "crash")
        bad = sp | ac
        excluded = bool(bad.mean() > F.IMPLAUSIBLE["trip_share"])
        mrow = meta.loc[eid] if eid in meta.index else None
        trips.append(dict(event_id=eid, category=cat,
                          outcome_type=otype,
                          in_study=in_study, excluded_implausible=excluded, event_time_ms=te,
                          start_ms=int(ts[0]), end_ms=int(ts[-1]), n_frames=len(ts), duplicate_times_dropped=int((~keep).sum()),
                          ego_frames_masked=int(bad.sum()),
                          ego_length=float(mrow["ego_length"]) if mrow is not None and "ego_length" in mrow else np.nan,
                          ego_width=float(mrow["ego_width"]) if mrow is not None and "ego_width" in mrow else np.nan))
        if not in_study or (excluded and mask):
            continue
        v, psi, acc = g["v_ekf"].to_numpy(float), g["psi_ekf"].to_numpy(float), g["acc_ekf"].to_numpy(float)
        e = pd.DataFrame(dict(dataset="SHRP2", scene_id=f"SHRP2/{eid}", time_s=ts / 1000.0, agent_id="ego", is_ego=True,
                              x=g["x_ekf"].to_numpy(float), y=g["y_ekf"].to_numpy(float), vx=v * np.cos(psi),
                              vy=v * np.sin(psi), ax=acc * np.cos(psi), ay=acc * np.sin(psi), heading=psi,
                              length=trips[-1]["ego_length"], width=trips[-1]["ego_width"], agent_type="subject",
                              lane_id=-1, driving_direction=0, event_id=eid, timestamp_ms=ts,
                              raw_timestamp=g["timestamp"].to_numpy(np.int64), masked=bad if mask else False))
        if mask:
            e.loc[bad, KIN] = np.nan
        parts = [e]
        s = sur_by.get(eid)
        if s is not None and len(s):
            s = dedup_objects(s)                                    # duplicate (object, grid time): keep first
            sfm = clock_ms(s["time"])
            osp, ojp = object_flags(s)
            obad = osp | ojp
            sv, sp_ = s["v_ekf"].to_numpy(float), s["psi_ekf"].to_numpy(float)
            o = pd.DataFrame(dict(dataset="SHRP2", scene_id=f"SHRP2/{eid}", time_s=sfm / 1000.0,
                                  agent_id=s["target_id"].astype(np.int64).astype(str).to_numpy(), is_ego=False,
                                  x=s["x_ekf"].to_numpy(float), y=s["y_ekf"].to_numpy(float), vx=sv * np.cos(sp_),
                                  vy=sv * np.sin(sp_), ax=np.nan, ay=np.nan, heading=sp_, length=np.nan, width=np.nan,
                                  agent_type="radar_object", lane_id=-1, driving_direction=0, event_id=eid,
                                  timestamp_ms=sfm, raw_timestamp=s["timestamp"].to_numpy(np.int64),
                                  masked=obad if mask else False))
            if mask:
                o.loc[obad, KIN] = np.nan
            parts.append(o)
        out.append(pd.concat(parts, ignore_index=True))
    rows = pd.concat(out, ignore_index=True)[COLS + EXTRA] if out else pd.DataFrame(columns=COLS + EXTRA)
    return rows, trips


def main(root, out, mask=True):
    t0 = time.time()
    rt, st = find_dirs(root)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(rt / "metadata_birdseye.csv").set_index("event_id")
    em, env = load_event_tables(st)
    emi = em.set_index("event_id") if len(em) else pd.DataFrame(columns=["included", "impact_timestamp", "sev"])
    all_trips, n_rows = [], 0
    for cat, (ef, sf) in category_files(rt).items():
        sur_map = {p.name.replace(F.SUR_FILE, ""): p for p in sf}
        for p in ef:
            suffix = p.name.replace(F.EGO_FILE, "")              # ".h5" or "_<chunk>.h5"
            ego = read_h5(p).assign(category=cat)
            sur = read_h5(sur_map[suffix]) if suffix in sur_map else pd.DataFrame(columns=F.SUR_COLS_PUBLIC)
            rows, trips = convert_file(ego, sur, meta, emi, mask)
            name = f"traj_{cat}{suffix.replace('.h5', '')}.parquet"
            rows.to_parquet(out / name, index=False)
            n_rows += len(rows); all_trips += trips
            print(f"  {name}: {rows['scene_id'].nunique() if len(rows) else 0} trips, {len(rows)} rows | {time.time() - t0:.0f}s", flush=True)
    T = pd.DataFrame(all_trips)
    T["event_time_ms"] = T["event_time_ms"].astype("Int64")
    if len(env):
        e = env.set_index("event_id")
        for f in F.ENV_COLS[1:]:
            T[f] = T.event_id.map(e[f]) if f in e else np.nan
            T[f + "_bin"] = [binary_factor(f, v) if pd.notna(v) else None for v in T[f]]
    T.to_parquet(out / "trips.parquet", index=False)
    print(f"trips: {len(T)} | in study: {int(T.in_study.sum())} | excluded (implausible): {int(T.excluded_implausible.sum())} "
          f"| rows written: {n_rows} | mask={'on' if mask else 'OFF (sensitivity 8)'} | {time.time() - t0:.0f}s")
    return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--out", default="data/processed/shrp2")
    ap.add_argument("--no-mask", action="store_true")
    a = ap.parse_args()
    if not Path(a.out).as_posix().startswith("data/"):
        raise SystemExit("--out must be inside data/ (outside Git): outputs contain licensed trajectories")
    main(a.root, a.out if not a.no_mask else a.out.rstrip("/\\") + "_nomask", mask=not a.no_mask)
