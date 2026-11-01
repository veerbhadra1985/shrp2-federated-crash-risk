"""
SYNTHETIC SHRP2-format dataset with KNOWN answers, for testing the readiness kit (no SHRP2 data involved).
Layout, file names, h5 key and columns follow src/shrp2/shrp2_format.py (the dataset authors' documentation).
The ground truth is computed here with plain per-frame loops, independently of the audit code, and saved to
<out>/truth.json, so the tests compare two separate implementations.

Planted cases: events with short pre-event history (horizon retention), event trips failing the inclusion
filters (duration_enough False / conflict 'none'), unknown environment values, a baseline trip with > 10 %
implausible ego frames (excluded), an implausible acceleration inside and one outside the 1-s crash window,
an object position jump > 10 m, an object above 60 m/s, a trip that is listed in the metadata but not
reconstructed, a short trip below the 100-frame eligibility, an object reference offset in the test set.
Usage : python src/shrp2/make_synthetic.py --out data/synthetic_shrp2 [--variant complete|public|driver_id|missing_col]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shrp2_format as F

REF_OFFSET = 2.25          # planted: test-set x_sur = Surrounding x_ekf + 2.25 m along the object's heading


def ego_track(rng, n, t0_ms, v0):
    psi = np.cumsum(np.r_[rng.uniform(-np.pi, np.pi), rng.normal(0, 0.004, n - 1)])
    v = np.clip(v0 + np.cumsum(rng.normal(0, 0.05, n)), 0.5, 40.0)
    acc = np.gradient(v, 0.1)
    x = np.cumsum(v * np.cos(psi) * 0.1); y = np.cumsum(v * np.sin(psi) * 0.1)
    ts = t0_ms + 100 * np.arange(n)
    return ts, x, y, psi, v, acc


def build(out, variant="complete", seed=131, scale=1):
    rng = np.random.default_rng(seed)
    out = Path(out); rt, st = out / F.RT_DIR, out / F.ST_DIR
    plan = [("Crash", 6 * scale), ("NearCrash", 10 * scale), ("Crash-NearCrash", 2 * scale), (F.BASELINE_CATEGORY, 20 * scale)]
    eid, tid = 9_000_000, 0
    meta, ego_by_cat, sur_by_cat, emeta, env, edata = [], {}, {}, [], [], {}
    trips = []                                  # ground-truth per trip
    for cat, n_trips in plan:
        for j in range(n_trips):
            eid += 7
            is_base = cat == F.BASELINE_CATEGORY
            dur_s = rng.uniform(18, 25) if is_base else rng.uniform(20, 35)
            n = int(round(dur_s * 10))
            if cat == "NearCrash" and j == 9:
                n = 105                             # planted short trip (10.5 s) -> below 100 evaluable frames
            t0_ms = int(rng.integers(0, 50)) * 100
            ts, x, y, psi, v, acc = ego_track(rng, n, t0_ms, rng.uniform(8, 25))
            sev1 = 3 if cat.startswith("Crash") else (2 if cat == "NearCrash" else 0)
            sev2 = 2 if cat == "Crash-NearCrash" else 0
            if cat == "Crash-NearCrash" and j == 1:
                sev1, sev2 = 2, 0                   # planted: category says crash, coded severities say near-crash (C25)
            te_ms = None
            if not is_base:
                pre = rng.uniform(4.0, 6.5) if j in (0, 1) else rng.uniform(8.0, (n / 10) - 2.5)  # 2 short histories
                if n == 105:
                    pre = 9.0
                te_ms = int(t0_ms + round(pre * 1000))          # ms, not on the 0.1-s grid in general
            # planted implausible values
            if is_base and j == 3:
                bad = rng.choice(n, size=int(0.15 * n), replace=False); v[bad] = 70.0      # 15 % -> excluded
            if cat == "Crash" and j == 2:
                k_in = int(np.argmin(np.abs(ts - (te_ms - 500)))); acc[k_in] = 20.0      # within 1 s: allowed
                k_out = int(np.argmin(np.abs(ts - (te_ms - 3000)))); acc[k_out] = -20.0  # outside: masked
            if cat == "Crash-NearCrash" and j == 1:
                acc[int(np.argmin(np.abs(ts - (te_ms - 400))))] = 18.0   # within 1 s of a crash BY CATEGORY: allowed
            ev = np.zeros(n, int)
            if te_ms is not None:
                ev[(ts >= te_ms - 3000) & (ts <= te_ms + 1000)] = 1
            reconstructed = not (cat == "NearCrash" and j == 8)                # planted: listed, not reconstructed
            meta.append(dict(event_id=eid, event_category=cat, first="leading" if not is_base else "none",
                             second="none", ego_width=1.8, ego_length=4.6, target_width=1.75, target_length=4.5,
                             other_width=np.nan, other_length=np.nan, ego_reconstructed=int(reconstructed),
                             surrounding_reconstructed=int(reconstructed), time_series_das="x.csv",
                             time_series_honda=np.nan, file_dir="RawData/", file2use="x.csv", note=np.nan))
            if not reconstructed:
                continue
            raw = ts + rng.integers(-40, 41, n)                                     # raw ms stamps (jittered)
            raw[rng.random(n) < 0.1] = -1                                          # -1 = interpolated by the authors
            ego = pd.DataFrame(dict(event_id=eid, timestamp=raw.astype(int), time=np.round(ts / 1000, 1),
                                    x_ekf=x, y_ekf=y, psi_ekf=psi, v_ekf=v, omega_ekf=np.gradient(psi, 0.1),
                                    acc_ekf=acc, event=ev))
            if variant != "public":
                for c in F.EGO_COLS_RAW_ONLY:
                    ego[c] = 0.0
            if is_base and j == 1:                                                # planted duplicate grid time
                dup = ego.iloc[[50]].copy(); dup["timestamp"] = dup["timestamp"] + 20
                ego = pd.concat([ego.iloc[:51], dup, ego.iloc[51:]], ignore_index=True)
            if variant == "driver_id":
                ego["driver_id"] = 77
            ego_by_cat.setdefault(cat, []).append(ego)
            # surrounding objects ahead of the subject
            n_obj = 0 if (is_base and j == 5) else int(rng.integers(1, 4))
            objs = []
            for o in range(n_obj):
                a = int(rng.integers(0, max(1, n - 80))); b = int(min(n, a + rng.integers(60, 200)))
                k = np.arange(a, b); gap = rng.uniform(10, 30)
                fx = x[k] + np.cos(psi[k]) * (2.3 + gap); fy = y[k] + np.sin(psi[k]) * (2.3 + gap)
                vs = np.clip(v[k] + rng.normal(0, 0.3, len(k)), 0, 45)
                if cat == "NearCrash" and j == 4 and o == 0:
                    fx = fx.copy(); fx[len(k) // 2:] += 15.0                    # planted jump > 10 m
                if is_base and j == 7 and o == 0:
                    vs = vs.copy(); vs[:5] = 65.0                               # planted object speed > 60
                df = pd.DataFrame(dict(event_id=eid, target_id=tid, timestamp=np.where(rng.random(len(k)) < 0.1, -1, ts[k]).astype(int),
                                       time=np.round(ts[k] / 1000, 1), x_ekf=fx, y_ekf=fy, v_ekf=vs,
                                       psi_ekf=psi[k]))
                if variant != "public":
                    for c in F.SUR_COLS_RAW_ONLY:
                        df[c] = 0.0
                if is_base and j == 1 and o == 0:                                  # planted duplicate object frame
                    df = pd.concat([df.iloc[:11], df.iloc[[10]], df.iloc[11:]], ignore_index=True)
                objs.append(df); tid += 1
            sur = pd.concat(objs, ignore_index=True) if objs else None
            if sur is not None:
                sur_by_cat.setdefault(cat, []).append(sur)
            # safety-critical test set (events only)
            dur_enough = bool(sur is not None and sur.groupby("target_id").size().max() >= 50)
            if cat == "NearCrash" and j == 6:
                dur_enough = False                                              # planted filter failure
            conflict = "none" if (cat == "NearCrash" and j == 7) else ("leading" if not is_base else None)
            if not is_base:
                emeta.append(dict(event_id=eid, event_category=cat, first="leading", second="none",
                                  ego_width=1.8, ego_length=4.6, target_width=1.75, target_length=4.5,
                                  other_width=np.nan, other_length=np.nan, start_timestamp=te_ms - 3000,
                                  end_timestamp=te_ms + 1000, impact_timestamp=te_ms,
                                  reaction_timestamp=te_ms - 1500, severity_first=sev1, severity_second=sev2,
                                  conflict=conflict, duration_enough=dur_enough,
                                  **({"narrative": "synthetic"} if variant != "public" else {})))
                lit = ["Daylight", "Darkness, lighted", "Dusk", "Unknown"][int(rng.integers(0, 4))]
                wea = ["No Adverse Conditions", "Raining", "Unknown"][int(rng.integers(0, 3))]
                sur_c = ["Dry", "Wet", "Snowy", "Unknown"][int(rng.integers(0, 4))]
                tra = ["A1", "A2", "B", "C", "D", "E", "F", "Unknown"][int(rng.integers(0, 8))]
                env.append(dict(event_id=eid, lighting=lit, weather=wea, surfaceCondition=sur_c, trafficDensity=tra))
                if sur is not None:
                    m = ego.merge(sur, on=["event_id", "time"], suffixes=("_e", "_s"))
                    ed = pd.DataFrame(dict(target_id=m.target_id, time=m.time, event_id=m.event_id,
                                           x_ego=m.x_ekf_e, y_ego=m.y_ekf_e, v_ego=m.v_ekf_e, psi_ego=m.psi_ekf_e,
                                           acc_ego=m.acc_ekf, hx_ego=np.cos(m.psi_ekf_e), hy_ego=np.sin(m.psi_ekf_e),
                                           x_sur=m.x_ekf_s + REF_OFFSET * np.cos(m.psi_ekf_s),
                                           y_sur=m.y_ekf_s + REF_OFFSET * np.sin(m.psi_ekf_s),
                                           v_sur=m.v_ekf_s, psi_sur=m.psi_ekf_s, hx_sur=np.cos(m.psi_ekf_s),
                                           hy_sur=np.sin(m.psi_ekf_s))).set_index(["target_id", "time"])
                    edata.setdefault(cat, []).append(ed)
            parts_ = cat.split("-")                 # independent form of C25: a component 'Crash', or SecondaryCrash
            trips.append(dict(event_id=eid, category=cat, ts=ts, v=v, acc=acc, te_ms=te_ms, sev=max(sev1, sev2),
                              crash=("Crash" in parts_ or cat == "SecondaryCrash"),
                              included=(is_base or (dur_enough and conflict != "none")), sur=sur,
                              env=env[-1] if not is_base else None))
    # ---- write files ----
    rt.mkdir(parents=True, exist_ok=True)
    M = pd.DataFrame(meta)
    if variant == "missing_col":
        M = M.drop(columns=["ego_length"])
    M.to_csv(rt / "metadata_birdseye.csv", index=False)
    pd.DataFrame(M.event_category.value_counts()).to_csv(rt / "event_counts.csv")
    pd.DataFrame([dict(ego_uncertainty_init=1.0, sur_uncertainty_init=15.0)]).to_csv(rt / "ekf_parameters.csv", index=False)
    (rt / "readme_data_structure.md").write_text("synthetic\n")
    for cat, frames in ego_by_cat.items():
        d = rt / cat; d.mkdir(exist_ok=True)
        E = pd.concat(frames, ignore_index=True)
        S = pd.concat(sur_by_cat.get(cat, []), ignore_index=True) if sur_by_cat.get(cat) else \
            pd.DataFrame(columns=F.SUR_COLS_PUBLIC)
        if cat == F.BASELINE_CATEGORY:                                  # 5 chunks, as documented
            ids = E.event_id.unique(); parts = np.array_split(ids, 5)
            # planted metadata flags as the authors' reconstruct_birdseye.py writes them for chunked baselines:
            # ego-only trips (1, 0) in every chunk; trips with objects (1, 1) only in the LAST chunk, else (0, 0)
            has_sur = set(S.event_id.unique())
            for c, p in enumerate(parts):
                for e_ in p:
                    f_ = (1, 0) if e_ not in has_sur else ((1, 1) if c == len(parts) - 1 else (0, 0))
                    M.loc[M.event_id == e_, ["ego_reconstructed", "surrounding_reconstructed"]] = f_
            M.to_csv(rt / "metadata_birdseye.csv", index=False)
            chunk_truth = [dict(in_ego=len(p), in_sur=len(set(p) & has_sur),
                                flag_both=len(set(p) & has_sur) if c == len(parts) - 1 else 0,
                                ego_only=len(set(p) - has_sur)) for c, p in enumerate(parts)]
            for c, p in enumerate(parts):
                E[E.event_id.isin(p)].to_hdf(d / f"{F.EGO_FILE}_{c}.h5", key=F.H5_KEY)
                S[S.event_id.isin(p)].to_hdf(d / f"{F.SUR_FILE}_{c}.h5", key=F.H5_KEY)
        else:
            E.to_hdf(d / f"{F.EGO_FILE}.h5", key=F.H5_KEY); S.to_hdf(d / f"{F.SUR_FILE}.h5", key=F.H5_KEY)
    EM, EN = pd.DataFrame(emeta), pd.DataFrame(env)
    for cat in EM.event_category.unique():
        d = st / cat; d.mkdir(parents=True, exist_ok=True)
        EM[EM.event_category == cat].to_csv(d / "event_meta.csv", index=False)
        if variant != "public":
            EN[EN.event_id.isin(EM[EM.event_category == cat].event_id)].to_csv(d / "environment.csv", index=False)
        if edata.get(cat):
            pd.concat(edata[cat]).to_hdf(d / "event_data.h5", key=F.H5_KEY)
    T_ = truth(trips, M, variant); T_["baseline_chunks"] = chunk_truth
    (out / "truth.json").write_text(json.dumps(T_, indent=1, default=int))
    return out


# --------------------------- independent ground truth (plain loops) ---------------------------
def truth(trips, M, variant):
    T = dict(listed_trips=len(M), reconstructed_trips=len(trips),
             listed_by_category={k: int(v) for k, v in M.event_category.value_counts().items()})
    inc = [t for t in trips if t["category"] != F.BASELINE_CATEGORY and t["included"]]
    T["included_events"] = len(inc)
    T["included_crash"] = sum(t["crash"] for t in inc); T["included_near_crash"] = sum(not t["crash"] for t in inc)
    T["baseline_trips"] = sum(t["category"] == F.BASELINE_CATEGORY for t in trips)
    # horizon retention
    T["retention"] = {}
    for h in F.HORIZONS_S:
        ok = [(t["te_ms"] - t["ts"][0]) >= (h + 2) * 1000 for t in inc]
        T["retention"][str(h)] = dict(share=float(np.mean(ok)), kept=bool(np.mean(ok) >= 0.8))
    # implausible ego frames and excluded trips
    excl = []; n_speed = n_acc = 0
    for t in trips:
        bad = 0
        for i in range(len(t["ts"])):
            sp = t["v"][i] > 60.0
            in_win = t["crash"] and t["te_ms"] is not None and abs(t["ts"][i] - t["te_ms"]) <= 1000
            ac = abs(t["acc"][i]) > 15.0 and not in_win
            n_speed += sp; n_acc += ac; bad += (sp or ac)
        if bad / len(t["ts"]) > 0.10:
            excl.append(t["event_id"])
    T["ego_frames_speed_gt60"] = n_speed; T["ego_frames_acc_gt15_outside_window"] = n_acc
    T["excluded_trips"] = excl
    n_jump = n_osp = 0
    for t in trips:
        if t["sur"] is None:
            continue
        for _, g in t["sur"].groupby("target_id"):
            g = g.sort_values("time"); xs, ys, vs = g.x_ekf.to_numpy(), g.y_ekf.to_numpy(), g.v_ekf.to_numpy()
            for i in range(len(g)):
                n_osp += vs[i] > 60.0
                if i > 0 and np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) > 10.0:
                    n_jump += 1
    T["object_frames_speed_gt60"] = n_osp; T["object_jumps_gt10"] = n_jump
    # evaluable frames at 3 s, per unit (included trips that are not excluded)
    units = {}
    for t in trips:
        if not t["included"] or t["event_id"] in excl:
            continue
        ts = t["ts"]; n_eval = n_pos = n_neg = 0
        for i in range(F.HISTORY_FRAMES - 1, len(ts)):
            tm = ts[i]
            in_win = t["crash"] and t["te_ms"] is not None and abs(tm - t["te_ms"]) <= 1000
            if t["v"][i] > 60.0 or (abs(t["acc"][i]) > 15.0 and not in_win):
                continue                                   # masked subject frame: no prediction at t
            if t["te_ms"] is not None:
                tte = t["te_ms"] - tm
                if tte <= 0 or (5000 < tte <= 7000):
                    continue
                if tte <= 3000:
                    n_eval += 1; n_pos += 1; continue
            if ts[-1] - tm >= 3000:
                n_eval += 1; n_neg += 1
        units[str(t["event_id"])] = dict(eval=n_eval, pos=n_pos, neg=n_neg, sev=t["sev"])
    T["units_total"] = len(units)
    T["units_eligible"] = sum(u["eval"] >= F.MIN_EVAL_UNIT for u in units.values())
    T["eval_frames_total"] = sum(u["eval"] for u in units.values())
    T["pos_frames_total"] = sum(u["pos"] for u in units.values())
    T["neg_frames_total"] = sum(u["neg"] for u in units.values())
    T["units"] = units
    # environment
    if variant != "public":
        lv = {}
        for t in inc:
            for f in ["lighting", "weather", "surfaceCondition", "trafficDensity"]:
                val = t["env"][f]
                if f == "trafficDensity":
                    b = None if val == "Unknown" else (0 if val[0] in "AB" else 1)
                else:
                    b = F.FACTOR_MAPS[f].get(val)
                lv.setdefault(f, {"0": 0, "1": 0, "unknown": 0})["unknown" if b is None else str(b)] += 1
        T["factor_levels_included_events"] = lv
    T["baselines_with_environment"] = 0
    T["ref_offset"] = REF_OFFSET
    return T


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/synthetic_shrp2")
    ap.add_argument("--variant", default="complete", choices=["complete", "public", "driver_id", "missing_col"])
    ap.add_argument("--scale", type=int, default=1, help="multiply trip counts (timing check)")
    a = ap.parse_args(); print("written:", build(a.out, a.variant, scale=a.scale))
