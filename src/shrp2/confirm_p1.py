"""
SHRP2 CONFIRMATORY ANALYSIS, PART P1: H1a and H1b under the GSSM protocol (pre-registration Sections 1, 5, 6;
Amendment 4). Run in GSSM's environment (it imports the authors' evaluation functions through p1_eval.py).
Written and tested on synthetic data before any SHRP2 access.

REGISTERED TESTS (part of the nine-test Holm family; the rest is in confirm_p2.py)
  H1a  P1 AUPRC: proposed model (5 seeds, mean) - GSSM h-C > 0
  H1b  P1 AUPRC: proposed model - best classical measure > 0; the best of TTC, DRAC, TTC2D and TTC2D moving is the one
       with the highest P1 AUPRC on the VALIDATION events (Amendment 4 Section 2), chosen before the test events are
       evaluated
  Protocol P1 = the authors' first-stage evaluation (Amendment 4 Sections 2-4), run by p1_eval.py on the test
  workspace; frame-level scores (our model, LightGBM, classical measures) are copied to every object row at that
  frame. Bootstrap over test EVENTS (B 2,000, seed 131); not testable if fewer than 50 test events have a recorded
  danger period for either method (then p = NaN, reported as not testable, never replaced).
ALSO HERE (descriptive / sensitivity; not in the Holm family)
  P1 AUPRC and mTTI (99% sign-test interval) of every method: model seeds, LightGBM seeds, h-C, S-C, each classical
  measure; sensitivity 2 (registration Section 7): H1a and H1b on GSSM's published voted subset (third stage; only if
  the voted file is given) and on the method-independent subset (C21); H1 key secondary vs GSSM under P2 (C22).

Inputs  data/gssm_ws_test, data/gssm_ws_val (gssm_workspace.py; the validation workspace only needs its events),
        ResultData/EventEvaluation/highD_current.h5 (+ SafeBaseline_current.h5) in the test workspace (evaluate_safety),
        data/processed/shrp2/preds/p1_central_<head>_aug_s<seed>.parquet, p1_lgbm_s<seed>.parquet (train_shrp2.py p1),
        data/processed/shrp2/ssm_shrp2.parquet (classical measures per subject frame),
        data/processed/shrp2/preds/haz_central_<head>_aug_s<seed>.npz + cache test meta (P2 secondary vs GSSM)
Usage   <GPY> src/shrp2/confirm_p1.py --gssm baselines/GSSM [--B 2000] [--voted path/to/Voted_conflicting_targets.csv]
Output  results/shrp2_confirm_p1_tests.csv, results/shrp2_confirm_p1_methods.csv, results/shrp2_confirm_p1_summary.txt
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import p1_eval as P
import analysis.stats as S

SEEDS = [131, 132, 133, 134, 135]
RUN = os.environ.get("SHRP2_RUN", "")                 # as in train_shrp2.py: "" registered run, "nomask" sensitivity 8 (C23)
if RUN not in ("", "nomask"):
    raise SystemExit(f"SHRP2_RUN must be empty or 'nomask', not {RUN!r}")
SFX = f"_{RUN}" if RUN else ""
SH, OUT = Path(f"data/processed/shrp2{SFX}"), Path("results")
CLASSICAL = {"TTC": ("ttc_min", "TTC2D", "TTC2D"), "DRAC": ("drac_max", "EI", "EI_DRAC"),
             "TTC2D": ("ttc2d_min", "TTC2D", "TTC2D"), "TTC2D_moving": ("ttc2d_mov", "TTC2D", "TTC2D")}
HALF_LANE = 1.75                       # C21: in-lane = |lateral offset| <= half of a 3.5 m lane
MIN_EVENTS = 50


def head_choice():
    r = json.loads(Path(f"results/shrp2_head_selection{SFX}.json").read_text())
    if r.get("smoke"):
        raise SystemExit("the head selection file is from a smoke run")
    return r["choice"]


def classical_scores(ssm, event_ids, col):
    """Frame scores (event_id, frame, score) of one classical measure on the given events; frame = round(time * 10)."""
    s = ssm[ssm["event_id"].isin(event_ids)]
    return pd.DataFrame({"event_id": s["event_id"].to_numpy(), "frame": s["frame"].to_numpy(), "score": s[col].to_numpy(float)})


def read_ssm():
    s = pd.read_parquet(SH / "ssm_shrp2.parquet")
    s = s[s["agent_id"].astype(str) == "ego"] if "agent_id" in s else s
    out = pd.DataFrame({"event_id": s["scene_id"].astype(str).str.split("/").str[1].astype(np.int64),
                        "frame": np.round(s["time_s"].to_numpy(float) * 10).astype(np.int64)})
    for c in ("ttc_min", "drac_max", "ttc2d_min", "ttc2d_mov"):
        out[c] = s[c].to_numpy(float)
    return out.drop_duplicates(["event_id", "frame"])


def method_independent(meta, data):
    """C21: included leading-vehicle conflicts; conflicting object = the nearest object AHEAD of the subject vehicle
    within half a lane width of its heading line at the impact frame (impact_timestamp / 1000 s, nearest 0.1-s frame).
    Returns DataFrame(event_id -> target_id) of the events where such an object is recorded."""
    m = meta[(meta["conflict"] == "leading") & meta["duration_enough"].astype(bool)]
    d = data.reset_index()
    d = d[d["event_id"].isin(m.index)]
    d["frame"] = np.round(d["time"].to_numpy(float) * 10).astype(np.int64)
    out = {}
    for e, g in d.groupby("event_id"):
        fi = int(round(m.loc[e, "impact_timestamp"] / 100.0))
        g = g[g["frame"] == fi]
        if not len(g):
            continue
        dx, dy = g["x_sur"] - g["x_ego"], g["y_sur"] - g["y_ego"]
        lon = dx * g["hx_ego"] + dy * g["hy_ego"]; lat = -dx * g["hy_ego"] + dy * g["hx_ego"]
        ok = (lon > 0) & (lat.abs() <= HALF_LANE)
        if ok.any():
            out[e] = g.loc[lon[ok].idxmin(), "target_id"]
    return pd.DataFrame({"target_id": pd.Series(out, dtype=object)}).rename_axis("event_id")


def covered(vt, tables, meta):
    """C27: boolean array over vt (event_id -> target_id): True if EVERY table (event_id, target_id, time, ...) has at
    least one row of that event's shared object inside its danger period [danger_start, danger_end] (ms in meta)."""
    q = vt.reset_index()[["event_id", "target_id"]].copy()
    q["event_id"] = q["event_id"].astype(np.int64); q["target_id"] = q["target_id"].astype(np.int64)
    q = q.merge(meta[["danger_start", "danger_end"]].reset_index().rename(columns={meta.index.name or "index": "event_id"}),
                on="event_id", how="left")
    ok = np.ones(len(q), bool)
    for t in tables:
        r = t[["event_id", "target_id", "time"]].copy()
        r["event_id"] = r["event_id"].astype(np.int64); r["target_id"] = r["target_id"].astype(np.int64)
        m = q.merge(r, on=["event_id", "target_id"], how="left")
        inside = (m["time"] * 1000 >= m["danger_start"]) & (m["time"] * 1000 <= m["danger_end"])
        hit = inside.groupby([m["event_id"], m["target_id"]]).any()
        ok &= hit.reindex(pd.MultiIndex.from_arrays([q["event_id"], q["target_id"]])).fillna(False).to_numpy(bool)
    return ok


def p2_vs_gssm(UE, data, gssm_scores, head, B):
    """C22 (H1 key secondary vs GSSM under P2): on test samples of EVENT trips only (GSSM scores exist only for events),
    GSSM's frame score = its highest intensity over the objects at that frame (no object -> lowest score); AUPRC at
    3 s, model (seed mean) - GSSM; bootstrap over event trips."""
    import calibration as C
    sys.path.insert(0, str(HERE)); import train_shrp2 as T
    mt = pd.read_parquet(T.CACHE / "shrp2_test_meta.parquet")
    ev_ids = set(data["event_id"].unique())
    keep = mt["event_id"].isin(ev_ids).to_numpy()
    y, kn = C.labels_at(mt["event_bin"].to_numpy(), mt["obs_bins"].to_numpy(), 2)
    g = gssm_scores.copy(); g["frame"] = np.round(g["time"].to_numpy(float) * 10).astype(np.int64)
    gm = g.groupby(["event_id", "frame"])["intensity"].max()
    key = pd.MultiIndex.from_arrays([mt["event_id"].to_numpy(), mt["frame"].to_numpy()])
    gs = gm.reindex(key).to_numpy(float)
    Tix = np.flatnonzero(keep & kn)
    if len(Tix) == 0 or y[Tix].sum() == 0:
        return dict(test="H1_P2 model-GSSM (event trips)", estimate=np.nan, p=np.nan, note="NOT TESTABLE: no event-trip positives")
    models = []
    for s in SEEDS:
        z = T.npz(T.PRED / f"haz_central_{head}_aug_s{s}.npz"); sel = np.asarray(z["split"]).astype(str) == "test"
        if not np.array_equal(np.asarray(z["frame"])[sel], mt["frame"].to_numpy()):
            raise SystemExit("P2 predictions not aligned with the cache")
        models.append(z["risk"][sel][:, 2][Tix])
    boot = S.ClusterBootstrap(mt["event_id"].to_numpy()[Tix], B=B, seed=131)
    g_ = gs[Tix]; fin = np.isfinite(g_)
    gsT = np.where(fin, g_, (g_[fin].min() - 1.0) if fin.any() else 0.0)       # no object -> below every GSSM score
    r = S.paired_test(boot, S.seed_mean_difference(S.average_precision, y[Tix], models, [gsT] * len(SEEDS)))
    return dict(test="H1_P2 model-GSSM (event trips)", family="secondary", delta0=0.0,
                note=f"{len(Tix)} event-trip test frames; GSSM frame score = max over objects", **r)


def run(repo, B=2000, voted_file=None, ws_test="data/gssm_ws_test", ws_val="data/gssm_ws_val", n_jobs=1):
    sfx = SFX
    t0 = time.time()
    UE, UM = P.gssm(repo)
    head = head_choice()
    meta_t, data_t = P.load_events(ws_test, UE); meta_v, data_v = P.load_events(ws_val, UE)
    ev_t, ev_v = np.sort(meta_t.index.unique()), np.sort(meta_v.index.unique())
    ssm = read_ssm()
    G = P.GRIDS
    methods, mats, sevals = [], {}, {}

    def evaluate(name, seval, indicator, grid, meta, data, events, voted=None, keep=True):
        rec = P.run_records(UE, seval, data, meta, indicator, grid, n_jobs, voted)
        M = P.count_matrices(rec, events)
        if keep:
            mats[name] = M; sevals[name] = (seval, indicator)
        return M

    # ---- best classical measure on VALIDATION events (before any test evaluation of classical measures)
    val_ap = {}
    for c, (col, ind, grid) in CLASSICAL.items():
        sv = P.frame_scores_to_rows(data_v, classical_scores(ssm, ev_v, col), ind)
        val_ap[c] = P.auprc(evaluate(f"val_{c}", sv, ind, G[grid], meta_v, data_v, ev_v, keep=False), UM)
    finite = {c: v for c, v in val_ap.items() if np.isfinite(v)}
    best = max(finite, key=finite.get) if finite else None
    print("validation P1 AUPRC: " + ", ".join(f"{c} {v:.4f}" for c, v in val_ap.items()) + f" -> best = {best}", flush=True)
    # ---- test events: model seeds, LightGBM seeds, h-C, S-C, classical measures
    for s in SEEDS:
        fs = pd.read_parquet(SH / "preds" / f"p1_central_{head}_aug_s{s}.parquet"); fs = fs[fs["split"] == "test"]
        evaluate(f"model_s{s}", P.frame_scores_to_rows(data_t, fs, "GSSM"), "GSSM", G["RISK"], meta_t, data_t, ev_t)
        fl = SH / "preds" / f"p1_lgbm_s{s}.parquet"
        if fl.exists():
            fz = pd.read_parquet(fl); fz = fz[fz["split"] == "test"]
            evaluate(f"lgbm_s{s}", P.frame_scores_to_rows(data_t, fz, "GSSM"), "GSSM", G["RISK"], meta_t, data_t, ev_t)
    ee = Path(ws_test) / "ResultData" / "EventEvaluation"
    for nm, f in (("GSSM_hC", "highD_current.h5"), ("GSSM_SC", "SafeBaseline_current.h5")):
        if (ee / f).exists():
            g = pd.read_hdf(ee / f, "data")[["event_id", "target_id", "time", "intensity"]]
            evaluate(nm, g, "GSSM", G["GSSM"], meta_t, data_t, ev_t)
    for c, (col, ind, grid) in CLASSICAL.items():
        evaluate(c, P.frame_scores_to_rows(data_t, classical_scores(ssm, ev_t, col), ind), ind, G[grid], meta_t, data_t, ev_t)
    model_mats = [mats[f"model_s{s}"] for s in SEEDS]
    rows = []

    def h1(name, comp, family="registered", events=ev_t, mm=None, cm=None, note=""):
        mm = mm or model_mats; cm = cm if cm is not None else mats.get(comp)
        if cm is None:
            rows.append(dict(test=name, family=family, estimate=np.nan, p=np.nan, note=f"NOT TESTABLE: {comp} scores missing")); return
        r = P.h1_test(UM, mm, cm, events, B=B, seed=131)
        nt = not r["testable"]
        rows.append(dict(test=name, family=family, delta0=0.0, comparator=comp,
                         **{k: (np.nan if (nt and k == "p") else v) for k, v in r.items()},
                         note=("NOT TESTABLE: fewer than 50 test events with a recorded danger period; " if nt else "") + note))
        print(f"  {name:28} est {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p {rows[-1]['p']:.4f} "
              f"| recorded events model {r['n_events_recorded_model']}, comparator {r['n_events_recorded_comparator']}", flush=True)

    print("\nRegistered (P1 AUPRC, first stage, all included test events):")
    h1("H1a", "GSSM_hC", note="comparator GSSM h-C (highD/current)")
    if best is None:
        rows.append(dict(test="H1b", family="registered", estimate=np.nan, p=np.nan, note="NOT TESTABLE: no classical measure with a defined validation AUPRC"))
    else:
        h1("H1b", best, note=f"best classical on validation = {best}")
    # ---- sensitivity 2: subsets (third stage, one shared conflicting object)
    subsets = {}
    if voted_file:
        v = pd.read_csv(voted_file)
        v = v.rename(columns={c: "event_id" for c in v.columns if c.lower() in ("event_id", "eventid")})
        v = v[v["event_id"].isin(ev_t)].drop_duplicates("event_id").set_index("event_id")[["target_id"]]
        present = data_t.reset_index().groupby("event_id")["target_id"].apply(set)
        v = v[[t in present.get(e, set()) for e, t in zip(v.index, v["target_id"])]]
        subsets["voted"] = v
    subsets["method-independent"] = method_independent(meta_t, data_t)
    print("\nSensitivity 2 (third stage, shared conflicting object):")
    # C27 (26 Oct 2026, technical fix after a crash in this sensitivity analysis; registered tests unaffected): the authors'
    # third-stage function needs the shared object to have rows in the danger period of EVERY compared method. GSSM h-C
    # does not score every object, so an event is kept only if all compared methods (model seeds, h-C, best classical)
    # have >= 1 row of the shared object in its danger period; the number dropped is reported. Same events, same object.
    comp_tables = [sevals[f"model_s{s}"][0] for s in SEEDS] + [sevals[c][0] for c in ("GSSM_hC", best) if c is not None and c in sevals]
    dropped = {}
    for sname in list(subsets):
        vt = subsets[sname]
        if len(vt):
            keep = covered(vt, comp_tables, meta_t)
            dropped[sname] = int((~keep).sum()); subsets[sname] = vt[keep]
    for sname, vt in subsets.items():
        if len(vt) == 0:
            for t in ("H1a", "H1b"):
                rows.append(dict(test=f"{t} [2 {sname}]", family="sensitivity", estimate=np.nan, p=np.nan, note="NOT TESTABLE: empty subset"))
            continue
        evs = np.sort(vt.index.to_numpy())
        mm = [evaluate(f"{sname}_model_s{s}", sevals[f"model_s{s}"][0], "GSSM", G["RISK"], meta_t, data_t, evs, vt, keep=False) for s in SEEDS]
        for t, comp in (("H1a", "GSSM_hC"), ("H1b", best)):
            if comp is None or comp not in sevals:
                rows.append(dict(test=f"{t} [2 {sname}]", family="sensitivity", estimate=np.nan, p=np.nan, note="NOT TESTABLE: comparator missing")); continue
            sv, ind = sevals[comp]
            cm = evaluate(f"{sname}_{comp}", sv, ind, G["GSSM" if comp.startswith("GSSM") else CLASSICAL[comp][2]], meta_t, data_t, evs, vt, keep=False)
            h1(f"{t} [2 {sname}]", comp, "sensitivity", evs, mm, cm,
               note=f"{len(evs)} events ({dropped.get(sname, 0)} dropped: shared object not scored in the danger period by every method, C27)")
    # ---- descriptive: every method's P1 AUPRC and mTTI
    desc = []
    for nm, M in mats.items():
        sv, ind = sevals[nm]
        mt_ = P.mtti(UE, UM, M, sv, meta_t, ind)
        desc.append(dict(method=nm, auprc=P.auprc(M, UM), events_recorded=int(M["recorded"].sum()), **{f"mtti_{k}": v for k, v in mt_.items()}))
    for c, v in val_ap.items():
        desc.append(dict(method=f"validation {c}", auprc=v))
    # ---- H1 key secondary vs GSSM under P2 (event trips only)
    if "GSSM_hC" in sevals:
        rows.append(p2_vs_gssm(UE, data_t, sevals["GSSM_hC"][0], head, B))
    OUT.mkdir(exist_ok=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / f"shrp2_confirm_p1_tests{sfx}.csv", index=False)
    Dd = pd.DataFrame(desc); Dd.to_csv(OUT / f"shrp2_confirm_p1_methods{sfx}.csv", index=False)
    cols = [c for c in ["test", "family", "comparator", "estimate", "ci_low", "ci_high", "p", "n_events_recorded_model",
                        "n_events_recorded_comparator", "testable", "note"] if c in R]
    lines = ["SHRP2 CONFIRMATORY ANALYSIS, P1 PART (GSSM protocol) - aggregate results only (shareable)",
             f"risk head {head}; B = {B}; clusters = test events; best classical (validation) = {best}", "",
             R[cols].round(5).to_string(index=False), "", "Per method (P1 AUPRC, mTTI with 99% sign-test interval):",
             Dd.round(4).to_string(index=False)]
    (OUT / f"shrp2_confirm_p1_summary{sfx}.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved results/shrp2_confirm_p1_*.csv | {time.time() - t0:.0f}s")
    return R, Dd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gssm", required=True); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--voted", default=None, help="the authors' published voted file (only after the licence allows)")
    ap.add_argument("--ws-test", default="data/gssm_ws_test"); ap.add_argument("--ws-val", default="data/gssm_ws_val")
    ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    run(a.gssm, a.B, a.voted, a.ws_test, a.ws_val, a.n_jobs)
