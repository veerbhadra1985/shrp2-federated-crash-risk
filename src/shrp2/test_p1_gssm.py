"""
Tests of the H1 / protocol-P1 code (Amendment 4): GSSM workspace, GSSM event features, P1 engine, bootstrap.
Hand-built events with answers worked out by hand + an end-to-end run of the authors' own code on a synthetic
SHRP2-format workspace. No SHRP2 data. Needs the GSSM repository (default baselines/GSSM; or --gssm PATH).
Usage : python src/shrp2/test_p1_gssm.py [--gssm baselines/GSSM]      (~1-3 min, temporary folder only)
"""
import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "loaders"))
import p1_eval as P
RES = []


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def hand_events():
    """Three events, 0-30 s at 0.1 s; start 20 s, impact 22 s, end 23 s -> danger 17.5-22.5 s (GSSM rule)."""
    t = np.round(np.arange(0, 30.0001, 0.1), 1)
    meta = pd.DataFrame(dict(event_id=[1, 2, 3], start_timestamp=20000, impact_timestamp=22000, end_timestamp=23000,
                             reaction_timestamp=21000, duration_enough=True, conflict="leading", ego_width=1.8,
                             ego_length=4.6, target_width=1.8, target_length=4.5, other_width=np.nan,
                             other_length=np.nan)).set_index("event_id")
    rows = []
    for e in (1, 2, 3):
        for tid, lo, hi in ((e * 100 + 1, 10.0, 25.0), (e * 100 + 2, 0.0, 8.0)):
            for tt in t[(t >= lo - 1e-9) & (t <= hi + 1e-9)]:
                rows.append(dict(event_id=e, target_id=tid, time=float(tt), acc_ego=0.0))
    data = pd.DataFrame(rows).set_index(["target_id", "time"])
    fs = []
    for e in (1, 2, 3):
        s = np.full(len(t), 0.1)
        if e == 1:
            s[(t >= 17.5 - 1e-9) & (t <= 22.5 + 1e-9)] = 0.9
        if e == 2:
            s[(t >= 3.0 - 1e-9) & (t <= 4.0 + 1e-9)] = 0.6
        if e == 3:
            s[(t >= 17.5 - 1e-9) & (t <= 22.5 + 1e-9)] = 0.4
        fs.append(pd.DataFrame(dict(event_id=e, frame=np.round(t * 10).astype(int), score=s)))
    meta["danger_start"] = np.minimum(meta["impact_timestamp"] - 4500, meta["start_timestamp"])
    meta["danger_end"] = np.minimum(meta["impact_timestamp"] + 500, meta["end_timestamp"])
    return meta, data, pd.concat(fs, ignore_index=True)


def expected(thr):
    """Hand answer per threshold (strict '>'): TP/FN for events 1 and 3. Event 2 has no recorded danger (same risk
    before and in danger), and GSSM's parallel_records then skips the event entirely - its safe object (risk 0.6
    at 3-4 s) is NOT counted. FP/TN: one safe object each in events 1 and 3 (risk 0.1)."""
    tp = (thr < 0.9).astype(int) + (thr < 0.4).astype(int); fn = 2 - tp
    fp = 2 * (thr < 0.1).astype(int); tn = 2 - fp
    return tp, fn, fp, tn


def own_auprc(recall, precision):
    o = np.argsort(recall); r, p = recall[o], precision[o]
    vr = np.linspace(1e-6, 1 - 1e-6, 1000)
    y = np.interp(vr, r, p)
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(vr)))


def main(repo):
    repo = os.path.abspath(repo)          # the test changes folder later; a relative path would break
    UE, UM = P.gssm(repo)
    # ---------------- G1 hand events through the authors' parallel_records ----------------
    meta, data, fs = hand_events()
    seval = P.frame_scores_to_rows(data, fs, "GSSM")
    thr = P.GRIDS["RISK"]
    rec = P.run_records(UE, seval, data, meta, "GSSM", thr)
    M = P.count_matrices(rec, np.array([1, 2, 3]))
    tp, fn, fp, tn = expected(M["thresholds"])
    check("G1 danger periods 17.5-22.5 s (earlier of start and impact - 4.5 s; impact + 0.5 s)",
          (meta["danger_start"] == 17500).all() and (meta["danger_end"] == 22500).all())
    check("G1 TP/FN/FP/TN per threshold = hand answer (GSSM first stage)",
          np.array_equal(M["tp"].sum(1), tp) and np.array_equal(M["fn"].sum(1), fn)
          and np.array_equal(M["fp"].sum(1), fp) and np.array_equal(M["tn"].sum(1), tn))
    check("G1 event 2: no recorded danger, so GSSM skips it (no safe-period counts either)",
          list(M["recorded"]) == [True, False, True] and list(M["safety"]) == [True, False, True])
    a_eng = P.auprc(M, UM)
    rcl = tp / np.maximum(1e-6, tp + fn); prc = tp / np.maximum(1e-6, tp + fp)
    a_hand = own_auprc(rcl, prc)
    a_gssm = UM.get_eval_metrics(rec.assign(indicator="x", model="x"), {"roc": [0.8], "prc": [0.8], "tti": None})["auprc"]
    check("G2 AUPRC: engine = GSSM get_eval_metrics = independent trapezoid", np.isclose(a_eng, a_gssm, atol=1e-12)
          and np.isclose(a_eng, a_hand, atol=1e-9), f"{a_eng:.6f} {a_gssm:.6f} {a_hand:.6f}")
    check("G2 AUPRC with unit weights equals the original sample", np.isclose(P.auprc(M, UM, np.ones(3)), a_eng))
    w = np.array([2.0, 0.0, 1.0])                                         # event 1 twice, event 3 once
    tpw = 2 * (M["thresholds"] < 0.9) + (M["thresholds"] < 0.4); fnw = 3 - tpw; fpw = 2 * (M["thresholds"] < 0.1) + (M["thresholds"] < 0.1)
    check("G2 event multiplicities (bootstrap weights) enter the counts", np.isclose(P.auprc(M, UM, w),
          own_auprc(tpw / np.maximum(1e-6, tpw + fnw), tpw / np.maximum(1e-6, tpw + fpw)), atol=1e-9))
    mt = P.mtti(UE, UM, M, seval, meta, "GSSM")
    check("G3 mTTI* at the max-F1 threshold = 4.5 s (switch at 17.5 s, impact 22 s)", np.isclose(mt["mTTI"], 4.5)
          and mt["n"] >= 1, str(mt))
    # missing scores never alarm
    fs_m = fs.copy(); fs_m.loc[(fs_m.event_id == 2), "score"] = np.nan
    M2 = P.count_matrices(P.run_records(UE, P.frame_scores_to_rows(data, fs_m, "GSSM"), data, meta, "GSSM", thr), np.array([1, 2, 3]))
    check("G4 missing scores never exceed a threshold (event 2 cannot be recorded)", not M2["recorded"][1]
          and np.array_equal(M2["tp"][:, 1], np.zeros(len(thr))))
    # smaller-is-riskier path (TTC2D column) mirrors the larger-is-riskier one
    fs_t = fs.assign(score=1.0 / fs["score"])                            # 0.9 -> 1.11 s etc.: smaller = riskier
    M3 = P.count_matrices(P.run_records(UE, P.frame_scores_to_rows(data, fs_t, "TTC2D"), data, meta, "TTC2D",
                                        np.array([1.0, 1.2, 2.0, 3.0, 11.0])), np.array([1, 2, 3]))
    check("G5 TTC-type measures: '<' threshold path (GSSM TTC2D rule)", list(M3["tp"].sum(1)) == [0, 1, 1, 2, 2]
          and list(M3["fp"].sum(1)) == [0, 0, 0, 0, 2], f"{M3['tp'].sum(1)} {M3['fp'].sum(1)}")
    # third stage with a shared (voted) object
    voted = pd.DataFrame({"target_id": [101, 201, 301]}, index=pd.Index([1, 2, 3], name="event_id"))
    M4 = P.count_matrices(P.run_records(UE, seval, data, meta, "GSSM", thr, voted=voted), np.array([1, 2, 3]))
    tp4 = (M4["thresholds"] < 0.9).astype(int) + (M4["thresholds"] < 0.4).astype(int) + (M4["thresholds"] < 0.1).astype(int)
    check("G6 third stage (voted object): all three events recorded; event 2 TP only below its 0.1 risk",
          np.array_equal(M4["tp"].sum(1), tp4) and M4["recorded"].all())
    # load_events applies GSSM's danger formula: start earlier than impact - 4.5 s -> danger starts at the start
    tmpw = Path(tempfile.mkdtemp(prefix="p1ws_"))
    try:
        d = tmpw / "ResultData/EventData/Cat"; d.mkdir(parents=True)
        mm = meta.drop(columns=["danger_start", "danger_end"]).copy(); mm.loc[1, "start_timestamp"] = 15000; mm.loc[2, "end_timestamp"] = 22200
        mm.reset_index().to_csv(d / "event_meta.csv", index=False); data.to_hdf(d / "event_data.h5", key="data")
        lm, _ = P.load_events(tmpw, UE)
        check("G1 load_events: danger start = earlier of start and impact - 4.5 s; end = earlier of end and impact + 0.5 s",
              lm.loc[1, "danger_start"] == 15000 and lm.loc[3, "danger_start"] == 17500 and lm.loc[2, "danger_end"] == 22200
              and lm.loc[3, "danger_end"] == 22500)
    finally:
        shutil.rmtree(tmpw, ignore_errors=True)
    # ---------------- G7 bootstrap ----------------
    worse = fs.copy(); worse.loc[worse.event_id == 1, "score"] = 0.1; worse.loc[(worse.event_id == 1) & (worse.frame >= 175) & (worse.frame <= 225), "score"] = 0.05   # event 1 missed
    Mw = P.count_matrices(P.run_records(UE, P.frame_scores_to_rows(data, worse, "GSSM"), data, meta, "GSSM", thr), np.array([1, 2, 3]))
    r = P.h1_test(UM, [M, M], Mw, np.array([1, 2, 3]), B=200, seed=131)
    r2 = P.h1_test(UM, [M, M], Mw, np.array([1, 2, 3]), B=200, seed=131)
    check("G7 bootstrap estimate = AUPRC difference; deterministic; small sets flagged not testable",
          np.isclose(r["estimate"], P.auprc(M, UM) - P.auprc(Mw, UM)) and abs(r["estimate"]) > 1e-6 and r == r2
          and r["ci_low"] < r["ci_high"]
          and not r["testable"], f"{r['estimate']:.4f} p {r['p']:.3f}")
    # ---------------- G8 end to end on a synthetic workspace with the authors' scripts ----------------
    tmp = Path(tempfile.mkdtemp(prefix="p1gssm_"))
    try:
        import make_synthetic as MK, shrp2 as LD, make_split as MS, gssm_workspace as GW, gssm_features as GF
        root = MK.build(tmp / "syn", "complete", scale=5)
        work = tmp / "data/processed/shrp2"; LD.main(root, work, mask=True)
        MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        ws = tmp / "data/gssm_ws_test"
        man = GW.build(root, work / "split.csv", "test", ws)
        S = pd.read_csv(work / "split.csv")
        em = pd.concat([pd.read_csv(p) for p in (ws / "ResultData/EventData").glob("*/event_meta.csv")])
        base = pd.concat([pd.read_hdf(p, "data") for p in (ws / "ProcessedData/SHRP2/SafeBaseline").glob("Ego_*.h5")])
        check("G8 workspace holds exactly the test events and only training-split baselines",
              set(em.event_id) == set(S[(S.split == "test") & ~S.stratum.str.startswith("baseline")].event_id) & set(em.event_id)
              and set(em.event_id) <= set(S[S.split == "test"].event_id)
              and set(base.event_id) <= set(S[(S.split == "train") & S.stratum.str.startswith("baseline")].event_id), str(man["events"]))
        cwd = os.getcwd(); os.chdir(ws)
        try:
            made = GF.main(repo)
            ok_feat = all((ws / "ResultData/EventData" / c / "event_features.npz").exists() for c, n in made.items() if n)
            env = dict(os.environ); env["PYTHONPATH"] = os.path.abspath(repo)
            p = subprocess.run([sys.executable, os.path.join(os.path.abspath(repo), "src_safety_evaluation", "evaluate_safety.py")],
                               capture_output=True, text=True, env=env)
            ok_ssm = (ws / "ResultData/EventEvaluation/TAdv_TTC2D_ACT_EI.h5").exists()
        finally:
            os.chdir(cwd)
        check("G8 authors' feature lines + evaluate_safety (2D SSMs) run unchanged on the workspace", ok_feat and ok_ssm,
              str(made) if ok_ssm else str(made) + " | " + p.stderr[-600:])
        if not ok_ssm:
            raise SystemExit("evaluate_safety did not run - see the message above")
        meta8, data8 = P.load_events(ws, UE)
        ev8 = pd.read_hdf(ws / "ResultData/EventEvaluation/TAdv_TTC2D_ACT_EI.h5", "data")
        rec8 = P.run_records(UE, ev8, data8, meta8, "TTC2D", P.GRIDS["TTC2D"][:20])
        M8 = P.count_matrices(rec8, np.sort(meta8.index.unique()))
        check("G8 GSSM's own TTC2D scores flow through the P1 engine (counts consistent)",
              np.all(M8["tp"] + M8["fn"] == M8["recorded"][None, :].astype(float)), f"{int(M8['recorded'].sum())} recorded")
        # a GSSM model in the authors' layout (UNTRAINED weights: plumbing only) -> evaluate_safety -> P1 engine
        import torch
        sys.path.insert(0, os.path.abspath(repo))
        from src_posterior_inference.inference_utils.utils_train_eval_test import train_val_test
        mdir = tmp / "models"; cwd = os.getcwd(); os.chdir(tmp)
        try:
            torch.manual_seed(0)
            pl = train_val_test("cpu", str(mdir) + "/", ["highD"], ["current"])
            torch.save(torch.optim.swa_utils.AveragedModel(pl._model).state_dict(), mdir / "PosteriorInference/highD/current/untrained.pth")
        finally:
            os.chdir(cwd)
        GW.add_models(ws, mdir / "PosteriorInference", ["highD/current"])
        os.chdir(ws)
        try:
            p = subprocess.run([sys.executable, os.path.join(os.path.abspath(repo), "src_safety_evaluation", "evaluate_safety.py")],
                               capture_output=True, text=True, env=env)
        finally:
            os.chdir(cwd)
        out = ws / "ResultData/EventEvaluation/highD_current.h5"
        ok = out.exists()
        if ok:
            g = pd.read_hdf(out, "data")
            rec9 = P.run_records(UE, g, data8, meta8, "GSSM", P.GRIDS["GSSM"][::10])
            M9 = P.count_matrices(rec9, np.sort(meta8.index.unique()))
            ok = {"event_id", "target_id", "time", "intensity"} <= set(g.columns) and np.all(M9["tp"] + M9["fn"] == M9["recorded"][None, :])
        check("G9 a model placed in the authors' layout is scored by evaluate_safety and read by the P1 engine", ok,
              p.stderr[-300:] if not ok else "")
        # a frame-level method (project TTC2D, subject frames) mapped onto the workspace's object rows
        import ssm_shrp2 as SSn
        R = SSn.main(work, "asgiven", "car")
        fsc = pd.DataFrame(dict(event_id=R.scene_id.str.split("/").str[1].astype(np.int64),
                                frame=np.round(R.time_s * 10).astype(np.int64), score=R.ttc2d_min))
        rows = P.frame_scores_to_rows(data8, fsc, "TTC2D")
        check("G10 frame scores reach every object row with a subject frame (coverage)", rows["TTC2D"].notna().mean() > 0.95
              and len(rows) == len(data8), f"{rows['TTC2D'].notna().mean():.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = sum(ok for _, ok in RES); print(f"\n{n}/{len(RES)} PASS"); sys.exit(0 if n == len(RES) else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gssm", default="baselines/GSSM")
    main(ap.parse_args().gssm)
