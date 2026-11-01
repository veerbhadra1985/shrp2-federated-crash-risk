"""
Tests of the SHRP2 confirmatory analysis, P1 part (src/shrp2/confirm_p1.py), on synthetic SHRP2-format workspaces built
by gssm_workspace.py, with fabricated frame scores whose properties are designed, and the authors' evaluation functions.
Run with GSSM's environment:  <GPY> src/shrp2/test_confirm_p1.py --gssm baselines/GSSM      (~3-6 min)
"""
import argparse, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "loaders"))
RES = []
SEEDS = [131, 132, 133, 134, 135]


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def subject_frames(data):
    d = data.reset_index()
    return d[["event_id"]].assign(frame=np.round(d["time"].to_numpy(float) * 10).astype(np.int64)).drop_duplicates()


def main(repo):
    repo = os.path.abspath(repo)
    import make_synthetic as MK, shrp2 as LD, make_split as MS, gssm_workspace as GW
    import p1_eval as P, confirm_p1 as CF
    tmp = Path(tempfile.mkdtemp(prefix="confp1_")); cwd = os.getcwd()
    try:
        os.chdir(tmp); rng = np.random.default_rng(11)
        root = MK.build(tmp / "syn", "complete", scale=5)
        work = tmp / "data/processed/shrp2"; LD.main(root, work, mask=True)
        MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        GW.build(root, work / "split.csv", "test", tmp / "data/gssm_ws_test")
        GW.build(root, work / "split.csv", "val", tmp / "data/gssm_ws_val")
        UE, UM = P.gssm(repo)
        P.GRIDS = {k: v[::5] for k, v in P.GRIDS.items()}          # coarser grids: same logic, faster test
        meta_t, data_t = P.load_events(tmp / "data/gssm_ws_test", UE); meta_v, data_v = P.load_events(tmp / "data/gssm_ws_val", UE)
        # ---- plant the method-independent geometry in one test event (before anything reads event_data)
        inc = meta_t[(meta_t.conflict == "leading") & meta_t.duration_enough.astype(bool)]
        e0 = inc.index[0]; fi = int(round(inc.loc[e0, "impact_timestamp"] / 100))
        d = data_t.reset_index(); d["frame"] = np.round(d["time"] * 10).astype(int)
        at = d[(d.event_id == e0) & (d.frame == fi)]
        tids = list(at.target_id.unique())
        check("P0 synthetic test event has >= 2 objects at its impact frame (needed for the geometry test)", len(tids) >= 2, str(len(tids)))
        ix_ahead = at.index[at.target_id == tids[0]][0]; ix_side = at.index[at.target_id == tids[1]][0]
        hx, hy = d.loc[ix_ahead, "hx_ego"], d.loc[ix_ahead, "hy_ego"]; xe, ye = d.loc[ix_ahead, "x_ego"], d.loc[ix_ahead, "y_ego"]
        d.loc[ix_ahead, ["x_sur", "y_sur"]] = [xe + 6 * hx, ye + 6 * hy]                        # 6 m ahead, in lane
        d.loc[ix_side, ["x_sur", "y_sur"]] = [xe + 3 * hx - 3 * hy, ye + 3 * hy + 3 * hx]        # 3 m ahead, 3 m to the side
        for t_ in tids[2:]:
            j = at.index[at.target_id == t_][0]
            d.loc[j, ["x_sur", "y_sur"]] = [xe - 10 * hx, ye - 10 * hy]                         # behind
        behind = d.loc[[ix_ahead]].copy(); behind["target_id"] = int(d.target_id.max()) + 1000   # in lane, 4 m BEHIND: never "ahead"
        behind[["x_sur", "y_sur"]] = [[xe - 4 * hx, ye - 4 * hy]]
        d = pd.concat([d, behind], ignore_index=True)
        cat = [c for c in (tmp / "data/gssm_ws_test/ResultData/EventData").iterdir() if (pd.read_csv(c / "event_meta.csv").event_id == e0).any()][0]
        ed = pd.read_hdf(cat / "event_data.h5", "data").reset_index()
        dd = d.drop(columns="frame"); dd = dd[dd.event_id.isin(ed.event_id.unique())]
        dd.set_index(["target_id", "time"]).to_hdf(cat / "event_data.h5", key="data")
        meta_t, data_t = P.load_events(tmp / "data/gssm_ws_test", UE)
        mi = CF.method_independent(meta_t, data_t)
        check("P1 method-independent object: the nearest object AHEAD within half a lane (6 m ahead), not the nearer one 3 m to "
              "the side, nor the in-lane one 4 m behind",
              mi.loc[e0, "target_id"] == tids[0])
        # ---- fabricated frame scores
        (tmp / "results").mkdir(exist_ok=True); (work / "preds").mkdir(exist_ok=True)
        (tmp / "results/shrp2_head_selection.json").write_text(json.dumps(dict(choice="hazard", smoke=False)))
        sf_t, sf_v = subject_frames(data_t), subject_frames(data_v)

        def danger_flag(sf, meta):
            m = meta.reindex(sf.event_id)
            ds = np.minimum(m.impact_timestamp - 4500, m.start_timestamp).to_numpy() / 100
            de = np.minimum(m.impact_timestamp + 500, m.end_timestamp).to_numpy() / 100
            return (sf.frame.to_numpy() >= ds) & (sf.frame.to_numpy() <= de)
        dt, dv = danger_flag(sf_t, meta_t), danger_flag(sf_v, meta_v)
        for s in SEEDS:
            p = pd.DataFrame(dict(event_id=sf_t.event_id, frame=sf_t.frame, split="test",
                                  score=np.clip(0.15 + 0.7 * dt + rng.normal(0, 0.12, len(sf_t)), 0, 1)))
            p.to_parquet(work / "preds" / f"p1_central_hazard_aug_s{s}.parquet", index=False)
            p.assign(score=np.clip(0.3 + 0.2 * dt + rng.normal(0, 0.2, len(sf_t)), 0, 1)).to_parquet(work / "preds" / f"p1_lgbm_s{s}.parquet", index=False)
        # classical: on VALIDATION, TTC is informative and DRAC is noise; on TEST the reverse (choice must come from validation)
        ssm = []
        for sf, dflag, good in ((sf_v, dv, "ttc"), (sf_t, dt, "drac")):
            n = len(sf)
            ttc = np.where(dflag, rng.uniform(0.5, 2.0, n), rng.uniform(4, 9, n)) if good == "ttc" else rng.uniform(0.5, 9, n)
            drac = np.where(dflag, rng.uniform(3, 6, n), rng.uniform(0, 1, n)) if good == "drac" else rng.uniform(0, 6, n)
            ssm.append(pd.DataFrame(dict(scene_id="SHRP2/" + sf.event_id.astype(str), time_s=sf.frame / 10.0, agent_id="ego",
                                         ttc_min=ttc, drac_max=drac, ttc2d_min=rng.uniform(0.5, 9, n), ttc2d_mov=np.inf)))
        pd.concat(ssm).to_parquet(work / "ssm_shrp2.parquet", index=False)
        ee = tmp / "data/gssm_ws_test/ResultData/EventEvaluation"; ee.mkdir(parents=True)
        g = data_t.reset_index()[["event_id", "target_id", "time"]].copy()
        gf = np.round(g.time * 10).astype(int)
        gd = danger_flag(pd.DataFrame(dict(event_id=g.event_id, frame=gf)), meta_t)
        g["intensity"] = np.clip(0.5 + 2.0 * gd + rng.normal(0, 0.8, len(g)), 0, 6)
        # C27: as on SHRP2, GSSM h-C has no score for one method-independent object inside its event's danger period
        # (planted for e0's shared object); sensitivity 2 must drop that event, not crash
        ds0, de0 = meta_t.loc[e0, "danger_start"] / 1000, meta_t.loc[e0, "danger_end"] / 1000
        g = g[~((g.event_id == e0) & (g.target_id == tids[0]) & (g.time >= ds0) & (g.time <= de0))].reset_index(drop=True)
        gf = np.round(g.time * 10).astype(int)
        g.to_hdf(ee / "highD_current.h5", key="data")
        # P2 cache meta + predictions for the P2-vs-GSSM secondary
        sys.path.insert(0, str(HERE)); import train_shrp2 as T
        T.CACHE.mkdir(parents=True, exist_ok=True)
        cm = sf_t.copy(); cm["event_bin"] = np.where(dt, 0, -1); cm["obs_bins"] = 4; cm["current"] = 0
        cm.to_parquet(T.CACHE / "shrp2_test_meta.parquet", index=False)
        for s in SEEDS:
            r = np.clip(0.1 + 0.6 * dt + rng.normal(0, 0.1, len(cm)), 0.001, 0.999)
            np.savez_compressed(T.PRED / f"haz_central_hazard_aug_s{s}.npz", event_id=cm.event_id.to_numpy(), frame=cm.frame.to_numpy(),
                                split=np.full(len(cm), "test"), risk=np.repeat(r[:, None], 4, 1).astype(np.float32))
        e_v = int(np.sort(meta_t.index.unique())[0]); t_v = data_t.reset_index().query("event_id == @e_v").target_id.iloc[0]
        others = [int(e) for e in np.sort(meta_t.index.unique())[1:3]]
        voted = pd.DataFrame(dict(event_id=[e_v] + others + [99999999], target_id=[t_v, -5, -5, t_v]))
        voted.to_csv(tmp / "voted.csv", index=False)

        # ---------------- run ----------------
        R, Dd = CF.run(repo, B=99, voted_file=tmp / "voted.csv", ws_test=tmp / "data/gssm_ws_test", ws_val=tmp / "data/gssm_ws_val")
        Rt = R.set_index("test")
        # independent recomputation with the engine on the right files, split, indicator and grid
        ev_t = np.sort(meta_t.index.unique()); ev_v = np.sort(meta_v.index.unique())
        def M_(seval, ind, grid, meta, data, evs):
            return P.count_matrices(P.run_records(UE, seval, data, meta, ind, grid), evs)
        mod = [M_(P.frame_scores_to_rows(data_t, pd.read_parquet(work / "preds" / f"p1_central_hazard_aug_s{s}.parquet"), "GSSM"), "GSSM",
                  P.GRIDS["RISK"], meta_t, data_t, ev_t) for s in SEEDS]
        hc = M_(g, "GSSM", P.GRIDS["GSSM"], meta_t, data_t, ev_t)
        exp_a = np.mean([P.auprc(m, UM) for m in mod]) - P.auprc(hc, UM)
        check("P2 H1a = mean over seeds of the model's P1 AUPRC (risk grid) - h-C's (GSSM grid), test events, first stage",
              np.isclose(Rt.loc["H1a", "estimate"], exp_a), f"{exp_a:+.4f}")
        ss = CF.read_ssm()
        def cls(name, meta, data, evs):
            col, ind, grid = CF.CLASSICAL[name]
            return M_(P.frame_scores_to_rows(data, CF.classical_scores(ss, evs, col), ind), ind, P.GRIDS[grid], meta, data, evs)
        val = {c: P.auprc(cls(c, meta_v, data_v, ev_v), UM) for c in CF.CLASSICAL}
        tst = {c: P.auprc(cls(c, meta_t, data_t, ev_t), UM) for c in CF.CLASSICAL}
        fin = {c: v for c, v in val.items() if np.isfinite(v)}
        best_v = max(fin, key=fin.get)
        check("P3 H1b comparator = best classical on VALIDATION (planted: TTC there), not the test's best (planted: DRAC)",
              Rt.loc["H1b", "comparator"] == best_v == "TTC", f"validation {val}, test {tst}")
        exp_b = np.mean([P.auprc(m, UM) for m in mod]) - tst[best_v]
        check("P4 H1b estimate = model - that measure on the test events; TTC under GSSM's '<' rule (smaller = riskier)",
              np.isclose(Rt.loc["H1b", "estimate"], exp_b), f"{exp_b:+.4f}")
        check("P5 fewer than 50 recorded test events -> NOT TESTABLE with p = NaN (never replaced); estimate still reported",
              Rt.loc["H1a", "note"].startswith("NOT TESTABLE") and np.isnan(Rt.loc["H1a", "p"]) and np.isfinite(Rt.loc["H1a", "estimate"]))
        real = P.h1_test
        P.h1_test = lambda *a, **k: dict(real(*a, **k), testable=True)
        try:
            R2, _ = CF.run(repo, B=99, ws_test=tmp / "data/gssm_ws_test", ws_val=tmp / "data/gssm_ws_val")
        finally:
            P.h1_test = real
        r2 = R2.set_index("test")
        check("P6 a testable comparison keeps its bootstrap p-value (registered one-sided formula)",
              np.isfinite(r2.loc["H1a", "p"]) and 0 < r2.loc["H1a", "p"] <= 1 and not str(r2.loc["H1a", "note"]).startswith("NOT"))
        vt = Rt.loc[[i for i in Rt.index if "[2 voted]" in i]]; mt_ = Rt.loc[[i for i in Rt.index if "[2 method-independent]" in i]]
        check("P7 sensitivity 2: voted subset (events outside the test set or with an absent object dropped) and the method-"
              "independent subset, third stage", len(vt) == 2 and len(mt_) == 2 and "1 events" in vt.iloc[0].note
              and f"{len(mi) - 1} events (1 dropped" in mt_.iloc[0].note, f"voted 1 event, method-independent {len(mi) - 1} events")
        check("P7b C27: an event whose shared object GSSM h-C did not score in the danger period is dropped from the subset "
              "(not a crash), and the count is reported", "1 dropped" in mt_.iloc[0].note and "C27" in mt_.iloc[0].note,
              str(mt_.iloc[0].note))
        D = Dd.set_index("method")
        check("P8 descriptive: P1 AUPRC and mTTI for every method (5 model seeds, 5 LightGBM seeds, h-C, 4 classical) + validation AUPRCs",
              all(f"model_s{s}" in D.index and f"lgbm_s{s}" in D.index for s in SEEDS) and "GSSM_hC" in D.index
              and all(c in D.index for c in CF.CLASSICAL) and "mtti_mTTI" in D.columns and "validation TTC" in D.index)
        # P2 secondary vs GSSM (event trips)
        import calibration as C
        y, kn = C.labels_at(cm.event_bin.to_numpy(), cm.obs_bins.to_numpy(), 2)
        gm = g.assign(frame=gf).groupby(["event_id", "frame"])["intensity"].max()
        gs = gm.reindex(pd.MultiIndex.from_arrays([cm.event_id, cm.frame])).to_numpy(float)
        gs = np.where(np.isfinite(gs), gs, np.nanmin(gs) - 1)
        mods = [T.npz(T.PRED / f"haz_central_hazard_aug_s{s}.npz")["risk"][:, 2] for s in SEEDS]
        import analysis.stats as S
        exp_p2 = np.mean([S.average_precision(y[kn], m[kn]) - S.average_precision(y[kn], gs[kn]) for m in mods])
        check("P9 H1 key secondary vs GSSM under P2: GSSM frame score = max intensity over objects; model - GSSM at 3 s",
              np.isclose(Rt.loc["H1_P2 model-GSSM (event trips)", "estimate"], exp_p2), f"{exp_p2:+.4f}")
        import subprocess
        out = subprocess.run([sys.executable, "-c", f"import sys; sys.path[:0]=[r'{HERE}', r'{HERE.parent}']; import confirm_p1 as C; print(C.SH.as_posix(), C.SFX)"],
                             capture_output=True, text=True, env=dict(os.environ, SHRP2_RUN="nomask")).stdout.split()
        check("P11 SHRP2_RUN=nomask (sensitivity 8): the P1 analysis reads the _nomask data and writes _nomask outputs",
              out == ["data/processed/shrp2_nomask", "_nomask"], " ".join(out))
        txt = (tmp / "results/shrp2_confirm_p1_summary.txt").read_text()
        check("P10 outputs written; the summary holds aggregates only (no event IDs)",
              (tmp / "results/shrp2_confirm_p1_tests.csv").exists() and not any(str(e) in txt for e in ev_t))
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gssm", default="baselines/GSSM")
    sys.exit(0 if main(ap.parse_args().gssm) else 1)
