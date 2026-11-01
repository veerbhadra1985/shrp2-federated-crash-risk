"""
Tests of the exploratory analyses E1 and E2 under protocol P1 (src/shrp2/explore_p1.py) on the synthetic SHRP2-format
workspace of test_confirm_p1.py (gssm_workspace.py), fabricated frame scores, and the authors' evaluation functions.
Expected values are recomputed independently (threshold search by plain loops, the P1 engine re-run on the shared
events only, a separate bootstrap loop).
Run with GSSM's environment:  <GPY> src/shrp2/test_explore_p1.py --gssm baselines/GSSM      (~3-6 min)
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
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
    import p1_eval as P, confirm_p1 as CF, explore_p1 as X
    import analysis.stats as S
    tmp = Path(tempfile.mkdtemp(prefix="explp1_")); cwd = os.getcwd()
    try:
        os.chdir(tmp); rng = np.random.default_rng(11)
        root = MK.build(tmp / "syn", "complete", scale=5)
        work = tmp / "data/processed/shrp2"; LD.main(root, work, mask=True)
        MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        GW.build(root, work / "split.csv", "test", tmp / "data/gssm_ws_test")
        UE, UM = P.gssm(repo)
        P.GRIDS = {k: v[::5] for k, v in P.GRIDS.items()}          # coarser grids: same logic, faster test
        meta_t, data_t = P.load_events(tmp / "data/gssm_ws_test", UE)
        ev_t = np.sort(meta_t.index.unique())
        (tmp / "results").mkdir(exist_ok=True); (work / "preds").mkdir(exist_ok=True)
        (tmp / "results/shrp2_head_selection.json").write_text(json.dumps(dict(choice="hazard", smoke=False)))
        sf_t = subject_frames(data_t)

        def danger_flag(sf, meta, lead=4500):
            m = meta.reindex(sf.event_id)
            ds = np.minimum(m.impact_timestamp - lead, m.start_timestamp).to_numpy() / 100
            de = np.minimum(m.impact_timestamp + 500, m.end_timestamp).to_numpy() / 100
            return (sf.frame.to_numpy() >= ds) & (sf.frame.to_numpy() <= de)
        dt = danger_flag(sf_t, meta_t); dearly = danger_flag(sf_t, meta_t, 6000)
        for s in SEEDS:
            noise = rng.normal(0, 0.12, len(sf_t))
            p = pd.DataFrame(dict(event_id=sf_t.event_id, frame=sf_t.frame, split="test", score=np.clip(0.15 + 0.7 * dt + noise, 0, 1)))
            p.to_parquet(work / "preds" / f"p1_central_hazard_aug_s{s}.parquet", index=False)
            p.assign(score=np.clip(0.2 + 0.6 * dearly + noise, 0, 1)).to_parquet(work / "preds" / f"p1_central_hazard_aug_s{s}_h5s.parquet", index=False)
        ssm = pd.DataFrame(dict(scene_id="SHRP2/" + sf_t.event_id.astype(str), time_s=sf_t.frame / 10.0, agent_id="ego",
                                ttc_min=np.where(dt, rng.uniform(0.5, 3, len(sf_t)), rng.uniform(2, 9, len(sf_t))),
                                drac_max=rng.uniform(0, 3, len(sf_t)), ttc2d_min=rng.uniform(0.5, 9, len(sf_t)), ttc2d_mov=np.inf))
        ssm.to_parquet(work / "ssm_shrp2.parquet", index=False)
        ee = tmp / "data/gssm_ws_test/ResultData/EventEvaluation"; ee.mkdir(parents=True)
        g = data_t.reset_index()[["event_id", "target_id", "time"]].copy()
        gd = danger_flag(pd.DataFrame(dict(event_id=g.event_id, frame=np.round(g.time * 10).astype(int))), meta_t)
        g["intensity"] = np.clip(0.5 + 2.0 * gd + rng.normal(0, 0.8, len(g)), 0, 6)
        drop_ev = ev_t[:3]                                               # h-C scores nothing in 3 events: shared set < all
        g = g[~g.event_id.isin(drop_ev)].reset_index(drop=True)
        g.to_hdf(ee / "highD_current.h5", key="data")

        # ---------------- Q0 matched_index ties -> lower threshold ----------------
        M0 = dict(tp=np.array([[4.], [3.], [2.], [1.]]), fn=np.array([[0.], [1.], [2.], [3.]]), fp=np.zeros((4, 1)),
                  thresholds=np.array([0.1, 0.2, 0.3, 0.4]))
        check("Q0 matched recall: closest recall, ties -> the lower threshold (recalls 1, .75, .5, .25; target .625 -> 0.2)",
              X.matched_index(M0, 0.625) == 1 and X.matched_index(M0, 0.5) == 2 and X.matched_index(M0, 2.0) == 0)

        # ---------------- run ----------------
        B = 49
        R = X.run(repo, B=B, ws_test=tmp / "data/gssm_ws_test", out_dir=str(tmp / "results"))
        G = P.GRIDS

        def M_(seval, ind, grid, meta=meta_t, data=data_t, evs=ev_t):
            return P.count_matrices(P.run_records(UE, seval, data, meta, ind, grid), evs)
        sev = {}
        for s in SEEDS:
            for tag, suf in (("3s", ""), ("5s", "_h5s")):
                f = pd.read_parquet(work / "preds" / f"p1_central_hazard_aug_s{s}{suf}.parquet")
                sev[f"{tag}_{s}"] = P.frame_scores_to_rows(data_t, f, "GSSM")
        mats = {k: M_(v, "GSSM", G["RISK"]) for k, v in sev.items()}
        hc = M_(g, "GSSM", G["GSSM"])
        # ---------------- Q1 best-F1 rows = registered mtti; h-C reference ----------------
        ok1 = True
        ref = R[R.method == "GSSM_hC"].iloc[0]; mh = P.mtti(UE, UM, hc, g, meta_t, "GSSM")
        ok1 &= np.isclose(ref.threshold, mh["threshold"]) and np.isclose(ref.mTTI, mh["mTTI"], equal_nan=True) \
            and np.isclose(ref.auprc, P.auprc(hc, UM)) and np.isclose(ref.ci_low, mh["ci99"][0], equal_nan=True)
        for s in SEEDS:
            for tag in ("3s", "5s"):
                r = R[(R.analysis == f"E1 best-F1 ({tag} score)") & (R.method == f"model_s{s}")].iloc[0]
                m = P.mtti(UE, UM, mats[f"{tag}_{s}"], sev[f"{tag}_{s}"], meta_t, "GSSM")
                ok1 &= np.isclose(r.threshold, m["threshold"]) and np.isclose(r.mTTI, m["mTTI"], equal_nan=True) \
                    and np.isclose(r.auprc, P.auprc(mats[f"{tag}_{s}"], UM)) and int(r.events_recorded) == int(mats[f"{tag}_{s}"]["recorded"].sum())
        check("Q1 E1 best-F1 rows (3 s and 5 s scores) and the h-C reference = p1_eval.mtti / auprc on independently built "
              "matrices (5 s rows read the _h5s files)", ok1)
        # ---------------- Q2 matched recall by plain loops ----------------
        tp, fn, fp = hc["tp"].sum(1), hc["fn"].sum(1), hc["fp"].sum(1)
        f1 = [tp[i] / max(1e-6, tp[i] + 0.5 * (fp[i] + fn[i])) for i in range(len(tp))]
        ib = int(np.argmax(f1)); target = tp[ib] / max(1e-6, tp[ib] + fn[ib])
        ok2 = True
        for s in SEEDS:
            M = mats[f"3s_{s}"]; best, bi = None, None
            for i, th in enumerate(M["thresholds"]):
                rc = M["tp"][i].sum() / max(1e-6, M["tp"][i].sum() + M["fn"][i].sum())
                if best is None or abs(rc - target) < best - 1e-15:
                    best, bi = abs(rc - target), i
            th = float(M["thresholds"][bi])
            w = UE.issue_warning("GSSM", th, sev[f"3s_{s}"], meta_t)
            tti = ((w["impact_timestamp"] - w["warning_timestamp"]) / 1000.0) if "warning_timestamp" in w else pd.Series([], dtype=float)
            tti = tti[np.isfinite(tti) & (tti < 10)]
            r = R[(R.analysis == "E1a matched recall (3 s score)") & (R.method == f"model_s{s}")].iloc[0]
            ok2 &= np.isclose(r.threshold, th) and np.isclose(r.mTTI, np.median(tti) if len(tti) else np.nan, equal_nan=True)
        sm = R[R.analysis == "E1a matched recall (3 s score) - seed mean"].iloc[0]
        check("Q2 E1a: threshold of the closest recall to h-C's best-F1 recall (plain loop), mTTI there (issue_warning); "
              "seed-mean row with the reading rule", ok2 and ("comparable" in sm.note), f"target recall {target:.3f}; {sm.note}")
        # ---------------- Q3 E2 shared events = engine re-run on those events only ----------------
        ok3, det = True, []
        for comp, cm, cs, ind, grid in (("GSSM_hC", hc, g, "GSSM", G["GSSM"]),):
            ests, boots = [], []
            for s in SEEDS:
                M = mats[f"3s_{s}"]; sh = M["recorded"] & cm["recorded"]; evs = ev_t[sh]
                mt = meta_t.loc[meta_t.index.isin(evs)]
                dd = data_t.reset_index(); dd = dd[dd.event_id.isin(evs)].set_index(data_t.index.names)
                a = M_(sev[f"3s_{s}"][sev[f"3s_{s}"].event_id.isin(evs)], "GSSM", G["RISK"], mt, dd, evs)
                c = M_(cs[cs.event_id.isin(evs)], ind, grid, mt, dd, evs)
                ests.append(P.auprc(a, UM) - P.auprc(c, UM))
                bt = S.ClusterBootstrap(evs.astype(str), B=B, seed=131)
                boots.append([P.auprc(a, UM, bt.counts[b][np.argsort(np.argsort(evs.astype(str)))]) -
                              P.auprc(c, UM, bt.counts[b][np.argsort(np.argsort(evs.astype(str)))]) for b in range(B)])
                det.append(int(sh.sum()))
            Bm = np.array(boots); okb = np.isfinite(Bm).all(0); mb = Bm[:, okb].mean(0)
            r = R[R.analysis == f"E2 shared events vs {comp}"].iloc[0]
            ok3 &= np.isclose(r.estimate, np.mean(ests)) and np.isclose(r.ci_low, np.quantile(mb, 0.025)) and np.isclose(r.ci_high, np.quantile(mb, 0.975))
            ok3 &= all(n < len(ev_t) for n in det) and str(det) in r.note
        check("Q3 E2: shared-event AUPRC difference = P1 engine re-run on the shared events only; bootstrap interval from a "
              "separate loop (cluster counts, per-seed sets, seed mean per resample); h-C-less events excluded", ok3, f"shared {det}")
        trow = R[R.analysis == "E2 shared events vs TTC"]
        check("Q4 E2 vs TTC reported; outputs aggregate (no event-ID column)", len(trow) == 1 and np.isfinite(trow.estimate.iloc[0])
              and "event_id" not in pd.read_csv(tmp / "results/shrp2_explore_E1E2.csv").columns
              and (tmp / "results/shrp2_explore_E1E2_summary.txt").exists())
        r_ = subprocess.run([sys.executable, str(HERE / "explore_p1.py"), "--gssm", repo], capture_output=True, text=True,
                            env=dict(os.environ, SHRP2_RUN="nomask"))
        check("Q5 refuses SHRP2_RUN (registered run only)", r_.returncode != 0 and "registered run only" in r_.stderr + r_.stdout)
        os.remove(work / "preds" / "p1_central_hazard_aug_s133_h5s.parquet")
        try:
            X.run(repo, B=5, ws_test=tmp / "data/gssm_ws_test", out_dir=str(tmp / "results")); stop = False
        except SystemExit as e:
            stop = "explore_e1_scores" in str(e)
        check("Q6 a missing 5 s score file stops the run with the instruction to run explore_e1_scores.py first", stop)
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gssm", default="baselines/GSSM")
    sys.exit(0 if main(ap.parse_args().gssm) else 1)
