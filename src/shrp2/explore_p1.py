"""
EXPLORATORY ANALYSES E1 and E2 under protocol P1 (post-hoc; plan docs/exploratory_plan_E1_E5.md, written 30 Oct 2026
before this script; committed with it before its single run). Run in GSSM's environment. Registered run only.

E1a  matched recall: for each model seed, the threshold on the registered grid whose P1 recall is closest to GSSM
     h-C's recall at h-C's best-F1 threshold (ties: the lower threshold); recall, precision and mTTI (GSSM's 99%
     sign-test interval) at that threshold. GSSM h-C reported at its best-F1 threshold.
E1b  5 s horizon: the same five models scored with their cumulative risk at 5 s (explore_e1_scores.py, files *_h5s);
     P1 engine unchanged, same grid; AUPRC, recorded events and mTTI at the best-F1 threshold.
E2   shared events: per model seed, the test events with a recorded danger period for BOTH that seed and the
     comparator (GSSM h-C; TTC); AUPRC of each on that set (P1 engine unchanged); seed-mean difference with a 95%
     percentile interval from a cluster bootstrap over each seed's shared events (B 2,000, seed 131; differences
     averaged over seeds within each resample). Descriptive: no p-value is used for any decision.
Nothing here changes a registered rule, grid, protocol or confirmatory verdict. The functions used are those of the
confirmatory P1 analysis (p1_eval.py, confirm_p1.py), unchanged.

Usage   <GPY> src/shrp2/explore_p1.py --gssm baselines/GSSM
Output  results/shrp2_explore_E1E2.csv, results/shrp2_explore_E1E2_summary.txt   (aggregate only)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import p1_eval as P
import confirm_p1 as CF
import analysis.stats as S

SEEDS = CF.SEEDS
B_DEFAULT = 2000


# ------------------------------------------------------------------------------------------ helpers
def rates(M):
    """Recall and precision at every threshold of a count matrix (original sample), as get_statistics computes."""
    tp, fn, fp = M["tp"].sum(1), M["fn"].sum(1), M["fp"].sum(1)
    return tp / np.maximum(P.EPS, tp + fn), tp / np.maximum(P.EPS, tp + fp)


def best_f1_index(M):
    """Index of the best-F1 threshold, exactly as p1_eval.mtti chooses it."""
    tp, fn, fp = M["tp"].sum(1), M["fn"].sum(1), M["fp"].sum(1)
    return int(np.argmax(tp / np.maximum(P.EPS, tp + 0.5 * (fp + fn))))


def matched_index(M, target_recall):
    """Threshold index whose recall is closest to target_recall; ties -> the lower threshold (thresholds ascending)."""
    rec, _ = rates(M)
    d = np.abs(rec - target_recall)
    return int(np.flatnonzero(d == d.min())[0])


def mtti_at(UE, UM, th, seval, event_meta, indicator):
    """p1_eval.mtti at a GIVEN threshold (same TTI rule and 99% sign-test interval)."""
    w = UE.issue_warning(indicator, th, seval, event_meta)
    if "warning_timestamp" not in w:
        w = w.assign(warning_timestamp=np.nan)
    tti = (w["impact_timestamp"] - w["warning_timestamp"]) / 1000.0
    tti = tti[np.isfinite(tti) & (tti < 10)]
    lo, hi = UM._median_ci_bounds(tti, alpha=0.01) if len(tti) else (np.nan, np.nan)
    return dict(mTTI=float(np.median(tti)) if len(tti) else np.nan, ci_low=float(lo), ci_high=float(hi), n=int(len(tti)))


def subset(M, mask):
    """Count matrix restricted to the events (columns) where mask is True."""
    mask = np.asarray(mask, bool)
    out = {k: M[k][:, mask] for k in ("tp", "fn", "fp", "tn")}
    out["recorded"], out["safety"], out["thresholds"] = M["recorded"][mask], M["safety"][mask], M["thresholds"]
    return out


def shared_diff(UM, model_mats, comp_mat, events, B=B_DEFAULT, seed=131):
    """E2: per seed, events recorded by both; AUPRC difference on that set; bootstrap per seed's set, averaged."""
    events = np.asarray(events)
    per = []
    for M in model_mats:
        sh = M["recorded"] & comp_mat["recorded"]
        a, c = subset(M, sh), subset(comp_mat, sh)
        boot = S.ClusterBootstrap(events[sh].astype(str), B=B, seed=seed)
        n = int(sh.sum())
        diffs = np.full(B, np.nan)
        for b in range(B):
            w = np.bincount(boot.indices(b), minlength=n)
            x, y = P.auprc(a, UM, w), P.auprc(c, UM, w)
            if np.isfinite(x) and np.isfinite(y):
                diffs[b] = x - y
        per.append(dict(n_shared=n, auprc_model=P.auprc(a, UM), auprc_comp=P.auprc(c, UM), diffs=diffs))
    D = np.array([p["diffs"] for p in per])                     # seeds x B
    ok = np.isfinite(D).all(axis=0)
    mean_b = D[:, ok].mean(axis=0)
    est = float(np.mean([p["auprc_model"] - p["auprc_comp"] for p in per]))
    lo, hi = (np.quantile(mean_b, [0.025, 0.975]) if ok.any() else (np.nan, np.nan))
    return dict(estimate=est, ci_low=float(lo), ci_high=float(hi), B_used=int(ok.sum()),
                n_shared=[p["n_shared"] for p in per], auprc_model=[p["auprc_model"] for p in per],
                auprc_comp=[p["auprc_comp"] for p in per])


# ------------------------------------------------------------------------------------------ driver
def run(repo, B=B_DEFAULT, ws_test="data/gssm_ws_test", out_dir="results", n_jobs=1):
    if os.environ.get("SHRP2_RUN", ""):
        raise SystemExit("E1-E5 run on the registered run only: remove SHRP2_RUN")
    t0 = time.time()
    UE, UM = P.gssm(repo)
    head = CF.head_choice()
    meta_t, data_t = P.load_events(ws_test, UE)
    ev_t = np.sort(meta_t.index.unique())
    G = P.GRIDS
    mats, sev = {}, {}

    def evaluate(name, seval, indicator, grid):
        rec = P.run_records(UE, seval, data_t, meta_t, indicator, grid, n_jobs)
        mats[name] = P.count_matrices(rec, ev_t); sev[name] = (seval, indicator)

    for s in SEEDS:
        for tag, suf in (("3s", ""), ("5s", "_h5s")):
            f = CF.SH / "preds" / f"p1_central_{head}_aug_s{s}{suf}.parquet"
            if not f.exists():
                raise SystemExit(f"missing {f}" + (" - run explore_e1_scores.py first" if suf else ""))
            fs = pd.read_parquet(f); fs = fs[fs["split"] == "test"]
            evaluate(f"model{tag}_s{s}", P.frame_scores_to_rows(data_t, fs, "GSSM"), "GSSM", G["RISK"])
    fh = Path(ws_test) / "ResultData" / "EventEvaluation" / "highD_current.h5"
    if not fh.exists():
        raise SystemExit(f"missing {fh}")
    g = pd.read_hdf(Path(ws_test) / "ResultData" / "EventEvaluation" / "highD_current.h5", "data")[["event_id", "target_id", "time", "intensity"]]
    evaluate("GSSM_hC", g, "GSSM", G["GSSM"])
    ssm = CF.read_ssm()
    col, ind, grid = CF.CLASSICAL["TTC"]
    evaluate("TTC", P.frame_scores_to_rows(data_t, CF.classical_scores(ssm, ev_t, col), ind), ind, G[grid])

    rows, lines = [], ["SHRP2 EXPLORATORY ANALYSES E1-E2 (protocol P1; post-hoc; registered run; aggregate only)",
                       f"risk head {head}; test events {len(ev_t)}; B = {B}", ""]
    # GSSM h-C reference at its best-F1 threshold
    Mg = mats["GSSM_hC"]; ig = best_f1_index(Mg); rg, pg = rates(Mg)
    tg = float(Mg["thresholds"][ig]); mg = mtti_at(UE, UM, tg, sev["GSSM_hC"][0], meta_t, "GSSM")
    reg = P.mtti(UE, UM, Mg, sev["GSSM_hC"][0], meta_t, "GSSM")
    if not (np.isclose(reg["threshold"], tg) and np.isclose(reg["mTTI"], mg["mTTI"], equal_nan=True)):
        raise SystemExit("check failed: mtti_at at the best-F1 threshold differs from p1_eval.mtti")
    rows.append(dict(analysis="reference", method="GSSM_hC", threshold=tg, recall=rg[ig], precision=pg[ig],
                     auprc=P.auprc(Mg, UM), events_recorded=int(Mg["recorded"].sum()), **mg))
    # E1a matched recall (3 s scores) and E1b 5 s scores at best F1
    for s in SEEDS:
        for tag in ("3s", "5s"):
            M = mats[f"model{tag}_s{s}"]; rec, prec = rates(M); seval, indic = sev[f"model{tag}_s{s}"]
            ib = best_f1_index(M); tb = float(M["thresholds"][ib])
            mb = mtti_at(UE, UM, tb, seval, meta_t, indic)
            if tag == "3s":
                regm = P.mtti(UE, UM, M, seval, meta_t, indic)
                if not (np.isclose(regm["threshold"], tb) and np.isclose(regm["mTTI"], mb["mTTI"], equal_nan=True)):
                    raise SystemExit(f"check failed: seed {s} best-F1 mTTI differs from the registered function")
            rows.append(dict(analysis=f"E1 best-F1 ({tag} score)", method=f"model_s{s}", threshold=tb, recall=rec[ib],
                             precision=prec[ib], auprc=P.auprc(M, UM), events_recorded=int(M["recorded"].sum()), **mb))
            if tag == "3s":
                im = matched_index(M, rg[ig]); tm = float(M["thresholds"][im])
                mm = mtti_at(UE, UM, tm, seval, meta_t, indic)
                rows.append(dict(analysis="E1a matched recall (3 s score)", method=f"model_s{s}", threshold=tm,
                                 recall=rec[im], precision=prec[im], auprc=np.nan,
                                 events_recorded=int(M["recorded"].sum()), **mm))
    R = pd.DataFrame(rows)
    gi = R[R.method == "GSSM_hC"].iloc[0]
    for an in ("E1 best-F1 (3s score)", "E1a matched recall (3 s score)", "E1 best-F1 (5s score)"):
        q = R[R.analysis == an]
        verdict = "comparable" if gi.ci_low <= q.mTTI.mean() <= gi.ci_high else "not comparable"
        rows.append(dict(analysis=f"{an} - seed mean", method="model (5 seeds)", threshold=np.nan, recall=q.recall.mean(),
                         precision=q.precision.mean(), auprc=q.auprc.mean(), events_recorded=q.events_recorded.mean(),
                         mTTI=q.mTTI.mean(), ci_low=np.nan, ci_high=np.nan, n=np.nan,
                         note=f"{verdict} with GSSM h-C (mTTI {gi.mTTI:.2f} s, 99% interval [{gi.ci_low:.2f}, {gi.ci_high:.2f}])"))
    # E2 shared events
    for comp in ("GSSM_hC", "TTC"):
        r = shared_diff(UM, [mats[f"model3s_s{s}"] for s in SEEDS], mats[comp], ev_t, B=B)
        rows.append(dict(analysis=f"E2 shared events vs {comp}", method="model (5 seeds)", estimate=r["estimate"],
                         ci_low=r["ci_low"], ci_high=r["ci_high"], B_used=r["B_used"],
                         note=f"shared events per seed {r['n_shared']}; AUPRC model {np.round(r['auprc_model'], 4).tolist()}, "
                              f"{comp} {np.round(r['auprc_comp'], 4).tolist()}"))
    R = pd.DataFrame(rows)
    Path(out_dir).mkdir(exist_ok=True)
    R.to_csv(Path(out_dir) / "shrp2_explore_E1E2.csv", index=False)
    lines.append(R.round(4).to_string(index=False))
    Path(out_dir, "shrp2_explore_E1E2_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\nSaved {out_dir}/shrp2_explore_E1E2.csv | {time.time() - t0:.0f}s")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gssm", required=True); ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--ws-test", default="data/gssm_ws_test"); ap.add_argument("--n-jobs", type=int, default=1)
    a = ap.parse_args()
    run(a.gssm, a.B, a.ws_test, n_jobs=a.n_jobs)
