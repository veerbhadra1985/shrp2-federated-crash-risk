"""
REGISTERED DESCRIPTIVE ANALYSES NOT YET REPORTED, plus one post-hoc comparison (written 1 Nov 2026 after an
independent review; committed before its single run on the registered run). Project environment. No training.

Part A  Absolute P2 metrics (Methods Section 4.8 promises AUROC; registration: AUPRC primary, AUROC and adaptive ECE
        reported): on the labelled test frames, per horizon (1, 2, 3, 5 s), AUPRC (analysis.stats.average_precision),
        AUROC (scikit-learn roc_auc_score) and adaptive ECE (analysis.stats.ece_adaptive; probabilistic methods only) for
        the central model (chosen head, augmentation on and off), LightGBM, the classical measure chosen on validation
        (TTC; ranking only, no ECE), and, on the client test frames, the five federated regimes (local, central, FedAvg,
        FedProx, FedPer). Seed mean, minimum and maximum. Descriptive.
Part B  H5 comparators in the deployment design (Amendment 2, Section 5: published comparators reported descriptively;
        run_h5.py leaves them out): calibration units = the eligible VALIDATION trips (each trip = one client), new
        units = the eligible TEST trips, per seed, at 0.05, 0.10 and eps_op. Methods: HP, CRC, C1 (registered, from
        src/phase2/domain_crc.py), C2 Lu et al. (2023), C4 Shahid (2026), C5 Plassier et al. (2023) (src/phase2/
        comparators.py, unchanged), and C3 Humbert et al. (2023) with ONE stated change: with 3,555 clients the exact
        law of the k-th smallest client quantile (a Poisson-binomial recursion over 3,555 clients) is too slow on the
        laptop, so P(count <= k - 1) is computed with the normal approximation with continuity correction; everything
        else as in comparators.c3_humbert_qq (tested against the exact form on small federations).
        Reported: share of test trips above the level, mean and pooled false-alarm rate, worst trip, recall at 3 s.
        Built-in check: HP, CRC and C1 reproduce results/h5_shrp2_secondary.csv (descriptive DEPLOY rows) exactly.
Part C  POST-HOC (reviewer's point that HP and C1 run at different false-alarm rates): C1 at a level alpha* chosen on
        the VALIDATION trips only, so that C1's pooled validation recall matches HP's at 0.05 (grid 0.001-0.050, step
        0.001; ties -> the larger alpha); then applied to the test trips. Per seed. Labelled post-hoc.
Nothing here changes a registered rule or verdict. Outputs are aggregate.

Usage   python src/shrp2/report_p2_extra.py [all|a|b]
Output  results/shrp2_p2_extra_A.csv, results/shrp2_p2_extra_B.csv, results/shrp2_p2_extra_summary.txt
"""
import argparse, math, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import analysis.stats as S
import train_shrp2 as T
import confirm_p2 as CP
from phase2 import run_h5 as RH, domain_crc as D, comparators as K

SEEDS = T.SEEDS
REGIMES = ["local", "central", "fedavg", "fedprox", "fedper"]
OUT = Path("results")


# ================================================================================================ Part A
def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def metrics(y, s, prob=True):
    return dict(auprc=S.average_precision(y, s), auroc=auroc(y, s), ece=S.ece_adaptive(y, s) if prob else np.nan)


def summarise(rows_seed, method, horizon, frames, positives):
    M = pd.DataFrame(rows_seed)
    out = dict(part="A", method=method, horizon_s=horizon, frames=frames, positives=positives, seeds=len(M))
    for c in ("auprc", "auroc", "ece"):
        out[f"{c}_mean"], out[f"{c}_min"], out[f"{c}_max"] = M[c].mean(), M[c].min(), M[c].max()
    return out


def part_a(bank):
    rows = []
    cl = bank.classical()["test"]["TTC"]
    lg = {s: bank.lgbm(s)["test"] for s in SEEDS}
    fed = {r: [bank.fed(r, s) for s in SEEDS] for r in REGIMES}
    for k, h in enumerate(CP.HORIZONS):
        y_t, kn = bank.labels("test", k); ix = np.flatnonzero(kn); y = y_t[ix]
        for aug in ("aug", "noaug"):
            ms = [bank.model(f"central_{bank.head}_{aug}_s{s}") for s in SEEDS]
            if any(m is None for m in ms):
                continue
            rows.append(summarise([metrics(y, m["test"][0][ix, k]) for m in ms], f"central model ({aug})", h, len(ix), int(y.sum())))
        rows.append(summarise([metrics(y, lg[s][ix, k]) for s in SEEDS], "LightGBM", h, len(ix), int(y.sum())))
        rows.append(summarise([metrics(y, cl[ix], prob=False)], "TTC (chosen on validation)", h, len(ix), int(y.sum())))
        for r in REGIMES:
            if any(z is None for z in fed[r]):
                continue
            sub = np.isfinite(fed[r][0]["test"][0][ix, k])
            rows.append(summarise([metrics(y[sub], z["test"][0][ix, k][sub]) for z in fed[r]], f"federated: {r}", h,
                                  int(sub.sum()), int(y[sub].sum())))
    return rows


# ================================================================================================ Part B
_X = K._X


def m_lk_normal(sizes, l):
    """comparators._m_lk_all_k with the Poisson-binomial count replaced by its normal approximation (continuity
    correction): M_lk = integral over x of P(count(x) <= k - 1), count(x) = sum_j Bernoulli(P(Q_j <= x))."""
    sizes = np.asarray(sizes)
    u, cnt = np.unique(sizes, return_counts=True)
    P = np.stack([stats.beta.cdf(_X, l, n + 1 - l) for n in u])                 # (distinct sizes, X)
    mu = (cnt[:, None] * P).sum(0); var = (cnt[:, None] * P * (1 - P)).sum(0)
    m = len(sizes); kk = np.arange(1, m + 1)
    sd = np.sqrt(var)[:, None]
    z = (kk[None, :] - 0.5 - mu[:, None]) / np.where(sd > 0, sd, 1.0)
    surv = np.where(sd > 0, stats.norm.cdf(z), (kk[None, :] - 1 >= mu[:, None]).astype(float))
    return np.trapezoid(surv, _X, axis=0) if hasattr(np, "trapezoid") else np.trapz(surv, _X, axis=0)


_MC = {}


def c3_humbert_qq_normal(client_scores, a, coarse=60):
    """comparators.c3_humbert_qq with m_lk_normal in place of the exact recursion (search rule unchanged)."""
    sizes = tuple(sorted(len(s) for s in client_scores)); nmin = sizes[0]

    def M(l):
        if (sizes, l) not in _MC:
            _MC[(sizes, l)] = m_lk_normal(sizes, l)
        return _MC[(sizes, l)]

    def best_on(ls):
        best = None
        for l in ls:
            Mv = M(l); ok = np.flatnonzero(Mv >= 1 - a)
            if len(ok):
                k = int(ok[0]) + 1
                if best is None or Mv[k - 1] < best[2]:
                    best = (l, k, float(Mv[k - 1]))
        return best
    step = max(1, int(math.ceil(nmin / coarse)))
    b = best_on(sorted(set(list(range(nmin, 0, -step)) + [nmin])))
    if b is None:
        return dict(lam=np.inf, feasible=False)
    b2 = best_on(range(max(1, b[0] - step), min(nmin, b[0] + step) + 1))
    l, k, Mv = b2 if (b2 is not None and b2[2] <= b[2]) else b
    q = np.sort([np.sort(s)[l - 1] for s in client_scores])
    return dict(lam=float(q[k - 1]), feasible=True, l=l, k=k, M=Mv)


def thresholds(cal, lv, ln):
    curves = np.array([c["curve"] for c in cal]); scores = [c["neg"] for c in cal]
    th = {"HP": D.hp_threshold(curves, lv, RH.DELTA)["lam"], "C1": D.pooled_split_conformal(np.concatenate(scores), lv),
          "C2": K.c2_lu_fcp(scores, lv)["lam"], "C3": c3_humbert_qq_normal(scores, lv)["lam"],
          "C4": K.c4_shahid_unseen(curves, [len(s) for s in scores], lv, D.GRID)["lam"],
          "C5": K.c5_plassier_class0(scores, [c["share0"] for c in cal], lv)["lam"]}
    if ln != "op":
        th["CRC"] = D.crc_threshold(curves, lv)["lam"]
    return th


def evaluate(test, lam):
    fa = np.array([D.false_alarm_rate(t["neg"], lam) for t in test]); nn = np.array([len(t["neg"]) for t in test])
    caught = sum(int((t["pos"] > lam).sum()) for t in test); npos = sum(len(t["pos"]) for t in test)
    return fa, nn, caught, npos


def matched_alpha(val, lam_hp, grid=np.round(np.arange(0.001, 0.0505, 0.001), 3)):
    """Part C: alpha on the grid whose C1 threshold gives the pooled VALIDATION recall closest to HP's (ties -> larger)."""
    pos = np.concatenate([v["pos"] for v in val]); negs = np.concatenate([v["neg"] for v in val])
    target = float((pos > lam_hp).mean())
    rec = np.array([(pos > D.pooled_split_conformal(negs, a)).mean() for a in grid])
    d = np.abs(rec - target); best = np.flatnonzero(d == d.min())[-1]
    return float(grid[best]), target, float(rec[best])


def part_b(h5_path=None, min_eval=RH.MIN_EVAL_DEFAULT, check_file=Path("results/h5_shrp2_secondary.csv")):
    h5_path = h5_path or T.SH / "h5_input_shrp2.npz"
    inp = dict(np.load(h5_path, allow_pickle=True))
    seeds = [s for s in SEEDS if f"score_s{s}" in inp]
    fph = float(inp["frames_per_hour"]); levels = [("0.05", 0.05), ("0.10", 0.10), ("op", 1.0 / fph)]
    units, _ = RH.build_units(inp["unit"], inp["split"], inp["known"], inp["y"], {s: inp[f"score_s{s}"] for s in seeds}, min_eval)
    test_ids = sorted(u for u in units if units[u]["split"] == "test"); val_ids = sorted(u for u in units if units[u]["split"] == "val")
    per = []
    for s in seeds:
        def unit(u):
            neg, pos = units[u]["neg"][s], units[u]["pos"][s]
            return dict(neg=neg, pos=pos, curve=D.loss_curve(neg), share0=len(neg) / (len(neg) + len(pos)))
        val, test = [unit(u) for u in val_ids], [unit(u) for u in test_ids]
        for ln, lv in levels:
            th = thresholds(val, lv, ln)
            if ln == "0.05":
                a_star, tgt, got = matched_alpha(val, th["HP"])
                th["C1 recall-matched (post-hoc)"] = D.pooled_split_conformal(np.concatenate([v["neg"] for v in val]), a_star)
            for m, lam in th.items():
                fa, nn, caught, npos = evaluate(test, lam)
                per.append(dict(seed=s, level=ln, eps=lv, method=m, lam=lam, share_over=float((fa > lv).mean()),
                                mean_fa=float(fa.mean()), pooled_fa=float((fa * nn).sum() / nn.sum()), worst_fa=float(fa.max()),
                                recall=caught / max(npos, 1), alpha_star=(a_star if m.startswith("C1 recall") else np.nan)))
        print(f"  seed {s} done", flush=True)
    P = pd.DataFrame(per)
    R = P.groupby(["level", "method"], sort=False).mean(numeric_only=True).reset_index().drop(columns="seed")
    note = f"{check_file} absent - cross-check not run"
    if check_file.exists():
        ref = pd.read_csv(check_file); ref = ref[ref["analysis"] == "descriptive DEPLOY"]
        for _, r in R[R.method.isin(["HP", "CRC", "C1"])].iterrows():
            q = ref[(ref.level.astype(str) == r.level) & (ref.method == r.method)]
            if len(q) and not np.allclose([q.share_over.iloc[0], q.recall.iloc[0], q.mean_fa.iloc[0]],
                                          [r.share_over, r.recall, r.mean_fa], atol=1e-9):
                raise SystemExit(f"check failed: {r.method} at {r.level} differs from {check_file}")
        note = f"HP, CRC and C1 reproduce {check_file}"
    R.insert(0, "part", "B"); R["n_test_trips"] = len(test_ids); R["k_validation_trips"] = len(val_ids); R["note"] = note
    return R


# ================================================================================================ driver
def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("what", nargs="?", default="all", choices=["all", "a", "b"])
    a = ap.parse_args(argv)
    if os.environ.get("SHRP2_RUN", ""):
        raise SystemExit("registered run only: remove SHRP2_RUN (Remove-Item Env:SHRP2_RUN)")
    t0 = time.time(); OUT.mkdir(exist_ok=True); txt = ["SHRP2 P2 EXTRA (registered descriptive analyses + one post-hoc comparison; aggregate only)", ""]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if a.what in ("all", "a"):
        A = pd.DataFrame(part_a(CP.Bank())); A.to_csv(OUT / "shrp2_p2_extra_A.csv", index=False)
        txt += ["PART A - absolute metrics on test frames (seed mean [min, max])", A.round(4).to_string(index=False), ""]
    if a.what in ("all", "b"):
        B = part_b(); B.to_csv(OUT / "shrp2_p2_extra_B.csv", index=False)
        txt += ["PART B - H5 comparators, deployment design (seed means); PART C row = C1 recall-matched (post-hoc)",
                B.round(4).to_string(index=False), ""]
    (OUT / "shrp2_p2_extra_summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt)); print(f"Saved results/shrp2_p2_extra_*.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
