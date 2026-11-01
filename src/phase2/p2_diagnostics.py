"""
Phase 2, step P2-2: DIAGNOSTICS after the negative P2-1 result (EM prior estimation failed).
Exploratory development work on INTERACTION; nothing here is confirmatory.

Which lever closes the calibration gap in an unseen domain? For every testable H2 pair a -> b
(step 9) and seed, starting from the source model with temperature fitted on SOURCE validation:
  (a) ORACLE PRIOR   binary prior correction at 3 s with the TRUE target prevalence:
                     p' = p r1 / (p r1 + (1 - p) r0), r1 = pi_t / pi_s, r0 = (1 - pi_t) / (1 - pi_s),
                     pi_s = mean calibrated risk on source validation. Upper bound of any
                     prevalence correction (not usable in practice).
  (b) BURN-IN        delayed outcomes: ONE target test recording is used as a labelled burn-in,
                     the rest is evaluated; repeated with every target recording as burn-in.
                       b1 prior correction with the burn-in prevalence
                       b2 temperature re-fitted on the burn-in (hazard likelihood)
  (c) COVARIATE WEIGHTS  importance-weighted temperature (Park et al., 2020): a logistic domain
                     classifier separates source-validation from target-test inputs (unlabelled;
                     8 kinematic features of step 6), weights w = P(target|x)/P(source|x) * n_s/n_t,
                     clipped at 20; temperature fitted by weighted survival likelihood.
Every method is scored by log(ECE_method / ECE_in-domain) at 3 s on the SAME evaluation samples
(lower is better; 0 = as good as the in-domain model), plus AUPRC.

DECISION RULE (fixed before the first run): a lever is carried into the Phase 2 method design if
its mean log ratio is at least 0.10 lower than temperature-only on the same samples. (a) is
reported as a ceiling only.

Usage : python src/phase2/p2_diagnostics.py
Needs : data/processed/preds/haz_dom_*.npz, results/rehearsal_h2_pairs.csv, SSM parquet (features)
Output: results/p2_diagnostics.csv, printed summary
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import analysis.stats as S
import calibration as C
from train_pretrain import split_by_scene

PRED = Path("data/processed/preds")
SEEDS, K3, W_CLIP, GRID = [131, 132, 133, 134, 135], 2, 20.0, np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
warnings.filterwarnings("ignore", category=RuntimeWarning)


def nll_per_sample(logits, ev, ob):
    n, K = logits.shape; j = np.arange(K)[None, :]
    is_ev = (ev >= 0)[:, None]; upto = np.where(is_ev, ev[:, None], ob[:, None])
    ls = lambda z: -np.logaddexp(0.0, -z)
    return -((ls(-logits) * (j < upto)).sum(1) + (ls(logits) * (is_ev & (j == ev[:, None]))).sum(1))


def temperature(logits, ev, ob, w=None):
    w = np.ones(len(logits)) if w is None else w
    return float(GRID[int(np.argmin([np.average(nll_per_sample(logits / T, ev, ob), weights=w) for T in GRID]))])


def prior_correct(p, pi_s, pi_t):
    pi_t = np.clip(pi_t, 1e-4, 0.5)
    r1, r0 = pi_t / pi_s, (1 - pi_t) / (1 - pi_s)
    return p * r1 / (p * r1 + (1 - p) * r0)


def main():
    from rehearsal.lgbm_baseline import build_features, FEATURES
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    N = len(lab["current"]); keep = lab["current"] == 0
    scene = d["scene"].astype(str); ev, ob = lab["event_bin"], lab["obs_bins"]
    tr_k, va_k, te_k = split_by_scene(scene[keep], d["location"][keep].astype(str))
    va, te = np.zeros(N, bool), np.zeros(N, bool); va[keep], te[keep] = va_k, te_k
    y_all, known = C.labels_at(ev, ob, K3)
    Xk = build_features(d, keep)
    X = np.full((N, len(FEATURES)), np.nan); X[np.flatnonzero(keep)] = Xk.to_numpy(float)
    X = np.where(np.isnan(X), np.nanmedian(X[keep], axis=0), X)             # simple imputation
    fac = pd.read_csv("results/rehearsal_factors_by_recording.csv").set_index("scene_id")
    pairs = pd.read_csv("results/rehearsal_h2_pairs.csv"); pairs = pairs[pairs.testable == True]
    rows = []
    for _, pr in pairs.iterrows():
        f = pr.factor; a, b = (int(x) for x in pr.direction.split("->"))
        lev = fac.loc[scene, f].to_numpy()
        sv = np.flatnonzero(va & (lev == a))                                   # source validation (labelled)
        tg = np.flatnonzero(te & known & (lev == b))                           # target evaluation samples
        tg_all = np.flatnonzero(te & (lev == b))                               # target inputs (unlabelled)
        y = y_all[tg]; recs = np.unique(scene[tg])
        # domain classifier for (c), shared by all seeds of this pair
        Z = np.r_[X[sv], X[tg_all]]; dlab = np.r_[np.zeros(len(sv)), np.ones(len(tg_all))]
        sc = StandardScaler().fit(Z); clf = LogisticRegression(max_iter=2000).fit(sc.transform(Z), dlab)
        pt = clf.predict_proba(sc.transform(X[sv]))[:, 1]
        w = np.clip(pt / (1 - pt) * len(sv) / len(tg_all), 0, W_CLIP)
        for s in SEEDS:
            zs = np.load(PRED / f"haz_dom_{f}_{a}_s{s}.npz"); zi = np.load(PRED / f"haz_dom_{f}_{b}_s{s}.npz")
            L = np.full((N, 4), np.nan); L[zs["idx"]] = zs["logits"]
            Li = np.full((N, 4), np.nan); Li[zi["idx"]] = zi["logits"]
            T = temperature(L[sv], ev[sv], ob[sv])
            pi_s = C.risk_from_logits(L[sv] / T)[:, K3].mean()
            risk_T = C.risk_from_logits(L / np.where(np.isnan(L), 1, T))[:, K3]
            inn = C.risk_from_logits(Li)[:, K3]
            Tw = temperature(L[sv], ev[sv], ob[sv], w)
            risk_W = C.risk_from_logits(L / np.where(np.isnan(L), 1, Tw))[:, K3]
            lr = lambda p, idx: np.log(S.ece_adaptive(y_all[idx], p[idx]) / S.ece_adaptive(y_all[idx], inn[idx]))
            ap = lambda p, idx: S.average_precision(y_all[idx], p[idx])
            oracle = prior_correct(risk_T, pi_s, y.mean())
            row = dict(factor=f, direction=pr.direction, seed=s, T=T, T_weighted=Tw, ess_weights=w.sum() ** 2 / (w ** 2).sum() / len(w),
                       lr_temp=lr(risk_T, tg), lr_oracle=lr(oracle, tg), lr_weighted=lr(risk_W, tg),
                       auprc_temp=ap(risk_T, tg), auprc_oracle=ap(oracle, tg), auprc_weighted=ap(risk_W, tg))
            # (b) burn-in: each target recording in turn
            b_temp, b_prior, b_refit = [], [], []
            for r in recs:
                bi = tg[scene[tg] == r]; ev_idx = tg[scene[tg] != r]
                if y_all[ev_idx].sum() == 0 or len(np.unique(scene[ev_idx])) == 0:
                    continue
                bi_all = np.flatnonzero(te & (lev == b) & (scene == r))
                p_prior = prior_correct(risk_T, pi_s, y_all[bi].mean())
                Tb = temperature(L[bi_all], ev[bi_all], ob[bi_all])
                p_refit = C.risk_from_logits(L / np.where(np.isnan(L), 1, Tb))[:, K3]
                b_temp.append(lr(risk_T, ev_idx)); b_prior.append(lr(p_prior, ev_idx)); b_refit.append(lr(p_refit, ev_idx))
            row.update(lr_burnin_temp=np.mean(b_temp), lr_burnin_prior=np.mean(b_prior), lr_burnin_refitT=np.mean(b_refit),
                       burnin_splits=len(b_temp))
            rows.append(row)
    r = pd.DataFrame(rows); r.to_csv("results/p2_diagnostics.csv", index=False)
    pd.set_option("display.width", 220)
    cols = ["lr_temp", "lr_oracle", "lr_weighted", "lr_burnin_temp", "lr_burnin_prior", "lr_burnin_refitT"]
    print("Per pair (mean over 5 seeds; log ratio to in-domain ECE, lower is better):")
    print(r.groupby(["factor", "direction"])[cols + ["ess_weights"]].mean().round(3).to_string())
    m = r[cols + ["auprc_temp", "auprc_oracle", "auprc_weighted"]].mean()
    print("\nOverall mean:"); print(m.round(4).to_string())
    print("\nDECISION RULE (carry a lever if >= 0.10 better than temperature on the same samples):")
    print(f"  (a) oracle prior      {m.lr_temp - m.lr_oracle:+.3f}  (ceiling only)")
    print(f"  (b1) burn-in prior    {m.lr_burnin_temp - m.lr_burnin_prior:+.3f}  -> {'CARRY' if m.lr_burnin_temp - m.lr_burnin_prior >= 0.10 else 'drop'}")
    print(f"  (b2) burn-in refit T  {m.lr_burnin_temp - m.lr_burnin_refitT:+.3f}  -> {'CARRY' if m.lr_burnin_temp - m.lr_burnin_refitT >= 0.10 else 'drop'}")
    print(f"  (c) weighted T        {m.lr_temp - m.lr_weighted:+.3f}  -> {'CARRY' if m.lr_temp - m.lr_weighted >= 0.10 else 'drop'}")
    print("Saved results/p2_diagnostics.csv")


if __name__ == "__main__":
    main()