"""
Phase 2, step P2-1: FEASIBILITY CHECK of label-shift (prior-shift) adaptation for crash-risk
calibration in an unseen domain WITHOUT target labels. Development work on INTERACTION only;
nothing here is confirmatory (the Phase 2 method is confirmed on SHRP2 after amendment 2).

Motivation (rehearsal, step 9): the per-pair calibration loss grew with the base-rate shift
between domains (Pearson 0.68), i.e. the shift looks largely like a change of outcome
prevalence. Label-shift adaptation estimates the new prevalence from UNLABELLED target data
(Saerens et al., 2002, EM; Alexandari et al., 2020, calibration first) and re-weights the
predicted outcome distribution. The discrete-time hazard model already gives a categorical
distribution over 5 outcomes: event in (0,1], (1,2], (2,3], (3,5] s, or no event within 5 s:
    p(bin k) = h_k * prod_{j<k} (1 - h_j),   p(none) = prod_j (1 - h_j).

Procedure for every testable H2 factor-direction pair a -> b (step 9) and seed:
  1. calibrate the source model on SOURCE validation data: temperature on the hazard logits
     (survival likelihood; the bias-corrected-calibration step of Alexandari et al.)
  2. source prior pi_s = mean calibrated outcome distribution on source validation data
  3. EM on the TARGET test samples WITHOUT their labels (max 100 iterations, tol 1e-8):
        pi_t <- mean_i normalise( p_i * pi_t / pi_s )
  4. adjusted p_i ∝ p_i * pi_t / pi_s; risk at 3 s = p(bin1) + p(bin2) + p(bin3)
Compared on the target test data (labels used ONLY for evaluation): adaptive ECE and AUPRC at
3 s for (i) raw source model, (ii) temperature only, (iii) temperature + EM, against (iv) the
in-domain model. Statistic = mean over pairs and seeds of log(ECE_method / ECE_in-domain),
i.e. the H2 statistic (lower = closer to in-domain calibration).

FEASIBILITY CRITERION (fixed before the first run on real data): label-shift adaptation is worth
building on if (iii) reduces the mean log ECE ratio to at most HALF of (i), AND is lower than (ii)
(i.e. it adds something beyond standard recalibration), with AUPRC within 0.01 of (i).
Also reported: estimated vs true target prevalence at 3 s (how well EM recovers the base rate).

Usage : python src/phase2/p2_feasibility.py
Needs : data/processed/preds/haz_dom_*.npz (step 8), results/rehearsal_h2_pairs.csv (step 9)
Output: results/p2_feasibility.csv (one row per pair x seed), printed summary
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import analysis.stats as S
import calibration as C
from train_pretrain import split_by_scene

PRED = Path("data/processed/preds")
SEEDS, K3 = [131, 132, 133, 134, 135], 2


def outcome_dist(logits):
    """(n, 4) hazard logits -> (n, 5) probabilities of event in bin 1..4 or no event."""
    h = 1.0 / (1.0 + np.exp(-logits))
    surv = np.cumprod(np.c_[np.ones(len(h)), 1 - h], axis=1)          # survival before each bin
    return np.c_[h * surv[:, :4], surv[:, 4]]


def em_prior(p, pi_s, iters=100, tol=1e-8):
    pi_t = pi_s.copy()
    for it in range(iters):
        w = p * (pi_t / pi_s); w /= w.sum(1, keepdims=True)
        new = w.mean(0)
        if np.abs(new - pi_t).max() < tol:
            pi_t = new; break
        pi_t = new
    return pi_t, it + 1


def adjust(p, pi_s, pi_t):
    w = p * (pi_t / pi_s)
    return w / w.sum(1, keepdims=True)


def main():
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    N = len(lab["current"]); keep = lab["current"] == 0
    scene = d["scene"].astype(str); ev, ob = lab["event_bin"], lab["obs_bins"]
    tr_k, va_k, te_k = split_by_scene(scene[keep], d["location"][keep].astype(str))
    va, te = np.zeros(N, bool), np.zeros(N, bool); va[keep], te[keep] = va_k, te_k
    y_all, known = C.labels_at(ev, ob, K3)
    fac = pd.read_csv("results/rehearsal_factors_by_recording.csv").set_index("scene_id")
    pairs = pd.read_csv("results/rehearsal_h2_pairs.csv")
    pairs = pairs[pairs.testable == True]
    rows = []
    for _, pr in pairs.iterrows():
        f = pr.factor; a, b = (int(x) for x in pr.direction.split("->"))
        lev = fac.loc[scene, f].to_numpy()
        src_val = np.flatnonzero(va & (lev == a))                      # labelled source validation
        tgt = np.flatnonzero(te & known & (lev == b))                  # target test (labels for evaluation only)
        tgt_all = np.flatnonzero(te & (lev == b))                      # ALL target test samples for EM (unlabelled)
        y = y_all[tgt]
        for s in SEEDS:
            zs = np.load(PRED / f"haz_dom_{f}_{a}_s{s}.npz"); zi = np.load(PRED / f"haz_dom_{f}_{b}_s{s}.npz")
            L = np.full((N, 4), np.nan); L[zs["idx"]] = zs["logits"]
            Li = np.full((N, 4), np.nan); Li[zi["idx"]] = zi["logits"]
            T = C.Temperature().fit(L[src_val], ev[src_val], ob[src_val]).T
            p_src = outcome_dist(L[src_val] / T); pi_s = p_src.mean(0)
            pi_t, n_it = em_prior(outcome_dist(L[tgt_all] / T), pi_s)
            raw = C.risk_from_logits(L[tgt])[:, K3]
            temp = C.risk_from_logits(L[tgt] / T)[:, K3]
            emr = adjust(outcome_dist(L[tgt] / T), pi_s, pi_t)[:, :3].sum(1)
            inn = C.risk_from_logits(Li[tgt])[:, K3]
            e = {k: S.ece_adaptive(y, v) for k, v in (("raw", raw), ("temp", temp), ("em", emr), ("in", inn))}
            ap = {k: S.average_precision(y, v) for k, v in (("raw", raw), ("temp", temp), ("em", emr), ("in", inn))}
            rows.append(dict(factor=f, direction=pr.direction, seed=s, T=T, em_iterations=n_it,
                             prev_true=y.mean(), prev_source=pi_s[:3].sum(), prev_em=pi_t[:3].sum(),
                             mean_pred_raw=raw.mean(), mean_pred_em=emr.mean(),
                             **{f"ece_{k}": v for k, v in e.items()}, **{f"auprc_{k}": v for k, v in ap.items()},
                             **{f"logratio_{k}": np.log(e[k] / e["in"]) for k in ("raw", "temp", "em")}))
    r = pd.DataFrame(rows); r.to_csv("results/p2_feasibility.csv", index=False)
    pd.set_option("display.width", 200)
    g = r.groupby(["factor", "direction"])[["prev_true", "prev_source", "prev_em", "logratio_raw",
                                            "logratio_temp", "logratio_em", "auprc_raw", "auprc_em"]].mean()
    print("Per pair (mean over 5 seeds; prevalence = share with an event within 3 s):")
    print(g.round(4).to_string())
    m = r[["logratio_raw", "logratio_temp", "logratio_em", "auprc_raw", "auprc_temp", "auprc_em", "auprc_in"]].mean()
    print("\nOverall mean (pairs x seeds):")
    print(m.round(4).to_string())
    err_src = (r.prev_source - r.prev_true).abs().mean(); err_em = (r.prev_em - r.prev_true).abs().mean()
    print(f"\nbase-rate recovery: mean |error| source prior {err_src:.4f} -> EM estimate {err_em:.4f}")
    c1 = m.logratio_em <= 0.5 * m.logratio_raw
    c2 = m.logratio_em < m.logratio_temp
    c3 = abs(m.auprc_em - m.auprc_raw) <= 0.01
    print(f"criterion: (1) EM <= half of raw: {c1} | (2) EM better than temperature alone: {c2} | "
          f"(3) AUPRC within 0.01: {c3}")
    print(f"FEASIBILITY CRITERION: {'MET' if (c1 and c2 and c3) else 'NOT MET'}")
    print("Saved results/p2_feasibility.csv")


if __name__ == "__main__":
    main()