"""
Phase 2, step B4, test E1c (spec v1.1, Section 0, item 3): the high-probability form at the OPERATIONAL
false-alarm level on INTERACTION, recording level. POST-HOC development analysis (E1 at alpha already seen).

eps_op = 1/3600 per sample = 1 false alarm per agent-hour (INTERACTION: one sample per agent-second).
HP: per-recording threshold t_j = first grid point with FA_j <= eps_op (zero false alarms if the recording
has fewer than 3,600 negatives); lam = r-th smallest t_j over the k calibration recordings,
r = ceil((k+1)(1-delta)), delta = 0.10.  Guarantee: P(FA_new > eps_op) <= delta (exchangeable recordings).
The expectation form is infeasible here (needs k >= 3,599) and is only reported as such.
Same units, scores, eligibility and designs as E1 (src/phase2/b4_e1_recording.py):
  E1a leave one recording out (k = n - 1);  E1b 1,000 random partitions, k = number of validation recordings.
Comparator C1 (pooled split conformal at eps_op) on the same calibration data.
VALIDITY RULE (spec v1.1): share of held-out recordings with FA > eps_op <= delta + 3 SE
(binomial SE over distinct held-out recordings), for both designs and all 5 seeds.
Usage : python src/phase2/b4_e1c_operational.py
Output: results/p2_domain_e1c_units.csv, results/p2_domain_e1c_summary.csv
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import calibration as C
from phase2 import domain_crc as D

PRED = Path("data/processed/preds")
SEEDS, K3, DELTA, MIN_EVAL, PARTS = [131, 132, 133, 134, 135], 2, 0.10, 100, 1000
EPS_OP = 1.0 / 3600


def main():
    t0 = time.time()
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene = d["scene"].astype(str)
    y_all, known = C.labels_at(lab["event_bin"], lab["obs_bins"], K3)
    rows = []
    for s in SEEDS:
        z = np.load(PRED / f"haz_central_aug_s{s}.npz")
        idx, split, r3 = z["idx"], z["split"].astype(str), z["risk"][:, K3].astype(float)
        sc, kn, y = scene[idx], known[idx], y_all[idx]
        recs = np.unique(sc)
        elig = [u for u in recs if int(kn[sc == u].sum()) >= MIN_EVAL]
        val = [u for u in elig if (split[sc == u] == "val").all()]
        U = {}
        for u in elig:
            m = (sc == u) & kn
            neg = r3[m & (y == 0)]; pos = r3[m & (y == 1)]
            cv = D.loss_curve(neg)
            U[u] = dict(neg=neg, pos=pos, curve=cv, t=D.unit_threshold(cv, EPS_OP))
        if s == SEEDS[0]:
            small = sum(len(U[u]["neg"]) < 1 / EPS_OP for u in elig)
            print(f"eligible recordings {len(elig)} ({len(val)} validation); eps_op = 1/3600 = {EPS_OP:.6f}")
            print(f"recordings with fewer than 3,600 negatives (per-unit threshold = zero false alarms): {small} of {len(elig)}")
            print(f"negatives per recording: median {np.median([len(U[u]['neg']) for u in elig]):.0f}, "
                  f"min {min(len(U[u]['neg']) for u in elig)}, max {max(len(U[u]['neg']) for u in elig)}", flush=True)

        def run(cal, u, variant, part):
            k = len(cal); r = int(np.ceil((k + 1) * (1 - DELTA)))
            hp = np.sort([U[v]["t"] for v in cal])[r - 1] if r <= k else np.inf
            c1 = D.pooled_split_conformal(np.concatenate([U[v]["neg"] for v in cal]), EPS_OP)
            for meth, lam in (("HP", hp), ("C1", c1)):
                fa = D.false_alarm_rate(U[u]["neg"], lam)
                rows.append(dict(seed=s, variant=variant, method=meth, held_out=u, part=part, k=k, lam=lam,
                                 feasible=bool(np.isfinite(lam)), fa=fa, exceeds=fa > EPS_OP,
                                 n_neg=len(U[u]["neg"]), n_pos=len(U[u]["pos"]), caught=int((U[u]["pos"] > lam).sum())))

        for u in elig:
            run([v for v in elig if v != u], u, "E1a", -1)
        rng = np.random.default_rng(131); kb = len(val)
        for p in range(PARTS):
            perm = rng.permutation(len(elig)); run([elig[i] for i in perm[:kb]], elig[perm[kb]], "E1b", p)
        print(f"  seed {s} done | {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); Path("results").mkdir(exist_ok=True)
    R.to_csv("results/p2_domain_e1c_units.csv", index=False)
    out = []
    for (var, meth, s), g in R.groupby(["variant", "method", "seed"]):
        nd = g.held_out.nunique(); p = g.exceeds.mean()
        out.append(dict(variant=var, method=meth, seed=s, k=g.k.iloc[0], share_FA_over_eps=p,
                        SE_share=np.sqrt(max(p * (1 - p), 1e-12) / nd), mean_FA=g.fa.mean(),
                        FA_per_agent_hour=g.fa.mean() * 3600, worst_FA_per_agent_hour=g.fa.max() * 3600,
                        recall=g.caught.sum() / max(g.n_pos.sum(), 1), feasible=g.feasible.mean()))
    S = pd.DataFrame(out); S.to_csv("results/p2_domain_e1c_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nMean over seeds:")
    print(S.groupby(["variant", "method"])[["k", "share_FA_over_eps", "SE_share", "FA_per_agent_hour",
                                            "worst_FA_per_agent_hour", "recall", "feasible"]].mean().round(4).to_string())
    hp = S[S.method == "HP"].copy()
    hp["pass"] = (hp.share_FA_over_eps <= DELTA + 3 * hp.SE_share) & (hp.feasible == 1)
    print(f"\nVALIDITY RULE (HP: share of held-out recordings with FA > eps_op <= {DELTA} + 3 SE):")
    print(hp[["variant", "seed", "k", "share_FA_over_eps", "SE_share", "recall", "pass"]].round(4).to_string(index=False))
    print(f"=> {'HP VALID at the operational level' if hp['pass'].all() else 'HP NOT VALID - investigate'}")
    print(f"Saved results/p2_domain_e1c_units.csv and results/p2_domain_e1c_summary.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()