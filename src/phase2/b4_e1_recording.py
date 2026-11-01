"""
Phase 2, step B4, test E1 (spec docs/phase2_method_spec_B2.md v1.0, Section 6): R-LEVEL validity and cost
of the domain-level false-alarm guarantee on INTERACTION. No retraining.

Unit = recording. Scores = RAW cumulative risk at 3 s from the central models (augmentation on,
seeds 131-135), saved for validation + test recordings by src/rehearsal/run_jobs.py. The central model
was trained on TRAINING recordings only, so every validation and test recording is unseen by it; with
random assignment of recordings, a held-out recording is exchangeable with the calibration recordings.
Negatives: not in conflict at t, evaluable at 3 s, no event within 3 s. Eligible unit: >= 100 samples
evaluable at 3 s (decided on counts, before any score is used).
  E1a  leave-one-recording-out over all n pooled eligible val+test recordings (k = n - 1)
  E1b  1,000 random partitions (seed 131): k calibration recordings, k = number of eligible VALIDATION
       recordings (deployment-sized), and one held-out recording drawn from the rest
Methods at alpha in {0.05, 0.10}: CRC (domain-level, expectation form), HP (high-probability form,
eps = alpha, delta = 0.10), C1 (pooled split conformal = registered H2 method), C6 (oracle: the held-out
recording's own threshold; reference only).
Per held-out recording: false-alarm rate (FA); pooled recall of positives at 3 s.
SE of the mean FA = SD over held-out recordings / sqrt(number of distinct held-out recordings)
(E1b: distinct recordings, not partitions, so the SE is not understated by re-using recordings).
DECISION RULE (spec Section 8, fixed before this run): the method is carried to amendment 2 if, for CRC,
mean held-out FA <= alpha + 3 SE for both alpha values, both variants and all 5 seeds.
Usage : python src/phase2/b4_e1_recording.py
Needs : data/processed/preds/haz_central_aug_s13{1..5}.npz, pretrain_samples.npz, hazard_labels.npz
Output: results/p2_domain_e1_units.csv (one row per held-out recording/partition x method x alpha x seed),
        results/p2_domain_e1_summary.csv, printed summary and decision
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import calibration as C
from phase2 import domain_crc as D

warnings.filterwarnings("ignore", category=DeprecationWarning)
PRED = Path("data/processed/preds")
SEEDS, K3, ALPHAS, DELTA, MIN_EVAL, PARTS = [131, 132, 133, 134, 135], 2, [0.05, 0.10], 0.10, 100, 1000


def thresholds(cal, units, alpha):
    curves = np.array([units[u]["curve"] for u in cal])
    pooled = np.concatenate([units[u]["neg"] for u in cal])
    crc = D.crc_threshold(curves, alpha); hp = D.hp_threshold(curves, alpha, DELTA)
    return {"CRC": (crc["lam"], crc["feasible"]), "HP": (hp["lam"], hp["feasible"]),
            "C1": (D.pooled_split_conformal(pooled, alpha), True)}


def score_unit(u, units, lam):
    x = units[u]
    return dict(fa=D.false_alarm_rate(x["neg"], lam), n_neg=len(x["neg"]), n_pos=len(x["pos"]),
                caught=int((x["pos"] > lam).sum()))


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
        n_eval = {u: int(kn[sc == u].sum()) for u in recs}
        elig = [u for u in recs if n_eval[u] >= MIN_EVAL]
        val = [u for u in elig if (split[sc == u] == "val").all()]
        if s == SEEDS[0]:
            print(f"recordings saved: {len(recs)} ({len(np.unique(sc[split == 'val']))} validation, "
                  f"{len(np.unique(sc[split == 'test']))} test); eligible (>= {MIN_EVAL} evaluable at 3 s): {len(elig)} "
                  f"({len(val)} validation) -> E1a k = {len(elig) - 1}, E1b k = {len(val)}")
            print(f"  excluded (too small): {sorted(set(recs) - set(elig))}")
            print(f"  feasibility limit 1/(k+1): E1a {1 / len(elig):.4f}, E1b {1 / (len(val) + 1):.4f}", flush=True)
        units = {}
        for u in elig:
            m = (sc == u) & kn
            neg = r3[m & (y == 0)]; pos = r3[m & (y == 1)]
            units[u] = dict(neg=neg, pos=pos, curve=D.loss_curve(neg))
        # ---- E1a: leave one recording out ----
        for alpha in ALPHAS:
            for u in elig:
                th = thresholds([v for v in elig if v != u], units, alpha)
                th["C6"] = (D.unit_threshold(units[u]["curve"], alpha), True)
                for meth, (lam, feas) in th.items():
                    rows.append(dict(seed=s, variant="E1a", alpha=alpha, method=meth, held_out=u, part=-1,
                                     feasible=feas, lam=lam, k=len(elig) - 1, **score_unit(u, units, lam)))
        # ---- E1b: deployment-sized random partitions ----
        rng = np.random.default_rng(131); kb = len(val)
        for p in range(PARTS):
            perm = rng.permutation(len(elig)); cal = [elig[i] for i in perm[:kb]]; u = elig[perm[kb]]
            for alpha in ALPHAS:
                th = thresholds(cal, units, alpha)
                th["C6"] = (D.unit_threshold(units[u]["curve"], alpha), True)
                for meth, (lam, feas) in th.items():
                    rows.append(dict(seed=s, variant="E1b", alpha=alpha, method=meth, held_out=u, part=p,
                                     feasible=feas, lam=lam, k=kb, **score_unit(u, units, lam)))
        print(f"  seed {s} done | {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True); R.to_csv("results/p2_domain_e1_units.csv", index=False)

    out = []                                   # explicit loop: identical across pandas versions
    for (var, alpha, meth, s), g in R.groupby(["variant", "alpha", "method", "seed"]):
        n_distinct = g["held_out"].nunique()
        out.append(dict(variant=var, alpha=alpha, method=meth, seed=s, mean_FA=g.fa.mean(),
                        SE=g.fa.std(ddof=1) / np.sqrt(n_distinct), share_FA_over_alpha=(g.fa > alpha).mean(),
                        worst_FA=g.fa.max(), recall=g.caught.sum() / max(g.n_pos.sum(), 1),
                        FA_per_agent_hour=g.fa.mean() * 3600, feasible=g.feasible.mean(), k=g.k.iloc[0]))
    S = pd.DataFrame(out)
    S.to_csv("results/p2_domain_e1_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nMean over seeds (per-seed rows in results/p2_domain_e1_summary.csv):")
    print(S.groupby(["variant", "alpha", "method"])[["mean_FA", "SE", "share_FA_over_alpha", "worst_FA", "recall",
                                                     "FA_per_agent_hour", "feasible", "k"]].mean().round(4).to_string())
    crc = S[S.method == "CRC"].copy()
    crc["structurally_infeasible"] = crc.alpha < 1.0 / (crc.k + 1)     # spec Section 5: reported, not a failure
    crc["pass"] = (crc.mean_FA <= crc.alpha + 3 * crc.SE) & (crc.feasible == 1)
    print("\nDECISION RULE (CRC: mean held-out FA <= alpha + 3 SE, both alpha, both variants, all 5 seeds):")
    print(crc[["variant", "alpha", "seed", "k", "mean_FA", "SE", "feasible", "structurally_infeasible", "pass"]].round(4).to_string(index=False))
    if crc.structurally_infeasible.any():
        print("NOTE: rows with alpha < 1/(k+1) are INFEASIBLE by construction (spec Section 5) - reported, excluded from the rule")
    ok = bool(crc.loc[~crc.structurally_infeasible, "pass"].all())
    print(f"=> {'CARRY the method to amendment 2' if ok else 'STOP and investigate (implementation or assumption)'}")
    print(f"Saved results/p2_domain_e1_units.csv and results/p2_domain_e1_summary.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()