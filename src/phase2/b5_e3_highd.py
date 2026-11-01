"""
Phase 2, step B5, tests E3a/E3b/E3c on highD (spec docs/phase2_method_spec_B2.md v1.1, Sections 0 and 6).
Scores: RAW cumulative risk at 3 s from the INTERACTION central models (seeds 131-135; src/highd/score_highd.py).
Labels: PRIMARY highD label TTC2D-moving < 3.0 s (threshold rule, A1; A+C decision). Negatives: not in conflict
at t, evaluable at 3 s, no event within 3 s. Unit = recording; eligible: >= 100 samples evaluable at 3 s.
Levels: alpha in {0.05, 0.10} (CRC, HP, C1), eps_op = 1/3600 per sample (HP, C1); delta = 0.10.

E3a  R-LEVEL WITHIN highD (guarantee applies: the model never saw highD, recordings are exchangeable under random
     assignment). Two designs: (a) leave one recording out, k = n - 1; (b) 1,000 random partitions (seed 131)
     with k = floor(n / 2) calibration recordings and one held-out recording from the rest
     [k = floor(n/2) fixed here, 10 Oct 2026, before any highD result; the spec left k open for E3a].
     VALIDITY RULES (spec Sections 8 and 0): CRC mean held-out FA <= alpha + 3 SE; HP share of held-out recordings
     with FA > eps <= delta + 3 SE; every seed, both designs.
E3b  ACROSS DATASETS (stress test; NOT covered by the guarantee): thresholds from the 23 INTERACTION validation
     recordings (same seed, same model) applied to every eligible highD recording.
E3c  L-LEVEL WITHIN highD: 6 locations, k = 5. CRC needs alpha >= 1/6 and HP needs delta >= 1/6, so both are
     INFEASIBLE at the registered levels and are reported as such. C1 leave-one-location-out is reported for
     information only (no guarantee).
OPTION-B TRIGGER (fixed in spec Section 6, B5; recall statistic fixed here before any result): the larger highD
     rebuild runs only if the 95% cluster-bootstrap interval (recordings, B = 2000, seed 131) of pooled recall at
     3 s for CRC at alpha = 0.10 in E3a design (a), seed-averaged, is WIDER than 0.30.
Per-location positive-based results are NOT reported (A+C decision). Recall is pooled only.
Usage : python src/phase2/b5_e3_highd.py
Needs : data/processed/preds/highd_central_aug_s13{1..5}.npz, data/processed/highd_samples.npz,
        data/processed/highd_labels_ttc30.npz, data/processed/preds/haz_central_aug_s13{1..5}.npz (E3b),
        data/processed/pretrain_samples.npz, data/processed/hazard_labels.npz (E3b)
Output: results/p2_domain_e3_units.csv, results/p2_domain_e3_summary.csv, results/p2_domain_e3_trigger.csv
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
SEEDS, K3, DELTA, MIN_EVAL, PARTS, B = [131, 132, 133, 134, 135], 2, 0.10, 100, 1000, 2000
EPS_OP = 1.0 / 3600
LEVELS = [("0.05", 0.05), ("0.10", 0.10), ("op", EPS_OP)]


def make_units(score, unit, known, y):
    U = {}
    for u in np.unique(unit):
        m = (unit == u) & known
        if m.sum() < MIN_EVAL:
            continue
        neg = score[m & (y == 0)]; pos = score[m & (y == 1)]
        if len(neg):
            U[u] = dict(neg=neg, pos=pos, curve=D.loss_curve(neg))
    return U


def thresholds(cal, lv, lname):
    curves = np.array([c["curve"] for c in cal]); pooled = np.concatenate([c["neg"] for c in cal])
    out = {"HP": D.hp_threshold(curves, lv, DELTA), "C1": dict(lam=D.pooled_split_conformal(pooled, lv), feasible=True)}
    if lname != "op":
        out["CRC"] = D.crc_threshold(curves, lv)
    return out


def main():
    t0 = time.time()
    hs = np.load("data/processed/highd_samples.npz", allow_pickle=True)
    hl = np.load("data/processed/highd_labels_ttc30.npz")
    assert float(hl["threshold"]) == 3.0, "primary highD label must be TTC2D < 3.0 s"
    keep = hl["current"] == 0
    y_h, kn_h = C.labels_at(hl["event_bin"], hl["obs_bins"], K3)
    rec_h, loc_h = hs["scene"].astype(str), hs["location"].astype(str)
    # INTERACTION validation recordings (E3b calibration source)
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene_i = d["scene"].astype(str)
    y_i, kn_i = C.labels_at(lab["event_bin"], lab["obs_bins"], K3)
    rows = []
    for s in SEEDS:
        r = np.load(PRED / f"highd_central_aug_s{s}.npz")["risk"][:, K3].astype(float)
        assert len(r) == len(keep), "highD predictions and labels differ in length"
        R_ = make_units(r[keep], rec_h[keep], kn_h[keep], y_h[keep])
        recs = sorted(R_)
        if s == SEEDS[0]:
            n_pos = sum(len(R_[u]["pos"]) for u in recs); n_neg = sum(len(R_[u]["neg"]) for u in recs)
            print(f"highD eligible recordings {len(recs)} | negatives {n_neg:,} | positives at 3 s {n_pos} | "
                  f"E3a(a) k = {len(recs) - 1}, E3a(b) k = {len(recs) // 2}", flush=True)
        # ---- E3a (a) leave one recording out ----
        for lname, lv in LEVELS:
            for u in recs:
                for m, res in thresholds([R_[v] for v in recs if v != u], lv, lname).items():
                    lam = res["lam"]; x = R_[u]
                    rows.append(dict(seed=s, test="E3a", design="a", eps_name=lname, eps=lv, method=m, held_out=u,
                                     part=-1, k=len(recs) - 1, feasible=bool(res["feasible"]), lam=lam,
                                     fa=D.false_alarm_rate(x["neg"], lam), n_neg=len(x["neg"]), n_pos=len(x["pos"]),
                                     caught=int((x["pos"] > lam).sum())))
        # ---- E3a (b) random partitions, k = floor(n/2) ----
        rng = np.random.default_rng(131); kb = len(recs) // 2
        for p in range(PARTS):
            perm = rng.permutation(len(recs)); cal = [R_[recs[i]] for i in perm[:kb]]; u = recs[perm[kb]]; x = R_[u]
            for lname, lv in LEVELS:
                for m, res in thresholds(cal, lv, lname).items():
                    lam = res["lam"]
                    rows.append(dict(seed=s, test="E3a", design="b", eps_name=lname, eps=lv, method=m, held_out=u,
                                     part=p, k=kb, feasible=bool(res["feasible"]), lam=lam,
                                     fa=D.false_alarm_rate(x["neg"], lam), n_neg=len(x["neg"]), n_pos=len(x["pos"]),
                                     caught=int((x["pos"] > lam).sum())))
        # ---- E3b across datasets: INTERACTION validation recordings -> highD recordings ----
        z = np.load(PRED / f"haz_central_aug_s{s}.npz")
        vi = z["idx"][z["split"].astype(str) == "val"]; rv = z["risk"][z["split"].astype(str) == "val", K3].astype(float)
        IV = make_units(rv, scene_i[vi], kn_i[vi], y_i[vi])
        for lname, lv in LEVELS:
            for m, res in thresholds(list(IV.values()), lv, lname).items():
                lam = res["lam"]
                for u in recs:
                    x = R_[u]
                    rows.append(dict(seed=s, test="E3b", design="-", eps_name=lname, eps=lv, method=m, held_out=u,
                                     part=-1, k=len(IV), feasible=bool(res["feasible"]), lam=lam,
                                     fa=D.false_alarm_rate(x["neg"], lam), n_neg=len(x["neg"]), n_pos=len(x["pos"]),
                                     caught=int((x["pos"] > lam).sum())))
        # ---- E3c location level within highD (k = 5) ----
        Lu = make_units(r[keep], loc_h[keep], kn_h[keep], y_h[keep]); locs = sorted(Lu)
        for lname, lv in LEVELS:
            for t in locs:
                for m, res in thresholds([Lu[j] for j in locs if j != t], lv, lname).items():
                    lam = res["lam"]; x = Lu[t]
                    rows.append(dict(seed=s, test="E3c", design="-", eps_name=lname, eps=lv, method=m, held_out=t,
                                     part=-1, k=len(locs) - 1, feasible=bool(res["feasible"]), lam=lam,
                                     fa=D.false_alarm_rate(x["neg"], lam), n_neg=len(x["neg"]), n_pos=len(x["pos"]),
                                     caught=int((x["pos"] > lam).sum())))
        print(f"  seed {s} done | {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); R["exceeds"] = R.fa > R.eps
    Path("results").mkdir(exist_ok=True); R.to_csv("results/p2_domain_e3_units.csv", index=False)
    out = []
    for (te, de, en, m, s), g in R.groupby(["test", "design", "eps_name", "method", "seed"], sort=False):
        nd = g.held_out.nunique(); p = g.exceeds.mean()
        out.append(dict(test=te, design=de, eps=en, method=m, seed=s, k=g.k.iloc[0], feasible=g.feasible.mean(),
                        mean_FA=g.fa.mean(), SE_FA=g.fa.std(ddof=1) / np.sqrt(nd) if nd > 1 else np.nan,
                        share_over=p, SE_share=np.sqrt(max(p * (1 - p), 1e-12) / nd), worst_FA=g.fa.max(),
                        FA_per_vehicle_hour=g.fa.mean() * 3600, recall=g.caught.sum() / max(g.n_pos.sum(), 1),
                        positives=int(g.drop_duplicates("held_out").n_pos.sum())))
    S = pd.DataFrame(out); S.to_csv("results/p2_domain_e3_summary.csv", index=False)
    pd.set_option("display.width", 230)
    print("\nMean over seeds:")
    print(S.groupby(["test", "design", "eps", "method"], sort=False)[["k", "feasible", "mean_FA", "share_over", "worst_FA",
          "FA_per_vehicle_hour", "recall"]].mean().round(4).to_string())
    # ---- validity rules (E3a) ----
    a = S[S.test == "E3a"].copy()
    a["structurally_infeasible"] = np.where(a.method == "CRC", a.eps.map({"0.05": .05, "0.10": .10}).fillna(0) < 1 / (a.k + 1),
                                            DELTA < 1 / (a.k + 1))
    crc = a[a.method == "CRC"]; hp = a[a.method == "HP"]
    ok_crc = ((crc.mean_FA <= crc.eps.astype(float) + 3 * crc.SE_FA) & (crc.feasible == 1)) | crc.structurally_infeasible
    ok_hp = ((hp.share_over <= DELTA + 3 * hp.SE_share) & (hp.feasible == 1)) | hp.structurally_infeasible
    print(f"\nVALIDITY RULES E3a: CRC {int(ok_crc.sum())}/{len(crc)} rows pass | HP {int(ok_hp.sum())}/{len(hp)} rows pass")
    print(f"=> {'E3a VALID (guarantee holds within highD)' if ok_crc.all() and ok_hp.all() else 'E3a: some rows fail - see summary'}")
    # ---- option-B trigger ----
    g = R[(R.test == "E3a") & (R.design == "a") & (R.eps_name == "0.10") & (R.method == "CRC")]
    per = g.groupby("held_out").agg(caught=("caught", "mean"), pos=("n_pos", "first"))
    rng = np.random.default_rng(131); idx = np.arange(len(per)); bs = []
    for _ in range(B):
        j = rng.choice(idx, len(idx), replace=True); P = per.pos.to_numpy()[j].sum()
        if P > 0:
            bs.append(per.caught.to_numpy()[j].sum() / P)
    lo, hi = np.percentile(bs, [2.5, 97.5]); est = per.caught.sum() / max(per.pos.sum(), 1)
    trig = (hi - lo) > 0.30
    pd.DataFrame([dict(recall=est, ci_low=lo, ci_high=hi, width=hi - lo, trigger=trig, resamples_used=len(bs))]).to_csv(
        "results/p2_domain_e3_trigger.csv", index=False)
    print(f"\nOPTION-B TRIGGER: pooled recall (CRC, alpha 0.10, E3a-a, seed-averaged) {est:.3f} "
          f"[{lo:.3f}, {hi:.3f}], width {hi - lo:.3f} -> {'RUN option B (larger rebuild)' if trig else 'option B NOT run'}")
    print("E3c: CRC and HP are infeasible with 6 locations at the registered levels (need alpha, delta >= 1/6).")
    print(f"Saved results/p2_domain_e3_units.csv, _summary.csv, _trigger.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()