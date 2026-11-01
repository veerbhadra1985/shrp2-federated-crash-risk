"""
Phase 2, step B4, test E2 (spec docs/phase2_method_spec_B2.md v1.1, Sections 0, 4.1 and 6): the domain-level
false-alarm guarantee at UNSEEN LOCATIONS on INTERACTION. Uses the 12 leave-one-location-out fold models
(Phase 1 settings, seed 131), whose predictions were saved by: python src/lolo_calibration.py --save-preds
(re-run reproduced Phase 1 exactly). No training here.

Score = RAW cumulative risk at 3 s. Negative = evaluable at 3 s, no event within 3 s (current-conflict samples
already removed). Eligible unit: >= 100 samples evaluable at 3 s.

(i)  R-LEVEL THRESHOLD AT A NEW LOCATION ("cost of shift"; NOT covered by any guarantee).
     Fold t: calibration units = the VALIDATION recordings of the other 11 locations, scored by fold model t
     (unseen recordings of seen locations). The threshold is applied to ALL recordings of location t.
(ii) L-LEVEL, symmetric leave-one-location-out scheme (spec 4.1a, decision D3). Unit j = location j; its loss
     curve = false-alarm rate of ALL its recordings under the fold model that excluded j. For target t the
     calibration units are the 11 other locations (k = 11). By the symmetry of the procedure the 12 losses
     are exchangeable if locations are exchangeable draws; the guarantee then covers the held-out location
     under its own fold model. Caveat (spec 4.1a): exact as an evaluation of the procedure; approximate as a
     deployment rule for a 13th location.
Methods: CRC (expectation form) at alpha in {0.05, 0.10}; HP (high-probability form, delta = 0.10) at
eps in {0.05, 0.10, eps_op = 1/3600}; C1 pooled split conformal (registered method) at the same levels.
Infeasible settings are reported as such (CRC needs alpha >= 1/(k+1); HP needs delta >= 1/(k+1)).
Per target location: false-alarm rate (FA), FA per vehicle-hour, share of its recordings with FA > level,
recall at 3 s. Summary over the 12 locations: mean FA (the quantity CRC bounds), number of locations with
FA > level (the quantity HP bounds: expected share <= delta), pooled recall.
NOTE (fixed before the run): 12 locations give ONE realisation of a 12-unit scheme, so E2 cannot "test" the
guarantee statistically; it shows what the guarantee delivers here and what it costs. Reported, not tuned.
Usage : python src/phase2/b4_e2_location.py
Needs : data/processed/preds/lolo_<location>_s131.npz (12 files), pretrain_samples.npz, hazard_labels.npz
Output: results/p2_domain_e2_locations.csv, results/p2_domain_e2_summary.csv
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
K3, DELTA, MIN_EVAL, EPS_OP = 2, 0.10, 100, 1.0 / 3600
LEVELS = [("0.05", 0.05), ("0.10", 0.10), ("op", EPS_OP)]


def risk3(logits):
    return C.risk_from_logits(np.asarray(logits, float))[:, K3]


def main():
    t0 = time.time()
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene, loc = d["scene"].astype(str), d["location"].astype(str)
    y_all, known = C.labels_at(lab["event_bin"], lab["obs_bins"], K3)
    LOCS = sorted(np.unique(loc))
    miss = [L for L in LOCS if not (PRED / f"lolo_{L}_s131.npz").exists()]
    if miss:
        raise SystemExit(f"missing fold predictions for {miss} - run lolo_calibration.py --save-preds first")
    F = {}
    for L in LOCS:
        z = np.load(PRED / f"lolo_{L}_s131.npz")
        ic, it = z["idx_cal"], z["idx_tgt"]
        assert (loc[it] == L).all() and not (loc[ic] == L).any(), f"fold {L}: index check failed"
        F[L] = dict(ic=ic, rc=risk3(z["logits_cal"]), it=it, rt=risk3(z["logits_tgt"]))
    # ---- unit data ----
    def units_from(idx, r, by):
        out = {}
        for u in np.unique(by[idx]):
            m = (by[idx] == u) & known[idx]
            if m.sum() < MIN_EVAL:
                continue
            neg = r[m & (y_all[idx] == 0)]; pos = r[m & (y_all[idx] == 1)]
            if len(neg):
                out[u] = dict(neg=neg, pos=pos, curve=D.loss_curve(neg))
        return out
    LOCU = {L: units_from(F[L]["it"], F[L]["rt"], loc)[L] for L in LOCS}          # location units (own fold model)
    print(f"12 folds loaded | negatives per location: " +
          ", ".join(f"{L.split('_', 1)[-1]} {len(LOCU[L]['neg'])}" for L in LOCS), flush=True)
    rows = []
    for t in LOCS:
        tgt = LOCU[t]
        trec = units_from(F[t]["it"], F[t]["rt"], scene)                              # recordings of t (for shares)
        # (i) R-level calibration units: validation recordings of the other 11 locations, fold model t
        rcal = units_from(F[t]["ic"], F[t]["rc"], scene)
        # (ii) L-level calibration units: the other 11 locations, each under its own fold model
        lcal = [LOCU[j] for j in LOCS if j != t]
        for level, (lname, lv) in [("R", x) for x in LEVELS] + [("L", x) for x in LEVELS]:
            cal = list(rcal.values()) if level == "R" else lcal
            curves = np.array([u["curve"] for u in cal]); pooled = np.concatenate([u["neg"] for u in cal])
            k = len(cal)
            meths = {"HP": D.hp_threshold(curves, lv, DELTA), "C1": dict(lam=D.pooled_split_conformal(pooled, lv), feasible=True)}
            if lname != "op":
                meths["CRC"] = D.crc_threshold(curves, lv)
            for m, res in meths.items():
                lam = res["lam"]
                fa = D.false_alarm_rate(tgt["neg"], lam)
                rec_share = np.mean([D.false_alarm_rate(u["neg"], lam) > lv for u in trec.values()])
                rows.append(dict(target=t, level=level, eps_name=lname, eps=lv, method=m, k=k, feasible=bool(res["feasible"]),
                                 lam=lam, fa=fa, exceeds=fa > lv, fa_per_vehicle_hour=fa * 3600,
                                 share_recordings_over=rec_share, n_recordings=len(trec), n_neg=len(tgt["neg"]),
                                 n_pos=len(tgt["pos"]), caught=int((tgt["pos"] > lam).sum())))
    R = pd.DataFrame(rows); Path("results").mkdir(exist_ok=True)
    R.to_csv("results/p2_domain_e2_locations.csv", index=False)
    out = []
    for (lev, en, m), g in R.groupby(["level", "eps_name", "method"], sort=False):
        out.append(dict(level=lev, eps=en, method=m, k=g.k.median(), feasible_folds=int(g.feasible.sum()),
                        mean_FA=g.fa.mean(), locations_over=int(g.exceeds.sum()), share_locations_over=g.exceeds.mean(),
                        worst_FA=g.fa.max(), mean_FA_per_vehicle_hour=g.fa_per_vehicle_hour.mean(),
                        mean_share_recordings_over=g.share_recordings_over.mean(),
                        recall=g.caught.sum() / max(g.n_pos.sum(), 1)))
    S = pd.DataFrame(out); S.to_csv("results/p2_domain_e2_summary.csv", index=False)
    pd.set_option("display.width", 230)
    print("\nSummary over the 12 held-out locations (level R = threshold from other locations' recordings;"
          " level L = symmetric location-level scheme):")
    print(S.round(4).to_string(index=False))
    print("\nPer location, alpha/eps = 0.05 (FA; * = above the level):")
    p = R[R.eps_name == "0.05"].assign(cell=lambda x: np.where(x.feasible, x.fa.map(lambda v: f"{v:.3f}") + np.where(x.exceeds, "*", " "), "infeas."))
    print(p.pivot_table(index="target", columns=["level", "method"], values="cell", aggfunc="first").to_string())
    print("\nReading guide: CRC bounds the MEAN FA over new locations by alpha; HP bounds the expected SHARE of new"
          f" locations above eps by delta = {DELTA} (here: at most about 1 of 12 on average). Level R has no guarantee"
          " at a new location (shift). 12 locations = one realisation: descriptive, not a statistical test.")
    print(f"Saved results/p2_domain_e2_locations.csv and results/p2_domain_e2_summary.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()