"""
Phase 2, step B7 (spec docs/phase2_method_spec_B2.md v1.1, Sections 3.3, 6 "B7" and 7): FEDERATED FORM and the
COMPARISON with published methods, on identical units, scores and levels. No training.

Settings (each unit = one federated client; the held-out unit is the NEW client):
  A  INTERACTION, recording level, leave one recording out (as E1a), central models, seeds 131-135
  B  INTERACTION, location level, symmetric leave-one-location-out (as E2 level L), fold models, seed 131
  C  highD, recording level, leave one recording out (as E3a design a), INTERACTION central models, seeds 131-135
Levels: a in {0.05, 0.10}; eps_op = 1/3600 per sample. delta = 0.10 for HP.
Methods: CRC, HP (ours; domain_crc.py); C1 pooled split conformal (registered); C2 Lu et al. 2023; C3 Humbert et al.
2023; C4 Shahid 2026 (unseen-site form); C5 Plassier et al. 2023 (class-0 reduction); C6 oracle (held-out unit's own
threshold; reference only). Definitions and the unseen-client reading of each comparator: src/phase2/comparators.py.
CRC is not defined at eps_op (needs k >= 3,599) and is omitted there.
Reported per setting x level x method (mean over seeds): mean FA over new clients (what CRC bounds), share of new
clients with FA > level (what HP bounds), POOLED FA = negatives-weighted FA (what the sample-level methods C1, C2, C3,
C5 target), worst FA, pooled recall at 3 s, feasibility, upload per client.
FEDERATED CHECK (spec 3.3): in setting A, seed 131, the threshold from the client messages (server sum of curves) must
equal the centralised threshold for EVERY held-out client and both levels.
Descriptive comparison; no new confirmatory test (those are worded for SHRP2 in amendment 2).
Usage : python src/phase2/b7_comparison.py           (about 20-45 min on a laptop; C3 dominates)
Output: results/p2_b7_units.csv, results/p2_b7_summary.csv
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import calibration as C
from phase2 import domain_crc as D
from phase2 import comparators as K

PRED = Path("data/processed/preds")
SEEDS, K3, DELTA, MIN_EVAL, EPS_OP = [131, 132, 133, 134, 135], 2, 0.10, 100, 1.0 / 3600
LEVELS = [("0.05", 0.05), ("0.10", 0.10), ("op", EPS_OP)]
UPLOAD = {"CRC": "2,802 floats (curve)", "HP": "1 float", "C1": "all scores", "C2": "T-Digest sketch (paper)",
          "C3": "1 float", "C4": "curve + count", "C5": "all scores + share"}


def units(score, unit, known, y):
    U = {}
    for u in np.unique(unit):
        m = (unit == u) & known
        if m.sum() < MIN_EVAL:
            continue
        neg, pos = score[m & (y == 0)], score[m & (y == 1)]
        if len(neg):
            U[u] = dict(neg=neg, pos=pos, curve=D.loss_curve(neg), share0=len(neg) / int(m.sum()))
    return U


def all_thresholds(cal, lv, ln):
    curves = np.array([c["curve"] for c in cal]); scores = [c["neg"] for c in cal]
    out = {"HP": D.hp_threshold(curves, lv, DELTA),
           "C1": dict(lam=D.pooled_split_conformal(np.concatenate(scores), lv)),
           "C2": K.c2_lu_fcp(scores, lv), "C3": K.c3_humbert_qq(scores, lv),
           "C4": K.c4_shahid_unseen(curves, [len(s) for s in scores], lv, D.GRID),
           "C5": K.c5_plassier_class0(scores, [c["share0"] for c in cal], lv)}
    if ln != "op":
        out["CRC"] = D.crc_threshold(curves, lv)
    return out


def run_setting(name, U, seed, rows, t0):
    ids = sorted(U)
    for ln, lv in LEVELS:
        for u in ids:
            cal = [U[v] for v in ids if v != u]; x = U[u]
            th = all_thresholds(cal, lv, ln); th["C6"] = dict(lam=D.unit_threshold(x["curve"], lv))
            for m, r in th.items():
                lam = r["lam"]
                rows.append(dict(setting=name, seed=seed, level=ln, eps=lv, method=m, held_out=u, k=len(cal),
                                 feasible=bool(np.isfinite(lam)), lam=lam, fa=D.false_alarm_rate(x["neg"], lam),
                                 n_neg=len(x["neg"]), n_pos=len(x["pos"]), caught=int((x["pos"] > lam).sum())))
    print(f"  {name} seed {seed}: {len(ids)} clients done | {time.time() - t0:.0f}s", flush=True)


def main():
    t0 = time.time(); rows = []
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene, loc = d["scene"].astype(str), d["location"].astype(str)
    y_i, kn_i = C.labels_at(lab["event_bin"], lab["obs_bins"], K3)
    # ---- federated check (setting A, seed 131) ----
    z = np.load(PRED / "haz_central_aug_s131.npz"); idx = z["idx"]; r3 = z["risk"][:, K3].astype(float)
    UA = units(r3, scene[idx], kn_i[idx], y_i[idx]); ids = sorted(UA); same = 0; total = 0
    for lv in (0.05, 0.10):
        for u in ids:
            msgs = [D.client_message(UA[v]["neg"]) for v in ids if v != u]
            fed = D.server_threshold(msgs, lv); cen = D.crc_threshold(np.array([UA[v]["curve"] for v in ids if v != u]), lv)
            same += int(fed == cen); total += 1
    print(f"FEDERATED CHECK (setting A, seed 131): federated = centralised in {same}/{total} cases "
          f"-> {'PASS' if same == total else 'FAIL'}", flush=True)
    # ---- A: INTERACTION recordings ----
    for s in SEEDS:
        z = np.load(PRED / f"haz_central_aug_s{s}.npz"); idx = z["idx"]
        run_setting("A_INTERACTION_recordings", units(z["risk"][:, K3].astype(float), scene[idx], kn_i[idx], y_i[idx]),
                    s, rows, t0)
    # ---- B: INTERACTION locations (symmetric LOLO, seed 131) ----
    UB = {}
    for L in sorted(np.unique(loc)):
        z = np.load(PRED / f"lolo_{L}_s131.npz"); it = z["idx_tgt"]
        rt = C.risk_from_logits(z["logits_tgt"].astype(float))[:, K3]
        UB.update(units(rt, loc[it], kn_i[it], y_i[it]))
    run_setting("B_INTERACTION_locations", UB, 131, rows, t0)
    # ---- C: highD recordings ----
    hs = np.load("data/processed/highd_samples.npz", allow_pickle=True)
    hl = np.load("data/processed/highd_labels_ttc30.npz"); keep = hl["current"] == 0
    y_h, kn_h = C.labels_at(hl["event_bin"], hl["obs_bins"], K3); rec_h = hs["scene"].astype(str)
    for s in SEEDS:
        r = np.load(PRED / f"highd_central_aug_s{s}.npz")["risk"][:, K3].astype(float)
        run_setting("C_highD_recordings", units(r[keep], rec_h[keep], kn_h[keep], y_h[keep]), s, rows, t0)
    R = pd.DataFrame(rows); R["exceeds"] = R.fa > R.eps
    Path("results").mkdir(exist_ok=True); R.to_csv("results/p2_b7_units.csv", index=False)
    out = []
    for (st, ln, m, s), g in R.groupby(["setting", "level", "method", "seed"], sort=False):
        out.append(dict(setting=st, level=ln, method=m, seed=s, k=g.k.iloc[0], feasible=g.feasible.mean(),
                        mean_FA=g.fa.mean(), share_over=g.exceeds.mean(),
                        pooled_FA=(g.fa * g.n_neg).sum() / g.n_neg.sum(), worst_FA=g.fa.max(),
                        recall=g.caught.sum() / max(g.n_pos.sum(), 1)))
    S = pd.DataFrame(out).groupby(["setting", "level", "method"], sort=False).mean(numeric_only=True).reset_index()
    S["upload"] = S.method.map(UPLOAD).fillna("-")
    S.drop(columns="seed").to_csv("results/p2_b7_summary.csv", index=False)
    pd.set_option("display.width", 240)
    order = {m: i for i, m in enumerate(["CRC", "HP", "C1", "C2", "C3", "C4", "C5", "C6"])}
    for st in S.setting.unique():
        print(f"\n=== {st} (mean over seeds; level = target FA; share_over = share of NEW clients above the level) ===")
        T = S[S.setting == st].copy(); T["o"] = T.method.map(order)
        print(T.sort_values(["level", "o"])[["level", "method", "k", "feasible", "mean_FA", "share_over", "pooled_FA",
                                              "worst_FA", "recall", "upload"]].round(4).to_string(index=False))
    print("\nReading guide: CRC bounds mean_FA; HP bounds share_over (<= delta = 0.10); C1/C2/C3/C5 target pooled_FA"
          " (sample-level); C4 = sample-weighted CRC for an unseen site; C6 = oracle reference (uses the new client's"
          " own outcomes).")
    print(f"Saved results/p2_b7_units.csv and results/p2_b7_summary.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()