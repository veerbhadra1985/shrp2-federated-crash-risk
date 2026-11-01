"""
Phase 2, step B6 (spec docs/phase2_method_spec_B2.md v1.1, Section 6 "B6", Section 0 item 4): LEARNING CURVE of
delayed outcomes at an UNSEEN location. Exploratory; protocol fixed in the spec and here BEFORE any result.

For each held-out location t (12 leave-one-location-out fold models, seed 131; predictions saved by
lolo_calibration.py --save-preds): the recordings of t "arrive" in order of recording ID (proxy for time; stated
as a limitation). After the first m recordings (m = 1 .. n_t - 1) their outcomes are known (a negative at time t
is confirmed at t + 3 s); the REMAINING recordings of t are evaluated.
Score = RAW cumulative risk at 3 s. Levels: alpha in {0.05, 0.10}; eps_op = 1/3600 (HP and C1 only); delta 0.10.
Methods along the curve:
  SRC-CRC, SRC-HP   source-only thresholds from the validation recordings of the 11 other locations (fold model t);
                    constant reference
  LOC-C1            split conformal on the pooled burn-in negatives of t (no unit-level guarantee)
  LOC-CRC, LOC-HP   domain-level methods with units = consecutive 60 s segments of the burn-in recordings
                    (segment = recording x floor(frame / 600)); eligible segment: >= 20 samples evaluable at 3 s
                    [eligibility 20 fixed here, 10 Oct 2026, before any result]. Used only when feasible
                    (CRC: alpha >= 1/(k+1); HP: delta >= 1/(k+1)); otherwise reported as infeasible.
  Calibration       temperature fitted on the source validation data (T_src) vs re-fitted on the burn-in (T_burn);
                    adaptive ECE at 3 s on the remaining recordings.
x-axis: m, burn-in samples, and burn-in minutes of recording (sum over burn-in recordings of (last - first frame)/600).
READ-OUTS per location (fixed in the spec; positive rule fixed here before any result):
  (1) first m at which a LOCAL method has FA on the remainder <= level AND recall >= the SRC method's recall at the
      same level (compared like for like: LOC-CRC vs SRC-CRC, LOC-HP vs SRC-HP); evaluated only at m where the
      remainder holds >= 5 positives [fixed here]; reported as "not reached" otherwise;
  (2) first m at which the ECE with T_burn is below the ECE with T_src on the same remainder.
Summary: median over locations (with range) of the minutes needed, and the number of locations that reach it.
Usage : python src/phase2/b6_learning_curve.py
Needs : data/processed/preds/lolo_<location>_s131.npz, pretrain_samples.npz, hazard_labels.npz
Output: results/p2_b6_curve.csv (one row per location x m x method x level), results/p2_b6_readouts.csv,
        results/p2_b6_ece.csv
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
K3, DELTA, MIN_EVAL_REC, MIN_EVAL_SEG, MIN_POS_REMAIN, EPS_OP = 2, 0.10, 100, 20, 5, 1.0 / 3600
LEVELS = [("0.05", 0.05), ("0.10", 0.10), ("op", EPS_OP)]


def rid(s):
    """Numeric part of a recording id for ordering (falls back to the string)."""
    tail = s.split("/")[-1]; digits = "".join(ch for ch in tail if ch.isdigit())
    return (int(digits) if digits else 10 ** 9, tail)


def main():
    t0 = time.time()
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene, loc, frame = d["scene"].astype(str), d["location"].astype(str), d["frame"].astype(np.int64)
    ev, ob = lab["event_bin"], lab["obs_bins"]
    y_all, known = C.labels_at(ev, ob, K3)
    curve_rows, read_rows, ece_rows = [], [], []
    for t in sorted(np.unique(loc)):
        z = np.load(PRED / f"lolo_{t}_s131.npz")
        ic, it = z["idx_cal"], z["idx_tgt"]; lc, lt = z["logits_cal"].astype(float), z["logits_tgt"].astype(float)
        rc, rt = C.risk_from_logits(lc)[:, K3], C.risk_from_logits(lt)[:, K3]
        # source reference thresholds (validation recordings of the other locations, fold model t)
        src_units = []
        for u in np.unique(scene[ic]):
            m = (scene[ic] == u) & known[ic]
            if m.sum() >= MIN_EVAL_REC and (m & (y_all[ic] == 0)).any():
                src_units.append(D.loss_curve(rc[m & (y_all[ic] == 0)]))
        src_units = np.array(src_units)
        SRCT = {}
        for ln, lv in LEVELS:
            if ln != "op":
                SRCT[("SRC-CRC", ln)] = D.crc_threshold(src_units, lv)["lam"]
            SRCT[("SRC-HP", ln)] = D.hp_threshold(src_units, lv, DELTA)["lam"]
        T_src = C.Temperature().fit(lc, ev[ic], ob[ic]).T
        # target recordings in arrival order
        recs = sorted(np.unique(scene[it]), key=rid)
        if len(recs) < 2:
            print(f"{t}: {len(recs)} recording(s) - no learning curve possible"); continue
        sc_t, kn_t, y_t, fr_t = scene[it], known[it], y_all[it], frame[it]
        seg_t = np.array([f"{s}#{f // 600}" for s, f in zip(sc_t, fr_t)])
        rec_minutes = {u: (fr_t[sc_t == u].max() - fr_t[sc_t == u].min()) / 600.0 for u in recs}
        # per-segment curves and per-level unit thresholds (computed once)
        SEG = {}
        for g in np.unique(seg_t):
            m = (seg_t == g) & kn_t
            neg = rt[m & (y_t == 0)]
            if m.sum() >= MIN_EVAL_SEG and len(neg):
                cv = D.loss_curve(neg)
                SEG[g] = dict(rec=g.split("#")[0], curve=cv, t={ln: D.unit_threshold(cv, lv) for ln, lv in LEVELS})
        reached = {}; ece_reached = None
        for m_ in range(1, len(recs)):
            burn, rest = set(recs[:m_]), set(recs[m_:])
            bi = np.isin(sc_t, list(burn)); ri = np.isin(sc_t, list(rest))
            rneg = rt[ri & kn_t & (y_t == 0)]; rpos = rt[ri & kn_t & (y_t == 1)]
            bneg = rt[bi & kn_t & (y_t == 0)]
            mins = sum(rec_minutes[u] for u in burn); n_burn = int((bi & kn_t).sum())
            segs = [v for v in SEG.values() if v["rec"] in burn]
            k = len(segs)
            meths = {}
            for ln, lv in LEVELS:
                if ln != "op":
                    meths[("SRC-CRC", ln)] = (SRCT[("SRC-CRC", ln)], True)
                    res = D.crc_threshold(np.array([s["curve"] for s in segs]), lv) if k else dict(lam=np.inf, feasible=False)
                    meths[("LOC-CRC", ln)] = (res["lam"], res["feasible"])
                meths[("SRC-HP", ln)] = (SRCT[("SRC-HP", ln)], True)
                r = int(np.ceil((k + 1) * (1 - DELTA)))
                meths[("LOC-HP", ln)] = ((np.sort([s["t"][ln] for s in segs])[r - 1], True) if (k and r <= k) else (np.inf, False))
                meths[("LOC-C1", ln)] = (D.pooled_split_conformal(bneg, lv) if len(bneg) else np.inf, len(bneg) > 0)
            for (meth, ln), (lam, feas) in meths.items():
                lv = dict(LEVELS)[ln]
                fa = D.false_alarm_rate(rneg, lam) if len(rneg) else np.nan
                rec = float((rpos > lam).mean()) if len(rpos) else np.nan
                curve_rows.append(dict(location=t, m=m_, n_recordings=len(recs), burn_minutes=mins, burn_samples=n_burn,
                                       method=meth, level=ln, eps=lv, k_segments=k, feasible=bool(feas), lam=lam, fa=fa,
                                       fa_ok=bool(fa <= lv) if np.isfinite(fa) else False, recall=rec,
                                       remain_pos=len(rpos), remain_neg=len(rneg)))
            # read-out (1)
            for fam in ("CRC", "HP"):
                for ln, lv in LEVELS:
                    if fam == "CRC" and ln == "op":
                        continue
                    key = (fam, ln)
                    if key in reached or len(rpos) < MIN_POS_REMAIN:
                        continue
                    lam_l, feas = meths[(f"LOC-{fam}", ln)]; lam_s = meths[(f"SRC-{fam}", ln)][0]
                    if feas and D.false_alarm_rate(rneg, lam_l) <= lv and (rpos > lam_l).mean() >= (rpos > lam_s).mean():
                        reached[key] = (m_, mins)
            # calibration: T_src vs T_burn on the remainder
            bmask = bi; rmask = ri & kn_t
            T_burn = C.Temperature().fit(lt[bmask], ev[it][bmask], ob[it][bmask]).T
            yr = y_t[rmask]
            if len(yr) and 0 < yr.sum() < len(yr):
                e_src = C.ece_adaptive(yr, C.risk_from_logits(lt[rmask] / T_src)[:, K3])
                e_brn = C.ece_adaptive(yr, C.risk_from_logits(lt[rmask] / T_burn)[:, K3])
                ece_rows.append(dict(location=t, m=m_, burn_minutes=mins, T_src=T_src, T_burn=T_burn,
                                     ece_src=e_src, ece_burn=e_brn, remain_pos=int(yr.sum())))
                if ece_reached is None and e_brn < e_src:
                    ece_reached = (m_, mins)
        for fam in ("CRC", "HP"):
            for ln, lv in LEVELS:
                if fam == "CRC" and ln == "op":
                    continue
                r_ = reached.get((fam, ln))
                read_rows.append(dict(location=t, n_recordings=len(recs), readout=f"LOC-{fam} beats SRC-{fam}", level=ln,
                                      reached=r_ is not None, m=r_[0] if r_ else np.nan, minutes=r_[1] if r_ else np.nan))
        read_rows.append(dict(location=t, n_recordings=len(recs), readout="T_burn ECE < T_src ECE", level="-",
                              reached=ece_reached is not None, m=ece_reached[0] if ece_reached else np.nan,
                              minutes=ece_reached[1] if ece_reached else np.nan))
        print(f"{t}: {len(recs)} recordings, {sum(rec_minutes.values()):.0f} min | readouts: " +
              ", ".join(f"{f}@{l}={'m=' + str(reached[(f, l)][0]) if (f, l) in reached else 'not reached'}"
                        for f in ("CRC", "HP") for l, _ in LEVELS if not (f == "CRC" and l == "op")) +
              f" | ECE: {'m=' + str(ece_reached[0]) if ece_reached else 'not reached'} | {time.time() - t0:.0f}s", flush=True)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(curve_rows).to_csv("results/p2_b6_curve.csv", index=False)
    pd.DataFrame(ece_rows).to_csv("results/p2_b6_ece.csv", index=False)
    Rd = pd.DataFrame(read_rows); Rd.to_csv("results/p2_b6_readouts.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nREAD-OUTS over locations (minutes of recording in the new location before the local method is better):")
    out = []
    for (ro, ln), g in Rd.groupby(["readout", "level"], sort=False):
        ok = g[g.reached]
        out.append(dict(readout=ro, level=ln, locations_reached=f"{len(ok)}/{len(g)}",
                        median_minutes=ok.minutes.median() if len(ok) else np.nan,
                        min_minutes=ok.minutes.min() if len(ok) else np.nan, max_minutes=ok.minutes.max() if len(ok) else np.nan,
                        median_recordings=ok.m.median() if len(ok) else np.nan))
    print(pd.DataFrame(out).round(1).to_string(index=False))
    print(f"\nSaved results/p2_b6_curve.csv, p2_b6_readouts.csv, p2_b6_ece.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()