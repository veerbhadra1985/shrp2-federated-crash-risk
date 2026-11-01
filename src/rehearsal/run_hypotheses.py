"""
Stage 1, step 9: the confirmatory analysis of the pre-registration, rehearsed on INTERACTION
(stand-in labels, 3 s horizon, protocol P2). Uses ONLY the analysis engine (src/analysis/stats.py)
and the saved predictions of steps 6 and 8. Nothing is tuned here.

Registered tests (one-sided; H3 follows separately, then Holm is recomputed over all 9):
  H1a  AUPRC: model - GSSM          (rehearsal stand-in for GSSM: the registered LightGBM baseline)
  H1b  AUPRC: model - best classical measure (TTC, DRAC, TTC2D, TTC2D-moving; chosen on VALIDATION)
  H2   mean over testable factor-direction pairs of log( ECE(model trained elsewhere) /
       ECE(model trained in-domain) ), adaptive ECE, same held-out test data
  H4a  AUPRC: augmented - non-augmented, on test times with natural neighbour dropout
       (an object present in the 2 s window missing in >= 20% of its frames; registered definition)
  H4b  AUPRC: augmented - non-augmented, under 50% synthetic frame flicker
  H4c  clean AUPRC: augmented - non-augmented > -0.02 (non-inferiority)
Also reported (not in the registered family):
  H2-matched   H2 with the larger level's model replaced by its size-matched model (amendment 1 candidate)
  H2-temp      H2 after temperature recalibration on each model's own validation data (sensitivity 6)
  H2-secondary conformal false-alarm thresholds (alpha 0.05, 0.01) from each model's own validation
               negatives, applied to the held-out level: violation by the registered frame-level
               Clopper-Pearson rule AND by the cluster-bootstrap interval (amendment 1 candidate)
Units: test recordings (cluster bootstrap, B = 2000, seed 131; same resamples for every test).

Usage : python src/rehearsal/run_hypotheses.py            (B = 2000, about 10-20 min)
        python src/rehearsal/run_hypotheses.py --B 200    (quick check)
Needs : data/processed/preds/haz_*.npz and lgbm_s*.npz, models/rehearsal/haz_central_*.pt (H4b),
        results/rehearsal_factors_by_recording.csv
Output: results/rehearsal_hypotheses.csv, results/rehearsal_h2_pairs.csv, results/rehearsal_conformal.csv
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import analysis.stats as S
import calibration as C
from train_pretrain import split_by_scene
from models.encoder import InteractionEncoder
from models.hazard import HazardHead
from robustness import perturb

PRED, ENC = Path("data/processed/preds"), Path("models/rehearsal")
FACTOR_FILE = Path("results/rehearsal_factors_by_recording.csv")
SEEDS = [131, 132, 133, 134, 135]
K3, MIN_EVENTS, ALPHAS, DROP_SHARE = 2, 50, (0.05, 0.01), 0.2


def load_pred(name, N):
    z = np.load(PRED / f"{name}.npz")
    risk = np.full((N, 4), np.nan); risk[z["idx"]] = z["risk"]
    logit = None
    if "logits" in z.files:
        logit = np.full((N, 4), np.nan); logit[z["idx"]] = z["logits"]
    return risk, logit


def classical_scores(d, keep, N):
    from rehearsal.lgbm_baseline import build_features
    X = build_features(d, keep)
    full = {}
    for name, col, sign, fill in [("TTC", "ttc", -1, 1e3), ("DRAC", "drac", 1, 0.0),
                                  ("TTC2D", "ttc2d", -1, 1e3), ("TTC2D_moving", "ttc2d_mov", -1, 1e3)]:
        v = np.full(N, np.nan); v[np.flatnonzero(keep)] = sign * np.nan_to_num(X[col].to_numpy(), nan=fill)
        full[name] = v
    return full


def natural_dropout(hist):
    nb = hist[:, 1:, :, 8] > 0                                   # (n, 8, 20) neighbour valid flags
    present = nb.any(-1)
    missing_share = 1 - nb.mean(-1)
    return (present & (missing_share >= DROP_SHARE)).any(-1)


def flicker_risk(hist_test, names):
    """Cumulative risk at 3 s under 50% frame flicker; the SAME perturbed inputs for every model."""
    Xp = torch.from_numpy(perturb(hist_test, "frames", 0.5, np.random.default_rng(131)))
    out = {}
    for n in names:
        f = ENC / f"haz_{n}.pt"
        if not f.exists():
            raise SystemExit(f"missing {f} - re-run: python src/rehearsal/run_jobs.py --budget registered --only central")
        ck = torch.load(f, map_location="cpu")
        enc, head = InteractionEncoder(), HazardHead(64, 4)
        enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
        with torch.no_grad():
            lg = torch.cat([head(enc(Xp[s:s + 1024])) for s in range(0, len(Xp), 1024)]).numpy()
        out[n] = C.risk_from_logits(lg)[:, K3]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--min-events", type=int, default=MIN_EVENTS, help="registered: 50 (change only for code tests)")
    a = ap.parse_args()
    t0 = time.time(); torch.set_num_threads(4)
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    N = len(lab["current"]); keep = lab["current"] == 0
    scene = d["scene"].astype(str); ev, ob = lab["event_bin"], lab["obs_bins"]
    tr_k, va_k, te_k = split_by_scene(scene[keep], d["location"][keep].astype(str))
    tr, va, te = (np.zeros(N, bool) for _ in range(3))
    tr[keep], va[keep], te[keep] = tr_k, va_k, te_k
    y_all, known_all = C.labels_at(ev, ob, K3)
    fac = pd.read_csv(FACTOR_FILE).set_index("scene_id")
    factors = [c for c in fac.columns if c[:2] in ("F1", "F2", "F3", "F4")]
    level = {f: fac.loc[scene, f].to_numpy() for f in factors}

    # ---- test space: not-in-conflict TEST samples whose 3 s outcome is known
    T = np.flatnonzero(te & known_all); y = y_all[T]; cl = scene[T]
    V = np.flatnonzero(va & known_all)
    print(f"test samples at 3 s: {len(T):,} ({y.sum()} positives) in {len(np.unique(cl))} recordings; "
          f"validation {len(V):,}")
    boot = S.ClusterBootstrap(cl, B=a.B, seed=131)

    # ---- predictions
    R = lambda name: load_pred(name, N)[0][:, K3]
    aug = [R(f"haz_central_aug_s{s}")[T] for s in SEEDS]
    noaug = [R(f"haz_central_noaug_s{s}")[T] for s in SEEDS]
    lgbm = [R(f"lgbm_s{s}")[T] for s in SEEDS]
    if any(np.isnan(x).any() for x in aug + noaug + lgbm):
        raise SystemExit("missing predictions on test samples")
    cls = classical_scores(d, keep, N)
    val_auprc = {k: S.average_precision(y_all[V], v[V]) for k, v in cls.items()}
    best = max(val_auprc, key=val_auprc.get)
    print("classical measures, validation AUPRC@3s: " + ", ".join(f"{k} {v:.3f}" for k, v in val_auprc.items())
          + f" -> best = {best}")
    best_cls = [cls[best][T]] * len(SEEDS)

    rows = []
    def add(name, family, stat, delta0=0.0, note=""):
        r = S.paired_test(boot, stat, delta0); r.update(test=name, family=family, delta0=delta0, note=note)
        rows.append(r); print(f"  {name:13} est {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                              f"p {r['p']:.4f}" + (" UNRELIABLE" if r["unreliable"] else ""), flush=True)

    AP = S.average_precision
    print("\nH1 / H4 (AUPRC at 3 s)")
    add("H1a", "registered", S.seed_mean_difference(AP, y, aug, lgbm), note="GSSM stand-in = LightGBM")
    add("H1b", "registered", S.seed_mean_difference(AP, y, aug, best_cls), note=f"best classical = {best}")
    H = d["hist"][T]                                               # test histories, loaded once
    drop = natural_dropout(H); print(f"  natural-dropout subset: {drop.sum():,} samples, {y[drop].sum()} positives")
    sub = lambda arrs: [x[drop] for x in arrs]
    boot_drop = S.ClusterBootstrap(cl[drop], B=a.B, seed=131)
    r = S.paired_test(boot_drop, S.seed_mean_difference(AP, y[drop], sub(aug), sub(noaug)))
    r.update(test="H4a", family="registered", delta0=0.0, note=f"natural dropout subset n={drop.sum()}"); rows.append(r)
    print(f"  {'H4a':13} est {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p {r['p']:.4f}")
    fl = flicker_risk(H, [f"central_aug_s{s}" for s in SEEDS] + [f"central_noaug_s{s}" for s in SEEDS])
    add("H4b", "registered", S.seed_mean_difference(AP, y, [fl[f"central_aug_s{s}"] for s in SEEDS],
                                                    [fl[f"central_noaug_s{s}"] for s in SEEDS]), note="50% frame flicker")
    add("H4c", "registered", S.seed_mean_difference(AP, y, aug, noaug), delta0=-0.02, note="non-inferiority")

    # ---- H2
    print("\nH2 (adaptive ECE at 3 s, same held-out test data)")
    dom = {(f, v): [load_pred(f"haz_dom_{f}_{v}_s{s}", N) for s in SEEDS] for f in factors for v in (0, 1)}
    match = {f: [load_pred(f"haz_match_{f}_s{s}", N) for s in SEEDS] for f in factors}
    ntrain = {(f, v): int((tr & (level[f] == v)).sum()) for f in factors for v in (0, 1)}
    big = {f: 0 if ntrain[(f, 0)] >= ntrain[(f, 1)] else 1 for f in factors}

    def temp_scaled(pred, f, v):
        risk, logit = pred
        vm = np.flatnonzero(va & (level[f] == v))
        T_ = C.Temperature().fit(logit[vm], ev[vm], ob[vm]).T
        out = np.full(N, np.nan); ok = np.isfinite(logit[:, 0])
        out[ok] = C.risk_from_logits(logit[ok] / T_)[:, K3]
        return out

    pair_rows, pairs = [], {"registered": [], "matched": [], "temperature": []}
    for f in factors:
        for a_, b_ in ((0, 1), (1, 0)):
            m = level[f][T] == b_
            npos = int(y[m].sum())
            if m.sum() == 0 or (m & (y == 0)).sum() == 0:
                pair_rows.append(dict(factor=f, direction=f"{a_}->{b_}", testable=False, test_samples=int(m.sum()),
                                      test_positives=npos))
                print(f"  {f} {a_}->{b_}: no held-out test data - not testable"); continue
            testable = npos >= a.min_events
            out_reg = [p[0][:, K3][T] for p in dom[(f, a_)]]; in_reg = [p[0][:, K3][T] for p in dom[(f, b_)]]
            out_m = [p[0][:, K3][T] for p in (match[f] if a_ == big[f] else dom[(f, a_)])]
            in_m = [p[0][:, K3][T] for p in (match[f] if b_ == big[f] else dom[(f, b_)])]
            out_t = [temp_scaled(p, f, a_)[T] for p in dom[(f, a_)]]; in_t = [temp_scaled(p, f, b_)[T] for p in dom[(f, b_)]]
            ece = lambda arrs: float(np.mean([S.ece_adaptive(y[m], x[m]) for x in arrs]))
            prev_src = float(y_all[tr & (level[f] == a_) & known_all].mean())
            pair_rows.append(dict(factor=f, direction=f"{a_}->{b_}", testable=testable,
                                  test_recordings=len(np.unique(cl[m])), test_samples=int(m.sum()),
                                  test_positives=npos, prev_heldout=float(y[m].mean()), prev_source_train=prev_src,
                                  mean_pred_out=float(np.mean([x[m].mean() for x in out_reg])),
                                  ece_out=ece(out_reg), ece_in=ece(in_reg), log_ratio=np.log(ece(out_reg) / ece(in_reg)),
                                  ece_out_matched=ece(out_m), ece_in_matched=ece(in_m),
                                  log_ratio_matched=np.log(ece(out_m) / ece(in_m)),
                                  log_ratio_temperature=np.log(ece(out_t) / ece(in_t))))
            if testable:
                for key, o, i in (("registered", out_reg, in_reg), ("matched", out_m, in_m), ("temperature", out_t, in_t)):
                    pairs[key].append(dict(y=y, mask=m, out=o, inn=i))
            # secondary: conformal thresholds from model a's own validation negatives
            vneg = np.flatnonzero(va & known_all & (level[f] == a_) & ~y_all.astype(bool))
            tneg = m & (y == 0)
            for al in ALPHAS:
                for s_i, s in enumerate(SEEDS):
                    tau = C.conformal_threshold(dom[(f, a_)][s_i][0][vneg, K3], al)
                    sc = out_reg[s_i]; k_ = int((sc[tneg] > tau).sum()); n_ = int(tneg.sum())
                    cp_lo = S.clopper_pearson(k_, n_)[0]
                    est, lo, hi = S.false_alarm_interval_cluster(boot, sc, tau, tneg)
                    conf_rows.append(dict(factor=f, direction=f"{a_}->{b_}", alpha=al, seed=s, false_alarm=est,
                                          cp_lower=cp_lo, cluster_lower=lo, cluster_upper=hi,
                                          violation_cp=cp_lo > al, violation_cluster=lo > al))
    pr = pd.DataFrame(pair_rows)
    print(pr[["factor", "direction", "testable", "test_recordings", "test_positives", "prev_heldout", "prev_source_train",
              "log_ratio", "log_ratio_matched", "log_ratio_temperature"]].round(4).to_string(index=False))
    print(f"  testable pairs: {int(pr.testable.sum())} of 8 (registered minimum 4)")
    if len(pairs["registered"]) >= 4:
        add("H2", "registered", S.mean_log_ratio(pairs["registered"]))
        add("H2_matched", "amendment", S.mean_log_ratio(pairs["matched"]), note="size-matched (amendment 1 candidate)")
        add("H2_temp", "sensitivity", S.mean_log_ratio(pairs["temperature"]), note="after temperature recalibration")
    else:
        rows.append(dict(test="H2", family="registered", estimate=np.nan, p=np.nan, note="not testable (< 4 pairs)"))
        print("  H2 not testable (fewer than 4 testable pairs)")

    # ---- Holm over the registered tests available so far
    res = pd.DataFrame(rows)
    reg = res.family == "registered"
    adj, rej = S.holm(res.loc[reg, "p"].to_numpy())
    res.loc[reg, "p_holm_partial"] = adj; res.loc[reg, "supported_partial"] = rej
    cf = pd.DataFrame(conf_rows)
    for p_, df_ in (("results/rehearsal_hypotheses.csv", res), ("results/rehearsal_h2_pairs.csv", pr),
                    ("results/rehearsal_conformal.csv", cf)):
        df_.to_csv(p_, index=False)

    print("\n=== Summary (Holm over the 6 registered tests available; recomputed with H3 later) ===")
    print(res[["test", "family", "estimate", "ci_low", "ci_high", "p", "p_holm_partial", "supported_partial",
               "discarded", "note"]].round(4).to_string(index=False))
    testable_dirs = set(pr.loc[pr.testable, "factor"] + pr.loc[pr.testable, "direction"])
    cft = cf[(cf.factor + cf.direction).isin(testable_dirs)]
    print("\nH2 secondary - held-out pairs flagged as violating the false-alarm guarantee "
          "(mean over 5 seeds, testable pairs):")
    print(cft.groupby("alpha")[["false_alarm", "violation_cp", "violation_cluster"]].agg(
        {"false_alarm": "mean", "violation_cp": "sum", "violation_cluster": "sum"}).rename(
        columns={"violation_cp": "flags_clopper_pearson", "violation_cluster": "flags_cluster"}).assign(
        pair_seed_combinations=len(cft) // len(ALPHAS)).to_string())
    print("  note: a cluster interval from only 5-9 held-out recordings is itself approximate (few clusters);"
          " on SHRP2 the clusters are drivers/events (hundreds).")
    print(f"\nSaved results/rehearsal_hypotheses.csv, rehearsal_h2_pairs.csv, rehearsal_conformal.csv "
          f"| {time.time() - t0:.0f} s")


conf_rows = []

if __name__ == "__main__":
    main()