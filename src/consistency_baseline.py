"""
Step 12 (ablation, part 2): post-hoc horizon consistency for separate per-horizon classifiers.

Reviewer question: "why not train independent classifiers and force consistency afterwards?"
Compares, on the TEST split:
  hazard       full model: consistent by construction (models/hazard_model_aug.pt)
  separate     independent sigmoid per horizon, no fix (models/hazard_model_aug_sep.pt)
  sep_cummax   separate + running maximum across horizons
  sep_pav      separate + isotonic projection across horizons (closest non-decreasing sequence)
Metrics per horizon: AUPRC, Brier, ECE, mean prediction; plus share of inconsistent samples.
Uncertainty: paired bootstrap over RECORDINGS (samples in one recording are correlated) gives
a 95% CI for each variant's AUPRC difference from the hazard model.
Usage : python src/consistency_baseline.py
Output: results/consistency_baseline.csv
"""
import csv, os, sys
from pathlib import Path
import numpy as np

HORIZONS = [1, 2, 3, 5]


def cummax_fix(risk):
    return np.maximum.accumulate(risk, axis=1)


def pav_fix(risk):
    """Per-sample pool-adjacent-violators: L2-closest non-decreasing sequence."""
    out = risk.copy()
    bad = np.where((np.diff(risk, axis=1) < 0).any(1))[0]
    for i in bad:
        vals, wts = [], []
        for v in risk[i]:
            vals.append(float(v)); wts.append(1.0)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]; m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
                vals[-2:] = [m]; wts[-2:] = [w]
        out[i] = np.repeat(vals, np.array(wts, int))
    return out


def cluster_bootstrap_delta(y, base, other, groups, metric, B=1000, seed=131):
    """Paired bootstrap over RECORDINGS (clusters): resample recordings with replacement,
    compute metric(other) - metric(base) on the same resample. Returns (delta, lo95, hi95, P(delta>0))."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in uniq}
    point = metric(y, other) - metric(y, base); deltas = []
    for _ in range(B):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if 0 < y[take].sum() < len(take):
            deltas.append(metric(y[take], other[take]) - metric(y[take], base[take]))
    deltas = np.array(deltas)
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float((deltas > 0).mean())


def main():
    import torch
    from sklearn.metrics import average_precision_score
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.encoder import InteractionEncoder
    from models.hazard import HazardHead, cumulative_risk
    from train_pretrain import split_by_scene
    import calibration as C
    torch.set_num_threads(os.cpu_count() or 1)
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz"); keep = lab["current"] == 0
    _, _, te = split_by_scene(d["scene"][keep], d["location"][keep])
    X = torch.from_numpy(d["hist"][keep][te]); ev = lab["event_bin"][keep][te]; ob = lab["obs_bins"][keep][te]
    rec = d["scene"][keep][te]
    print(f"test: {len(X):,} samples from {len(np.unique(rec))} recordings")

    def scores(path, to_risk):
        ck = torch.load(path, map_location="cpu")
        enc = InteractionEncoder(); enc.load_state_dict(ck["encoder"]); enc.eval()
        head = HazardHead(enc.d_model, len(HORIZONS)); head.load_state_dict(ck["head"]); head.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 1024):
                out.append(to_risk(head(enc(X[s:s + 1024]))))
        return torch.cat(out).numpy()

    hazard = scores("models/hazard_model_aug.pt", cumulative_risk)
    sep = scores("models/hazard_model_aug_sep.pt", torch.sigmoid)
    variants = {"hazard": hazard, "separate": sep, "sep_cummax": cummax_fix(sep), "sep_pav": pav_fix(sep)}
    rows = []
    print(f"{'variant':11} {'inconsistent':>12} | " + " | ".join(f"AUPRC {h}s" for h in HORIZONS) + " | Brier 3s | ECE 3s | mean pred 3s")
    for name, r in variants.items():
        incon = float(((-np.diff(r, axis=1)) > 1e-7).any(1).mean())
        vals = []
        for k, h in enumerate(HORIZONS):
            y, ok = C.labels_at(ev, ob, k); yk, pk = y[ok], r[ok, k]
            row = dict(variant=name, horizon_s=h, inconsistent_share=incon, AUPRC=average_precision_score(yk, pk),
                       Brier=float(np.mean((pk - yk) ** 2)), ECE=C.ece(yk, pk), mean_pred=float(pk.mean()), prevalence=float(yk.mean()))
            rows.append(row); vals.append(row)
        print(f"{name:11} {incon:12.2%} | " + " | ".join(f"{v['AUPRC']:9.3f}" for v in vals)
              + f" | {vals[2]['Brier']:.4f}   | {vals[2]['ECE']:.4f} | {vals[2]['mean_pred']:.2%}")
    print("\nPaired bootstrap over recordings (1,000 resamples): AUPRC difference vs hazard model")
    print(f"{'variant':11} {'h':>3} {'delta':>7} {'95% CI':>18} {'P(better)':>9}")
    for name in ("separate", "sep_cummax", "sep_pav"):
        for k, h in enumerate(HORIZONS):
            y, ok = C.labels_at(ev, ob, k)
            dlt, lo, hi, pb = cluster_bootstrap_delta(y[ok], variants["hazard"][ok, k], variants[name][ok, k],
                                                      rec[ok], average_precision_score)
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"{name:11} {h:>2}s {dlt:+7.3f} [{lo:+.3f}, {hi:+.3f}] {pb:9.2f}{sig}")
            for r in rows:
                if r["variant"] == name and r["horizon_s"] == h:
                    r.update(dAUPRC=dlt, dAUPRC_lo95=lo, dAUPRC_hi95=hi, P_better=pb)
    print("  * = 95% CI excludes zero")
    Path("results").mkdir(exist_ok=True)
    with open("results/consistency_baseline.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("Saved results/consistency_baseline.csv")


if __name__ == "__main__":
    main()