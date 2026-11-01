"""
Step 16: robustness of the hazard model to degraded NEIGHBOUR information
(mimics SHRP2 forward-radar limitations; the target vehicle's own history is untouched).

Conditions
  drop   p : each neighbour removed entirely with probability p (radar never detects it)
  frames q : each observed neighbour frame removed with probability q (track flicker)
  noise  s : Gaussian noise on neighbour positions (std s metres) and velocities (std s/2 m/s)
Evaluation-only: uses the trained Step 11 model (models/hazard_model.pt) on the project-wide
TEST split; 3 random seeds per condition (mean and std reported).
Usage : python src/robustness.py            (add --seeds 1 for a quick run)
        python src/robustness.py --model models/hazard_model_aug.pt --tag aug
Output: results/robustness.csv
"""
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

HORIZONS = [1, 2, 3, 5]
CONDITIONS = [("none", 0.0)] + [("drop", p) for p in (0.1, 0.25, 0.5, 0.75, 1.0)] + \
             [("frames", q) for q in (0.1, 0.3, 0.5)] + [("noise", s) for s in (0.5, 1.0, 2.0)]


def perturb(hist, kind, level, rng):
    """hist: (n, A, T, 9) numpy array; row 0 = target, rows 1.. = neighbours.
    Feature order: x, y, vx, vy, sin, cos, length, width, valid. Returns a perturbed copy."""
    h = hist.copy()
    nb = h[:, 1:]                                   # view on neighbours
    present = nb[..., 8].sum(-1) > 0                # (n, A-1)
    if kind == "drop":
        gone = present & (rng.random(present.shape) < level)
        nb[gone] = 0.0
    elif kind == "frames":
        valid = nb[..., 8] > 0                      # (n, A-1, T)
        lost = valid & (rng.random(valid.shape) < level)
        nb[lost] = 0.0
    elif kind == "noise":
        valid = nb[..., 8] > 0
        nb[..., 0:2] += rng.normal(0, level, nb[..., 0:2].shape) * valid[..., None]
        nb[..., 2:4] += rng.normal(0, level / 2, nb[..., 2:4].shape) * valid[..., None]
    return h


def main():
    import torch
    from sklearn.metrics import average_precision_score
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.encoder import InteractionEncoder
    from models.hazard import HazardHead, cumulative_risk
    from train_pretrain import split_by_scene
    import calibration as C
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", default="models/hazard_model.pt"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); torch.set_num_threads(os.cpu_count() or 1)
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    keep = lab["current"] == 0
    _, _, te = split_by_scene(d["scene"][keep], d["location"][keep])
    H = d["hist"][keep][te]; ev = lab["event_bin"][keep][te]; ob = lab["obs_bins"][keep][te]
    ck = torch.load(a.model, map_location="cpu")
    print(f"model: {a.model}")
    enc = InteractionEncoder(); enc.load_state_dict(ck["encoder"]); enc.eval()
    head = HazardHead(enc.d_model, len(HORIZONS)); head.load_state_dict(ck["head"]); head.eval()
    n_nb = (H[:, 1:, :, 8].sum(-1) > 0).sum(1)
    print(f"test samples {len(H):,} | mean neighbours per sample {n_nb.mean():.2f} | "
          f"samples with >=1 neighbour {np.mean(n_nb > 0):.1%}")

    def score(h):
        out = []
        with torch.no_grad():
            for s in range(0, len(h), 1024):
                out.append(cumulative_risk(head(enc(torch.from_numpy(h[s:s + 1024])))))
        return torch.cat(out).numpy()

    rows = []
    for kind, level in CONDITIONS:
        res = {f"AUPRC_{h}s": [] for h in HORIZONS}; res["ECE_3s"] = []; res["mean_pred_3s"] = []
        for seed in range(1 if kind == "none" else a.seeds):
            r = score(perturb(H, kind, level, np.random.default_rng(1000 + seed)))
            for k, h in enumerate(HORIZONS):
                y, ok = C.labels_at(ev, ob, k)
                res[f"AUPRC_{h}s"].append(average_precision_score(y[ok], r[ok, k]))
                if h == 3:
                    res["ECE_3s"].append(C.ece(y[ok], r[ok, k])); res["mean_pred_3s"].append(r[ok, k].mean())
        row = {"condition": kind, "level": level}
        for k_, v in res.items():
            row[k_] = float(np.mean(v)); row[k_ + "_sd"] = float(np.std(v))
        rows.append(row)
        print(f"{kind:7} {level:5} | AUPRC 1s {row['AUPRC_1s']:.3f} 3s {row['AUPRC_3s']:.3f} "
              f"(sd {row['AUPRC_3s_sd']:.3f}) 5s {row['AUPRC_5s']:.3f} | ECE 3s {row['ECE_3s']:.4f} "
              f"| mean pred 3s {row['mean_pred_3s']:.2%}", flush=True)
    base = rows[0]["AUPRC_3s"]
    print("\nRelative AUPRC@3s vs clean data:")
    for r in rows[1:]:
        print(f"  {r['condition']:7} {r['level']:5}: {r['AUPRC_3s'] / base:6.1%}")
    Path("results").mkdir(exist_ok=True)
    out = f"results/robustness{('_' + a.tag) if a.tag else ''}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()