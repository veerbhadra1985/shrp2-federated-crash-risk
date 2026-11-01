"""
Stage 1, step 10: rehearsal of H3 (federated training) and the FINAL Holm correction over all
nine registered tests. ONE training seed (131), as agreed on 9 Oct 2026 (federated learning was
already shown in Step 13; the rehearsal checks the analysis pipeline, not the effect).

Registered tests (pooled over clients, 3 s horizon, test recordings as bootstrap units):
  H3a   adaptive ECE: FedAvg - FedPer > 0          (FedPer better calibrated)
  H3b1  AUPRC: FedAvg - local > 0
  H3b2  AUPRC: FedPer - local > 0
Clients = the 12 INTERACTION locations (Step 13 design; on SHRP2 the registered clients are
lighting x surface x density cells). Models: python src/train_federated.py --method M
--rounds 30 --local-steps 50 --seed 131 --save-preds --tag h3   (M = local, fedavg, fedper).

Usage : python src/rehearsal/run_h3.py            (B = 2000)
Needs : data/processed/preds/fed_{local,fedavg,fedper}_s131.npz, results/rehearsal_hypotheses.csv
Output: results/rehearsal_h3.csv, results/rehearsal_final_holm.csv
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import analysis.stats as S
import calibration as C
from train_pretrain import split_by_scene

PRED = Path("data/processed/preds")
SEEDS, K3 = [131], 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args()
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    N = len(lab["current"]); keep = lab["current"] == 0
    scene = d["scene"].astype(str)
    _, _, te_k = split_by_scene(scene[keep], d["location"][keep].astype(str))
    te = np.zeros(N, bool); te[keep] = te_k
    y_all, known = C.labels_at(lab["event_bin"], lab["obs_bins"], K3)
    T = np.flatnonzero(te & known); y = y_all[T]
    print(f"test samples at 3 s: {len(T):,} ({y.sum()} positives) in {len(np.unique(scene[T]))} recordings")

    def risk(method, s):
        z = np.load(PRED / f"fed_{method}_s{s}.npz")
        r = np.full(N, np.nan); r[z["idx"]] = z["risk"][:, K3]
        if np.isnan(r[T]).any():
            raise SystemExit(f"fed_{method}_s{s}: predictions missing for some test samples")
        return r[T]

    P = {m: [risk(m, s) for s in SEEDS] for m in ("local", "fedavg", "fedper")}
    for m, v in P.items():
        print(f"  {m:7} pooled AUPRC@3s {S.average_precision(y, v[0]):.3f} | adaptive ECE@3s {S.ece_adaptive(y, v[0]):.4f}")
    boot = S.ClusterBootstrap(scene[T], B=a.B, seed=131)
    rows = []
    for name, metric, A, Bm in (("H3a", S.ece_adaptive, "fedavg", "fedper"),
                                ("H3b1", S.average_precision, "fedavg", "local"),
                                ("H3b2", S.average_precision, "fedper", "local")):
        r = S.paired_test(boot, S.seed_mean_difference(metric, y, P[A], P[Bm]))
        r.update(test=name, family="registered", delta0=0.0, note=f"{A} - {Bm}; one seed")
        rows.append(r)
        print(f"  {name:5} est {r['estimate']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p {r['p']:.4f}")
    h3 = pd.DataFrame(rows); h3.to_csv("results/rehearsal_h3.csv", index=False)

    prev = pd.read_csv("results/rehearsal_hypotheses.csv")
    fam = pd.concat([prev[prev.family == "registered"], h3], ignore_index=True)
    fam = fam[["test", "estimate", "ci_low", "ci_high", "p", "note"]]
    adj, rej = S.holm(fam["p"].to_numpy())
    fam["p_holm"], fam["supported"] = adj, rej
    order = ["H1a", "H1b", "H2", "H3a", "H3b1", "H3b2", "H4a", "H4b", "H4c"]
    fam = fam.set_index("test").loc[order].reset_index()
    fam.to_csv("results/rehearsal_final_holm.csv", index=False)
    print(f"\n=== FINAL Holm over all {len(fam)} registered tests (family-wise alpha 0.05) ===")
    print(fam.round(4).to_string(index=False))
    print("\nSaved results/rehearsal_h3.csv and results/rehearsal_final_holm.csv")


if __name__ == "__main__":
    main()