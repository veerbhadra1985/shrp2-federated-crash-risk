"""
Phase 2, step B7 - DIAGNOSTIC for comparator C3 (Humbert et al., 2023), written 11 Oct 2026 AFTER the B7 result
showed C3 far above its target on INTERACTION (setting A: mean FA 0.376 at a = 0.05). Purpose: decide whether this
is (i) a real property of C3 under client heterogeneity (the paper's Prop. 3.4 warns that coverage drops under
heterogeneity) or (ii) an implementation error. Post-hoc check of OUR implementation; not a new result about our method.

Design (setting A, seed 131, leave one recording out, a in {0.05, 0.10}):
  ACTUAL    C3 on the real per-recording negatives (as in B7)
  SHUFFLED  the same pooled negatives randomly re-assigned to recordings, keeping every recording's size,
            repeated for 20 independent shuffles (seeds 131..150) -> clients become i.i.d. by construction, which is
            exactly C3's assumption. 20 shuffles are needed because C3's threshold rests on a few order statistics
            and varies strongly from one data realisation to the next (one realisation can sit ~0.017 above or below
            its expectation; checked by Monte Carlo on 11 Oct 2026 before this run: fresh i.i.d. clients with the same
            size spread gave mean FA 0.047 at a = 0.05).
DECISION RULE (fixed here, before running on the real data): the implementation is accepted, and the B7 C3 result
attributed to heterogeneity, if the SHUFFLED mean FA (averaged over the 20 shuffles) <= a + 3 SE (SE over shuffles)
for both levels, while ACTUAL is far above a. Otherwise the implementation is suspect and C3 is withdrawn from B7.
Also printed: the chosen (l, k), M_lk, and client sizes, for transparency.
Usage : python src/phase2/b7_c3_diagnostic.py      (about 10-20 min)
Output: results/p2_b7_c3_diagnostic.csv
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import calibration as C
from phase2 import comparators as K

PRED, K3, MIN_EVAL = Path("data/processed/preds"), 2, 100


def main():
    t0 = time.time()
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene = d["scene"].astype(str)
    y, kn = C.labels_at(lab["event_bin"], lab["obs_bins"], K3)
    z = np.load(PRED / "haz_central_aug_s131.npz"); idx = z["idx"]; r = z["risk"][:, K3].astype(float)
    sc, k_, y_ = scene[idx], kn[idx], y[idx]
    negs = {}
    for u in np.unique(sc):
        m = (sc == u) & k_
        if m.sum() >= MIN_EVAL and (m & (y_ == 0)).any():
            negs[u] = r[m & (y_ == 0)]
    ids = sorted(negs); sizes = [len(negs[u]) for u in ids]
    pooled = np.concatenate([negs[u] for u in ids])
    print(f"{len(ids)} clients | negatives per client: min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}")
    rows = []
    def run(D_, name, a, rep):
        fas = []
        for u in ids:
            res = K.c3_humbert_qq([D_[v] for v in ids if v != u], a)
            fa = float(np.mean(D_[u] > res["lam"])); fas.append(fa)
            rows.append(dict(alpha=a, version=name, rep=rep, held_out=u, l=res["l"], k=res["k"], M=res["M"],
                             lam=res["lam"], fa=fa))
        return float(np.mean(fas))
    summary = []
    for a in (0.05, 0.10):
        act = run(negs, "ACTUAL", a, 0)
        sh = []
        for rep in range(20):
            perm = np.random.default_rng(131 + rep).permutation(pooled); D_, pos = {}, 0
            for u, n in zip(ids, sizes):
                D_[u] = perm[pos:pos + n]; pos += n
            sh.append(run(D_, "SHUFFLED", a, rep))
        m, se = float(np.mean(sh)), float(np.std(sh, ddof=1) / np.sqrt(len(sh)))
        summary.append(dict(alpha=a, actual_mean_FA=act, shuffled_mean_FA=m, shuffled_SE=se,
                            shuffled_min=min(sh), shuffled_max=max(sh), ok=m <= a + 3 * se))
        print(f"  a={a:.2f}: ACTUAL mean FA {act:.4f} | SHUFFLED mean FA {m:.4f} (SE {se:.4f}; range {min(sh):.4f}-{max(sh):.4f}) "
              f"| (l, k) = ({rows[-1]['l']}, {rows[-1]['k']}), M = {rows[-1]['M']:.4f} | {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows); Path("results").mkdir(exist_ok=True); R.to_csv("results/p2_b7_c3_diagnostic.csv", index=False)
    S = pd.DataFrame(summary); print("\n" + S.round(4).to_string(index=False))
    ok = bool(S.ok.all())
    print(f"=> {'IMPLEMENTATION ACCEPTED: C3 is valid when clients are i.i.d.; the B7 failure is due to client heterogeneity' if ok else 'IMPLEMENTATION SUSPECT: withdraw C3 from B7 and investigate'}")
    print(f"Saved results/p2_b7_c3_diagnostic.csv | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()