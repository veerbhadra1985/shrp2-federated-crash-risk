"""
Phase 2, step B5 (spec docs/phase2_method_spec_B2.md v1.1, Section 6): score ALL highD samples with the
INTERACTION central hazard models (augmentation on, seeds 131-135; weights saved by src/rehearsal/run_jobs.py).
The models never saw highD (pretrained and fine-tuned on INTERACTION training recordings only), so every highD
recording is unseen. No training, no tuning: inference only.

Usage : python src/highd/score_highd.py            (resumable per seed; about 5-20 min per seed on a laptop CPU)
        python src/highd/score_highd.py --max 5000 (quick code check on the first 5,000 samples; *_test files)
Needs : data/processed/highd_samples.npz, models/rehearsal/haz_central_aug_s13{1..5}.pt
Output: data/processed/preds/highd_central_aug_s<seed>.npz  (logits (N,4) float32, risk (N,4) float32)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from models.encoder import InteractionEncoder
from models.hazard import HazardHead
import calibration as C

ENC, PRED = Path("models/rehearsal"), Path("data/processed/preds")
SEEDS, BATCH = [131, 132, 133, 134, 135], 2048


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=None); a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1); PRED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = np.load("data/processed/highd_samples.npz", allow_pickle=True)
    X = torch.from_numpy(np.ascontiguousarray(d["hist"][: a.max] if a.max else d["hist"]).astype(np.float32, copy=False))
    tag = "_test" if a.max else ""
    print(f"highD samples {len(X):,} | input shape {tuple(X.shape)} | loaded in {time.time() - t0:.0f}s", flush=True)
    for s in SEEDS:
        out = PRED / f"highd_central_aug_s{s}{tag}.npz"
        if out.exists():
            print(f"  seed {s}: already done"); continue
        f = ENC / f"haz_central_aug_s{s}.pt"
        if not f.exists():
            raise SystemExit(f"missing {f} - the central weights from run_jobs.py are needed")
        ck = torch.load(f, map_location="cpu")
        enc, head = InteractionEncoder(), HazardHead(64, 4)
        enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"]); enc.eval(); head.eval()
        t1 = time.time(); L = []
        with torch.no_grad():
            for i in range(0, len(X), BATCH):
                L.append(head(enc(X[i:i + BATCH])).numpy())
        lg = np.concatenate(L)
        r = C.risk_from_logits(lg)
        assert np.isfinite(lg).all() and (np.diff(r, axis=1) >= -1e-9).all(), "non-finite logits or non-monotone risk"
        np.savez_compressed(out, logits=lg.astype(np.float32), risk=r.astype(np.float32, copy=False))
        print(f"  seed {s}: mean risk@3s {r[:, 2].mean():.4%} | max {r[:, 2].max():.3f} | {time.time() - t1:.0f}s -> {out}", flush=True)
    print(f"done | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()