"""
EXPLORATORY ANALYSIS E1b, step 1 (post-hoc; plan docs/exploratory_plan_E1_E5.md, written 30 Oct 2026 before this script):
P1 frame scores with the cumulative risk at 5 s instead of 3 s, from the same five registered central models (chosen
head, augmentation on). Inference only: no model is trained or changed.

The code is the registered P1 scoring routine (train_shrp2.p1_scores) with one difference: the risk column. With
column K3 (3 s) it reproduces the registered P1 files exactly (tested); E1b uses column 3 (5 s). Output files carry the
suffix _h5s and are never written over an existing file, so the registered P1 scores cannot be touched.

Registered run only (refuses SHRP2_RUN). Project environment:
    python src/shrp2/explore_e1_scores.py
Output data/processed/shrp2/preds/p1_central_<head>_aug_s<seed>_h5s.parquet   (event_id, frame, split, score)
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train_shrp2 as T

H5_COL = 3                       # interval index of the 5 s horizon: bins (0,1], (1,2], (2,3], (3,5] s
SUFFIX = "_h5s"


def p1_scores_at(names, col, suffix, overwrite=False):
    """train_shrp2.p1_scores with a chosen risk column; writes PRED/p1_<name><suffix>.parquet."""
    import torch
    if not suffix and not overwrite:
        raise SystemExit("refusing to write without a suffix: the registered P1 files must not be replaced")
    T.PRED.mkdir(parents=True, exist_ok=True)
    G = {grp: (np.load(T.CACHE / f"{grp}_hist.npy", mmap_mode="r"), pd.read_parquet(T.CACHE / f"{grp}_meta.parquet"))
         for grp in ("p1_val", "p1_test")}
    written = []
    for name in names:
        out_f = T.PRED / f"p1_{name}{suffix}.parquet"
        if out_f.exists() and not overwrite:
            raise SystemExit(f"{out_f} exists - refusing to overwrite")
        ck = torch.load(T.MODELS / f"haz_{name}.pt", map_location="cpu")
        enc, head = T.InteractionEncoder(), T.HazardHead(64, 4)
        enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
        parts = []
        for grp, (X, m) in G.items():
            lg = T.logits_of(enc, head, lambda j: np.asarray(X[j], np.float32), np.arange(len(m)))
            parts.append(pd.DataFrame(dict(event_id=m["event_id"].to_numpy(), frame=m["frame"].to_numpy(),
                                           split=grp.split("_")[1], score=T.risk_of(lg, ck["head_kind"])[:, col])))
        out = pd.concat(parts, ignore_index=True)
        if out.duplicated(["event_id", "frame"]).any():
            raise SystemExit("P1 samples have duplicate (event_id, frame)")
        out.to_parquet(out_f, index=False); written.append(out_f)
        print(f"  {out_f.name}: {len(out):,} frame scores (risk column {col})", flush=True)
    return written


def main():
    if os.environ.get("SHRP2_RUN", ""):
        raise SystemExit("E1-E5 run on the registered run only: remove SHRP2_RUN (Remove-Item Env:SHRP2_RUN)")
    head = T.chosen_head(False)
    p1_scores_at([f"central_{head}_aug_s{s}" for s in T.SEEDS], H5_COL, SUFFIX)


if __name__ == "__main__":
    main()
