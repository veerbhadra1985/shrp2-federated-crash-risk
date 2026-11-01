"""
SHRP2 SAMPLES AND REGISTERED LABELS for the hazard model (pre-registration osf.io/y5zrp, Sections 3-4).
Produces the same arrays as the development pipeline (src/data/build_samples.py + build_hazard_labels.py),
so the frozen models and training code take them unchanged.

Per sample (subject vehicle only; radar objects are inputs):
  hist (9, 20, 9)  subject (row 0) and up to 8 nearest radar objects within 30 m at t, 2 s at 10 Hz, in the
                   subject's frame at t: x, y, vx, vy, sin(dh), cos(dh), length, width, valid
  fut (30, 2), fut_valid (30,)   subject's next 3 s (pretraining target; censored frames invalid)
  event_bin, obs_bins            registered labels (src/shrp2/labels.py; bins 1/2/3/5 s, censoring)
  current = 0                    excluded frames (t >= event, 5-7 s buffer) are NOT written at all
  scene 'SHRP2/<id>', location = split stratum, agent 'ego', frame = t / 0.1 s, split, event_id,
  outcome_type, tte_ms (event time minus t; -1 for baselines; used by protocol P1)
Prediction times (registered): every frame for validation and test; every 1 s for training, at frames
  first + 19, first + 29, ... (as build_samples.py: fr[H-1::10]). Valid times: src/shrp2/labels.py.
Radar objects: the audit-dependent rules of src/shrp2/objects.py (--object-ref, --object-size) are REQUIRED
  arguments, so no default can slip in before the author freezes them.
Inputs : data/processed/shrp2/traj_*.parquet, trips.parquet (src/loaders/shrp2.py), split.csv (make_split.py)
Outputs: data/processed/shrp2/samples/<split>_<k>.npz (shards of --shard samples) and
         results/shrp2_samples_summary.txt (aggregate counts; per-unit evaluable counts in
         data/processed/shrp2/samples/units.csv for the cross-check against the audit)
Usage  : python src/shrp2/build_samples_shrp2.py --object-ref asgiven --object-size car
         ... --p1   (Amendment 4: inference samples for protocol P1 - validation and test EVENTS, every valid
                     frame incl. after the event and the 5-7 s buffer, current = 1 there; to samples_p1/)
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import labels as LB
import objects as OB
from data.build_samples import to_local

H, F, K, RADIUS, STRIDE = 20, 30, 8, 30.0, 10
KIN = ["x", "y", "vx", "vy", "heading"]


def trip_samples(g, split, stratum, te_ms, outcome, ref, size, p1=False):
    """g: common-format rows of ONE trip (subject 'ego' + radar objects). Returns dict of arrays (or None).
    p1=True (Amendment 4, protocol P1 inference only): every valid prediction time is kept, including frames after
    the event and in the 5-7 s buffer (marked current = 1), with no training stride."""
    ego = g[g["is_ego"]].sort_values("timestamp_ms")
    ts = ego["timestamp_ms"].to_numpy(np.int64)
    f0 = int(ts[0] // 100); nfr = int(ts[-1] // 100) - f0 + 1
    fr = (ts // 100 - f0).astype(np.int64)
    # subject arrays on the full frame range (NaN = missing or masked)
    E = np.full((nfr, 7), np.nan)
    E[fr] = ego[["x", "y", "vx", "vy", "heading", "length", "width"]].to_numpy(float)
    masked = ego["masked"].to_numpy(bool)
    pidx = LB.prediction_index(ts, masked)
    if len(pidx) == 0:
        return None
    tpred = ts[pidx]
    excl, ev, obs = LB.label_arrays(tpred, te_ms, ts[-1])
    keep = np.ones(len(tpred), bool) if p1 else ~excl
    if split == "train" and not p1:
        keep &= ((tpred // 100 - f0) % STRIDE) == (H - 1) % STRIDE
    tpred, ev, obs, excl = tpred[keep], ev[keep], obs[keep], excl[keep]
    m = len(tpred)
    if m == 0:
        return None
    P = (tpred // 100 - f0).astype(np.int64)                       # frame index of each prediction time
    # radar objects -> arrays (n_obj, nfr, 7) after the frozen object rules
    ob = g[~g["is_ego"]]
    ids = np.sort(ob["agent_id"].unique()) if len(ob) else np.array([], dtype=object)
    A = np.full((len(ids), nfr, 7), np.nan)
    for a, aid in enumerate(ids):
        o = ob[ob["agent_id"] == aid]
        fo = (o["timestamp_ms"].to_numpy(np.int64) // 100 - f0)
        inr = (fo >= 0) & (fo < nfr); o, fo = o[inr], fo[inr]
        ex, ey = E[fo, 0], E[fo, 1]
        x, y, Lo, Wo = OB.apply(o["x"].to_numpy(float), o["y"].to_numpy(float), o["heading"].to_numpy(float),
                                ex, ey, ref, size)
        A[a, fo] = np.column_stack([x, y, o["vx"].to_numpy(float), o["vy"].to_numpy(float),
                                    o["heading"].to_numpy(float), Lo, Wo])
    # nearest K objects present at t within RADIUS (ties broken by object id order)
    ox, oy, oh = E[P, 0], E[P, 1], E[P, 4]
    if len(ids):
        d = np.hypot(A[:, P, 0] - ox[None], A[:, P, 1] - oy[None])          # (n_obj, m)
        d = np.where(np.isnan(d) | (d > RADIUS), np.inf, d)
        order = np.argsort(d, axis=0, kind="stable")[:K]                     # (<=K, m)
        dsel = np.take_along_axis(d, order, axis=0)
    else:
        order = np.zeros((0, m), int); dsel = np.zeros((0, m))
    win = P[:, None] + np.arange(-(H - 1), 1)[None]                          # (m, 20)
    hist = np.zeros((m, K + 1, H, 9), np.float32)

    def fill(slot, W7):                                                      # W7: (m, 20, 7)
        ok = ~np.isnan(W7[..., 0])
        x, y, vx, vy, hh = to_local(W7[..., 0], W7[..., 1], W7[..., 2], W7[..., 3], W7[..., 4],
                                    ox[:, None], oy[:, None], oh[:, None])
        feat = np.stack([x, y, vx, vy, np.sin(hh), np.cos(hh), W7[..., 5], W7[..., 6], ok.astype(float)], -1)
        hist[:, slot] = np.where(ok[..., None], np.nan_to_num(feat), 0.0)

    fill(0, E[win])
    for k in range(order.shape[0]):
        present = np.isfinite(dsel[k])                                       # (m,)
        W7 = A[order[k]][np.arange(m)[:, None], win]                        # (m, 20, 7)
        W7[~present] = np.nan
        fill(k + 1, W7)
    fw = P[:, None] + np.arange(1, F + 1)[None]
    inside = fw < nfr
    FW = np.full((m, F, 7), np.nan); FW[inside] = E[fw[inside]]
    fx, fy, _, _, _ = to_local(FW[..., 0], FW[..., 1], FW[..., 2], FW[..., 3], FW[..., 4],
                               ox[:, None], oy[:, None], oh[:, None])
    fval = ~np.isnan(FW[..., 0])
    eid = int(g["event_id"].iat[0])
    return dict(hist=hist, fut=np.nan_to_num(np.stack([fx, fy], -1)).astype(np.float32),
                fut_valid=fval.astype(np.float32), event_bin=ev, obs_bins=obs,
                current=excl.astype(np.int64), frame=(tpred // 100).astype(np.int64),
                tte_ms=(int(te_ms) - tpred) if te_ms is not None else np.full(m, -1, np.int64),
                event_id=np.full(m, eid, np.int64), scene=np.full(m, f"SHRP2/{eid}", dtype=object),
                agent=np.full(m, "ego", dtype=object), location=np.full(m, stratum, dtype=object),
                split=np.full(m, split, dtype=object), outcome_type=np.full(m, outcome, dtype=object))


def main(traj_dir, ref, size, out=None, shard=50000, res_dir="results", p1=False):
    OB.check(ref, size)
    t0 = time.time()
    traj_dir = Path(traj_dir); out = Path(out or traj_dir / ("samples_p1" if p1 else "samples")); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.npz"):
        old.unlink()                                                     # never mix shards of two runs
    T = pd.read_parquet(traj_dir / "trips.parquet").set_index("event_id")
    S = pd.read_csv(traj_dir / "split.csv").set_index("event_id")
    buf = {s: [] for s in ("train", "val", "test")}; nbuf = dict.fromkeys(buf, 0); nshard = dict.fromkeys(buf, 0)
    units = []

    def flush(s, final=False):
        if not buf[s] or (nbuf[s] < shard and not final):
            return
        D = {k: np.concatenate([b[k] for b in buf[s]]) for k in buf[s][0]}
        np.savez_compressed(out / f"{s}_{nshard[s]:03d}.npz", **D)
        nshard[s] += 1; buf[s].clear(); nbuf[s] = 0

    for p in sorted(traj_dir.glob("traj_*.parquet")):
        df = pd.read_parquet(p)
        for eid, g in df.groupby("event_id", sort=True):
            if eid not in S.index:
                continue                                                 # not a split unit
            if p1 and (S.at[eid, "split"] == "train" or T.at[eid, "outcome_type"] == "baseline"):
                continue                                                 # P1 scores validation/test events only
            te = T.at[eid, "event_time_ms"]
            te = None if pd.isna(te) else int(te)
            r = trip_samples(g, S.at[eid, "split"], S.at[eid, "stratum"], te, T.at[eid, "outcome_type"], ref, size, p1)
            n3 = 0
            if r is not None:
                y3, known3 = LB.known_at(r["event_bin"], r["obs_bins"], 2)
                n3 = int((known3 & (r["current"] == 0)).sum())
                s = S.at[eid, "split"]; buf[s].append(r); nbuf[s] += len(r["frame"]); flush(s)
            units.append(dict(event_id=eid, split=S.at[eid, "split"], n_samples=0 if r is None else len(r["frame"]),
                              n_known3=n3, n_pos3=0 if r is None else int(((r["event_bin"] >= 0) & (r["event_bin"] <= 2) & (r["current"] == 0)).sum())))
        print(f"  {p.name} done | {time.time() - t0:.0f}s", flush=True)
    for s in buf:
        flush(s, final=True)
    U = pd.DataFrame(units); U.to_csv(out / "units.csv", index=False)
    lines = ["SHRP2 SAMPLES - aggregate counts only (shareable)", f"object rules: ref={ref}, size={size}; "
             f"train stride {STRIDE} frames, val/test every frame", ""]
    for s in ("train", "val", "test"):
        u = U[U.split == s]
        lines.append(f"{s}: units {len(u)}, samples {int(u.n_samples.sum())}, known at 3 s {int(u.n_known3.sum())}, "
                     f"positive at 3 s {int(u.n_pos3.sum())}, shards {nshard[s]}")
    # sensitivity 8 (C23): a run on the unmasked data (traj_dir *_nomask or SHRP2_RUN=nomask) writes *_nomask summaries,
    # so the registered summaries are never overwritten (process note P9, 26 Oct 2026, before the repetition)
    tag = "_nomask" if (Path(traj_dir).name.endswith("_nomask") or os.environ.get("SHRP2_RUN", "") == "nomask") else ""
    Path(res_dir).mkdir(exist_ok=True)
    Path(res_dir, (f"shrp2_samples_p1_summary{tag}.txt" if p1 else f"shrp2_samples_summary{tag}.txt")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines) + f"\n{time.time() - t0:.0f}s")
    return U


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", default="data/processed/shrp2")
    ap.add_argument("--object-ref", required=True, choices=OB.REFS)
    ap.add_argument("--object-size", required=True, choices=OB.SIZES)
    ap.add_argument("--shard", type=int, default=50000)
    ap.add_argument("--p1", action="store_true", help="Amendment 4: P1 inference samples (val/test events, all frames)")
    a = ap.parse_args()
    main(a.traj_dir, a.object_ref, a.object_size, shard=a.shard, p1=a.p1)
