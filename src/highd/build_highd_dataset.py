"""
highD crash-risk dataset (Phase 2 cross-domain testing; fallback paper). Same code and same
definitions as INTERACTION wherever possible, so the two datasets are directly comparable:
  surrogate safety measures   src/baselines/ssm.py   compute()          (unchanged)
  samples                     src/data/build_samples.py scene_samples()  (2 s history, 8 neighbours
                                                                          within 30 m, 3 s future, 1 s stride)
  hazard labels               src/data/build_hazard_labels.py label_vectorized() (bins 1/2/3/5 s, censoring)

LABEL THRESHOLD RULE (fixed before any highD label is computed): conflicts are far rarer on
motorways (Step 20a). Labels are built for BOTH moving-pair TTC2D < 1.5 s (as INTERACTION) and
TTC2D < 3.0 s (a common motorway threshold). PRIMARY = 1.5 s if the highD samples contain at
least 500 positives at 3 s among samples not in conflict at t; otherwise PRIMARY = 3.0 s.
Both label files are kept and the choice is printed and saved.

Stages (each resumable; run in order):
  python src/highd/build_highd_dataset.py --stage ssm       # per recording, hours in total
  python src/highd/build_highd_dataset.py --stage samples   # about 200,000 samples
  python src/highd/build_highd_dataset.py --stage labels    # both thresholds + primary choice
Add --max-recordings 2 for a quick test (outputs get a _test suffix).
Inputs : data/processed/highd/highd_XX.parquet (src/loaders/highd.py)
Outputs: data/processed/highd_ssm/highd_XX.parquet, data/processed/highd_samples.npz,
         data/processed/highd_labels_ttc15.npz, data/processed/highd_labels_ttc30.npz,
         results/highd_dataset_summary.csv
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "baselines")); sys.path.insert(0, str(SRC / "data"))
import ssm as SSM
import build_samples as BS
import build_hazard_labels as BHL

IN_DIR, SSM_DIR = Path("data/processed/highd"), Path("data/processed/highd_ssm")
MAX_SAMPLES, SEED, MIN_POS, HZ = 200_000, 131, 500, 10


def recordings(maxr):
    files = sorted(IN_DIR.glob("highd_*.parquet"))
    if not files:
        raise SystemExit(f"no highD files in {IN_DIR} - run src/loaders/highd.py first")
    return files[:maxr] if maxr else files


def stage_ssm(files):
    SSM_DIR.mkdir(parents=True, exist_ok=True); t0 = time.time()
    for k, f in enumerate(files, 1):
        out = SSM_DIR / f.name
        if out.exists():
            print(f"  [{k}/{len(files)}] {f.name} already done"); continue
        df = pd.read_parquet(f, columns=["dataset", "scene_id", "time_s", "agent_id", "x", "y", "vx", "vy",
                                          "heading", "length", "width"])
        res = SSM.compute(df)
        res.to_parquet(out, index=False)
        print(f"  [{k}/{len(files)}] {f.name}: {len(res):,} rows | TTC2D-moving < 1.5 s {(res.ttc2d_mov < 1.5).mean():.4%}"
              f" | < 3.0 s {(res.ttc2d_mov < 3.0).mean():.4%} | {time.time() - t0:.0f}s", flush=True)


def stage_samples(files, tag):
    rng = np.random.default_rng(SEED); per = int(np.ceil(MAX_SAMPLES / len(files)))
    H_, F_, FV, SC, AG, FR = [], [], [], [], [], []
    for k, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=["scene_id", "time_s", "agent_id", "x", "y", "vx", "vy",
                                          "heading", "length", "width"])
        res = BS.scene_samples(df, rng, per)
        sc = df["scene_id"].iloc[0]
        for h, fu, v, a, t in res:
            H_.append(h); F_.append(fu); FV.append(v); SC.append(sc); AG.append(str(a)); FR.append(t)
        print(f"  [{k}/{len(files)}] {sc}: {len(res)} samples", flush=True)
    sc = np.array(SC); loc = np.array([s.split("/")[0] for s in SC])
    out = Path(f"data/processed/highd_samples{tag}.npz")
    np.savez_compressed(out, hist=np.stack(H_), fut=np.stack(F_), fut_valid=np.stack(FV), scene=sc, location=loc,
                        agent=np.array(AG), frame=np.array(FR, np.int64), H=BS.H, F=BS.F, K=BS.K, hz=BS.HZ)
    print(f"Saved {len(SC):,} samples from {len(set(SC))} recordings, {len(set(loc))} locations -> {out}")


def stage_labels(files, tag):
    d = np.load(f"data/processed/highd_samples{tag}.npz", allow_pickle=True)
    parts = [pd.read_parquet(SSM_DIR / f.name, columns=["scene_id", "time_s", "agent_id", "ttc2d_mov"]) for f in files]
    ssm = pd.concat(parts, ignore_index=True)
    keys = ssm["scene_id"].astype(str) + "|" + ssm["agent_id"].astype(str)
    codes, uniques = pd.factorize(keys)
    lookup = pd.Series(np.arange(len(uniques)), index=uniques)
    s_keys = pd.Series(d["scene"]).astype(str) + "|" + pd.Series(d["agent"]).astype(str)
    tr_s = lookup.reindex(s_keys).to_numpy()
    if np.isnan(tr_s.astype(float)).any():
        raise SystemExit("some samples have no SSM track - rebuild samples and SSM from the same files")
    fr_rows = np.round(ssm["time_s"].to_numpy() * HZ).astype(np.int64)
    rows, pos3 = [], {}
    for thr, name in ((1.5, "ttc15"), (3.0, "ttc30")):
        BHL.THRESH = thr                                       # label_vectorized reads this module constant
        ev, ob, now = BHL.label_vectorized(codes, fr_rows, ssm["ttc2d_mov"].to_numpy(),
                                           tr_s.astype(np.int64), d["frame"].astype(np.int64))
        cur = (now < thr).astype(np.int64)
        np.savez_compressed(f"data/processed/highd_labels_{name}{tag}.npz", event_bin=ev, obs_bins=ob, current=cur,
                            ttc2d_now=now, threshold=thr)
        keep = cur == 0
        for b, h in enumerate([1, 2, 3, 5]):
            p = keep & (ev >= 0) & (ev <= b); known = keep & (p | (ob >= b + 1))
            rows.append(dict(threshold_s=thr, horizon_s=h, samples=len(ev), in_conflict_now=int(cur.sum()),
                             evaluable=int(known.sum()), positives=int(p.sum()), prevalence=p.sum() / max(known.sum(), 1)))
            if h == 3:
                pos3[thr] = int(p.sum())
    s = pd.DataFrame(rows)
    primary = 1.5 if pos3[1.5] >= MIN_POS else 3.0
    s["primary"] = s["threshold_s"] == primary
    Path("results").mkdir(exist_ok=True); s.to_csv(f"results/highd_dataset_summary{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    print(f"\nTHRESHOLD RULE: positives at 3 s with TTC2D < 1.5 s = {pos3[1.5]} (minimum {MIN_POS}) "
          f"-> PRIMARY label threshold = {primary} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["ssm", "samples", "labels"])
    ap.add_argument("--max-recordings", type=int, default=None)
    a = ap.parse_args()
    files = recordings(a.max_recordings); tag = "_test" if a.max_recordings else ""
    print(f"{len(files)} highD recordings | stage {a.stage}")
    {"ssm": lambda: stage_ssm(files), "samples": lambda: stage_samples(files, tag),
     "labels": lambda: stage_labels(files, tag)}[a.stage]()


if __name__ == "__main__":
    main()