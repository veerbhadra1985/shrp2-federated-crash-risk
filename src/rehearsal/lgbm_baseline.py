"""
Stage 1, step 6: gradient-boosting baseline exactly as registered (Section 4):
"LightGBM on hand-crafted features (current TTC, DRAC and TTC2D, subject speed and
acceleration, number of objects, nearest gap); library defaults, with early stopping
on validation loss."

Implementation choices (fixed here, before any run; report them in the paper):
  - One binary classifier per horizon (1, 2, 3, 5 s); cumulative risk made
    non-decreasing across horizons with a running maximum (as in Step 12).
  - Features at prediction time t for the target vehicle:
      ttc        longitudinal TTC to the leader (s)             [ssm.py ttc_min]
      drac       DRAC w.r.t. the leader (m/s^2)                  [ssm.py drac_max]
      ttc2d      2D TTC, all pairs (s)                            [ssm.py ttc2d_min]
      ttc2d_mov  2D TTC, moving pairs (s)                         [ssm.py ttc2d_mov]
      speed      target speed (m/s), last history frame
      accel      longitudinal acceleration (m/s^2) over the last 0.5 s
      n_objects  neighbours present in the last history frame (<= 8 within 30 m)
      gap        centre distance to the nearest neighbour (m)
    "No leader / no conflict" (infinite TTC) is passed as missing (NaN); LightGBM
    handles missing values natively.
  - Library defaults (LGBMClassifier(), n_estimators = 100) + early stopping after
    10 rounds without improvement in validation log-loss. With default settings
    LightGBM uses no random subsampling, so the 5 seeds are expected to give
    identical results; the script reports this rather than assuming it.
  - Same samples (not in conflict at t), same project-wide split as every step.

Usage : python src/rehearsal/lgbm_baseline.py
Inputs: data/processed/pretrain_samples.npz, data/processed/hazard_labels.npz,
        data/processed/interaction_ssm.parquet
Output: results/lgbm_test_metrics.csv
        data/processed/preds/lgbm_s<seed>.npz   (validation + test predictions, for the
                                                 analysis engine of step 7)
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from train_pretrain import split_by_scene
from calibration import labels_at, ece_adaptive

SAMPLES = Path("data/processed/pretrain_samples.npz")
LABELS = Path("data/processed/hazard_labels.npz")
SSM = Path("data/processed/interaction_ssm.parquet")
PRED_DIR = Path("data/processed/preds")
OUT = Path("results/lgbm_test_metrics.csv")
HZ, HORIZONS, SEEDS = 10, [1, 2, 3, 5], [131, 132, 133, 134, 135]
warnings.filterwarnings("ignore", message=".*eval_set.*")   # newer LightGBM renames it
FEATURES = ["ttc", "drac", "ttc2d", "ttc2d_mov", "speed", "accel", "n_objects", "gap"]


def build_features(d, keep):
    t0 = time.time()
    hist = d["hist"]                                # (N, 9, 20, 9): x, y, vx, vy, sin, cos, L, W, valid
    me = hist[:, 0]
    speed = np.hypot(me[:, -1, 2], me[:, -1, 3])
    accel = (me[:, -1, 2] - me[:, -6, 2]) / 0.5     # local frame: vx = forward speed at t
    accel = np.where(me[:, -6, 8] > 0, accel, np.nan)
    nb = hist[:, 1:, -1]                            # neighbours at the last frame
    present = nb[..., 8] > 0
    dist = np.where(present, np.hypot(nb[..., 0], nb[..., 1]), np.inf).min(1)
    gap = np.where(np.isfinite(dist), dist, np.nan)
    X = pd.DataFrame({"speed": speed[keep], "accel": accel[keep],
                      "n_objects": present.sum(1)[keep], "gap": gap[keep]})
    del hist, me, nb, present

    ssm = pd.read_parquet(SSM, columns=["scene_id", "time_s", "agent_id", "ttc_min", "drac_max",
                                        "ttc2d_min", "ttc2d_mov"])
    ssm["scene"] = ssm.pop("scene_id").astype(str)
    ssm["agent"] = ssm.pop("agent_id").astype(str)
    ssm["frame"] = np.round(ssm.pop("time_s").to_numpy() * HZ).astype(np.int64)
    n0 = len(ssm); ssm = ssm.drop_duplicates(["scene", "agent", "frame"])
    if len(ssm) < n0:
        print(f"  note: {n0 - len(ssm)} duplicate SSM rows dropped")
    s = pd.DataFrame({"scene": d["scene"][keep].astype(str), "agent": d["agent"][keep].astype(str),
                      "frame": d["frame"][keep].astype(np.int64)})
    m = s.merge(ssm, on=["scene", "agent", "frame"], how="left", validate="many_to_one")
    if len(m) != len(s) or m["ttc_min"].isna().mean() > 0.001:
        raise SystemExit("samples and SSM rows do not match - rebuild both from the same data")
    for new, old in [("ttc", "ttc_min"), ("drac", "drac_max"), ("ttc2d", "ttc2d_min"), ("ttc2d_mov", "ttc2d_mov")]:
        v = m[old].to_numpy(dtype=float)
        X[new] = np.where(np.isfinite(v), v, np.nan)
    print(f"  features built for {len(X):,} samples in {time.time() - t0:.0f} s "
          f"(missing share: " + ", ".join(f"{c} {X[c].isna().mean():.0%}" for c in FEATURES) + ")")
    return X[FEATURES]


def fit_predict(X, ev, ob, tr, va, seed):
    """One classifier per horizon; returns cumulative risk (n, 4) for all samples."""
    risk = np.zeros((len(X), len(HORIZONS))); info = []
    for k, h in enumerate(HORIZONS):
        y, known = labels_at(ev, ob, k)
        a, b = tr & known, va & known
        clf = lgb.LGBMClassifier(random_state=seed, verbose=-1)
        clf.fit(X[a], y[a], eval_set=[(X[b], y[b])], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(10, verbose=False)])
        risk[:, k] = clf.predict_proba(X, num_iteration=clf.best_iteration_)[:, 1]
        info.append(clf.best_iteration_)
    return np.maximum.accumulate(risk, axis=1), info


def evaluate(risk, ev, ob, te, extra):
    rows = []
    for k, h in enumerate(HORIZONS):
        y, known = labels_at(ev, ob, k)
        m = te & known
        rows.append(dict(extra, horizon_s=h, n=int(m.sum()), positives=int(y[m].sum()),
                         prevalence=y[m].mean(), auprc=average_precision_score(y[m], risk[m, k]),
                         auroc=roc_auc_score(y[m], risk[m, k]), ece_adaptive=ece_adaptive(y[m], risk[m, k]),
                         mean_pred=risk[m, k].mean()))
    return rows


def main():
    for p in (SAMPLES, LABELS, SSM):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    print("loading samples and labels ...")
    d = np.load(SAMPLES, allow_pickle=True)
    lab = np.load(LABELS)
    keep = lab["current"] == 0                                     # same filter as train_hazard.py
    ev, ob = lab["event_bin"][keep], lab["obs_bins"][keep]
    tr, va, te = split_by_scene(d["scene"][keep], d["location"][keep])
    print(f"samples (not in conflict at t): train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")
    X = build_features(d, keep)

    rows, risks = [], {}
    PRED_DIR.mkdir(parents=True, exist_ok=True); OUT.parent.mkdir(exist_ok=True)
    idx = np.flatnonzero(keep)                                     # index into the full sample arrays
    for seed in SEEDS:
        t0 = time.time()
        risk, iters = fit_predict(X, ev, ob, tr, va, seed)
        risks[seed] = risk
        rows += evaluate(risk, ev, ob, te, {"model": "lgbm", "seed": seed})
        sel = va | te
        np.savez_compressed(PRED_DIR / f"lgbm_s{seed}.npz", idx=idx[sel], split=np.where(te[sel], "test", "val"),
                            risk=risk[sel].astype(np.float32), horizons=np.array(HORIZONS))
        print(f"  seed {seed}: trees per horizon {iters}, {time.time() - t0:.0f} s")

    # classical reference on the same test samples: current moving-pair TTC2D (lower = riskier)
    ttc_score = -np.nan_to_num(X["ttc2d_mov"].to_numpy(), nan=1e3)
    for k, h in enumerate(HORIZONS):
        y, known = labels_at(ev, ob, k); m = te & known
        rows.append(dict(model="ttc2d_mov_now", seed=0, horizon_s=h, n=int(m.sum()), positives=int(y[m].sum()),
                         prevalence=y[m].mean(), auprc=average_precision_score(y[m], ttc_score[m]),
                         auroc=roc_auc_score(y[m], ttc_score[m]), ece_adaptive=np.nan, mean_pred=np.nan))
    res = pd.DataFrame(rows); res.to_csv(OUT, index=False)

    same = all(np.array_equal(risks[SEEDS[0]], risks[s]) for s in SEEDS[1:])
    print(f"\nidentical predictions across the 5 seeds: {same}")
    g = res[res.model == "lgbm"].groupby("horizon_s")
    summary = g[["prevalence", "auprc", "auroc", "ece_adaptive", "mean_pred"]].mean()
    summary = summary.join(g["auprc"].std().rename("auprc_sd_seeds"))
    summary["auprc_ttc2d_now"] = res[res.model == "ttc2d_mov_now"].set_index("horizon_s")["auprc"]
    pd.set_option("display.width", 200)
    print("\n=== LightGBM baseline, test split (mean over seeds) ===")
    print(summary.round(4).to_string())
    print("\nReference (Step 11, hazard model, same split): AUPRC 0.461 / 0.372 / 0.324 / 0.295; "
          "TTC2D-now 0.256 / 0.174 / 0.147 / 0.131")
    print(f"Saved {OUT} and {PRED_DIR}/lgbm_s<seed>.npz")


if __name__ == "__main__":
    main()