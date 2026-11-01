"""
REGISTERED SPLIT for SHRP2 (pre-registration osf.io/y5zrp, Section 5): units split 80/10/10 into training,
validation and test, stratified by outcome type (crash, near-crash, baseline) and by the lighting x surface
cell, with seed 131. Run ONCE, after the audit-dependent choices are frozen and BEFORE any model is trained.

Units  : trips (one event record each) - the registered rule when no driver identifier exists. If the audit
         finds a driver identifier, units become drivers; the stratum of a driver must then be fixed in the
         freeze note first, so this script refuses to run on drivers until that rule is written in.
Input  : data/processed/shrp2/trips.parquet (src/loaders/shrp2.py): in_study, excluded_implausible,
         outcome_type, lighting_bin, surfaceCondition_bin.
Strata : outcome_type x L{lighting_bin} x S{surface_bin}; an unknown level is its own level "U" (registered:
         unknown categories are excluded only from the analyses of that factor). Trips without environment
         data (expected: all baselines) therefore form the strata baseline x LU x SU.
Rule (fixed here, before any SHRP2 access):
         strata in sorted order; inside a stratum the event IDs are sorted, then permuted by ONE generator
         np.random.default_rng(131) used across strata in that order; n_val = n_test = floor(0.1 n + 0.5);
         the first n_val permuted units -> validation, the next n_test -> test, the rest -> training.
         Strata with fewer than 5 units therefore go entirely to training (reported).
Outputs: data/processed/shrp2/split.csv (event_id, split, stratum) - stays on the laptop;
         results/shrp2_split_summary.txt - aggregate counts only (shareable), with the SHA-256 of split.csv
         to be written into the deviations log ("splits frozen").
Usage  : python src/shrp2/make_split.py [--trips data/processed/shrp2/trips.parquet]
"""
import argparse, hashlib, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

SEED, SHARE = 131, 0.10


def stratum_of(row):
    def lv(v):
        return "U" if pd.isna(v) else str(int(v))
    return f"{row['outcome_type']}|L{lv(row.get('lighting_bin'))}|S{lv(row.get('surfaceCondition_bin'))}"


def make_split(trips):
    if "driver_id" in trips.columns:
        raise SystemExit("driver identifier present: write the driver-stratum rule into the freeze note first")
    u = trips[trips["in_study"].astype(bool) & ~trips["excluded_implausible"].astype(bool)].copy()
    if u["event_id"].duplicated().any():
        raise SystemExit("event_id repeated in trips table")
    u["stratum"] = u.apply(stratum_of, axis=1)
    rng = np.random.default_rng(SEED)
    out = []
    for s in sorted(u["stratum"].unique()):
        ids = np.sort(u.loc[u["stratum"] == s, "event_id"].to_numpy())
        perm = ids[rng.permutation(len(ids))]
        k = int(math.floor(SHARE * len(ids) + 0.5))
        lab = np.array(["val"] * k + ["test"] * k + ["train"] * (len(ids) - 2 * k))
        out.append(pd.DataFrame({"event_id": perm, "split": lab, "stratum": s}))
    S = pd.concat(out, ignore_index=True).sort_values("event_id").reset_index(drop=True)
    return S


def summary(S):
    tab = S.pivot_table(index="stratum", columns="split", values="event_id", aggfunc="count", fill_value=0)
    tab = tab.reindex(columns=["train", "val", "test"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    return tab


def main(trips_path, out_dir="data/processed/shrp2", res_dir="results"):
    T = pd.read_parquet(trips_path)
    S = make_split(T)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / "split.csv"
    S.to_csv(f, index=False, lineterminator="\n")
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    tab = summary(S)
    small = tab[tab["total"] < 5].index.tolist()
    lines = ["SHRP2 REGISTERED SPLIT - aggregate counts only (shareable)", f"seed {SEED}; 80/10/10 per stratum; "
             f"n_val = n_test = floor(0.1 n + 0.5)", f"split.csv SHA-256: {sha}", "",
             tab.to_string(), "", f"totals: " + ", ".join(f"{c} {int(tab[c].sum())}" for c in ["train", "val", "test", "total"]),
             f"strata with < 5 units (all training): {len(small)}"]
    Path(res_dir).mkdir(exist_ok=True)
    Path(res_dir, "shrp2_split_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return S, sha


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trips", default="data/processed/shrp2/trips.parquet")
    a = ap.parse_args(); main(a.trips)
