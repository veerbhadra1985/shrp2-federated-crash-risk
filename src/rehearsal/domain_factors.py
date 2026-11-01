"""
Stage 1, step 5 (dress rehearsal): binary DOMAIN FACTORS on INTERACTION that mimic
the structure of the SHRP2 factors registered for H2 and H3.

Why: on SHRP2, H2 holds out one level of a binary factor (lighting, weather, surface,
traffic density). Those levels are unequal in size (>80% of events in non-adverse
conditions) and each level spans many trips. The rehearsal needs the same structure.

RULES - fixed in this file BEFORE any model is trained on these factors. Only
covariates (no outcomes) are used to define the levels; event counts are reported
afterwards for the testability check only, as in the registered SHRP2 audit.

  F1 region   : 1 = non-USA location, 0 = USA                    (location level, strong shift)
  F2 road     : 1 = roundabout,       0 = intersection or merge  (location level, strong shift)
  F3 density  : 1 = recording in the top quartile of mean vehicles per frame
                    WITHIN its own location, 0 = otherwise         (recording level, mild shift)
  F4 speed    : 1 = recording in the bottom quartile of mean speed WITHIN its own
                    location, 0 = otherwise                         (recording level, mild shift)
      Pre-declared fallback: if |phi(F3, F4)| > 0.5 over recordings (near-duplicate
      factors), F4 is replaced by HEAVY: 1 = recording in the top quartile of the share
      of long vehicles (> 7 m) within its own location (locations without any long
      vehicle: all 0).

Testability (registered rule, copied): a held-out level with fewer than 50 test events
at 3 s makes that factor-direction pair not testable; H2 needs at least 4 of 8 pairs.

Usage : python src/rehearsal/domain_factors.py
Inputs: data/processed/interaction.parquet, data/processed/pretrain_samples.npz,
        data/processed/hazard_labels.npz
Output: results/rehearsal_factors_by_recording.csv  (one row per recording)
        results/rehearsal_factor_summary.csv        (one row per factor level)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from train_pretrain import split_by_scene          # THE project-wide split

PARQUET = Path("data/processed/interaction.parquet")
SAMPLES = Path("data/processed/pretrain_samples.npz")
LABELS = Path("data/processed/hazard_labels.npz")
OUT_REC = Path("results/rehearsal_factors_by_recording.csv")
OUT_SUM = Path("results/rehearsal_factor_summary.csv")
HZ, QUART, LONG_M, PHI_MAX, MIN_TEST_EVENTS, H3S = 10, 0.25, 7.0, 0.5, 50, 2   # H3S = bin index of 3 s


def within_location_flag(rec, col, top):
    """1 for recordings in the top (or bottom) quartile of `col` inside their location."""
    q = rec.groupby("location")[col].transform(lambda s: s.quantile(1 - QUART if top else QUART))
    return ((rec[col] > q) if top else (rec[col] < q)).astype(int)


def phi(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def recording_covariates():
    df = pd.read_parquet(PARQUET, columns=["scene_id", "time_s", "agent_id", "vx", "vy", "length"])
    df["scene_id"] = df["scene_id"].astype(str)
    df["frame"] = np.round(df["time_s"] * HZ).astype(np.int64)
    df["speed"] = np.hypot(df["vx"], df["vy"])
    per_frame = df.groupby(["scene_id", "frame"]).size().groupby(level=0).mean().rename("veh_per_frame")
    speed = df.groupby("scene_id")["speed"].mean().rename("mean_speed")
    agents = df.groupby(["scene_id", "agent_id"])["length"].max()
    long_share = (agents > LONG_M).groupby(level=0).mean().rename("long_share")
    n_veh = agents.groupby(level=0).size().rename("vehicles")
    rec = pd.concat([per_frame, speed, long_share, n_veh], axis=1).reset_index()
    rec = rec.rename(columns={"index": "scene_id"})
    rec["scene_id"] = rec["scene_id"].astype(str)
    rec["location"] = rec["scene_id"].str.split("/").str[0]
    return rec


def define_factors(rec):
    loc = rec["location"].str.upper()
    rec["F1_region"] = (~loc.str.contains("USA")).astype(int)
    rec["F2_road"] = loc.str.contains("ROUNDABOUT").astype(int)
    rec["F3_density"] = within_location_flag(rec, "veh_per_frame", top=True)
    rec["F4_speed"] = within_location_flag(rec, "mean_speed", top=False)
    r34 = phi(rec["F3_density"], rec["F4_speed"])
    print(f"phi(F3 density, F4 speed) over recordings = {r34:+.2f} (limit |phi| <= {PHI_MAX})")
    if abs(r34) > PHI_MAX:
        print("  -> pre-declared fallback: F4 = HEAVY (top quartile of long-vehicle share)")
        rec = rec.drop(columns="F4_speed")
        rec["F4_heavy"] = within_location_flag(rec, "long_share", top=True)
        rec.loc[rec.groupby("location")["long_share"].transform("max") == 0, "F4_heavy"] = 0
    return rec


def main():
    for p in (PARQUET, SAMPLES, LABELS):
        if not p.exists():
            raise SystemExit(f"missing {p} - run the pipeline steps first")
    print("computing recording covariates ...")
    rec = define_factors(recording_covariates())
    factors = [c for c in rec.columns if c[:2] in ("F1", "F2", "F3", "F4")]

    print("loading samples and labels ...")
    d = np.load(SAMPLES, allow_pickle=True)
    lab = np.load(LABELS)
    keep = lab["current"] == 0                          # same filter as train_hazard.py
    scene = d["scene"][keep].astype(str); location = d["location"][keep].astype(str)
    ev, ob = lab["event_bin"][keep], lab["obs_bins"][keep]
    tr, va, te = split_by_scene(scene, location)       # identical call to train_hazard.py
    pos3 = (ev >= 0) & (ev <= H3S)
    known3 = pos3 | (ob >= H3S + 1)

    missing = set(np.unique(scene)) - set(rec["scene_id"])
    if missing:
        raise SystemExit(f"{len(missing)} sample recordings have no covariates - rebuild from the same data")
    split_of = pd.Series(np.select([tr, va, te], ["train", "val", "test"], "none"), index=scene)
    rec["split"] = rec["scene_id"].map(split_of.groupby(level=0).first()).fillna("no samples")
    lut = rec.set_index("scene_id")

    rows = []
    for f in factors:
        lev = lut.loc[scene, f].to_numpy()
        for v in (0, 1):
            m = lev == v
            recs = rec[rec[f] == v]
            rows.append(dict(
                factor=f, level=v,
                recordings=len(recs), locations=recs["location"].nunique(),
                share_of_samples=m.mean(),
                train_samples=int((m & tr).sum()), train_events_3s=int((m & tr & pos3).sum()),
                val_events_3s=int((m & va & pos3).sum()),
                test_recordings=int(recs["split"].eq("test").sum()),
                test_samples=int((m & te & known3).sum()), test_events_3s=int((m & te & pos3).sum()),
                test_rate_3s=(m & te & pos3).sum() / max((m & te & known3).sum(), 1)))
    s = pd.DataFrame(rows)
    s["testable_as_heldout"] = s["test_events_3s"] >= MIN_TEST_EVENTS

    OUT_REC.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(OUT_REC, index=False); s.to_csv(OUT_SUM, index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\n=== Factor levels (3 s horizon; test counts = held-out evaluation data) ===")
    show = s[["factor", "level", "recordings", "locations", "share_of_samples", "train_samples",
              "train_events_3s", "test_recordings", "test_events_3s", "test_rate_3s", "testable_as_heldout"]].copy()
    show["share_of_samples"] = show["share_of_samples"].map("{:.1%}".format)
    show["test_rate_3s"] = show["test_rate_3s"].map("{:.2%}".format)
    print(show.to_string(index=False))

    print("\n=== Checks ===")
    n_pairs = int(s["testable_as_heldout"].sum())
    print(f"testable factor-direction pairs: {n_pairs} of 8 (registered minimum 4) -> "
          f"{'OK' if n_pairs >= 4 else 'NOT ENOUGH'}")
    for f in factors:
        sub = s[s.factor == f].set_index("level")
        if (sub["train_samples"] == 0).any() or (sub["test_recordings"] == 0).any():
            print(f"  WARNING {f}: a level has no training samples or no test recording")
        ratio = sub.loc[0, "train_samples"] / max(sub.loc[1, "train_samples"], 1)
        print(f"  {f}: training-size ratio level0/level1 = {ratio:.1f} (size-matching relevant if far from 1)")
    print("Correlation between factors over recordings (phi):")
    for i, a in enumerate(factors):
        for b in factors[i + 1:]:
            print(f"  {a} x {b}: {phi(rec[a], rec[b]):+.2f}")
    print(f"\nSaved {OUT_REC} and {OUT_SUM}")


if __name__ == "__main__":
    main()