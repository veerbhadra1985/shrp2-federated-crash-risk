"""
SHRP2 AUDIT, BASELINE FLAG CHECK (20 Oct 2026; written after the audit supplement and BEFORE any sample,
split or model). Aggregate counts only (no event IDs). Baselines carry no outcome, so nothing here relates
any trip to an outcome.

Why: 30,741 safe baseline trips are in the trajectory files, but the metadata flag ego_reconstructed marks only
10,919 of them (the readme figure). In the dataset authors' code (GSSM, src_trajectory_reconstruction/
reconstruct_birdseye.py), the flags are written in two places:
  * record_invalid(...) during the loop: (1, 0) for a trip whose subject vehicle is reconstructed but that has
    no usable object - in EVERY chunk;
  * after the loop: ego_reconstructed = surrounding_reconstructed = 1 for the trips in data_ego / data_sur.
    For the baselines, which are saved in 5 chunks of about 6,600 trips, data_ego is emptied after each saved
    chunk, so at that point it holds the LAST chunk only.
If that is what happened, the flags (and the readme counts taken from them) miss the trips with objects in
chunks 0-3, although those trips were reconstructed by the same code and are in the files.

The check counts, per baseline chunk file: trips in the Ego and Surrounding files, and the metadata flags.
The artefact explanation predicts ALL of:
  P1 chunks 0..3: no trip flagged (1, 1);
  P2 last chunk: flagged (1, 1) = its trips with objects;
  P3 every baseline without objects (ego only) is flagged (1, 0), in every chunk;
  P4 every unflagged baseline in the files has objects.
If audit_trips.parquet (local, from audit_counts.py) exists, it also compares the unflagged and the flagged
baselines with objects on record length, share of frames with an object, share of implausible frames and the
implausible-value exclusion (aggregate medians and shares only).
Usage : python src/shrp2/audit_chunks.py --root data/raw/shrp2/extracted [--trips data/processed/shrp2/audit_trips.parquet]
Output: results/shrp2_audit_chunks.txt (shareable)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shrp2_format as F
from audit_counts import find_dirs, read_h5


def check(root, trips_path=None, out="results"):
    rt, _ = find_dirs(root)
    meta = pd.read_csv(rt / "metadata_birdseye.csv")
    meta = meta[meta["event_category"] == F.BASELINE_CATEGORY].set_index("event_id")
    fe = meta["ego_reconstructed"].fillna(0).astype(float) > 0.5
    fs = meta["surrounding_reconstructed"].fillna(0).astype(float) > 0.5
    d = rt / F.BASELINE_CATEGORY
    chunks, groups = [], {}
    ego_files = sorted(d.glob(f"{F.EGO_FILE}_*.h5"), key=lambda p: int(p.stem.split("_")[-1]))
    for p in ego_files:
        c = p.stem.split("_")[-1]
        e = set(read_h5(p)["event_id"].unique())
        sp = d / f"{F.SUR_FILE}_{c}.h5"
        s = set(read_h5(sp)["event_id"].unique()) if sp.exists() else set()
        ids = pd.Index(sorted(e))
        f_e, f_s = fe.reindex(ids, fill_value=False), fs.reindex(ids, fill_value=False)
        has = ids.isin(list(s))
        chunks.append(dict(chunk=int(c), in_ego=len(e), in_sur=int(has.sum()), sur_not_in_ego=len(s - e),
                           flag_11=int((f_e & f_s).sum()), flag_10=int((f_e & ~f_s).sum()), flag_00=int((~f_e & ~f_s).sum()),
                           flag_01=int((~f_e & f_s).sum()),
                           with_obj_flag_11=int((has & f_e & f_s).sum()), with_obj_flag_00=int((has & ~f_e & ~f_s).sum()),
                           ego_only=int((~has).sum()), ego_only_flag_10=int((~has & f_e & ~f_s).sum()),
                           not_in_files_flagged=0))
        for i, h, a, b in zip(ids, has, f_e, f_s):
            groups[i] = "with_objects_unflagged" if (h and not a) else ("with_objects_flagged" if h else "ego_only")
    C = pd.DataFrame(chunks).set_index("chunk")
    in_files = set(groups)
    C.loc[C.index[-1], "not_in_files_flagged"] = int((fe & ~meta.index.isin(list(in_files))).sum())
    last = C.index.max()
    P = dict(P1_chunks_before_last_have_no_flag_11=bool((C.loc[C.index != last, "flag_11"] == 0).all()),
             P2_last_chunk_flag_11_equals_its_trips_with_objects=bool(C.at[last, "flag_11"] == C.at[last, "in_sur"]
                                                                       == C.at[last, "with_obj_flag_11"]),
             P3_every_ego_only_trip_flagged_10=bool((C["ego_only"] == C["ego_only_flag_10"]).all()),
             P4_every_unflagged_trip_in_files_has_objects=bool((C["flag_00"] == C["with_obj_flag_00"]).all()
                                                               and (C["flag_01"] == 0).all()),
             no_flagged_trip_missing_from_files=bool(C["not_in_files_flagged"].sum() == 0))
    P["artefact_explanation_holds"] = all(P.values())
    lines = ["SHRP2 AUDIT - BASELINE FLAG CHECK - AGGREGATE COUNTS ONLY (no event IDs)", "=" * 72,
             "[per_chunk]", C.to_string(), "",
             "[totals]", "  " + str({k: int(v) for k, v in C.sum().items()}), "", "[predictions]"]
    lines += [f"  {k}: {v}" for k, v in P.items()]
    if trips_path and Path(trips_path).exists():
        T = pd.read_parquet(trips_path)
        T = T[T["category"] == F.BASELINE_CATEGORY].copy()
        T["group"] = T["event_id"].map(groups)
        T["share_frames_with_object"] = T["frames_with_object"] / T["n_frames"].clip(lower=1)
        Q = T.groupby("group").agg(trips=("event_id", "size"), duration_s_median=("duration_s", "median"),
                                   duration_s_p25=("duration_s", lambda x: np.percentile(x, 25)),
                                   share_frames_with_object_median=("share_frames_with_object", "median"),
                                   share_implausible_mean=("share_implausible", "mean"),
                                   excluded_implausible=("excluded_implausible", "sum"),
                                   eligible_100_frames=("n_eval3", lambda x: int((x >= F.MIN_EVAL_UNIT).sum())),
                                   max_objects_median=("max_objects", "median"))
        lines += ["", "[quality_by_group] (baselines only; no outcome involved)", Q.round(4).to_string()]
    else:
        lines += ["", "[quality_by_group] skipped (audit_trips.parquet not found)"]
    Path(out).mkdir(exist_ok=True)
    Path(out, "shrp2_audit_chunks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines)); print(f"\nSaved {out}/shrp2_audit_chunks.txt (shareable)")
    return C, P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True)
    ap.add_argument("--trips", default="data/processed/shrp2/audit_trips.parquet")
    ap.add_argument("--out-results", default="results")
    a = ap.parse_args(); check(a.root, a.trips, a.out_results)
