"""
SHRP2 DATA AUDIT - step (1) of the registered order of work (pre-registration osf.io/y5zrp, Section 8):
"counts, identifiers, horizon retention and domain sizes; event counts are used, but no model is trained
or evaluated". This script reads the data files and COUNTS. It loads no model, computes no risk score and
no surrogate safety measure, and relates nothing to outcomes beyond the registered event counts.

Outputs
  results/shrp2_audit_summary.txt and .json   AGGREGATE counts only (no event IDs) - the only files that may
                                              be shared (e.g. pasted to the assistant) or committed to Git
  data/processed/shrp2/audit_trips.parquet    one row per trip WITH event IDs - stays on the encrypted laptop,
                                              never shared, never committed (data/ is outside Git)
Sections of the summary
  A version and files      B schema check            C identifiers (unit rule)
  D trip and event counts vs published counts         E record lengths, time grid, duplicate times
  F horizon retention (registered 80 % rule)          G implausible values (registered masking rules)
  H evaluable frames and unit eligibility (Amendment 2) I environment coverage and domain sizes
  J radar objects per frame                           K object reference point (radar point vs centre)
Usage : python src/shrp2/audit_counts.py --root <folder that contains ReconstructedTrajectories and
        SafetyCriticalTestSet>   [--out-results results] [--out-local data/processed/shrp2]
Definitions are imported from src/shrp2/shrp2_format.py (single source of truth).
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shrp2_format as F
import labels as L


# ------------------------------------------------------------------ discovery and reading
def find_dirs(root):
    root = Path(root)
    hits = sorted(root.rglob("metadata_birdseye.csv"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one metadata_birdseye.csv under {root}, found {len(hits)}: {hits}")
    rt = hits[0].parent
    st_c = [p for p in root.rglob(F.ST_DIR) if p.is_dir()]
    st = st_c[0] if len(st_c) == 1 else None
    if len(st_c) > 1:
        raise SystemExit(f"more than one {F.ST_DIR} folder under {root}: {st_c}")
    return rt, st


def read_h5(path):
    return pd.read_hdf(path, key=F.H5_KEY)


def category_files(rt):
    """{category: (ego files, surrounding files)}; a category = sub-folder holding Ego_birdseye*.h5."""
    out = {}
    for d in sorted(p for p in rt.iterdir() if p.is_dir()):
        ego = sorted(d.glob(f"{F.EGO_FILE}*.h5")); sur = sorted(d.glob(f"{F.SUR_FILE}*.h5"))
        if ego:
            out[d.name] = (ego, sur)
    return out


# ------------------------------------------------------------------ per-trip computations
def evaluable_counts(ts_ms, te_ms, masked=None):
    """Registered labels at the primary horizon (3 s), from src/shrp2/labels.py (the same code the sample
    builder uses). Prediction times: contiguous 2-s history and an unmasked subject frame at t. Excluded:
    t >= event time; time to event in (5, 7] s. Positive: time to event in (0, 3] s. Negative: otherwise, if
    the trip is observed >= 3 s ahead. Returns (evaluable, positive, negative, n_prediction_times)."""
    ts = np.asarray(ts_ms, np.int64)
    idx = L.prediction_index(ts, masked)
    if len(idx) == 0:
        return 0, 0, 0, 0
    excl, ev, obs = L.label_arrays(ts[idx], te_ms, ts[-1])
    y, known = L.known_at(ev[~excl], obs[~excl], 2)
    pos = int(y.sum()); kn = int(known.sum())
    return kn, pos, kn - pos, int(len(idx))


def clock_ms(time_s):
    """The trip clock. In the released files 'time' is the 0.1-s grid (seconds) and 'timestamp' is the raw
    millisecond stamp where a raw sample existed, -1 where the authors interpolated ([D4] process_ego,
    ekf_sur). So every computation uses round(time * 1000); 'timestamp' is only counted."""
    return np.round(np.asarray(time_s, float) * 1000.0).astype(np.int64)


def implausible_ego(ts_ms, v, acc, te_ms, is_crash):
    sp = np.asarray(v, float) > F.IMPLAUSIBLE["speed"]
    win = np.zeros(len(ts_ms), bool)
    if is_crash and te_ms is not None:
        win = np.abs(np.asarray(ts_ms, np.int64) - int(te_ms)) <= int(F.IMPLAUSIBLE["acc_crash_window_s"] * 1000)
    ac = (np.abs(np.asarray(acc, float)) > F.IMPLAUSIBLE["acc"]) & ~win
    return sp, ac


def dedup_objects(sur):
    """Sort object frames by object and grid time; keep the first row of duplicate (object, grid time) pairs."""
    s = sur.sort_values(["target_id", "time"], kind="mergesort")
    fm = clock_ms(s["time"]); tid = s["target_id"].to_numpy()
    return s[np.r_[True, (np.diff(fm) != 0) | (np.diff(tid) != 0)]]


def object_flags(sur):
    """Per object frame of a DEDUPLICATED, sorted frame (dedup_objects): speed > 60 m/s; position jump > 10 m
    from the previous frame of the same object. Flags are aligned with the rows of the input."""
    if sur is None or len(sur) == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    s = sur
    dx = s.groupby("target_id")["x_ekf"].diff(); dy = s.groupby("target_id")["y_ekf"].diff()
    jump = (np.hypot(dx, dy) > F.IMPLAUSIBLE["jump"]).to_numpy()
    return (s["v_ekf"].to_numpy() > F.IMPLAUSIBLE["speed"]), jump


def binary_factor(f, val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    if F.FACTOR_MAPS[f] == "LOS":
        # the data write 'Level-of-service B: Flow with some restrictions' (audit of 20 Oct 2026); the level is
        # the letter after the prefix. Bare codes ('A1', 'C') are still accepted. Rule unchanged: A-B -> 0, C-F -> 1.
        if val in ("Unknown", ""):
            return None
        m = re.match(r"^(?:level[- ]of[- ]service\s*)?([A-F])(?:\d|\b|:)", val, flags=re.IGNORECASE)
        if not m:
            return "UNMAPPED"                      # reported by the audit, never guessed
        return 0 if m.group(1).upper() in F.LOS_LOW else 1
    return F.FACTOR_MAPS[f].get(val, "UNMAPPED")


def outcome_of(cat):
    """Outcome type of a trip from its event category (C25); 'UNMAPPED' stops the audit and the loader."""
    return F.OUTCOME_BY_CATEGORY.get(cat, "UNMAPPED")


def q(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), min=round(float(x.min()), 2), p25=round(float(np.percentile(x, 25)), 2),
                median=round(float(np.median(x)), 2), p75=round(float(np.percentile(x, 75)), 2),
                max=round(float(x.max()), 2))


def trip_rows(E, SU, cat, em, inc_all):
    """Per-trip counts for one category file (E: subject frames, SU: object frames)."""
    out = []
    SU_by = dict(tuple(SU.groupby("event_id"))) if len(SU) else {}
    for eid, g0 in E.sort_values(["event_id", "time"], kind="mergesort").groupby("event_id", sort=False):
        is_base = cat == F.BASELINE_CATEGORY
        fm0 = clock_ms(g0["time"])
        keep = np.r_[True, np.diff(fm0) != 0]                      # duplicate grid times: keep the first row
        g = g0[keep]
        te = int(em.at[eid, "impact_timestamp"]) if (eid in inc_all and pd.notna(em.at[eid, "impact_timestamp"])) else None
        in_study = is_base or (eid in inc_all and te is not None)        # same rule as src/loaders/shrp2.py
        otype = outcome_of(cat)                                     # C25: event category, as the GSSM code
        ts = fm0[keep]                                              # clock in ms on the 0.1-s grid
        gaps = int((np.diff(ts) != 100).sum())
        raw_missing = int((g["timestamp"].to_numpy() < 0).sum())    # -1 = frame interpolated by the authors
        sp, ac = implausible_ego(ts, g["v_ekf"].to_numpy(), g["acc_ekf"].to_numpy(), te, otype == "crash")
        share_bad = float((sp | ac).mean())
        excluded = share_bad > F.IMPLAUSIBLE["trip_share"]
        ne, npos, nneg, npt = evaluable_counts(ts, te, sp | ac) if (in_study and not excluded) else (0, 0, 0, 0)
        su = SU_by.get(eid)
        if su is not None and len(su):
            per_frame = su.groupby(su["time"].round(1))["target_id"].nunique()
            f_with = int(np.isin(np.round(g["time"].to_numpy(), 1), per_frame.index.to_numpy()).sum())
            max_obj = int(per_frame.max())
        else:
            f_with, max_obj = 0, 0
        out.append(dict(event_id=eid, category=cat, outcome_type=otype, in_study=in_study, excluded_implausible=excluded,
                        n_frames=len(ts), duplicate_times=int((~keep).sum()), raw_timestamp_missing=raw_missing,
                        duration_s=(ts[-1] - ts[0]) / 1000.0, gaps=gaps,
                        pre_event_s=(te - ts[0]) / 1000.0 if te is not None else np.nan,
                        ego_speed_gt60=int(sp.sum()), ego_acc_gt15=int(ac.sum()), share_implausible=share_bad,
                        n_eval3=ne, n_pos3=npos, n_neg3=nneg, n_pred_times=npt, frames_with_object=f_with,
                        max_objects=max_obj))
    return out


# ------------------------------------------------------------------ the audit
def audit(root):
    t0 = time.time()
    rt, st = find_dirs(root)
    S = {"A_version_and_files": {}, "B_schema": {}, "C_identifiers": {}, "D_counts": {}, "E_lengths": {},
         "F_retention": {}, "G_implausible": {}, "H_eligibility": {}, "I_environment": {}, "J_objects": {},
         "K_reference_point": {}}
    cats = category_files(rt)
    meta = pd.read_csv(rt / "metadata_birdseye.csv")
    em_files = sorted(st.rglob("event_meta.csv")) if st else []
    env_files = sorted(st.rglob("environment.csv")) if st else []
    ed_files = sorted(st.rglob("event_data.h5")) if st else []
    EM = pd.concat([pd.read_csv(p) for p in em_files], ignore_index=True) if em_files else pd.DataFrame()
    EN = pd.concat([pd.read_csv(p) for p in env_files], ignore_index=True) if env_files else pd.DataFrame()

    # ---- event tables first (small) ----
    need_em = ["event_id", "severity_first", "severity_second", "duration_enough", "conflict", "impact_timestamp"]
    if len(EM) and any(c not in EM.columns for c in need_em):
        S["STOP"] = f"event_meta.csv lacks required columns {[c for c in need_em if c not in EM.columns]} - send this summary"
        S["B_schema"] = dict(event_meta_columns=sorted(EM.columns))
        return S, pd.DataFrame()
    dup_em = int(EM["event_id"].duplicated().sum()) if len(EM) else 0
    dup_env = int(EN["event_id"].duplicated().sum()) if len(EN) else 0
    if dup_em:
        S["STOP"] = f"{dup_em} event IDs appear more than once in event_meta.csv files - send this summary"
        return S, pd.DataFrame()
    if dup_env:
        EN = EN.drop_duplicates("event_id", keep="first")             # reported in section I
    if len(EM):
        EM["sev"] = EM[["severity_first", "severity_second"]].max(axis=1)
        EM["included"] = EM["duration_enough"].astype(str).str.lower().isin(["true", "1"]) & (EM["conflict"].astype(str) != "none")
    em = EM.set_index("event_id") if len(EM) else pd.DataFrame(columns=["sev", "included", "impact_timestamp"])
    inc_all = set(em.index[em["included"].astype(bool)]) if len(em) else set()
    ed_by_cat = {p.parent.name: p for p in ed_files}

    # ---- stream over category files: one ego/surrounding pair in memory at a time ----
    ego_cols, sur_cols, skipped = set(), set(), []
    rows = []; obj_speed = obj_jump = 0; gaps_total = 0; k_along, k_dist = [], []
    for cat, (ef, sf) in cats.items():
        sur_map = {p.name.replace(F.SUR_FILE, ""): p for p in sf}
        ED = None
        for p in ef:
            suffix = p.name.replace(F.EGO_FILE, "")
            E = read_h5(p); ego_cols |= set(E.columns)
            need_e = ["event_id", "timestamp", "time", "v_ekf", "acc_ekf"]
            if any(c not in E.columns for c in need_e):
                skipped.append(p.name); continue
            E = E[need_e].copy()
            SU = read_h5(sur_map[suffix]) if suffix in sur_map else pd.DataFrame(columns=F.SUR_COLS_PUBLIC)
            sur_cols |= set(SU.columns)
            need_s = ["event_id", "target_id", "time", "timestamp", "x_ekf", "y_ekf", "v_ekf", "psi_ekf"]
            SU = SU[[c for c in need_s if c in SU.columns]].copy()
            if len(SU):
                SU = dedup_objects(SU)
            rows += trip_rows(E, SU, cat, em, inc_all)
            if len(SU):
                os_, oj = object_flags(SU); obj_speed += int(os_.sum()); obj_jump += int(oj.sum())
            # K reference point: test-set x_sur vs Surrounding x_ekf for the same object and time
            if cat in ed_by_cat and len(SU):
                if ED is None:
                    ED = read_h5(ed_by_cat[cat]).reset_index(); ED["time"] = ED["time"].round(1)
                m = ED.merge(SU.assign(time=SU["time"].round(1))[["event_id", "target_id", "time", "x_ekf", "y_ekf", "psi_ekf"]],
                             on=["event_id", "target_id", "time"])
                if len(m):
                    dx, dy = (m.x_sur - m.x_ekf).to_numpy(), (m.y_sur - m.y_ekf).to_numpy()
                    k_along.append(dx * np.cos(m.psi_ekf.to_numpy()) + dy * np.sin(m.psi_ekf.to_numpy()))
                    k_dist.append(np.hypot(dx, dy))
            del E, SU
    T = pd.DataFrame(rows)
    gaps_total = int(T.gaps.sum()) if len(T) else 0

    # ---- A version and files ----
    complete = all(c in ego_cols for c in F.EGO_COLS_RAW_ONLY) and bool(env_files)
    S["A_version_and_files"] = dict(
        version_detected="complete" if complete else ("public-like (no raw signals / no environment)"),
        categories=sorted(cats), files_per_category={c: [p.name for p in v[0] + v[1]] for c, v in cats.items()},
        test_set_found=st is not None, event_meta_files=len(em_files), environment_files=len(env_files),
        event_data_files=len(ed_files),
        event_features_files=len(sorted(st.rglob("event_features.npz"))) if st else 0)   # GSSM features shipped? (Amendment 4)

    # ---- B schema ----
    def miss(have, need):
        return sorted(set(need) - set(have))
    B = dict(ego_missing=miss(ego_cols, F.EGO_COLS_PUBLIC), surrounding_missing=miss(sur_cols, F.SUR_COLS_PUBLIC),
             metadata_missing=miss(meta.columns, F.META_COLS),
             event_meta_missing=miss(EM.columns, F.EVENT_META_COLS) if len(EM) else "no event_meta",
             environment_missing=miss(EN.columns, F.ENV_COLS) if len(EN) else "no environment files",
             ego_extra=sorted(ego_cols - set(F.EGO_COLS_PUBLIC + F.EGO_COLS_RAW_ONLY)),
             surrounding_extra=sorted(sur_cols - set(F.SUR_COLS_PUBLIC + F.SUR_COLS_RAW_ONLY)),
             metadata_extra=sorted(set(meta.columns) - set(F.META_COLS + F.META_COLS_OPTIONAL)))
    B["schema_ok"] = not (B["ego_missing"] or B["surrounding_missing"] or B["metadata_missing"]
                          or (isinstance(B["event_meta_missing"], list) and B["event_meta_missing"]))
    S["B_schema"] = B
    if B["ego_missing"] or "event_id" not in meta.columns:
        S["STOP"] = "required ego/metadata columns missing - audit stopped; send this summary"
        return S, pd.DataFrame()

    # ---- C identifiers ----
    tables = {"ego": ego_cols, "surrounding": sur_cols, "metadata": set(meta.columns), "event_meta": set(EM.columns),
              "environment": set(EN.columns)}
    hits = {k: sorted(c for c in cols if any(pat in c.lower() for pat in F.ID_PATTERNS)) for k, cols in tables.items()}
    hits = {k: v for k, v in hits.items() if v}
    S["C_identifiers"] = dict(identifier_columns_found=hits,
                              unit_rule=("DRIVER IDENTIFIER CANDIDATE FOUND - author must confirm before any split"
                                         if hits else "no driver identifier -> unit = event (trip), registered rule"))

    # ---- D counts ----
    trip_cat = T.groupby("event_id")["category"].first() if len(T) else pd.Series(dtype=str)
    base_ids = set(trip_cat[trip_cat == F.BASELINE_CATEGORY].index)
    inc = em[em["included"].astype(bool)] if len(em) else pd.DataFrame(columns=["sev", "impact_timestamp"])
    inc_ids = inc_all & set(trip_cat.index)
    D = dict(listed_trips=int(len(meta)), listed_by_category=meta["event_category"].value_counts().to_dict(),
             reconstructed_trips_in_h5=int(trip_cat.size),
             reconstructed_by_category=trip_cat.value_counts().to_dict(),
             metadata_flag_ego_reconstructed=int(meta.get("ego_reconstructed", pd.Series(dtype=float)).fillna(0).astype(bool).sum()),
             trips_in_more_than_one_file=int(T.event_id.duplicated().sum()) if len(T) else 0,
             ego_files_skipped_missing_columns=skipped,
             baseline_trips=len(base_ids),
             test_set_events=int(len(em)), test_set_included=int(len(inc)),
             included_events_with_trajectory=len(inc_ids),
             included_crash=int((inc["event_category"].map(outcome_of) == "crash").sum()) if len(inc) else 0,
             included_near_crash=int((inc["event_category"].map(outcome_of) == "near-crash").sum()) if len(inc) else 0,
             included_unmapped_category=int((inc["event_category"].map(outcome_of) == "UNMAPPED").sum()) if len(inc) else 0,
             outcome_rule="event category as in the GSSM code (C25); coded severities shown for information only",
             included_by_severity_info=dict(crash=int((inc["sev"] == 3).sum()), near_crash=int((inc["sev"] == 2).sum()),
                                            other=int((~inc["sev"].isin([2, 3])).sum())) if len(inc) else {},
             category_folder_vs_metadata_mismatch=int((T.groupby("event_id")["category"].first()
                                                       != meta.set_index("event_id")["event_category"].reindex(
                                                           T.groupby("event_id")["category"].first().index)).sum()) if len(T) else 0,
             excluded_by_filters=dict(duration_enough_false=int((~EM["duration_enough"].astype(str).str.lower().isin(["true", "1"])).sum()) if len(EM) else 0,
                                      conflict_none=int((EM["conflict"].astype(str) == "none").sum()) if len(EM) else 0),
             included_missing_impact_timestamp=int(inc["impact_timestamp"].isna().sum()) if len(inc) else 0)
    D["published_comparison"] = dict(
        test_set_included=[D["test_set_included"], F.PUBLISHED["test_set_events"]],
        crashes=[D["included_crash"], F.PUBLISHED["crashes"]], near_crashes=[D["included_near_crash"], F.PUBLISHED["near_crashes"]],
        baselines_reconstructed=[D["baseline_trips"], F.PUBLISHED["baselines_reconstructed"]],
        events_reconstructed=[int(trip_cat.size - len(base_ids)), F.PUBLISHED["events_reconstructed"]],
        note="[obtained, published]; differences are documented, never adjusted (registration Section 2)")
    S["D_counts"] = D
    unm = sorted(set(trip_cat) - set(F.OUTCOME_BY_CATEGORY))
    if unm or D["included_unmapped_category"] or D["category_folder_vs_metadata_mismatch"]:
        S["STOP"] = (f"event categories without an outcome type {unm} or folder/metadata category mismatch "
                     f"({D['category_folder_vs_metadata_mismatch']}) - send this summary")
    study = T[T.in_study]

    # ---- E lengths ----
    S["E_lengths"] = dict(
        duration_s_by_outcome={k: q(v) for k, v in study.groupby("outcome_type")["duration_s"]},
        time_grid_gaps_total=gaps_total, trips_with_gaps=int((T.gaps > 0).sum()),
        duplicate_grid_times_total=int(T.duplicate_times.sum()), trips_with_duplicate_times=int((T.duplicate_times > 0).sum()),
        share_frames_without_raw_timestamp=round(float(T.raw_timestamp_missing.sum() / max(T.n_frames.sum(), 1)), 4),
        note="duration = last minus first subject-vehicle timestamp; grid expected 100 ms")

    # ---- F retention ----
    ev = study[study.outcome_type.isin(["crash", "near-crash"])]
    R = {}
    for h in F.HORIZONS_S:
        share = float((ev["pre_event_s"] >= h + F.RETENTION_EXTRA_S).mean()) if len(ev) else float("nan")
        R[f"{h}s"] = dict(share=round(share, 4), kept=bool(share >= F.RETENTION_SHARE))
    R["primary_3s_kept"] = R["3s"]["kept"]
    R["pre_event_s"] = q(ev["pre_event_s"])
    S["F_retention"] = R

    # ---- G implausible ----
    S["G_implausible"] = dict(
        ego_frames_speed_gt60=int(T.ego_speed_gt60.sum()), ego_frames_acc_gt15_outside_crash_window=int(T.ego_acc_gt15.sum()),
        object_frames_speed_gt60=obj_speed, object_jumps_gt10m=obj_jump,
        trips_excluded_gt10pct=T[T.excluded_implausible].groupby("outcome_type").size().to_dict(),
        trips_excluded_total=int(T.excluded_implausible.sum()))

    # ---- H eligibility ----
    units = study[~study.excluded_implausible]
    H = dict(units_total=int(len(units)), units_eligible=int((units.n_eval3 >= F.MIN_EVAL_UNIT).sum()),
             eligible_by_outcome=units[units.n_eval3 >= F.MIN_EVAL_UNIT].groupby("outcome_type").size().to_dict(),
             eval_frames_total=int(units.n_eval3.sum()), pos_frames_total=int(units.n_pos3.sum()),
             neg_frames_total=int(units.n_neg3.sum()),
             events_with_positive_frame=int(((units.n_pos3 > 0) & (units.outcome_type != "baseline")).sum()),
             eval_frames_per_unit=q(units.n_eval3), negatives_per_unit=q(units.n_neg3))
    el = units[units.n_eval3 >= F.MIN_EVAL_UNIT]
    H["eligible_units_below_1_over_eps"] = {k: int((el.n_neg3 < 1.0 / v).sum()) for k, v in F.EPS_LEVELS.items()}
    H["baseline_hours_eligible"] = round(float(el[el.outcome_type == "baseline"].n_neg3.sum()) / 36000.0, 2)
    S["H_eligibility"] = H

    # ---- I environment ----
    I = dict(environment_rows=int(len(EN)), environment_duplicate_ids_dropped=dup_env)
    if len(EN):
        env = EN.set_index("event_id")
        I["baselines_with_environment"] = int(len(set(env.index) & base_ids))
        I["included_events_with_environment"] = int(len(set(env.index) & inc_ids))
        I["raw_values"] = {c: EN[c].astype(str).value_counts().to_dict() for c in F.ENV_COLS[1:] if c in EN}
        lev, unm = {}, {}
        ids = sorted(inc_ids & set(env.index))
        for f in F.ENV_COLS[1:]:
            vals = [binary_factor(f, env.at[i, f]) for i in ids]
            lev[f] = {"0": vals.count(0), "1": vals.count(1), "unknown": vals.count(None), "UNMAPPED": vals.count("UNMAPPED")}
            unm[f] = sorted({str(env.at[i, f]) for i in ids if binary_factor(f, env.at[i, f]) == "UNMAPPED"})
        I["factor_levels_included_events"] = lev; I["unmapped_values"] = unm
        cell = {}
        for i in ids:
            l, s_ = binary_factor("lighting", env.at[i, "lighting"]), binary_factor("surfaceCondition", env.at[i, "surfaceCondition"])
            k = f"L{l}xS{s_}"; cell[k] = cell.get(k, 0) + 1
        I["lighting_x_surface_cells_included_events"] = cell
    else:
        I["note"] = "no environment.csv found (public version or missing files)"
        I["baselines_with_environment"] = 0
    S["I_environment"] = I

    # ---- J objects ----
    S["J_objects"] = dict(share_frames_with_object_by_outcome={
        k: round(float(g.frames_with_object.sum() / max(g.n_frames.sum(), 1)), 4) for k, g in study.groupby("outcome_type")},
        trips_without_any_object_by_outcome={k: int((g.frames_with_object == 0).sum()) for k, g in study.groupby("outcome_type")},
        max_simultaneous_objects=int(T.max_objects.max()) if len(T) else 0,
        trips_with_more_than_8_objects_at_once=int((T.max_objects > 8).sum()))

    # ---- K reference point ----
    K = {}
    if k_along:
        al, di = np.concatenate(k_along), np.concatenate(k_dist)
        K = dict(matched_rows=int(len(al)), along_heading_m=q(al), distance_m=q(di),
                 note="test-set x_sur minus Surrounding x_ekf; ~0 -> same reference point; ~half a car length "
                      "-> test set uses the object centre")
    S["K_reference_point"] = K or dict(note="not computed (no event_data.h5 or no objects)")
    S["runtime_s"] = round(time.time() - t0, 1)
    return S, T


def write_outputs(S, T, out_results, out_local):
    out_results, out_local = Path(out_results), Path(out_local)
    out_results.mkdir(parents=True, exist_ok=True); out_local.mkdir(parents=True, exist_ok=True)
    js = json.dumps(S, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (out_results / "shrp2_audit_summary.json").write_text(js)
    lines = ["SHRP2 DATA AUDIT - AGGREGATE COUNTS ONLY (safe to share; contains no event IDs)", "=" * 78]
    for sec, val in S.items():
        lines.append(f"\n[{sec}]")
        if isinstance(val, dict):
            for k, v in val.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {val}")
    (out_results / "shrp2_audit_summary.txt").write_text("\n".join(lines))
    if len(T):
        T.to_parquet(out_local / "audit_trips.parquet", index=False)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-results", default="results")
    ap.add_argument("--out-local", default="data/processed/shrp2")
    a = ap.parse_args()
    if not str(Path(a.out_local).as_posix()).startswith("data/"):
        raise SystemExit("--out-local must be inside data/ (outside Git): trip-level files contain event IDs")
    S, T = audit(a.root)
    print(write_outputs(S, T, a.out_results, a.out_local))
    print(f"\nSaved {a.out_results}/shrp2_audit_summary.txt/.json (shareable) and {a.out_local}/audit_trips.parquet (LOCAL ONLY)")
