"""
PROTOCOL P1 (GSSM evaluation) for H1 - Amendment 4. Every method is evaluated by the AUTHORS' OWN functions
(github.com/Yiru-Jiao/GSSM, src_safety_evaluation/validation_utils): parallel_records (first-stage records per
event and threshold), evaluate (third stage, one shared conflicting object), issue_warning, get_statistics,
get_auprc. Nothing in them is changed; this module only feeds them and sums their records.

Methods with one score per subject frame (proposed model, LightGBM, TTC / DRAC / TTC2D) get that score copied to
every object row (event x object x time) of the test set's event_data at that frame (Amendment 4, Section 3);
GSSM keeps its own pairwise intensity. Missing scores stay NaN and never exceed a threshold.

Indicator mapping (GSSM's determine_conflicts): 'GSSM' -> column 'intensity', larger = riskier (proposed model,
LightGBM, GSSM); 'TTC2D' -> column 'TTC2D', smaller = riskier (TTC, TTC2D); 'EI' -> column 'EI', larger = riskier
(DRAC). Threshold grids: Amendment 4, Section 2 (GRIDS below).
Bootstrap (Amendment 4, Section 6): clusters = test events (src/analysis/stats.py ClusterBootstrap, B 2,000, seed
131); per resample the per-threshold counts are re-summed with the event multiplicities and AUPRC recomputed with
get_auprc; delta = mean over seeds of the model's AUPRC minus the comparator's; a resample without any true positive
is undefined (discarded, counted).
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analysis import stats as ST

EPS = 1e-6
GRIDS = {
    "GSSM": np.unique(np.round(np.arange(0, 6, 0.06) - 0.06, 2)),                    # GSSM's own grid
    "TTC2D": np.unique(np.round(np.arange(0, 1.94, 0.0135) ** 7, 2)),                # GSSM's TTC2D grid
    "EI_DRAC": np.round(np.arange(0, 10.0 + 1e-9, 0.05), 2),                         # DRAC 0-10 m/s^2
    "RISK": 1.0 / (1.0 + np.exp(-np.round(np.arange(-10, 10 + 1e-9, 0.1), 1))),      # model / LightGBM risk
}
COLUMN = {"GSSM": "intensity", "TTC2D": "TTC2D", "EI": "EI"}


def gssm(repo):
    """Import the authors' evaluation functions from the GSSM repository."""
    sys.path.insert(0, os.path.abspath(repo))
    from src_safety_evaluation.validation_utils import utils_evaluation as UE, utils_eval_metrics as UM
    return UE, UM


def load_events(ws, UE):
    """event_meta (with GSSM's danger_start/danger_end, as in analyse_events.py) and event_data of a workspace."""
    event_meta, event_data = UE.read_events(str(Path(ws) / "ResultData" / "EventData") + "/")
    # analyse_events.py, lines 46-49 (unchanged formula)
    event_meta["danger_start"] = np.minimum(event_meta["impact_timestamp"].values - 4500, event_meta["start_timestamp"].values)
    event_meta["danger_end"] = np.minimum(event_meta["impact_timestamp"].values + 500, event_meta["end_timestamp"].values)
    return event_meta, event_data


def frame_scores_to_rows(event_data, frame_scores, indicator):
    """frame_scores: DataFrame(event_id, frame, score) at 0.1-s frames (frame = round(time * 10)).
    Returns the safety_evaluation table GSSM's functions read: event_id, target_id, time, <indicator column>."""
    r = event_data.reset_index()[["event_id", "target_id", "time"]].copy()
    r["frame"] = np.round(r["time"].to_numpy(float) * 10).astype(np.int64)
    fs = frame_scores[["event_id", "frame", "score"]].copy()
    fs["event_id"] = fs["event_id"].astype(np.int64); fs["frame"] = fs["frame"].astype(np.int64)
    if fs.duplicated(["event_id", "frame"]).any():
        raise ValueError("frame_scores has duplicate (event_id, frame) rows")
    r = r.merge(fs, on=["event_id", "frame"], how="left").drop(columns="frame")
    return r.rename(columns={"score": COLUMN[indicator]})


def run_records(UE, seval, event_data, event_meta, indicator, thresholds, n_jobs=1, voted=None):
    """First stage (voted=None): GSSM parallel_records per threshold. Third stage: GSSM evaluate with the voted
    targets (DataFrame indexed by event_id with column target_id). Returns concatenated records (event_id column)."""
    if voted is None:
        fn = lambda th: UE.parallel_records(th, seval, event_data, event_meta, indicator)
    else:
        fn = lambda th: UE.evaluate(indicator, th, seval, event_data, event_meta, voted)
    if n_jobs == 1:
        out = [fn(th) for th in thresholds]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs)(delayed(fn)(th) for th in thresholds)
    rec = pd.concat(out)
    rec.index.name = "event_id"
    return rec.reset_index()


def _flag(s):
    """Boolean flag column of GSSM's records; missing (NaN / None) = False."""
    return s.map(lambda v: False if pd.isna(v) else bool(v)).astype(bool)


def count_matrices(rec, events):
    """Per threshold x event: TP, FN (danger-recorded events) and FP, TN (safe objects), as get_statistics counts."""
    rec = rec.copy()
    dr = _flag(rec["danger_recorded"]); sr = _flag(rec["safety_recorded"])
    tw = pd.to_numeric(rec["true_warning"], errors="coerce")
    rec["tp"] = (dr & (tw > 0.5)).astype(float); rec["fn"] = (dr & (tw < 0.5)).astype(float)
    rec["fp"] = np.where(sr, pd.to_numeric(rec["num_false_warning"], errors="coerce").fillna(0), 0.0)
    rec["tn"] = np.where(sr, pd.to_numeric(rec["num_true_non_warning"], errors="coerce").fillna(0), 0.0)
    ths = np.sort(rec["threshold"].unique())
    M = {}
    for k in ("tp", "fn", "fp", "tn"):
        M[k] = rec.pivot_table(index="threshold", columns="event_id", values=k, aggfunc="sum", fill_value=0.0) \
                  .reindex(index=ths, columns=events, fill_value=0.0).to_numpy(float)
    M["recorded"] = dr.groupby(rec["event_id"]).any() \
                       .reindex(events, fill_value=False).to_numpy()
    M["safety"] = sr.groupby(rec["event_id"]).any() \
                     .reindex(events, fill_value=False).to_numpy()
    M["thresholds"] = ths
    return M


def auprc(M, UM, w=None):
    """AUPRC from count matrices with event multiplicities w (None = original sample), via GSSM's get_auprc.
    NaN when undefined (no event with a recorded danger or safe period, or no true positive at any threshold)."""
    w = np.ones(M["tp"].shape[1]) if w is None else np.asarray(w, float)
    if not ((M["recorded"] | M["safety"]) * w).any():
        return np.nan
    tp, fn, fp = M["tp"] @ w, M["fn"] @ w, M["fp"] @ w
    if tp.max() <= 0:
        return np.nan
    recall = tp / np.maximum(EPS, tp + fn); precision = tp / np.maximum(EPS, tp + fp)
    return float(UM.get_auprc(recall, precision))


def h1_test(UM, model_mats, comp_mat, events, B=2000, seed=131):
    """Registered paired cluster bootstrap over test events: delta = mean_s AUPRC(model_s) - AUPRC(comparator)."""
    boot = ST.ClusterBootstrap(np.asarray(events).astype(str), B=B, seed=seed)
    n = len(events)

    def stat(idx):
        w = np.bincount(idx, minlength=n)
        a = [auprc(M, UM, w) for M in model_mats]; c = auprc(comp_mat, UM, w)
        return float(np.mean(a) - c) if np.all(np.isfinite(a)) and np.isfinite(c) else np.nan
    res = ST.paired_test(boot, stat, 0.0)
    res["n_events_recorded_model"] = int(min(M["recorded"].sum() for M in model_mats))
    res["n_events_recorded_comparator"] = int(comp_mat["recorded"].sum())
    res["testable"] = min(res["n_events_recorded_model"], res["n_events_recorded_comparator"]) >= 50
    return res


def mtti(UE, UM, M, seval, event_meta, indicator):
    """mTTI* (GSSM): threshold with the highest F1 on the original sample, issue_warning there, TTI = impact time -
    warning time (s), median over events with a warning and TTI < 10 s; exact sign-test interval (alpha 0.01, as
    the paper states; the code's helper default is 0.05)."""
    tp, fn, fp = M["tp"].sum(1), M["fn"].sum(1), M["fp"].sum(1)
    f1 = tp / np.maximum(EPS, tp + 0.5 * (fp + fn))
    th = float(M["thresholds"][int(np.argmax(f1))])
    w = UE.issue_warning(indicator, th, seval, event_meta)
    if "warning_timestamp" not in w:                  # GSSM adds this column only when some event records a danger period
        w = w.assign(warning_timestamp=np.nan)
    tti = (w["impact_timestamp"] - w["warning_timestamp"]) / 1000.0
    tti = tti[np.isfinite(tti) & (tti < 10)]
    lo, hi = UM._median_ci_bounds(tti, alpha=0.01) if len(tti) else (np.nan, np.nan)
    return dict(threshold=th, f1=float(f1.max()), mTTI=float(np.median(tti)) if len(tti) else np.nan,
                ci99=(float(lo), float(hi)), n=int(len(tti)))
