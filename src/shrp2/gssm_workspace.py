"""
GSSM WORKSPACE for H1 (Amendment 4, Sections 4-5): lays out the folders the authors' GSSM code expects
(github.com/Yiru-Jiao/GSSM, run from a working folder with relative paths), restricted to OUR registered split,
so that the authors' scripts run UNCHANGED on exactly our events.

  <ws>/ResultData/EventData/<category>/  event_meta.csv, environment.csv, event_data.h5 of the SafetyCriticalTestSet,
                                         restricted to the events of --split (validation or test)
  <ws>/ProcessedData/SHRP2/              metadata_birdseye.csv (those events + training-split baselines) and
                                         SafeBaseline/Ego_birdseye_<0-4>.h5, Surrounding_birdseye_<0-4>.h5 restricted
                                         to TRAINING-split baselines (only S-C, the secondary model, learns from them)
  <ws>/RawData/SHRP2/FileToUse/InsightTables/Event_Table.csv
                                         the authors' code reads the four environment categories from this raw SHRP2
                                         table, which the bird's-eye dataset does not ship; it is rebuilt from the
                                         dataset's environment.csv ("Unknown" where a trip has none, e.g. baselines).
                                         Only GSSM's environment features use it; h-C and S-C use current features.
Nothing in GSSM's code is edited. A manifest (<ws>/workspace_manifest.json) records counts and a hash of the event
list. The workspace holds licensed data: it must stay inside data/ (outside Git) on the encrypted laptop.
Usage: python src/shrp2/gssm_workspace.py --root data/raw/shrp2/extracted --split test
       (then run the authors' scripts from <ws> - see docs/SHRP2_runbook.md, GSSM section)
"""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shrp2_format as F
from audit_counts import find_dirs, read_h5

ENV = ["lighting", "weather", "surfaceCondition", "trafficDensity"]


def build(root, split_csv, split, ws, with_baselines=True, models_dir=None, models=("highD/current",)):
    rt, st = find_dirs(root)
    if st is None:
        raise SystemExit("SafetyCriticalTestSet not found - GSSM evaluation needs it")
    S = pd.read_csv(split_csv)
    ids = set(S.loc[S["split"] == split, "event_id"].astype(np.int64))
    base_tr = set(S.loc[(S["split"] == "train") & S["stratum"].str.startswith("baseline"), "event_id"].astype(np.int64))
    ws = Path(ws)
    if not ws.as_posix().startswith("data/") and not ws.is_absolute():
        raise SystemExit("workspace must be inside data/ (outside Git): it holds licensed data")
    if ws.exists():
        shutil.rmtree(ws)                                           # never mix two builds
    ev_dir = ws / "ResultData" / "EventData"
    kept, env_all, cats = [], [], {}
    for d in sorted(p for p in st.iterdir() if p.is_dir()):
        em = pd.read_csv(d / "event_meta.csv")
        em = em[em["event_id"].astype(np.int64).isin(ids)]
        if not len(em):
            continue
        out = ev_dir / d.name; out.mkdir(parents=True, exist_ok=True)
        em.to_csv(out / "event_meta.csv", index=False)
        if (d / "environment.csv").exists():
            en = pd.read_csv(d / "environment.csv"); en = en[en["event_id"].isin(em["event_id"])]
            en.to_csv(out / "environment.csv", index=False); env_all.append(en)
        ed = read_h5(d / "event_data.h5")
        ed = ed[ed["event_id"].astype(np.int64).isin(set(em["event_id"]))]
        ed.to_hdf(out / "event_data.h5", key=F.H5_KEY)
        kept += em["event_id"].astype(np.int64).tolist(); cats[d.name] = int(len(em))
    if not kept:
        raise SystemExit(f"no {split} events found in the SafetyCriticalTestSet")
    # ProcessedData: metadata for kept events + training baselines; baseline trajectories (training split only)
    pdir = ws / "ProcessedData" / "SHRP2"; pdir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(rt / "metadata_birdseye.csv")
    keep_meta = set(kept) | (base_tr if with_baselines else set())
    meta[meta["event_id"].astype(np.int64).isin(keep_meta)].to_csv(pdir / "metadata_birdseye.csv", index=False)
    nb = 0
    if with_baselines:
        bdir = pdir / F.BASELINE_CATEGORY; bdir.mkdir(exist_ok=True)
        src = rt / F.BASELINE_CATEGORY
        for kind in (F.EGO_FILE, F.SUR_FILE):
            for c in range(5):
                f = src / f"{kind}_{c}.h5"
                d = read_h5(f) if f.exists() else pd.DataFrame(columns=["event_id"])
                d = d[d["event_id"].astype(np.int64).isin(base_tr)]
                d.to_hdf(bdir / f"{kind}_{c}.h5", key=F.H5_KEY)
                if kind == F.EGO_FILE:
                    nb += int(d["event_id"].nunique()) if len(d) else 0
    # RawData: environment table the authors' code reads (eventID + four categories)
    raw = ws / "RawData" / "SHRP2" / "FileToUse" / "InsightTables"; raw.mkdir(parents=True, exist_ok=True)
    env = pd.concat(env_all, ignore_index=True) if env_all else pd.DataFrame(columns=["event_id"] + ENV)
    tab = pd.DataFrame({"eventID": sorted(keep_meta)}).merge(env.rename(columns={"event_id": "eventID"}), on="eventID", how="left")
    for c in ENV:
        tab[c] = tab[c].fillna("Unknown") if c in tab else "Unknown"
    tab[["eventID"] + ENV].to_csv(raw / "Event_Table.csv", index=False)
    models_done = add_models(ws, models_dir, models) if models_dir else []
    man = dict(split=split, events=len(kept), events_by_category=cats, training_baselines=nb, gssm_models=models_done,
               event_list_sha256=hashlib.sha256(",".join(map(str, sorted(kept))).encode()).hexdigest(),
               environment_rows_unknown={c: int((tab[c] == "Unknown").sum()) for c in ENV})
    (ws / "workspace_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return man


def add_models(ws, models_dir, models):
    """Copy trained GSSM models (<models_dir>/<dataset>/<encoder>/*.pth, the authors' PreparedData/PosteriorInference
    layout) into the workspace and write the evaluation.csv that evaluate_safety.py loops over (one row per model,
    columns as the authors' file; mixrate empty = single-dataset model)."""
    out = Path(ws) / "PreparedData" / "PosteriorInference"; rows = []
    for m in models:
        dataset, encoder = m.split("/")
        src = Path(models_dir) / dataset / encoder
        pth = sorted(src.glob("*.pth"))
        if len(pth) != 1:
            raise SystemExit(f"expected exactly one .pth in {src}, found {len(pth)}")
        dst = out / dataset / encoder; dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pth[0], dst / pth[0].name)
        rows.append(dict(dataset=dataset, encoder_selection=encoder))
    cols = ["dataset", "encoder_selection", "val_loss", "num_samples", "num_epochs_trained", "training_time",
            "whole_model", "current_encoder", "environment_encoder", "profiles_encoder", "attention_decoder", "mixrate"]
    pd.DataFrame(rows).reindex(columns=cols).to_csv(out / "evaluation.csv", index=False)
    return list(models)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--split-csv", default="data/processed/shrp2/split.csv")
    ap.add_argument("--ws", default=None)
    ap.add_argument("--no-baselines", action="store_true", help="skip the training baselines (h-C only)")
    ap.add_argument("--models-dir", default=None, help="folder in the authors' PreparedData/PosteriorInference layout")
    ap.add_argument("--models", default="highD/current", help="comma-separated <dataset>/<encoder> (h-C = highD/current)")
    a = ap.parse_args()
    build(a.root, a.split_csv, a.split, a.ws or f"data/gssm_ws_{a.split}", not a.no_baselines, a.models_dir,
          tuple(a.models.split(",")))
