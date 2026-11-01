"""
Tests of the S-C retraining path (Amendment 4, Section 5): training-split GSSM workspace, the authors' segmentation
(gssm_segment_sc.py) and the authors' UNCHANGED training script, then S-C scored by evaluate_safety in the test
workspace and read by the P1 engine. Synthetic SHRP2-format data only.
Run with GSSM's environment:  <GPY> src/shrp2/test_gssm_sc.py --gssm baselines/GSSM      (~5-10 min)
"""
import argparse, hashlib, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "loaders"))
import shrp2_format as F
RES = []


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def ids_in_ws(ws):
    """Every event ID a workspace holds, by place."""
    out = {}
    out["metadata"] = set(pd.read_csv(ws / "ProcessedData/SHRP2/metadata_birdseye.csv")["event_id"])
    ch = []
    for f in (ws / "ProcessedData/SHRP2/SafeBaseline").glob("*.h5"):
        d = pd.read_hdf(f, "data")
        ch += list(d["event_id"].unique()) if len(d) else []
    out["baseline_chunks"] = set(ch)
    out["event_table"] = set(pd.read_csv(ws / "RawData/SHRP2/FileToUse/InsightTables/Event_Table.csv")["eventID"])
    out["event_data"] = set(pd.concat([pd.read_csv(p) for p in (ws / "ResultData/EventData").glob("*/event_meta.csv")])["event_id"])
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(repo):
    repo = os.path.abspath(repo)
    import make_synthetic as MK, shrp2 as LD, make_split as MS, gssm_workspace as GW, gssm_features as GF, gssm_segment_sc as SC
    import p1_eval as P
    tmp = Path(tempfile.mkdtemp(prefix="gssmsc_")); cwd = os.getcwd()
    try:
        root = MK.build(tmp / "syn", "complete", scale=5)
        work = tmp / "data/processed/shrp2"; LD.main(root, work, mask=True)
        MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        S = pd.read_csv(work / "split.csv")
        train_ids, other_ids = set(S[S.split == "train"].event_id), set(S[S.split != "train"].event_id)
        # plant object sizes: training events vary, validation/test events 9.9 m (must never reach S-C)
        mf = root / F.RT_DIR / "metadata_birdseye.csv"; meta = pd.read_csv(mf)
        ev_tr = meta.event_id.isin(train_ids) & meta.target_length.notna()
        meta.loc[ev_tr, "target_length"] = np.round(np.linspace(3.5, 6.0, int(ev_tr.sum())), 2)
        meta.loc[meta.event_id.isin(other_ids) & meta.target_length.notna(), "target_length"] = 9.9
        meta.to_csv(mf, index=False)
        lengths_tr = set(meta.loc[ev_tr, "target_length"])

        # ---------------- S1 the wrapper's loop body is the authors' code, byte for byte ----------------
        src = Path(repo, "src_data_preparation", "segment_datasets.py").read_text()
        body = src[src.index("        path_save = f'{path_prepared}{dataset}/'"):src.index("    print('--- Total time")].rstrip("\n") + "\n"
        check("S1 gssm_segment_sc.py contains the authors' segment_datasets.main loop body byte for byte",
              body in Path(HERE, "gssm_segment_sc.py").read_text(), f"{len(body.splitlines())} lines")

        # ---------------- S2 training-split workspace holds training-split trips only ----------------
        wtr = tmp / "data/gssm_ws_train"; GW.build(root, work / "split.csv", "train", wtr)
        got = ids_in_ws(wtr)
        base_tr = set(S[(S.split == "train") & S.stratum.str.startswith("baseline")].event_id)
        check("S2 training workspace: metadata, baseline chunks, environment table and events are ALL training-split trips",
              all(v <= train_ids for v in got.values()) and got["baseline_chunks"] == base_tr and len(got["event_data"]) > 0,
              ", ".join(f"{k} {len(v)}" for k, v in got.items()))
        wte = tmp / "data/gssm_ws_test"; GW.build(root, work / "split.csv", "test", wte)
        os.chdir(wte)
        try:
            SC.main(repo); refused = False
        except SystemExit:
            refused = True
        os.chdir(cwd)
        check("S3 the segmentation refuses to run in a validation or test workspace", refused)

        # ---------------- S4-S6 segmentation in the training workspace ----------------
        os.chdir(wtr)
        try:
            n = SC.main(repo)
            seg = wtr / "PreparedData/Segments/SafeBaseline"
            cur = {k: pd.read_hdf(seg / f"current_features_SafeBaseline_{k}.h5", "features") for k in ("train", "val")}
            env = {k: pd.read_hdf(seg / f"environment_features_SafeBaseline_{k}.h5", "features") for k in ("train", "val")}
            sys.path.insert(0, os.path.join(repo, "src_data_preparation"))
            from represent_utils.utils_data_segmentation import read_dataset
            eids = read_dataset("SafeBaseline", "./ProcessedData/", 131)["event_id"].unique()
            tr_e = set(np.random.RandomState(131).choice(eids, int(0.8 * len(eids)), replace=False)); va_e = set(eids) - tr_e
            check("S4 scenes come from training-split baselines only, divided 80/20 by the authors' rule (RandomState(131))",
                  set(env["train"].event_id) <= tr_e and set(env["val"].event_id) <= va_e and set(eids) <= base_tr
                  and n["train"] > 0 and n["val"] > 0, f"{n['train']} train / {n['val']} val scenes")
            ls = set(np.round(pd.concat(cur.values())["l_sur"], 2))
            check("S5 random object sizes are drawn from training events only (planted 9.9 m of validation/test events absent)",
                  ls <= lengths_tr and 9.9 not in ls and len(ls) > 1, f"{len(ls)} distinct lengths")
            keep = {k: v.copy() for k, v in cur.items()}
            shutil.rmtree(seg); SC.main(repo)
            check("S6 segmentation is reproducible (re-run gives identical scenes)",
                  all(keep[k].equals(pd.read_hdf(seg / f"current_features_SafeBaseline_{k}.h5", "features")) for k in keep))
            # ---------------- S7 the authors' UNCHANGED training script ----------------
            env_ = dict(os.environ); env_["PYTHONPATH"] = repo
            p = subprocess.run([sys.executable, os.path.join(repo, "src_posterior_inference", "pi_train_eval.py"), "--stage", "1"],
                               capture_output=True, text=True, env=env_)
        finally:
            os.chdir(cwd)
        pth = sorted((wtr / "PreparedData/PosteriorInference/SafeBaseline/current").glob("*.pth"))
        ev = pd.read_csv(wtr / "PreparedData/PosteriorInference/evaluation.csv")
        row = ev[(ev.dataset == "SafeBaseline") & (ev.encoder_selection == "current")]
        check("S7 authors' pi_train_eval.py --stage 1, unchanged: S-C trained (one .pth, evaluation row with all scenes)",
              p.returncode == 0 and len(pth) == 1 and len(row) == 1 and int(row.num_samples.iloc[0]) == n["train"] + n["val"],
              (f"{int(row.num_epochs_trained.iloc[0])} epochs, {row.training_time.iloc[0]:.0f} s" if len(row) else p.stderr[-300:]))

        # ---------------- S8 S-C scored in the test workspace and read by the P1 engine ----------------
        mdir = tmp / "data/gssm_models/SafeBaseline/current"; mdir.mkdir(parents=True)
        shutil.copy2(pth[0], mdir / pth[0].name)
        GW.build(root, work / "split.csv", "test", wte, models_dir=tmp / "data/gssm_models", models=("SafeBaseline/current",))
        placed = list((wte / "PreparedData/PosteriorInference/SafeBaseline/current").glob("*.pth"))
        os.chdir(wte)
        try:
            GF.main(repo)
            p = subprocess.run([sys.executable, os.path.join(repo, "src_safety_evaluation", "evaluate_safety.py")],
                               capture_output=True, text=True, env=env_)
        finally:
            os.chdir(cwd)
        out = wte / "ResultData/EventEvaluation/SafeBaseline_current.h5"
        ok = out.exists() and len(placed) == 1 and sha(placed[0]) == sha(pth[0])
        if ok:
            UE, UM = P.gssm(repo)
            meta8, data8 = P.load_events(wte, UE)
            g = pd.read_hdf(out, "data")
            M = P.count_matrices(P.run_records(UE, g, data8, meta8, "GSSM", P.GRIDS["GSSM"][::10]), np.sort(meta8.index.unique()))
            ok = {"event_id", "target_id", "time", "intensity"} <= set(g.columns) and np.all(M["tp"] + M["fn"] == M["recorded"][None, :])
        check("S8 the trained S-C, copied unchanged into the test workspace, is scored by evaluate_safety and read by the P1 engine",
              ok, "" if ok else p.stderr[-300:])
        check("S9 nothing of S-C's training was done in the test workspace (no segments there)",
              not (wte / "PreparedData/Segments").exists())
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gssm", default="baselines/GSSM")
    sys.exit(0 if main(ap.parse_args().gssm) else 1)
