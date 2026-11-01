"""
Tests of src/shrp2/report_p2_extra.py on the FABRICATED workspace of test_confirm_p2.py; expected values computed
independently (scikit-learn, the exact comparator code, hand loops). No SHRP2 data, no training.
Usage : python src/shrp2/test_report_p2_extra.py      (~2-5 min, temporary folder only)
"""
import os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train_shrp2 as T
import confirm_p2 as CP
import calibration as C
import test_confirm_p2 as TC
import report_p2_extra as X
from phase2 import run_h5 as RH, domain_crc as D, comparators as K
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
RES = []
SEEDS = T.SEEDS


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="p2extra_")); cwd = os.getcwd()
    try:
        os.chdir(tmp); rng = np.random.default_rng(5)
        Tr, metas, hists = TC.build(tmp, rng)
        bank = CP.Bank()
        # ---------------- Y1 Part A against scikit-learn ----------------
        A = pd.DataFrame(X.part_a(bank)).set_index(["method", "horizon_s"])
        ok = True
        for k, h in enumerate(CP.HORIZONS):
            y_t, kn = C.labels_at(metas["test"].event_bin.to_numpy(), metas["test"].obs_bins.to_numpy(), k)
            ix = np.flatnonzero(kn); y = y_t[ix]
            r = [bank.model(f"central_hazard_aug_s{s}")["test"][0][ix, k] for s in SEEDS]
            ok &= np.isclose(A.loc[("central model (aug)", h), "auprc_mean"], np.mean([AP(y, x) for x in r]))
            ok &= np.isclose(A.loc[("central model (aug)", h), "auroc_max"], max(AUC(y, x) for x in r))
            e = []
            for x in r:
                o = np.argsort(x, kind="mergesort"); e.append(sum(len(c) / len(x) * abs(y[c].mean() - x[c].mean()) for c in np.array_split(o, 10)))
            ok &= np.isclose(A.loc[("central model (aug)", h), "ece_mean"], np.mean(e))
            fz = [bank.fed("fedavg", s)["test"][0][ix, k] for s in SEEDS]; sub = np.isfinite(fz[0])
            ok &= np.isclose(A.loc[("federated: fedavg", h), "auroc_mean"], np.mean([AUC(y[sub], x[sub]) for x in fz]))
            ok &= int(A.loc[("federated: fedavg", h), "frames"]) == int(sub.sum())
            ok &= np.isnan(A.loc[("TTC (chosen on validation)", h), "ece_mean"])
        missing = ("federated: central", 3) not in A.index
        check("Y1 Part A: AUPRC, AUROC (scikit-learn) and adaptive ECE (hand) per horizon; federated metrics on client "
              "frames only; no ECE for TTC; regimes without prediction files are skipped", ok and missing)
        # ---------------- Y2 C3 normal approximation against the exact recursion ----------------
        r2 = np.random.default_rng(1)
        sizes = tuple(sorted(r2.integers(40, 400, 150)))
        dmax = max(float(np.max(np.abs(X.m_lk_normal(sizes, l) - K._m_lk_all_k(list(sizes), l)))) for l in (1, 5, 20, 40))
        sc = [r2.random(n) for n in r2.integers(40, 400, 120)]
        ex, nm = K.c3_humbert_qq(sc, 0.05), X.c3_humbert_qq_normal(sc, 0.05)
        check("Y2 C3: normal approximation of the order-statistic law within 0.005 of the exact recursion (150 clients); "
              "threshold close to the exact method's", dmax < 0.005 and abs(ex["lam"] - nm["lam"]) < 0.02,
              f"max |dM| {dmax:.5f}; lam exact {ex['lam']:.4f} vs normal {nm['lam']:.4f}")
        # ---------------- Y3 Part B: deployment cross-check and comparators ----------------
        CP.h5_input(); inp = dict(np.load(T.SH / "h5_input_shrp2.npz", allow_pickle=True))
        RH.analyse(inp, "shrp2", 5)
        B = X.part_b(min_eval=5).set_index(["level", "method"])
        units, _ = RH.build_units(inp["unit"], inp["split"], inp["known"], inp["y"], {s: inp[f"score_s{s}"] for s in SEEDS}, 5)
        vid = sorted(u for u in units if units[u]["split"] == "val"); tid = sorted(u for u in units if units[u]["split"] == "test")
        sh = []
        for s in SEEDS:
            lam = K.c2_lu_fcp([units[u]["neg"][s] for u in vid], 0.05)["lam"]
            sh.append(np.mean([np.mean(units[u]["neg"][s] > lam) > 0.05 for u in tid]))
        ok3 = np.isclose(B.loc[("0.05", "C2"), "share_over"], np.mean(sh)) and B.note.iloc[0].startswith("HP, CRC and C1 reproduce")
        ok3 &= ("op", "CRC") not in B.index and all((lv, m) in B.index for lv in ("0.05", "0.10", "op") for m in ("HP", "C1", "C2", "C3", "C4", "C5"))
        sec = pd.read_csv(tmp / "results/h5_shrp2_secondary.csv")
        sec.loc[sec.analysis == "descriptive DEPLOY", "recall"] += 0.01; sec.to_csv(tmp / "results/h5_shrp2_secondary.csv", index=False)
        try:
            X.part_b(min_eval=5); bites = False
        except SystemExit:
            bites = True
        check("Y3 Part B: HP, CRC, C1 reproduce the registered deployment rows (check bites when altered); C2 = Lu et al. "
              "threshold from validation trips applied to test trips (hand); CRC omitted at eps_op", ok3 and bites)
        # ---------------- Y4 Part C: alpha chosen on validation only ----------------
        a1 = B.loc[("0.05", "C1 recall-matched (post-hoc)"), "alpha_star"]
        inp2 = dict(inp); te = np.asarray(inp["split"]).astype(str) == "test"
        for s in SEEDS:
            z = np.array(inp2[f"score_s{s}"], float); z[te] = rng.random(te.sum()); inp2[f"score_s{s}"] = z
        np.savez_compressed(tmp / "alt.npz", **inp2)
        (tmp / "results/h5_shrp2_secondary.csv").unlink()
        B2 = X.part_b(h5_path=tmp / "alt.npz", min_eval=5).set_index(["level", "method"])
        val = []
        for u in vid:
            val.append(dict(neg=units[u]["neg"][SEEDS[0]], pos=units[u]["pos"][SEEDS[0]]))
        curves = np.array([D.loss_curve(v["neg"]) for v in val]); hp = D.hp_threshold(curves, 0.05, 0.10)["lam"]
        pos = np.concatenate([v["pos"] for v in val]); negs = np.concatenate([v["neg"] for v in val]); tgt = (pos > hp).mean()
        best, ba = None, None
        for a in np.round(np.arange(0.001, 0.0505, 0.001), 3):
            d = abs((pos > D.pooled_split_conformal(negs, a)).mean() - tgt)
            if best is None or d <= best + 1e-15:
                best, ba = d, a
        a_s1, _, _ = X.matched_alpha(val, hp)
        check("Y4 Part C: recall-matched alpha depends on validation trips only (unchanged when test scores are replaced); "
              "equals a hand search (ties -> larger alpha)", np.isclose(a1, B2.loc[("0.05", "C1 recall-matched (post-hoc)"), "alpha_star"])
              and np.isclose(a_s1, ba), f"alpha* seed mean {a1:.4f}; seed 131 {a_s1}")
        # ---------------- Y5 registered run only; aggregate output ----------------
        r_ = subprocess.run([sys.executable, str(HERE / "report_p2_extra.py"), "a"], capture_output=True, text=True,
                            env=dict(os.environ, SHRP2_RUN="nomask"))
        r1 = subprocess.run([sys.executable, str(HERE / "report_p2_extra.py"), "a"], capture_output=True, text=True)
        cols = set(pd.read_csv(tmp / "results/shrp2_p2_extra_A.csv").columns) if r1.returncode == 0 else {"x"}
        check("Y5 refuses SHRP2_RUN; output written with aggregate columns only (no trip or frame identifiers)",
              r_.returncode != 0 and r1.returncode == 0 and not {"event_id", "unit", "frame", "trip"} & cols, r1.stderr[-300:])
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
