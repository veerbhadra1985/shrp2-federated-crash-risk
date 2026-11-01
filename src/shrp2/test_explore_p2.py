"""
Tests of the exploratory analyses E1b step 1 (explore_e1_scores.py) and E3-E5 (explore_p2.py) on the FABRICATED
workspace of test_confirm_p2.py, with planted dropout patterns and spread frames, so every expected value is computed
independently (hand loops, scikit-learn AUPRC, pandas). No SHRP2 data, no training.
Usage : python src/shrp2/test_explore_p2.py      (~2-4 min, temporary folder only)
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train_shrp2 as T
import confirm_p2 as CP
import calibration as C
import test_confirm_p2 as TC
import explore_p2 as E
import explore_e1_scores as E1
from phase2 import run_h5 as RH, domain_crc as DC
from sklearn.metrics import average_precision_score as AP
RES = []
SEEDS = T.SEEDS


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plant_dropout(Tr, mt):
    """Planted objects in the test histories (subject at x = 0): object 2 interior gap (frames 5-9 missing) at 27 m;
    object 3 leaving (15-19 missing) at 5 m; object 4 both edges (0-2 and 17-19 missing) at 8 m."""
    f = T.CACHE / "shrp2_test_hist.npy"; H = np.load(f).copy()
    noobj = Tr.set_index("event_id")["noobj"].reindex(mt.event_id).to_numpy()
    ev = mt.event_id.to_numpy()
    for i, e in enumerate(ev):
        if noobj[i]:
            continue
        if e % 7 == 0:
            H[i, 2, :, 8] = 1; H[i, 2, 5:10, 8] = 0; H[i, 2, :, 0] = np.where(H[i, 2, :, 8] > 0, 27.0, 0.0)
        if e % 11 == 0:
            H[i, 3, :, 8] = 1; H[i, 3, 15:, 8] = 0; H[i, 3, :, 0] = np.where(H[i, 3, :, 8] > 0, 5.0, 0.0)
        if e % 13 == 0:
            H[i, 4, :, 8] = 1; H[i, 4, :3, 8] = 0; H[i, 4, 17:, 8] = 0; H[i, 4, :, 0] = np.where(H[i, 4, :, 8] > 0, 8.0, 0.0)
    np.save(f, H)
    return H


def hand_classes(H):
    """Plain loops over samples and objects."""
    cnt = dict(dropout=0, entering=0, leaving=0, both=0, interior=0, far=0)
    drop_s, inter_s = np.zeros(len(H), bool), np.zeros(len(H), bool)
    for i in range(len(H)):
        for j in range(1, 9):
            v = H[i, j, :, 8] > 0
            if not v.any() or (1 - v.mean()) < 0.2:
                continue
            cnt["dropout"] += 1; drop_s[i] = True
            pres = np.flatnonzero(v); a, b = pres[0], pres[-1]
            inner = (~v[a:b + 1]).any()
            if inner:
                cnt["interior"] += 1; inter_s[i] = True
            elif a > 0 and b < 19:
                cnt["both"] += 1
            elif a > 0:
                cnt["entering"] += 1
            elif b < 19:
                cnt["leaving"] += 1
            if np.hypot(H[i, j, b, 0] - H[i, 0, b, 0], H[i, j, b, 1] - H[i, 0, b, 1]) > 25:
                cnt["far"] += 1
    return cnt, drop_s, inter_s


def hand_episodes(frames_alarm, frames_all):
    """Runs of consecutive alarm samples (adjacent in frame order among the trip's samples), merged when fewer than 10
    frames separate them."""
    f = np.sort(frames_alarm)
    if len(f) == 0:
        return 0
    runs = [[f[0], f[0]]]
    for x in f[1:]:
        if x - runs[-1][1] - 1 < 10:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return len(runs)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="explp2_")); cwd = os.getcwd()
    try:
        os.chdir(tmp); rng = np.random.default_rng(3)
        Tr, metas, hists = TC.build(tmp, rng)
        pd.DataFrame(dict(event_id=Tr.event_id, split=Tr.split)).to_csv(T.SH / "split.csv", index=False)
        B = 199
        # ---------------- X0 episode rule on hand cases ----------------
        cases = [([1, 2, 3, 14, 30, 31], 3), ([1, 11], 1), ([1, 12], 2), ([], 0), ([5], 1), ([1, 3, 5, 7, 40], 2)]
        check("X0 E5 episode rule: new episode when the frame gap > 10 (fewer than 10 frames between runs -> merged)",
              all(E.episodes(f) == n == hand_episodes(f, None) for f, n in cases), str([E.episodes(f) for f, _ in cases]))
        # ---------------- X1 E1b scores ----------------
        for grp, s in (("p1_val", "val"), ("p1_test", "test")):
            m = metas[s][["event_id", "frame"]].iloc[:300].reset_index(drop=True)
            m.to_parquet(T.CACHE / f"{grp}_meta.parquet", index=False)
            np.save(T.CACHE / f"{grp}_hist.npy", hists[s][:300])
        names = [f"central_hazard_aug_s{s}" for s in SEEDS]
        T.p1_scores(names)
        reg_hash = {n: sha(T.PRED / f"p1_{n}.parquet") for n in names}
        E1.p1_scores_at(names, T.K3, "_chk")
        same = all(pd.read_parquet(T.PRED / f"p1_{n}_chk.parquet").equals(pd.read_parquet(T.PRED / f"p1_{n}.parquet")) for n in names)
        E1.main()
        a, b = pd.read_parquet(T.PRED / f"p1_{names[0]}.parquet"), pd.read_parquet(T.PRED / f"p1_{names[0]}_h5s.parquet")
        mono = a[["event_id", "frame", "split"]].equals(b[["event_id", "frame", "split"]]) and (b.score.to_numpy() >= a.score.to_numpy() - 1e-7).all() \
            and (b.score.to_numpy() > a.score.to_numpy() + 1e-6).any()
        untouched = all(sha(T.PRED / f"p1_{n}.parquet") == reg_hash[n] for n in names)
        try:
            E1.main(); refused = False
        except SystemExit:
            refused = True
        try:
            E1.p1_scores_at(names, 3, ""); refused2 = False
        except SystemExit:
            refused2 = True
        check("X1 E1b scores: column K3 reproduces the registered P1 files exactly; column 3 (5 s) >= 3 s risk, same rows; "
              "registered files untouched; no overwrite; no empty suffix", same and mono and untouched and refused and refused2)
        # ---------------- X2 E3 ----------------
        allk = np.ones(len(metas["test"]), bool); bank = CP.Bank()
        pairs = []
        for nm, h in E.VARIANT_K.items():
            _, p, _ = CP.endpoints(bank, dict(name=nm, k=CP.HORIZONS.index(h), keep=allk, tests=["H2"]), B, {})
            pairs += p
        P = pd.DataFrame(pairs)
        P["prev_heldout"] = P["prev_heldout"] * (1 + 0.15 * (np.arange(len(P)) % 7))     # synthetic levels share one base rate: vary it
        P.to_csv(tmp / "pairs.csv", index=False); P = pd.read_csv(tmp / "pairs.csv")
        mt = pd.read_parquet(T.CACHE / "shrp2_train_meta.parquet")
        rows = pd.DataFrame(E.e3(P, bank.trips, mt))
        tri = Tr.set_index("event_id")
        ok3, n3 = True, 0
        from scipy.stats import spearmanr
        for nm, h in E.VARIANT_K.items():
            k = CP.HORIZONS.index(h)
            q = rows[(rows.analysis == "E3 pair") & (rows.variant == nm)]
            sh = []
            for _, r in q.iterrows():
                f, a_ = r.factor, int(r.direction[0])
                num = den = 0
                for e_, eb, ob in zip(mt.event_id, mt.event_bin, mt.obs_bins):
                    if tri.at[e_, T.FACTORS[f]] == a_:
                        yy = 0 <= eb <= k; known = yy or ob >= k + 1
                        if known:
                            den += 1; num += yy
                p0 = P[(P.variant == nm) & (P.factor == f) & (P.direction == r.direction)].prev_heldout.iloc[0]
                ok3 &= np.isclose(r.prev_source_train, num / den) and np.isclose(r["shift"], abs(np.log(p0 / (num / den))))
                sh.append(r["shift"]); n3 += 1
            rho = rows[(rows.analysis == "E3 Spearman shift vs log_ratio") & (rows.variant == nm)].estimate.iloc[0]
            if len(q) >= 3:
                ok3 &= np.isclose(rho, spearmanr(sh, q.log_ratio).correlation)
        check("X2 E3: source training prevalence (hand loop), base-rate shift and Spearman per horizon variant",
              ok3 and n3 > 0, f"{n3} pairs")
        # ---------------- X3 E4a ----------------
        class FakeD:
            def __init__(s_):
                s_.g = {"shrp2_train": {"meta": mt}}
            def level_of(s_, f, ids):
                return tri[T.FACTORS[f]].reindex(ids).to_numpy(float)
        r4 = pd.DataFrame(E.e4a(FakeD()))
        y2, kn2 = C.labels_at(mt.event_bin.to_numpy(), mt.obs_bins.to_numpy(), 2)
        lab = {e_: f"L{int(tri.at[e_, 'lighting_bin'])}-S{int(tri.at[e_, 'surfaceCondition_bin'])}"
               for e_ in tri.index if tri.loc[e_, ["lighting_bin", "surfaceCondition_bin", "trafficDensity_bin"]].notna().all()}
        exp = pd.DataFrame(dict(c=[lab.get(e_) for e_ in mt.event_id], y=y2, k=kn2))
        exp = exp[exp.k & exp.c.notna()].groupby("c").y.mean()
        got = r4[r4.analysis == "E4a client"].set_index("variant").estimate
        ratio = r4[r4.analysis == "E4a max/min base rate"].estimate.iloc[0]
        from train_federated_shrp2 import client_table
        _, rep = client_table(FakeD())
        (tmp / "results/shrp2_fed_clients.json").write_text(json.dumps(dict(testable=True, training_events=rep["training_events"])))
        E.e4a(FakeD())                                           # matching file: passes
        (tmp / "results/shrp2_fed_clients.json").write_text(json.dumps(dict(testable=True, training_events={"L0-S0": 1})))
        try:
            E.e4a(FakeD()); bad = False
        except SystemExit:
            bad = True
        check("X3 E4a: client base rates at 3 s (hand grouping: traffic dropped, 4 lighting x surface clients), max/min ratio; "
              "refuses if the client rule does not reproduce the registered client file",
              sorted(got.index) == sorted(exp.index) and np.allclose(got[exp.index], exp) and np.isclose(ratio, exp.max() / exp.min()) and bad,
              f"{dict(got.round(3))}")
        (tmp / "results/shrp2_fed_clients.json").write_text(json.dumps(dict(testable=True)))
        # ---------------- X4 E4b ----------------
        Hp = plant_dropout(Tr, metas["test"])
        cnt, ds, gs = hand_classes(Hp)
        bank = CP.Bank()
        r5 = pd.DataFrame(E.e4b(bank, B))
        oc = r5[r5.analysis == "E4b dropout objects"].set_index("variant").n
        okc = all(int(oc[k]) == v for k, v in cnt.items()) and min(cnt["entering"], cnt["leaving"], cnt["both"], cnt["interior"], cnt["far"]) > 0
        y3, kn3 = C.labels_at(metas["test"].event_bin.to_numpy(), metas["test"].obs_bins.to_numpy(), 2)
        Tix = np.flatnonzero(kn3); y = y3[Tix]
        R = lambda nm: bank.model(nm)["test"][0][Tix, 2]
        okh = True
        sp = r5[r5.analysis == "E4b H4a split"].set_index("variant")
        for nm, sub in (("all natural-dropout (registered H4a set)", ds[Tix]), ("with an interior-gap object", ds[Tix] & gs[Tix]),
                        ("without an interior-gap object", ds[Tix] & ~gs[Tix])):
            e_ = np.mean([AP(y[sub], R(f"central_hazard_aug_s{s}")[sub]) - AP(y[sub], R(f"central_hazard_noaug_s{s}")[sub]) for s in SEEDS])
            okh &= np.isclose(sp.loc[nm, "estimate"], e_) and int(sp.loc[nm, "n"]) == int(sub.sum()) and sp.loc[nm, "ci_low"] <= sp.loc[nm, "ci_high"]
        check("X4 E4b: object classes and far share = hand loops (all four classes planted); dropout set = registered flag; "
              "H4a split = scikit-learn AUPRC differences, seed mean", okc and okh, str(cnt))
        # ---------------- X5 E5 ----------------
        CP.h5_input()
        inp = dict(np.load(T.SH / "h5_input_shrp2.npz", allow_pickle=True))
        RH.analyse(inp, "shrp2", 5)
        bank = CP.Bank()
        bank.meta["test"] = bank.meta["test"].assign(frame=10000 + 3 * (bank.meta["test"]["frame"] - 10000))   # spread: gaps
        r6 = pd.DataFrame(E.e5(bank, min_eval=5))
        units, _ = RH.build_units(inp["unit"], inp["split"], inp["known"], inp["y"], {s: inp[f"score_s{s}"] for s in SEEDS}, 5)
        tid = sorted(u for u in units if units[u]["split"] == "test"); vid = sorted(u for u in units if units[u]["split"] == "val")
        fr = np.r_[bank.meta["val"]["frame"].to_numpy(), bank.meta["test"]["frame"].to_numpy()]
        ok6 = r6[r6.analysis == "E5 check"].note.iloc[0].startswith("deployment shares and recall reproduce")
        fph = 36000.0; multi = 0
        for ln, lv in (("0.05", 0.05), ("op", 1 / fph)):
            for m in ("HP", "C1"):
                per = []
                for s in SEEDS:
                    if m == "HP":
                        lam = DC.hp_threshold([DC.loss_curve(units[u]["neg"][s]) for u in vid], lv, 0.10)["lam"]
                    else:
                        lam = DC.pooled_split_conformal(np.concatenate([units[u]["neg"][s] for u in vid]), lv)
                    ep, hr = [], []
                    for u in tid:
                        ni = units[u]["neg_ix"]; al = units[u]["neg"][s] > lam
                        ep.append(hand_episodes(fr[ni][al], fr[ni])); hr.append(len(ni) / fph)
                    ep, hr = np.array(ep, float), np.array(hr)
                    multi += int((ep > 1).sum())
                    per.append([(ep / hr).mean(), np.median(ep / hr), ep.sum() / hr.sum(), (ep > 0).mean()])
                v = np.mean(per, axis=0)
                q = r6[(r6.analysis == "E5 episodes per hour") & (r6.variant == f"{m} at {ln}")].iloc[0]
                ok6 &= np.allclose([q["mean"], q["median"], q.estimate, q.share_trips_with_episode], v)
        sec = pd.read_csv(tmp / "results/h5_shrp2_secondary.csv")
        sec.loc[sec.analysis == "descriptive DEPLOY", "recall"] += 0.01; sec.to_csv(tmp / "results/h5_shrp2_secondary.csv", index=False)
        try:
            E.e5(bank, min_eval=5); bad5 = False
        except SystemExit:
            bad5 = True
        check("X5 E5: deployment thresholds (validation units), episodes by run merging (hand), per-hour rates, pooled rate, "
              "share of trips; cross-check against h5_shrp2_secondary.csv bites", ok6 and bad5 and multi > 0, f"{multi} trip-seeds with > 1 episode")
        # ---------------- X6 registered run only; aggregate output ----------------
        r_ = subprocess.run([sys.executable, str(HERE / "explore_p2.py"), "e3"], capture_output=True, text=True,
                            env=dict(os.environ, SHRP2_RUN="nomask"))
        r1 = subprocess.run([sys.executable, str(HERE / "explore_e1_scores.py")], capture_output=True, text=True,
                            env=dict(os.environ, SHRP2_RUN="nomask"))
        (tmp / "results/shrp2_confirm_h2_pairs.csv").write_bytes((tmp / "pairs.csv").read_bytes())
        r2 = subprocess.run([sys.executable, str(HERE / "explore_p2.py"), "e3"], capture_output=True, text=True)
        txt = (tmp / "results/shrp2_explore_E3E5_e3_summary.txt").read_text() if r2.returncode == 0 else ""
        check("X6 both scripts refuse SHRP2_RUN (registered run only); the summary holds aggregates only (no trip IDs)",
              r_.returncode != 0 and "registered run only" in r_.stderr + r_.stdout and r1.returncode != 0
              and r2.returncode == 0 and "event_id" not in txt and not {"event_id", "unit", "trip"} & set(pd.read_csv(tmp / "results/shrp2_explore_E3E5_e3.csv").columns),
              r2.stderr[-300:])
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
