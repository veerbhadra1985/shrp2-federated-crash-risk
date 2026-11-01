"""
EXPLORATORY ANALYSES E3, E4a, E4b and E5 on the frame-level predictions (plan docs/exploratory_plan_E1_E5.md, written
30 Oct 2026 before this script; committed with it before its single run). Project environment. Registered run only.
E3 and E5 are registered exploratory analyses (osf.io/y5zrp; Amendment 2); E4a and E4b are post-hoc.

E3   per testable factor-direction pair (results/shrp2_confirm_h2_pairs.csv): base-rate shift
     |log(prevalence of the held-out level's test samples / prevalence of the source level's training samples)| and the
     log ECE ratio (registered and size-matched); Spearman correlation, no p-value. Primary horizon 3 s and the
     horizon variants 1, 2, 5 s. NOTE: the pairs file stores log(seed-mean ECE out / seed-mean ECE in) per pair.
E4a  per federated client (registered client rule, train_federated_shrp2.client_table): base rate at 3 s = positive /
     labelled training samples; ratio of the highest to the lowest base rate. Descriptive.
E4b  within the registered natural-dropout set (confirm_p2.natural_dropout), each dropout object (present in the 2 s
     window, missing in >= 20% of its frames) is classed by where its missing frames lie: entering (only before its
     first present frame), leaving (only after its last present frame), both edges (before and after, none between),
     interior gap (>= 1 missing frame between two present frames). Share of dropout objects whose last present
     position is beyond 25 m from the subject (same frame). H4a (augmented - plain AUPRC at 3 s, seed mean) on the
     prediction times with >= 1 interior-gap object and on the rest, 95% cluster-bootstrap intervals (trips, B 2,000,
     seed 131), engine of confirm_p2 unchanged. NOTE: objects are selected as present at the prediction time, so
     "leaving" and "both edges" are expected to be empty; they are reported, not assumed.
E5   alarm episodes in the deployment design (run_h5.per_unit_results, design DEPLOY: thresholds from the validation
     units, as for H5a and H5c): within each test unit (trip), negative samples above the threshold, in frame order;
     a new episode starts when the frame gap to the previous alarm frame exceeds 10 (runs separated by fewer than 10
     frames, 1 s, are merged). HP and C1 at 5% and at eps_op, per seed. Per trip: episodes per hour of negative driving
     (negatives / frames_per_hour); over trips: mean, median, 90th percentile, maximum, pooled rate and share of trips
     with >= 1 episode; seed means. Cross-check: the deployment shares above the level and recall reproduce
     results/h5_shrp2_secondary.csv.
Nothing here changes a registered rule or verdict. Outputs are aggregate.

Usage   python src/shrp2/explore_p2.py [all|e3|e4a|e4b|e5] [--B 2000]
Output  results/shrp2_explore_E3E5.csv, results/shrp2_explore_E3E5_summary.txt
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import analysis.stats as S
import train_shrp2 as T
import confirm_p2 as CP
from phase2 import run_h5 as RH

SEEDS = T.SEEDS
K3 = 2
GAP = 10                       # frames (1 s at 10 Hz)
FAR_M = 25.0
VARIANT_K = {"primary": 3, "3 horizon 1 s": 1, "3 horizon 2 s": 2, "3 horizon 5 s": 5}
OUT = Path("results")


# ================================================================================================ E3
def prevalence_train(trips, mt, factor, level, k):
    """Prevalence at horizon index k among the labelled training samples of trips at the given factor level."""
    col = T.FACTORS[factor]
    lv = trips[col].reindex(mt["event_id"].to_numpy()).to_numpy(float) if col in trips else np.full(len(mt), np.nan)
    import calibration as C
    y, kn = C.labels_at(mt["event_bin"].to_numpy(), mt["obs_bins"].to_numpy(), k)
    m = kn & (lv == level)
    return float(y[m].mean()) if m.any() else np.nan


def e3(pairs, trips, mt):
    from scipy.stats import spearmanr
    rows = []
    for var, h in VARIANT_K.items():
        k = CP.HORIZONS.index(h)
        P = pairs[(pairs["variant"] == var) & (pairs["testable"].astype(str).str.lower() == "true")].copy()
        if P.empty:
            rows.append(dict(analysis="E3", variant=var, note="no testable pairs")); continue
        P["prev_source_train"] = [prevalence_train(trips, mt, f, int(d.split("->")[0]), k) for f, d in zip(P.factor, P.direction)]
        P["shift"] = np.abs(np.log(P["prev_heldout"] / P["prev_source_train"]))
        for _, r in P.iterrows():
            rows.append(dict(analysis="E3 pair", variant=var, factor=r.factor, direction=r.direction,
                             prev_heldout_test=r.prev_heldout, prev_source_train=r.prev_source_train, shift=r["shift"],
                             log_ratio=r.log_ratio, log_ratio_matched=r.get("log_ratio_matched", np.nan)))
        for col in ("log_ratio", "log_ratio_matched"):
            ok = P[col].notna() & np.isfinite(P["shift"]) if col in P else pd.Series(False, index=P.index)
            rho = float(spearmanr(P.loc[ok, "shift"], P.loc[ok, col]).correlation) if ok.sum() >= 3 else np.nan
            rows.append(dict(analysis=f"E3 Spearman shift vs {col}", variant=var, estimate=rho, n=int(ok.sum()),
                             note="descriptive; no p-value (few pairs); log ratio of seed-mean ECEs"))
    return rows


# ================================================================================================ E4a
def e4a(D=None):
    import train_federated_shrp2 as TF
    import calibration as C
    D = D or T.Data(need_public=False)
    lab, rep = TF.client_table(D)
    cj = Path("results/shrp2_fed_clients.json")
    if cj.exists():
        reg = json.loads(cj.read_text()).get("training_events")
        if reg is not None and reg != rep["training_events"]:
            raise SystemExit("check failed: client rule does not reproduce results/shrp2_fed_clients.json")
    mt = D.g["shrp2_train"]["meta"]
    y, kn = C.labels_at(mt["event_bin"].to_numpy(), mt["obs_bins"].to_numpy(), K3)
    cli = lab.reindex(mt["event_id"].to_numpy()).to_numpy()
    rows, br = [], {}
    for c in sorted(x for x in pd.unique(cli) if x is not None and not (isinstance(x, float) and np.isnan(x))):
        m = kn & (cli == c)
        br[c] = float(y[m].mean())
        rows.append(dict(analysis="E4a client", variant=c, n=int(m.sum()), positives=int(y[m].sum()), estimate=br[c],
                         note=f"training events {rep['training_events'].get(c)}"))
    v = np.array(list(br.values()))
    rows.append(dict(analysis="E4a max/min base rate", estimate=float(v.max() / v.min()) if v.min() > 0 else np.inf,
                     n=len(v), note="compare: about 40-fold spread of the INTERACTION development clients"))
    return rows


# ================================================================================================ E4b
def classify_objects(H):
    """H: (n, 9, 20, 9) histories. For each sample: flags of dropout objects (registered definition) and their class.
    Returns dict of (n, 8) bool arrays: dropout, entering, leaving, both, interior, far."""
    v = H[:, 1:, :, 8] > 0                                        # (n, 8, 20)
    present = v.any(-1); drop = present & ((1 - v.mean(-1)) >= CP.DROP_SHARE)
    nf = v.shape[-1]; idx = np.arange(nf)
    first = np.where(present, np.argmax(v, -1), 0)
    last = np.where(present, nf - 1 - np.argmax(v[..., ::-1], -1), -1)
    before = present & (first > 0)
    after = present & (last < nf - 1)
    inside = (idx[None, None] > first[..., None]) & (idx[None, None] < last[..., None])
    interior = present & (inside & ~v).any(-1)
    ent = drop & before & ~after & ~interior
    lea = drop & after & ~before & ~interior
    both = drop & before & after & ~interior
    li = np.clip(last, 0, nf - 1)
    take = lambda a: np.take_along_axis(a, li[..., None], -1)[..., 0]
    ox, oy = take(H[:, 1:, :, 0]), take(H[:, 1:, :, 1])
    sx = H[:, 0, :, 0][np.arange(len(H))[:, None], li]; sy = H[:, 0, :, 1][np.arange(len(H))[:, None], li]
    far = drop & (np.hypot(ox - sx, oy - sy) > FAR_M)
    return dict(dropout=drop, entering=ent, leaving=lea, both=both, interior=drop & interior, far=far)


def e4b(bank, B, chunk=50000):
    X = np.load(T.CACHE / "shrp2_test_hist.npy", mmap_mode="r")
    n = len(bank.meta["test"])
    cnt = {k: 0 for k in ("dropout", "entering", "leaving", "both", "interior", "far")}
    drop_s, inter_s = np.zeros(n, bool), np.zeros(n, bool)
    for lo in range(0, n, chunk):
        H = np.asarray(X[lo:lo + chunk], np.float32)
        c = classify_objects(H)
        if not np.array_equal(c["dropout"].any(-1), CP.natural_dropout(H)):
            raise SystemExit("check failed: dropout objects do not reproduce the registered natural-dropout flag")
        for k in cnt:
            cnt[k] += int(c[k].sum())
        drop_s[lo:lo + len(H)] = c["dropout"].any(-1); inter_s[lo:lo + len(H)] = c["interior"].any(-1)
    rows = [dict(analysis="E4b dropout objects", variant=k, n=v, estimate=v / max(cnt["dropout"], 1),
                 note="share of dropout objects (natural-dropout test samples)") for k, v in cnt.items()]
    y_t, kn_t = bank.labels("test", K3)
    Tix = np.flatnonzero(kn_t); y = y_t[Tix]; cl = bank.ev("test")[Tix]
    H_ = bank.head
    A = [bank.model(f"central_{H_}_aug_s{s}")["test"][0][Tix, K3] for s in SEEDS]
    N = [bank.model(f"central_{H_}_noaug_s{s}")["test"][0][Tix, K3] for s in SEEDS]
    d, g = drop_s[Tix], inter_s[Tix]
    for nm, sub in (("all natural-dropout (registered H4a set)", d), ("with an interior-gap object", d & g),
                    ("without an interior-gap object", d & ~g)):
        if sub.sum() == 0 or y[sub].sum() == 0:
            rows.append(dict(analysis="E4b H4a split", variant=nm, n=int(sub.sum()), note="no positives")); continue
        bt = S.ClusterBootstrap(cl[sub], B=B, seed=131)
        r = S.paired_test(bt, S.seed_mean_difference(S.average_precision, y[sub], [a[sub] for a in A], [b[sub] for b in N]), 0.0)
        rows.append(dict(analysis="E4b H4a split", variant=nm, n=int(sub.sum()), positives=int(y[sub].sum()),
                         trips=int(len(np.unique(cl[sub]))), estimate=r["estimate"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                         B_used=r["B_used"], note="augmented - plain AUPRC at 3 s; descriptive interval (p not used)"))
    return rows


# ================================================================================================ E5
def episodes(frames, gap=GAP):
    """Number of alarm episodes among alarm frames: a new episode when the gap to the previous alarm frame > gap."""
    f = np.sort(np.asarray(frames))
    return 0 if len(f) == 0 else int(1 + np.sum(np.diff(f) > gap))


def e5(bank, h5_path=None, min_eval=RH.MIN_EVAL_DEFAULT, check_file=Path("results/h5_shrp2_secondary.csv")):
    h5_path = h5_path or T.SH / "h5_input_shrp2.npz"
    inp = dict(np.load(h5_path, allow_pickle=True))
    fr = np.concatenate([bank.meta[s]["frame"].to_numpy() for s in ("val", "test")]).astype(np.int64)
    ev = np.concatenate([bank.ev(s) for s in ("val", "test")]).astype(str)
    if len(fr) != len(inp["unit"]) or not np.array_equal(ev, np.asarray(inp["unit"]).astype(str)):
        raise SystemExit("check failed: the H5 input is not in the cache order of the validation and test samples")
    seeds = [s for s in SEEDS if f"score_s{s}" in inp]
    fph = float(inp["frames_per_hour"]); levels = [("0.05", 0.05), ("op", 1.0 / fph)]
    units, _ = RH.build_units(inp["unit"], inp["split"], inp["known"], inp["y"], {s: inp[f"score_s{s}"] for s in seeds}, min_eval)
    test_ids = sorted(u for u in units if units[u]["split"] == "test")
    val_ids = sorted(u for u in units if units[u]["split"] == "val")
    RD = RH.per_unit_results(units, test_ids, seeds, levels, "DEPLOY", val_ids)
    rows = []
    if check_file.exists():
        ref = pd.read_csv(check_file); ref = ref[ref["analysis"] == "descriptive DEPLOY"]
        mine = pd.DataFrame(RH.fa_quantiles(RD, "DEPLOY"))
        for _, r in mine.iterrows():
            q = ref[(ref.level.astype(str) == r.level) & (ref.method == r.method)]
            if len(q) and not np.allclose([q.share_over.iloc[0], q.recall.iloc[0]], [r.share_over, r.recall], atol=1e-9):
                raise SystemExit(f"check failed: deployment results differ from {check_file} ({r.level}, {r.method})")
        rows.append(dict(analysis="E5 check", note=f"deployment shares and recall reproduce {check_file}"))
    else:
        rows.append(dict(analysis="E5 check", note=f"{check_file} absent - cross-check not run"))
    for ln, _ in levels:
        for m in ("HP", "C1"):
            per = []
            for s in seeds:
                g = RD[(RD.level == ln) & (RD.method == m) & (RD.seed == s)].set_index("unit")
                ep, hrs = [], []
                for u in test_ids:
                    ni = units[u]["neg_ix"]; al = units[u]["neg"][s] > g.at[u, "lam"]
                    ep.append(episodes(fr[ni][al])); hrs.append(len(ni) / fph)
                ep, hrs = np.array(ep, float), np.array(hrs)
                rate = ep / hrs
                per.append([rate.mean(), np.median(rate), np.quantile(rate, 0.9), rate.max(), ep.sum() / hrs.sum(),
                            (ep > 0).mean(), ep.sum(), hrs.sum()])
            v = np.mean(per, axis=0)
            rows.append(dict(analysis="E5 episodes per hour", variant=f"{m} at {ln}", estimate=v[4], mean=v[0], median=v[1],
                             p90=v[2], max=v[3], share_trips_with_episode=v[5], episodes_total=v[6], hours_negative=v[7],
                             n=len(test_ids), note=f"deployment design; seed means over {len(seeds)} seeds; estimate = pooled rate"))
    return rows


# ================================================================================================ driver
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "e3", "e4a", "e4b", "e5"])
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args(argv)
    if os.environ.get("SHRP2_RUN", ""):
        raise SystemExit("E1-E5 run on the registered run only: remove SHRP2_RUN (Remove-Item Env:SHRP2_RUN)")
    t0 = time.time(); rows = []
    bank = CP.Bank()
    if a.what in ("all", "e3"):
        rows += e3(pd.read_csv(OUT / "shrp2_confirm_h2_pairs.csv"), bank.trips, pd.read_parquet(T.CACHE / "shrp2_train_meta.parquet"))
    if a.what in ("all", "e4a"):
        rows += e4a()
    if a.what in ("all", "e4b"):
        rows += e4b(bank, a.B)
    if a.what in ("all", "e5"):
        rows += e5(bank)
    R = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    sfx = "" if a.what == "all" else f"_{a.what}"
    R.to_csv(OUT / f"shrp2_explore_E3E5{sfx}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    txt = ["SHRP2 EXPLORATORY ANALYSES E3-E5 (registered run; E3, E5 registered exploratory; E4 post-hoc; aggregate only)",
           f"risk head {bank.head}; B = {a.B}", "", R.round(4).to_string(index=False)]
    (OUT / f"shrp2_explore_E3E5{sfx}_summary.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt)); print(f"\nSaved results/shrp2_explore_E3E5{sfx}.csv | {time.time() - t0:.0f}s")
    return R


if __name__ == "__main__":
    main()
