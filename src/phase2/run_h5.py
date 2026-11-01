"""
CONFIRMATORY ANALYSIS of hypotheses H5a-H5c, as registered in OSF Amendment 2 (https://osf.io/zd8jx, filed 11 Oct 2026;
docs/osf/OSF_Amendment_2_2026-10-11.pdf) and corrected by OSF Amendment 3 (H5a and H5c on the deployment design).
Written and tested BEFORE any SHRP2 data exist. The same code runs on SHRP2 (confirmatory) and on
INTERACTION (rehearsal only, development data).

Registered design (Amendment 2, Sections 2-6)
  Score     raw cumulative risk at 3 s; one score array per training seed (131-135).
  Negative  frame evaluable at 3 s with no crash/near-crash in (t, t + 3 s]. Positive: evaluable, event in (t, t + 3 s].
  Unit      driver if an ID exists, else trip (SHRP2); recording (INTERACTION rehearsal).
  Eligible  >= MIN_EVAL frames evaluable at 3 s (decided on counts only). Units with no negatives are counted and
            excluded (their false-alarm rate is undefined).
  Units     the n eligible TEST units. Leave one test unit out: each test unit's thresholds are set on the other
            n - 1 test units (k = n - 1), separately for every seed.
  Methods   HP  t_j = smallest grid value with L_j <= eps; threshold = r-th smallest of the k values,
                r = ceil((k + 1)(1 - delta)), delta = 0.10                             (primary)
            CRC smallest grid value with sum_j L_j / (k + 1) + 1 / (k + 1) <= alpha    (secondary)
            C1  registered split conformal on the pooled negatives of the k calibration units
  Levels    eps = alpha = 0.05 (primary), 0.10 (secondary); eps_op = one alarm frame per hour (HP only).
  AS AMENDED BY AMENDMENT 3 (filed before any SHRP2 access; under leave-one-out the HP exceedance count is fixed
  by ranks, so the original H5a and H5c could not fail):
  H5a  deployment design (HP threshold set once on the k eligible VALIDATION units, applied to every test unit, per
       seed); x = number of test units with FA > 0.05, averaged over seeds and rounded UP; exact one-sided binomial
       test of H0: share >= 0.20, p = P(Bin(n, 0.20) <= x). Not testable if k < 9 or n < 50.
  H5b  unchanged: recall(HP) - recall(C1) at 3 s on the leave-one-out design, recall = test positive frames above
       their own unit's threshold, pooled over test units, averaged over seeds within each resample; paired cluster
       bootstrap over test units, B = 2000, seed 131; p = (1 + #{D*_b <= -0.15}) / (B + 1). Not testable if n < 20
       or fewer than 50 test events with >= 1 positive frame at 3 s.
  H5c  as H5a, with FA > eps_op (one alarm frame per hour).
  Family  Holm at 0.05 over the testable tests among H5a-H5c. Bootstrap resamples with an undefined statistic are
       discarded and counted; > 5 % -> "unreliable".
  Approximation (stated in the amendment): per-unit results are computed once; the leave-one-out thresholds are not
       recomputed inside bootstrap resamples.
Secondary and pre-specified descriptive (no confirmatory claim): H5a (corrected form) and H5b at the 10 % level; CRC;
  the original leave-one-out shares above 5 % and eps_op for HP and C1 with bootstrap intervals; per-unit FA median,
  90th percentile and maximum, and C1 mean/pooled FA, for both designs; counts of units excluded by eligibility, with zero negatives, and below 1/eps negatives at every level.
  Not in this script: published comparators (B7 code), strata shares, episode-level false alarms, Holm over all 12
  tests (needs the nine H1-H4 p-values; use --h1h4 to pass them).

Input (generic, .npz): unit, split ('val'/'test'), known (bool, evaluable at 3 s), y (0/1 at 3 s), event (event id
  per frame, used only to count test events), score_s<seed> (one float array per seed), frames_per_hour (scalar).
  --interaction builds this file from the INTERACTION rehearsal predictions (recording = unit, 1 sample per
  agent-second, event = recording + agent; documented proxies).
Usage
  python src/phase2/run_h5.py --interaction            (rehearsal on INTERACTION, development data)
  python src/phase2/run_h5.py --input data/processed/h5_input_shrp2.npz --tag shrp2   (confirmatory, after audit)
Outputs results/h5_<tag>_units.csv, results/h5_<tag>_tests.csv, results/h5_<tag>_secondary.csv, results/h5_<tag>_counts.csv
"""
import argparse, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from phase2 import domain_crc as D
from analysis import stats as S

SEEDS = [131, 132, 133, 134, 135]
DELTA, ALPHA_FAMILY, B, BOOT_SEED = 0.10, 0.05, 2000, 131
MIN_EVAL_DEFAULT, N_MIN_ALL, N_MIN_DEPLOY_TEST, EVENTS_MIN_H5B = 100, 20, 50, 50
H5B_MARGIN, H0_SHARE, MIN_DEPLOY_UNITS = -0.15, 0.20, 9


# ------------------------------------------------------------------------------------------- units
def build_units(unit, split, known, y, scores_by_seed, min_eval):
    """Eligibility on counts only, then per-unit negatives/positives for every seed.
    Returns (units dict, counts dict). units[u] = dict(split, neg={seed: array}, pos={seed: array})."""
    unit = np.asarray(unit).astype(str); split = np.asarray(split).astype(str)
    known = np.asarray(known, bool); y = np.asarray(y).astype(int)
    order = np.argsort(unit, kind="mergesort"); us, start = np.unique(unit[order], return_index=True)
    bounds = list(zip(start, list(start[1:]) + [len(order)]))
    units, cnt = {}, dict(units_total=len(us), excluded_eligibility=0, excluded_zero_negatives=0,
                          excluded_mixed_split=0)
    for u, (a, b) in zip(us, bounds):
        ix = order[a:b]
        if len(np.unique(split[ix])) != 1:                 # a unit must sit in one split (registered splits)
            cnt["excluded_mixed_split"] += 1; continue
        kn = known[ix]
        if int(kn.sum()) < min_eval:                       # counts only, before any score is used
            cnt["excluded_eligibility"] += 1; continue
        neg_ix, pos_ix = ix[kn & (y[ix] == 0)], ix[kn & (y[ix] == 1)]
        if len(neg_ix) == 0:
            cnt["excluded_zero_negatives"] += 1; continue
        units[u] = dict(split=split[ix[0]], neg_ix=neg_ix, pos_ix=pos_ix,
                        neg={s: np.asarray(sc, float)[neg_ix] for s, sc in scores_by_seed.items()},
                        pos={s: np.asarray(sc, float)[pos_ix] for s, sc in scores_by_seed.items()})
    return units, cnt


# ------------------------------------------------------------------------------- leave-one-out rules
def loo_hp(t_values, delta=DELTA):
    """HP threshold for each unit from the OTHER units' t_j: r-th smallest of k = n - 1 values."""
    t = np.asarray(t_values, float); n = len(t); k = n - 1
    r = int(math.ceil((k + 1) * (1 - delta)))
    if r > k:
        return np.full(n, np.inf), False
    o = np.argsort(t, kind="mergesort"); srt = t[o]; rank = np.empty(n, int); rank[o] = np.arange(n)
    # removing unit i shifts every value above it down by one position
    out = np.where(rank >= r, srt[r - 1], srt[r] if r < n else np.inf)
    return out, True


def loo_crc(curves, alpha):
    """CRC threshold for each unit from the sum of the OTHER units' curves (total minus own curve)."""
    curves = np.asarray(curves, float); n = len(curves); k = n - 1
    total = D.exact_sum(curves)
    lam = np.empty(n); feas = np.empty(n, bool)
    for i in range(n):
        r = D.crc_from_sum(total - curves[i], k, alpha); lam[i] = r["lam"]; feas[i] = r["feasible"]
    return lam, feas


def loo_c1(neg_list, alpha):
    """C1 (registered split conformal) on the pooled negatives of the OTHER units, without re-sorting:
    the r-th smallest of the others = smallest value v with #others(<= v) >= r."""
    allv = np.sort(np.concatenate(neg_list)); N = len(allv); out = np.empty(len(neg_list))
    for i, own in enumerate(neg_list):
        own = np.sort(own); M = N - len(own); r = int(math.ceil((M + 1) * (1 - alpha)))
        if r > M:
            out[i] = np.inf; continue
        lo, hi = 0, N - 1                                   # binary search over candidate values allv
        while lo < hi:
            mid = (lo + hi) // 2; v = allv[mid]
            if np.searchsorted(allv, v, "right") - np.searchsorted(own, v, "right") >= r:
                hi = mid
            else:
                lo = mid + 1
        out[i] = allv[lo]
    return out


# ------------------------------------------------------------------------------ per-unit results
def per_unit_results(units, ids, seeds, levels, design="LOO", cal_ids=None):
    """One row per (unit, seed, level, method): FA, exceed, caught, n_pos, n_neg, lam, feasible.
    design 'LOO': calibration = the other test units. design 'DEPLOY': calibration = cal_ids (validation units)."""
    rows = []
    for s in seeds:
        curves = {u: D.loss_curve(units[u]["neg"][s]) for u in set(ids) | set(cal_ids or [])}
        for ln, lv in levels:
            if design == "LOO":
                t = [D.unit_threshold(curves[u], lv) for u in ids]
                hp, hp_ok = loo_hp(t)
                th = {"HP": (hp, np.full(len(ids), hp_ok))}
                if ln != "op":
                    th["CRC"] = loo_crc([curves[u] for u in ids], lv)
                th["C1"] = (loo_c1([units[u]["neg"][s] for u in ids], lv), np.ones(len(ids), bool))
            else:
                cc = [curves[u] for u in cal_ids]
                r = D.hp_threshold(cc, lv, DELTA); th = {"HP": (np.full(len(ids), r["lam"]), np.full(len(ids), r["feasible"]))}
                if ln != "op":
                    r = D.crc_threshold(cc, lv); th["CRC"] = (np.full(len(ids), r["lam"]), np.full(len(ids), r["feasible"]))
                pooled = np.concatenate([units[u]["neg"][s] for u in cal_ids])
                th["C1"] = (np.full(len(ids), D.pooled_split_conformal(pooled, lv)), np.ones(len(ids), bool))
            for m, (lam, feas) in th.items():
                for i, u in enumerate(ids):
                    neg, pos = units[u]["neg"][s], units[u]["pos"][s]
                    fa = float(np.mean(neg > lam[i]))
                    rows.append(dict(design=design, unit=u, seed=s, level=ln, eps=lv, method=m, lam=float(lam[i]),
                                     feasible=bool(feas[i]), fa=fa, exceed=bool(fa > lv), n_neg=len(neg),
                                     n_pos=len(pos), caught=int((pos > lam[i]).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- tests
def unit_arrays(R, level, ids, seeds, methods=("HP", "C1")):
    """Arrays [seed, unit] for the given methods at one level: exceed, caught, n_pos."""
    out = {}
    for m in methods:
        g = R[(R.level == level) & (R.method == m)]
        out[m] = {f: g.pivot(index="seed", columns="unit", values=f).loc[seeds, ids].to_numpy(float)
                  for f in ("exceed", "caught", "n_pos")}
    return out


def share_diff_stat(A):
    """Seed-averaged share difference C1 - HP (DESCRIPTIVE since Amendment 3); idx = resampled unit positions."""
    d = (A["C1"]["exceed"] - A["HP"]["exceed"]).mean(axis=0)
    return lambda idx: float(d[idx].mean())


def share_stat(A, m):
    """Seed-averaged share of units above the level for one method (descriptive interval)."""
    d = A[m]["exceed"].mean(axis=0)
    return lambda idx: float(d[idx].mean())


def h5b_stat(A):
    """Seed-averaged difference of pooled recall HP - C1; undefined (nan) if the resample has no positives."""
    def stat(idx):
        npos = A["HP"]["n_pos"][:, idx].sum(axis=1)
        if np.any(npos == 0):
            return np.nan
        rec_hp = A["HP"]["caught"][:, idx].sum(axis=1) / npos
        rec_c1 = A["C1"]["caught"][:, idx].sum(axis=1) / npos
        return float(np.mean(rec_hp - rec_c1))
    return stat


def binomial_share_test(exceed, n, null=H0_SHARE):
    """exceed: [seed, unit] exceedances under the DEPLOYED threshold. x = ceil(seed mean of counts);
    exact one-sided test of H0: share >= null, p = P(Bin(n, null) <= x). Valid because the test units are
    independent given the deployed threshold; averaging over seeds and rounding up are conservative."""
    mean_count = float(np.mean(exceed.sum(axis=1)))
    x = int(math.ceil(round(mean_count, 9)))                  # round() guards against 3.0000000001 -> 4
    return dict(x=x, mean_count=mean_count, share=mean_count / n, p=float(binom.cdf(x, n, null)))


h5c_test = binomial_share_test                                # name kept for the unit tests of the counting rule


def deploy_test(RD, level, ids, seeds, name, n, k_val):
    if RD is None or k_val < MIN_DEPLOY_UNITS or n < N_MIN_DEPLOY_TEST:
        why = (f"{k_val} eligible validation units < {MIN_DEPLOY_UNITS}" if k_val < MIN_DEPLOY_UNITS
               else f"n = {n} test units < {N_MIN_DEPLOY_TEST}")
        return dict(test=name, estimate=np.nan, p=np.nan, note=f"not testable: {why}")
    g = RD[(RD.level == level) & (RD.method == "HP")]
    ex = g.pivot(index="seed", columns="unit", values="exceed").loc[seeds, ids].to_numpy(float)
    c = binomial_share_test(ex, n)
    return dict(test=name, estimate=c["share"], p=c["p"], x=c["x"], mean_count=c["mean_count"], n=n, k=k_val,
                delta0=H0_SHARE, note="deployment design: HP threshold from validation units, exact binomial")


def run_tests(R, RD, ids, seeds, n_events, boot, k_val):
    """Confirmatory family as amended (Amendment 3): H5a, H5c on the deployment design; H5b leave-one-out."""
    n = len(ids); res = []
    res.append(deploy_test(RD, "0.05", ids, seeds, "H5a", n, k_val))
    A5 = unit_arrays(R, "0.05", ids, seeds)
    if n >= N_MIN_ALL and n_events >= EVENTS_MIN_H5B:
        r = S.paired_test(boot, h5b_stat(A5), H5B_MARGIN)
        res.append(dict(test="H5b", **r, delta0=H5B_MARGIN, note="leave one test unit out, paired cluster bootstrap"))
    else:
        why = f"n = {n} < {N_MIN_ALL}" if n < N_MIN_ALL else f"test events = {n_events} < {EVENTS_MIN_H5B}"
        res.append(dict(test="H5b", estimate=np.nan, p=np.nan, note=f"not testable: {why}"))
    res.append(deploy_test(RD, "op", ids, seeds, "H5c", n, k_val))
    T = pd.DataFrame(res)
    adj, rej = S.holm(T.p.values, ALPHA_FAMILY)
    T["p_holm"] = adj; T["supported"] = rej
    return T


def fa_quantiles(RR, design):
    """Per-unit false-alarm distribution and C1 mean/pooled FA (descriptive), averaged over seeds."""
    out = []
    for (ln, m), g in RR.groupby(["level", "method"], sort=False):
        ps = []                                               # explicit loop: identical across pandas versions
        for _, h in g.groupby("seed"):
            ps.append([h.fa.mean(), (h.fa * h.n_neg).sum() / h.n_neg.sum(), h.fa.median(), h.fa.quantile(0.9),
                       h.fa.max(), h.exceed.mean(), h.caught.sum() / max(h.n_pos.sum(), 1), h.feasible.mean()])
        v = np.mean(ps, axis=0)
        out.append(dict(analysis=f"descriptive {design}", level=ln, method=m, mean_fa=v[0], pooled_fa=v[1],
                        median_fa=v[2], p90_fa=v[3], max_fa=v[4], share_over=v[5], recall=v[6], feasible=v[7]))
    return out


# -------------------------------------------------------------------------------------- driver
def analyse(inp, tag, min_eval, h1h4=None):
    t0 = time.time()
    seeds = [s for s in SEEDS if f"score_s{s}" in inp]
    fph = float(inp["frames_per_hour"]); eps_op = 1.0 / fph
    levels = [("0.05", 0.05), ("0.10", 0.10), ("op", eps_op)]
    scores = {s: inp[f"score_s{s}"] for s in seeds}
    units, cnt = build_units(inp["unit"], inp["split"], inp["known"], inp["y"], scores, min_eval)
    test_ids = sorted(u for u in units if units[u]["split"] == "test")
    val_ids = sorted(u for u in units if units[u]["split"] == "val")
    n, k_val = len(test_ids), len(val_ids)
    ev = np.asarray(inp["event"]).astype(str)
    pos_ix = np.concatenate([units[u]["pos_ix"] for u in test_ids]) if n else np.array([], int)
    n_events = int(len(np.unique(ev[pos_ix]))) if len(pos_ix) else 0
    cnt.update(seeds=len(seeds), eps_op=eps_op, frames_per_hour=fph, test_units=n, validation_units=k_val,
               test_events=n_events, test_positive_frames=int(len(pos_ix)))
    for ln, lv in levels:
        cnt[f"test_units_below_1/eps_{ln}"] = sum(len(units[u]["neg_ix"]) < 1.0 / lv for u in test_ids)
    print(f"[{tag}] seeds {seeds} | eligible test units n = {n}, validation units k = {k_val}, "
          f"test events = {n_events}, test positive frames = {len(pos_ix)} | eps_op = 1/{fph:.0f}", flush=True)
    if n < 2:
        raise SystemExit("fewer than 2 eligible test units - nothing can be computed")
    R = per_unit_results(units, test_ids, seeds, levels, "LOO")
    RD = per_unit_results(units, test_ids, seeds, levels, "DEPLOY", val_ids) if k_val >= MIN_DEPLOY_UNITS else None
    boot = S.ClusterBootstrap(np.array(test_ids), B=B, seed=BOOT_SEED)   # one row per unit -> unit-level resampling
    T = run_tests(R, RD, test_ids, seeds, n_events, boot, k_val)
    # ---- secondary and pre-specified descriptive analyses (Amendment 2 Section 5; Amendment 3 Section 3) ----
    sec = []
    sec.append(dict(analysis="H5a at 10% (secondary, corrected form)", **deploy_test(RD, "0.10", test_ids, seeds, "H5a10", n, k_val)))
    A10 = unit_arrays(R, "0.10", test_ids, seeds)
    if n >= N_MIN_ALL and n_events >= EVENTS_MIN_H5B:
        r = S.paired_test(boot, h5b_stat(A10), H5B_MARGIN); sec.append(dict(analysis="H5b at 10% (secondary)", **r))
    if n >= N_MIN_ALL:
        for ln in ("0.05", "op"):
            A = unit_arrays(R, ln, test_ids, seeds)
            r = S.paired_test(boot, share_diff_stat(A), 0.0)
            sec.append(dict(analysis=f"descriptive LOO share above level {ln}: C1 - HP (no p-value; HP share <= (n-r)/n by construction)",
                            estimate=r["estimate"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
            for m in ("HP", "C1"):
                r = S.paired_test(boot, share_stat(A, m), 0.0)
                sec.append(dict(analysis=f"descriptive LOO share above level {ln}: {m}", estimate=r["estimate"],
                                ci_low=r["ci_low"], ci_high=r["ci_high"]))
    sec += fa_quantiles(R, "LOO")
    if RD is not None:
        sec += fa_quantiles(RD, "DEPLOY")
    else:
        sec.append(dict(analysis="deployment design", note=f"not run: {k_val} eligible validation units < {MIN_DEPLOY_UNITS}"))
    for row in sec:
        if "mean_fa" in row:
            row["alarm_frames_per_hour"] = row["mean_fa"] * fph
    if h1h4 is not None:
        p12 = np.r_[np.asarray(h1h4, float), T.p.values]
        adj, rej = S.holm(p12, ALPHA_FAMILY)
        sec.append(dict(analysis="Holm over all 12 (sensitivity)", note="; ".join(
            f"{nm}: adj p {a:.4g} {'rejected' if r_ else 'not rejected'}" for nm, a, r_ in
            zip([f"H1-H4 test {i+1}" for i in range(len(h1h4))] + ["H5a", "H5b", "H5c"], adj, rej))))
    Sec = pd.DataFrame(sec)
    Path("results").mkdir(exist_ok=True)
    pd.concat([R] + ([RD] if RD is not None else [])).to_csv(f"results/h5_{tag}_units.csv", index=False)
    T.to_csv(f"results/h5_{tag}_tests.csv", index=False)
    Sec.to_csv(f"results/h5_{tag}_secondary.csv", index=False)
    pd.DataFrame([cnt]).T.rename(columns={0: "value"}).to_csv(f"results/h5_{tag}_counts.csv")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\nCOUNTS:"); [print(f"  {k}: {v}") for k, v in cnt.items()]
    cols = [c for c in ["test", "estimate", "ci_low", "ci_high", "p", "p_holm", "supported", "B_used", "discarded",
                        "unreliable", "x", "mean_count", "n", "k", "note"] if c in T]
    print("\nCONFIRMATORY FAMILY H5a-H5c as amended (Holm at 0.05 within the family):")
    print(T[cols].round(4).to_string(index=False))
    print("\nSECONDARY AND DESCRIPTIVE:"); print(Sec.round(4).to_string(index=False))
    print(f"\nSaved results/h5_{tag}_*.csv | {time.time() - t0:.0f}s")
    return R, T, Sec, cnt


def interaction_input():
    """Rehearsal input from the INTERACTION central-model predictions (development data only)."""
    import calibration as C
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    scene, agent = d["scene"].astype(str), d["agent"].astype(str)
    y_all, known = C.labels_at(lab["event_bin"], lab["obs_bins"], 2)
    ref = None; out = {}
    for s in SEEDS:
        z = np.load(f"data/processed/preds/haz_central_aug_s{s}.npz")
        if ref is None:
            ref = z["idx"]; out.update(unit=scene[ref], split=z["split"].astype(str), known=known[ref], y=y_all[ref],
                                       event=np.char.add(np.char.add(scene[ref], "|"), agent[ref]))
        assert np.array_equal(z["idx"], ref), "seed predictions must cover the same samples in the same order"
        out[f"score_s{s}"] = z["risk"][:, 2].astype(float)
    out["frames_per_hour"] = 3600.0                         # INTERACTION samples: one per agent-second
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--interaction", action="store_true", help="rehearsal on INTERACTION (development data)")
    g.add_argument("--input", help="generic H5 input .npz (e.g. SHRP2, built after the data audit)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--min-eval", type=int, default=MIN_EVAL_DEFAULT)
    ap.add_argument("--h1h4", default=None, help="comma-separated nine raw p-values of H1-H4 (Holm-12 sensitivity)")
    a = ap.parse_args()
    inp = interaction_input() if a.interaction else dict(np.load(a.input, allow_pickle=True))
    tag = a.tag or ("interaction_rehearsal" if a.interaction else Path(a.input).stem)
    analyse(inp, tag, a.min_eval, [float(x) for x in a.h1h4.split(",")] if a.h1h4 else None)
