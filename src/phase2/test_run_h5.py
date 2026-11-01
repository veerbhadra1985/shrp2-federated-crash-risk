"""
Unit tests for src/phase2/run_h5.py (H5a-H5c, OSF Amendments 2 and 3), synthetic data with answers known in advance.
Run BEFORE the script is used on any real data:   python src/phase2/test_run_h5.py   (about 1-3 minutes)
"""
import math, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2 import run_h5 as H
from phase2 import domain_crc as D
from analysis import stats as S

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def synth_units(rng, n, shift_sd=0.6, size=(150, 600), pos_rate=0.02):
    """n heterogeneous but exchangeable units: unit-specific logit shift, frames with scores and labels."""
    unit, y, sc = [], [], []
    for j in range(n):
        m = int(rng.integers(*size)); mu = rng.normal(0, shift_sd)
        lab = (rng.random(m) < pos_rate).astype(int)
        z = rng.normal(mu, 1, m) + 2.0 * lab
        unit += [f"u{j:04d}"] * m; y += list(lab); sc += list(1 / (1 + np.exp(-(z - 4))))
    return np.array(unit), np.array(y), np.array(sc)


def t1_loo_hp_exact():
    rng = np.random.default_rng(131); ok = True
    for _ in range(300):
        n = int(rng.integers(10, 60)); t = rng.choice(D.GRID[:400], n)          # ties likely
        fast, feas = H.loo_hp(t)
        for i in range(n):
            others = np.delete(t, i); r = math.ceil(n * (1 - H.DELTA))         # k + 1 = n
            ref = np.sort(others)[r - 1] if r <= n - 1 else np.inf
            ok &= fast[i] == ref
    check("T1 leave-one-out HP threshold = brute force (with ties)", ok)


def t2_loo_crc_c1_exact():
    rng = np.random.default_rng(132); ok_crc = ok_c1 = True; nflip = 0
    for _ in range(40):
        n = int(rng.integers(5, 30))
        negs = [np.round(rng.beta(1, 25, int(rng.integers(20, 300))), 3) for _ in range(n)]   # rounding -> ties
        curves = [D.loss_curve(x) for x in negs]
        for a in (0.05, 0.10, 0.2):
            lam, feas = H.loo_crc(curves, a); c1 = H.loo_c1(negs, a)
            for i in range(n):
                oth = [c for j, c in enumerate(curves) if j != i]
                r = D.crc_threshold(oth, a)
                if not (lam[i] == r["lam"] and feas[i] == r["feasible"]):
                    nflip += 1
                ref = D.pooled_split_conformal(np.concatenate([x for j, x in enumerate(negs) if j != i]), a)
                ok_c1 &= c1[i] == ref
    check("T2a leave-one-out CRC (total minus own curve) = brute force", nflip == 0, f"{nflip} mismatches")
    check("T2b leave-one-out C1 (no re-sort) = brute-force split conformal (with ties)", ok_c1)


def t3_h5c_binomial():
    ex = np.zeros((5, 50)); counts = [3, 3, 4, 3, 3]                         # mean 3.2 -> x = 4
    for s, c in enumerate(counts):
        ex[s, :c] = 1
    r = H.h5c_test(ex, 50)
    ex2 = np.zeros((5, 60)); ex2[:, :6] = 1                                  # mean exactly 6.0 -> x = 6
    r2 = H.h5c_test(ex2, 60)
    check("T3 H5c: x = ceil(seed mean), p = P(Bin(n, 0.20) <= x)",
          r["x"] == 4 and abs(r["p"] - binom.cdf(4, 50, 0.2)) < 1e-15 and r2["x"] == 6
          and abs(r2["p"] - binom.cdf(6, 60, 0.2)) < 1e-15, f"x {r['x']}, p {r['p']:.6f}; x {r2['x']}")


def t4_holm():
    adj, rej = S.holm([0.01, 0.04, np.nan], 0.05)
    check("T4 Holm within family (NaN = not testable, never rejected)",
          abs(adj[0] - 0.02) < 1e-12 and abs(adj[1] - 0.04) < 1e-12 and np.isnan(adj[2])
          and list(rej) == [True, True, False])


def make_input(rng, n_test, n_val=12, pos_rate=0.02, seeds=(131, 132), fph=3600.0):
    u, y, sc = synth_units(rng, n_test + n_val, pos_rate=pos_rate)
    ids = np.unique(u); split = np.where(np.isin(u, ids[:n_val]), "val", "test")
    inp = dict(unit=u, split=split, known=np.ones(len(u), bool), y=y,
               event=np.array([f"e{i}" for i in range(len(u))]), frames_per_hour=fph)
    for s in seeds:
        inp[f"score_s{s}"] = np.clip(sc + rng.normal(0, 0.002, len(sc)), 0, 1)
    return inp


def t5_not_testable_rules():
    rng = np.random.default_rng(133)
    H.SEEDS = [131, 132]; H.B = 200
    _, T19, _, _ = H.analyse(make_input(rng, 19), "test_n19", 100)
    _, T30, _, _ = H.analyse(make_input(rng, 30), "test_n30", 100)
    _, T60, _, c60 = H.analyse(make_input(rng, 60, pos_rate=0.0008), "test_fewevents", 100)
    _, Tk5, _, _ = H.analyse(make_input(rng, 60, n_val=5), "test_k5", 100)
    p19 = T19.set_index("test").p; p30 = T30.set_index("test").p
    p60 = T60.set_index("test").p; pk5 = Tk5.set_index("test").p
    check("T5 not-testable rules: n<20 -> all; n<50 -> H5a,H5c; <50 events -> H5b; <9 validation units -> H5a,H5c",
          p19.isna().all() and np.isfinite(p30["H5b"]) and p30[["H5a", "H5c"]].isna().all()
          and c60["test_events"] < 50 and np.isnan(p60["H5b"]) and p60[["H5a", "H5c"]].notna().all()
          and pk5[["H5a", "H5c"]].isna().all() and np.isfinite(pk5["H5b"]),
          f"events in few-events case {c60['test_events']}")


def t11_deployment_not_mechanical():
    """Deployment design through the real pipeline: share random (not fixed), mean <= delta + MC error."""
    rng = np.random.default_rng(136); sh = []
    for _ in range(40):
        u, y, sc = synth_units(rng, 100)
        ids = np.unique(u); val, test = list(ids[:40]), list(ids[40:])
        units = {k: dict(split="val" if k in val else "test", neg={131: sc[(u == k) & (y == 0)]},
                         pos={131: sc[(u == k) & (y == 1)]}, neg_ix=np.flatnonzero((u == k) & (y == 0)),
                         pos_ix=np.flatnonzero((u == k) & (y == 1))) for k in ids}
        RD = H.per_unit_results(units, test, [131], [("0.05", 0.05)], "DEPLOY", val)
        sh.append(RD[RD.method == "HP"].exceed.mean())
    m, sd = np.mean(sh), np.std(sh, ddof=1)
    check("T11 deployment design: HP share varies across datasets and its mean is <= delta + 3 SE",
          sd > 0.01 and m <= 0.10 + 3 * sd / np.sqrt(len(sh)), f"mean {m:.3f}, SD {sd:.3f}")


def t12_deployment_detects_optimism():
    """If validation units look safer than new units, H5a must NOT be supported."""
    rng = np.random.default_rng(137)
    inp = make_input(rng, 80, n_val=30)
    val = inp["split"] == "val"
    for s in (131, 132):
        inp[f"score_s{s}"] = np.where(val, inp[f"score_s{s}"] * 0.3, inp[f"score_s{s}"])   # validation looks safer
    _, T, _, _ = H.analyse(inp, "test_optimism", 100)
    t = T.set_index("test")
    check("T12 deployment design detects optimistic validation units (H5a not supported)",
          (not bool(t.loc["H5a", "supported"])) and t.loc["H5a", "estimate"] > 0.2, f"H5a share {t.loc['H5a', 'estimate']:.3f}")


def t6_bootstrap_direction():
    n = 40; boot = S.ClusterBootstrap(np.array([f"u{i:02d}" for i in range(n)]), B=500, seed=131)
    A_pos = {"C1": {"exceed": np.ones((2, n))}, "HP": {"exceed": np.zeros((2, n))}}
    A_null = {"C1": {"exceed": np.zeros((2, n))}, "HP": {"exceed": np.zeros((2, n))}}
    r1 = S.paired_test(boot, H.share_diff_stat(A_pos), 0.0); r0 = S.paired_test(boot, H.share_diff_stat(A_null), 0.0)
    check("T6 bootstrap p (share difference): clear effect -> 1/(B+1); no effect -> 1",
          abs(r1["p"] - 1 / 501) < 1e-12 and r0["p"] == 1.0 and r1["estimate"] == 1.0, f"{r1['p']:.4f}, {r0['p']}")


def t7_recall_pooled():
    A = {"HP": {"caught": np.array([[1, 2, 0]]), "n_pos": np.array([[2, 4, 4]])},
         "C1": {"caught": np.array([[2, 3, 1]]), "n_pos": np.array([[2, 4, 4]])}}
    st = H.h5b_stat(A)
    v = st(np.array([0, 1, 2])); v2 = st(np.array([1, 1])); v0 = H.h5b_stat(
        {"HP": {"caught": np.array([[0]]), "n_pos": np.array([[0]])}, "C1": {"caught": np.array([[0]]), "n_pos": np.array([[0]])}})(np.array([0]))
    check("T7 H5b recall pooled over units (frames), undefined when no positives",
          abs(v - (3 / 10 - 6 / 10)) < 1e-12 and abs(v2 - (4 / 8 - 6 / 8)) < 1e-12 and np.isnan(v0))


def t8_validity_monte_carlo():
    """Exchangeable units: per-unit exceedance of HP at eps must be <= delta (+ MC error), C1 typically higher."""
    rng = np.random.default_rng(134); sh_hp, sh_c1 = [], []
    for _ in range(60):
        u, y, sc = synth_units(rng, 40)
        ids = np.unique(u)
        units = {k: dict(split="test", neg={131: sc[(u == k) & (y == 0)]}, pos={131: sc[(u == k) & (y == 1)]},
                         neg_ix=np.flatnonzero((u == k) & (y == 0)), pos_ix=np.flatnonzero((u == k) & (y == 1)))
                 for k in ids}
        R = H.per_unit_results(units, list(ids), [131], [("0.05", 0.05)], "LOO")
        sh_hp.append(R[R.method == "HP"].exceed.mean()); sh_c1.append(R[R.method == "C1"].exceed.mean())
    m, se = np.mean(sh_hp), np.std(sh_hp, ddof=1) / np.sqrt(len(sh_hp))
    check("T8 validity (Monte Carlo): HP share of units above eps <= delta + 3 SE; C1 higher under heterogeneity",
          m <= 0.10 + 3 * se and np.mean(sh_c1) > m, f"HP {m:.3f} (SE {se:.3f}), C1 {np.mean(sh_c1):.3f}")


def t9_determinism():
    rng1, rng2 = np.random.default_rng(135), np.random.default_rng(135)
    _, Ta, _, _ = H.analyse(make_input(rng1, 55), "test_det_a", 100)
    _, Tb, _, _ = H.analyse(make_input(rng2, 55), "test_det_b", 100)
    check("T9 identical input -> identical results (fixed bootstrap seed)",
          Ta[["estimate", "p"]].fillna(-1).equals(Tb[["estimate", "p"]].fillna(-1)))


def t10_eligibility_counts():
    u = np.array(["a"] * 150 + ["b"] * 80 + ["c"] * 120); y = np.zeros(len(u), int); y[230:] = 1
    sc = np.linspace(0, 1, len(u)); kn = np.ones(len(u), bool)
    units, cnt = H.build_units(u, np.array(["test"] * len(u)), kn, y, {131: sc}, 100)
    check("T10 eligibility on counts only (b < 100 excluded), unit with no negatives excluded and counted",
          set(units) == {"a"} and cnt["excluded_eligibility"] == 1 and cnt["excluded_zero_negatives"] == 1)


if __name__ == "__main__":
    t0 = time.time()
    for f in (t1_loo_hp_exact, t2_loo_crc_c1_exact, t3_h5c_binomial, t4_holm, t5_not_testable_rules,
              t6_bootstrap_direction, t7_recall_pooled, t8_validity_monte_carlo, t9_determinism, t10_eligibility_counts,
              t11_deployment_not_mechanical, t12_deployment_detects_optimism):
        f()
    for f in Path("results").glob("h5_test_*.csv"):
        f.unlink()                                          # remove the synthetic test outputs
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(0 if all(RESULTS) else 1)
