"""
Unit tests for the analysis engine (src/analysis/stats.py). Every test has an answer
known in advance. Run:  python src/analysis/test_stats.py
Prints PASS/FAIL per test; exits with an error if any test fails.
Test 9 also measures how often each false-alarm interval puts its lower bound above the
TRUE false-alarm rate when frames are autocorrelated - evidence for amendment 1.
"""
import sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis.stats as S
from calibration import ece_adaptive as ece_reference, conformal_threshold

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def t1_average_precision():
    rng = np.random.default_rng(0); worst = 0.0
    for _ in range(200):
        n = rng.integers(20, 400); y = (rng.random(n) < rng.uniform(0.02, 0.5)).astype(int)
        if y.sum() == 0:
            continue
        s = np.round(rng.random(n) + y * rng.uniform(0, 1), rng.integers(1, 3))   # many ties
        worst = max(worst, abs(S.average_precision(y, s) - average_precision_score(y, s)))
    check("1 AUPRC identical to sklearn (200 random cases, with ties)", worst < 1e-12, f"max diff {worst:.1e}")


def t2_ece():
    rng = np.random.default_rng(1); p = rng.random(1000); y = (rng.random(1000) < p).astype(int)
    a, b = S.ece_adaptive(y, p), ece_reference(y, p)
    check("2 adaptive ECE identical to calibration.py", abs(a - b) < 1e-12, f"{a:.6f} vs {b:.6f}")


def t3_holm():
    adj, rej = S.holm([0.01, 0.04, 0.03, 0.005])
    ok = np.allclose(adj, [0.03, 0.06, 0.06, 0.02]) and list(rej) == [True, False, False, True]
    adj2, rej2 = S.holm([0.01, np.nan, 0.02])
    ok2 = np.allclose(adj2[[0, 2]], [0.02, 0.02]) and np.isnan(adj2[1]) and not rej2[1]
    check("3 Holm adjustment (textbook example; NaN = not testable)", ok and ok2, f"{np.round(adj, 3)}")


def t4_clopper_pearson():
    lo, hi = S.clopper_pearson(5, 10); lo0, hi0 = S.clopper_pearson(0, 10)
    ok = abs(lo - 0.187086) < 1e-5 and abs(hi - 0.812914) < 1e-5 and lo0 == 0 and abs(hi0 - (1 - 0.025 ** 0.1)) < 1e-9
    check("4 Clopper-Pearson (k=5,n=10 -> 0.1871-0.8129; k=0 -> 0-0.3085)", ok, f"{lo:.4f}-{hi:.4f}, {lo0}-{hi0:.4f}")


def t5_bootstrap_se():
    rng = np.random.default_rng(2); x = rng.normal(size=400)
    boot = S.ClusterBootstrap(np.arange(400), B=2000, seed=3)
    se = np.std([x[boot.indices(b)].mean() for b in range(boot.B)])
    check("5 bootstrap SE of a mean (independent data) ~ 1/sqrt(400) = 0.050", 0.045 < se < 0.055, f"{se:.4f}")


def t6_cluster_effect():
    rng = np.random.default_rng(4); clusters = np.repeat(np.arange(20), 100)
    x = rng.normal(size=20)[clusters] + 0.1 * rng.normal(size=2000)        # strong within-cluster correlation
    se_c = np.std([x[i].mean() for i in (S.ClusterBootstrap(clusters, B=1000, seed=5).indices(b) for b in range(1000))])
    se_n = np.std([x[i].mean() for i in (S.ClusterBootstrap(np.arange(2000), B=1000, seed=5).indices(b) for b in range(1000))])
    check("6 cluster bootstrap widens SE under clustering (ratio > 5)", se_c / se_n > 5, f"cluster {se_c:.3f} vs naive {se_n:.4f}")


def t7_mask_and_pairing():
    clusters = np.repeat(["a", "b", "c"], [2, 3, 4]); boot = S.ClusterBootstrap(clusters, B=50, seed=6)
    ok = True
    for b in range(50):
        idx = boot.indices(b); n_exp = int(boot.counts[b] @ np.array([2, 3, 4]))
        ok &= len(idx) == n_exp and set(clusters[idx]) <= set("abc")
        sub = boot.indices(b, mask=clusters == "c"); ok &= len(sub) == boot.counts[b][2] * 4
    check("7 resample sizes and masks consistent", ok)


def t8_paired_test_null_and_signal():
    rng = np.random.default_rng(7); n = 3000; clusters = np.repeat(np.arange(60), 50)
    y = (rng.random(n) < 0.05).astype(int)
    good = [y + rng.normal(0, 0.6, n) for _ in range(5)]
    rand = [rng.random(n) for _ in range(5)]
    boot = S.ClusterBootstrap(clusters, B=500, seed=8)
    sig = S.paired_test(boot, S.seed_mean_difference(S.average_precision, y, good, rand))
    same = S.paired_test(boot, S.seed_mean_difference(S.average_precision, y, good, good))
    check("8a real AUPRC difference -> p < 0.01", sig["p"] < 0.01, f"est {sig['estimate']:.3f}, p {sig['p']:.4f}")
    check("8b identical models -> estimate 0 and p = 1", same["estimate"] == 0 and same["p"] == 1.0, f"p {same['p']}")
    # H2 statistic: out-of-domain predictions twice too high -> log ratio > 0; identical -> 0
    p_true = np.clip(rng.beta(1, 20, n), 1e-4, 0.5); y2 = (rng.random(n) < p_true).astype(int)
    pair = dict(y=y2, mask=np.ones(n, bool), inn=[p_true] * 5, out=[np.clip(2 * p_true, 0, 1)] * 5)
    shifted = S.paired_test(boot, S.mean_log_ratio([pair]))
    pair0 = dict(pair, out=[p_true] * 5)
    null = S.paired_test(boot, S.mean_log_ratio([pair0]))
    check("8c H2 log-ratio detects 2x over-prediction (p < 0.01)", shifted["p"] < 0.01, f"est {shifted['estimate']:.2f}")
    check("8d H2 log-ratio = 0 for identical predictions (p = 1)", null["estimate"] == 0 and null["p"] == 1.0)


def t9_false_alarm_intervals(reps=200):
    """Frames inside a cluster are strongly correlated (cluster effect sd 1, frame noise
    sd 0.2). For each replicate, a conformal threshold is set on calibration clusters and
    the false-alarm rate is measured on new test clusters. The TRUE false-alarm rate for
    that threshold is known exactly: P(N(0, 1.04) > tau). A valid 95% interval should have
    its lower bound above the true rate in only about 2.5% of replicates."""
    from scipy.stats import norm
    rng = np.random.default_rng(9); alpha = 0.05; n_cl, per = 40, 100
    cp_miss = cl_miss = 0
    for r in range(reps):
        cal = rng.normal(size=n_cl)[:, None] + 0.2 * rng.normal(size=(n_cl, per))
        tau = conformal_threshold(cal.ravel(), alpha)
        true_fa = norm.sf(tau / np.sqrt(1.04))
        test = rng.normal(size=n_cl)[:, None] + 0.2 * rng.normal(size=(n_cl, per))
        scores = test.ravel(); clusters = np.repeat(np.arange(n_cl), per)
        k, n = int((scores > tau).sum()), scores.size
        cp_miss += S.clopper_pearson(k, n)[0] > true_fa
        boot = S.ClusterBootstrap(clusters, B=300, seed=r)
        cl_miss += S.false_alarm_interval_cluster(boot, scores, tau, np.ones(n, bool))[1] > true_fa
    cp_rate, cl_rate = cp_miss / reps, cl_miss / reps
    check("9a cluster interval: lower bound wrongly above the true rate rarely (<= 6%; nominal 2.5%)",
          cl_rate <= 0.06, f"{cl_rate:.1%}")
    check("9b Clopper-Pearson on correlated frames: wrongly above the true rate far more often (> 4x)",
          cp_rate > 4 * max(cl_rate, 0.01), f"CP {cp_rate:.1%} vs cluster {cl_rate:.1%}")
    return cp_rate, cl_rate


def t10_discard_rule():
    y = np.zeros(1000, int); y[:5] = 1; clusters = np.repeat(np.arange(10), 100)   # positives in one cluster only
    boot = S.ClusterBootstrap(clusters, B=400, seed=10); s = np.random.default_rng(11).random(1000)
    res = S.paired_test(boot, lambda idx: S.average_precision(y[idx], s[idx]))
    expected = (0.9 ** 10) * 400                                                    # resamples missing cluster 0
    check("10 undefined resamples discarded, counted and flagged", res["discarded"] > 0 and res["unreliable"]
          and abs(res["discarded"] - expected) < 4 * np.sqrt(expected), f"{res['discarded']} discarded, ~{expected:.0f} expected")


if __name__ == "__main__":
    t0 = time.time()
    for t in (t1_average_precision, t2_ece, t3_holm, t4_clopper_pearson, t5_bootstrap_se, t6_cluster_effect,
              t7_mask_and_pairing, t8_paired_test_null_and_signal, t9_false_alarm_intervals, t10_discard_rule):
        t()
    print(f"\n{sum(RESULTS)} of {len(RESULTS)} checks passed in {time.time() - t0:.0f} s")
    if not all(RESULTS):
        sys.exit(1)