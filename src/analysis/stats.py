"""
Stage 1, step 7: the confirmatory ANALYSIS ENGINE (pre-registration, Section 6).

Everything the registered tests need, in one place, independent of any model:
  ClusterBootstrap   paired cluster bootstrap over units (drivers / events; recordings in
                     the rehearsal). ONE set of resamples is drawn and reused for every
                     model, seed and hypothesis, so all comparisons are paired.
  paired_test        one-sided test of a statistic against delta0:
                       p = (1 + #{stat_b <= delta0}) / (B + 1)      (registered formula)
                     with the point estimate on the full test data and a 95% percentile
                     interval. Resamples where the statistic is undefined are discarded
                     and counted; > 5% discarded -> flagged "unreliable" (registered rule).
  holm               Holm step-down adjustment for the confirmatory family.
  average_precision  fast AUPRC, identical to sklearn's average_precision_score.
  clopper_pearson    exact binomial interval (registered H2 secondary test; assumes
                     independent frames).
  false_alarm_interval_cluster   the same false-alarm rate with a cluster-bootstrap
                     interval (proposed amendment: respects dependence between frames).

Conventions
  - A statistic is a function stat(idx) -> float, where idx are sample indices (with
    repetitions in a bootstrap resample). It closes over the arrays it needs.
  - "Higher statistic = hypothesis direction". For lower-is-better metrics (ECE), write
    the statistic as worse - better (e.g. ECE_FedAvg - ECE_FedPer), as registered.
"""
import numpy as np
from scipy.stats import beta

B_DEFAULT, SEED_DEFAULT, MAX_DISCARD = 2000, 131, 0.05


# ----------------------------------------------------------------------------- metrics
def average_precision(y, s):
    """AUPRC = sum_n (R_n - R_{n-1}) P_n over distinct thresholds (sklearn definition)."""
    y = np.asarray(y); s = np.asarray(s, float)
    npos = y.sum()
    if npos == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort"); y, s = y[o], s[o]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]          # last index of each tie group
    tp = np.cumsum(y)[last]; fp = (last + 1) - tp
    precision = tp / (tp + fp); recall = tp / npos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def ece_adaptive(y, p, bins=10):
    """Equal-mass ECE, identical to calibration.ece_adaptive."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(p) == 0:
        return np.nan
    o = np.argsort(p, kind="mergesort")
    return float(sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean()) for c in np.array_split(o, bins) if len(c)))


# --------------------------------------------------------------------------- bootstrap
class ClusterBootstrap:
    """Resample whole clusters with replacement. counts[b, c] = copies of cluster c in resample b."""

    def __init__(self, clusters, B=B_DEFAULT, seed=SEED_DEFAULT):
        self.codes, self.names = _codes(clusters)
        self.C, self.B = len(self.names), B
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.C, size=(B, self.C))
        self.counts = np.stack([np.bincount(r, minlength=self.C) for r in draws])
        self.order = np.argsort(self.codes, kind="mergesort")         # samples grouped by cluster
        self.code_of_order = self.codes[self.order]

    def indices(self, b=None, mask=None):
        """Sample indices of resample b (None = original data), optionally restricted by mask."""
        if b is None:
            idx = np.arange(len(self.codes))
        else:
            idx = np.repeat(self.order, self.counts[b][self.code_of_order])
        return idx if mask is None else idx[np.asarray(mask)[idx]]


def _codes(clusters):
    names, codes = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    return codes, names


def paired_test(boot, stat, delta0=0.0, alpha_ci=0.05):
    """One-sided bootstrap test of H: stat > delta0 (registered p-value formula)."""
    est = stat(boot.indices(None))
    vals = np.array([stat(boot.indices(b)) for b in range(boot.B)], float)
    ok = np.isfinite(vals); v = vals[ok]; used = int(ok.sum())
    discarded = boot.B - used
    if used == 0:
        return dict(estimate=est, ci_low=np.nan, ci_high=np.nan, p=np.nan, B_used=0,
                    discarded=discarded, unreliable=True)
    p = (1 + np.sum(v <= delta0)) / (used + 1)
    lo, hi = np.quantile(v, [alpha_ci / 2, 1 - alpha_ci / 2])
    return dict(estimate=est, ci_low=float(lo), ci_high=float(hi), p=float(p), B_used=used,
                discarded=discarded, unreliable=discarded / boot.B > MAX_DISCARD)


def holm(pvals, alpha=0.05):
    """Holm step-down: adjusted p-values and rejections (NaN p = not testable, never rejected)."""
    p = np.asarray(pvals, float); adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p)); m = len(ok)
    if m:
        o = ok[np.argsort(p[ok], kind="mergesort")]
        adj[o] = np.minimum(1.0, np.maximum.accumulate(p[o] * (m - np.arange(m))))
    return adj, np.where(np.isfinite(adj), adj <= alpha, False)


# --------------------------------------------------------------- statistics builders
def seed_mean_difference(metric, y, scores_a, scores_b):
    """stat(idx) = mean over seeds of metric(A_s) - metric(B_s). scores_*: list of arrays (one per seed)."""
    def stat(idx):
        d = [metric(y[idx], a[idx]) - metric(y[idx], b[idx]) for a, b in zip(scores_a, scores_b)]
        return float(np.mean(d))
    return stat


def mean_log_ratio(pairs, metric=ece_adaptive):
    """H2 statistic. pairs: list of dicts with keys
         y (labels), mask (held-out test samples of the pair), out (list per seed: model trained
         elsewhere), inn (list per seed: model trained in-domain), all on the SAME samples.
       stat(idx) = mean over pairs of [ mean over seeds of log(metric(out_s) / metric(in_s)) ]."""
    def stat(idx):
        per_pair = []
        for pr in pairs:
            j = idx[pr["mask"][idx]]
            if len(j) == 0 or pr["y"][j].sum() == 0:
                return np.nan
            r = []
            for o, i in zip(pr["out"], pr["inn"]):
                a, b = metric(pr["y"][j], o[j]), metric(pr["y"][j], i[j])
                if not (a > 0 and b > 0):
                    return np.nan
                r.append(np.log(a / b))
            per_pair.append(np.mean(r))
        return float(np.mean(per_pair))
    return stat


# -------------------------------------------------------------------- false alarms
def clopper_pearson(k, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def false_alarm_interval_cluster(boot, scores, tau, neg_mask):
    """False-alarm rate on negatives (score > tau) with a cluster-bootstrap percentile interval."""
    alarm = np.asarray(scores) > tau; neg = np.asarray(neg_mask)
    def rate(idx):
        j = idx[neg[idx]]
        return alarm[j].mean() if len(j) else np.nan
    est = rate(boot.indices(None))
    v = np.array([rate(boot.indices(b)) for b in range(boot.B)]); v = v[np.isfinite(v)]
    lo, hi = np.quantile(v, [0.025, 0.975])
    return float(est), float(lo), float(hi)