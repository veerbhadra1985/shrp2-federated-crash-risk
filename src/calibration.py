"""
Calibration methods (Steps 14-15), kept modular so a Phase 2 method can replace them.
NumPy/scikit-learn only. All methods work on multi-horizon hazard outputs:
  logits  (n, K)  per-bin hazard logits from the model
  risk    (n, K)  cumulative risk P(event within horizon k), non-decreasing in k
Labels per sample: event_bin (-1 = no event observed), obs_bins (fully observed bins).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def _logsig(z):
    return -np.logaddexp(0.0, -z)


def risk_from_logits(logits):
    return 1.0 - np.exp(np.cumsum(_logsig(-logits), axis=1))


def hazard_nll_np(logits, event_bin, obs_bins):
    n, K = logits.shape
    j = np.arange(K)[None, :]
    is_ev = (event_bin >= 0)[:, None]
    upto = np.where(is_ev, event_bin[:, None], obs_bins[:, None])
    surv = (j < upto)
    ev = is_ev & (j == event_bin[:, None])
    return -((_logsig(-logits) * surv).sum(1) + (_logsig(logits) * ev).sum(1)).mean()


def labels_at(event_bin, obs_bins, k):
    """Binary label for horizon k and mask of samples whose outcome is known."""
    y = (event_bin >= 0) & (event_bin <= k)
    return y.astype(int), (y | (obs_bins >= k + 1))


class Raw:
    name = "raw"
    def fit(self, logits, ev, ob): return self
    def transform(self, logits): return risk_from_logits(logits)


class Temperature:
    """One temperature for all hazard logits, fitted by minimising survival NLL.
    Keeps risk non-decreasing across horizons by construction."""
    name = "temperature"
    def fit(self, logits, ev, ob):
        grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
        nll = [hazard_nll_np(logits / T, ev, ob) for T in grid]
        self.T = float(grid[int(np.argmin(nll))]); return self
    def transform(self, logits): return risk_from_logits(logits / self.T)


class Isotonic:
    """Per-horizon isotonic regression on cumulative risk; a running maximum
    across horizons restores non-decreasing risk after calibration."""
    name = "isotonic"
    def fit(self, logits, ev, ob):
        r = risk_from_logits(logits); self.models = []
        for k in range(r.shape[1]):
            y, ok = labels_at(ev, ob, k)
            self.models.append(IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(r[ok, k], y[ok]))
        return self
    def transform(self, logits):
        r = risk_from_logits(logits)
        out = np.column_stack([m.predict(r[:, k]) for k, m in enumerate(self.models)])
        return np.maximum.accumulate(out, axis=1)


def conformal_threshold(scores_neg, alpha):
    """Split-conformal threshold on NEGATIVE calibration scores: warn if score > tau.
    Under exchangeability, P(false alarm) <= alpha."""
    s = np.sort(np.asarray(scores_neg)); n = len(s)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if r > n else float(s[r - 1])


def ece(y, p, bins=10):
    """Equal-width ECE."""
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def ece_adaptive(y, p, bins=10):
    """Equal-mass ECE (each bin holds ~n/bins samples) - more informative for rare events."""
    o = np.argsort(p); chunks = np.array_split(o, bins)
    return sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c))


def reliability_bins(y, p, bins=10):
    """Equal-mass reliability-diagram points: mean prediction, observed rate, count."""
    o = np.argsort(p)
    return [(float(p[c].mean()), float(y[c].mean()), int(len(c))) for c in np.array_split(o, bins) if len(c)]