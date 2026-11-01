"""
Phase 2, step B7 (spec docs/phase2_method_spec_B2.md v1.1, Section 7): published comparators, applied to the SAME
task as our method - a threshold on the risk score of NEGATIVES such that the false-alarm rate (FA) in a new unit is
controlled at level a. "Coverage 1 - a of negative scores" = "FA <= a". Each comparator is implemented from its paper;
how it is applied to an UNSEEN client is stated explicitly, because none of the papers defines that case.

C2  Lu et al. (2023, ICML, PMLR 202:22942-22964), FCP. Server quantile of the pooled client scores at rank
    ceil((1 - a)(N + K)) (N = total scores, K = clients); guarantee for a test point from the client MIXTURE with
    weights (n_k + 1)/(N + K). Clients send T-Digest sketches in the paper; here the quantile is computed EXACTLY
    (equivalent to a sketch with zero error; Corollary 4.5 with eps = 0). Unseen client: the same global threshold.
C3  Humbert et al. (2023, ICML, PMLR 202), one-shot quantile of quantiles (Fed CP-QQ). Each client sends its l-th
    smallest score; the server returns the k-th smallest of these; (l, k) = argmin { M_lk : M_lk >= 1 - a }, where
    M_lk = P(test score <= threshold) for i.i.d. continuous scores (Theorem 3.2). The paper assumes equal n; for
    unequal client sizes M_lk is computed EXACTLY here for the actual sizes (uniform-score representation:
    Q_j ~ Beta(l, n_j + 1 - l) independently; M_lk = E[k-th smallest Q_j], via a Poisson-binomial recursion and
    numerical integration); l ranges over 1..min_j n_j. Unseen client: the same global threshold.
C4  Shahid (2026, arXiv:2606.20115), site-conditional federated CRC with risk-curve shrinkage. Site k uses
    w_k R_k + (1 - w_k) R_global + corr_k, w_k = n_k/(n_k + n0), R_global = sum n_k R_k / N,
    corr_k = w_k B/(n_k + 1) + (1 - w_k) B/(N + 1), B = 1, threshold = inf{lam : shrunk risk <= a} after a running
    minimum over increasing lam. Unseen client: it has no data, so n_k = 0 -> w_k = 0 and the rule becomes
    R_global(lam) + 1/(N + 1) <= a (sample-weighted pooled CRC). n0 is then irrelevant.
C5  Plassier et al. (2023, ICML, PMLR 202:27907-27947), label-shift weighted federated conformal. For scores of
    NEGATIVES only (class-conditional false-alarm control), the label-shift weight of a calibration point of agent j
    reduces to a per-agent constant proportional to 1/pi_j(0) (pi_j(0) = share of negatives at agent j); the target
    prior pi_*(0) of an unseen agent cancels in the normalisation. Weighted split-conformal quantile with the test
    point carrying the mean weight. (Class-0 reduction of the method; stated as such.)
All functions return dict(lam, feasible); lam = inf means "never warn" (infeasible).
"""
import math
import numpy as np
from scipy import stats


def c2_lu_fcp(client_scores, a):
    s = np.sort(np.concatenate(client_scores)); N, K = len(s), len(client_scores)
    r = int(math.ceil((1 - a) * (N + K)))
    return dict(lam=float(s[r - 1]) if r <= N else np.inf, feasible=r <= N, rank=r)


_X = np.unique(np.r_[0.0, np.logspace(-9, np.log10(0.5), 2000), 1.0 - np.logspace(np.log10(0.5), -9, 2000), 1.0])  # dense near 0 and 1


def _m_lk_all_k(sizes, l):
    """M_{l,k} for k = 1..m (exact up to quadrature): M = E[k-th smallest of independent Beta(l, n_j + 1 - l)]."""
    m = len(sizes)
    P = np.stack([stats.beta.cdf(_X, l, n + 1 - l) for n in sizes])          # P(Q_j <= x), shape (m, X)
    dist = np.zeros((len(_X), m + 1)); dist[:, 0] = 1.0                       # Poisson-binomial count distribution
    for j in range(m):
        p = P[j][:, None]
        new = dist * (1 - p); new[:, 1:] += dist[:, :-1] * p; dist = new
    cdf_count = np.cumsum(dist, axis=1)                                       # P(count <= c)
    surv = cdf_count[:, :m]                                                   # P(X_(k) > x) = P(count <= k-1), k=1..m
    return np.trapezoid(surv, _X, axis=0) if hasattr(np, "trapezoid") else np.trapz(surv, _X, axis=0)


_MCACHE = {}


def _M(sizes, l):
    key = (sizes, l)
    if key not in _MCACHE:
        _MCACHE[key] = _m_lk_all_k(list(sizes), l)
    return _MCACHE[key]


def c3_humbert_qq(client_scores, a, coarse=60):
    """(l, k) = argmin {M_lk : M_lk >= 1 - a}. l is searched on a grid of about `coarse` values, then refined
    around the best grid point (any selected pair has M_lk >= 1 - a, so validity never depends on the search)."""
    sizes = tuple(sorted(len(s) for s in client_scores)); nmin = sizes[0]
    def best_on(ls):
        best = None
        for l in ls:
            M = _M(sizes, l); ok = np.flatnonzero(M >= 1 - a)
            if len(ok):
                k = int(ok[0]) + 1
                if best is None or M[k - 1] < best[2]:
                    best = (l, k, float(M[k - 1]))
        return best
    step = max(1, int(math.ceil(nmin / coarse)))
    b = best_on(sorted(set(list(range(nmin, 0, -step)) + [nmin])))
    if b is None:
        return dict(lam=np.inf, feasible=False, l=None, k=None, M=None)
    b2 = best_on(range(max(1, b[0] - step), min(nmin, b[0] + step) + 1))
    l, k, Mv = b2 if (b2 is not None and b2[2] <= b[2]) else b
    q = np.sort([np.sort(s)[l - 1] for s in client_scores])
    return dict(lam=float(q[k - 1]), feasible=True, l=l, k=k, M=Mv)


def c4_shahid_unseen(curves, sizes, a, grid):
    """Unseen site: w = 0 -> R_global + 1/(N+1) <= a (running minimum applied as in Algorithm 1)."""
    sizes = np.asarray(sizes, float); N = sizes.sum()
    R = (np.asarray(curves) * sizes[:, None]).sum(0) / N
    R = np.minimum.accumulate(R)                                               # curves already non-increasing
    ok = R + 1.0 / (N + 1) <= a
    if not ok.any():
        return dict(lam=np.inf, feasible=False)
    return dict(lam=float(grid[int(np.argmax(ok))]), feasible=True)


def c5_plassier_class0(client_scores, client_neg_share, a):
    w_agent = 1.0 / np.asarray(client_neg_share, float)
    s = np.concatenate(client_scores)
    w = np.concatenate([np.full(len(c), wj) for c, wj in zip(client_scores, w_agent)])
    o = np.argsort(s); s, w = s[o], w[o]
    w_test = w.mean(); cw = np.cumsum(w) / (w.sum() + w_test)
    i = np.searchsorted(cw, 1 - a - 1e-12)                                     # smallest score with mass >= 1 - a
    return dict(lam=float(s[i]) if i < len(s) else np.inf, feasible=i < len(s))


def message_bytes(method, n_scores, grid_len=2802):
    """Upload per client, in bytes (float64), for the federated form of each method."""
    return {"CRC": grid_len * 8, "HP": 8, "C1": n_scores * 8, "C2": None, "C3": 8, "C4": grid_len * 8 + 8,
            "C5": n_scores * 8 + 8}[method]