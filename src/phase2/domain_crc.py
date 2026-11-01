"""
Phase 2 core method (spec docs/phase2_method_spec_B2.md v1.0, approved 10 Oct 2026):
DOMAIN-LEVEL conformal risk control of the false-alarm rate for an UNSEEN domain unit.

  unit j's loss   L_j(lam) = share of unit j's negatives with score > lam   (non-increasing, in [0,1], L_j(1)=0)
  expectation     lam_hat = min{lam in GRID : (k/(k+1)) * mean_j L_j(lam) + 1/(k+1) <= alpha}
                  -> E[L_new(lam_hat)] <= alpha if the k+1 units are exchangeable
                     (Angelopoulos et al., 2024, Thm 1, one unit = one loss, B = 1)
  high-prob.      t_j = min{lam : L_j(lam) <= eps};  lam_delta = r-th smallest t_j, r = ceil((k+1)(1-delta))
                  -> P(L_new(lam_delta) > eps) <= delta  (domain-level Neyman-Pearson umbrella)
  federated       client sends its curve on the fixed GRID once; server needs only the SUM of curves
                  (exactly rounded with math.fsum, so the result does not depend on client order)
Infeasible (no lam satisfies the rule) is returned as feasible=False, lam=inf: never warn. Never replaced.
Comparators here: C1 pooled split conformal (registered H2 method) and C6 per-unit oracle.
C2-C5 (Lu, Humbert, Shahid, Plassier) are added in step B7.
"""
import math
import numpy as np

GRID = np.r_[1.0 / (1.0 + np.exp(-np.round(np.arange(-14.0, 14.0 + 1e-9, 0.01), 2))), 1.0]   # 2,801 + lam_max
assert len(GRID) == 2802 and np.all(np.diff(GRID) > 0)


def loss_curve(scores_neg):
    """False-alarm rate of one unit at every GRID threshold (warn if score > lam)."""
    s = np.sort(np.asarray(scores_neg, float))
    if len(s) == 0:
        raise ValueError("unit has no negatives")
    return 1.0 - np.searchsorted(s, GRID, side="right") / len(s)


def unit_curves(scores, is_negative, unit, eligible=None):
    """Curves for every unit that is eligible and has at least one negative.
    Returns (unit ids sorted, (k, G) array)."""
    scores, is_negative, unit = np.asarray(scores, float), np.asarray(is_negative, bool), np.asarray(unit)
    ids = np.unique(unit if eligible is None else unit[np.asarray(eligible, bool)])
    keep, rows = [], []
    for u in ids:
        m = (unit == u) & is_negative
        if m.any():
            keep.append(u); rows.append(loss_curve(scores[m]))
    return np.array(keep), np.array(rows)


def exact_sum(curves):
    """Column sums, exactly rounded (order-independent): what secure aggregation would return."""
    c = np.asarray(curves, float)
    return np.array([math.fsum(c[:, g]) for g in range(c.shape[1])])


def crc_from_sum(curve_sum, k, alpha):
    """Server rule (expectation form). Returns dict(lam, index, feasible, k)."""
    ok = (curve_sum / (k + 1)) + 1.0 / (k + 1) <= alpha            # (k/(k+1)) * mean + 1/(k+1), with mean = sum/k
    if not ok.any():
        return dict(lam=np.inf, index=-1, feasible=False, k=int(k))
    i = int(np.argmax(ok))
    return dict(lam=float(GRID[i]), index=i, feasible=True, k=int(k))


def crc_threshold(curves, alpha):
    curves = np.asarray(curves, float)
    return crc_from_sum(exact_sum(curves), len(curves), alpha)


def unit_threshold(curve, eps):
    """Smallest GRID threshold at which this unit's false-alarm rate is <= eps (C6 oracle / t_j)."""
    return float(GRID[int(np.argmax(np.asarray(curve) <= eps))])        # L(1) = 0, so always defined


def hp_threshold(curves, eps, delta):
    """High-probability form. Returns dict(lam, feasible, k, r)."""
    k = len(curves); r = int(math.ceil((k + 1) * (1 - delta)))
    if r > k:
        return dict(lam=np.inf, feasible=False, k=k, r=r)
    t = np.sort([unit_threshold(c, eps) for c in curves])
    return dict(lam=float(t[r - 1]), feasible=True, k=k, r=r)


def pooled_split_conformal(scores_neg_pooled, alpha):
    """C1: registered split-conformal threshold on pooled negatives (same rule as calibration.py)."""
    s = np.sort(np.asarray(scores_neg_pooled, float)); n = len(s)
    r = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if r > n else float(s[r - 1])


def false_alarm_rate(scores_neg, lam):
    s = np.asarray(scores_neg, float)
    return float(np.mean(s > lam)) if len(s) else np.nan


# ---------------- federated form (what actually travels) ----------------
def client_message(scores_neg):
    """Client side: one curve of 2,802 float64 values (~22 kB) - no samples, scores or labels leave the client."""
    return loss_curve(scores_neg)


def server_threshold(messages, alpha):
    """Server side: needs only the sum and the number of clients."""
    return crc_from_sum(exact_sum(np.asarray(messages)), len(messages), alpha)