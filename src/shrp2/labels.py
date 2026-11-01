"""
Registered prediction times and outcome labels for SHRP2 (pre-registration osf.io/y5zrp, Section 3;
Amendment 2, Section 2). Single implementation used by the audit (counts), the sample builder (training and
evaluation samples) and the tests, so that counted and used frames can never differ.

Prediction time t (a subject-vehicle frame on the 0.1-s grid) is valid when
  * the 20 grid frames t-1.9 s .. t all exist (2 s of history, as build_samples.py: window t-19..t), and
  * the subject-vehicle frame at t is not masked as implausible (masked frames count as missing; registered).
Labels (integer milliseconds; te = annotated impact / closest-approach time):
  * excluded  if t >= te (frames after the event) or 5 s < te - t <= 7 s (ambiguous buffer);
  * event_bin 0, 1, 2, 3 if te - t lies in (0,1], (1,2], (2,3], (3,5] s; -1 if no event within 5 s
    (all frames of baseline trips);
  * obs_bins  number of the bins (1, 2, 3, 5 s) fully observed before the trip's last subject frame
    (censoring, as build_hazard_labels.py).
Label at horizon k (0-based; 3 s = k 2) and "known" follow calibration.labels_at:
  y = 0 <= event_bin <= k;  known = y or obs_bins >= k + 1.
"""
import numpy as np

HIST = 20                                   # frames of history including t
BIN_END_MS = np.array([1000, 2000, 3000, 5000])
BUFFER_MS = (5000, 7000)


def prediction_index(ts_ms, masked=None):
    """Indices i of valid prediction times in a sorted, de-duplicated grid clock ts_ms."""
    ts = np.asarray(ts_ms, np.int64); n = len(ts); h = HIST - 1
    if n <= h:
        return np.zeros(0, np.int64)
    i = np.arange(h, n)
    ok = (ts[i] - ts[i - h]) == h * 100
    if masked is not None:
        ok &= ~np.asarray(masked, bool)[i]
    return i[ok]


def label_arrays(t_ms, te_ms, t_last_ms):
    """For prediction times t_ms: (excluded, event_bin, obs_bins), integer ms arithmetic."""
    t = np.asarray(t_ms, np.int64)
    excl = np.zeros(len(t), bool); ev = np.full(len(t), -1, np.int64)
    if te_ms is not None:
        tte = int(te_ms) - t
        excl = (tte <= 0) | ((tte > BUFFER_MS[0]) & (tte <= BUFFER_MS[1]))
        inwin = (~excl) & (tte <= BIN_END_MS[-1])
        ev[inwin] = np.searchsorted(BIN_END_MS, tte[inwin], side="left")   # (0,1000] -> 0 ... (3000,5000] -> 3
    last_rel = np.minimum(int(t_last_ms) - t, BIN_END_MS[-1])
    obs = (last_rel[:, None] >= BIN_END_MS[None, :]).sum(1).astype(np.int64)
    return excl, ev, obs


def known_at(ev, obs, k):
    y = (ev >= 0) & (ev <= k)
    return y.astype(int), y | (obs >= k + 1)
