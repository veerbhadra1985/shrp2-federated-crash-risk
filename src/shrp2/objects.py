"""
Radar-object conventions for SHRP2 model inputs and surrogate safety measures - the audit-dependent choice A5
(docs/SHRP2_runbook.md). Both options are implemented and tested; the author freezes ONE of each after the
audit (section K measures the reference point) and BEFORE any model is trained.

  size  'zero' : object length and width unknown -> 0 (a point; build_samples.py turns missing sizes into 0)
        'car'  : 4.5 m x 1.75 m for every radar object (GSSM's 'CAR' dimensions, Jiao et al. code)
  ref   'asgiven'          : use x_ekf, y_ekf as delivered (the radar-detected point)
        'centre_from_edge' : the radar sees the object's edge nearest to the subject; move the point by half
                             the object length along the object's heading axis, AWAY from the subject:
                             centre = p + (L/2) * u,  u = +h if (p - subject) . h >= 0 else -h
                             (needs size 'car'; a zero length would make it a no-op). If the subject frame
                             is missing or masked, u = +h (object taken as ahead).
"""
import numpy as np

CAR_L, CAR_W = 4.5, 1.75
SIZES, REFS = ("zero", "car"), ("asgiven", "centre_from_edge")


def check(ref, size):
    if size not in SIZES or ref not in REFS:
        raise ValueError(f"object rules must be one of size {SIZES}, ref {REFS}")
    if ref == "centre_from_edge" and size != "car":
        raise ValueError("centre_from_edge needs a length: use size 'car'")


def apply(ox, oy, oh, ex, ey, ref, size):
    """Object positions (arrays) -> (x, y, length, width) under the chosen rules.
    ex, ey: subject-vehicle position at the same frames (same shape)."""
    check(ref, size)
    L = np.full(np.shape(ox), CAR_L if size == "car" else 0.0)
    W = np.full(np.shape(ox), CAR_W if size == "car" else 0.0)
    if ref == "asgiven":
        return np.asarray(ox, float), np.asarray(oy, float), L, W
    hx, hy = np.cos(oh), np.sin(oh)
    dot = (np.asarray(ox) - ex) * hx + (np.asarray(oy) - ey) * hy
    away = np.where(np.isnan(dot), True, dot >= 0)          # subject frame missing/masked: treat as ahead (+h)
    s = np.where(away, 1.0, -1.0)
    return ox + s * (L / 2) * hx, oy + s * (L / 2) * hy, L, W
