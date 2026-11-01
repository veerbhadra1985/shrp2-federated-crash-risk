"""
Step 16b: training-time augmentation of NEIGHBOUR information (target history untouched).

For each sample in a batch, with probability P_APPLY, draw its own random levels:
  neighbour drop  p ~ U(0, MAX_DROP)    remove whole neighbours (missed detections)
  frame dropout   q ~ U(0, MAX_FRAMES)  remove individual neighbour frames (track flicker)
  noise           s ~ U(0, MAX_NOISE)   position noise s m, velocity noise s/2 m/s
Labels are NOT changed: a vehicle that is not observed can still cause a conflict,
so the model learns that missing information does not mean safety.
"""
import numpy as np

P_APPLY, MAX_DROP, MAX_FRAMES, MAX_NOISE = 0.5, 0.5, 0.3, 1.0


def augment(hist, rng):
    """hist: (B, A, T, 9) numpy array; returns an augmented copy."""
    h = hist.copy()
    B = h.shape[0]
    nb = h[:, 1:]
    on = rng.random(B) < P_APPLY
    p = rng.uniform(0, MAX_DROP, B) * on
    q = rng.uniform(0, MAX_FRAMES, B) * on
    s = rng.uniform(0, MAX_NOISE, B) * on
    present = nb[..., 8].sum(-1) > 0
    nb[present & (rng.random(present.shape) < p[:, None])] = 0.0
    valid = nb[..., 8] > 0
    nb[valid & (rng.random(valid.shape) < q[:, None, None])] = 0.0
    valid = (nb[..., 8] > 0)[..., None]
    nb[..., 0:2] += rng.normal(size=nb[..., 0:2].shape) * s[:, None, None, None] * valid
    nb[..., 2:4] += rng.normal(size=nb[..., 2:4].shape) * (s / 2)[:, None, None, None] * valid
    return h