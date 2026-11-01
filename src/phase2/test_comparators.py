"""
Step B7: unit tests for the published comparators (src/phase2/comparators.py), answers known in advance. Run:
    python src/phase2/test_comparators.py        (about 1-3 minutes)
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2 import comparators as K
from phase2 import domain_crc as D
from calibration import conformal_threshold

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def t1_lu_single_client():
    rng = np.random.default_rng(131); ok = True
    for _ in range(200):
        s = rng.random(rng.integers(5, 400)); a = rng.choice([0.01, 0.05, 0.1])
        ok &= K.c2_lu_fcp([s], a)["lam"] == conformal_threshold(s, a)
    check("T1 C2 (Lu) with one client = split conformal", ok)


def t2_humbert_formula():
    e1 = abs(K._m_lk_all_k([100] * 10, 100)[-1] - 1000 / 1001)
    e2 = abs(K._m_lk_all_k([30] * 4, 1)[0] - 1 / 121)
    rng = np.random.default_rng(131); sizes = [40, 60, 90]; l = 30
    mc = np.array([np.sort([np.sort(rng.random(n))[l - 1] for n in sizes]) for _ in range(20000)]).mean(0)
    e3 = np.abs(K._m_lk_all_k(sizes, l) - mc).max()
    check("T2 C3 (Humbert) coverage M_lk exact (closed forms; unequal sizes vs Monte Carlo)",
          e1 < 1e-5 and e2 < 1e-5 and e3 < 0.004, f"errors {e1:.1e}, {e2:.1e}, MC {e3:.4f}")


def t3_humbert_validity():
    rng = np.random.default_rng(131); hits, Ms = [], []
    sizes = [50, 80, 120, 200, 65]
    for _ in range(3000):
        cs = [rng.random(n) for n in sizes]
        r = K.c3_humbert_qq(cs, 0.10, coarse=20); Ms.append(r["M"])
        hits.append(rng.random() <= r["lam"])
    p = np.mean(hits); se = np.sqrt(p * (1 - p) / len(hits))
    check("T3 C3 valid on i.i.d. scores (coverage >= 1 - a) and M >= 1 - a", p >= 0.90 - 3 * se and min(Ms) >= 0.90,
          f"coverage {p:.3f} (SE {se:.3f}); min M {min(Ms):.4f}")


def t4_plassier_equal_shares():
    rng = np.random.default_rng(131); ok = True
    for _ in range(200):
        cs = [rng.random(rng.integers(5, 100)) for _ in range(rng.integers(1, 6))]; a = rng.choice([0.05, 0.1])
        ok &= K.c5_plassier_class0(cs, [0.98] * len(cs), a)["lam"] == conformal_threshold(np.concatenate(cs), a)
    check("T4 C5 (Plassier, class 0) with equal negative shares = pooled split conformal", ok)


def t5_shahid_unseen():
    rng = np.random.default_rng(131); ok = True
    for _ in range(100):
        cs = [rng.beta(1, 30, rng.integers(20, 300)) for _ in range(rng.integers(2, 8))]; a = rng.choice([0.05, 0.1])
        r = K.c4_shahid_unseen([D.loss_curve(c) for c in cs], [len(c) for c in cs], a, D.GRID)
        pooled = np.concatenate(cs); N = len(pooled)
        fa = np.array([np.mean(pooled > g) for g in D.GRID]); ref = D.GRID[int(np.argmax(fa + 1 / (N + 1) <= a))]
        ok &= r["lam"] == ref
    check("T5 C4 (Shahid) for an unseen site = sample-weighted pooled CRC (w = 0)", ok)


def t6_lu_mixture_validity():
    rng = np.random.default_rng(131); hits = []
    for _ in range(3000):
        n = rng.integers(20, 200, 6); mus = rng.normal(0, 1, 6)
        cs = [rng.normal(mus[j], 1, n[j]) for j in range(6)]
        lam = K.c2_lu_fcp(cs, 0.10)["lam"]
        j = rng.choice(6, p=(n + 1) / (n + 1).sum()); hits.append(rng.normal(mus[j], 1) <= lam)
    p = np.mean(hits); se = np.sqrt(p * (1 - p) / len(hits))
    check("T6 C2 valid for a test point from the client mixture (weights n_k + 1)", p >= 0.90 - 3 * se,
          f"coverage {p:.3f} (SE {se:.3f})")


if __name__ == "__main__":
    t0 = time.time()
    for f in (t1_lu_single_client, t2_humbert_formula, t3_humbert_validity, t4_plassier_equal_shares,
              t5_shahid_unseen, t6_lu_mixture_validity):
        f()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(0 if all(RESULTS) else 1)