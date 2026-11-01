"""
Step B3: unit tests for src/phase2/domain_crc.py on synthetic domains with answers known in advance
(spec docs/phase2_method_spec_B2.md, Section 6, tests 1-6). Run:
    python src/phase2/test_domain_crc.py            (about 2-4 minutes on a laptop)
Prints PASS/FAIL per test; exits with an error if any test fails. Seed 131 throughout.
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2 import domain_crc as D
from calibration import conformal_threshold

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)


def domains(rng, k1, shift_last=0.0):
    """k1 heterogeneous domains drawn i.i.d. (exchangeable): negatives' risk scores on the probability
    scale, unequal sizes 50-400. shift_last moves the LAST domain's mean (non-exchangeable case)."""
    mu = rng.normal(-4.0, 1.0, k1); sd = rng.uniform(0.5, 1.5, k1); n = rng.integers(50, 401, k1)
    mu[-1] += shift_last
    return [1 / (1 + np.exp(-rng.normal(mu[j], sd[j], n[j]))) for j in range(k1)]


def mc_fa(rng, k, alpha, reps, shift_last=0.0):
    fa, feas = [], 0
    for _ in range(reps):
        S = domains(rng, k + 1, shift_last)
        res = D.crc_threshold([D.loss_curve(s) for s in S[:k]], alpha)
        if res["feasible"]:
            feas += 1; fa.append(D.false_alarm_rate(S[k], res["lam"]))
    fa = np.array(fa)
    return fa.mean(), fa.std(ddof=1) / np.sqrt(len(fa)), feas / reps


def t1_validity():
    rng = np.random.default_rng(131)
    for k, alpha in ((5, 0.25), (11, 0.10), (22, 0.10), (22, 0.05), (50, 0.05)):
        m, se, f = mc_fa(rng, k, alpha, 2000)
        check(f"T1 validity k={k} alpha={alpha}", f == 1.0 and m <= alpha + 3 * se,
              f"mean unseen-domain FA {m:.4f} (MC SE {se:.4f}) <= {alpha}; feasible {f:.2f}")


def t2_feasibility():
    rng = np.random.default_rng(131); ok = True
    for k in (3, 5, 11, 19, 22, 43):
        S = domains(rng, k); C = [D.loss_curve(s) for s in S]
        a0 = 1.0 / (k + 1)
        ok &= D.crc_threshold(C, a0)["feasible"] and not D.crc_threshold(C, a0 - 1e-9)["feasible"]
        # at the boundary the rule needs ZERO calibration false alarms: lam = first grid point >= max score
        top = max(s.max() for s in S)
        ok &= D.crc_threshold(C, a0)["lam"] == D.GRID[np.searchsorted(D.GRID, top, side="left")]
    check("T2 infeasible exactly when alpha < 1/(k+1)", ok)


def t3_high_probability():
    rng = np.random.default_rng(131)
    for k, eps, delta in ((11, 0.10, 0.10), (22, 0.05, 0.10), (43, 0.05, 0.10)):
        exc = []
        for _ in range(2000):
            S = domains(rng, k + 1)
            res = D.hp_threshold([D.loss_curve(s) for s in S[:k]], eps, delta)
            exc.append(D.false_alarm_rate(S[k], res["lam"]) > eps)
        p = np.mean(exc); se = np.sqrt(p * (1 - p) / len(exc))
        check(f"T3 high-probability k={k} eps={eps} delta={delta}", p <= delta + 3 * se,
              f"P(FA > eps) {p:.3f} (SE {se:.3f}) <= {delta}")
    check("T3b high-probability infeasible when delta < 1/(k+1)",
          not D.hp_threshold([D.loss_curve(s) for s in domains(rng, 8)], 0.1, 0.10)["feasible"])


def t4_federated():
    rng = np.random.default_rng(131); S = domains(rng, 30)
    msgs = [D.client_message(s) for s in S]
    central = D.crc_threshold([D.loss_curve(s) for s in S], 0.10)
    same, sums = True, []
    for _ in range(20):
        order = rng.permutation(len(msgs))
        fed = D.server_threshold([msgs[i] for i in order], 0.10)
        same &= fed == central; sums.append(D.exact_sum(np.array([msgs[i] for i in order])))
    bitwise = all(np.array_equal(sums[0], s) for s in sums)
    check("T4 federated = centralised, invariant to client order (bit for bit)", same and bitwise,
          f"lam {central['lam']:.6g}; message size {msgs[0].nbytes / 1024:.1f} kB per client")


def t5_grid():
    rng = np.random.default_rng(131); mono, conservative = True, True
    for _ in range(300):
        S = domains(rng, 12); C = np.array([D.loss_curve(s) for s in S[:11]])
        mono &= bool(np.all(np.diff(C, axis=1) <= 0)) and bool(np.all(C[:, -1] == 0))
        g = D.crc_threshold(C, 0.10)
        # continuous-scale lam_hat: the rule can only change at a pooled score value (or 0)
        cand = np.r_[0.0, np.unique(np.concatenate(S[:11]))]
        mean_at = np.array([np.mean([np.mean(s > c) for s in S[:11]]) for c in cand])
        ok = (11 / 12) * mean_at + 1 / 12 <= 0.10
        lam_c = cand[np.argmax(ok)] if ok.any() else np.inf
        conservative &= g["lam"] >= lam_c and D.false_alarm_rate(S[11], g["lam"]) <= D.false_alarm_rate(S[11], lam_c)
    check("T5 curves non-increasing with L(1)=0; grid lam_hat >= continuous lam_hat", mono and conservative)


def t6_known_failure():
    rng = np.random.default_rng(131)
    m, se, _ = mc_fa(rng, 22, 0.05, 1000, shift_last=+2.0)
    check("T6 shifted (non-exchangeable) new domain -> violation detected", m > 0.05 + 3 * se,
          f"mean FA {m:.4f} > 0.05 as expected; the guarantee does not cover this case")


def t7_c1_matches_registered():
    rng = np.random.default_rng(131); ok = True
    for _ in range(200):
        s = rng.random(rng.integers(5, 500)); a = rng.choice([0.05, 0.01, 0.1])
        ok &= D.pooled_split_conformal(s, a) == conformal_threshold(s, a)
    check("T7 comparator C1 identical to the registered split-conformal rule", ok)


if __name__ == "__main__":
    t0 = time.time()
    for f in (t1_validity, t2_feasibility, t3_high_probability, t4_federated, t5_grid, t6_known_failure,
              t7_c1_matches_registered):
        f()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(0 if all(RESULTS) else 1)