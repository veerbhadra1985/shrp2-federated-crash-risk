"""
POST-HOC DEVELOPMENT DIAGNOSTIC (INTERACTION only; no SHRP2 data; no change to any registered rule).
Why: in the H5 rehearsal run on INTERACTION (11 Oct 2026, after commit a411dfb), the deployment design at eps_op put 28.2% of the 22 test
recordings above one alarm per vehicle-hour (HP), although HP targets <= 10% of new units, and although E1b
(1,000 random validation/test partitions, b4_e1c_operational.py) gave 7.1%. At 5% the same design gave 0.9%.

Questions, fixed BEFORE this script was first run on real data:
  Q1  Are the 22 test recordings larger (more negatives) than the 23 validation recordings?
      Read-out: median negatives per recording, validation vs test; Mann-Whitney U, two-sided.
  Q2  At eps_op every recording has fewer than 3,600 negatives, so its own threshold t_j is its largest
      negative score ("zero false alarms"). Check: t_j equals the grid value just at/above max(neg) for all units.
  Q3  Within the test set, do recordings that exceed eps_op under deployment HP have more negatives?
      Read-out: median negatives of exceeding vs non-exceeding test recordings (seed 131) and the Spearman
      correlation between negatives and the per-recording maximum negative score (all 45, seed 131).
  Q4  Is 28.2% unusual for ONE validation/test partition? Read-out: share above eps_op (seed-mean, HP,
      deployment design) over 1,000 random partitions of the same 45 recordings into 23 + 22 (seed 131);
      report the percentile of the actual partition.
Interpretation rule (fixed in advance): if Q1 shows test recordings clearly larger AND Q3 links exceedance to
size AND the actual partition sits in the upper tail of Q4 (>= 95th percentile), the result is attributed to a
UNIT-SIZE effect of the "zero alarms" target on short recordings (units not exchangeable in size), not to an
implementation error. Otherwise it is reported as unexplained. Either way the result is reported as found.
Usage : python src/phase2/diag_h5_deploy_op.py            (INTERACTION)
        python src/phase2/diag_h5_deploy_op.py --synthetic (code check only, not evidence)
Output: printed read-outs; results/h5_diag_deploy_op_<tag>.csv (Q4 partition shares)
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from phase2 import run_h5 as H
from phase2 import domain_crc as D

PARTS, PART_SEED = 1000, 131


def deploy_share_op(units, cal, test, seeds, eps):
    """Seed-mean share of test units with FA > eps under HP with threshold set on cal (deployment design)."""
    shares = []
    for s in seeds:
        lam = D.hp_threshold([D.loss_curve(units[u]["neg"][s]) for u in cal], eps, H.DELTA)["lam"]
        shares.append(np.mean([np.mean(units[u]["neg"][s] > lam) > eps for u in test]))
    return float(np.mean(shares))


def main(inp, tag):
    seeds = [s for s in H.SEEDS if f"score_s{s}" in inp]
    eps = 1.0 / float(inp["frames_per_hour"])
    scores = {s: inp[f"score_s{s}"] for s in seeds}
    units, _ = H.build_units(inp["unit"], inp["split"], inp["known"], inp["y"], scores, H.MIN_EVAL_DEFAULT)
    ids = sorted(units); val = [u for u in ids if units[u]["split"] == "val"]
    test = [u for u in ids if units[u]["split"] == "test"]
    nneg = {u: len(units[u]["neg_ix"]) for u in ids}
    s0 = seeds[0]
    print(f"[{tag}] {len(val)} validation, {len(test)} test recordings | eps_op = 1/{1 / eps:.0f} | seeds {seeds}")

    # Q1
    a, b = [nneg[u] for u in val], [nneg[u] for u in test]
    p1 = mannwhitneyu(a, b, alternative="two-sided").pvalue
    print(f"\nQ1 negatives per recording: validation median {np.median(a):.0f} (IQR {np.percentile(a, 25):.0f}-"
          f"{np.percentile(a, 75):.0f}) | test median {np.median(b):.0f} (IQR {np.percentile(b, 25):.0f}-"
          f"{np.percentile(b, 75):.0f}) | Mann-Whitney p = {p1:.3f}")

    # Q2
    ok = 0; below = sum(nneg[u] < 1 / eps for u in ids)
    for u in ids:
        neg = units[u]["neg"][s0]
        t = D.unit_threshold(D.loss_curve(neg), eps)
        g = D.GRID[np.searchsorted(D.GRID, neg.max(), side="left")]
        ok += int(np.isclose(t, g)) if nneg[u] < 1 / eps else 1
    print(f"\nQ2 recordings with < 1/eps negatives: {below}/{len(ids)}; t_j = grid value at/above max(neg): {ok}/{len(ids)}")

    # Q3
    lam = D.hp_threshold([D.loss_curve(units[u]["neg"][s0]) for u in val], eps, H.DELTA)["lam"]
    exc = [u for u in test if np.mean(units[u]["neg"][s0] > lam) > eps]
    non = [u for u in test if u not in exc]
    mx = [units[u]["neg"][s0].max() for u in ids]
    rho = spearmanr([nneg[u] for u in ids], mx)
    print(f"\nQ3 seed {s0}: {len(exc)}/{len(test)} test recordings exceed | median negatives exceeding "
          f"{np.median([nneg[u] for u in exc]) if exc else float('nan'):.0f} vs not exceeding "
          f"{np.median([nneg[u] for u in non]) if non else float('nan'):.0f}")
    print(f"   Spearman(negatives, max negative score), all {len(ids)} recordings: rho = {rho.correlation:.2f}, "
          f"p = {rho.pvalue:.3f}")

    # Q4
    actual = deploy_share_op(units, val, test, seeds, eps)
    rng = np.random.default_rng(PART_SEED); kv = len(val); sh = []
    for _ in range(PARTS):
        perm = rng.permutation(len(ids))
        sh.append(deploy_share_op(units, [ids[i] for i in perm[:kv]], [ids[i] for i in perm[kv:]], seeds, eps))
    sh = np.array(sh)
    pct = 100 * np.mean(sh < actual) + 50 * np.mean(sh == actual)
    print(f"\nQ4 actual partition share {actual:.4f} | random partitions ({PARTS}): mean {sh.mean():.4f}, median "
          f"{np.median(sh):.4f}, 5-95% {np.percentile(sh, 5):.4f}-{np.percentile(sh, 95):.4f} | actual at "
          f"percentile {pct:.1f}")
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(dict(partition=np.arange(PARTS), share_op=sh)).to_csv(f"results/h5_diag_deploy_op_{tag}.csv", index=False)

    size_effect = (np.median(b) > np.median(a) and p1 < 0.05 and exc and non
                   and np.median([nneg[u] for u in exc]) > np.median([nneg[u] for u in non]) and pct >= 95)
    print("\nINTERPRETATION (rule fixed in advance): " + ("UNIT-SIZE effect of the zero-alarm target on short recordings"
          if size_effect else "not attributed to unit size by the pre-set rule -> reported as unexplained"))
    print(f"Saved results/h5_diag_deploy_op_{tag}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
    if a.synthetic:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_run_h5 as T
        main(T.make_input(np.random.default_rng(7), 22, n_val=23, seeds=(131, 132)), "synthetic")
    else:
        main(H.interaction_input(), "interaction")
