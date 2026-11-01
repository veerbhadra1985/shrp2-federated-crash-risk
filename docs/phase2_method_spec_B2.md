# Phase 2 method specification (step B2): a domain-level false-alarm guarantee for unseen domains

Veer Bhadra Pratap Singh, Galgotias University. Version 1.0, APPROVED by the author 10 Oct 2026 (16:20 IST): D1, D2 and D3 as recommended.
**Version 1.1, APPROVED by the author 10 Oct 2026 (17:32 IST)** — see Section 0. The v1.0 text below is kept unchanged as the audit trail; where Section 0 differs, Section 0 applies.

## 0. Changes in version 1.1 (made AFTER the B4 E1 result on development data; logged as process note P2)

Why: E1 showed that (i) at R-level the pooled method meets α on average but 43–48% of individual
recordings exceed it, so per-unit control is what the method adds there, and (ii) sample-level
α = 0.05 at 1 Hz means about 100–350 false alarms per agent-hour, far from the registered
operational target of 1 false alarm per hour of baseline driving. The expectation form cannot
reach such rates, because α ≥ 1/(k+1). The high-probability form can, because only δ is limited
by k (δ ≥ 1/(k+1)); ε can be as small as needed.

What changes:
1. **The high-probability form (3.2) becomes CO-PRIMARY** with the expectation form (3.1). Its
   guarantee: P(FA in a new unit > ε) ≤ δ, with δ = 0.10 (needs k ≥ 9).
2. **Operational level added.** ε_op = 1 false alarm per hour of driving, expressed per sample:
   - ε_op = 1/3600 for INTERACTION and highD (one sample per agent-second);
   - ε_op = 1/36000 for SHRP2 (10 Hz frames; amendment 2).

   It is applied with the high-probability form only. The expectation form at ε_op is reported
   as infeasible (it would need k ≥ 3,599).
   - A unit with fewer than 1/ε_op negatives cannot show any false alarm below ε_op. Its per-unit
     threshold is then the one with zero false alarms (the first grid point ≥ its highest
     negative score). This is conservative and is reported with the number of such units.
3. **E1c (new, exploratory addendum to E1, same data):** HP at ε_op with δ = 0.10, E1a and E1b
   designs, 5 seeds.
   - Reported: share of held-out recordings with FA > ε_op (must be ≤ δ + 3 SE, binomial SE over
     distinct recordings); mean FA; false alarms per agent-hour; recall at 3 s.
   - Because E1 at α has already been seen, E1c is labelled a post-hoc development analysis.
4. **All remaining tests (E2, E3, B6, B7) report HP at ε_op** next to CRC and HP at α ∈ {0.05, 0.10}.
5. **Section 8 additions.**
   - HP validity rule: share of held-out units with FA > ε ≤ δ + 3 SE, for every ε, in E1 (and
     E1c) and in E3a.
   - Candidate SHRP2 hypotheses are reworded around per-unit control:
     - H5a: share of held-out units above ε, HP < C1;
     - H5b: recall of HP non-inferior to C1, margin fixed from B4–B5;
     - H5c (operational): at ε_op the share of units above ε_op is ≤ δ.

   Final wording is still fixed only in amendment 2, before SHRP2 access.
6. Nothing else changes: units, grid, scores, eligibility, comparators, levels, α values, seeds.

Status: a development specification. This document must be committed to Git **before** any of the
tests in Section 6 are run. Rules and thresholds fixed here are not changed after results are
seen; any change goes to the deviations log with its date. The confirmatory version for SHRP2
goes into OSF amendment 2 (step B8).

Author decisions (10 Oct 2026):
- D1: level R (new recording / new driver) is the guarantee level, level L is the reach, level X
  is a stress test only; for SHRP2 the R-level unit is the driver if an ID exists, else the trip.
- D2: α = 0.05 (primary) and 0.10 at R-level; α = 0.10 at L-level.
- D3: symmetric leave-one-location-out scheme (12 models), not leave-two-out.

---

## 1. Problem

The registered split-conformal false-alarm threshold is fitted on pooled source validation
negatives. Phase 1 (leave-one-location-out, Steps 14–15) showed two failures:

- At 4 of 12 unseen locations the realised false-alarm rate exceeded the 5% target (up to 8.0%).
- At low-risk locations the threshold was over-conservative, with zero recall.

The rehearsal confirmed this (mean held-out false-alarm rate 7.4% at α = 0.05). Label-free repair
of calibration failed (P2-1, P2-2).

The cause: pooled split conformal treats every source frame as exchangeable with every target
frame. Under domain shift this is false.

**Aim:** a threshold whose false-alarm rate in a *new, unseen domain* is controlled by
construction, under an assumption that is stated openly and can be checked. It must:

- run in federated form;
- report the price paid in recall;
- come with a learning curve showing how much delayed data from the new domain is needed before
  local recalibration is better.

## 2. Notation and definitions (fixed)

- **Score.** s(x) = cumulative risk at the 3 s horizon from the hazard model, **raw** (no
  temperature or isotonic step). Raw scores are fixed so that the threshold does not depend on a
  recalibrator fitted on other data. Horizons 1, 2 and 5 s are exploratory.
- **Negative.** A sample not in conflict at t (current = 0), evaluable at 3 s (event observed or
  followed for at least 3 s), with no event within 3 s. False alarms are counted only on
  negatives.
- **Domain unit.** The unit that is assumed exchangeable (Section 3). Its loss is
  L_j(λ) = (number of negatives in unit j with s > λ) / (number of negatives in unit j).
  This loss is non-increasing in λ, lies in [0, 1], and L_j(1) = 0.
- **Grid.** Λ = {σ(t) : t = −14, −13.99, …, +14} (2,801 points, logit-spaced, covering risk
  values from about 8 × 10⁻⁷ to 1 − 8 × 10⁻⁷), plus λ_max = 1. The grid is fixed now and does not
  depend on any data. Taking the minimum over a grid keeps the guarantee valid, because it can
  only raise λ̂.
- **Eligible unit.** At least 100 samples evaluable at 3 s. This is decided on sample counts
  before scores or outcomes are looked at, and units below it are listed and excluded.

## 3. The method

### 3.1 Primary: domain-level conformal risk control (expectation form)

Given k calibration units with loss curves L_1, …, L_k:

  λ̂ = min { λ ∈ Λ : (k / (k+1)) · (1/k) Σ_j L_j(λ) + 1/(k+1) ≤ α }

**Guarantee** (Angelopoulos et al., 2024, Theorem 1, applied with one unit = one exchangeable
loss function, B = 1): if the k calibration units and the new unit are exchangeable, then

  E[ L_new(λ̂) ] ≤ α.

In words: averaged over new units, the false-alarm rate is at most α.

- Each unit counts once, whatever its size. This is the equal-weighting idea of Lee et al. (2023).
- Arbitrary dependence between frames inside a unit is allowed, because the loss is the unit's
  own rate. This removes the autocorrelation problem found in Stage 1 (test 9).
- **Feasibility:** λ̂ exists only if α ≥ 1/(k+1), and it is useful only if α is clearly larger.
  If it does not exist, the result is reported as "infeasible", meaning the method declines to
  set a threshold. It is never replaced by another rule.

### 3.2 Secondary: high-probability form

For each unit, t_j(ε) = min{λ ∈ Λ : L_j(λ) ≤ ε}. Set λ̂_δ = the r-th smallest of t_1, …, t_k,
with r = ⌈(k+1)(1−δ)⌉, which requires δ ≥ 1/(k+1).

Then P( L_new(λ̂_δ) > ε ) ≤ δ under exchangeability. Proof sketch: if t_new ≤ λ̂_δ, then
L_new(λ̂_δ) ≤ L_new(t_new) ≤ ε, and t_new exceeds the r-th smallest of the others with
probability at most (k+1−r)/(k+1) ≤ δ.

This is the domain-level form of the Neyman–Pearson umbrella algorithm (Tong et al., 2018). It
also matches the (α, δ) environment coverage of Duchi et al. (2024).
- Development values: ε = α, δ = 0.10.

### 3.3 Federated form

- Each client j computes L_j on the fixed grid from its own validation negatives and sends G
  numbers, in **one** communication round. No samples, scores or labels leave the client.
- The expectation form needs only the **sum** of the curves, so secure aggregation can hide each
  client's curve completely. This is a privacy property that the pooled and per-site methods do
  not have.
- The high-probability form needs one scalar t_j per client.
- Communication: 2,802 float64 values ≈ 22 kB per client, once.
- Exactness: the federated result equals the centralised result exactly. This is a unit test.

### 3.4 What is and is not new

Already exists (cited, not claimed):
- conformal risk control (Angelopoulos et al., 2024);
- new-group conformal coverage (Dunn et al., 2023; Lee et al., 2023; Duchi et al., 2024);
- clients sharing risk curves on a grid (Shahid, 2026, arXiv:2606.20115, a heuristic for
  *participating* sites only, segmentation false-negative rate);
- federated conformal methods for participating clients (Lu et al., 2023; Plassier et al., 2023;
  Humbert et al., 2023).

Claimed as new, to be stated carefully:
- (i) a finite-sample false-alarm guarantee for an **unseen** client in federated crash-risk
  forecasting, with one-round aggregation that secure aggregation can protect. Duchi et al.
  (2024) name risk control across environments as future work, so the theory step itself is
  close to known; the contribution is its use, its federated form and its evaluation;
- (ii) the evidence on *which domain level* the guarantee can honestly be given at (recording or
  driver versus location or environment), and what each costs in recall;
- (iii) the learning curve for delayed outcomes in a new domain;
- (iv) the stress test across datasets.

## 4. Domain levels [D1: APPROVED]

Exchangeability holds only if the scoring model saw **neither** the calibration units **nor** the
new unit, or treated them symmetrically. This fixes which levels are valid.

| Level | INTERACTION | highD | SHRP2 (amendment 2) | Status of the guarantee |
|---|---|---|---|---|
| **R: new recording / new driver** | validation + test recordings (~22 + ~22), model trained on training recordings | all 60 recordings, scored by the INTERACTION model (never saw highD) | test-split units (drivers if an ID exists, else trips), calibrated on validation-split units | **Exact** under random assignment of units (permutation argument); approximate under stratified assignment (exact within strata) |
| **L: new location / new environment** | 12 locations, via the symmetric leave-one-location-out scheme (Section 4.1) | 6 locations: k = 5, so α ≥ 1/6 — infeasible at 0.10 | lighting × surface × density cells, ≤ 8 (k ≤ 7, α ≥ 0.125) | Valid for the scheme in 4.1, but **conservative** (k = 11) or infeasible |
| **X: across datasets** | INTERACTION units → highD units | — | INTERACTION + highD → SHRP2 | **Not covered**: units are not exchangeable. Reported as a stress test only |

**Decision D1 (approved):** make level R the guarantee level, level L the honest "how far does it
stretch" analysis, and level X a stress test only.
- For SHRP2, the R-level unit is the **driver** if the audit finds a driver ID, otherwise the
  **trip**. This follows the registered rule ("driver if available, else event") at trip level,
  because a single event is too small to be a unit.
- The claim in the paper becomes: "a false-alarm guarantee for a new driver joining the fleet",
  which is the operationally relevant federated case. The reach to a new location or environment
  is shown as an open limit with numbers.

### 4.1 L-level on INTERACTION: two ways [D3: APPROVED]

- **(a) Symmetric leave-one-location-out (APPROVED, D3).** Re-run the 12 Phase 1 LOLO folds and
  save predictions. Unit j's loss is the false-alarm rate of location j under the fold model that
  excluded j (model₋ⱼ). The 12 losses are exchangeable by the symmetry of the procedure (the
  jackknife argument), so domain-level CRC with k = 11 bounds the mean false-alarm rate of the
  held-out location under its own fold model.
  - Cost: 12 fold models (Phase 1 settings, 5 + 5 epochs, seed 131); roughly the "several hours"
    of Phase 1.
  - Caveat, stated in the paper: model₋ⱼ for j ≠ t was trained with location t's training
    recordings. The scheme is therefore exact as an evaluation of the procedure, and approximate
    as a deployment rule for a 13th location (11- versus 12-location models; a stability
    assumption).
- **(b) Leave-two-out (exact for deployment).** One model per pair of locations, 66 models,
  about 25–30 h of CPU. Not used: the gain is a second-order correction, and the
  cost blocks B5–B7.

## 5. Settings fixed now [D2: APPROVED]

- **R-level:** α ∈ {0.05, 0.10}. Primary α = 0.05, the registered value. It is feasible with
  k ≈ 43 (E1a: 1/44 ≈ 0.023). With k ≈ 22 (E1b) it is only barely feasible (1/23 ≈ 0.043), so the
  threshold will be conservative there. That is reported, not tuned.
  - The exact numbers of validation and test recordings come from `split_by_scene` and are
    printed by the B4 script before anything is computed. If k ≤ 19 in E1b, α = 0.05 is reported
    as infeasible for that variant.
- **L-level:** α = 0.10 (0.05 is infeasible at k = 11).
- **High-probability form:** ε = α, δ = 0.10.
- **Score and horizon:** raw cumulative risk at 3 s. Models: the existing central models
  (augmentation on, seeds 131–135) for R-level; LOLO fold models (seed 131) for L-level.

## 6. Development tests (B3–B7)

### B3: unit tests on synthetic domains (known answers), committed with the code

1. Exchangeable heterogeneous domains of unequal sizes, 2,000 Monte Carlo repetitions: mean
   held-out FA ≤ α + 3 Monte Carlo standard errors, for k ∈ {5, 11, 22, 50}.
2. Infeasible exactly when α < 1/(k+1); feasible at α ≥ 1/(k+1).
3. High-probability form: share with FA > ε is ≤ δ + 3 SE.
4. Federated sum of curves = centralised λ̂, bit for bit; invariant to client order.
5. Grid: the loss curves are non-increasing; λ̂ on the grid ≥ λ̂ on the continuous scale.
6. A **known failure** case (the new domain is shifted, not exchangeable): FA > α is reproduced,
   so the stress test can detect violations.

### B4: INTERACTION

- **E1, R-level validity and cost** (no retraining). For each seed, pool the n validation and test
  recordings (all unseen by the central model; n ≈ 44). Two variants:
  - (a) **leave-one-recording-out**: every recording in turn is held out, and the other n − 1 are
    the calibration units (k ≈ 43, so α ≥ 1/44 ≈ 0.023 and α = 0.05 has a real margin);
  - (b) **deployment-sized**: 1,000 random partitions with k equal to the number of validation
    recordings (k ≈ 22, as a federation would have).

  Both are exact under the permutation argument. Report:
  - mean held-out FA against α (validity);
  - the share of held-out recordings with FA > α;
  - recall at 3 s;
  - FA per agent-hour.

  Compare against the comparators in Section 7 on the same partitions.
- **E2, L-level** (12 LOLO folds, predictions saved).
  - (i) The R-level threshold from the source validation recordings, applied to the held-out
    location. This is the cost of shift when the guarantee does not cover it.
  - (ii) The symmetric L-level scheme (4.1a) at α = 0.10.
  - (iii) Comparators.
  - Per location: FA, recall, violation. Summary: the mean FA over the 12 locations (the quantity
    the guarantee targets) and the number of locations violating α.

### B5: highD and across datasets

- Score all 175,593 highD samples with the INTERACTION central models (5 seeds); highD is unseen.
- **E3a, R-level within highD:** 1,000 random partitions of the 60 recordings (eligible ones) into
  calibration and held-out. The guarantee holds for a recording drawn at random from highD.
- **E3b, across datasets:** the threshold set on INTERACTION validation recordings, applied to
  highD recordings. A stress test; violations are expected and reported.
- **E3c, L-level on highD:** k = 5 locations, so α ≥ 1/6. Reported as **infeasible** at
  α = 0.10. That is a finding, not a failure.
- Pooled recall at 3 s with a 95% cluster bootstrap over recordings. Per-location positive-based
  results are **not** reported (A+C decision).
- **Option B trigger (fixed now):** the larger highD rebuild is run only if the 95% interval for
  pooled recall at 3 s in E3a (α = 0.10, seed-averaged) is wider than 0.30. If it is run, it is a
  pre-specified sensitivity analysis.

### B6: learning curve for delayed outcomes (exploratory, protocol fixed now)

- For each held-out location t (LOLO), the recordings of t arrive in order of recording ID.
  Recording ID is a proxy for time; this is stated as a limitation.
- After the first m recordings (m = 1, 2, …, all but one), their samples are labelled. A
  negative at time t is confirmed at t + 3 s; an event is known when it happens.
- The rest of the location is evaluated. Methods along the curve:
  1. the source-only threshold (constant reference);
  2. sample-level split conformal on the burn-in negatives of t;
  3. domain-level CRC with units = consecutive 60 s segments of the burn-in, used only when
     feasible (α ≥ 1/(k+1));
  4. for calibration: temperature refitted on the burn-in (as P2-2, now as a curve), with
     adaptive ECE at 3 s.
- x-axis: minutes of observed driving in t. Read-out per location: the first m at which FA on the
  remainder is ≤ α **and** recall is at least the source-only recall. Also the first m at which
  the refitted ECE is below the source ECE.
- Summary: the median over locations, with the range.

### B7: federated version and comparison

- Run the federated form (client curves → server) and confirm it is identical to the centralised
  result.
- Comparators on identical data (Section 7).
- Communication cost per client.

## 7. Comparators (all on the same units, scores and α)

| Code | Method | Source |
|---|---|---|
| C1 | Pooled split conformal on validation negatives (the registered H2 method) | Phase 1 / pre-registration |
| C2 | Federated conformal, client mixture quantile | Lu et al. (2023) |
| C3 | One-shot quantile of quantiles | Humbert et al. (2023) |
| C4 | Risk-curve shrinkage, participating sites | Shahid (2026) |
| C5 | Label-shift weighted federated conformal | Plassier et al. (2023) |
| C6 | Oracle: per-unit threshold from the target's own negatives (uses target labels; reference only) | — |

The temperature-scaled fixed operating threshold from the original B7 plan is the same threshold
as C1 under raw scores. It is dropped as a separate comparator, and this is noted here.

## 8. Decision and reporting rules (fixed now)

- **Implementation accepted** if all B3 tests pass. **Method carried to amendment 2** if, in B4
  E1, the mean held-out FA is ≤ α + 3 SE for both α values and all 5 seeds. If not, the
  implementation or the assumption is wrong; stop and investigate.
- Every result is reported with recall. A guarantee that never warns is not a result (the
  Phase 1 reporting rule applies).
- Violations at level X, and at L where the scheme does not cover them, are reported as findings,
  never "fixed" by tuning on the same data.
- Candidate confirmatory hypotheses for SHRP2 (worded only after B4–B6, filed in amendment 2
  **before** SHRP2 access):
  - H5a: at R-level, the share of held-out units whose FA exceeds α is lower for domain-level CRC
    than for pooled split conformal (C1);
  - H5b: the recall at 3 s of domain-level CRC is non-inferior to C1 (margin to be fixed from
    B4);
  - learning-curve read-outs: exploratory.

## 9. Code and outputs (B3)

- `src/phase2/domain_crc.py`: loss curves, CRC (both forms), federated aggregation, comparators
  C1–C6.
- `src/phase2/test_domain_crc.py`: tests 1–6 in Section 6 (B3).
- Runner change: `src/lolo_calibration.py --save-preds` saves fold predictions
  (`data/processed/preds/lolo_<location>_s131.npz`) without changing any result. It is checked
  against the existing `results/lolo_conformal.csv`.
- `src/highd/score_highd.py`: highD inference with the saved central weights.
- Outputs: `results/p2_domain_*.csv`; a dossier entry after each step.

## 10. Limitations stated in advance

1. Exchangeability of units is an assumption. At level R it is guaranteed by random assignment;
   at levels L and X it is not, and those results are shown as stress tests.
2. The guarantee is on the **average** over new units (expectation form), or holds with
   probability 1 − δ (secondary form). It is **not** a guarantee for every unit.
3. With few domains, the guarantee is conservative or infeasible. This is a structural limit and
   it is reported.
4. Recording ID as a proxy for time (B6).
5. INTERACTION outcomes are a surrogate (TTC2D-moving < 1.5 s), not crashes.

## References (verified in batches 8–9)

Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L., & Schuster, T. (2024). Conformal risk control.
*ICLR 2024*.

Duchi, J. C., Gupta, S., Jiang, K., & Sur, P. (2024). Predictive inference in multi-environment
scenarios. arXiv:2403.16336.

Dunn, R., Wasserman, L., & Ramdas, A. (2023). Distribution-free prediction sets for two-layer
hierarchical models. *JASA*, 118(544), 2491–2502.

Humbert, P., Le Bars, B., Bellet, A., & Arlot, S. (2023). One-shot federated conformal
prediction. *ICML 2023*, PMLR 202.

Lee, Y., Barber, R. F., & Willett, R. (2023). Distribution-free inference with hierarchical data.
arXiv:2306.06342.

Lu, C., Yu, Y., Karimireddy, S. P., Jordan, M., & Raskar, R. (2023). Federated conformal
predictors for distributed uncertainty quantification. *ICML 2023*, PMLR 202, 22942–22964.

Plassier, V., Makni, M., Rubashevskii, A., Moulines, E., & Panov, M. (2023). Conformal prediction
for federated uncertainty quantification under label shift. *ICML 2023*, PMLR 202, 27907–27947.

Shahid, N. F. (2026). When average calibration fails: Site-conditional federated conformal risk
control. arXiv:2606.20115 (v3).

Tong, X., Feng, Y., & Li, J. J. (2018). Neyman–Pearson classification algorithms and NP receiver
operating characteristics. *Science Advances*, 4(2), eaao1659.
