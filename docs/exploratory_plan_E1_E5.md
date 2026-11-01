# Exploratory analyses E1–E5: plan fixed before any script is written or run

Veer Bhadra Pratap Singh, Galgotias University. 30 October 2026.

## Status and rules

- These analyses are **exploratory**. They come after the confirmatory analysis (Holm and H5, committed 0e863fb, 26 Oct 2026) and never change a confirmatory verdict.
- E3 and E5 were listed as exploratory in the registration (osf.io/y5zrp; Amendment 2). E1, E2 and E4 are **post-hoc**: defined on 30 Oct 2026, after the confirmatory results were known, to examine weak points found in them. They are labelled post-hoc wherever they are reported.
- Every definition below is fixed before the first line of code. The scripts are tested on synthetic data, committed, and only then run once on the registered run's data (not the sensitivity-8 repetition). Results are reported in full, whatever they show.
- No model is retrained. No threshold grid, protocol rule or test from the registration is changed. No p-values are used for decisions; intervals are descriptive.
- All analyses use the test set of the registered split (02a37afe…e081), five seeds (131–135), and the paired cluster bootstrap of the registration (B = 2,000, seed 131) where an interval is given.

## E1 — Lead time under protocol P1 (post-hoc)

Question: is the shorter lead time of the proposed model (mTTI 1.38–1.80 s against GSSM h-C 2.11 s) a fixed property, or does it depend on the threshold and the horizon?

- **E1a, matched recall.** For each model seed, take the threshold on the registered grid (logistic of −10, −9.9, …, +10) whose P1 recall is closest to GSSM h-C's recall at h-C's best-F1 threshold (ties: the lower threshold). Report, per seed and as the seed mean: recall, precision, mTTI with GSSM's 99% sign-test interval. GSSM h-C is reported at its best-F1 threshold, unchanged.
- **E1b, 5 s horizon score.** Score the P1 frames of the test events with the cumulative risk at 5 s (F₄) of the same five central models (chosen head, augmentation on), by inference only. Run the P1 engine unchanged with the same grid. Report AUPRC, recorded events, and mTTI at the best-F1 threshold, per seed and as the seed mean, beside the registered 3 s results.
- Reading rule, fixed now: lead times are compared descriptively. A seed-mean mTTI within GSSM h-C's 99% interval is described as "comparable"; otherwise the difference is reported as it stands.

## E2 — Shared events under P1 (post-hoc)

Question: does the H1a advantage hold when both methods are scored on the same events, each with its own conflicting object?

- For each model seed, the shared set is the test events with a recorded danger period for both that seed and GSSM h-C. AUPRC is computed for both methods on that set with the P1 engine unchanged.
- Report: number of shared events per seed; AUPRC of each method; the seed-mean difference (model − h-C) with a 95% paired cluster-bootstrap interval over the shared events of each seed (events resampled within each seed's set, differences averaged over seeds). The same for TTC.

## E3 — Base-rate shift and calibration error (registered exploratory)

Question: does the H2 effect grow with the shift in base rate between the source and the held-out level?

- For each testable factor-direction pair at 3 s (from results/shrp2_confirm_h2_pairs.csv), compute the base-rate shift |log(prevalence of the held-out level's test frames / prevalence of the source level's training frames)| and the seed-mean log ECE ratio (registered and size-matched).
- Report the seven pairs in a table and the Spearman correlation between shift and log ratio, with no p-value claim (seven points). Repeat at 1, 2 and 5 s.

## E4 — Checks of the explanations for the two null results (post-hoc)

- **E4a (H3a, FedPer not better calibrated).** For the seven clients, report the base rate at 3 s (positive frames / labelled frames, training split) and the ratio of the highest to the lowest base rate, to compare with the about 40-fold spread of the INTERACTION development clients. Descriptive only.
- **E4b (H4a, no gain under natural dropout).** Within the registered natural-dropout set (test prediction times at which an object present in the 2 s window is missing in at least 20% of its frames), classify each such object by where its missing frames lie: only at the start of the window (entering), only at the end (leaving), or with at least one missing frame between two present frames (interior gap, a true dropout). Also report the share of these objects whose last present position is beyond 25 m (near the 30 m input radius). Then report the H4a difference (augmented − plain AUPRC at 3 s) separately for prediction times whose dropout objects include an interior gap and for the rest, with 95% paired cluster-bootstrap intervals.

## E5 — False alarms as episodes per hour (registered exploratory)

Question: how many separate false warnings would a driver experience?

- An alarm episode is a run of consecutive negative frames above the threshold within one trip; runs separated by fewer than 10 frames (1 s) are merged.
- Thresholds: HP and C1 in the deployment design (thresholds set on the validation trips, as for H5a and H5c), at the 5% level and at ε_op, for each seed.
- Report per test trip the episodes per hour of negative driving, and over trips the mean, median, 90th percentile and maximum, plus the pooled rate (total episodes / total hours), seed-averaged; and the share of trips with at least one episode.

## Reporting

A new Section 5.9 "Exploratory analyses", every heading labelled "registered exploratory" or "post-hoc exploratory". The Discussion may use these results only to explain the confirmatory findings, never to restate them. The deviations log records this plan, its commit, and each run.
