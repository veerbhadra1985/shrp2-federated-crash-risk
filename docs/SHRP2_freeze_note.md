# SHRP2 freeze note: audit-dependent choices

Veer Bhadra Pratap Singh, Galgotias University. 20 October 2026.

This note is step (2) of the registered order of work (pre-registration osf.io/y5zrp, Section 8): (1) audit, (2) freeze the audit-dependent choices and log them, (3) train and select, (4) evaluate on the test set once. It was written after the audit and **before any sample, split or model existed**. The audit counts trips, frames and categories; it computes no risk score, no surrogate safety measure and no model output.

## Record

| Item | Value |
|---|---|
| Data | Complete bird's-eye trajectory reconstruction of the SHRP2 NDS (Jiao & Calvert, 2025; DOI 10.15787/VTT1/EFYEJR). Licence approved 19 Oct 2026; downloaded 20 Oct 2026 (hashes in the deviations log and dossier) |
| Audit code | Commit 66cfac8 (`audit_counts.py`, `audit_extra.py`, `audit_chunks.py`; kit tests 33/33 on the laptop) |
| Final audit outputs | `results/shrp2_audit_summary.txt` / `.json`, `results/shrp2_audit_chunks.txt` (aggregate counts only; committed with this note) |
| Earlier audit outputs | First audit, commit 15fb8ee; after the trafficDensity fix (P6) and supplement, commit 1f4c23f |
| Author approval | A1–A6, C25, C26: 20 Oct 2026, 17:25 IST. C24 (all baselines in the files): 20 Oct 2026, 21:36 IST, after the flag check |

## Frozen choices

| # | Choice | Frozen rule | Audit basis |
|---|---|---|---|
| A1 | Unit | **Trip** (registered rule when no driver identifier exists) | No identifier column in any table (section C) |
| A2 | Horizons | **1, 2, 3 and 5 s all kept; 3 s primary** | Retention 0.9992 / 0.9992 / 0.9982 / 0.9975 against the registered 0.80 rule (section F) |
| A3 | Domain factors | Mappings as fixed before access in `shrp2_format.FACTOR_MAPS`; trafficDensity read as the level-of-service letter (P6): A–B → 0, C–F → 1. Unknown values are excluded from that factor only | No unmapped value. Included events, level 0 / 1 / unknown: lighting 3,963 / 912 / 0; weather 4,392 / 483 / 0; surface 4,128 / 745 / 2; traffic 3,420 / 1,450 / 5 (section I) |
| A4 | Baselines without environment data | Registered rule: own split stratum (baseline × L unknown × S unknown); excluded from the domain-factor analyses H2 and H3 (C3, C4, C11) | No baseline has environment data (0 of 30,741); all 4,875 included events have it |
| A5 | Radar object position and size | **Position as given** (`--object-ref asgiven`); **size 4.5 m × 1.75 m** for every object (`--object-size car`, the GSSM default) | Test-set object positions equal the trajectory files exactly (0.0 m offset over 3,700,583 matched rows, section K); the GSSM code uses these positions with these dimensions; a size is needed for 2D TTC |
| A6 | Eligibility (Amendment 2) | A unit is eligible for the unit-level analyses (H5) if it has **≥ 100 frames evaluable at 3 s**, decided on counts only | 35,573 of 35,610 units eligible (section H, below) |
| C24 | Baseline population | **All safe baseline trips in the trajectory files (30,741)**, i.e. all reconstructed baselines, as the loader's registered rule states. The metadata flag `ego_reconstructed` is not used | See "Baseline count" below |
| C25 | Outcome type of an event | **The SHRP2 event category, mapped as in the GSSM code** (`utils_tabfig.category`): a category with a 'Crash' component, or SecondaryCrash → crash; otherwise near-crash. Used for the split strata, the crash-only sensitivity analysis (C16) and the ±1 s crash window of the implausible-acceleration rule. The coded severities are not used | Reproduces the published 282 crashes / 4,593 near-crashes exactly. The maximum coded severity would give 215 / 4,636 / 24 unclassified |
| C26 | Events without an impact time | **Not units.** Excluded from the split and from all analyses, P1 included | 5 of the 4,875 included events (1 crash, 4 near-crashes). The registered labels need the event time; the GSSM event-organising code (`organise_events.py`) likewise removes the trajectories of events with a missing start, end or impact time from the event data it evaluates. It is not a choice between results: no event time, no label |

## Baseline count (C24)

The readme of the dataset reports 10,919 reconstructed safe baselines (3,893 with at least one object). The files hold 30,741. The difference comes from how the dataset authors' code writes the metadata flags (GSSM repository, `src_trajectory_reconstruction/reconstruct_birdseye.py`):

- a trip whose subject vehicle is reconstructed but that has no usable object is flagged (1, 0) during the loop, in every file;
- trips with objects are flagged (1, 1) only after the loop, from the data still in memory. The baselines are written in five files of about 6,600 trips, and the memory is emptied after each file is saved, so at that point it holds the **last file only**.

The count check (`audit_chunks.py`, commit 66cfac8) confirmed every prediction of this explanation:

| Baseline file | Trips | With objects | Flagged (1, 1) | No objects (all flagged (1, 0)) |
|---|---|---|---|---|
| 0 | 6,597 | 5,292 | 0 | 1,305 |
| 1 | 6,594 | 5,272 | 0 | 1,322 |
| 2 | 6,587 | 5,346 | 0 | 1,241 |
| 3 | 5,676 | 3,912 | 0 | 1,764 |
| 4 (last) | 5,287 | 3,893 | 3,893 | 1,394 |
| Total | 30,741 | 23,715 | 3,893 | 7,026 |

3,893 + 7,026 = 10,919, the readme count. Further checks:

- **Quality.** The 19,822 unflagged baselines with objects equal the 3,893 flagged ones in median length (20.8 s), share of frames with an object (median 0.995), implausible values (none) and eligibility (19,818 of 19,822 against 3,892 of 3,893).
- **Use by the dataset authors.** The GSSM data-preparation code (`represent_utils/utils_data_segmentation.py`) reads all five baseline files without any flag filter, so the authors' own SafeBaseline model was trained on all of them.
- **Plausibility.** Event trips (one file per category, flags complete) are 91 % reconstructed (8,111 of 8,895 listed) and 18 % have no object. Baselines in the files: 95 % of listed (30,741 of 32,509), 23 % with no object. Under the flags they would be 34 % and 64 %.

Keeping only the flagged 10,919 would select baselines by a bookkeeping step, and would raise the share without any surrounding object from 23 % to 64 %. The registered rule ("all reconstructed safe baseline trips enter") is applied to the trips that were reconstructed. The difference from the readme is reported and explained, never adjusted (registration Section 2). Baselines carry no outcome, so no result could have informed this choice. A note to the dataset authors asks them to confirm the explanation.

## Counts after the freeze (final audit, 20 Oct 2026)

| Quantity | Value |
|---|---|
| Listed trips / reconstructed in the files | 41,404 / 38,852 |
| Test-set events (published filters) | 6,664 events; 4,875 included (published 4,875): **282 crashes, 4,593 near-crashes** (published 282 / 4,593) |
| Events without an impact time (C26) | 5 (1 crash, 4 near-crashes) → 4,870 event trips in the study |
| Baseline trips (C24) | 30,741 |
| Trips excluded for implausible values (> 10 % of subject frames) | 1 study trip (a near-crash); the audit lists 2 because it also counts a reconstructed event outside the study |
| **Units (split population)** | **35,610** = 30,741 baselines + 4,869 events (281 crashes, 4,588 near-crashes) |
| Units eligible (≥ 100 frames at 3 s) | **35,573**: 30,736 baselines, 271 crashes, 4,566 near-crashes |
| Frames evaluable at 3 s | 5,814,930: 145,948 positive (2.51 %), 5,668,982 negative |
| Events with at least one positive frame | 4,867 |
| Eligible baseline driving time | 135.6 h of negative frames |
| Implausible values (all reconstructed trips) | subject speed > 60 m/s: 144 frames; \|acc\| > 15 m/s² outside a crash window: 1,424; object speed > 60 m/s: 1,332; object jumps > 10 m: 1,362 (masked, as registered) |
| Time grid | 0 gaps, 0 duplicate times |
| Objects | Share of frames with an object: baselines 0.60, crashes 0.79, near-crashes 0.90; 7,026 baselines with no object; at most 11 at once (237 trips above 8) |

## What follows from the registered rules (expected, not decided here)

These are consequences of rules already registered, written down now so they cannot be mistaken for later choices:

- **Split.** 80/10/10 per stratum (outcome × lighting × surface), seed 131 (`make_split.py`). Baselines form one stratum. Expected test split: about 3,074 baselines and about 487 events, of which about 28 crashes.
- **H2 testability (C15).** A pair needs ≥ 50 test events of the held-out level with a positive frame. Weather level 1 has 483 included events, so its test split will hold about 48. The pair evaluated on weather level 1 may therefore be "not testable" under the registered rule; H2 needs at least 4 of 8 pairs and should be tested. Weather is not a split stratum, so the exact number is known only after the split.
- **Crash-only sensitivity (C16).** About 27 eligible test crashes. This is a registered sensitivity analysis, reported as such.
- **H5.** About 3,560 eligible test units and as many validation units, well above the minima (20, and 50 for H5c). Every unit has fewer than 36,000 negative frames, so ε_op = 1/36,000 per frame is reached at unit level only with zero false alarms in that unit; this is how the registered HP rule works (Amendment 2), not a new choice.
- **Sensitivity 7 (C20).** Restricting to baselines with at least one detected object is already registered and covers the 23 % of baselines without objects.

## Next steps (registered order)

1. Commit this note, the final audit outputs and the deviations log ("Data audit completed", "Audit-dependent choices frozen").
2. Loader: `python src\loaders\shrp2.py --root data\raw\shrp2\extracted`.
3. Split, once: `python src\shrp2\make_split.py`; record the SHA-256 of `split.csv` in the log and commit.
4. Samples, measures and features with `--object-ref asgiven --object-size car`, then training, as in the runbook.
