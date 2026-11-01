# Federated crash risk prediction with per-trip conformal false-alarm control

Code, tests, protocols and aggregate results for the study

> Singh, V.B.P. *Federated crash risk prediction with per-trip conformal false-alarm control: a pre-registered SHRP2 study of calibration under domain shift.* Manuscript submitted to *Transportation Research Part C: Emerging Technologies*.

The study forecasts the probability of a crash or near-crash within 1, 2, 3 and 5 s with a discrete-time hazard model. It tests calibration when driving conditions change, trains the model federated across driving conditions, and controls false alarms per trip with a group-level conformal threshold. Five hypotheses were registered on OSF before any SHRP2 data were accessed and were tested once on held-out trips.

## Pre-registration

| Record | DOI |
| --- | --- |
| Registration | https://doi.org/10.17605/OSF.IO/Y5ZRP |
| Amendment 1 | https://doi.org/10.17605/OSF.IO/CFDKJ |
| Amendment 2 | https://doi.org/10.17605/OSF.IO/ZD8JX |
| Amendment 3 | https://doi.org/10.17605/OSF.IO/HBFM9 |
| Amendment 4 | https://doi.org/10.17605/OSF.IO/27G8T |

Every point the registration left open, and how it was settled, is recorded with its date in `docs/deviations_log.md`. The git history of the original repository shows that each analysis script was committed before its first run on the confirmatory data.

## What is in this repository

| Folder | Contents |
| --- | --- |
| `src/loaders/` | Loaders for INTERACTION, highD and the SHRP2 bird's-eye trajectory reconstruction |
| `src/data/`, `src/models/` | Samples, labels, interaction encoder and hazard head (development stage) |
| `src/shrp2/` | SHRP2 pipeline: audit, split, samples, surrogate measures, features, training (central, domain, size-matched, federated, LightGBM), the confirmatory analysis (`confirm_p1.py`, `confirm_p2.py`), the exploratory analyses (`explore_*.py`) and the extra descriptive analyses (`report_p2_extra.py`) |
| `src/phase2/` | The per-trip false-alarm threshold (`domain_crc.py`), the H5 analysis (`run_h5.py`), published federated conformal comparators (`comparators.py`) and the development experiments on INTERACTION and highD |
| `src/analysis/` | Cluster bootstrap, paired one-sided tests, Holm correction, AUPRC, adaptive ECE |
| `src/rehearsal/`, `src/sim/`, other `src/*.py` | Development-stage rehearsal on public data and the SUMO work of the original plan |
| `test_*.py` | Tests on synthetic data with known answers; no licensed data are needed to run them |
| `docs/` | Deviations log, OSF registration amendments (PDF), SHRP2 freeze note, exploratory-analysis plan, method specification and runbook |
| `logs/` | Run logs of the development experiments on public data |
| `results/` | Aggregate results only (counts, estimates, intervals); no trip, event or driver identifier |
| `GSSM_VERSION.txt` | The commit of the GSSM code used unchanged as a baseline |
| `MANIFEST.sha256` | SHA-256 of every file in this release |

## What is not in this repository, and why

The SHRP2 trajectory reconstruction is licensed by the Virginia Tech Transportation Institute (VTTI) and cannot be redistributed. This repository therefore holds **no SHRP2 data, no files derived from them at trip or frame level, no trip or event identifiers, and no model weights trained on them**. The split is reproduced exactly from the licensed data by `src/shrp2/make_split.py` (registered rule, seed 131); the resulting `split.csv` has SHA-256 `02a37afe90c2ace9ac496c35e641d9655dff33ec525554878133550beb20e081`, so a licensee can check that the same split was obtained. See `DATA_ACCESS.md` for how to obtain each dataset.

The GSSM baseline code and model weights belong to their authors and are not copied here; clone them from their sources (see `DATA_ACCESS.md` and `GSSM_VERSION.txt`).

## Environments

Two Python environments are used, as in the study:

1. **Project environment** (Python 3.10): `requirements.txt` lists the exact package versions used.
2. **GSSM environment**: the GSSM authors' own environment, used unchanged to run their code (protocol P1). Create it from the GSSM repository's instructions.

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Running the tests (no licensed data needed)

```
python src/analysis/test_stats.py
python src/phase2/test_domain_crc.py
python src/phase2/test_run_h5.py
python src/phase2/test_comparators.py
python src/shrp2/test_shrp2_kit.py
python src/shrp2/test_shrp2_pipeline.py
python src/shrp2/test_train_shrp2.py
python src/shrp2/test_train_federated_shrp2.py
python src/shrp2/test_confirm_p2.py
python src/shrp2/test_explore_p2.py
python src/shrp2/test_report_p2_extra.py
<GSSM python> src/shrp2/test_p1_gssm.py   --gssm baselines/GSSM
<GSSM python> src/shrp2/test_gssm_sc.py    --gssm baselines/GSSM
<GSSM python> src/shrp2/test_confirm_p1.py --gssm baselines/GSSM
<GSSM python> src/shrp2/test_explore_p1.py --gssm baselines/GSSM
```

Each test builds a synthetic dataset in the SHRP2 format, runs the code and compares every result with an independent computation. Each prints `n/n PASS` at the end.

## Reproducing the study with the licensed data

The full order of work (audit, freeze, split, samples, training, confirmatory analysis once, sensitivity analyses, exploratory analyses) is given step by step in `docs/SHRP2_runbook.md`. The registered order must be kept: every choice that depends on the data is frozen after the audit and before any model is trained, and the test set is used only by the registered analyses.

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`). The datasets and the GSSM code keep their own licences.

## Citation

Please cite the article (details will be added on publication) and this repository (see `CITATION.cff`).
