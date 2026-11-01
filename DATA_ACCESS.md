# Data and third-party code

None of the datasets below is included in this repository. Each must be obtained from its provider under the provider's terms.

| Resource | Use in the study | How to obtain |
| --- | --- | --- |
| SHRP2 bird's-eye trajectory reconstruction (Jiao and Calvert, 2025; DOI 10.15787/VTT1/EFYEJR) | Confirmatory data (all crashes, near-crashes and baselines) | Data-use licence from the Virginia Tech Transportation Institute (VTTI). The data may not be redistributed. |
| INTERACTION dataset (Zhan et al., 2019) | Method development and encoder pretraining | From the dataset providers, under their licence |
| highD dataset (Krajewski et al., 2018) | Method development and encoder pretraining | From the dataset providers, under their licence |
| GSSM code (Jiao et al., 2026) | Baseline, run unchanged in its own environment | https://github.com/Yiru-Jiao/GSSM; the commit used is in `GSSM_VERSION.txt` |
| GSSM model weights, h-C (Jiao et al., 2025b; DOI 10.4121/9caa1e6c-9abd-4e36-ae28-c9ea4542d940) | Baseline (highD, current-state features) | 4TU.ResearchData |

## Where the code expects the data

```
data/raw/shrp2/extracted/        SHRP2 files as delivered by VTTI (never committed)
data/processed/                  everything derived from the data (never committed)
baselines/GSSM/                  a clone of the GSSM repository
models/                          trained models (never committed)
```

The split is not distributed. It is reproduced from the licensed data by `src/shrp2/make_split.py` with seed 131; the expected SHA-256 of `split.csv` is `02a37afe90c2ace9ac496c35e641d9655dff33ec525554878133550beb20e081`.

## Rules followed in the study

Licensed data never left the encrypted analysis computer and were never committed to version control. Only aggregate results (counts, estimates and intervals) are published.
