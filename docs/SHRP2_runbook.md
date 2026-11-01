# SHRP2 arrival runbook

Veer Bhadra Pratap Singh, Galgotias University. Written 11 October 2026, before any SHRP2 data were opened. Updated the same day for the split, sample, measure and feature scripts, for the GSSM comparison (H1, Amendment 4), and for the training runner.

This page applies the registered order of work (pre-registration osf.io/y5zrp, Section 8): **(1) audit, (2) freeze the audit-dependent choices and log them, (3) train and select, (4) evaluate on the test set once.** Do not go past step 1 until step 2 is committed.

## Before the data arrive (once)

1. The kit is committed, and `python src\shrp2\test_shrp2_kit.py` prints **27/27 PASS** on the laptop.
2. `python -c "import tables"` runs without an error. If it fails, run `uv pip install tables`, which pandas needs to read the `.h5` files.
3. 7-Zip is installed, for split archives such as `.zip.001`.
4. The data security plan is in place: device encryption on, project outside OneDrive, recovery key backed up (all checked on 11 Oct 2026).

## Arrival day

1. **Download** only into `data\raw\shrp2\` inside the project. `data\` is outside Git. Never use OneDrive, Google Drive, e-mail or any cloud service.
2. **Record the files.** For every downloaded file, record its name, size and `Get-FileHash -Algorithm SHA256`. Add the deviations-log rows "SHRP2 licence approved" and "SHRP2 data downloaded" with the date, the version and these hashes. Commit the log, not the data.
3. **Extract** with 7-Zip into `data\raw\shrp2\extracted\`. For split archives, extract from `.zip.001` only.
4. **Run the audit**, which only counts and loads no model:
   `python src\shrp2\audit_counts.py --root data\raw\shrp2\extracted`
5. **Share the summary only.** `results\shrp2_audit_summary.txt` holds aggregate counts and no event IDs; send only this file to the assistant. `data\processed\shrp2\audit_trips.parquet` holds event IDs and **never leaves the laptop**.
6. **Stop and send the summary** if any of these happens:
   - the summary shows `STOP` or `schema_ok: False`;
   - an identifier column is found (section C);
   - any value is `UNMAPPED` (section I);
   - the counts differ strongly from the published counts (section D).

## Step 2: freeze the audit-dependent choices, before any training

The assistant drafts the freeze note from the summary, and the author approves it. Then commit it and log it as "Audit-dependent choices frozen" before running any model.

| # | Choice | Rule |
|---|---|---|
| A1 | Unit | Driver if an identifier exists, else trip (registered). The documentation shows no driver ID, so the unit is expected to be the trip. |
| A2 | Horizons kept | 80 % rule (registered). If 3 s fails, the longest retained horizon below 3 s becomes primary, logged as a deviation. |
| A3 | Domain factors | Category mappings fixed in `src/shrp2/shrp2_format.py` (`FACTOR_MAPS`) before access. Unknown values are excluded from that factor only (registered). |
| A4 | Baselines without environment data | Registered rule: excluded from the domain-factor analyses (H2, H3) and placed in their own stratum for the split. Confirm with VTTI first. |
| A5 | Radar object position and size | The author decides: keep the radar point or shift it to the object centre; leave the size missing or use a fixed default. Section K of the audit informs this. |
| A6 | Eligibility | ≥ 100 frames evaluable at 3 s (Amendment 2), counted in section H. |

These were already fixed in the code on 11 Oct 2026, before access, and are tested:
- Clock: the 0.1-s grid `time`. The raw `timestamp` is −1 where the dataset authors interpolated, so it is not used as the clock.
- Event time: `impact_timestamp` (ms).
- Prediction times need a contiguous 2-s history and an unmasked subject frame at t. This was added 11 Oct, before access, so that the audit and the sample builder count exactly the same frames.
- Labels: time to event in (0, 3] s is positive; time to event in (5, 7] s is excluded; frames at or after the event are excluded.
- Inclusion: `duration_enough` and `conflict ≠ 'none'`.
- Implausible values: masked as registered.
- Duplicate grid times: the first row is kept.

## Step 3 onwards (scripts written and tested before access; each run once, in this order)

Requirement: A5 (object reference point and size) is frozen and committed. The scripts below **require** it as arguments.

1. Loader: `python src\loaders\shrp2.py --root data\raw\shrp2\extracted`, which writes `data\processed\shrp2\` (trajectories and `trips.parquet`).
2. Split, run once: `python src\shrp2\make_split.py`.
   - Rule: 80/10/10 per stratum (outcome × lighting × surface), seed 131.
   - Record the SHA-256 printed in `results\shrp2_split_summary.txt` in the deviations log, then commit.
3. Samples and labels: `python src\shrp2\build_samples_shrp2.py --object-ref <A5> --object-size <A5>`.
   - Training samples every 1 s; validation and test samples at every frame.
   - The counts in `results\shrp2_samples_summary.txt` must equal the audit's evaluable frames (tested).
4. Classical measures: `python src\shrp2\ssm_shrp2.py --object-ref <A5> --object-size <A5>`, using the same project code as in development.
5. LightGBM features: `python src\shrp2\features_shrp2.py`, using the same function as the rehearsal.
6. P1 inference samples and features (every valid frame of validation and test events, for the H1 comparison under Amendment 4): `python src\shrp2\build_samples_shrp2.py --object-ref <A5> --object-size <A5> --p1`, then `python src\shrp2\features_shrp2.py --p1`.
7. Training: see the next two sections.
8. The test set is touched once per confirmatory analysis.

## Training (H1, H2, H4, H5): src\shrp2\train_shrp2.py

Requirement: Steps 1–6 are done, the clarifications in the deviations log are filed (the row "SHRP2 training: implementation clarifications", written before access), and the public sample files exist (`data\processed\pretrain_samples.npz`, `hazard_labels.npz`, `highd_samples.npz`). The budget is the registered one only (20 pretraining + 10 fine-tuning epochs). There is no reduced option, and budgets are never extended.

1. Cache, once: `python src\shrp2\train_shrp2.py cache`. This writes uncompressed copies in `data\processed\shrp2\cache\`, which needs several GB of disk. The runner refuses to start if any sample file changes after this.
2. Optional timing: `python src\shrp2\train_shrp2.py probe`. It gives projected hours, for planning only.
3. Code check, then delete the check's outputs:
   - `python src\shrp2\train_shrp2.py run --stage A --smoke --only central_hazard_aug_s131`
   - `Remove-Item models\shrp2\*_smoke*, data\processed\shrp2\preds\*_smoke*`
   - `Remove-Item results\shrp2_jobs.csv` (the smoke row is in it).
4. Stage A: `python src\shrp2\train_shrp2.py run --stage A`. This trains 15 models: the hazard head with and without augmentation, and the separate-classifier head with augmentation, for 5 seeds. There are 5 pretrainings. The run is resumable: if it stops, run the same line again.
5. Head choice (registered, validation only): `python src\shrp2\train_shrp2.py select-head`. Commit `results\shrp2_head_selection.txt` and `results\shrp2_head_selection.json` before stage B.
6. Stage B: `python src\shrp2\train_shrp2.py run --stage B`. This trains 40 domain models and 20 size-matched models with the chosen head, plus 5 no-augmentation models if the separate head was chosen. If baselines carry no environment data (expected), every H2 model reuses its seed's central encoder.
7. LightGBM: `python src\shrp2\train_shrp2.py lgbm`.
8. P1 frame scores for H1: `python src\shrp2\train_shrp2.py p1`. This writes `data\processed\shrp2\preds\p1_*.parquet`, which `p1_eval.frame_scores_to_rows` reads.
9. Record in the deviations log:
   - every row of `results\shrp2_divergence.csv` (if the file exists);
   - the head choice;
   - the number of models marked "not trainable" in `results\shrp2_jobs.csv`.

Tests (no SHRP2 data): `python src\shrp2\test_train_shrp2.py` must print 44/44 PASS. It takes about 6 minutes (longer on the laptop).

## H3 federated training: src\shrp2\train_federated_shrp2.py

Requirement: training stage A and `select-head` are done (H3 uses the chosen head), and clarifications C10–C13 are in the deviations log.

1. Clients: `python src\shrp2\train_federated_shrp2.py clients`. This writes `results\shrp2_fed_clients.txt`, with aggregate counts only. Commit it before training.
   - If it says "testable: False", H3 is not testable under the registered rule. Record that in the deviations log and train nothing.
2. Code check, then delete the check's outputs:
   - `python src\shrp2\train_federated_shrp2.py run --smoke --only fedavg_s131`
   - `Remove-Item models\shrp2\*_smoke*, data\processed\shrp2\preds\*_smoke*, results\shrp2_fed_jobs.csv, results\shrp2_fed_rounds.csv`
3. Run: `python src\shrp2\train_federated_shrp2.py run`. This trains 5 regimes × 5 seeds, 30 rounds × 50 steps each, with 5 public-only pretrainings. The run is resumable.
4. Record every row of `results\shrp2_divergence.csv` that starts with `fed_`.

Tests (no SHRP2 data): `python src\shrp2\test_train_federated_shrp2.py` must print 32/32 PASS. It takes about 6 minutes (longer on the laptop).

## H1 comparison with GSSM (protocol P1, Amendment 4)

Requirement: Amendment 4 is filed on OSF before any SHRP2 access. Everything below runs only after Step 2 is committed and after our model is trained and selected. The GSSM code in `baselines\GSSM` is used **unchanged**; run its scripts with its own environment (`baselines\GSSM\.venv\Scripts\python.exe`, below called GPY).

Before access (once, any time):
1. Download the authors' trained models from 4TU (doi 10.4121/9caa1e6c-9abd-4e36-ae28-c9ea4542d940, the PreparedData archive). Extract **only** the folder `PosteriorInference\highD\current` into `data\gssm_models\highD\current\`. It must hold exactly one `.pth` file. Record its SHA-256 in the deviations log.
2. If the weights cannot be obtained or do not load, the fallback is retraining h-C on highD with the authors' scripts, as registered.

After access, S-C first (secondary model; training split only):
1. Build a training-split workspace: `python src\shrp2\gssm_workspace.py --root data\raw\shrp2\extracted --split train --ws data\gssm_ws_train`. It holds training-split trips only. This matters because the authors' segmentation draws random object sizes from the metadata of the events in its workspace.
2. Segment, with the authors' segmentation code for SafeBaseline only (`gssm_segment_sc.py`; it refuses to run in any other workspace):
   `cd data\gssm_ws_train` then `<GPY> ..\..\src\shrp2\gssm_segment_sc.py --gssm ..\..\baselines\GSSM`
3. Train, with the authors' unchanged script, from the same folder: `<GPY> ..\..\baselines\GSSM\src_posterior_inference\pi_train_eval.py --stage 1`. It trains SafeBaseline/current and skips highD and ArgoverseHV, which have no data here. Up to 150 epochs, with early stopping. Rough estimate on the laptop's CPU: 5–10 hours (synthetic rehearsal: about 2 s per epoch per 1,500 scenes; expect about 340,000 scenes). It writes `PreparedData\PosteriorInference\SafeBaseline\current\model_final_<n>epoch.pth`. Then `cd ..\..`.
4. Copy that one `.pth` file into `data\gssm_models\SafeBaseline\current\`, next to h-C, and record its SHA-256 in the deviations log:
   `New-Item -ItemType Directory -Force data\gssm_models\SafeBaseline\current` then `Copy-Item data\gssm_ws_train\PreparedData\PosteriorInference\SafeBaseline\current\*.pth data\gssm_models\SafeBaseline\current\`

After access, for the test split (the same steps on `--split val` are allowed for checks only):
1. Build the workspace (it holds licensed data, so it stays in `data\`):
   `python src\shrp2\gssm_workspace.py --root data\raw\shrp2\extracted --split test --ws data\gssm_ws_test --models-dir data\gssm_models --models highD/current,SafeBaseline/current`
   Keep the printed manifest (event count, event-list SHA-256, unknown environment counts) for the deviations log.
2. Event features (works around the authors' UnboundLocalError when event_data.h5 already exists; same feature lines as theirs):
   `cd data\gssm_ws_test` then `<GPY> ..\..\src\shrp2\gssm_features.py --gssm ..\..\baselines\GSSM`
3. GSSM and the classical measures, with the authors' script, from the workspace:
   `<GPY> ..\..\baselines\GSSM\src_safety_evaluation\evaluate_safety.py`
   This writes `ResultData\EventEvaluation\highD_current.h5` (h-C), `SafeBaseline_current.h5` (S-C) and `TAdv_TTC2D_ACT_EI.h5` (TTC2D, EI and others). Then `cd ..\..`.
4. Our model's and LightGBM's frame scores on the P1 samples come from `train_shrp2.py p1` (training step 8). They are copied to every object row at that frame (`p1_eval.frame_scores_to_rows`).
5. The P1 engine (`src\shrp2\p1_eval.py`, run with GPY) gives first-stage records with the authors' functions, AUPRC with their `get_auprc`, the paired cluster bootstrap over events (B 2,000, seed 131, Δ averaged over the 5 seeds), and mTTI with a 99% sign-test interval. H1 is not testable if fewer than 50 events have a recorded danger period for either method.
6. Sensitivity analyses, after the licence allows: the authors' published voted conflicting objects (third stage) and the method-independent subset.

S-C is descriptive only (not tested), as registered in Amendment 4. Its rehearsal on synthetic data is `test_gssm_sc.py` (below).

Tests (no SHRP2 data): `<GPY> src\shrp2\test_p1_gssm.py --gssm baselines\GSSM` must print 17/17 PASS. The test harness needs pyarrow in GSSM's environment. Install it once, naming GSSM's Python explicitly: `uv pip install --python baselines\GSSM\.venv\Scripts\python.exe pyarrow==19.0.1`. Do not rely on the current folder: with the project environment activated, `uv pip install` goes into the project environment instead. GSSM's own code does not use pyarrow.
S-C rehearsal (no SHRP2 data): `<GPY> src\shrp2\test_gssm_sc.py --gssm baselines\GSSM` must print 9/9 PASS. It takes about 5–10 minutes.

## Confirmatory analysis: the test set, once (confirm_p2.py, confirm_p1.py, run_h5.py)

Requirement: all training is done (stage A, head choice, stage B, LightGBM, P1 scores, H3), GSSM's `evaluate_safety.py` has scored the test workspace, and Amendment 4 (with C1–C23) is filed. Every output below is aggregate and shareable. If the audit dropped the 3 s horizon, the primary horizon is the longest retained one below 3 s (registered contingency). Pass it as `--primary-horizon`, and the retained horizons as `--horizons`.

1. Validation workspace (events only; for choosing the best classical measure under P1, Amendment 4 §2):
   `python src\shrp2\gssm_workspace.py --root data\raw\shrp2\extracted --split val --ws data\gssm_ws_val --no-baselines`
2. P2 part (H2, H3, H4, the H1 key secondary endpoint, sensitivity analyses 1 and 3–7): `python src\shrp2\confirm_p2.py run`. This takes several hours: each test uses 2,000 bootstrap resamples.
3. P1 part (H1a, H1b, sensitivity 2; GSSM's environment): `<GPY> src\shrp2\confirm_p1.py --gssm baselines\GSSM`. Add `--voted <file>` only after the licence allows the authors' voted file.
4. Holm over the nine registered tests: `python src\shrp2\confirm_p2.py holm`. This writes `results\shrp2_confirm_holm.csv` and `results\shrp2_confirm_h1h4_pvalues.txt`.
5. H5 (Amendments 2–3): `python src\shrp2\confirm_p2.py h5-input`, then
   `python src\phase2\run_h5.py --input data\processed\shrp2\h5_input_shrp2.npz --tag shrp2 --h1h4 <the line in results\shrp2_confirm_h1h4_pvalues.txt>`
6. Sensitivity 8 (no handling of implausible values; **full repetition**, C23). Run this only after the registered analysis (steps 1–5) is complete and committed. It takes about as long again as all the training (2–4 days). Everything goes to `*_nomask` folders and files, so the registered run is never touched.
   - `$env:SHRP2_RUN = "nomask"` (every script below then reads and writes the `_nomask` data, models, logs and results; any other value is refused)
   - `python src\loaders\shrp2.py --root data\raw\shrp2\extracted --out data\processed\shrp2_nomask --no-mask`
   - `Copy-Item data\processed\shrp2\split.csv data\processed\shrp2_nomask\` (the registered split stays fixed)
   - Samples, measures and features exactly as in Step 3 onwards items 3–6, with `--traj-dir data\processed\shrp2_nomask` added to each of those commands.
   - Training exactly as in the Training section (cache, stage A, select-head, stage B, lgbm, p1) and the H3 section (clients, run), with the same commands. The variable sends everything to `_nomask`.
   - Analysis exactly as in steps 2–4 above (`confirm_p2.py run`, `confirm_p1.py`, `confirm_p2.py holm`); the outputs end in `_nomask`. The GSSM workspaces and GSSM's scores are reused unchanged.
   - `Remove-Item Env:SHRP2_RUN` (always, so nothing later runs in the repetition by mistake)
   - Record in the deviations log: the repetition's head choice, every row of `results\shrp2_divergence_nomask.csv`, and the date.
7. Commit the `results\shrp2_confirm_*` files and record the run in the deviations log ("Test set first evaluated").

Tests (no SHRP2 data):
- `python src\shrp2\test_confirm_p2.py` must print 20/20 PASS (about 6 minutes).
- `<GPY> src\shrp2\test_confirm_p1.py --gssm baselines\GSSM` must print 12/12 PASS (about 5 minutes).
