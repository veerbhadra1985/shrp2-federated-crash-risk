"""
One-off repair (7 Oct 2026): results/dist_<model>_runs.csv got round-2 rows with one extra column
(cutin_gap_min) under the round-1 header. Round-1 rows used the fixed minimum gap of 10 m.
This inserts 'cutin_gap_min' into the header and 10.0 into the round-1 rows, at the position where
round-2 rows carry it (right after cutin_drac_max). A backup of the original file is kept.
Usage: python src/sim/repair_dist_runs.py --model Krauss
"""
import argparse, csv, shutil
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
f = Path(f"results/dist_{a.model}_runs.csv")
rows = list(csv.reader(open(f, newline="")))
header, body = rows[0], rows[1:]
if "cutin_gap_min" in header:
    print("header already contains cutin_gap_min: nothing to repair"); raise SystemExit
pos = header.index("cutin_drac_max") + 1
n_old, n_new = len(header), len(header) + 1
fixed, n1, n2 = [], 0, 0
for r in body:
    if len(r) == n_old:                         # round-1 row: insert the fixed 10 m gap
        fixed.append(r[:pos] + ["10.0"] + r[pos:]); n1 += 1
    elif len(r) == n_new:                       # round-2 row: already has the gap
        fixed.append(r); n2 += 1
    else:
        raise SystemExit(f"unexpected row length {len(r)}; file left unchanged")
backup = f.with_name(f.stem + "_backup_before_repair.csv"); shutil.copy(f, backup)
with open(f, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(header[:pos] + ["cutin_gap_min"] + header[pos:]); w.writerows(fixed)
print(f"repaired {f}: {n1} round-1 rows (gap 10 m inserted), {n2} round-2 rows kept; backup: {backup}")