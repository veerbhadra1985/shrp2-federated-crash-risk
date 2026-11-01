"""
SHRP2 AUDIT, SUPPLEMENT (20 Oct 2026; written after the first audit summary and BEFORE any sample, split or model).
Aggregate counts only (no event IDs), to settle three points the first audit raised:
  (1) the baseline count: 30,741 baseline trips are in the h5 files, the readme reports 10,919 reconstructed,
      and the metadata flags count 19,030 (= 10,919 + 8,111). Cross-tabulates, per category, the metadata flags
      ego_reconstructed / surrounding_reconstructed (raw values shown) against presence in the Ego and
      Surrounding h5 files;
  (2) the outcome type of included test-set events: severity rule used so far (max of the coded severities)
      versus the event-category rule of the GSSM code (utils_tabfig.category, which gives the published
      282 crashes / 4,593 near-crashes);
  (3) trafficDensity levels of included events after the parsing fix (strings start with 'Level-of-service X').
Usage : python src/shrp2/audit_extra.py --root data/raw/shrp2/extracted
Output: results/shrp2_audit_extra.txt (shareable)
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shrp2_format as F
from audit_counts import find_dirs, category_files, read_h5, binary_factor

GSSM_CATEGORY = {k: v for k, v in F.OUTCOME_BY_CATEGORY.items() if k != F.BASELINE_CATEGORY}   # C25 (GSSM utils_tabfig)


def main(root, out="results"):
    rt, st = find_dirs(root)
    meta = pd.read_csv(rt / "metadata_birdseye.csv")
    R = {}
    R["flag_raw_values"] = {c: {str(k): int(v) for k, v in meta[c].value_counts(dropna=False).items()}
                            for c in ["ego_reconstructed", "surrounding_reconstructed"] if c in meta.columns}
    fe = meta["ego_reconstructed"].fillna(0).astype(float) > 0.5
    fs = meta["surrounding_reconstructed"].fillna(0).astype(float) > 0.5
    ego_ids, sur_ids = set(), set()
    for cat, (ef, sf) in category_files(rt).items():
        for p in ef:
            ego_ids |= set(read_h5(p)["event_id"].unique())
        for p in sf:
            sur_ids |= set(read_h5(p)["event_id"].unique())
    in_e = meta["event_id"].isin(ego_ids); in_s = meta["event_id"].isin(sur_ids)
    rows = {}
    for cat, g in meta.groupby("event_category"):
        i = g.index
        rows[cat] = dict(listed=len(g), flag_ego=int(fe[i].sum()), flag_sur=int(fs[i].sum()),
                         flag_both=int((fe[i] & fs[i]).sum()), in_ego_h5=int(in_e[i].sum()), in_sur_h5=int(in_s[i].sum()),
                         in_ego_h5_flag_ego=int((in_e[i] & fe[i]).sum()), in_ego_h5_not_flag_ego=int((in_e[i] & ~fe[i]).sum()),
                         flag_ego_not_in_h5=int((fe[i] & ~in_e[i]).sum()),
                         in_both_h5=int((in_e[i] & in_s[i]).sum()), in_both_h5_flag_both=int((in_e[i] & in_s[i] & fe[i] & fs[i]).sum()))
    R["by_category"] = rows
    b = rows.get(F.BASELINE_CATEGORY, {})
    R["baseline_summary"] = b

    # (2) outcome type of included events
    EM = pd.concat([pd.read_csv(p) for p in sorted(st.rglob("event_meta.csv"))], ignore_index=True)
    EM["sev"] = EM[["severity_first", "severity_second"]].max(axis=1)
    inc = EM[EM["duration_enough"].astype(str).str.lower().isin(["true", "1"]) & (EM["conflict"].astype(str) != "none")].copy()
    inc["by_severity"] = inc["sev"].map({3: "crash", 2: "near-crash"}).fillna("other")
    inc["by_category"] = inc["event_category"].map(GSSM_CATEGORY).fillna("UNMAPPED")
    R["included_events"] = int(len(inc))
    R["included_by_category_rule"] = inc["by_category"].value_counts().to_dict()
    R["included_by_severity_rule"] = inc["by_severity"].value_counts().to_dict()
    R["crosstab_category_rule_x_severity_rule"] = {f"{a}|{s}": int(n) for (a, s), n in
                                                   inc.groupby(["by_category", "by_severity"]).size().items()}
    R["included_by_event_category"] = inc["event_category"].value_counts().to_dict()
    R["included_missing_impact_by_category_rule"] = inc[inc["impact_timestamp"].isna()]["by_category"].value_counts().to_dict()

    # (3) trafficDensity after the parsing fix
    EN = pd.concat([pd.read_csv(p) for p in sorted(st.rglob("environment.csv"))], ignore_index=True).drop_duplicates("event_id")
    env = EN.set_index("event_id").reindex(inc["event_id"])
    lv = env["trafficDensity"].map(lambda v: binary_factor("trafficDensity", v))
    R["trafficDensity_levels_included"] = {str(k): int(v) for k, v in lv.value_counts(dropna=False).items()}
    lt = env["lighting"].map(lambda v: binary_factor("lighting", v))
    R["traffic_x_lighting_included"] = {f"T{a}|L{c}": int(n) for (a, c), n in
                                        pd.crosstab(lv.fillna("U"), lt.fillna("U")).stack().items()}

    Path(out).mkdir(exist_ok=True)
    txt = ["SHRP2 AUDIT SUPPLEMENT - AGGREGATE COUNTS ONLY (no event IDs)", "=" * 62]
    for k, v in R.items():
        txt.append(f"[{k}]")
        if isinstance(v, dict):
            for kk, vv in v.items():
                txt.append(f"  {kk}: {vv}")
        else:
            txt.append(f"  {v}")
    Path(out, "shrp2_audit_extra.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    print("\n".join(txt)); print(f"\nSaved {out}/shrp2_audit_extra.txt (shareable)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out-results", default="results")
    a = ap.parse_args(); main(a.root, a.out_results)
