"""
Tests of the SHRP2 readiness kit on SYNTHETIC SHRP2-format data with known answers (no SHRP2 data).
The ground truth comes from make_synthetic.py's own per-frame loops (a separate implementation).
Usage : python src/shrp2/test_shrp2_kit.py        (writes only to a temporary folder; ~10-30 s)
"""
import json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "loaders")); sys.path.insert(0, str(HERE.parent))
import shrp2_format as F
import make_synthetic as M
import audit_counts as A
import shrp2 as L
import audit_chunks as AC

RES = []


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


def main():
    tmp = Path(tempfile.mkdtemp(prefix="shrp2kit_"))
    try:
        root = M.build(tmp / "complete", "complete")
        tr = json.loads((root / "truth.json").read_text())
        S, T = A.audit(root)
        D, H, G, R, I = S["D_counts"], S["H_eligibility"], S["G_implausible"], S["F_retention"], S["I_environment"]

        # T1 counts
        check("T1 listed / reconstructed trips", D["listed_trips"] == tr["listed_trips"] and
              D["reconstructed_trips_in_h5"] == tr["reconstructed_trips"], f"{D['listed_trips']}/{D['reconstructed_trips_in_h5']}")
        check("T1 included events, crash, near-crash, baselines",
              (D["test_set_included"], D["included_crash"], D["included_near_crash"], D["baseline_trips"]) ==
              (tr["included_events"], tr["included_crash"], tr["included_near_crash"], tr["baseline_trips"]),
              f"{D['test_set_included']}, {D['included_crash']}, {D['included_near_crash']}, {D['baseline_trips']}")
        # T2 retention
        ok = all(abs(R[f"{h}s"]["share"] - tr["retention"][str(h)]["share"]) < 1e-4 and R[f"{h}s"]["kept"] == tr["retention"][str(h)]["kept"]
                 for h in F.HORIZONS_S)
        check("T2 horizon retention shares and decisions", ok, str({h: R[f'{h}s'] for h in F.HORIZONS_S}))
        # T3 implausible values
        check("T3 implausible ego/object counts", (G["ego_frames_speed_gt60"], G["ego_frames_acc_gt15_outside_crash_window"],
              G["object_frames_speed_gt60"], G["object_jumps_gt10m"]) == (tr["ego_frames_speed_gt60"],
              tr["ego_frames_acc_gt15_outside_window"], tr["object_frames_speed_gt60"], tr["object_jumps_gt10"]),
              f"{G['ego_frames_speed_gt60']}, {G['ego_frames_acc_gt15_outside_crash_window']}, {G['object_frames_speed_gt60']}, {G['object_jumps_gt10m']}")
        check("T3 excluded trips (>10 % implausible)", sorted(T[T.excluded_implausible].event_id.tolist()) == sorted(tr["excluded_trips"]))
        # T4 evaluable frames per unit, exactly
        U = T[T.in_study & ~T.excluded_implausible].set_index("event_id")
        mism = [k for k, u in tr["units"].items() if int(k) not in U.index or
                (U.at[int(k), "n_eval3"], U.at[int(k), "n_pos3"], U.at[int(k), "n_neg3"]) != (u["eval"], u["pos"], u["neg"])]
        check("T4 evaluable / positive / negative frames per unit (exact)", not mism and len(U) == tr["units_total"],
              f"{len(U)} units, mismatches {mism[:3]}")
        check("T4 eligible units and frame totals", (H["units_eligible"], H["eval_frames_total"], H["pos_frames_total"], H["neg_frames_total"]) ==
              (tr["units_eligible"], tr["eval_frames_total"], tr["pos_frames_total"], tr["neg_frames_total"]),
              f"{H['units_eligible']} eligible, {H['eval_frames_total']} evaluable")
        # T5 environment
        check("T5 factor levels of included events", I["factor_levels_included_events"] ==
              {f: {**v, "UNMAPPED": 0} for f, v in tr["factor_levels_included_events"].items()})
        check("T5 baselines without environment detected", I["baselines_with_environment"] == 0)
        # T6 reference point
        K = S["K_reference_point"]
        check("T6 object reference offset recovered", abs(K["along_heading_m"]["median"] - tr["ref_offset"]) < 1e-6, str(K["along_heading_m"]))
        # T7 no driver id; then a driver id is detected
        check("T7 no identifier in documented layout -> unit = trip", not S["C_identifiers"]["identifier_columns_found"])
        S2, _ = A.audit(M.build(tmp / "drv", "driver_id"))
        check("T7 driver identifier detected when present", "driver_id" in S2["C_identifiers"]["identifier_columns_found"].get("ego", []))
        # T8 missing column -> schema failure reported
        S3, _ = A.audit(M.build(tmp / "miss", "missing_col"))
        check("T8 missing metadata column flagged", (not S3["B_schema"]["schema_ok"]) and "ego_length" in S3["B_schema"]["metadata_missing"])
        # T9 public variant
        S4, T4 = A.audit(M.build(tmp / "pub", "public"))
        check("T9 public version detected; counts unchanged", S4["A_version_and_files"]["version_detected"].startswith("public")
              and S4["H_eligibility"]["eval_frames_total"] == H["eval_frames_total"])
        # T10 label boundaries by hand (integer ms)
        ts = np.arange(0, 20000, 100)                       # 20 s, 200 frames; first prediction time 1.9 s
        te = 15000
        tt = ts[19:]; tte = te - tt
        exp_pos = int(((tte > 0) & (tte <= 3000)).sum()); exp_excl = int(((tte <= 0) | ((tte > 5000) & (tte <= 7000))).sum())
        exp_neg = int(((tte > 3000) & ~((tte > 5000) & (tte <= 7000)) & ((19900 - tt) >= 3000)).sum())
        got = A.evaluable_counts(ts, te)
        check("T10 boundaries: (0,3] positive, (5,7] excluded, t>=te excluded", got[:3] == (exp_pos + exp_neg, exp_pos, exp_neg)
              and exp_pos == 30 and got[3] == 181, f"{got} pos {exp_pos} neg {exp_neg}")
        gap = np.r_[np.arange(0, 3000, 100), np.arange(3500, 10000, 100)]
        check("T10 a gap breaks the 2-s history", A.evaluable_counts(gap, None)[3] == (len(gap) - 30 - 19) + (30 - 19))
        # T11 summary contains no event IDs
        txt = A.write_outputs(S, T, tmp / "res", tmp / "data/processed/shrp2")
        leaked = [str(e) for e in T.event_id.unique() if str(e) in txt]
        check("T11 shareable summary contains no event IDs", not leaked, f"leaked {leaked[:3]}")
        # T12 every documented category string is mapped
        doc = {"lighting": ["Daylight", "Dawn", "Dusk", "Darkness, lighted", "Darkness, not lighted", "Unknown"],
               "weather": ["No Adverse Conditions", "Fog", "Sleeting", "Mist/Light Rain", "Rain and Fog", "Raining",
                           "Snow/Sleet and Fog", "Snowing", "Unknown"],
               "surfaceCondition": ["Dry", "Wet", "Gravel over Asphalt", "Gravel/Dirt Road", "Muddy", "Icy", "Snowy", "Unknown"],
               "trafficDensity": ["A1", "A2", "B", "C", "D", "E", "F", "Unknown"]}
        un = [(f, v) for f, vs in doc.items() for v in vs if A.binary_factor(f, v) == "UNMAPPED"]
        check("T12 all documented category strings mapped", not un and A.binary_factor("lighting", "Darkness, lighted") == 1
              and A.binary_factor("trafficDensity", "B") == 0 and A.binary_factor("trafficDensity", "C") == 1, str(un))

        # T12b the trafficDensity strings as written in the data (audit of 20 Oct 2026: 'Level-of-service X: ...');
        # the first parser read the first character ('L') and set every value to unknown
        real = {"Level-of-service A1: Free flow, no lead traffic": 0,
                "Level-of-service A2: Free flow, leading traffic present": 0,
                "Level-of-service B: Flow with some restrictions": 0,
                "Level-of-service C: Stable flow, maneuverability and speed are more restricted": 1,
                "Level-of-service D: Unstable flow - temporary restrictions substantially slow driver": 1,
                "Level-of-service E: Flow is unstable, vehicles are unable to pass, temporary stoppages, etc.": 1,
                "Level-of-service F: Forced traffic flow condition with low speeds and traffic volumes that are below capacity": 1,
                "Unknown": None}
        wrong = [(v, A.binary_factor("trafficDensity", v)) for v, want in real.items() if A.binary_factor("trafficDensity", v) != want]
        check("T12b trafficDensity strings as written in the data: A-B -> 0, C-F -> 1, Unknown -> unknown; anything else UNMAPPED",
              not wrong and A.binary_factor("trafficDensity", "Level-of-service X: none") == "UNMAPPED", str(wrong))

        # T13 outcome type by event category (C25, frozen 20 Oct 2026)
        EMs = pd.concat([pd.read_csv(p) for p in sorted((root / F.ST_DIR).rglob("event_meta.csv"))])
        pl = EMs[(EMs.event_category == "Crash-NearCrash") & (EMs.severity_first == 2)]
        pl_id = int(pl.event_id.iat[0]) if len(pl) else None
        check("T13 C25: category decides the outcome type; coded severities disagree for the planted event",
              pl_id is not None and T.set_index("event_id").at[pl_id, "outcome_type"] == "crash"
              and D["included_by_severity_info"]["crash"] < D["included_crash"] and D["included_unmapped_category"] == 0
              and "STOP" not in S, f"planted {pl_id}, by severity {D['included_by_severity_info']}")
        cats14 = ["Crash", "Crash-Crash", "Crash-CrashRelevant", "Crash-NearCrash", "Crash-OtherConflict", "CrashRelevant-Crash",
                  "CrashRelevant-NearCrash", "NearCrash", "NearCrash-Crash", "NearCrash-CrashRelevant", "NearCrash-NearCrash",
                  "NearCrash-OtherConflict", "SecondaryCrash", "SecondaryNearCrash"]       # as found by the audit, 20 Oct 2026
        indep = {c: ("crash" if ("Crash" in c.split("-") or c == "SecondaryCrash") else "near-crash") for c in cats14}
        same = all(A.outcome_of(c) == indep[c] for c in cats14) and A.outcome_of(F.BASELINE_CATEGORY) == "baseline" \
            and A.outcome_of("SomethingElse") == "UNMAPPED" and len(F.OUTCOME_BY_CATEGORY) == 15
        gssm_tab = HERE.parents[1] / "baselines" / "GSSM" / "src_visualisation" / "visual_utils" / "utils_tabfig.py"
        note = "GSSM file not found here; independent rule only"
        if gssm_tab.exists():
            import ast
            tree = ast.parse(gssm_tab.read_text(encoding="utf-8"))
            lit = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(getattr(t, "id", "") == "category" for t in n.targets)]
            word = {"Crashes": "crash", "Near-crashes": "near-crash", "Safe interactions": "baseline"}
            same = same and bool(lit) and {k: word[v] for k, v in lit[0].items()} == F.OUTCOME_BY_CATEGORY
            note = "equal to the GSSM code's category table" if same else "DIFFERS from the GSSM code's table"
        check("T13b C25 mapping: all 14 categories, independent rule and GSSM table agree", same, note)
        eg = pd.read_hdf(root / F.RT_DIR / "Crash-NearCrash" / f"{F.EGO_FILE}.h5", key=F.H5_KEY)
        eg = eg[eg.event_id == pl_id].sort_values("time")
        te_pl = int(pl.impact_timestamp.iat[0]); ts_pl = A.clock_ms(eg["time"])
        _, ac_cat = A.implausible_ego(ts_pl, eg.v_ekf.to_numpy(), eg.acc_ekf.to_numpy(), te_pl, True)    # crash by category
        _, ac_sev = A.implausible_ego(ts_pl, eg.v_ekf.to_numpy(), eg.acc_ekf.to_numpy(), te_pl, False)   # near-crash by severity
        check("T13c +-1 s crash window follows the category: planted spike allowed (would be masked under the severity rule)",
              int(ac_sev.sum()) - int(ac_cat.sum()) == 1
              and tr["ego_frames_acc_gt15_outside_window"] == G["ego_frames_acc_gt15_outside_crash_window"],
              f"{int(ac_cat.sum())} vs {int(ac_sev.sum())}")

        # T14 baseline flag check (audit_chunks.py): counts per chunk = truth; artefact predictions; no IDs
        Cc, Pc = AC.check(root, tmp / "data/processed/shrp2" / "audit_trips.parquet", tmp / "res")
        bt = tr["baseline_chunks"]
        okc = all(Cc.loc[i, "in_ego"] == b["in_ego"] and Cc.loc[i, "in_sur"] == b["in_sur"] and Cc.loc[i, "flag_11"] == b["flag_both"]
                  and Cc.loc[i, "ego_only"] == b["ego_only"] for i, b in enumerate(bt))
        txtc = (tmp / "res" / "shrp2_audit_chunks.txt").read_text()
        leakc = [str(e) for e in T.event_id.unique() if str(e) in txtc]
        check("T14 baseline flag check: per-chunk counts exact, artefact predictions hold, no IDs",
              okc and Pc["artefact_explanation_holds"] and not leakc and "quality_by_group" in txtc, str(Pc))
        alt = tmp / "alt"; shutil.copytree(root, alt)
        mm = pd.read_csv(alt / F.RT_DIR / "metadata_birdseye.csv")
        mm.loc[mm.event_category == F.BASELINE_CATEGORY, ["ego_reconstructed", "surrounding_reconstructed"]] = 1
        mm.to_csv(alt / F.RT_DIR / "metadata_birdseye.csv", index=False)
        _, Pn = AC.check(alt, None, tmp / "res_alt")
        check("T14b negative control: correct flags -> artefact explanation rejected", not Pn["artefact_explanation_holds"], str(Pn))

        # ---------------- loader ----------------
        TL = L.main(root, tmp / "data/processed/shrp2", mask=True)
        traj = pd.concat([pd.read_parquet(p) for p in sorted((tmp / "data/processed/shrp2").glob("traj_*.parquet"))])
        excl = set(tr["excluded_trips"]); study = set(int(k) for k in tr["units"])
        check("L1 rows only for study trips; none for the excluded trip", set(traj.event_id.unique()) == study and not (excl & set(traj.event_id)))
        ego = pd.concat([pd.read_hdf(p, key=F.H5_KEY) for p in sorted((root / F.RT_DIR).rglob(f"{F.EGO_FILE}*.h5"))])
        ego["grid_ms"] = np.round(ego["time"] * 1000).astype(np.int64)
        ego = ego.drop_duplicates(["event_id", "grid_ms"], keep="first")
        m = traj[traj.is_ego & ~traj.masked].merge(ego, left_on=["event_id", "timestamp_ms"], right_on=["event_id", "grid_ms"])
        err = max(np.abs(m.vx - m.v_ekf * np.cos(m.psi_ekf)).max(), np.abs(m.vy - m.v_ekf * np.sin(m.psi_ekf)).max(),
                  np.abs(m.x - m.x_ekf).max(), np.abs(m.time_s - m.time).max(), np.abs(m.raw_timestamp - m.timestamp).max())
        check("L1 subject kinematics, grid clock and raw stamp exact", err < 1e-9 and len(m) == (traj.is_ego & ~traj.masked).sum(), f"max err {err:.2e}")
        check("L1 duplicate grid time dropped (1 subject, 1 object planted)", int(TL.duplicate_times_dropped.sum()) == 1
              and S["E_lengths"]["duplicate_grid_times_total"] == 1
              and not traj.duplicated(["event_id", "agent_id", "timestamp_ms"]).any())
        check("E raw -1 stamps counted, not used as clock", 0.05 < S["E_lengths"]["share_frames_without_raw_timestamp"] < 0.15)
        check("L2 masked subject frames = truth (speed + acceleration)", int(TL.ego_frames_masked.sum()) ==
              tr["ego_frames_speed_gt60"] + tr["ego_frames_acc_gt15_outside_window"])
        om = traj[~traj.is_ego & traj.masked]
        check("L3 masked object frames = truth (speed + jumps), kinematics NaN", len(om) == tr["object_frames_speed_gt60"] + tr["object_jumps_gt10"]
              and om[L.KIN].isna().all().all(), f"{len(om)}")
        TN = L.main(root, tmp / "data/processed/shrp2_nomask", mask=False)
        trn = pd.concat([pd.read_parquet(p) for p in sorted((tmp / "data/processed/shrp2_nomask").glob("traj_*.parquet"))])
        check("L4 --no-mask keeps the excluded trip and masks nothing", (excl <= set(trn.event_id)) and not trn.masked.any()
              and trn[trn.is_ego][["x", "y"]].notna().all().all())
        check("L5 trips table: environment bins for events, none for baselines",
              TL[TL.outcome_type == "baseline"]["lighting"].isna().all() and TL[TL.in_study & (TL.outcome_type != "baseline")]["lighting"].notna().all())
        # L6 the common format is consumable by the project's sample builder
        from data import build_samples as BS
        one = traj[traj.event_id == sorted(study)[0]]
        smp = BS.scene_samples(one[["scene_id", "time_s", "agent_id", "x", "y", "vx", "vy", "heading", "length", "width"]],
                               np.random.default_rng(0), 5)
        check("L6 build_samples.scene_samples accepts the output", len(smp) > 0 and smp[0][0].shape == (BS.K + 1, BS.H, 9), f"{len(smp)} samples")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()
