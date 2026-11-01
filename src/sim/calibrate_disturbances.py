"""
Step 21: calibrate disturbances so SUMO's conflict rate matches highD at L1 (rules fixed in the
dossier BEFORE any result, 7 Oct 2026).

Fixed: v3 driver parameters (results/calib_<model>_L3_best_v3.json), L1 inputs (volume, trucks),
gaps for cut-ins 10-30 m, hard braking 4 per vehicle-hour at 3-6 m/s^2 for 1-3 s, emergency braking
of every vehicle capped at 6 m/s^2 (highD max 6.6). Tuned: cut-in rate and the gating threshold
(deceleration the new follower / the cut-in car would need, DRAC).
Round 1: grid cut-in rate {10, 20, 30, 45} per veh-h x gating {1.5, 3.0} m/s^2.
Round 2 (only if round 1 does not meet the rules; LAST round): cut-in rate fixed at the model's round-1
best; INTENSITY varied: gating {4.5, 6.0} m/s^2 x minimum cut-in gap {5, 10} m. Changed from the
originally scripted rate refinement after round 1 showed the rate has little effect and the gating a
strong one; recorded in the dossier BEFORE round 2 was run (within the recorded rules: rates and
intensities, at most two rounds, unchanged guard rails). No further rounds (stopping rule).
PRIMARY: TTC<3 s share within +-25% of highD L1. SECONDARY (validation only): TTC<1.5 s within a
factor of 2. GUARD RAILS: the 7 traffic criteria within 2x tolerance; 0 collisions; max braking
<= 6 m/s^2. REALISM (reported): share of frames braking >4 m/s^2 vs highD 0.003%; forced cut-ins per
vehicle-km (must stay well below highD's total lane-change rate).
TraCI runs are strictly SEQUENTIAL (parallel runs contaminate each other).
Usage : python src/sim/calibrate_disturbances.py --model Krauss --round 1
        python src/sim/calibrate_disturbances.py --model Krauss --round 2      (only if needed)
        python src/sim/calibrate_disturbances.py --model Krauss --validate
Output: results/dist_<model>_runs.csv, results/dist_<model>_validation.csv, _best.json (resumable)
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_sumo as C
import disturbances as D

HIGHD_SHARE_DECEL_GT4 = 0.00003      # measured on highD L1 (dossier, 7 Oct 2026)
ELIGIBLE_KM = 0.6 + 0.42             # disturbances act on the last 600 m of the approach + the zone
FIXED = dict(brake_rate=4.0, brake_decel_min=3.0, brake_decel_max=6.0,
             cutin_gap_min=10.0, cutin_gap_max=30.0, emergency_decel=6.0)


def l1_targets():
    t = pd.read_csv("results/highd_targets_by_direction.csv")
    x = t[(t["location"] == "highD_L1") & (t["lanes"] == 3) & (t["state"] == "free-flowing")]
    s = x.median(numeric_only=True)
    tg = {k: float(s[k]) for k in C.CRITERIA if k in s and k != "flow_veh_h"}
    tg.update(flow_veh_h=float((x["flow_veh_h"] / x["lanes"]).median() * 3), truck_share=float(s["truck_share"]),
              ttc_below_3s_share=float(s["ttc_below_3s_share"]), ttc_below_1_5s_share=float(s["ttc_below_1_5s_share"]),
              directions=len(x))
    return tg


def judge(sim, tg):
    e = C.errors(sim, tg)
    traffic = {k: v for k, v in e.items() if C.CRITERIA[k][2]}
    ratio = sim["ttc_below_3s_share"] / tg["ttc_below_3s_share"] if tg["ttc_below_3s_share"] > 0 else np.nan
    return dict(ttc3_ratio=ratio, primary_ok=bool(0.75 <= ratio <= 1.25),
                traffic_ok=all(abs(v) <= 2 for v in traffic.values()),
                worst_traffic=max(traffic, key=lambda k: abs(traffic[k])), worst_traffic_x=max(abs(v) for v in traffic.values()),
                collisions_ok=sim["collisions"] == 0, braking_ok=sim["max_decel_ms2"] <= 6.0 + 1e-6,
                cutins_per_veh_km=sim["n_cutin"] / max(sim["vehicles_in_zone"], 1) / ELIGIBLE_KM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Krauss", choices=["Krauss", "IDM"])
    ap.add_argument("--round", type=int, choices=[1, 2])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="test only: 2 settings, short runs")
    a = ap.parse_args()
    p = json.load(open(f"results/calib_{a.model}_L3_best_v3.json"))["params"]
    tg = l1_targets()
    print(f"model {a.model} | v3 driver parameters frozen | L1: {tg['directions']} recording-directions, "
          f"{tg['flow_veh_h']:.0f} veh/h, trucks {tg['truck_share']:.1%}")
    print(f"targets: TTC<3 s share {tg['ttc_below_3s_share']:.6f} (+-25% -> {0.75*tg['ttc_below_3s_share']:.6f}-"
          f"{1.25*tg['ttc_below_3s_share']:.6f}) | TTC<1.5 s share {tg['ttc_below_1_5s_share']:.7f}")
    out = Path("results") / f"dist_{a.model}_runs.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    work = Path("sim/calib") / f"dist_{a.model}"

    if a.round:
        if a.round == 1:
            grid = [(r, g, FIXED["cutin_gap_min"]) for r in (10, 20, 30, 45) for g in (1.5, 3.0)]
        else:
            r1 = done[done["round"] == 1].sort_values("score").iloc[0]
            grid = [(float(r1["cutin_rate"]), g, gm) for g in (4.5, 6.0) for gm in (5.0, 10.0)]
            print(f"round 2 (last): cut-in rate fixed at round-1 best {r1['cutin_rate']}; gating 4.5/6.0 m/s2 x minimum gap 5/10 m")
        if a.quick:
            grid, a.minutes, a.seeds = grid[:2], 2, 1
        for rate, gate, gmin in grid:
            key = f"r{a.round}_{rate}_{gate}_{gmin}"
            if not done.empty and key in set(done["key"]):
                continue
            t0 = time.time()
            dp = dict(FIXED, cutin_rate=rate, cutin_drac_max=gate, cutin_gap_min=gmin)
            sim = D.simulate_disturbed(p, dp, a.model, 3, tg, a.minutes, list(range(1, a.seeds + 1)), work / key)
            j = judge(sim, tg)
            ok_all = j["traffic_ok"] and j["collisions_ok"] and j["braking_ok"]
            score = abs(np.log(j["ttc3_ratio"])) if j["ttc3_ratio"] > 0 else 9.0
            score += 0 if ok_all else 10                     # guard-rail violations rank last
            row = {"key": key, "round": a.round, "cutin_rate": rate, "cutin_drac_max": gate, "cutin_gap_min": gmin, "score": score, **j,
                   **{f"sim_{k}": v for k, v in sim.items() if isinstance(v, (int, float))}}
            # rewrite the whole file so rows with new columns can never misalign with the header
            prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
            pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(out, index=False)
            print(f"  cut-in {rate:5.1f}/veh-h gating {gate} min gap {gmin:.0f} m | TTC<3 x{j['ttc3_ratio']:.2f} {'OK' if j['primary_ok'] else '--'} | "
                  f"traffic worst {j['worst_traffic']} {j['worst_traffic_x']:.2f}x {'OK' if j['traffic_ok'] else 'FAIL'} | "
                  f"collisions {sim['collisions']} | max braking {sim['max_decel_ms2']:.1f} | "
                  f"cut-ins/veh-km {j['cutins_per_veh_km']:.3f} | {time.time()-t0:.0f}s", flush=True)
        r = pd.read_csv(out)
        best = r.sort_values("score").iloc[0]
        meets = bool(best["primary_ok"] and best["score"] < 10)
        print(f"\nbest so far: {best['key']} | TTC<3 ratio {best['ttc3_ratio']:.2f} | meets primary + guard rails: {meets}")
        if not meets and a.round == 1:
            print("-> run round 2 (the last round under the stopping rule).")
        elif not meets:
            print("-> stopping rule: no further rounds; validate the best setting and report it as the best achieved.")
        else:
            print("-> validate this setting.")

    if a.validate:
        r = pd.read_csv(out); best = r.sort_values("score").iloc[0]
        gmin = float(best["cutin_gap_min"]) if "cutin_gap_min" in best and pd.notna(best["cutin_gap_min"]) else FIXED["cutin_gap_min"]
        dp = dict(FIXED, cutin_rate=float(best["cutin_rate"]), cutin_drac_max=float(best["cutin_drac_max"]), cutin_gap_min=gmin)
        mins, seeds = (2, [101]) if a.quick else (60, [101, 102, 103])
        print(f"validating {best['key']} with {len(seeds)} fresh seeds x {mins} simulated minutes (sequential) ...")
        sim = D.simulate_disturbed(p, dp, a.model, 3, tg, mins, seeds, work / "validate")
        j = judge(sim, tg)
        r15 = sim["ttc_below_1_5s_share"] / tg["ttc_below_1_5s_share"] if tg["ttc_below_1_5s_share"] > 0 else np.nan
        rows = [("PRIMARY: TTC<3 s share", tg["ttc_below_3s_share"], sim["ttc_below_3s_share"], "ratio 0.75-1.25", j["primary_ok"]),
                ("SECONDARY: TTC<1.5 s share", tg["ttc_below_1_5s_share"], sim["ttc_below_1_5s_share"], "ratio 0.5-2", bool(0.5 <= r15 <= 2)),
                ("GUARD: traffic criteria (worst, x tolerance)", 2.0, j["worst_traffic_x"], f"<= 2 ({j['worst_traffic']})", j["traffic_ok"]),
                ("GUARD: collisions", 0, sim["collisions"], "0", j["collisions_ok"]),
                ("GUARD: max braking m/s2", 6.0, sim["max_decel_ms2"], "<= 6.0", j["braking_ok"]),
                ("REALISM: frames braking >4 m/s2", HIGHD_SHARE_DECEL_GT4, sim["share_decel_gt4"], "report", ""),
                ("REALISM: forced cut-ins per veh-km", 0.32, j["cutins_per_veh_km"], "well below 0.32 (highD all lane changes)", "")]
        rep = pd.DataFrame(rows, columns=["check", "highD_or_limit", "sumo", "rule", "passed"])
        pd.set_option("display.width", 200); print(rep.to_string(index=False))
        accepted = bool(j["primary_ok"] and j["traffic_ok"] and j["collisions_ok"] and j["braking_ok"])
        print(f"\nSTEP 21 VERDICT ({a.model}): {'ACCEPTED' if accepted else 'NOT MET -> report as best achieved, labelled'} | secondary check: {'met' if 0.5 <= r15 <= 2 else 'not met'}")
        rep.to_csv(Path("results") / f"dist_{a.model}_validation.csv", index=False)
        json.dump({"model": a.model, "driver_params_v3": p, "disturbance": dp, "accepted": accepted,
                   "ttc3_ratio": j["ttc3_ratio"], "ttc15_ratio": r15, "validation": rows},
                  open(Path("results") / f"dist_{a.model}_best.json", "w"), indent=2, default=str)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()