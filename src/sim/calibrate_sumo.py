"""
Step 20b: calibrate the SUMO motorway scenario against highD (free-flowing traffic).

Validation: traffic volume and truck share are INPUTS taken from each location set; the validation\nrun uses the validation locations' own inputs, so it tests whether driver behaviour transfers.\nTargets: results/highd_targets_by_direction.csv (Step 20a), free-flowing recording-directions
with the chosen number of lanes. highD locations are split in two: CALIBRATION locations
(every other location, sorted) and VALIDATION locations (the rest). Targets are medians.
Fixed from data: flow per lane and truck share (calibration targets).
Tuned (car drivers): desired-speed factor mean and spread, time headway tau and its spread
across 5 driver types (tau_cv), Krauss sigma, minimum gap, lane-change eagerness
(lcSpeedGain, lcKeepRight), truck maximum speed; IDM also acceleration and deceleration.
Search: random sampling (--samples), then refinement around the best (--refine); the --recheck best\ncandidates are re-tested on fresh seeds and the most robust one is selected. Arrivals are random\n(Poisson), which reproduces the bunching of real traffic (equally spaced insertion did not).
Loss: mean squared normalised error; |error| <= 1 means the acceptance criterion is met:
  speed p15/p50/p85 +-5%; median time gap +-10%; share of gaps <1 s +-5 pp;
  lane changes per veh-km +-25%; flow +-5%   (truck mean speed +-5% is in the loss only)
Lane use (car share in left/right lane, truck share in right lane) enters the loss only.
Trucks may not use the leftmost lane on 3+ lane roads (highD: 0.7% of truck frames).
Conflict rates (TTC) are reported, never tuned.
Usage : python src/sim/calibrate_sumo.py --model Krauss --lanes 3 --samples 40 --refine 20
        (quick test: --samples 2 --refine 1 --minutes 2 --seeds 1)
Output: results/calib_<model>_L<lanes>_runs.csv, results/calib_<model>_L<lanes>_best.json
Resumable: runs already in the CSV are skipped.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import highway as HW

BOUNDS = {"car_sf_mean": (0.70, 1.05), "car_sf_dev": (0.05, 0.25), "truck_vmax": (22.0, 27.0),
          "tau": (0.6, 1.8), "tau_cv": (0.0, 0.6), "min_gap": (1.0, 3.0),
          "lc_speed_gain": (0.2, 2.0), "lc_keep_right": (0.2, 5.0),
          "truck_lc_keep_right": (0.5, 10.0), "truck_lc_speed_gain": (0.05, 1.0)}
KRAUSS_ONLY = {"sigma": (0.1, 0.9)}
IDM_ONLY = {"accel": (0.8, 2.5), "decel": (1.5, 4.0)}
# metric: (target column, tolerance, kind, in acceptance?)  kind: rel = relative, abs = absolute
CRITERIA = {"speed_p15_kmh": (0.05, "rel", True), "speed_p50_kmh": (0.05, "rel", True),
            "speed_p85_kmh": (0.05, "rel", True), "time_gap_median_s": (0.10, "rel", True),
            "time_gap_below_1s_share": (0.05, "abs", True), "lane_changes_per_veh_km": (0.25, "rel", True),
            "flow_veh_h": (0.05, "rel", True), "truck_speed_mean_kmh": (0.05, "rel", False),
            # lane use (measured in highD): in the loss only, NOT acceptance criteria
            "car_share_left_lane": (0.05, "abs", False), "car_share_right_lane": (0.05, "abs", False),
            "truck_share_right_lane": (0.05, "abs", False)}
REPORT_ONLY = ["ttc_below_3s_share", "ttc_below_1_5s_share", "speed_mean_kmh", "car_speed_mean_kmh"]


def targets(lanes):
    t = pd.read_csv("results/highd_targets_by_direction.csv")
    t = t[(t["lanes"] == lanes) & (t["state"] == "free-flowing")].copy()
    locs = sorted(t["location"].unique())
    if len(locs) >= 2:
        cal_l, val_l = locs[0::2], locs[1::2]
        cal, val = t[t["location"].isin(cal_l)], t[t["location"].isin(val_l)]
        how = f"calibration locations {cal_l} | validation locations {val_l}"
    else:                                                   # one location: split recordings instead
        recs = sorted(t["recording"].unique())
        cal, val = t[t["recording"].isin(recs[0::2])], t[t["recording"].isin(recs[1::2])]
        how = f"single location {locs}: recordings split alternately"
    def summ(x):
        s = x.median(numeric_only=True)
        return {**{k: float(s[k]) for k in CRITERIA if k in s and k != "flow_veh_h"},
                "flow_veh_h": float((x["flow_veh_h"] / x["lanes"]).median() * lanes),
                "truck_share": float(s["truck_share"]),
                **{k: float(s[k]) for k in REPORT_ONLY if k in s}, "directions": len(x)}
    return summ(cal), summ(val), how


def errors(sim, tgt):
    e = {}
    for k, (tol, kind, _) in CRITERIA.items():
        if k in sim and k in tgt and np.isfinite(sim[k]) and np.isfinite(tgt[k]):
            e[k] = (sim[k] - tgt[k]) / (tol * tgt[k] if kind == "rel" else tol)
    return e


TRUCK_LEFT_BAN = None   # None = default (3+ lanes); True/False forces it (set by --truck-left-ban)


def simulate(p, model, lanes, tgt, minutes, seeds, workdir):
    flow_lane = tgt["flow_veh_h"] / lanes
    res = []
    for sd in seeds:
        d = workdir / f"s{sd}"; d.mkdir(parents=True, exist_ok=True)
        zone = HW.build_network(d, lanes, False, TRUCK_LEFT_BAN)
        HW.build_demand(d, lanes, False, flow_lane, 0, tgt["truck_share"], minutes, model, p)
        HW.run(d, minutes, sd)
        res.append(HW.summarise(d, zone, minutes, lanes))
        shutil.rmtree(d, ignore_errors=True)               # fcd files are large; keep only the summary
    keys = set(res[0]) - {"collisions"}
    return {k: float(np.nanmean([r[k] for r in res])) for k in keys} | {"collisions": int(sum(r["collisions"] for r in res))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Krauss", choices=["Krauss", "IDM"])
    ap.add_argument("--lanes", type=int, default=3, choices=[2, 3])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--refine", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--recheck", type=int, default=5, help="best candidates re-tested on fresh seeds before selection")
    ap.add_argument("--rng", type=int, default=131)
    ap.add_argument("--truck-left-ban", default="auto", choices=["auto", "on", "off"],
                    help="trucks barred from the leftmost lane: auto = only on 3+ lane roads")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    global TRUCK_LEFT_BAN
    TRUCK_LEFT_BAN = {"auto": None, "on": True, "off": False}[a.truck_left_ban]
    bounds = {**BOUNDS, **(KRAUSS_ONLY if a.model == "Krauss" else IDM_ONLY)}
    tcal, tval, how = targets(a.lanes)
    print(f"model {a.model} | lanes {a.lanes} | {how}")
    print("calibration targets:", {k: round(v, 3) for k, v in tcal.items()})
    tag = f"calib_{a.model}_L{a.lanes}" + (f"_{a.tag}" if a.tag else "")
    out = Path("results") / f"{tag}_runs.csv"; out.parent.mkdir(exist_ok=True)
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rng = np.random.default_rng(a.rng)
    seeds = list(range(1, a.seeds + 1))
    work = Path("sim/calib") / tag
    plan = [("sample", {k: float(rng.uniform(*b)) for k, b in bounds.items()}) for _ in range(a.samples)]

    def evaluate(i, stage, p):
        if not done.empty and i in set(done["run"]):
            r = done[done["run"] == i].iloc[0]
            return float(r["loss"]), {k: float(r[k]) for k in bounds}
        t0 = time.time()
        sim = simulate(p, a.model, a.lanes, tcal, a.minutes, seeds, work / f"run{i}")
        e = errors(sim, tcal)
        loss = float(np.mean([v ** 2 for v in e.values()]))
        accepted = all(abs(e[k]) <= 1 for k, (_, _, acc) in CRITERIA.items() if acc and k in e)
        row = {"run": i, "stage": stage, **p, **{f"sim_{k}": v for k, v in sim.items()},
               **{f"err_{k}": v for k, v in e.items()}, "loss": loss, "accepted": accepted}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        worst = max(e, key=lambda k: abs(e[k]))
        print(f"  run {i:3d} [{stage}] loss {loss:7.2f} | accepted {accepted} | worst: {worst} ({e[worst]:+.1f}) | {time.time()-t0:.0f}s", flush=True)
        return loss, p

    best = (np.inf, None)
    for i, (stage, p) in enumerate(plan):
        best = min(best, evaluate(i, stage, p), key=lambda x: x[0])
    step = 0.15
    for j in range(a.refine):
        i = a.samples + j
        p = {k: float(np.clip(best[1][k] + rng.normal(0, step * (b[1] - b[0])), *b)) for k, b in bounds.items()}
        cand = evaluate(i, "refine", p)
        if cand[0] < best[0]:
            best = cand
        else:
            step *= 0.9
    # robust selection: re-test the best candidates on fresh seeds (guards against lucky seeds)
    hist = pd.read_csv(out).sort_values("loss").head(a.recheck)
    cands = []
    for _, r in hist.iterrows():
        p = {k: float(r[k]) for k in bounds}
        sim = simulate(p, a.model, a.lanes, tcal, a.minutes, [11, 12, 13], work / f"recheck{int(r['run'])}")
        e = errors(sim, tcal); l2 = float(np.mean([v ** 2 for v in e.values()]))
        print(f"  recheck run {int(r['run']):3d}: search loss {r['loss']:.2f} -> fresh-seed loss {l2:.2f}", flush=True)
        cands.append((l2, p, int(r["run"])))
    best_l, p_best, best_run = min(cands, key=lambda x: x[0])
    best = (best_l, p_best)
    print(f"selected run {best_run} (fresh-seed loss {best_l:.2f})")
    print(f"\nbest loss {best[0]:.2f}; validating with 3 new seeds. Traffic volume and truck share are INPUTS:")
    print(f"  calibration run uses the calibration locations' inputs; validation run uses the validation locations' own")
    print(f"  inputs ({tval['flow_veh_h']:.0f} veh/h, {tval['truck_share']:.1%} trucks; {tval['directions']} recording-directions),")
    print(f"  so the validation tests whether the calibrated DRIVER BEHAVIOUR transfers to other roads.")
    sim = simulate(p_best, a.model, a.lanes, tcal, a.minutes, [101, 102, 103], work / "validate_cal")
    simv = simulate(p_best, a.model, a.lanes, tval, a.minutes, [101, 102, 103], work / "validate_val")
    ecal, eval_ = errors(sim, tcal), errors(simv, tval)
    rows = []
    for k, (tol, kind, acc) in CRITERIA.items():
        rows.append(dict(metric=k, highD_cal=round(tcal.get(k, np.nan), 3), sumo_cal=round(sim.get(k, np.nan), 3),
                         pass_cal=abs(ecal.get(k, np.inf)) <= 1, highD_val=round(tval.get(k, np.nan), 3),
                         sumo_val=round(simv.get(k, np.nan), 3), pass_val=abs(eval_.get(k, np.inf)) <= 1,
                         tolerance=f"±{tol:.0%}" if kind == "rel" else f"±{tol}", in_acceptance=acc))
    for k in REPORT_ONLY:
        rows.append(dict(metric=k, highD_cal=round(tcal.get(k, np.nan), 6), sumo_cal=round(sim.get(k, np.nan), 6), pass_cal="",
                         highD_val=round(tval.get(k, np.nan), 6), sumo_val=round(simv.get(k, np.nan), 6), pass_val="",
                         tolerance="report only", in_acceptance=False))
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    acc = rep[rep["in_acceptance"] == True]
    verdict = dict(calibration=bool(acc["pass_cal"].all()), validation=bool(acc["pass_val"].all()))
    print(f"\nACCEPTANCE: calibration locations {'PASS' if verdict['calibration'] else 'FAIL'} | "
          f"validation locations {'PASS' if verdict['validation'] else 'FAIL'} | collisions in validation runs: {sim['collisions'] + simv['collisions']}")
    if tval["directions"] < 20:
        print(f"note: the validation set has only {tval['directions']} recording-directions; report this when interpreting it.")
    json.dump({"model": a.model, "lanes": a.lanes, "params": p_best, "loss": best[0], "verdict": verdict,
               "split": how, "report": rows}, open(Path("results") / f"{tag}_best.json", "w"), indent=2, default=str)
    rep.to_csv(Path("results") / f"{tag}_validation.csv", index=False)
    shutil.rmtree(work, ignore_errors=True)
    print(f"saved results/{tag}_runs.csv, _best.json, _validation.csv")


if __name__ == "__main__":
    main()