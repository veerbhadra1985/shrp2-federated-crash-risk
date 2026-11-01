"""
Step 20b, final round: multi-condition calibration with leave-one-location-out validation.

Each highD location (3 lanes, free-flowing) is one traffic condition, simulated with its OWN
traffic volume and truck share (inputs). A parameter set's loss is the EQUAL-WEIGHT average of
its per-location losses, so the parameters must work across traffic levels.
Validation: leave-one-location-out (LOLO). For each location h, the run with the lowest average
loss over the OTHER locations is selected and re-simulated on h with fresh seeds.
Caveat (reported): the refinement step is guided by all locations, so candidate generation is not
independent of h; only the selection is held out. A fully independent version needs one search
per held-out location (about 4x the compute).
Acceptance criteria and tolerances: identical to calibrate_sumo.py (7 criteria; lane use and truck
speed in the loss only). Stopping rule: misses below 2x tolerance are accepted and reported.
Usage : python src/sim/calibrate_multi.py --model Krauss --samples 40 --refine 20
        (quick test: --samples 2 --refine 1 --minutes 2 --seeds 1)
Output: results/calibmulti_<model>_L3_runs.csv, _lolo.csv, _best.json   (resumable)
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_sumo as C


def location_targets(lanes):
    t = pd.read_csv("results/highd_targets_by_direction.csv")
    t = t[(t["lanes"] == lanes) & (t["state"] == "free-flowing")]
    out = {}
    for loc, x in t.groupby("location"):
        s = x.median(numeric_only=True)
        tg = {k: float(s[k]) for k in C.CRITERIA if k in s and k != "flow_veh_h"}
        tg.update(flow_veh_h=float((x["flow_veh_h"] / x["lanes"]).median() * lanes),
                  truck_share=float(s["truck_share"]), directions=len(x))
        out[loc] = tg
    return out


def run_conditions(p, model, lanes, locs, minutes, seeds, work):
    per = {}
    for loc, tg in locs.items():
        sim = C.simulate(p, model, lanes, tg, minutes, seeds, work / loc.replace("/", "_"))
        e = C.errors(sim, tg)
        per[loc] = dict(sim=sim, err=e, loss=float(np.mean([v ** 2 for v in e.values()])))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Krauss", choices=["Krauss", "IDM"])
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--refine", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rng", type=int, default=131)
    a = ap.parse_args()
    locs = location_targets(a.lanes)
    bounds = {**C.BOUNDS, **(C.KRAUSS_ONLY if a.model == "Krauss" else C.IDM_ONLY)}
    print(f"model {a.model} | lanes {a.lanes} | conditions (equal weight):")
    for loc, tg in locs.items():
        print(f"  {loc}: {tg['directions']} recording-directions | {tg['flow_veh_h']:.0f} veh/h | trucks {tg['truck_share']:.1%} | median speed {tg['speed_p50_kmh']:.1f} km/h")
    tag = f"calibmulti_{a.model}_L{a.lanes}"
    out = Path("results") / f"{tag}_runs.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rng = np.random.default_rng(a.rng); seeds = list(range(1, a.seeds + 1)); work = Path("sim/calib") / tag
    plan = [{k: float(rng.uniform(*b)) for k, b in bounds.items()} for _ in range(a.samples)]

    def evaluate(i, stage, p):
        if not done.empty and i in set(done["run"]):
            r = done[done["run"] == i].iloc[0]
            return float(r["loss"]), {k: float(r[k]) for k in bounds}
        t0 = time.time()
        per = run_conditions(p, a.model, a.lanes, locs, a.minutes, seeds, work / f"run{i}")
        loss = float(np.mean([v["loss"] for v in per.values()]))
        row = {"run": i, "stage": stage, **p, "loss": loss, **{f"loss_{l}": v["loss"] for l, v in per.items()}}
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  run {i:3d} [{stage}] mean loss {loss:6.2f} | " +
              " ".join(f"{l.split('_')[1]} {v['loss']:5.2f}" for l, v in per.items()) + f" | {time.time()-t0:.0f}s", flush=True)
        return loss, p

    best = (np.inf, None)
    for i, p in enumerate(plan):
        best = min(best, evaluate(i, "sample", p), key=lambda x: x[0])
    step = 0.15
    for j in range(a.refine):
        p = {k: float(np.clip(best[1][k] + rng.normal(0, step * (b[1] - b[0])), *b)) for k, b in bounds.items()}
        cand = evaluate(a.samples + j, "refine", p)
        if cand[0] < best[0]:
            best = cand
        else:
            step *= 0.9

    hist = pd.read_csv(out)
    loss_cols = [f"loss_{l}" for l in locs]
    rows, verdicts = [], {}
    fresh = [101, 102, 103]
    for h in list(locs) + ["ALL (in-sample)"]:
        others = [c for c in loss_cols if c != f"loss_{h}"]
        r = hist.loc[hist[others].mean(axis=1).idxmin()]
        p = {k: float(r[k]) for k in bounds}
        test_locs = locs if h.startswith("ALL") else {h: locs[h]}
        per = run_conditions(p, a.model, a.lanes, test_locs, a.minutes, fresh, work / f"val_{h.split()[0]}")
        for loc, v in per.items():
            for k, (tol, kind, acc) in C.CRITERIA.items():
                if k in v["err"]:
                    rows.append(dict(held_out=h, location=loc, run=int(r["run"]), metric=k, highD=round(locs[loc][k], 4),
                                     sumo=round(v["sim"][k], 4), err_x_tol=round(v["err"][k], 2),
                                     passed=abs(v["err"][k]) <= 1, within_2x=abs(v["err"][k]) <= 2, in_acceptance=acc))
        sub = [x for x in rows if x["held_out"] == h and x["in_acceptance"]]
        verdicts[h] = dict(run=int(r["run"]), all_pass=all(x["passed"] for x in sub), all_within_2x=all(x["within_2x"] for x in sub),
                           worst=max(sub, key=lambda x: abs(x["err_x_tol"]))["metric"] if sub else None,
                           worst_x=max(abs(x["err_x_tol"]) for x in sub) if sub else None)
    rep = pd.DataFrame(rows)
    rep.to_csv(Path("results") / f"{tag}_lolo.csv", index=False)
    pd.set_option("display.width", 200)
    for h, v in verdicts.items():
        print(f"\n=== {'HELD-OUT ' + h if not h.startswith('ALL') else h} | selected run {v['run']} ===")
        sub = rep[(rep.held_out == h) & (rep.in_acceptance)]
        print(sub[["location", "metric", "highD", "sumo", "err_x_tol", "passed", "within_2x"]].to_string(index=False))
        print(f"  all criteria pass: {v['all_pass']} | all within 2x tolerance (stopping rule): {v['all_within_2x']} | worst: {v['worst']} ({v['worst_x']}x)")
    json.dump({"model": a.model, "lanes": a.lanes, "conditions": locs, "verdicts": verdicts,
               "best_all_params": {k: float(hist.loc[hist['loss'].idxmin(), k]) for k in bounds},
               "caveat": "refinement guided by all locations; selection held out"},
              open(Path("results") / f"{tag}_best.json", "w"), indent=2, default=str)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nsaved results/{tag}_runs.csv, _lolo.csv, _best.json")


if __name__ == "__main__":
    main()