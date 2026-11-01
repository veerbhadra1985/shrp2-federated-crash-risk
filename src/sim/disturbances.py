"""
Step 21: realistic disturbances in the calibrated SUMO motorway (via TraCI).

Two disturbance types, applied to random CARS on the approach (last 600 m) and in the zone:
  hard braking : speed reduced at a deceleration drawn from U(brake_decel_min, brake_decel_max)
                 (max 6 m/s^2, a strong but plausible emergency stop) for a duration ~ U(1, 3) s
  cut-in       : forced change into an adjacent lane where the new follower is 5-20 m behind
                 (front-to-rear gap) AND the follower's required deceleration DRAC = dv^2/(2 gap)
                 is at most cutin_drac_max; lane-change safety checks are suspended only for that manoeuvre
Rates are events per vehicle-hour (Poisson). Driver parameters are NOT changed.
Measured afterwards with the same definitions as the calibration (highway.summarise), plus the
maximum deceleration actually realised by any vehicle in the zone.
"""
import os, sys, random, shutil, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import highway as HW
try:
    import traci
except ImportError:
    import sumo
    sys.path.append(os.path.join(sumo.SUMO_HOME, "tools"))
    import traci

DEFAULT_LC_MODE = 1621          # SUMO default lane-change mode
FORCED_LC_MODE = 0              # TraCI request executed without safety checks (cut-in only)
DT = 0.1


def eligible(vid):
    e = traci.vehicle.getRoadID(vid)
    if not traci.vehicle.getTypeID(vid).startswith("car"):
        return False
    if e == "zone":
        return True
    return e == "approach" and traci.vehicle.getLanePosition(vid) > HW.APPROACH - 600


def run_disturbed(d, minutes, seed, dp, lanes):
    """dp: brake_rate, cutin_rate (events per vehicle-hour), brake_decel_min/max (m/s^2),
    cutin_gap_min/max (m). Returns dict with event counts."""
    d = Path(d).resolve()                      # absolute: SUMO resolves relative paths from the config folder
    HW.write_config(d, minutes, seed)
    rng = random.Random(seed * 7919 + 17); nrng = np.random.default_rng(seed * 104729 + 3)
    traci.start([HW.sumo_bin("sumo"), "-c", str(d / "sim.sumocfg"), *[a if a != "ssm.xml" else str(d / "ssm.xml") for a in HW.SSM_ARGS]],
                label=f"d{seed}_{os.getpid()}")
    restore, busy = {}, {}
    n_brake = n_cut = n_cut_tried = 0
    end = minutes * 60 + 120
    try:
        while traci.simulation.getTime() < end:
            traci.simulationStep()
            t = traci.simulation.getTime()
            for vid, tr in list(restore.items()):
                if t >= tr:
                    if vid in traci.vehicle.getIDList():
                        traci.vehicle.setLaneChangeMode(vid, DEFAULT_LC_MODE)
                    del restore[vid]
            busy = {v: tb for v, tb in busy.items() if tb > t}
            cand = [v for v in traci.vehicle.getIDList() if v not in busy and eligible(v)]
            if not cand:
                continue
            lam = len(cand) * DT / 3600
            for _ in range(nrng.poisson(dp["brake_rate"] * lam)):
                v = rng.choice(cand); sp = traci.vehicle.getSpeed(v)
                if sp < 15:
                    continue
                dec = rng.uniform(dp["brake_decel_min"], min(dp["brake_decel_max"], 6.0)); dur = rng.uniform(1.0, 3.0)
                traci.vehicle.slowDown(v, max(5.0, sp - dec * dur), dur)
                busy[v] = t + dur + 2; n_brake += 1
            for _ in range(nrng.poisson(dp["cutin_rate"] * lam)):
                v = rng.choice(cand); n_cut_tried += 1
                lane = traci.vehicle.getLaneIndex(v); nl = lanes
                sides = [s for s in (+1, -1) if 0 <= lane + s < nl]
                rng.shuffle(sides)
                for s in sides:
                    fol = traci.vehicle.getLeftFollowers(v) if s > 0 else traci.vehicle.getRightFollowers(v)
                    sp_v = traci.vehicle.getSpeed(v)
                    ok = []
                    for fid, g in fol:
                        if not (dp["cutin_gap_min"] <= g <= dp["cutin_gap_max"]):
                            continue
                        dv = traci.vehicle.getSpeed(fid) - sp_v          # follower closing speed
                        drac = dv * dv / (2 * g) if dv > 0 else 0.0      # deceleration the follower needs
                        if drac <= dp.get("cutin_drac_max", 99.0):
                            ok.append(g)
                    if ok:   # the cut-in car must also be able to follow its NEW leader without hard braking
                        lead = traci.vehicle.getLeftLeaders(v) if s > 0 else traci.vehicle.getRightLeaders(v)
                        for lid, lg in lead:
                            dvl = sp_v - traci.vehicle.getSpeed(lid)
                            if lg < dp["cutin_gap_min"] or (dvl > 0 and dvl * dvl / (2 * lg) > dp.get("cutin_drac_max", 99.0)):
                                ok = []
                    allowed = traci.vehicle.getRoadID(v) and traci.lane.getDisallowed(f"{traci.vehicle.getRoadID(v)}_{lane + s}")
                    if ok and "passenger" not in (allowed or []):
                        traci.vehicle.setLaneChangeMode(v, FORCED_LC_MODE)
                        traci.vehicle.changeLane(v, lane + s, 2.0)
                        restore[v] = t + 3.0; busy[v] = t + 5.0; n_cut += 1
                        break
    finally:
        traci.close()
    return dict(n_brake=n_brake, n_cutin=n_cut, n_cutin_tried=n_cut_tried)


def decel_stats(d):
    """Largest deceleration realised by any vehicle in the zone, and the share of vehicle-frames
    braking harder than 4 m/s^2 (highD L1: 0.003%), from consecutive fcd speeds."""
    last, worst, n, hard = {}, 0.0, 0, 0
    for _, el in ET.iterparse(d / "fcd.xml", events=("end",)):
        if el.tag == "timestep":
            t = float(el.get("time"))
            for v in el:
                vid, sp = v.get("id"), float(v.get("speed"))
                if vid in last and abs(t - last[vid][0] - DT) < 1e-6:
                    dec = (last[vid][1] - sp) / DT
                    worst = max(worst, dec); n += 1; hard += dec > 4.0
                last[vid] = (t, sp)
            el.clear()
    return worst, (hard / n if n else float("nan"))


def max_decel(d):
    return decel_stats(d)[0]


def simulate_disturbed(p, dp, model, lanes, tgt, minutes, seeds, workdir):
    """Same as calibrate_sumo.simulate, with disturbances; returns seed-averaged summary.
    Emergency braking of all vehicles is capped at dp['emergency_decel'] (highD: never above 6.6 m/s^2)."""
    p = {**p, "emergency_decel": dp.get("emergency_decel", 9.0)}
    res = []
    for sd in seeds:
        d = workdir / f"s{sd}"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        zone = HW.build_network(d, lanes, False)
        HW.build_demand(d, lanes, False, tgt["flow_veh_h"] / lanes, 0, tgt["truck_share"], minutes, model, p)
        ev = run_disturbed(d, minutes, sd, dp, lanes)
        s = HW.summarise(d, zone, minutes, lanes)
        s.update(ev); s["max_decel_ms2"], s["share_decel_gt4"] = decel_stats(d)
        res.append(s); shutil.rmtree(d, ignore_errors=True)
    keys = set(res[0]) - {"collisions", "max_decel_ms2"}
    out = {k: float(np.nanmean([r[k] for r in res])) for k in keys}
    out["collisions"] = int(sum(r["collisions"] for r in res)); out["max_decel_ms2"] = float(max(r["max_decel_ms2"] for r in res))
    return out