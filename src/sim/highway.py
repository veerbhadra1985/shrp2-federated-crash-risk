"""
Step 19: highD-like motorway scenario in SUMO.

Geometry (x in metres): 1 km approach | 420 m OBSERVATION ZONE (as in highD) | 1 km exit.
Optional on-ramp with a 250 m acceleration lane inside the observation zone (merging conflicts).
Traffic: cars + trucks, 10 Hz simulation step (as in highD), SUMO SSM device logs TTC/DRAC.
Driver-model parameters are PLACEHOLDERS until Step 20 calibrates them against highD.
NOTE: SUMO's default driver models are collision-free and very cautious, so default traffic
produces almost no conflicts; Steps 20-21 add calibration and realistic disturbances.
The summary computes TTC and time gaps independently from trajectories (bumper to bumper)
as a cross-check on SUMO's SSM device.

Usage:
    python src/sim/highway.py --lanes 2 --minutes 10
    python src/sim/highway.py --lanes 3 --ramp --flow 1600 --cf IDM --minutes 10
Outputs (in sim/out/<name>/): network, routes, config, fcd.xml (observation zone only),
    ssm.xml, collisions.xml, summary.csv
"""
import argparse, csv, os, subprocess, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

APPROACH, ZONE, EXIT = 1000.0, 420.0, 1000.0
MERGE_AT, ACCEL_LEN = 100.0, 250.0            # ramp joins 100 m into the zone; acceleration lane 250 m
VMAX = 36.11                                  # 130 km/h edge speed (cars may exceed via speedFactor)


def sumo_bin(name):
    try:
        import sumo
        p = Path(sumo.SUMO_HOME) / "bin" / name
        for c in (p, p.with_suffix(".exe")):
            if c.exists():
                return str(c)
    except ImportError:
        pass
    return name


def write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def build_network(d, lanes, ramp, truck_left_ban=None):
    """truck_left_ban: trucks may not use the leftmost lane (default: on roads with 3+ lanes,
    as observed in highD (0.7% of truck frames) and required for heavy trucks in Germany)."""
    if truck_left_ban is None:
        truck_left_ban = lanes >= 3
    x0, x1, x2 = APPROACH, APPROACH + ZONE, APPROACH + ZONE + EXIT
    nodes = [("in", 0, 0), ("z0", x0, 0), ("z1", x1, 0), ("out", x2, 0)]
    if not ramp:
        edges = [("approach", "in", "z0", lanes), ("zone", "z0", "z1", lanes), ("exit", "z1", "out", lanes)]
        cons = []
    else:
        m, a = x0 + MERGE_AT, x0 + MERGE_AT + ACCEL_LEN
        nodes += [("m", m, 0), ("a", a, 0), ("r0", m - 400, -60)]
        edges = [("approach", "in", "z0", lanes), ("zone1", "z0", "m", lanes),
                 ("accel", "m", "a", lanes + 1), ("zone2", "a", "z1", lanes),
                 ("exit", "z1", "out", lanes), ("ramp", "r0", "m", 1)]
        cons = [(f"zone1", i, "accel", i + 1) for i in range(lanes)] + [("ramp", 0, "accel", 0)] + \
               [("accel", i + 1, "zone2", i) for i in range(lanes)] + [("accel", 0, "zone2", 0)]
    write(d / "net.nod.xml", "<nodes>\n" + "".join(
        f'  <node id="{n}" x="{x}" y="{y}" type="priority"/>\n' for n, x, y in nodes) + "</nodes>\n")
    def edge_xml(e, f, t, k):
        head = f'  <edge id="{e}" from="{f}" to="{t}" numLanes="{k}" speed="{VMAX if e != "ramp" else 22.22}" priority="{1 if e == "ramp" else 2}"'
        if truck_left_ban and e != "ramp" and k >= 2:
            return head + f'>\n    <lane index="{k - 1}" disallow="truck"/>\n  </edge>\n'
        return head + "/>\n"
    write(d / "net.edg.xml", "<edges>\n" + "".join(edge_xml(e, f, t, k) for e, f, t, k in edges) + "</edges>\n")
    write(d / "net.con.xml", "<connections>\n" + "".join(
        f'  <connection from="{f}" fromLane="{fl}" to="{t}" toLane="{tl}"/>\n' for f, fl, t, tl in cons) + "</connections>\n")
    cmd = [sumo_bin("netconvert"), "-n", str(d / "net.nod.xml"), "-e", str(d / "net.edg.xml"),
           "-x", str(d / "net.con.xml"), "-o", str(d / "net.net.xml"), "--no-turnarounds", "true",
           "--no-internal-links", "false", "--offset.disable-normalization", "true"]
    subprocess.run(cmd, check=True, capture_output=True)
    zone_edges = ["zone"] if not ramp else ["zone1", "accel", "zone2"]
    write(d / "zone_edges.txt", "\n".join(f"edge:{e}" for e in zone_edges) + "\n")
    return zone_edges


DEFAULT_PARAMS = dict(car_sf_mean=1.0, car_sf_dev=0.10, truck_vmax=25.0, tau=1.0, tau_cv=0.0, sigma=0.5,
                      min_gap=2.5, accel=2.6, decel=4.5, lc_speed_gain=1.0, lc_keep_right=1.0,
                      truck_lc_keep_right=1.0, truck_lc_speed_gain=1.0)
TAU_Z = (-1.2816, -0.5244, 0.0, 0.5244, 1.2816)   # quantiles of 5 equal-probability driver types


def car_taus(tau, cv):
    """Time headways of 5 equally likely car-driver types (lognormal spread with mean tau)."""
    if cv <= 0:
        return [tau]
    return [max(0.3, tau * np.exp(cv * z - cv * cv / 2)) for z in TAU_Z]


def build_demand(d, lanes, ramp, flow, ramp_flow, truck_share, minutes, cf, p=None, arrivals="random"):
    """arrivals='random': exponential headways (Poisson arrivals, realistic bunching);
    arrivals='even': SUMO vehsPerHour, i.e. equally spaced insertions (Step 19 behaviour)."""
    p = {**DEFAULT_PARAMS, **(p or {})}
    end = minutes * 60
    route = "approach zone exit" if not ramp else "approach zone1 accel zone2 exit"
    m, dv = p["car_sf_mean"], p["car_sf_dev"]
    sf = f"normc({m:.3f},{dv:.3f},{max(0.4, m - 3 * dv):.3f},{m + 3 * dv:.3f})"
    sigma = f' sigma="{p["sigma"]:.3f}"' if cf == "Krauss" else ""
    common = (f'accel="{p["accel"]:.3f}" decel="{p["decel"]:.3f}" emergencyDecel="{p.get("emergency_decel", 9.0):.3f}" minGap="{p["min_gap"]:.3f}" '
              f'carFollowModel="{cf}"{sigma}')
    taus = car_taus(p["tau"], p["tau_cv"])
    cars = "".join(
        f'    <vType id="car_{i}" vClass="passenger" length="4.5" width="1.8" maxSpeed="44" tau="{t:.3f}" '
        f'speedFactor="{sf}" {common} lcStrategic="1" lcCooperative="1" lcSpeedGain="{p["lc_speed_gain"]:.3f}" '
        f'lcKeepRight="{p["lc_keep_right"]:.3f}" probability="{1 / len(taus):.4f}"/>\n' for i, t in enumerate(taus))
    car_block = f'  <vTypeDistribution id="car">\n{cars}  </vTypeDistribution>'
    truck = (f'<vType id="truck" vClass="truck" length="15" width="2.5" maxSpeed="{p["truck_vmax"]:.3f}" '
             f'accel="1.0" decel="3.5" emergencyDecel="{min(7.0, p.get("emergency_decel", 9.0)):.3f}" tau="1.2" minGap="3.0" '
             f'speedFactor="normc(1.0,0.05,0.9,1.1)" carFollowModel="{cf}"{sigma} '
             f'lcKeepRight="{p["truck_lc_keep_right"]:.3f}" lcSpeedGain="{p["truck_lc_speed_gain"]:.3f}"/>')   # dynamics fixed as in Step 19
    main = flow * lanes
    flows = [("main_car", "car", main * (1 - truck_share), "main"), ("main_truck", "truck", main * truck_share, "main")]
    if ramp:
        flows.append(("ramp_car", "car", ramp_flow, "rampr"))
    routes = f'  <route id="main" edges="{route}"/>\n'
    if ramp:
        routes += '  <route id="rampr" edges="ramp accel zone2 exit"/>\n'
    def flow_line(i, t, v, r):
        rate = f'period="exp({v / 3600:.6f})"' if arrivals == "random" else f'vehsPerHour="{v:.0f}"'
        lane = "free" if r == "main" else "0"
        return (f'  <flow id="{i}" type="{t}" route="{r}" begin="0" end="{end}" {rate} '
                f'departLane="{lane}" departSpeed="max"/>\n')
    write(d / "routes.rou.xml", "<routes>\n" + car_block + "\n  " + truck + "\n" + routes +
          "".join(flow_line(i, t, v, r) for i, t, v, r in flows if v > 0) + "</routes>\n")


SSM_ARGS = ["--device.ssm.probability", "1", "--device.ssm.measures", "TTC DRAC",
            "--device.ssm.thresholds", "3.0 3.4", "--device.ssm.range", "100", "--device.ssm.file", "ssm.xml"]


def write_config(d, minutes, seed):
    """Simulation configuration shared by plain runs (run) and disturbed runs (disturbances.py)."""
    cfg = f"""<configuration>
  <input><net-file value="net.net.xml"/><route-files value="routes.rou.xml"/></input>
  <time><begin value="0"/><end value="{minutes * 60 + 120}"/><step-length value="0.1"/></time>
  <processing><collision.action value="warn"/><collision.check-junctions value="true"/>
    <time-to-teleport value="-1"/></processing>
  <random_number><seed value="{seed}"/></random_number>
  <output><fcd-output value="fcd.xml"/><fcd-output.filter-edges.input-file value="zone_edges.txt"/>
    <collision-output value="collisions.xml"/></output>
  <report><no-step-log value="true"/><no-warnings value="true"/></report>
</configuration>"""
    write(d / "sim.sumocfg", cfg)


def run(d, minutes, seed):
    write_config(d, minutes, seed)
    subprocess.run([sumo_bin("sumo"), "-c", "sim.sumocfg", *SSM_ARGS], check=True, cwd=d, capture_output=True)


LENGTH = {"car": 4.5, "truck": 15.0}


def lane_shares(lane_use, lanes):
    """Share of car and of truck frames in the rightmost (index 0) and leftmost lane."""
    out = {}
    for kind in ("car", "truck"):
        tot = sum(v for (k, _), v in lane_use.items() if k == kind)
        out[f"{kind}_share_right_lane"] = lane_use.get((kind, 0), 0) / tot if tot else float("nan")
        out[f"{kind}_share_left_lane"] = lane_use.get((kind, lanes - 1), 0) / tot if tot else float("nan")
    return out


def summarise(d, zone_edges, minutes, lanes, warmup=60.0):
    """Statistics in the observation zone. Positions are SUMO lane positions of the FRONT
    bumper, so gap = pos_leader - length_leader - pos_follower (bumper to bumper)."""
    speeds, lane_of, changes, first_seen, gaps_t, ttcs = [], {}, 0, {}, [], []
    car_sp, truck_sp = [], []
    lane_use = {}
    for _, el in ET.iterparse(d / "fcd.xml", events=("end",)):
        if el.tag == "timestep":
            t = float(el.get("time")); by_lane = {}
            for v in el:
                vid, ln, ty = v.get("id"), v.get("lane"), v.get("type")
                pos, sp = float(v.get("pos")), float(v.get("speed"))
                speeds.append(sp); first_seen.setdefault(vid, t)
                (truck_sp if ty.startswith("truck") else car_sp).append(sp)
                kind = "truck" if ty.startswith("truck") else "car"
                lane_use[(kind, int(ln.rsplit("_", 1)[1]))] = lane_use.get((kind, int(ln.rsplit("_", 1)[1])), 0) + 1
                if vid in lane_of and lane_of[vid] != ln and lane_of[vid].rsplit("_", 1)[0] == ln.rsplit("_", 1)[0]:
                    changes += 1
                lane_of[vid] = ln
                by_lane.setdefault(ln, []).append((pos, sp, 15.0 if ty.startswith("truck") else 4.5))
            for cars in by_lane.values():
                cars.sort()
                for (pf, vf, _), (pl, vl, ll) in zip(cars[:-1], cars[1:]):
                    gap = pl - ll - pf
                    if vf > 5 and gap > 0:
                        gaps_t.append(gap / vf)                       # time gap (s)
                        if vf > vl: ttcs.append(gap / (vf - vl))      # time-to-collision (s)
            el.clear()
    ttc_ssm, drac_ssm = [], []
    if (d / "ssm.xml").exists():
        for _, el in ET.iterparse(d / "ssm.xml", events=("end",)):
            if el.tag == "conflict":
                for m in el:
                    if m.tag == "minTTC" and m.get("value") not in (None, "NA"): ttc_ssm.append(float(m.get("value")))
                    if m.tag == "maxDRAC" and m.get("value") not in (None, "NA"): drac_ssm.append(float(m.get("value")))
                el.clear()
    collisions = sum(1 for _ in ET.parse(d / "collisions.xml").getroot()) if (d / "collisions.xml").exists() else 0
    window_h = (minutes * 60 - warmup) / 3600                                 # steady-state entry window
    entered = sum(warmup <= t0 <= minutes * 60 for t0 in first_seen.values())
    sp = np.array(speeds) * 3.6; g = np.array(gaps_t); tt = np.array(ttcs)
    s = dict(vehicles_in_zone=len(first_seen), flow_veh_h=round(entered / window_h),
             speed_mean_kmh=round(sp.mean(), 1), speed_p15_kmh=round(np.percentile(sp, 15), 1),
             speed_p50_kmh=round(np.percentile(sp, 50), 1),
             car_speed_mean_kmh=round(np.mean(car_sp) * 3.6, 1) if car_sp else np.nan,
             truck_speed_mean_kmh=round(np.mean(truck_sp) * 3.6, 1) if truck_sp else np.nan,
             speed_p85_kmh=round(np.percentile(sp, 85), 1), lane_changes=changes,
             lane_changes_per_veh_km=round(changes / (sp.sum() / 3.6 * 0.1 / 1000), 3) if sp.sum() > 0 else np.nan,
             time_gap_median_s=round(float(np.median(g)), 2) if len(g) else np.nan,
             time_gap_below_1s_share=round(float((g < 1).mean()), 3) if len(g) else np.nan,
             ttc_min_s=round(float(tt.min()), 2) if len(tt) else np.nan,
             ttc_below_3s_frames=int((tt < 3).sum()), ttc_below_1_5s_frames=int((tt < 1.5).sum()),
             ttc_below_3s_share=float((tt < 3).sum() / max(len(g), 1)), ttc_below_1_5s_share=float((tt < 1.5).sum() / max(len(g), 1)),
             **lane_shares(lane_use, lanes),
             ssm_conflicts=len(ttc_ssm), ssm_ttc_below_1_5s=int(np.sum(np.array(ttc_ssm) < 1.5)),
             ssm_drac_above_3_35=int(np.sum(np.array(drac_ssm) > 3.35)), collisions=collisions)
    with open(d / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(s)); w.writeheader(); w.writerow(s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", type=int, default=2, choices=[2, 3])
    ap.add_argument("--ramp", action="store_true")
    ap.add_argument("--flow", type=float, default=1500, help="mainline veh/h PER LANE")
    ap.add_argument("--ramp-flow", type=float, default=400)
    ap.add_argument("--truck-share", type=float, default=0.15)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--cf", default="Krauss", choices=["Krauss", "IDM"])
    ap.add_argument("--seed", type=int, default=131)
    a = ap.parse_args()
    name = f"L{a.lanes}{'_ramp' if a.ramp else ''}_{a.cf}_f{int(a.flow)}_s{a.seed}"
    d = Path("sim/out") / name; d.mkdir(parents=True, exist_ok=True)
    print(f"scenario {name}: building network ...", flush=True)
    zone = build_network(d, a.lanes, a.ramp)
    build_demand(d, a.lanes, a.ramp, a.flow, a.ramp_flow, a.truck_share, a.minutes, a.cf)
    print(f"running SUMO ({a.minutes:g} simulated minutes at 10 Hz) ...", flush=True)
    run(d, a.minutes, a.seed)
    s = summarise(d, zone, a.minutes, a.lanes)
    print("\nObservation zone (420 m) summary")
    for k, v in s.items():
        print(f"  {k:24} {v}")
    print(f"\nSaved outputs in {d}")


if __name__ == "__main__":
    main()