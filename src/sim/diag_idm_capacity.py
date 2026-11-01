"""
Step 21 diagnostic: does the IDM v3 calibration reach its capacity at L1 traffic volume?
Runs IDM v3 WITHOUT any disturbances (plain SUMO) at several demands and prints the delivered flow.
Usage: python src/sim/diag_idm_capacity.py            (about 1-2 minutes per demand)
"""
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import highway as HW

p = json.load(open("results/calib_IDM_L3_best_v3.json"))["params"]
for q in (3500, 3694, 3788):
    d = Path(f"sim/diag/idm_{q}").resolve()
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    zone = HW.build_network(d, 3, False)
    HW.build_demand(d, 3, False, q / 3, 0, 0.191, 10, "IDM", p)
    HW.run(d, 10, 1)
    s = HW.summarise(d, zone, 10, 3)
    print(f"demand {q} veh/h -> delivered {s['flow_veh_h']:.0f} veh/h ({s['flow_veh_h'] / q - 1:+.1%}) | "
          f"median speed {s['speed_p50_kmh']:.1f} km/h | median gap {s['time_gap_median_s']:.2f} s", flush=True)
    shutil.rmtree(d, ignore_errors=True)