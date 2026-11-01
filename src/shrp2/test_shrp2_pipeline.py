"""
Tests of the SHRP2 pipeline after the loader - registered split, samples and labels, surrogate safety
measures, LightGBM features - on HAND-BUILT scenes with answers worked out by hand, plus a cross-check
against the audit on the synthetic SHRP2-format dataset. No SHRP2 data involved.
Usage : python src/shrp2/test_shrp2_pipeline.py     (temporary folder only; ~20-60 s)
"""
import shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "loaders"))
import labels as LB
import objects as OB
import make_split as MS
import build_samples_shrp2 as BS
import ssm_shrp2 as SS
import make_synthetic as M
import audit_counts as A
import shrp2 as LD

RES = []


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ----------------------------------------------------------------------------- hand-built scenes
def scene(eid, n=300, t0=0, v=10.0, heading=0.0, objects=(), masked=(), ego_L=4.6, ego_W=1.8):
    """Subject drives straight at speed v along `heading` from (0,0). objects: dicts with keys
    id, lon (m ahead of the subject centre along the subject heading), lat (m to the left), speed, heading_rel,
    frames (range of frame indices). Returns common-format rows as written by src/loaders/shrp2.py."""
    rows = []
    ts = t0 + 100 * np.arange(n); hx, hy = np.cos(heading), np.sin(heading)
    ex, ey = v * (ts - t0) / 1000.0 * hx, v * (ts - t0) / 1000.0 * hy
    for i in range(n):
        m = i in masked
        rows.append(dict(agent_id="ego", is_ego=True, x=np.nan if m else ex[i], y=np.nan if m else ey[i],
                         vx=np.nan if m else v * hx, vy=np.nan if m else v * hy, heading=np.nan if m else heading,
                         length=ego_L, width=ego_W, timestamp_ms=int(ts[i]), masked=m))
    for o in objects:
        oh = heading + o.get("heading_rel", 0.0)
        for i in o["frames"]:
            px = ex[i] + o["lon"] * hx - o["lat"] * hy; py = ey[i] + o["lon"] * hy + o["lat"] * hx
            rows.append(dict(agent_id=str(o["id"]), is_ego=False, x=px, y=py, vx=o["speed"] * np.cos(oh),
                             vy=o["speed"] * np.sin(oh), heading=oh, length=np.nan, width=np.nan,
                             timestamp_ms=int(ts[i]), masked=False))
    g = pd.DataFrame(rows)
    g["dataset"] = "SHRP2"; g["scene_id"] = f"SHRP2/{eid}"; g["time_s"] = g["timestamp_ms"] / 1000.0
    g["event_id"] = eid; g["ax"] = np.nan; g["ay"] = np.nan
    return g


def main():
    tmp = Path(tempfile.mkdtemp(prefix="shrp2pipe_"))
    try:
        # ---------------- S1 registered split ----------------
        T = pd.DataFrame(dict(event_id=np.arange(1, 40), in_study=True, excluded_implausible=False,
                              outcome_type=["baseline"] * 25 + ["crash"] * 4 + ["near-crash"] * 10,
                              lighting_bin=[None] * 25 + [1] * 4 + [0] * 10,
                              surfaceCondition_bin=[None] * 25 + [0] * 4 + [1] * 10))
        T.loc[38, "excluded_implausible"] = True                      # one excluded trip: never split
        S = MS.make_split(T)
        tab = MS.summary(S)
        exp = {"baseline|LU|SU": (19, 3, 3), "crash|L1|S0": (4, 0, 0), "near-crash|L0|S1": (7, 1, 1)}
        got = {k: tuple(int(x) for x in tab.loc[k, ["train", "val", "test"]]) for k in exp}
        check("S1 counts per stratum: n_val = n_test = floor(0.1 n + 0.5)", got == exp, str(got))
        check("S1 every unit once; excluded trip not split", len(S) == 38 and S.event_id.is_unique and 39 not in set(S.event_id))
        S2 = MS.make_split(T.sample(frac=1, random_state=7))           # row order must not matter
        check("S1 deterministic (seed 131) and independent of row order", S.equals(S2))
        MS.SEED = 132; S3 = MS.make_split(T); MS.SEED = 131
        check("S1 a different seed gives a different split", not S.equals(S3))
        bad = T.assign(driver_id=1)
        try:
            MS.make_split(bad); ok = False
        except SystemExit:
            ok = True
        check("S1 refuses drivers until the driver-stratum rule is frozen", ok)

        # ---------------- S3 labels by hand ----------------
        ts = 100 * np.arange(300); te = 20050
        idx = LB.prediction_index(ts); t = ts[idx]
        excl, ev, obs = LB.label_arrays(t, te, ts[-1])
        exp_ev, exp_ex = [], []
        for tm in t:
            tte = (te - tm) / 1000.0
            if tte <= 0 or 5.0 < tte <= 7.0:
                exp_ex.append(True); exp_ev.append(-1); continue
            exp_ex.append(False)
            exp_ev.append(0 if tte <= 1 else 1 if tte <= 2 else 2 if tte <= 3 else 3 if tte <= 5 else -1)
        check("S3 event bins, buffer and post-event exclusion (te off the grid)", list(excl) == exp_ex and
              list(ev[~excl]) == [e for e, x in zip(exp_ev, exp_ex) if not x] and
              (ev == 0).sum() == 10 and (ev == 3).sum() == 20 and excl.sum() == 20 + (len(t) - int((t < te).sum())))
        for te_g in (20000,):                                       # event exactly on the grid: every boundary hit
            exg, evg, _ = LB.label_arrays(t, te_g, ts[-1]); eg, xg = [], []
            for tm in t:
                tte = (te_g - tm) / 1000.0
                xg.append(tte <= 0 or 5.0 < tte <= 7.0)
                eg.append(-1 if xg[-1] else 0 if tte <= 1 else 1 if tte <= 2 else 2 if tte <= 3 else 3 if tte <= 5 else -1)
            check("S3 boundaries on the grid: t = te excluded, 1/2/3/5 s closed on the right, 5 s kept, 7 s excluded",
                  list(exg) == xg and list(evg[~exg]) == [e for e, x in zip(eg, xg) if not x]
                  and (evg == 0).sum() == 10 and 15000 in set(t[~exg]) and 13000 not in set(t[~exg]) and 20000 not in set(t[~exg]))
        o_end = LB.label_arrays(np.array([27500, 28000, 29000]), None, 29900)[2]
        check("S3 censoring: obs_bins near the trip end", list(o_end) == [2, 1, 0], str(o_end))

        # ---------------- S4/S5/S6/S7 samples by hand ----------------
        objs = [dict(id=11, lon=20, lat=0, speed=8, frames=range(0, 300)),
                dict(id=12, lon=25, lat=3, speed=10, frames=range(0, 300)),
                dict(id=13, lon=35, lat=0, speed=10, frames=range(0, 300)),          # beyond 30 m
                dict(id=14, lon=26, lat=-2, speed=12, heading_rel=np.pi, frames=range(150, 300))]  # oncoming
        for hd in (0.0, np.pi / 2):
            g = scene(1, heading=hd, objects=objs)
            r = BS.trip_samples(g, "test", "s", None, "baseline", "asgiven", "zero")
            k = int(np.flatnonzero(r["frame"] == 100)[0]); h = r["hist"][k]
            c1 = np.allclose(h[0, -1, :6], [0, 0, 10, 0, 0, 1], atol=1e-9) and h[0, -1, 8] == 1 and np.isclose(h[0, 0, 0], -19.0)
            c2 = np.allclose(h[1, -1, :4], [20, 0, 8, 0], atol=1e-9) and np.allclose(h[2, -1, :4], [25, 3, 10, 0], atol=1e-9)
            c3 = h[3:, :, 8].sum() == 0                                   # 35 m object excluded, no 4th object yet
            check(f"S4 local frame, history, nearest objects, 30 m radius (heading {hd:.2f})", c1 and c2 and c3)
        g = scene(1, objects=objs)
        r = BS.trip_samples(g, "test", "s", None, "baseline", "asgiven", "zero")
        k = int(np.flatnonzero(r["frame"] == 160)[0]); h = r["hist"][k]
        check("S4 oncoming object enters when present (sorted by distance: 20, 25.2, 26.1 m); absent rows invalid",
              np.allclose(h[3, -1, :3], [26, -2, -12], atol=1e-9) and h[3, 0, 8] == 0 and h[3, 9, 8] == 1 and h[3, 8, 8] == 0)
        rc = BS.trip_samples(g, "test", "s", None, "baseline", "centre_from_edge", "car")
        hc = rc["hist"][k]
        check("S5 centre_from_edge: ahead +L/2 (22.25), oncoming further away (28.25), sizes 4.5 x 1.75",
              np.isclose(hc[1, -1, 0], 22.25) and np.isclose(hc[3, -1, 0], 28.25) and np.allclose(hc[1, -1, 6:8], [4.5, 1.75]))
        try:
            OB.check("centre_from_edge", "zero"); ok = False
        except ValueError:
            ok = True
        check("S5 centre rule without a size is refused", ok)
        last = r["frame"].max(); kl = int(np.flatnonzero(r["frame"] == last)[0]); k0 = int(np.flatnonzero(r["frame"] == 100)[0])
        check("S6 future: 1 m per frame ahead; censored at trip end", np.allclose(r["fut"][k0, :, 0], np.arange(1, 31))
              and np.allclose(r["fut"][k0, :, 1], 0) and r["fut_valid"][kl].sum() == 0 and r["fut_valid"][k0].sum() == 30)
        gm = scene(1, objects=objs, masked={100})
        rm = BS.trip_samples(gm, "test", "s", None, "baseline", "asgiven", "zero")
        k5 = int(np.flatnonzero(rm["frame"] == 105)[0])
        check("S7 masked subject frame: no prediction at it; invalid row in later histories",
              100 not in set(rm["frame"]) and rm["hist"][k5, 0, 14, 8] == 0 and rm["hist"][k5, 0, 13, 8] == 1)
        rp = BS.trip_samples(g, "test", "s", te, "crash", "asgiven", "zero", p1=True)
        check("S12 P1 inference samples (Amendment 4): every valid frame kept, excluded ones flagged current = 1",
              len(rp["frame"]) == len(t) and int(rp["current"].sum()) == int(excl.sum()) and np.array_equal(rp["current"] == 1, excl))
        rt = BS.trip_samples(g, "train", "s", None, "baseline", "asgiven", "zero")
        check("S10 training stride: frames 19, 29, 39, ... from the first frame", list(rt["frame"][:3]) == [19, 29, 39]
              and len(rt["frame"]) == 29)
        re_ = BS.trip_samples(g, "test", "s", te, "crash", "asgiven", "zero")
        check("S3 samples carry the same labels; excluded frames not written", len(re_["frame"]) == int((~excl).sum())
              and np.array_equal(re_["event_bin"], ev[~excl]) and np.all(re_["tte_ms"] == te - re_["frame"] * 100))

        # ---------------- S8 surrogate safety measures by hand ----------------
        lead = [dict(id=21, lon=20, lat=0, speed=2, frames=range(0, 50))]
        g8 = scene(8, n=50, objects=lead)
        res = {}
        for ref, size in (("asgiven", "zero"), ("centre_from_edge", "car"), ("asgiven", "car")):
            from baselines import ssm as SSMM
            R = SSMM.compute(SS.scene_frame(g8, ref, size)); R = R[R.agent_id == "ego"].set_index("time_s")
            res[(ref, size)] = R.loc[0.0]
        a, b, c = res[("asgiven", "zero")], res[("centre_from_edge", "car")], res[("asgiven", "car")]
        check("S8 TTC = 17.7 / 8 = 2.2125 s and DRAC = 64 / 35.4 (point object)", np.isclose(a.ttc_min, 2.2125) and np.isclose(a.drac_max, 64 / 35.4))
        check("S8 centre rule + car size gives the same gap (17.7 m) as the point object", np.isclose(b.ttc_min, 2.2125))
        check("S8 radar point + car size shortens the gap by L/2 (15.45 m)", np.isclose(c.ttc_min, 15.45 / 8))
        check("S8 TTC2D first overlap on the 0.1-s grid = 2.3 s; moving filter keeps it", np.isclose(a.ttc2d_min, 2.3) and np.isclose(a.ttc2d_mov, 2.3))

        # ---------------- S2 cross-check with the audit, S9 features, S11 model ----------------
        root = M.build(tmp / "syn", "complete", scale=5)
        work = tmp / "data/processed/shrp2"
        LD.main(root, work, mask=True)
        MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        U = BS.main(work, "asgiven", "car", res_dir=tmp / "res").set_index("event_id")
        Sa, Ta = A.audit(root); Ta = Ta.set_index("event_id")
        ev_units = U[U.split != "train"]
        mism = [e for e in ev_units.index if ev_units.at[e, "n_known3"] != Ta.at[e, "n_eval3"] or ev_units.at[e, "n_pos3"] != Ta.at[e, "n_pos3"]]
        check("S2 validation/test samples known at 3 s = audit evaluable frames, per unit", not mism and len(ev_units) > 0, f"{len(ev_units)} units")
        tr = U[U.split == "train"]
        exp_tr = 0
        ALL = pd.concat([pd.read_parquet(p) for p in work.glob("traj_*.parquet")]); ALL = ALL[ALL.is_ego]
        for e in tr.index:
            gg = ALL[ALL.event_id == e]
            tse = np.sort(gg["timestamp_ms"].to_numpy()); m_ = gg.sort_values("timestamp_ms")["masked"].to_numpy()
            ii = LB.prediction_index(tse, m_); tp = tse[ii]
            te_e = Ta.at[e, "pre_event_s"]; te_e = None if pd.isna(te_e) else int(round(te_e * 1000)) + int(tse[0])
            ex_, _, _ = LB.label_arrays(tp, te_e, tse[-1])
            exp_tr += int((((tp // 100 - tse[0] // 100) % 10) == 9)[~ex_].sum())
        check("S2 training samples = stride frames of valid, non-excluded times", int(tr.n_samples.sum()) == exp_tr, f"{exp_tr}")
        SS.main(work, "asgiven", "car")
        import features_shrp2 as FE
        X = FE.main(work)
        d = dict(np.load(sorted((work / "samples").glob("test_*.npz"))[0], allow_pickle=True))
        sp = np.hypot(d["hist"][:, 0, -1, 2], d["hist"][:, 0, -1, 3])
        nob = (d["hist"][:, 1:, -1, 8] > 0).sum(1)
        check("S9 features: speed and object count equal the samples; rows aligned", np.allclose(X["test"]["speed"], sp)
              and np.array_equal(X["test"]["n_objects"], nob) and np.array_equal(X["test"]["frame"], d["frame"]))
        import torch
        from models.encoder import InteractionEncoder
        from models.hazard import HazardHead, hazard_nll
        enc, head = InteractionEncoder(), HazardHead(64, 4)
        xb = torch.from_numpy(d["hist"][:64]); lo = hazard_nll(head(enc(xb)), torch.from_numpy(d["event_bin"][:64]),
                                                            torch.from_numpy(d["obs_bins"][:64]))
        check("S11 frozen encoder + hazard head accept the samples (finite loss)", bool(torch.isfinite(lo)), f"{float(lo.detach()):.4f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = sum(ok for _, ok in RES); print(f"\n{n}/{len(RES)} PASS"); sys.exit(0 if n == len(RES) else 1)


if __name__ == "__main__":
    main()
