"""
Tests of the SHRP2 confirmatory analysis, P2 part (src/shrp2/confirm_p2.py), on FABRICATED prediction files whose
properties are designed, so every expected value is computed independently (scikit-learn AUPRC, calibration.py ECE,
hand-built masks). No SHRP2 data, no training.
Usage : python src/shrp2/test_confirm_p2.py      (~2-4 min, temporary folder only)
"""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train_shrp2 as T
import confirm_p2 as CP
import calibration as C
from sklearn.metrics import average_precision_score as AP
RES = []
STOP = os.environ.get("TEST_STOP_AFTER", "")
SEEDS = T.SEEDS
FRAMES = 20


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def build(tmp, rng):
    """Trips, cache metas and histories, features, predictions of every model, client file, head selection."""
    sh = tmp / "data/processed/shrp2"; (sh / "cache").mkdir(parents=True); (tmp / "models/shrp2").mkdir(parents=True)
    (sh / "preds").mkdir(); (tmp / "results").mkdir()
    trips, eid = [], 1000
    for split, n_ev, n_base in (("train", 400, 300), ("val", 160, 120), ("test", 240, 160)):
        for i in range(n_ev):
            out = "crash" if i % 4 == 0 else "near-crash"
            lv = dict(lighting_bin=float(i % 2), weather_bin=float(i % 2 == 0 and i % 6 == 0),  # weather: level 1 small
                      surfaceCondition_bin=float((i // 2) % 2), trafficDensity_bin=float(i % 3 == 0))
            trips.append(dict(event_id=eid, split=split, outcome_type=out, **lv)); eid += 1
        for i in range(n_base):
            trips.append(dict(event_id=eid, split=split, outcome_type="baseline", lighting_bin=np.nan, weather_bin=np.nan,
                              surfaceCondition_bin=np.nan, trafficDensity_bin=np.nan, noobj=(i < 10))); eid += 1
    Tr = pd.DataFrame(trips); Tr["noobj"] = Tr["noobj"].fillna(False).astype(bool)
    Tr.drop(columns=["split", "noobj"]).to_parquet(sh / "trips.parquet", index=False)
    metas, hists = {}, {}
    for s in ("train", "val", "test"):
        rows, H = [], []
        for _, t in Tr[Tr.split == s].iterrows():
            for fr in range(FRAMES):
                if t.outcome_type == "baseline":
                    eb, ob = -1, 4
                else:
                    d = FRAMES - 1 - fr                      # last frames are closest to the event
                    eb, ob = (min(d // 2, 3) if d < 8 else -1), 4
                if fr == FRAMES - 1 and t.outcome_type == "baseline":
                    ob = 1                                   # censored at the trip end: unknown at 3 s
                rows.append(dict(event_id=t.event_id, frame=10000 + fr, event_bin=eb, obs_bins=ob, current=0, split=s,
                                 outcome_type=t.outcome_type, location="x"))
            if s != "train":
                h = np.zeros((FRAMES, 9, 20, 9), np.float32); h[:, 0, :, 8] = 1; h[:, 0, :, 2] = 10
                if not t.noobj:
                    h[:, 1, :, 8] = 1; h[:, 1, :, 0] = 15
                    if t.event_id % 5 == 0:                  # natural dropout: object missing in 30% of its frames
                        h[:, 1, :6, 8] = 0; h[:, 1, :6, 0] = 0
                H.append(h)
        metas[s] = pd.DataFrame(rows)
        metas[s].to_parquet(sh / "cache" / f"shrp2_{s}_meta.parquet", index=False)
        if s != "train":
            hists[s] = np.concatenate(H); np.save(sh / "cache" / f"shrp2_{s}_hist.npy", hists[s])
    # predictions: hazard logits -> cumulative risk; quality and calibration designed per model
    lv = {s: {f: Tr.set_index("event_id")[c].reindex(metas[s].event_id).to_numpy(float)
              for f, c in T.FACTORS.items()} for s in ("val", "test")}
    y3 = {s: C.labels_at(metas[s].event_bin.to_numpy(), metas[s].obs_bins.to_numpy(), 2)[0] for s in ("val", "test")}

    def preds(sig, bias, seed, head="hazard", factor=None, level=None):
        out = {}
        r_ = np.random.default_rng(seed)
        for s in ("val", "test"):
            n = len(metas[s]); y = y3[s]
            shift = bias if factor is None else np.where(lv[s][factor] == level, 0.0, bias)
            p = 1 / (1 + np.exp(-(sig * (y - 0.5) * 4 - 2.5 + shift + r_.normal(0, 1, n))))
            lg = np.repeat(logit(1 - (1 - p) ** (1 / 3))[:, None], 4, 1) + r_.normal(0, 0.05, (n, 4))
            out[s] = lg
        return out

    def save(name, P, head="hazard", seed_used=131):
        ev = np.r_[metas["val"].event_id, metas["test"].event_id]; fr = np.r_[metas["val"].frame, metas["test"].frame]
        sp = np.r_[np.full(len(metas["val"]), "val"), np.full(len(metas["test"]), "test")]
        lg = np.concatenate([P["val"], P["test"]])
        np.savez_compressed(sh / "preds" / f"haz_{name}.npz", event_id=ev, frame=fr, split=sp, logits=lg.astype(np.float32),
                            risk=T.risk_of(lg, head).astype(np.float32), head=head, seed_used=seed_used)
    for i, s in enumerate(SEEDS):
        save(f"central_hazard_aug_s{s}", preds(1.0, 0.0, s))
        save(f"central_hazard_noaug_s{s}", preds(0.8, 0.0, 100 + s))
        save(f"central_sep_aug_s{s}", preds(0.9, 0.0, 200 + s, "sep"), head="sep")
        for f in T.FACTORS:
            for v in (0, 1):
                save(f"dom_{f}_{v}_hazard_s{s}", preds(1.0, 1.5, 300 + s + 10 * v, factor=f, level=v))   # miscalibrated off its level
            save(f"match_{f}_hazard_s{s}", preds(0.9, 1.5, 400 + s, factor=f, level=0))
        # LightGBM and classical
        for split in ("val", "test"):
            pass
        zl = preds(0.6, 0.0, 500 + s)
        ev = np.r_[metas["val"].event_id, metas["test"].event_id]; fr = np.r_[metas["val"].frame, metas["test"].frame]
        sp = np.r_[np.full(len(metas["val"]), "val"), np.full(len(metas["test"]), "test")]
        np.savez_compressed(sh / "preds" / f"lgbm_s{s}.npz", event_id=ev, frame=fr, split=sp,
                            risk=C.risk_from_logits(np.concatenate([zl["val"], zl["test"]])).astype(np.float32))
        # federated: clients = lighting level of event trips; baselines in no client
        for reg, sig in (("local", 0.5, ), ("fedavg", 1.0), ("fedper", 0.9)):
            P = preds(sig, 0.8 if reg == "fedavg" else 0.0, 600 + s + {"local": 0, "fedavg": 1, "fedper": 2}[reg])
            E, Fr, Sp, Cl, L = [], [], [], [], []
            for split in ("val", "test"):
                m = np.isfinite(lv[split]["lighting"])
                E.append(metas[split].event_id.to_numpy()[m]); Fr.append(metas[split].frame.to_numpy()[m])
                Sp.append(np.full(m.sum(), split)); Cl.append(np.where(lv[split]["lighting"][m] == 0, "L0", "L1")); L.append(P[split][m])
            lg = np.concatenate(L)
            np.savez_compressed(sh / "preds" / f"fed_{reg}_s{s}.npz", event_id=np.concatenate(E), frame=np.concatenate(Fr),
                                split=np.concatenate(Sp), client=np.concatenate(Cl), logits=lg.astype(np.float32),
                                risk=C.risk_from_logits(lg).astype(np.float32), head="hazard", seed_used=s)
        # small real models for the flicker test (aug, noaug, both heads)
        for nm, hk in ((f"central_hazard_aug_s{s}", "hazard"), (f"central_hazard_noaug_s{s}", "hazard"), (f"central_sep_aug_s{s}", "sep")):
            torch.manual_seed(hash(nm) % 1000)
            e, h = T.InteractionEncoder(), T.HazardHead(64, 4)
            torch.save({"encoder": e.state_dict(), "head": h.state_dict(), "head_kind": hk, "seed_used": s}, tmp / "models/shrp2" / f"haz_{nm}.pt")
    for s in ("val", "test"):
        F = metas[s][["event_id", "frame"]].copy()
        F["ttc"] = np.where(y3[s] == 1, rng.uniform(1, 4, len(F)), rng.uniform(2, 9, len(F)))
        F.loc[rng.random(len(F)) < 0.3, "ttc"] = np.nan
        F["drac"] = rng.uniform(0, 3, len(F)); F["ttc2d"] = rng.uniform(1, 9, len(F)); F["ttc2d_mov"] = np.nan
        F.to_parquet(sh / f"features_{s}.parquet", index=False)
    (tmp / "results/shrp2_head_selection.json").write_text(json.dumps(dict(choice="hazard", smoke=False)))
    (tmp / "results/shrp2_fed_clients.json").write_text(json.dumps(dict(testable=True)))
    return Tr, metas, hists


def main():
    tmp = Path(tempfile.mkdtemp(prefix="confp2_")); cwd = os.getcwd()
    try:
        os.chdir(tmp); rng = np.random.default_rng(3)
        Tr, metas, hists = build(tmp, rng)
        bank = CP.Bank()
        B = 199
        mt = metas["test"]; y3, kn3 = C.labels_at(mt.event_bin.to_numpy(), mt.obs_bins.to_numpy(), 2)
        Tix = np.flatnonzero(kn3); y = y3[Tix]
        R = lambda nm, k=2, s="test": bank.model(nm)[s][0][:, k]
        names = [f"central_{h}_{a}_s{s}" for h in ("hazard", "sep") for a in ("aug", "noaug") for s in SEEDS
                 if (T.MODELS / f"haz_central_{h}_{a}_s{s}.pt").exists()]
        drop, flick = CP.test_hist_pass(bank, names)
        extra = dict(lgbm={s: bank.lgbm(s) for s in SEEDS}, classical=bank.classical(), drop=drop, flicker=flick)
        V = CP.variants(bank, 2, [1, 2, 5])
        rows, pairs, conf = CP.endpoints(bank, V[0], B, extra)
        Rw = pd.DataFrame(rows).set_index("test")

        # ---------------- K1 H1 key secondary (P2) ----------------
        exp = np.mean([AP(y, R(f"central_hazard_aug_s{s}")[Tix]) - AP(y, extra["lgbm"][s]["test"][Tix, 2]) for s in SEEDS])
        cs = extra["classical"]; yv, knv = C.labels_at(metas["val"].event_bin.to_numpy(), metas["val"].obs_bins.to_numpy(), 2)
        vb = {c: AP(yv[knv], cs["val"][c][knv]) for c in cs["val"]}; best = max(vb, key=vb.get)
        exp2 = np.mean([AP(y, R(f"central_hazard_aug_s{s}")[Tix]) - AP(y, cs["test"][best][Tix]) for s in SEEDS])
        check("K1 H1-P2: model - LightGBM and model - best classical (chosen on validation) = scikit-learn AUPRC, seed mean",
              np.isclose(Rw.loc["H1_P2 model-LightGBM", "estimate"], exp) and np.isclose(Rw.loc["H1_P2 model-best classical", "estimate"], exp2)
              and best in Rw.loc["H1_P2 model-best classical", "note"], f"best classical {best}")
        # ---------------- K2-K4 H4 ----------------
        hand_drop = np.zeros(len(mt), bool)
        for i, (e_, fr) in enumerate(zip(mt.event_id, mt.frame)):
            t = Tr.set_index("event_id").loc[e_]
            hand_drop[i] = (not t.noobj) and e_ % 5 == 0
        d = hand_drop[Tix]
        exp4a = np.mean([AP(y[d], R(f"central_hazard_aug_s{s}")[Tix][d]) - AP(y[d], R(f"central_hazard_noaug_s{s}")[Tix][d]) for s in SEEDS])
        check("K2 H4a: natural-dropout flag = hand construction (object missing in 6 of 20 frames); estimate on that subset",
              np.array_equal(drop, hand_drop) and np.isclose(Rw.loc["H4a", "estimate"], exp4a), f"{int(d.sum())} frames")
        from robustness import perturb
        Hp = perturb(hists["test"], "frames", 0.5, np.random.default_rng(131))
        def fwd(nm):
            ck = torch.load(T.MODELS / f"haz_{nm}.pt"); e, h = T.InteractionEncoder(), T.HazardHead(64, 4)
            e.load_state_dict(ck["encoder"]); h.load_state_dict(ck["head"]); e.eval(); h.eval()
            with torch.no_grad():
                return T.risk_of(h(e(torch.from_numpy(Hp))).numpy(), ck["head_kind"])
        exp4b = np.mean([AP(y, fwd(f"central_hazard_aug_s{s}")[Tix, 2]) - AP(y, fwd(f"central_hazard_noaug_s{s}")[Tix, 2]) for s in SEEDS])
        check("K3 H4b: 50% flicker = independent perturbation (rng 131) + forward pass of the saved models",
              np.isclose(Rw.loc["H4b", "estimate"], exp4b, atol=1e-6), f"{exp4b:+.4f}")
        exp4c = np.mean([AP(y, R(f"central_hazard_aug_s{s}")[Tix]) - AP(y, R(f"central_hazard_noaug_s{s}")[Tix]) for s in SEEDS])
        check("K4 H4c: clean AUPRC difference, tested against the non-inferiority margin -0.02",
              np.isclose(Rw.loc["H4c", "estimate"], exp4c) and Rw.loc["H4c", "delta0"] == -0.02)

        # ---------------- K5-K7 H2 ----------------
        P = pd.DataFrame(pairs).set_index(["factor", "direction"])
        ok, lrs = True, []
        for f in T.FACTORS:
            lt = bank.level(f, "test")[Tix]
            for a_, b_ in ((0, 1), (1, 0)):
                m = lt == b_
                n_ev = len(np.unique(mt.event_id.to_numpy()[Tix][m & (y == 1)]))
                tst = n_ev >= 50
                ok &= bool(P.loc[(f, f"{a_}->{b_}"), "testable"]) == tst
                if tst:
                    eo = np.mean([C.ece_adaptive(y[m], R(f"dom_{f}_{a_}_hazard_s{s}")[Tix][m]) for s in SEEDS])
                    ei = np.mean([C.ece_adaptive(y[m], R(f"dom_{f}_{b_}_hazard_s{s}")[Tix][m]) for s in SEEDS])
                    ok &= np.isclose(P.loc[(f, f"{a_}->{b_}"), "log_ratio"], np.log(eo / ei))
                    # registered statistic: mean over seeds of log ratios inside the pair
                    lrs.append(np.mean([np.log(C.ece_adaptive(y[m], R(f"dom_{f}_{a_}_hazard_s{s}")[Tix][m]) /
                                               C.ece_adaptive(y[m], R(f"dom_{f}_{b_}_hazard_s{s}")[Tix][m])) for s in SEEDS]))
        n_t = int(P.testable.sum())
        check("K5 H2: testable pairs = >= 50 held-out test events with a positive (weather level 1 too small); "
              "estimate = mean over pairs of seed-mean log ECE ratios", ok and 4 <= n_t < 8 and np.isclose(Rw.loc["H2", "estimate"], np.mean(lrs)),
              f"{n_t} of 8 testable, H2 = {np.mean(lrs):+.4f}")
        mtr = metas["train"]; okm = True
        for f, c in T.FACTORS.items():
            ltr = Tr.set_index("event_id")[c].reindex(mtr.event_id).to_numpy(float)
            big = 0 if (ltr == 0).sum() >= (ltr == 1).sum() else 1
            lt = bank.level(f, "test")[Tix]
            for a_, b_ in ((0, 1), (1, 0)):
                if not P.loc[(f, f"{a_}->{b_}"), "testable"]:
                    continue
                m = lt == b_
                mo = (lambda s: f"match_{f}_hazard_s{s}") if a_ == big else (lambda s: f"dom_{f}_{a_}_hazard_s{s}")
                mi = (lambda s: f"match_{f}_hazard_s{s}") if b_ == big else (lambda s: f"dom_{f}_{b_}_hazard_s{s}")
                eo = np.mean([C.ece_adaptive(y[m], R(mo(s))[Tix][m]) for s in SEEDS]); ei = np.mean([C.ece_adaptive(y[m], R(mi(s))[Tix][m]) for s in SEEDS])
                okm &= np.isclose(P.loc[(f, f"{a_}->{b_}"), "log_ratio_matched"], np.log(eo / ei)) and P.loc[(f, f"{a_}->{b_}"), "big_level"] == big
        check("K6 H2 size-matched: the larger level (most training samples) is replaced by its matched model in BOTH directions",
              okm and "H2_matched" in Rw.index)
        Cf = pd.DataFrame(conf); okc = len(Cf) > 0
        lvv = bank.level("lighting", "val"); ltt = bank.level("lighting", "test")[Tix]
        r_ = Cf[(Cf.factor == "lighting") & (Cf.direction == "0->1") & (Cf.alpha == 0.05)].iloc[0]
        vneg = np.flatnonzero(knv & (lvv == 0) & (yv == 0)); neg = (ltt == 1) & (y == 0)
        taus = [C.conformal_threshold(R(f"dom_lighting_0_hazard_s{s}", s="val")[vneg], 0.05) for s in SEEDS]
        fa = np.mean([(R(f"dom_lighting_0_hazard_s{s}")[Tix][neg] > t).mean() for s, t in zip(SEEDS, taus)])
        check("K7 H2 secondary: thresholds from the source level's validation negatives, false alarms on the held-out level's "
              "test negatives (seed mean), violation = cluster lower bound > alpha",
              okc and np.isclose(r_.false_alarm, fa) and bool(r_.violation) == (r_.cluster_lower > 0.05) and r_.cluster_lower <= fa <= r_.cluster_upper,
              f"FA {fa:.4f} [{r_.cluster_lower:.4f}, {r_.cluster_upper:.4f}]")

        # ---------------- K8 H3 ----------------
        def fedr(reg, s):
            z = bank.fed(reg, s); return z["test"][0][:, 2][Tix]
        sub = np.isfinite(fedr("fedavg", 131))
        e3a = np.mean([C.ece_adaptive(y[sub], fedr("fedavg", s)[sub]) - C.ece_adaptive(y[sub], fedr("fedper", s)[sub]) for s in SEEDS])
        e3b = np.mean([AP(y[sub], fedr("fedper", s)[sub]) - AP(y[sub], fedr("local", s)[sub]) for s in SEEDS])
        check("K8 H3: pooled over client test frames only (baselines in no client); ECE FedAvg - FedPer and AUPRC FedPer - local",
              np.isclose(Rw.loc["H3a", "estimate"], e3a) and np.isclose(Rw.loc["H3b2", "estimate"], e3b)
              and sub.sum() < len(Tix) and "client test frames" in Rw.loc["H3a", "note"], f"{int(sub.sum())} of {len(Tix)} frames")
        if STOP == "K8":
            raise SystemExit
        # ---------------- K9 sensitivity variants ----------------
        byname = {v["name"]: v for v in V}
        r1, _, _ = CP.endpoints(bank, byname["1 crash-only"], B, extra)
        R1 = pd.DataFrame(r1).set_index("test")
        keep = (bank.outcome("test") != "near-crash")[Tix]
        e1 = np.mean([AP(y[keep], R(f"central_hazard_aug_s{s}")[Tix][keep]) - AP(y[keep], R(f"central_hazard_noaug_s{s}")[Tix][keep]) for s in SEEDS])
        check("K9 sensitivity 1 crash-only: near-crash trips left out; H2 pairs then not testable (< 50 crash events)",
              np.isclose(R1.loc["H4c [1 crash-only]", "estimate"], e1) and R1.loc["H2 [1 crash-only]", "note"].startswith("NOT TESTABLE"))
        r3, _, _ = CP.endpoints(bank, byname["3 horizon 1 s"], B, extra); R3 = pd.DataFrame(r3).set_index("test")
        y1, kn1 = C.labels_at(mt.event_bin.to_numpy(), mt.obs_bins.to_numpy(), 0); T1 = np.flatnonzero(kn1)
        e3 = np.mean([AP(y1[T1], R(f"central_hazard_aug_s{s}", 0)[T1]) - AP(y1[T1], R(f"central_hazard_noaug_s{s}", 0)[T1]) for s in SEEDS])
        check("K10 sensitivity 3 horizon 1 s: labels and risk at 1 s (column 0)", np.isclose(R3.loc["H4c [3 horizon 1 s]", "estimate"], e3))
        r7, _, _ = CP.endpoints(bank, byname["7 baselines with >= 1 object"], B, extra); R7 = pd.DataFrame(r7).set_index("test")
        k7 = ~np.isin(mt.event_id.to_numpy(), Tr[Tr.noobj].event_id.to_numpy())[Tix]
        e7 = np.mean([AP(y[k7], R(f"central_hazard_aug_s{s}")[Tix][k7]) - AP(y[k7], R(f"central_hazard_noaug_s{s}")[Tix][k7]) for s in SEEDS])
        check("K11 sensitivity 7: baseline trips without any detected object left out (10 planted)",
              np.isclose(R7.loc["H4c [7 baselines with >= 1 object]", "estimate"], e7) and (~k7).sum() > 0)
        r5, p5, _ = CP.endpoints(bank, byname["5 equal-width ECE"], B, extra); R5 = pd.DataFrame(r5).set_index("test")
        e5 = np.mean([C.ece(y[sub], fedr("fedavg", s)[sub]) - C.ece(y[sub], fedr("fedper", s)[sub]) for s in SEEDS])
        check("K12 sensitivity 5: equal-width ECE (calibration.ece) for H2 and H3a", np.isclose(R5.loc["H3a [5 equal-width ECE]", "estimate"], e5)
              and set(R5.index) == {"H2 [5 equal-width ECE]", "H2_matched [5 equal-width ECE]", "H3a [5 equal-width ECE]"})
        r6, p6, _ = CP.endpoints(bank, byname["6 temperature"], B, extra)
        P6 = pd.DataFrame(p6).set_index(["factor", "direction"])
        f, a_, b_ = "lighting", 0, 1
        lt = bank.level(f, "test")[Tix]; m = lt == b_; lvv = bank.level(f, "val"); mv = metas["val"]
        def tcal(nm, v):
            z = bank.model(nm); fv = np.flatnonzero(lvv == v)
            Tt = C.Temperature().fit(z["val"][1][fv], mv.event_bin.to_numpy()[fv], mv.obs_bins.to_numpy()[fv])
            return Tt.transform(z["test"][1])[:, 2][Tix]
        eo = np.mean([C.ece_adaptive(y[m], tcal(f"dom_{f}_{a_}_hazard_s{s}", a_)[m]) for s in SEEDS])
        ei = np.mean([C.ece_adaptive(y[m], tcal(f"dom_{f}_{b_}_hazard_s{s}", b_)[m]) for s in SEEDS])
        check("K13 sensitivity 6 temperature: each model recalibrated on its OWN level's validation data (calibration.Temperature)",
              np.isclose(P6.loc[(f, "0->1"), "log_ratio"], np.log(eo / ei)))
        r4, _, _ = CP.endpoints(bank, byname["4 other head (sep)"], B, extra); R4 = pd.DataFrame(r4).set_index("test")
        e4 = np.mean([AP(y, R(f"central_sep_aug_s{s}")[Tix]) - AP(y, extra["lgbm"][s]["test"][Tix, 2]) for s in SEEDS])
        check("K14 sensitivity 4 other head: H1-P2 with the separate head; its H4 not testable (no separate-head no-aug models)",
              np.isclose(R4.loc["H1_P2 model-LightGBM [4 other head (sep)]", "estimate"], e4)
              and R4.loc["H4c [4 other head (sep)]", "note"].startswith("NOT TESTABLE"))

        # ---------------- K15 calibration for the separate head ----------------
        z = bank.model("central_sep_aug_s131"); mvv = metas["val"]
        rt = CP.calibrate("sep", z["val"][1], mvv.event_bin.to_numpy(), mvv.obs_bins.to_numpy(), z["test"][1], "temperature")
        ri = CP.calibrate("sep", z["val"][1], mvv.event_bin.to_numpy(), mvv.obs_bins.to_numpy(), z["test"][1], "isotonic")
        grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
        Tt = grid[int(np.argmin([T.bce_np(z["val"][1] / t, mvv.event_bin.to_numpy(), mvv.obs_bins.to_numpy()) for t in grid]))]
        check("K15 separate-head recalibration: temperature by BCE on the registered grid, then projection; isotonic monotone",
              np.allclose(rt, T.isotonic_rows(1 / (1 + np.exp(-z["test"][1] / Tt)))) and (np.diff(ri, axis=1) >= -1e-12).all())

        # ---------------- K16 not testable paths, alignment guard ----------------
        (tmp / "results/shrp2_fed_clients.json").write_text(json.dumps(dict(testable=False)))
        rn, _, _ = CP.endpoints(bank, dict(name="primary", k=2, keep=np.ones(len(mt), bool), tests=["H3a", "H3b1", "H3b2"]), B, extra)
        (tmp / "results/shrp2_fed_clients.json").write_text(json.dumps(dict(testable=True)))
        CP.MIN_EVENTS = 10 ** 6
        rq, _, _ = CP.endpoints(bank, dict(name="primary", k=2, keep=np.ones(len(mt), bool), tests=["H2"]), B, extra)
        CP.MIN_EVENTS = 50
        f_ = T.PRED / "haz_central_hazard_aug_s135.npz"; z = T.npz(f_); z["frame"] = z["frame"][::-1].copy()
        bad = tmp / "bad.npz"; np.savez(bad, **z); shutil.copy(f_, tmp / "good.npz"); shutil.copy(bad, f_)
        try:
            CP.Bank().model("central_hazard_aug_s135"); guard = False
        except SystemExit:
            guard = True
        shutil.copy(tmp / "good.npz", f_)
        check("K16 not testable is reported, never replaced: H3 when the client rule fails, H2 with < 4 testable pairs; "
              "misaligned prediction files are refused",
              all(r["note"].startswith("NOT TESTABLE") and np.isnan(r["p"]) for r in rn + rq) and guard)

        # ---------------- K17 full run, outputs, Holm, H5 input ----------------
        R_, P_, C_ = CP.run(B)
        prim = R_[R_.family == "registered"]
        check("K17 run: one row per registered P2 test in the primary analysis, sensitivity rows labelled, aggregate files written",
              sorted(prim.test) == sorted(["H2", "H3a", "H3b1", "H3b2", "H4a", "H4b", "H4c"])
              and (R_.family == "sensitivity").sum() > 10 and (tmp / "results/shrp2_confirm_p2_summary.txt").exists()
              and "event_id" not in (tmp / "results/shrp2_confirm_p2_summary.txt").read_text())
        p1 = pd.DataFrame([dict(test="H1a", estimate=0.1, ci_low=0.0, ci_high=0.2, p=0.004, note=""),
                           dict(test="H1b", estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, note="NOT TESTABLE: x")])
        p1.to_csv(tmp / "results/shrp2_confirm_p1_tests.csv", index=False)
        H = CP.holm()
        ps = np.r_[0.004, np.nan, prim.set_index("test").loc[["H2", "H3a", "H3b1", "H3b2", "H4a", "H4b", "H4c"], "p"].to_numpy()]
        adj, rej = CP.S.holm(ps)
        check("K18 Holm over the nine registered tests in the registered order; a not-testable test is never supported",
              list(H.test) == CP.REGISTERED and np.allclose(H.p_holm.to_numpy(float), adj, equal_nan=True)
              and not H.set_index("test").loc["H1b", "supported"] and not H.set_index("test").loc["H1b", "testable"]
              and np.allclose(np.array((tmp / "results/shrp2_confirm_h1h4_pvalues.txt").read_text().strip().split(","), float), ps, equal_nan=True))
        # sensitivity 8 run (C23): with SHRP2_RUN=nomask the Holm step reads the _nomask results and writes _nomask files only
        import subprocess
        for f_ in ("shrp2_confirm_p1_tests", "shrp2_confirm_p2_tests"):
            d_ = pd.read_csv(tmp / f"results/{f_}.csv"); d_.loc[d_.test == "H2", "p"] = 0.5
            d_.to_csv(tmp / f"results/{f_}_nomask.csv", index=False)
        before = (tmp / "results/shrp2_confirm_holm.csv").read_bytes()
        r_ = subprocess.run([sys.executable, str(HERE / "confirm_p2.py"), "holm"], capture_output=True, text=True,
                            env=dict(os.environ, SHRP2_RUN="nomask"))
        Hn = pd.read_csv(tmp / "results/shrp2_confirm_holm_nomask.csv") if (tmp / "results/shrp2_confirm_holm_nomask.csv").exists() else None
        check("K20 SHRP2_RUN=nomask: Holm reads the _nomask results, writes _nomask files, leaves the registered ones untouched",
              r_.returncode == 0 and Hn is not None and float(Hn.set_index("test").loc["H2", "p"]) == 0.5
              and (tmp / "results/shrp2_confirm_holm.csv").read_bytes() == before, r_.stderr[-200:])
        path = CP.h5_input(); z = T.npz(path)
        ok5 = float(z["frames_per_hour"]) == 36000.0 and len(z["unit"]) == len(metas["val"]) + len(metas["test"])
        ok5 &= np.allclose(z["score_s133"][len(metas["val"]):], R("central_hazard_aug_s133")) and set(z["split"]) == {"val", "test"}
        ok5 &= np.array_equal(z["known"][len(metas["val"]):], kn3) and np.array_equal(z["y"][len(metas["val"]):], y3)
        y5_, _ = C.labels_at(mt.event_bin.to_numpy(), mt.obs_bins.to_numpy(), 3)
        ok5 &= not np.array_equal(y5_, y3)                     # the labels at 3 s and 5 s differ here, so the check bites
        check("K19 H5 input: unit = trip, known and y at 3 s, scores = H1 models' raw risk at 3 s per seed, 36,000 frames per hour", ok5)
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
