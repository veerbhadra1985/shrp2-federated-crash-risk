"""
Tests of the H3 federated runner (src/shrp2/train_federated_shrp2.py) on synthetic SHRP2-format data and synthetic
public data. No SHRP2 data. Known answers: hand-built client tables, the development code (src/train_federated.py)
for bit-identity, and exact equalities that must hold with one client.
Usage : python src/shrp2/test_train_federated_shrp2.py      (~4-8 min, temporary folder only)
"""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "loaders"))
import make_synthetic as M
import shrp2 as LD
import make_split as MS
import build_samples_shrp2 as BS
import train_shrp2 as T
import train_federated_shrp2 as FD
from test_train_shrp2 import public_data, npz, open_npz_files
RES = []
STOP = os.environ.get("TEST_STOP_AFTER", "")            # mutation runs only


class _Stop(Exception):
    pass


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def table(cells, extra=()):
    """cells: {(L, S, T): number of TRAINING event trips}; extra: rows (lighting, surface, traffic, split, outcome)."""
    rows, eid = [], 1
    for (l, s, t), n in cells.items():
        for _ in range(n):
            rows.append(dict(event_id=eid, split="train", outcome_type="near-crash", lighting=l, surface=s, traffic=t)); eid += 1
    for l, s, t, sp, oc in extra:
        rows.append(dict(event_id=eid, split=sp, outcome_type=oc, lighting=l, surface=s, traffic=t)); eid += 1
    return pd.DataFrame(rows).set_index("event_id")


def same_state(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fedshrp2_")); cwd = os.getcwd()
    try:
        # ---------------- F1 client rule, hand cases (registered 100 training events) ----------------
        full = {(l, s, t): 120 for l in (0, 1) for s in (0, 1) for t in (0, 1)}
        lab, rep = FD.define_clients(table(full))
        check("F1 every cell >= 100 training events -> 8 clients, nothing merged",
              rep["testable"] and rep["n_clients"] == 8 and not rep["dropped_traffic"] and not rep["dropped_lighting"])
        cx = dict(full); cx[(0, 0, 1)] = 100
        lab, rep = FD.define_clients(table(cx))
        check("F1 a cell with exactly 100 training events stays a client (rule: fewer than 100 is merged)",
              rep["n_clients"] == 8 and not rep["dropped_traffic"])
        c2 = dict(full); c2[(0, 0, 1)] = 50
        lab, rep = FD.define_clients(table(c2))
        check("F1 one small cell -> traffic density dropped for that (lighting, surface) pair only -> 7 clients",
              rep["n_clients"] == 7 and rep["dropped_traffic"] == ["L0-S0"] and rep["training_events"]["L0-S0"] == 170
              and rep["training_events"]["L1-S0-T1"] == 120 and rep["testable"])
        c3 = dict(full); c3[(1, 1, 0)] = 30; c3[(1, 1, 1)] = 40
        lab, rep = FD.define_clients(table(c3))
        check("F1 merged pair still below 100 -> lighting dropped for that surface level -> client S1 (310 events)",
              rep["dropped_lighting"] == ["S1"] and rep["training_events"].get("S1") == 120 + 120 + 70
              and rep["n_clients"] == 5 and rep["testable"], str(rep["training_events"]))
        c4 = {(l, s, t): 20 for l in (0, 1) for s in (0, 1) for t in (0, 1)}
        lab, rep = FD.define_clients(table(c4))
        check("F1 all small: S0 = S1 = 80 < 100 -> NOT TESTABLE", not rep["testable"] and rep["n_clients"] == 2)
        c5 = {(0, 0, 0): 300, (0, 0, 1): 10}
        lab, rep = FD.define_clients(table(c5))
        check("F1 fewer than 2 clients after merging -> NOT TESTABLE", not rep["testable"] and rep["n_clients"] == 1)
        tb = table(full, extra=[(np.nan, 0, 0, "train", "near-crash")] * 30 + [(0, 0, 0, "val", "crash")] * 500
                   + [(1, 1, 1, "train", "baseline")] * 500)
        c6 = {k: v for k, v in full.items()}; c6[(0, 1, 0)] = 90
        tb6 = table(c6, extra=[(0, 1, 0, "val", "near-crash")] * 50 + [(0, 1, 0, "train", "baseline")] * 50)
        lab, rep = FD.define_clients(tb)
        lab6, rep6 = FD.define_clients(tb6)
        check("F1 unknown-level trips get no client; validation trips and baselines do not count toward the 100",
              rep["unknown_level_trips"] == 30 and lab[tb.lighting.isna()].isna().all() and rep["n_clients"] == 8
              and rep6["dropped_traffic"] == ["L0-S1"] and lab6[(tb6.split == "val") & (tb6.surface == 1) & (tb6.lighting == 0)].eq("L0-S1").all())

        if STOP == "F1":
            raise _Stop
        # ---------------- F3 local update and averaging = development code ----------------
        import train_federated as DEV
        g = torch.Generator().manual_seed(1)
        X = torch.randn(300, 9, 20, 9, generator=g); X[..., 8] = 1.0
        E = torch.randint(-1, 4, (300,), generator=g); O = torch.full((300,), 4)
        out = []
        for impl in ("dev", "runner"):
            torch.manual_seed(5); enc, head = FD.InteractionEncoder(), FD.HazardHead(64, 4)
            torch.manual_seed(9)
            if impl == "dev":
                DEV.local_train(enc, head, X, E, O, 64, steps=7, mu=0.01)
            else:
                FD.local_train(enc, head, X, E, O, 64, 7, mu=0.01)
            out.append((enc.state_dict(), head.state_dict()))
        check("F3 local update (incl. FedProx term) = development local_train, bit for bit (steps wrap the data)",
              same_state(out[0][0], out[1][0]) and same_state(out[0][1], out[1][1]))
        s1 = {"w": torch.tensor([1.0, 2.0])}; s3 = {"w": torch.tensor([3.0, 6.0])}
        check("F6 averaging weighted by client sample size: (1 x 1 + 3 x 3) / 4 = 2.5",
              torch.allclose(FD.average([s1, s3], [1, 3])["w"], torch.tensor([2.5, 5.0]))
              and same_state(FD.average([out[0][0], out[1][0]], [2, 5]), DEV.average([out[0][0], out[1][0]], [2, 5])))

        # ---------------- synthetic SHRP2 + public data ----------------
        os.chdir(tmp); rng = np.random.default_rng(7); (tmp / "data/processed").mkdir(parents=True)
        root = M.build(tmp / "syn", "complete", scale=6)
        work = tmp / "data/processed/shrp2"
        LD.main(root, work, mask=True); MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        BS.main(work, "asgiven", "car", res_dir=tmp / "res")
        public_data(tmp, rng); T.build_cache()
        D = T.Data()

        # ---------------- F9 registered threshold on the small synthetic set -> not testable, nothing trained ----------------
        T.HEADSEL.parent.mkdir(exist_ok=True)
        T.HEADSEL.write_text(json.dumps(dict(choice="hazard", smoke=True)))
        rep9 = FD.main(["run", "--smoke"])
        nothing = not (T.PRED.exists() and list(T.PRED.glob("fed_*"))) and not list(T.MODELS.glob("enc_*"))
        check("F9 fewer than 100 training events per client -> H3 NOT TESTABLE, reported, nothing trained",
              not rep9["testable"] and "testable: False" in FD.CLIENTS_TXT.read_text() and nothing)
        FD.MIN_EVENTS = 8                                          # small enough for the synthetic set
        lab, rep = FD.client_table(D)
        check("F9 synthetic clients at a threshold of 8 (for the runs below)", rep["testable"] and rep["n_clients"] >= 2,
              str(rep["training_events"]))

        try:
            FD.main(["run"]); refused = False                       # smoke selection file, real run
        except SystemExit:
            refused = True
        T.HEADSEL.write_text(json.dumps(dict(choice="hazard", smoke=False)))
        check("F13 head choice: a real run refuses a smoke selection; a smoke check may use the real one",
              refused and FD.head_for(True) == "hazard" and FD.head_for(False) == "hazard" and not list(T.MODELS.glob("fed_*")))
        T.HEADSEL.write_text(json.dumps(dict(choice="hazard", smoke=True)))
        # ---------------- F2 public-only pretraining ----------------
        ptr, pva = FD.public_sets(D)
        lo = {p[0]: p[1] for p in D.parts}
        check("F2 H3 pretraining = public training/validation recordings only (no SHRP2 trip at all)",
              ptr.max() < lo["shrp2_train"] and pva.max() < lo["shrp2_train"]
              and len(ptr) == sum(int((G["meta"]["split"] == "train").sum()) for n, _, G in D.parts if n in T.PUBLIC))

        # ---------------- F4 exact one-client equalities; F7 step counts; F10 augmentation ----------------
        data = FD.client_data(D, lab, "hazard")
        one = dict(data); c0 = data["clients"]
        one["clients"] = ["ALL"]
        one["Xtr"] = {"ALL": torch.cat([data["Xtr"][c] for c in c0])}; one["Etr"] = {"ALL": torch.cat([data["Etr"][c] for c in c0])}
        one["Otr"] = {"ALL": torch.cat([data["Otr"][c] for c in c0])}
        one["idx"] = {g: {"ALL": np.concatenate([data["idx"][g][c] for c in c0])} for g in data["idx"]}
        torch.manual_seed(0); enc0 = FD.InteractionEncoder().state_dict()
        def vl(models):                                            # pooled validation loss of the one client
            enc, head = models("ALL"); j = one["idx"]["val"]["ALL"]
            lg = T.logits_of(enc, head, lambda q: D.get("shrp2_val", q), j); e, o = D.labels("shrp2_val", j)
            return float(T.C.hazard_nll_np(lg, e, o))
        one["val_loss"] = vl
        res = {m: FD.run_regime(m, enc0, one, 131, "hazard", True, 3, 4) for m in ("central", "fedavg", "local", "fedper")}
        ref = res["central"][0]("ALL")
        eq = all(same_state(res[m][0]("ALL")[0].state_dict(), ref[0].state_dict()) and same_state(res[m][0]("ALL")[1].state_dict(), ref[1].state_dict())
                 and res[m][2:4] == res["central"][2:4] for m in res)
        check("F4 one client: FedAvg = central = local = FedPer, bit for bit (same best round and loss)", eq,
              f"best round {res['central'][2]}, loss {res['central'][3]:.5f}")
        mu0 = FD.MU; FD.MU = 0.0
        rp0 = FD.run_regime("fedprox", enc0, one, 131, "hazard", True, 3, 4)
        FD.MU = mu0
        rp = FD.run_regime("fedprox", enc0, one, 131, "hazard", True, 3, 4)
        check("F4 FedProx with mu = 0 equals FedAvg; with mu = 0.01 it differs (the proximal term acts)",
              same_state(rp0[0]("ALL")[0].state_dict(), ref[0].state_dict())
              and not same_state(rp[0]("ALL")[0].state_dict(), ref[0].state_dict()))
        calls, real_lt = [], FD.local_train

        def spy(enc, head, X, E, O, batch, steps, *a, **k):
            calls.append((len(X), steps)); return real_lt(enc, head, X, E, O, batch, steps, *a, **k)
        FD.local_train = spy
        try:
            calls.clear(); FD.run_regime("central", enc0, data, 131, "hazard", True, 2, 4); cc = list(calls)
            calls.clear(); FD.run_regime("fedavg", enc0, data, 131, "hazard", True, 2, 4); cf = list(calls)
        finally:
            FD.local_train = real_lt
        AV, real_av = [], FD.average
        FD.average = lambda states, w: (AV.append(list(w)), real_av(states, w))[1]
        try:
            FD.run_regime("fedavg", enc0, data, 131, "hazard", True, 1, 2)
        finally:
            FD.average = real_av
        k = len(data["clients"]); sizes = [len(data["Xtr"][c]) for c in data["clients"]]
        check("F6 in a FedAvg round the encoder and head are averaged with weights = client training sample sizes",
              AV == [sizes, sizes] and len(set(sizes)) > 1, str(sizes))
        check("F7 central: 4 x (number of clients) steps per round on the pooled data; FedAvg: 4 steps per client per round",
              cc == [(sum(sizes), 4 * k)] * 2 and cf == [(n, 4) for n in sizes] * 2, f"{k} clients")
        ra = FD.run_regime("fedavg", enc0, data, 131, "hazard", True, 2, 4)
        rb = FD.run_regime("fedavg", enc0, data, 131, "hazard", False, 2, 4)
        rc = FD.run_regime("fedavg", enc0, data, 131, "hazard", True, 2, 4)
        c = data["clients"][0]
        check("F10 neighbour augmentation is applied (changes the model) and is reproducible (same seed, same model)",
              not same_state(ra[0](c)[0].state_dict(), rb[0](c)[0].state_dict()) and same_state(ra[0](c)[0].state_dict(), rc[0](c)[0].state_dict()))
        # best round = lowest pooled validation loss; restored model reproduces it
        rr = FD.run_regime("fedper", enc0, data, 132, "hazard", True, 4, 3)
        curve = dict(rr[4]); br = min(curve, key=curve.get)
        check("F5 restored models reproduce the logged pooled validation loss of the best round",
              rr[2] == br and np.isclose(data["val_loss"](rr[0]), curve[br], rtol=1e-9), f"round {br} of 4")
        # best round before the last: a scripted validation loss 3, 1, 2, 4 -> round 2 must be restored, and the
        # restored models must equal those of the same run stopped after 2 rounds (training is deterministic)
        scripted = dict(data); seq = iter([3.0, 1.0, 2.0, 4.0]); scripted["val_loss"] = lambda m: next(seq)
        r4 = FD.run_regime("fedper", enc0, scripted, 132, "hazard", True, 4, 3)
        seq2 = iter([3.0, 1.0]); scripted["val_loss"] = lambda m: next(seq2)
        r2 = FD.run_regime("fedper", enc0, scripted, 132, "hazard", True, 2, 3)
        same = all(same_state(r4[0](c)[0].state_dict(), r2[0](c)[0].state_dict()) and same_state(r4[0](c)[1].state_dict(), r2[0](c)[1].state_dict())
                   for c in data["clients"])
        check("F5 best round BEFORE the last (scripted losses 3, 1, 2, 4): round 2 is restored, models = the 2-round run",
              r4[2] == 2 and r4[3] == 1.0 and same)

        # local: a client's model depends on its own data only (another client's data changed -> same model)
        alt = dict(data); alt["Xtr"] = dict(data["Xtr"]); cb = data["clients"][-1]
        alt["Xtr"][cb] = data["Xtr"][cb] + 0.5                     # same size, different values
        la = FD.run_regime("local", enc0, data, 131, "hazard", True, 2, 3)
        lb = FD.run_regime("local", enc0, alt, 131, "hazard", True, 2, 3)
        ca = data["clients"][0]
        check("F11 local: each client trains on its own data only (changing another client's data leaves it unchanged)",
              same_state(la[0](ca)[0].state_dict(), lb[0](ca)[0].state_dict()) and not same_state(la[0](cb)[0].state_dict(), lb[0](cb)[0].state_dict()))
        # separate-classifier head (if the registered selection picks it): BCE loss for training and for the round choice
        ds = FD.client_data(D, lab, "sep")
        bce_calls, real_bce = [], FD.bce_per_horizon
        FD.bce_per_horizon = lambda *a: (bce_calls.append(1), real_bce(*a))[1]
        try:
            rs = FD.run_regime("fedavg", enc0, ds, 131, "sep", True, 1, 2)
        finally:
            FD.bce_per_horizon = real_bce
        Ls, es, os_ = ds["pooled"](rs[0])
        check("F12 separate head: trained with the per-horizon BCE; round chosen by the pooled validation BCE",
              len(bce_calls) == 2 * len(ds["clients"]) and np.isclose(rs[3], T.bce_np(np.concatenate(Ls), es, os_), rtol=1e-12))
        if STOP == "F10":
            raise _Stop
        # ---------------- F5 end-to-end smoke run ----------------
        FD.main(["run", "--smoke"])
        L = pd.read_csv(FD.FLOG)
        check("F5 smoke run: 5 regimes x 5 seeds, one public-only encoder per seed shared by all regimes",
              len(L) == 25 and L.groupby("seed_slot")["encoder_set"].nunique().eq(1).all()
              and set(L.encoder_set) == {T.set_key(ptr, pva)} and (L["head"] == "hazard").all())
        ckc = torch.load(T.MODELS / "fed_central_s131_smoke.pt")
        enc = FD.InteractionEncoder(); enc.load_state_dict(ckc["state"]["enc"]); head = FD.HazardHead(64, 4); head.load_state_dict(ckc["state"]["head"])
        mvm = D.g["shrp2_val"]["meta"]; jv = np.flatnonzero(mvm["event_id"].map(lab).notna().to_numpy())
        lgv = T.logits_of(enc, head, lambda q: D.get("shrp2_val", q), jv)
        ind = T.C.hazard_nll_np(lgv, mvm["event_bin"].to_numpy()[jv], mvm["obs_bins"].to_numpy()[jv])
        logged = float(L[L.model == "fed_central_s131_smoke"].val_loss.iloc[0])
        check("F5 logged pooled validation loss = independent recomputation on the validation samples of all client trips",
              np.isclose(ind, logged, atol=1e-5), f"{ind:.6f} vs {logged:.6f}")
        z = npz(T.PRED / "fed_fedper_s131_smoke.npz")
        nv = sum(len(data["idx"]["val"][c]) for c in data["clients"]); nt = sum(len(data["idx"]["test"][c]) for c in data["clients"])
        keys = pd.DataFrame(dict(e=z["event_id"], f=z["frame"]))
        mv = D.g["shrp2_val"]["meta"]
        ok_cov = len(keys) == nv + nt and not keys.duplicated().any() and set(z["client"]) == set(data["clients"])
        ok_cl = all(lab.get(e) == c for e, c in zip(z["event_id"][:200], z["client"][:200]))
        check("F5 predictions: every validation and test sample of a client trip exactly once, tagged with its client",
              ok_cov and ok_cl, f"{nv} val + {nt} test")
        ck = torch.load(T.MODELS / "fed_fedper_s131_smoke.pt")
        st = ck["state"]; ok = True; heads_differ = False
        for c in data["clients"]:
            enc = FD.InteractionEncoder(); enc.load_state_dict(st["enc"]); head = FD.HazardHead(64, 4); head.load_state_dict(st["heads"][c])
            m = (z["client"] == c) & (z["split"] == "val")
            j = data["idx"]["val"][c][:5]
            lg = T.logits_of(enc, head, lambda q: D.get("shrp2_val", q), j)
            ok &= np.allclose(z["logits"][m][:5], lg, atol=1e-6)
        h0, h1 = (st["heads"][c]["mlp.2.bias"] for c in data["clients"][:2])
        heads_differ = not torch.equal(h0, h1)
        check("F5 FedPer: each client's samples scored by the shared encoder + THAT client's head (heads differ)", ok and heads_differ)
        f_ = T.PRED / "fed_local_s133_smoke.npz"; b0 = f_.read_bytes()
        FD.main(["run", "--smoke"])
        check("F5 resume: a second run skips finished models and changes nothing", f_.read_bytes() == b0 and len(pd.read_csv(FD.FLOG)) == 25)
        r0 = npz(f_)["risk"]; f_.unlink(); (T.MODELS / "fed_local_s133_smoke.pt").unlink()
        FD.main(["run", "--smoke", "--only", "local_s133"])
        check("F5 refit of a deleted model reproduces its predictions exactly", np.array_equal(npz(f_)["risk"], r0))

        leaks = open_npz_files()
        check("F5 no .npz file left open (Windows cannot delete or replace an open file)", not leaks,
              f"{len(leaks)} open" if leaks else ("checked via /proc" if Path("/proc/self/fd").exists() else "not checkable on this system"))
        # ---------------- F8 divergence rule ----------------
        for p in (T.PRED / "fed_fedavg_s132_smoke.npz", T.MODELS / "fed_fedavg_s132_smoke.pt"):
            p.unlink()
        T.FORCE_DIVERGE.clear(); T.FORCE_DIVERGE.add(("federated", 132, 1.0))
        FD.main(["run", "--smoke", "--only", "fedavg_s132"])
        last = pd.read_csv(FD.FLOG).iloc[-1]; dv = pd.read_csv(T.DIVLOG)
        check("F8 non-finite loss -> regime re-run once at half the learning rates (logged)",
              float(last.lr_scale) == 0.5 and int(last.seed_used) == 132 and (dv.stage == "federated").sum() == 1)
        for p in (T.PRED / "fed_fedavg_s132_smoke.npz", T.MODELS / "fed_fedavg_s132_smoke.pt"):
            p.unlink()
        T.FORCE_DIVERGE.add(("federated", 132, 0.5))
        FD.main(["run", "--smoke", "--only", "fedavg_s132"])
        last = pd.read_csv(FD.FLOG).iloc[-1]
        check("F8 failing twice -> replaced by the next seed (136), recorded", int(last.seed_used) == 136
              and int(npz(T.PRED / "fed_fedavg_s132_smoke.npz")["seed_used"]) == 136)
        T.FORCE_DIVERGE.clear()
        import subprocess
        out = subprocess.run([sys.executable, "-c", f"import sys; sys.path[:0]=[r'{HERE}', r'{HERE.parent}']; import train_federated_shrp2 as F; "
                              "print(F.CLIENTS_TXT.as_posix(), F.FLOG.as_posix(), F.RLOG.as_posix(), F.T.MODELS.as_posix(), F.T.SH.as_posix())"],
                             capture_output=True, text=True, env=dict(os.environ, SHRP2_RUN="nomask")).stdout.split()
        check("F14 SHRP2_RUN=nomask (sensitivity 8): client file, logs, models and data of the federated run carry the tag",
              out == ["results/shrp2_fed_clients_nomask.txt", "results/shrp2_fed_jobs_nomask.csv", "results/shrp2_fed_rounds_nomask.csv",
                      "models/shrp2_nomask", "data/processed/shrp2_nomask"], " ".join(out))
    except _Stop:
        print(f"(stopped after {STOP}: mutation run)")
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
