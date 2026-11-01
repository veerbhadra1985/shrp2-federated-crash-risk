"""
H3 FEDERATED RUNNER for SHRP2 (pre-registration osf.io/y5zrp, Sections 2-6). Written and tested on synthetic data
before any SHRP2 access. Uses the cache, data reader, pretraining loop and divergence rule of train_shrp2.py.

REGISTERED DESIGN
  clients     cells of lighting x road surface x traffic density; a cell with fewer than 100 TRAINING EVENTS is merged
              by dropping factors in a fixed order (traffic density first, then lighting) until every client has at
              least 100; weather is not used
  pretraining PUBLIC DATA ONLY (INTERACTION + highD training recordings; best epoch by validation ADE), 20 epochs,
              so that no client's SHRP2 data is pooled centrally
  regimes     local, central, fedavg, fedprox (mu = 0.01), fedper (encoder averaged, hazard head kept per client);
              30 rounds x 50 local minibatch steps of 256; central takes 50 x (number of clients) steps per round;
              averaging weights proportional to client training sample size; after every round the POOLED
              validation loss is computed (each client's samples scored by that client's model) and the round with
              the lowest value is kept, for every regime alike
  seeds       131-135; test data are only scored, never used for any choice
  H3 tests    (analysis code, not here) pooled adaptive ECE at 3 s FedAvg - FedPer; pooled AUPRC at 3 s FedAvg - local
              and FedPer - local

CLARIFICATIONS (fixed here before access; deviations log C10-C13)
  C10 neighbour augmentation ON (the registered default for the model; the development H3 runs had none), and the
      risk head chosen by the registered selection (results/shrp2_head_selection.json, train_shrp2.py select-head);
      the pooled validation loss is that head's training loss (survival NLL, or per-horizon BCE)
  C11 trips with an unknown level of lighting, surface or traffic density belong to no client (registered: unknown
      categories are excluded from that factor's analyses); if safe baselines carry no environment data, H3 uses
      event trips only
  C12 merging, applied in two steps: (1) every (lighting, surface) pair with a cell below 100 training events
      becomes one client (traffic density dropped for that pair only); (2) every surface level with a client
      below 100 becomes one client (lighting dropped for that surface level only). If a client is still below
      100, or fewer than 2 clients remain, H3 is NOT TESTABLE (reported, nothing is trained)
  C13 a regime run that diverges is re-run once at half the learning rates, then with the next replacement seed
      (registered rule), logged in results/shrp2_divergence.csv
The local update, averaging and round loop are those of the development code (src/train_federated.py:
local_train, average, main), with augmentation added; test_train_federated_shrp2.py checks bit-identity with
local_train and average, and the exact equalities fedavg = central = local = fedper with one client.

Usage (after train_shrp2.py cache, run --stage A and select-head):
  python src/shrp2/train_federated_shrp2.py clients        # client definition (aggregate counts; shareable)
  python src/shrp2/train_federated_shrp2.py run            # 5 regimes x 5 seeds, resumable; --only <prefix>
Outputs: results/shrp2_fed_clients.txt, results/shrp2_fed_jobs.csv, results/shrp2_fed_rounds.csv
         models/shrp2/fed_<regime>_s<seed>.pt
         data/processed/shrp2/preds/fed_<regime>_s<seed>.npz  event_id, frame, split, client, logits, risk for every
                                                              validation and test sample of a client trip
"""
import argparse, copy, csv, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import train_shrp2 as T
from models.encoder import InteractionEncoder
from models.hazard import HazardHead, hazard_nll
from train_hazard import bce_per_horizon
from augment import augment
import calibration as C

REGIMES = ["local", "central", "fedavg", "fedprox", "fedper"]
ROUNDS, LOCAL_STEPS, BATCH, MU, MIN_EVENTS = 30, 50, 256, 0.01, 100
CELL = ["lighting", "surface", "traffic"]                 # factor names as in train_shrp2.FACTORS
CLIENTS_TXT = Path(f"results/shrp2_fed_clients{T.SFX}.txt")          # run tag from train_shrp2 (SHRP2_RUN)
FLOG, RLOG = Path(f"results/shrp2_fed_jobs{T.SFX}.csv"), Path(f"results/shrp2_fed_rounds{T.SFX}.csv")


# ================================================================================================ clients
def define_clients(trips):
    """trips: DataFrame with event_id, split, outcome_type and lighting/surface/traffic levels (0/1/NaN).
    Returns (client per trip as a Series of labels or None, report dict). Rule: module docstring C11-C12."""
    t = trips.copy()
    known = t[CELL].notna().all(axis=1)
    lab = pd.Series(None, index=t.index, dtype=object)
    key = {i: ("L%d" % t.at[i, "lighting"], "S%d" % t.at[i, "surface"], "T%d" % t.at[i, "traffic"]) for i in t.index[known]}
    train_ev = (t["split"] == "train") & (t["outcome_type"] != "baseline")

    def counts(k):
        s = pd.Series(k); return s[train_ev.reindex(s.index)].value_counts()
    # step 1: drop traffic density for (L, S) pairs with a small cell
    n = counts(key)
    small_pairs = {(a, b) for (a, b, c) in key.values() if n.get((a, b, c), 0) < MIN_EVENTS}
    key = {i: ((a, b) if (a, b) in small_pairs else (a, b, c)) for i, (a, b, c) in key.items()}
    # step 2: drop lighting for surface levels with a small client
    n = counts(key)
    small_surf = {k[1] for k in set(key.values()) if n.get(k, 0) < MIN_EVENTS}
    key = {i: ((k[1],) if k[1] in small_surf else k) for i, k in key.items()}
    n = counts(key)
    clients = sorted(set(key.values()))
    ok = len(clients) >= 2 and all(n.get(k, 0) >= MIN_EVENTS for k in clients)
    for i, k in key.items():
        lab[i] = "-".join(k)
    rep = dict(testable=ok, n_clients=len(clients), dropped_traffic=sorted("-".join(p) for p in small_pairs),
               dropped_lighting=sorted(small_surf), unknown_level_trips=int((~known).sum()),
               training_events={"-".join(k): int(n.get(k, 0)) for k in clients})
    if not ok:
        rep["reason"] = "fewer than 2 clients" if len(clients) < 2 else "a client below 100 training events after merging"
    return lab, rep


def client_table(D):
    T_ = pd.read_parquet(T.SH / "trips.parquet")
    S = pd.read_csv(T.SH / "split.csv")
    t = S.merge(T_[["event_id", "outcome_type"]], on="event_id", how="left")
    for f in CELL:
        t[f] = D.level_of(f, t["event_id"].to_numpy())
    lab, rep = define_clients(t.set_index("event_id"))
    return lab, rep


def write_clients(rep):
    lines = ["SHRP2 H3 CLIENTS (registered rule + clarifications C11-C12; aggregate counts only, shareable)",
             f"testable: {rep['testable']}" + ("" if rep["testable"] else f" ({rep['reason']})"),
             f"clients: {rep['n_clients']}; trips with an unknown level (no client): {rep['unknown_level_trips']}",
             f"traffic density dropped for: {rep['dropped_traffic'] or 'none'}; lighting dropped for: {rep['dropped_lighting'] or 'none'}"]
    lines += [f"  {k}: {v} training events" for k, v in rep["training_events"].items()]
    CLIENTS_TXT.parent.mkdir(exist_ok=True); CLIENTS_TXT.write_text("\n".join(lines) + "\n")
    CLIENTS_TXT.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print("\n".join(lines))


# ================================================================================================ training
def local_train(enc, head, X, E, O, batch, steps, mu=0.0, aug=False, rng=None, head_kind="hazard", lr_scale=1.0,
                seed=None):
    """= src/train_federated.py local_train (fixed number of steps), + optional neighbour augmentation, + the
    separate-classifier loss, + a divergence check. X: torch tensor (n, 9, 20, 9) of ONE client's training samples."""
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4 * lr_scale},
                             {"params": head.parameters(), "lr": 1e-3 * lr_scale}], weight_decay=1e-4)
    loss_fn = hazard_nll if head_kind == "hazard" else bce_per_horizon
    ref = [p.detach().clone() for p in params] if mu > 0 else None
    n = len(X)
    enc.train(); head.train()
    perm, pos = torch.randperm(n), 0
    for st in range(steps):
        if pos >= n:
            perm, pos = torch.randperm(n), 0
        i = perm[pos:pos + batch]; pos += batch
        xb = X[i]
        if aug:
            xb = torch.from_numpy(augment(xb.numpy(), rng))
        loss = loss_fn(head(enc(xb)), E[i], O[i])
        if mu > 0:
            loss = loss + mu / 2 * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
        loss = T._check(loss, "federated", seed, lr_scale, st == 0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()


def average(states, weights):
    """= src/train_federated.py average."""
    w = np.asarray(weights, float) / np.sum(weights)
    return {k: sum(wi * s[k].float() for wi, s in zip(w, states)) for k in states[0]}


def run_regime(method, enc_state, data, seed, head_kind="hazard", aug=True, rounds=ROUNDS, steps=LOCAL_STEPS,
               lr_scale=1.0, log=None):
    """One regime, the round loop of src/train_federated.py main. data = dict(clients, Xtr, Etr, Otr (per client,
    torch), val_logits_fn(models) -> (pooled val loss, per-client logits)). Returns (models_fn, state, best_round,
    best_loss, curve)."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    clients = data["clients"]
    mu = MU if method == "fedprox" else 0.0

    def new_model():
        enc = InteractionEncoder(); enc.load_state_dict(enc_state)
        return enc, HazardHead(enc.d_model, 4)
    if method == "central":
        enc, head = new_model()
        Xall = torch.cat([data["Xtr"][c] for c in clients]); Eall = torch.cat([data["Etr"][c] for c in clients])
        Oall = torch.cat([data["Otr"][c] for c in clients])
        state = lambda: {"enc": copy.deepcopy(enc.state_dict()), "head": copy.deepcopy(head.state_dict())}
        models = lambda c: (enc, head)
    elif method == "local":
        loc_models = {c: new_model() for c in clients}
        state = lambda: {c: (copy.deepcopy(m[0].state_dict()), copy.deepcopy(m[1].state_dict())) for c, m in loc_models.items()}
        models = lambda c: loc_models[c]
    else:
        g_enc, g_head = new_model(); heads = {c: copy.deepcopy(g_head) for c in clients}
        state = lambda: {"enc": copy.deepcopy(g_enc.state_dict()), "head": copy.deepcopy(g_head.state_dict()),
                         "heads": {c: copy.deepcopy(h.state_dict()) for c, h in heads.items()}}
        models = lambda c: (g_enc, heads[c] if method == "fedper" else g_head)
    best, curve = (np.inf, None, 0), []
    for r in range(1, rounds + 1):
        if method == "central":
            local_train(enc, head, Xall, Eall, Oall, BATCH, steps * len(clients), 0.0, aug, rng, head_kind, lr_scale, seed)
        elif method == "local":
            for c in clients:
                local_train(*loc_models[c], data["Xtr"][c], data["Etr"][c], data["Otr"][c], BATCH, steps, 0.0, aug, rng,
                            head_kind, lr_scale, seed)
        else:
            enc_states, head_states, w = [], [], []
            for c in clients:
                enc_c = copy.deepcopy(g_enc)
                head_c = heads[c] if method == "fedper" else copy.deepcopy(g_head)
                local_train(enc_c, head_c, data["Xtr"][c], data["Etr"][c], data["Otr"][c], BATCH, steps, mu, aug, rng,
                            head_kind, lr_scale, seed)
                enc_states.append(enc_c.state_dict()); head_states.append(head_c.state_dict()); w.append(len(data["Xtr"][c]))
            g_enc.load_state_dict(average(enc_states, w))
            if method != "fedper":
                g_head.load_state_dict(average(head_states, w))
        vloss = data["val_loss"](models)
        if not np.isfinite(vloss):
            raise T.Diverged("federated-val")
        curve.append((r, vloss))
        if vloss < best[0]:
            best = (vloss, state(), r)
        if log:
            log(r, vloss)
    s = best[1]                                         # restore the best round (validation only)
    if method == "central":
        enc.load_state_dict(s["enc"]); head.load_state_dict(s["head"])
    elif method == "local":
        for c in clients:
            loc_models[c][0].load_state_dict(s[c][0]); loc_models[c][1].load_state_dict(s[c][1])
    else:
        g_enc.load_state_dict(s["enc"]); g_head.load_state_dict(s["head"])
        for c in clients:
            heads[c].load_state_dict(s["heads"][c])
    return models, s, best[2], best[0], curve


# ================================================================================================ data per client
def client_data(D, lab, head_kind):
    """Training tensors per client (in memory) and index lists for validation and test (read in batches)."""
    G = {g: D.g[f"shrp2_{g}"]["meta"] for g in ("train", "val", "test")}
    clients = sorted(set(lab.dropna()))
    idx = {g: {c: np.flatnonzero(G[g]["event_id"].map(lab).to_numpy() == c) for c in clients} for g in G}
    data = dict(clients=clients, idx=idx, Xtr={}, Etr={}, Otr={})
    for c in clients:
        j = idx["train"][c]
        data["Xtr"][c] = torch.from_numpy(D.get("shrp2_train", j))
        e, o = D.labels("shrp2_train", j)
        data["Etr"][c], data["Otr"][c] = torch.from_numpy(np.asarray(e, np.int64)), torch.from_numpy(np.asarray(o, np.int64))
    vloss_np = C.hazard_nll_np if head_kind == "hazard" else T.bce_np

    def pooled(models, grp="val"):
        L, E, O = [], [], []
        for c in clients:
            enc, head = models(c); j = idx[grp][c]
            L.append(T.logits_of(enc, head, lambda q: D.get(f"shrp2_{grp}", q), j))
            e, o = D.labels(f"shrp2_{grp}", j); E.append(e); O.append(o)
        return L, np.concatenate(E), np.concatenate(O)

    def val_loss(models):
        L, e, o = pooled(models)
        return float(vloss_np(np.concatenate(L), e, o))
    data["val_loss"], data["pooled"] = val_loss, pooled
    return data


def save_fed(name, models, data, D, head_kind, seed_used):
    parts = []
    for grp in ("val", "test"):
        L, _, _ = data["pooled"](models, grp)
        m = D.g[f"shrp2_{grp}"]["meta"]
        for c, lg in zip(data["clients"], L):
            j = data["idx"][grp][c]
            parts.append((m["event_id"].to_numpy()[j], m["frame"].to_numpy()[j], np.full(len(j), grp), np.full(len(j), c), lg))
    ev, fr, sp, cl, lg = (np.concatenate([p[k] for p in parts]) for k in range(5))
    T.PRED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(T.PRED / f"fed_{name}.npz", event_id=ev, frame=fr, split=sp, client=cl, logits=lg.astype(np.float32),
                        risk=T.risk_of(lg, head_kind).astype(np.float32), head=head_kind, seed_used=seed_used)


# ================================================================================================ running
FED_FIELDS = ["model", "regime", "seed_slot", "seed_used", "head", "augment", "clients", "train_samples", "rounds",
              "local_steps", "best_round", "val_loss", "encoder_set", "lr_scale", "seconds", "smoke"]


def log_fed(path, row, fields):
    path.parent.mkdir(parents=True, exist_ok=True); new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader() if new else None; w.writerow(row)


def fed_done(name):
    return (T.PRED / f"fed_{name}.npz").exists() and (T.MODELS / f"fed_{name}.pt").exists()


def used_fed_seeds(regime):
    used = set()
    if T.DIVLOG.exists():
        d = pd.read_csv(T.DIVLOG)
        if len(d):
            used |= set(d.loc[d["model"].astype(str).str.rsplit("_s", n=1).str[0] == f"fed_{regime}", "seed"].astype(int))
    if FLOG.exists():
        d = pd.read_csv(FLOG)
        if len(d):
            used |= set(d.loc[d["regime"] == regime, "seed_used"].astype(int))
    return used


def run_one(D, data, regime, slot, head_kind, smoke, pre_ep, rounds, steps):
    name = f"{regime}_s{slot}" + ("_smoke" if smoke else "")
    if fed_done(name):
        print(f"  [skip] fed_{name} already done"); return
    ptr, pva = public_sets(D)
    for seed in [slot] + T.REPLACEMENT_SEEDS:
        if seed != slot and seed in used_fed_seeds(regime) - {slot}:
            continue
        try:
            for lr_scale in (1.0, 0.5):
                try:
                    enc, ade, pep, key, _ = T.encoder_for(D, ptr, pva, seed, pre_ep, smoke, lr_scale)
                    break
                except T.Diverged:
                    T.log_row(T.DIVLOG, dict(model=f"fed_{regime}_s{slot}", seed=seed, stage="pretrain", lr_scale=lr_scale))
            else:
                raise T.Diverged("pretrain twice")
            for lr_scale in (1.0, 0.5):
                try:
                    t0 = time.time(); curve = []
                    models, st, br, bl, curve = run_regime(regime, enc.state_dict(), data, seed, head_kind, True, rounds, steps,
                                                           lr_scale, log=lambda r, v: print(f"    round {r}: val loss {v:.5f}", flush=True))
                    break
                except T.Diverged:
                    T.log_row(T.DIVLOG, dict(model=f"fed_{regime}_s{slot}", seed=seed, stage="federated", lr_scale=lr_scale))
            else:
                raise T.Diverged("federated twice")
        except T.Diverged:
            T.log_row(T.DIVLOG, dict(model=f"fed_{regime}_s{slot}", seed=seed, stage="replaced", lr_scale=""))
            continue
        torch.save({"state": st, "regime": regime, "clients": data["clients"], "head_kind": head_kind, "seed_used": seed},
                   T.MODELS / f"fed_{name}.pt")
        save_fed(name, models, data, D, head_kind, seed)
        for r, v in curve:
            log_fed(RLOG, dict(model=f"fed_{name}", round=r, val_loss=round(v, 6)), ["model", "round", "val_loss"])
        log_fed(FLOG, dict(model=f"fed_{name}", regime=regime, seed_slot=slot, seed_used=seed, head=head_kind, augment=True,
                           clients=len(data["clients"]), train_samples=int(sum(len(x) for x in data["Xtr"].values())),
                           rounds=rounds, local_steps=steps, best_round=br, val_loss=round(bl, 6), encoder_set=key,
                           lr_scale=lr_scale, seconds=round(time.time() - t0), smoke=smoke), FED_FIELDS)
        print(f"  fed_{name}: best round {br}/{rounds}, pooled val loss {bl:.5f}", flush=True)
        return
    raise SystemExit(f"fed_{name}: diverged with every seed - stop and report")


def head_for(smoke):
    """The registered head choice. A real run needs the real selection (train_shrp2.chosen_head refuses a smoke file);
    a smoke run (code check) may use either, so it can be run after the real selection, as the runbook does."""
    if not smoke:
        return T.chosen_head(False)
    if not T.HEADSEL.exists():
        raise SystemExit("run train_shrp2.py select-head first (after stage A)")
    return json.loads(T.HEADSEL.read_text())["choice"]


def public_sets(D):
    """Registered H3 pretraining: public data only."""
    tr, va = [], []
    for name, lo, G in D.parts:
        if name in T.PUBLIC:
            s = G["meta"]["split"].to_numpy()
            tr.append(lo + np.flatnonzero(s == "train")); va.append(lo + np.flatnonzero(s == "val"))
    if not tr:
        raise SystemExit("no public pretraining data in the cache")
    return np.concatenate(tr), np.concatenate(va)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clients", "run"])
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true", help="code check: 1 pretraining epoch (5 batches), 2 rounds x 3 steps")
    a = ap.parse_args(argv)
    torch.set_num_threads(os.cpu_count() or 1)
    T.MODELS.mkdir(parents=True, exist_ok=True)
    D = T.Data(need_public=(a.cmd == "run"))
    lab, rep = client_table(D)
    write_clients(rep)
    if a.cmd == "clients":
        return rep
    if not rep["testable"]:
        print("H3 NOT TESTABLE under the registered client rule - nothing trained (record in the deviations log)")
        return rep
    head_kind = head_for(a.smoke)
    data = client_data(D, lab, head_kind)
    pre_ep, rounds, steps = (1, 2, 3) if a.smoke else (T.BUDGET[0], ROUNDS, LOCAL_STEPS)
    for slot in T.SEEDS:
        for regime in REGIMES:
            if not f"{regime}_s{slot}".startswith(a.only):
                continue
            print(f"\n[{regime} seed {slot}] {len(data['clients'])} clients, {rounds} rounds x {steps} steps", flush=True)
            run_one(D, data, regime, slot, head_kind, a.smoke, pre_ep, rounds, steps)


if __name__ == "__main__":
    main()
