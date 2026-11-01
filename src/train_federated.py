"""
Step 13: federated training of the hazard model, clients = INTERACTION locations.

Methods (--method):
  local     each client trains only on its own data (lower bound)
  central   all data pooled on one server (upper bound, same compute budget)
  fedavg    federated averaging of encoder + head (McMahan et al., 2017)
  fedprox   fedavg + proximal term mu/2*||w - w_global||^2 (Li et al., 2020)
  fedper    shared encoder averaged, hazard head kept LOCAL per client (personalised)
Training budget per round:
  --local-steps N   each client takes N minibatch steps per round (recommended);
                    central takes N x (number of clients) steps per round (same compute)
  (if not given)    each client runs one full local epoch per round (original setting)
Model selection: after every round the pooled VALIDATION NLL is computed; the round
with the lowest value is kept (same rule for every method). Test data is used once, at the end.
All methods use the project-wide split (recordings split within each location).
Sanity check (fedavg with one pooled client must reproduce central exactly):
    python src/train_federated.py --method fedavg  --rounds 3 --local-steps 50 --pool-clients --tag pooltest
    python src/train_federated.py --method central --rounds 3 --local-steps 50 --pool-clients --tag pooltest
Usage:
    python src/train_federated.py --method fedavg --smoke
    python src/train_federated.py --method fedavg --rounds 40 --local-steps 50 --tag s50
Outputs: results/federated_<method>[_<tag>].csv          (test, per client + pooled)
         results/federated_<method>[_<tag>]_rounds.csv   (validation curve per round)
"""
import argparse, copy, csv, os, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoder import InteractionEncoder
from models.hazard import HazardHead, cumulative_risk, hazard_nll
from train_hazard import ece
from train_pretrain import split_by_scene

HORIZONS = [1, 2, 3, 5]


def load():
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    keep = lab["current"] == 0
    X = torch.from_numpy(d["hist"][keep])
    ev, ob = lab["event_bin"][keep], lab["obs_bins"][keep]
    loc = d["location"][keep]
    tr, va, te = split_by_scene(d["scene"][keep], loc)          # project-wide split
    return X, ev, ob, loc, tr, va, te


def new_model():
    enc = InteractionEncoder()
    enc.load_state_dict(torch.load("models/encoder_pretrained.pt", map_location="cpu")["encoder"])
    return enc, HazardHead(enc.d_model, len(HORIZONS))


def local_train(enc, head, X, E, O, batch, steps=None, epochs=1, smoke=False, mu=0.0):
    """Train for a fixed number of minibatch steps, or for full epochs if steps is None."""
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    ref = [p.detach().clone() for p in params] if mu > 0 else None
    n = len(X)
    if steps is None:
        steps = epochs * int(np.ceil(n / batch))
    if smoke:
        steps = min(steps, 3)
    enc.train(); head.train()
    perm, pos = torch.randperm(n), 0
    for _ in range(steps):
        if pos >= n:
            perm, pos = torch.randperm(n), 0
        i = perm[pos:pos + batch]; pos += batch
        loss = hazard_nll(head(enc(X[i])), E[i], O[i])
        if mu > 0:
            loss = loss + mu / 2 * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()


def average(states, weights):
    w = np.asarray(weights, float) / np.sum(weights)
    return {k: sum(wi * s[k].float() for wi, s in zip(w, states)) for k in states[0]}


def predict(enc, head, X, bs=1024):
    enc.eval(); head.eval(); L = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            L.append(head(enc(X[s:s + bs])))
    return torch.cat(L) if L else torch.zeros(0, len(HORIZONS))


def safe_metrics(risk, ev, ob):
    out = {}
    for k, h in enumerate(HORIZONS):
        y = (ev >= 0) & (ev <= k); ok = y | (ob >= k + 1)
        yk, pk = y[ok].astype(int), risk[ok, k]
        both = 0 < yk.sum() < len(yk)
        out[f"pos_{h}s"] = int(yk.sum())
        out[f"AUPRC_{h}s"] = average_precision_score(yk, pk) if both else np.nan
        out[f"AUROC_{h}s"] = roc_auc_score(yk, pk) if both else np.nan
        out[f"ECE_{h}s"] = ece(yk, pk) if len(yk) else np.nan
    return out


def evaluate(models, X, ev, ob, idx):
    """Pooled NLL and metrics over all clients, each client scored by its own model."""
    logits, EV, OB, per = [], [], [], {}
    for c, ii in idx.items():
        enc, head = models(c)
        lg = predict(enc, head, X[ii]); logits.append(lg); EV.append(ev[ii]); OB.append(ob[ii])
        per[c] = cumulative_risk(lg).numpy()
    lg = torch.cat(logits); e = np.concatenate(EV); o = np.concatenate(OB)
    nll = hazard_nll(lg, torch.from_numpy(e), torch.from_numpy(o)).item()
    return nll, safe_metrics(cumulative_risk(lg).numpy(), e, o), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["local", "central", "fedavg", "fedprox", "fedper"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--local-steps", type=int, default=None)
    ap.add_argument("--mu", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=131, help="training seed (Step 13 used 131)")
    ap.add_argument("--save-preds", action="store_true",
                    help="save per-sample TEST risk to data/processed/preds/fed_<method>_s<seed>.npz (H3 rehearsal)")
    ap.add_argument("--pool-clients", action="store_true",
                    help="sanity check: treat all locations as ONE client (fedavg must then equal central)")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(os.cpu_count() or 1)
    X, ev, ob, loc, tr, va, te = load()
    if a.pool_clients:
        loc = np.full(len(loc), "ALL_POOLED")
    E, O = torch.from_numpy(ev), torch.from_numpy(ob)
    clients = sorted(np.unique(loc))
    idx_tr = {c: np.where(tr & (loc == c))[0] for c in clients}
    idx_va = {c: np.where(va & (loc == c))[0] for c in clients}
    idx_te = {c: np.where(te & (loc == c))[0] for c in clients}
    print(f"method {a.method} | {len(clients)} clients | local steps/round: "
          f"{a.local_steps if a.local_steps else 'one full epoch'}")
    for c in clients:
        print(f"  {c:26} train {len(idx_tr[c]):6d} | val {len(idx_va[c]):5d} | test {len(idx_te[c]):5d}")
    empty = [c for c in clients if min(len(idx_tr[c]), len(idx_va[c]), len(idx_te[c])) == 0]
    if empty:
        raise SystemExit(f"Clients without train/val/test data: {empty}")
    rounds = 2 if a.smoke else a.rounds
    mu = a.mu if a.method == "fedprox" else 0.0
    name = f"federated_{a.method}" + (f"_{a.tag}" if a.tag else "")
    t0, best, curve = time.time(), (np.inf, None, 0), []

    if a.method == "central":
        enc, head = new_model(); all_tr = np.concatenate([idx_tr[c] for c in clients])
        state = lambda: {"enc": copy.deepcopy(enc.state_dict()), "head": copy.deepcopy(head.state_dict())}
        models = lambda c: (enc, head)
    elif a.method == "local":
        loc_models = {c: new_model() for c in clients}
        state = lambda: {c: (copy.deepcopy(m[0].state_dict()), copy.deepcopy(m[1].state_dict()))
                         for c, m in loc_models.items()}
        models = lambda c: loc_models[c]
    else:
        g_enc, g_head = new_model(); heads = {c: copy.deepcopy(g_head) for c in clients}
        state = lambda: {"enc": copy.deepcopy(g_enc.state_dict()), "head": copy.deepcopy(g_head.state_dict()),
                         "heads": {c: copy.deepcopy(h.state_dict()) for c, h in heads.items()}}
        models = lambda c: (g_enc, heads[c] if a.method == "fedper" else g_head)

    for r in range(1, rounds + 1):
        if a.method == "central":
            steps = a.local_steps * len(clients) if a.local_steps else None
            local_train(enc, head, X[all_tr], E[all_tr], O[all_tr], a.batch, steps, smoke=a.smoke)
        elif a.method == "local":
            for c in clients:
                local_train(*loc_models[c], X[idx_tr[c]], E[idx_tr[c]], O[idx_tr[c]], a.batch,
                            a.local_steps, smoke=a.smoke)
        else:
            enc_states, head_states, w = [], [], []
            for c in clients:
                enc_c = copy.deepcopy(g_enc)
                head_c = heads[c] if a.method == "fedper" else copy.deepcopy(g_head)
                local_train(enc_c, head_c, X[idx_tr[c]], E[idx_tr[c]], O[idx_tr[c]], a.batch,
                            a.local_steps, smoke=a.smoke, mu=mu)
                enc_states.append(enc_c.state_dict()); head_states.append(head_c.state_dict())
                w.append(len(idx_tr[c]))
            g_enc.load_state_dict(average(enc_states, w))
            if a.method != "fedper":
                g_head.load_state_dict(average(head_states, w))
        vnll, vm, _ = evaluate(models, X, ev, ob, idx_va)
        curve.append((r, vnll, vm["AUPRC_3s"], vm["ECE_3s"]))
        flag = ""
        if vnll < best[0]:
            best = (vnll, state(), r); flag = "  <- best"
        print(f"  round {r}/{rounds} | val NLL {vnll:.4f} | val AUPRC@3s {vm['AUPRC_3s']:.3f} "
              f"| val ECE@3s {vm['ECE_3s']:.4f} | {time.time()-t0:.0f}s{flag}", flush=True)

    # restore the best round (chosen on validation only)
    s = best[1]
    if a.method == "central":
        enc.load_state_dict(s["enc"]); head.load_state_dict(s["head"])
    elif a.method == "local":
        for c in clients:
            loc_models[c][0].load_state_dict(s[c][0]); loc_models[c][1].load_state_dict(s[c][1])
    else:
        g_enc.load_state_dict(s["enc"]); g_head.load_state_dict(s["head"])
        for c in clients:
            heads[c].load_state_dict(s["heads"][c])

    _, pooled, per = evaluate(models, X, ev, ob, idx_te)
    if a.save_preds:                                 # added 9 Oct 2026 for the H3 rehearsal
        full = np.flatnonzero(np.load("data/processed/hazard_labels.npz")["current"] == 0)   # keep -> full index
        Path("data/processed/preds").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f"data/processed/preds/fed_{a.method}_s{a.seed}.npz",
                            idx=np.concatenate([full[idx_te[c]] for c in clients]),
                            risk=np.concatenate([per[c] for c in clients]).astype(np.float32),
                            client=np.concatenate([[c] * len(idx_te[c]) for c in clients]))
    rows = [{"client": c, "n_test": len(idx_te[c]), **safe_metrics(per[c], ev[idx_te[c]], ob[idx_te[c]])}
            for c in clients]
    rows.append({"client": "POOLED", "n_test": int(sum(len(v) for v in idx_te.values())), **pooled})
    mono = all(bool((np.diff(p, axis=1) >= -1e-7).all()) for p in per.values())
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{name}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    with open(f"results/{name}_rounds.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["round", "val_nll", "val_auprc_3s", "val_ece_3s"]); wr.writerows(curve)
    print(f"\nbest round (by validation NLL): {best[2]} of {rounds}")
    print(f"TEST by client ({a.method}{' ' + a.tag if a.tag else ''}) | horizon consistency: {mono}")
    print(f"{'client':26} {'n':>6} {'pos3s':>5} {'AUPRC1s':>7} {'AUPRC3s':>7} {'AUPRC5s':>7} {'ECE3s':>6}")
    f = lambda v: "   n/a" if np.isnan(v) else f"{v:7.3f}"
    for r in rows:
        print(f"{r['client']:26} {r['n_test']:6d} {r['pos_3s']:5d} {f(r['AUPRC_1s'])} {f(r['AUPRC_3s'])} "
              f"{f(r['AUPRC_5s'])} {r['ECE_3s']:6.4f}")
    macro = np.nanmean([r["AUPRC_3s"] for r in rows[:-1]])
    print(f"\nmean of client AUPRC@3s: {macro:.3f} | total time {time.time()-t0:.0f}s | saved results/{name}.csv")


if __name__ == "__main__":
    main()