"""
Step 18: continual learning across road types (stand-in task on INTERACTION).

Stages arrive in sequence:  1 Intersection -> 2 Roundabout -> 3 Merging
(Merges have almost no test conflicts, so they come last; any stage with fewer than
MIN_EVENTS test events is reported but EXCLUDED from the summary measures.)
Methods:
  naive   train on each new stage only (fine-tuning)
  replay  as naive, plus a fixed buffer of REPLAY_FRAC of each earlier stage's training data
  joint   all stages at once (upper bound; same total number of training epochs)
The encoder is pretrained ONLY on stage-1 data, so later stages are genuinely new.
After every stage each model is evaluated on the TEST data of all stages -> matrix R[i, j]
(performance on stage j after training stage i) for AUPRC@3s and ECE@3s.
Summary measures (Lopez-Paz & Ranzato style; verify citation before use):
  final average  mean_j R[T, j]
  forgetting     mean_{j<T} (R[T, j] - R[j, j])   (negative = forgetting)
  zero-shot      R[j-1, j] for j>1: performance on a stage BEFORE training on it
Usage : python src/continual.py [--epochs 3] [--smoke]
Output: results/continual_matrix.csv, results/continual_summary.csv
"""
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np

STAGES = ["Intersection", "Roundabout", "Merging"]
MIN_EVENTS = 20
REPLAY_FRAC = 0.10


def road_type(loc):
    return next(s for s in STAGES if s in loc)


def cl_summary(R, valid=None, sequential=True):
    """R: (T, T) matrix, R[i, j] = metric on stage j after training stage i.
    valid: boolean per stage (enough test events); invalid stages are excluded.
    sequential=False (joint training): forgetting and zero-shot are not defined."""
    T = R.shape[0]; valid = np.ones(T, bool) if valid is None else np.asarray(valid)
    out = dict(final_average=float(np.nanmean(R[-1][valid])))
    f = [R[-1, j] - R[j, j] for j in range(T - 1) if valid[j]]
    z = [R[j - 1, j] for j in range(1, T) if valid[j]]
    out["forgetting"] = float(np.mean(f)) if (sequential and f) else float("nan")
    out["zero_shot"] = float(np.mean(z)) if (sequential and z) else float("nan")
    return out


def main():
    import torch
    from sklearn.metrics import average_precision_score
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.encoder import InteractionEncoder
    from models.hazard import HazardHead, hazard_nll, cumulative_risk
    from train_pretrain import split_by_scene
    from lolo_calibration import pretrain
    from augment import augment
    import calibration as C
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    X = torch.from_numpy(d["hist"]); Y = torch.from_numpy(d["fut"]); V = torch.from_numpy(d["fut_valid"])
    ev, ob, keep = lab["event_bin"], lab["obs_bins"], lab["current"] == 0
    E, O = torch.from_numpy(ev), torch.from_numpy(ob)
    loc = d["location"]; tr, va, te = split_by_scene(d["scene"], loc)
    stage = np.array([STAGES.index(road_type(l)) for l in loc])
    tr_idx = [np.where(tr & keep & (stage == s))[0] for s in range(3)]
    te_idx = [np.where(te & keep & (stage == s))[0] for s in range(3)]
    n_ev = []
    for s in range(3):
        yy, okk = C.labels_at(ev[te_idx[s]], ob[te_idx[s]], 2); n_ev.append(int(yy[okk].sum()))
        flag = "" if n_ev[-1] >= MIN_EVENTS else f"  <- fewer than {MIN_EVENTS} test events: excluded from summaries"
        print(f"stage {s + 1} {STAGES[s]:12} train {len(tr_idx[s]):6,} | test {len(te_idx[s]):6,} | test events @3s {n_ev[-1]}{flag}")
    valid = np.array(n_ev) >= MIN_EVENTS

    torch.manual_seed(131); np.random.seed(131)
    s1 = stage == 0
    enc0, ade = pretrain(X, Y, V, s1 & tr, s1 & va, 1 if a.smoke else 5, a.smoke)
    print(f"encoder pretrained on stage 1 only (val ADE {ade:.2f} m)")
    init = {k: v.clone() for k, v in enc0.state_dict().items()}

    def evaluate(enc, head):
        enc.eval(); head.eval(); out = []
        for s in range(3):
            idx = te_idx[s]; R = []
            with torch.no_grad():
                for b in range(0, len(idx), 1024):
                    R.append(cumulative_risk(head(enc(X[idx[b:b + 1024]]))))
            r = torch.cat(R).numpy()
            y, ok = C.labels_at(ev[idx], ob[idx], 2)
            out.append((average_precision_score(y[ok], r[ok, 2]) if 0 < y[ok].sum() < ok.sum() else np.nan,
                        C.ece(y[ok], r[ok, 2])))
        return out

    def train(enc, head, idx, epochs, rng):
        opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4},
                                 {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
        enc.train(); head.train()
        for _ in range(epochs):
            perm = rng.permutation(idx)
            for b, s in enumerate(range(0, len(perm), 256)):
                if a.smoke and b >= 5: break
                i = perm[s:s + 256]
                xb = torch.from_numpy(augment(X[i].numpy(), rng))
                loss = hazard_nll(head(enc(xb)), E[i], O[i])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()

    ep = 1 if a.smoke else a.epochs
    rows, summ = [], []
    for method in ("naive", "replay", "joint"):
        t0 = time.time(); rng = np.random.default_rng(131); torch.manual_seed(131)
        enc = InteractionEncoder(); enc.load_state_dict(init); head = HazardHead(enc.d_model, 4)
        AUP, ECE = np.full((3, 3), np.nan), np.full((3, 3), np.nan)
        if method == "joint":
            train(enc, head, np.concatenate(tr_idx), ep, rng)
            res = evaluate(enc, head)
            for i in range(3):
                AUP[i] = [r[0] for r in res]; ECE[i] = [r[1] for r in res]
        else:
            buffer = np.array([], int)
            for s in range(3):
                idx = tr_idx[s] if method == "naive" else np.concatenate([tr_idx[s], buffer])
                train(enc, head, idx, ep, rng)
                res = evaluate(enc, head)
                AUP[s] = [r[0] for r in res]; ECE[s] = [r[1] for r in res]
                if method == "replay":
                    k = int(np.ceil(REPLAY_FRAC * len(tr_idx[s])))
                    buffer = np.concatenate([buffer, rng.choice(tr_idx[s], k, replace=False)])
                print(f"  {method:6} after stage {s + 1}: AUPRC@3s " +
                      " | ".join(f"{STAGES[j][:5]} {AUP[s, j]:.3f}" for j in range(3)) +
                      " | ECE@3s " + " ".join(f"{ECE[s, j]:.4f}" for j in range(3)), flush=True)
        for i in range(3):
            for j in range(3):
                rows.append(dict(method=method, after_stage=i + 1, eval_stage=STAGES[j],
                                 AUPRC_3s=AUP[i, j], ECE_3s=ECE[i, j]))
        seq = method != "joint"
        sa, se = cl_summary(AUP, valid, seq), cl_summary(ECE, valid, seq)
        summ.append(dict(method=method, **{f"AUPRC_{k}": v for k, v in sa.items()},
                         **{f"ECE_{k}": v for k, v in se.items()}, seconds=round(time.time() - t0)))
        f = lambda v, fmt: "n/a" if np.isnan(v) else format(v, fmt)
        print(f"{method:6} final avg AUPRC {sa['final_average']:.3f} | forgetting {f(sa['forgetting'], '+.3f')} | "
              f"zero-shot {f(sa['zero_shot'], '.3f')} | final avg ECE {se['final_average']:.4f} | "
              f"ECE change on old stages {f(se['forgetting'], '+.4f')} | {time.time() - t0:.0f}s\n", flush=True)
    Path("results").mkdir(exist_ok=True)
    for name, data in (("continual_matrix", rows), ("continual_summary", summ)):
        with open(f"results/{name}{'_smoke' if a.smoke else ''}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print("Saved results/continual_matrix.csv and results/continual_summary.csv")


if __name__ == "__main__":
    main()