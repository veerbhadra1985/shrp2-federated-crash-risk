"""
Step 11: train the multi-horizon hazard model on top of the pretrained encoder.
Stand-in task on INTERACTION: P(TTC2D conflict < 1.5 s within 1, 2, 3, 5 s).

Usage:
    python src/train_hazard.py --smoke
    python src/train_hazard.py --epochs 10
Step 12 ablations (combine freely; output names get a suffix, nothing is overwritten):
    --no-pretrain      encoder starts from random weights
    --separate-heads   one independent sigmoid classifier per horizon (BCE), instead of the hazard head
    --no-neighbours    model sees only the target vehicle (neighbours removed in train, val and test)
Inputs : data/processed/pretrain_samples.npz, data/processed/hazard_labels.npz,
         models/encoder_pretrained.pt
Outputs: models/hazard_model.pt, results/hazard_test_metrics.csv
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoder import InteractionEncoder
from models.hazard import HazardHead, cumulative_risk, hazard_nll
from train_pretrain import split_by_scene
from augment import augment

HORIZONS = [1, 2, 3, 5]


def ece(y, p, bins=10):
    """Expected calibration error with equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def bce_per_horizon(logits, event_bin, obs_bins):
    """Ablation: independent binary classifier per horizon; only samples whose outcome
    is known at that horizon contribute."""
    k = torch.arange(logits.shape[1], device=logits.device)[None, :]
    y = ((event_bin[:, None] >= 0) & (event_bin[:, None] <= k)).float()
    known = (y > 0) | (obs_bins[:, None] >= k + 1)
    l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l * known).sum() / known.sum().clamp(min=1)


def risk_of(enc, head, X, bs=1024, to_risk=cumulative_risk):
    enc.eval(); head.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(to_risk(head(enc(X[s:s + bs]))))
    return torch.cat(out).numpy()


def monotonic_violations(risk, tol=1e-7):
    """Share of samples whose risk DEcreases at some longer horizon, and the mean size of
    the largest decrease among those samples (percentage points)."""
    drop = -np.diff(risk, axis=1)
    bad = (drop > tol).any(1)
    return float(bad.mean()), float(drop.max(1)[bad].mean() * 100) if bad.any() else 0.0


def metrics(risk, ev, ob, ttc_now):
    rows = []
    for k, h in enumerate(HORIZONS):
        y = (ev >= 0) & (ev <= k)
        ok = y | (ob >= k + 1)                      # outcome known up to this horizon
        yk, pk = y[ok].astype(int), risk[ok, k]
        base = np.where(np.isfinite(ttc_now[ok]), 1.0 / np.maximum(ttc_now[ok], 1e-3), 0.0)
        prev = yk.mean()
        rows.append(dict(horizon_s=h, n=int(ok.sum()), positives=int(yk.sum()), prevalence=prev,
                         AUPRC=average_precision_score(yk, pk), AUROC=roc_auc_score(yk, pk),
                         Brier=np.mean((pk - yk) ** 2), Brier_baserate=prev * (1 - prev),
                         ECE=ece(yk, pk), mean_pred=pk.mean(),
                         AUPRC_TTC2D_now=average_precision_score(yk, base),
                         AUROC_TTC2D_now=roc_auc_score(yk, base)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--augment", action="store_true",
                    help="Step 16b: randomly drop/flicker/noise neighbours during training")
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--separate-heads", action="store_true")
    ap.add_argument("--no-neighbours", action="store_true")
    args = ap.parse_args()
    suffix = "".join(t for t, on in (("_aug", args.augment), ("_nopre", args.no_pretrain),
                                      ("_sep", args.separate_heads), ("_nonb", args.no_neighbours)) if on)
    loss_fn = bce_per_horizon if args.separate_heads else hazard_nll
    to_risk = torch.sigmoid if args.separate_heads else cumulative_risk
    print(f"variant: {suffix or 'full model'}")
    aug_rng = np.random.default_rng(131)
    torch.manual_seed(131); np.random.seed(131)
    torch.set_num_threads(os.cpu_count() or 1)

    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    keep = lab["current"] == 0                      # predict ONSET: drop already-in-conflict
    X = torch.from_numpy(d["hist"][keep])
    if args.no_neighbours:
        X = X.clone(); X[:, 1:] = 0.0                # target vehicle only
    ev = lab["event_bin"][keep]; ob = lab["obs_bins"][keep]; now = lab["ttc2d_now"][keep]
    tr, va, te = split_by_scene(d["scene"][keep], d["location"][keep])
    print(f"samples (not in conflict at t): train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")

    enc = InteractionEncoder()
    if not args.no_pretrain:
        ck = torch.load("models/encoder_pretrained.pt", map_location="cpu")
        enc.load_state_dict(ck["encoder"])
    head = HazardHead(enc.d_model, len(HORIZONS))
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    Ev, Ob = torch.from_numpy(ev), torch.from_numpy(ob)
    Xtr, Etr, Otr = X[tr], Ev[tr], Ob[tr]
    best, epochs = float("inf"), (1 if args.smoke else args.epochs)
    Path("models").mkdir(exist_ok=True); Path("results").mkdir(exist_ok=True)
    for ep in range(1, epochs + 1):
        enc.train(); head.train(); t0 = time.time(); perm = torch.randperm(len(Xtr)); L = []
        for b, s in enumerate(range(0, len(perm), args.batch)):
            if args.smoke and b >= 20:
                break
            i = perm[s:s + args.batch]
            xb = Xtr[i]
            if args.augment:
                xb = torch.from_numpy(augment(xb.numpy(), aug_rng))
            loss = loss_fn(head(enc(xb)), Etr[i], Otr[i])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step(); L.append(loss.item())
        enc.eval(); head.eval()
        with torch.no_grad():
            Xv, Ev_, Ov = X[va], Ev[va], Ob[va]
            vl = np.mean([loss_fn(head(enc(Xv[s:s + 1024])), Ev_[s:s + 1024], Ov[s:s + 1024]).item()
                          for s in range(0, int(va.sum()), 1024)])
        print(f"epoch {ep}: train loss {np.mean(L):.4f} | val loss {vl:.4f} | {time.time() - t0:.0f}s")
        if vl < best:
            best = vl
            torch.save({"encoder": enc.state_dict(), "head": head.state_dict(), "horizons": HORIZONS},
                       f"models/hazard_model{suffix}.pt")
    ck = torch.load(f"models/hazard_model{suffix}.pt", map_location="cpu")
    enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    risk = risk_of(enc, head, X[te], to_risk=to_risk)
    viol_share, viol_size = monotonic_violations(risk)
    rows = metrics(risk, ev[te], ob[te], now[te])
    for r in rows:
        r["variant"] = suffix or "full"; r["mono_violation_share"] = viol_share; r["mono_violation_pp"] = viol_size
    print(f"\nTEST (unseen recordings) | samples where risk DEcreases with horizon: {viol_share:.2%}"
          + (f" (mean largest drop {viol_size:.2f} percentage points)" if viol_share > 0 else ""))
    print(f"{'h':>3} {'prev':>7} {'AUPRC':>6} {'TTC2D':>6} {'AUROC':>6} {'Brier':>7} {'base':>7} {'ECE':>6}")
    for r in rows:
        print(f"{r['horizon_s']:>2}s {r['prevalence']:7.3%} {r['AUPRC']:6.3f} {r['AUPRC_TTC2D_now']:6.3f} "
              f"{r['AUROC']:6.3f} {r['Brier']:7.4f} {r['Brier_baserate']:7.4f} {r['ECE']:6.4f}")
    import csv
    with open(f"results/hazard_test_metrics{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Saved models/hazard_model{suffix}.pt and results/hazard_test_metrics{suffix}.csv")


if __name__ == "__main__":
    main()