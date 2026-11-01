"""
Step 10: pretrain the interaction encoder on trajectory forecasting.

Usage:
    python src/train_pretrain.py --smoke            # quick check (1 epoch, few batches)
    python src/train_pretrain.py --epochs 10        # full run
Input : data/processed/pretrain_samples.npz   (from src/data/build_samples.py)
Output: models/encoder_pretrained.pt, results/pretrain_log.csv
Split : by RECORDING within each location (project-wide split, see split_by_scene).
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoder import InteractionEncoder, ForecastHead

DATA = Path("data/processed/pretrain_samples.npz")


def split_by_scene(scene, loc, seed=131, frac=(0.8, 0.1)):
    """THE project-wide split (used by every step): recordings are split 80/10/10
    INSIDE each location, so no recording is in two splits and every location
    (federated client) has train, validation and test recordings."""
    rng = np.random.default_rng(seed)
    tr_s, va_s, te_s = set(), set(), set()
    for c in sorted(np.unique(loc)):
        rec = np.unique(scene[loc == c]); rng.shuffle(rec); n = len(rec)
        n_te = max(1, int(round((1 - frac[0] - frac[1]) * n)))
        n_va = max(1, int(round(frac[1] * n))) if n >= 3 else 0
        te_s.update(rec[:n_te]); va_s.update(rec[n_te:n_te + n_va]); tr_s.update(rec[n_te + n_va:])
    return (np.isin(scene, list(tr_s)), np.isin(scene, list(va_s)), np.isin(scene, list(te_s)))


def errors(pred, fut, valid):
    """Average and final displacement error over valid future steps (metres)."""
    d = torch.linalg.norm(pred - fut, dim=-1) * valid
    ade = d.sum(1) / valid.sum(1).clamp(min=1)
    last = (valid * torch.arange(valid.shape[1], device=valid.device)).argmax(1)
    fde = d[torch.arange(len(d)), last]
    return ade, fde


def const_vel(x, steps, hz=10.0):
    """Constant-velocity forecast of the target in its own frame: (B, steps, 2)."""
    t = torch.arange(1, steps + 1, dtype=torch.float32, device=x.device) / hz
    return x[:, 0, -1, 2:4][:, None, :] * t[None, :, None]


def predict(enc, head, x):
    """Model predicts a CORRECTION to the constant-velocity forecast (residual learning)."""
    return const_vel(x, head.future) + head(enc(x))


def evaluate(enc, head, X, Y, V, bs=1024):
    enc.eval(); head.eval(); A, Fd, Acv = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            x, y, v = X[s:s + bs], Y[s:s + bs], V[s:s + bs]
            a, f = errors(predict(enc, head, x), y, v); A.append(a); Fd.append(f)
            Acv.append(errors(const_vel(x, y.shape[1]), y, v)[0])   # baseline
    return torch.cat(A).mean().item(), torch.cat(Fd).mean().item(), torch.cat(Acv).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(131); np.random.seed(131)
    torch.set_num_threads(os.cpu_count() or 1)

    d = np.load(DATA, allow_pickle=True)
    X = torch.from_numpy(d["hist"]); Y = torch.from_numpy(d["fut"]); V = torch.from_numpy(d["fut_valid"])
    tr, va, te = split_by_scene(d["scene"], d["location"])
    print(f"samples: train {tr.sum():,} | val {va.sum():,} | test {te.sum():,} (split by recording)")

    enc, head = InteractionEncoder(), ForecastHead(future=Y.shape[1])
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)
    Xtr, Ytr, Vtr = X[tr], Y[tr], V[tr]
    epochs = 1 if args.smoke else args.epochs
    Path("results").mkdir(exist_ok=True); Path("models").mkdir(exist_ok=True)
    log = ["epoch,train_loss,val_ADE,val_FDE,val_ADE_constvel,seconds"]
    best = float("inf")
    for ep in range(1, epochs + 1):
        enc.train(); head.train(); t0 = time.time()
        perm = torch.randperm(len(Xtr)); losses = []
        for b, s in enumerate(range(0, len(perm), args.batch)):
            if args.smoke and b >= 20:
                break
            idx = perm[s:s + args.batch]
            pred = predict(enc, head, Xtr[idx])
            v = Vtr[idx][..., None]
            loss = (Fn.smooth_l1_loss(pred, Ytr[idx], reduction="none") * v).sum() / v.sum().clamp(min=1) / 2
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step(); losses.append(loss.item())
        ade, fde, ade_cv = evaluate(enc, head, X[va], Y[va], V[va])
        sec = time.time() - t0
        print(f"epoch {ep}: loss {np.mean(losses):.3f} | val ADE {ade:.2f} m, FDE {fde:.2f} m "
              f"| constant-velocity ADE {ade_cv:.2f} m | {sec:.0f}s")
        log.append(f"{ep},{np.mean(losses):.4f},{ade:.4f},{fde:.4f},{ade_cv:.4f},{sec:.1f}")
        if ade < best:
            best = ade
            torch.save({"encoder": enc.state_dict(), "config": {"d_model": enc.d_model},
                        "val_ADE": ade}, "models/encoder_pretrained.pt")
    ade, fde, ade_cv = evaluate(enc, head, X[te], Y[te], V[te])
    print(f"\nTEST (unseen recordings): ADE {ade:.2f} m, FDE {fde:.2f} m | constant-velocity ADE {ade_cv:.2f} m")
    Path("results/pretrain_log.csv").write_text("\n".join(log) + f"\ntest,,{ade:.4f},{fde:.4f},{ade_cv:.4f},\n")
    print("Saved models/encoder_pretrained.pt and results/pretrain_log.csv")


if __name__ == "__main__":
    main()