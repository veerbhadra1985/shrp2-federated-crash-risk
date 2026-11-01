"""
Step 17: communication cost, inference latency and membership-inference privacy test.

1. Communication (federated, per client): model upload + download per round, for FedAvg
   (encoder + head) and FedPer (encoder only), vs uploading the client's raw training data.
2. Latency: CPU inference time for one vehicle (batch 1) and for a batch of 256.
3. Privacy: loss-threshold membership-inference attack on the trained hazard model:
   can an attacker with the model tell training data (members) from unseen data
   (non-members) by their loss? Attack AUROC 0.5 = no leakage, 1.0 = full leakage.
   Reported at sample level and at RECORDING level (the unit a data owner would protect).
   Members = train split, non-members = test split (never used in training).
Usage : python src/costs_privacy.py [--model models/hazard_model_aug.pt] [--rounds 30]
Output: results/costs_privacy.csv
"""
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np


def attack_auroc(member_loss, nonmember_loss, rng, n=None):
    """AUROC of the loss-threshold attack (lower loss -> 'member'), balanced classes."""
    from sklearn.metrics import roc_auc_score
    m, nm = np.asarray(member_loss), np.asarray(nonmember_loss)
    n = min(len(m), len(nm)) if n is None else min(n, len(m), len(nm))
    m, nm = rng.choice(m, n, replace=False), rng.choice(nm, n, replace=False)
    y = np.r_[np.ones(n), np.zeros(n)]
    return float(roc_auc_score(y, -np.r_[m, nm])), n


def per_sample_nll(logits, ev, ob):
    """Survival NLL per sample (numpy), same definition as training."""
    ls = lambda z: -np.logaddexp(0.0, -z)
    K = logits.shape[1]; j = np.arange(K)[None, :]
    is_ev = (ev >= 0)[:, None]
    upto = np.where(is_ev, ev[:, None], ob[:, None])
    return -((ls(-logits) * (j < upto)).sum(1) + (ls(logits) * (is_ev & (j == ev[:, None]))).sum(1))


def main():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.encoder import InteractionEncoder
    from models.hazard import HazardHead
    from train_pretrain import split_by_scene
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/hazard_model_aug.pt")
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args(); torch.set_num_threads(os.cpu_count() or 1); rng = np.random.default_rng(131)
    rows = []

    enc, head = InteractionEncoder(), HazardHead(64, 4)
    ck = torch.load(a.model, map_location="cpu"); enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
    enc.eval(); head.eval()
    n_enc = sum(p.numel() for p in enc.parameters()); n_head = sum(p.numel() for p in head.parameters())
    mb = lambda nparams: nparams * 4 / 1e6                               # float32
    print(f"model: {a.model}\nparameters: encoder {n_enc:,} | hazard head {n_head:,} | total {n_enc + n_head:,}")

    # ---- 1. communication -------------------------------------------------------------
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    keep = lab["current"] == 0
    loc, scene = d["location"][keep], d["scene"][keep]
    tr, va, te = split_by_scene(scene, loc)
    bytes_per_sample = d["hist"][0].nbytes + 2 * 8                      # history + two int64 labels
    fedavg_round = 2 * mb(n_enc + n_head); fedper_round = 2 * mb(n_enc)   # upload + download
    print(f"\n1. COMMUNICATION per client ({a.rounds} rounds; upload + download)")
    print(f"   FedAvg  {fedavg_round:.2f} MB/round -> {fedavg_round * a.rounds:.1f} MB total")
    print(f"   FedPer  {fedper_round:.2f} MB/round -> {fedper_round * a.rounds:.1f} MB total (head stays local)")
    for c in sorted(np.unique(loc)):
        raw = (tr & (loc == c)).sum() * bytes_per_sample / 1e6
        rows.append(dict(part="communication", item=c, value_1=round(raw, 2),
                         value_2=round(fedavg_round * a.rounds, 2), value_3=round(fedper_round * a.rounds, 2),
                         note="raw train data MB | FedAvg total MB | FedPer total MB"))
    raws = [r["value_1"] for r in rows]
    print(f"   raw training data per client: {min(raws):.1f}-{max(raws):.1f} MB (processed samples; raw trajectories are larger)")

    # ---- 2. latency -------------------------------------------------------------------
    x1 = torch.from_numpy(d["hist"][:1]); xb = torch.from_numpy(d["hist"][:256])
    def timeit(x, reps):
        with torch.no_grad():
            for _ in range(5): head(enc(x))
            t = []
            for _ in range(reps):
                t0 = time.perf_counter(); head(enc(x)); t.append(time.perf_counter() - t0)
        return np.median(t) * 1000
    t1, t256 = timeit(x1, 200), timeit(xb, 30)
    print(f"\n2. LATENCY (CPU, {torch.get_num_threads()} threads)")
    print(f"   one vehicle: {t1:.2f} ms | batch of 256: {t256:.1f} ms ({t256 / 256:.3f} ms per vehicle)")
    print(f"   data arrive every 100 ms (10 Hz) -> real-time margin {100 / t1:.0f}x for a single vehicle")
    rows.append(dict(part="latency", item="batch1_ms", value_1=round(t1, 3), value_2="", value_3="", note="median ms"))
    rows.append(dict(part="latency", item="batch256_ms", value_1=round(t256, 2), value_2=round(t256 / 256, 4), value_3="",
                     note="median ms | ms per vehicle"))

    # ---- 3. membership inference ------------------------------------------------------
    X = d["hist"][keep]; ev, ob = lab["event_bin"][keep], lab["obs_bins"][keep]
    def nll_of(mask):
        idx = np.where(mask)[0]; out = []
        with torch.no_grad():
            for s in range(0, len(idx), 1024):
                out.append(head(enc(torch.from_numpy(X[idx[s:s + 1024]]))).numpy())
        lg = np.concatenate(out)
        return per_sample_nll(lg, ev[idx], ob[idx]), scene[idx], ev[idx]
    m_loss, m_scene, m_ev = nll_of(tr); n_loss, n_scene, n_ev = nll_of(te)
    auc_s, n_s = attack_auroc(m_loss, n_loss, rng, n=20000)
    auc_e, n_e = attack_auroc(m_loss[m_ev >= 0], n_loss[n_ev >= 0], rng)        # conflict samples only
    rec = lambda L, S: np.array([L[S == s].mean() for s in np.unique(S)])
    auc_r, n_r = attack_auroc(rec(m_loss, m_scene), rec(n_loss, n_scene), rng)
    print(f"\n3. MEMBERSHIP INFERENCE (loss-threshold attack; 0.5 = no leakage)")
    print(f"   sample level        AUROC {auc_s:.3f}  ({n_s:,} members vs {n_s:,} non-members)")
    print(f"   conflict samples    AUROC {auc_e:.3f}  ({n_e:,} vs {n_e:,}) - rare events are most exposed")
    print(f"   recording level     AUROC {auc_r:.3f}  ({n_r} vs {n_r} recordings)")
    print(f"   mean loss: members {m_loss.mean():.4f} | non-members {n_loss.mean():.4f}")
    for item, v, n in (("sample", auc_s, n_s), ("conflict_samples", auc_e, n_e), ("recording", auc_r, n_r)):
        rows.append(dict(part="membership_inference", item=item, value_1=round(v, 4), value_2=n, value_3="",
                         note="attack AUROC | n per class"))
    Path("results").mkdir(exist_ok=True)
    with open("results/costs_privacy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nSaved results/costs_privacy.csv")


if __name__ == "__main__":
    main()