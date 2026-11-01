"""
Steps 14-15: leave-one-location-out (LOLO) calibration test on INTERACTION (stand-in task).

For each held-out location L (never seen in ANY training step):
  1. pretrain a fresh encoder on the other 11 locations (train recordings)
  2. train the hazard model on the other 11 locations; best epoch by their validation NLL
  3. fit calibrators on the 11 locations' VALIDATION recordings:
       raw | temperature | isotonic | conformal false-alarm thresholds (alpha)
  4. evaluate on ALL recordings of L
Reports per fold and horizon: AUPRC, Brier, ECE, adaptive ECE, mean prediction vs observed
rate (calibration-in-the-large), base-rate shift, and conformal false-alarm rate vs target.
Resumable: finished folds are skipped when re-run.
Usage:
    python src/lolo_calibration.py --smoke --folds DR_USA_Roundabout_SR
    python src/lolo_calibration.py                      # all 12 folds (several hours)
    python src/lolo_calibration.py --save-preds         # Phase 2 B4 (added 10 Oct 2026): re-run all folds with
                                                        # identical settings, save fold predictions to
                                                        # data/processed/preds/lolo_<loc>_s131.npz, results to *_rerun.csv
Outputs: results/lolo_summary.csv, results/lolo_conformal.csv, results/lolo_reliability.csv
"""
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.encoder import InteractionEncoder, ForecastHead
from models.hazard import HazardHead, hazard_nll
from train_pretrain import split_by_scene, predict as forecast, errors
import calibration as C

HORIZONS = [1, 2, 3, 5]
ALPHAS = [0.05, 0.01]
OUT = Path("results")


def batches(n, bs, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def pretrain(X, Y, V, tr, va, epochs, smoke):
    enc, head = InteractionEncoder(), ForecastHead(future=Y.shape[1])
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    Xt, Yt, Vt = X[tr], Y[tr], V[tr]; Xv, Yv, Vv = X[va], Y[va], V[va]
    best, best_state = np.inf, None
    for ep in range(epochs):
        enc.train(); head.train()
        for b, i in enumerate(batches(len(Xt), 256)):
            if smoke and b >= 5: break
            v = Vt[i][..., None]
            loss = (Fn.smooth_l1_loss(forecast(enc, head, Xt[i]), Yt[i], reduction="none") * v).sum() / v.sum().clamp(min=1) / 2
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval(); A = []
        with torch.no_grad():
            for i in batches(len(Xv), 1024, False):
                A.append(errors(forecast(enc, head, Xv[i]), Yv[i], Vv[i])[0])
        ade = torch.cat(A).mean().item()
        if ade < best:
            best, best_state = ade, {k: v.clone() for k, v in enc.state_dict().items()}
    enc.load_state_dict(best_state)
    return enc, best


def train_hazard(enc, X, E, O, tr, va, epochs, smoke):
    head = HazardHead(enc.d_model, len(HORIZONS))
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    best, best_state = np.inf, None
    for ep in range(epochs):
        enc.train(); head.train()
        for b, i in enumerate(batches(len(tr), 256)):
            if smoke and b >= 5: break
            j = tr[i]
            loss = hazard_nll(head(enc(X[j])), E[j], O[j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()
        lg = logits_of(enc, head, X[va])
        nll = C.hazard_nll_np(lg, E[va].numpy(), O[va].numpy())
        if nll < best:
            best = nll
            best_state = ({k: v.clone() for k, v in enc.state_dict().items()},
                          {k: v.clone() for k, v in head.state_dict().items()})
    enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    return enc, head


def logits_of(enc, head, X):
    enc.eval(); head.eval(); L = []
    with torch.no_grad():
        for i in batches(len(X), 1024, False):
            L.append(head(enc(X[i])))
    return torch.cat(L).numpy()


def done_folds():
    f = OUT / "lolo_summary.csv"
    if not f.exists(): return set()
    with open(f) as fh:
        return {r["held_out"] for r in csv.DictReader(fh)}


def append(path, rows):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new: w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--pre-epochs", type=int, default=5)
    ap.add_argument("--haz-epochs", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    X = torch.from_numpy(d["hist"]); Y = torch.from_numpy(d["fut"]); V = torch.from_numpy(d["fut_valid"])
    ev, ob, cur = lab["event_bin"], lab["obs_bins"], lab["current"]
    E, O = torch.from_numpy(ev), torch.from_numpy(ob)
    loc = d["location"]; tr, va, te = split_by_scene(d["scene"], loc)
    keep = cur == 0
    folds = a.folds or sorted(np.unique(loc))
    PRED = Path("data/processed/preds"); PRED.mkdir(parents=True, exist_ok=True)
    if a.save_preds:      # re-run every fold; resumable through the saved prediction files; original CSVs untouched
        skip = set() if a.smoke else {L for L in np.unique(loc) if (PRED / f"lolo_{L}_s131.npz").exists()}
    else:
        skip = set() if a.smoke else done_folds()
    OUT.mkdir(exist_ok=True)
    tag = "_smoke" if a.smoke else ("_rerun" if a.save_preds else "")
    for L in folds:
        if L in skip:
            print(f"[skip] {L} already done"); continue
        torch.manual_seed(131); np.random.seed(131); t0 = time.time()
        src = loc != L
        print(f"\n=== held-out location: {L} ===", flush=True)
        enc, ade = pretrain(X, Y, V, src & tr, src & va, 1 if a.smoke else a.pre_epochs, a.smoke)
        print(f"  pretrained encoder without {L}: source val ADE {ade:.2f} m | {time.time()-t0:.0f}s", flush=True)
        htr = np.where(src & tr & keep)[0]; hva = src & va & keep
        enc, head = train_hazard(enc, X, E, O, torch.from_numpy(htr), torch.from_numpy(np.where(hva)[0]),
                                 1 if a.smoke else a.haz_epochs, a.smoke)
        print(f"  hazard model trained | {time.time()-t0:.0f}s", flush=True)
        cal_lg, cal_ev, cal_ob = logits_of(enc, head, X[hva]), ev[hva], ob[hva]
        tgt = (loc == L) & keep
        t_lg, t_ev, t_ob = logits_of(enc, head, X[tgt]), ev[tgt], ob[tgt]
        if a.save_preds and not a.smoke:
            np.savez_compressed(PRED / f"lolo_{L}_s131.npz", idx_cal=np.flatnonzero(hva), logits_cal=cal_lg.astype(np.float32),
                                idx_tgt=np.flatnonzero(tgt), logits_tgt=t_lg.astype(np.float32))
        src_tr_ev, src_tr_ob = ev[src & tr & keep], ob[src & tr & keep]
        cals = [C.Raw().fit(cal_lg, cal_ev, cal_ob), C.Temperature().fit(cal_lg, cal_ev, cal_ob),
                C.Isotonic().fit(cal_lg, cal_ev, cal_ob)]
        summary, conf, rel = [], [], []
        for M in cals:
            r = M.transform(t_lg)
            mono = bool((np.diff(r, axis=1) >= -1e-9).all())
            for k, h in enumerate(HORIZONS):
                y, ok = C.labels_at(t_ev, t_ob, k); ys, oks = C.labels_at(src_tr_ev, src_tr_ob, k)
                yk, pk = y[ok], r[ok, k]
                both = 0 < yk.sum() < len(yk)
                summary.append(dict(held_out=L, method=M.name, horizon_s=h, n=int(ok.sum()), events=int(yk.sum()),
                                    prev_target=yk.mean(), prev_source=ys[oks].mean(),
                                    mean_pred=pk.mean(), AUPRC=average_precision_score(yk, pk) if both else np.nan,
                                    Brier=np.mean((pk - yk) ** 2), ECE=C.ece(yk, pk), ECE_adaptive=C.ece_adaptive(yk, pk),
                                    temperature=getattr(M, "T", np.nan), monotone=mono))
                if h == 3:
                    rel += [dict(held_out=L, method=M.name, bin=b, mean_pred=mp, observed=obs, count=c)
                            for b, (mp, obs, c) in enumerate(C.reliability_bins(yk, pk))]
        r_cal, r_t = C.risk_from_logits(cal_lg), C.risk_from_logits(t_lg)
        for k, h in enumerate(HORIZONS):
            yc, okc = C.labels_at(cal_ev, cal_ob, k); yt, okt = C.labels_at(t_ev, t_ob, k)
            neg_cal = r_cal[okc & (yc == 0), k]; st, ytk = r_t[okt, k], yt[okt]
            for al in ALPHAS:
                tau = C.conformal_threshold(neg_cal, al)
                conf.append(dict(held_out=L, horizon_s=h, alpha=al, threshold=tau,
                                 false_alarm_rate=float(np.mean(st[ytk == 0] > tau)) if (ytk == 0).any() else np.nan,
                                 recall=float(np.mean(st[ytk == 1] > tau)) if (ytk == 1).any() else np.nan,
                                 events=int(ytk.sum())))
        append(OUT / f"lolo_summary{tag}.csv", summary)
        append(OUT / f"lolo_conformal{tag}.csv", conf)
        if rel: append(OUT / f"lolo_reliability{tag}.csv", rel)
        s3 = {m["method"]: m for m in summary if m["horizon_s"] == 3}
        c3 = [c for c in conf if c["horizon_s"] == 3]
        print(f"  @3s target events {s3['raw']['events']} | rate target {s3['raw']['prev_target']:.2%} vs source {s3['raw']['prev_source']:.2%}")
        for m in ("raw", "temperature", "isotonic"):
            x = s3[m]
            print(f"  {m:11} mean pred {x['mean_pred']:.2%} | AUPRC {x['AUPRC']:.3f} | ECE {x['ECE']:.4f} | adaptive ECE {x['ECE_adaptive']:.4f}"
                  + (f" | T={x['temperature']:.2f}" if m == "temperature" else ""))
        for c in c3:
            print(f"  conformal alpha={c['alpha']}: false-alarm rate {c['false_alarm_rate']:.3f} (target <= {c['alpha']}) | recall {c['recall']:.3f}")
        print(f"  fold time {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()