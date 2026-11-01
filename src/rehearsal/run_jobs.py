"""
Stage 1, step 8: train every model the rehearsal of H1, H2 and H4 needs, 5 seeds each,
and save their predictions for the analysis engine. (H3, federated, follows separately.)

JOBS (seeds 131-135; augmentation ON unless stated, as frozen in the registration)
  central      encoder pretrained on all training recordings; hazard model fine-tuned on
               all training samples. Two fine-tunings share one pretrained encoder per seed:
               central_aug (default model; H1, H4) and central_noaug (H4).
  dom          H2: for each frozen factor F1-F4 and level 0/1, a model whose pretraining AND
               fine-tuning use ONLY that level's training recordings; selection on that
               level's validation recordings. (8 models x 5 seeds)
  match        H2 size-matched variant (amendment candidate): the LARGER level of each factor
               re-trained on a random subset of its training recordings (seeded) with about
               the same number of hazard-training samples as the smaller level. (4 x 5)

BUDGET RULE (fixed before the timing probe was run):
  registered budget = pretraining 20 epochs + fine-tuning 10 epochs (pre-registration,
  Section 4). If the probe projects more than 60 laptop-hours for all jobs, every job uses
  the reduced budget of Steps 14-15 (5 + 5 epochs) instead, identically for all jobs, and
  the projected cost of the registered budget is reported for SHRP2 planning.

Usage (from the project folder):
  python src/rehearsal/run_jobs.py --probe                  # timing on 30 batches + projection, no files
  python src/rehearsal/run_jobs.py --budget reduced         # run all jobs (resumable, sequential)
  python src/rehearsal/run_jobs.py --budget reduced --only central   # subset by name prefix
  python src/rehearsal/run_jobs.py --smoke --only central_s131       # quick code check
Outputs: data/processed/preds/haz_<job>.npz  (idx, split, logits, risk for ALL val+test samples)
         models/rehearsal/enc_<key>_s<seed>.pt (pretrained encoders, reused when resuming)
         models/rehearsal/haz_central_*.pt     (fine-tuned central models, for the H4 flicker test;
                                                added 9 Oct 2026 - an older central run without
                                                them is re-fitted and its predictions compared)
         results/rehearsal_jobs.csv            (one row per finished job: sizes, epochs, time)
"""
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from models.encoder import InteractionEncoder, ForecastHead
from models.hazard import HazardHead, hazard_nll, cumulative_risk
from train_pretrain import split_by_scene, predict as forecast, errors
from augment import augment
import calibration as C

SEEDS = [131, 132, 133, 134, 135]
BUDGETS = {"registered": (20, 10), "reduced": (5, 5)}
MAX_HOURS = 60
PRED, ENC, LOG = Path("data/processed/preds"), Path("models/rehearsal"), Path("results/rehearsal_jobs.csv")
FACTOR_FILE = Path("results/rehearsal_factors_by_recording.csv")


def factor_names():
    """The four frozen factors (step 5), read from the frozen file (F4 = F4_heavy after the fallback)."""
    cols = pd.read_csv(FACTOR_FILE, nrows=1).columns
    return [c for c in cols if c[:2] in ("F1", "F2", "F3", "F4")]


TIMING = {"pre_train": 0.0, "pre_val": 0.0, "haz_train": 0.0, "haz_val": 0.0}


# ------------------------------------------------------------------------------ data
def load():
    d = np.load("data/processed/pretrain_samples.npz", allow_pickle=True)
    lab = np.load("data/processed/hazard_labels.npz")
    D = dict(X=torch.from_numpy(d["hist"]), Y=torch.from_numpy(d["fut"]), V=torch.from_numpy(d["fut_valid"]),
             E=torch.from_numpy(lab["event_bin"]), O=torch.from_numpy(lab["obs_bins"]),
             keep=lab["current"] == 0, scene=d["scene"].astype(str), loc=d["location"].astype(str))
    # the SAME split as every step: computed on the not-in-conflict samples, then mapped to recordings
    k = D["keep"]
    tr, va, te = split_by_scene(D["scene"][k], D["loc"][k])
    rec_split = pd.Series(np.select([tr, va, te], ["train", "val", "test"], "none"), index=D["scene"][k])
    rec_split = rec_split.groupby(level=0).first()
    split = pd.Series(D["scene"]).map(rec_split).fillna("none").to_numpy()
    D.update(tr=split == "train", va=split == "val", te=split == "test")
    fac = pd.read_csv(FACTOR_FILE).set_index("scene_id")
    for f in factor_names():
        D[f] = fac.loc[D["scene"], f].to_numpy()
    return D


def jobs():
    J = []
    for s in SEEDS:
        J.append(dict(name=f"central_s{s}", seed=s, kind="central"))
    for f in factor_names():
        for v in (0, 1):
            for s in SEEDS:
                J.append(dict(name=f"dom_{f}_{v}_s{s}", seed=s, kind="dom", factor=f, level=v))
    for f in factor_names():
        for s in SEEDS:
            J.append(dict(name=f"match_{f}_s{s}", seed=s, kind="match", factor=f))
    return J


def masks_for(job, D):
    """Recording-level masks (all samples) for pretraining/fine-tuning train and val."""
    if job["kind"] == "central":
        return D["tr"], D["va"], "all"
    f = job["factor"]
    if job["kind"] == "dom":
        lv = D[f] == job["level"]
        return D["tr"] & lv, D["va"] & lv, f"{f}_{job['level']}"
    # match: larger level subsampled by whole recordings to the smaller level's fine-tuning size
    n = {v: int((D["tr"] & D["keep"] & (D[f] == v)).sum()) for v in (0, 1)}
    big, small = (0, 1) if n[0] >= n[1] else (1, 0)
    rng = np.random.default_rng(job["seed"])
    recs = np.unique(D["scene"][D["tr"] & (D[f] == big)]); rng.shuffle(recs)
    per_rec = pd.Series(D["keep"] & D["tr"]).groupby(D["scene"]).sum()
    chosen, total = [], 0
    for r in recs:
        if total >= n[small]:
            break
        chosen.append(r); total += int(per_rec[r])
    sub = np.isin(D["scene"], chosen)
    job["level"] = big
    return D["tr"] & sub, D["va"] & (D[f] == big), f"{f}_{big}_match"


# --------------------------------------------------------------------------- training
def batches(n, bs, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def pretrain(D, tr, va, epochs, smoke, max_batches=None):
    enc, head = InteractionEncoder(), ForecastHead(future=D["Y"].shape[1])
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    ti, vi = torch.from_numpy(np.flatnonzero(tr)), torch.from_numpy(np.flatnonzero(va))
    best, state, best_ep = np.inf, None, 0
    for ep in range(1, epochs + 1):
        enc.train(); head.train(); t0 = time.time()
        for b, i in enumerate(batches(len(ti), 256)):
            if (smoke and b >= 5) or (max_batches and b >= max_batches):
                break
            j = ti[i]; v = D["V"][j][..., None]
            loss = (Fn.smooth_l1_loss(forecast(enc, head, D["X"][j]), D["Y"][j], reduction="none") * v).sum() \
                / v.sum().clamp(min=1) / 2
            opt.zero_grad(); loss.backward(); opt.step()
        TIMING["pre_train"] += time.time() - t0; t0 = time.time()
        enc.eval(); head.eval(); A = []
        with torch.no_grad():
            for i in batches(len(vi), 1024, False):
                j = vi[i]; A.append(errors(forecast(enc, head, D["X"][j]), D["Y"][j], D["V"][j])[0])
        ade = torch.cat(A).mean().item(); TIMING["pre_val"] += time.time() - t0
        if ade < best:
            best, best_ep, state = ade, ep, {k: t.clone() for k, t in enc.state_dict().items()}
    enc.load_state_dict(state)
    return enc, best, best_ep


def finetune(enc, D, tr, va, epochs, aug, seed, smoke, max_batches=None):
    head = HazardHead(enc.d_model, 4)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    ti, vi = torch.from_numpy(np.flatnonzero(tr)), np.flatnonzero(va)
    best, state, best_ep = np.inf, None, 0
    for ep in range(1, epochs + 1):
        enc.train(); head.train(); t0 = time.time()
        for b, i in enumerate(batches(len(ti), 256)):
            if (smoke and b >= 5) or (max_batches and b >= max_batches):
                break
            j = ti[i]; xb = D["X"][j]
            if aug:
                xb = torch.from_numpy(augment(xb.numpy(), rng))
            loss = hazard_nll(head(enc(xb)), D["E"][j], D["O"][j])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()
        TIMING["haz_train"] += time.time() - t0; t0 = time.time()
        lg = logits_of(enc, head, D["X"], vi)
        nll = C.hazard_nll_np(lg, D["E"][vi].numpy(), D["O"][vi].numpy()); TIMING["haz_val"] += time.time() - t0
        if nll < best:
            best, best_ep = nll, ep
            state = ({k: t.clone() for k, t in enc.state_dict().items()},
                     {k: t.clone() for k, t in head.state_dict().items()})
    enc.load_state_dict(state[0]); head.load_state_dict(state[1])
    return enc, head, best, best_ep


def logits_of(enc, head, X, idx):
    enc.eval(); head.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            out.append(head(enc(X[torch.from_numpy(idx[s:s + 1024])])))
    return torch.cat(out).numpy()


def save_preds(name, enc, head, D):
    sel = np.flatnonzero((D["va"] | D["te"]) & D["keep"])
    lg = logits_of(enc, head, D["X"], sel)
    np.savez_compressed(PRED / f"haz_{name}.npz", idx=sel, split=np.where(D["te"][sel], "test", "val"),
                        logits=lg.astype(np.float32), risk=C.risk_from_logits(lg).astype(np.float32))


def log_row(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader() if new else None; w.writerow(row)


def weights_file(name):
    return ENC / f"haz_{name}.pt"


def is_done(job, name):
    """Central models must also have saved weights (needed for the H4 flicker test)."""
    return (PRED / f"haz_{name}.npz").exists() and (job["kind"] != "central" or weights_file(name).exists())


def out_names(job, smoke):
    s = job["seed"]; tags = ["aug", "noaug"] if job["kind"] == "central" else [""]
    return [(job["name"].replace(f"_s{s}", f"_{t}_s{s}") if t else job["name"]) + ("_smoke" if smoke else "") for t in tags]


def run_job(job, D, pre_ep, haz_ep, smoke):
    if all(is_done(job, n) for n in out_names(job, smoke)):
        print("  [skip] already done"); return
    s = job["seed"]; torch.manual_seed(s); np.random.seed(s); t0 = time.time()
    ptr, pva, key = masks_for(job, D)
    enc_file = ENC / f"enc_{key}_s{s}{'_smoke' if smoke else ''}.pt"
    if enc_file.exists():
        enc = InteractionEncoder(); ck = torch.load(enc_file, map_location="cpu")
        enc.load_state_dict(ck["encoder"]); ade, pep = ck["ade"], ck["epoch"]
    else:
        enc, ade, pep = pretrain(D, ptr, pva, pre_ep, smoke)
        torch.save({"encoder": enc.state_dict(), "ade": ade, "epoch": pep}, enc_file)
    t_pre = time.time() - t0
    variants = [("aug", True), ("noaug", False)] if job["kind"] == "central" else [("", True)]
    htr, hva = ptr & D["keep"], pva & D["keep"]
    for tag, aug in variants:
        name = job["name"].replace(f"_s{s}", f"_{tag}_s{s}") if tag else job["name"]
        name += "_smoke" if smoke else ""
        if is_done(job, name):
            print(f"  [skip] {name} already done"); continue
        old = PRED / f"haz_{name}.npz"
        old_risk = np.load(old)["risk"] if old.exists() else None    # refit of an earlier central run
        t1 = time.time(); torch.manual_seed(s)
        e = InteractionEncoder(); e.load_state_dict(enc.state_dict())
        e, head, nll, hep = finetune(e, D, htr, hva, haz_ep, aug, s, smoke)
        save_preds(name, e, head, D)
        if job["kind"] == "central":
            torch.save({"encoder": e.state_dict(), "head": head.state_dict()}, weights_file(name))
        if old_risk is not None:
            diff = float(np.abs(np.load(old)["risk"] - old_risk).max())
            print(f"  REPRODUCIBILITY: refit vs earlier predictions, max |difference| = {diff:.2e}")
        row = dict(job=name, kind=job["kind"], factor=job.get("factor", ""), level=job.get("level", ""), seed=s,
                   augment=aug, pretrain_recordings=len(np.unique(D["scene"][ptr])), pretrain_samples=int(ptr.sum()),
                   train_samples=int(htr.sum()), val_samples=int(hva.sum()), pre_epochs=pre_ep, haz_epochs=haz_ep,
                   best_pre_epoch=pep, best_haz_epoch=hep, val_ade=round(ade, 4), val_nll=round(float(nll), 5),
                   pretrain_seconds=round(t_pre), finetune_seconds=round(time.time() - t1), smoke=smoke)
        if not smoke and old_risk is None:          # refits are not logged again (same model, same row)
            log_row(row)
        print(f"  {name}: train {row['train_samples']:,} | best epochs {pep}/{hep} | val ADE {ade:.3f} "
              f"NLL {nll:.4f} | {row['pretrain_seconds']}s + {row['finetune_seconds']}s", flush=True)
        t_pre = 0                                   # shared encoder counted once


def probe(D):
    """Time 30 batches of pretraining and fine-tuning (and one validation pass each) on the
    central job; project the cost of every job under both budgets."""
    nb = 30; torch.manual_seed(131)
    enc, _, _ = pretrain(D, D["tr"], D["va"], 1, False, max_batches=nb)
    finetune(enc, D, D["tr"] & D["keep"], D["va"] & D["keep"], 1, True, 131, False, max_batches=nb)
    n_pv, n_hv = int(D["va"].sum()), int((D["va"] & D["keep"]).sum())
    c = dict(pt=TIMING["pre_train"] / (nb * 256), pv=TIMING["pre_val"] / n_pv,
             ht=TIMING["haz_train"] / (nb * 256), hv=TIMING["haz_val"] / n_hv)
    tot = {}
    for budget, (pe, he) in BUDGETS.items():
        sec = 0.0
        for job in jobs():
            ptr, pva, _ = masks_for(dict(job), D)
            reps = 2 if job["kind"] == "central" else 1
            pre = pe * (int(ptr.sum()) * c["pt"] + int(pva.sum()) * c["pv"])
            haz = he * (int((ptr & D["keep"]).sum()) * c["ht"] + int((pva & D["keep"]).sum()) * c["hv"])
            sec += pre + reps * haz
        tot[budget] = sec / 3600
    print(f"\nprobe (ms per sample): pretrain train {c['pt']*1e3:.2f} / val {c['pv']*1e3:.2f}; "
          f"fine-tune train {c['ht']*1e3:.2f} / val {c['hv']*1e3:.2f} (augmentation on)")
    for b, h in tot.items():
        print(f"  projected total for all {len(jobs())} jobs, {b} budget {BUDGETS[b]}: {h:.1f} laptop-hours")
    choice = "registered" if tot["registered"] <= MAX_HOURS else "reduced"
    print(f"BUDGET RULE (limit {MAX_HOURS} h) -> use '{choice}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", choices=list(BUDGETS), default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    for p in (PRED, ENC):
        p.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(exist_ok=True)
    print("loading data ..."); D = load()
    print(f"samples: train {int(D['tr'].sum()):,} | val {int(D['va'].sum()):,} | test {int(D['te'].sum()):,} "
          f"(all; not-in-conflict train {int((D['tr'] & D['keep']).sum()):,})")
    if a.probe:
        probe(D); return
    if a.budget is None and not a.smoke:
        raise SystemExit("choose --budget registered|reduced (run --probe first)")
    pre_ep, haz_ep = (1, 1) if a.smoke else BUDGETS[a.budget]
    todo = [j for j in jobs() if j["name"].startswith(a.only)]
    print(f"{len(todo)} jobs, budget {pre_ep}+{haz_ep} epochs")
    for k, job in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {job['name']}", flush=True)
        run_job(job, D, pre_ep, haz_ep, a.smoke)


if __name__ == "__main__":
    main()