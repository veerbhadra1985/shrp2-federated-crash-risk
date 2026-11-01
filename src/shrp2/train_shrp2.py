"""
SHRP2 TRAINING RUNNER (pre-registration osf.io/y5zrp Sections 2, 4-6; Amendment 1; Amendment 4 for P1 scores).
Trains every model H1, H2, H4 and H5 need, 5 seeds each, with the REGISTERED budget only (pretraining 20 epochs,
fine-tuning 10 epochs; no reduced budget exists for SHRP2 and budgets are never extended), and writes the
predictions the confirmatory analysis reads. H3 (federated) has its own runner. Written and tested on synthetic
data before any SHRP2 access.

The training loops are those of the rehearsal (src/rehearsal/run_jobs.py: pretrain, finetune), reproduced line for
line with one change only: samples are read from disk in batches (the every-frame validation and test sets do not
fit in memory). test_train_shrp2.py checks that both give bit-identical models on the same data.

MODELS (seeds 131-135; neighbour augmentation ON unless stated, as frozen)
  Stage A   central_hazard_aug, central_hazard_noaug, central_sep_aug - one pretrained encoder per seed
  select    registered risk-head choice on VALIDATION data only (Section 6): the separate per-horizon classifiers
            with per-sample isotonic projection replace the hazard head only if, averaged over the 5 seeds, their
            AUPRC at 3 s is higher by >= 0.01 AND their adaptive ECE at 3 s is no more than 20% worse.
  Stage B   with the chosen head <h>: dom_<factor>_<level>_<h> (H2: 4 factors x 2 levels), match_<factor>_<h>
            (Amendment 1 size-matched larger level), and central_sep_noaug if 'sep' is chosen (H4 needs aug vs
            noaug with the chosen head)
  lgbm      registered LightGBM baseline, 5 seeds (rehearsal code: lgbm_baseline.fit_predict)
  p1        protocol P1 frame scores (cumulative risk at 3 s, Amendment 4) of the H1 models and LightGBM

DATA SETS PER MODEL (registered Section 2 "Pretraining data"; trips without a factor level, expected: all
baselines, are "unknown" for that factor and are excluded only from that factor's analyses)
  pretraining   public data (INTERACTION + highD training recordings; validation recordings for the best epoch)
                + SHRP2 SAFE BASELINE trips of the training split (validation-split baselines for the best epoch).
                Never a crash or near-crash event, never a test-split trip.
                H2 models (dom, match): minus every baseline trip of the held-out level; for match also minus the
                larger level's training baselines outside its size-matched subset. Identical pretraining sets give
                identical encoders, so one encoder is trained per distinct set and seed and shared (exact).
  fine-tuning   central: all training-split samples; selection on all validation samples.
                dom F=v: training samples of trips with F = v; selection on validation trips with F = v.
                match F: the larger level's training trips in random order (np.random.default_rng(seed)
                permutation of the sorted trip IDs), whole trips added until the training sample count reaches or
                first exceeds the smaller level's; selection on the larger level's full validation data.
PUBLIC SPLITS (fixed here): INTERACTION as in the rehearsal (recordings split by split_by_scene on the
  not-in-conflict samples, mapped back to all samples); highD by split_by_scene(scene, location) on all samples.
DIVERGENCE (Section 6): a non-finite loss -> that stage is re-run once with half the learning rate; if it fails
  again the model is replaced by the next seed (136, 137, ...); every event goes to results/shrp2_divergence.csv.

Usage (project folder; licensed data stay in data/):
  python src/shrp2/train_shrp2.py cache          # once: sample shards -> uncompressed cache + manifest (SHA-256)
  python src/shrp2/train_shrp2.py probe          # timing of 30 batches, projected hours (information only)
  python src/shrp2/train_shrp2.py run --stage A  # resumable; --only <prefix> for a subset
  python src/shrp2/train_shrp2.py select-head    # writes results/shrp2_head_selection.json/.txt
  python src/shrp2/train_shrp2.py run --stage B
  python src/shrp2/train_shrp2.py lgbm
  python src/shrp2/train_shrp2.py p1
  Sensitivity 8 (C23, full repetition): the same commands with the environment variable SHRP2_RUN=nomask; data, models,
  logs and the head choice then live in *_nomask folders and files (runbook)
Outputs: models/shrp2/enc_<set>_s<seed>.pt, models/shrp2/haz_<model>.pt (weights; H4 flicker test, P1 scores)
         data/processed/shrp2/preds/haz_<model>.npz   event_id, frame, split, logits (n,4), risk (n,4) for EVERY
                                                      validation and test sample (registered labels; P2)
         data/processed/shrp2/preds/lgbm_s<seed>.npz  same keys, risk (n,4)
         data/processed/shrp2/preds/p1_<model>.parquet event_id, frame, split, score (P1 frame scores)
         results/shrp2_jobs.csv (aggregate counts per model; shareable), results/shrp2_divergence.csv
"""
import argparse, csv, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC)); sys.path.insert(0, str(HERE))
from models.encoder import InteractionEncoder, ForecastHead
from models.hazard import HazardHead, hazard_nll
from train_pretrain import split_by_scene, predict as forecast, errors
from train_hazard import bce_per_horizon
from augment import augment
import calibration as C
import analysis.stats as ST

SEEDS = [131, 132, 133, 134, 135]
REPLACEMENT_SEEDS = [136, 137, 138, 139, 140]
BUDGET = (20, 10)                                   # registered: pretraining, fine-tuning epochs
FACTORS = {"lighting": "lighting_bin", "weather": "weather_bin", "surface": "surfaceCondition_bin",
           "traffic": "trafficDensity_bin"}
K3, BS_TRAIN, BS_EVAL = 2, 256, 1024
AUPRC_GAIN, ECE_WORSE = 0.01, 1.20                  # registered head-selection rule

RUN = os.environ.get("SHRP2_RUN", "")                 # "" = the registered run; "nomask" = sensitivity 8 (C23), a full repetition
if RUN not in ("", "nomask"):
    raise SystemExit(f"SHRP2_RUN must be empty or 'nomask', not {RUN!r}")
SFX = f"_{RUN}" if RUN else ""
SH = Path(f"data/processed/shrp2{SFX}")                # every path of a run carries its tag, so the runs never mix
CACHE, PRED = SH / "cache", SH / "preds"
MODELS = Path(f"models/shrp2{SFX}")
LOG, DIVLOG = Path(f"results/shrp2_jobs{SFX}.csv"), Path(f"results/shrp2_divergence{SFX}.csv")
HEADSEL = Path(f"results/shrp2_head_selection{SFX}.json")
PUBLIC = {"interaction": (Path("data/processed/pretrain_samples.npz"), Path("data/processed/hazard_labels.npz")),
          "highd": (Path("data/processed/highd_samples.npz"), None)}
FORCE_DIVERGE = set()           # tests only: {(stage, seed, lr_scale)} -> a NaN loss is injected at the first step


class Diverged(Exception):
    pass


# ================================================================================================ cache
def npz(p):
    """Read a whole .npz into a dict and CLOSE the file (Windows cannot replace or delete a file that is still open)."""
    with np.load(p, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _sources():
    src = {}
    for grp, sub, splits in (("shrp2", "samples", ("train", "val", "test")), ("p1", "samples_p1", ("val", "test"))):
        for s in splits:
            src[f"{grp}_{s}"] = sorted((SH / sub).glob(f"{s}_*.npz"))
    for name, (f, lab) in PUBLIC.items():
        src[name] = [f] + ([lab] if lab else [])
    return src


def build_cache():
    """Shards -> one uncompressed array per group (memory-mapped later) + meta table + manifest of source hashes."""
    t0 = time.time(); CACHE.mkdir(parents=True, exist_ok=True)
    man = {"sources": {}, "rows": {}}
    for grp, files in _sources().items():
        if not files or not all(f.exists() for f in files):
            if grp.startswith("shrp2"):
                raise SystemExit(f"missing sample shards for {grp} - run build_samples_shrp2.py first")
            print(f"  {grp}: not present (skipped)"); continue
        man["sources"][grp] = {str(f): sha256(f) for f in files}
        if grp in PUBLIC:
            n = _cache_public(grp, files)
        else:
            n = _cache_shrp2(grp, files)
        man["rows"][grp] = n
        print(f"  {grp}: {n:,} rows | {time.time() - t0:.0f}s", flush=True)
    (CACHE / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"cache written to {CACHE} | {time.time() - t0:.0f}s")


def _cache_shrp2(grp, files):
    keys = ["hist"] + (["fut", "fut_valid"] if grp.startswith("shrp2") else [])
    n = [len(npz(f)["frame"]) for f in files]
    shape = {}
    with np.load(files[0], allow_pickle=True) as z:
        for k in keys:
            shape[k] = z[k].shape[1:]
    out = {k: np.lib.format.open_memmap(CACHE / f"{grp}_{k}.npy", mode="w+", dtype=np.float32,
                                        shape=(sum(n),) + shape[k]) for k in keys}
    metas, lo = [], 0
    for f, m in zip(files, n):
        z = npz(f)
        for k in keys:
            out[k][lo:lo + m] = z[k]
        metas.append(pd.DataFrame({c: z[c] for c in ("event_id", "frame", "event_bin", "obs_bins", "current",
                                                    "split", "outcome_type", "location")}))
        lo += m
    for a in out.values():
        a.flush()
    meta = pd.concat(metas, ignore_index=True)
    for c in ("split", "outcome_type", "location"):
        meta[c] = meta[c].astype(str)
    meta.to_parquet(CACHE / f"{grp}_meta.parquet", index=False)
    return int(sum(n))


def interaction_split(scene, loc, current):
    """Rehearsal rule (run_jobs.load): split_by_scene on the not-in-conflict samples, mapped to recordings."""
    k = current == 0
    tr, va, te = split_by_scene(scene[k], loc[k])
    rec = pd.Series(np.select([tr, va, te], ["train", "val", "test"], "none"), index=scene[k]).groupby(level=0).first()
    return pd.Series(scene).map(rec).fillna("none").to_numpy()


def _cache_public(grp, files):
    d = npz(files[0])
    scene, loc = d["scene"].astype(str), d["location"].astype(str)
    if grp == "interaction":
        split = interaction_split(scene, loc, npz(files[1])["current"])
    else:
        tr, va, te = split_by_scene(scene, loc)
        split = np.select([tr, va, te], ["train", "val", "test"], "none")
    keep = np.flatnonzero((split == "train") | (split == "val"))          # public test recordings never used
    for k in ("hist", "fut", "fut_valid"):
        np.save(CACHE / f"{grp}_{k}.npy", np.ascontiguousarray(d[k][keep], dtype=np.float32))
    pd.DataFrame({"scene": scene[keep], "split": split[keep]}).to_parquet(CACHE / f"{grp}_meta.parquet", index=False)
    return int(len(keep))


def check_cache():
    f = CACHE / "manifest.json"
    if not f.exists():
        raise SystemExit("no cache - run: python src/shrp2/train_shrp2.py cache")
    man = json.loads(f.read_text())
    cur = _sources()
    for grp, files in man["sources"].items():
        if sorted(files) != sorted(map(str, cur.get(grp, []))):
            raise SystemExit(f"cache stale ({grp}: source files changed) - re-run the cache step")
        for p, h in files.items():
            if sha256(p) != h:
                raise SystemExit(f"cache stale ({p} changed since the cache was built) - re-run the cache step")
    for grp in cur:
        if grp not in man["sources"] and cur[grp] and all(Path(p).exists() for p in cur[grp]) and grp.startswith(("shrp2", "p1")):
            raise SystemExit(f"cache stale ({grp} appeared after the cache was built) - re-run the cache step")
    return man


# ================================================================================================ data
class Data:
    """Memory-mapped samples. Pretraining index space = public groups then SHRP2 train and val baselines."""

    def __init__(self, need_public=True):
        self.man = check_cache()
        self.g = {}
        for grp in ("shrp2_train", "shrp2_val", "shrp2_test"):
            self.g[grp] = self._open(grp, ("hist", "fut", "fut_valid"))
        T = pd.read_parquet(SH / "trips.parquet").set_index("event_id")
        self.levels = {f: T[c] if c in T else pd.Series(np.nan, index=T.index) for f, c in FACTORS.items()}
        if "driver_id" in T.columns:
            raise SystemExit("driver identifier present: write the driver rule into the freeze note first")
        self.pub = []
        if need_public:
            for name in PUBLIC:
                if name in self.man["sources"]:
                    self.pub.append((name, self._open(name, ("hist", "fut", "fut_valid"))))
            if not self.pub:
                raise SystemExit("no public pretraining data in the cache (INTERACTION / highD samples)")
        # pretraining index space
        parts, lo = [], 0
        for name, G in self.pub:
            parts.append((name, lo, G)); lo += len(G["meta"])
        for grp in ("shrp2_train", "shrp2_val"):
            parts.append((grp, lo, self.g[grp])); lo += len(self.g[grp]["meta"])
        self.parts, self.n_pre = parts, lo

    def _open(self, grp, keys):
        G = {k: np.load(CACHE / f"{grp}_{k}.npy", mmap_mode="r") for k in keys if (CACHE / f"{grp}_{k}.npy").exists()}
        G["meta"] = pd.read_parquet(CACHE / f"{grp}_meta.parquet")
        return G

    def level_of(self, factor, event_ids):
        """Factor level (0/1, NaN = unknown) of each trip."""
        return self.levels[factor].reindex(event_ids).to_numpy(float)

    # ------------------------------------------------------------ pretraining sets (global indices)
    def pretrain_sets(self, excl_level=None, keep_train_ids=None):
        """excl_level = (factor, level): baselines of that level are left out (held-out level, H2).
        keep_train_ids = (factor, level, ids): training baselines of that level are kept only if in ids (match)."""
        tr, va = [], []
        for name, lo, G in self.parts:
            m = G["meta"]
            if name in PUBLIC:
                tr.append(lo + np.flatnonzero(m["split"].to_numpy() == "train"))
                va.append(lo + np.flatnonzero(m["split"].to_numpy() == "val"))
                continue
            base = m["outcome_type"].to_numpy() == "baseline"
            if excl_level is not None:
                lv = self.level_of(excl_level[0], m["event_id"].to_numpy())
                base &= ~(lv == excl_level[1])
            if keep_train_ids is not None and name == "shrp2_train":
                f, v, ids = keep_train_ids
                lv = self.level_of(f, m["event_id"].to_numpy())
                base &= ~((lv == v) & ~np.isin(m["event_id"].to_numpy(), ids))
            (tr if name == "shrp2_train" else va).append(lo + np.flatnonzero(base))
        return np.concatenate(tr), np.concatenate(va)

    def pre_get(self, idx, key):
        out = None
        for name, lo, G in self.parts:
            hi = lo + len(G["meta"])
            m = (idx >= lo) & (idx < hi)
            if m.any():
                a = G[key][np.sort(idx[m] - lo)]
                order = np.argsort(np.argsort(idx[m] - lo, kind="stable"), kind="stable")
                if out is None:
                    out = np.empty((len(idx),) + a.shape[1:], np.float32)
                out[m] = a[order]
        return out

    # ------------------------------------------------------------ fine-tuning sets (indices into one group)
    def get(self, grp, idx, key="hist"):
        idx = np.asarray(idx)
        s = np.argsort(idx, kind="stable")
        out = np.empty((len(idx),) + self.g[grp][key].shape[1:], np.float32)
        out[s] = self.g[grp][key][idx[s]]
        return out

    def labels(self, grp, idx):
        m = self.g[grp]["meta"]
        return m["event_bin"].to_numpy()[idx], m["obs_bins"].to_numpy()[idx]


# ================================================================================================ jobs
def jobs(stage, head=None):
    J = []
    if stage == "A":
        for s in SEEDS:
            for h, aug in (("hazard", True), ("hazard", False), ("sep", True)):
                J.append(dict(name=f"central_{h}_{'aug' if aug else 'noaug'}_s{s}", kind="central", head=h, aug=aug, seed=s))
    elif stage == "B":
        if head not in ("hazard", "sep"):
            raise SystemExit("stage B needs the head selection (run select-head first)")
        for f in FACTORS:
            for v in (0, 1):
                for s in SEEDS:
                    J.append(dict(name=f"dom_{f}_{v}_{head}_s{s}", kind="dom", factor=f, level=v, head=head, aug=True, seed=s))
        for f in FACTORS:
            for s in SEEDS:
                J.append(dict(name=f"match_{f}_{head}_s{s}", kind="match", factor=f, head=head, aug=True, seed=s))
        if head == "sep":
            for s in SEEDS:
                J.append(dict(name=f"central_sep_noaug_s{s}", kind="central", head="sep", aug=False, seed=s))
    return J


def match_subset(D, factor, seed):
    """Amendment 1: larger level's training trips permuted by the model's seed, whole trips added until the
    training sample count reaches or first exceeds the smaller level's. Returns (big, small, ids) or None."""
    m = D.g["shrp2_train"]["meta"]
    lv = D.level_of(factor, m["event_id"].to_numpy())
    n = {v: int((lv == v).sum()) for v in (0, 1)}
    big, small = (0, 1) if n[0] >= n[1] else (1, 0)
    if n[small] == 0:
        return None
    per = pd.Series(lv == big).groupby(m["event_id"].to_numpy()).sum()
    ids = np.sort(per[per > 0].index.to_numpy())
    perm = ids[np.random.default_rng(seed).permutation(len(ids))]
    chosen, total = [], 0
    for e in perm:
        if total >= n[small]:
            break
        chosen.append(e); total += int(per[e])
    return big, small, np.array(chosen)


def job_sets(D, job, seed):
    """(pretrain_tr, pretrain_va, finetune_tr, finetune_va, info) - indices; see the module docstring."""
    mt, mv = D.g["shrp2_train"]["meta"], D.g["shrp2_val"]["meta"]
    info = {}
    if job["kind"] == "central":
        ptr, pva = D.pretrain_sets()
        ftr, fva = np.arange(len(mt)), np.arange(len(mv))
    elif job["kind"] == "dom":
        f, v = job["factor"], job["level"]
        ptr, pva = D.pretrain_sets(excl_level=(f, 1 - v))
        ftr = np.flatnonzero(D.level_of(f, mt["event_id"].to_numpy()) == v)
        fva = np.flatnonzero(D.level_of(f, mv["event_id"].to_numpy()) == v)
    else:
        f = job["factor"]
        ms = match_subset(D, f, seed)
        if ms is None:
            return None
        big, small, ids = ms
        ptr, pva = D.pretrain_sets(excl_level=(f, small), keep_train_ids=(f, big, ids))
        ftr = np.flatnonzero(np.isin(mt["event_id"].to_numpy(), ids))
        fva = np.flatnonzero(D.level_of(f, mv["event_id"].to_numpy()) == big)
        info = dict(level=big, matched_trips=len(ids))
    return ptr, pva, ftr, fva, info


def set_key(ptr, pva):
    h = hashlib.sha256(); h.update(np.asarray(ptr, np.int64).tobytes()); h.update(b"|"); h.update(np.asarray(pva, np.int64).tobytes())
    return h.hexdigest()[:12]


# ================================================================================================ training
def batches(n, bs, shuffle=True):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def _check(loss, stage, seed, lr_scale, first):
    if first and (stage, seed, lr_scale) in FORCE_DIVERGE:
        loss = loss * float("nan")
    if not torch.isfinite(loss):
        raise Diverged(stage)
    return loss


def pretrain(get, ti, vi, epochs, seed, lr_scale=1.0, max_batches=None):
    """= rehearsal run_jobs.pretrain; get(idx, key) reads samples. ti, vi: index arrays."""
    enc, head = InteractionEncoder(), ForecastHead(future=get(vi[:1], "fut").shape[1])
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3 * lr_scale, weight_decay=1e-4)
    ti, vi = torch.from_numpy(np.asarray(ti)), torch.from_numpy(np.asarray(vi))
    best, state, best_ep, first = np.inf, None, 0, True
    for ep in range(1, epochs + 1):
        enc.train(); head.train()
        for b, i in enumerate(batches(len(ti), BS_TRAIN)):
            if max_batches and b >= max_batches:
                break
            j = ti[i].numpy()
            X, Y, V = (torch.from_numpy(get(j, k)) for k in ("hist", "fut", "fut_valid"))
            v = V[..., None]
            loss = (Fn.smooth_l1_loss(forecast(enc, head, X), Y, reduction="none") * v).sum() / v.sum().clamp(min=1) / 2
            loss = _check(loss, "pretrain", seed, lr_scale, first); first = False
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval(); A = []
        with torch.no_grad():
            for i in batches(len(vi), BS_EVAL, False):
                j = vi[i].numpy()
                A.append(errors(forecast(enc, head, torch.from_numpy(get(j, "hist"))), torch.from_numpy(get(j, "fut")),
                                torch.from_numpy(get(j, "fut_valid")))[0])
        ade = torch.cat(A).mean().item()
        if ade < best:
            best, best_ep, state = ade, ep, {k: t.clone() for k, t in enc.state_dict().items()}
    if state is None:
        raise Diverged("pretrain-val")
    enc.load_state_dict(state)
    return enc, best, best_ep


def logits_of(enc, head, get, idx):
    enc.eval(); head.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(idx), BS_EVAL):
            out.append(head(enc(torch.from_numpy(get(idx[s:s + BS_EVAL])))))
    return torch.cat(out).numpy() if out else np.zeros((0, 4), np.float32)


def bce_np(logits, ev, ob):
    """Validation loss of the separate-classifier head: bce_per_horizon over all samples (exact global mean)."""
    k = np.arange(logits.shape[1])[None, :]
    y = ((ev[:, None] >= 0) & (ev[:, None] <= k)).astype(float)
    known = (y > 0) | (ob[:, None] >= k + 1)
    l = np.logaddexp(0.0, logits) - y * logits
    return float((l * known).sum() / max(known.sum(), 1))


def finetune(enc, get, labels, ti, vi, epochs, aug, seed, head_kind="hazard", lr_scale=1.0, max_batches=None):
    """= rehearsal run_jobs.finetune (head_kind 'sep': separate per-horizon classifiers, BCE loss as in
    train_hazard.py --separate-heads). Selection: lowest validation loss."""
    head = HazardHead(enc.d_model, 4)
    loss_fn = hazard_nll if head_kind == "hazard" else bce_per_horizon
    val_loss = C.hazard_nll_np if head_kind == "hazard" else bce_np
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4 * lr_scale},
                             {"params": head.parameters(), "lr": 1e-3 * lr_scale}], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    ti, vi = torch.from_numpy(np.asarray(ti)), np.asarray(vi)
    Ev, Ov = labels(vi)
    best, state, best_ep, first = np.inf, None, 0, True
    for ep in range(1, epochs + 1):
        enc.train(); head.train()
        for b, i in enumerate(batches(len(ti), BS_TRAIN)):
            if max_batches and b >= max_batches:
                break
            j = ti[i].numpy(); xb = get(j)
            if aug:
                xb = augment(xb, rng)
            e, o = labels(j)
            loss = loss_fn(head(enc(torch.from_numpy(xb))), torch.from_numpy(e), torch.from_numpy(o))
            loss = _check(loss, "finetune", seed, lr_scale, first); first = False
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()
        lg = logits_of(enc, head, get, vi)
        nll = val_loss(lg, Ev, Ov)
        if nll < best:
            best, best_ep = nll, ep
            state = ({k: t.clone() for k, t in enc.state_dict().items()},
                     {k: t.clone() for k, t in head.state_dict().items()})
    if state is None:
        raise Diverged("finetune-val")
    enc.load_state_dict(state[0]); head.load_state_dict(state[1])
    return enc, head, best, best_ep


def isotonic_rows(p):
    """Per-sample isotonic (non-decreasing, equal weights) projection across horizons: the least-squares
    non-decreasing fit, by the max-min formula fit_i = max_{j<=i} min_{k>=i} mean(p_j..p_k)."""
    p = np.asarray(p, float); n, K = p.shape
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(p, 1)], 1)
    mean = lambda j, k: (cs[:, k + 1] - cs[:, j]) / (k - j + 1)
    out = np.empty_like(p)
    for i in range(K):
        out[:, i] = np.max(np.stack([np.min(np.stack([mean(j, k) for k in range(i, K)], 1), 1) for j in range(i + 1)], 1), 1)
    return out


def risk_of(logits, head_kind):
    return C.risk_from_logits(logits) if head_kind == "hazard" else isotonic_rows(1.0 / (1.0 + np.exp(-logits)))


# ================================================================================================ running
LOG_FIELDS = ["model", "kind", "factor", "level", "head", "augment", "seed_slot", "seed_used", "pre_epochs", "haz_epochs",
              "matched_trips", "encoder_set", "pretrain_train", "pretrain_val", "train_samples", "val_samples",
              "train_trips", "best_pre_epoch", "best_haz_epoch", "val_ade", "val_loss", "pretrain_seconds",
              "finetune_seconds", "lr_scale", "trees_per_horizon", "note", "smoke"]
DIV_FIELDS = ["model", "seed", "stage", "lr_scale"]


def log_row(path, row):
    """Fixed columns per log, so rows of different kinds stay in one readable table."""
    fields = DIV_FIELDS if path == DIVLOG else LOG_FIELDS
    extra = set(row) - set(fields)
    if extra:
        raise ValueError(f"unknown log columns {extra}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=""); w.writeheader() if new else None; w.writerow(row)


def encoder_for(D, ptr, pva, seed, pre_ep, smoke, lr_scale=1.0):
    key = set_key(ptr, pva)
    f = MODELS / f"enc_{key}_s{seed}{'' if lr_scale == 1.0 else '_lr' + str(lr_scale)}{'_smoke' if smoke else ''}.pt"
    if f.exists():
        ck = torch.load(f, map_location="cpu")
        enc = InteractionEncoder(); enc.load_state_dict(ck["encoder"])
        return enc, ck["ade"], ck["epoch"], key, 0.0
    if f.with_suffix(".diverged").exists():          # a shared encoder that already diverged is not re-run
        raise Diverged("pretrain (recorded)")
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
    try:
        enc, ade, ep = pretrain(D.pre_get, ptr, pva, pre_ep, seed, lr_scale, 5 if smoke else None)
    except Diverged:
        f.with_suffix(".diverged").write_text("non-finite loss\n"); raise
    torch.save({"encoder": enc.state_dict(), "ade": ade, "epoch": ep, "n_train": len(ptr), "n_val": len(pva)}, f)
    return enc, ade, ep, key, time.time() - t0


def save_predictions(name, enc, head, D, head_kind, seed_used):
    parts = []
    for grp, s in (("shrp2_val", "val"), ("shrp2_test", "test")):
        idx = np.arange(len(D.g[grp]["meta"]))
        lg = logits_of(enc, head, lambda j: D.get(grp, j), idx)
        m = D.g[grp]["meta"]
        parts.append((m["event_id"].to_numpy(), m["frame"].to_numpy(), np.full(len(idx), s), lg))
    ev, fr, sp, lg = (np.concatenate([p[k] for p in parts]) for k in range(4))
    PRED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED / f"haz_{name}.npz", event_id=ev, frame=fr, split=sp, logits=lg.astype(np.float32),
                        risk=risk_of(lg, head_kind).astype(np.float32), head=head_kind, seed_used=seed_used)


def train_model(D, job, seed, pre_ep, haz_ep, smoke):
    """One model with the divergence rule at each stage. Returns (enc, head, record) or raises Diverged."""
    sets = job_sets(D, job, seed)
    if sets is None:
        return None
    ptr, pva, ftr, fva, info = sets
    if len(fva) == 0 or len(ftr) == 0:
        return None
    rec = dict(info)
    for lr_scale in (1.0, 0.5):
        try:
            enc, ade, pep, key, t_pre = encoder_for(D, ptr, pva, seed, pre_ep, smoke, lr_scale)
            break
        except Diverged:
            log_row(DIVLOG, dict(model=job["name"], seed=seed, stage="pretrain", lr_scale=lr_scale))
    else:
        raise Diverged("pretrain twice")
    for lr_scale in (1.0, 0.5):
        try:
            torch.manual_seed(seed)
            e = InteractionEncoder(); e.load_state_dict(enc.state_dict()); t1 = time.time()
            e, head, nll, hep = _finetune_split(e, D, ftr, fva, haz_ep, job, seed, lr_scale, smoke)
            break
        except Diverged:
            log_row(DIVLOG, dict(model=job["name"], seed=seed, stage="finetune", lr_scale=lr_scale))
    else:
        raise Diverged("finetune twice")
    rec.update(encoder_set=key, pretrain_train=len(ptr), pretrain_val=len(pva), train_samples=len(ftr),
               val_samples=len(fva), train_trips=int(D.g["shrp2_train"]["meta"]["event_id"].iloc[ftr].nunique()),
               best_pre_epoch=pep, best_haz_epoch=hep, val_ade=round(float(ade), 4), val_loss=round(float(nll), 6),
               pretrain_seconds=round(t_pre), finetune_seconds=round(time.time() - t1), lr_scale=lr_scale)
    return e, head, rec


def _finetune_split(enc, D, ftr, fva, haz_ep, job, seed, lr_scale, smoke):
    """finetune() reads training batches from the training group and validation chunks from the validation group;
    the two index spaces are kept apart by offsetting validation indices (never mixed, never the test group)."""
    nt = len(D.g["shrp2_train"]["meta"])

    def get(j):
        j = np.asarray(j); out = np.empty((len(j),) + D.g["shrp2_train"]["hist"].shape[1:], np.float32)
        a = j < nt
        if a.any():
            out[a] = D.get("shrp2_train", j[a])
        if (~a).any():
            out[~a] = D.get("shrp2_val", j[~a] - nt)
        return out

    def labels(j):
        j = np.asarray(j); a = j < nt
        e = np.empty(len(j), np.int64); o = np.empty(len(j), np.int64)
        if a.any():
            e[a], o[a] = D.labels("shrp2_train", j[a])
        if (~a).any():
            e[~a], o[~a] = D.labels("shrp2_val", j[~a] - nt)
        return e, o
    return finetune(enc, get, labels, ftr, fva + nt, haz_ep, job["aug"], seed, job["head"], lr_scale,
                    5 if smoke else None)


def done(name):
    return (PRED / f"haz_{name}.npz").exists() and (MODELS / f"haz_{name}.pt").exists()


def run_job(D, job, pre_ep, haz_ep, smoke):
    name = job["name"] + ("_smoke" if smoke else "")
    if done(name):
        print(f"  [skip] {name} already done"); return
    tried = []
    for seed in [job["seed"]] + REPLACEMENT_SEEDS:
        if seed != job["seed"] and seed in used_replacements(job):
            continue
        tried.append(seed)
        try:
            r = train_model(D, job, seed, pre_ep, haz_ep, smoke)
        except Diverged:
            log_row(DIVLOG, dict(model=job["name"], seed=seed, stage="replaced", lr_scale=""))
            continue
        if r is None:
            log_row(LOG, dict(model=name, kind=job["kind"], note="not trainable (empty training or validation set)"))
            print(f"  {name}: not trainable (empty set)"); return
        enc, head, rec = r
        torch.save({"encoder": enc.state_dict(), "head": head.state_dict(), "head_kind": job["head"], "seed_used": seed},
                   MODELS / f"haz_{name}.pt")
        save_predictions(name, enc, head, D, job["head"], seed)
        row = dict(model=name, kind=job["kind"], factor=job.get("factor", ""), level=rec.pop("level", job.get("level", "")),
                   head=job["head"], augment=job["aug"], seed_slot=job["seed"], seed_used=seed,
                   pre_epochs=pre_ep, haz_epochs=haz_ep, **rec, smoke=smoke)
        log_row(LOG, row)
        print(f"  {name}: train {row['train_samples']:,} | best epochs {row['best_pre_epoch']}/{row['best_haz_epoch']} | "
              f"val ADE {row['val_ade']:.3f} loss {row['val_loss']:.4f} | {row['pretrain_seconds']}s + {row['finetune_seconds']}s", flush=True)
        return
    raise SystemExit(f"{name}: diverged with seeds {tried} - stop and report")


def used_replacements(job):
    """Replacement seeds already taken by this model family (tried and failed, or in use by another slot)."""
    fam = job["name"].rsplit("_s", 1)[0]; used = set()
    if DIVLOG.exists():
        d = pd.read_csv(DIVLOG)
        if len(d):
            used |= set(d.loc[d["model"].astype(str).str.rsplit("_s", n=1).str[0] == fam, "seed"].astype(int))
    if LOG.exists():
        d = pd.read_csv(LOG)
        if len(d) and "seed_used" in d:
            d = d[d["model"].astype(str).str.replace("_smoke", "").str.rsplit("_s", n=1).str[0] == fam]
            used |= set(pd.to_numeric(d["seed_used"], errors="coerce").dropna().astype(int))
    return used - {job["seed"]}


# ================================================================================================ head selection
def val_metrics(name):
    z = npz(PRED / f"haz_{name}.npz")
    meta = pd.read_parquet(CACHE / "shrp2_val_meta.parquet")
    v = z["split"] == "val"
    if not (np.array_equal(z["event_id"][v], meta["event_id"].to_numpy()) and np.array_equal(z["frame"][v], meta["frame"].to_numpy())):
        raise SystemExit(f"{name}: validation predictions not aligned with the cache")
    y, known = C.labels_at(meta["event_bin"].to_numpy(), meta["obs_bins"].to_numpy(), K3)
    r = z["risk"][v][known, K3].astype(float); y = y[known]
    return ST.average_precision(y, r), ST.ece_adaptive(y, r)


def head_rule(auprc_h, ece_h, auprc_s, ece_s):
    """Registered rule (Section 6), on seed means. Rounded to 10 decimals so 0.01 exactly counts as >= 0.01."""
    gain = round(float(np.mean(auprc_s) - np.mean(auprc_h)), 10)
    ratio = round(float(np.mean(ece_s) / np.mean(ece_h)), 10)
    return ("sep" if (gain >= AUPRC_GAIN and ratio <= ECE_WORSE) else "hazard"), gain, ratio


def select_head(smoke=False):
    suf = "_smoke" if smoke else ""
    M = {h: [val_metrics(f"central_{h}_aug_s{s}{suf}") for s in SEEDS] for h in ("hazard", "sep")}
    for h in M:
        if len(M[h]) != len(SEEDS):
            raise SystemExit("all five seeds of both heads are needed")
    choice, gain, ratio = head_rule([a for a, _ in M["hazard"]], [e for _, e in M["hazard"]],
                                    [a for a, _ in M["sep"]], [e for _, e in M["sep"]])
    res = dict(choice=choice, auprc_gain=gain, ece_ratio=ratio, rule=f"sep if gain >= {AUPRC_GAIN} and ratio <= {ECE_WORSE}",
               per_seed={h: [dict(seed=s, auprc3=round(a, 6), ece_adaptive3=round(e, 6)) for s, (a, e) in zip(SEEDS, M[h])] for h in M},
               data="validation split only", smoke=smoke)
    HEADSEL.parent.mkdir(exist_ok=True); HEADSEL.write_text(json.dumps(res, indent=1))
    lines = ["SHRP2 RISK-HEAD SELECTION (registered Section 6; validation data only; aggregate, shareable)",
             f"hazard head: AUPRC@3s mean {np.mean([a for a, _ in M['hazard']]):.4f} | adaptive ECE@3s mean {np.mean([e for _, e in M['hazard']]):.5f}",
             f"separate + isotonic: AUPRC@3s mean {np.mean([a for a, _ in M['sep']]):.4f} | adaptive ECE@3s mean {np.mean([e for _, e in M['sep']]):.5f}",
             f"AUPRC gain {gain:+.4f} (needs >= {AUPRC_GAIN}); ECE ratio {ratio:.3f} (needs <= {ECE_WORSE})", f"CHOICE: {choice}"]
    HEADSEL.with_suffix(".txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    return res


def chosen_head(smoke=False):
    if not HEADSEL.exists():
        raise SystemExit("run select-head first (after stage A)")
    r = json.loads(HEADSEL.read_text())
    if r.get("smoke", False) != smoke:
        raise SystemExit("head selection file is from a smoke run - re-run select-head on the real models")
    return r["choice"]


# ================================================================================================ P1, LightGBM
def p1_scores(names):
    """Amendment 4: cumulative risk at 3 s on every P1 sample (validation and test events, all valid frames)."""
    PRED.mkdir(parents=True, exist_ok=True)
    G = {grp: (np.load(CACHE / f"{grp}_hist.npy", mmap_mode="r"), pd.read_parquet(CACHE / f"{grp}_meta.parquet"))
         for grp in ("p1_val", "p1_test")}
    for name in names:
        ck = torch.load(MODELS / f"haz_{name}.pt", map_location="cpu")
        enc, head = InteractionEncoder(), HazardHead(64, 4)
        enc.load_state_dict(ck["encoder"]); head.load_state_dict(ck["head"])
        parts = []
        for grp, (X, m) in G.items():
            lg = logits_of(enc, head, lambda j: np.asarray(X[j], np.float32), np.arange(len(m)))
            parts.append(pd.DataFrame(dict(event_id=m["event_id"].to_numpy(), frame=m["frame"].to_numpy(),
                                           split=grp.split("_")[1], score=risk_of(lg, ck["head_kind"])[:, K3])))
        out = pd.concat(parts, ignore_index=True)
        if out.duplicated(["event_id", "frame"]).any():
            raise SystemExit("P1 samples have duplicate (event_id, frame)")
        out.to_parquet(PRED / f"p1_{name}.parquet", index=False)
        print(f"  p1_{name}: {len(out):,} frame scores")


def run_lgbm():
    """Registered LightGBM (rehearsal lgbm_baseline.fit_predict, unchanged) on the SHRP2 features; P1 rows are
    predicted by the same classifiers (they are appended to X but are in neither the training nor the validation
    mask)."""
    from rehearsal import lgbm_baseline as LG
    parts = []
    for s in ("train", "val", "test"):
        f = SH / f"features_{s}.parquet"
        if not f.exists():
            raise SystemExit(f"missing {f} - run features_shrp2.py")
        parts.append(pd.read_parquet(f).assign(_part=s))
    for s in ("val", "test"):
        f = SH / f"features_p1_{s}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f).assign(_part=f"p1_{s}"))
    A = pd.concat(parts, ignore_index=True)
    X = A[LG.FEATURES].reset_index(drop=True)
    ev, ob, part = A["event_bin"].to_numpy(), A["obs_bins"].to_numpy(), A["_part"].to_numpy()
    tr, va = part == "train", part == "val"
    PRED.mkdir(parents=True, exist_ok=True); risks = {}
    for seed in SEEDS:
        risk, iters = LG.fit_predict(X, ev, ob, tr, va, seed); risks[seed] = risk
        sel = (part == "val") | (part == "test")
        np.savez_compressed(PRED / f"lgbm_s{seed}.npz", event_id=A["event_id"].to_numpy()[sel], frame=A["frame"].to_numpy()[sel],
                            split=part[sel], risk=risk[sel].astype(np.float32))
        p1 = np.isin(part, ["p1_val", "p1_test"])
        if p1.any():
            pd.DataFrame(dict(event_id=A["event_id"].to_numpy()[p1], frame=A["frame"].to_numpy()[p1],
                              split=np.char.replace(part[p1].astype(str), "p1_", ""), score=risk[p1, K3])) \
              .to_parquet(PRED / f"p1_lgbm_s{seed}.parquet", index=False)
        log_row(LOG, dict(model=f"lgbm_s{seed}", kind="lgbm", trees_per_horizon=" ".join(map(str, iters)),
                          train_samples=int(tr.sum()), val_samples=int(va.sum())))
        print(f"  lgbm seed {seed}: trees per horizon {iters}")
    same = all(np.array_equal(risks[SEEDS[0]], risks[s]) for s in SEEDS[1:])
    print(f"identical LightGBM predictions across the 5 seeds: {same}")
    return risks


# ================================================================================================ probe
def probe(D):
    """Seconds per sample from 30 batches of the central sets; projected hours for stages A and B (information only:
    the registered budget is fixed and is not changed by this number)."""
    nb = 30; torch.manual_seed(131)
    ptr, pva, ftr, fva, _ = job_sets(D, jobs("A")[0], 131)
    t0 = time.time(); pretrain(D.pre_get, ptr[:nb * BS_TRAIN], pva[:2048], 1, 131, max_batches=nb); tp = time.time() - t0
    enc = InteractionEncoder(); t0 = time.time()
    _finetune_split(enc, D, ftr[:nb * BS_TRAIN], fva[:2048], 1, dict(aug=True, head="hazard"), 131, 1.0, False)
    th = time.time() - t0
    cp, ch = tp / (nb * BS_TRAIN + 2048), th / (nb * BS_TRAIN + 2048)
    keys, tot = set(), 0.0
    for stage, head in (("A", None), ("B", "hazard")):
        for j in jobs(stage, head):
            s = job_sets(D, j, j["seed"])
            if s is None:
                continue
            k = (set_key(s[0], s[1]), j["seed"])
            if k not in keys:
                keys.add(k); tot += BUDGET[0] * (len(s[0]) + len(s[1])) * cp
            tot += BUDGET[1] * (len(s[2]) + len(s[3])) * ch + 2 * ch * sum(len(D.g[g]["meta"]) for g in ("shrp2_val", "shrp2_test"))
    print(f"probe: {cp * 1e3:.2f} ms/sample pretraining, {ch * 1e3:.2f} ms/sample fine-tuning (incl. validation)")
    print(f"distinct encoders {len(keys)}; projected stages A + B at the registered budget: {tot / 3600:.1f} hours "
          f"(rehearsal: real time was ~1.3 x the probe)")


# ================================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "probe", "run", "select-head", "lgbm", "p1"])
    ap.add_argument("--stage", choices=["A", "B"])
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true", help="code check: 1 + 1 epochs, 5 batches; outputs end in _smoke")
    a = ap.parse_args(argv)
    torch.set_num_threads(os.cpu_count() or 1)
    MODELS.mkdir(parents=True, exist_ok=True)
    if a.cmd == "cache":
        return build_cache()
    if a.cmd == "select-head":
        return select_head(a.smoke)
    if a.cmd == "lgbm":
        return run_lgbm()
    if a.cmd == "p1":
        check_cache()
        h = chosen_head(a.smoke); suf = "_smoke" if a.smoke else ""
        return p1_scores([f"central_{h}_aug_s{s}{suf}" for s in SEEDS])
    D = Data(need_public=True)
    if a.cmd == "probe":
        return probe(D)
    if a.stage is None:
        raise SystemExit("choose --stage A or B")
    pre_ep, haz_ep = (1, 1) if a.smoke else BUDGET
    todo = [j for j in jobs(a.stage, chosen_head(a.smoke) if a.stage == "B" else None) if j["name"].startswith(a.only)]
    print(f"{len(todo)} models, budget {pre_ep} + {haz_ep} epochs")
    for k, job in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {job['name']}", flush=True)
        run_job(D, job, pre_ep, haz_ep, a.smoke)


if __name__ == "__main__":
    main()
