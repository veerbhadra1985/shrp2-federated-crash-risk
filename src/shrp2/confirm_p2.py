"""
SHRP2 CONFIRMATORY ANALYSIS, PART P2 (frame-level predictions; pre-registration osf.io/y5zrp Sections 5-7, Amendment 1,
Amendments 2-3 for the H5 input, Amendment 4 Appendix A). Written and tested on synthetic data before any SHRP2 access.
Uses ONLY the analysis engine of the rehearsal (src/analysis/stats.py: cluster bootstrap, paired one-sided test,
Holm, AUPRC, adaptive ECE, Clopper-Pearson) and the calibration module (src/calibration.py).

REGISTERED TESTS COMPUTED HERE (the nine-test family also needs H1a and H1b under P1: confirm_p1.py)
  H2    mean over testable factor-direction pairs of log(ECE_adaptive(model trained elsewhere) / ECE_adaptive(model
        trained in-domain)) at the primary horizon, both models on the SAME held-out test samples; mean over seeds
        inside each pair; testable pair = >= 50 held-out test EVENTS with a positive frame at the primary horizon (C15);
        H2 testable if >= 4 of the 8 pairs are
  H3a   pooled adaptive ECE: FedAvg - FedPer > 0;   H3b1 pooled AUPRC FedAvg - local > 0;   H3b2 FedPer - local > 0
  H4a   AUPRC augmented - non-augmented on test frames with natural radar dropout (an object present in the 2-s window
        missing in >= 20% of its frames; registered definition) > 0
  H4b   the same under 50% synthetic frame flicker (identical perturbed inputs for every model) > 0
  H4c   clean AUPRC augmented - non-augmented > -0.02 (non-inferiority)
  Bootstrap: paired cluster bootstrap over test TRIPS (units), B = 2,000, seed 131, statistic = metric difference
  averaged over the 5 seeds; p = (1 + #{D*_b <= delta0}) / (B + 1); > 5% undefined resamples -> "unreliable"
  (registered). A subset (H3, H4a) is resampled over its own trips, as in the rehearsal.
ALSO HERE (not in the Holm family)
  H1 key secondary endpoint (P2): AUPRC at the primary horizon, model - LightGBM and model - best classical measure
     (chosen on VALIDATION among TTC, DRAC, TTC2D, TTC2D moving), as in the rehearsal; model - GSSM under P2 is in
     confirm_p1.py (GSSM scores exist only in the GSSM workspace).
  H2 size-matched (Amendment 1) and H2 secondary (conformal false-alarm guarantee, alpha 0.05 and 0.01, thresholds
     from the source model's own validation negatives; violation = lower bound of the 95% cluster-bootstrap interval of
     the seed-mean false-alarm rate > alpha (Amendment 1, C17); Clopper-Pearson per seed reported).
  Sensitivity analyses (registration Section 7; numbers as there): 1 crash-only outcome (C16); 3 horizons 1, 2, 5 s
     (those retained by the audit); 4 the other risk head (central models; C18); 5 equal-width ECE; 6 temperature and
     isotonic recalibration on each model's own validation data (C19); 7 baselines restricted to trips with >= 1
     detected object. 2 and 8 are in confirm_p1.py and in the runbook.
  h5-input: the input file of src/phase2/run_h5.py (Amendments 2-3), from the H1 models at 3 s.
  holm: the final Holm correction over the nine registered tests (needs confirm_p1.py's results), then H5 with the
     12-test sensitivity.

Usage (project environment, after training; licensed data stay in data/; outputs are aggregate):
  python src/shrp2/confirm_p2.py run [--B 2000] [--primary-horizon 3] [--horizons 1,2,5]
  python src/shrp2/confirm_p2.py h5-input
  python src/shrp2/confirm_p2.py holm            (after confirm_p1.py)
  Sensitivity 8 (C23, full repetition): the same commands with SHRP2_RUN=nomask (inputs and outputs carry "_nomask")
Outputs: results/shrp2_confirm_p2_tests.csv (all tests, primary and sensitivity), results/shrp2_confirm_h2_pairs.csv,
         results/shrp2_confirm_h2_conformal.csv, results/shrp2_confirm_p2_summary.txt, results/shrp2_confirm_holm.csv
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import analysis.stats as S
import calibration as C
import train_shrp2 as T

SEEDS = T.SEEDS
HORIZONS = [1, 2, 3, 5]
MIN_EVENTS, ALPHAS, NONINF, DROP_SHARE = 50, (0.05, 0.01), -0.02, 0.2
FACTORS = list(T.FACTORS)
REGISTERED = ["H1a", "H1b", "H2", "H3a", "H3b1", "H3b2", "H4a", "H4b", "H4c"]
OUT = Path("results")


# ================================================================================================ data
class Bank:
    """Labels, trip levels and every prediction, aligned to the cache order of the validation and test samples."""

    def __init__(self, head=None):
        self.meta = {s: pd.read_parquet(T.CACHE / f"shrp2_{s}_meta.parquet") for s in ("val", "test")}
        self.trips = pd.read_parquet(T.SH / "trips.parquet").set_index("event_id")
        self.head = head or T.chosen_head(False)
        self.other = "sep" if self.head == "hazard" else "hazard"
        self._m = {}

    def ev(self, s):
        return self.meta[s]["event_id"].to_numpy()

    def labels(self, s, k):
        m = self.meta[s]
        y, known = C.labels_at(m["event_bin"].to_numpy(), m["obs_bins"].to_numpy(), k)
        return y, known

    def level(self, f, s):
        col = T.FACTORS[f]
        lv = self.trips[col] if col in self.trips else pd.Series(np.nan, index=self.trips.index)
        return lv.reindex(self.ev(s)).to_numpy(float)

    def outcome(self, s):
        return self.trips["outcome_type"].reindex(self.ev(s)).to_numpy().astype(str)

    def _align(self, z, s, name):
        sel = np.asarray(z["split"]).astype(str) == s
        m = self.meta[s]
        if not (np.array_equal(np.asarray(z["event_id"])[sel], m["event_id"].to_numpy())
                and np.array_equal(np.asarray(z["frame"])[sel], m["frame"].to_numpy())):
            raise SystemExit(f"{name}: predictions are not aligned with the {s} samples of the cache")
        return sel

    def model(self, name):
        """dict(val=(risk, logits), test=(risk, logits)); None if the model was not trained (file absent)."""
        if name in self._m:
            return self._m[name]
        f = T.PRED / f"haz_{name}.npz"
        if not f.exists():
            self._m[name] = None; return None
        z = T.npz(f); out = {"head": str(z["head"])}
        for s in ("val", "test"):
            sel = self._align(z, s, name)
            out[s] = (z["risk"][sel].astype(float), z["logits"][sel].astype(float))
        self._m[name] = out
        return out

    def keyed(self, df, s, cols):
        """Rows of df (event_id, frame, cols) re-ordered to the cache order of split s (NaN where absent)."""
        idx = pd.MultiIndex.from_arrays([self.meta[s]["event_id"].to_numpy(), self.meta[s]["frame"].to_numpy()])
        d = df.set_index(["event_id", "frame"])
        if d.index.duplicated().any():
            raise SystemExit("duplicate (event_id, frame) rows")
        return d.reindex(idx)[cols].to_numpy()

    def lgbm(self, seed):
        z = T.npz(T.PRED / f"lgbm_s{seed}.npz")
        out = {}
        for s in ("val", "test"):
            sel = np.asarray(z["split"]).astype(str) == s
            df = pd.DataFrame({"event_id": z["event_id"][sel], "frame": z["frame"][sel]})
            for j in range(4):
                df[f"r{j}"] = z["risk"][sel][:, j]
            out[s] = self.keyed(df, s, [f"r{j}" for j in range(4)]).astype(float)
        return out

    def classical(self):
        """Rehearsal scores (run_hypotheses.classical_scores): larger = riskier; no leader / no conflict -> safest."""
        out = {}
        for s in ("val", "test"):
            F = pd.read_parquet(T.SH / f"features_{s}.parquet")
            X = self.keyed(F, s, ["ttc", "drac", "ttc2d", "ttc2d_mov"]).astype(float)
            out[s] = {"TTC": -np.nan_to_num(X[:, 0], nan=1e3), "DRAC": np.nan_to_num(X[:, 1], nan=0.0),
                      "TTC2D": -np.nan_to_num(X[:, 2], nan=1e3), "TTC2D_moving": -np.nan_to_num(X[:, 3], nan=1e3)}
        return out

    def fed(self, regime, seed):
        f = T.PRED / f"fed_{regime}_s{seed}.npz"
        if not f.exists():
            return None
        z = T.npz(f); out = {"head": str(z["head"])}
        for s in ("val", "test"):
            sel = np.asarray(z["split"]).astype(str) == s
            df = pd.DataFrame({"event_id": z["event_id"][sel], "frame": z["frame"][sel], "client": z["client"][sel]})
            for j in range(4):
                df[f"r{j}"] = z["risk"][sel][:, j]; df[f"l{j}"] = z["logits"][sel][:, j]
            A = self.keyed(df, s, [f"r{j}" for j in range(4)] + [f"l{j}" for j in range(4)] + ["client"])
            out[s] = (A[:, :4].astype(float), A[:, 4:8].astype(float), A[:, 8])
        return out


# ================================================================================================ calibration
def calibrate(head, logits_fit, ev, ob, logits_apply, kind):
    """Recalibrated risk (n, 4). Hazard head: src/calibration.py Temperature / Isotonic. Separate head (C19): one
    temperature on all per-horizon logits minimising the per-horizon BCE (grid 0.25-4, as registered), then the
    per-sample isotonic projection; isotonic = per-horizon isotonic regression on the projected risk + running max."""
    if head == "hazard":
        cal = (C.Temperature() if kind == "temperature" else C.Isotonic()).fit(logits_fit, ev, ob)
        return cal.transform(logits_apply)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    if kind == "temperature":
        grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 81))
        Tt = float(grid[int(np.argmin([T.bce_np(logits_fit / t, ev, ob) for t in grid]))])
        return T.isotonic_rows(sig(logits_apply / Tt))
    from sklearn.isotonic import IsotonicRegression
    rf, ra = T.isotonic_rows(sig(logits_fit)), T.isotonic_rows(sig(logits_apply))
    out = []
    for k in range(4):
        y, ok = C.labels_at(ev, ob, k)
        out.append(IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(rf[ok, k], y[ok]).predict(ra[:, k]))
    return np.maximum.accumulate(np.column_stack(out), axis=1)


# ================================================================================================ helpers
def natural_dropout(hist):
    """Registered H4a definition (= rehearsal run_hypotheses.natural_dropout)."""
    nb = hist[:, 1:, :, 8] > 0
    present = nb.any(-1)
    missing_share = 1 - nb.mean(-1)
    return (present & (missing_share >= DROP_SHARE)).any(-1)


def test_hist_pass(bank, names, chunk=50000):
    """One pass over the test histories: natural-dropout flag per test sample, and every named central model's risk
    (all horizons) under 50% frame flicker. The perturbation is drawn ONCE per chunk (np.random.default_rng(131),
    chunks in cache order) and the same perturbed inputs go to every model (rehearsal: flicker_risk)."""
    import torch
    from robustness import perturb
    X = np.load(T.CACHE / "shrp2_test_hist.npy", mmap_mode="r")
    n = len(bank.meta["test"]); drop = np.zeros(n, bool)
    rng = np.random.default_rng(131)
    models = {}
    for nm in names:
        ck = torch.load(T.MODELS / f"haz_{nm}.pt", map_location="cpu")
        e, h = T.InteractionEncoder(), T.HazardHead(64, 4)
        e.load_state_dict(ck["encoder"]); h.load_state_dict(ck["head"]); e.eval(); h.eval()
        models[nm] = (e, h, ck["head_kind"])
    fl = {nm: np.zeros((n, 4)) for nm in names}
    for lo in range(0, n, chunk):
        H = np.asarray(X[lo:lo + chunk], np.float32)
        drop[lo:lo + len(H)] = natural_dropout(H)
        Hp = perturb(H, "frames", 0.5, rng)
        for nm, (e, h, hk) in models.items():
            lg = T.logits_of(e, h, lambda j: Hp[j], np.arange(len(Hp)))
            fl[nm][lo:lo + len(H)] = T.risk_of(lg, hk)
    return drop, fl


def test_row(name, family, r, delta0, note=""):
    return dict(test=name, family=family, delta0=delta0, note=note, **r)


def nt_row(name, family, note):
    return dict(test=name, family=family, estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, B_used=0,
                discarded=0, unreliable=False, note=f"NOT TESTABLE: {note}")


def boot_ci_mean_fa(boot, alarms, neg):
    """Seed-mean false-alarm rate on negatives with its 95% cluster-bootstrap percentile interval (C17)."""
    def rate(idx):
        j = idx[neg[idx]]
        return float(np.mean([a[j].mean() for a in alarms])) if len(j) else np.nan
    est = rate(boot.indices(None))
    v = np.array([rate(boot.indices(b)) for b in range(boot.B)]); v = v[np.isfinite(v)]
    lo, hi = (np.quantile(v, [0.025, 0.975]) if len(v) else (np.nan, np.nan))
    return est, float(lo), float(hi)


# ================================================================================================ endpoints
def endpoints(bank, V, B, extra):
    """All P2 endpoints for one analysis variant V: dict(name, k, keep (bool over test rows), ece, cal, tests).
    extra: shared precomputations (classical scores, lgbm, drop flags, flicker risks). Returns (rows, pairs, conf)."""
    k, name = V["k"], V["name"]
    ece = V.get("ece", S.ece_adaptive); cal = V.get("cal"); tests = V["tests"]
    y_t, kn_t = bank.labels("test", k); y_v, kn_v = bank.labels("val", k)
    keep_v = V.get("keep_val", np.ones(len(y_v), bool))
    Tix = np.flatnonzero(kn_t & V["keep"]); y = y_t[Tix]; cl = bank.ev("test")[Tix]
    rows, pairs_out, conf = [], [], []
    tag = lambda t: t if name == "primary" else f"{t} [{name}]"
    fam = "registered" if name == "primary" else "sensitivity"
    if len(Tix) == 0 or y.sum() == 0:
        return [nt_row(tag(t), fam, "no test positives in this variant") for t in tests], [], []
    boot = S.ClusterBootstrap(cl, B=B, seed=131)
    AP = S.average_precision

    def risk(nm, split="test"):
        m = bank.model(nm)
        return None if m is None else m[split][0][:, k]

    def cal_risk(nm, fit_rows):
        m = bank.model(nm)
        if m is None:
            return None
        if cal is None:
            return m["test"][0][:, k]
        mv = bank.meta["val"]
        r = calibrate(m["head"], m["val"][1][fit_rows], mv["event_bin"].to_numpy()[fit_rows], mv["obs_bins"].to_numpy()[fit_rows],
                      m["test"][1], cal)
        return r[:, k]

    def seeds(fmt):
        L = [risk(fmt.format(s=s)) for s in SEEDS]
        return None if any(x is None for x in L) else L

    H = V.get("head", bank.head)
    aug, noaug = seeds(f"central_{H}_aug_s{{s}}"), seeds(f"central_{H}_noaug_s{{s}}")

    def add(t, stat, d0=0.0, note="", bt=None):
        r = S.paired_test(bt or boot, stat, d0); rows.append(test_row(tag(t), fam if t in REGISTERED else "secondary", r, d0, note))

    # ---------------- H1 key secondary (P2) ----------------
    if "H1_P2" in tests:
        if aug is None:
            rows.append(nt_row(tag("H1_P2 model-LightGBM"), "secondary", f"central_{H}_aug models missing"))
        else:
            lg = [extra["lgbm"][s]["test"][:, k][Tix] for s in SEEDS]
            add("H1_P2 model-LightGBM", S.seed_mean_difference(AP, y, [a[Tix] for a in aug], lg), note=f"head {H}")
            cs = extra["classical"]
            vv = np.flatnonzero(kn_v & keep_v)
            val_ap = {c: AP(y_v[vv], cs["val"][c][vv]) for c in cs["val"]}
            best = max(val_ap, key=val_ap.get)
            add("H1_P2 model-best classical", S.seed_mean_difference(AP, y, [a[Tix] for a in aug], [cs["test"][best][Tix]] * len(SEEDS)),
                note=f"best classical (validation) = {best}; head {H}")
    # ---------------- H4 ----------------
    if any(t in tests for t in ("H4a", "H4b", "H4c")):
        if aug is None or noaug is None:
            for t in ("H4a", "H4b", "H4c"):
                if t in tests:
                    rows.append(nt_row(tag(t), fam, f"central_{H}_aug/noaug models missing"))
        else:
            A, N = [a[Tix] for a in aug], [b[Tix] for b in noaug]
            if "H4a" in tests:
                d = extra["drop"][Tix]
                if d.sum() == 0 or y[d].sum() == 0:
                    rows.append(nt_row(tag("H4a"), fam, "no natural-dropout test frames with a positive"))
                else:
                    bd = S.ClusterBootstrap(cl[d], B=B, seed=131)
                    add("H4a", S.seed_mean_difference(AP, y[d], [a[d] for a in A], [b[d] for b in N]),
                        note=f"natural-dropout subset: {int(d.sum())} frames, {int(y[d].sum())} positives", bt=bd)
            if "H4b" in tests:
                fa = [extra["flicker"][f"central_{H}_aug_s{s}"][Tix, k] for s in SEEDS]
                fn = [extra["flicker"][f"central_{H}_noaug_s{s}"][Tix, k] for s in SEEDS]
                add("H4b", S.seed_mean_difference(AP, y, fa, fn), note="50% frame flicker")
            if "H4c" in tests:
                add("H4c", S.seed_mean_difference(AP, y, A, N), NONINF, note="non-inferiority, margin -0.02")
    # ---------------- H2 ----------------
    if "H2" in tests:
        mt = pd.read_parquet(T.CACHE / "shrp2_train_meta.parquet")
        reg, mat = [], []
        for f in FACTORS:
            lt, lvv = bank.level(f, "test")[Tix], bank.level(f, "val")
            ltr = bank.trips[T.FACTORS[f]].reindex(mt["event_id"].to_numpy()).to_numpy(float) if T.FACTORS[f] in bank.trips else np.full(len(mt), np.nan)
            ntr = {v: int((ltr == v).sum()) for v in (0, 1)}
            big = 0 if ntr[0] >= ntr[1] else 1
            dom = {v: [f"dom_{f}_{v}_{H}_s{s}" for s in SEEDS] for v in (0, 1)}
            match = [f"match_{f}_{H}_s{s}" for s in SEEDS]
            fit = {v: np.flatnonzero((lvv == v) & keep_v) for v in (0, 1)}
            for a_, b_ in ((0, 1), (1, 0)):
                m = lt == b_
                n_ev = int(len(np.unique(cl[m & (y == 1)])))
                missing = [nm for nm in dom[a_] + dom[b_] if bank.model(nm) is None]
                pr = dict(variant=name, factor=f, direction=f"{a_}->{b_}", test_trips=int(len(np.unique(cl[m]))),
                          test_samples=int(m.sum()), test_events_with_positive=n_ev, test_positives=int(y[m].sum()))
                if missing or n_ev < MIN_EVENTS or not (m & (y == 0)).any():
                    pr.update(testable=False, reason=("models missing" if missing else f"{n_ev} test events < {MIN_EVENTS}"))
                    pairs_out.append(pr); continue
                out = [cal_risk(nm, fit[a_])[Tix] for nm in dom[a_]]; inn = [cal_risk(nm, fit[b_])[Tix] for nm in dom[b_]]
                mo = match if a_ == big else dom[a_]; mi = match if b_ == big else dom[b_]
                have_m = all(bank.model(nm) is not None for nm in match)
                e_ = lambda arrs: float(np.mean([ece(y[m], x[m]) for x in arrs]))
                pr.update(testable=True, big_level=big, ece_out=e_(out), ece_in=e_(inn), log_ratio=np.log(e_(out) / e_(inn)),
                          prev_heldout=float(y[m].mean()))
                reg.append(dict(y=y, mask=m, out=out, inn=inn))
                if have_m:
                    om = [cal_risk(nm, fit[big] if nm.startswith("match") else fit[a_])[Tix] for nm in mo]
                    im = [cal_risk(nm, fit[big] if nm.startswith("match") else fit[b_])[Tix] for nm in mi]
                    pr.update(log_ratio_matched=np.log(e_(om) / e_(im)))
                    mat.append(dict(y=y, mask=m, out=om, inn=im))
                pairs_out.append(pr)
                # H2 secondary: conformal guarantee (primary variant only)
                if V.get("conformal"):
                    vneg = np.flatnonzero(kn_v & (lvv == a_) & (y_v == 0) & keep_v)
                    neg = m & (y == 0)
                    for al in ALPHAS:
                        taus = [C.conformal_threshold(risk(nm, "val")[vneg], al) for nm in dom[a_]]
                        if not all(np.isfinite(taus)):
                            conf.append(dict(factor=f, direction=f"{a_}->{b_}", alpha=al, testable=False,
                                             note="too few validation negatives for a finite threshold")); continue
                        alarms = [o > t for o, t in zip(out if cal is None else [risk(nm)[Tix] for nm in dom[a_]], taus)]
                        est, lo, hi = boot_ci_mean_fa(boot, alarms, neg)
                        cps = [S.clopper_pearson(int(al_[neg].sum()), int(neg.sum()))[0] for al_ in alarms]
                        conf.append(dict(factor=f, direction=f"{a_}->{b_}", alpha=al, testable=True, false_alarm=est,
                                         cluster_lower=lo, cluster_upper=hi, violation=lo > al,
                                         cp_lower_per_seed=" ".join(f"{c:.4f}" for c in cps),
                                         cp_violations_of_5=int(sum(c > al for c in cps))))
        n_t = len(reg)
        if n_t >= 4:
            add("H2", S.mean_log_ratio(reg, ece), note=f"{n_t} of 8 pairs testable")
            if len(mat) == n_t:
                add("H2_matched", S.mean_log_ratio(mat, ece), note="size-matched (Amendment 1)")
            else:
                rows.append(nt_row(tag("H2_matched"), "secondary", "size-matched models missing"))
        else:
            rows.append(nt_row(tag("H2"), fam, f"{n_t} testable pairs < 4"))
    # ---------------- H3 ----------------
    if any(t in tests for t in ("H3a", "H3b1", "H3b2")):
        F_ = {r: [bank.fed(r, s) for s in SEEDS] for r in ("local", "fedavg", "fedper")}
        cj = Path(f"results/shrp2_fed_clients{T.SFX}.json")
        if cj.exists() and not json.loads(cj.read_text()).get("testable", False):
            for t in ("H3a", "H3b1", "H3b2"):
                if t in tests:
                    rows.append(nt_row(tag(t), fam, "client rule (results/shrp2_fed_clients.txt)"))
        elif any(x is None for v in F_.values() for x in v):
            for t in ("H3a", "H3b1", "H3b2"):
                if t in tests:
                    rows.append(nt_row(tag(t), fam, "federated predictions missing"))
        else:
            mv = bank.meta["val"]
            def fr(rg, si):
                z = F_[rg][si]; rk = z["test"][0][:, k]
                if cal is None:
                    return rk
                cli_v, cli_t = z["val"][2], z["test"][2]
                out = np.full(len(rk), np.nan)
                groups = [(c, (cli_v == c) & keep_v, cli_t == c) for c in pd.unique(cli_t[pd.notna(cli_t)])] if rg in ("local", "fedper") \
                    else [(None, pd.notna(cli_v) & keep_v, pd.notna(cli_t))]
                for _, fv, ft in groups:
                    fv = np.flatnonzero(fv); ft = np.asarray(ft, bool)
                    out[ft] = calibrate(z["head"], z["val"][1][fv], mv["event_bin"].to_numpy()[fv], mv["obs_bins"].to_numpy()[fv],
                                        z["test"][1][ft], cal)[:, k]
                return out
            Rr = {rg: [fr(rg, i)[Tix] for i in range(len(SEEDS))] for rg in F_}
            sub = np.isfinite(Rr["fedavg"][0])
            if sub.sum() == 0 or y[sub].sum() == 0:
                for t in ("H3a", "H3b1", "H3b2"):
                    if t in tests:
                        rows.append(nt_row(tag(t), fam, "no client test frames with a positive"))
            else:
                b3 = S.ClusterBootstrap(cl[sub], B=B, seed=131); ys = y[sub]
                R3 = {rg: [x[sub] for x in v] for rg, v in Rr.items()}
                if "H3a" in tests:
                    add("H3a", S.seed_mean_difference(ece, ys, R3["fedavg"], R3["fedper"]), bt=b3,
                        note=f"{int(sub.sum())} client test frames")
                if "H3b1" in tests:
                    add("H3b1", S.seed_mean_difference(AP, ys, R3["fedavg"], R3["local"]), bt=b3)
                if "H3b2" in tests:
                    add("H3b2", S.seed_mean_difference(AP, ys, R3["fedper"], R3["local"]), bt=b3)
    return rows, pairs_out, conf


def variants(bank, primary_k, horizons):
    """Primary analysis and the registered sensitivity analyses 1, 3, 4, 5, 6, 7 (Section 7)."""
    n_t = len(bank.meta["test"]); allk = np.ones(n_t, bool)
    ALL = ["H1_P2", "H2", "H3a", "H3b1", "H3b2", "H4a", "H4b", "H4c"]
    V = [dict(name="primary", k=primary_k, keep=allk, tests=ALL, conformal=True)]
    out_t, out_v = bank.outcome("test"), bank.outcome("val")
    V.append(dict(name="1 crash-only", k=primary_k, keep=out_t != "near-crash", keep_val=out_v != "near-crash", tests=ALL))
    for h in horizons:
        V.append(dict(name=f"3 horizon {h} s", k=HORIZONS.index(h), keep=allk, tests=ALL))
    V.append(dict(name=f"4 other head ({bank.other})", k=primary_k, keep=allk, head=bank.other, tests=["H1_P2", "H4a", "H4b", "H4c"]))
    V.append(dict(name="5 equal-width ECE", k=primary_k, keep=allk, ece=C.ece, tests=["H2", "H3a"]))
    V.append(dict(name="6 temperature", k=primary_k, keep=allk, cal="temperature", tests=["H2", "H3a"]))
    V.append(dict(name="6 isotonic", k=primary_k, keep=allk, cal="isotonic", tests=["H2", "H3a"]))
    V.append(dict(name="7 baselines with >= 1 object", k=primary_k, keep=keep_with_objects(bank, "test"),
                  keep_val=keep_with_objects(bank, "val"), tests=ALL))
    return V


def keep_with_objects(bank, s):
    """Sensitivity 7 (C20): every event-trip sample, and baseline-trip samples only of trips in which a radar object is
    detected in at least one sample's 2-s window."""
    X = np.load(T.CACHE / f"shrp2_{s}_hist.npy", mmap_mode="r")
    ev = bank.ev(s); n = len(ev); seen = np.zeros(n, bool)
    for lo in range(0, n, 50000):
        seen[lo:lo + 50000] = (np.asarray(X[lo:lo + 50000, 1:, :, 8]) > 0).any(axis=(1, 2))
    trips_seen = set(pd.Series(seen).groupby(ev).any().loc[lambda s_: s_].index)
    base = bank.outcome(s) == "baseline"
    return ~base | np.isin(ev, list(trips_seen))


def run(B, primary_h=3, horizons=(1, 2, 5)):
    sfx, tag = T.SFX, T.RUN
    t0 = time.time()
    bank = Bank()
    k = HORIZONS.index(primary_h)
    print(f"head {bank.head}; primary horizon {primary_h} s; B = {B}", flush=True)
    names = [f"central_{h}_{a}_s{s}" for h in (bank.head, bank.other) for a in ("aug", "noaug") for s in SEEDS
             if (T.MODELS / f"haz_central_{h}_{a}_s{s}.pt").exists()]
    drop, flicker = test_hist_pass(bank, names)
    extra = dict(lgbm={s: bank.lgbm(s) for s in SEEDS}, classical=bank.classical(), drop=drop, flicker=flicker)
    allrows, allpairs, allconf = [], [], []
    for V in variants(bank, k, [h for h in horizons if h != primary_h]):
        print(f"\n--- {V['name']} ---", flush=True)
        r, p, c = endpoints(bank, V, B, extra)
        allrows += r; allpairs += p; allconf += c
        for x in r:
            print(f"  {x['test']:40} est {x['estimate']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}] p {x['p']:.4f}"
                  + (" UNRELIABLE" if x.get("unreliable") else "") + f"  {x.get('note', '')}", flush=True)
    OUT.mkdir(exist_ok=True)
    R = pd.DataFrame(allrows); R.to_csv(OUT / f"shrp2_confirm_p2_tests{sfx}.csv", index=False)
    pd.DataFrame(allpairs).to_csv(OUT / f"shrp2_confirm_h2_pairs{sfx}.csv", index=False)
    Cf = pd.DataFrame(allconf); Cf.to_csv(OUT / f"shrp2_confirm_h2_conformal{sfx}.csv", index=False)
    lines = ["SHRP2 CONFIRMATORY ANALYSIS, P2 PART - aggregate results only (shareable)" + (f" - {tag}" if tag else ""),
             f"risk head {bank.head}; primary horizon {primary_h} s; B = {B}; clusters = test trips", ""]
    lines += [R[["test", "family", "estimate", "ci_low", "ci_high", "p", "discarded", "unreliable", "note"]].round(5).to_string(index=False)]
    if len(Cf):
        t_ = Cf[Cf.testable]
        lines += ["", "H2 secondary (conformal false-alarm guarantee): violating held-out domains per alpha (cluster rule, Amendment 1)",
                  t_.groupby("alpha")["violation"].agg(["sum", "count"]).rename(columns={"sum": "violating", "count": "testable"}).to_string()]
    (OUT / f"shrp2_confirm_p2_summary{sfx}.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved results/shrp2_confirm_p2_*.csv, shrp2_confirm_h2_*.csv | {time.time() - t0:.0f}s")
    return R, pd.DataFrame(allpairs), Cf


# ================================================================================================ H5 input, Holm
def h5_input(path=T.SH / "h5_input_shrp2.npz"):
    """Input of src/phase2/run_h5.py (Amendments 2-3): unit = trip; known/y at 3 s; one score per seed from the H1 models
    (central, chosen head, augmentation on), raw cumulative risk at 3 s; 36,000 frames per hour (10 Hz)."""
    bank = Bank()
    parts = {k_: [] for k_ in ("unit", "split", "known", "y", "event")}
    sc = {s: [] for s in SEEDS}
    for s in ("val", "test"):
        y, kn = bank.labels(s, 2); ev = bank.ev(s)
        parts["unit"].append(ev.astype(str)); parts["split"].append(np.full(len(ev), s)); parts["known"].append(kn)
        parts["y"].append(y); parts["event"].append(ev.astype(str))
        for sd in SEEDS:
            m = bank.model(f"central_{bank.head}_aug_s{sd}")
            if m is None:
                raise SystemExit(f"missing central_{bank.head}_aug_s{sd}")
            sc[sd].append(m[s][0][:, 2])
    out = {k_: np.concatenate(v) for k_, v in parts.items()}
    out.update({f"score_s{sd}": np.concatenate(v) for sd, v in sc.items()}); out["frames_per_hour"] = 36000.0
    np.savez_compressed(path, **out)
    print(f"H5 input written: {path} ({len(out['unit']):,} frames)")
    return path


def holm():
    sfx = T.SFX
    """Final Holm over the nine registered tests (NaN = not testable, never rejected); p-values for run_h5's 12-test
    sensitivity in the registered order."""
    rows = []
    for f, tests in ((OUT / f"shrp2_confirm_p1_tests{sfx}.csv", ["H1a", "H1b"]), (OUT / f"shrp2_confirm_p2_tests{sfx}.csv", REGISTERED[2:])):
        if not f.exists():
            raise SystemExit(f"missing {f}")
        d = pd.read_csv(f)
        for t in tests:
            r = d[d.test == t]
            if len(r) != 1:
                raise SystemExit(f"{t}: expected one row in {f}, found {len(r)}")
            rows.append(r.iloc[0].to_dict())
    H = pd.DataFrame(rows)
    H["p_holm"], H["supported"] = S.holm(H["p"].to_numpy(float))
    H["testable"] = np.isfinite(H["p"].to_numpy(float))
    H = H.set_index("test").loc[REGISTERED].reset_index()
    H.to_csv(OUT / f"shrp2_confirm_holm{sfx}.csv", index=False)
    pv = ",".join("nan" if not np.isfinite(x) else repr(float(x)) for x in H["p"].to_numpy(float))
    (OUT / f"shrp2_confirm_h1h4_pvalues{sfx}.txt").write_text(pv + "\n")          # for run_h5.py --h1h4 (12-test sensitivity)
    print(H[["test", "estimate", "ci_low", "ci_high", "p", "p_holm", "supported", "testable", "note"]].round(5).to_string(index=False))
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "h5-input", "holm"])
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--primary-horizon", type=int, default=3, choices=HORIZONS,
                    help="registered 3 s; 2 or 1 only if the audit drops 3 s (Section 6 contingency)")
    ap.add_argument("--horizons", default="1,2,5", help="horizons retained by the audit, for sensitivity 3")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.B, a.primary_horizon, [int(h) for h in a.horizons.split(",") if h])
    if a.cmd == "h5-input":
        return h5_input()
    return holm()


if __name__ == "__main__":
    main()
