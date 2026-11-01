"""
Tests of the SHRP2 training runner (src/shrp2/train_shrp2.py) on synthetic SHRP2-format data and synthetic public
pretraining data. No SHRP2 data. Answers are worked out independently (by hand, by the rehearsal code, or by
scikit-learn), never by calling the function under test twice.
Usage : python src/shrp2/test_train_shrp2.py      (~3-6 min, temporary folder only)
"""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "loaders"))
import make_synthetic as M
import shrp2 as LD
import make_split as MS
import build_samples_shrp2 as BS
import ssm_shrp2 as SS
import features_shrp2 as FE
import train_shrp2 as T
RES = []
STOP = os.environ.get("TEST_STOP_AFTER", "")            # mutation runs only: stop after a section (e.g. R9)


class _Stop(Exception):
    pass


def section_done(tag):
    if STOP == tag:
        raise _Stop


def npz(p):
    with np.load(p, allow_pickle=True) as z:          # closed at once: Windows cannot delete an open file
        return {k: z[k] for k in z.files}


def open_npz_files():
    """.npz files this process still holds open (Linux /proc only; elsewhere an empty list)."""
    fd = Path("/proc/self/fd")
    if not fd.exists():
        return []
    out = []
    for f in fd.iterdir():
        try:
            t = os.readlink(f)
        except OSError:
            continue
        if t.endswith(".npz"):
            out.append(t)
    return out


def check(name, cond, detail=""):
    RES.append((name, bool(cond))); print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}", flush=True)


def public_data(root, rng):
    """Small INTERACTION-like and highD-like sample files in the development format."""
    def make(n_loc, n_rec, per, tag):
        H, F, V, SC, LO = [], [], [], [], []
        for l in range(n_loc):
            for r in range(n_rec):
                h = rng.normal(0, 3, (per, 9, 20, 9)).astype(np.float32)
                h[..., 8] = (rng.random((per, 9, 1)) < 0.7).astype(np.float32); h[:, 0, :, 8] = 1
                H.append(h); F.append(rng.normal(0, 5, (per, 30, 2)).astype(np.float32))
                V.append((rng.random((per, 30)) < 0.95).astype(np.float32))
                SC += [f"{tag}{l}/rec{r}"] * per; LO += [f"{tag}{l}"] * per
        return dict(hist=np.concatenate(H), fut=np.concatenate(F), fut_valid=np.concatenate(V),
                    scene=np.array(SC), location=np.array(LO))
    I = make(3, 10, 12, "LOC"); n = len(I["scene"])
    np.savez_compressed(root / "data/processed/pretrain_samples.npz", **I)
    np.savez_compressed(root / "data/processed/hazard_labels.npz", event_bin=np.full(n, -1), obs_bins=np.full(n, 4),
                        current=(rng.random(n) < 0.1).astype(np.int64), ttc2d_now=np.full(n, np.inf))
    Hd = make(2, 10, 10, "HD")
    np.savez_compressed(root / "data/processed/highd_samples.npz", **Hd)
    return I, Hd


def main():
    tmp = Path(tempfile.mkdtemp(prefix="trainshrp2_")); cwd = os.getcwd()
    try:
        os.chdir(tmp); rng = np.random.default_rng(7)
        (tmp / "data/processed").mkdir(parents=True)
        root = M.build(tmp / "syn", "complete", scale=6)
        work = tmp / "data/processed/shrp2"
        LD.main(root, work, mask=True)
        MS.main(work / "trips.parquet", out_dir=work, res_dir=tmp / "res")
        BS.main(work, "asgiven", "car", res_dir=tmp / "res")
        BS.main(work, "asgiven", "car", res_dir=tmp / "res", p1=True, shard=700)   # several P1 shards
        SS.main(work, "asgiven", "car"); FE.main(work); FE.main(work, "samples_p1", "_p1")
        I, Hd = public_data(tmp, rng)

        # ---------------- R1 cache = shards exactly; public splits ----------------
        T.build_cache()
        ok = True
        for s in ("train", "val", "test"):
            sh = [npz(p) for p in sorted((work / "samples").glob(f"{s}_*.npz"))]
            h = np.load(T.CACHE / f"shrp2_{s}_hist.npy"); m = pd.read_parquet(T.CACHE / f"shrp2_{s}_meta.parquet")
            ok &= np.array_equal(h, np.concatenate([z["hist"] for z in sh]))
            ok &= np.array_equal(m["frame"].to_numpy(), np.concatenate([z["frame"] for z in sh]))
            ok &= np.array_equal(m["event_bin"].to_numpy(), np.concatenate([z["event_bin"] for z in sh]))
        p1sh = sorted((work / "samples_p1").glob("test_*.npz"))
        ok &= np.array_equal(np.load(T.CACHE / "p1_test_hist.npy"), np.concatenate([npz(p)["hist"] for p in p1sh]))
        check("R1 cache arrays and labels equal the sample shards exactly (incl. several P1 shards)", ok and len(p1sh) > 1, f"{len(p1sh)} P1 test shards")
        from train_pretrain import split_by_scene
        mI = pd.read_parquet(T.CACHE / "interaction_meta.parquet")
        cur = npz(tmp / "data/processed/hazard_labels.npz")["current"]
        k = cur == 0; tr, va, te = split_by_scene(I["scene"][k], I["location"][k])
        exp = {}
        for sc, a, b, c in zip(I["scene"][k], tr, va, te):
            exp.setdefault(sc, "train" if a else "val" if b else "test")
        got = dict(zip(mI["scene"], mI["split"]))
        trH, vaH, teH = split_by_scene(Hd["scene"], Hd["location"])
        mH = pd.read_parquet(T.CACHE / "highd_meta.parquet")
        check("R1 public splits: INTERACTION = rehearsal rule, highD = split_by_scene; public test recordings left out",
              all(exp[s] == v for s, v in got.items()) and "test" not in set(mI["split"]) and
              len(mH) == int((trH | vaH).sum()) and len(mI) == sum(v != "test" for sc, v in exp.items() for _ in range(12)),
              f"INTERACTION {len(mI)} rows, highD {len(mH)} rows")

        # ---------------- R2 stale cache refused ----------------
        f0 = sorted((work / "samples").glob("val_*.npz"))[0]; keep_bytes = f0.read_bytes()
        f0.write_bytes(keep_bytes + b"x")
        try:
            T.check_cache(); stale = False
        except SystemExit:
            stale = True
        f0.write_bytes(keep_bytes)
        check("R2 a changed sample shard makes the cache stale (refused)", stale and T.check_cache() is not None)

        # ---------------- R3 pretraining and fine-tuning sets (central) ----------------
        D = T.Data()
        mt, mv = D.g["shrp2_train"]["meta"], D.g["shrp2_val"]["meta"]
        ptr, pva, ftr, fva, _ = T.job_sets(D, T.jobs("A")[0], 131)
        names = [p[0] for p in D.parts]
        n_pub_tr = int((mI.split == "train").sum() + (mH.split == "train").sum())
        lo_tr = dict((p[0], p[1]) for p in D.parts)["shrp2_train"]; lo_va = dict((p[0], p[1]) for p in D.parts)["shrp2_val"]
        sh_tr = ptr[ptr >= lo_tr] - lo_tr; sh_va = pva[pva >= lo_va] - lo_va
        check("R3 pretraining: public training recordings + SHRP2 TRAINING-split baselines only (no event, no test trip)",
              "shrp2_test" not in names and len(ptr) - len(sh_tr) == n_pub_tr
              and np.array_equal(sh_tr, np.flatnonzero(mt.outcome_type.to_numpy() == "baseline"))
              and np.array_equal(sh_va, np.flatnonzero(mv.outcome_type.to_numpy() == "baseline")) and len(sh_tr) > 0,
              f"public {n_pub_tr} + SHRP2 baselines {len(sh_tr)}; val baselines {len(sh_va)}")
        check("R3 central fine-tuning: every training sample; selection on every validation sample",
              np.array_equal(ftr, np.arange(len(mt))) and np.array_equal(fva, np.arange(len(mv))))
        X1 = D.pre_get(np.array([lo_tr + sh_tr[3], 5, lo_va + sh_va[0]]), "hist")
        jj = np.array([9, 2, 5, 2])
        Xa = D.pre_get(lo_tr + sh_tr[jj], "fut")
        check("R3 readers keep the requested order (unsorted and repeated indices, one source)",
              np.array_equal(D.get("shrp2_val", jj), D.g["shrp2_val"]["hist"][jj]) and np.array_equal(Xa, D.g["shrp2_train"]["fut"][sh_tr[jj]]))
        check("R3 pretraining reader returns the right rows from each source",
              np.array_equal(X1[0], D.g["shrp2_train"]["hist"][sh_tr[3]]) and np.array_equal(X1[1], np.load(T.CACHE / "interaction_hist.npy")[5])
              and np.array_equal(X1[2], D.g["shrp2_val"]["hist"][sh_va[0]]))

        # ---------------- R4 H2 sets; R6 encoder sharing ----------------
        jd = dict(name="dom_lighting_0_hazard_s131", kind="dom", factor="lighting", level=0, head="hazard", aug=True, seed=131)
        p0, v0, f0_, fv0, _ = T.job_sets(D, jd, 131)
        lv_t = D.level_of("lighting", mt.event_id.to_numpy()); lv_v = D.level_of("lighting", mv.event_id.to_numpy())
        check("R4 dom lighting=0: fine-tuning on trips with lighting 0 only (unknown excluded); selection likewise",
              np.array_equal(f0_, np.flatnonzero(lv_t == 0)) and np.array_equal(fv0, np.flatnonzero(lv_v == 0))
              and np.isnan(lv_t[mt.outcome_type.to_numpy() == "baseline"]).all(), f"{len(f0_)} train / {len(fv0)} val samples")
        check("R6 baselines have no level -> H2 pretraining set = central set -> one shared encoder (same key)",
              T.set_key(p0, v0) == T.set_key(ptr, pva))
        # give some training baselines a lighting level (as if the audit found environment for baselines)
        Tr = pd.read_parquet(work / "trips.parquet")
        btr = np.sort(mt.loc[mt.outcome_type == "baseline", "event_id"].unique())
        bva = np.sort(mv.loc[mv.outcome_type == "baseline", "event_id"].unique())
        Tr.loc[Tr.event_id.isin(btr[:3]), "lighting_bin"] = 1; Tr.loc[Tr.event_id.isin(btr[3:5]), "lighting_bin"] = 0
        Tr.loc[Tr.event_id.isin(bva[:1]), "lighting_bin"] = 1
        Tr.to_parquet(work / "trips.parquet", index=False)
        D2 = T.Data()
        p0b, v0b, f0b, _, _ = T.job_sets(D2, jd, 131)
        sh = p0b[p0b >= lo_tr] - lo_tr; shv = v0b[v0b >= lo_va] - lo_va
        exp_tr = np.flatnonzero((mt.outcome_type.to_numpy() == "baseline") & ~mt.event_id.isin(btr[:3]).to_numpy())
        exp_va = np.flatnonzero((mv.outcome_type.to_numpy() == "baseline") & ~mv.event_id.isin(bva[:1]).to_numpy())
        check("R4 dom lighting=0 pretraining leaves out exactly the held-out level's baselines (lighting 1); unknown kept",
              np.array_equal(sh, exp_tr) and np.array_equal(shv, exp_va))
        check("R4 those level-0 baselines now enter level 0 fine-tuning", np.isin(btr[3:5], mt.event_id.to_numpy()[f0b]).all())
        check("R6 different pretraining set -> different encoder key", T.set_key(p0b, v0b) != T.set_key(ptr, pva))
        # match: hand computation, independent loop
        ms = T.match_subset(D2, "lighting", 132)
        lvt = D2.level_of("lighting", mt.event_id.to_numpy())
        n = {v: int((lvt == v).sum()) for v in (0, 1)}; big = 0 if n[0] >= n[1] else 1
        per = pd.Series(lvt == big).groupby(mt.event_id.to_numpy()).sum(); ids = np.sort(per[per > 0].index.to_numpy())
        order = ids[np.random.default_rng(132).permutation(len(ids))]
        k_, tot = 0, 0
        while tot < n[1 - big]:
            tot += int(per[order[k_]]); k_ += 1
        check("R5 size-matched subset: larger level's trips in seeded order until the sample count reaches the smaller level's",
              ms[0] == big and np.array_equal(ms[2], order[:k_]) and tot >= n[1 - big] > tot - per[order[k_ - 1]],
              f"big level {big}: {k_} trips, {tot} samples vs {n[1 - big]}")
        ms2 = T.match_subset(D2, "lighting", 133)
        check("R5 subset depends on the seed and is reproducible", not np.array_equal(ms[2], ms2[2]) and np.array_equal(ms[2], T.match_subset(D2, "lighting", 132)[2]))
        class Stub:                                              # hand case with an exact tie: 5 big-level trips of 3 samples, small level 6
            g = {"shrp2_train": {"meta": pd.DataFrame({"event_id": np.r_[np.repeat([11, 12, 13, 14, 15], 3), [16] * 6]})}}
            def level_of(self, f, ids):
                return np.where(np.asarray(ids) == 16, 1.0, 0.0)
        st = T.match_subset(Stub(), "lighting", 131)
        exp2 = np.array([11, 12, 13, 14, 15])[np.random.default_rng(131).permutation(5)][:2]
        check("R5 exact tie: stops as soon as the count REACHES the smaller level's (2 trips x 3 = 6), not one trip later",
              st[0] == 0 and st[1] == 1 and np.array_equal(st[2], exp2), f"{st[2]}")
        jm = dict(name="match_lighting_hazard_s132", kind="match", factor="lighting", head="hazard", aug=True, seed=132)
        pm, vm, fm, fvm, info = T.job_sets(D2, jm, 132)
        shm = pm[pm >= lo_tr] - lo_tr
        small = 1 - big
        exp_m = np.flatnonzero((mt.outcome_type.to_numpy() == "baseline") & ~(lvt == small)
                               & ~((lvt == big) & ~mt.event_id.isin(ms[2]).to_numpy()))
        check("R4 match pretraining: minus held-out level baselines and larger-level baselines outside the subset; "
              "fine-tuning on the subset; selection on the larger level's full validation data",
              np.array_equal(shm, exp_m) and np.array_equal(fm, np.flatnonzero(mt.event_id.isin(ms[2]).to_numpy()))
              and np.array_equal(fvm, np.flatnonzero(D2.level_of("lighting", mv.event_id.to_numpy()) == big)))
        Tr.loc[Tr.event_id.isin(np.concatenate([btr, bva])), "lighting_bin"] = None
        Tr.to_parquet(work / "trips.parquet", index=False)           # back to the documented layout

        # ---------------- R7 bit-identical to the rehearsal training loops ----------------
        from rehearsal import run_jobs as RJ
        D = T.Data()
        a_tr, a_va = ptr[::7][:700], pva[::5][:300]
        allidx = np.concatenate([a_tr, a_va])
        Dr = dict(X=torch.from_numpy(D.pre_get(allidx, "hist")), Y=torch.from_numpy(D.pre_get(allidx, "fut")),
                  V=torch.from_numpy(D.pre_get(allidx, "fut_valid")))
        mtr = np.r_[np.ones(len(a_tr), bool), np.zeros(len(a_va), bool)]
        torch.manual_seed(131); e1, ade1, ep1 = RJ.pretrain(Dr, mtr, ~mtr, 4, False)
        torch.manual_seed(131); e2, ade2, ep2 = T.pretrain(lambda j, key: Dr[{"hist": "X", "fut": "Y", "fut_valid": "V"}[key]][np.asarray(j)].numpy(),
                                                          np.arange(len(a_tr)), np.arange(len(a_tr), len(allidx)), 4, 131)
        same_pre = all(torch.equal(e1.state_dict()[k], e2.state_dict()[k]) for k in e1.state_dict()) and ade1 == ade2 and ep1 == ep2
        check("R7 pretraining = rehearsal run_jobs.pretrain, bit for bit (same seed, same data)", same_pre, f"ADE {ade1:.4f}, best epoch {ep1}")
        NT, NV, EP = 300, 300, 8
        Xt = np.concatenate([D.get("shrp2_train", np.arange(NT)), D.get("shrp2_val", np.arange(NV))])
        Et = np.concatenate([D.labels("shrp2_train", np.arange(NT))[0], D.labels("shrp2_val", np.arange(NV))[0]])
        Ot = np.concatenate([D.labels("shrp2_train", np.arange(NT))[1], D.labels("shrp2_val", np.arange(NV))[1]])
        Dh = dict(X=torch.from_numpy(Xt), E=torch.from_numpy(Et), O=torch.from_numpy(Ot))
        mt2 = np.r_[np.ones(NT, bool), np.zeros(NV, bool)]
        out = []
        for impl in ("rehearsal", "runner"):
            torch.manual_seed(131); e = RJ.InteractionEncoder(); e.load_state_dict(e1.state_dict())
            if impl == "rehearsal":
                out.append(RJ.finetune(e, Dh, mt2, ~mt2, EP, True, 131, False))
            else:
                out.append(T.finetune(e, lambda j: Xt[np.asarray(j)], lambda j: (Et[np.asarray(j)], Ot[np.asarray(j)]),
                                      np.arange(NT), np.arange(NT, NT + NV), EP, True, 131))
        (ea, ha, la, pa), (eb, hb, lb, pb) = out
        same_ft = all(torch.equal(ea.state_dict()[k], eb.state_dict()[k]) for k in ea.state_dict()) and \
            all(torch.equal(ha.state_dict()[k], hb.state_dict()[k]) for k in ha.state_dict()) and la == lb and pa == pb
        check("R7 fine-tuning (augmentation on) = rehearsal run_jobs.finetune, bit for bit", same_ft, f"val NLL {la:.5f}, best epoch {pa} of {EP}")
        # selection exercised: validation targets that get WORSE as training proceeds, so the best epoch is early
        from train_pretrain import const_vel
        Dq = dict(Dr); cv = const_vel(Dr["X"], 30)
        Dq["Y"] = torch.where(torch.from_numpy(mtr)[:, None, None], cv + torch.tensor([5.0, 0.0]), cv); Dq["V"] = torch.ones_like(Dr["V"])
        torch.manual_seed(131); q1 = RJ.pretrain(Dq, mtr, ~mtr, 4, False)
        torch.manual_seed(131); q2 = T.pretrain(lambda j, key: Dq[{"hist": "X", "fut": "Y", "fut_valid": "V"}[key]][np.asarray(j)].numpy(),
                                               np.arange(len(a_tr)), np.arange(len(a_tr), len(allidx)), 4, 131)
        Eq = np.where(mt2, -1, 0); Oq = np.where(mt2, 4, 0)            # training: no events; validation: all events in bin 1
        Dhq = dict(X=Dh["X"], E=torch.from_numpy(Eq), O=torch.from_numpy(Oq))
        torch.manual_seed(131); e = RJ.InteractionEncoder(); e.load_state_dict(e1.state_dict()); r1 = RJ.finetune(e, Dhq, mt2, ~mt2, 4, True, 131, False)
        torch.manual_seed(131); e = RJ.InteractionEncoder(); e.load_state_dict(e1.state_dict())
        r2 = T.finetune(e, lambda j: Xt[np.asarray(j)], lambda j: (Eq[np.asarray(j)], Oq[np.asarray(j)]), np.arange(NT), np.arange(NT, NT + NV), 4, True, 131)
        same_q = all(torch.equal(q1[0].state_dict()[k], q2[0].state_dict()[k]) for k in q1[0].state_dict()) and q1[1:] == q2[1:] \
            and all(torch.equal(r1[1].state_dict()[k], r2[1].state_dict()[k]) for k in r1[1].state_dict()) and r1[2:] == r2[2:]
        check("R7 best epoch BEFORE the last (validation targets made to worsen): both implementations keep the same early epoch",
              same_q and q1[2] < 4 and r1[3] < 4, f"pretraining best {q1[2]}/4, fine-tuning best {r1[3]}/4")

        section_done("R7")
        # ---------------- R8 isotonic projection; R9 head rule ----------------
        P = np.array([[0.3, 0.2, 0.4, 0.35], [0.1, 0.2, 0.3, 0.4], [0.5, 0.4, 0.3, 0.2], [0.2, 0.6, 0.1, 0.7]])
        E8 = np.array([[0.25, 0.25, 0.375, 0.375], [0.1, 0.2, 0.3, 0.4], [0.35, 0.35, 0.35, 0.35], [0.2, 0.35, 0.35, 0.7]])
        from sklearn.isotonic import IsotonicRegression
        R = rng.random((300, 4)); ir = np.array([IsotonicRegression().fit_transform(np.arange(4), r) for r in R])
        check("R8 per-sample isotonic projection: hand cases and scikit-learn on 300 random rows",
              np.allclose(T.isotonic_rows(P), E8) and np.allclose(T.isotonic_rows(R), ir, atol=1e-12))
        r9 = [T.head_rule([0.30] * 5, [0.10] * 5, [0.31] * 5, [0.12] * 5)[0], T.head_rule([0.30] * 5, [0.10] * 5, [0.3099] * 5, [0.10] * 5)[0],
              T.head_rule([0.30] * 5, [0.10] * 5, [0.40] * 5, [0.12001] * 5)[0], T.head_rule([0.2, 0.4], [0.1, 0.1], [0.31, 0.31], [0.1, 0.1])[0]]
        check("R9 head rule: gain 0.01 and ECE x1.20 exactly -> separate; 0.0099 -> hazard; x1.2001 -> hazard; seed means",
              r9 == ["sep", "hazard", "hazard", "sep"], str(r9))

        section_done("R9")
        # ---------------- R10 stage A smoke run: coverage, consistency, sharing, resume, determinism ----------------
        T.main(["run", "--stage", "A", "--smoke"])
        L = pd.read_csv(T.LOG)
        encs = sorted(T.MODELS.glob("enc_*_smoke.pt"))
        check("R10 stage A: 15 models, one shared encoder per seed (5 pretrainings)",
              len(L) == 15 and len(encs) == 5 and L.groupby("seed_slot")["encoder_set"].nunique().eq(1).all())
        z = npz(T.PRED / "haz_central_hazard_aug_s131_smoke.npz")
        mvv, mte = D.g["shrp2_val"]["meta"], D.g["shrp2_test"]["meta"]
        keys = pd.DataFrame(dict(e=z["event_id"], f=z["frame"]))
        check("R10 predictions: every validation and test sample exactly once, in cache order",
              len(z["frame"]) == len(mvv) + len(mte) and not keys.duplicated().any()
              and np.array_equal(z["frame"], np.r_[mvv.frame.to_numpy(), mte.frame.to_numpy()])
              and set(z["split"]) == {"val", "test"})
        ok = True
        for nm in ("central_hazard_aug_s131_smoke", "central_sep_aug_s132_smoke"):
            ck = torch.load(T.MODELS / f"haz_{nm}.pt"); e = T.InteractionEncoder(); h = T.HazardHead(64, 4)
            e.load_state_dict(ck["encoder"]); h.load_state_dict(ck["head"]); e.eval(); h.eval()
            zz = npz(T.PRED / f"haz_{nm}.npz")
            sel = np.r_[0, 7, 31, len(mvv) + 3]
            X = np.concatenate([D.get("shrp2_val", sel[:3]), D.get("shrp2_test", sel[3:] - len(mvv))])
            with torch.no_grad():
                lg = h(e(torch.from_numpy(X))).numpy()
            rr = T.C.risk_from_logits(lg) if "hazard" in nm else np.array([IsotonicRegression().fit_transform(np.arange(4), 1 / (1 + np.exp(-l))) for l in lg])
            ok &= np.allclose(zz["logits"][sel], lg, atol=1e-6) and np.allclose(zz["risk"][sel], rr, rtol=1e-5, atol=1e-9)
            if "sep" in nm:
                ok &= (np.diff(zz["risk"], axis=1) >= -1e-7).all()
        check("R10 stored logits and risk = saved weights re-run on those samples (hazard; separate + isotonic, monotone)", ok)
        row = L[L.model == "central_hazard_noaug_s133_smoke"].iloc[0]
        ck = torch.load(T.MODELS / "haz_central_hazard_noaug_s133_smoke.pt"); e = T.InteractionEncoder(); h = T.HazardHead(64, 4)
        e.load_state_dict(ck["encoder"]); h.load_state_dict(ck["head"])
        lg = T.logits_of(e, h, lambda j: D.get("shrp2_val", j), np.arange(len(mvv)))
        ev, ob = D.labels("shrp2_val", np.arange(len(mvv)))
        check("R10 logged selection loss = validation loss of the saved model (best epoch kept)",
              np.isclose(row.val_loss, T.C.hazard_nll_np(lg, ev, ob), atol=1e-6), f"{row.val_loss}")
        f_ = T.PRED / "haz_central_sep_aug_s134_smoke.npz"; b0 = f_.read_bytes()
        T.main(["run", "--stage", "A", "--smoke"])
        check("R10 resume: a second run skips finished models and changes nothing", f_.read_bytes() == b0 and len(pd.read_csv(T.LOG)) == 15)
        r0 = npz(f_)["risk"]; f_.unlink(); (T.MODELS / "haz_central_sep_aug_s134_smoke.pt").unlink()
        T.main(["run", "--stage", "A", "--smoke", "--only", "central_sep_aug_s134"])
        check("R10 refit of a deleted model reproduces its predictions exactly", np.array_equal(npz(f_)["risk"], r0))

        # ---------------- R11 head selection and stage B ----------------
        try:
            T.main(["run", "--stage", "B", "--smoke"]); refused = False
        except SystemExit:
            refused = True
        res = T.select_head(smoke=True)
        hz = [r["auprc3"] for r in res["per_seed"]["hazard"]]
        from analysis.stats import average_precision
        zv = npz(T.PRED / "haz_central_hazard_aug_s131_smoke.npz"); v = zv["split"] == "val"
        y3, kn = T.C.labels_at(mvv.event_bin.to_numpy(), mvv.obs_bins.to_numpy(), 2)
        check("R11 stage B refused before selection; selection uses validation AUPRC at 3 s per seed",
              refused and np.isclose(hz[0], average_precision(y3[kn], zv["risk"][v][kn, 2]), atol=1e-6)
              and res["choice"] in ("hazard", "sep") and T.HEADSEL.with_suffix(".txt").exists(), f"choice {res['choice']}")
        T.main(["run", "--stage", "B", "--smoke"])
        L = pd.read_csv(T.LOG)
        B_ = L[L.kind.isin(["dom", "match"])]
        exp_n = len(T.jobs("B", res["choice"])) - (5 if res["choice"] == "sep" else 0)
        trainable = B_[B_.note.isna()] if "note" in B_ else B_
        ok = True
        for _, r in trainable[trainable.kind == "dom"].iterrows():
            f, lvl = r.factor, int(r.level)
            ok &= int(r.train_samples) == int((D.level_of(f, mt.event_id.to_numpy()) == lvl).sum())
        check("R11 stage B: 40 domain + 20 size-matched models; logged training sizes = trips of that level (hand count)",
              len(B_) == exp_n and ok, f"{len(trainable)} trainable of {len(B_)}")
        check("R11 H2 models share the central encoder when baselines carry no level (no extra pretraining)",
              len(list(T.MODELS.glob("enc_*_smoke.pt"))) == 5
              and set(trainable.encoder_set) <= set(L[L.kind == "central"].encoder_set))

        section_done("R11")
        # ---------------- R12 divergence rule ----------------
        def drop(nm):
            for p in (T.PRED / f"haz_{nm}.npz", T.MODELS / f"haz_{nm}.pt"):
                p.unlink(missing_ok=True)
        D = T.Data()
        job = T.jobs("A")[0]                                     # central_hazard_aug_s131
        drop(job["name"] + "_smoke"); T.FORCE_DIVERGE.clear(); T.FORCE_DIVERGE.add(("finetune", 131, 1.0))
        LRS, real_adamw = [], torch.optim.AdamW

        def spy(params, *a, **kw):
            params = list(params); LRS.append([g.get("lr", kw.get("lr")) for g in params] if isinstance(params[0], dict) else [kw.get("lr")])
            return real_adamw(params, *a, **kw)
        torch.optim.AdamW = spy
        try:
            T.run_job(D, job, 1, 1, True)
        finally:
            torch.optim.AdamW = real_adamw
        L = pd.read_csv(T.LOG); dv = pd.read_csv(T.DIVLOG)
        last = L[L.model == job["name"] + "_smoke"].iloc[-1]
        check("R12 a non-finite loss -> the stage is re-run once with half the learning rates (3e-4/1e-3 -> 1.5e-4/5e-4; logged)",
              LRS == [[3e-4, 1e-3], [1.5e-4, 5e-4]] and float(last.lr_scale) == 0.5 and int(last.seed_used) == 131 and len(dv) == 1 and dv.iloc[0].stage == "finetune")
        drop(job["name"] + "_smoke"); T.FORCE_DIVERGE.update({("finetune", 131, 0.5)})
        T.run_job(D, job, 1, 1, True)
        Lq = pd.read_csv(T.LOG); last = Lq[Lq.model == job["name"] + "_smoke"].iloc[-1]
        z = npz(T.PRED / f"haz_{job['name']}_smoke.npz")
        check("R12 failing twice -> replaced by the next seed (136), recorded in the model and the logs",
              int(last.seed_used) == 136 and int(z["seed_used"]) == 136 and "replaced" in set(pd.read_csv(T.DIVLOG).stage))
        j2 = T.jobs("A")[3]                                      # central_hazard_aug_s132
        drop(j2["name"] + "_smoke"); T.FORCE_DIVERGE.update({("finetune", 132, 1.0), ("finetune", 132, 0.5)})
        T.run_job(D, j2, 1, 1, True)
        Lq = pd.read_csv(T.LOG); last = Lq[Lq.model == j2["name"] + "_smoke"].iloc[-1]
        check("R12 a second replaced slot of the same model takes the next unused seed (137)", int(last.seed_used) == 137)
        T.FORCE_DIVERGE.clear(); T.FORCE_DIVERGE.add(("pretrain", 135, 1.0))
        j3 = dict(T.jobs("A")[12]); j3["name"] = "central_hazard_aug_s135x"
        for p in T.MODELS.glob("enc_*_s135_smoke.pt"):
            p.unlink()
        T.run_job(D, j3, 1, 1, True)
        check("R12 pretraining divergence: marker written, encoder re-trained at half rate and saved for the models sharing it",
              len(list(T.MODELS.glob("enc_*_s135_smoke.diverged"))) == 1 and len(list(T.MODELS.glob("enc_*_s135_lr0.5_smoke.pt"))) == 1)
        T.FORCE_DIVERGE.clear()

        section_done("R12")
        # ---------------- R13 P1 frame scores ----------------
        T.main(["p1", "--smoke"])
        h = T.chosen_head(True)
        p = pd.read_parquet(T.PRED / f"p1_central_{h}_aug_s131_smoke.parquet")
        m1 = pd.concat([pd.read_parquet(T.CACHE / f"p1_{s}_meta.parquet") for s in ("val", "test")], ignore_index=True)
        zz = npz(T.PRED / f"haz_central_{h}_aug_s131_smoke.npz")
        p2 = pd.DataFrame(dict(event_id=zz["event_id"], frame=zz["frame"], r3=zz["risk"][:, 2], r2=zz["risk"][:, 1]))
        mm = p.merge(m1[["event_id", "frame", "current"]], on=["event_id", "frame"]).merge(p2, on=["event_id", "frame"], how="left")
        both = mm[mm.current == 0]
        ego = pd.concat([pd.read_parquet(f, columns=["event_id", "timestamp_ms", "is_ego"]) for f in work.glob("traj_*.parquet")])
        ego = set(zip(ego[ego.is_ego].event_id, ego[ego.is_ego].timestamp_ms // 100))
        check("R13 P1: one score per P1 sample; where a P1 frame is also a P2 sample its score = the P2 risk at 3 s",
              len(p) == len(m1) and not p.duplicated(["event_id", "frame"]).any() and both.r3.notna().all()
              and np.allclose(both.score, both.r3, rtol=1e-5, atol=0) and (mm.current == 1).any(),
              f"{len(p)} frames, {len(both)} shared with P2, {(mm.current == 1).sum()} excluded frames scored")
        # horizon check on a hazard-head model: its cumulative risk at 3 s is strictly above that at 2 s
        T.p1_scores(["central_hazard_aug_s133_smoke"])
        ph = pd.read_parquet(T.PRED / "p1_central_hazard_aug_s133_smoke.parquet")
        zh = npz(T.PRED / "haz_central_hazard_aug_s133_smoke.npz")
        qh = ph.merge(pd.DataFrame(dict(event_id=zh["event_id"], frame=zh["frame"], r3=zh["risk"][:, 2], r2=zh["risk"][:, 1])), on=["event_id", "frame"])
        check("R13 P1 score is the 3 s risk, not the 2 s risk (hazard model: the two always differ)",
              len(qh) > 0 and (qh.r3 > qh.r2).all() and np.allclose(qh.score, qh.r3, rtol=1e-5, atol=0)
              and not np.isclose(qh.score, qh.r2, rtol=1e-5, atol=0).any(), f"{len(qh)} shared frames")
        check("R13 P1 frame = subject clock / 0.1 s (the frame key p1_eval.frame_scores_to_rows uses)",
              all((e, f) in ego for e, f in zip(p.event_id, p.frame)))

        leaks = open_npz_files()
        check("R13 no .npz file left open by the runner or the tests (Windows cannot delete or replace an open file)",
              not leaks, f"{len(leaks)} open" if leaks else ("checked via /proc" if Path("/proc/self/fd").exists() else "not checkable on this system"))
        # ---------------- R14 LightGBM ----------------
        from rehearsal import lgbm_baseline as LG
        MASKS, real_fp = [], LG.fit_predict

        def spy_fp(X, ev, ob, tr, va, seed):
            MASKS.append((tr.copy(), va.copy(), len(X))); return real_fp(X, ev, ob, tr, va, seed)
        LG.fit_predict = spy_fp
        try:
            risks = T.run_lgbm()
        finally:
            LG.fit_predict = real_fp
        n_tr, n_va = (len(pd.read_parquet(work / f"features_{s}.parquet")) for s in ("train", "val"))
        tr0, va0, nX = MASKS[0]
        check("R14 LightGBM masks: training = training-split rows, early stopping = validation-split rows only (P1 rows in neither)",
              len(MASKS) == 5 and np.array_equal(np.flatnonzero(tr0), np.arange(n_tr)) and np.array_equal(np.flatnonzero(va0), np.arange(n_tr, n_tr + n_va))
              and nX > n_tr + n_va + len(pd.read_parquet(work / "features_test.parquet")), f"X rows {nX}")
        A = pd.concat([pd.read_parquet(work / f"features_{s}.parquet").assign(_p=s) for s in ("train", "val", "test")], ignore_index=True)
        r_ref, _ = LG.fit_predict(A[LG.FEATURES].reset_index(drop=True), A.event_bin.to_numpy(), A.obs_bins.to_numpy(),
                                  A._p.to_numpy() == "train", A._p.to_numpy() == "val", 131)
        zl = npz(T.PRED / "lgbm_s131.npz")
        sel = A._p.to_numpy() != "train"
        pl = pd.read_parquet(T.PRED / "p1_lgbm_s131.parquet")
        q = pl.merge(pd.DataFrame(dict(event_id=zl["event_id"], frame=zl["frame"], r=zl["risk"][:, 2])), on=["event_id", "frame"])
        check("R14 LightGBM = rehearsal fit_predict on the SHRP2 features (P1 rows do not change the fit)",
              np.allclose(zl["risk"], r_ref[sel], atol=0) and np.array_equal(zl["frame"], A.frame.to_numpy()[sel]))
        check("R14 LightGBM P1 scores = its P2 risk at 3 s on shared frames", len(q) > 0 and np.allclose(q.score, q.r, rtol=1e-6, atol=0), f"{len(q)} shared")

        # ---------------- R15-R16 sensitivity 8 = a full repetition in separate folders (C23, SHRP2_RUN=nomask) ----------------
        import subprocess, hashlib
        def tree_hash(paths):
            h = hashlib.sha256()
            for p_ in sorted(x for d_ in paths for x in Path(d_).rglob("*") if x.is_file() and "nomask" not in x.name):
                h.update(str(p_).encode()); h.update(p_.read_bytes())
            return h.hexdigest()
        reg = [T.MODELS, T.PRED, Path("results")]
        h0 = tree_hash(reg)
        shutil.copytree(work, tmp / "data/processed/shrp2_nomask", ignore=shutil.ignore_patterns("cache", "preds"))
        env = dict(os.environ, SHRP2_RUN="nomask"); runner = str(HERE / "train_shrp2.py")
        r1 = subprocess.run([sys.executable, runner, "cache"], capture_output=True, text=True, env=env)
        r2 = subprocess.run([sys.executable, runner, "run", "--stage", "A", "--smoke", "--only", "central_hazard_noaug_s134"],
                            capture_output=True, text=True, env=env)
        nm = "central_hazard_noaug_s134_smoke"
        zr, zn = npz(T.PRED / f"haz_{nm}.npz"), npz(tmp / "data/processed/shrp2_nomask/preds" / f"haz_{nm}.npz")
        jn = pd.read_csv("results/shrp2_jobs_nomask.csv") if Path("results/shrp2_jobs_nomask.csv").exists() else pd.DataFrame()
        check("R15 SHRP2_RUN=nomask: data, cache, predictions, models and job log go to *_nomask; the registered run is untouched",
              r1.returncode == 0 and r2.returncode == 0 and (Path("models/shrp2_nomask") / f"haz_{nm}.pt").exists()
              and len(jn) == 1 and tree_hash(reg) == h0, (r1.stderr + r2.stderr)[-300:])
        check("R16 the repetition runs the SAME pipeline: on an identical copy of the data it reproduces the registered model exactly",
              np.array_equal(zr["risk"], zn["risk"]) and np.array_equal(zr["logits"], zn["logits"]))
        bad = subprocess.run([sys.executable, "-c", f"import sys; sys.path[:0]=[r'{HERE}', r'{HERE.parent}']; import train_shrp2"],
                             capture_output=True, text=True, env=dict(os.environ, SHRP2_RUN="nomaks"))
        check("R17 any other run tag is refused (a typo can never start a third, unregistered run)", bad.returncode != 0 and "nomaks" in bad.stderr)
    except _Stop:
        print(f"(stopped after {STOP}: mutation run)")
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in RES)
    print(f"\n{n_ok}/{len(RES)} PASS")
    return n_ok == len(RES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
