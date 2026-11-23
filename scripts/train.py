"""Train the full system from the extracted features (GPU for E0; about 12 min on one recent GPU).

  python scripts/train.py --data /path/to/MIVIA-IWDD --features features --scope all499 --out weights_new

Steps (all inside the scope; with --scope all499 this is exactly how the IWDD 2027 submission was built):
  1. E0 out-of-fold margins: for each camera fold k, four members trained on the other folds (each early-stopped and
     calibrated on one of them) score fold k.
  2. E0 final members: five members, member j trained on all folds but j and early-stopped / calibrated on fold j.
  3. Out-of-fold V-JEPA action scores and localizer beliefs (heads fitted on four folds, applied to the fifth).
  4. Fusion statistics, window-mass temperature (out-of-fold hit rate) and decision threshold (out-of-fold F1).
  5. Final action head and localizer fitted on every video of the scope.
Writes <out>/e0_models/ (members.json + fp16 weights), <out>/params.npz, <out>/e0_oof.json, <out>/build_info.json.
The printed out-of-fold F1 reuses the out-of-fold predictions to choose the threshold, so it is optimistic; the
paper's nested estimate for this procedure on train399 is 0.808 (contest window).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iwdd import e0, vjepa, windowmass  # noqa: E402
from iwdd.data import load_scope  # noqa: E402
from iwdd.metric import CONTEST_WINDOW, PAPER_WINDOW, best_threshold, evaluate  # noqa: E402

TEMPERATURES = [0.5, 1.0, 2.0, 4.0]


def load_features(root, vids):
    root = Path(root)
    P, A, C4, C8 = {}, {}, {}, {}
    for v in vids:
        with np.load(root / "siglip576" / f"{v}.npz") as d:
            P[v], A[v] = d["p"].astype(np.float32), d["a"].astype(np.float32)
        for name, dst in (("vjepa_4fps", C4), ("vjepa_8fps", C8)):
            with np.load(root / name / f"{v}.npz") as d:
                dst[v] = (d["start"].astype(np.float32), d["logits"].astype(np.float32))
    return P, A, C4, C8


def e0_out_of_fold(vids, onsets, by, P, A, seed, dev, log):
    acc = {v: [] for v in vids}
    for k in range(5):
        inner = [j for j in range(5) if j != k]
        for j in inner:
            tr = [v for i in inner if i != j for v in by[i]]
            m, th, sg, _ = e0.train_member(tr, by[j], onsets, P, A, seed, dev)
            for v in by[k]:
                s, t = e0.predict(m, P[v], dev)
                acc[v].append(((s - th) / sg, t))
            del m
            torch.cuda.empty_cache()
        log(f"E0 out-of-fold: fold {k} done")
    return {v: dict(margin=float(np.mean([a for a, _ in x])), t=float(np.mean([b for _, b in x])))
            for v, x in acc.items()}


def e0_final(onsets, by, P, A, seed, dev, out, log):
    (out / "e0_models").mkdir(parents=True, exist_ok=True)
    meta = []
    for j in range(5):
        tr = [v for i in range(5) if i != j for v in by[i]]
        m, th, sg, ep = e0.train_member(tr, by[j], onsets, P, A, seed, dev)
        f = f"e0_seed{seed}_val{j}.pt"
        torch.save({k: v.half().cpu() for k, v in m.state_dict().items()}, out / "e0_models" / f)
        meta.append(dict(file=f, th=float(th), sigma=float(sg), seed=seed, val_fold=j, epoch=int(ep)))
        log(f"E0 member {j}: epoch {ep}, theta {th:+.3f}, sigma {sg:.2f}")
        del m
        torch.cuda.empty_cache()
    (out / "e0_models" / "members.json").write_text(json.dumps(meta, indent=1))


def vjepa_out_of_fold(vids, onsets, by, C4, seqs):
    """Action score and localizer belief of every video from heads fitted on the other four folds."""
    score, curve = {}, {}
    for j in range(5):
        tr = [v for f in range(5) if f != j for v in by[f]]
        fn, _ = vjepa.fit_detector(C4, onsets, tr)
        loc = vjepa.fit_localizer(seqs, onsets, tr)
        for v in by[j]:
            s = fn(C4[v][1]).max(1)
            s = uniform_filter1d(s, 3, mode="nearest") if len(s) > 2 else s
            score[v] = float(s.max())
            curve[v] = vjepa.localizer_curve(seqs[v][0], seqs[v][1], loc)
    return score, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="dataset root (default: $IWDD_DATA)")
    ap.add_argument("--features", default="features")
    ap.add_argument("--scope", choices=["all499", "train399"], default="all499")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-e0", default=None, help="take e0_models/ and e0_oof.json from this folder (skip 1-2)")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[{(time.time() - t0) / 60:5.1f} min] {s}", flush=True)
    torch.set_num_threads(8)       # the CPU heads were fitted with 8 threads; other counts can move 1-3 videos
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    labels, folds = load_scope(a.scope, a.data)
    vids = sorted(labels)
    onsets = {v: labels[v]["onset"] for v in vids}
    by = {k: [v for v in vids if folds[v] == k] for k in range(5)}
    P, A, C4, C8 = load_features(a.features, vids)
    dur = {v: len(P[v]) / 8.0 for v in vids}
    log(f"scope {a.scope}: {len(vids)} videos ({sum(g is not None for g in onsets.values())} dumping) | {dev}")

    if a.reuse_e0:
        src = Path(a.reuse_e0)
        e0_oof = json.loads((src / "e0_oof.json").read_text())
        import shutil
        if src.resolve() != out.resolve():
            shutil.copytree(src / "e0_models", out / "e0_models", dirs_exist_ok=True)
        log(f"E0 reused from {src}")
    else:
        e0_oof = e0_out_of_fold(vids, onsets, by, P, A, a.seed, dev, log)
        e0_final(onsets, by, P, A, a.seed, dev, out, log)
    (out / "e0_oof.json").write_text(json.dumps(e0_oof))

    seqs = {v: (st, vjepa.localizer_features(L)) for v, (st, L) in C8.items()}
    vj, curve = vjepa_out_of_fold(vids, onsets, by, C4, seqs)
    X = np.array([[e0_oof[v]["margin"], vj[v]] for v in vids])
    z_mu, z_sd = X.mean(0), X.std(0) + 1e-9
    z = dict(zip(vids, ((X - z_mu) / z_sd).mean(1)))
    T, rates = windowmass.choose_temperature(curve, onsets, dur, TEMPERATURES, windowmass.CONTEST)
    log("window-mass temperature (out-of-fold hit rate on dumping videos): "
        + ", ".join(f"T={t:g}: {r:.3f}" for t, r in rates.items()) + f" -> T = {T:g}")
    when = {v: windowmass.decode(*curve[v], dur[v], T, windowmass.CONTEST)[0] for v in vids}
    th = best_threshold(vids, onsets, z, when, CONTEST_WINDOW)
    pred = {v: (when[v] if z[v] >= th else None) for v in vids}
    when_p = {v: windowmass.decode(*curve[v], dur[v], T, windowmass.PAPER)[0] for v in vids}
    rc = evaluate(onsets, pred, CONTEST_WINDOW)
    rp = evaluate(onsets, {v: (when_p[v] if z[v] >= th else None) for v in vids}, PAPER_WINDOW)
    log(f"out-of-fold (threshold tuned on the same predictions): contest F1 {rc.f1:.3f} {rc.counts()} | "
        f"paper-window F1 {rp.f1:.3f} | threshold {th:+.4f}")

    _, mil = vjepa.fit_detector(C4, onsets, vids)
    loc = vjepa.fit_localizer(seqs, onsets, vids)
    np.savez(out / "params.npz", **mil, **loc, z_mu=z_mu, z_sd=z_sd, threshold=np.float64(th), loc_T=np.float64(T))
    (out / "build_info.json").write_text(json.dumps(dict(
        scope=a.scope, n_videos=len(vids), seed=a.seed, threshold=th, loc_T=T, oof_contest_f1=rc.f1,
        oof_paper_f1=rp.f1, date=time.strftime("%Y-%m-%d")), indent=1))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
