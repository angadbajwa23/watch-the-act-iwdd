"""Shortcut audit: does a dumping detector watch the act or recognise the scene? (paper, Sec. 4 and Table 1)

Five checks on a detector's out-of-fold predictions on the 399 training videos (folds in splits/folds_train399.json):
  1. Beats the scene?   F1 minus the scene-only floor on the same folds. The floor is a linear classifier on the
                        first frame's SigLIP2 embedding (threshold tuned on the other folds) that always answers
                        "video midpoint + 2 s".
  2. Follows the scene? Spearman rho, on normal videos, between the detector's score and the first-frame scene score
                        (near 1: the score mostly reflects the place).
  3. Same-camera false alarms. False-alarm rate on the 81 normal videos whose camera also films dumping vs the 119
                        from other cameras, plus the threshold-free AUROC of dumping videos against each group.
  4. Knows when?        Partial correlation, on answered dumping videos, between the answer and the true onset given
                        the video midpoint (0: the time says nothing beyond "the middle of the video").
  5. Sees the act?      Within each dumping video, AUROC of frame scores during the act [g, g + 4 s] against frames
                        before g - 4 s, averaged over videos (0.5: blind to the act). Needs per-frame scores.

Command line (labels are read from your copy of the dataset):
  python -m iwdd.audit --data /path/to/MIVIA-IWDD --predictions my_oof.csv [--curves my_curves.npz]
predictions CSV: columns video, score, answer (answer in seconds; empty = no dumping). `score` must be comparable
across folds: if your threshold differs per fold, give score minus the threshold of the video's fold.
curves NPZ (optional, question 5): one array per video id, shape [N, 2] = (time in s, frame or clip score).
"""
import argparse
import csv
import json
from typing import Dict, Optional

import numpy as np
from scipy.stats import spearmanr

from .data import SPLITS_DIR, camera_groups, load_folds, load_labels, same_camera_negatives
from .metric import CONTEST_WINDOW, PAPER_WINDOW, best_threshold, evaluate

SCENE_CSV = SPLITS_DIR / "scene_train399.csv"


# ---------------- statistics ----------------
def auroc(y, s) -> float:
    """Rank AUROC (ties count half). y: bool array, s: scores."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or not y.any():
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def partial_corr(x, y, z) -> float:
    """corr(x, y | z): correlation of the residuals of straight-line fits of x and y on z."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    Z = np.c_[np.ones_like(z), z]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------- the scene-only reference ----------------
def scene_scores(first_frames: Dict[str, np.ndarray], onsets, folds, C: float = 0.01) -> Dict[str, float]:
    """Out-of-fold P(dumping) logit from the first frame's embedding (standardised features + logistic regression)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    vids = sorted(first_frames)
    X = np.stack([first_frames[v] for v in vids]).astype(np.float64)
    y = np.array([onsets[v] is not None for v in vids])
    s = np.zeros(len(vids))
    for k in sorted({folds[v] for v in vids}):
        tr = np.array([folds[v] != k for v in vids])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        s[~tr] = clf.decision_function(sc.transform(X[~tr]))
    return dict(zip(vids, s.tolist()))


def scene_floor(scene, onsets, durations, folds, window=CONTEST_WINDOW):
    """F1 of the scene-only floor: answer 'midpoint + 2 s' iff the scene score passes a threshold tuned on the
    other folds. Returns (Result, predictions)."""
    vids = sorted(scene)
    t = {v: int(durations[v] / 2 + 2) for v in vids}
    pred = {}
    for k in sorted({folds[v] for v in vids}):
        th = best_threshold([v for v in vids if folds[v] != k], onsets, scene, t, window)
        pred.update({v: (t[v] if scene[v] >= th else None) for v in vids if folds[v] == k})
    return evaluate({v: onsets[v] for v in vids}, pred, window), pred


def load_scene_csv(path=SCENE_CSV):
    """(scene score, decoded duration in s) per training video, as shipped in splits/."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return ({r["video"]: float(r["scene_score"]) for r in rows}, {r["video"]: float(r["duration_s"]) for r in rows})


# ---------------- the five checks ----------------
def within_video_auroc(curves, onsets, act=4.0, gap=4.0):
    """Mean over dumping videos of AUROC(frames in [g, g + act] vs frames before g - gap). curves: vid -> (t, s)."""
    out = []
    for v, (t, s) in curves.items():
        g = onsets[v]
        if g is None:
            continue
        t, s = np.asarray(t, float), np.asarray(s, float)
        ev, pre = (t >= g) & (t <= g + act), t < g - gap
        if ev.any() and pre.any():
            out.append(auroc(np.r_[np.ones(ev.sum()), np.zeros(pre.sum())].astype(bool), np.r_[s[ev], s[pre]]))
    return (float(np.mean(out)) if out else float("nan")), len(out)


def audit(score: Dict[str, float], answer: Dict[str, Optional[float]], onsets: Dict[str, Optional[float]],
          durations: Dict[str, float], folds: Dict[str, int], groups: Dict[str, int], scene: Dict[str, float],
          curves=None, window=CONTEST_WINDOW) -> dict:
    """All five checks for one system. Every dict is keyed by video id over the same videos."""
    vids = sorted(onsets)
    for name, d in (("score", score), ("answer", answer), ("scene", scene), ("durations", durations),
                    ("folds", folds)):
        missing = [v for v in vids if v not in d]
        if missing:
            raise ValueError(f"{name}: {len(missing)} videos missing, e.g. {missing[:3]}")
    early, late = window
    r = evaluate(onsets, {v: answer[v] for v in vids}, window)
    floor, _ = scene_floor({v: scene[v] for v in vids}, onsets, durations, folds, window)
    neg = [v for v in vids if onsets[v] is None]
    pos = [v for v in vids if onsets[v] is not None]
    same = same_camera_negatives(onsets, groups)
    other = [v for v in neg if v not in same]
    same = sorted(same)
    det = [v for v in pos if answer[v] is not None]
    mid = {v: durations[v] / 2 for v in vids}
    out = dict(
        n_videos=len(vids), f1=r.f1, precision=r.precision, recall=r.recall, counts=r.counts(),
        scene_floor_f1=floor.f1, f1_minus_floor=r.f1 - floor.f1,                                       # 1
        rho_scene=float(spearmanr([score[v] for v in neg], [scene[v] for v in neg])[0]),               # 2
        fa_same_camera=float(np.mean([answer[v] is not None for v in same])), n_same_camera=len(same),  # 3
        fa_other_cameras=float(np.mean([answer[v] is not None for v in other])), n_other_cameras=len(other),
        auroc_vs_same_camera=auroc(np.r_[np.ones(len(pos)), np.zeros(len(same))].astype(bool),
                                   [score[v] for v in pos + same]),
        auroc_vs_other_cameras=auroc(np.r_[np.ones(len(pos)), np.zeros(len(other))].astype(bool),
                                     [score[v] for v in pos + other]),
        partial_r=partial_corr([answer[v] for v in det], [onsets[v] for v in det], [mid[v] for v in det]),  # 4
        hit_rate_answered=float(np.mean([onsets[v] - early <= answer[v] <= onsets[v] + late for v in det])),
        hit_rate_midpoint_guess=float(np.mean([onsets[v] - early <= int(mid[v] + 2) <= onsets[v] + late
                                               for v in det])),
    )
    if curves is not None:                                                                              # 5
        out["act_auroc"], out["n_act_videos"] = within_video_auroc(curves, onsets)
    return out


def format_report(a: dict) -> str:
    c = a["counts"]
    lines = [
        f"videos {a['n_videos']} | F1 {a['f1']:.3f} (P {a['precision']:.2f}, R {a['recall']:.2f}) | "
        f"TP {c['tp']}, mistimed {c['mistimed']}, missed {c['missed']}, false alarms {c['false_alarm']}",
        f"1. beats the scene?    scene-only floor {a['scene_floor_f1']:.3f} -> F1 - floor {a['f1_minus_floor']:+.3f}",
        f"2. follows the scene?  rho on normal videos {a['rho_scene']:.2f}  (lower is better)",
        f"3. same-camera FAs     {a['fa_same_camera']:.0%} of {a['n_same_camera']} same-camera vs "
        f"{a['fa_other_cameras']:.0%} of {a['n_other_cameras']} other | AUROC vs same-camera "
        f"{a['auroc_vs_same_camera']:.3f}, vs other {a['auroc_vs_other_cameras']:.3f}",
        f"4. knows when?         partial r {a['partial_r']:.2f} | hit rate {a['hit_rate_answered']:.2f} vs "
        f"'midpoint + 2 s' {a['hit_rate_midpoint_guess']:.2f} on the same videos",
    ]
    if "act_auroc" in a:
        lines.append(f"5. sees the act?       within-video AUROC {a['act_auroc']:.2f} over {a['n_act_videos']} videos")
    return "\n".join(lines)


def read_predictions(path):
    score, answer = {}, {}
    with open(path) as f:
        for r in csv.DictReader(f):
            score[r["video"]] = float(r["score"])
            answer[r["video"]] = float(r["answer"]) if r["answer"].strip() not in ("", "None", "nan") else None
    return score, answer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="dataset root (default: $IWDD_DATA)")
    ap.add_argument("--predictions", required=True, help="CSV with columns video, score, answer (out-of-fold)")
    ap.add_argument("--curves", default=None, help="optional NPZ: video id -> [N, 2] (time s, score), question 5")
    ap.add_argument("--window", choices=["contest", "paper"], default="contest")
    ap.add_argument("--scene-features", default=None,
                    help="recompute the scene scores from features/siglip576 (default: the shipped splits/ CSV)")
    ap.add_argument("--json", default=None, help="also write the numbers to this JSON file")
    a = ap.parse_args()
    labels = load_labels("train", a.data)
    folds = load_folds("train399")
    onsets = {v: labels[v]["onset"] for v in folds}
    if a.scene_features:
        from pathlib import Path
        first, durations = {}, {}
        for v in folds:
            with np.load(Path(a.scene_features) / f"{v}.npz") as d:
                first[v], durations[v] = d["p"][0], len(d["p"]) / 8.0
        scene = scene_scores(first, onsets, folds)
    else:
        scene, durations = load_scene_csv()
    score, answer = read_predictions(a.predictions)
    curves = None
    if a.curves:
        with np.load(a.curves) as d:
            curves = {v: (d[v][:, 0], d[v][:, 1]) for v in d.files if v in onsets}
    window = CONTEST_WINDOW if a.window == "contest" else PAPER_WINDOW
    res = audit(score, answer, onsets, durations, folds, camera_groups(), scene, curves, window)
    print(format_report(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
