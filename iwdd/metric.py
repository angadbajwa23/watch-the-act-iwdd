"""Exact IWDD evaluation metric (Greco et al., Image and Vision Computing 2026; IWDD 2027 contest rules).

A positive video with onset g and prediction p is a TP iff g - early <= p <= g + late.
Per-video outcome categories (mutually exclusive):
  tp          positive, detection inside the window           -> TP
  mistimed    positive, detection outside the window          -> FP + FN
  missed      positive, no detection                          -> FN
  false_alarm negative, detection                             -> FP
  tn          negative, no detection                          -> TN

All times are in seconds. A prediction of None means "no dumping detected".
"""
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

PAPER_WINDOW = (5.0, 10.0)    # [g-5, g+10]  (IVC paper / official benchmark)
CONTEST_WINDOW = (3.0, 10.0)  # [g-3, g+10]  (IWDD 2027 contest)


@dataclass
class Result:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    mean_delay: Optional[float]  # mean |p - g| over TPs (Eq. 15); None if no TP
    d_norm: float                # max(0, 1 - D / t_max); 0 if no TP
    categories: Dict[str, str] = field(default_factory=dict)  # video_id -> category

    def counts(self) -> Dict[str, int]:
        out = {c: 0 for c in ("tp", "mistimed", "missed", "false_alarm", "tn")}
        for c in self.categories.values():
            out[c] += 1
        return out


def categorize(g: Optional[float], p: Optional[float], delta_t: float, t_max: float) -> str:
    """g is the ground-truth onset (None for a negative video); p the prediction."""
    if g is None:
        return "tn" if p is None else "false_alarm"
    if p is None:
        return "missed"
    return "tp" if g - delta_t <= p <= g + t_max else "mistimed"


def prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def evaluate(gt: Dict[str, Optional[float]], pred: Dict[str, Optional[float]],
             window=PAPER_WINDOW) -> Result:
    """gt: video_id -> onset seconds (None = negative). pred: video_id -> seconds or None.

    Every gt video must be evaluated; a video missing from pred is treated as "no detection".
    Predictions for videos not in gt are an error (likely an ID mismatch).
    """
    extra = set(pred) - set(gt)
    if extra:
        raise ValueError(f"{len(extra)} predictions for unknown videos, e.g. {sorted(extra)[:3]}")
    delta_t, t_max = window
    cats, delays = {}, []
    for vid, g in gt.items():
        p = pred.get(vid)
        cats[vid] = c = categorize(g, p, delta_t, t_max)
        if c == "tp":
            delays.append(abs(p - g))
    n = {c: 0 for c in ("tp", "mistimed", "missed", "false_alarm", "tn")}
    for c in cats.values():
        n[c] += 1
    tp, tn = n["tp"], n["tn"]
    fp = n["mistimed"] + n["false_alarm"]
    fn = n["mistimed"] + n["missed"]
    precision, recall, f1 = prf1(tp, fp, fn)
    mean_delay = sum(delays) / len(delays) if delays else None
    d_norm = max(0.0, 1 - mean_delay / t_max) if delays else 0.0
    return Result(tp, fp, fn, tn, precision, recall, f1, mean_delay, d_norm, cats)


def bootstrap_f1_ci(gt, pred, window=PAPER_WINDOW, n_resamples=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI on F1, resampling videos with replacement."""
    cats = evaluate(gt, pred, window).categories
    vids = list(cats)
    rng = random.Random(seed)
    f1s = []
    for _ in range(n_resamples):
        n = {c: 0 for c in ("tp", "mistimed", "missed", "false_alarm", "tn")}
        for _ in vids:
            n[cats[rng.choice(vids)]] += 1
        f1s.append(prf1(n["tp"], n["mistimed"] + n["false_alarm"], n["mistimed"] + n["missed"])[2])
    f1s.sort()
    lo = f1s[int(alpha / 2 * n_resamples)]
    hi = f1s[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi


def localization_accuracy(gt, pred, tolerances=(1.0, 2.0, 5.0)):
    """Fraction of positives-with-a-detection whose |p - g| <= tol (paper Sec. 5.4 style)."""
    errs = [abs(pred[v] - g) for v, g in gt.items() if g is not None and pred.get(v) is not None]
    return {t: (sum(e <= t for e in errs) / len(errs) if errs else 0.0) for t in tolerances}


def best_threshold(vids, gt, score, answer, window=CONTEST_WINDOW) -> float:
    """Threshold maximising F1 of 'answer iff score >= threshold' over `vids` (ties: the lowest threshold).

    gt: video_id -> onset or None; score: video_id -> float; answer: video_id -> the time given when answering."""
    best_f1, best_th = -1.0, None
    for th in sorted(set(float(score[v]) for v in vids)):
        f1 = evaluate({v: gt[v] for v in vids}, {v: (answer[v] if score[v] >= th else None) for v in vids}, window).f1
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return best_th
