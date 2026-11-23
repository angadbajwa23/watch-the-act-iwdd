"""Window-mass decoding: the F1-optimal timestamp under a tolerance-window metric (paper, Sec. 5.4).

Setting. A detector has already decided to answer a video. It gives a belief over when the event started:
clip j stands for time u_j and gets score s_j, and p = softmax(s / T). An answer a is correct iff the true onset g
satisfies  g - early <= a <= g + late,  i.e.  g in [a - late, a + early].

Once the set of answered videos is fixed, F1 = 2H / (2 D+ + D- + M) with only H (the number of hits) depending on the
answered times, so the expected F1 is maximised by maximising, per video, the probability that the onset falls in the
answer's window:

    a* = argmax_a  sum_j p_j [ Phi((a + early + edge - u_j) / sigma) - Phi((a - late - edge - u_j) / sigma) ]

edge = 0.5 widens the window by half a second on each side because onsets are annotated in whole seconds, and the
Gaussian CDF Phi (sigma = 1 s) softens the edges to allow for error in u_j. The usual "peak" readout (argmax_j p_j)
is the right answer only for an exact-hit metric.

Only numpy and scipy are needed; copy this file into your own project if you like.
"""
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

CONTEST = (3.0, 10.0)   # IWDD 2027: [g - 3, g + 10]
PAPER = (5.0, 10.0)     # IWDD 2026 / IVC paper: [g - 5, g + 10]


def belief(scores, T: float) -> np.ndarray:
    """Softmax of clip scores at temperature T (sums to 1)."""
    s = np.asarray(scores, dtype=np.float64)
    p = np.exp((s - s.max()) / T)
    return p / p.sum()


def window_mass(p, u, answers, window=CONTEST, sigma: float = 1.0, edge: float = 0.5) -> np.ndarray:
    """P(answer is a hit) for every candidate answer. p: belief over clips, u: clip times (s), answers: [A] (s)."""
    early, late = window
    p, u = np.asarray(p, np.float64), np.asarray(u, np.float64)
    A = np.asarray(answers, np.float64)[:, None]
    return (p[None] * (ndtr((A + early + edge - u[None]) / sigma) - ndtr((A - late - edge - u[None]) / sigma))).sum(1)


def decode(scores, u, duration: float, T: float, window=CONTEST, sigma: float = 1.0,
           edge: float = 0.5) -> Tuple[int, float]:
    """Best whole-second answer in [0, duration] and its hit probability.

    scores: [N] clip scores (any real numbers; the belief is softmax(scores / T)); u: [N] the onset time each clip
    stands for, in seconds; duration: video length in seconds."""
    answers = np.arange(0, int(np.floor(duration)) + 1, dtype=np.float64)
    mass = window_mass(belief(scores, T), u, answers, window, sigma, edge)
    i = int(np.argmax(mass))
    return int(answers[i]), float(mass[i])


def hit_rate(curves: Dict[str, Tuple[Sequence[float], Sequence[float]]], onsets: Dict[str, float],
             durations: Dict[str, float], T: float, window=CONTEST) -> float:
    """Fraction of videos whose decoded answer lands in the window. curves: vid -> (scores, u)."""
    early, late = window
    hits = [onsets[v] - early <= decode(s, u, durations[v], T, window)[0] <= onsets[v] + late
            for v, (s, u) in curves.items()]
    return float(np.mean(hits))


def choose_temperature(curves, onsets: Dict[str, Optional[float]], durations, temperatures: Iterable[float],
                       window=CONTEST) -> Tuple[float, Dict[float, float]]:
    """The temperature with the highest hit rate on the positives among `curves` (ties: the first in the list).

    Use out-of-fold curves: the temperature is the decoder's only setting."""
    pos = {v: c for v, c in curves.items() if onsets[v] is not None}
    rates = {T: hit_rate(pos, onsets, durations, T, window) for T in temperatures}
    return max(rates, key=rates.get), rates
