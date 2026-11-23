import numpy as np
import pytest

from iwdd.windowmass import CONTEST, PAPER, belief, choose_temperature, decode, window_mass


def test_belief_is_a_distribution():
    p = belief([1.0, 2.0, 3.0], T=0.5)
    assert p.sum() == pytest.approx(1.0) and np.all(np.diff(p) > 0)


def test_single_sharp_bump_is_answered_inside_the_window():
    u = np.arange(0, 30, 0.5)
    s = np.where(u == 10.0, 50.0, 0.0)            # all belief on an onset at 10 s
    a, mass = decode(s, u, 30.0, T=1.0)
    assert 10 - 3 <= a <= 10 + 10 and mass > 0.99


def test_two_bumps_are_covered_by_one_answer():
    """The peak answers one bump; the 13-s window can hold both (paper, Fig. 3)."""
    u = np.arange(0, 40, 0.5)
    s = np.full(len(u), -50.0)
    s[u == 2.0] = 0.0
    s[u == 10.0] = -0.01                           # slightly lower: the peak readout picks 2 s
    a, mass = decode(s, u, 40.0, T=1.0)
    assert mass > 0.95
    assert a - 10.5 <= 2.0 and 10.0 <= a + 3.5     # [a - 10, a + 3] (+-0.5 s labels) covers both candidates
    peak = int(u[np.argmax(s)])
    assert window_mass(belief(s, 1.0), u, [peak])[0] < 0.6


def test_mass_equals_monte_carlo_hit_rate():
    """mass(a) = P(g - 3 <= a <= g + 10) when g = round(u_j + N(0, 1)) with j ~ p and onsets in whole seconds."""
    rng = np.random.default_rng(0)
    u = np.array([3.0, 9.5, 20.0])
    p = np.array([0.5, 0.3, 0.2])
    n = 400_000
    g = np.floor(u[rng.choice(3, n, p=p)] + rng.standard_normal(n) + 0.5)       # round half up
    for a in (5, 12, 18, 25):
        mc = np.mean((g - 3 <= a) & (a <= g + 10))
        assert window_mass(p, u, [a])[0] == pytest.approx(mc, abs=0.004)


def test_paper_window_allows_earlier_answers():
    u = np.arange(0, 30, 0.5)
    s = np.where(u == 10.0, 50.0, 0.0)
    assert window_mass(belief(s, 1.0), u, [5], PAPER)[0] > window_mass(belief(s, 1.0), u, [5], CONTEST)[0]


def test_short_video_answers_zero():
    assert decode([0.0], [0.5], 0.6, T=1.0)[0] == 0


def test_choose_temperature_ignores_normal_videos_and_breaks_ties_in_order():
    curves = {"a": (np.array([0.0, 5.0]), np.array([2.0, 20.0])), "n": (np.array([0.0]), np.array([1.0]))}
    T, rates = choose_temperature(curves, {"a": 20.0, "n": None}, {"a": 30.0, "n": 5.0}, [0.5, 1.0, 2.0])
    assert set(rates) == {0.5, 1.0, 2.0} and T == 0.5 and rates[0.5] == 1.0
