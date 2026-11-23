import pytest

from iwdd.metric import (CONTEST_WINDOW, PAPER_WINDOW, best_threshold, bootstrap_f1_ci, categorize, evaluate,
                         localization_accuracy)


@pytest.mark.parametrize("g,p,window,expected", [
    (None, None, PAPER_WINDOW, "tn"),
    (None, 3.0, PAPER_WINDOW, "false_alarm"),
    (10.0, None, PAPER_WINDOW, "missed"),
    (10.0, 10.0, PAPER_WINDOW, "tp"),
    (10.0, 5.0, PAPER_WINDOW, "tp"),          # window edges are inclusive
    (10.0, 20.0, PAPER_WINDOW, "tp"),
    (10.0, 4.99, PAPER_WINDOW, "mistimed"),
    (10.0, 20.01, PAPER_WINDOW, "mistimed"),
    (10.0, 7.0, CONTEST_WINDOW, "tp"),
    (10.0, 6.99, CONTEST_WINDOW, "mistimed"),
    (10.0, 20.0, CONTEST_WINDOW, "tp"),
    (10.0, 6.0, PAPER_WINDOW, "tp"),          # early answer: a hit in 2026, a miss in 2027
    (10.0, 6.0, CONTEST_WINDOW, "mistimed"),
    (0.0, 0.0, CONTEST_WINDOW, "tp"),
])
def test_categorize(g, p, window, expected):
    assert categorize(g, p, *window) == expected


def test_mistimed_positive_counts_as_fp_and_fn():
    r = evaluate({"a": 10.0}, {"a": 30.0})
    assert (r.tp, r.fp, r.fn, r.tn) == (0, 1, 1, 0)
    assert r.categories == {"a": "mistimed"}


def test_empty_predictions():
    r = evaluate({"p1": 5.0, "p2": 8.0, "n1": None, "n2": None}, {})
    assert (r.tp, r.fp, r.fn, r.tn) == (0, 0, 2, 2)
    assert r.f1 == 0 and r.mean_delay is None


def test_unknown_video_id_raises():
    with pytest.raises(ValueError):
        evaluate({"a": 1.0}, {"b": 1.0})


def baseline_like(n_mistimed=6):
    """100 videos: 50 dumping (42 TP, 6 mistimed, 2 missed), 50 normal (9 false alarms)."""
    gt, pred = {}, {}
    for i in range(50):
        gt[f"pos{i}"] = 20.0
        if i < 42 or 42 + n_mistimed <= i < 48:
            pred[f"pos{i}"] = 21.0
        elif i < 42 + n_mistimed:
            pred[f"pos{i}"] = 40.0
    for i in range(50):
        gt[f"neg{i}"] = None
        if i < 9:
            pred[f"neg{i}"] = 3.0
    return gt, pred


def test_reproduces_published_baseline_confusion_matrix():
    """Greco et al. 2026 report TP 42 / FP 15 / FN 8 / TN 41 -> P 0.74, R 0.84, F1 0.79 on 100 videos: this only adds
    up if a mistimed answer counts as both a false positive and a false negative."""
    r = evaluate(*baseline_like())
    assert (r.tp, r.fp, r.fn, r.tn) == (42, 15, 8, 41)
    assert (round(r.precision, 2), round(r.recall, 2), round(r.f1, 2)) == (0.74, 0.84, 0.79)


def test_fixing_timing_alone_gives_f1_090():
    assert round(evaluate(*baseline_like(n_mistimed=0)).f1, 2) == 0.90


def test_bootstrap_ci_brackets_point_estimate():
    gt, pred = baseline_like()
    lo, hi = bootstrap_f1_ci(gt, pred, seed=42)
    assert lo < evaluate(gt, pred).f1 < hi
    assert bootstrap_f1_ci(gt, pred, seed=42) == (lo, hi)


def test_localization_accuracy():
    gt = {"a": 10.0, "b": 10.0, "c": 10.0, "d": 10.0, "n": None}
    acc = localization_accuracy(gt, {"a": 10.5, "b": 8.5, "c": 14.0, "n": 1.0})
    assert acc == {1.0: pytest.approx(1 / 3), 2.0: pytest.approx(2 / 3), 5.0: pytest.approx(1.0)}


def test_best_threshold_picks_lowest_of_tied_maxima():
    gt = {"p1": 10.0, "p2": 10.0, "n1": None}
    score = {"p1": 3.0, "p2": 2.0, "n1": 1.0}
    answer = {"p1": 10, "p2": 10, "n1": 10}
    assert best_threshold(list(gt), gt, score, answer) == 2.0
