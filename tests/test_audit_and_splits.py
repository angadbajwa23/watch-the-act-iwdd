import json

import numpy as np
import pytest

from iwdd.audit import audit, auroc, load_scene_csv, partial_corr, within_video_auroc
from iwdd.data import EXCLUDE, SPLITS_DIR, camera_groups, load_folds, same_camera_negatives


# ---------------- statistics ----------------
def test_auroc():
    assert auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auroc([1, 0], [0.5, 0.5]) == 0.5
    assert auroc([1, 1, 0], [0.9, 0.1, 0.5]) == 0.5


def test_partial_corr():
    rng = np.random.default_rng(0)
    z = rng.uniform(5, 60, 2000)
    assert abs(partial_corr(z + rng.normal(0, 1, 2000), z + rng.normal(0, 1, 2000), z)) < 0.1   # only z in common
    e = rng.normal(0, 3, 2000)
    assert partial_corr(z + e, z + e, z) == pytest.approx(1.0)


def test_within_video_auroc():
    t = np.arange(0, 30, 0.5)
    curves = {"a": (t, (t >= 10).astype(float)), "n": (t, t)}
    assert within_video_auroc(curves, {"a": 10.0, "n": None}) == (1.0, 1)


def test_audit_flags_a_pure_scene_classifier():
    """A detector whose score IS the scene score follows the scene perfectly (rho = 1) and adds no timing."""
    rng = np.random.default_rng(1)
    vids = [f"v{i:03d}" for i in range(200)]
    groups = {v: i // 10 for i, v in enumerate(vids)}                       # 20 cameras of 10 videos
    dumping_cams = set(range(10))
    onsets = {v: (float(rng.integers(5, 25)) if groups[v] in dumping_cams and rng.random() < 0.7 else None)
              for v in vids}
    scene = {v: float(groups[v] in dumping_cams) + rng.normal(0, 0.3) for v in vids}
    durations = {v: 40.0 for v in vids}
    folds = {v: groups[v] % 5 for v in vids}
    answer = {v: (float(rng.integers(15, 25)) if scene[v] > 0.5 else None) for v in vids}
    a = audit(scene, answer, onsets, durations, folds, groups, scene)
    assert a["rho_scene"] == pytest.approx(1.0)
    assert a["fa_same_camera"] > a["fa_other_cameras"]
    assert a["n_same_camera"] + a["n_other_cameras"] == sum(g is None for g in onsets.values())


# ---------------- shipped splits ----------------
def test_train399_folds_and_groups():
    folds, groups = load_folds("train399"), camera_groups()
    assert len(folds) == 399 and set(folds) == set(groups) and not EXCLUDE & set(folds)
    assert set(folds.values()) == {0, 1, 2, 3, 4}
    for g in set(groups.values()):                      # a camera group never spans two folds
        assert len({folds[v] for v in folds if groups[v] == g}) == 1


def test_all499_folds():
    folds = load_folds("all499")
    assert len(folds) == 499 and not EXCLUDE & set(folds) and set(folds.values()) == {0, 1, 2, 3, 4}
    assert set(load_folds("train399")) < set(folds)


def test_scene_csv_covers_train399():
    scene, dur = load_scene_csv()
    assert set(scene) == set(load_folds("train399")) and min(dur.values()) > 0


def test_same_camera_negatives():
    groups = {"a": 0, "b": 0, "c": 1, "d": 1}
    assert same_camera_negatives({"a": 5.0, "b": None, "c": None, "d": None}, groups) == {"b"}


def test_meta_documents_the_method():
    meta = json.loads((SPLITS_DIR / "folds_train399.json").read_text())["meta"]
    assert meta["n"] == 399 and "0.88" in meta["method"]
