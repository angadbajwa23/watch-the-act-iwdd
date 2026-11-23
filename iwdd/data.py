"""MIVIA-IWDD-500 labels, the camera-disjoint folds and the camera groups.

The dataset is not redistributed here. Download it from https://github.com/MiviaLab/IllegalWasteDumping and point
IWDD_DATA (or --data) at the folder that holds TRAINING_SET/ and VALIDATION_SET/, each with videos/ and labels/.
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO / "splits"
SPLIT_DIRS = {"train": "TRAINING_SET", "val": "VALIDATION_SET"}   # official 400 / 100 split
EXCLUDE = {"vid0481"}            # its annotated onset falls after the last frame; excluded everywhere
SCOPES = {                       # scope -> (label splits, folds file)
    "train399": (("train",), "folds_train399.json"),          # model selection / paper cross-validation
    "all499": (("train", "val"), "folds_all499.json"),        # the IWDD 2027 submission
}


def data_root(root=None) -> Path:
    root = Path(root or os.environ.get("IWDD_DATA", ""))
    if not (root / "TRAINING_SET" / "labels").is_dir():
        raise FileNotFoundError(f"no TRAINING_SET/labels under {str(root)!r}: set IWDD_DATA or pass --data")
    return root


def load_labels(split: str, root=None) -> Dict[str, dict]:
    """video id -> dict(onset: float seconds or None, event_type, time_of_day, lighting, video: path).

    Label quirks handled here: `label` is the string "true"/"false"; `timestamp` is in whole seconds."""
    base = data_root(root) / SPLIT_DIRS[split]
    out = {}
    for f in sorted((base / "labels").glob("*.json")):
        d = json.loads(f.read_text())
        m = d["metadata"]
        assert d["label"] in ("true", "false"), f
        out[f.stem] = dict(onset=float(d["timestamp"]) if d["label"] == "true" else None,
                           event_type=m["event_type"], time_of_day=m["time_of_day"], lighting=m["lighting"],
                           video=str(base / "videos" / m["video_id"]))
    return out


def load_folds(scope: str = "train399") -> Dict[str, int]:
    """video id -> fold 0..4 (camera-disjoint)."""
    d = json.loads((SPLITS_DIR / SCOPES[scope][1]).read_text())
    return {v: int(k) for v, k in d["folds"].items()}


def load_scope(scope: str, root=None) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """(labels of the scope's videos, folds). Every video of the folds file must be in the dataset."""
    splits, _ = SCOPES[scope]
    labels = {}
    for s in splits:
        labels.update(load_labels(s, root))
    folds = load_folds(scope)
    missing = sorted(set(folds) - set(labels))
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos of {scope} have no label file, e.g. {missing[:3]}")
    return {v: labels[v] for v in sorted(folds)}, folds


def camera_groups() -> Dict[str, int]:
    """video id -> camera group (train399 only): k-means on first-frame SigLIP2 embeddings + merge at cosine 0.88."""
    return {v: int(g) for v, g in json.loads((SPLITS_DIR / "camera_groups_train399.json").read_text()).items()}


def same_camera_negatives(onsets: Dict[str, Optional[float]], groups: Dict[str, int]) -> set:
    """Normal videos whose camera group also contains dumping videos (81 of the 200 in train399)."""
    dumping_groups = {groups[v] for v, g in onsets.items() if g is not None}
    return {v for v, g in onsets.items() if g is None and groups[v] in dumping_groups}
