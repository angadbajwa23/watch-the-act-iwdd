# Camera-disjoint splits for MIVIA-IWDD-500

Many cameras in MIVIA-IWDD contribute several videos, and most cameras film mostly one class. A random split therefore
rewards recognising the place (of the 100 official validation videos, 46 have a near-duplicate first frame in the
training set). These files let you evaluate on cameras the model has never seen.

| file | content |
|---|---|
| `folds_train399.json` | 5 camera-disjoint folds of the official training set (399 videos; `vid0481` excluded). Use these for model selection and for the shortcut audit. |
| `camera_groups_train399.json` | camera group of each of those 399 videos (70 groups). A group never spans two folds. |
| `scene_train399.csv` | per video: decoded duration (s) and the out-of-fold scene score of the scene-only reference (see below). |
| `folds_all499.json` | 5 camera-disjoint folds over all 499 usable videos (training + validation). We used them only to train the contest submission; nothing is held out. |

**How the groups were built.** Each video is represented by the SigLIP2 so400m embedding of its first frame
(576 patches, L2-normalised). k-means with k = 100 gives initial clusters; two clusters are merged whenever any two of
their first frames have cosine similarity >= 0.88. The groups are then split into 5 folds with scikit-learn's
`StratifiedGroupKFold`, stratified by static / dynamic / normal, keeping the most balanced of 50 shuffles. Visual
grouping is imperfect: a few same-camera pairs with cosine 0.86-0.88 may still cross folds.

**`vid0481` is excluded** everywhere: its annotated onset falls after its last frame.

## train399 at a glance

| fold | videos | static | dynamic | normal | same-camera normals |
|---|---|---|---|---|---|
| 0 | 73 | 17 | 16 | 40 | 37 |
| 1 | 96 | 29 | 27 | 40 | 5 |
| 2 | 75 | 20 | 15 | 40 | 5 |
| 3 | 78 | 14 | 24 | 40 | 2 |
| 4 | 77 | 20 | 17 | 40 | 32 |

**Same-camera normals** (81 in total, `iwdd.data.same_camera_negatives`) are normal videos from a camera that also
films dumping. Their background is the same as that of the dumping videos, so only what people do separates them.
The 37 in fold 0 come mostly from one camera that films dumping in 2 of its 34 videos; the 32 in fold 4 come from
cameras that film dumping about half the time. **Fold 4 is therefore the stress test** of whether a detector watches
the act; the paper reports its F1 separately.

## The scene-only reference

`scene_score` is the out-of-fold decision value of a logistic regression (C = 0.01, standardised inputs) on the
first-frame SigLIP2 embedding, trained on the other four folds of `folds_train399.json`
(`iwdd.audit.scene_scores`). It never sees the act, yet separates dumping from normal videos with AUROC 0.87. Answering
"video midpoint + 2 s" whenever it passes a threshold tuned on the other folds gives the **scene-only floor: F1 = 0.687**
in the 2027 window `[g-3, g+10]`. A detector that watches the act should clearly beat it.
