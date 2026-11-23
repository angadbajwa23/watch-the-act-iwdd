# Watch the Act, Not the Scene

Code for **"Watch the Act, Not the Scene: Detecting and Timing Illegal Waste Dumping"**
(Shreya Ajay Kale and Angad Ripudaman Singh Bajwa, WACV 2027 Workshops), and the system we submitted to the
IWDD 2027 challenge.

The task: given one untrimmed surveillance video, say whether someone dumps waste and, if so, when it starts. An
answer counts if it lands in `[g-3, g+10]` s around the annotated onset `g` (`[g-5, g+10]` in 2026); a mistimed answer
counts as both a false positive and a false negative.

On MIVIA-IWDD we found that a strong appearance-based detector mostly learns *where* a video was filmed, not *what*
happens in it: on unseen cameras, a classifier that sees only the first frame does as well. This repository contains:

| | |
|---|---|
| [`iwdd/metric.py`](iwdd/metric.py) | the exact challenge scorer (both windows, error categories, bootstrap CI) |
| [`splits/`](splits/) | camera-disjoint folds, camera groups and the scene-only reference |
| [`iwdd/audit.py`](iwdd/audit.py) | the five-check shortcut audit: does your detector watch the act or the scene? |
| [`iwdd/windowmass.py`](iwdd/windowmass.py) | window-mass decoding: the F1-optimal timestamp under a tolerance window |
| [`submission/`](submission/) + released weights | the system we submitted, ready to run |
| [`scripts/`](scripts/) | feature extraction and training, to rebuild that system from the videos |

## Results

| | unseen cameras (nested CV, 399 videos) | official validation split | hidden 2027 test (new locations) | peak GPU memory |
|---|---|---|---|---|
| Baseline (Greco et al. 2026) | 0.575 (our reproduction) | 0.79 (published) | – | 3.15 GB |
| Scene-only floor (first frame) | 0.687 | – | – | – |
| **Ours** | **0.808 ± 0.001** | **0.86** (P 0.86, R 0.86) | 0.68 (P 0.63, R 0.74) | **2.58 GB** |

F1 in the 2027 window, except the hidden test, which the organizers scored in the 2026 window `[g-5, g+10]`. The
validation number comes from a system trained on the 399 training videos and run once; 61 of those 100 videos come
from camera sites seen in training, so we made every design choice on the camera-disjoint cross-validation. Most of the
drop on the hidden test is precision: false alarms in new scenes remain the main open problem (see
[Limitations](#limitations)).

## Install

```bash
git clone https://github.com/angadbajwa23/watch-the-act-iwdd.git && cd watch-the-act-iwdd
pip install -e .            # scorer, splits, audit, window-mass decoding (numpy, scipy, scikit-learn)
pip install -e ".[model]"   # + the detector (PyTorch, transformers, OpenCV)
```
[`requirements.txt`](requirements.txt) lists the exact versions our results were verified with.

**Data.** MIVIA-IWDD-500 is not redistributed here. Get it from
[MiviaLab/IllegalWasteDumping](https://github.com/MiviaLab/IllegalWasteDumping) and point `IWDD_DATA` at the folder
that holds `TRAINING_SET/` and `VALIDATION_SET/` (each with `videos/` and `labels/`). The loader handles the label
quirks: `label` is the string `"true"`/`"false"` and onsets are whole seconds. `vid0481` is excluded because its onset
falls after its last frame. Frames are sampled on an exact 8 fps grid by their decoded timestamps, because 12 videos
play at a different rate than their metadata says.

## 1. Score and evaluate on unseen cameras

```python
from iwdd.metric import evaluate, CONTEST_WINDOW
r = evaluate(gt, pred, CONTEST_WINDOW)     # gt / pred: video id -> seconds, or None for "no dumping"
print(r.f1, r.counts())                    # {'tp', 'mistimed', 'missed', 'false_alarm', 'tn'}

from iwdd.data import load_folds
folds = load_folds("train399")             # video id -> fold 0..4; no camera appears in two folds
```

Train on four folds, predict the fifth, and report F1 over the pooled out-of-fold predictions. Fit everything
(thresholds included) inside the training folds. [`splits/README.md`](splits/README.md) describes how the folds were
built and why fold 4 is the stress test.

## 2. Audit your detector

Write out-of-fold predictions on `folds_train399.json` as a CSV with columns `video,score,answer` (`answer` empty for
"no dumping"). Optionally, write per-frame scores as an NPZ with one `[N, 2]` array (time in s, score) per video.
Then run:

```bash
python -m iwdd.audit --predictions my_oof.csv --curves my_curves.npz
```

Our submitted system (seed 42) gives:
```
videos 399 | F1 0.808 (P 0.83, R 0.78) | TP 156, mistimed 9, missed 34, false alarms 22
1. beats the scene?    scene-only floor 0.687 -> F1 - floor +0.121
2. follows the scene?  rho on normal videos 0.77  (lower is better)
3. same-camera FAs     19% of 81 same-camera vs 6% of 119 other | AUROC vs same-camera 0.900, vs other 0.981
4. knows when?         partial r 0.66 | hit rate 0.95 vs 'midpoint + 2 s' 0.85 on the same videos
5. sees the act?       within-video AUROC 0.73 over 133 videos
```

| Paper, Table 1 (means over 3 seeds) | F1 | vs. floor | ρ scene ↓ | false alarms: same camera / other | AUROC vs. same camera ↑ | partial r ↑ | act AUROC ↑ |
|---|---|---|---|---|---|---|---|
| Baseline, frozen (reproduced)* | 0.575 | −0.131 | 0.69 | 62% / 10% | 0.70 | 0.03 | 0.64 |
| E0 appearance detector | 0.710 | +0.023 | 0.85 | 44% / 5% | 0.80 | 0.25 | 0.64 |
| V-JEPA action detector | 0.724 | +0.037 | 0.24 | 26% / 13% | 0.84 | 0.80 | 0.73 |
| Full system (final) | 0.810 | +0.123 | 0.76 | 33% / 4% | 0.88 | 0.68 | 0.73 |

\* The baseline rows were computed on an earlier partition built the same way (scene-only floor 0.706 there).

What the checks mean: (1) F1 minus the scene-only floor on the same folds; (2) on normal videos, the rank correlation
between your score and the first-frame scene score; (3) false alarms on normal videos from cameras that also film
dumping, plus the threshold-free AUROC; (4) whether your timestamp follows the onset beyond "the middle of the video";
(5) whether frame scores rise during the act compared with earlier frames of the same video. The docstring of
[`iwdd/audit.py`](iwdd/audit.py) and Sec. 4 of the paper give the details.

## 3. Window-mass decoding

The scoring window is wide (13 s) and asymmetric, so the best answer is not the most likely onset. It is the time
whose window holds the most onset probability. Once the set of answered videos is fixed, F1 is proportional to the
number of hits, so this choice maximizes expected F1. It needs no training and works with any score curve:

```python
from iwdd.windowmass import decode, choose_temperature
answer, p_hit = decode(scores, times, duration, T=0.5, window=(3, 10))   # whole seconds, P(answer is a hit)
```

`scores[j]` is any score for "the event starts at `times[j]`". The belief is `softmax(scores / T)`, and the single
setting `T` is chosen on out-of-fold predictions with `choose_temperature`. Applied to the reproduced baseline, it
raises unseen-camera F1 from 0.575 to 0.686 without retraining.

## 4. Run the submitted system

```bash
python scripts/download_weights.py                   # our trained parts (~300 MB) -> weights/
python submission/test.py --videos my_videos/ --results my_results/
```

This writes `my_results/<video file name>.txt` with the onset in whole seconds, or an empty file for "no dumping",
which is the challenge's I/O format. On first use, SigLIP2 so400m and V-JEPA 2 ViT-L are downloaded from Hugging Face
at pinned revisions (~6 GB). On a Colab T4 a video takes about 50 s and peak GPU memory is 2.58 GB. See
[`submission/colab.ipynb`](submission/colab.ipynb).

> The released weights are the contest submission. They were trained on all 499 usable videos, **including
> `VALIDATION_SET`**, so do not evaluate them on it. To reproduce the 0.86 on the validation split, train with
> `--scope train399` (below).

How it works (paper, Fig. 2):
- **Appearance detector (E0).** Frozen SigLIP2 features (576 patches per frame) feed 5 local-global temporal models,
  trained with a random CCTV-style text overlay so that on-screen text stops predicting the label.
- **Action detector.** Frozen V-JEPA 2, read only through its 174 Something-Something-v2 action scores, on 4-s clips
  in five views (the whole frame and four zoomed tiles), with a linear head on top.
- **Decision.** The two scores are standardized and averaged, and the system answers when the average passes a
  threshold.
- **Timing.** A linear onset localizer on V-JEPA 2 scores of 2-s clips gives the onset belief, read out by
  window-mass decoding.

## 5. Train it yourself

```bash
python scripts/extract_features.py --scope all499 --out features              # ~2 h on one GPU, 0.5 GB
python scripts/train.py --features features --scope all499 --out weights_new   # ~12 min
python submission/test.py --weights weights_new --videos my_videos/ --results my_results/
```

`--scope all499` is exactly how the submission was built. `--scope train399` builds the system we evaluated once on
the validation split. `train.py` prints its out-of-fold F1. For all499 it should print `contest F1 0.829`,
`T = 0.5` and `threshold -0.0891`. Its threshold is tuned on those same predictions, so it is optimistic; the nested
estimate is the 0.808 above.

Reproducibility, as we verified it on one GPU (RTX PRO 4500 Blackwell):
- Given our cached features, `train.py` reproduces the submitted weights bit for bit: the same five E0 members
  (epochs, thresholds, weights), action head, localizer, fusion statistics, temperature and threshold.
- Re-extracted V-JEPA features are bit-identical.
- Re-extracted SigLIP2 features agree with ours only up to float16 GPU noise (cosine ≥ 0.9995), so a full retrain
  gives slightly different E0 members.
- The inference code reproduces the submitted system's scores and answers bit for bit.

## Limitations

- On the hidden test set from new locations, precision fell to 0.63. The fused score still follows the scene
  (ρ = 0.76 on normal videos), and a new place that looks like a dumping spot can trigger false alarms that no
  threshold tuned on 499 videos anticipates.
- It misses acts that stay tiny even in a zoomed tile, and sometimes fires when people move around parked cars.
- Everything was developed on one dataset (about 70 training cameras); no second dumping benchmark was available.
- It is a research prototype. It is not suitable for automated enforcement; every alarm needs human review.

## Citation

```bibtex
@inproceedings{kale2027watch,
  author    = {Kale, Shreya Ajay and Bajwa, Angad Ripudaman Singh},
  title     = {Watch the Act, Not the Scene: Detecting and Timing Illegal Waste Dumping},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```
Please also cite the dataset:
```bibtex
@article{greco2026benchmarking,
  author  = {Greco, Antonio and Ricciardi, Andrea Vincenzo and Sansone, Carlo and Vento, Bruno},
  title   = {Benchmarking illegal waste dumping detection: A public video dataset and a reference baseline},
  journal = {Image and Vision Computing},
  volume  = {174},
  pages   = {106125},
  year    = {2026},
  doi     = {10.1016/j.imavis.2026.106125}
}
```

## License

The code is MIT-licensed ([`LICENSE`](LICENSE)). The released weights were trained on MIVIA-IWDD; follow the dataset's
terms when you use them. The frozen encoders are downloaded from their original sources:
[SigLIP2](https://huggingface.co/google/siglip2-so400m-patch16-naflex) (Apache-2.0) and
[V-JEPA 2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) (MIT).

We thank the MIVIA Lab (University of Salerno) for the dataset and for organizing the IWDD challenge.
