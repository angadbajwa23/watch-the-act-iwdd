"""Action detector and onset localizer on V-JEPA 2 action scores (paper, Secs. 5.2 and 5.3).

The frozen V-JEPA 2 ViT-L with its released Something-Something-v2 head scores 174 hand-object actions. Every clip is
seen in five 256x256 views: the whole frame and four overlapping tiles of 60 % of its width and height, so small or
distant acts fill more of the input. We read ONLY these 174 action logits; a head on the 1024-d features under them
learns the scene again (paper, Sec. 5.2).

Action detector: 4-s clips (16 frames at 4 fps), one per second. A linear head maps one view's 174 logits to a
dumping score, trained by multiple-instance learning: per dumping video the positive is the best view of each clip
that contains the onset, negatives are every view of clips of normal videos and of clips far from the onset.
Video score = max over clips of the 3-clip moving average of the max over views.

Onset localizer: 2-s clips (16 frames at 8 fps) every 0.5 s. Clip features = [whole-frame logits, max over tiles,
mean over tiles] and their change from the previous to the next clip. A linear scorer + softmax over the clips of a
video, trained on dumping videos only against a Gaussian (sigma 1 s) centred on the onset. The belief is read out by
window-mass decoding (iwdd.windowmass).
"""
import numpy as np
import torch
import torch.nn.functional as TF

from .video import GRID_FPS

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
CLIP, SIZE, TILE_FRAC, N_VIEWS = 16, 256, 0.6, 5
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DETECTOR_CLIPS = dict(fps=4, hop=1.0)      # 4-s clips every 1 s
LOCALIZER_CLIPS = dict(fps=8, hop=0.5)     # 2-s clips every 0.5 s
LOC_ANCHOR = 0.5                           # a 2-s clip stands for the onset time start + 0.5 s (clip length / 4)
DUMP_CLASSES = [8, 9, 10, 11, 12,          # Dropping [something] behind / in front of / into / next to / onto
                102, 104, 105, 106, 107, 108, 109, 112, 114,   # Putting [something] ...
                151, 155,                  # Throwing [something]; Throwing [something] onto a surface
                159]                       # Tipping [something] with [something in it] over, so it falls out


# ---------------- V-JEPA 2 clip logits ----------------
def make_views(x):
    """x [B, T, 3, H, W] in [0, 1] -> [B, 5, T, 3, 256, 256]: whole frame + 2x2 overlapping tiles."""
    B, T, C, H, W = x.shape
    flat = x.reshape(B * T, C, H, W)
    rs = lambda im: TF.interpolate(im, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    views = [rs(flat)]
    th, tw = int(round(H * TILE_FRAC)), int(round(W * TILE_FRAC))
    for i in range(2):
        for j in range(2):
            y0, x0 = round(i * (H - th)), round(j * (W - tw))
            views.append(rs(flat[:, :, y0:y0 + th, x0:x0 + tw]))
    return torch.stack(views, 1).reshape(B, T, len(views), C, SIZE, SIZE).transpose(1, 2)


def clip_starts(n, fps, hop_s):
    """Start frames: one every hop_s seconds while at least 1 s of real frames remains (at least one clip)."""
    step = max(1, int(round(hop_s * fps)))
    last = max(0, n - fps)
    return list(range(0, last + 1, step)) if n > fps else [0]


@torch.no_grad()
def clip_logits(model, frames, fps, hop, dev):
    """frames: [n, H, W, 3] uint8 on the 8 fps grid -> (clip starts in s [N], SSv2 logits [N, 5, 174]).

    Clips shorter than 16 frames are padded by repeating the last frame. Frames stay on the CPU; one clip (5 views)
    goes through the model at a time. Logits are rounded to fp16 like the stored training features."""
    fr = frames[::GRID_FPS // fps]
    n = len(fr)
    starts = clip_starts(n, fps, hop)
    idx = np.array([[min(s + j, n - 1) for j in range(CLIP)] for s in starts])
    fr = torch.from_numpy(np.ascontiguousarray(fr))
    mean = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1).half()
    std = torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1).half()
    out = []
    for b in range(len(idx)):
        x = fr[torch.from_numpy(idx[b:b + 1])].to(dev).permute(0, 1, 4, 2, 3).half().div(255)
        x = make_views(x)
        x = x.reshape(N_VIEWS, *x.shape[2:]).sub(mean).div(std)
        hid = model.vjepa2(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        out.append(model.classifier(model.pooler(hid)).float().cpu().reshape(-1, N_VIEWS, 174))
    logits = torch.cat(out).numpy().astype(np.float16).astype(np.float32)
    return np.array(starts, np.float32) / fps, logits


# ---------------- action detector (MIL linear head) ----------------
def clip_labels(onset, start):
    """Per 4-s clip: 1 = contains the onset (starts in [g - 3, g]), 0 = far from it or a normal video, -1 = ignored."""
    if onset is None:
        return np.zeros(len(start), int)
    lab = np.full(len(start), -1)
    lab[(start >= onset - 3) & (start <= onset)] = 1
    lab[(start + 4 <= onset - 1) | (start >= onset + 6)] = 0
    return lab


def _lse(x, axis=-1):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def fit_detector(clips, onsets, train_vids, C=0.1, rounds=2):
    """MIL training. clips: vid -> (starts [N], logits [N, 5, 174]). Returns (score_fn, params).

    Round 0 picks, in each onset clip, the view most likely to show a drop/put/throw action (zero-shot over
    DUMP_CLASSES); every later round re-picks the view with the current head."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    pick = {v: _lse(clips[v][1][..., DUMP_CLASSES]) - _lse(clips[v][1]) for v in train_vids}
    for _ in range(rounds):
        X, y = [], []
        for v in train_vids:
            st, L = clips[v]
            lab = clip_labels(onsets[v], st)
            for i in np.where(lab == 1)[0]:
                X.append(L[i, int(np.argmax(pick[v][i]))][None]); y.append([1])
            neg = np.where(lab == 0)[0]
            if len(neg):
                X.append(L[neg].reshape(-1, L.shape[-1])); y.append(np.zeros(len(neg) * L.shape[1], int))
        X, y = np.concatenate(X), np.concatenate(y)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
        def score(L, sc=sc, clf=clf):
            return clf.decision_function(sc.transform(L.reshape(-1, L.shape[-1]))).reshape(L.shape[:2])
        pick = {v: score(clips[v][1]) for v in train_vids}
    params = dict(mil_mean=sc.mean_, mil_scale=sc.scale_, mil_coef=clf.coef_[0], mil_intercept=float(clf.intercept_[0]))
    return score, params


def smoothed_max(view_scores):
    """[N, 5] clip x view scores -> video score: max over clips of the 3-clip moving average of the max over views."""
    s = view_scores.max(1)
    if len(s) > 2:
        pad = np.concatenate([s[:1], s, s[-1:]])
        s = (pad[:-2] + pad[1:-1] + pad[2:]) / 3          # = scipy uniform_filter1d(s, 3, mode="nearest")
    return float(s.max())


def detector_score(logits, P):
    """Video score from the stored head parameters (inference; no scikit-learn needed)."""
    z = ((logits - P["mil_mean"]) / P["mil_scale"]) @ P["mil_coef"] + float(P["mil_intercept"])
    return smoothed_max(z)


# ---------------- onset localizer ----------------
def localizer_features(logits):
    """[N, 5, 174] -> [N, 522]: whole frame, max over tiles, mean over tiles."""
    return np.concatenate([logits[:, 0], logits[:, 1:].max(1), logits[:, 1:].mean(1)], 1)


def _context(f):
    """[f_k, f_{k+1} - f_{k-1}] per clip (edges repeat the first / last clip)."""
    pad = np.concatenate([f[:1], f, f[-1:]])
    return np.concatenate([pad[1:-1], pad[2:] - pad[:-2]], 1).astype(np.float32)


def fit_localizer(seqs, onsets, train_vids, wd=1.0, epochs=300, lr=1e-2, sigma=1.0, seed=0):
    """seqs: vid -> (starts [N], features [N, 522]). Trained on the dumping videos among train_vids."""
    torch.manual_seed(seed)
    pos = [v for v in train_vids if onsets[v] is not None]
    X = [_context(seqs[v][1]) for v in pos]
    allx = np.concatenate(X)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    n = max(len(x) for x in X)
    Xb = torch.zeros(len(X), n, X[0].shape[1])
    M = torch.zeros(len(X), n, dtype=torch.bool)
    T = torch.zeros(len(X), n)
    for i, (v, x) in enumerate(zip(pos, X)):
        st = seqs[v][0]
        k = np.clip(onsets[v] - LOC_ANCHOR, 0, st[-1])
        t = np.exp(-0.5 * ((st - k) / sigma) ** 2)
        Xb[i, :len(x)] = torch.from_numpy((x - mu) / sd)
        M[i, :len(x)] = True
        T[i, :len(x)] = torch.from_numpy((t / t.sum()).astype(np.float32))
    w = torch.zeros(Xb.shape[2], requires_grad=True)
    opt = torch.optim.Adam([w], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        s = (Xb @ w).masked_fill(~M, -1e9)
        loss = -(T * torch.log_softmax(s, 1)).sum(1).mean()
        loss.backward()
        opt.step()
    return dict(loc_w=w.detach().numpy(), loc_mu=mu, loc_sd=sd, loc_anchor=LOC_ANCHOR)


def localizer_curve(starts, feats, P):
    """(clip scores s_j, onset times u_j = start + anchor) for window-mass decoding."""
    s = ((_context(feats) - P["loc_mu"]) / P["loc_sd"]) @ P["loc_w"]
    return s, starts.astype(np.float64) + float(P["loc_anchor"])
