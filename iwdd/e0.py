"""E0, the appearance detector (paper, Sec. 5.1).

Frozen SigLIP2 so400m frame features (576 NaFlex patches, pooled, 1152-d) -> two-layer local-global temporal (LGT)
adapter -> one logit per frame. Trained with the baseline's losses (Greco et al. 2026): MIL classification (attention
pooling for dumping clips, mean pooling for normal clips), soft-argmax onset loss and sparsity loss. Changes against
the baseline: 576 instead of 256 patches, a random CCTV-style text overlay on half of the training clips of both
classes (removes the on-screen-text shortcut), and five members, each early-stopped and calibrated on one held-out
camera fold. The video margin is the mean over members of (top-5 mean logit - theta_k) / sigma_k.
"""
import copy
import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF

from .metric import CONTEST_WINDOW, PAPER_WINDOW, evaluate
from .video import GRID_FPS

ARCH = dict(d=1152, layers=2, heads=8, win=8, drop=0.3)
WIN = 64                       # frames per clip (8 s)
STEP = 1.0 / GRID_FPS
SIGLIP_PATCHES = 576
OFFSET = 2.0                   # time readout used only to calibrate theta: soft-argmax + 2 s
POS_PRE = 5.0                  # a dumping clip starts 5 s before the onset
MAX_EPOCHS, PATIENCE, BATCH = 200, 25, 32
OVERLAY_P = 0.5                # share of training clips drawn from the overlay-augmented features


# ---------------- model ----------------
class LocalBlock(nn.Module):
    def __init__(s, d, heads, win, drop):
        super().__init__()
        s.win, s.heads = win, heads
        s.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        s.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        s.n1, s.n2, s.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(drop)

    def forward(s, x, m):
        L = x.shape[1]
        i = torch.arange(L, device=x.device)
        same_block = (i[:, None] // s.win) == (i[None, :] // s.win)
        eye = torch.eye(L, dtype=torch.bool, device=x.device)
        allowed = same_block[None] & (m[:, None, :] | eye[None])
        h = s.n1(x)
        x = x + s.drop(s.attn(h, h, h, attn_mask=~allowed.repeat_interleave(s.heads, 0), need_weights=False)[0])
        x = x + s.drop(s.ff(s.n2(x)))
        return x * m[..., None]


class GlobalGCN(nn.Module):
    def __init__(s, d, drop, sim_th=0.7, sigma=1.0):
        super().__init__()
        s.w_sim, s.w_dis, s.out = nn.Linear(d, d // 2), nn.Linear(d, d // 2), nn.Linear(d, d)
        s.n, s.drop, s.th, s.sigma = nn.LayerNorm(d), nn.Dropout(drop), sim_th, sigma

    def forward(s, x, m):
        L = x.shape[1]
        h = s.n(x)
        pad_col = ~m[:, None, :]
        xn = TF.normalize(h, dim=-1)
        sim = xn @ xn.transpose(1, 2)
        sim = sim.masked_fill(sim < s.th, -1e4).masked_fill(pad_col, -1e4)
        A_sim = torch.softmax(sim, -1)
        i = torch.arange(L, device=x.device).float()
        dis = torch.exp(-(i[:, None] - i[None, :]).abs() / s.sigma)[None].expand(x.shape[0], L, L)
        dis = dis.masked_fill(pad_col, 0)
        A_dis = dis / dis.sum(-1, keepdim=True).clamp_min(1e-6)
        g = torch.cat([A_sim @ s.w_sim(h), A_dis @ s.w_dis(h)], -1)
        x = x + s.drop(s.out(TF.gelu(g)))
        return x * m[..., None]


class LGTDetector(nn.Module):
    """Local-global temporal adapter (VadCLIP-style) + MLP head -> one logit per frame (padded frames -1e4)."""

    def __init__(s, d=1152, layers=2, heads=8, win=8, drop=0.3):
        super().__init__()
        s.blocks = nn.ModuleList([nn.ModuleList([LocalBlock(d, heads, win, drop), GlobalGCN(d, drop)])
                                  for _ in range(layers)])
        s.mlp = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop))
        s.cls = nn.Linear(4 * d, 1)

    def forward(s, x, m):
        x = x * m[..., None]
        for loc, glob in s.blocks:
            x = glob(loc(x, m), m)
        return s.cls(s.mlp(x)).squeeze(-1).masked_fill(~m, -1e4)


def clip(F, start, win=WIN):
    """Frames [start, start + win) of F, padded by repeating the last frame; mask marks real frames."""
    x = F[start:start + win]
    valid = len(x)
    if valid < win:
        x = np.concatenate([x, np.repeat(x[-1:], win - valid, 0)])
    m = np.zeros(win, bool)
    m[:valid] = True
    return x, m


@torch.no_grad()
def predict(model, F, dev):
    """Whole video in consecutive 64-frame clips -> (top-5 mean logit, soft-argmax time in s)."""
    zs = []
    for s in range(0, len(F), WIN):
        x, m = clip(F, s)
        z = model(torch.from_numpy(x[None]).to(dev), torch.from_numpy(m[None]).to(dev))[0]
        zs.append(z[:int(m.sum())].float().cpu().numpy())
    z = np.concatenate(zs)
    w = np.exp(z - z.max())
    w /= w.sum()
    idx = float((w * np.arange(len(z))).sum())
    t = float(np.interp(idx, np.arange(len(z)), np.arange(len(z)) * STEP))
    return float(np.sort(z)[-min(5, len(z)):].mean()), t


def margin(members, F, dev) -> float:
    """E0 video margin: mean over members (model, theta, sigma) of (score - theta) / sigma."""
    return float(np.mean([(predict(m, F, dev)[0] - th) / sg for m, th, sg in members]))


# ---------------- SigLIP2 features ----------------
@torch.no_grad()
def siglip_features(enc, proc, frames, dev, dtype, batch=8):
    """Pooled SigLIP2 features [n, 1152] for RGB uint8 frames, rounded to fp16 like the stored training features."""
    out = []
    for b in range(0, len(frames), batch):
        x = proc(images=frames[b:b + batch], max_num_patches=SIGLIP_PATCHES, return_tensors="pt")
        x = {k: v.to(dev) for k, v in x.items()}
        x["pixel_values"] = x["pixel_values"].to(dtype)
        out.append(enc(**x).pooler_output.float().cpu().numpy().astype(np.float16).astype(np.float32))
    return np.concatenate(out)


# ---------------- CCTV-style text overlay (training augmentation) ----------------
DATE_FMTS = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m-%d-%Y %I:%M:%S %p", "%d-%m-%Y %a %H:%M:%S"]
NAMES = ["CAM 01", "Camera 05", "IPCamera", "Polizia Locale", "CH2", "Garage 2", "ENTRANCE", "Via Roma", "NVR-3",
         "Street Cam"]
COLORS = [(255, 255, 255), (255, 255, 0), (0, 255, 0), (255, 255, 255)]


def overlay_style(i):
    """Overlay style number i (deterministic): position of date and camera name, date format, colour, size, logo."""
    r = np.random.default_rng(1000 + i)
    return {"date_pos": r.choice(["tl", "tr", "bl", "br"]), "name_pos": r.choice(["tl", "tr", "bl", "br"]),
            "fmt": DATE_FMTS[r.integers(len(DATE_FMTS))], "name": NAMES[r.integers(len(NAMES))],
            "color": COLORS[r.integers(len(COLORS))], "scale": float(r.uniform(0.7, 1.4)),
            "logo": bool(r.random() < 0.3),
            "start": datetime.datetime(2016 + int(r.integers(8)), 1 + int(r.integers(12)), 1 + int(r.integers(28)),
                                       int(r.integers(24)), int(r.integers(60)))}


def draw_overlay(img, t_sec, st):
    """Draw a running timestamp (and a camera name, maybe a small logo) on an RGB frame shown at t_sec seconds."""
    out = img.copy()
    h, w = out.shape[:2]
    fs = max(0.4, w / 640 * 0.6 * st["scale"])
    th = max(1, round(fs * 2))
    stamp = (st["start"] + datetime.timedelta(seconds=int(t_sec))).strftime(st["fmt"])

    def org(pos, text):
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x = int(0.02 * w) if pos[1] == "l" else w - tw - int(0.02 * w)
        y = int(0.03 * h) + tht if pos[0] == "t" else h - int(0.04 * h)
        return x, y

    items = [(stamp, st["date_pos"])] + ([(st["name"], st["name_pos"])] if st["name_pos"] != st["date_pos"] else [])
    for text, pos in items:
        o = org(pos, text)
        cv2.putText(out, text, o, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(out, text, o, cv2.FONT_HERSHEY_SIMPLEX, fs, st["color"], th, cv2.LINE_AA)
    if st["logo"]:
        s = max(4, int(0.02 * w))
        y0 = int(0.78 * h)
        for c, col in enumerate([(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 255, 0)]):
            cv2.rectangle(out, (int(0.03 * w) + c * (s + 2), y0), (int(0.03 * w) + c * (s + 2) + s, y0 + s), col, -1)
    return out


# ---------------- losses (Greco et al. 2026, Sec. 4.3) ----------------
def baseline_loss(logits, mask, y, tau_gt, w_cls=1.0, w_loc=0.5, w_sp=0.5, beta=1.0):
    """logits [B, T] (padded -1e4), mask [B, T], y [B] in {0, 1}, tau_gt [B]: onset frame (1-based) / T.
    L_cls: BCE on attention-pooled (dumping) or mean-pooled (normal) logits; L_loc: SmoothL1 between the soft-argmax
    centre of mass and tau_gt (dumping only); L_sp: mean sigmoid over frames (normal only)."""
    B, T = logits.shape
    valid = mask.float()
    n_valid = valid.sum(1)
    w = torch.softmax(logits, dim=1)
    z_att = (w * logits.masked_fill(~mask, 0.0)).sum(1)
    z_mean = (logits * valid).sum(1) / n_valid
    l_cls = TF.binary_cross_entropy_with_logits(torch.where(y > 0.5, z_att, z_mean), y)
    f = torch.arange(1, T + 1, device=logits.device, dtype=logits.dtype)
    tau_hat = (w * f).sum(1) / T
    l_loc = (TF.smooth_l1_loss(tau_hat, tau_gt, beta=beta, reduction="none") * y).mean()
    l_sp = ((torch.sigmoid(logits) * valid).sum(1) / n_valid * (1.0 - y)).mean()
    return w_cls * l_cls + w_loc * l_loc + w_sp * l_sp


# ---------------- training ----------------
def make_clip(feat, kind, onset, rng):
    """(x [64, 1152], mask, tau). Dumping: 8 s window from 5 s before the onset (shifted to fit). Normal: random."""
    F = len(feat)
    if kind == "pos":
        times = np.arange(F) * STEP
        of = min(int(np.searchsorted(times, onset - 1e-6)), F - 1)          # first frame at/after the onset
        start = int(np.clip(of - round(POS_PRE / STEP), 0, max(0, F - WIN)))
        tau = float(np.clip((of - start + 1) / WIN, 1e-3, 1.0))
    else:
        start = int(rng.integers(0, max(1, F - WIN + 1)))
        tau = 0.0
    x, m = clip(feat, start)
    return x, m, tau


def best_theta(scores, times, onsets):
    """theta maximising contest-window F1 of 'answer t + 2 s iff score >= theta' (ties: the lowest theta)."""
    ids = list(range(len(scores)))
    gt = {i: onsets[i] for i in ids}
    best_th, best_f1 = float(np.median(scores)), -1.0
    for th in np.unique(scores):
        f1 = evaluate(gt, {i: (times[i] + OFFSET if scores[i] >= th else None) for i in ids}, CONTEST_WINDOW).f1
        if f1 > best_f1:
            best_f1, best_th = f1, float(th)
    return best_th


def train_member(train_ids, val_ids, onsets, feats, feats_overlay, seed, dev, log=None):
    """Train one LGT member on train_ids; early-stop and calibrate on val_ids (unmodified features).

    feats / feats_overlay: vid -> float32 [F, 1152] without / with the text overlay.
    Returns (model, theta, sigma, best epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    sources = (feats, feats_overlay)
    mix_p = np.array([1.0 - OVERLAY_P, OVERLAY_P])
    model = LGTDetector(**ARCH).to(dev)
    head = {id(p) for p in model.cls.parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head], "lr": 8e-5, "weight_decay": 1e-4},
        {"params": list(model.cls.parameters()), "lr": 1e-4, "weight_decay": 5e-5}])
    items = [(v, "pos" if onsets[v] is not None else "neg") for v in train_ids]
    gts = [onsets[v] for v in val_ids]

    def val_outputs():
        model.eval()
        out = [predict(model, feats[v], dev) for v in val_ids]
        return np.array([o[0] for o in out]), [o[1] for o in out]

    best, best_state, best_ep, wait = None, None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(items))
        for b in range(0, len(order), BATCH):
            xs, ms, ys, taus = [], [], [], []
            for i in order[b:b + BATCH]:
                v, kind = items[i]
                src = sources[rng.choice(2, p=mix_p)]
                x, m, tau = make_clip(src[v], kind, onsets[v], rng)
                xs.append(x); ms.append(m); ys.append(1.0 if kind == "pos" else 0.0); taus.append(tau)
            x = torch.from_numpy(np.stack(xs)).to(dev)
            m = torch.from_numpy(np.stack(ms)).to(dev)
            y = torch.tensor(ys, device=dev)
            tau = torch.tensor(taus, device=dev, dtype=torch.float32)
            loss = baseline_loss(model(x, m), m, y, tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        # early stopping: answer at the soft-argmax time iff sigmoid(score) > 0.5, F1 in the [g-5, g+10] window
        sc, tm = val_outputs()
        f1 = evaluate(dict(enumerate(gts)),
                      {i: (tm[i] if 1 / (1 + np.exp(-sc[i])) > 0.5 else None) for i in range(len(gts))},
                      PAPER_WINDOW).f1
        if best is None or f1 > best:
            best, best_ep, wait, best_state = f1, epoch, 0, copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    sc, tm = val_outputs()
    return model, best_theta(sc, tm, gts), float(sc.std()) or 1.0, best_ep
