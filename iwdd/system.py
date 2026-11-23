"""The full IWDD 2027 system, as submitted (paper, Fig. 2).

Per video, after one decode at 8 fps (longest side <= 640 px):
  1. E0 margin m: frozen SigLIP2 features -> 5 LGT members -> mean of (top-5 logit - theta_k) / sigma_k.
  2. Action score a: V-JEPA 2 on 4-s clips x 5 views -> linear MIL head -> smoothed max over clips.
  3. Fusion z = ((m - mu_m) / sd_m + (a - mu_a) / sd_a) / 2; answer iff z >= threshold (statistics and threshold
     from out-of-fold predictions on the training videos).
  4. When: V-JEPA 2 on 2-s clips every 0.5 s -> localizer belief -> window-mass decoding for [g - 3, g + 10].

The two encoders are downloaded from Hugging Face at pinned revisions; the trained parts (e0_models/, params.npz,
about 300 MB) come from the release (scripts/download_weights.py) or from scripts/train.py.
"""
import json
import os

import numpy as np
import torch

from . import e0, vjepa, windowmass
from .video import GRID_FPS, read_frames

SIGLIP_ID, SIGLIP_REV = "google/siglip2-so400m-patch16-naflex", "cc24074f717b612951c2dead130904ab9b65a81e"
VJEPA_ID, VJEPA_REV = vjepa.MODEL_ID, "4aa02df83918538fc21cfaf576382fa20e489a80"


def load_encoders(dev, dtype):
    """(SigLIP2 vision tower, its image processor, V-JEPA 2 + SSv2 head), frozen, in `dtype` on `dev`."""
    from transformers import AutoImageProcessor, Siglip2VisionModel, VJEPA2ForVideoClassification
    enc = Siglip2VisionModel.from_pretrained(SIGLIP_ID, revision=SIGLIP_REV, dtype=dtype).eval().to(dev)
    # The torchvision backend is the one our features were computed with; the PIL backend gives slightly different
    # pixels, so we ask for torchvision explicitly rather than let transformers fall back silently.
    proc = AutoImageProcessor.from_pretrained(SIGLIP_ID, revision=SIGLIP_REV, backend="torchvision")
    vj = VJEPA2ForVideoClassification.from_pretrained(VJEPA_ID, revision=VJEPA_REV, dtype=dtype).eval().to(dev)
    return enc, proc, vj


def fuse(m_e0, s_vj, P):
    return ((m_e0 - float(P["z_mu"][0])) / float(P["z_sd"][0]) + (s_vj - float(P["z_mu"][1])) / float(P["z_sd"][1])) / 2


def load_members(weights_dir, dev):
    """E0 members as (model, theta, sigma), stored in fp16 and run in fp32."""
    meta = json.load(open(os.path.join(weights_dir, "e0_models", "members.json")))
    members = []
    for mm in meta:
        m = e0.LGTDetector(**e0.ARCH)
        sd = torch.load(os.path.join(weights_dir, "e0_models", mm["file"]), map_location="cpu", weights_only=True)
        m.load_state_dict({k: v.float() for k, v in sd.items()})
        members.append((m.eval().to(dev), mm["th"], mm["sigma"]))
    return members


class System:
    def __init__(self, weights_dir, dev=None):
        self.dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        self.P = dict(np.load(os.path.join(weights_dir, "params.npz")))
        self.enc, self.proc, self.vj = load_encoders(self.dev, self.dtype)
        self.members = load_members(weights_dir, self.dev)

    def predict(self, path):
        """-> (answer: onset in whole seconds as a string, or '' for no dumping; diagnostics dict)."""
        frames = read_frames(path)
        if not frames:
            return "", dict(n=0)
        F = e0.siglip_features(self.enc, self.proc, frames, self.dev, self.dtype)
        m_e0 = e0.margin(self.members, F, self.dev)
        arr = np.stack(frames)
        _, L4 = vjepa.clip_logits(self.vj, arr, **vjepa.DETECTOR_CLIPS, dev=self.dev)
        s_vj = vjepa.detector_score(L4, self.P)
        st8, L8 = vjepa.clip_logits(self.vj, arr, **vjepa.LOCALIZER_CLIPS, dev=self.dev)
        s, u = vjepa.localizer_curve(st8, vjepa.localizer_features(L8), self.P)
        t, mass = windowmass.decode(s, u, len(frames) / GRID_FPS, float(self.P["loc_T"]), windowmass.CONTEST)
        z = fuse(m_e0, s_vj, self.P)
        return (str(t) if z >= float(self.P["threshold"]) else ""), dict(n=len(frames), m_e0=m_e0, s_vj=s_vj, z=z,
                                                                          t=t, p_hit=mass)
