"""Extract the frozen-encoder features used for training (one decode per video, GPU).

Writes, per video id:
  <out>/siglip576/<vid>.npz    p [F, 1152], a [F, 1152]   SigLIP2 pooled features (576 patches) of every 8 fps frame,
                                                           without (p) and with (a) a random CCTV-style text overlay
  <out>/vjepa_4fps/<vid>.npz   start [N], logits [N, 5, 174]   V-JEPA 2 SSv2 logits, 4-s clips every 1 s (detector)
  <out>/vjepa_8fps/<vid>.npz   start [N], logits [N, 5, 174]   2-s clips every 0.5 s (localizer)
All arrays fp16 (~0.5 GB for the 499 videos). Finished videos are skipped, so the script can be resumed.
Takes about 2 h for all 499 videos on one recent GPU.

  python scripts/extract_features.py --data /path/to/MIVIA-IWDD --scope all499 --out features
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iwdd import e0, vjepa  # noqa: E402
from iwdd.data import EXCLUDE, load_scope  # noqa: E402
from iwdd.system import load_encoders  # noqa: E402
from iwdd.video import GRID_FPS, read_frames  # noqa: E402

# The overlay style of a video depends only on its id (style i = its position among vid0001..vid0500 without the
# excluded video), so the augmented features do not depend on which videos you extract.
ALL_IDS = [f"vid{i:04d}" for i in range(1, 501) if f"vid{i:04d}" not in EXCLUDE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dataset root (default: $IWDD_DATA)")
    ap.add_argument("--scope", choices=["all499", "train399"], default="all499")
    ap.add_argument("--out", default="features")
    ap.add_argument("--vids", nargs="*", default=None, help="only these video ids")
    ap.add_argument("--decode-workers", type=int, default=4, help="videos decoded ahead of the GPU")
    ap.add_argument("--siglip-batch", type=int, default=64)
    a = ap.parse_args()
    labels, _ = load_scope(a.scope, a.data)
    vids = a.vids or sorted(labels)
    out = Path(a.out)
    dirs = {k: out / k for k in ("siglip576", "vjepa_4fps", "vjepa_8fps")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    todo = [v for v in vids if not all((d / f"{v}.npz").exists() for d in dirs.values())]
    print(f"{len(vids)} videos, {len(todo)} to extract -> {out}", flush=True)
    if not todo:
        return

    dev = "cuda"
    enc, proc, vj = load_encoders(dev, torch.float16)
    t0 = time.time()
    with ThreadPoolExecutor(a.decode_workers) as pool:
        for i, (v, frames) in enumerate(zip(todo, pool.map(lambda v: read_frames(labels[v]["video"]), todo))):
            if not frames:
                print(f"WARNING {v}: no frames decoded", flush=True)
                continue
            st = e0.overlay_style(ALL_IDS.index(v))
            over = [e0.draw_overlay(f, k / GRID_FPS, st) for k, f in enumerate(frames)]
            p = e0.siglip_features(enc, proc, frames, dev, torch.float16, a.siglip_batch)
            aug = e0.siglip_features(enc, proc, over, dev, torch.float16, a.siglip_batch)
            np.savez(dirs["siglip576"] / f"{v}.npz", p=p.astype(np.float16), a=aug.astype(np.float16))
            arr = np.stack(frames)
            for name, cfg in (("vjepa_4fps", vjepa.DETECTOR_CLIPS), ("vjepa_8fps", vjepa.LOCALIZER_CLIPS)):
                start, logits = vjepa.clip_logits(vj, arr, **cfg, dev=dev)
                np.savez(dirs[name] / f"{v}.npz", start=start, logits=logits.astype(np.float16))
            el = time.time() - t0
            print(f"{i + 1}/{len(todo)} {v}: {len(frames)} frames | {el / 60:.1f} min, "
                  f"ETA {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
