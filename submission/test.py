"""IWDD 2027 submission: one TXT file per video with the dumping onset, or empty.

  python submission/test.py --videos <folder> --results <folder> [--weights weights/]

Same input / output as the organizers' sample test.py: every file in --videos is read, and <video file name>.txt is
written to --results with the onset in whole seconds, or nothing if no dumping is detected. A video that cannot be
read gets an empty file. Peak GPU memory is printed at the end (2.58 GB on a Colab T4 for a 180-s video).
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=str, default="foo_videos/")
    ap.add_argument("--results", type=str, default="foo_results/")
    ap.add_argument("--weights", type=str, default=str(REPO / "weights"))
    ap.add_argument("--verbose", action="store_true", help="print the per-video scores")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)

    from iwdd.system import System
    t_load = time.time()
    system = System(a.weights)
    print(f"models loaded in {time.time() - t_load:.0f} s | device {system.dev}", flush=True)

    files = sorted(f for f in os.listdir(a.videos)
                   if not f.startswith(".") and os.path.isfile(os.path.join(a.videos, f)))
    print(f"{len(files)} files", flush=True)
    t_all = time.time()
    for name in files:
        t0, answer, info = time.time(), "", {}
        try:
            answer, info = system.predict(os.path.join(a.videos, name))
        except Exception:
            print(f"ERROR on {name}; writing an empty result")
            traceback.print_exc()
        with open(os.path.join(a.results, name + ".txt"), "w") as f:
            f.write(answer)
        extra = (f" | E0 {info['m_e0']:+.2f} V-JEPA {info['s_vj']:+.2f} z {info['z']:+.2f} t {info['t']}"
                 if a.verbose and "z" in info else "")
        print(f'{name}: {info.get("n", 0)} frames -> "{answer}" ({time.time() - t0:.1f} s){extra}', flush=True)
    print(f"total {time.time() - t_all:.0f} s for {len(files)} videos")
    if system.dev == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
