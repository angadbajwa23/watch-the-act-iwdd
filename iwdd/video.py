"""Video decoding on an exact 8 fps grid (OpenCV only; no torch, safe in worker processes).

For every grid time k / 8 s we take the real decoded frame nearest to it, using the frame times OpenCV reports rather
than the stated frame rate (12 MIVIA-IWDD videos play at a different rate than their metadata says). Frames are shrunk
to a longest side of at most 640 px. Frame k of every feature stream therefore sits at time k / 8 s.
"""
import cv2

GRID_FPS = 8
MAX_SIDE = 640


def _prep(f):
    h, w = f.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)


def grid_frames(path):
    """Yield RGB uint8 frames, one per 1/8 s, from 0 s to the last decoded frame."""
    step = 1.0 / GRID_FPS
    cap = cv2.VideoCapture(path)
    prev_t = prev_img = None
    k = 0
    while cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, f = cap.retrieve()
        if not ok or f is None:
            continue
        cur = _prep(f)
        while k * step <= t + 1e-9:
            g = k * step
            yield prev_img if (prev_t is not None and (g - prev_t) < (t - g)) else cur
            k += 1
        prev_t, prev_img = t, cur
    cap.release()


def read_frames(path):
    """All grid frames as a list of [H, W, 3] uint8 arrays (empty if nothing could be decoded)."""
    return list(grid_frames(path))
