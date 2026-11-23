import numpy as np
import pytest

torch = pytest.importorskip("torch")
scipy_ndimage = pytest.importorskip("scipy.ndimage")

from iwdd import e0, vjepa  # noqa: E402


def test_lgt_detector_masks_padding():
    m = e0.LGTDetector(**e0.ARCH).eval()
    x = torch.randn(2, e0.WIN, 1152)
    mask = torch.ones(2, e0.WIN, dtype=torch.bool)
    mask[1, 40:] = False
    z = m(x, mask)
    assert z.shape == (2, e0.WIN) and torch.all(z[1, 40:] == -1e4)


def test_e0_predict_handles_short_and_long_videos():
    m = e0.LGTDetector(**e0.ARCH).eval()
    for n in (3, 64, 150):
        s, t = e0.predict(m, np.random.default_rng(n).normal(size=(n, 1152)).astype(np.float32), "cpu")
        assert np.isfinite(s) and 0 <= t <= (n - 1) / 8


def test_overlay_style_is_deterministic():
    a, b = e0.overlay_style(7), e0.overlay_style(7)
    assert a == b
    img = np.zeros((360, 640, 3), np.uint8)
    assert e0.draw_overlay(img, 3.0, a).any()


def test_detector_score_matches_scipy_smoothing():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(9, 5, 174)).astype(np.float32)
    P = dict(mil_mean=rng.normal(size=174), mil_scale=rng.uniform(0.5, 2, 174), mil_coef=rng.normal(size=174),
             mil_intercept=np.float64(0.3))
    z = ((L - P["mil_mean"]) / P["mil_scale"]) @ P["mil_coef"] + 0.3
    ref = scipy_ndimage.uniform_filter1d(z.max(1), 3, mode="nearest").max()
    assert vjepa.detector_score(L, P) == pytest.approx(ref)


def test_clip_starts():
    assert vjepa.clip_starts(3, 4, 1.0) == [0]                      # under 1 s: one clip
    assert vjepa.clip_starts(40, 4, 1.0) == list(range(0, 37, 4))   # 10 s at 4 fps: one clip per second
    assert vjepa.clip_starts(80, 8, 0.5) == list(range(0, 73, 4))   # 10 s at 8 fps: every 0.5 s


def test_views_shape():
    x = torch.rand(1, 16, 3, 360, 640)
    assert vjepa.make_views(x).shape == (1, 5, 16, 3, 256, 256)


def test_clip_labels():
    st = np.arange(0, 20, 1.0)
    lab = vjepa.clip_labels(10.0, st)
    assert set(np.where(lab == 1)[0]) == {7, 8, 9, 10}
    assert lab[5] == 0 and lab[16] == 0 and lab[12] == -1
    assert np.all(vjepa.clip_labels(None, st) == 0)
