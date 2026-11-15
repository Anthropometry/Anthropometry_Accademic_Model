import numpy as np
import pytest
from PIL import Image

from anthro.preprocess import check_inputs, check_silhouette, crop_pad_resize, load_mask


def _body(h=300, w=200, top=20, bh=240, bw=60):
    m = np.zeros((h, w), bool)
    m[top:top + bh, w // 2 - bw // 2:w // 2 + bw // 2] = True
    return m


def test_output_shape_and_type():
    out = crop_pad_resize(_body(), 256, 128)
    assert out.shape == (256, 128) and out.dtype == np.uint8


def test_distance_from_camera_is_removed():
    """Same body, photographed nearer or further: after cropping the input should match (D-011)."""
    near = crop_pad_resize(_body(600, 400, 40, 480, 120), 256, 128).astype(float)
    far = crop_pad_resize(_body(300, 200, 20, 240, 60), 256, 128).astype(float)
    assert np.abs(near - far).mean() < 5          # out of 255


def test_proportions_are_not_stretched():
    wide = crop_pad_resize(_body(bh=150, bw=150), 256, 128) > 127
    cols = np.nonzero(wide.any(0))[0]
    rows = np.nonzero(wide.any(1))[0]
    # a square body stays roughly square after padding, instead of being squashed to 2:1
    assert 0.8 < (np.ptp(rows) + 1) / (np.ptp(cols) + 1) < 1.25


def test_empty_mask_is_refused():
    r = check_silhouette(np.zeros((100, 50), bool))
    assert not r.ok and "no body" in r.problems[0]


def test_full_frame_is_refused():
    assert not check_silhouette(np.ones((100, 50), bool)).ok


def test_cut_off_body_warns():
    m = _body()
    m[0, 100] = True
    assert check_silhouette(m).warnings


@pytest.mark.parametrize("h,w,ok", [(170, 70, True), (50, 70, False), (170, 400, False), (170, None, True)])
def test_input_limits(h, w, ok):
    assert check_inputs(h, w).ok is ok


def test_out_of_training_range_warns_but_accepts():
    r = check_inputs(205, 70)
    assert r.ok and r.warnings


def test_oversized_image_refused(tmp_path, monkeypatch):
    import anthro.preprocess as pp
    monkeypatch.setattr(pp, "_MAX_PIXELS", 100)
    p = tmp_path / "big.png"
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(p)
    with pytest.raises(ValueError, match="too large"):
        load_mask(p)
