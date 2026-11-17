"""S3b geometry: the ellipse formula, width/depth measurement, levels, flags and refusals.

Synthetic bodies with known sizes, so every expected value is exact and checkable by hand.
"""

import numpy as np
import pytest

from anthro import geometry as geo

H = 1000  # body height in pixels -> 170 cm / 1000 px = 0.17 cm per pixel


def synthetic_body(arms_apart=True, torso_w=200, depth=150, gap_at_centre=False):
    """Head, torso, two legs; arms either clear of the torso (A-pose) or touching it."""
    f = np.zeros((H + 40, 600), bool)
    s = np.zeros((H + 40, 400), bool)
    t = 20
    f[t:t + 120, 260:340] = True                                        # head
    if gap_at_centre:                                                    # two blobs, nothing in the middle
        f[t + 120:t + 500, 150:280] = True
        f[t + 120:t + 500, 320:450] = True
    else:
        f[t + 120:t + 500, 300 - torso_w // 2:300 + torso_w // 2] = True  # torso
    f[t + 500:t + H, 220:290] = True                                     # legs
    f[t + 500:t + H, 310:380] = True
    left, right = 300 - torso_w // 2, 300 + torso_w // 2
    if arms_apart:
        f[t + 170:t + 480, left - 60:left - 10] = True
        f[t + 170:t + 480, right + 10:right + 60] = True
    else:
        f[t + 170:t + 480, left - 50:right + 50] = True
    s[t:t + 120, 170:230] = True
    s[t + 120:t + 500, 200 - depth // 2:200 + depth // 2] = True
    s[t + 500:t + H, 170:230] = True
    return f, s


def test_ramanujan_is_exact_for_a_circle():
    assert geo.ramanujan(10, 10) == pytest.approx(2 * np.pi * 10, rel=1e-12)


def test_ramanujan_matches_numerical_ellipse_perimeter():
    a, b = 16.0, 12.0
    t = np.linspace(0, 2 * np.pi, 400_001)
    numeric = np.hypot(np.diff(a * np.cos(t)), np.diff(b * np.sin(t))).sum()
    assert geo.ramanujan(a, b) == pytest.approx(numeric, rel=1e-4)


def test_width_and_depth_recovered_in_cm():
    f, s = synthetic_body(torso_w=200, depth=150)
    p = geo.profile(f, s, 170.0)
    feats = geo.level_features(p)
    cm = 170.0 / H
    for lv in geo.LEVELS:
        assert feats[lv]["W"] == pytest.approx(200 * cm, abs=cm)        # within one pixel
        assert feats[lv]["D"] == pytest.approx(150 * cm, abs=cm)
        assert feats[lv]["C0"] == pytest.approx(geo.ramanujan(100 * cm, 75 * cm), rel=0.01)
        assert not feats[lv]["merge"]                                   # arms clear: trusted


def test_arms_touching_the_body_are_flagged_not_trusted():
    f, s = synthetic_body(arms_apart=False)
    feats = geo.level_features(geo.profile(f, s, 170.0))
    assert all(feats[lv]["merge"] for lv in geo.LEVELS)


def test_scale_comes_from_height():
    f, s = synthetic_body()
    w150 = geo.level_features(geo.profile(f, s, 150.0))["waist"]["W"]
    w180 = geo.level_features(geo.profile(f, s, 180.0))["waist"]["W"]
    assert w180 / w150 == pytest.approx(180 / 150, rel=1e-6)


def test_empty_mask_is_refused():
    with pytest.raises(geo.GeometryError, match="no body"):
        geo.profile(np.zeros((500, 300), bool), np.zeros((500, 300), bool), 170.0)


def test_height_outside_limits_is_refused():
    f, s = synthetic_body()
    with pytest.raises(geo.GeometryError, match="outside"):
        geo.profile(f, s, 40.0)


def test_no_body_at_torso_centre_is_refused():
    f, s = synthetic_body(gap_at_centre=True)
    with pytest.raises(geo.GeometryError):
        geo.level_features(geo.profile(f, s, 170.0))


def test_batch_levels_agree_with_single_profile():
    profs = [geo.profile(*synthetic_body(torso_w=w, depth=d), 170.0) for w, d in ((180, 140), (220, 170))]
    w = np.stack([p["w_cm"] for p in profs])
    d = np.stack([p["d_cm"] for p in profs])
    n = np.stack([p["n_runs"] for p in profs])
    batch = geo.levels_batch(w, d, n)
    for i, p in enumerate(profs):
        one = geo.find_levels(p["w_cm"], p["d_cm"], p["n_runs"])
        assert {lv: int(batch["rows"][lv][i]) for lv in geo.LEVELS} == one["rows"]


def test_fullness_is_higher_for_a_heavier_body_with_the_same_outline():
    p = geo.profile(*synthetic_body(), 170.0)
    assert geo.fullness(p["sil_volume_l"], 90) > geo.fullness(p["sil_volume_l"], 60)


def test_edge_softness_flags_blurry_masks():
    crisp = np.zeros((100, 100), np.uint8)
    crisp[20:80, 30:70] = 255
    blurry = crisp.copy()
    blurry[20:80, 28:30] = 128
    blurry[20:80, 70:72] = 128
    assert geo.edge_softness(crisp) == 0.0
    assert geo.edge_softness(blurry) > 0.05
