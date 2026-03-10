from __future__ import annotations

import numpy as np

from track_ap_ellipse_all_frames import circle_seed_points_from_ellipse, smooth_ellipse


def test_circle_seed_points_from_ellipse() -> None:
    ellipse = ((100.0, 120.0), (80.0, 40.0), 25.0)
    pts = circle_seed_points_from_ellipse(ellipse, n_pts=32)
    assert len(pts) == 32

    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    assert abs(cx - 100.0) < 2.0
    assert abs(cy - 120.0) < 2.0


def test_smooth_ellipse_limits_large_jump() -> None:
    prev = ((200.0, 200.0), (100.0, 80.0), 10.0)
    curr = ((500.0, 500.0), (220.0, 20.0), 140.0)
    smoothed = smooth_ellipse(prev, curr, (600, 600, 3), alpha=0.5)

    (cx, cy), (maj, minor), _ = smoothed
    dist = float(np.hypot(cx - prev[0][0], cy - prev[0][1]))

    assert dist < 60.0
    assert 65.0 <= maj <= 135.0
    assert 52.0 <= minor <= 108.0

