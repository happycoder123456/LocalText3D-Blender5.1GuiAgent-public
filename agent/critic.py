"""Reward critic: SSIM + viewport absdiff → float score."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _viewport_crop(image: np.ndarray) -> np.ndarray:
    """Heuristic: drop top/left UI chrome; keep central 3D viewport."""
    h, w = image.shape[:2]
    top = int(h * 0.08)
    left = int(w * 0.12)
    bottom = int(h * 0.96)
    right = int(w * 0.98)
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    bottom = max(top + 1, min(bottom, h))
    right = max(left + 1, min(right, w))
    return image[top:bottom, left:right]


def _mean_absdiff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    ga = _to_gray(a).astype(np.float32)
    gb = _to_gray(b).astype(np.float32)
    return float(np.mean(np.abs(ga - gb)) / 255.0)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    ga = _to_gray(a)
    gb = _to_gray(b)
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]), interpolation=cv2.INTER_AREA)
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return 1.0 - _mean_absdiff(a, b)
    # Win size must be odd and <= min dimension.
    side = min(ga.shape[0], ga.shape[1])
    win = min(7, side if side % 2 == 1 else side - 1)
    if win < 3:
        return 1.0 - _mean_absdiff(a, b)
    score = structural_similarity(ga, gb, win_size=win)
    return float(score)


def _viewport_looks_empty(viewport: np.ndarray) -> bool:
    gray = _to_gray(viewport).astype(np.float32)
    # Near-uniform / dark viewport after a destructive action.
    return float(np.std(gray)) < 8.0


def score_transition(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    """
    Map visual change to a reward float.

    near-zero change → small negative (missed)
    moderate viewport change → positive
    huge change + empty-looking viewport → large negative (accidental delete)
    """
    if before is None or after is None or before.size == 0 or after.size == 0:
        return {"reward": -0.5, "ssim": 0.0, "viewport_delta": 0.0, "note": "missing frames"}

    ssim = _ssim(before, after)
    change = 1.0 - ssim
    vp_before = _viewport_crop(before)
    vp_after = _viewport_crop(after)
    vp_delta = _mean_absdiff(vp_before, vp_after)
    empty = _viewport_looks_empty(vp_after)

    if empty and vp_delta > 0.15:
        reward = -1.0
        note = "large change and empty viewport (possible delete)"
    elif change < 0.01 and vp_delta < 0.01:
        reward = -0.3
        note = "almost no change (missed click / empty space)"
    elif 0.02 <= vp_delta <= 0.35:
        # Meaningful geometric/UI change in viewport.
        reward = float(min(1.0, 0.2 + vp_delta * 2.0))
        note = "meaningful viewport change"
    elif vp_delta > 0.35 and not empty:
        reward = 0.4
        note = "large viewport change"
    else:
        reward = float(-0.1 + vp_delta)
        note = "weak or ambiguous change"

    return {
        "reward": float(max(-1.0, min(1.0, reward))),
        "ssim": float(ssim),
        "viewport_delta": float(vp_delta),
        "note": note,
    }
