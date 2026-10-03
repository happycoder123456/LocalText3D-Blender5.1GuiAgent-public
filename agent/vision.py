"""OpenCV template-matching vision for Blender UI anchors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from agent.paths import default_templates_dir


class BlenderVision:
    """Detect UI anchors by comparing screenshots to small template PNGs."""

    def __init__(
        self,
        templates_dir: Path | None = None,
        threshold: float = 0.8,
        scales: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2),
    ):
        self.templates_dir = Path(templates_dir) if templates_dir else default_templates_dir()
        self.threshold = float(threshold)
        self.scales = scales
        self._templates: dict[str, np.ndarray] = {}
        self.reload()

    def reload(self) -> None:
        self._templates.clear()
        if not self.templates_dir.is_dir():
            return
        for path in sorted(self.templates_dir.glob("*.png"))[:24]:
            try:
                if path.stat().st_size > 2_000_000:
                    continue
            except OSError:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            self._templates[path.stem] = img

    @property
    def class_names(self) -> list[str]:
        return sorted(self._templates.keys())

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return [{class, bbox: [x, y, w, h], score}] for matches above threshold."""
        if image is None or image.size == 0:
            return []
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image
        h, w = bgr.shape[:2]
        hits: list[dict[str, Any]] = []
        for name, tmpl in self._templates.items():
            best: dict[str, Any] | None = None
            th, tw = tmpl.shape[:2]
            tmpl_std = float(np.std(tmpl.astype(np.float32)))
            use_sqdiff = tmpl_std < 5.0
            for scale in self.scales:
                nw = max(1, int(tw * scale))
                nh = max(1, int(th * scale))
                if nw >= w or nh >= h:
                    continue
                scaled = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_AREA)
                if use_sqdiff:
                    result = cv2.matchTemplate(bgr, scaled, cv2.TM_SQDIFF_NORMED)
                    min_v, _max_v, min_loc, _max_loc = cv2.minMaxLoc(result)
                    score = float(1.0 - min_v)
                    loc = min_loc
                else:
                    result = cv2.matchTemplate(bgr, scaled, cv2.TM_CCOEFF_NORMED)
                    _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(result)
                    score = float(max_v)
                    loc = max_loc
                if score < self.threshold:
                    continue
                x, y = int(loc[0]), int(loc[1])
                cand = {
                    "class": name,
                    "bbox": [x, y, nw, nh],
                    "score": score,
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand
            if best is not None:
                hits.append(best)
        hits.sort(key=lambda d: d["score"], reverse=True)
        return hits

    def center_of(self, detection: dict[str, Any]) -> tuple[int, int]:
        x, y, bw, bh = detection["bbox"]
        return int(x + bw / 2), int(y + bh / 2)
