"""
landmarks.py — face landmarks for the liveness FALLBACK (blink + head-turn) only.

Uses MediaPipe FaceMesh when available (guarded import so this file compiles without it).
The PRIMARY liveness is OAK-D depth; landmarks are only consulted when depth is weak or
unavailable, so the cost of MediaPipe is paid rarely.
"""

from typing import Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

# 6-point Eye-Aspect-Ratio indices (MediaPipe FaceMesh)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class LandmarkProvider:
    """Returns {'ear': float, 'yaw': float in ~[-1,1]} for a frame, or None."""

    def __init__(self):
        self.available = mp is not None
        self._mesh = None
        if self.available:
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.5,
            )

    def _ear(self, pts, idx):
        p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
        v = _dist(p2, p6) + _dist(p3, p5)
        h = 2 * _dist(p1, p4)
        return v / h if h else 1.0

    def analyze(self, rgb_bgr) -> Optional[dict]:
        if not self.available:
            return None
        res = self._mesh.process(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        pts = [(p.x, p.y) for p in lm]
        ear = (self._ear(pts, LEFT_EYE) + self._ear(pts, RIGHT_EYE)) / 2.0
        # yaw proxy: nose position between cheeks, centered at 0 (frontal),
        # negative when turned one way, positive the other. ~[-1, 1].
        lx, rx = pts[LEFT_CHEEK][0], pts[RIGHT_CHEEK][0]
        span = (rx - lx) or 1e-6
        yaw = ((pts[NOSE_TIP][0] - lx) / span - 0.5) * 2.0
        return {"ear": ear, "yaw": yaw}
