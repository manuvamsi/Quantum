"""
checks.py - liveness 

PRIMARY: OAK-D depth 3D-region anti-spoof (+ motion). A real face has depth relief across
the face box; a photo/screen is ~flat and gets rejected. 


`process(rgb_bgr, depth_mm, bbox)` is called per frame and returns:
  {state, prompt, done, passed}
"""

import time
from typing import Optional

import cv2
import numpy as np


class LivenessChecker:
    def __init__(self, cfg: dict, landmark_provider=None):
        self.L = cfg["liveness"]
        self.mode = self.L.get("mode", "fast")
        self.landmarks = landmark_provider
        self.reset()

    def reset(self):
        self.state = "WAIT"
        self.t_start: Optional[float] = None
        self.substep_t: Optional[float] = None
        self.frames = 0
        self.depth_vars: list[float] = []
        self.depth_ratios: list[float] = []
        self.motion_peak = 0.0
        self.prev_gray = None
        self.eyes_were_open = True

    # ── geometry / signals ────────────────────────────────────────────────────
    def _face_ok(self, bbox, shape) -> bool:
        h, w = shape[:2]
        x, y, bw, bh = bbox
        ratio = bw / w
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        tol = self.L["center_tolerance"]
        return ratio >= self.L["min_face_ratio"] and abs(cx - 0.5) < tol and abs(cy - 0.5) < tol + 0.1

    def _depth_stats(self, depth, bbox):
        """(depth spread in mm, valid-pixel ratio) inside the face box, or None."""
        if depth is None:
            return None
        x, y, bw, bh = bbox
        roi = depth[max(0, y):y + bh, max(0, x):x + bw].astype(np.float32)
        if roi.size == 0:
            return (0.0, 0.0)
        valid = roi[roi > 0]
        if valid.size < 50:
            return (0.0, 0.0)
        # robust spread: middle 90% to ignore edge outliers
        lo, hi = np.percentile(valid, [5, 95])
        return (float(hi - lo) / 2.0, float(valid.size) / float(roi.size))

    def _motion(self, rgb, bbox) -> float:
        x, y, bw, bh = bbox
        g = cv2.cvtColor(rgb[max(0, y):y + bh, max(0, x):x + bw], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (64, 64))
        if self.prev_gray is None:
            self.prev_gray = g
            return 0.0
        d = float(np.mean(cv2.absdiff(g, self.prev_gray)))
        self.prev_gray = g
        return d

    def _res(self, prompt, done=False, passed=False):
        return {"state": self.state, "prompt": prompt, "done": done, "passed": passed}

    # ── main step ─────────────────────────────────────────────────────────────
    def process(self, rgb, depth, bbox, manual_trigger: bool = False):
        now = time.time()

        if bbox is None or not self._face_ok(bbox, rgb.shape):
            if self.state not in ("PASSED", "FAILED"):
                self.reset()
            return self._res("Come closer and center your face")

        if self.t_start is None:
            self.t_start = now
        if now - self.t_start > self.L["timeout_sec"] and self.state not in ("PASSED", "FAILED"):
            self.state = "FAILED"
            return self._res("Liveness timed out - try again", done=True)

        motion = self._motion(rgb, bbox)

        # ---- depth assessment phase ----
        if self.state in ("WAIT", "ASSESS"):
            self.state = "ASSESS"
            self.frames += 1
            self.motion_peak = max(self.motion_peak, motion)
            ds = self._depth_stats(depth, bbox)
            if ds is not None:
                self.depth_vars.append(ds[0])
                self.depth_ratios.append(ds[1])
            if self.frames < self.L["assess_frames"]:
                return self._res("Hold still...")

            # Laptop testing (no depth camera): accept a present, moving face.
            if self.L.get("local_test_mode"):
                self.state = "PASSED"
                return self._res("Live (local test)", done=True, passed=True)

            var = float(np.median(self.depth_vars)) if self.depth_vars else 0.0
            ratio = float(np.median(self.depth_ratios)) if self.depth_ratios else 0.0
            motion_ok = self.motion_peak >= self.L["motion_threshold"]
            has_depth = depth is not None and ratio >= self.L["depth_region_min_ratio"]
            flat = has_depth and var < self.L["depth_min_variance_mm"]
            strong = has_depth and var >= self.L["depth_strong_variance_mm"]

            if flat:
                self.state = "FAILED"
                return self._res("Spoof detected (flat surface) - access denied", done=True)

            # fast mode: strong depth + motion -> instant pass
            if self.mode == "fast" and strong and motion_ok:
                self.state = "PASSED"
                return self._res("Live face confirmed", done=True, passed=True)

            # otherwise fall back to gestures (or, in fast mode with usable depth but no
            # landmarks, accept on 'real & not flat + motion')
            if self.landmarks is None or not self.landmarks.available:
                if self.mode == "fast" and has_depth and not flat and motion_ok:
                    self.state = "PASSED"
                    return self._res("Live face confirmed (depth)", done=True, passed=True)
                self.state = "FAILED"
                return self._res("Cannot verify liveness (no depth/landmarks)", done=True)

            self.state = "BLINK"
            self.substep_t = now
            return self._res("Please blink")

        # ---- gesture fallback phase ----
        return self._gestures(rgb, now)

    def _gestures(self, rgb, now):
        fb = self.L["fallback"]
        if self.substep_t and now - self.substep_t > fb["step_timeout_sec"]:
            self.state = "FAILED"
            return self._res("Gesture timed out - try again", done=True)

        data = self.landmarks.analyze(rgb) if self.landmarks else None
        if data is None:
            return self._res("Keep your face in view")
        ear, yaw = data["ear"], data["yaw"]
        yaw_thr = fb["yaw_threshold_deg"] / 90.0  # yaw proxy is ~[-1,1]

        if self.state == "BLINK":
            if not fb.get("require_blink", True):
                self.state = "TURN_LEFT"; self.substep_t = now
                return self._res("Turn your head LEFT")
            if ear < fb["blink_ear_threshold"]:
                self.eyes_were_open = False
            elif not self.eyes_were_open and ear > fb["blink_ear_threshold"] + 0.05:
                # blink completed (closed then open)
                if not fb.get("require_head_turn", True) and self.mode != "strict":
                    self.state = "PASSED"
                    return self._res("Live face confirmed", done=True, passed=True)
                self.state = "TURN_LEFT"; self.substep_t = now
                return self._res("Turn your head LEFT")
            return self._res("Please blink")

        if self.state == "TURN_LEFT":
            if yaw < -yaw_thr:
                self.state = "TURN_RIGHT"; self.substep_t = now
                return self._res("Now turn your head RIGHT")
            return self._res("Turn your head LEFT")

        if self.state == "TURN_RIGHT":
            if yaw > yaw_thr:
                self.state = "PASSED"
                return self._res("Live face confirmed", done=True, passed=True)
            return self._res("Now turn your head RIGHT")

        return self._res("...")
