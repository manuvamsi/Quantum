"""
webcam.py — a plain USB/laptop webcam backend (cv2.VideoCapture) with the same interface
as OakCamera, for running the app LOCALLY without an OAK-D. `read()` returns (rgb_bgr, None)
— there is no depth, so pair this with `liveness.local_test_mode: true` on a dev machine.
"""

import cv2


class WebcamCamera:
    def __init__(self, cfg: dict):
        self.index = int(cfg.get("camera", {}).get("webcam_index", 0))
        self.cap = None

    def __enter__(self) -> "WebcamCamera":
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.index}")
        return self

    def read(self):
        ok, frame = self.cap.read()
        return (frame if ok else None), None  # (rgb_bgr, depth=None)

    def __exit__(self, *exc):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
