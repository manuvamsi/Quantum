'''
Webcam camera backend is not mandatory as of now,you can choose "auto" in the "NVIDIA_App/Testing_App_deployment/app/camera/__init__.py"
'''

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
