"""
realsense.py — Intel RealSense backend, COLOR (RGB) only, no depth.

Grabs the RealSense color stream via pyrealsense2 and returns (rgb_bgr, None) so it
drops into the same interface as WebcamCamera. Depth is intentionally not enabled —
pair with `liveness.local_test_mode: true` on a laptop (no depth-based anti-spoof).
"""

import numpy as np

try:
    import pyrealsense2 as rs
except Exception:  # pyrealsense2 not installed
    rs = None


class RealSenseCamera:
    def __init__(self, cfg: dict):
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is not installed — run: pip install pyrealsense2"
            )
        cam = cfg.get("camera", {})
        # 640x480 @ 30 is a universally supported RealSense color profile. Note: RealSense
        # color only allows 6/15/30/60 fps — do NOT reuse the OAK `rgb_fps` (20) here.
        self.width = int(cam.get("realsense_width", 640))
        self.height = int(cam.get("realsense_height", 480))
        self.fps = int(cam.get("realsense_fps", 30))
        self.pipeline = None

    def __enter__(self) -> "RealSenseCamera":
        self.pipeline = rs.pipeline()
        config = rs.config()
        # COLOR only — no depth stream is enabled.
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        try:
            self.pipeline.start(config)
        except Exception as e:
            self.pipeline = None
            raise RuntimeError(f"Cannot start RealSense color stream: {e}")
        return self

    def read(self):
        """Return (rgb_bgr, None). Waits briefly for the next color frame."""
        try:
            frames = self.pipeline.wait_for_frames(2000)  # ms
        except Exception:
            return None, None
        color = frames.get_color_frame()
        if not color:
            return None, None
        return np.asanyarray(color.get_data()), None  # (rgb_bgr, depth=None)

    def __exit__(self, *exc):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
