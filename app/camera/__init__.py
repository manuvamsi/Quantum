"""Camera backends: OAK-D Lite (depth), Intel RealSense (RGB), or a plain webcam."""


def get_camera(cfg: dict):
    """Pick a camera backend from config `camera.backend`: auto | oak | realsense | webcam.

    auto -> OAK-D if depthai is present, else RealSense if pyrealsense2 is present,
            else a plain webcam.
    """
    backend = cfg.get("camera", {}).get("backend", "auto")

    if backend == "webcam":
        from app.camera.webcam import WebcamCamera
        return WebcamCamera(cfg)
    if backend == "oak":
        from app.camera.oak import OakCamera
        return OakCamera(cfg)
    if backend == "realsense":
        from app.camera.realsense import RealSenseCamera
        return RealSenseCamera(cfg)

    # auto: OAK-D -> RealSense -> plain webcam
    try:
        import depthai  # noqa: F401
        from app.camera.oak import OakCamera
        return OakCamera(cfg)
    except Exception:
        pass
    try:
        import pyrealsense2  # noqa: F401
        from app.camera.realsense import RealSenseCamera
        return RealSenseCamera(cfg)
    except Exception:
        from app.camera.webcam import WebcamCamera
        return WebcamCamera(cfg)
