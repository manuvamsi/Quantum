"""Camera backends: OAK-D Lite (DepthAI, on-device) or a plain webcam (local testing)."""


def get_camera(cfg: dict):
    """Pick a camera backend from config `camera.backend`: auto | oak | webcam.

    auto  -> OAK-D if depthai is importable, otherwise a plain webcam.
    """
    backend = cfg.get("camera", {}).get("backend", "auto")

    if backend == "webcam":
        from app.camera.webcam import WebcamCamera
        return WebcamCamera(cfg)
    if backend == "oak":
        from app.camera.oak import OakCamera
        return OakCamera(cfg)

    # auto
    try:
        import depthai  # noqa: F401
        from app.camera.oak import OakCamera
        return OakCamera(cfg)
    except Exception:
        from app.camera.webcam import WebcamCamera
        return WebcamCamera(cfg)
