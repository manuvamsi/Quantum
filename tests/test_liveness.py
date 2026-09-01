"""CI test: liveness FSM rejects flat (photo) depth and passes real 3D depth. No torch/depthai."""

import numpy as np

from app.config import load_config
from app.liveness.checks import LivenessChecker

BBOX = (220, 100, 200, 200)  # centered, large enough


def _run(depth_fn):
    cfg = load_config()
    cfg["liveness"]["local_test_mode"] = False  # test the real depth logic, not the laptop bypass
    chk = LivenessChecker(cfg, landmark_provider=None)
    final = None
    for _ in range(12):
        rgb = np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)  # motion
        final = chk.process(rgb, depth_fn(), BBOX)
        if final["done"]:
            break
    return final


def test_flat_photo_rejected():
    r = _run(lambda: np.full((400, 640), 500, np.uint16))
    assert r["state"] == "FAILED" and not r["passed"]


def test_real_3d_passes():
    def d3():
        d = np.zeros((400, 640), np.uint16)
        grad = np.linspace(460, 520, 200).astype(np.uint16)[None, :].repeat(200, 0)
        d[100:300, 220:420] = grad
        return d

    assert _run(d3)["passed"]
