
'''
OAK-D camera backend is not mandatory as of now,you can choose "auto" in the "C:/Users\rosha/Documents/M.Tech/MU/Major_Project/New_Model_QCNN/NVIDIA_App/Testing_App_deployment\app\camera\__init__.py"
'''
from typing import Optional, Tuple

import numpy as np

try:
    import depthai as dai
except Exception:  #  without depthai will raise ImportError
    dai = None

PREVIEW_W, PREVIEW_H = 640, 400  #the depth output size,this is the size of the depth frame that will be sent to the host. The depth frame is aligned to the RGB frame,so it should be the same size as the RGB preview frame.


class OakCamera:
    def __init__(self, cfg: dict):
        if dai is None:
            raise RuntimeError("depthai is not installed — run scripts/setup_jetson.sh on the Jetson")
        cam = cfg.get("camera", {})
        self.fps = int(cam.get("rgb_fps", 20))
        self.depth_enabled = bool(cam.get("depth_enabled", True))
        self._device = None
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self):
        p = dai.Pipeline()

        cam_rgb = p.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewSize(PREVIEW_W, PREVIEW_H)
        cam_rgb.setFps(self.fps)
        xout_rgb = p.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        if self.depth_enabled:
            left = p.create(dai.node.MonoCamera)
            right = p.create(dai.node.MonoCamera)
            left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
            right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
            for m in (left, right):
                m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
                m.setFps(self.fps)

            stereo = p.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to the RGB frame
            stereo.setOutputSize(PREVIEW_W, PREVIEW_H)
            left.out.link(stereo.left)
            right.out.link(stereo.right)

            xout_d = p.create(dai.node.XLinkOut)
            xout_d.setStreamName("depth")
            stereo.depth.link(xout_d.input)

        return p

    # context manager so the device is always closed
    def __enter__(self) -> "OakCamera":
        self._device = dai.Device(self._pipeline)
        self.q_rgb = self._device.getOutputQueue("rgb", maxSize=2, blocking=False)
        self.q_depth = (
            self._device.getOutputQueue("depth", maxSize=2, blocking=False)
            if self.depth_enabled else None
        )
        return self

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (rgb_bgr, depth_mm). rgb blocks briefly; depth is best-effort."""
        rgb_msg = self.q_rgb.get()
        rgb = rgb_msg.getCvFrame() if rgb_msg is not None else None
        depth = None
        if self.q_depth is not None:
            d = self.q_depth.tryGet()
            if d is not None:
                depth = d.getFrame()  # usame as  aligned to rgb
        return rgb, depth

    def __exit__(self, *exc):
        if self._device is not None:
            self._device.close()
            self._device = None
