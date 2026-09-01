"""
Stage-2 device test (run ON THE JETSON with the OAK-D Lite plugged in):

    python scripts/test_camera.py

Shows the RGB preview + depth colormap, detects the face (Haar), and runs the liveness
state machine (depth-primary; blink/turn fallback). Prints LIVE / FAIL per attempt.
Press 'q' to quit. Needs a display (the small screen or an X session).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NVIDIA_App/

import cv2  # noqa: E402
from app.config import load_config  # noqa: E402
from app.camera import get_camera  # noqa: E402
from app.liveness.checks import LivenessChecker  # noqa: E402
from app.liveness.landmarks import LandmarkProvider  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> int:
    cfg = load_config()
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    landmarks = LandmarkProvider()
    print(f"MediaPipe landmarks available: {landmarks.available}")
    checker = LivenessChecker(cfg, landmarks)

    with get_camera(cfg) as cam:
        print("Camera open. Look at the camera. Press 'q' to quit.")
        while True:
            rgb, depth = cam.read()
            if rgb is None:
                continue

            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            bbox = tuple(max(faces, key=lambda f: f[2] * f[3])) if len(faces) else None

            status = checker.process(rgb, depth, bbox)

            disp = rgb.copy()
            if bbox is not None:
                x, y, w, h = bbox
                color = (0, 255, 0) if status["state"] != "FAILED" else (0, 0, 255)
                cv2.rectangle(disp, (x, y), (x + w, y + h), color, 2)
            cv2.putText(disp, status["prompt"], (10, 28), FONT, 0.7, (0, 255, 255), 2)
            cv2.putText(disp, f"state={status['state']}", (10, 56), FONT, 0.55, (200, 200, 200), 1)

            if depth is not None:
                dvis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET)
                cv2.imshow("depth (aligned)", dvis)
            cv2.imshow("ReQAgnIze — liveness test", disp)

            if status["done"]:
                print("RESULT:", "LIVE ✔" if status["passed"] else "FAIL ✖", "—", status["prompt"])
                cv2.waitKey(1200)
                checker.reset()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
