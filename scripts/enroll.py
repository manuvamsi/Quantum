"""
enroll.py — register a person on the device using the OAK-D camera.

    python scripts/enroll.py --name "Jane Doe" --phone 9876543210 --age 28 --shots 3

Press SPACE to capture each shot (needs a detected face), 'q' to cancel. Embeddings go to
the device ChromaDB; name/phone/age are encrypted with PQC-NTRU.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NVIDIA_App/

import cv2  # noqa: E402
from app.config import load_config  # noqa: E402
from app.recognition.engine import RecognitionEngine, EngineError  # noqa: E402
from app.camera import get_camera  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--phone", default=None)
    ap.add_argument("--age", type=int, default=None)
    ap.add_argument("--shots", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config()
    eng = RecognitionEngine(cfg)
    print("Loading model…")
    eng.load()
    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    shots = []
    with get_camera(cfg) as cam:
        print(f"Capturing {args.shots} shots — SPACE=capture, q=cancel")
        while len(shots) < args.shots:
            rgb, _ = cam.read()
            if rgb is None:
                continue
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            disp = rgb.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(disp, f"{len(shots)}/{args.shots}  SPACE=capture  q=cancel",
                        (10, 30), FONT, 0.6, (0, 255, 255), 2)
            cv2.imshow("Enroll", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" ") and len(faces):
                shots.append(rgb.copy())
                print(f"  captured {len(shots)}/{args.shots}")
            elif k == ord("q"):
                print("cancelled")
                cv2.destroyAllWindows()
                return 1
    cv2.destroyAllWindows()

    try:
        res = eng.enroll(args.name, shots, phone=args.phone, age=args.age)
        print("ENROLLED:", res)
        return 0
    except EngineError as e:
        print("FAILED:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
