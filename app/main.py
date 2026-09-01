"""
main.py — the ReQAgnIze edge appliance state machine.

    OFF ──(button)──▶ LOADING ──▶ READY/ACTIVE ──(face+liveness)──▶ RECOGNIZE ──▶ RESULT ──▶ ACTIVE
     ▲                                                                                  │
     └───────────────────────────────(button)──────────────────────────────────────────┘

Power is toggled by the physical GPIO button OR the on-screen button (OR 'p' on a keyboard
in mock mode). On power-on the quantum model + camera load once and stay warm; recognition
runs only after liveness passes. Run on the Jetson:  python -m app.main   (or scripts/run.sh)
"""

import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NVIDIA_App/

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import load_config  # noqa: E402
from app.recognition.engine import RecognitionEngine, EngineError  # noqa: E402
from app.liveness.checks import LivenessChecker  # noqa: E402
from app.liveness.landmarks import LandmarkProvider  # noqa: E402
from app.camera import get_camera  # noqa: E402
from app.gpio.io import DeviceIO  # noqa: E402
from app.ui.kiosk import Kiosk  # noqa: E402
from app.metrics.exporter import Metrics  # noqa: E402
from app.events.store import EventStore  # noqa: E402


class AppController:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.engine = RecognitionEngine(cfg)
        self.checker = LivenessChecker(cfg, LandmarkProvider())
        self.io = DeviceIO(cfg, on_power_toggle=self.request_toggle)
        self.kiosk = Kiosk(cfg)
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.metrics = Metrics(cfg)
        self.metrics.start()
        self.events = EventStore(cfg)

        self.power = False
        self._toggle_req = False
        self.phase = "idle"          # idle | loading | active | result
        self.cam = None
        self._ready = False
        self._load_error = None
        self.result = None
        self.result_until = 0.0
        self.last_frame = None
        self.enroll = None            # in-app registration state
        self._enroll_frame = None
        self._enroll_bbox = None

    # ── power ─────────────────────────────────────────────────────────────────
    def request_toggle(self):
        self._toggle_req = True       # applied on the main thread (GPIO cb is off-thread)

    def _power_on(self):
        self.power = True
        self.phase = "loading"
        self._ready = False
        self._load_error = None
        self.io.set_led(True)
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self.engine.load()
            self.cam = get_camera(self.cfg)
            self.cam.__enter__()
            self._ready = True
            self.metrics.set_model_loaded(True)
            self.metrics.set_users(self.engine.count())
        except Exception as e:  # surfaced on the loading screen
            self._load_error = str(e)

    def _power_off(self):
        self.power = False
        self.phase = "idle"
        self._ready = False
        self.io.set_led(False)
        self.metrics.set_model_loaded(False)
        if self.cam is not None:
            try:
                self.cam.__exit__(None, None, None)
            except Exception:
                pass
            self.cam = None
        self.checker.reset()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _detect(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return tuple(max(faces, key=lambda f: f[2] * f[3])) if len(faces) else None

    def _recognize(self, rgb):
        try:
            res = self.engine.recognize(rgb)
        except EngineError as e:
            res = {"allowed": False, "message": str(e)}
        except Exception as e:
            res = {"allowed": False, "message": f"error: {e}"}

        self.metrics.record_result(bool(res.get("allowed")), res.get("latency_sec"))
        self.events.record(
            user=res.get("user"), allowed=bool(res.get("allowed")),
            score=res.get("score"), votes=res.get("votes"),
            liveness=self.checker.mode, latency=res.get("latency_sec"),
        )
        if res.get("allowed"):
            name = (res.get("metadata") or {}).get("name") or res.get("user") or "User"
            self.result = {"allowed": True, "name": name}
            self.io.pulse_relay()
            # Stage 4b hook: record + sync the access event here.
        else:
            self.result = {"allowed": False, "name": "",
                           "sub": res.get("message", "Not recognized")}
        self.phase = "result"
        self.result_until = time.time() + 4.0

    # ── in-app registration (enrollment) ──────────────────────────────────────
    def _start_enroll(self):
        self.enroll = {"step": "name", "name": "", "phone": "", "age": "", "shots": []}
        self.phase = "enroll"

    def _enroll_step(self):
        e = self.enroll
        step = e["step"]
        if step in ("name", "phone", "age"):
            hints = {
                "name": "Type name  -  ENTER = next, Esc = cancel",
                "phone": "Phone (optional)  -  ENTER = next, Esc = cancel",
                "age": "Age (optional)  -  ENTER = next, Esc = cancel",
            }
            self.kiosk.show_enroll_field(step, e[step], hints[step])
        elif step == "capture":
            rgb, _ = self.cam.read()
            if rgb is not None:
                self._enroll_frame = rgb
                self._enroll_bbox = self._detect(rgb)
                self.kiosk.show_enroll_capture(rgb, len(e["shots"]), 3, self._enroll_bbox)
        elif step == "saving":
            self.kiosk.show_message("Enrolling...", "please wait")
            self.kiosk.key()
            self._do_enroll()
        elif step == "done":
            r = e.get("result", {})
            self.kiosk.show_message(r.get("title", ""), r.get("sub", ""),
                                    (94, 197, 34) if r.get("ok") else (60, 60, 235))
            if time.time() > e.get("until", 0):
                self.enroll = None
                self.checker.reset()
                self.phase = "active"

    def _do_enroll(self):
        e = self.enroll
        try:
            res = self.engine.enroll(
                e["name"].strip(), e["shots"],
                phone=e["phone"].strip() or None,
                age=int(e["age"]) if e["age"].isdigit() else None,
            )
            self.metrics.set_users(self.engine.count())
            e["result"] = {"ok": True, "title": f"Registered {res['name']}",
                           "sub": f"{res['embeddings']} samples stored"}
        except EngineError as ex:
            e["result"] = {"ok": False, "title": "Registration failed", "sub": str(ex)}
        except Exception as ex:
            e["result"] = {"ok": False, "title": "Registration error", "sub": str(ex)}
        e["step"] = "done"
        e["until"] = time.time() + 3.0

    def _handle_enroll_key(self, k):
        if k == 255 or self.enroll is None:
            return
        e = self.enroll
        step = e["step"]
        ENTER, BKSP, ESC = 13, 8, 27
        if k == ESC:
            self.enroll = None
            self.checker.reset()
            self.phase = "active"
            return
        if step in ("name", "phone", "age"):
            if k in (ENTER, 10):
                if step == "name" and len(e["name"].strip()) >= 2:
                    e["step"] = "phone"
                elif step == "phone":
                    e["step"] = "age"
                elif step == "age":
                    e["step"] = "capture"
                return
            if k in (BKSP, 127):
                e[step] = e[step][:-1]
                return
            if 32 <= k <= 126:
                ch = chr(k)
                if step == "age" and not ch.isdigit():
                    return
                if len(e[step]) < 40:
                    e[step] += ch
            return
        if step == "capture":
            if k == ord(" "):
                if (self._enroll_frame is not None and self._enroll_bbox is not None
                        and len(e["shots"]) < 3):
                    e["shots"].append(self._enroll_frame.copy())
            elif k in (ENTER, 10) and len(e["shots"]) >= 1:
                e["step"] = "saving"

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        try:
            while True:
                if self._toggle_req:
                    self._toggle_req = False
                    (self._power_off if self.power else self._power_on)()

                if not self.power:
                    self.kiosk.show_idle()

                elif self.phase == "loading":
                    if self._load_error:
                        self.kiosk.show_loading(f"Load failed: {self._load_error[:36]}")
                    elif self._ready:
                        self.checker.reset()
                        self.phase = "active"
                    else:
                        self.kiosk.show_loading()

                elif self.phase == "active":
                    rgb, depth = self.cam.read()
                    if rgb is not None:
                        self.last_frame = rgb
                        bbox = self._detect(rgb)
                        status = self.checker.process(rgb, depth, bbox)
                        if status["done"]:
                            if status["passed"]:
                                self.kiosk.show_active(rgb, "Recognizing...", "RECOGNIZE", bbox)
                                self.kiosk.key()
                                self._recognize(rgb)
                            else:
                                self.metrics.record_liveness_fail()
                                self.events.record(user=None, allowed=False, liveness="liveness_fail")
                                self.result = {"allowed": False, "name": "",
                                               "sub": status["prompt"]}
                                self.phase = "result"
                                self.result_until = time.time() + 2.5
                        else:
                            self.kiosk.show_active(rgb, status["prompt"], status["state"], bbox)

                elif self.phase == "result":
                    frame = self.last_frame if self.last_frame is not None \
                        else np.zeros((self.kiosk.H, self.kiosk.W, 3), np.uint8)
                    r = self.result or {"allowed": False}
                    self.kiosk.show_result(frame, r["allowed"], r.get("name", ""), r.get("sub", ""))
                    if time.time() > self.result_until:
                        self.checker.reset()
                        self.phase = "active"

                elif self.phase == "enroll":
                    self._enroll_step()

                k = self.kiosk.key()
                if self.phase == "enroll":
                    # keys are text/capture input here (Esc cancels); don't treat 'q' as quit
                    self._handle_enroll_key(k)
                else:
                    if k == ord("q"):
                        break
                    clicked = self.kiosk.poll_click()
                    if k == ord("p") or clicked == "power":
                        self.request_toggle()
                    elif (k == ord("r") or clicked == "register") and self.power and self.phase == "active":
                        self._start_enroll()
        finally:
            self._power_off()
            self.io.cleanup()
            self.events.close()
            self.kiosk.close()


def main():
    cfg = load_config()
    AppController(cfg).run()


if __name__ == "__main__":
    main()
