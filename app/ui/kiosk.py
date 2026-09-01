"""
kiosk.py — the small-display UI, rendered with OpenCV HighGUI (fullscreen).

Chosen over a heavy GUI toolkit for lowest latency and minimal deps on the Jetson.
Renders the camera feed + prompts + big result, and draws an on-screen POWER button
(click toggles power, alongside the physical GPIO button). All colours are BGR.
"""

import cv2
import numpy as np

WIN = "ReQAgnIze"

# Theme (BGR)
BG = (14, 10, 12)
CRIMSON = (71, 43, 222)     # #de2b47
CYAN = (255, 224, 40)       # #28e0ff
GREEN = (94, 197, 34)
RED = (60, 60, 235)
WHITE = (245, 245, 245)
GREY = (150, 150, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Kiosk:
    def __init__(self, cfg: dict):
        d = cfg.get("device", {}).get("display", {})
        self.W = int(d.get("width", 800))
        self.H = int(d.get("height", 480))
        self.fullscreen = bool(d.get("fullscreen", True))
        self._click = None            # (x, y) of last unhandled click
        self._power_rect = None       # (x1, y1, x2, y2)
        self._register_rect = None

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, self.W, self.H)
        if self.fullscreen:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(WIN, self._on_mouse)

    # ── input ─────────────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click = (x, y)

    def poll_click(self):
        """Return 'power' | 'register' | None for the button clicked since last poll."""
        if not self._click:
            return None
        x, y = self._click
        self._click = None
        for name, rect in (("power", self._power_rect), ("register", self._register_rect)):
            if rect:
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return name
        return None

    def key(self) -> int:
        return cv2.waitKey(1) & 0xFF

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _canvas(self):
        c = np.empty((self.H, self.W, 3), np.uint8)
        c[:] = BG
        return c

    def _center_text(self, img, text, y, scale, color, thick=2):
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
        cv2.putText(img, text, ((self.W - tw) // 2, y), FONT, scale, color, thick, cv2.LINE_AA)

    def _fit_frame(self, frame):
        """Scale a camera frame to fill the canvas (keep aspect, center-crop)."""
        fh, fw = frame.shape[:2]
        scale = max(self.W / fw, self.H / fh)
        r = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
        y0 = (r.shape[0] - self.H) // 2
        x0 = (r.shape[1] - self.W) // 2
        return r[y0:y0 + self.H, x0:x0 + self.W]

    def _power_button(self, img, on: bool):
        w, h = 120, 44
        x1, y1 = self.W - w - 16, 16
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), CRIMSON if on else GREY, -1)
        label = "POWER" if on else "START"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.putText(img, label, (x1 + (w - tw) // 2, y1 + (h + th) // 2),
                    FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        self._power_rect = (x1, y1, x2, y2)

    def _register_button(self, img):
        w, h = 130, 44
        x1, y1 = self.W - 120 - 16 - w - 10, 16   # sits left of POWER
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), (110, 150, 40), -1)
        (tw, th), _ = cv2.getTextSize("REGISTER", FONT, 0.55, 2)
        cv2.putText(img, "REGISTER", (x1 + (w - tw) // 2, y1 + (h + th) // 2),
                    FONT, 0.55, WHITE, 2, cv2.LINE_AA)
        self._register_rect = (x1, y1, x2, y2)

    def _show(self, img):
        cv2.imshow(WIN, img)

    # ── screens ───────────────────────────────────────────────────────────────
    def show_idle(self):
        c = self._canvas()
        self._register_rect = None
        self._center_text(c, "ReQAgnIze", self.H // 2 - 40, 1.4, CRIMSON, 3)
        self._center_text(c, "Quantum Face Access", self.H // 2, 0.7, GREY, 1)
        self._center_text(c, "Press START to power on", self.H // 2 + 50, 0.7, WHITE, 2)
        self._power_button(c, on=False)
        self._show(c)

    def show_loading(self, msg="Loading quantum model..."):
        c = self._canvas()
        self._register_rect = None
        self._center_text(c, msg, self.H // 2, 0.8, WHITE, 2)
        self._center_text(c, "please wait", self.H // 2 + 40, 0.6, GREY, 1)
        self._power_button(c, on=True)
        self._show(c)

    def show_active(self, frame, prompt: str, state: str, bbox=None):
        c = self._fit_frame(frame).copy()
        # dim strip for text legibility
        cv2.rectangle(c, (0, 0), (self.W, 64), (0, 0, 0), -1)
        cv2.addWeighted(c, 1.0, c, 0, 0, c)
        if bbox is not None:
            # bbox is in the camera-frame coords; approximate onto the fitted canvas
            fh, fw = frame.shape[:2]
            sx, sy = self.W / fw, self.H / fh
            x, y, w, h = bbox
            cv2.rectangle(c, (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)),
                          CYAN if state != "FAILED" else RED, 2)
        self._center_text(c, prompt, 42, 0.85, CYAN, 2)
        self._power_button(c, on=True)
        self._register_button(c)
        self._show(c)

    def show_result(self, frame, allowed: bool, name: str = "", sub: str = ""):
        c = self._fit_frame(frame).copy()
        self._register_rect = None
        overlay = c.copy()
        cv2.rectangle(overlay, (0, 0), (self.W, self.H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, c, 0.45, 0, c)
        if allowed:
            self._center_text(c, "ACCESS ALLOWED", self.H // 2 - 20, 1.3, GREEN, 3)
            self._center_text(c, name, self.H // 2 + 40, 1.0, WHITE, 2)
        else:
            self._center_text(c, "ACCESS DENIED", self.H // 2 - 10, 1.3, RED, 3)
            if sub:
                self._center_text(c, sub, self.H // 2 + 45, 0.7, GREY, 1)
        self._power_button(c, on=True)
        self._show(c)

    def show_enroll_field(self, label: str, typed: str, hint: str):
        c = self._canvas()
        self._register_rect = None
        self._center_text(c, "REGISTER NEW USER", 74, 1.0, CRIMSON, 2)
        self._center_text(c, label.upper(), self.H // 2 - 58, 0.7, GREY, 1)
        bw, bh = int(self.W * 0.7), 56
        x1, y1 = (self.W - bw) // 2, self.H // 2 - 28
        cv2.rectangle(c, (x1, y1), (x1 + bw, y1 + bh), (40, 40, 52), -1)
        cv2.rectangle(c, (x1, y1), (x1 + bw, y1 + bh), CRIMSON, 2)
        cv2.putText(c, typed + "_", (x1 + 16, y1 + 38), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
        self._center_text(c, hint, self.H // 2 + 74, 0.55, GREY, 1)
        self._power_button(c, on=True)
        self._show(c)

    def show_enroll_capture(self, frame, count: int, needed: int, bbox=None):
        c = self._fit_frame(frame).copy()
        self._register_rect = None
        cv2.rectangle(c, (0, 0), (self.W, 64), (0, 0, 0), -1)
        if bbox is not None:
            fh, fw = frame.shape[:2]
            sx, sy = self.W / fw, self.H / fh
            x, y, w, h = bbox
            cv2.rectangle(c, (int(x * sx), int(y * sy)),
                          (int((x + w) * sx), int((y + h) * sy)), (110, 150, 40), 2)
        self._center_text(c, f"REGISTER  {count}/{needed}  -  SPACE = capture", 42, 0.7, CYAN, 2)
        for i in range(needed):
            cx = self.W // 2 - (needed * 18) // 2 + i * 18 + 9
            cv2.circle(c, (cx, self.H - 22), 6, (110, 150, 40) if i < count else (90, 90, 100), -1)
        self._center_text(c, "ENTER = save   Esc = cancel", self.H - 46, 0.5, GREY, 1)
        self._power_button(c, on=True)
        self._show(c)

    def show_message(self, title: str, sub: str = "", color=WHITE):
        c = self._canvas()
        self._register_rect = None
        self._center_text(c, title, self.H // 2 - 8, 1.0, color, 2)
        if sub:
            self._center_text(c, sub, self.H // 2 + 40, 0.6, GREY, 1)
        self._power_button(c, on=True)
        self._show(c)

    def close(self):
        cv2.destroyAllWindows()
