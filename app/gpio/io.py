"""
io.py — physical device I/O via Jetson.GPIO.

- power button (momentary, active-low): a press fires `on_power_toggle`
- status LED: on when the device is powered/ready
- optional relay: pulsed on ACCESS ALLOWED to drive a door strike/lock

Jetson.GPIO is guarded so this compiles/runs on a dev machine (mock mode: no-ops,
keyboard 'p' in the kiosk substitutes for the physical button).
"""

import threading
import time

try:
    import Jetson.GPIO as GPIO
except Exception:  # pragma: no cover - dev machine
    GPIO = None


class DeviceIO:
    def __init__(self, cfg: dict, on_power_toggle=None):
        g = cfg.get("gpio", {})
        self.on_power_toggle = on_power_toggle
        self.btn = int(g.get("power_button_pin", 0))
        self.led = int(g.get("status_led_pin", 0))
        self.relay = int(g.get("relay_pin", 0))
        self.relay_pulse = float(g.get("relay_pulse_sec", 1.5))
        self.enabled = bool(g.get("enabled", True)) and GPIO is not None and self.btn > 0

        if not self.enabled:
            self.mock = True
            return
        self.mock = False
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.btn, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        if self.led:
            GPIO.setup(self.led, GPIO.OUT, initial=GPIO.LOW)
        if self.relay:
            GPIO.setup(self.relay, GPIO.OUT, initial=GPIO.LOW)
        GPIO.add_event_detect(self.btn, GPIO.FALLING, callback=self._on_press, bouncetime=350)

    def _on_press(self, _channel):
        if self.on_power_toggle:
            self.on_power_toggle()

    def set_led(self, on: bool):
        if not self.mock and self.led:
            GPIO.output(self.led, GPIO.HIGH if on else GPIO.LOW)

    def pulse_relay(self):
        """Pulse the door relay (non-blocking)."""
        if self.mock or not self.relay:
            return

        def _pulse():
            GPIO.output(self.relay, GPIO.HIGH)
            time.sleep(self.relay_pulse)
            GPIO.output(self.relay, GPIO.LOW)

        threading.Thread(target=_pulse, daemon=True).start()

    def cleanup(self):
        if not self.mock:
            try:
                GPIO.cleanup()
            except Exception:
                pass
