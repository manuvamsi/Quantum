"""
exporter.py — application Prometheus metrics.

Exposes /metrics on `metrics.port` (default 9108) so Prometheus can scrape access events,
recognition latency, liveness failures, model status, and DB size. prometheus_client is
guarded so the app still runs if it isn't installed (metrics simply become no-ops).
"""

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


class Metrics:
    def __init__(self, cfg: dict):
        m = cfg.get("metrics", {})
        self.enabled = bool(m.get("enabled", True)) and _HAVE
        self.port = int(m.get("port", 9108))
        self.device = cfg.get("device", {}).get("id", "reqagnize-edge")
        self._started = False
        if not self.enabled:
            return
        L = ["device"]
        self.c_allow = Counter("reqagnize_access_allowed_total", "Access allowed", L)
        self.c_deny = Counter("reqagnize_access_denied_total", "Access denied (not recognized)", L)
        self.c_livefail = Counter("reqagnize_liveness_fail_total", "Liveness checks failed", L)
        self.h_lat = Histogram("reqagnize_recognition_latency_seconds", "Recognition latency", L,
                               buckets=(0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0))
        self.g_users = Gauge("reqagnize_db_users", "Enrolled users in the vector DB", L)
        self.g_model = Gauge("reqagnize_model_loaded", "1 if the model is loaded", L)

    def start(self):
        if self.enabled and not self._started:
            start_http_server(self.port)
            self._started = True

    # ── recorders (all no-op when disabled) ───────────────────────────────────
    def record_result(self, allowed: bool, latency: float | None):
        if not self.enabled:
            return
        (self.c_allow if allowed else self.c_deny).labels(self.device).inc()
        if latency is not None:
            self.h_lat.labels(self.device).observe(latency)

    def record_liveness_fail(self):
        if self.enabled:
            self.c_livefail.labels(self.device).inc()

    def set_users(self, n: int):
        if self.enabled:
            self.g_users.labels(self.device).set(n)

    def set_model_loaded(self, on: bool):
        if self.enabled:
            self.g_model.labels(self.device).set(1 if on else 0)
