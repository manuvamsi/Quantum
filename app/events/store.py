"""
This is not used to run the app,later for cloud sync it is declared.So, it is not mandatory to check this file for running the app.
"""

import os
import sqlite3
import threading
import time

try:
    import requests
except Exception:  
    requests = None

from app.config import abspath

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL,
    device   TEXT,
    user     TEXT,
    allowed  INTEGER,
    score    REAL,
    votes    INTEGER,
    liveness TEXT,
    latency  REAL,
    synced   INTEGER DEFAULT 0
)
"""
COLS = ["id", "ts", "device", "user", "allowed", "score", "votes", "liveness", "latency"]


class EventStore:
    def __init__(self, cfg: dict):
        e = cfg.get("events", {})
        self.device = cfg.get("device", {}).get("id", "reqagnize-edge")
        self.path = abspath(cfg, e.get("sqlite_path", "events.db"))
        self.api = str(e.get("cloud_api_url", "")).rstrip("/")
        self.token = e.get("device_token", "")
        self.sync_enabled = bool(e.get("sync_enabled", True)) and bool(self.api) and requests is not None

        os.makedirs(os.path.dirname(self.path), exist_ok=True) if os.path.dirname(self.path) else None
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(SCHEMA)
        self._conn.commit()
        self._stop = False
        if self.sync_enabled:
            threading.Thread(target=self._sync_loop, daemon=True).start()

    def record(self, user, allowed, score=None, votes=None, liveness=None, latency=None):
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(ts,device,user,allowed,score,votes,liveness,latency) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (time.time(), self.device, user or "", 1 if allowed else 0,
                 score, votes, liveness, latency),
            )
            self._conn.commit()

    # ── sync ──────────────────────────────────────────────────────────────────
    def _sync_loop(self):
        while not self._stop:
            try:
                self.sync_once()
            except Exception:
                pass
            time.sleep(10)

    def sync_once(self) -> int:
        """Push unsynced rows to the cloud. Returns number synced."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {','.join(COLS)} FROM events WHERE synced=0 ORDER BY id LIMIT 200"
            ).fetchall()
        if not rows:
            return 0
        payload = [dict(zip(COLS, r)) for r in rows]
        resp = requests.post(
            f"{self.api}/api/events",
            json={"device": self.device, "events": payload},
            headers={"X-Device-Token": self.token},
            timeout=5,
        )
        if resp.status_code == 200:
            ids = [(r[0],) for r in rows]
            with self._lock:
                self._conn.executemany("UPDATE events SET synced=1 WHERE id=?", ids)
                self._conn.commit()
            return len(ids)
        return 0

    def close(self):
        self._stop = True
        try:
            self._conn.close()
        except Exception:
            pass
