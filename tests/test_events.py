"""CI test: on-device event store writes rows to SQLite. No network."""

import os
import sqlite3
import tempfile

from app.config import load_config
from app.events.store import EventStore


def test_event_store_writes():
    cfg = load_config()
    cfg["events"]["sqlite_path"] = os.path.join(tempfile.mkdtemp(), "e.db")
    cfg["events"]["sync_enabled"] = False
    es = EventStore(cfg)
    es.record(user="Alice", allowed=True, score=0.9, votes=3, liveness="fast", latency=1.0)
    es.record(user=None, allowed=False, liveness="liveness_fail")
    count, allowed = sqlite3.connect(cfg["events"]["sqlite_path"]).execute(
        "SELECT COUNT(*), SUM(allowed) FROM events"
    ).fetchone()
    es.close()
    assert count == 2 and allowed == 1
