"""Config loader — reads NVIDIA_App/config.yaml and resolves paths against the app root."""

import os
import yaml

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NVIDIA_App/


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(APP_ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_root"] = APP_ROOT
    return cfg


def abspath(cfg: dict, rel: str) -> str:
    """Resolve a config-relative path to an absolute path under the app root."""
    if os.path.isabs(rel):
        return rel
    return os.path.abspath(os.path.join(cfg["_root"], rel))


def qfr_path(cfg: dict) -> str:
    """Absolute path to the quantum model code/weights (env QFR_PATH wins)."""
    env = os.environ.get("QFR_PATH")
    if env:
        return os.path.abspath(env)
    return abspath(cfg, cfg.get("paths", {}).get("qfr_path", "../quantum_face_recognition"))
