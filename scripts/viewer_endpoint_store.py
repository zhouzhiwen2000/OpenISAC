from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


SETTINGS_DIR = Path.home() / ".openisac_viewer_settings"


def default_settings_key() -> str:
    name = Path(sys.argv[0] or "viewer").stem
    return name or "viewer"


def _settings_path(settings_key: str) -> Path:
    safe_key = (settings_key or "viewer").strip() or "viewer"
    return SETTINGS_DIR / f"{safe_key}.json"


def load_settings(settings_key: str) -> dict[str, Any]:
    path = _settings_path(settings_key)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(settings_key: str, updates: dict[str, Any]) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_settings(settings_key)
    for key, value in updates.items():
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
        data[key] = value
    path = _settings_path(settings_key)
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def load_endpoint(
    settings_key: str,
    default_host: str = "127.0.0.1",
    default_port: int = 0,
) -> tuple[str, int]:
    data = load_settings(settings_key)
    host = str(data.get("host") or default_host or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(data.get("port", default_port))
    except (TypeError, ValueError):
        port = int(default_port)
    return host, port


def save_endpoint(settings_key: str, host: str, port: int) -> None:
    save_settings(
        settings_key,
        {
            "host": str(host or "127.0.0.1").strip() or "127.0.0.1",
            "port": int(port),
        },
    )
