from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SETTINGS_VERSION = 1


def save_settings(path: str | Path, settings: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(settings)
    payload["settings_version"] = SETTINGS_VERSION
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_settings(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("settings_version", 1)
    if version != SETTINGS_VERSION:
        raise ValueError(f"Unsupported settings version: {version}")
    return payload


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value

