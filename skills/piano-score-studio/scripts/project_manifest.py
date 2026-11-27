"""Canonical Project Manifest (IR v3.0) for Piano Score Studio.

Serves as the single source of truth shared across Stages 0-5, the one-command CLI
(`piano_studio.py`), and Agent repair loops (`bar_overrides`).
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "3.0"

DEFAULT_STAGE_STATUS = {
    "stage0_triage": "pending",
    "stage1_calibrate_transcribe": "pending",
    "stage2_pitch_audit": "pending",
    "stage4_1_dynamics_plan": "pending",
    "stage3_engrave_score": "pending",
    "stage4_render_audio": "pending",
    "stage5_ab_and_eval": "pending",
}


def create_manifest(
    inputs: list[str],
    out_dir: Path,
    song_name: str,
    title: str,
    engine: str = "auto",
    device: str = "auto",
    style_profile: str = "auto",
) -> dict[str, Any]:
    """Initialize a fresh v3.0 project manifest dictionary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "project": {
            "song_name": song_name,
            "title": title,
            "inputs": [str(Path(x).resolve()) for x in inputs],
            "out_dir": str(out_dir.resolve()),
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "engines": {
            "transcription_engine_requested": engine,
            "transcription_engine_used": None,
            "device": device,
            "style_profile": style_profile,
        },
        "stage0_triage": {},
        "stage1_calibration": {},
        "artifacts": {},
        "bar_overrides": {},
        "stage_status": dict(DEFAULT_STAGE_STATUS),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    """Load an existing project_manifest.json and ensure required v3.0 keys exist."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("engines", {})
    data.setdefault("stage0_triage", {})
    data.setdefault("stage1_calibration", {})
    data.setdefault("artifacts", {})
    data.setdefault("bar_overrides", {})
    status = dict(DEFAULT_STAGE_STATUS)
    status.update(data.get("stage_status") or {})
    data["stage_status"] = status
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> Path:
    """Atomically write project_manifest.json with updated timestamp."""
    data.setdefault("project", {})
    data["project"]["updated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def set_stage_status(manifest: dict[str, Any], stage_key: str, status: str) -> None:
    manifest.setdefault("stage_status", dict(DEFAULT_STAGE_STATUS))[stage_key] = status


def is_stage_done(manifest: dict[str, Any], stage_key: str) -> bool:
    return (manifest.get("stage_status") or {}).get(stage_key) == "done"