"""Canonical Project Manifest (IR v3.0) for Piano Score Studio.

Serves as the single source of truth shared across Stages 0-5, the one-command CLI
(`piano_studio.py`), and Agent repair loops (`bar_overrides`).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "3.0"

DEFAULT_STAGE_STATUS = {
    "stage0_triage": "pending",
    "stage1_calibrate_transcribe": "pending",
    "stage1_5_style_analysis": "pending",
    "stage2_pitch_audit": "pending",
    "stage4_1_dynamics_plan": "pending",
    "stage3_engrave_score": "pending",
    "stage4_render_audio": "pending",
    "stage5_ab_and_eval": "pending",
}

STAGE_ORDER = [
    "stage0_triage",
    "stage1_calibrate_transcribe",
    "stage1_5_style_analysis",
    "stage2_pitch_audit",
    "stage4_1_dynamics_plan",
    "stage3_engrave_score",
    "stage4_render_audio",
    "stage5_ab_and_eval",
]

# Parameter groups mapped to the earliest stage they invalidate when changed
PARAM_STAGE_ROOTS = {
    "inputs": "stage0_triage",
    "engine": "stage1_calibrate_transcribe",
    "device": "stage1_calibrate_transcribe",
    "bpm": "stage1_calibrate_transcribe",
    "time_sig": "stage1_calibrate_transcribe",
    "tempo_mode": "stage1_calibrate_transcribe",
    "rubato_mode": "stage1_calibrate_transcribe",
    "subdivision": "stage1_calibrate_transcribe",
    "style": "stage1_5_style_analysis",
    "key_fifths": "stage1_5_style_analysis",
    "title": "stage3_engrave_score",
    "skip_pdf": "stage3_engrave_score",
    "soundfont": "stage4_render_audio",
    "render_backend": "stage4_render_audio",
    "auto_repair": "stage5_ab_and_eval",
}


def compute_run_params_hash(params: dict[str, Any]) -> str:
    """Deterministic SHA-256 prefix (16 hex chars) for CLI run parameters."""
    canonical = json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def invalidate_stale_stages(manifest: dict[str, Any], new_params: dict[str, Any]) -> list[str]:
    """Compare new CLI parameters against saved manifest run_params (P2-5).

    If any parameter changed, marks the earliest affected stage and all downstream
    stages as 'pending', updates manifest['run_params'] and manifest['run_params_hash'],
    and returns the invalidated stage keys.
    """
    new_hash = compute_run_params_hash(new_params)
    old_params = manifest.get("run_params")
    old_hash = manifest.get("run_params_hash")

    if not old_params and not old_hash:
        manifest["run_params"] = dict(new_params)
        manifest["run_params_hash"] = new_hash
        return []

    if old_hash == new_hash:
        return []

    earliest_idx = len(STAGE_ORDER)
    all_keys = set(new_params.keys()) | set((old_params or {}).keys())
    for k in all_keys:
        if (old_params or {}).get(k) != new_params.get(k):
            root_stage = PARAM_STAGE_ROOTS.get(k, "stage1_calibrate_transcribe")
            if root_stage in STAGE_ORDER:
                earliest_idx = min(earliest_idx, STAGE_ORDER.index(root_stage))
            else:
                earliest_idx = 0

    invalidated: list[str] = []
    status = manifest.setdefault("stage_status", dict(DEFAULT_STAGE_STATUS))
    for st in STAGE_ORDER[earliest_idx:]:
        if status.get(st) != "pending":
            status[st] = "pending"
            invalidated.append(st)

    manifest["run_params"] = dict(new_params)
    manifest["run_params_hash"] = new_hash
    return invalidated


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
