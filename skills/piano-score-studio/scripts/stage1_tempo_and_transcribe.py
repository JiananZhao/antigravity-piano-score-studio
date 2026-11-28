"""Stage 1: Beat tracking (beat_this), circular BPM & phase-drift calibration,
meter & first-downbeat locking, coarsest quantization grid selection,
and baseline MIDI + MusicXML transcription via MuScriptor.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage0_audio_triage import detect_first_note_sec
from tempo_map import TempoMap
from transcription_engines import (
    quantize_performance_midi,
    resolve_transcription_engine,
    run_transkun_transcribe,
)


def find_plugin_root() -> Path:
    # scripts/ -> piano-score-studio/ -> skills/ -> plugin root
    return Path(__file__).resolve().parents[3]


def load_config(plugin_root: Path, cfg_arg: str | None) -> dict:
    p = Path(cfg_arg) if cfg_arg else (plugin_root / "config.json")
    if not p.is_file():
        p = plugin_root / "config.example.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def apply_cache_env(plugin_root: Path, cfg: dict) -> None:
    """Configure HF_HOME and TORCH_HOME from config.json before importing ML libraries."""
    for cfg_key, env_key in (("hf_home", "HF_HOME"), ("torch_home", "TORCH_HOME")):
        val = cfg.get(cfg_key)
        if val and env_key not in os.environ:
            p = Path(val)
            if not p.is_absolute():
                p = plugin_root / p
            p.mkdir(parents=True, exist_ok=True)
            os.environ[env_key] = str(p)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_musescore(explicit: str | None, cfg: dict) -> str | None:
    for cand in (explicit, cfg.get("musescore_exe"), os.environ.get("MUSCRIPTOR_MUSESCORE")):
        if cand and Path(cand).is_file():
            return str(Path(cand))
    for name in ("mscore", "musescore", "MuseScore4", "musescore4", "mscore4"):
        hit = shutil.which(name)
        if hit:
            return hit
    for p in (
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe",
        Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
        Path("/usr/bin/mscore"),
        Path("/usr/bin/musescore"),
    ):
        if p.is_file():
            return str(p)
    return None


def resolve_muscriptor_weights(plugin_root: Path, cfg: dict, explicit: str | None) -> str | None:
    if explicit:
        p = Path(explicit)
        if p.is_dir() and (p / "model.safetensors").is_file():
            return str(p / "model.safetensors")
        return explicit
    rel = cfg.get("muscriptor_model_dir") or "models/medium"
    model_dir = Path(rel) if Path(rel).is_absolute() else (plugin_root / rel)
    safetensors = model_dir / "model.safetensors"
    if safetensors.is_file():
        return str(safetensors)
    return None


def load_audio_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float64 mono; fall back to beat_this.inference.load_audio for exotic containers."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float64", always_2d=True)
        return data.mean(axis=1), int(sr)
    except Exception:
        from beat_this.inference import load_audio

        sig, sr = load_audio(str(audio_path), dtype="float64")
        if sig.ndim > 1:
            sig = sig.mean(axis=-1)
        return np.asarray(sig, dtype=np.float64), int(sr)


def extract_onset_features(mono: np.ndarray, sr: int) -> dict[str, np.ndarray | float]:
    """Compute 10 ms spectral onset flux (full + bass + chroma change) and first active note time."""
    hop_10ms = max(1, int(round(sr * 0.01)))
    first_note_sec = detect_first_note_sec(mono, sr)

    n_fft = 2048
    nf = (len(mono) - n_fft) // hop_10ms + 1
    if nf < 4:
        empty = np.zeros(1, dtype=np.float64)
        return {
            "times": empty,
            "full_flux": empty,
            "bass_flux": empty,
            "chroma_flux": empty,
            "first_note_sec": first_note_sec,
        }

    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    full_mask = (freqs >= 80.0) & (freqs <= 5120.0)
    bass_mask = (freqs >= 45.0) & (freqs <= 260.0)
    chroma_mask = (freqs >= 65.0) & (freqs <= 2093.0)
    chroma_bins = np.round(12.0 * np.log2(np.maximum(freqs[chroma_mask], 1e-6) / 440.0) + 69.0).astype(int) % 12

    full_flux = np.zeros(nf - 1, dtype=np.float64)
    bass_flux = np.zeros(nf - 1, dtype=np.float64)
    chroma_flux = np.zeros(nf - 1, dtype=np.float64)

    prev_full: np.ndarray | None = None
    prev_bass: np.ndarray | None = None
    prev_chroma: np.ndarray | None = None

    chunk_frames = 2048
    for f0 in range(0, nf, chunk_frames):
        f1 = min(nf, f0 + chunk_frames)
        idx = np.arange(n_fft)[None, :] + np.arange(f0, f1)[:, None] * hop_10ms
        frames = mono[idx].astype(np.float32) * win[None, :]
        mag = np.abs(np.fft.rfft(frames, axis=1))

        log_full = np.log1p(1000.0 * mag[:, full_mask])
        log_bass = np.log1p(1000.0 * mag[:, bass_mask])

        c_mag = mag[:, chroma_mask] ** 2
        chroma = np.zeros((f1 - f0, 12), dtype=np.float64)
        for pc in range(12):
            pc_cols = chroma_bins == pc
            if np.any(pc_cols):
                chroma[:, pc] = np.sum(c_mag[:, pc_cols], axis=1)
        chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-12

        if prev_full is not None:
            full_flux[f0 - 1] = float(np.mean(np.maximum(0.0, log_full[0] - prev_full)))
            bass_flux[f0 - 1] = float(np.mean(np.maximum(0.0, log_bass[0] - prev_bass)))
            chroma_flux[f0 - 1] = float(1.0 - np.dot(chroma[0], prev_chroma))
        if len(log_full) > 1:
            full_flux[f0 : f1 - 1] = np.mean(np.maximum(0.0, np.diff(log_full, axis=0)), axis=1)
            bass_flux[f0 : f1 - 1] = np.mean(np.maximum(0.0, np.diff(log_bass, axis=0)), axis=1)
            chroma_flux[f0 : f1 - 1] = 1.0 - np.sum(chroma[1:] * chroma[:-1], axis=1)

        prev_full = log_full[-1]
        prev_bass = log_bass[-1]
        prev_chroma = chroma[-1]

    times = (np.arange(1, nf) * hop_10ms + 0.5 * n_fft) / float(sr)
    return {
        "times": times,
        "full_flux": full_flux,
        "bass_flux": bass_flux,
        "chroma_flux": np.maximum(0.0, chroma_flux),
        "first_note_sec": round(first_note_sec, 3),
    }


def regularize_beats(beats: np.ndarray) -> np.ndarray:
    """Fill missed beats and drop extra double-triggered beats relative to median beat spacing."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 3:
        return beats
    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return beats
    kept = [float(beats[0])]
    for beat in beats[1:]:
        ratio = (float(beat) - kept[-1]) / period
        if ratio < 0.7:
            continue
        steps = max(1, int(round(ratio)))
        if steps > 1:
            for k in range(1, steps):
                kept.append(kept[-1] + (float(beat) - kept[-1]) * (k / steps))
        kept.append(float(beat))
    return np.asarray(kept, dtype=np.float64)


def run_beat_this(mono: np.ndarray, sr: int, device: str, checkpoint: str = "final0") -> dict:
    """Run CPJKU/beat_this inference and summarize regularized beats & downbeats."""
    from beat_this.inference import Audio2Beats

    tracker = Audio2Beats(checkpoint_path=checkpoint, device=device, dbn=False)
    raw_beats, raw_downbeats = tracker(mono.astype(np.float32), sr)
    beats = regularize_beats(np.asarray(raw_beats, dtype=np.float64))
    downbeats = np.asarray(raw_downbeats, dtype=np.float64)

    if len(beats) >= 2:
        idx = np.arange(len(beats), dtype=np.float64)
        slope, intercept = np.polyfit(idx, beats, 1)
        fit_bpm = float(60.0 / max(slope, 1e-6))
        residual_ms = float(np.std(beats - (slope * idx + intercept)) * 1000.0)
    else:
        fit_bpm, residual_ms = 120.0, 999.0

    # Infer raw beat_this beats-per-bar from beat-index gaps between consecutive downbeats
    bt_bpb = None
    if len(downbeats) >= 3 and len(beats) >= 4:
        db_idx = np.abs(downbeats[:, None] - beats[None, :]).argmin(axis=1)
        counts = np.diff(db_idx)
        counts = counts[(counts >= 2) & (counts <= 12)]
        if len(counts):
            vals, freqs = np.unique(counts, return_counts=True)
            best_i = int(np.argmax(freqs))
            if freqs[best_i] / len(counts) >= 0.5:
                bt_bpb = int(vals[best_i])

    return {
        "raw_beat_count": int(len(raw_beats)),
        "regularized_beat_count": int(len(beats)),
        "downbeat_count": int(len(downbeats)),
        "beats": beats,
        "downbeats": downbeats,
        "beat_this_bpm": round(fit_bpm, 3),
        "beat_residual_ms": round(residual_ms, 2),
        "beat_this_bpb": bt_bpb,
    }


def _onset_weights(flux: np.ndarray) -> np.ndarray:
    thr = float(np.median(flux) + 1.5 * np.std(flux))
    return np.where(flux > thr, flux - thr, 0.0)


def eval_circular_bpm(
    times: np.ndarray,
    weights: np.ndarray,
    bpm: float,
    ref_time: float,
    dur_s: float,
) -> tuple[float, float, float]:
    """Return (circular_R, max_20s_drift_ms, phase_shift_sec) matching eval_gate_runner.py."""
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-12:
        return 0.0, 999.0, 0.0
    beat_s = 60.0 / bpm
    phi = 2.0 * np.pi * ((times - ref_time) / beat_s)
    z_all = np.sum(weights * np.exp(1j * phi)) / w_sum
    r_val = float(np.abs(z_all))
    global_ang = float(np.angle(z_all))
    phase_shift_sec = (global_ang / (2.0 * np.pi)) * beat_s

    drifts: list[float] = []
    for t0 in range(0, max(20, int(dur_s)), 20):
        m_win = (times >= t0) & (times < t0 + 20.0)
        w_win = float(np.sum(weights[m_win]))
        if w_win > 0.02 * w_sum:
            z_w = np.sum(weights[m_win] * np.exp(1j * phi[m_win])) / w_win
            ang_diff = float(np.angle(z_w * np.exp(-1j * global_ang)))
            drifts.append(abs(ang_diff) / (2.0 * np.pi) * beat_s * 1000.0)
    max_drift_ms = float(max(drifts)) if drifts else 0.0
    return r_val, max_drift_ms, phase_shift_sec


def calibrate_bpm_and_mode(
    times: np.ndarray,
    full_flux: np.ndarray,
    dur_s: float,
    bt_info: dict,
    bpm_override: float | None,
    mode_arg: str,
) -> dict:
    """Sweep circular resultant |R| around beat_this BPM and inspect 20 s segment phase drift."""
    weights = _onset_weights(full_flux)
    ref_t = float(bt_info["downbeats"][0]) if len(bt_info["downbeats"]) else (float(bt_info["beats"][0]) if len(bt_info["beats"]) else 0.0)

    if bpm_override is not None:
        best_bpm = float(bpm_override)
        r_val, max_drift_ms, phase_shift = eval_circular_bpm(times, weights, best_bpm, ref_t, dur_s)
        cand_summary = [{"bpm": round(best_bpm, 3), "circular_R": round(r_val, 3), "max_drift_ms": round(max_drift_ms, 1), "source": "override"}]
    else:
        b0 = float(bt_info["beat_this_bpm"])
        # Prioritize beat_this base tempo; check half/double only when base is outside comfortable piano range
        anchors = [(b0, 1.0)]
        for mult, prior in ((0.5, 0.92), (2.0, 0.92), (1.5, 0.88), (2.0 / 3.0, 0.88)):
            cand = b0 * mult
            if 42.0 <= cand <= 210.0:
                anchors.append((cand, prior))

        best_score = -1.0
        best_bpm = b0
        cand_summary = []
        for anchor, prior in anchors:
            lo, hi = max(35.0, anchor * 0.96), min(240.0, anchor * 1.04)
            coarse = np.arange(lo, hi + 1e-6, 0.1)
            local_best_r, local_best_b = -1.0, anchor
            for b in coarse:
                r, _, _ = eval_circular_bpm(times, weights, float(b), ref_t, dur_s)
                if r > local_best_r:
                    local_best_r, local_best_b = r, float(b)
            # Fine refinement (0.02 BPM)
            fine = np.arange(max(35.0, local_best_b - 0.2), min(240.0, local_best_b + 0.2) + 1e-6, 0.02)
            for b in fine:
                r, _, _ = eval_circular_bpm(times, weights, float(b), ref_t, dur_s)
                if r > local_best_r:
                    local_best_r, local_best_b = r, float(b)

            r_fin, drift_fin, _ = eval_circular_bpm(times, weights, local_best_b, ref_t, dur_s)
            score = r_fin * prior
            cand_summary.append(
                {
                    "anchor_bpm": round(anchor, 2),
                    "refined_bpm": round(local_best_b, 3),
                    "circular_R": round(r_fin, 3),
                    "max_drift_ms": round(drift_fin, 1),
                    "weighted_score": round(score, 4),
                }
            )
            if score > best_score:
                best_score, best_bpm = score, local_best_b

        # Guard: fixed-tempo circular |R| < 0.45 is statistically noise on rubato / expressive pieces.
        # Never let noise pick a bogus 1.5x / 2x / 0.5x multiplier that corrupts beat_this's true pulse!
        max_cand_r = max((float(c["circular_R"]) for c in cand_summary), default=0.0)
        if max_cand_r < 0.45:
            best_bpm = b0

        r_val, max_drift_ms, phase_shift = eval_circular_bpm(times, weights, best_bpm, ref_t, dur_s)

    steady_recommended = (r_val >= 0.70) and (max_drift_ms <= 30.0)
    if mode_arg == "fixed-tempo":
        tempo_mode = "fixed-tempo"
    elif mode_arg == "tempo-map":
        tempo_mode = "tempo-map"
    else:
        tempo_mode = "fixed-tempo" if steady_recommended else "tempo-map"

    return {
        "bpm": round(best_bpm, 3),
        "circular_R": round(r_val, 3),
        "max_abs_drift_ms": round(max_drift_ms, 1),
        "phase_shift_sec": round(phase_shift, 4),
        "steady_recommended": steady_recommended,
        "tempo_mode": tempo_mode,
        "candidates": cand_summary,
    }


def _extract_beat_grid_times(bpm: float, ref_t: float, dur_s: float) -> np.ndarray:
    beat_s = 60.0 / bpm
    k_start = math.floor((0.0 - ref_t) / beat_s)
    k_end = math.ceil((dur_s - ref_t) / beat_s)
    grid = ref_t + np.arange(k_start, k_end + 1, dtype=np.float64) * beat_s
    return grid[(grid >= 0.0) & (grid <= dur_s)]


def _sample_feature_at_beats(
    times: np.ndarray,
    feat: np.ndarray,
    beat_times: np.ndarray,
    half_win_s: float = 0.045,
) -> np.ndarray:
    vals = np.zeros(len(beat_times), dtype=np.float64)
    for i, tb in enumerate(beat_times):
        i0 = int(np.searchsorted(times, tb - half_win_s, side="left"))
        i1 = int(np.searchsorted(times, tb + half_win_s, side="right"))
        if i1 > i0:
            vals[i] = float(np.max(feat[i0:i1]))
    return vals


def calibrate_meter_and_downbeat(
    times: np.ndarray,
    full_flux: np.ndarray,
    bass_flux: np.ndarray,
    chroma_flux: np.ndarray,
    dur_s: float,
    first_note_sec: float,
    bpm_info: dict,
    bt_info: dict,
    bpb_override: int | None,
    beat_type_override: int | None,
    downbeat_override: float | None,
) -> dict:
    """Lock meter (beats_per_bar, beat_type) and first_downbeat via bass/harmonic cycle dominance + beat_this."""
    bpm = float(bpm_info["bpm"])
    beat_s = 60.0 / bpm
    raw_ref = float(bt_info["downbeats"][0]) if len(bt_info["downbeats"]) else (float(bt_info["beats"][0]) if len(bt_info["beats"]) else first_note_sec)
    aligned_ref = raw_ref + float(bpm_info["phase_shift_sec"])

    grid_beats = _extract_beat_grid_times(bpm, aligned_ref, dur_s)
    if len(grid_beats) < 4:
        bpb = bpb_override or bt_info.get("beat_this_bpb") or 4
        bt = beat_type_override or 4
        return {
            "beats_per_bar": int(bpb),
            "beat_type": int(bt),
            "bpm_for_meter": round(bpm, 3),
            "first_downbeat": round(downbeat_override if downbeat_override is not None else first_note_sec, 4),
            "pickup_sec": 0.0,
            "meter_scores": [],
        }

    f_full = _sample_feature_at_beats(times, full_flux, grid_beats)
    f_bass = _sample_feature_at_beats(times, bass_flux, grid_beats)
    f_harm = _sample_feature_at_beats(times, chroma_flux, grid_beats, half_win_s=0.08)

    # Normalize each feature to zero-mean unit-std for comparable weighting
    def norm(x: np.ndarray) -> np.ndarray:
        s = float(np.std(x))
        return (x - float(np.mean(x))) / (s + 1e-12)

    combined = 0.45 * norm(f_bass) + 0.35 * norm(f_harm) + 0.20 * norm(f_full)

    # Downbeat votes from beat_this projected onto grid_beats
    db_votes = np.zeros(len(grid_beats), dtype=np.float64)
    for db in bt_info["downbeats"]:
        j = int(np.argmin(np.abs(grid_beats - db)))
        if abs(grid_beats[j] - db) <= 0.35 * beat_s:
            db_votes[j] = 1.0

    # Evaluate candidate bar cycles
    if bpb_override is not None:
        candidates = [(int(bpb_override), int(beat_type_override or 4))]
    elif beat_type_override == 8:
        candidates = [(6, 8), (12, 8)]
    elif beat_type_override == 4:
        candidates = [(4, 4), (3, 4), (2, 4)]
    else:
        candidates = [(4, 4), (3, 4), (6, 8), (2, 4)]

    meter_scores = []
    best_meter = candidates[0]
    best_phase = 0
    best_total = -1e9

    for bpb, btype in candidates:
        # In 6/8 or 12/8 when beat_this tracked dotted-quarter beats (bpm < 105), each bar has bpb/3 compound beats
        # When beat_this tracked eighth beats (bpm >= 105), each bar has bpb tracked beats
        cycle_len = bpb
        if btype == 8 and bpb % 3 == 0 and bpm < 105.0:
            cycle_len = bpb // 3

        phase_scores = []
        for p in range(cycle_len):
            beat1_vals = combined[p::cycle_len]
            other_mask = np.ones(len(combined), dtype=bool)
            other_mask[p::cycle_len] = False
            other_vals = combined[other_mask]
            if not len(beat1_vals) or not len(other_vals):
                continue
            contrast = float(np.mean(beat1_vals) - np.mean(other_vals))
            db_agree = float(np.mean(db_votes[p::cycle_len])) if np.any(db_votes > 0) else 0.0
            # Mild prior matching beat_this inferred bpb
            bt_bonus = 0.25 if (bt_info.get("beat_this_bpb") == cycle_len) else 0.0
            # Penalize 2/4 when 4/4 is equally strong (avoid over-splitting 4/4 bars)
            meter_prior = -0.15 if (bpb == 2 and btype == 4) else 0.0
            if btype == 8 and bpb == 6:
                # 6/8 requires compound feel (either secondary accent on beat 4 of 6 or beat_this saw 2/6)
                if cycle_len == 6:
                    b4_vals = combined[(p + 3) % 6 :: 6]
                    b2356 = np.concatenate([combined[(p + o) % 6 :: 6] for o in (1, 2, 4, 5)])
                    compound_bonus = 0.20 * float(np.mean(b4_vals) - np.mean(b2356))
                else:
                    compound_bonus = -0.08  # prefer 3/4 or 4/4 unless clearly compound
            else:
                compound_bonus = 0.0

            total = contrast + 0.85 * db_agree + bt_bonus + meter_prior + compound_bonus
            phase_scores.append((total, p, contrast, db_agree))

        if not phase_scores:
            continue
        phase_scores.sort(reverse=True)
        top_total, top_p, top_contrast, top_db = phase_scores[0]
        meter_scores.append(
            {
                "meter": f"{bpb}/{btype}",
                "cycle_len": cycle_len,
                "best_phase": int(top_p),
                "score": round(top_total, 4),
                "beat1_contrast": round(top_contrast, 4),
                "beat_this_downbeat_agree": round(top_db, 4),
            }
        )
        if top_total > best_total:
            best_total = top_total
            best_meter = (bpb, btype)
            best_phase = top_p

    sel_bpb, sel_btype = best_meter
    # Convert beat-unit BPM when compound /8 meter was tracked at dotted-quarter pulse
    if sel_btype == 8 and sel_bpb % 3 == 0 and bpm < 105.0:
        bpm_for_meter = bpm * 3.0  # eighth-note BPM = 3 * dotted-quarter BPM
        bar_s = (sel_bpb / 3.0) * beat_s
    else:
        bpm_for_meter = bpm
        bar_s = sel_bpb * (60.0 / bpm_for_meter)

    if downbeat_override is not None:
        first_downbeat = float(downbeat_override)
    else:
        d0 = float(grid_beats[best_phase])
        # Snap to the first barline of actual music so Bar 0 pickup aligns cleanly with stage3
        tol = 0.25 * (60.0 / bpm_for_meter)
        k = math.ceil((first_note_sec - tol - d0) / bar_s)
        first_downbeat = max(0.0, d0 + k * bar_s)

    pickup_sec = max(0.0, first_downbeat - first_note_sec)
    if pickup_sec < 0.25 * (60.0 / bpm_for_meter):
        pickup_sec = 0.0

    return {
        "beats_per_bar": int(sel_bpb),
        "beat_type": int(sel_btype),
        "bpm_for_meter": round(bpm_for_meter, 3),
        "first_downbeat": round(first_downbeat, 4),
        "pickup_sec": round(pickup_sec, 4),
        "meter_scores": meter_scores,
    }


def resample_beats_to_quarter_pulse(
    beats: np.ndarray,
    tracked_bpm: float,
    quarter_bpm: float,
    first_downbeat: float,
) -> tuple[np.ndarray, float]:
    """Resample tracked beat_this beat times onto the quarter-note pulse expected by MuScriptor.

    Handles 6/8 / 12/8 compound meters (1 dotted-quarter beat -> 1.5 quarter notes, or 1 eighth
    beat -> 0.5 quarter notes) as well as half/double-time BPM adjustments while keeping every
    barline anchored on `first_downbeat`.
    """
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 2 or tracked_bpm <= 0.0 or quarter_bpm <= 0.0:
        return beats, 1.0

    raw_ratio = float(quarter_bpm) / float(tracked_bpm)
    # Snap to exact musical multipliers when within 8% so barlines remain locked to real downbeats
    clean_ratios = (0.25, 0.5, 2.0 / 3.0, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    nearest_clean = min(clean_ratios, key=lambda r: abs(raw_ratio - r) / r)
    quarter_per_tracked = nearest_clean if abs(raw_ratio - nearest_clean) / nearest_clean <= 0.08 else raw_ratio

    if math.isclose(quarter_per_tracked, 1.0, rel_tol=1e-4):
        return beats, 1.0

    k0 = int(np.argmin(np.abs(beats - first_downbeat)))
    q_offset_k0 = (float(beats[k0]) - float(first_downbeat)) * (quarter_bpm / 60.0)
    if abs(q_offset_k0) <= 0.25:
        q_offset_k0 = 0.0
    else:
        q_offset_k0 = round(q_offset_k0 * 2.0) / 2.0

    tracked_qpos = q_offset_k0 + (np.arange(len(beats), dtype=np.float64) - float(k0)) * quarter_per_tracked
    q_min = math.ceil(float(tracked_qpos[0]) - 1e-9)
    q_max = math.floor(float(tracked_qpos[-1]) + 1e-9)
    if q_max <= q_min:
        return beats, round(quarter_per_tracked, 4)

    target_qpos = np.arange(q_min, q_max + 1, dtype=np.float64)
    resampled = np.interp(target_qpos, tracked_qpos, beats)
    return resampled, round(quarter_per_tracked, 4)


def select_quantization_grid(
    onset_times: np.ndarray,
    quarter_bpm: float,
    first_downbeat: float,
    beat_type: int,
    grid_override: int | None,
    tempo_map: TempoMap | None = None,
) -> dict:
    """Pick the coarsest grid (2, 3, or 4 subdivisions per quarter note) covering >= 95% of onsets."""
    allowed = (2, 4) if beat_type == 8 else (2, 3, 4)
    quarter_s = 60.0 / quarter_bpm
    onsets = np.asarray(onset_times, dtype=np.float64)
    use_rubato = tempo_map is not None and tempo_map.is_rubato and len(onsets) > 0
    q_onsets = np.array([tempo_map.sec_to_q(float(t)) for t in onsets], dtype=np.float64) if use_rubato else None

    coverages: dict[str, float] = {}
    for g in allowed:
        if len(onsets) == 0:
            coverages[str(g)] = 100.0
            continue
        step_s = quarter_s / float(g)
        tol_s = min(0.22 * step_s, 0.040)
        if use_rubato and q_onsets is not None:
            rel = q_onsets * float(g)
            dist_s = (np.abs(rel - np.round(rel)) / float(g)) * quarter_s
        else:
            rel = (onsets - first_downbeat) / step_s
            dist_s = np.abs(rel - np.round(rel)) * step_s
        coverages[str(g)] = round(float(np.mean(dist_s <= tol_s) * 100.0), 2)

    if grid_override is not None:
        chosen = int(grid_override)
        reason = "manual_override"
    else:
        chosen = None
        for g in allowed:
            if coverages[str(g)] >= 95.0:
                chosen = g
                reason = f"coarsest_ge_95pct ({coverages[str(g)]:.1f}%)"
                break
        if chosen is None:
            # Pick highest coverage; break near-ties (<= 1.5%) in favor of coarser grid
            best_cov = max(coverages.values())
            for g in allowed:
                if coverages[str(g)] >= best_cov - 1.5:
                    chosen = g
                    reason = f"best_available_coarsest ({coverages[str(g)]:.1f}%)"
                    break
            chosen = chosen or allowed[0]

    return {
        "grid": int(chosen),
        "reason": reason,
        "coverage_pct_by_grid": coverages,
        "onset_count_evaluated": int(len(onsets)),
    }


def detect_onset_peak_times(times: np.ndarray, full_flux: np.ndarray) -> np.ndarray:
    """Pick local spectral flux maxima above threshold for pre-transcription grid estimation."""
    if len(full_flux) < 3:
        return np.array([], dtype=np.float64)
    thr = float(np.median(full_flux) + 1.5 * np.std(full_flux))
    peaks = []
    last_t = -1.0
    for i in range(1, len(full_flux) - 1):
        if full_flux[i] > thr and full_flux[i] >= full_flux[i - 1] and full_flux[i] >= full_flux[i + 1]:
            if times[i] - last_t >= 0.05:
                peaks.append(float(times[i]))
                last_t = float(times[i])
    return np.asarray(peaks, dtype=np.float64)


def _shift_muscriptor_events(events: list, pad_sec: float) -> list:
    """Subtract leading silence padding from MuScriptor NoteStartEvent / NoteEndEvent stream."""
    if pad_sec <= 0.0:
        return events
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    start_map: dict[int, NoteStartEvent] = {}
    shifted = []
    next_idx = 0
    for ev in events:
        if isinstance(ev, NoteStartEvent):
            st = ev.start_time - pad_sec
            if st < -0.15:
                continue
            new_start = NoteStartEvent(
                pitch=ev.pitch,
                start_time=max(0.0, st),
                index=next_idx,
                instrument=ev.instrument,
            )
            start_map[ev.index] = new_start
            shifted.append(new_start)
            next_idx += 1
        elif isinstance(ev, NoteEndEvent):
            orig_idx = getattr(ev, "start_event_index", None)
            if orig_idx is None and getattr(ev, "start_event", None) is not None:
                orig_idx = ev.start_event.index
            new_start = start_map.get(orig_idx)
            if new_start is None:
                continue
            et = max(new_start.start_time + 0.02, ev.end_time - pad_sec)
            shifted.append(NoteEndEvent(end_time=et, start_event=new_start))
    return shifted


def _build_filtered_dataclass(cls, kwargs: dict):
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in kwargs.items() if k in valid})


def transcribe_with_muscriptor(
    audio_path: Path,
    weights_path: str | None,
    device: str,
    instruments: list[str] | None,
    pad_sec: float,
    quarter_bpm: float,
    beats_per_bar_quarter: int,
    first_downbeat: float,
    beats_array: np.ndarray,
    tempo_mode: str,
    grid_override: int | None,
    beat_type: int,
    out_dir: Path,
) -> dict:
    """Run MuScriptor transcription, refine grid on actual note onsets, and write score.mid + score_raw.mid."""
    from muscriptor.events import NoteStartEvent
    from muscriptor.transcription_model import TranscriptionModel
    from muscriptor.utils.beats import BeatGrid
    from muscriptor.utils.midi import ExportOptions

    model = TranscriptionModel.load_model(weights_path=weights_path, device=device)

    with tempfile.TemporaryDirectory(prefix="stage1-pad-") as tmpd:
        transcribe_input = audio_path
        if pad_sec > 0.0:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            pad_frames = int(round(pad_sec * sr))
            padded = np.vstack([np.zeros((pad_frames, data.shape[1]), dtype=np.float32), data])
            transcribe_input = Path(tmpd) / "padded_input.wav"
            sf.write(str(transcribe_input), padded, sr, subtype="PCM_24")

        t_sig = inspect.signature(model.transcribe)
        t_kwargs: dict = {}
        if "instruments" in t_sig.parameters and instruments:
            t_kwargs["instruments"] = instruments
        if "prelude_forcing" in t_sig.parameters:
            t_kwargs["prelude_forcing"] = True

        events = list(model.transcribe(str(transcribe_input), **t_kwargs))

    events = _shift_muscriptor_events(events, pad_sec)
    note_onsets = np.asarray([ev.start_time for ev in events if isinstance(ev, NoteStartEvent)], dtype=np.float64)

    grid_info = select_quantization_grid(
        onset_times=note_onsets,
        quarter_bpm=quarter_bpm,
        first_downbeat=first_downbeat,
        beat_type=beat_type,
        grid_override=grid_override,
    )
    chosen_grid = int(grid_info["grid"])

    use_tempo_map = tempo_mode == "tempo-map" and len(beats_array) >= 2
    beat_grid = _build_filtered_dataclass(
        BeatGrid,
        {
            "bpm": float(quarter_bpm),
            "beats_per_bar": int(beats_per_bar_quarter),
            "first_downbeat": float(first_downbeat),
            "beats": np.asarray(beats_array, dtype=np.float64) if len(beats_array) >= 2 else None,
            "beat_subdivision": chosen_grid,
            "steady": not use_tempo_map,
        },
    )

    # 1) Unquantized raw MIDI for listening/debugging
    raw_opts = _build_filtered_dataclass(
        ExportOptions,
        {
            "bpm": None if use_tempo_map else float(quarter_bpm),
            "beats_per_bar": int(beats_per_bar_quarter),
            "first_downbeat": float(first_downbeat),
            "tempo_map": use_tempo_map,
            "quantize": False,
            "trim_leading_bars": False,
        },
    )
    # 2) Quantized baseline MIDI aligned for Stage 2 & Stage 3 (trim_leading_bars preserves Bar 0 pickup)
    quant_opts = _build_filtered_dataclass(
        ExportOptions,
        {
            "bpm": None if use_tempo_map else float(quarter_bpm),
            "beats_per_bar": int(beats_per_bar_quarter),
            "first_downbeat": float(first_downbeat),
            "tempo_map": use_tempo_map,
            "quantize": True,
            "subdivision": chosen_grid,
            "strength": 1.0,
            "fill_gaps": 0.0,
            "trim_leading_bars": True,
        },
    )

    e_sig = inspect.signature(model.events_to_midi_bytes)
    if "export" in e_sig.parameters:
        raw_bytes = model.events_to_midi_bytes(iter(events), beat_grid=beat_grid, export=raw_opts)
        quant_bytes = model.events_to_midi_bytes(iter(events), beat_grid=beat_grid, export=quant_opts)
    else:
        raw_bytes = model.events_to_midi_bytes(iter(events), beat_grid=beat_grid, quantize=False)
        quant_bytes = model.events_to_midi_bytes(iter(events), beat_grid=beat_grid, quantize=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_mid_path = out_dir / "score_raw.mid"
    quant_mid_path = out_dir / "score.mid"
    raw_mid_path.write_bytes(raw_bytes)
    quant_mid_path.write_bytes(quant_bytes)

    return {
        "note_onset_count": int(len(note_onsets)),
        "grid_refined": grid_info,
        "raw_midi": str(raw_mid_path),
        "quantized_midi": str(quant_mid_path),
    }


def export_baseline_musicxml(midi_path: Path, out_xml: Path, musescore_exe: str) -> Path:
    """Convert quantized baseline MIDI to MusicXML via MuseScore 4 headless CLI."""
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if platform.system() == "Linux":
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.run(
        [musescore_exe, "-o", str(out_xml), str(midi_path)],
        capture_output=True,
        text=True,
        errors="replace",
        env=env,
        timeout=180,
    )
    if proc.returncode != 0 or not out_xml.is_file() or out_xml.stat().st_size == 0:
        raise RuntimeError(f"MuseScore failed to export baseline MusicXML: {proc.stderr or proc.stdout}")
    return out_xml


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Stage 1: beat_this + circular BPM/meter/downbeat/grid calibration + MuScriptor baseline MIDI/MusicXML"
    )
    ap.add_argument("--audio", required=True, help="Input solo piano audio (WAV/MP3/FLAC)")
    ap.add_argument("--out-dir", required=True, help="Output directory for calibration JSON, score.mid, score.musicxml")
    ap.add_argument("--triage", default=None, help="Optional Stage 0 audio_triage.json (auto-triggers 2s leading silence guard)")
    ap.add_argument("--config", default=None, help="Optional config.json path")
    ap.add_argument("--engine", default="auto", choices=("auto", "transkun", "muscriptor"), help="Transcription engine (auto prefers MIT TransKun v2, falls back to MuScriptor)")
    ap.add_argument("--model", default=None, help="Optional model weight path/dir/name (TransKun .pt or MuScriptor model dir)")
    ap.add_argument("--instruments", default="acoustic_piano", help="Comma-separated MuScriptor instrument names (empty/'all' disables filter)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    ap.add_argument("--mode", default="auto", choices=("auto", "fixed-tempo", "tempo-map"))
    ap.add_argument("--bpm", type=float, default=None, help="Manual beat-unit BPM override")
    ap.add_argument("--beats-per-bar", type=int, default=None, help="Manual beats-per-bar override (e.g. 3, 4, 6)")
    ap.add_argument("--beat-type", type=int, default=None, choices=(4, 8), help="Manual time-signature denominator (4 or 8)")
    ap.add_argument("--first-downbeat", type=float, default=None, help="Manual Bar 1 downbeat timestamp in seconds")
    ap.add_argument("--grid", type=int, default=None, choices=(2, 3, 4), help="Manual quantization subdivisions per quarter note")
    ap.add_argument("--pad-leading-silence", type=float, default=-1.0, help="Seconds of leading silence before transcription (-1=auto 2.0s if first_note<0.15s, 0=off)")
    ap.add_argument("--musescore", default=None, help="Optional MuseScore 4 binary path")
    ap.add_argument("--analyze-only", action="store_true", help="Run beat_this + calibration only (skip MuScriptor & MusicXML)")
    ap.add_argument("--skip-musicxml", action="store_true", help="Skip MuseScore baseline MusicXML export")
    args = ap.parse_args()

    plugin_root = find_plugin_root()
    cfg = load_config(plugin_root, args.config)
    apply_cache_env(plugin_root, cfg)

    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    mono, sr = load_audio_mono(audio_path)
    dur_s = len(mono) / float(sr)

    feats = extract_onset_features(mono, sr)
    first_note_sec = float(feats["first_note_sec"])
    if args.triage and Path(args.triage).is_file():
        triage = json.loads(Path(args.triage).read_text(encoding="utf-8"))
        rec_name = triage.get("recommended_filename")
        cand = next((c for c in triage.get("candidates", []) if c.get("filename") == rec_name), None)
        if cand and "first_note_sec" in cand:
            first_note_sec = float(cand["first_note_sec"])

    pad_sec = (2.0 if first_note_sec < 0.15 else 0.0) if args.pad_leading_silence < 0 else max(0.0, float(args.pad_leading_silence))

    print(f"[Stage 1] Running beat_this on {audio_path.name} (sr={sr}, dur={dur_s:.2f}s, device={device})...")
    bt_info = run_beat_this(mono, sr, device=device)

    bpm_info = calibrate_bpm_and_mode(
        times=feats["times"],
        full_flux=feats["full_flux"],
        dur_s=dur_s,
        bt_info=bt_info,
        bpm_override=args.bpm,
        mode_arg=args.mode,
    )
    meter_info = calibrate_meter_and_downbeat(
        times=feats["times"],
        full_flux=feats["full_flux"],
        bass_flux=feats["bass_flux"],
        chroma_flux=feats["chroma_flux"],
        dur_s=dur_s,
        first_note_sec=first_note_sec,
        bpm_info=bpm_info,
        bt_info=bt_info,
        bpb_override=args.beats_per_bar,
        beat_type_override=args.beat_type,
        downbeat_override=args.first_downbeat,
    )

    sel_bpb = int(meter_info["beats_per_bar"])
    sel_btype = int(meter_info["beat_type"])
    sel_bpm = float(meter_info["bpm_for_meter"])
    first_downbeat = float(meter_info["first_downbeat"])
    quarter_bpm = sel_bpm * (4.0 / sel_btype)
    beats_per_bar_quarter = max(1, int(round((sel_bpb * 4) / float(sel_btype))))

    quarter_beats, beat_resample_ratio = resample_beats_to_quarter_pulse(
        beats=bt_info["beats"],
        tracked_bpm=float(bt_info["beat_this_bpm"]),
        quarter_bpm=quarter_bpm,
        first_downbeat=first_downbeat,
    )

    # Pre-transcription onset-flux grid estimate
    flux_peaks = detect_onset_peak_times(feats["times"], feats["full_flux"])
    grid_info = select_quantization_grid(
        onset_times=flux_peaks,
        quarter_bpm=quarter_bpm,
        first_downbeat=first_downbeat,
        beat_type=sel_btype,
        grid_override=args.grid,
    )

    report: dict = {
        "stage": "stage_1_tempo_meter_grid",
        "audio": str(audio_path),
        "sample_rate": sr,
        "duration_sec": round(dur_s, 3),
        "first_note_sec": round(first_note_sec, 3),
        "leading_silence_pad_sec": pad_sec,
        "device": device,
        "beat_this": {
            "raw_beat_count": bt_info["raw_beat_count"],
            "regularized_beat_count": bt_info["regularized_beat_count"],
            "downbeat_count": bt_info["downbeat_count"],
            "beat_this_bpm": bt_info["beat_this_bpm"],
            "beat_residual_ms": bt_info["beat_residual_ms"],
            "beat_this_bpb": bt_info["beat_this_bpb"],
        },
        "calibration": {
            "bpm": sel_bpm,
            "quarter_bpm": round(quarter_bpm, 3),
            "beats_per_bar": sel_bpb,
            "beat_type": sel_btype,
            "first_downbeat": first_downbeat,
            "pickup_sec": meter_info["pickup_sec"],
            "has_pickup": bool(meter_info["pickup_sec"] > 0.05),
            "bar1_offset_quarters": 0.0,
            "circular_R": bpm_info["circular_R"],
            "max_abs_drift_ms": bpm_info["max_abs_drift_ms"],
            "tempo_mode": bpm_info["tempo_mode"],
            "beat_resample_ratio": beat_resample_ratio,
            "quarter_beat_count": int(len(quarter_beats)),
            "grid": grid_info["grid"],
            "grid_pre_transcription": grid_info,
            "bpm_candidates": bpm_info["candidates"],
            "meter_scores": meter_info["meter_scores"],
        },
        "tempo_map": {
            "mode": bpm_info["tempo_mode"],
            "first_downbeat": first_downbeat,
            "quarter_bpm": round(quarter_bpm, 3),
            "beats_per_bar": sel_bpb,
            "beat_type": sel_btype,
            "quarter_beats": [round(float(t), 4) for t in quarter_beats],
        },
        "outputs": {},
    }

    if not args.analyze_only:
        chosen_engine = resolve_transcription_engine(args.engine)
        report["transcription_engine"] = chosen_engine
        if chosen_engine == "transkun":
            print(
                f"[Stage 1] Running TransKun v2 (MIT, zero-HF-login, device={device}, "
                f"pad={pad_sec:.1f}s, mode={bpm_info['tempo_mode']})..."
            )
            raw_mid_path = out_dir / "score_raw.mid"
            quant_mid_path = out_dir / "score.mid"
            tk_notes = run_transkun_transcribe(
                audio_path=audio_path,
                out_raw_mid=raw_mid_path,
                device=device,
                pad_sec=pad_sec,
                weight_path=args.model,
            )
            note_onsets = np.asarray([n["start_sec"] for n in tk_notes], dtype=np.float64)
            # P0-2 guard: never leave notes more than 1 bar before first_downbeat
            if len(note_onsets) > 0:
                min_note_t = float(np.min(note_onsets))
                bar_s_approx = beats_per_bar_quarter * (60.0 / max(1e-6, quarter_bpm))
                if min_note_t < first_downbeat - bar_s_approx * 1.05:
                    shift_bars = int(math.floor((first_downbeat - min_note_t) / bar_s_approx))
                    new_fd = max(0.0, first_downbeat - shift_bars * bar_s_approx)
                    if len(quarter_beats) >= 2:
                        k_new = int(np.argmin(np.abs(quarter_beats - new_fd)))
                        new_fd = float(quarter_beats[k_new])
                    first_downbeat = round(new_fd, 4)
                    report["calibration"]["first_downbeat"] = first_downbeat
                    report["tempo_map"]["first_downbeat"] = first_downbeat
            k0 = int(np.argmin(np.abs(quarter_beats - first_downbeat))) if len(quarter_beats) else 0
            q_arr = (np.arange(len(quarter_beats), dtype=np.float64) - float(k0)) if len(quarter_beats) >= 2 else None
            tmap_s1 = TempoMap(
                bpm=sel_bpm,
                beats_per_bar=sel_bpb,
                beat_type=sel_btype,
                first_downbeat=float(quarter_beats[k0]) if len(quarter_beats) >= 2 else first_downbeat,
                mode=bpm_info["tempo_mode"],
                q_grid=q_arr,
                t_grid=quarter_beats if len(quarter_beats) >= 2 else None,
            )
            grid_refined = select_quantization_grid(
                onset_times=note_onsets,
                quarter_bpm=quarter_bpm,
                first_downbeat=first_downbeat,
                beat_type=sel_btype,
                grid_override=args.grid,
                tempo_map=tmap_s1,
            )
            q_res = quantize_performance_midi(
                note_events=tk_notes,
                tempo_map=tmap_s1,
                grid=int(grid_refined["grid"]),
                beats_per_bar=sel_bpb,
                beat_type=sel_btype,
                out_mid=quant_mid_path,
                trim_leading_bars=True,
            )
            report["calibration"]["has_pickup"] = bool(q_res["has_pickup"])
            report["calibration"]["bar1_offset_quarters"] = float(q_res.get("bar1_offset_quarters", 0.0))
            tr_out = {
                "note_onset_count": int(len(note_onsets)),
                "grid_refined": grid_refined,
                "raw_midi": str(raw_mid_path),
                "quantized_midi": str(quant_mid_path),
            }
        else:
            inst_raw = (args.instruments or "").strip()
            instruments = None if (not inst_raw or inst_raw.lower() == "all") else [x.strip() for x in inst_raw.split(",") if x.strip()]
            weights_path = resolve_muscriptor_weights(plugin_root, cfg, args.model)
            print(
                f"[Stage 1] Running MuScriptor (weights={weights_path or 'HF default'}, "
                f"instruments={instruments or 'auto'}, pad={pad_sec:.1f}s, mode={bpm_info['tempo_mode']})..."
            )
            tr_out = transcribe_with_muscriptor(
                audio_path=audio_path,
                weights_path=weights_path,
                device=device,
                instruments=instruments,
                pad_sec=pad_sec,
                quarter_bpm=quarter_bpm,
                beats_per_bar_quarter=beats_per_bar_quarter,
                first_downbeat=first_downbeat,
                beats_array=quarter_beats,
                tempo_mode=bpm_info["tempo_mode"],
                grid_override=args.grid,
                beat_type=sel_btype,
                out_dir=out_dir,
            )
        report["calibration"]["grid"] = tr_out["grid_refined"]["grid"]
        report["calibration"]["grid_post_transcription"] = tr_out["grid_refined"]
        report["calibration"]["transcribed_note_onsets"] = tr_out["note_onset_count"]
        report["outputs"]["score_raw_mid"] = tr_out["raw_midi"]
        report["outputs"]["score_mid"] = tr_out["quantized_midi"]

        if not args.skip_musicxml:
            ms = find_musescore(args.musescore, cfg)
            if not ms:
                raise FileNotFoundError(
                    "MuseScore 4 binary not found for baseline MusicXML export; "
                    "pass --musescore, set MUSCRIPTOR_MUSESCORE, or pass --skip-musicxml."
                )
            xml_path = export_baseline_musicxml(Path(tr_out["quantized_midi"]), out_dir / "score.musicxml", ms)
            report["outputs"]["score_musicxml"] = str(xml_path)

    final_grid = report["calibration"]["grid"]
    cal_path = out_dir / "stage1_calibration.json"
    baseline_xml = out_dir / "score.musicxml"
    baseline_mid = out_dir / "score.mid"
    pitch_audit_out = out_dir.parent / "analysis" / "pitch_audit.json"
    study_xml = out_dir.parent / "score" / "score_learning.musicxml"
    study_pdf = out_dir.parent / "score" / "score_learning.pdf"
    report["next_commands"] = {
        "stage2_pitch_audit": (
            f'python skills/piano-score-studio/scripts/stage2_pitch_audit.py '
            f'--audio "{audio_path}" --xml "{baseline_xml}" '
            f'--bpm {sel_bpm} --beats-per-bar {sel_bpb} --beat-type {sel_btype} '
            f'--first-downbeat {first_downbeat} --stage1-cal "{cal_path}" '
            f'--out "{pitch_audit_out}"'
        ),
        "stage3_engrave": (
            f'python skills/piano-score-studio/scripts/stage3_engrave_study_score.py '
            f'--midi "{baseline_mid}" --audio "{audio_path}" '
            f'--bpm {sel_bpm} --beats-per-bar {sel_bpb} --beat-type {sel_btype} '
            f'--grid {final_grid} --first-downbeat {first_downbeat} --stage1-cal "{cal_path}" '
            f'--out-xml "{study_xml}" --out-pdf "{study_pdf}"'
        ),
    }
    cal_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Stage 1 Complete: BPM={sel_bpm} ({sel_bpb}/{sel_btype}), downbeat={first_downbeat:.3f}s, "
        f"|R|={bpm_info['circular_R']:.3f}, drift={bpm_info['max_abs_drift_ms']:.1f}ms "
        f"({bpm_info['tempo_mode']}), grid={final_grid} -> {cal_path}"
    )


if __name__ == "__main__":
    main()
