#!/usr/bin/env python3
"""v3.0 Phase 3 — Chroma-Flux Dynamic Pedaling & TransKun Real Velocity Fusion.

Replaces blind per-bar CC64=127 pedaling and purely acoustic-envelope velocity estimation with:
  1. Style-aware, chroma-flux-driven sustain pedaling:
       - dry_articulated (Baroque):    No blind sustain pedal (CC64=0 only)
       - rhythmic_half_pedal (Jazz):   Half-pedal (CC64=68) on harmonic changes only
       - legato_harmonic (Pop Ballad): Full syncopated pedal changes on bar & mid-bar chord changes
       - romantic_rubato (Classical):  Sensitive chroma-flux syncopated pedal changes
  2. TransKun v2 real MIDI velocity fusion:
       - Matches engraved MusicXML notes to nearest TransKun performance-MIDI notes (pitch + time)
       - Blends real TransKun touch velocity with Stage 4 acoustic-calibrated velocity
       - Automatically skips blending if source MIDI has flat placeholder velocities (e.g. MuScriptor)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _chroma_vector(pitches: Sequence[int]) -> np.ndarray:
    v = np.zeros(12, dtype=np.float64)
    for p in pitches:
        v[int(p) % 12] += 1.0
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        v /= norm
    return v


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, 1.0 - sim))


def compute_dynamic_pedal_events(
    notes: Sequence[dict],
    tempo_map,
    max_bar: int,
    rate: int,
    pedal_style: str = "legato_harmonic",
    beats_per_bar: int = 4,
    beat_type: int = 4,
) -> list[dict]:
    """Generate style-aware, chroma-flux-driven CC64 sustain pedal controller events."""
    _, t_end = tempo_map.bar_bounds_sec(max(1, max_bar))
    if pedal_style == "dry_articulated":
        # Baroque / Bach: clean dry articulation, zero sustain pedal wash
        return [{"sample": 0, "channel": 1, "controller": 64, "value": 0}]

    if pedal_style == "rhythmic_half_pedal":
        pedal_depth = 68
        flux_thresh = 0.48
        always_on_bar = False
    elif pedal_style == "romantic_rubato":
        pedal_depth = 127
        flux_thresh = 0.36
        always_on_bar = True
    else:  # legato_harmonic (default)
        pedal_depth = 127
        flux_thresh = 0.42
        always_on_bar = True

    # Group notes into beat buckets per bar to detect harmonic changes via chroma flux
    bar_q = beats_per_bar * (4.0 / beat_type)
    n_sub = max(1, int(round(bar_q)))  # quarter-note harmonic check steps per bar
    controllers: list[dict] = []
    prev_chroma = np.zeros(12, dtype=np.float64)
    pedal_is_down = False

    start_bar = 0 if any(int(n.get("measure", 1)) == 0 for n in notes) else 1
    for m in range(start_bar, max_bar + 1):
        tb_raw, te = tempo_map.bar_bounds_sec(m)
        tb = max(0.0, tb_raw) if m == 0 else tb_raw
        bar_dur = max(0.05, te - tb)
        step_dur = bar_dur / float(n_sub)

        for k in range(n_sub):
            t0 = tb + k * step_dur
            t1 = t0 + step_dur
            step_notes = [n for n in notes if (t0 - 0.03) <= float(n["st"]) < (t1 - 0.02)]
            if not step_notes:
                continue
            cur_chroma = _chroma_vector([int(n["pitch"]) for n in step_notes])
            flux = _cosine_distance(prev_chroma, cur_chroma) if float(prev_chroma.sum()) > 0 else 1.0

            is_bar_start = (k == 0)
            should_change = (is_bar_start and always_on_bar) or (flux >= flux_thresh)

            if should_change:
                if pedal_is_down or is_bar_start:
                    controllers.append(
                        {"sample": max(0, int(round((t0 - 0.018) * rate))), "channel": 1, "controller": 64, "value": 0}
                    )
                # Slightly reduce depth on very dense low-register clusters to avoid bass mud
                low_count = sum(1 for n in step_notes if int(n["pitch"]) < 52)
                depth = min(pedal_depth, 96) if (low_count >= 3 and pedal_depth > 96) else pedal_depth
                controllers.append(
                    {"sample": max(0, int(round((t0 + 0.028) * rate))), "channel": 1, "controller": 64, "value": depth}
                )
                pedal_is_down = True
                prev_chroma = cur_chroma
            else:
                # Blend sustained chroma smoothly
                prev_chroma = _chroma_vector(
                    [int(n["pitch"]) for n in step_notes]
                ) if float(prev_chroma.sum()) <= 0 else (0.7 * prev_chroma + 0.3 * cur_chroma)

    controllers.append({"sample": max(0, int(round((t_end + 0.10) * rate))), "channel": 1, "controller": 64, "value": 0})
    return controllers


def read_midi_pedal_events(midi_path: Path, rate: int = 48000) -> list[dict]:
    """Read real CC64 sustain pedal events from a performance MIDI (e.g. TransKun score_raw.mid).

    Returns an empty list if the source MIDI contains no active sustain pedal presses (value >= 32).
    """
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = max(1, int(mid.ticks_per_beat))
    cur_t = 0.0
    cur_tempo = 500000
    events: list[dict] = []
    has_press = False
    for msg in mido.merge_tracks(mid.tracks):
        cur_t += mido.tick2second(msg.time, tpb, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "control_change" and msg.control == 64:
            val = int(msg.value)
            if val >= 32:
                has_press = True
            events.append(
                {
                    "sample": max(0, int(round(cur_t * rate))),
                    "channel": 1,
                    "controller": 64,
                    "value": val,
                }
            )
    return events if has_press else []


def read_midi_velocity_events(midi_path: Path) -> list[tuple[float, int, int]]:
    """Read (start_sec, pitch, velocity) tuples from a MIDI file using its tempo map."""
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = max(1, int(mid.ticks_per_beat))
    cur_t = 0.0
    cur_tempo = 500000
    events: list[tuple[float, int, int]] = []
    for msg in mido.merge_tracks(mid.tracks):
        cur_t += mido.tick2second(msg.time, tpb, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            events.append((cur_t, int(msg.note), int(msg.velocity)))
    return events


def fuse_transkun_velocities(
    notes: Sequence[dict],
    source_vel_events: Sequence[tuple[float, int, int]],
    alpha: float = 0.55,
    tol_sec: float = 0.14,
    lh_ceiling_vel: int = 90,
) -> int:
    """Blend real TransKun MIDI velocities into Stage 4 notes in-place.

    Skips blending if source velocities are flat/placeholder (std < 2.0).
    Returns number of notes successfully fused.
    """
    if not notes or not source_vel_events:
        return 0
    raw_vels = np.asarray([v for _, _, v in source_vel_events], dtype=np.float64)
    if len(raw_vels) < 3 or float(raw_vels.std()) < 2.0:
        return 0

    by_pitch: dict[int, list[tuple[float, int]]] = {}
    for t_sec, pitch, vel in source_vel_events:
        by_pitch.setdefault(int(pitch), []).append((float(t_sec), int(vel)))

    fused = 0
    for n in notes:
        cands = by_pitch.get(int(n["pitch"]))
        if not cands:
            continue
        t_target = float(n["st"])
        best_vel = None
        best_err = tol_sec + 1.0
        for t_s, v_s in cands:
            err = abs(t_s - t_target)
            if err <= tol_sec and err < best_err:
                best_err = err
                best_vel = v_s
        if best_vel is not None:
            base_v = float(n.get("calc_vel", 64))
            blended = alpha * float(best_vel) + (1.0 - alpha) * base_v
            if int(n.get("staff", 1)) == 2:
                blended = min(blended, float(lh_ceiling_vel))
            n["calc_vel"] = int(np.clip(round(blended), 12, 118))
            n["vel_fused_transkun"] = True
            fused += 1
    return fused


def align_microtiming_to_transkun(
    notes: Sequence[dict],
    source_vel_events: Sequence[tuple[float, int, int]],
    blend: float = 0.80,
    max_shift_sec: float = 0.10,
) -> int:
    """Gently shift rendered note onsets toward TransKun performance micro-timing.

    Keeps the MusicXML score cleanly quantized while restoring natural human phrasing,
    subtle chord rolls, and rubato breathing in the rendered audio.
    """
    if not notes or not source_vel_events or blend <= 0.0:
        return 0

    by_pitch: dict[int, list[list]] = {}
    for t_sec, pitch, _ in source_vel_events:
        by_pitch.setdefault(int(pitch), []).append([float(t_sec), False])

    aligned = 0
    for n in notes:
        cands = by_pitch.get(int(n["pitch"]))
        if not cands:
            continue
        t_target = float(n["st"])
        best_idx = -1
        best_err = max_shift_sec + 1.0
        for idx, item in enumerate(cands):
            if item[1]:
                continue
            err = abs(item[0] - t_target)
            if err <= max_shift_sec and err < best_err:
                best_err = err
                best_idx = idx
        if best_idx >= 0:
            cands[best_idx][1] = True
            delta = blend * (cands[best_idx][0] - t_target)
            new_st = max(0.0, t_target + delta)
            dur = max(0.08, float(n["et"]) - t_target)
            n["st"] = new_st
            n["et"] = new_st + dur
            n["timing_aligned_transkun"] = True
            aligned += 1
    return aligned
