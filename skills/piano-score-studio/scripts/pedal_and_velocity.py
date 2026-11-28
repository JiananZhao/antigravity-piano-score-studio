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
    release_t: float | None = None,
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

    final_rel = max(t_end + 0.10, float(release_t)) if release_t is not None else (t_end + 0.10)
    controllers.append({"sample": max(0, int(round(final_rel * rate))), "channel": 1, "controller": 64, "value": 0})
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


def read_midi_soft_pedal_events(midi_path: Path, rate: int = 48000) -> list[dict]:
    """Read raw CC67 (una corda / soft pedal) controller events from a performance MIDI."""
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = max(1, int(mid.ticks_per_beat))
    cur_t = 0.0
    cur_tempo = 500000
    events: list[dict] = []
    for msg in mido.merge_tracks(mid.tracks):
        cur_t += mido.tick2second(msg.time, tpb, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "control_change" and msg.control == 67:
            events.append(
                {
                    "sample": max(0, int(round(cur_t * rate))),
                    "channel": 1,
                    "controller": 67,
                    "value": int(msg.value),
                }
            )
    return events


def read_midi_soft_pedal_spans(midi_path: Path) -> list[tuple[float, float]]:
    """Read CC67 (soft pedal) active intervals in seconds (`value >= 64` = pressed).

    Any span still open at end-of-track is closed at the timestamp of the last MIDI event.
    """
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = max(1, int(mid.ticks_per_beat))
    cur_t = 0.0
    cur_tempo = 500000
    spans: list[tuple[float, float]] = []
    down_t: float | None = None
    for msg in mido.merge_tracks(mid.tracks):
        cur_t += mido.tick2second(msg.time, tpb, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "control_change" and msg.control == 67:
            if int(msg.value) >= 64:
                if down_t is None:
                    down_t = cur_t
            else:
                if down_t is not None:
                    if cur_t > down_t:
                        spans.append((down_t, cur_t))
                    down_t = None
    if down_t is not None and cur_t > down_t:
        spans.append((down_t, cur_t))
    return spans


def read_midi_velocity_events(midi_path: Path) -> list[tuple[float, int, int]]:
    """Read (start_sec, pitch, velocity) tuples from a MIDI file using its tempo map."""
    return [(st, p, v) for st, _, p, v in read_midi_performance_events(midi_path)]


def read_midi_performance_events(midi_path: Path) -> list[tuple[float, float, int, int]]:
    """Read (start_sec, end_sec, pitch, velocity) tuples from a MIDI file using its tempo map."""
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpb = max(1, int(mid.ticks_per_beat))
    cur_t = 0.0
    cur_tempo = 500000
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    events: list[tuple[float, float, int, int]] = []
    for msg in mido.merge_tracks(mid.tracks):
        cur_t += mido.tick2second(msg.time, tpb, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            key = (getattr(msg, "channel", 0), int(msg.note))
            active.setdefault(key, []).append((cur_t, int(msg.velocity)))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (getattr(msg, "channel", 0), int(msg.note))
            stk = active.get(key)
            if stk:
                st_s, vel = stk.pop(0)
                events.append((st_s, max(st_s + 0.04, cur_t), int(msg.note), vel))
    for (_, pitch), stk in active.items():
        for st_s, vel in stk:
            events.append((st_s, st_s + 0.25, int(pitch), vel))
    events.sort(key=lambda x: (x[0], x[2]))
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
    source_events: Sequence[tuple],
    blend: float = 0.80,
    max_shift_sec: float = 0.22,
) -> int:
    """Shift rendered note onsets (and restore real sustain durations when 4-tuples are given)
    toward TransKun performance timing.

    Accepts either 3-tuples `(start_sec, pitch, velocity)` or 4-tuples
    `(start_sec, end_sec, pitch, velocity)`. When `end_sec` is present, restores
    finger-hold durations that were shortened during study-score chord simplification,
    clamped safely before the next onset of the same pitch.
    """
    if not notes or not source_events or blend <= 0.0:
        return 0

    by_pitch: dict[int, list[list]] = {}
    for ev in source_events:
        if len(ev) >= 4:
            st_s, et_s, pitch = float(ev[0]), float(ev[1]), int(ev[2])
        else:
            st_s, et_s, pitch = float(ev[0]), None, int(ev[1])
        by_pitch.setdefault(pitch, []).append([st_s, et_s, False])

    aligned = 0
    for n in notes:
        cands = by_pitch.get(int(n["pitch"]))
        if not cands:
            continue
        t_target = float(n["st"])
        best_idx = -1
        best_err = max_shift_sec + 1.0
        for idx, item in enumerate(cands):
            if item[2]:
                continue
            err = abs(item[0] - t_target)
            if err <= max_shift_sec and err < best_err:
                best_err = err
                best_idx = idx
        if best_idx >= 0:
            raw_st, raw_et, _ = cands[best_idx]
            cands[best_idx][2] = True
            delta = blend * (raw_st - t_target)
            new_st = max(0.0, t_target + delta)
            score_dur = max(0.08, float(n["et"]) - t_target)
            if raw_et is not None:
                raw_dur = max(0.08, raw_et - raw_st)
                eff_dur = max(score_dur, raw_dur)
            else:
                eff_dur = score_dur
            n["st"] = new_st
            n["et"] = new_st + eff_dur
            n["timing_aligned_transkun"] = True
            aligned += 1

    # Prevent same-pitch overlaps after duration restoration
    same_pitch: dict[int, list[dict]] = {}
    for n in notes:
        same_pitch.setdefault(int(n["pitch"]), []).append(n)
    for grp in same_pitch.values():
        grp.sort(key=lambda x: x["st"])
        for i in range(len(grp) - 1):
            nxt_st = float(grp[i + 1]["st"])
            if float(grp[i]["et"]) > nxt_st - 0.015:
                grp[i]["et"] = max(float(grp[i]["st"]) + 0.05, nxt_st - 0.015)

    return aligned


def calibrate_chord_voicing_from_audio(
    notes: Sequence[dict],
    mono_orig: np.ndarray,
    sr_orig: int,
    cluster_sec: float = 0.045,
) -> int:
    """Attenuate inner/background chord tones whose real acoustic level in `mono_orig`
    is markedly softer than the lead tone in the same simultaneous chord.

    Compensates for compressed MIDI velocity predictions and Salamander Grand Piano's
    shallow velocity-to-dB curve so soft harmonic/inner notes never stick out harshly.
    """
    if not notes or len(mono_orig) < 256:
        return 0

    def _note_onset_db(t_on: float, pitch: int) -> float:
        f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        i0 = max(0, int((t_on - 0.005) * sr_orig))
        i1 = min(len(mono_orig), int((t_on + 0.140) * sr_orig))
        ch = mono_orig[i0:i1]
        if len(ch) < 128:
            return -80.0
        n_fft = 16384
        sp = np.abs(np.fft.rfft(ch * np.hanning(len(ch)), n=n_fft)) / (len(ch) + 1e-12)
        fs = np.fft.rfftfreq(n_fft, 1.0 / sr_orig)
        bw = max(f0 * 0.022, 3.5)
        m1 = (fs >= f0 - bw) & (fs <= f0 + bw)
        db1 = 20.0 * np.log10(float(np.max(sp[m1])) + 1e-12) if np.any(m1) else -80.0
        f2 = 2.0 * f0
        if f2 < sr_orig * 0.48:
            m2 = (fs >= f2 - 2.0 * bw) & (fs <= f2 + 2.0 * bw)
            db2 = 20.0 * np.log10(float(np.max(sp[m2])) + 1e-12) if np.any(m2) else -80.0
            return max(db1, db2 - 4.0)
        return db1

    adjusted = 0
    for staff_id in (1, 2):
        st_notes = sorted(
            [n for n in notes if int(n.get("staff", 1)) == staff_id and int(n["pitch"]) >= 48],
            key=lambda x: (x["st"], x["pitch"]),
        )
        clusters: list[list[dict]] = []
        for n in st_notes:
            if not clusters or (float(n["st"]) - float(clusters[-1][0]["st"])) > cluster_sec:
                clusters.append([n])
            else:
                clusters[-1].append(n)
        for cl in clusters:
            if len(cl) < 2:
                continue
            dbs = [_note_onset_db(float(x["st"]), int(x["pitch"])) for x in cl]
            lead_idx = int(np.argmax(dbs))
            max_db = dbs[lead_idx]
            if max_db < -65.0:
                continue
            ref_vel = max(int(cl[lead_idx].get("calc_vel", 64)), max(int(x.get("calc_vel", 64)) for x in cl))
            for x, ndb in zip(cl, dbs):
                rel_db = ndb - max_db
                if rel_db < -4.0:
                    ceil_vel = int(np.clip(round(ref_vel * (10.0 ** (rel_db / 26.0))), 8, 118))
                    cur_v = int(x.get("calc_vel", 64))
                    if ceil_vel < cur_v:
                        x["calc_vel"] = ceil_vel
                        x["voicing_calibrated"] = True
                        adjusted += 1
    return adjusted


def load_confirmed_ghosts_from_audit(pitch_audit_path: Path | None) -> list[tuple[float, int]]:
    """Return `(t_on, midi_pitch)` pairs marked `CONFIRMED_GHOST` in Stage 2 pitch_audit.json."""
    if pitch_audit_path is None or not pitch_audit_path.is_file():
        return []
    import json

    try:
        data = json.loads(pitch_audit_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    ghosts: list[tuple[float, int]] = []
    for item in data.get("minor2nd_clashes", []):
        if item.get("verdict") == "CONFIRMED_GHOST":
            try:
                ghosts.append((float(item["t_on"]), int(item["weaker_midi"])))
            except (KeyError, TypeError, ValueError):
                continue
    return ghosts


def build_performance_notes_from_raw_midi(
    raw_events: Sequence[tuple[float, float, int, int]],
    xml_notes: Sequence[dict],
    tempo_map=None,
    confirmed_ghosts: Sequence[tuple[float, int]] = (),
    split_pitch: int = 60,
    max_bar: int = 1,
    rh_cap: int = 94,
    lh_cap: int = 90,
) -> tuple[list[dict], int]:
    """Build Stage 4 performance notes directly from raw TransKun MIDI (`score_raw.mid`).

    Decouples audio/performance MIDI rendering from Study Score quantization so rubato
    arpeggios and finger-hold durations never suffer grid-snapping collisions or truncation.
    Borrows only `staff` (RH/LH) and `measure` metadata from `xml_notes` without altering
    `st`, `et`, or `calc_vel`.

    Returns `(perf_notes, removed_ghost_count)`.
    """
    if not raw_events:
        return [], 0

    # 1. Remove Stage 2 confirmed ghost notes (1-to-1 closest match within 0.18s)
    kept: list[tuple[float, float, int, int]] = list(raw_events)
    removed_ghosts = 0
    for g_time, g_pitch in confirmed_ghosts:
        best_idx = -1
        best_err = 0.181
        for idx, (st_s, _, p_s, _) in enumerate(kept):
            if int(p_s) != int(g_pitch):
                continue
            err = abs(float(st_s) - float(g_time))
            if err < best_err:
                best_err = err
                best_idx = idx
        if best_idx >= 0:
            kept.pop(best_idx)
            removed_ghosts += 1

    # 2. Index XML notes by pitch solely to borrow engraved staff & measure labels
    xml_by_pitch: dict[int, list[list]] = {}
    for xn in xml_notes:
        xml_by_pitch.setdefault(int(xn["pitch"]), []).append(
            [float(xn["st"]), int(xn.get("staff", 1)), int(xn.get("measure", 1)), False]
        )

    bar_bounds: list[tuple[int, float, float]] = []
    if tempo_map is not None:
        for m in range(0, max(1, max_bar) + 1):
            try:
                t0, t1 = tempo_map.bar_bounds_sec(m)
                bar_bounds.append((m, float(t0), float(t1)))
            except Exception:
                break

    def _infer_measure(t_sec: float) -> int:
        for m, t0, t1 in bar_bounds:
            if t0 - 0.05 <= t_sec < t1:
                return m
        return bar_bounds[-1][0] if bar_bounds and t_sec >= bar_bounds[-1][1] else 1

    perf_notes: list[dict] = []
    for st_s, et_s, pitch, vel in kept:
        p = int(pitch)
        st = max(0.0, float(st_s))
        et = max(st + 0.05, float(et_s))
        staff = 1 if p >= split_pitch else 2
        measure = _infer_measure(st)

        cands = xml_by_pitch.get(p)
        if cands:
            best_i = -1
            best_err = 0.28
            for idx, item in enumerate(cands):
                if item[3]:
                    continue
                err = abs(item[0] - st)
                if err < best_err:
                    best_err = err
                    best_i = idx
            if best_i >= 0:
                cands[best_i][3] = True
                staff = cands[best_i][1]
                measure = cands[best_i][2]

        perf_notes.append(
            {
                "measure": measure,
                "st": st,
                "et": et,
                "pitch": p,
                "staff": staff,
                "arpeggiate": False,
                "calc_vel": int(np.clip(int(vel), 10, rh_cap if staff == 1 else lh_cap)),
                "timing_aligned_transkun": True,
                "vel_fused_transkun": True,
                "source_raw_midi": True,
            }
        )

    # Prevent same-pitch overlaps cleanly without truncating real sustains
    same_pitch: dict[int, list[dict]] = {}
    for n in perf_notes:
        same_pitch.setdefault(int(n["pitch"]), []).append(n)
    for grp in same_pitch.values():
        grp.sort(key=lambda x: x["st"])
        for i in range(len(grp) - 1):
            nxt_st = float(grp[i + 1]["st"])
            if float(grp[i]["et"]) > nxt_st - 0.012:
                grp[i]["et"] = max(float(grp[i]["st"]) + 0.04, nxt_st - 0.012)

    perf_notes.sort(key=lambda x: (x["st"], x["staff"], x["pitch"]))
    return perf_notes, removed_ghosts


def closed_loop_velocity_calibration(
    notes: Sequence[dict],
    orig: np.ndarray,
    sr_orig: int,
    rend: np.ndarray,
    sr_rend: int,
    deadband_db: float = 2.0,
    damping: float = 0.72,
    max_corr_db: float = 6.0,
    vel_db_per_decade: float = 30.0,
    cluster_sec: float = 0.030,
    rh_cap: int = 94,
    lh_cap: int = 90,
    min_vel: int = 10,
    max_rise_vel: int = 16,
) -> tuple[int, float]:
    """Closed-loop Render-Measure-Correct per-note velocity calibration.

    Accepts 1D mono or 2D `(N, C)` stereo arrays for `orig` and `rend`.
    Computes per-channel windowed FFT power spectra and averages POWER across
    channels (never summing L+R waveforms before FFT), preventing stereo AB-mic
    comb-filter nulls (e.g. 3 ms delay cancelling ~175 Hz / ~525 Hz) from causing
    false velocity boosts.

    Returns `(adjusted_note_count, median_abs_error_db)`.
    """
    orig_2d = np.asarray(orig, dtype=np.float64)
    rend_2d = np.asarray(rend, dtype=np.float64)
    if orig_2d.ndim == 1:
        orig_2d = orig_2d[:, None]
    if rend_2d.ndim == 1:
        rend_2d = rend_2d[:, None]

    if not notes or orig_2d.shape[0] < 512 or rend_2d.shape[0] < 512:
        return 0, 0.0

    # Global active-region RMS offset using channel-averaged RMS^2 (no waveform folding)
    t_first = max(0.0, min(float(n["st"]) for n in notes))
    t_last = min(
        orig_2d.shape[0] / float(sr_orig),
        rend_2d.shape[0] / float(sr_rend),
        max(float(n["et"]) for n in notes) + 0.5,
    )
    if t_last <= t_first + 0.2:
        return 0, 0.0

    o_seg = orig_2d[int(t_first * sr_orig) : int(t_last * sr_orig)]
    r_seg = rend_2d[int(t_first * sr_rend) : int(t_last * sr_rend)]
    rms2_o = float(np.mean(o_seg ** 2)) if len(o_seg) else 0.0
    rms2_r = float(np.mean(r_seg ** 2)) if len(r_seg) else 0.0
    if rms2_o < 1e-12 or rms2_r < 1e-12:
        return 0, 0.0
    global_diff_db = 10.0 * float(np.log10(rms2_r / rms2_o))

    n_fft = 16384
    fs_o = np.fft.rfftfreq(n_fft, 1.0 / sr_orig)
    fs_r = np.fft.rfftfreq(n_fft, 1.0 / sr_rend)

    def _spec_at(sig_2d: np.ndarray, sr: int, t0: float, dur: float) -> np.ndarray | None:
        i0 = max(0, int(round(t0 * sr)))
        i1 = min(sig_2d.shape[0], int(round((t0 + dur) * sr)))
        seg = sig_2d[i0:i1]
        if seg.shape[0] < 128:
            return None
        w = np.hanning(seg.shape[0])
        w_sum = float(np.sum(w)) * 0.5 + 1e-12
        n_ch = seg.shape[1]
        acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
        for ch in range(n_ch):
            acc += (np.abs(np.fft.rfft(seg[:, ch] * w, n=n_fft)) / w_sum) ** 2
        return acc / float(n_ch)

    def _pitch_db(power_spec: np.ndarray, fs: np.ndarray, sr: int, pitch: int, chord_pitches: set[int]) -> float:
        f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        if pitch >= 52:
            harm_weights = ((1, 1.0),) if (pitch + 12) in chord_pitches else ((1, 1.0), (2, 0.35))
        else:
            harm_weights = ((1, 1.0), (2, 1.0), (3, 0.75))
        tot = 0.0
        for h, w_h in harm_weights:
            ft = f0 * h
            if ft >= sr * 0.48:
                continue
            bw = max(ft * 0.015, 2.5)
            m = (fs >= ft - bw) & (fs <= ft + bw)
            if np.any(m):
                tot += w_h * float(np.max(power_spec[m]))
        return 10.0 * float(np.log10(max(tot, 1e-12)))

    ordered = sorted(notes, key=lambda x: (float(x["st"]), int(x["pitch"])))
    clusters: list[list[dict]] = []
    for n in ordered:
        if not clusters or (float(n["st"]) - float(clusters[-1][0]["st"])) > cluster_sec:
            clusters.append([n])
        else:
            clusters[-1].append(n)

    adjusted = 0
    abs_errs: list[float] = []
    for cl in clusters:
        t_win = min(float(x["st"]) for x in cl)
        sp_o = _spec_at(orig_2d, sr_orig, t_win + 0.010, 0.165)
        sp_r = _spec_at(rend_2d, sr_rend, t_win + 0.010, 0.165)
        if sp_o is None or sp_r is None:
            continue
        sp_o_pre = _spec_at(orig_2d, sr_orig, max(0.0, t_win - 0.090), 0.075)
        chord_pitches = {int(x["pitch"]) for x in cl}

        for n in cl:
            p = int(n["pitch"])
            orig_db = _pitch_db(sp_o, fs_o, sr_orig, p, chord_pitches)
            rend_db = _pitch_db(sp_r, fs_r, sr_rend, p, chord_pitches)
            if max(orig_db, rend_db) < -55.0:
                continue
            if sp_o_pre is not None:
                pre_db = _pitch_db(sp_o_pre, fs_o, sr_orig, p, chord_pitches)
                if pre_db > orig_db + 4.0 and pre_db > -45.0:
                    continue

            err_db = (rend_db - orig_db) - global_diff_db
            abs_errs.append(abs(err_db))
            if abs(err_db) <= deadband_db:
                continue

            corr_db = float(np.clip(damping * err_db, -max_corr_db, max_corr_db))
            cur_v = int(n.get("calc_vel", 64))
            staff_cap = int(rh_cap) if int(n.get("staff", 1)) == 1 else int(lh_cap)
            raw_new_v = int(round(cur_v * (10.0 ** (-corr_db / vel_db_per_decade))))
            if raw_new_v > cur_v + int(max_rise_vel):
                raw_new_v = cur_v + int(max_rise_vel)
            new_v = int(np.clip(raw_new_v, int(min_vel), staff_cap))
            if new_v != cur_v:
                n["calc_vel"] = new_v
                n["closed_loop_calibrated"] = True
                adjusted += 1

    med_err = float(np.median(abs_errs)) if abs_errs else 0.0
    return adjusted, med_err
