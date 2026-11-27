"""Pluggable solo-piano transcription engines (TransKun v2 default + MuScriptor optional)
and TempoMap-aware Performance-MIDI quantization preserving real MIDI velocities.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from tempo_map import TempoMap


def is_package_installed(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None


def resolve_transcription_engine(requested: str = "auto") -> str:
    """Select transcription backend: 'auto' prefers MIT-licensed TransKun v2 (zero HF login)."""
    req = (requested or "auto").strip().lower()
    if req in ("transkun", "muscriptor"):
        return req
    if is_package_installed("transkun"):
        return "transkun"
    if is_package_installed("muscriptor"):
        return "muscriptor"
    raise RuntimeError(
        "Neither 'transkun' nor 'muscriptor' is installed. "
        "Install TransKun v2 (MIT, no HuggingFace login required) via `pip install transkun`."
    )


def read_midi_note_events_sec(midi_path: Path, pad_sec: float = 0.0) -> list[dict]:
    """Extract note events in wall-clock seconds with real velocities from any MIDI file."""
    mid = mido.MidiFile(str(midi_path))
    tpb = mid.ticks_per_beat

    # Merge all tracks so tempo changes apply globally in chronological order
    merged = mido.merge_tracks(mid.tracks)
    cur_sec = 0.0
    cur_tempo = 500_000  # default 120 BPM
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    notes: list[dict] = []

    for msg in merged:
        if msg.time > 0:
            cur_sec += mido.tick2second(msg.time, tpb, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            key = (getattr(msg, "channel", 0), msg.note)
            active.setdefault(key, []).append((cur_sec, int(msg.velocity)))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (getattr(msg, "channel", 0), msg.note)
            stack = active.get(key)
            if stack:
                st_raw, vel = stack.pop(0)
                st = st_raw - pad_sec
                et = cur_sec - pad_sec
                if st < -0.15:
                    continue
                st_c = max(0.0, st)
                et_c = max(st_c + 0.03, et)
                notes.append(
                    {
                        "start_sec": st_c,
                        "end_sec": et_c,
                        "pitch": int(msg.note),
                        "velocity": max(1, min(127, vel)),
                    }
                )

    # Flush any unterminated notes
    for (_, pitch), stack in active.items():
        for st_raw, vel in stack:
            st = st_raw - pad_sec
            if st >= -0.15:
                st_c = max(0.0, st)
                notes.append(
                    {
                        "start_sec": st_c,
                        "end_sec": st_c + 0.25,
                        "pitch": int(pitch),
                        "velocity": max(1, min(127, vel)),
                    }
                )

    notes.sort(key=lambda n: (n["start_sec"], n["pitch"]))
    return notes


def quantize_performance_midi(
    note_events: list[dict],
    tempo_map: TempoMap,
    grid: int,
    beats_per_bar: int,
    beat_type: int,
    out_mid: Path,
    trim_leading_bars: bool = True,
) -> dict:
    """Quantize wall-clock performance notes onto `tempo_map` + `grid`, preserving real velocities.

    Output MIDI tick layout matches Stage 3 (`stage3_engrave_study_score.py`):
    - If a Bar 0 pickup exists (notes in `(-bar_slots, 0)` relative to Bar 1 beat 1),
      Bar 0 occupies `[0, bar_slots)` and Bar 1 starts at `bar_slots`.
    - Otherwise Bar 1 starts at tick `0`.
    """
    tpb = 480
    ticks_per_slot = tpb / float(grid)
    bar_slots = (beats_per_bar * grid * 4) // beat_type

    quantized: list[tuple[int, int, int, int]] = []  # (start_slot, dur_slots, pitch, vel)
    for n in note_events:
        q_on = tempo_map.sec_to_q(n["start_sec"])
        q_off = tempo_map.sec_to_q(n["end_sec"])
        s_on = int(round(q_on * grid))
        s_off = max(s_on + 1, int(round(q_off * grid)))
        quantized.append((s_on, s_off - s_on, int(n["pitch"]), int(n["velocity"])))

    if not quantized:
        raise ValueError("No notes available to quantize.")

    # Deduplicate same (start_slot, pitch) keeping strongest velocity/longest duration,
    # and trim overlapping same-pitch tails to prevent hanging MIDI notes.
    dedup_map: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for s_on, dur, pitch, vel in quantized:
        key = (s_on, pitch)
        prev = dedup_map.get(key)
        if prev is None:
            dedup_map[key] = (s_on, dur, pitch, vel)
        else:
            dedup_map[key] = (s_on, max(prev[1], dur), pitch, max(prev[3], vel))
    by_pitch: dict[int, list[tuple[int, int, int, int]]] = {}
    for item in dedup_map.values():
        by_pitch.setdefault(item[2], []).append(item)
    cleaned_q: list[tuple[int, int, int, int]] = []
    for pitch_notes in by_pitch.values():
        pitch_notes.sort(key=lambda x: x[0])
        for idx, (s_on, dur, pitch, vel) in enumerate(pitch_notes):
            if idx + 1 < len(pitch_notes):
                nxt_on = pitch_notes[idx + 1][0]
                dur = max(1, min(dur, nxt_on - s_on))
            cleaned_q.append((s_on, dur, pitch, vel))
    cleaned_q.sort(key=lambda x: (x[0], x[2]))
    quantized = cleaned_q

    min_slot = min(x[0] for x in quantized)
    # Shift so Bar 1 starts at bar_slots if there is a Bar 0 pickup, else at 0
    if trim_leading_bars:
        if min_slot < 0:
            # Pickup notes before Bar 1: place Bar 1 at bar_slots (or multiple if pickup > 1 bar)
            bars_before = int(math.ceil(-min_slot / float(bar_slots)))
            slot_offset = bars_before * bar_slots
        else:
            slot_offset = 0
    else:
        slot_offset = max(0, -min_slot)

    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    meta.name = "Score Meta"
    meta.append(mido.MetaMessage("time_signature", numerator=beats_per_bar, denominator=beat_type, time=0))
    quarter_us = max(100_000, min(2_500_000, int(round(60_000_000.0 / tempo_map.quarter_bpm))))
    meta.append(mido.MetaMessage("set_tempo", tempo=quarter_us, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    tr = mido.MidiTrack()
    tr.name = "Acoustic Grand Piano"
    tr.append(mido.Message("program_change", channel=0, program=0, time=0))

    raw_evs: list[tuple[int, int, int, int]] = []  # (tick, order, pitch, vel)
    for s_on, dur, pitch, vel in quantized:
        tk_on = max(0, int(round((s_on + slot_offset) * ticks_per_slot)))
        tk_off = max(tk_on + 1, int(round((s_on + dur + slot_offset) * ticks_per_slot)))
        raw_evs.append((tk_on, 1, pitch, vel))
        raw_evs.append((tk_off, 0, pitch, 0))

    raw_evs.sort(key=lambda x: (x[0], x[1], x[2]))
    cur_tk = 0
    for tk, kind, pitch, vel in raw_evs:
        dt = max(0, tk - cur_tk)
        cur_tk = tk
        if kind == 1:
            tr.append(mido.Message("note_on", channel=0, note=pitch, velocity=vel, time=dt))
        else:
            tr.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=dt))
    tr.append(mido.MetaMessage("end_of_track", time=tpb))
    mid.tracks.append(tr)

    out_mid.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_mid))
    return {
        "note_count": len(quantized),
        "slot_offset": slot_offset,
        "has_pickup": min_slot < 0,
        "quantized_midi": str(out_mid),
    }


def run_transkun_transcribe(
    audio_path: Path,
    out_raw_mid: Path,
    device: str = "cpu",
    pad_sec: float = 0.0,
    weight_path: str | None = None,
) -> list[dict]:
    """Run TransKun v2 (`python -m transkun.transcribe`) and return wall-clock note events."""
    out_raw_mid.parent.mkdir(parents=True, exist_ok=True)
    # TransKun v2 CLI supports 'cuda' and 'cpu'; map 'mps' safely to 'cpu' to avoid backend errors
    tk_device = "cuda" if device == "cuda" else "cpu"

    with tempfile.TemporaryDirectory(prefix="transkun-pad-") as tmpd:
        in_audio = audio_path
        if pad_sec > 0.0:
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
            pad_frames = int(round(pad_sec * sr))
            padded = np.vstack([np.zeros((pad_frames, data.shape[1]), dtype=np.float32), data])
            in_audio = Path(tmpd) / "padded_input.wav"
            sf.write(str(in_audio), padded, sr, subtype="PCM_24")

        tmp_mid = Path(tmpd) / "transkun_raw.mid"
        cmd = [sys.executable, "-m", "transkun.transcribe", str(in_audio), str(tmp_mid), "--device", tk_device]
        if weight_path:
            cmd.extend(["--weight", str(weight_path)])

        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=900)
        if proc.returncode != 0 and tk_device != "cpu":
            # Graceful fallback to CPU if CUDA invocation failed
            cmd_cpu = [sys.executable, "-m", "transkun.transcribe", str(in_audio), str(tmp_mid), "--device", "cpu"]
            if weight_path:
                cmd_cpu.extend(["--weight", str(weight_path)])
            proc = subprocess.run(cmd_cpu, capture_output=True, text=True, errors="replace", timeout=900)

        if proc.returncode != 0 or not tmp_mid.is_file():
            raise RuntimeError(f"TransKun v2 transcription failed: {proc.stderr or proc.stdout}")

        notes = read_midi_note_events_sec(tmp_mid, pad_sec=pad_sec)
        # Save raw performance MIDI (shifted if padded, or verbatim if pad_sec == 0)
        if pad_sec <= 0.0:
            out_raw_mid.write_bytes(tmp_mid.read_bytes())
        else:
            _write_raw_performance_midi(notes, out_raw_mid)
        return notes


def _write_raw_performance_midi(notes: list[dict], out_mid: Path) -> None:
    """Write unquantized performance MIDI in real seconds (120 BPM reference, 960 ticks/sec)."""
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    tr.name = "TransKun Raw Performance"
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    tr.append(mido.Message("program_change", channel=0, program=0, time=0))

    ticks_per_sec = 960.0
    evs: list[tuple[int, int, int, int]] = []
    for n in notes:
        tk_on = max(0, int(round(n["start_sec"] * ticks_per_sec)))
        tk_off = max(tk_on + 1, int(round(n["end_sec"] * ticks_per_sec)))
        evs.append((tk_on, 1, n["pitch"], n["velocity"]))
        evs.append((tk_off, 0, n["pitch"], 0))
    evs.sort(key=lambda x: (x[0], x[1], x[2]))

    cur_tk = 0
    for tk, kind, pitch, vel in evs:
        dt = max(0, tk - cur_tk)
        cur_tk = tk
        if kind == 1:
            tr.append(mido.Message("note_on", channel=0, note=pitch, velocity=vel, time=dt))
        else:
            tr.append(mido.Message("note_off", channel=0, note=pitch, velocity=0, time=dt))
    tr.append(mido.MetaMessage("end_of_track", time=tpb))
    mid.tracks.append(tr)
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_mid))
