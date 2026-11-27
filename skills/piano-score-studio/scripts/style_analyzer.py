#!/usr/bin/env python3
"""v3.0 Phase 2 — Auto Key Detection (Krumhansl-Schmuckler) & 6-Dim Musical DNA Style Analyzer.

Replaces hardcoded Bb-major (-2) key signatures and Pop-Ballad-only thresholds with:
  1. Krumhansl-Schmuckler duration-weighted key-finding algorithm (major & minor).
  2. 6-dimensional Musical DNA extraction from quantized MIDI + Stage 1 TempoMap:
       - tempo_cv:               Tempo elasticity (coefficient of variation across bars)
       - subdivision_density:    Share of fast (<= 16th-note) inter-onset intervals
       - syncopation_index:      Share of off-beat / weak-beat onsets
       - chromaticism_rate:      Duration-weighted out-of-scale pitch fraction
       - polyphony_independence: Share of overlapping notes with independent onset/offset
       - swing_ratio:            Long-to-short eighth-note pair IOI ratio (1.0=straight, ~1.5-2.0=swing)
  3. Transparent rule-scored classifier selecting among 4 declarative profiles:
       pop_ballad | classical_romantic | baroque_polyphonic | jazz_groove
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# Canonical Krumhansl-Kessler (1982) key profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64)
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64)

MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
MINOR_INTERVALS = (0, 2, 3, 5, 7, 8, 10)

# Circle-of-fifths mapping (prefer standard piano key signatures -6..+6)
MAJOR_PC_TO_FIFTHS = {
    0: 0,   # C
    7: 1,   # G
    2: 2,   # D
    9: 3,   # A
    4: 4,   # E
    11: 5,  # B
    6: 6,   # F#
    1: -5,  # Db
    8: -4,  # Ab
    3: -3,  # Eb
    10: -2, # Bb
    5: -1,  # F
}
MINOR_PC_TO_FIFTHS = {
    9: 0,   # Am
    4: 1,   # Em
    11: 2,  # Bm
    6: 3,   # F#m
    1: 4,   # C#m
    8: 5,   # G#m
    3: -6,  # Ebm
    10: -5, # Bbm
    5: -4,  # Fm
    0: -3,  # Cm
    7: -2,  # Gm
    2: -1,  # Dm
}

MAJOR_PC_NAMES = {
    0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B",
}
MINOR_PC_NAMES = {
    0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "Bb", 11: "B",
}

VALID_PROFILES = ("pop_ballad", "classical_romantic", "baroque_polyphonic", "jazz_groove")


def scale_pcs_from_key_fifths(key_fifths: int) -> set[int]:
    """Return the 7 diatonic pitch classes (0..11) for a given circle-of-fifths key signature."""
    tonic_major = (int(key_fifths) * 7) % 12
    return {(tonic_major + iv) % 12 for iv in MAJOR_INTERVALS}


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a0, b0) / denom)


def detect_key_krumhansl_schmuckler(
    notes: Sequence[tuple[int, float, float, int]],
) -> dict:
    """Estimate key signature via duration-weighted Krumhansl-Schmuckler correlation.

    Args:
        notes: Sequence of (pitch, start_q, end_q, velocity) tuples.
    Returns:
        Dict with key_name, tonic_pc, mode, key_fifths, confidence, runner_up, scale_pcs.
    """
    hist = np.zeros(12, dtype=np.float64)
    for pitch, start_q, end_q, vel in notes:
        dur = max(0.0625, float(end_q) - float(start_q))
        # Subtle velocity emphasis (0.75..1.25) so structural tones outweigh faint grace artifacts
        vel_w = 0.75 + 0.5 * (max(1, min(127, int(vel))) / 127.0)
        hist[int(pitch) % 12] += dur * vel_w

    if float(hist.sum()) <= 1e-9:
        return {
            "key_name": "C major",
            "tonic_pc": 0,
            "mode": "major",
            "key_fifths": 0,
            "confidence": 0.0,
            "runner_up": None,
            "scale_pcs": sorted(scale_pcs_from_key_fifths(0)),
        }

    candidates: list[tuple[float, int, str, int, str]] = []
    for tonic in range(12):
        rotated = np.roll(hist, -tonic)
        r_maj = _pearson_corr(rotated, KK_MAJOR)
        r_min = _pearson_corr(rotated, KK_MINOR)
        candidates.append((r_maj, tonic, "major", MAJOR_PC_TO_FIFTHS[tonic], f"{MAJOR_PC_NAMES[tonic]} major"))
        candidates.append((r_min, tonic, "minor", MINOR_PC_TO_FIFTHS[tonic], f"{MINOR_PC_NAMES[tonic]} minor"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_r, best_tonic, best_mode, best_fifths, best_name = candidates[0]
    ru_r, _, _, ru_fifths, ru_name = candidates[1]
    intervals = MAJOR_INTERVALS if best_mode == "major" else MINOR_INTERVALS
    scale_pcs = sorted({(best_tonic + iv) % 12 for iv in intervals})

    return {
        "key_name": best_name,
        "tonic_pc": int(best_tonic),
        "mode": best_mode,
        "key_fifths": int(best_fifths),
        "confidence": round(float(best_r), 4),
        "runner_up": {"key_name": ru_name, "key_fifths": int(ru_fifths), "confidence": round(float(ru_r), 4)},
        "scale_pcs": scale_pcs,
    }


def load_midi_notes_in_quarters(midi_path: Path) -> list[tuple[int, float, float, int]]:
    """Read a quantized MIDI file into (pitch, start_q, end_q, velocity) tuples."""
    import mido

    mid = mido.MidiFile(str(midi_path))
    tpq = max(1, int(mid.ticks_per_beat))
    notes: list[tuple[int, float, float, int]] = []
    for tr in mid.tracks:
        cur_tick = 0
        active: dict[int, list[tuple[int, int]]] = {}
        for msg in tr:
            cur_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append((cur_tick, int(msg.velocity)))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                stack = active.get(msg.note)
                if stack:
                    st_tick, vel = stack.pop(0)
                    st_q = st_tick / tpq
                    en_q = max(st_q + 0.0625, cur_tick / tpq)
                    notes.append((int(msg.note), st_q, en_q, vel))
        for pitch, stack in active.items():
            for st_tick, vel in stack:
                st_q = st_tick / tpq
                notes.append((int(pitch), st_q, st_q + 0.5, vel))
    notes.sort(key=lambda x: (x[1], x[0]))
    return notes


def extract_musical_dna(
    notes: Sequence[tuple[int, float, float, int]],
    tempo_map_dict: dict | None = None,
    beats_per_bar: int = 4,
    beat_type: int = 4,
    key_info: dict | None = None,
) -> dict:
    """Compute the 6-dimensional Musical DNA vector from note events and Stage 1 TempoMap."""
    if key_info is None:
        key_info = detect_key_krumhansl_schmuckler(notes)

    if not notes:
        return {
            "tempo_cv": 0.0,
            "subdivision_density": 0.0,
            "syncopation_index": 0.0,
            "chromaticism_rate": 0.0,
            "polyphony_independence": 0.0,
            "swing_ratio": 1.0,
            "note_count": 0,
            "mean_notes_per_beat": 0.0,
            "detected_key": key_info,
        }

    # 1. tempo_cv from Stage 1 TempoMap (bar_tempos_bpm or tempo_points)
    tempo_cv = 0.0
    if isinstance(tempo_map_dict, dict):
        bpms = []
        if tempo_map_dict.get("bar_tempos_bpm"):
            bpms = [float(x) for x in tempo_map_dict["bar_tempos_bpm"] if float(x) > 0]
        elif tempo_map_dict.get("tempo_points"):
            bpms = [float(p.get("bpm", 0)) for p in tempo_map_dict["tempo_points"] if float(p.get("bpm", 0)) > 0]
        if len(bpms) >= 3:
            arr = np.asarray(bpms, dtype=np.float64)
            mean_b = float(arr.mean())
            if mean_b > 1e-6:
                tempo_cv = float(arr.std() / mean_b)

    # Quarter-note length of one bar
    bar_q = beats_per_bar * (4.0 / beat_type)
    starts = sorted({round(float(n[1]), 4) for n in notes})

    # 2. subdivision_density: share of consecutive unique onset IOIs <= 0.30 quarter notes (16th-note activity)
    iois = [starts[i + 1] - starts[i] for i in range(len(starts) - 1) if (starts[i + 1] - starts[i]) > 0.02]
    if iois:
        fast_iois = sum(1 for d in iois if d <= 0.30)
        subdivision_density = fast_iois / len(iois)
    else:
        subdivision_density = 0.0

    # 3. syncopation_index: share of unique onsets landing off main quarter-note beats AND not aligned with downbeat
    offbeat_count = 0
    for st in starts:
        beat_frac = st % 1.0
        dist_to_beat = min(beat_frac, 1.0 - beat_frac)
        if dist_to_beat > 0.14:
            offbeat_count += 1
    syncopation_index = offbeat_count / max(1, len(starts))

    # 4. chromaticism_rate: duration-weighted pitch fraction outside detected diatonic scale
    scale_pcs = set(key_info.get("scale_pcs", []))
    total_dur = 0.0
    oos_dur = 0.0
    for pitch, st_q, en_q, _ in notes:
        dur = max(0.0625, float(en_q) - float(st_q))
        total_dur += dur
        if (int(pitch) % 12) not in scale_pcs:
            oos_dur += dur
    chromaticism_rate = oos_dur / max(1e-9, total_dur)

    # 5. polyphony_independence: fraction of notes overlapping another note that starts at a DIFFERENT time
    #    (high in Baroque fugues/inventions; low in block-chord homophony)
    sorted_notes = sorted(notes, key=lambda x: (x[1], x[0]))
    independent_overlaps = 0
    total_overlaps = 0
    n_len = len(sorted_notes)
    for i in range(n_len):
        p_i, st_i, en_i, _ = sorted_notes[i]
        has_indep = False
        has_any_overlap = False
        # Scan nearby window (bounded for O(N) performance on long pieces)
        j = i + 1
        while j < n_len and sorted_notes[j][1] < en_i - 0.04:
            st_j = sorted_notes[j][1]
            has_any_overlap = True
            if abs(st_j - st_i) >= 0.12:
                has_indep = True
                break
            j += 1
        if has_any_overlap:
            total_overlaps += 1
            if has_indep:
                independent_overlaps += 1
    polyphony_independence = independent_overlaps / max(1, n_len)

    # 6. swing_ratio: inspect consecutive IOI pairs summing to ~1.0 quarter note (0.85..1.15)
    #    where first IOI > second IOI (long-short eighth feel)
    swing_ratios: list[float] = []
    for i in range(len(iois) - 1):
        d1, d2 = iois[i], iois[i + 1]
        pair_sum = d1 + d2
        if 0.85 <= pair_sum <= 1.15 and d1 >= 0.30 and d2 >= 0.20:
            ratio = d1 / d2
            if 0.85 <= ratio <= 2.4:
                swing_ratios.append(ratio)
    swing_ratio = float(np.median(swing_ratios)) if len(swing_ratios) >= 4 else 1.0
    swing_ratio = max(1.0, min(2.2, swing_ratio))

    total_beats = max(1.0, (max(n[2] for n in notes) - min(n[1] for n in notes)))
    mean_notes_per_beat = len(notes) / total_beats

    return {
        "tempo_cv": round(float(tempo_cv), 4),
        "subdivision_density": round(float(subdivision_density), 4),
        "syncopation_index": round(float(syncopation_index), 4),
        "chromaticism_rate": round(float(chromaticism_rate), 4),
        "polyphony_independence": round(float(polyphony_independence), 4),
        "swing_ratio": round(float(swing_ratio), 4),
        "note_count": len(notes),
        "bar_quarters": round(float(bar_q), 4),
        "mean_notes_per_beat": round(float(mean_notes_per_beat), 3),
        "detected_key": key_info,
    }


def classify_style_profile(dna: dict) -> tuple[str, float, dict[str, float], str]:
    """Deterministically score all 4 solo-piano style profiles from the 6-dim Musical DNA."""
    tempo_cv = float(dna.get("tempo_cv", 0.0))
    subdiv = float(dna.get("subdivision_density", 0.0))
    syncop = float(dna.get("syncopation_index", 0.0))
    chrom = float(dna.get("chromaticism_rate", 0.0))
    poly = float(dna.get("polyphony_independence", 0.0))
    swing = float(dna.get("swing_ratio", 1.0))

    scores: dict[str, float] = {}

    # 1. Jazz / Groove: swing ratio, chromatic extensions, off-beat syncopation
    s_jazz = (
        2.6 * max(0.0, min(1.0, (swing - 1.18) / 0.55))
        + 1.8 * max(0.0, min(1.0, (chrom - 0.12) / 0.22))
        + 1.5 * max(0.0, min(1.0, (syncop - 0.35) / 0.35))
        - 0.6 * max(0.0, min(1.0, (tempo_cv - 0.10) / 0.12))
    )
    scores["jazz_groove"] = round(float(s_jazz), 4)

    # 2. Baroque / Polyphonic: high independent voice overlap + steady tempo + motoric 16ths + low swing
    s_baroque = (
        2.4 * max(0.0, min(1.0, (poly - 0.32) / 0.38))
        + 1.6 * max(0.0, min(1.0, (subdiv - 0.30) / 0.45))
        + 0.8 * max(0.0, 1.0 - tempo_cv / 0.08)
        - 1.5 * max(0.0, min(1.0, (swing - 1.22) / 0.40))
        - 0.8 * max(0.0, min(1.0, (chrom - 0.18) / 0.20))
    )
    scores["baroque_polyphonic"] = round(float(s_baroque), 4)

    # 3. Classical / Romantic: rubato tempo elasticity + moderate/high subdivision or chromatic color
    s_classical = (
        2.8 * max(0.0, min(1.0, (tempo_cv - 0.055) / 0.10))
        + 0.9 * max(0.0, min(1.0, (subdiv - 0.20) / 0.45))
        + 0.7 * max(0.0, min(1.0, (chrom - 0.06) / 0.18))
        + 0.5 * max(0.0, min(1.0, (poly - 0.18) / 0.40))
        - 1.2 * max(0.0, min(1.0, (swing - 1.25) / 0.40))
    )
    scores["classical_romantic"] = round(float(s_classical), 4)

    # 4. Pop / Modern Ballad: steady tempo, diatonic, straight rhythm, homophonic (low independent counterpoint)
    s_pop = (
        1.1 * max(0.0, 1.0 - tempo_cv / 0.08)
        + 1.1 * max(0.0, 1.0 - chrom / 0.16)
        + 1.0 * max(0.0, 1.0 - max(0.0, poly - 0.15) / 0.35)
        + 0.8 * max(0.0, 1.0 - max(0.0, subdiv - 0.15) / 0.45)
        - 1.5 * max(0.0, min(1.0, (swing - 1.22) / 0.40))
    )
    scores["pop_ballad"] = round(float(s_pop), 4)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_sc = ranked[0]
    second_sc = ranked[1][1]
    margin = max(0.0, best_sc - second_sc)
    # Soft confidence in [0.50, 0.99]
    confidence = round(float(0.50 + 0.49 * math.tanh(margin / 1.2)), 4)

    rationale = (
        f"Selected '{best_id}' (score={best_sc:.2f}, runner_up={ranked[1][0]}:{second_sc:.2f}) "
        f"from tempo_cv={tempo_cv:.3f}, subdiv={subdiv:.2f}, syncop={syncop:.2f}, "
        f"chrom={chrom:.2f}, poly={poly:.2f}, swing={swing:.2f}."
    )
    return best_id, confidence, scores, rationale


def default_profiles_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "profiles"


def resolve_profile_path(profile_id_or_path: str, profiles_dir: Path | None = None) -> Path:
    p = Path(profile_id_or_path)
    if p.is_file():
        return p
    base = profiles_dir if profiles_dir is not None else default_profiles_dir()
    candidate = base / f"{profile_id_or_path}.json"
    if candidate.is_file():
        return candidate
    fallback = base / "pop_ballad.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Style profile '{profile_id_or_path}' not found in {base}")


def load_style_profile(profile_id_or_path: str | Path | None, profiles_dir: Path | None = None) -> dict:
    """Load a declarative style profile JSON (defaults to pop_ballad if None)."""
    if not profile_id_or_path:
        profile_id_or_path = "pop_ballad"
    path = resolve_profile_path(str(profile_id_or_path), profiles_dir=profiles_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_profile_path"] = str(path)
    return data


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 1.5 Krumhansl-Schmuckler Key & 6-Dim Musical DNA Style Analyzer")
    ap.add_argument("--midi", required=True, help="Quantized MIDI from Stage 1")
    ap.add_argument("--stage1-cal", default=None, help="Optional Stage 1 stage1_calibration.json")
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--beat-type", type=int, default=4, choices=(4, 8))
    ap.add_argument(
        "--style",
        default="auto",
        choices=("auto",) + VALID_PROFILES,
        help="Auto-classify style profile or force a specific profile",
    )
    ap.add_argument("--profiles-dir", default=None, help="Optional custom profiles directory")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. style_analysis.json)")
    args = ap.parse_args()

    tempo_map_dict = None
    beats_per_bar = args.beats_per_bar
    beat_type = args.beat_type
    if args.stage1_cal and Path(args.stage1_cal).is_file():
        cal = json.loads(Path(args.stage1_cal).read_text(encoding="utf-8"))
        tempo_map_dict = cal.get("tempo_map")
        beats_per_bar = int(cal.get("beats_per_bar", beats_per_bar))
        beat_type = int(cal.get("beat_type", beat_type))

    notes = load_midi_notes_in_quarters(Path(args.midi))
    key_info = detect_key_krumhansl_schmuckler(notes)
    dna = extract_musical_dna(
        notes,
        tempo_map_dict=tempo_map_dict,
        beats_per_bar=beats_per_bar,
        beat_type=beat_type,
        key_info=key_info,
    )
    auto_profile_id, confidence, scores, rationale = classify_style_profile(dna)
    selected_profile_id = auto_profile_id if args.style == "auto" else args.style
    pdir = Path(args.profiles_dir) if args.profiles_dir else None
    profile_path = resolve_profile_path(selected_profile_id, profiles_dir=pdir)
    profile_data = load_style_profile(profile_path)

    report = {
        "schema_version": "3.0",
        "selected_profile": selected_profile_id,
        "profile_source": "auto" if args.style == "auto" else "user_override",
        "profile_path": str(profile_path),
        "auto_classified_profile": auto_profile_id,
        "classification_confidence": confidence,
        "profile_scores": scores,
        "rationale": rationale,
        "detected_key": key_info,
        "musical_dna": dna,
        "profile": {k: v for k, v in profile_data.items() if not k.startswith("_")},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Style Analyzer: key={key_info['key_name']} (fifths={key_info['key_fifths']}, r={key_info['confidence']:.2f}) | "
        f"profile={selected_profile_id} (conf={confidence:.2f}, source={report['profile_source']})"
    )


if __name__ == "__main__":
    main()
