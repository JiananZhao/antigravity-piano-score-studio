"""Stage 3: Generalized pedagogical piano score engraver (MusicXML + MuseScore 4 PDF).

Produces a clean, human-readable study edition for any solo piano MIDI/audio:
- Proper anacrusis (Bar 0 implicit=yes) or full-bar start
- Locked RH (G2) and LH (F4) clefs with zero intra-bar clef flipping
- Span-aware hand splitting with <arpeggiate/> on wide chords (>12 semitones)
- Max 2 voices per hand, hidden structural spacer rests, audio-aware legato gap fill
- Configurable quantization grid (8th / triplet-8th / 16th) & simple/compound meters (beat-type 4 or 8)
- Beat-grouped beaming and key-consistent enharmonic spelling
- Phrase dynamics derived automatically from bar_lh_rh_envelope.csv (when provided)
- Balanced A4 4-bar phrase pagination
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

DIVISIONS = 2  # Default divisions per quarter note (2 = eighth-note grid)

SHARP_PC_SPELLING = {
    0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0), 5: ("F", 0),
    6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0), 10: ("A", 1), 11: ("B", 0),
}
FLAT_PC_SPELLING = {
    0: ("C", 0), 1: ("D", -1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0), 5: ("F", 0),
    6: ("F", 1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0), 10: ("B", -1), 11: ("B", 0),
}
SHARP_ORDER = ["F", "C", "G", "D", "A", "E", "B"]
FLAT_ORDER = ["B", "E", "A", "D", "G", "C", "F"]


def key_sig_alter(fifths: int) -> dict[str, int]:
    if fifths > 0:
        return {s: 1 for s in SHARP_ORDER[:fifths]}
    if fifths < 0:
        return {s: -1 for s in FLAT_ORDER[:-fifths]}
    return {}


@dataclasses.dataclass
class RawNote:
    start_slot: int
    dur: int
    pitch: int
    hand: int = 1
    arpeggiate: bool = False


@dataclasses.dataclass
class BarSegment:
    bar: int
    slot: int
    dur: int
    pitch: int
    hand: int
    tie_stop: bool = False
    tie_start: bool = False
    arpeggiate: bool = False


@dataclasses.dataclass
class ChordEvent:
    slot: int
    dur: int
    segs: list[BarSegment]


@dataclasses.dataclass
class RenderedAtom:
    slot: int
    dur: int
    type_str: str
    dotted: bool
    is_rest: bool
    print_object: bool
    pitches: list[tuple[int, bool, bool, bool]]
    beam1: str | None = None
    beam2: str | None = None
    tuplet: tuple[int, int] | None = None


def compute_grid_metrics(beats_per_bar: int, beat_type: int, grid: int) -> tuple[int, int, int, int]:
    """Return (divisions, bar_slots, beat_slots, beam_group_slots)."""
    divisions = grid  # MusicXML divisions = slots per quarter note (2=8th, 3=triplet-8th, 4=16th)
    if beat_type == 8 and divisions == 3:
        raise ValueError("--grid 3 (quarter-note triplets) is only valid with --beat-type 4; use --grid 2 or 4 for 6/8.")
    bar_slots = (beats_per_bar * divisions * 4) // beat_type
    if beat_type == 8 and beats_per_bar % 3 == 0:
        beat_slots = 3 * (divisions // 2)  # compound dotted-quarter beat (3 eighths)
    else:
        beat_slots = divisions if beat_type == 4 else max(1, divisions // 2)
    beam_group_slots = beat_slots
    return divisions, bar_slots, beat_slots, beam_group_slots


def load_midi_notes(path: Path, divisions: int, bar_slots: int) -> tuple[list[RawNote], int]:
    """Load MIDI notes quantized to the chosen division grid; detect Bar 0 pickup if first bar is partial."""
    mid = mido.MidiFile(path)
    ticks_per_slot = mid.ticks_per_beat / float(divisions)
    notes: list[RawNote] = []
    for tr in mid.tracks:
        t = 0
        active: dict[int, int] = {}
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in active:
                    st = active.pop(msg.note)
                    if t > st:
                        notes.append(
                            RawNote(
                                int(round(st / ticks_per_slot)),
                                max(1, int(round((t - st) / ticks_per_slot))),
                                msg.note,
                            )
                        )
                active[msg.note] = t
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    st = active.pop(msg.note)
                    if t > st:
                        notes.append(
                            RawNote(
                                int(round(st / ticks_per_slot)),
                                max(1, int(round((t - st) / ticks_per_slot))),
                                msg.note,
                            )
                        )
    notes.sort(key=lambda n: (n.start_slot, n.pitch))
    if not notes:
        return notes, 0
    first_slot = notes[0].start_slot
    pickup_start = first_slot if (0 < first_slot < bar_slots) else 0
    return notes, pickup_start


def load_slot_rms_db(
    audio_path: Path,
    bpm: float,
    beat_type: int,
    divisions: int,
    first_downbeat: float,
    bar1_slot: int,
    total_slots: int,
) -> np.ndarray:
    data, sr = sf.read(str(audio_path), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    quarter_s = (60.0 / bpm) * (beat_type / 4.0)
    slot_s = quarter_s / float(divisions)
    dbs = np.full(total_slots, -60.0, dtype=np.float64)
    for s in range(total_slots):
        t0 = first_downbeat + (s - bar1_slot) * slot_s
        t1 = t0 + slot_s
        i0, i1 = max(0, int(round(t0 * sr))), min(len(mono), int(round(t1 * sr)))
        if i1 > i0:
            dbs[s] = 20.0 * np.log10(float(np.sqrt(np.mean(mono[i0:i1] ** 2))) + 1e-12)
    return dbs


def split_hands(notes: list[RawNote]) -> None:
    by_slot: dict[int, list[RawNote]] = {}
    for n in notes:
        by_slot.setdefault(n.start_slot, []).append(n)
    active_lh: list[tuple[int, int]] = []
    active_rh: list[tuple[int, int]] = []
    prev_lh_c, prev_rh_c = 50.0, 72.0
    last_lh_slot, last_lh_pitch = -999, 50

    for t in sorted(by_slot):
        active_lh = [(e, p) for e, p in active_lh if e > t]
        active_rh = [(e, p) for e, p in active_rh if e > t]
        grp = sorted(by_slot[t], key=lambda x: x.pitch)

        if len(grp) >= 2:
            best_cost, best_s = float("inf"), 0
            for s in range(len(grp) + 1):
                lh, rh = grp[:s], grp[s:]
                cost = 0.0
                if lh:
                    sp = lh[-1].pitch - lh[0].pitch
                    if sp > 12:
                        cost += 25.0 * (sp - 12) ** 2
                    for x in lh:
                        if x.pitch > 65:
                            cost += 40.0 * (x.pitch - 65) ** 2
                        cost += 1.5 * max(0.0, x.pitch - 63.5)
                    cost += 0.5 * abs(sum(x.pitch for x in lh) / len(lh) - prev_lh_c)
                if rh:
                    sp = rh[-1].pitch - rh[0].pitch
                    if sp > 12:
                        cost += 25.0 * (sp - 12) ** 2
                    for x in rh:
                        if x.pitch < 58:
                            cost += 30.0 * (58 - x.pitch) ** 2
                        cost += 1.5 * max(0.0, 63.5 - x.pitch)
                    cost += 0.5 * abs(sum(x.pitch for x in rh) / len(rh) - prev_rh_c)
                if cost < best_cost:
                    best_cost, best_s = cost, s
            lh_n, rh_n = grp[:best_s], grp[best_s:]
            for x in lh_n:
                x.hand = 2
            for x in rh_n:
                x.hand = 1
            if len(rh_n) >= 2 and (rh_n[-1].pitch - rh_n[0].pitch) > 12:
                for x in rh_n:
                    x.arpeggiate = True
            if len(lh_n) >= 2 and (lh_n[-1].pitch - lh_n[0].pitch) > 12:
                for x in lh_n:
                    x.arpeggiate = True
        else:
            x = grp[0]
            p = x.pitch
            if p <= 59:
                x.hand = 2
            elif p >= 66:
                x.hand = 1
            elif any(ap >= p + 10 for _, ap in active_rh):
                x.hand = 2
            elif any(ap <= p - 13 for _, ap in active_lh):
                x.hand = 1
            elif last_lh_slot == t - 1 and abs(last_lh_pitch - p) <= 9 and any(ap > p for _, ap in active_rh):
                x.hand = 2
            else:
                x.hand = 2 if p <= 62 else 1
            lh_n = [x] if x.hand == 2 else []
            rh_n = [x] if x.hand == 1 else []

        for x in lh_n:
            active_lh.append((x.start_slot + x.dur, x.pitch))
        for x in rh_n:
            active_rh.append((x.start_slot + x.dur, x.pitch))
        if lh_n:
            prev_lh_c = 0.65 * prev_lh_c + 0.35 * (sum(x.pitch for x in lh_n) / len(lh_n))
            last_lh_slot, last_lh_pitch = t, lh_n[-1].pitch
        if rh_n:
            prev_rh_c = 0.65 * prev_rh_c + 0.35 * (sum(x.pitch for x in rh_n) / len(rh_n))


def slice_into_bars(
    notes: list[RawNote], pickup_start: int, bar_slots: int
) -> tuple[dict[tuple[int, int], list[BarSegment]], int]:
    bar1_slot = bar_slots if pickup_start > 0 else 0
    max_end = max(n.start_slot + n.dur for n in notes)
    max_bar = max(1, (max_end - bar1_slot + bar_slots - 1) // bar_slots)
    bars: dict[tuple[int, int], list[BarSegment]] = {}
    for n in notes:
        cur, rem, first = n.start_slot, n.dur, True
        while rem > 0:
            if pickup_start > 0 and cur < bar_slots:
                bar, b_start, b_end = 0, pickup_start, bar_slots
            else:
                bar = ((cur - bar1_slot) // bar_slots) + 1
                b_start = bar1_slot + (bar - 1) * bar_slots
                b_end = b_start + bar_slots
            if bar > max_bar:
                break
            take = min(rem, b_end - cur)
            bars.setdefault((bar, n.hand), []).append(
                BarSegment(
                    bar=bar, slot=cur - b_start, dur=take, pitch=n.pitch, hand=n.hand,
                    tie_stop=not first, tie_start=(rem > take) and (bar < max_bar),
                    arpeggiate=n.arpeggiate if first else False,
                )
            )
            cur += take
            rem -= take
            first = False
    return bars, max_bar


def bridge_gaps(
    events: list[ChordEvent],
    bar_global_start: int,
    bar_len: int,
    slot_rms: np.ndarray,
    divisions: int,
) -> None:
    eighth_slots = max(1, divisions // 2)
    quarter_slots = divisions
    for i, ev in enumerate(events):
        end_sl = ev.slot + ev.dur
        if i + 1 < len(events):
            nxt = events[i + 1].slot
            if 0 < (nxt - end_sl) <= eighth_slots:
                if all(
                    (bar_global_start + s) < len(slot_rms) and slot_rms[bar_global_start + s] > -38.0
                    for s in range(end_sl, nxt)
                ):
                    ev.dur = nxt - ev.slot
        else:
            gap = bar_len - end_sl
            if 0 < gap <= quarter_slots and all(
                slot_rms[bar_global_start + s] > -38.0
                for s in range(end_sl, bar_len)
                if bar_global_start + s < len(slot_rms)
            ):
                ev.dur = bar_len - ev.slot


def simplify_bar_hand(
    segs: list[BarSegment],
    bar: int,
    bar_global_start: int,
    bar_len: int,
    slot_rms: np.ndarray,
    divisions: int,
) -> tuple[list[ChordEvent], list[ChordEvent] | None]:
    if not segs:
        return [], None
    by_s: dict[int, list[BarSegment]] = {}
    for s in segs:
        by_s.setdefault(s.slot, []).append(s)
    slots = sorted(by_s)

    if bar == 0:
        v1 = []
        for i, sl in enumerate(slots):
            grp = sorted(by_s[sl], key=lambda x: x.pitch)
            nxt = slots[i + 1] if i + 1 < len(slots) else bar_len
            v1.append(ChordEvent(sl, max(1, min(min(x.dur for x in grp), nxt - sl)), grp))
        return v1, None

    anchor_min_dur = int(round(1.5 * divisions))
    anchor_segs: list[BarSegment] = []
    anchor_is_lower = True
    for sl in slots:
        for seg in by_s[sl]:
            if seg.dur >= anchor_min_dur:
                later = [s2 for s2 in slots if sl < s2 < sl + seg.dur]
                if len(later) >= 2:
                    lp = [x.pitch for s2 in later for x in by_s[s2]]
                    if seg.pitch < min(lp):
                        anchor_segs.append(seg)
                        anchor_is_lower = True
                    elif seg.pitch > max(lp):
                        anchor_segs.append(seg)
                        anchor_is_lower = False

    if anchor_segs:
        a_ids = {id(s) for s in anchor_segs}
        v_a = [s for s in segs if id(s) in a_ids]
        v_m = [s for s in segs if id(s) not in a_ids]
        if v_a and v_m:
            def pack(vsegs: list[BarSegment]) -> list[ChordEvent]:
                vb: dict[int, list[BarSegment]] = {}
                for x in vsegs:
                    vb.setdefault(x.slot, []).append(x)
                vslots = sorted(vb)
                out = []
                for idx, sl in enumerate(vslots):
                    g = sorted(vb[sl], key=lambda x: x.pitch)
                    nxt = vslots[idx + 1] if idx + 1 < len(vslots) else bar_len
                    out.append(ChordEvent(sl, max(1, min(max(x.dur for x in g), nxt - sl)), g))
                return out
            eva, evm = pack(v_a), pack(v_m)
            bridge_gaps(evm, bar_global_start, bar_len, slot_rms, divisions)
            return (evm, eva) if anchor_is_lower else (eva, evm)

    v1 = []
    for i, sl in enumerate(slots):
        grp = sorted(by_s[sl], key=lambda x: x.pitch)
        nxt = slots[i + 1] if i + 1 < len(slots) else bar_len
        d = (bar_len - sl) if (nxt == bar_len and any(x.tie_start for x in grp)) else min(min(x.dur for x in grp), nxt - sl)
        v1.append(ChordEvent(sl, max(1, d), grp))
    bridge_gaps(v1, bar_global_start, bar_len, slot_rms, divisions)
    return v1, None


def split_duration(
    slot: int,
    dur: int,
    is_rest: bool,
    divisions: int,
    beat_slots: int,
) -> list[tuple[int, int, str, bool, tuple[int, int] | None]]:
    """Split arbitrary slot duration into MusicXML-valid rhythmic atoms aligned to beat boundaries."""
    if divisions == 2:
        table = [
            (8, "whole", False, None, lambda c: c == 0),
            (6, "half", True, None, lambda c: (not is_rest or beat_slots == 3) and c == 0),
            (4, "half", False, None, lambda c: c % 2 == 0 and beat_slots != 3),
            (3, "quarter", True, None, lambda c: (c % beat_slots == 0) if (not is_rest or beat_slots == 3) else False),
            (2, "quarter", False, None, lambda c: c % 2 == 0),
            (1, "eighth", False, None, lambda c: True),
        ]
    elif divisions == 4:
        table = [
            (16, "whole", False, None, lambda c: c == 0),
            (12, "half", True, None, lambda c: (not is_rest or beat_slots == 6) and c == 0),
            (8, "half", False, None, lambda c: c % 4 == 0 and beat_slots != 6),
            (6, "quarter", True, None, lambda c: (c % 2 == 0) if not is_rest else (beat_slots == 6 and c % 6 == 0)),
            (4, "quarter", False, None, lambda c: c % 4 == 0),
            (3, "eighth", True, None, lambda c: (not is_rest) and c % 2 == 0),
            (2, "eighth", False, None, lambda c: c % 2 == 0),
            (1, "16th", False, None, lambda c: True),
        ]
    else:  # divisions == 3 (eighth-note triplets per quarter beat)
        table = [
            (12, "whole", False, None, lambda c: c == 0),
            (9, "half", True, None, lambda c: (not is_rest) and c == 0),
            (6, "half", False, None, lambda c: c % 3 == 0),
            (3, "quarter", False, None, lambda c: c % 3 == 0),
            (2, "quarter", False, (3, 2), lambda c: (not is_rest) and c % 3 == 0),
            (1, "eighth", False, (3, 2), lambda c: True),
        ]

    out: list[tuple[int, int, str, bool, tuple[int, int] | None]] = []
    c, rem = slot, dur
    while rem > 0:
        max_step = (beat_slots - (c % beat_slots)) if (c % beat_slots != 0) else rem
        limit = min(rem, max_step)
        for step, tp, dot, tup, ok in table:
            if step <= limit and ok(c):
                out.append((c, step, tp, dot, tup))
                c += step
                rem -= step
                break
    return out


def build_voice_atoms(
    events: list[ChordEvent],
    bar_len: int,
    other: list[ChordEvent] | None,
    divisions: int,
    beat_slots: int,
    beam_group_slots: int,
) -> list[RenderedAtom]:
    if not events:
        return []
    covered = {s for oe in (other or []) for s in range(oe.slot, oe.slot + oe.dur)}
    atoms: list[RenderedAtom] = []
    cur = 0
    for ev in events:
        if ev.slot > cur:
            for ps, pd, pt, pdot, ptup in split_duration(cur, ev.slot - cur, True, divisions, beat_slots):
                hide = other is not None and all(s in covered for s in range(ps, ps + pd))
                atoms.append(RenderedAtom(ps, pd, pt, pdot, True, not hide, [], tuplet=ptup))
        pieces = split_duration(ev.slot, ev.dur, False, divisions, beat_slots)
        for idx, (ps, pd, pt, pdot, ptup) in enumerate(pieces):
            is_f, is_l = (idx == 0), (idx == len(pieces) - 1)
            plist = []
            for seg in ev.segs:
                t_stop = seg.tie_stop if is_f else True
                t_start = True if not is_l else (seg.tie_start if ev.slot + ev.dur == bar_len else False)
                plist.append((seg.pitch, t_stop, t_start, seg.arpeggiate and is_f))
            atoms.append(RenderedAtom(ps, pd, pt, pdot, False, True, plist, tuplet=ptup))
        cur = ev.slot + ev.dur
    if cur < bar_len:
        for ps, pd, pt, pdot, ptup in split_duration(cur, bar_len - cur, True, divisions, beat_slots):
            hide = other is not None and all(s in covered for s in range(ps, ps + pd))
            atoms.append(RenderedAtom(ps, pd, pt, pdot, True, not hide, [], tuplet=ptup))

    beamable = {"eighth", "16th"}
    run: list[RenderedAtom] = []

    def flush_run() -> None:
        if len(run) >= 2:
            for i, at in enumerate(run):
                at.beam1 = "begin" if i == 0 else ("end" if i == len(run) - 1 else "continue")
                if at.type_str == "16th":
                    prev_16 = i > 0 and run[i - 1].type_str == "16th"
                    next_16 = i + 1 < len(run) and run[i + 1].type_str == "16th"
                    if prev_16 and next_16:
                        at.beam2 = "continue"
                    elif next_16:
                        at.beam2 = "begin"
                    elif prev_16:
                        at.beam2 = "end"
                    else:
                        at.beam2 = "forward hook" if i == 0 else "backward hook"
        run.clear()

    prev_grp = -1
    for at in atoms:
        grp = at.slot // beam_group_slots
        if (not at.is_rest) and (at.type_str in beamable) and (not run or (grp == prev_grp and at.slot == run[-1].slot + run[-1].dur)):
            run.append(at)
            prev_grp = grp
        else:
            flush_run()
            if (not at.is_rest) and (at.type_str in beamable):
                run.append(at)
                prev_grp = grp
            else:
                prev_grp = -1
    flush_run()
    return atoms


def derive_dynamics_from_csv(csv_path: Path | None, max_bar: int) -> dict[int, list[tuple[str, str]]]:
    """Build sparse phrase dynamics (pp..f) at 4-bar boundaries from bar_lh_rh_envelope.csv."""
    out: dict[int, list[tuple[str, str]]] = {}
    if not csv_path or not csv_path.is_file():
        return out
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return out
    prev = None
    for i in range(0, len(rows), 4):
        ch = rows[i : i + 4]
        b = int(ch[0]["bar"])
        votes = Counter(r["suggested_dynamic"] for r in ch)
        dyn = votes.most_common(1)[0][0]
        if dyn != prev and b <= max_bar:
            out.setdefault(b, []).append(("dyn", dyn))
            prev = dyn
    return out


def find_musescore(explicit: str | None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    plugin_root = Path(__file__).resolve().parents[3]
    cfg_path = plugin_root / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg_ms = cfg.get("musescore_exe")
        if cfg_ms and Path(cfg_ms).is_file():
            return str(Path(cfg_ms))
    env_p = os.environ.get("MUSCRIPTOR_MUSESCORE")
    if env_p and Path(env_p).is_file():
        return env_p
    for name in ("mscore", "musescore", "MuseScore4", "musescore4"):
        h = shutil.which(name)
        if h:
            return h
    for p in (
        Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"),
        Path(r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe"),
        Path.home() / "AppData" / "Local" / "Programs" / "MuseScore 4" / "bin" / "MuseScore4.exe",
        Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore"),
    ):
        if p.is_file():
            return str(p)
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 3 generalized pedagogical piano score engraver")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--title", default="Piano Piece")
    ap.add_argument("--subtitle", default="Piano Solo (Study Edition)")
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--beat-type", type=int, default=4, choices=(4, 8), help="Time signature denominator (4 or 8)")
    ap.add_argument("--grid", type=int, default=DIVISIONS, choices=(2, 3, 4), help="Subdivisions per quarter note (2=8th, 3=triplet-8th, 4=16th)")
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--key-fifths", type=int, default=-2)
    ap.add_argument("--dynamics-csv", default=None)
    ap.add_argument("--musescore", default=None)
    ap.add_argument("--out-xml", required=True)
    ap.add_argument("--out-pdf", default=None)
    args = ap.parse_args()

    divisions, bar_slots, beat_slots, beam_group_slots = compute_grid_metrics(args.beats_per_bar, args.beat_type, args.grid)
    notes, pickup_start = load_midi_notes(Path(args.midi), divisions, bar_slots)
    if not notes:
        raise ValueError(f"No playable notes found in MIDI: {args.midi}")
    bar1_slot = bar_slots if pickup_start > 0 else 0
    total_slots = max(n.start_slot + n.dur for n in notes) + bar_slots
    slot_rms = load_slot_rms_db(Path(args.audio), args.bpm, args.beat_type, divisions, args.first_downbeat, bar1_slot, total_slots)
    split_hands(notes)
    bars_map, max_bar = slice_into_bars(notes, pickup_start, bar_slots)

    pc_table = FLAT_PC_SPELLING if args.key_fifths <= 0 else SHARP_PC_SPELLING
    k_alter = key_sig_alter(args.key_fifths)
    dyns = derive_dynamics_from_csv(Path(args.dynamics_csv) if args.dynamics_csv else None, max_bar)

    root = ET.Element("score-partwise", version="4.0")
    ident = ET.SubElement(root, "identification")
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "Piano Score Studio Engraver v2"
    for el_name in ("accidental", "beam", "stem"):
        ET.SubElement(enc, "supports", element=el_name, type="yes")
    for attr_name in ("new-page", "new-system"):
        ET.SubElement(enc, "supports", element="print", attribute=attr_name, type="yes", value="yes")

    defaults = ET.SubElement(root, "defaults")
    sc = ET.SubElement(defaults, "scaling")
    ET.SubElement(sc, "millimeters").text = "7"
    ET.SubElement(sc, "tenths").text = "40"
    pl = ET.SubElement(defaults, "page-layout")
    ET.SubElement(pl, "page-height").text = "1697.14"
    ET.SubElement(pl, "page-width").text = "1200"
    for mtype in ("even", "odd"):
        pm = ET.SubElement(pl, "page-margins", type=mtype)
        for tag, val in (("left-margin", "80"), ("right-margin", "80"), ("top-margin", "75"), ("bottom-margin", "75")):
            ET.SubElement(pm, tag).text = val
    ET.SubElement(ET.SubElement(defaults, "system-layout"), "system-distance").text = "112"
    ET.SubElement(ET.SubElement(defaults, "staff-layout"), "staff-distance").text = "82"

    cr1 = ET.SubElement(root, "credit", page="1")
    ET.SubElement(cr1, "credit-type").text = "title"
    ET.SubElement(cr1, "credit-words", {"default-x": "600", "default-y": "1615", "justify": "center", "valign": "top", "font-size": "24", "font-weight": "bold"}).text = args.title
    cr2 = ET.SubElement(root, "credit", page="1")
    ET.SubElement(cr2, "credit-type").text = "subtitle"
    ET.SubElement(cr2, "credit-words", {"default-x": "600", "default-y": "1565", "justify": "center", "valign": "top", "font-size": "13", "font-style": "italic"}).text = args.subtitle

    plist = ET.SubElement(root, "part-list")
    sp = ET.SubElement(plist, "score-part", id="P1")
    ET.SubElement(sp, "part-name", {"print-object": "no"}).text = "Piano"
    ET.SubElement(ET.SubElement(sp, "score-instrument", id="P1-I1"), "instrument-name").text = "Piano"

    part = ET.SubElement(root, "part", id="P1")
    start_bar = 0 if pickup_start > 0 else 1
    sys_count = 0

    for b in range(start_bar, max_bar + 1):
        bar_len = (bar_slots - pickup_start) if b == 0 else bar_slots
        b_global = pickup_start if b == 0 else (bar1_slot + (b - 1) * bar_slots)
        m_el = ET.SubElement(part, "measure", number=str(b))
        if b == 0:
            m_el.set("implicit", "yes")

        # 4-bar systems; 4 systems on Page 1, 5 systems on subsequent pages
        if b == start_bar:
            pr = ET.SubElement(m_el, "print")
            ET.SubElement(ET.SubElement(pr, "system-layout"), "top-system-distance").text = "165"
            ET.SubElement(ET.SubElement(pr, "staff-layout", number="2"), "staff-distance").text = "82"
        elif (b - 1) % 4 == 0 and b > 1 and (max_bar - b >= 2):
            sys_count += 1
            pr = ET.SubElement(m_el, "print")
            if sys_count in (4,) or (sys_count > 4 and (sys_count - 4) % 5 == 0):
                pr.set("new-page", "yes")
            else:
                pr.set("new-system", "yes")

        if b == start_bar:
            attr = ET.SubElement(m_el, "attributes")
            ET.SubElement(attr, "divisions").text = str(divisions)
            ET.SubElement(ET.SubElement(attr, "key"), "fifths").text = str(args.key_fifths)
            tm = ET.SubElement(attr, "time")
            ET.SubElement(tm, "beats").text = str(args.beats_per_bar)
            ET.SubElement(tm, "beat-type").text = str(args.beat_type)
            ET.SubElement(attr, "staves").text = "2"
            c1 = ET.SubElement(attr, "clef", number="1")
            ET.SubElement(c1, "sign").text = "G"; ET.SubElement(c1, "line").text = "2"
            c2 = ET.SubElement(attr, "clef", number="2")
            ET.SubElement(c2, "sign").text = "F"; ET.SubElement(c2, "line").text = "4"

            d_top = ET.SubElement(m_el, "direction", placement="above", system="only-top")
            ET.SubElement(ET.SubElement(d_top, "direction-type"), "words", {"font-weight": "bold", "font-size": "11", "relative-y": "24"}).text = "Moderato espressivo"
            met = ET.SubElement(ET.SubElement(d_top, "direction-type"), "metronome", {"parentheses": "no", "relative-y": "24", "default-x": "12"})
            ET.SubElement(met, "beat-unit").text = "eighth" if args.beat_type == 8 else "quarter"
            ET.SubElement(met, "per-minute").text = str(int(round(args.bpm)))
            ET.SubElement(d_top, "staff").text = "1"
            quarter_bpm = args.bpm * (4.0 / args.beat_type)
            ET.SubElement(d_top, "sound", tempo=str(round(quarter_bpm, 2)))

        if b == 1:
            dp = ET.SubElement(m_el, "direction", placement="below")
            ET.SubElement(ET.SubElement(dp, "direction-type"), "words", {"font-style": "italic", "font-size": "10", "relative-y": "-14"}).text = "con pedale"
            ET.SubElement(dp, "staff").text = "2"

        for _, val in dyns.get(b, []):
            dd = ET.SubElement(m_el, "direction", placement="below")
            ET.SubElement(ET.SubElement(ET.SubElement(dd, "direction-type"), "dynamics", {"relative-y": "-18"}), val)
            ET.SubElement(dd, "staff").text = "1"

        rh_v1, rh_v2 = simplify_bar_hand(bars_map.get((b, 1), []), b, b_global, bar_len, slot_rms, divisions)
        lh_v1, lh_v2 = simplify_bar_hand(bars_map.get((b, 2), []), b, b_global, bar_len, slot_rms, divisions)

        def emit_staff(v1: list[ChordEvent], v2: list[ChordEvent] | None, staff: int, v_id1: int, v_id2: int) -> None:
            acc_state: dict[tuple[str, int], int] = {}
            if not v1 and not v2:
                if b == 0:
                    for _, pd, pt, pdot, ptup in split_duration(0, bar_len, True, divisions, beat_slots):
                        n = ET.SubElement(m_el, "note")
                        ET.SubElement(n, "rest")
                        ET.SubElement(n, "duration").text = str(pd)
                        ET.SubElement(n, "voice").text = str(v_id1)
                        ET.SubElement(n, "type").text = pt
                        if pdot:
                            ET.SubElement(n, "dot")
                        if ptup:
                            tmod = ET.SubElement(n, "time-modification")
                            ET.SubElement(tmod, "actual-notes").text = str(ptup[0])
                            ET.SubElement(tmod, "normal-notes").text = str(ptup[1])
                        ET.SubElement(n, "staff").text = str(staff)
                else:
                    n = ET.SubElement(m_el, "note")
                    ET.SubElement(n, "rest", measure="yes")
                    ET.SubElement(n, "duration").text = str(bar_len)
                    ET.SubElement(n, "voice").text = str(v_id1)
                    ET.SubElement(n, "staff").text = str(staff)
                return

            def write_atoms(atoms: list[RenderedAtom], voice: int, stem: str | None) -> None:
                for at in atoms:
                    if at.is_rest:
                        n = ET.SubElement(m_el, "note")
                        if not at.print_object:
                            n.set("print-object", "no")
                        ET.SubElement(n, "rest")
                        ET.SubElement(n, "duration").text = str(at.dur)
                        ET.SubElement(n, "voice").text = str(voice)
                        ET.SubElement(n, "type").text = at.type_str
                        if at.dotted:
                            ET.SubElement(n, "dot")
                        if at.tuplet:
                            tmod = ET.SubElement(n, "time-modification")
                            ET.SubElement(tmod, "actual-notes").text = str(at.tuplet[0])
                            ET.SubElement(tmod, "normal-notes").text = str(at.tuplet[1])
                        ET.SubElement(n, "staff").text = str(staff)
                    else:
                        for idx, (midi, ts, te, arp) in enumerate(at.pitches):
                            n = ET.SubElement(m_el, "note")
                            if idx > 0:
                                ET.SubElement(n, "chord")
                            step, alter = pc_table[midi % 12]
                            octv = (midi // 12) - 1
                            pe = ET.SubElement(n, "pitch")
                            ET.SubElement(pe, "step").text = step
                            if alter != 0:
                                ET.SubElement(pe, "alter").text = str(alter)
                            ET.SubElement(pe, "octave").text = str(octv)
                            ET.SubElement(n, "duration").text = str(at.dur)
                            if ts:
                                ET.SubElement(n, "tie", type="stop")
                            if te:
                                ET.SubElement(n, "tie", type="start")
                            ET.SubElement(n, "voice").text = str(voice)
                            ET.SubElement(n, "type").text = at.type_str
                            if at.dotted:
                                ET.SubElement(n, "dot")
                            cur_st = acc_state.get((step, octv), k_alter.get(step, 0))
                            if not ts and alter != cur_st:
                                ET.SubElement(n, "accidental").text = {-1: "flat", 0: "natural", 1: "sharp"}[alter]
                                acc_state[(step, octv)] = alter
                            elif not ts:
                                acc_state[(step, octv)] = alter
                            if at.tuplet:
                                tmod = ET.SubElement(n, "time-modification")
                                ET.SubElement(tmod, "actual-notes").text = str(at.tuplet[0])
                                ET.SubElement(tmod, "normal-notes").text = str(at.tuplet[1])
                            if stem:
                                ET.SubElement(n, "stem").text = stem
                            ET.SubElement(n, "staff").text = str(staff)
                            if idx == 0 and at.beam1:
                                ET.SubElement(n, "beam", number="1").text = at.beam1
                            if idx == 0 and at.beam2:
                                ET.SubElement(n, "beam", number="2").text = at.beam2
                            if ts or te or arp:
                                nts = ET.SubElement(n, "notations")
                                if ts:
                                    ET.SubElement(nts, "tied", type="stop")
                                if te:
                                    ET.SubElement(nts, "tied", type="start")
                                if arp:
                                    ET.SubElement(nts, "arpeggiate")

            write_atoms(build_voice_atoms(v1, bar_len, v2, divisions, beat_slots, beam_group_slots), v_id1, "up" if v2 else None)
            if v2:
                ET.SubElement(ET.SubElement(m_el, "backup"), "duration").text = str(bar_len)
                write_atoms(build_voice_atoms(v2, bar_len, v1, divisions, beat_slots, beam_group_slots), v_id2, "down")

        emit_staff(rh_v1, rh_v2, 1, 1, 2)
        ET.SubElement(ET.SubElement(m_el, "backup"), "duration").text = str(bar_len)
        emit_staff(lh_v1, lh_v2, 2, 5, 6)

        if b == max_bar:
            ET.SubElement(ET.SubElement(m_el, "barline", location="right"), "bar-style").text = "light-heavy"

    out_xml = Path(args.out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
    )
    out_xml.write_text(header + ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    print(f"Stage 3 MusicXML written: {out_xml} (measures={max_bar - start_bar + 1})")

    if args.out_pdf:
        ms = find_musescore(args.musescore)
        if not ms:
            raise FileNotFoundError("MuseScore 4 binary not found; pass --musescore or set MUSCRIPTOR_MUSESCORE")
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run([ms, "-o", str(out_pdf), str(out_xml)], capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not out_pdf.is_file():
            raise RuntimeError(f"MuseScore failed: {proc.stderr or proc.stdout}")
        print(f"Stage 3 PDF rendered: {out_pdf} ({out_pdf.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
