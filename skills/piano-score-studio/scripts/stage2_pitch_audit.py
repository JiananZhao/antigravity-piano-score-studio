"""Stage 2: Deep harmonic pitch audit for minor-2nd clashes, ghost notes, and out-of-scale tones."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

STEP_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def parse_xml_onsets(xml_path: Path, bpm: float, beats_per_bar: int, first_downbeat: float) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    divisions = int(root.findtext(".//attributes/divisions", "2"))
    beat_s = 60.0 / bpm
    bar_s = beats_per_bar * beat_s

    notes = []
    for m in root.findall(".//part/measure"):
        b = int(m.attrib["number"])
        implicit = m.attrib.get("implicit") == "yes"
        # Compute bar note sum for pickup length
        cur = 0
        last_on = 0
        m_items = []
        max_end = 0
        for el in m:
            if el.tag == "backup":
                cur -= int(el.findtext("duration"))
            elif el.tag == "forward":
                cur += int(el.findtext("duration"))
            elif el.tag == "note":
                is_ch = el.find("chord") is not None
                dur = int(el.findtext("duration"))
                on = last_on if is_ch else cur
                if not is_ch:
                    last_on = cur
                    cur += dur
                    max_end = max(max_end, cur)
                p = el.find("pitch")
                if p is None:
                    continue
                if any(t.attrib.get("type") == "stop" for t in el.findall("tie")):
                    continue
                midi = (int(p.findtext("octave")) + 1) * 12 + STEP_MAP[p.findtext("step")] + int(p.findtext("alter", "0"))
                staff = int(el.findtext("staff", "1"))
                m_items.append((on, dur, midi, staff))

        bar_len_beats = max_end / divisions if implicit else float(beats_per_bar)
        t_bar_start = (first_downbeat - bar_len_beats * beat_s) if b == 0 else (first_downbeat + (b - 1) * bar_s)
        for on, dur, midi, staff in m_items:
            beat_on = on / divisions
            notes.append(
                {
                    "measure": b,
                    "beat": round(beat_on + 1.0, 2),
                    "t_on": t_bar_start + beat_on * beat_s,
                    "dur_s": (dur / divisions) * beat_s,
                    "midi": midi,
                    "name": f"{PC_NAMES[midi % 12]}{midi // 12 - 1}",
                    "staff": staff,
                }
            )
    return notes


def harmonic_evidence(mono: np.ndarray, sr: int, t_on: float, midi: int) -> dict:
    f0 = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    i0 = max(0, int((t_on - 0.01) * sr))
    i1 = min(len(mono), int((t_on + 0.14) * sr))
    ip0 = max(0, int((t_on - 0.12) * sr))
    ip1 = max(1, int((t_on - 0.02) * sr))
    ch, pre = mono[i0:i1], mono[ip0:ip1]
    if len(ch) < 128:
        return {"f0_db": -80.0, "snr_db": 0.0, "og_db": 0.0}
    spec = np.abs(np.fft.rfft(ch * np.hanning(len(ch)), n=16384)) / (len(ch) + 1e-12)
    wp = np.hanning(len(pre)) if len(pre) >= 64 else np.ones(max(1, len(pre)))
    spec_pre = np.abs(np.fft.rfft(pre * wp, n=16384)) / (len(pre) + 1e-12)
    fs = np.fft.rfftfreq(16384, 1.0 / sr)

    snrs, ogs = [], []
    f0_db = -80.0
    for h in (1, 2, 3):
        fh = f0 * h
        if fh > 6000:
            break
        bw = max(fh * 0.025, 4.0)
        m = (fs >= fh - bw) & (fs <= fh + bw)
        bg = (fs >= fh - 5 * bw) & (fs <= fh + 5 * bw) & (~m)
        pk = float(np.max(spec[m])) + 1e-12
        bg_m = float(np.median(spec[bg])) + 1e-12
        pre_pk = float(np.max(spec_pre[m])) + 1e-12
        if h == 1:
            f0_db = 20.0 * np.log10(pk)
        snrs.append(20.0 * np.log10(pk / bg_m))
        ogs.append(20.0 * np.log10(pk / pre_pk))
    return {
        "f0_db": round(f0_db, 2),
        "snr_db": round(float(np.mean(snrs)), 2),
        "og_db": round(float(np.max(ogs)), 2),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 2 pitch, clash, and ghost-note harmonic audit")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data, sr = sf.read(args.audio, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    notes = parse_xml_onsets(Path(args.xml), args.bpm, args.beats_per_bar, args.first_downbeat)

    pcs = np.bincount([n["midi"] % 12 for n in notes], minlength=12)
    scale_pcs = set(np.argsort(pcs)[-7:].tolist())

    # Group by quantized 40ms onset buckets to detect minor-2nd clashes
    by_bucket: dict[int, list[dict]] = {}
    for n in notes:
        ev = harmonic_evidence(mono, sr, n["t_on"], n["midi"])
        n.update(ev)
        by_bucket.setdefault(int(round(n["t_on"] / 0.04)), []).append(n)

    clashes = []
    confirmed_ghosts = []
    for _, grp in sorted(by_bucket.items()):
        grp_s = sorted(grp, key=lambda x: x["midi"])
        for i in range(len(grp_s)):
            for j in range(i + 1, len(grp_s)):
                if grp_s[j]["midi"] - grp_s[i]["midi"] == 1:
                    a, b = grp_s[i], grp_s[j]
                    weaker, stronger = (a, b) if a["f0_db"] < b["f0_db"] else (b, a)
                    gap_db = stronger["f0_db"] - weaker["f0_db"]
                    is_ghost = gap_db > 12.0 or weaker["og_db"] < 3.0
                    item = {
                        "measure": weaker["measure"],
                        "beat": weaker["beat"],
                        "clash_pair": [a["name"], b["name"]],
                        "weaker_note": weaker["name"],
                        "f0_gap_db": round(gap_db, 2),
                        "weaker_og_db": weaker["og_db"],
                        "verdict": "CONFIRMED_GHOST" if is_ghost else "REAL_DISSONANCE",
                    }
                    clashes.append(item)
                    if is_ghost:
                        confirmed_ghosts.append(item)

    oos_notes = [n for n in notes if (n["midi"] % 12) not in scale_pcs]
    weak_oos = [n for n in oos_notes if n["snr_db"] < 12.0 or n["og_db"] < 3.0]
    low_og_lh = [n for n in notes if n["staff"] == 2 and n["og_db"] < 3.0]

    report = {
        "stage": "stage_2_transcription_pitch_audit",
        "total_onsets": len(notes),
        "detected_scale_pcs": sorted(PC_NAMES[c] for c in scale_pcs),
        "minor2nd_clashes": clashes,
        "confirmed_ghost_count": len(confirmed_ghosts),
        "out_of_scale_count": len(oos_notes),
        "out_of_scale_checked_pct": 100.0,
        "weak_out_of_scale_count": len(weak_oos),
        "out_of_scale_notes": oos_notes,
        "lh_low_og_candidates_count": len(low_og_lh),
        "lh_low_og_candidates": low_og_lh[:30],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"Stage 2 Pitch Audit: {len(notes)} onsets | clashes={len(clashes)} "
        f"(ghosts={len(confirmed_ghosts)}) | out-of-scale={len(oos_notes)} (weak={len(weak_oos)})"
    )


if __name__ == "__main__":
    main()
