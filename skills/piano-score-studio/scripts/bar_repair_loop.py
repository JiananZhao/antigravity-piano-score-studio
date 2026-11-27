#!/usr/bin/env python3
"""v3.0 Phase 3 — Local Bar-Override Repair Loop for Agent-Driven Score Refinement.

Reads `bar_overrides` from `project_manifest.json` (or a standalone JSON file) and provides:
  1. `load_bar_overrides(path)`: loads per-bar overrides keyed by integer bar number.
  2. `apply_bar_overrides_to_segments(bars_map, overrides)`: applies local `drop_pitches`
     and `force_hand` overrides without re-running Stage 1 transcription.
  3. `propose_bar_repairs(eval_report, pitch_audit, xml_path)`: inspects failing Eval Gate
     metrics + MusicXML bars and emits targeted `bar_overrides` proposals for automatic
     local re-engraving (e.g. wide-span >12st chords, stubborn ghost clashes).
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

STEP_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def load_bar_overrides(path: str | Path | None) -> dict[int, dict]:
    """Load `bar_overrides` from either project_manifest.json or a direct overrides JSON."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    raw = data.get("bar_overrides", data) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        return {}
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            bar_num = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            out[bar_num] = v
    return out


def apply_bar_overrides_to_notes(notes: list, bar_slots: int, pickup_start: int, overrides: dict[int, dict]) -> dict[str, int]:
    """Apply bar-level `drop_pitches` and `force_hand` overrides in-place on Stage 3 RawNote list.

    Returns summary counts of applied edits.
    """
    if not overrides or not notes:
        return {"dropped": 0, "rehanded": 0}

    bar1_slot = bar_slots if pickup_start > 0 else 0
    dropped = 0
    rehanded = 0
    kept = []

    for n in notes:
        if pickup_start > 0 and n.start_slot < bar_slots:
            bar = 0
        else:
            bar = ((n.start_slot - bar1_slot) // bar_slots) + 1
        ov = overrides.get(bar)
        if ov:
            drop_set = {int(p) for p in ov.get("drop_pitches", [])}
            if int(n.pitch) in drop_set:
                dropped += 1
                continue
            force_hand = ov.get("force_hand") or {}
            # Supports {"62": 1} or {"pitch_62": 2}
            fh = force_hand.get(str(n.pitch), force_hand.get(int(n.pitch)))
            if fh in (1, 2):
                n.hand = int(fh)
                rehanded += 1
        kept.append(n)

    if dropped > 0:
        notes[:] = kept
    return {"dropped": dropped, "rehanded": rehanded}


def propose_bar_repairs(
    eval_report: dict | None,
    pitch_audit: dict | None,
    xml_path: Path | None,
) -> dict[str, dict]:
    """Inspect Eval Gate + Pitch Audit + MusicXML and propose bounded per-bar overrides."""
    proposals: dict[str, dict] = {}

    # 1. Any remaining confirmed ghost notes in pitch_audit -> drop in that specific bar
    if isinstance(pitch_audit, dict) and int(pitch_audit.get("confirmed_ghost_count", 0)) > 0:
        for cl in pitch_audit.get("minor2nd_clashes", []):
            if cl.get("verdict") == "CONFIRMED_GHOST" and "measure" in cl and "weaker_midi" in cl:
                b_key = str(int(cl["measure"]))
                entry = proposals.setdefault(b_key, {"reason": []})
                drops = entry.setdefault("drop_pitches", [])
                midi_p = int(cl["weaker_midi"])
                if midi_p not in drops:
                    drops.append(midi_p)
                entry["reason"].append(f"confirmed_ghost:{cl.get('weaker_note', midi_p)}")

    # 2. Scan MusicXML for any unmarked >12-semitone chord spans -> suggest hand-split adjustment
    if xml_path and xml_path.is_file():
        root = ET.parse(xml_path).getroot()
        for m in root.findall(".//part/measure"):
            m_num = int(m.attrib.get("number", "1"))
            cur, last_on = 0, 0
            chords: dict[tuple[str, str, int], list[tuple[int, bool]]] = {}
            for el in m:
                if el.tag == "backup":
                    cur -= int(el.findtext("duration", "0"))
                elif el.tag == "forward":
                    cur += int(el.findtext("duration", "0"))
                elif el.tag == "note":
                    v = el.findtext("voice", "1")
                    st = el.findtext("staff", "1")
                    is_ch = el.find("chord") is not None
                    dur = int(el.findtext("duration", "0"))
                    on = last_on if is_ch else cur
                    if not is_ch:
                        last_on = cur
                        cur += dur
                    p = el.find("pitch")
                    if p is not None:
                        midi_p = (int(p.findtext("octave")) + 1) * 12 + STEP_MAP[p.findtext("step")] + int(p.findtext("alter", "0"))
                        has_arp = el.find(".//notations/arpeggiate") is not None
                        chords.setdefault((st, v, on), []).append((midi_p, has_arp))
            for (st, _, _), grp in chords.items():
                if len(grp) >= 2:
                    pitches = sorted(x[0] for x in grp)
                    if (pitches[-1] - pitches[0]) > 12 and not any(x[1] for x in grp):
                        b_key = str(m_num)
                        entry = proposals.setdefault(b_key, {"reason": []})
                        fh = entry.setdefault("force_hand", {})
                        # Move extreme note to opposite staff to relieve span > 12st
                        if st == "1":
                            fh[str(pitches[0])] = 2
                        else:
                            fh[str(pitches[-1])] = 1
                        entry["reason"].append(f"wide_span_gt12:staff{st}")

    # 3. If eval_report flags high visible rests, disable gap-bridging suppression or enable bridging locally
    if isinstance(eval_report, dict):
        for st in eval_report.get("stages", []):
            for chk in st.get("checks", []):
                if chk.get("id") == "s3_r1_rests_durations" and chk.get("status") == "FAIL":
                    # Suggest enabling gap bridging on bar 1 as a diagnostic signal
                    proposals.setdefault("1", {"reason": []}).setdefault("enable_gap_bridging", True)
                    proposals["1"]["reason"].append("eval_s3_rests_fail")

    return proposals


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="v3.0 Phase 3 local bar_overrides repair analyzer")
    ap.add_argument("--manifest", required=True, help="Path to project_manifest.json")
    ap.add_argument("--eval-report", default=None, help="Optional eval_report.json")
    ap.add_argument("--pitch-audit", default=None, help="Optional pitch_audit.json")
    ap.add_argument("--xml", default=None, help="Optional study MusicXML")
    ap.add_argument("--write", action="store_true", help="Merge proposed bar_overrides into project_manifest.json")
    args = ap.parse_args()

    m_path = Path(args.manifest)
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    eval_path = Path(args.eval_report) if args.eval_report else Path(manifest.get("artifacts", {}).get("eval_report_json", ""))
    pa_path = Path(args.pitch_audit) if args.pitch_audit else Path(manifest.get("artifacts", {}).get("pitch_audit_json", ""))
    xml_path = Path(args.xml) if args.xml else Path(manifest.get("artifacts", {}).get("study_xml", ""))

    eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else None
    pa_data = json.loads(pa_path.read_text(encoding="utf-8")) if pa_path.is_file() else None
    proposals = propose_bar_repairs(eval_data, pa_data, xml_path if xml_path.is_file() else None)

    if args.write and proposals:
        existing = manifest.setdefault("bar_overrides", {})
        for b_key, ov in proposals.items():
            cur = existing.setdefault(b_key, {})
            if "drop_pitches" in ov:
                cur["drop_pitches"] = sorted(set(cur.get("drop_pitches", [])) | set(ov["drop_pitches"]))
            if "force_hand" in ov:
                cur.setdefault("force_hand", {}).update(ov["force_hand"])
            if "enable_gap_bridging" in ov:
                cur["enable_gap_bridging"] = ov["enable_gap_bridging"]
            cur["reason"] = sorted(set(cur.get("reason", [])) | set(ov.get("reason", [])))
        m_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Bar Repair Analyzer: {len(proposals)} bar(s) proposed for local override.")


if __name__ == "__main__":
    main()
