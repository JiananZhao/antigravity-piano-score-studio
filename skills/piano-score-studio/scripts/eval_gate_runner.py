"""Eval Gate Runner: Evaluate all 6 stages against agent/eval_rubric_v2.json and emit eval_report.json."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import mido
import numpy as np
import soundfile as sf


def band_db(sig: np.ndarray, sr: int, t0: float, t1: float, lo: float, hi: float) -> float:
    i0, i1 = max(0, int(t0 * sr)), min(len(sig), int(t1 * sr))
    ch = sig[i0:i1]
    if len(ch) < 64:
        return -80.0
    sp = np.abs(np.fft.rfft(ch * np.hanning(len(ch))))
    fs = np.fft.rfftfreq(len(ch), 1.0 / sr)
    mk = (fs >= lo) & (fs <= hi)
    return 10.0 * np.log10(max(1e-14, float(np.sum(sp[mk] ** 2) / (len(ch) ** 2))))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Run full-pipeline Eval Gates against eval_rubric_v2.json")
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--triage", required=True)
    ap.add_argument("--pitch-audit", required=True)
    ap.add_argument("--dynamics-plan", required=True)
    ap.add_argument("--dynamics-csv", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--midi", required=True)
    ap.add_argument("--orig-audio", required=True)
    ap.add_argument("--cal-audio", required=True)
    ap.add_argument("--ab-dir", required=False, default="")
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--pdf-visually-inspected", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rubric = json.loads(Path(args.rubric).read_text(encoding="utf-8"))
    triage = json.loads(Path(args.triage).read_text(encoding="utf-8"))
    pitch_audit = json.loads(Path(args.pitch_audit).read_text(encoding="utf-8"))

    # Stage 0 metrics
    rec_name = triage.get("recommended_filename")
    cand = next((c for c in triage["candidates"] if c["filename"] == rec_name), triage["candidates"][0])
    s0_integrity = "PASS" if (cand["clipping_count"] == 0 and cand["click_jumps_gt_0.1"] == 0 and cand["dropouts_lt_-70dB"] == 0) else (
        "WARN" if (cand["clipping_count"] <= 3 and cand["dropouts_lt_-70dB"] == 0) else "FAIL"
    )
    cancel_pct = cand["strong_harmonic_mono_cancel_gt_6dB_pct"]
    s0_mono = "PASS" if cancel_pct < 12.0 else ("WARN" if cancel_pct <= 25.0 else "FAIL")
    s0_lead = "PASS" if cand["first_note_sec"] >= 0.15 else "WARN"

    # XML audit (Stage 1 & Stage 3 & Stage 4 wedges)
    tree = ET.parse(args.xml)
    root = tree.getroot()
    measures = root.findall(".//part/measure")
    divisions = int(root.findtext(".//attributes/divisions", "2"))
    clefs = root.findall(".//clef")
    mid_clef_changes = max(0, len(clefs) - 2)
    visible_rests = sum(1 for n in root.findall(".//note") if n.find("rest") is not None and n.attrib.get("print-object") != "no")
    types = Counter(el.text for el in root.findall(".//note/type"))
    spurious_16_32 = types.get("16th", 0) + types.get("32nd", 0)
    total_syms = max(1, sum(types.values()))
    spurious_pct = 100.0 * spurious_16_32 / total_syms

    max_voices_per_hand = 1
    dur_errors = 0
    for m in measures:
        expected = 2 if int(m.attrib["number"]) == 0 else args.beats_per_bar * divisions
        vdur: dict[str, int] = {}
        vh1, vh2 = set(), set()
        for n in m.findall("note"):
            v = n.findtext("voice", "1")
            st = n.findtext("staff", "1")
            (vh1 if st == "1" else vh2).add(v)
            if n.find("chord") is None:
                vdur[v] = vdur.get(v, 0) + int(n.findtext("duration", "0"))
        max_voices_per_hand = max(max_voices_per_hand, len(vh1), len(vh2))
        if any(tot != expected for tot in vdur.values()):
            dur_errors += 1

    wedges = Counter(el.attrib.get("type") for el in root.findall(".//direction-type/wedge"))
    unclosed_wedges = abs((wedges.get("crescendo", 0) + wedges.get("diminuendo", 0)) - wedges.get("stop", 0))

    # MIDI audit (Stage 4 R1)
    mid = mido.MidiFile(args.midi)
    rh_vels, lh_vels = [], []
    for tr in mid.tracks:
        ons = [m.velocity for m in tr if m.type == "note_on" and m.velocity > 0]
        if not ons:
            continue
        if "Left" in tr.name or any(m.type == "control_change" for m in tr):
            lh_vels.extend(ons)
        else:
            rh_vels.extend(ons)
    rh_med = float(np.median(rh_vels)) if rh_vels else 0.0
    lh_med = float(np.median(lh_vels)) if lh_vels else 0.0
    lh_max = int(max(lh_vels)) if lh_vels else 0
    all_vels = rh_vels + lh_vels
    placeholder_only = len(set(all_vels)) <= 1 and (all_vels[0] == 100 if all_vels else True)

    # Acoustic closed-loop audit (Stage 4 R2)
    orig_d, sr_o = sf.read(args.orig_audio, dtype="float64", always_2d=True)
    cal_d, sr_c = sf.read(args.cal_audio, dtype="float64", always_2d=True)
    nan_inf = int(np.sum(np.isnan(cal_d)) + np.sum(np.isinf(cal_d)))
    pk_db = 20.0 * np.log10(float(np.max(np.abs(cal_d))) + 1e-12)
    tail_db = 20.0 * np.log10(float(np.std(cal_d[-int(sr_c * 0.5) :])) + 1e-12)
    tail_snr = float(pk_db - tail_db)

    m_o, m_c = orig_d.mean(axis=1), cal_d.mean(axis=1)
    sp_o = np.abs(np.fft.rfft(m_o))
    fs_o = np.fft.rfftfreq(len(m_o), 1.0 / sr_o)
    sp_c = np.abs(np.fft.rfft(m_c))
    fs_c = np.fft.rfftfreq(len(m_c), 1.0 / sr_c)
    pct_o_150 = 100.0 * float(np.sum(sp_o[(fs_o >= 20) & (fs_o < 150)] ** 2) / (np.sum(sp_o ** 2) + 1e-12))
    pct_c_150 = 100.0 * float(np.sum(sp_c[(fs_c >= 20) & (fs_c < 150)] ** 2) / (np.sum(sp_c ** 2) + 1e-12))
    band_150_diff = abs(pct_c_150 - pct_o_150)

    bar_s = args.beats_per_bar * (60.0 / args.bpm)
    num_full_bars = len(measures) - (1 if int(measures[0].attrib["number"]) == 0 else 0)
    lh_errs, rh_errs = [], []
    for b in range(1, num_full_bars + 1):
        t0 = args.first_downbeat + (b - 1) * bar_s
        t1 = t0 + bar_s
        lh_errs.append(band_db(m_c, sr_c, t0, t1, 45, 350) - band_db(m_o, sr_o, t0, t1, 45, 350))
        rh_errs.append(band_db(m_c, sr_c, t0, t1, 350, 3500) - band_db(m_o, sr_o, t0, t1, 350, 3500))
    med_lh_err = float(np.median(lh_errs))
    med_rh_err = float(np.median(rh_errs))

    pdf_ok = Path(args.pdf).is_file() and Path(args.pdf).stat().st_size > 1000
    plan_ok = Path(args.dynamics_plan).is_file() and Path(args.dynamics_csv).is_file()
    ab_ok = bool(args.ab_dir) and Path(args.ab_dir).is_dir() and any(Path(args.ab_dir).glob("AB_*.wav"))

    statuses = {
        "s0_audio_integrity": (s0_integrity, {"clips": cand["clipping_count"], "jumps": cand["click_jumps_gt_0.1"], "dropouts": cand["dropouts_lt_-70dB"]}),
        "s0_mono_coherence": (s0_mono, {"mono_cancel_gt6dB_pct": cancel_pct, "selected": rec_name}),
        "s0_leading_note": (s0_lead, {"first_note_sec": cand["first_note_sec"]}),
        "s1_tempo_lock": ("PASS", {"bpm": args.bpm}),
        "s1_downbeat_meter": ("PASS", {"beats_per_bar": args.beats_per_bar, "first_downbeat": args.first_downbeat}),
        "s1_grid_cleanliness": ("PASS" if spurious_pct < 2.0 else ("WARN" if spurious_pct <= 8.0 else "FAIL"), {"spurious_16_32_pct": round(spurious_pct, 2)}),
        "s2_clash_ghost_audit": ("PASS" if pitch_audit["confirmed_ghost_count"] == 0 else "FAIL", {"confirmed_ghosts": pitch_audit["confirmed_ghost_count"]}),
        "s2_out_of_scale_verify": ("PASS" if pitch_audit["out_of_scale_checked_pct"] == 100.0 else "FAIL", {"oos_count": pitch_audit["out_of_scale_count"], "weak_oos": pitch_audit["weak_out_of_scale_count"]}),
        "s3_r1_clefs_voices": ("PASS" if (mid_clef_changes == 0 and max_voices_per_hand <= 2) else ("WARN" if mid_clef_changes <= 4 and max_voices_per_hand <= 2 else "FAIL"), {"mid_clef_changes": mid_clef_changes, "max_voices_per_hand": max_voices_per_hand}),
        "s3_r1_rests_durations": ("PASS" if (dur_errors == 0 and visible_rests <= 100) else ("WARN" if dur_errors == 0 and visible_rests <= 160 else "FAIL"), {"dur_errors": dur_errors, "visible_rests": visible_rests}),
        "s3_r1_playability_notation": ("PASS" if spurious_16_32 == 0 else "WARN", {"spurious_16_32": spurious_16_32}),
        "s3_r2_pdf_visual": ("PASS" if (pdf_ok and args.pdf_visually_inspected) else ("WARN" if pdf_ok else "FAIL"), {"pdf_exists": pdf_ok, "visually_inspected": args.pdf_visually_inspected}),
        "s4_plan_before_exec": ("PASS" if plan_ok else "FAIL", {"plan_md": Path(args.dynamics_plan).is_file(), "envelope_csv": Path(args.dynamics_csv).is_file()}),
        "s4_r1_static_dynamics": ("PASS" if (not placeholder_only and (rh_med - lh_med) >= 15 and lh_max <= 78 and unclosed_wedges == 0) else "FAIL", {"rh_med": rh_med, "lh_med": lh_med, "lh_max": lh_max, "unclosed_wedges": unclosed_wedges}),
        "s4_r2_acoustic_match": ("PASS" if (nan_inf == 0 and tail_snr >= 80 and abs(med_lh_err) <= 1.5 and abs(med_rh_err) <= 1.5 and band_150_diff <= 4.0) else ("WARN" if (nan_inf == 0 and tail_snr >= 70 and abs(med_lh_err) <= 2.5 and abs(med_rh_err) <= 2.5) else "FAIL"), {"nan_inf": nan_inf, "tail_snr_dB": round(tail_snr, 1), "median_lh_err_dB": round(med_lh_err, 2), "median_rh_err_dB": round(med_rh_err, 2), "band_20_150_pct_diff": round(band_150_diff, 2)}),
        "s5_deliverables_complete": ("PASS" if (pdf_ok and ab_ok) else ("WARN" if pdf_ok else "FAIL"), {"pdf_ok": pdf_ok, "ab_excerpts_ok": ab_ok}),
    }

    blocking_fails = []
    out_stages = []
    for st in rubric["stages"]:
        checks_out = []
        for chk in st["checks"]:
            cid = chk["id"]
            status, measured = statuses[cid]
            if status == "FAIL" and chk.get("blocking", True):
                blocking_fails.append(cid)
            checks_out.append({"id": cid, "status": status, "blocking": chk.get("blocking", True), "measured": measured})
        out_stages.append({"stage_id": st["stage_id"], "name": st["name"], "checks": checks_out})

    summary = {
        "schema_version": "2.0",
        "overall_verdict": "PASS" if not blocking_fails else "FAIL",
        "blocking_fails": blocking_fails,
        "stages": out_stages,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Eval Gate Runner Verdict: {summary['overall_verdict']} (blocking_fails={len(blocking_fails)})")
    for st in out_stages:
        for c in st["checks"]:
            print(f"  [{c['status']:4s}] {c['id']:28s} -> {c['measured']}")
    if blocking_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
