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

STEP_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def band_db(sig: np.ndarray, sr: int, t0: float, t1: float, lo: float, hi: float) -> float:
    i0, i1 = max(0, int(t0 * sr)), min(len(sig), int(t1 * sr))
    ch = sig[i0:i1]
    if len(ch) < 64:
        return -80.0
    sp = np.abs(np.fft.rfft(ch * np.hanning(len(ch))))
    fs = np.fft.rfftfreq(len(ch), 1.0 / sr)
    mk = (fs >= lo) & (fs <= hi)
    return 10.0 * np.log10(max(1e-14, float(np.sum(sp[mk] ** 2) / (len(ch) ** 2))))


def measure_tempo_lock(mono: np.ndarray, sr: int, bpm: float, first_downbeat: float) -> tuple[float, float]:
    """Compute circular resultant |R| and 20s-window max phase drift (ms) from 10ms onset spectral flux."""
    n_fft = 2048
    hop = max(1, int(round(sr * 0.01)))
    nf = (len(mono) - n_fft) // hop + 1
    if nf < 4:
        return 0.0, 999.0

    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= 80.0) & (freqs <= 5120.0)

    flux = np.zeros(nf - 1, dtype=np.float64)
    prev_log: np.ndarray | None = None
    chunk_frames = 2048
    for f0 in range(0, nf, chunk_frames):
        f1 = min(nf, f0 + chunk_frames)
        idx = np.arange(n_fft)[None, :] + np.arange(f0, f1)[:, None] * hop
        frames = mono[idx].astype(np.float32) * win[None, :]
        mag = np.abs(np.fft.rfft(frames, axis=1))[:, band]
        log_mag = np.log1p(1000.0 * mag)
        if prev_log is not None:
            flux[f0 - 1] = float(np.mean(np.maximum(0.0, log_mag[0] - prev_log)))
        if len(log_mag) > 1:
            flux[f0 : f1 - 1] = np.mean(np.maximum(0.0, np.diff(log_mag, axis=0)), axis=1)
        prev_log = log_mag[-1]

    thr = float(np.median(flux) + 1.5 * np.std(flux))
    w = np.where(flux > thr, flux - thr, 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        return 0.0, 999.0

    times = (np.arange(1, nf) * hop + 0.5 * n_fft) / float(sr)
    beat_s = 60.0 / bpm
    phi = 2.0 * np.pi * ((times - first_downbeat) / beat_s)
    z_all = np.sum(w * np.exp(1j * phi)) / w_sum
    r_val = float(np.abs(z_all))
    global_ang = float(np.angle(z_all))

    dur_s = len(mono) / float(sr)
    drifts: list[float] = []
    for t0 in range(0, max(20, int(dur_s)), 20):
        m_win = (times >= t0) & (times < t0 + 20.0)
        w_win = float(np.sum(w[m_win]))
        if w_win > 0.02 * w_sum:
            z_w = np.sum(w[m_win] * np.exp(1j * phi[m_win])) / w_win
            ang_diff = float(np.angle(z_w * np.exp(-1j * global_ang)))
            drifts.append(abs(ang_diff) / (2.0 * np.pi) * beat_s * 1000.0)
    max_drift_ms = float(max(drifts)) if drifts else 0.0
    return round(r_val, 3), round(max_drift_ms, 1)


def evaluate_s4_r1_static_dynamics(
    placeholder_only: bool,
    rh_med: float,
    lh_med: float,
    lh_max: int,
    unclosed_wedges: int,
) -> str:
    """Evaluate s4_r1_static_dynamics against eval_rubric_v2.json PASS/WARN/FAIL bands."""
    if placeholder_only or unclosed_wedges != 0:
        return "FAIL"
    diff = rh_med - lh_med
    if diff >= 15.0 and lh_max <= 78:
        return "PASS"
    if diff >= 10.0 and lh_max <= 88:
        return "WARN"
    return "FAIL"


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
    ap.add_argument("--stage1-cal", default=None, help="Optional Stage 1 stage1_calibration.json")
    ap.add_argument("--pdf-visually-inspected", action="store_true")
    ap.add_argument("--rubato-verified", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rubric = json.loads(Path(args.rubric).read_text(encoding="utf-8"))
    triage = json.loads(Path(args.triage).read_text(encoding="utf-8"))
    pitch_audit = json.loads(Path(args.pitch_audit).read_text(encoding="utf-8"))
    stage1_cal = (
        json.loads(Path(args.stage1_cal).read_text(encoding="utf-8"))
        if args.stage1_cal and Path(args.stage1_cal).is_file()
        else {}
    )

    # Stage 0 metrics
    rec_name = triage.get("recommended_filename")
    cand = next((c for c in triage["candidates"] if c["filename"] == rec_name), triage["candidates"][0])
    s0_integrity = "PASS" if (cand["clipping_count"] == 0 and cand["click_jumps_gt_0.1"] == 0 and cand["dropouts_lt_-70dB"] == 0) else (
        "WARN" if (cand["clipping_count"] <= 3 and cand["dropouts_lt_-70dB"] == 0) else "FAIL"
    )
    cancel_pct = cand["strong_harmonic_mono_cancel_gt_6dB_pct"]
    s0_mono = "PASS" if cancel_pct < 12.0 else ("WARN" if cancel_pct <= 25.0 else "FAIL")
    pad2s_verified = float(stage1_cal.get("leading_silence_pad_sec", 0.0)) >= 1.5
    s0_lead = "PASS" if (cand["first_note_sec"] >= 0.15 or pad2s_verified) else "WARN"

    # XML audit (Stage 1 & Stage 3 & Stage 4 wedges)
    tree = ET.parse(args.xml)
    root = tree.getroot()
    measures = root.findall(".//part/measure")
    divisions = int(root.findtext(".//attributes/divisions", "2"))
    beat_type = int(root.findtext(".//attributes/time/beat-type", "4"))
    full_bar_units = (args.beats_per_bar * divisions * 4) // beat_type
    beat_units = max(1, (divisions * 4) // beat_type)
    clefs = root.findall(".//clef")
    mid_clef_changes = max(0, len(clefs) - 2)
    visible_rests = sum(1 for n in root.findall(".//note") if n.find("rest") is not None and n.attrib.get("print-object") != "no")
    types = Counter(el.text for el in root.findall(".//note/type"))
    spurious_16_32 = types.get("16th", 0) + types.get("32nd", 0)
    total_syms = max(1, sum(types.values()))
    spurious_pct = 100.0 * spurious_16_32 / total_syms

    max_voices_per_hand = 1
    dur_errors = 0
    pickup_aligned = True
    unmarked_gt12 = 0
    unclosed_ties = 0
    open_ties: dict[tuple[str, int], int] = {}
    beat_onsets = [0] * args.beats_per_bar

    for m in measures:
        m_num = int(m.attrib["number"])
        is_pickup = (m_num == 0) or (m.attrib.get("implicit") == "yes")
        vdur: dict[str, int] = {}
        vh1, vh2 = set(), set()
        chords: dict[tuple[str, str, int], list[tuple[int, bool]]] = {}
        cur, last_on = 0, 0

        for el in m:
            if el.tag == "backup":
                cur -= int(el.findtext("duration", "0"))
            elif el.tag == "forward":
                cur += int(el.findtext("duration", "0"))
            elif el.tag == "note":
                v = el.findtext("voice", "1")
                st = el.findtext("staff", "1")
                (vh1 if st == "1" else vh2).add(v)
                is_ch = el.find("chord") is not None
                dur = int(el.findtext("duration", "0"))
                on = last_on if is_ch else cur
                if not is_ch:
                    last_on = cur
                    cur += dur
                    vdur[v] = vdur.get(v, 0) + dur
                p = el.find("pitch")
                if p is not None:
                    midi_p = (int(p.findtext("octave")) + 1) * 12 + STEP_MAP[p.findtext("step")] + int(p.findtext("alter", "0"))
                    has_arp = el.find(".//notations/arpeggiate") is not None
                    chords.setdefault((st, v, on), []).append((midi_p, has_arp))
                    ts = any(t.attrib.get("type") == "stop" for t in el.findall("tie"))
                    te = any(t.attrib.get("type") == "start" for t in el.findall("tie"))
                    if not ts and not is_pickup:
                        b_idx = min(args.beats_per_bar - 1, max(0, on // beat_units))
                        beat_onsets[b_idx] += 1
                    tkey = (st, midi_p)
                    if ts:
                        if open_ties.get(tkey, 0) > 0:
                            open_ties[tkey] -= 1
                        else:
                            unclosed_ties += 1
                    if te:
                        open_ties[tkey] = open_ties.get(tkey, 0) + 1

        for grp in chords.values():
            if len(grp) >= 2:
                pitches = [x[0] for x in grp]
                if (max(pitches) - min(pitches)) > 12 and not any(x[1] for x in grp):
                    unmarked_gt12 += 1

        max_voices_per_hand = max(max_voices_per_hand, len(vh1), len(vh2))
        if is_pickup:
            uniq = set(vdur.values())
            if not uniq or len(uniq) > 1 or not (0 < next(iter(uniq)) < full_bar_units):
                dur_errors += 1
                pickup_aligned = False
        else:
            if any(tot != full_bar_units for tot in vdur.values()):
                dur_errors += 1
    unclosed_ties += sum(open_ties.values())

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

    # Real Stage 1 tempo & downbeat checks (replacing hardcoded PASS)
    r_val, max_drift_ms = measure_tempo_lock(m_o, sr_o, args.bpm, args.first_downbeat)
    if r_val >= 0.70 and max_drift_ms <= 30.0:
        s1_tempo_status = "PASS"
    elif args.rubato_verified or (r_val >= 0.50 and max_drift_ms <= 60.0):
        s1_tempo_status = "WARN"
    else:
        s1_tempo_status = "FAIL"

    other_beats = beat_onsets[1:] if len(beat_onsets) > 1 else [0]
    if pickup_aligned and beat_onsets[0] > max(other_beats):
        s1_downbeat_status = "PASS"
    elif pickup_aligned and beat_onsets[0] >= float(np.mean(other_beats)):
        s1_downbeat_status = "WARN"
    else:
        s1_downbeat_status = "FAIL"

    if unmarked_gt12 > 0 or unclosed_ties > 0:
        s3_play_status = "FAIL"
    elif spurious_16_32 == 0:
        s3_play_status = "PASS"
    elif spurious_16_32 <= 5:
        s3_play_status = "WARN"
    else:
        s3_play_status = "FAIL"

    pdf_ok = Path(args.pdf).is_file() and Path(args.pdf).stat().st_size > 1000
    plan_ok = Path(args.dynamics_plan).is_file() and Path(args.dynamics_csv).is_file()
    ab_ok = bool(args.ab_dir) and Path(args.ab_dir).is_dir() and any(Path(args.ab_dir).glob("AB_*.wav"))

    statuses = {
        "s0_audio_integrity": (s0_integrity, {"clips": cand["clipping_count"], "jumps": cand["click_jumps_gt_0.1"], "dropouts": cand["dropouts_lt_-70dB"]}),
        "s0_mono_coherence": (s0_mono, {"mono_cancel_gt6dB_pct": cancel_pct, "selected": rec_name}),
        "s0_leading_note": (s0_lead, {"first_note_sec": cand["first_note_sec"], "pad2s_verified": pad2s_verified}),
        "s1_tempo_lock": (s1_tempo_status, {"bpm": args.bpm, "circular_R": r_val, "max_drift_ms": max_drift_ms}),
        "s1_downbeat_meter": (s1_downbeat_status, {"beats_per_bar": args.beats_per_bar, "beat_type": beat_type, "beat_onsets": beat_onsets, "pickup_aligned": pickup_aligned}),
        "s1_grid_cleanliness": ("PASS" if spurious_pct < 2.0 else ("WARN" if spurious_pct <= 8.0 else "FAIL"), {"spurious_16_32_pct": round(spurious_pct, 2)}),
        "s2_clash_ghost_audit": ("PASS" if pitch_audit["confirmed_ghost_count"] == 0 else "FAIL", {"confirmed_ghosts": pitch_audit["confirmed_ghost_count"]}),
        "s2_out_of_scale_verify": ("PASS" if pitch_audit["out_of_scale_checked_pct"] == 100.0 else "FAIL", {"oos_count": pitch_audit["out_of_scale_count"], "weak_oos": pitch_audit["weak_out_of_scale_count"]}),
        "s3_r1_clefs_voices": ("PASS" if (mid_clef_changes == 0 and max_voices_per_hand <= 2) else ("WARN" if mid_clef_changes <= 4 and max_voices_per_hand <= 2 else "FAIL"), {"mid_clef_changes": mid_clef_changes, "max_voices_per_hand": max_voices_per_hand}),
        "s3_r1_rests_durations": ("PASS" if (dur_errors == 0 and visible_rests <= 100) else ("WARN" if dur_errors == 0 and visible_rests <= 160 else "FAIL"), {"dur_errors": dur_errors, "visible_rests": visible_rests}),
        "s3_r1_playability_notation": (s3_play_status, {"unmarked_gt12": unmarked_gt12, "unclosed_ties": unclosed_ties, "spurious_16_32": spurious_16_32}),
        "s3_r2_pdf_visual": ("PASS" if (pdf_ok and args.pdf_visually_inspected) else ("WARN" if pdf_ok else "FAIL"), {"pdf_exists": pdf_ok, "visually_inspected": args.pdf_visually_inspected}),
        "s4_plan_before_exec": ("PASS" if plan_ok else "FAIL", {"plan_md": Path(args.dynamics_plan).is_file(), "envelope_csv": Path(args.dynamics_csv).is_file()}),
        "s4_r1_static_dynamics": (evaluate_s4_r1_static_dynamics(placeholder_only, rh_med, lh_med, lh_max, unclosed_wedges), {"rh_med": rh_med, "lh_med": lh_med, "lh_max": lh_max, "unclosed_wedges": unclosed_wedges}),
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