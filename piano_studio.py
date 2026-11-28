#!/usr/bin/env python3
"""Piano Score Studio (v3.0) — One-Command Solo Piano Audio-to-Score & Calibrated Render CLI.

Usage:
    python piano_studio.py input.wav -o output/my_song
    python piano_studio.py input_dir/ -o output/my_song --engine transkun --skip-render
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "skills" / "piano-score-studio" / "scripts"
REFS_DIR = ROOT / "skills" / "piano-score-studio" / "references"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_manifest import (  # noqa: E402
    compute_run_params_hash,
    create_manifest,
    invalidate_stale_stages,
    is_stage_done,
    load_manifest,
    save_manifest,
    set_stage_status,
)


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return cleaned or "PianoPiece"


def _run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== [{label}] ===")
    print("  $", " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(f"Stage '{label}' failed with exit code {proc.returncode}.")


def _count_full_bars_from_xml(xml_path: Path) -> int:
    root = ET.parse(xml_path).getroot()
    measures = root.findall(".//part/measure")
    if not measures:
        return 1
    return max(1, len(measures) - (1 if int(measures[0].attrib.get("number", "1")) == 0 else 0))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        prog="piano_studio.py",
        description="One-command solo piano audio-to-score (PDF/MusicXML) & calibrated Salamander audio pipeline.",
    )
    ap.add_argument("inputs", nargs="+", help="Input solo piano audio file(s) or directory (WAV/MP3/FLAC)")
    ap.add_argument("-o", "--out-dir", default=None, help="Output project directory (default: output/<song_name>)")
    ap.add_argument("--title", default=None, help="Score title printed on Page 1")
    ap.add_argument("--song-name", default=None, help="File stem for deliverables (default: derived from input)")
    ap.add_argument("--engine", default="auto", choices=("auto", "transkun", "muscriptor"), help="Transcription engine (auto prefers MIT TransKun v2)")
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"), help="Inference device")
    ap.add_argument("--mode", default="auto", choices=("auto", "fixed-tempo", "tempo-map"), help="Tempo alignment mode")
    ap.add_argument("--bpm", type=float, default=None, help="Optional manual beat-unit BPM override")
    ap.add_argument("--beats-per-bar", type=int, default=None, help="Optional manual beats-per-bar override")
    ap.add_argument("--beat-type", type=int, default=None, choices=(4, 8), help="Optional time-signature denominator")
    ap.add_argument("--first-downbeat", type=float, default=None, help="Optional Bar 1 downbeat timestamp (sec)")
    ap.add_argument("--grid", type=int, default=None, choices=(2, 3, 4), help="Optional quantization grid (2/3/4)")
    ap.add_argument("--style", default="auto", choices=("auto", "pop_ballad", "classical_romantic", "baroque_polyphonic", "jazz_groove"), help="Style profile (default: auto via 6-dim Musical DNA)")
    ap.add_argument("--key-fifths", type=int, default=None, help="Optional manual key signature fifths (-7..+7); auto-detected via Krumhansl-Schmuckler when omitted")
    ap.add_argument("--config", default=None, help="Optional config.json path")
    ap.add_argument("--musescore", default=None, help="Optional MuseScore 4 binary path")
    ap.add_argument("--analyze-only", action="store_true", help="Stop after Stage 0 triage + Stage 1 calibration")
    ap.add_argument("--skip-pdf", action="store_true", help="Skip MuseScore PDF export (MusicXML only)")
    ap.add_argument("--skip-render", action="store_true", help="Skip Stage 4 Salamander audio render & Stage 5 A/B")
    ap.add_argument("--resume", action="store_true", help="Resume from existing project_manifest.json in --out-dir")
    ap.add_argument("--auto-repair", action="store_true", help="Run 1 bounded local bar_overrides repair pass if Eval Gate flags repairable bar issues")
    args = ap.parse_args()

    first_in = Path(args.inputs[0])
    default_stem = first_in.stem if first_in.is_file() else first_in.name
    song_name = _slugify(args.song_name or default_stem)
    title = args.title or default_stem.replace("_", " ").replace("-", " ").strip() or "Piano Piece"
    out_dir = Path(args.out_dir) if args.out_dir else (ROOT / "output" / song_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_params = {
        "inputs": [str(Path(x).resolve()) for x in args.inputs],
        "engine": args.engine,
        "device": args.device,
        "bpm": args.bpm,
        "time_sig": (args.beats_per_bar, args.beat_type),
        "tempo_mode": args.mode,
        "first_downbeat": args.first_downbeat,
        "subdivision": args.grid,
        "style": args.style,
        "key_fifths": args.key_fifths,
        "title": title,
        "skip_pdf": bool(args.skip_pdf),
        "auto_repair": bool(args.auto_repair),
    }

    manifest_path = out_dir / "project_manifest.json"
    if args.resume and manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        invalidated = invalidate_stale_stages(manifest, run_params)
        if invalidated:
            print(f"Resuming from {manifest_path} (parameters changed; invalidating: {', '.join(invalidated)})")
            save_manifest(manifest_path, manifest)
        else:
            print(f"Resuming from {manifest_path} (params hash={manifest.get('run_params_hash')})")
    else:
        manifest = create_manifest(
            inputs=args.inputs,
            out_dir=out_dir,
            song_name=song_name,
            title=title,
            engine=args.engine,
            device=args.device,
        )
        manifest["run_params"] = run_params
        manifest["run_params_hash"] = compute_run_params_hash(run_params)
        save_manifest(manifest_path, manifest)

    py = sys.executable
    analysis_dir = out_dir / "analysis"
    baseline_dir = out_dir / "baseline"
    score_dir = out_dir / "score"
    deliverables_dir = out_dir / "deliverables"

    triage_json = analysis_dir / "audio_triage.json"
    cal_json = baseline_dir / "stage1_calibration.json"
    pitch_audit_json = analysis_dir / "pitch_audit.json"
    cleaned_mid = baseline_dir / "score_cleaned.mid"
    dynamics_md = analysis_dir / "dynamics_plan.md"
    dynamics_csv = analysis_dir / "bar_lh_rh_envelope.csv"
    study_xml = score_dir / "score_learning.musicxml"
    study_pdf = score_dir / "score_learning.pdf"

    # Stage 0: Audio Triage & Source Selection
    if not (args.resume and is_stage_done(manifest, "stage0_triage") and triage_json.is_file()):
        _run_step(
            "Stage 0: Audio Triage",
            [py, str(SCRIPTS_DIR / "stage0_audio_triage.py"), "--inputs", *args.inputs, "--out", str(triage_json)],
        )
        triage_data = json.loads(triage_json.read_text(encoding="utf-8"))
        rec_name = triage_data.get("recommended_filename")
        cand = next((c for c in triage_data["candidates"] if c["filename"] == rec_name), triage_data["candidates"][0])
        manifest["stage0_triage"] = {
            "recommended_source": triage_data["recommended_source"],
            "recommended_filename": rec_name,
            "first_note_sec": cand["first_note_sec"],
            "duration_sec": cand["duration_sec"],
            "report_path": str(triage_json),
        }
        manifest["artifacts"]["triage_json"] = str(triage_json)
        set_stage_status(manifest, "stage0_triage", "done")
        save_manifest(manifest_path, manifest)

    best_audio = Path(manifest["stage0_triage"]["recommended_source"])

    # Stage 1: Beat Tracking, Calibration & Transcription
    if not (args.resume and is_stage_done(manifest, "stage1_calibrate_transcribe") and cal_json.is_file()):
        s1_cmd = [
            py,
            str(SCRIPTS_DIR / "stage1_tempo_and_transcribe.py"),
            "--audio",
            str(best_audio),
            "--triage",
            str(triage_json),
            "--out-dir",
            str(baseline_dir),
            "--engine",
            args.engine,
            "--device",
            args.device,
            "--mode",
            args.mode,
        ]
        if args.config:
            s1_cmd.extend(["--config", args.config])
        if args.musescore:
            s1_cmd.extend(["--musescore", args.musescore])
        if args.bpm is not None:
            s1_cmd.extend(["--bpm", str(args.bpm)])
        if args.beats_per_bar is not None:
            s1_cmd.extend(["--beats-per-bar", str(args.beats_per_bar)])
        if args.beat_type is not None:
            s1_cmd.extend(["--beat-type", str(args.beat_type)])
        if args.first_downbeat is not None:
            s1_cmd.extend(["--first-downbeat", str(args.first_downbeat)])
        if args.grid is not None:
            s1_cmd.extend(["--grid", str(args.grid)])
        if args.analyze_only:
            s1_cmd.append("--analyze-only")
        if args.skip_pdf:
            s1_cmd.append("--skip-musicxml")
        _run_step("Stage 1: Tempo/Meter Calibration & Transcription", s1_cmd)

        cal_data = json.loads(cal_json.read_text(encoding="utf-8"))
        cal = cal_data["calibration"]
        manifest["engines"]["transcription_engine_used"] = cal_data.get("transcription_engine", args.engine)
        manifest["stage1_calibration"] = {
            "bpm": cal["bpm"],
            "quarter_bpm": cal["quarter_bpm"],
            "beats_per_bar": cal["beats_per_bar"],
            "beat_type": cal["beat_type"],
            "first_downbeat": cal["first_downbeat"],
            "grid": cal["grid"],
            "tempo_mode": cal["tempo_mode"],
            "key_fifths": args.key_fifths,
            "report_path": str(cal_json),
        }
        manifest["artifacts"].update(cal_data.get("outputs") or {})
        manifest["artifacts"]["stage1_cal_json"] = str(cal_json)
        set_stage_status(manifest, "stage1_calibrate_transcribe", "done")
        save_manifest(manifest_path, manifest)

    if args.analyze_only:
        print(f"\nAnalyze-only complete. Manifest saved to: {manifest_path}")
        return

    s1 = manifest["stage1_calibration"]
    bpm = str(s1["bpm"])
    bpb = str(s1["beats_per_bar"])
    btype = str(s1["beat_type"])
    t0 = str(s1["first_downbeat"])
    grid = str(s1["grid"])
    baseline_mid = Path(manifest["artifacts"]["score_mid"])
    baseline_xml = manifest["artifacts"].get("score_musicxml")

    # Stage 1.5: Auto Key Detection (Krumhansl-Schmuckler) & 6-Dim Musical DNA Style Analysis
    style_json = analysis_dir / "style_analysis.json"
    if not (args.resume and is_stage_done(manifest, "stage1_5_style_analysis") and style_json.is_file()):
        _run_step(
            "Stage 1.5: Key Detection & 6-Dim Musical DNA Style Analysis",
            [
                py,
                str(SCRIPTS_DIR / "style_analyzer.py"),
                "--midi",
                str(baseline_mid),
                "--stage1-cal",
                str(cal_json),
                "--beats-per-bar",
                bpb,
                "--beat-type",
                btype,
                "--style",
                args.style,
                "--out",
                str(style_json),
            ],
        )
        st_data = json.loads(style_json.read_text(encoding="utf-8"))
        manifest["style_profile"] = st_data["selected_profile"]
        manifest["style_profile_path"] = st_data["profile_path"]
        manifest["musical_dna"] = st_data["musical_dna"]
        manifest["detected_key"] = st_data["detected_key"]
        manifest["artifacts"]["style_analysis_json"] = str(style_json)
        set_stage_status(manifest, "stage1_5_style_analysis", "done")
        save_manifest(manifest_path, manifest)

    style_profile_arg = manifest.get("style_profile_path") or manifest.get("style_profile") or "pop_ballad"
    resolved_key_fifths = args.key_fifths if args.key_fifths is not None else int((manifest.get("detected_key") or {}).get("key_fifths", 0))
    key_fifths = str(resolved_key_fifths)

    # Stage 2: Harmonic Pitch & Ghost Audit (+ automatic --clean-midi)
    # P1-5: Reads score.mid + TempoMap directly without MuseScore MIDI->MusicXML round-trip
    midi_for_stage3 = baseline_mid
    if baseline_mid.is_file():
        if not (args.resume and is_stage_done(manifest, "stage2_pitch_audit") and pitch_audit_json.is_file()):
            s2_cmd = [
                py,
                str(SCRIPTS_DIR / "stage2_pitch_audit.py"),
                "--audio",
                str(best_audio),
                "--midi",
                str(baseline_mid),
                "--bpm",
                bpm,
                "--beats-per-bar",
                bpb,
                "--beat-type",
                btype,
                "--first-downbeat",
                t0,
                "--stage1-cal",
                str(cal_json),
                "--clean-midi",
                str(cleaned_mid),
                "--style-profile",
                str(style_profile_arg),
                "--key-fifths",
                key_fifths,
                "--out",
                str(pitch_audit_json),
            ]
            raw_mid_for_s2 = manifest["artifacts"].get("score_raw_mid")
            if raw_mid_for_s2 and Path(raw_mid_for_s2).is_file():
                s2_cmd.extend(["--raw-midi", str(raw_mid_for_s2)])
            if baseline_xml and Path(baseline_xml).is_file():
                s2_cmd.extend(["--xml", str(baseline_xml)])
            _run_step("Stage 2: Harmonic Pitch Audit & Ghost Cleaning", s2_cmd)
            manifest["artifacts"]["pitch_audit_json"] = str(pitch_audit_json)
            manifest["artifacts"]["cleaned_mid"] = str(cleaned_mid)
            set_stage_status(manifest, "stage2_pitch_audit", "done")
            save_manifest(manifest_path, manifest)
        if cleaned_mid.is_file():
            midi_for_stage3 = cleaned_mid
    else:
        print("[SKIPPED] Stage 2 Pitch Audit: baseline score.mid not found")
        set_stage_status(manifest, "stage2_pitch_audit", "skipped")
        save_manifest(manifest_path, manifest)

    # Stage 4.1: Two-Band Dynamics Plan (run before Stage 3 so dynamics embed into Study Score)
    est_bars = max(4, int(round((manifest["stage0_triage"]["duration_sec"] - float(t0)) / (float(bpb) * 60.0 / float(bpm)))))
    if not (args.resume and is_stage_done(manifest, "stage4_1_dynamics_plan") and dynamics_csv.is_file()):
        _run_step(
            "Stage 4.1: Two-Band Dynamics Plan",
            [
                py,
                str(SCRIPTS_DIR / "stage4_dynamics_plan.py"),
                "--audio",
                str(best_audio),
                "--bpm",
                bpm,
                "--beats-per-bar",
                bpb,
                "--beat-type",
                btype,
                "--first-downbeat",
                t0,
                "--stage1-cal",
                str(cal_json),
                "--num-bars",
                str(est_bars),
                "--out-dir",
                str(analysis_dir),
            ],
        )
        manifest["artifacts"]["dynamics_plan_md"] = str(dynamics_md)
        manifest["artifacts"]["dynamics_csv"] = str(dynamics_csv)
        set_stage_status(manifest, "stage4_1_dynamics_plan", "done")
        save_manifest(manifest_path, manifest)

    # Stage 3: Study Score Engraving (MusicXML + optional MuseScore 4 PDF)
    if not (args.resume and is_stage_done(manifest, "stage3_engrave_score") and study_xml.is_file()):
        s3_cmd = [
            py,
            str(SCRIPTS_DIR / "stage3_engrave_study_score.py"),
            "--midi",
            str(midi_for_stage3),
            "--audio",
            str(best_audio),
            "--title",
            title,
            "--bpm",
            bpm,
            "--beats-per-bar",
            bpb,
            "--beat-type",
            btype,
            "--grid",
            grid,
            "--first-downbeat",
            t0,
            "--stage1-cal",
            str(cal_json),
            "--key-fifths",
            key_fifths,
            "--dynamics-csv",
            str(dynamics_csv),
            "--style-profile",
            str(style_profile_arg),
            "--bar-overrides",
            str(manifest_path),
            "--out-xml",
            str(study_xml),
        ]
        if not args.skip_pdf:
            s3_cmd.extend(["--out-pdf", str(study_pdf)])
        if args.musescore:
            s3_cmd.extend(["--musescore", args.musescore])
        _run_step("Stage 3: Study Score Engraving", s3_cmd)
        manifest["artifacts"]["study_xml"] = str(study_xml)
        if not args.skip_pdf and study_pdf.is_file():
            manifest["artifacts"]["study_pdf"] = str(study_pdf)
        set_stage_status(manifest, "stage3_engrave_score", "done")
        save_manifest(manifest_path, manifest)

    if args.skip_render:
        set_stage_status(manifest, "stage4_render_audio", "skipped")
        set_stage_status(manifest, "stage5_ab_and_eval", "skipped")
        save_manifest(manifest_path, manifest)
        print(f"\nScore engraving complete (--skip-render). Manifest: {manifest_path}")
        return

    # Stage 4.2-4.4: Calibrated Salamander Grand Piano Render
    cal_mid = deliverables_dir / f"{song_name}_Full_Calibrated_Salamander.mid"
    cal_wav = deliverables_dir / f"{song_name}_Full_Calibrated_Salamander.wav"
    if not (args.resume and is_stage_done(manifest, "stage4_render_audio") and cal_wav.is_file()):
        s4_cmd = [
            py,
            str(SCRIPTS_DIR / "stage4_render_calibrated.py"),
            "--xml",
            str(study_xml),
            "--orig-audio",
            str(best_audio),
            "--bpm",
            bpm,
            "--beats-per-bar",
            bpb,
            "--first-downbeat",
            t0,
            "--stage1-cal",
            str(cal_json),
            "--key-fifths",
            key_fifths,
            "--song-name",
            song_name,
            "--style-profile",
            str(style_profile_arg),
            "--transkun-midi",
            str(Path(manifest["artifacts"].get("score_raw_mid") or baseline_mid)),
            "--out-dir",
            str(deliverables_dir),
        ]
        if args.config:
            s4_cmd.extend(["--config", args.config])
        _run_step("Stage 4: Calibrated Salamander Render", s4_cmd)
        manifest["artifacts"]["calibrated_mid"] = str(cal_mid)
        manifest["artifacts"]["calibrated_wav"] = str(cal_wav)
        cal_mp3 = deliverables_dir / f"{song_name}_Full_Calibrated_Salamander.mp3"
        if cal_mp3.is_file():
            manifest["artifacts"]["calibrated_mp3"] = str(cal_mp3)
        set_stage_status(manifest, "stage4_render_audio", "done")
        save_manifest(manifest_path, manifest)

    # Stage 5: A/B Excerpts + Eval Gate Runner
    ab_dir = deliverables_dir / "ab_excerpts"
    eval_json = deliverables_dir / "eval_report.json"
    if not (args.resume and is_stage_done(manifest, "stage5_ab_and_eval") and eval_json.is_file()):
        _run_step(
            "Stage 5.1: A/B Comparison Excerpts",
            [
                py,
                str(SCRIPTS_DIR / "stage5_ab_excerpts.py"),
                "--orig",
                str(best_audio),
                "--calibrated",
                str(cal_wav),
                "--bpm",
                bpm,
                "--beats-per-bar",
                bpb,
                "--first-downbeat",
                t0,
                "--out-dir",
                str(ab_dir),
            ],
        )
        manifest["artifacts"]["ab_dir"] = str(ab_dir)
        if pitch_audit_json.is_file() and (study_pdf.is_file() or args.skip_pdf):
            eval_cmd = [
                py,
                str(SCRIPTS_DIR / "eval_gate_runner.py"),
                "--rubric",
                str(REFS_DIR / "eval_rubric_v2.json"),
                "--triage",
                str(triage_json),
                "--pitch-audit",
                str(pitch_audit_json),
                "--dynamics-plan",
                str(dynamics_md),
                "--dynamics-csv",
                str(dynamics_csv),
                "--xml",
                str(study_xml),
                *(["--pdf", str(study_pdf)] if study_pdf.is_file() else ["--skip-pdf"]),
                "--midi",
                str(cal_mid),
                "--orig-audio",
                str(best_audio),
                "--cal-audio",
                str(cal_wav),
                "--ab-dir",
                str(ab_dir),
                "--bpm",
                bpm,
                "--beats-per-bar",
                bpb,
                "--first-downbeat",
                t0,
                "--stage1-cal",
                str(cal_json),
                "--style-profile",
                str(style_profile_arg),
                "--out",
                str(eval_json),
            ]
            # Run eval gate without aborting manifest save on FAIL so user can inspect eval_report.json
            print("\n=== [Stage 5.2: Eval Gate Runner] ===")
            proc = subprocess.run(eval_cmd)
            manifest["artifacts"]["eval_report_json"] = str(eval_json)
            set_stage_status(manifest, "stage5_ab_and_eval", "done" if proc.returncode == 0 else "failed")
            save_manifest(manifest_path, manifest)
            if proc.returncode != 0 and args.auto_repair:
                print("\n=== [Phase 3 Local Repair Loop: Proposing bar_overrides] ===")
                _run_step(
                    "Bar Repair Proposal",
                    [
                        py,
                        str(SCRIPTS_DIR / "bar_repair_loop.py"),
                        "--manifest",
                        str(manifest_path),
                        "--eval-report",
                        str(eval_json),
                        "--pitch-audit",
                        str(pitch_audit_json),
                        "--xml",
                        str(study_xml),
                        "--write",
                    ],
                )
                manifest = load_manifest(manifest_path)
                if manifest.get("bar_overrides"):
                    print("Re-engraving Stage 3 & re-evaluating with local bar_overrides...")
                    s3_retry = [
                        py, str(SCRIPTS_DIR / "stage3_engrave_study_score.py"),
                        "--midi", str(midi_for_stage3), "--audio", str(best_audio),
                        "--title", title, "--bpm", bpm, "--beats-per-bar", bpb, "--beat-type", btype,
                        "--grid", grid, "--first-downbeat", t0, "--stage1-cal", str(cal_json),
                        "--key-fifths", key_fifths, "--dynamics-csv", str(dynamics_csv),
                        "--style-profile", str(style_profile_arg), "--bar-overrides", str(manifest_path),
                        "--out-xml", str(study_xml),
                    ]
                    if not args.skip_pdf:
                        s3_retry.extend(["--out-pdf", str(study_pdf)])
                    if args.musescore:
                        s3_retry.extend(["--musescore", args.musescore])
                    _run_step("Stage 3 (Local Repair Pass)", s3_retry)
                    proc = subprocess.run(eval_cmd)
                    set_stage_status(manifest, "stage5_ab_and_eval", "done" if proc.returncode == 0 else "failed")
                    save_manifest(manifest_path, manifest)
            if proc.returncode != 0:
                print(f"\nPipeline finished with Eval Gate warnings/failures. Inspect: {eval_json}")
                sys.exit(proc.returncode)
        else:
            reason = "missing_study_pdf" if not study_pdf.is_file() else "missing_pitch_audit"
            print(f"\n[SKIPPED] Stage 5.2 Eval Gate skipped ({reason}).")
            manifest["stage5_eval_skip_reason"] = reason
            set_stage_status(manifest, "stage5_ab_and_eval", "skipped")
            save_manifest(manifest_path, manifest)

    print(f"\nPiano Score Studio v3.0 complete! Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
