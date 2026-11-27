"""Lightweight synthetic unit tests for Piano Score Studio core scripts.

Runs without GPU, PyTorch, MuScriptor, beat_this, MuseScore, or sfizz.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "skills" / "piano-score-studio" / "scripts"


def _load_script(name: str):
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stage3_split_duration_conservation() -> None:
    s3 = _load_script("stage3_engrave_study_score.py")
    checked = 0
    for bpb, bt, grid in (
        (4, 4, 2),
        (4, 4, 3),
        (4, 4, 4),
        (3, 4, 2),
        (3, 4, 3),
        (3, 4, 4),
        (6, 8, 2),
        (6, 8, 4),
        (12, 8, 2),
        (12, 8, 4),
    ):
        div, bar_slots, beat_slots, _ = s3.compute_grid_metrics(bpb, bt, grid)
        for is_rest in (False, True):
            for slot in range(bar_slots):
                for dur in range(1, bar_slots - slot + 1):
                    pieces = s3.split_duration(slot, dur, is_rest, div, beat_slots)
                    assert sum(p[1] for p in pieces) == dur
                    checked += 1
    assert checked > 1500


def test_stage1_compound_meter_beat_resampling() -> None:
    s1 = _load_script("stage1_tempo_and_transcribe.py")
    # 6/8 at dotted-quarter = 60 BPM (1.0 s/dotted-quarter, 2.0 s/bar -> quarter_bpm = 90)
    dq_beats = np.arange(0.5, 10.5, 1.0)
    resampled, ratio = s1.resample_beats_to_quarter_pulse(
        dq_beats, tracked_bpm=60.0, quarter_bpm=90.0, first_downbeat=0.5
    )
    assert abs(ratio - 1.5) < 1e-6
    assert abs(resampled[0] - 0.5) < 1e-6
    assert abs(resampled[3] - 2.5) < 1e-6  # Bar 2 downbeat locked


def test_stage2_parse_xml_onsets_compound_6_8(tmp_path: Path) -> None:
    s2 = _load_script("stage2_pitch_audit.py")
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>2</divisions>
        <time><beats>6</beats><beat-type>8</beat-type></time>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><staff>1</staff></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><staff>1</staff></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><staff>1</staff></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><staff>1</staff></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><staff>1</staff></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>
"""
    xml_path = tmp_path / "test_6_8.musicxml"
    xml_path.write_text(xml_text, encoding="utf-8")

    # 6/8 at eighth-note BPM = 180 -> 1 eighth = 1/3 sec, 6 eighths = 2.0 sec
    notes = s2.parse_xml_onsets(xml_path, bpm=180.0, beats_per_bar=6, first_downbeat=1.0)
    assert len(notes) == 6
    expected_times = [1.0 + i / 3.0 for i in range(6)]
    for note, exp_t in zip(notes, expected_times):
        assert abs(note["t_on"] - exp_t) < 1e-4
    assert [n["beat"] for n in notes] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_stage2_clean_midi_removes_confirmed_ghost(tmp_path: Path) -> None:
    s2 = _load_script("stage2_pitch_audit.py")
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM (0.5 s/beat)
    # Simultaneous C4 (60) and C#4 (61) at t = 0.5s (tick 480)
    tr.append(mido.Message("note_on", note=60, velocity=80, time=480))
    tr.append(mido.Message("note_on", note=61, velocity=40, time=0))
    tr.append(mido.Message("note_off", note=60, velocity=0, time=480))
    tr.append(mido.Message("note_off", note=61, velocity=0, time=0))
    tr.append(mido.MetaMessage("end_of_track", time=0))

    in_mid = tmp_path / "in.mid"
    out_mid = tmp_path / "cleaned.mid"
    mid.save(str(in_mid))

    removed = s2.clean_ghost_notes_from_midi(
        in_mid, out_mid, [{"weaker_midi": 61, "t_on": 0.5}]
    )
    assert removed == 1
    cleaned = mido.MidiFile(str(out_mid))
    remaining_notes = [
        msg.note for track in cleaned.tracks for msg in track if msg.type == "note_on" and msg.velocity > 0
    ]
    assert remaining_notes == [60]


def test_eval_gate_s4_r1_warn_band() -> None:
    eg = _load_script("eval_gate_runner.py")
    assert eg.evaluate_s4_r1_static_dynamics(False, 75.0, 55.0, 74, 0) == "PASS"
    assert eg.evaluate_s4_r1_static_dynamics(False, 68.0, 56.0, 78, 0) == "WARN"
    assert eg.evaluate_s4_r1_static_dynamics(False, 75.0, 55.0, 82, 0) == "WARN"
    assert eg.evaluate_s4_r1_static_dynamics(False, 65.0, 58.0, 74, 0) == "FAIL"
    assert eg.evaluate_s4_r1_static_dynamics(True, 75.0, 55.0, 74, 0) == "FAIL"
    assert eg.evaluate_s4_r1_static_dynamics(False, 75.0, 55.0, 74, 1) == "FAIL"


def test_stage0_audio_triage_wav_and_m4a_rejection(tmp_path: Path) -> None:
    s0 = _load_script("stage0_audio_triage.py")
    assert ".m4a" not in s0.AUDIO_EXTS
    sr = 44100
    t = np.arange(sr * 2) / sr
    sig = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.column_stack([sig, sig])
    wav = tmp_path / "tone.wav"
    sf.write(str(wav), stereo, sr, subtype="PCM_16")
    res = s0.analyze_audio(wav)
    assert res["clipping_count"] == 0
    assert res["sample_rate"] == sr


def test_rubato_tempo_map_stage2_and_stage4_midi_roundtrip(tmp_path: Path) -> None:
    tm = _load_script("tempo_map.py")
    s2 = _load_script("stage2_pitch_audit.py")
    s4 = _load_script("stage4_render_calibrated.py")

    # Ritardando: Bar 1 @ 120 BPM (0.5 s/q -> [0.5..2.5]), Bar 2 @ 80 BPM (0.75 s/q -> [2.5..5.5])
    q_beats = [0.5, 1.0, 1.5, 2.0, 2.5, 3.25, 4.0, 4.75, 5.5]
    cal_path = tmp_path / "stage1_calibration.json"
    cal_path.write_text(
        json.dumps(
            {
                "calibration": {"tempo_mode": "tempo-map"},
                "tempo_map": {
                    "mode": "tempo-map",
                    "first_downbeat": 0.5,
                    "quarter_bpm": 100.0,
                    "beats_per_bar": 4,
                    "beat_type": 4,
                    "quarter_beats": q_beats,
                },
            }
        ),
        encoding="utf-8",
    )
    tmap = tm.load_tempo_map(cal_path, bpm=100.0, beats_per_bar=4, beat_type=4, first_downbeat=0.5)
    assert tmap.is_rubato
    assert abs(tmap.bar_bounds_sec(1)[1] - 2.5) < 1e-6
    assert abs(tmap.bar_bounds_sec(2)[1] - 5.5) < 1e-6

    # Verify Stage 2 MusicXML onset timing follows rubato curve
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><staff>1</staff></note>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><staff>1</staff></note>
    </measure>
    <measure number="2">
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration><staff>1</staff></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>
"""
    xml_path = tmp_path / "rubato.musicxml"
    xml_path.write_text(xml_text, encoding="utf-8")
    notes = s2.parse_xml_onsets(xml_path, bpm=100.0, beats_per_bar=4, first_downbeat=0.5, tempo_map=tmap)
    expected_onsets = [0.5, 1.5, 2.5, 4.0]
    for n, exp in zip(notes, expected_onsets):
        assert abs(n["t_on"] - exp) < 1e-4

    # Verify Stage 4 export_midi writes a real rubato tempo map matching wall-clock note times within 3 ms
    rate = 48000
    render_notes = [
        {"sample": int(round(t * rate)), "duration_samples": int(round(0.4 * rate)), "note": 60 + i, "velocity": 0.6, "channel": 1, "staff": 1}
        for i, t in enumerate(expected_onsets)
    ]
    out_mid = tmp_path / "rubato_out.mid"
    s4.export_midi(render_notes, [], rate, bpm=100.0, beats_per_bar=4, beat_type=4, key_fifths=0, out_mid=out_mid, tempo_map=tmap)

    mid_back = mido.MidiFile(str(out_mid))
    # Reconstruct real-time playback timestamps from merged tracks
    cur_t = 0.0
    cur_tempo = 500000
    observed_onsets = []
    for msg in mido.merge_tracks(mid_back.tracks):
        cur_t += mido.tick2second(msg.time, mid_back.ticks_per_beat, cur_tempo)
        if msg.type == "set_tempo":
            cur_tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity > 0:
            observed_onsets.append(cur_t)
    assert len(observed_onsets) == len(expected_onsets)
    for obs, exp in zip(observed_onsets, expected_onsets):
        assert abs(obs - exp) < 0.003


def test_v3_manifest_roundtrip_and_transkun_quantization(tmp_path: Path) -> None:
    pm = _load_script("project_manifest.py")
    tm = _load_script("tempo_map.py")
    te = _load_script("transcription_engines.py")

    m_path = tmp_path / "project_manifest.json"
    manifest = pm.create_manifest(["demo.wav"], tmp_path, "demo", "Demo Piece", engine="transkun")
    pm.set_stage_status(manifest, "stage0_triage", "done")
    pm.save_manifest(m_path, manifest)
    loaded = pm.load_manifest(m_path)
    assert loaded["schema_version"] == "3.0"
    assert pm.is_stage_done(loaded, "stage0_triage")

    # Quantize Rubato performance notes (including Bar 0 pickup) while preserving real velocities
    q_beats = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.75, 4.5, 5.25, 6.0], dtype=np.float64)
    q_grid = np.arange(len(q_beats), dtype=np.float64)  # Bar 1 beat 1 at t=1.0s
    tmap = tm.TempoMap(bpm=100.0, beats_per_bar=4, beat_type=4, first_downbeat=1.0, mode="tempo-map", q_grid=q_grid, t_grid=q_beats)
    perf_notes = [
        {"start_sec": 0.50, "end_sec": 0.95, "pitch": 67, "velocity": 52},  # Bar 0 pickup (q=-1)
        {"start_sec": 1.01, "end_sec": 1.98, "pitch": 60, "velocity": 88},  # Bar 1 beat 1 (q=0)
        {"start_sec": 3.02, "end_sec": 4.48, "pitch": 64, "velocity": 71},  # Bar 2 beat 1 (q=4)
    ]
    out_mid = tmp_path / "score.mid"
    res = te.quantize_performance_midi(perf_notes, tmap, grid=2, beats_per_bar=4, beat_type=4, out_mid=out_mid)
    assert res["has_pickup"] is True
    assert res["note_count"] == 3
    q_mid = mido.MidiFile(str(out_mid))
    vels = [m.velocity for tr in q_mid.tracks for m in tr if m.type == "note_on" and m.velocity > 0]
    assert vels == [52, 88, 71]


def test_krumhansl_schmuckler_key_detection() -> None:
    """Verify Krumhansl-Schmuckler detects C major, A minor, and Bb major accurately."""
    sa = _load_script("style_analyzer.py")

    # C major: strong C-E-G-F-G-C cadence
    c_maj_notes = [
        (60, 0.0, 1.0, 90), (64, 0.0, 1.0, 82), (67, 0.0, 1.0, 80),
        (65, 1.0, 2.0, 80), (69, 1.0, 2.0, 78), (60, 1.0, 2.0, 85),
        (67, 2.0, 3.0, 84), (71, 2.0, 3.0, 80), (62, 2.0, 3.0, 76),
        (60, 3.0, 5.0, 95), (64, 3.0, 5.0, 84), (67, 3.0, 5.0, 82),
    ]
    res_c = sa.detect_key_krumhansl_schmuckler(c_maj_notes)
    assert res_c["key_name"] == "C major"
    assert res_c["key_fifths"] == 0
    assert res_c["confidence"] > 0.7

    # Bb major (key_fifths = -2)
    bb_maj_notes = [(p - 2, s, e, v) for p, s, e, v in c_maj_notes]
    res_bb = sa.detect_key_krumhansl_schmuckler(bb_maj_notes)
    assert res_bb["key_name"] == "Bb major"
    assert res_bb["key_fifths"] == -2

    # A minor (key_fifths = 0, mode = minor)
    a_min_notes = [
        (57, 0.0, 2.0, 92), (60, 0.0, 2.0, 84), (64, 0.0, 2.0, 82),
        (62, 2.0, 3.0, 80), (65, 2.0, 3.0, 80), (57, 2.0, 3.0, 84),
        (64, 3.0, 4.0, 85), (68, 3.0, 4.0, 80), (59, 3.0, 4.0, 78),
        (57, 4.0, 6.0, 95), (60, 4.0, 6.0, 84), (64, 4.0, 6.0, 82),
    ]
    res_am = sa.detect_key_krumhansl_schmuckler(a_min_notes)
    assert res_am["key_name"] == "A minor"
    assert res_am["key_fifths"] == 0
    assert res_am["mode"] == "minor"


def test_musical_dna_and_style_classification() -> None:
    """Verify 6-dim Musical DNA classifies Pop, Classical Rubato, Baroque, and Jazz distinctly."""
    sa = _load_script("style_analyzer.py")

    # 1. Pop Ballad: steady tempo, diatonic block chords, quarter/half notes
    pop_notes = []
    for bar in range(8):
        b0 = bar * 4.0
        for p in (48, 55, 60, 64, 67):
            pop_notes.append((p, b0, b0 + 2.0, 80))
        for p in (53, 57, 60, 65, 69):
            pop_notes.append((p, b0 + 2.0, b0 + 4.0, 80))
    dna_pop = sa.extract_musical_dna(pop_notes, tempo_map_dict={"bar_tempos_bpm": [72.0] * 8})
    prof_pop, _, _, _ = sa.classify_style_profile(dna_pop)
    assert prof_pop == "pop_ballad"

    # 2. Classical Romantic: strong rubato tempo_cv + expressive 16th-note runs
    cl_notes = []
    for bar in range(8):
        b0 = bar * 4.0
        cl_notes.append((48, b0, b0 + 2.0, 72))
        for k in range(8):
            cl_notes.append((60 + (k % 7), b0 + k * 0.25, b0 + (k + 1) * 0.25, 84))
    rubato_bpms = [62.0, 68.0, 78.0, 84.0, 74.0, 64.0, 80.0, 60.0]
    dna_cl = sa.extract_musical_dna(cl_notes, tempo_map_dict={"bar_tempos_bpm": rubato_bpms})
    prof_cl, _, _, _ = sa.classify_style_profile(dna_cl)
    assert prof_cl == "classical_romantic"

    # 3. Baroque Polyphonic: steady tempo + continuous 16th notes + staggered overlapping voices
    baroque_notes = []
    for step in range(64):
        t = step * 0.25
        baroque_notes.append((72 + (step % 5), t, t + 0.5, 82))
        if step % 2 == 1:
            baroque_notes.append((55 + (step % 4), t, t + 0.75, 80))
    dna_bq = sa.extract_musical_dna(baroque_notes, tempo_map_dict={"bar_tempos_bpm": [96.0] * 8})
    prof_bq, _, _, _ = sa.classify_style_profile(dna_bq)
    assert prof_bq == "baroque_polyphonic"

    # 4. Jazz Groove: swing long-short eighths (0.67 / 0.33) + chromatic extensions + off-beat syncopation
    jazz_notes = []
    for beat in range(24):
        t0 = float(beat)
        t_sw = t0 + 0.67
        jazz_notes.append((60 + (beat % 12), t0, t0 + 0.60, 86))
        jazz_notes.append((63 + ((beat * 5) % 12), t_sw, t_sw + 0.28, 82))
    dna_jz = sa.extract_musical_dna(jazz_notes, tempo_map_dict={"bar_tempos_bpm": [128.0] * 6})
    prof_jz, _, _, _ = sa.classify_style_profile(dna_jz)
    assert prof_jz == "jazz_groove"


def test_style_profiles_and_dynamic_eval_rubric() -> None:
    """Verify all 4 declarative profiles load cleanly and adapt eval_gate_runner static dynamics."""
    sa = _load_script("style_analyzer.py")
    eg = _load_script("eval_gate_runner.py")

    required_sections = {"transcription", "pitch_audit", "engraving", "dynamics", "eval_rubric"}
    for pid in sa.VALID_PROFILES:
        prof = sa.load_style_profile(pid)
        assert prof["profile_id"] == pid
        assert required_sections.issubset(prof.keys())

    # Balanced Baroque counterpoint (rh_med=84, lh_med=82, lh_max=98) fails legacy Pop rubric
    # but PASSES under baroque_polyphonic dynamic rubric.
    assert eg.evaluate_s4_r1_static_dynamics(False, 84.0, 82.0, 98, 0) == "FAIL"
    baroque_rubric = sa.load_style_profile("baroque_polyphonic")["eval_rubric"]
    assert eg.evaluate_s4_r1_static_dynamics(False, 84.0, 82.0, 98, 0, eval_rubric=baroque_rubric) == "PASS"
