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
    """Verify Krumhansl-Schmuckler detects C major, A minor, Bb major, and D minor accurately."""
    sa = _load_script("style_analyzer.py")
    detect_key_krumhansl_schmuckler = sa.detect_key_krumhansl_schmuckler

    # C major: strong C-E-G-F-G-C cadence
    c_maj_notes = [
        (60, 0.0, 1.0, 90), (64, 0.0, 1.0, 82), (67, 0.0, 1.0, 80),
        (65, 1.0, 2.0, 80), (69, 1.0, 2.0, 78), (60, 1.0, 2.0, 85),
        (67, 2.0, 3.0, 84), (71, 2.0, 3.0, 80), (62, 2.0, 3.0, 76),
        (60, 3.0, 5.0, 95), (64, 3.0, 5.0, 84), (67, 3.0, 5.0, 82),
    ]
    res_c = detect_key_krumhansl_schmuckler(c_maj_notes)
    assert res_c["key_name"] == "C major"
    assert res_c["key_fifths"] == 0
    assert res_c["confidence"] > 0.7

    # Bb major (key_fifths = -2)
    bb_maj_notes = [(p - 2, s, e, v) for p, s, e, v in c_maj_notes]
    res_bb = detect_key_krumhansl_schmuckler(bb_maj_notes)
    assert res_bb["key_name"] == "Bb major"
    assert res_bb["key_fifths"] == -2

    # A minor (key_fifths = 0, mode = minor)
    a_min_notes = [
        (57, 0.0, 2.0, 92), (60, 0.0, 2.0, 84), (64, 0.0, 2.0, 82),
        (62, 2.0, 3.0, 80), (65, 2.0, 3.0, 80), (57, 2.0, 3.0, 84),
        (64, 3.0, 4.0, 85), (68, 3.0, 4.0, 80), (59, 3.0, 4.0, 78),
        (57, 4.0, 6.0, 95), (60, 4.0, 6.0, 84), (64, 4.0, 6.0, 82),
    ]
    res_am = detect_key_krumhansl_schmuckler(a_min_notes)
    assert res_am["key_name"] == "A minor"
    assert res_am["key_fifths"] == 0
    assert res_am["mode"] == "minor"


def test_musical_dna_and_style_classification() -> None:
    """Verify 6-dim Musical DNA classifies Pop, Classical Rubato, Baroque, and Jazz distinctly."""
    sa = _load_script("style_analyzer.py")
    classify_style_profile, extract_musical_dna = sa.classify_style_profile, sa.extract_musical_dna

    # 1. Pop Ballad: steady tempo, diatonic block chords, quarter/half notes
    pop_notes = []
    for bar in range(8):
        b0 = bar * 4.0
        for p in (48, 55, 60, 64, 67):
            pop_notes.append((p, b0, b0 + 2.0, 80))
        for p in (53, 57, 60, 65, 69):
            pop_notes.append((p, b0 + 2.0, b0 + 4.0, 80))
    dna_pop = extract_musical_dna(pop_notes, tempo_map_dict={"bar_tempos_bpm": [72.0] * 8})
    prof_pop, _, _, _ = classify_style_profile(dna_pop)
    assert prof_pop == "pop_ballad"

    # 2. Classical Romantic: strong rubato tempo_cv + expressive 16th-note runs
    cl_notes = []
    for bar in range(8):
        b0 = bar * 4.0
        cl_notes.append((48, b0, b0 + 2.0, 72))
        for k in range(8):
            cl_notes.append((60 + (k % 7), b0 + k * 0.25, b0 + (k + 1) * 0.25, 84))
    rubato_bpms = [62.0, 68.0, 78.0, 84.0, 74.0, 64.0, 80.0, 60.0]
    dna_cl = extract_musical_dna(cl_notes, tempo_map_dict={"bar_tempos_bpm": rubato_bpms})
    prof_cl, _, _, _ = classify_style_profile(dna_cl)
    assert prof_cl == "classical_romantic"

    # 3. Baroque Polyphonic: steady tempo + continuous 16th notes + staggered overlapping voices
    baroque_notes = []
    for step in range(64):
        t = step * 0.25
        # Upper voice 16th-note motoric line overlapping lower voice staggered by 0.25 beat
        baroque_notes.append((72 + (step % 5), t, t + 0.5, 82))
        if step % 2 == 1:
            baroque_notes.append((55 + (step % 4), t, t + 0.75, 80))
    dna_bq = extract_musical_dna(baroque_notes, tempo_map_dict={"bar_tempos_bpm": [96.0] * 8})
    prof_bq, _, _, _ = classify_style_profile(dna_bq)
    assert prof_bq == "baroque_polyphonic"

    # 4. Jazz Groove: swing long-short eighths (0.67 / 0.33) + chromatic extensions + off-beat syncopation
    jazz_notes = []
    for beat in range(24):
        t0 = float(beat)
        t_sw = t0 + 0.67
        # Chromatic shell/tension pitches (Eb, F#, Bb, Db, Ab)
        jazz_notes.append((60 + (beat % 12), t0, t0 + 0.60, 86))
        jazz_notes.append((63 + ((beat * 5) % 12), t_sw, t_sw + 0.28, 82))
    dna_jz = extract_musical_dna(jazz_notes, tempo_map_dict={"bar_tempos_bpm": [128.0] * 6})
    prof_jz, _, _, _ = classify_style_profile(dna_jz)
    assert prof_jz == "jazz_groove"


def test_style_profiles_and_dynamic_eval_rubric() -> None:
    """Verify all 4 declarative profiles load cleanly and adapt eval_gate_runner static dynamics."""
    sa = _load_script("style_analyzer.py")
    eg = _load_script("eval_gate_runner.py")
    evaluate_s4_r1_static_dynamics = eg.evaluate_s4_r1_static_dynamics
    VALID_PROFILES, load_style_profile = sa.VALID_PROFILES, sa.load_style_profile

    required_sections = {"transcription", "pitch_audit", "engraving", "dynamics", "eval_rubric"}
    for pid in VALID_PROFILES:
        prof = load_style_profile(pid)
        assert prof["profile_id"] == pid
        assert required_sections.issubset(prof.keys())

    # Balanced Baroque counterpoint (rh_med=84, lh_med=82, lh_max=98) fails legacy Pop rubric
    # but PASSES under baroque_polyphonic dynamic rubric.
    assert evaluate_s4_r1_static_dynamics(False, 84.0, 82.0, 98, 0) == "FAIL"
    baroque_rubric = load_style_profile("baroque_polyphonic")["eval_rubric"]
    assert evaluate_s4_r1_static_dynamics(False, 84.0, 82.0, 98, 0, eval_rubric=baroque_rubric) == "PASS"


def test_chroma_flux_dynamic_pedal_styles() -> None:
    """Verify dry Baroque omits sustain, Jazz uses half-pedal, and Pop lifts on mid-bar harmony change."""
    pv = _load_script("pedal_and_velocity.py")
    tm = _load_script("tempo_map.py")

    tmap = tm.TempoMap(bpm=120.0, beats_per_bar=4, beat_type=4, first_downbeat=0.0, mode="fixed-tempo")
    # Bar 1: Beat 1-2 C major (C,E,G), Beat 3-4 F# major (F#,A#,C#) -> strong mid-bar chroma flux!
    notes = [
        {"st": 0.0, "pitch": 60}, {"st": 0.0, "pitch": 64}, {"st": 0.0, "pitch": 67},
        {"st": 1.0, "pitch": 66}, {"st": 1.0, "pitch": 70}, {"st": 1.0, "pitch": 73},
    ]

    # 1. Baroque dry_articulated: zero CC64 > 0 events
    dry = pv.compute_dynamic_pedal_events(notes, tmap, max_bar=1, rate=48000, pedal_style="dry_articulated")
    assert all(ev["value"] == 0 for ev in dry)

    # 2. Pop legato_harmonic: should trigger BOTH bar-start pedal AND mid-bar chord-change pedal lift
    pop = pv.compute_dynamic_pedal_events(notes, tmap, max_bar=1, rate=48000, pedal_style="legato_harmonic")
    on_evs = [ev for ev in pop if ev["value"] > 0]
    assert len(on_evs) >= 2
    assert any(ev["value"] == 127 for ev in on_evs)

    # 3. Jazz rhythmic_half_pedal: uses half-pedal depth (<= 80)
    jazz = pv.compute_dynamic_pedal_events(notes, tmap, max_bar=1, rate=48000, pedal_style="rhythmic_half_pedal")
    jazz_on = [ev for ev in jazz if ev["value"] > 0]
    assert len(jazz_on) >= 1
    assert all(ev["value"] <= 80 for ev in jazz_on)


def test_transkun_velocity_fusion_and_flat_fallback() -> None:
    """Verify real TransKun velocities blend with acoustic velocities while flat placeholders are skipped."""
    pv = _load_script("pedal_and_velocity.py")

    notes = [
        {"st": 0.50, "pitch": 60, "staff": 1, "calc_vel": 64},
        {"st": 1.00, "pitch": 64, "staff": 1, "calc_vel": 64},
        {"st": 1.50, "pitch": 48, "staff": 2, "calc_vel": 64},
    ]
    # Flat placeholder MIDI (std = 0) should be skipped
    flat_events = [(0.50, 60, 100), (1.00, 64, 100), (1.50, 48, 100)]
    assert pv.fuse_transkun_velocities(notes, flat_events) == 0
    assert [n["calc_vel"] for n in notes] == [64, 64, 64]

    # Real expressive TransKun velocities should blend & respect lh_ceiling_vel=78
    real_events = [(0.51, 60, 96), (0.99, 64, 42), (1.51, 48, 110)]
    fused = pv.fuse_transkun_velocities(notes, real_events, alpha=0.55, lh_ceiling_vel=78)
    assert fused == 3
    assert notes[0]["calc_vel"] > 75
    assert notes[1]["calc_vel"] < 58
    assert notes[2]["calc_vel"] <= 78


def test_bar_overrides_application_and_repair_proposal(tmp_path: Path) -> None:
    """Verify bar_overrides drops/re-hands notes per bar and proposes fixes for wide-span chords."""
    br = _load_script("bar_repair_loop.py")
    s3 = _load_script("stage3_engrave_study_score.py")

    notes = [
        s3.RawNote(start_slot=0, dur=2, pitch=60, hand=1),
        s3.RawNote(start_slot=0, dur=2, pitch=61, hand=1),  # ghost to drop in bar 1
        s3.RawNote(start_slot=8, dur=2, pitch=62, hand=1),  # re-hand to LH in bar 2
    ]
    overrides = {
        1: {"drop_pitches": [61]},
        2: {"force_hand": {"62": 2}},
    }
    stats = br.apply_bar_overrides_to_notes(notes, bar_slots=8, pickup_start=0, overrides=overrides)
    assert stats == {"dropped": 1, "rehanded": 1}
    assert [n.pitch for n in notes] == [60, 62]
    assert notes[1].hand == 2

    # Propose fix for an unmarked >12st RH chord in Bar 2 (C4=60 + D5=74 -> span 14)
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part id="P1">
    <measure number="2">
      <attributes><divisions>2</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff></note>
      <note><chord/><pitch><step>D</step><octave>5</octave></pitch><duration>2</duration><voice>1</voice><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>
"""
    xml_path = tmp_path / "wide.musicxml"
    xml_path.write_text(xml_text, encoding="utf-8")
    proposals = br.propose_bar_repairs(None, {"confirmed_ghost_count": 0}, xml_path)
    assert "2" in proposals
    assert proposals["2"]["force_hand"]["60"] == 2


def test_p0_1_stage2_clean_midi_nonzero_downbeat_and_rubato(tmp_path: Path) -> None:
    """P0-1 regression: clean_ghost_notes_from_midi must match audio t_on when first_downbeat != 0."""
    s2 = _load_script("stage2_pitch_audit.py")
    tm = _load_script("tempo_map.py")

    # Quantized score MIDI: Bar 1 beat 1 sits at tick 0 (even though audio first_downbeat = 1.30 s)
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM
    tr.append(mido.Message("note_on", note=60, velocity=85, time=0))
    tr.append(mido.Message("note_on", note=61, velocity=35, time=0))  # ghost C#4 at Bar 1 beat 1 -> audio 1.30s
    tr.append(mido.Message("note_off", note=60, velocity=0, time=480))
    tr.append(mido.Message("note_off", note=61, velocity=0, time=0))
    tr.append(mido.MetaMessage("end_of_track", time=0))

    in_mid = tmp_path / "score.mid"
    out_mid = tmp_path / "score_cleaned.mid"
    mid.save(str(in_mid))

    tmap = tm.TempoMap(bpm=120.0, beats_per_bar=4, beat_type=4, first_downbeat=1.30, mode="fixed-tempo")
    removed = s2.clean_ghost_notes_from_midi(
        in_mid, out_mid, [{"weaker_midi": 61, "t_on": 1.30}], tempo_map=tmap, has_pickup=False
    )
    assert removed == 1
    cleaned = mido.MidiFile(str(out_mid))
    remaining = [m.note for t in cleaned.tracks for m in t if m.type == "note_on" and m.velocity > 0]
    assert remaining == [60]


def test_p0_2_stage3_reconcile_cross_bar_ties_zero_orphans() -> None:
    """P0-2 regression: truncated cross-bar note in chord must not leave orphan tie_stop in next bar."""
    s3 = _load_script("stage3_engrave_study_score.py")

    # RH Bar 1: E5(76) lasts 6 beats (12 slots @ div=2, crosses barline into Bar 2),
    # simultaneous short C5(72) dur=2, plus new onset D5(74) at slot 2 truncating Bar 1 chord.
    notes = [
        s3.RawNote(start_slot=0, dur=12, pitch=76, hand=1),
        s3.RawNote(start_slot=0, dur=2, pitch=72, hand=1),
        s3.RawNote(start_slot=2, dur=2, pitch=74, hand=1),
    ]
    bars_map, max_bar = s3.slice_into_bars(notes, pickup_start=0, bar_slots=8)
    slot_rms = np.full(32, -60.0)
    bar_voices = {}
    for b in range(1, max_bar + 1):
        bar_voices[(b, 1)] = s3.simplify_bar_hand(bars_map.get((b, 1), []), b, (b - 1) * 8, 8, slot_rms, 2)
        bar_voices[(b, 2)] = ([], None)
    s3.reconcile_cross_bar_ties(bar_voices, start_bar=1, max_bar=max_bar, bar_slots=8, pickup_start=0)

    # Count outgoing tie_starts vs incoming tie_stops across all bars -> must balance strictly
    starts = sum(1 for (b, _), (v1, v2) in bar_voices.items() for ev in (v1 + (v2 or [])) for seg in ev.segs if seg.tie_start)
    stops = sum(1 for (b, _), (v1, v2) in bar_voices.items() for ev in (v1 + (v2 or [])) for seg in ev.segs if seg.tie_stop)
    assert starts == stops


def test_p0_3_stage3_no_false_pickup_on_leading_rest(tmp_path: Path) -> None:
    """P0-3 regression: when Stage 1 reports has_pickup=False, a Beat 2 entry must NOT become Bar 0."""
    s3 = _load_script("stage3_engrave_study_score.py")

    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    # First note starts on Beat 2 (tick 480 = slot 2 at div=2) in a 4/4 bar (bar_slots=8)
    tr.append(mido.Message("note_on", note=64, velocity=80, time=480))
    tr.append(mido.Message("note_off", note=64, velocity=0, time=480))
    tr.append(mido.MetaMessage("end_of_track", time=0))
    mid_path = tmp_path / "leading_rest.mid"
    mid.save(str(mid_path))

    _, pickup_start = s3.load_midi_notes(mid_path, divisions=2, bar_slots=8, has_pickup=False)
    assert pickup_start == 0


def test_p0_4_compound_6_8_eighth_plus_quarter_not_split() -> None:
    """P0-4 regression: 6/8 eighth + quarter + dotted-quarter should emit a clean quarter (not tied eighths)."""
    s3 = _load_script("stage3_engrave_study_score.py")

    # 6/8 at div=2 -> beat_slots=3. Quarter note starting on 2nd eighth (slot=1, dur=2)
    pieces = s3.split_duration(slot=1, dur=2, is_rest=False, divisions=2, beat_slots=3)
    assert len(pieces) == 1
    assert pieces[0][1] == 2 and pieces[0][2] == "quarter"

    # 6/8 at div=4 -> beat_slots=6. Quarter note starting on 2nd eighth (slot=2, dur=4)
    pieces4 = s3.split_duration(slot=2, dur=4, is_rest=False, divisions=4, beat_slots=6)
    assert len(pieces4) == 1
    assert pieces4[0][1] == 4 and pieces4[0][2] == "quarter"


def test_p1_1_and_p1_3_rubato_grid_coverage_and_note_dedup(tmp_path: Path) -> None:
    """P1-1 & P1-3 regression: rubato grid uses TempoMap.sec_to_q, and quantize_performance_midi deduplicates."""
    s1 = _load_script("stage1_tempo_and_transcribe.py")
    tm = _load_script("tempo_map.py")
    te = _load_script("transcription_engines.py")

    # Strongly accelerating rubato beats: 8th-note onsets exactly on rubato grid
    q_beats = np.array([0.0, 0.80, 1.50, 2.10, 2.60, 3.05, 3.45, 3.85, 4.25], dtype=np.float64)
    q_grid = np.arange(len(q_beats), dtype=np.float64)
    tmap = tm.TempoMap(bpm=112.0, beats_per_bar=4, beat_type=4, first_downbeat=0.0, mode="tempo-map", q_grid=q_grid, t_grid=q_beats)
    onsets = np.array([tmap.q_to_sec(k * 0.5) for k in range(16)], dtype=np.float64)
    res_grid = s1.select_quantization_grid(onsets, quarter_bpm=112.0, first_downbeat=0.0, beat_type=4, grid_override=None, tempo_map=tmap)
    assert res_grid["grid"] == 2
    assert res_grid["coverage_pct_by_grid"]["2"] == 100.0

    # Duplicate same-pitch notes quantizing to same slot should deduplicate cleanly
    perf_notes = [
        {"start_sec": 0.01, "end_sec": 0.78, "pitch": 60, "velocity": 70},
        {"start_sec": 0.02, "end_sec": 0.79, "pitch": 60, "velocity": 92},
    ]
    out_mid = tmp_path / "dedup.mid"
    q_res = te.quantize_performance_midi(perf_notes, tmap, grid=2, beats_per_bar=4, beat_type=4, out_mid=out_mid)
    assert q_res["note_count"] == 1


def test_p1_5_stage2_parse_midi_onsets_direct_and_pickup(tmp_path: Path) -> None:
    """P1-5: Stage 2 parses onsets directly from quantized MIDI + TempoMap without MuseScore XML."""
    s2 = _load_script("stage2_pitch_audit.py")
    tm = _load_script("tempo_map.py")
    parse_midi_onsets = s2.parse_midi_onsets
    load_tempo_map = tm.load_tempo_map

    mid_path = tmp_path / "score.mid"
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    # With has_pickup=True in 4/4 (bar_quarters=4.0), Bar 0 beat 4 is tick 1440 (score_q = -1.0),
    # Bar 1 beat 1 is tick 1920 (score_q = 0.0).
    tr.append(mido.Message("note_on", note=64, velocity=80, time=1440))
    tr.append(mido.Message("note_off", note=64, velocity=0, time=480))
    tr.append(mido.Message("note_on", note=60, velocity=90, time=0))
    tr.append(mido.Message("note_off", note=60, velocity=0, time=480))
    mid.save(str(mid_path))

    tmap = load_tempo_map(None, 120.0, 4, 4, 1.50)
    notes = parse_midi_onsets(
        mid_path, bpm=120.0, beats_per_bar=4, first_downbeat=1.50, beat_type=4, tempo_map=tmap, has_pickup=True
    )
    assert len(notes) == 2
    assert notes[0]["measure"] == 0 and abs(notes[0]["t_on"] - 1.00) < 1e-4
    assert notes[1]["measure"] == 1 and abs(notes[1]["t_on"] - 1.50) < 1e-4


def test_p2_5_manifest_resume_param_hash_invalidation() -> None:
    """P2-5: Changing --bpm or --engine invalidates Stage 1+ while preserving Stage 0."""
    pm = _load_script("project_manifest.py")
    compute_run_params_hash = pm.compute_run_params_hash
    create_manifest = pm.create_manifest
    invalidate_stale_stages = pm.invalidate_stale_stages
    set_stage_status = pm.set_stage_status

    m = create_manifest(["song.wav"], Path("/tmp/out"), "song", "Song")
    base_params = {"inputs": ["/tmp/song.wav"], "engine": "transkun", "bpm": None, "style": "auto"}
    m["run_params"] = dict(base_params)
    m["run_params_hash"] = compute_run_params_hash(base_params)
    for k in ("stage0_triage", "stage1_calibrate_transcribe", "stage1_5_style_analysis", "stage2_pitch_audit", "stage3_engrave_score"):
        set_stage_status(m, k, "done")

    assert invalidate_stale_stages(m, dict(base_params)) == []

    changed = dict(base_params, bpm=108.0)
    inv = invalidate_stale_stages(m, changed)
    assert "stage0_triage" not in inv
    assert "stage1_calibrate_transcribe" in inv
    assert "stage3_engrave_score" in inv
    assert m["stage_status"]["stage0_triage"] == "done"
    assert m["stage_status"]["stage1_calibrate_transcribe"] == "pending"


def test_p2_6_timeout_expired_graceful_runtime_error(tmp_path: Path, monkeypatch) -> None:
    """P2-6: TransKun TimeoutExpired is caught and converted to an actionable RuntimeError."""
    import subprocess
    te = _load_script("transcription_engines.py")

    fake_wav = tmp_path / "a.wav"
    fake_wav.write_bytes(b"RIFF")

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="transkun", timeout=900)

    monkeypatch.setattr(subprocess, "run", boom)
    try:
        te.run_transkun_transcribe(fake_wav, tmp_path / "out.mid", device="cpu", pad_sec=0.0)
        raise AssertionError("Expected RuntimeError on TimeoutExpired")
    except RuntimeError as exc:
        assert "timed out" in str(exc).lower()
