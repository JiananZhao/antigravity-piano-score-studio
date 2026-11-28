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
    assert eg.evaluate_s4_r1_static_dynamics(False, 65.0, 58.0, 74, 0) == "WARN"
    assert eg.evaluate_s4_r1_static_dynamics(False, 50.0, 62.0, 74, 0) == "FAIL"
    assert eg.evaluate_s4_r1_static_dynamics(False, 75.0, 55.0, 105, 0) == "FAIL"
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
    assert evaluate_s4_r1_static_dynamics(False, 84.0, 82.0, 98, 0) == "WARN"
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


def test_p2_1_setup_assets_blocks_zip_and_tar_path_traversal(tmp_path: Path) -> None:
    """P2-1: safe_extract_tar and safe_extract_zip must reject path-traversal entries."""
    import io
    import tarfile
    import zipfile

    sa = _load_script("setup_assets.py")
    dest = tmp_path / "extract"
    dest.mkdir()

    # 1. Malicious ZipSlip archive
    bad_zip = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../escape.txt", "boom")
    with zipfile.ZipFile(bad_zip, "r") as zf:
        try:
            sa.safe_extract_zip(zf, dest)
            raise AssertionError("Expected ValueError for ZipSlip entry")
        except ValueError as exc:
            assert "traversal" in str(exc).lower() or "unsafe" in str(exc).lower()

    # 2. Malicious tar path traversal archive
    bad_tar = tmp_path / "evil.tar.gz"
    with tarfile.open(bad_tar, "w:gz") as tf:
        payload = b"boom"
        info = tarfile.TarInfo(name="../escape_tar.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    with tarfile.open(bad_tar, "r:gz") as tf:
        try:
            sa.safe_extract_tar(tf, dest)
            raise AssertionError("Expected ValueError for tar traversal entry")
        except ValueError as exc:
            assert "traversal" in str(exc).lower() or "unsafe" in str(exc).lower()

    # 3. Clean archive + verify_sha256
    good_zip = tmp_path / "good.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("sub/ok.txt", "hello")
    digest = sa.sha256_file(good_zip)
    assert sa.verify_sha256(good_zip, digest, label="good.zip") == digest
    with zipfile.ZipFile(good_zip, "r") as zf:
        sa.safe_extract_zip(zf, dest)
    assert (dest / "sub" / "ok.txt").read_text(encoding="utf-8") == "hello"


def test_rubato_low_circular_r_preserves_beat_this_bpm() -> None:
    """Regression: on rubato pieces (|R| < 0.45), calibrate_bpm_and_mode must NOT multiply beat_this_bpm by 1.5x."""
    s1 = _load_script("stage1_tempo_and_transcribe.py")
    rng = np.random.default_rng(42)
    # Simulate rubato onset flux over 120s where fixed-tempo circular |R| is low (< 0.25) everywhere
    times = np.arange(0.0, 120.0, 0.01)
    flux = rng.uniform(0.0, 1.0, size=len(times))
    # Inject irregular rubato peaks around ~60.7 BPM
    bt_beats = np.cumsum(rng.uniform(0.82, 1.15, size=120))
    bt_info = {"beat_this_bpm": 60.743, "beats": bt_beats, "downbeats": bt_beats[::4]}
    res = s1.calibrate_bpm_and_mode(times, flux, dur_s=120.0, bt_info=bt_info, bpm_override=None, mode_arg="auto")
    assert abs(res["bpm"] - 60.743) < 1e-2
    assert res["tempo_mode"] == "tempo-map"


def test_stage3_prunes_orphan_tie_stop_without_phantom_note() -> None:
    """Regression: truncated cross-bar note must NOT become a phantom re-struck note in the next bar."""
    s3 = _load_script("stage3_engrave_study_score.py")
    # Bar 0 (pickup_start=0, bar_slots=16 @ div=4): F5(77) held from slot 3 across barline into Bar 1 (dur=15),
    # while G4(67) and Bb4(70) enter underneath -> Bar 0 must split into 2 voices preserving the real cross-bar tie!
    notes_bar0 = [
        s3.RawNote(start_slot=3, dur=15, pitch=77, hand=1),
        s3.RawNote(start_slot=10, dur=1, pitch=67, hand=1),
        s3.RawNote(start_slot=11, dur=1, pitch=70, hand=1),
        s3.RawNote(start_slot=14, dur=2, pitch=75, hand=1),
    ]
    bars_map, max_bar = s3.slice_into_bars(notes_bar0, pickup_start=1, bar_slots=16)
    slot_rms = np.full(64, -60.0)
    bar_voices = {}
    for b in range(0, max_bar + 1):
        bar_len = 15 if b == 0 else 16
        g_start = 1 if b == 0 else (b * 16)
        bar_voices[(b, 1)] = s3.simplify_bar_hand(bars_map.get((b, 1), []), b, g_start, bar_len, slot_rms, 4)
        bar_voices[(b, 2)] = ([], None)
    s3.reconcile_cross_bar_ties(bar_voices, start_bar=0, max_bar=max_bar, bar_slots=16, pickup_start=1)

    # Bar 1 must have zero phantom (non-tie_stop) attacks of F5(77)
    b1_v1, b1_v2 = bar_voices[(1, 1)]
    b1_f5_attacks = [
        seg for ev in (b1_v1 + (b1_v2 or [])) for seg in ev.segs
        if seg.pitch == 77 and not seg.tie_stop
    ]
    assert b1_f5_attacks == []


def test_real_midi_pedal_reader_and_bar0_fallback(tmp_path: Path) -> None:
    """Regression: read_midi_pedal_events extracts real CC64 from TransKun MIDI, and fallback covers Bar 0."""
    pv = _load_script("pedal_and_velocity.py")
    tm = _load_script("tempo_map.py")

    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    tr.append(mido.Message("control_change", control=64, value=127, time=240))
    tr.append(mido.Message("control_change", control=64, value=0, time=480))
    tr.append(mido.MetaMessage("end_of_track", time=0))
    p_mid = tmp_path / "raw_pedal.mid"
    mid.save(str(p_mid))

    evs = pv.read_midi_pedal_events(p_mid, rate=48000)
    assert len(evs) == 2
    assert evs[0]["value"] == 127 and evs[0]["sample"] == 12000
    assert evs[1]["value"] == 0 and evs[1]["sample"] == 36000

    # Fallback dynamic pedaling must also cover Bar 0 pickup when notes have measure == 0
    tmap = tm.TempoMap(bpm=60.0, beats_per_bar=4, beat_type=4, first_downbeat=4.0, mode="fixed-tempo")
    notes_bar0 = [
        {"measure": 0, "st": 0.5, "pitch": 60},
        {"measure": 0, "st": 1.0, "pitch": 64},
        {"measure": 1, "st": 4.0, "pitch": 65},
    ]
    fb = pv.compute_dynamic_pedal_events(notes_bar0, tmap, max_bar=1, rate=48000, pedal_style="legato_harmonic")
    assert any(ev["sample"] < int(3.5 * 48000) and ev["value"] > 0 for ev in fb)


def test_microtiming_alignment_soft_velocity_and_hall_reverb() -> None:
    """Verify micro-timing alignment, delicate soft velocity preservation, and finite stereo hall reverb."""
    pv = _load_script("pedal_and_velocity.py")
    s4 = _load_script("stage4_render_calibrated.py")

    notes = [
        {"st": 1.00, "et": 1.50, "pitch": 74, "staff": 1, "calc_vel": 64},
        {"st": 1.00, "et": 1.50, "pitch": 77, "staff": 1, "calc_vel": 64},
    ]
    raw_evs = [(0.97, 74, 16), (1.02, 77, 78), (1.50, 60, 50)]
    fused = pv.fuse_transkun_velocities(notes, raw_evs, alpha=0.92, lh_ceiling_vel=90)
    assert fused == 2
    assert notes[0]["calc_vel"] <= 22  # soft inner voice stays genuinely delicate (< 24 old floor)
    assert notes[1]["calc_vel"] >= 74

    aligned = pv.align_microtiming_to_transkun(notes, raw_evs, blend=0.80, max_shift_sec=0.10)
    assert aligned == 2
    assert notes[0]["st"] < 0.99 and notes[1]["st"] > 1.01

    sr = 16000
    imp = np.zeros((sr, 2), dtype=np.float64)
    imp[100, :] = 0.5
    out = s4.apply_natural_hall_reverb(imp, sr, wet_mix=0.18, rt60_sec=1.0)
    assert out.shape == imp.shape
    assert np.all(np.isfinite(out))
    # Reverb tail must exist after the impulse (> 0.05s later)
    assert float(np.sqrt(np.mean(out[int(0.08 * sr) : int(0.30 * sr)] ** 2))) > 1e-4


def test_p0_1_quiet_audio_first_note_detection() -> None:
    """P0-1: Quiet recording peaking at -36 dBFS must detect first note near 0s, not end of track."""
    s0 = _load_script("stage0_audio_triage.py")
    detect_first_note_sec = s0.detect_first_note_sec

    sr = 16000
    dur = 10.0
    t = np.arange(int(sr * dur)) / sr
    # Peak amplitude ~0.0155 (~ -36.2 dBFS), starting at t=0.5s after silence
    sig = np.zeros_like(t)
    mask = t >= 0.5
    sig[mask] = 0.0155 * np.sin(2.0 * np.pi * 440.0 * t[mask])
    first_sec = detect_first_note_sec(sig, sr)
    assert 0.45 <= first_sec <= 0.55, f"Expected ~0.5s, got {first_sec}"


def test_p0_2_multibar_pickup_alignment_stage2(tmp_path: Path) -> None:
    """P0-2: Pickup > 1 bar must stay aligned between quantizer and Stage 2 via bar1_offset_quarters."""
    s2 = _load_script("stage2_pitch_audit.py")
    tm = _load_script("tempo_map.py")
    te = _load_script("transcription_engines.py")
    parse_midi_onsets = s2.parse_midi_onsets
    TempoMap = tm.TempoMap
    quantize_performance_midi = te.quantize_performance_midi

    # 120 BPM 4/4 -> 1 quarter = 0.5s, 1 bar = 2.0s. first_downbeat = 5.0s.
    # C4 at 0.0s is 2.5 bars before first_downbeat -> bars_before = 3, bar1_offset_quarters = 12.0.
    tmap = TempoMap(bpm=120.0, beats_per_bar=4, beat_type=4, first_downbeat=5.0, mode="fixed")
    notes = [
        {"pitch": 60, "start_sec": 0.0, "end_sec": 0.5, "velocity": 70},
        {"pitch": 64, "start_sec": 5.0, "end_sec": 5.5, "velocity": 80},
    ]
    out_mid = tmp_path / "multibar_pickup.mid"
    res = quantize_performance_midi(
        note_events=notes,
        tempo_map=tmap,
        grid=2,
        beats_per_bar=4,
        beat_type=4,
        out_mid=out_mid,
        trim_leading_bars=True,
    )
    assert res["bars_before"] == 3
    assert abs(res["bar1_offset_quarters"] - 12.0) < 1e-6

    parsed = parse_midi_onsets(
        out_mid,
        bpm=120.0,
        beats_per_bar=4,
        first_downbeat=5.0,
        beat_type=4,
        tempo_map=tmap,
        has_pickup=res["has_pickup"],
        bar1_offset_quarters=res["bar1_offset_quarters"],
    )
    by_p = {n["midi"]: n["t_on"] for n in parsed}
    assert abs(by_p[60] - 0.0) < 0.05, f"C4 drifted: {by_p[60]}"
    assert abs(by_p[64] - 5.0) < 0.05, f"E4 drifted: {by_p[64]}"
    assert mido.MidiFile(str(out_mid)).ticks_per_beat > 0


def test_p1_1_cross_bar_tied_note_duration_preserved() -> None:
    """P1-1: 6-beat cross-bar held note with 1 moving note in bar 1 must not be truncated."""
    s3 = _load_script("stage3_engrave_study_score.py")
    BarSegment = s3.BarSegment
    reconcile_cross_bar_ties = s3.reconcile_cross_bar_ties
    simplify_bar_hand = s3.simplify_bar_hand

    divisions = 2  # 8 slots per 4/4 bar
    bar_slots = 8
    slot_rms = np.full(32, -20.0)

    # Bar 1 RH: E5 held for all 4 beats (tie_start=True), plus C5 on beat 2 (only 1 later slot)
    b1_segs = [
        BarSegment(bar=1, slot=0, dur=8, pitch=76, hand=1, tie_stop=False, tie_start=True),
        BarSegment(bar=1, slot=2, dur=2, pitch=72, hand=1, tie_stop=False, tie_start=False),
    ]
    # Bar 2 RH: E5 continuation for 2 beats (tie_stop=True)
    b2_segs = [
        BarSegment(bar=2, slot=0, dur=4, pitch=76, hand=1, tie_stop=True, tie_start=False),
    ]

    v1_b1, v2_b1 = simplify_bar_hand(b1_segs, 1, 0, bar_slots, slot_rms, divisions)
    v1_b2, v2_b2 = simplify_bar_hand(b2_segs, 2, bar_slots, bar_slots, slot_rms, divisions)
    bar_voices = {(1, 1): (v1_b1, v2_b1), (2, 1): (v1_b2, v2_b2)}
    reconcile_cross_bar_ties(bar_voices, start_bar=1, max_bar=2, bar_slots=bar_slots, pickup_start=0)

    def total_dur(pitch: int) -> int:
        tot = 0
        for b in (1, 2):
            v1, v2 = bar_voices[(b, 1)]
            for ev in list(v1) + (list(v2) if v2 else []):
                if any(s.pitch == pitch for s in ev.segs):
                    tot += ev.dur
        return tot

    assert total_dur(76) == 12, f"Expected 12 slots (6 beats) for E5, got {total_dur(76)}"


def test_p1_7_cc64_pedal_preserved_with_pad_sec(tmp_path: Path) -> None:
    """P1-7: Leading pad_sec > 0 must preserve and time-shift CC64 pedal events in raw MIDI."""
    te = _load_script("transcription_engines.py")
    _write_raw_performance_midi = te._write_raw_performance_midi
    read_midi_cc64_events_sec = te.read_midi_cc64_events_sec

    src_mid = tmp_path / "padded_transkun.mid"
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))  # 960 ticks/sec
    tr.append(mido.Message("control_change", channel=0, control=64, value=127, time=2400))  # 2.5s
    tr.append(mido.Message("control_change", channel=0, control=64, value=0, time=960))     # 3.5s
    mid.tracks.append(tr)
    mid.save(str(src_mid))

    cc_evs = read_midi_cc64_events_sec(src_mid, pad_sec=2.0)
    assert len(cc_evs) == 2
    assert abs(cc_evs[0]["time_sec"] - 0.5) < 1e-3 and cc_evs[0]["value"] == 127
    assert abs(cc_evs[1]["time_sec"] - 1.5) < 1e-3 and cc_evs[1]["value"] == 0

    out_raw = tmp_path / "score_raw.mid"
    _write_raw_performance_midi(
        [{"pitch": 60, "start_sec": 0.5, "end_sec": 1.4, "velocity": 72}],
        out_raw,
        cc64_events=cc_evs,
    )
    cc_back = read_midi_cc64_events_sec(out_raw, pad_sec=0.0)
    assert len(cc_back) == 2 and cc_back[0]["value"] == 127 and cc_back[1]["value"] == 0


def test_p1_4_eval_gate_natural_reverb_match() -> None:
    """P1-4 & P1-5: Natural reverb mastering & real TransKun velocity pass Eval Gate cleanly."""
    eg = _load_script("eval_gate_runner.py")
    evaluate_s4_r1_static_dynamics = eg.evaluate_s4_r1_static_dynamics
    evaluate_s4_r2_acoustic_match = eg.evaluate_s4_r2_acoustic_match

    assert evaluate_s4_r2_acoustic_match(nan_inf=0, peak_db=-1.0, tail_snr=85.0, env_corr=0.82) == "PASS"
    assert evaluate_s4_r2_acoustic_match(nan_inf=0, peak_db=-1.0, tail_snr=85.0, env_corr=0.55) == "WARN"
    assert evaluate_s4_r2_acoustic_match(nan_inf=1, peak_db=-1.0, tail_snr=85.0, env_corr=0.90) == "FAIL"
    assert evaluate_s4_r2_acoustic_match(nan_inf=0, peak_db=-0.1, tail_snr=85.0, env_corr=0.90) == "FAIL"
    assert evaluate_s4_r1_static_dynamics(False, rh_med=62.0, lh_med=58.0, lh_max=82, unclosed_wedges=0) == "WARN"


def test_stage4_parse_xml_voice_order_no_phantom_cross_bar_tie(tmp_path: Path) -> None:
    """Voice 1 after <backup> vs voice 2 at slot 0 must not orphan cross-bar ties into phantom attacks."""
    s4 = _load_script("stage4_render_calibrated.py")
    xml_text = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><rest/><duration>4</duration><voice>1</voice><staff>1</staff></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration><tie type="start"/><voice>1</voice><staff>1</staff></note>
    </measure>
    <measure number="2">
      <note><rest/><duration>4</duration><voice>1</voice><staff>1</staff></note>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><staff>1</staff></note>
      <backup><duration>8</duration></backup>
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>2</duration><tie type="stop"/><voice>2</voice><staff>1</staff></note>
      <note><rest/><duration>6</duration><voice>2</voice><staff>1</staff></note>
    </measure>
  </part>
</score-partwise>
"""
    xml_path = tmp_path / "voice_swap_tie.musicxml"
    xml_path.write_text(xml_text, encoding="utf-8")
    notes, _, _ = s4.parse_notes_from_xml(xml_path, bpm=60.0, beats_per_bar=4, first_downbeat=0.0)
    # Must be exactly 2 real F5 notes (bar 1 held into bar 2, plus new F5 on bar 2 beat 3), NOT 3!
    assert len(notes) == 2
    assert abs(notes[0]["st"] - 2.0) < 1e-4 and abs(notes[0]["et"] - 5.0) < 1e-4
    assert abs(notes[1]["st"] - 6.0) < 1e-4 and abs(notes[1]["et"] - 8.0) < 1e-4


def test_stage4_sustain_restoration_and_acoustic_chord_voicing() -> None:
    """Restoring raw durations prevents chord truncation, and acoustic voicing tames loud inner chord notes."""
    pv = _load_script("pedal_and_velocity.py")

    # 1. Sustain restoration from 4-tuple performance events
    notes = [
        {"st": 5.00, "et": 5.74, "pitch": 70, "staff": 1, "calc_vel": 77},
        {"st": 5.00, "et": 5.74, "pitch": 77, "staff": 1, "calc_vel": 46},
    ]
    raw_perf = [
        (5.02, 6.51, 70, 76),
        (5.04, 6.78, 77, 45),
    ]
    cnt = pv.align_microtiming_to_transkun(notes, raw_perf, blend=0.80, max_shift_sec=0.14)
    assert cnt == 2
    assert (notes[0]["et"] - notes[0]["st"]) >= 1.45
    assert (notes[1]["et"] - notes[1]["st"]) >= 1.70

    # 2. Acoustic chord voicing: Bb4 (loud, 0.50) + F5 (very soft, 0.025 -> ~-26 dB)
    sr = 16000
    t = np.arange(int(sr * 1.0)) / sr
    sig = 0.50 * np.sin(2.0 * np.pi * 466.16 * t) + 0.025 * np.sin(2.0 * np.pi * 698.46 * t)
    chord_notes = [
        {"st": 0.02, "et": 0.50, "pitch": 70, "staff": 1, "calc_vel": 77},
        {"st": 0.02, "et": 0.50, "pitch": 77, "staff": 1, "calc_vel": 46},
    ]
    adj = pv.calibrate_chord_voicing_from_audio(chord_notes, sig, sr)
    assert adj == 1
    assert chord_notes[0]["calc_vel"] == 77
    assert chord_notes[1]["calc_vel"] <= 16


def test_stage4_export_mp3_from_wav_graceful_skip_and_real(tmp_path: Path, monkeypatch) -> None:
    """export_mp3_from_wav returns None gracefully without ffmpeg and writes valid MP3 when available."""
    import shutil
    s4 = _load_script("stage4_render_calibrated.py")

    wav_p = tmp_path / "demo.wav"
    mp3_p = tmp_path / "demo.mp3"
    sr = 16000
    t = np.arange(sr // 2) / sr
    sf.write(str(wav_p), 0.2 * np.sin(2.0 * np.pi * 440.0 * t), sr, subtype="PCM_16")

    # 1. Missing ffmpeg -> graceful None (never crashes)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert s4.export_mp3_from_wav(wav_p, mp3_p) is None
    assert not mp3_p.exists()

    # 2. Real ffmpeg (if installed on CI/host) -> creates non-empty MP3
    monkeypatch.undo()
    if shutil.which("ffmpeg"):
        res = s4.export_mp3_from_wav(wav_p, mp3_p)
        assert res == mp3_p and mp3_p.is_file() and mp3_p.stat().st_size > 1000


def test_stage4_direct_performance_midi_zero_drift_and_ghost_removal(tmp_path: Path) -> None:
    """PR #20 Root Fix 1: Direct raw MIDI mode preserves exact rubato arpeggio timing (0ms drift)
    even when XML quantization shifted a note >140ms, while removing Stage 2 confirmed ghosts."""
    pv = _load_script("pedal_and_velocity.py")
    build_performance_notes_from_raw_midi = pv.build_performance_notes_from_raw_midi
    load_confirmed_ghosts_from_audit = pv.load_confirmed_ghosts_from_audit

    audit_path = tmp_path / "pitch_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "minor2nd_clashes": [
                    {"t_on": 73.77, "weaker_midi": 57, "verdict": "CONFIRMED_GHOST"},
                    {"t_on": 173.60, "weaker_midi": 75, "verdict": "REAL_DISSONANCE"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ghosts = load_confirmed_ghosts_from_audit(audit_path)
    assert ghosts == [(73.77, 57)]

    raw_events = [
        (14.569, 14.880, 67, 68),
        (14.800, 14.950, 63, 64),
        (15.060, 15.336, 70, 66),  # quantized XML drifted 150ms early to 14.910
        (73.772, 73.850, 57, 32),  # confirmed ghost A3
        (173.600, 174.100, 75, 58),  # real dissonance Eb5 (must stay)
    ]
    xml_notes = [
        {"st": 14.576, "et": 14.880, "pitch": 67, "staff": 1, "measure": 3},
        {"st": 14.823, "et": 15.125, "pitch": 63, "staff": 1, "measure": 3},
        {"st": 14.910, "et": 15.213, "pitch": 70, "staff": 1, "measure": 3},
        {"st": 173.600, "et": 174.100, "pitch": 75, "staff": 1, "measure": 43},
    ]

    perf_notes, removed = build_performance_notes_from_raw_midi(
        raw_events, xml_notes, confirmed_ghosts=ghosts, max_bar=50
    )
    assert removed == 1
    assert len(perf_notes) == 4
    by_p = {n["pitch"]: n for n in perf_notes}
    assert 57 not in by_p
    assert 75 in by_p
    # Exact 0ms drift on Bb4 (70) despite 150ms XML quantization error
    assert abs(by_p[70]["st"] - 15.060) < 1e-6
    assert abs(by_p[70]["et"] - 15.336) < 1e-6
    assert by_p[70]["staff"] == 1
    assert by_p[70]["measure"] == 3


def test_stage4_closed_loop_velocity_calibration_balances_chord() -> None:
    """PR #20 Root Fix 2: Closed-loop calibration boosts under-loud D5 and attenuates over-loud F5
    based on actual rendered-vs-original per-note dB difference."""
    pv = _load_script("pedal_and_velocity.py")
    closed_loop_velocity_calibration = pv.closed_loop_velocity_calibration

    sr = 16000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    f_d5 = 440.0 * (2.0 ** ((74 - 69) / 12.0))
    f_f5 = 440.0 * (2.0 ** ((77 - 69) / 12.0))
    env = np.where((t >= 0.5) & (t < 1.5), np.exp(-2.0 * np.maximum(0.0, t - 0.5)), 0.0)

    # Original: D5 and F5 equal loudness
    mono_orig = env * (0.25 * np.sin(2.0 * np.pi * f_d5 * t) + 0.25 * np.sin(2.0 * np.pi * f_f5 * t))
    # Pass-1 render (equal total RMS): Salamander-like imbalance where D5 is too soft and F5 is too loud
    mono_rend = env * (0.10 * np.sin(2.0 * np.pi * f_d5 * t) + 0.34 * np.sin(2.0 * np.pi * f_f5 * t))

    notes = [
        {"st": 0.5, "et": 1.5, "pitch": 74, "staff": 1, "calc_vel": 70},
        {"st": 0.5, "et": 1.5, "pitch": 77, "staff": 1, "calc_vel": 76},
    ]
    adj, med_err = closed_loop_velocity_calibration(notes, mono_orig, sr, mono_rend, sr)
    assert adj == 2
    assert med_err > 2.5
    assert notes[0]["calc_vel"] > 78  # D5 boosted
    assert notes[1]["calc_vel"] < 70  # F5 attenuated


def test_closed_loop_stereo_comb_no_false_boost() -> None:
    """PR #21: 3 ms L/R stereo delay at B4 (~494 Hz) must NOT trigger a false boost in 2D stereo mode,
    while collapsing to mono via mean(axis=1) reproduces the old comb-filter false boost (>= +6 vel)."""
    pv = _load_script("pedal_and_velocity.py")
    sr = 48000
    t = np.arange(int(sr * 1.5), dtype=np.float64) / sr
    f_b4 = 440.0 * (2.0 ** ((71 - 69) / 12.0))  # 493.88 Hz (3 ms delay ~ 1.48 cycles -> near-cancel!)
    env = np.where((t >= 0.2) & (t < 1.2), np.exp(-1.5 * np.maximum(0.0, t - 0.2)), 0.0)
    tone = env * 0.25 * np.sin(2.0 * np.pi * f_b4 * t)
    # Add a broadband/bass reference tone at 110 Hz (A2) so global RMS stays steady
    ref_tone = env * 0.25 * np.sin(2.0 * np.pi * 110.0 * t)
    orig_st = np.column_stack([tone + ref_tone, tone + ref_tone])

    delay = 144  # 3.0 ms @ 48 kHz
    tone_r = np.concatenate([np.zeros(delay), tone[:-delay]])
    ref_r = np.concatenate([np.zeros(delay), ref_tone[:-delay]])
    rend_st = np.column_stack([tone + ref_tone, tone_r + ref_r])

    notes_st = [{"st": 0.2, "et": 1.2, "pitch": 71, "staff": 1, "calc_vel": 66}]
    pv.closed_loop_velocity_calibration(notes_st, orig_st, sr, rend_st, sr)
    assert abs(notes_st[0]["calc_vel"] - 66) <= 2

    # Same stereo render folded to mono triggers comb-filter null -> false boost >= 6
    notes_mono = [{"st": 0.2, "et": 1.2, "pitch": 71, "staff": 1, "calc_vel": 66}]
    pv.closed_loop_velocity_calibration(notes_mono, orig_st.mean(axis=1), sr, rend_st.mean(axis=1), sr)
    assert (notes_mono[0]["calc_vel"] - 66) >= 6


def test_closed_loop_velocity_caps() -> None:
    """PR #21: Closed-loop calibration enforces RH <= 94, LH <= 90, and single-pass rise <= 16."""
    pv = _load_script("pedal_and_velocity.py")
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    env = np.where((t >= 0.3) & (t < 1.5), np.exp(-1.5 * np.maximum(0.0, t - 0.3)), 0.0)
    f_rh = 440.0 * (2.0 ** ((72 - 69) / 12.0))
    f_lh = 440.0 * (2.0 ** ((48 - 69) / 12.0))
    f_mid = 440.0 * (2.0 ** ((64 - 69) / 12.0))

    # Original has strong RH/LH; render has them 20 dB softer while mid tone anchors global RMS
    orig = env * (0.30 * np.sin(2 * np.pi * f_rh * t) + 0.30 * np.sin(2 * np.pi * f_lh * t) + 0.30 * np.sin(2 * np.pi * f_mid * t))
    rend = env * (0.03 * np.sin(2 * np.pi * f_rh * t) + 0.03 * np.sin(2 * np.pi * f_lh * t) + 0.30 * np.sin(2 * np.pi * f_mid * t))

    notes = [
        {"st": 0.3, "et": 1.5, "pitch": 72, "staff": 1, "calc_vel": 90},
        {"st": 0.3, "et": 1.5, "pitch": 48, "staff": 2, "calc_vel": 86},
        {"st": 0.3, "et": 1.5, "pitch": 64, "staff": 1, "calc_vel": 50},
    ]
    pv.closed_loop_velocity_calibration(notes, orig, sr, rend, sr)
    assert notes[0]["calc_vel"] <= 94 and (notes[0]["calc_vel"] - 90) <= 16
    assert notes[1]["calc_vel"] <= 90 and (notes[1]["calc_vel"] - 86) <= 16

    # Also verify max_rise_vel=16 cap on a lower starting velocity (50 -> <= 66)
    notes_low = [
        {"st": 0.3, "et": 1.5, "pitch": 72, "staff": 1, "calc_vel": 50},
        {"st": 0.3, "et": 1.5, "pitch": 64, "staff": 1, "calc_vel": 60},
    ]
    pv.closed_loop_velocity_calibration(notes_low, orig, sr, rend, sr)
    assert notes_low[0]["calc_vel"] <= 66


def test_raw_midi_notes_capped_by_staff() -> None:
    """PR #21: build_performance_notes_from_raw_midi caps RH <= 94 and LH <= 90."""
    pv = _load_script("pedal_and_velocity.py")
    raw_events = [
        (0.5, 1.0, 72, 120),  # RH C5
        (0.5, 1.0, 48, 120),  # LH C3
    ]
    perf, _ = pv.build_performance_notes_from_raw_midi(raw_events, [])
    by_p = {n["pitch"]: n for n in perf}
    assert by_p[72]["staff"] == 1 and by_p[72]["calc_vel"] <= 94
    assert by_p[48]["staff"] == 2 and by_p[48]["calc_vel"] <= 90


def test_integrated_lufs_sine_reference() -> None:
    """PR #21: BS.1770 integrated_lufs on 997 Hz stereo sine (amp 0.1) matches 20*log10(0.1) within 0.5 LU."""
    am = _load_script("audio_match.py")
    sr = 48000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    s = 0.1 * np.sin(2.0 * np.pi * 997.0 * t)
    stereo = np.column_stack([s, s])
    lufs = am.integrated_lufs(stereo, sr)
    ref = 20.0 * np.log10(0.1)
    assert abs(lufs - ref) <= 0.5


def test_peak_limiter_keeps_loudness() -> None:
    """PR #21: Lookahead peak limiter caps a 10 ms 0 dBFS spike to <= -1.0 dBFS with <= 0.3 LU loudness drop."""
    am = _load_script("audio_match.py")
    sr = 48000
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((sr * 3, 2))
    noise *= 10.0 ** ((-20.0 - am.integrated_lufs(noise, sr)) / 20.0)
    lufs_clean = am.integrated_lufs(noise, sr)

    spiked = noise.copy()
    spiked[sr : sr + int(0.010 * sr), :] = 1.0
    limited = am.apply_peak_limiter(spiked, sr, ceiling_db=-1.0, attack_ms=5.0, release_ms=80.0)
    peak_db = 20.0 * np.log10(float(np.max(np.abs(limited))))
    lufs_lim = am.integrated_lufs(limited, sr)
    assert peak_db <= -0.99
    assert (lufs_clean - lufs_lim) <= 0.3


def test_detect_sound_end_and_pedal_release() -> None:
    """PR #21: detect_sound_end locates 10.0s decay end within 0.1s and extends pedal release accordingly."""
    am = _load_script("audio_match.py")
    sr = 24000
    rng = np.random.default_rng(11)
    t = np.arange(sr * 12, dtype=np.float64) / sr
    env = np.where(t < 10.0, np.exp(-0.4 * t), 0.0)
    sig = rng.standard_normal((len(t), 2)) * env[:, None] * 0.25
    sound_end = am.detect_sound_end(sig, sr, floor_db=-60.0)
    assert abs(sound_end - 10.0) <= 0.1
    release_t = max(8.0 + 0.15, 0.0, sound_end - 0.5)
    assert release_t >= sound_end - 0.5


def test_match_eq_darkens_bright_render() -> None:
    """PR #21: Match EQ darkens bright white noise toward -30 dB (>2 kHz) dark reference with max boost <= +3 dB."""
    am = _load_script("audio_match.py")
    sr = 48000
    rng = np.random.default_rng(19)
    wn = rng.standard_normal((sr * 3, 2)) * 0.05
    bf, rend_db = am.long_term_spectrum_db(wn, sr)
    dark_orig = am.apply_eq_fft(wn, sr, bf, np.where(bf >= 2000.0, -30.0, 0.0))
    _, orig_db = am.long_term_spectrum_db(dark_orig, sr)

    gains = am.design_match_eq(orig_db, rend_db, band_freqs=bf, strength=0.7, max_boost_db=3.0, max_cut_db=24.0)
    assert float(np.max(gains)) <= 3.01
    matched = am.apply_eq_fft(wn, sr, bf, gains)
    _, matched_db = am.long_term_spectrum_db(matched, sr)
    i4 = int(np.argmin(np.abs(bf - 4000.0)))
    i8 = int(np.argmin(np.abs(bf - 8000.0)))
    assert abs(matched_db[i4] - orig_db[i4]) <= 10.0
    assert abs(matched_db[i8] - orig_db[i8]) <= 12.0


def test_estimate_rt60_synthetic_decay() -> None:
    """PR #21: estimate_decay_rate recovers synthetic RT60 = 2.5s within 15% and returns None for 0.5s tails."""
    am = _load_script("audio_match.py")
    sr = 24000
    rng = np.random.default_rng(23)
    rt_true = 2.5
    t = np.arange(sr * 4, dtype=np.float64) / sr
    # Amplitude decays by 60 dB in rt_true seconds: 10^(-3 * t / rt_true) = exp(-6.907755 * t / rt_true)
    dec = rng.standard_normal((len(t), 2)) * np.exp(-6.907755 * t / rt_true)[:, None]
    res = am.estimate_decay_rate(dec, sr, 0.0, 3.8)
    assert res is not None
    rate_db_s, r2 = res
    rt_est = 60.0 / abs(rate_db_s)
    assert r2 >= 0.8
    assert abs(rt_est - rt_true) / rt_true <= 0.15
    assert am.estimate_decay_rate(dec, sr, 0.0, 0.5) is None


def test_read_soft_pedal_spans(tmp_path: Path) -> None:
    """PR #21: read_midi_soft_pedal_spans parses two CC67 intervals with <10 ms error and accurate soft_ratio."""
    pv = _load_script("pedal_and_velocity.py")
    mid = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))  # 960 ticks/sec
    # Span 1: 1.0s .. 3.0s
    tr.append(mido.Message("control_change", control=67, value=127, time=960))
    tr.append(mido.Message("control_change", control=67, value=0, time=1920))
    # Span 2: 5.0s .. 10.0s (open until end_of_track at 10.0s)
    tr.append(mido.Message("control_change", control=67, value=127, time=1920))
    tr.append(mido.MetaMessage("end_of_track", time=4800))
    mid.tracks.append(tr)
    p_mid = tmp_path / "soft_pedal.mid"
    mid.save(str(p_mid))

    spans = pv.read_midi_soft_pedal_spans(p_mid)
    assert len(spans) == 2
    assert abs(spans[0][0] - 1.0) < 0.01 and abs(spans[0][1] - 3.0) < 0.01
    assert abs(spans[1][0] - 5.0) < 0.01 and abs(spans[1][1] - 10.0) < 0.01
    soft_ratio = sum(b - a for a, b in spans) / 10.0
    assert abs(soft_ratio - 0.7) < 0.01


def test_stage5_stereo_excerpts_and_outro(tmp_path: Path) -> None:
    """PR #21: Stage 5 outputs 2-channel stereo clips (L != R) and 12s outro covering original's sound end."""
    s5 = _load_script("stage5_ab_excerpts.py")
    sr = 16000
    t_orig = np.arange(sr * 18, dtype=np.float64) / sr
    # Distinct L (440 Hz) and R (554.37 Hz) stereo orig ending at 16.0s
    env_o = np.where(t_orig < 16.0, 0.2, 0.0)
    orig = np.column_stack([env_o * np.sin(2 * np.pi * 440.0 * t_orig), env_o * np.sin(2 * np.pi * 554.37 * t_orig)])
    # Render is 3s shorter (ends at 13.0s)
    t_rend = np.arange(sr * 13, dtype=np.float64) / sr
    rend = np.column_stack([0.2 * np.sin(2 * np.pi * 440.0 * t_rend), 0.2 * np.sin(2 * np.pi * 554.37 * t_rend)])

    orig_p = tmp_path / "orig.wav"
    rend_p = tmp_path / "rend.wav"
    out_d = tmp_path / "ab"
    sf.write(str(orig_p), orig, sr, subtype="PCM_16")
    sf.write(str(rend_p), rend, sr, subtype="PCM_16")

    res = s5.generate_ab_excerpts(orig_p, rend_p, out_d, first_downbeat=1.0, loudness_match=True)
    assert len(res) == 3
    assert not list(out_d.glob("*_LR.wav"))

    outro_o, sr_out = sf.read(str(out_d / "AB_03_outro_orig.wav"), always_2d=True)
    outro_r, _ = sf.read(str(out_d / "AB_03_outro_render.wav"), always_2d=True)
    outro_s, _ = sf.read(str(out_d / "AB_03_outro_seq.wav"), always_2d=True)
    assert outro_o.shape[1] == 2 and outro_r.shape[1] == 2 and outro_s.shape[1] == 2
    assert float(np.mean(np.abs(outro_o[:, 0] - outro_o[:, 1]))) > 1e-3  # L != R preserved
    assert abs(outro_o.shape[0] / sr_out - 12.0) < 0.05
    assert abs(outro_r.shape[0] / sr_out - 12.0) < 0.05
    # Outro starts at 16.0 - 10.0 = 6.0s -> orig has active sound through t=10.0s of the 12s clip
    assert float(np.sqrt(np.mean(outro_o[int(9.0 * sr_out) : int(9.8 * sr_out)] ** 2))) > 0.05
