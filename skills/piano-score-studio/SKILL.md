---
name: piano-score-studio
description: Convert solo piano audio (WAV/MP3/FLAC) into a human-readable pedagogical piano score (PDF/MusicXML) and calibrated Salamander Grand Piano audio/MIDI through a v3.0 multi-style pipeline (MIT TransKun v2 default, 6-dim Musical DNA style profiling, direct MIDI pitch audit, chroma-flux dynamic pedaling, and bounded local bar repair). Use when transcribing solo piano recordings, cleaning up messy MIDI-imported scores for human practice, calibrating piano dynamics/pedaling, or rendering verification audio.
---

# Piano Score Studio (v3.0)

End-to-end workflow for turning **multi-style solo piano audio** (Pop/Ballad, Classical/Romantic rubato, Baroque/Bach polyphony, Jazz groove) into:
1. A **human-readable study sheet music edition** (`score_learning.pdf` + `score_learning.musicxml`)
2. A **calibrated high-quality Salamander Grand Piano render** (`*_Full_Calibrated_Salamander.wav`, `*_Full_Calibrated_Salamander.mid`, and 12s A/B comparison excerpts)

All scripts live under `scripts/`, style profiles under `profiles/`, and detailed reference specs under `references/`.

---

## One-Command CLI Quick Start (Recommended)

Run the full v3.0 pipeline (with `project_manifest.json` IR, parameter-hash `--resume`, and optional `--auto-repair`) from the repository root:

```bash
python piano_studio.py input/my_piano.wav -o output/my_piano
```

Useful flags:
- `--engine auto|transkun|muscriptor` — Default `auto` prefers MIT **TransKun v2** (zero HuggingFace login, preserves real velocities); falls back to `muscriptor` if installed.
- `--style auto|pop_ballad|classical_romantic|baroque_polyphonic|jazz_groove` — Default `auto` classifies the piece via Krumhansl-Schmuckler key detection + 6-dim Musical DNA (`Stage 1.5`).
- `--mode auto|fixed-tempo|tempo-map` — Automatically selects `tempo-map` rubato alignment when 20s phase drift exceeds ±30 ms.
- `--resume` — Resumes from `project_manifest.json` and automatically invalidates downstream stages if CLI parameters (`--bpm`, `--engine`, `--style`, etc.) changed.
- `--auto-repair` — Runs 1 bounded `bar_overrides` local repair pass if Eval Gate flags repairable bar issues.
- `--skip-pdf` / `--skip-render` / `--analyze-only` — Fast partial runs (Stage 2 direct MIDI pitch audit still runs under `--skip-pdf`).

---

## Core Principle: What Makes a Score Truly Pass for Human Reading?

A score NEVER passes merely because MuseScore exited `0` or every note matches the audio pitch. Before marking Stage 3 complete, the score MUST satisfy all 8 dimensions in [human_readable_score_checklist.md](references/human_readable_score_checklist.md):

1. **Clef Stability**: `0` intra-bar clef flips. Lock RH = Treble (`G2`), LH = Bass (`F4`). Only shift LH to Treble at a clean barline if `>= 2` consecutive full bars stay at `>= C4 (60)`.
2. **Voice & Rest Hygiene**: Proper anacrusis (`<measure number="0" implicit="yes">`, driven explicitly by Stage 1 `has_pickup`); default `1` voice per hand, strict max `2` voices (never `3`+); structural spacer rests in secondary voices hidden via `print-object="no"`; micro-gaps (`<= 1` eighth internal or `<= 1` beat bar-end) bridged when audio RMS `> -38 dBFS`; `>= 50%` visible-rest reduction vs raw import; `0` voice duration errors.
3. **Playable Hand-Splitting & Span**: Prefer simultaneous hand span `<= 12` semitones; mark genuine wide chords (`> 12` semitones) with `<arpeggiate/>`; keep RH melody out of LH; keep LH p95 `<= F4 (65)`.
4. **Metric Beaming & Tie Discipline**: Beat-grouped beaming (`3/4` = `2+2+2`, `4/4` = `4+4` or `2+2+2+2`, `6/8` = `3+3` with natural eighth+quarter grouping); split ties across strong beats with balanced cross-bar tie reconciliation (`0` orphaned ties); `< 2%` spurious `16th/32nd` notes; explicit `<tuplet>` brackets for triplets.
5. **Key-Consistent Enharmonic Spelling**: Spell accidentals according to the detected or specified key signature (default `--key-fifths 0` when run standalone) with clean bar-local accidental tracking.
6. **Restrained Pedagogical Markings**: Phrase-level dynamics (`pp`–`f`), `100%` paired/closed wedges (`cresc.` / `dim.`), tempo/character header, style-appropriate pedal indication (suppressed for Baroque `dry_articulated`), `rit.`, and final double barline (`light-heavy`).
7. **Balanced Book Layout**: A4 comfortable scaling (`7 mm = 40 tenths`), 4-bar phrase systems (5 bars only on sparse sections), `4` systems on Page 1 (with clean non-duplicated title/subtitle) and `5` systems on Pages 2+, zero orphan 1–2 bar last page.
8. **Mandatory 2-Round Review**:
   - **Round 1**: Automated XML audit (`0` duration errors, `<= 2` voices, `0` mid-piece clef flips, `0` unclosed ties).
   - **Round 2**: **Visual page-by-page inspection of ALL rendered PDF pages** (zero text/note collisions, zero duplicate titles, legible ledger lines, clean spacing). Fix and re-render immediately if any page fails.

---

## Mandatory 6-Stage Execution Pipeline (Individual Script Invocation)

Never skip a stage or bypass a blocking `FAIL` in [eval_rubric_v2.json](references/eval_rubric_v2.json) (thresholds adapt dynamically to the active `profiles/*.json`).

### Stage 0 — Audio Triage & Source Selection
Check `setup_assets.py --check` first if environment readiness is unknown, then triage all candidate inputs:
```bash
python scripts/stage0_audio_triage.py --inputs <input_dir_or_files> --out <job>/analysis/audio_triage.json
```
- **Blocking checks**: `0` clipping (`>= 0.999`), `0` click jumps (`> 0.1`), `0` dropouts (`< -70 dBFS`).
- **Multi-candidate selection**: Pick the candidate with lowest mono-sum `>6 dB` cancellation %, lowest negative-phase frame %, and cleanest harmonic peak-to-median SNR.
- **Leading-note guard**: If `first_note_sec < 0.15s`, automatic 2s-silence-padded transcription ensures the opening note is not swallowed.

### Stage 1 — Tempo, Meter, Downbeat, Grid Calibration & Transcription
Run the unified calibration and transcription runner (defaults to MIT **TransKun v2**):
```bash
python scripts/stage1_tempo_and_transcribe.py \
  --audio <best_audio> \
  --triage <job>/analysis/audio_triage.json \
  --engine auto --mode auto \
  --out-dir <job>/baseline
```
- Runs `beat_this` (`final0`), sweeps fixed-tempo circular resultant `|R|` over 10 ms onset flux, and checks 20s-window phase drift (`<= ±30 ms` → `fixed-tempo`; larger rubato → `tempo-map`).
- Compares bass flux (`45–260 Hz`) & 12-bin chroma harmonic change across candidate bar cycles (`4/4`, `3/4`, `6/8`, `2/4`) to lock `beats_per_bar`, `beat_type`, `first_downbeat`, and explicit `has_pickup`.
- Picks the coarsest `--grid` (`2`, `3`, or `4`) covering `>= 95%` of onsets (evaluated via `TempoMap.sec_to_q` in rubato mode), deduplicates quantized same-pitch collisions, and writes `stage1_calibration.json`, `score_raw.mid`, and `score.mid`.

### Stage 1.5 — Auto Key Detection & 6-Dim Musical DNA Style Profiling
```bash
python scripts/style_analyzer.py \
  --midi <job>/baseline/score.mid \
  --stage1-cal <job>/baseline/stage1_calibration.json \
  --beats-per-bar <BPB> --beat-type <BEAT_TYPE> --style auto \
  --out <job>/analysis/style_analysis.json
```
- Detects key signature via Krumhansl-Schmuckler and routes to `profiles/{pop_ballad,classical_romantic,baroque_polyphonic,jazz_groove}.json`.

### Stage 2 — Direct MIDI Harmonic Pitch, Clash & Ghost Audit
Reads quantized `score.mid` + `TempoMap` directly (no MuseScore MIDI→MusicXML round-trip needed) and removes confirmed ghost notes into `score_cleaned.mid`:
```bash
python scripts/stage2_pitch_audit.py \
  --audio <best_audio> --midi <job>/baseline/score.mid \
  --bpm <BPM> --beats-per-bar <BPB> --beat-type <BEAT_TYPE> --first-downbeat <T0> \
  --stage1-cal <job>/baseline/stage1_calibration.json \
  --clean-midi <job>/baseline/score_cleaned.mid \
  --style-profile <STYLE_PROFILE> --key-fifths <FIFTHS> \
  --out <job>/analysis/pitch_audit.json
```
- Preserves genuine chromatic/color tones (`SNR >= 15 dB` and `OG >= 6 dB`) with style-calibrated thresholds (e.g., Jazz allows intentional minor-2nd voicings).
- Removes confirmed sustain-tail ghosts using `TempoMap`-aligned audio timestamps (`first_downbeat` & rubato aware).

### Stage 3 — Pedagogical Study Score Engraving & 2-Round Review
Engrave the cleaned study edition from `score_cleaned.mid`, `bar_lh_rh_envelope.csv`, style profile, and optional `bar_overrides`:
```bash
python scripts/stage3_engrave_study_score.py \
  --midi <job>/baseline/score_cleaned.mid --audio <best_audio> \
  --title "<Song Title>" --bpm <BPM> --beats-per-bar <BPB> \
  --beat-type <BEAT_TYPE> --grid <GRID> \
  --first-downbeat <T0> --stage1-cal <job>/baseline/stage1_calibration.json \
  --key-fifths <FIFTHS> --dynamics-csv <job>/analysis/bar_lh_rh_envelope.csv \
  --style-profile <STYLE_PROFILE> --bar-overrides <job>/project_manifest.json \
  --out-xml <job>/score/score_learning.musicxml \
  --out-pdf <job>/score/score_learning.pdf
```
- Execute **Round 1** (XML audit printed by script) + **Round 2** (visually inspect EVERY page of `score_learning.pdf`).

### Stage 4 — Dynamics Plan → Velocity Fusion & Dynamic Pedaling → 2-Round Verification
Flat `velocity = 100` MIDI is strictly a placeholder and **MUST NEVER** be used for final audio rendering.

1. **Plan First (Stage 4.1)** — Separate original audio into `LH (45–350 Hz)` and `RH (350–3500 Hz)` bands:
   ```bash
   python scripts/stage4_dynamics_plan.py \
     --audio <best_audio> --bpm <BPM> --beats-per-bar <BPB> --beat-type <BEAT_TYPE> \
     --first-downbeat <T0> --stage1-cal <job>/baseline/stage1_calibration.json \
     --num-bars <NUM_BARS> --out-dir <job>/analysis
   ```
2. **Execute & Render (Stage 4.2–4.4)** — Fuse real TransKun performance velocities with acoustic loudness, generate chroma-flux dynamic `CC64` pedaling (`dry_articulated` / `legato_harmonic` / `expressive_rubato` / `rhythmic_half_pedal`), render via `Salamander Grand Piano` (`sfizz_render` CLI or `NativeHost`), and apply two-band phrase mastering:
   ```bash
   python scripts/stage4_render_calibrated.py \
     --xml <job>/score/score_learning.musicxml --orig-audio <best_audio> \
     --bpm <BPM> --beats-per-bar <BPB> --first-downbeat <T0> \
     --stage1-cal <job>/baseline/stage1_calibration.json \
     --key-fifths <FIFTHS> --song-name "<SongName>" \
     --style-profile <STYLE_PROFILE> --transkun-midi <job>/baseline/score.mid \
     --out-dir <job>/deliverables
   ```
   - **Round 1 (Static MIDI/XML check)**: Style-calibrated `RH_med - LH_med` & `LH_max` thresholds, `0` placeholder-only tracks, `100%` closed wedges.
   - **Round 2 (Closed-loop acoustic check)**: `NaN/Inf = 0`, Tail SNR `>= 80 dB`, per-bar `LH` & `RH` median error within `±1.5 dB` of original audio, `20–150 Hz` energy share diff `<= 4%`.

### Stage 5 — A/B Listening Excerpts, Eval Gate & Bounded Bar Repair
```bash
python scripts/stage5_ab_excerpts.py \
  --orig <best_audio> --calibrated <job>/deliverables/<SongName>_Full_Calibrated_Salamander.wav \
  --bpm <BPM> --beats-per-bar <BPB> --first-downbeat <T0> \
  --out-dir <job>/deliverables/ab_excerpts

python scripts/eval_gate_runner.py \
  --rubric references/eval_rubric_v2.json \
  --triage <job>/analysis/audio_triage.json \
  --pitch-audit <job>/analysis/pitch_audit.json \
  --dynamics-plan <job>/analysis/dynamics_plan.md \
  --dynamics-csv <job>/analysis/bar_lh_rh_envelope.csv \
  --xml <job>/score/score_learning.musicxml \
  --pdf <job>/score/score_learning.pdf \
  --midi <job>/deliverables/<SongName>_Full_Calibrated_Salamander.mid \
  --orig-audio <best_audio> \
  --cal-audio <job>/deliverables/<SongName>_Full_Calibrated_Salamander.wav \
  --ab-dir <job>/deliverables/ab_excerpts \
  --bpm <BPM> --beats-per-bar <BPB> --first-downbeat <T0> \
  --stage1-cal <job>/baseline/stage1_calibration.json \
  --style-profile <STYLE_PROFILE> --pdf-visually-inspected \
  --out <job>/deliverables/eval_report.json
```
Deliverables are complete only when `eval_report.json` reports `"overall_verdict": "PASS"` (`blocking_fails == []`). If repairable bar issues remain, run `scripts/bar_repair_loop.py --write` (or pass `--auto-repair` to `piano_studio.py`).
