---
name: piano-score-studio
description: Convert solo piano audio (WAV/MP3/FLAC) into a human-readable pedagogical piano score (PDF/MusicXML) and calibrated Salamander Grand Piano audio/MIDI through a 6-stage pipeline with strict Eval gates, 2-round score review, and 2-round two-band dynamics verification. Use when transcribing piano recordings, cleaning up messy MIDI-imported scores for human practice, calibrating piano dynamics/pedaling, or rendering verification audio.
---

# Piano Score Studio

End-to-end workflow for turning solo piano audio into:
1. A **human-readable study sheet music edition** (`score_learning.pdf` + `score_learning.musicxml`)
2. A **calibrated high-quality Salamander Grand Piano render** (`*_Full_Calibrated_Salamander.wav`, `*_Full_Calibrated_Salamander.mid`, and 12s A/B comparison excerpts)

All scripts live under `scripts/` inside this skill directory, and detailed reference specs live under `references/`.

---

## Core Principle: What Makes a Score Truly Pass for Human Reading?

A score NEVER passes merely because MuseScore exited `0` or every note matches the audio pitch. Before marking Stage 3 complete, the score MUST satisfy all 8 dimensions in [human_readable_score_checklist.md](references/human_readable_score_checklist.md):

1. **Clef Stability**: `0` intra-bar clef flips. Lock RH = Treble (`G2`), LH = Bass (`F4`). Only shift LH to Treble at a clean barline if `>= 2` consecutive full bars stay at `>= C4 (60)`.
2. **Voice & Rest Hygiene**: Proper anacrusis (`<measure number="0" implicit="yes">`); default `1` voice per hand, strict max `2` voices (never `3`+); structural spacer rests in secondary voices hidden via `print-object="no"`; micro-gaps (`<= 1` eighth internal or `<= 1` beat bar-end) bridged when audio RMS `> -38 dBFS`; `>= 50%` visible-rest reduction vs raw import; `0` voice duration errors.
3. **Playable Hand-Splitting & Span**: Prefer simultaneous hand span `<= 12` semitones; mark genuine wide chords (`> 12` semitones) with `<arpeggiate/>`; keep RH melody out of LH; keep LH p95 `<= F4 (65)`.
4. **Metric Beaming & Tie Discipline**: Beat-grouped beaming (`3/4` = `2+2+2`, `4/4` = `4+4` or `2+2+2+2`, `6/8` = `3+3`); split ties across strong beats; `< 2%` spurious `16th/32nd` notes; trim tiny 1-eighth barline spillover tails.
5. **Key-Consistent Enharmonic Spelling**: Spell accidentals according to the piece's key signature (e.g., in B♭ major use `E♭/A♭/D♭/F♯`, not `A♯/D♯`) with clean bar-local accidental tracking.
6. **Restrained Pedagogical Markings**: Phrase-level dynamics (`pp`–`f`), `100%` paired/closed wedges (`cresc.` / `dim.`), tempo/character header (`Moderato espressivo ♩ = ...`), `con pedale`, `rit.`, and final double barline (`light-heavy`).
7. **Balanced Book Layout**: A4 comfortable scaling (`7 mm = 40 tenths`), 4-bar phrase systems (5 bars only on sparse sections), `4` systems on Page 1 (with clean non-duplicated title/subtitle) and `5` systems on Pages 2+, zero orphan 1–2 bar last page.
8. **Mandatory 2-Round Review**:
   - **Round 1**: Automated XML audit (`0` duration errors, `<= 2` voices, `0` mid-piece clef flips).
   - **Round 2**: **Visual page-by-page inspection of ALL rendered PDF pages** (zero text/note collisions, zero duplicate titles, legible ledger lines, clean spacing). Fix and re-render immediately if any page fails.

---

## Mandatory 6-Stage Execution Pipeline

Never skip a stage or bypass a blocking `FAIL` in [eval_rubric_v2.json](references/eval_rubric_v2.json).

### Stage 0 — Audio Triage & Source Selection
Check `setup_assets.py --check` first if environment readiness is unknown, then triage all candidate inputs:
```bash
python scripts/stage0_audio_triage.py --inputs <input_dir_or_files> --out <job>/analysis/audio_triage.json
```
- **Blocking checks**: `0` clipping (`>= 0.999`), `0` click jumps (`> 0.1`), `0` dropouts (`< -70 dBFS`).
- **Multi-candidate selection**: Pick the candidate with lowest mono-sum `>6 dB` cancellation %, lowest negative-phase frame %, and cleanest harmonic peak-to-median SNR.
- **Leading-note guard**: If `first_note_sec < 0.15s`, run a 2s-silence-padded transcription check to ensure the opening note is not swallowed.

### Stage 1 — Tempo, Meter, Downbeat & Quantization Grid Calibration
Do NOT blindly accept `beat_this` output when compound meters (`3/4` vs `6/8`) confuse it:
- Sweep fixed-tempo circular resultant `|R|` over 10 ms onset flux and check 20s-window phase drift (`<= ±30 ms` → `--fixed-tempo`; larger rubato → `--tempo-map`).
- Compare bass flux & harmonic change across candidate bar cycles to lock `--beats-per-bar` and `--first-downbeat`.
- Pick the coarsest `--grid` (`2`, `3`, or `4`) covering `>= 95%` of onsets, then run transcription (e.g., via `scripts/project.py convert` or `muscriptor`).

### Stage 2 — Transcription Pitch, Clash & Ghost Audit
Run harmonic verification against the original audio before engraving:
```bash
python scripts/stage2_pitch_audit.py \
  --audio <best_audio> --xml <raw_musicxml> \
  --bpm <BPM> --beats-per-bar <BPB> --first-downbeat <T0> \
  --out <job>/analysis/pitch_audit.json
```
- Preserve genuine chromatic/color tones (`SNR >= 15 dB` and `OG >= 6 dB`).
- Remove confirmed sustain-tail ghosts (`f0` > 12 dB weaker than simultaneous harmony or `OG < 3 dB` minor-2nd clashes) and record them in `pitch_audit_log`.

### Stage 3 — Pedagogical Study Score Engraving & 2-Round Review
Engrave the cleaned study edition from the calibrated MIDI & audio:
```bash
python scripts/stage3_engrave_study_score.py \
  --midi <raw_quantized_mid> --audio <best_audio> \
  --title "<Song Title>" --bpm <BPM> --beats-per-bar <BPB> \
  --first-downbeat <T0> --key-fifths <FIFTHS> \
  --dynamics-csv <job>/analysis/bar_lh_rh_envelope.csv \
  --out-xml <job>/score/score_learning.musicxml \
  --out-pdf <job>/score/score_learning.pdf
```
*(Tip: If you want dynamics from Stage 4.1 already embedded in the XML, run Stage 4.1 `stage4_dynamics_plan.py` right before `stage3_engrave_study_score.py`, or re-run Stage 3 after generating `bar_lh_rh_envelope.csv`.)*
- Execute **Round 1** (XML audit printed by script) + **Round 2** (visually inspect EVERY page of `score_learning.pdf`).

### Stage 4 — Dynamics Plan → Execute → 2-Round Verification
Flat `velocity = 100` MIDI is strictly a Stage 2 placeholder and **MUST NEVER** be used for final audio rendering.

1. **Plan First (Stage 4.1)** — Separate original audio into `LH (45–350 Hz)` and `RH (350–3500 Hz)` bands:
   ```bash
   python scripts/stage4_dynamics_plan.py \
     --audio <best_audio> --bpm <BPM> --beats-per-bar <BPB> \
     --first-downbeat <T0> --num-bars <NUM_BARS> \
     --out-dir <job>/analysis
   ```
   Review `dynamics_plan.md` & `bar_lh_rh_envelope.csv` (check `LH - RH dB` median, 5-band spectral profile, and bass taper rules).
2. **Execute & Render (Stage 4.2–4.4)** — Calibrate per-note RH/LH velocities (taming LH bass out of harsh fortissimo layers and honoring `LH - RH` bar caps), generate bar-by-bar `CC64` pedaling, render via `Salamander Grand Piano` (`sfizz_render` CLI or `music-learning` `NativeHost`), and apply two-band phrase mastering:
   ```bash
   python scripts/stage4_render_calibrated.py \
     --xml <job>/score/score_learning.musicxml --orig-audio <best_audio> \
     --bpm <BPM> --beats-per-bar <BPB> --first-downbeat <T0> \
     --key-fifths <FIFTHS> --song-name "<SongName>" \
     --out-dir <job>/deliverables
   ```
   - **Round 1 (Static MIDI/XML check before/inside render)**: `0` `vel=100` placeholder-only tracks, `RH_med - LH_med >= 15` (when audio shows melody-dominant balance), `LH_max <= 78`, `100%` closed wedges.
   - **Round 2 (Closed-loop acoustic check)**: `NaN/Inf = 0`, Tail SNR `>= 80 dB`, per-bar `LH` & `RH` median error within `±1.5 dB` of original audio, `20–150 Hz` energy share diff `<= 4%`. Re-tune and re-render if any check fails.

### Stage 5 — A/B Listening Excerpts & Final Eval Gate
Generate 12s A/B excerpts (Opening, Climax, Outro) and run the full rubric gate:
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
  --pdf-visually-inspected \
  --out <job>/deliverables/eval_report.json
```
Deliverables are complete only when `eval_report.json` reports `"overall_verdict": "PASS"` (`blocking_fails == []`).
