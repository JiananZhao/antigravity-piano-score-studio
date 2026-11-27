# Antigravity Piano Score Studio (v3.0)

**Drop solo piano audio in, get publication-grade study sheet music (PDF / MusicXML) and calibrated Salamander Grand Piano audio/MIDI out — in one command.**

Designed for the open-source community and Antigravity agents, v3.0 generalizes across **multi-style solo piano repertoire** (Pop/Ballad, Classical/Romantic rubato, Baroque/Bach polyphony, and Jazz swing/groove) with zero manual threshold tuning.

---

## What's New in v3.0

1. **One-Command CLI & Canonical IR (`piano_studio.py` + `project_manifest.json`)**
   - Single entrypoint orchestrating Stages 0 → 1 → 1.5 → 2 → 4.1 → 3 → 4.2–4.4 → 5 with crash-safe `--resume`.
2. **Pluggable Transcription Engine (`TransKun v2` Default + `MuScriptor` Fallback)**
   - Uses MIT-licensed [TransKun v2](https://github.com/yujia-yan/transkun) (`transkun>=2.0.0`, bundled weights, zero HuggingFace login) by default, preserving real per-note touch velocities and Bar 0 pickup offsets on the Stage 1 `TempoMap` grid.
   - Automatically detects NVIDIA GPUs (`nvidia-smi`) during installation (`install.ps1` / `install.sh`) to install CUDA PyTorch (`cu124` with `cu121` fallback).
3. **End-to-End Rubato `TempoMap` Alignment (`tempo_map.py`)**
   - Full wall-clock-to-score-beat synchronization across Stages 1–5 and Eval Gate for expressive rubato performances.
4. **Auto Key Detection & 6-Dim Musical DNA Style Profiles (`style_analyzer.py` + `profiles/*.json`)**
   - Duration-weighted **Krumhansl-Schmuckler** major/minor key detection (no more hardcoded key signatures).
   - 6-dimensional Musical DNA (`tempo_cv`, `subdivision_density`, `syncopation_index`, `chromaticism_rate`, `polyphony_independence`, `swing_ratio`) automatically selects among 4 declarative profiles:
     - `pop_ballad`
     - `classical_romantic`
     - `baroque_polyphonic`
     - `jazz_groove`
   - Dynamic Eval Gate Rubric adapts downbeat, 16th-note, rest-density, and LH/RH balance criteria per style.
5. **Chroma-Flux Dynamic Pedaling, Real Velocity Fusion & Local Repair Loop (`pedal_and_velocity.py` + `bar_repair_loop.py`)**
   - Replaces blind per-bar sustain pedaling with style-aware chroma-flux pedaling (dry Baroque articulation, Jazz half-pedal, Pop/Romantic harmonic pedal lifts).
   - Blends real TransKun touch velocities with Stage 4 acoustic-band calibration.
   - `--auto-repair` inspects Eval Gate failures and applies bounded per-bar `bar_overrides` (`drop_pitches`, `force_hand`, `enable_gap_bridging`) without re-running Stage 1.

---

## Quick Start

### 1. Download & Install

Click **Code → Download ZIP** (or `git clone https://github.com/JiananZhao/antigravity-piano-score-studio.git`), then run the platform installer (auto-detects NVIDIA GPU & installs CUDA PyTorch):

- **Windows (PowerShell)**:
  ```powershell
  .\install.ps1
  ```
- **macOS / Linux**:
  ```bash
  bash install.sh
  ```

Optional external tools for full PDF engraving & Salamander rendering:
- **MuseScore 4** (for PDF export from MusicXML)
- **sfizz_render** + **Salamander Grand Piano V3 SFZ** (run `python skills/piano-score-studio/scripts/setup_assets.py --download-salamander`)

### 2. One-Command Usage

```bash
# Full automatic pipeline (TransKun v2 + auto GPU + auto key & style + PDF + calibrated audio)
python piano_studio.py input.wav -o output/my_song

# Enable automatic 1-pass local bar_overrides repair if Eval Gate flags bar-level issues
python piano_studio.py input.wav -o output/my_song --auto-repair

# Sheet music only (skip Salamander audio render & A/B excerpts)
python piano_studio.py input.wav -o output/my_song --skip-render

# Resume an interrupted run from project_manifest.json
python piano_studio.py input.wav -o output/my_song --resume

# Override style profile, key signature, or engine explicitly
python piano_studio.py bach_invention.wav --style baroque_polyphonic --engine transkun
python piano_studio.py jazz_ballad.wav --style jazz_groove --engine muscriptor
```

---

## Pipeline Architecture

| Stage | Script | Responsibility |
| :--- | :--- | :--- |
| **Stage 0** | `stage0_audio_triage.py` | Audio integrity check (clipping, dropouts, mono cancellation) & best-take selection |
| **Stage 1** | `stage1_tempo_and_transcribe.py` + `transcription_engines.py` + `tempo_map.py` | Beat tracking, fixed-tempo vs. rubato `TempoMap` calibration, TransKun v2 / MuScriptor transcription |
| **Stage 1.5** | `style_analyzer.py` + `profiles/*.json` | Krumhansl-Schmuckler key detection, 6-dim Musical DNA extraction, declarative style profile selection |
| **Stage 2** | `stage2_pitch_audit.py` | Harmonic FFT audit for minor-2nd clashes, ghost-note removal (`--clean-midi`), and key-aware out-of-scale checks |
| **Stage 4.1** | `stage4_dynamics_plan.py` | Two-band (LH 45–350 Hz / RH 350–3500 Hz) bar-by-bar dynamics envelope CSV & plan |
| **Stage 3** | `stage3_engrave_study_score.py` + `bar_repair_loop.py` | Pedagogical piano engraving (G2/F4 locked clefs, style-aware hand split & gap bridging, `bar_overrides`, MuseScore 4 PDF) |
| **Stage 4.2–4.4** | `stage4_render_calibrated.py` + `pedal_and_velocity.py` | TransKun velocity fusion, chroma-flux dynamic CC64 pedaling, Salamander Grand Piano render & two-band mastering |
| **Stage 5** | `stage5_ab_excerpts.py` + `eval_gate_runner.py` | A/B comparison excerpts & style-adaptive Eval Gate verification (`eval_report.json`) |

---

## Outputs

Each run writes a clean project folder under `--out-dir` (default `output/<song_name>/`):
- `project_manifest.json` — canonical v3.0 state, detected key, Musical DNA, `bar_overrides`, and artifact paths
- `score/score_learning.musicxml` & `score/score_learning.pdf` — human-readable study edition
- `deliverables/<song>_Full_Calibrated_Salamander.wav` & `.mid` — calibrated Salamander Grand Piano audio & MIDI
- `deliverables/ab_excerpts/` — matched original vs. calibrated A/B listening clips
- `deliverables/eval_report.json` — full style-adaptive quality audit

---

## Development & Testing

```bash
# Run lightweight synthetic unit tests (no GPU / PyTorch / MuseScore required)
pytest -v tests/
```

GitHub Actions CI automatically runs `py_compile`, CLI `--help` smoke checks across all scripts, and `pytest -v tests/` on Python 3.10 and 3.12 on every push and pull request.

---

## License

MIT License. See [LICENSE](LICENSE) and [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
