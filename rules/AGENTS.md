# Piano Score Studio — Always-On Guardrails

1. **No flat-velocity audio delivery**: Raw transcription MIDI (`velocity = 100`, no `CC64` pedal) is a Stage 2 placeholder only. Never render final verification or deliverable audio without first generating a two-band `dynamics_plan.md` (`LH 45-350 Hz` vs `RH 350-3500 Hz`) and completing 2 rounds of dynamics verification.
2. **Human-readable score gate**: A piano score is NOT passing just because MuseScore exited 0. It must pass both Round 1 (automated XML checks: 0 intra-bar clef flips, `<= 2` voices per hand, 0 duration errors, beat-grouped beaming, key-consistent spelling) and Round 2 (page-by-page PDF visual inspection).
3. **Evidence honesty & non-destructive hygiene**: Never claim audio/score accuracy without verified measurements; never overwrite original input audio or commit tokens/secrets (`hf_token`, `config.json`) into git repositories.
