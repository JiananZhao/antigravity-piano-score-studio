"""Stage 0: Audio triage, multi-candidate ranking, and leading-note check."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_audio(path: Path) -> dict:
    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Unsupported or unreadable audio container '{path.name}' for libsndfile ({exc}). "
            "Convert compressed formats like .m4a/.aac to WAV or FLAC first (e.g., `ffmpeg -i input.m4a output.wav`)."
        ) from exc
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    n_frames, n_ch = data.shape
    dur_s = n_frames / float(sr)

    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    clip_cnt = int(np.sum(np.abs(data) >= 0.999))
    diff = np.abs(np.diff(data, axis=0)) if n_frames > 1 else np.zeros((1, n_ch))
    click_jumps = int(np.sum(diff > 0.1))

    # Dropouts (10ms frames < -70 dBFS inside [1s, dur-2s])
    hop_10ms = max(1, int(sr * 0.01))
    s_i0, s_i1 = min(n_frames, sr), max(0, n_frames - 2 * sr)
    dropouts = 0
    if s_i1 > s_i0 + hop_10ms:
        seg = mono[s_i0:s_i1]
        nf = len(seg) // hop_10ms
        fdb = 10.0 * np.log10(np.mean(seg[: nf * hop_10ms].reshape(nf, hop_10ms) ** 2, axis=1) + 1e-15)
        dropouts = int(np.sum(fdb < -70.0))

    # First active note (> -35 dBFS in mono 10ms frames)
    nf_all = n_frames // hop_10ms
    mdb = 10.0 * np.log10(np.mean(mono[: nf_all * hop_10ms].reshape(nf_all, hop_10ms) ** 2, axis=1) + 1e-15)
    above = np.where(mdb > -35.0)[0]
    first_note_sec = float(above[0] * 0.01) if len(above) else dur_s

    # Stereo coherence & mono-sum cancellation
    lr_corr = 1.0
    neg_phase_pct = 0.0
    side_mid_ratio = 0.0
    mono_cancel_gt6_pct = 0.0
    quiet_side_p01_db = -90.0

    n_fft = 4096
    hop = 1024
    win = np.hanning(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    def stft_c(x: np.ndarray) -> np.ndarray:
        nf = (len(x) - n_fft) // hop + 1
        if nf <= 0:
            return np.zeros((len(freqs), 1), dtype=np.complex128)
        idx = np.arange(n_fft)[None, :] + np.arange(nf)[:, None] * hop
        frames = x[idx] * win[None, :]
        return np.fft.rfft(frames, axis=1).T

    CM = stft_c(mono)
    AM = np.abs(CM)

    if n_ch >= 2:
        L, R = data[:, 0], data[:, 1]
        side = 0.5 * (L - R)
        lr_corr = float(np.corrcoef(L, R)[0, 1]) if np.std(L) > 0 and np.std(R) > 0 else 1.0
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        mid_rms = float(np.sqrt(np.mean(mono ** 2)))
        side_mid_ratio = side_rms / (mid_rms + 1e-12)

        hop_100 = max(1, int(sr * 0.1))
        nf100 = n_frames // hop_100
        Lf = L[: nf100 * hop_100].reshape(nf100, hop_100)
        Rf = R[: nf100 * hop_100].reshape(nf100, hop_100)
        num = np.sum(Lf * Rf, axis=1)
        den = np.sqrt(np.sum(Lf ** 2, axis=1) * np.sum(Rf ** 2, axis=1)) + 1e-12
        neg_phase_pct = float(np.mean((num / den) < 0.0) * 100.0)

        nf_s = len(side) // hop_10ms
        sdb = 10.0 * np.log10(np.mean(side[: nf_s * hop_10ms].reshape(nf_s, hop_10ms) ** 2, axis=1) + 1e-15)
        quiet_side_p01_db = float(np.percentile(sdb, 1))

        CL, CR = stft_c(L), stft_c(R)
        pLR = 0.5 * (np.abs(CL) ** 2 + np.abs(CR) ** 2)
        pM = AM ** 2
        idx_h = (freqs >= 60) & (freqs <= 5000)
        pLR_s, pM_s = pLR[idx_h], pM[idx_h]
        thr = np.percentile(pLR_s, 95)
        mask = pLR_s > thr
        if np.any(mask):
            ratio = pM_s[mask] / (pLR_s[mask] + 1e-12)
            mono_cancel_gt6_pct = float(np.mean(ratio < 0.25) * 100.0)

    # Harmonic sharpness (peak/median dB in 320-5120 Hz) & onset contrast
    f_rms = np.sqrt(np.mean(AM ** 2, axis=0))
    active = f_rms > np.percentile(f_rms, 20)
    idx_harm = (freqs >= 320) & (freqs <= 5120)
    p_act = AM[idx_harm][:, active] ** 2 + 1e-18
    peak_med_db = float(np.mean(10.0 * np.log10(np.max(p_act, axis=0) / (np.median(p_act, axis=0) + 1e-18))))

    idx_p = (freqs >= 80) & (freqs <= 5120)
    lspec = np.log1p(1000.0 * AM[idx_p])
    flux = np.mean(np.maximum(0.0, np.diff(lspec, axis=1)), axis=0)
    flux_contrast = float(np.percentile(flux, 95) / (np.median(flux) + 1e-12))

    # Composite transcription friendliness score (higher = cleaner, drier, better phase)
    score = (
        (100.0 - mono_cancel_gt6_pct * 2.0)
        + peak_med_db * 1.5
        + flux_contrast * 5.0
        + lr_corr * 15.0
        - neg_phase_pct * 0.8
        - (clip_cnt * 10.0 + click_jumps * 10.0 + dropouts * 20.0)
    )

    return {
        "file": str(path),
        "filename": path.name,
        "sha256": sha256_file(path),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate": int(sr),
        "channels": int(n_ch),
        "duration_sec": round(dur_s, 3),
        "peak_dbfs": round(20.0 * np.log10(peak + 1e-12), 2),
        "rms_dbfs": round(20.0 * np.log10(rms + 1e-12), 2),
        "clipping_count": clip_cnt,
        "click_jumps_gt_0.1": click_jumps,
        "dropouts_lt_-70dB": dropouts,
        "first_note_sec": round(first_note_sec, 3),
        "needs_2s_pad_regression": first_note_sec < 0.15,
        "lr_corr": round(lr_corr, 4),
        "neg_phase_pct": round(neg_phase_pct, 2),
        "side_mid_ratio": round(side_mid_ratio, 4),
        "quiet_side_p01_dbfs": round(quiet_side_p01_db, 2),
        "strong_harmonic_mono_cancel_gt_6dB_pct": round(mono_cancel_gt6_pct, 2),
        "harmonic_peak_to_median_db": round(peak_med_db, 2),
        "onset_flux_p95_med": round(flux_contrast, 3),
        "transcription_readiness_score": round(score, 2),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 0 audio triage & candidate selection")
    ap.add_argument("--inputs", nargs="+", required=True, help="Audio files or directories")
    ap.add_argument("--out", required=True, help="Output JSON report path")
    args = ap.parse_args()

    files: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            for ch in sorted(p.iterdir()):
                if ch.is_file() and ch.suffix.lower() in AUDIO_EXTS:
                    files.append(ch)
        elif p.is_file():
            files.append(p)
    if not files:
        raise SystemExit("No audio files found in --inputs")

    results = [analyze_audio(f) for f in files]
    results.sort(key=lambda r: r["transcription_readiness_score"], reverse=True)
    recommended = results[0]

    report = {
        "stage": "stage_0_audio_triage",
        "candidate_count": len(results),
        "recommended_source": recommended["file"],
        "recommended_filename": recommended["filename"],
        "candidates": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Stage 0 Audio Triage complete ({len(results)} file(s)). Recommended: {recommended['filename']}")
    for r in results:
        print(
            f"  {r['filename']:24s} | score={r['transcription_readiness_score']:6.2f} | "
            f"mono_cancel>6dB={r['strong_harmonic_mono_cancel_gt_6dB_pct']:5.2f}% | "
            f"neg_phase={r['neg_phase_pct']:5.2f}% | first_note={r['first_note_sec']:.3f}s | "
            f"clips={r['clipping_count']}"
        )


if __name__ == "__main__":
    main()