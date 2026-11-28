#!/usr/bin/env python3
"""Stage 5.1: Stereo A/B comparison excerpts (opening, climax, outro).

Generates stereo paired excerpts (`{stem}_orig.wav`, `{stem}_render.wav`) and
sequential A/B comparison clips (`{stem}_seq.wav`: 6s orig -> 0.4s silence -> 6s render)
with optional LUFS loudness matching and zero-padded outro alignment to the original's
detected sound end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_match import apply_peak_limiter, detect_sound_end, integrated_lufs


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    """Convert 1D mono or (N, 1) array to float64 (N, 2) stereo; keep first 2 channels if >2."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return np.column_stack([arr, arr])
    if arr.shape[1] == 1:
        return np.column_stack([arr[:, 0], arr[:, 0]])
    return arr[:, :2]


def resample_linear(sig: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linearly resample 1D or 2D audio per channel to `sr_out`."""
    arr = ensure_stereo(sig)
    if sr_in == sr_out or len(arr) == 0:
        return arr
    n_in = arr.shape[0]
    n_out = max(1, int(round(n_in * float(sr_out) / float(sr_in))))
    t_in = np.linspace(0.0, 1.0, n_in, endpoint=False)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    out = np.empty((n_out, arr.shape[1]), dtype=np.float64)
    for ch in range(arr.shape[1]):
        out[:, ch] = np.interp(t_out, t_in, arr[:, ch])
    return out


def extract_padded_slice(sig: np.ndarray, sr: int, t0: float, dur_sec: float) -> np.ndarray:
    """Extract `[t0, t0 + dur_sec]` stereo slice, zero-padding cleanly if signal is shorter."""
    arr = ensure_stereo(sig)
    n_target = max(1, int(round(dur_sec * sr)))
    i0 = max(0, int(round(t0 * sr)))
    out = np.zeros((n_target, 2), dtype=np.float64)
    if i0 < arr.shape[0]:
        avail = min(n_target, arr.shape[0] - i0)
        out[:avail] = arr[i0 : i0 + avail]
    return out


def find_climax_start(orig_stereo: np.ndarray, sr: int, win_sec: float = 12.0) -> float:
    """Locate highest-energy window using stereo power sum across channels."""
    dur_s = orig_stereo.shape[0] / float(sr)
    if dur_s <= win_sec + 1.0:
        return 0.0
    hop = max(1, int(round(0.25 * sr)))
    win_frames = max(1, int(round(win_sec / 0.25)))
    n_frames = orig_stereo.shape[0] // hop
    if n_frames <= win_frames:
        return 0.0
    framed = orig_stereo[: n_frames * hop].reshape(n_frames, hop, 2)
    # Stereo power sum across L and R
    frame_pow = np.sum(framed ** 2, axis=(1, 2))
    csum = np.concatenate(([0.0], np.cumsum(frame_pow)))
    win_pow = csum[win_frames:] - csum[:-win_frames]
    best_idx = int(np.argmax(win_pow))
    return best_idx * (hop / float(sr))


def write_ab_pair(
    orig_slice: np.ndarray,
    rend_slice: np.ndarray,
    sr: int,
    out_dir: Path,
    stem: str,
    loudness_match: bool = True,
) -> dict:
    """Write stereo `{stem}_orig.wav`, `{stem}_render.wav`, and `{stem}_seq.wav`."""
    o_seg = ensure_stereo(orig_slice).copy()
    r_seg = ensure_stereo(rend_slice).copy()
    gain_db = 0.0

    if loudness_match:
        lufs_o = integrated_lufs(o_seg, sr)
        lufs_r = integrated_lufs(r_seg, sr)
        if lufs_o > -65.0 and lufs_r > -65.0:
            gain_db = float(np.clip(lufs_o - lufs_r, -12.0, 12.0))
            r_seg *= 10.0 ** (gain_db / 20.0)
            r_seg = apply_peak_limiter(r_seg, sr, ceiling_db=-1.0)
        print(f"  [{stem}] LUFS match: orig={lufs_o:.2f} LUFS, render={lufs_r:.2f} LUFS -> gain={gain_db:+.2f} dB")

    orig_path = out_dir / f"{stem}_orig.wav"
    rend_path = out_dir / f"{stem}_render.wav"
    seq_path = out_dir / f"{stem}_seq.wav"

    sf.write(str(orig_path), o_seg, sr, subtype="PCM_24")
    sf.write(str(rend_path), r_seg, sr, subtype="PCM_24")

    # Sequential A/B clip: 6s orig -> 0.4s silence -> 6s render (stereo)
    n6 = min(int(round(6.0 * sr)), o_seg.shape[0], r_seg.shape[0])
    gap = np.zeros((int(round(0.4 * sr)), 2), dtype=np.float64)
    fade = min(int(round(0.02 * sr)), n6 // 4)
    o6 = o_seg[:n6].copy()
    r6 = r_seg[:n6].copy()
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)[:, None]
        o6[:fade] *= ramp
        o6[-fade:] *= ramp[::-1]
        r6[:fade] *= ramp
        r6[-fade:] *= ramp[::-1]
    seq = np.vstack([o6, gap, r6])
    sf.write(str(seq_path), seq, sr, subtype="PCM_24")

    return {
        "stem": stem,
        "orig_wav": str(orig_path),
        "render_wav": str(rend_path),
        "seq_wav": str(seq_path),
        "gain_db": round(gain_db, 3),
    }


def generate_ab_excerpts(
    orig_path: Path,
    cal_path: Path,
    out_dir: Path,
    first_downbeat: float = 0.0,
    loudness_match: bool = True,
) -> list[dict]:
    """Generate opening, climax, and outro stereo A/B excerpts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    o_raw, sr_o = sf.read(str(orig_path), dtype="float64", always_2d=True)
    r_raw, sr_r = sf.read(str(cal_path), dtype="float64", always_2d=True)

    sr = sr_r
    orig = resample_linear(o_raw, sr_o, sr)
    rend = ensure_stereo(r_raw)

    sound_end_orig = detect_sound_end(orig, sr, floor_db=-60.0)
    opening_t0 = max(0.0, float(first_downbeat) - 1.0)
    climax_t0 = find_climax_start(orig, sr, win_sec=12.0)
    outro_t0 = max(0.0, sound_end_orig - 10.0)

    specs = [
        ("AB_01_opening", opening_t0, 12.0),
        ("AB_02_climax", climax_t0, 12.0),
        ("AB_03_outro", outro_t0, 12.0),
    ]
    results = []
    for stem, t0, dur in specs:
        o_sl = extract_padded_slice(orig, sr, t0, dur)
        r_sl = extract_padded_slice(rend, sr, t0, dur)
        results.append(write_ab_pair(o_sl, r_sl, sr, out_dir, stem, loudness_match=loudness_match))
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 5.1 stereo A/B comparison excerpt generator")
    ap.add_argument("--orig", required=True, help="Original reference audio file")
    ap.add_argument("--calibrated", required=True, help="Calibrated Salamander rendered WAV")
    ap.add_argument("--bpm", type=float, default=90.0)
    ap.add_argument("--beats-per-bar", type=int, default=4)
    ap.add_argument("--first-downbeat", type=float, default=0.0)
    ap.add_argument("--out-dir", required=True, help="Output directory for A/B excerpts")
    ap.add_argument("--no-loudness-match", action="store_true", help="Disable per-excerpt LUFS matching")
    args = ap.parse_args()

    res = generate_ab_excerpts(
        orig_path=Path(args.orig),
        cal_path=Path(args.calibrated),
        out_dir=Path(args.out_dir),
        first_downbeat=args.first_downbeat,
        loudness_match=not args.no_loudness_match,
    )
    print(f"Stage 5.1 Stereo A/B Excerpts Complete: {len(res)} pairs in {args.out_dir}")


if __name__ == "__main__":
    main()
