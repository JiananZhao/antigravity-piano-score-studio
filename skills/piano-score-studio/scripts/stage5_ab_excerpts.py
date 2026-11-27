"""Stage 5: Generate 12s A/B comparison WAV excerpts (Opening, Climax, Outro)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(t_out, t_in, x)


def write_ab_pair(
    orig_m: np.ndarray, cal_m: np.ndarray, sr: int, t0: float, dur: float, out_dir: Path, stem: str
) -> None:
    i0 = max(0, int(round(t0 * sr)))
    i1 = min(len(orig_m), len(cal_m), int(round((t0 + dur) * sr)))
    if i1 <= i0 + sr:
        return
    a = orig_m[i0:i1].copy()
    b = cal_m[i0:i1].copy()
    # Match RMS slightly so loudness difference doesn't bias timbre check, while preserving dynamics
    fade = min(len(a) // 10, int(0.05 * sr))
    ramp = np.linspace(0.0, 1.0, fade)
    for arr in (a, b):
        arr[:fade] *= ramp
        arr[-fade:] *= ramp[::-1]

    stereo_lr = np.column_stack([a, b])
    sf.write(str(out_dir / f"{stem}_LR.wav"), stereo_lr, sr, subtype="PCM_24")

    half = min(len(a), int(6.0 * sr))
    gap = np.zeros(int(0.4 * sr), dtype=np.float64)
    seq = np.concatenate([a[:half], gap, b[:half]])
    sf.write(str(out_dir / f"{stem}_seq.wav"), np.column_stack([seq, seq]), sr, subtype="PCM_24")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 5 A/B listening excerpts generator")
    ap.add_argument("--orig", required=True)
    ap.add_argument("--calibrated", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    orig_d, sr_o = sf.read(args.orig, dtype="float64", always_2d=True)
    cal_d, sr_c = sf.read(args.calibrated, dtype="float64", always_2d=True)
    orig_m = resample_linear(orig_d.mean(axis=1), sr_o, sr_c)
    cal_m = cal_d.mean(axis=1)

    dur_s = min(len(orig_m), len(cal_m)) / float(sr_c)
    # Find loudest 12s window (climax) in [20s, dur - 20s]
    win = int(12.0 * sr_c)
    step = int(2.0 * sr_c)
    best_t, best_e = max(20.0, dur_s * 0.4), -1.0
    for i0 in range(int(20.0 * sr_c), max(int(21.0 * sr_c), len(orig_m) - win - int(20.0 * sr_c)), step):
        e = float(np.mean(orig_m[i0 : i0 + win] ** 2))
        if e > best_e:
            best_e = e
            best_t = i0 / float(sr_c)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ab_pair(orig_m, cal_m, sr_c, 0.0, 12.0, out_dir, "AB_01_opening")
    write_ab_pair(orig_m, cal_m, sr_c, best_t, 12.0, out_dir, "AB_02_climax")
    write_ab_pair(orig_m, cal_m, sr_c, max(0.0, dur_s - 14.0), 12.0, out_dir, "AB_03_outro")
    print(f"Stage 5 A/B excerpts written to {out_dir} (opening=0s, climax={best_t:.1f}s, outro={max(0.0, dur_s-14.0):.1f}s)")


if __name__ == "__main__":
    main()
