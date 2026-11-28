#!/usr/bin/env python3
"""Pure-NumPy audio matching & mastering utilities for Stage 4 & Stage 5.

Provides ITU-R BS.1770 integrated LUFS metering, lookahead peak limiting,
sound-end detection, 1/3-octave long-term spectrum & match EQ, and
Schroeder backward-integration RT60 estimation — with zero dependencies
beyond NumPy (optional `pyloudnorm` used if installed).
"""

from __future__ import annotations

import numpy as np

# Standard IEC 1/3-octave center frequencies from 31.5 Hz to 16 kHz
ONE_THIRD_OCT_BANDS_HZ = np.array(
    [
        31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
        200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0,
        1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0,
        8000.0, 10000.0, 12500.0, 16000.0,
    ],
    dtype=np.float64,
)


def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure audio array is float64 with shape (N, C)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _bs1770_k_weight_mag(freqs: np.ndarray) -> np.ndarray:
    """ITU-R BS.1770-4 K-weighting magnitude response (Stage 1 pre-filter shelf * Stage 2 RLB HPF).

    Uses exact analog prototype poles/zeros evaluated at s = j*2*pi*f so it is
    sample-rate independent and matches BS.1770 within <0.05 dB at 997 Hz.
    """
    w = 2.0 * np.pi * freqs
    s = 1j * w

    # Stage 1: high-frequency shelf (+3.99984385397 dB @ 1681.974450955533 Hz, Q=0.7071752369554196)
    vh = 10.0 ** (3.999843853973347 / 20.0)
    vb = vh ** 0.4996667741545416
    w0_1 = 2.0 * np.pi * 1681.974450955533
    q1 = 0.7071752369554196
    h1 = (vh * (s ** 2) + (vb * w0_1 / q1) * s + (w0_1 ** 2)) / (
        (s ** 2) + (w0_1 / q1) * s + (w0_1 ** 2)
    )

    # Stage 2: RLB 2nd-order high-pass (f0 = 38.13547087602444 Hz, Q = 0.5003270373238773)
    w0_2 = 2.0 * np.pi * 38.13547087602444
    q2 = 0.5003270373238773
    h2 = (s ** 2) / ((s ** 2) + (w0_2 / q2) * s + (w0_2 ** 2))

    return np.abs(h1 * h2)


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    """Compute ITU-R BS.1770 integrated loudness (LUFS) for mono or multi-channel audio."""
    arr = ensure_2d(x)
    n_samples, n_ch = arr.shape
    if n_samples == 0 or float(np.max(np.abs(arr))) < 1e-12:
        return -70.0

    # Optional fast-path via pyloudnorm when installed and signal >= 0.4s
    if n_samples >= int(round(0.45 * sr)):
        try:
            import pyloudnorm as pyln  # type: ignore

            meter = pyln.Meter(sr)
            val = float(meter.integrated_loudness(arr if n_ch > 1 else arr[:, 0]))
            if np.isfinite(val):
                return val
        except Exception:
            pass

    # Pure-NumPy zero-phase frequency-domain BS.1770 K-weighting
    n_fft = 1 << (n_samples - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    k_mag = _bs1770_k_weight_mag(freqs)

    y = np.empty_like(arr)
    for ch in range(n_ch):
        sp = np.fft.rfft(arr[:, ch], n=n_fft)
        y[:, ch] = np.fft.irfft(sp * k_mag, n=n_fft)[:n_samples]

    block_len = max(1, int(round(0.400 * sr)))
    hop_len = max(1, int(round(0.100 * sr)))

    if n_samples < block_len:
        ch_pow = np.sum(np.mean(y ** 2, axis=0))
        return float(-0.691 + 10.0 * np.log10(max(ch_pow, 1e-12)))

    n_blocks = 1 + (n_samples - block_len) // hop_len
    # Compute cumulative channel power for O(N) block mean squares
    sq_sum = np.sum(y ** 2, axis=1)
    csum = np.concatenate(([0.0], np.cumsum(sq_sum)))
    starts = np.arange(n_blocks) * hop_len
    block_z = (csum[starts + block_len] - csum[starts]) / float(block_len)
    block_lufs = -0.691 + 10.0 * np.log10(np.maximum(block_z, 1e-12))

    # Stage 1 absolute gate (-70 LUFS)
    abs_mask = block_lufs > -70.0
    if not np.any(abs_mask):
        return -70.0
    z_abs_mean = float(np.mean(block_z[abs_mask]))
    rel_thresh = -0.691 + 10.0 * np.log10(max(z_abs_mean, 1e-12)) - 10.0

    # Stage 2 relative gate (-10 LU)
    rel_mask = abs_mask & (block_lufs > rel_thresh)
    if not np.any(rel_mask):
        return float(-0.691 + 10.0 * np.log10(max(z_abs_mean, 1e-12)))
    z_final = float(np.mean(block_z[rel_mask]))
    return float(-0.691 + 10.0 * np.log10(max(z_final, 1e-12)))


def _sliding_max_1d(arr: np.ndarray, win: int) -> np.ndarray:
    """O(N) vectorized sliding-window maximum (forward lookahead of length `win`)."""
    n = len(arr)
    if win <= 1 or n == 0:
        return arr.copy()
    win = min(win, n)
    pad = (win - (n % win)) % win
    padded = np.pad(arr, (0, pad), mode="constant", constant_values=-np.inf)
    blocks = padded.reshape(-1, win)
    prefix = np.maximum.accumulate(blocks, axis=1)
    suffix = np.maximum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1]
    out = np.empty(n, dtype=np.float64)
    # For index i, window [i, i + win - 1]: max(suffix[i], prefix[i + win - 1])
    suf_flat = suffix.ravel()
    pre_flat = prefix.ravel()
    # First n - win + 1 positions have full lookahead
    valid = n - win + 1
    out[:valid] = np.maximum(suf_flat[:valid], pre_flat[win - 1 : win - 1 + valid])
    if valid < n:
        # Tail positions: suffix up to end of array
        tail = np.maximum.accumulate(arr[valid:][::-1])[::-1]
        out[valid:] = tail
    return out


def apply_peak_limiter(
    x: np.ndarray,
    sr: int,
    ceiling_db: float = -1.0,
    attack_ms: float = 5.0,
    release_ms: float = 80.0,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Transparent lookahead peak limiter that only reduces gain on transient overs.

    Preserves whole-track loudness instead of scaling the entire file down to a single spike.
    """
    was_1d = np.asarray(x).ndim == 1
    arr = ensure_2d(x)
    n_samples = arr.shape[0]
    if n_samples == 0:
        return (arr[:, 0] if was_1d else arr, 0.0) if return_stats else (arr[:, 0] if was_1d else arr)

    ceiling = 10.0 ** (ceiling_db / 20.0)
    inst_peak = np.max(np.abs(arr), axis=1)
    if float(np.max(inst_peak)) <= ceiling:
        out = arr[:, 0].copy() if was_1d else arr.copy()
        return (out, 0.0) if return_stats else out

    att_samp = max(1, int(round(attack_ms * 1e-3 * sr)))
    # Forward lookahead peak envelope
    la_peak = _sliding_max_1d(inst_peak, att_samp)
    g_target = np.minimum(1.0, ceiling / np.maximum(la_peak, 1e-12))

    # Downsampled release smoothing (2 ms hop) to avoid per-sample Python loops while staying smooth
    hop = max(1, int(round(0.002 * sr)))
    n_frames = (n_samples + hop - 1) // hop
    pad_len = n_frames * hop - n_samples
    g_frames = np.min(np.pad(g_target, (0, pad_len), mode="edge").reshape(n_frames, hop), axis=1)

    alpha_rel = float(np.exp(-hop / max(1.0, release_ms * 1e-3 * sr)))
    g_env_f = np.empty(n_frames, dtype=np.float64)
    cur = 1.0
    for i in range(n_frames):
        gt = float(g_frames[i])
        if gt < cur:
            cur = gt
        else:
            cur = alpha_rel * cur + (1.0 - alpha_rel) * gt
        g_env_f[i] = cur

    # Linearly upsample frame gain envelope & enforce strict hard ceiling on exact samples
    frame_centers = np.arange(n_frames, dtype=np.float64) * hop
    sample_idx = np.arange(n_samples, dtype=np.float64)
    g_smooth = np.interp(sample_idx, frame_centers, g_env_f)
    g_final = np.minimum(g_smooth, g_target)

    # Smooth attack ramp on g_final with short moving average, then clamp exact overs
    if att_samp > 1:
        k = np.ones(att_samp, dtype=np.float64) / float(att_samp)
        g_ramp = np.convolve(g_final, k, mode="same")
        g_final = np.minimum(g_ramp, np.minimum(1.0, ceiling / np.maximum(inst_peak, 1e-12)))

    out_2d = arr * g_final[:, None]
    max_gr_db = float(-20.0 * np.log10(max(float(np.min(g_final)), 1e-6)))
    out = out_2d[:, 0] if was_1d else out_2d
    return (out, max_gr_db) if return_stats else out


def detect_sound_end(x: np.ndarray, sr: int, floor_db: float = -60.0) -> float:
    """Detect end of audible sound (seconds) using 50 ms stereo RMS frames."""
    arr = ensure_2d(x)
    n_samples = arr.shape[0]
    dur_s = n_samples / float(sr)
    if n_samples == 0:
        return 0.0

    frame_len = max(1, int(round(0.050 * sr)))
    n_frames = n_samples // frame_len
    if n_frames == 0:
        return dur_s

    trimmed = arr[: n_frames * frame_len].reshape(n_frames, frame_len, arr.shape[1])
    frame_rms = np.sqrt(np.mean(trimmed ** 2, axis=(1, 2)))
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))

    max_db = float(np.max(frame_db))
    if max_db < -95.0:
        return 0.0

    # Estimate noise floor from quietest ~0.5 s of frames (not 10% of song, which may be pp music)
    n_quiet = max(1, min(10, n_frames // 20))
    noise_floor_db = min(float(np.median(np.sort(frame_db)[:n_quiet])), max_db - 18.0)
    thresh_db = max(max_db + floor_db, noise_floor_db + 6.0)

    active = np.where(frame_db >= thresh_db)[0]
    if len(active) == 0:
        return dur_s
    last_idx = int(active[-1])
    return min(dur_s, (last_idx + 1) * (frame_len / float(sr)))


def long_term_spectrum_db(x: np.ndarray, sr: int, n_fft: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Compute channel-power-averaged Welch long-term spectrum aggregated to 1/3-octave bands (dB)."""
    arr = ensure_2d(x)
    n_samples, n_ch = arr.shape
    bands = ONE_THIRD_OCT_BANDS_HZ[ONE_THIRD_OCT_BANDS_HZ < 0.48 * sr]
    if n_samples < 64:
        return bands.copy(), np.full(len(bands), -120.0, dtype=np.float64)

    eff_fft = min(n_fft, n_samples)
    hop = max(1, eff_fft // 2)
    win = np.hanning(eff_fft)
    win_norm = float(np.sum(win ** 2)) + 1e-12

    n_frames = max(1, 1 + (n_samples - eff_fft) // hop)
    psd_acc = np.zeros(eff_fft // 2 + 1, dtype=np.float64)
    for idx in range(n_frames):
        s0 = idx * hop
        seg = arr[s0 : s0 + eff_fft]
        if len(seg) < eff_fft:
            seg = np.pad(seg, ((0, eff_fft - len(seg)), (0, 0)), mode="constant")
        frame_p = np.zeros_like(psd_acc)
        for ch in range(n_ch):
            sp = np.fft.rfft(seg[:, ch] * win, n=eff_fft)
            frame_p += (np.abs(sp) ** 2) / win_norm
        psd_acc += frame_p / float(n_ch)
    psd_mean = psd_acc / float(n_frames)

    freqs = np.fft.rfftfreq(eff_fft, 1.0 / float(sr))
    half_bw_ratio = 2.0 ** (1.0 / 6.0)
    band_db = np.full(len(bands), -120.0, dtype=np.float64)
    for i, fc in enumerate(bands):
        f_lo = fc / half_bw_ratio
        f_hi = fc * half_bw_ratio
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if np.any(mask):
            band_db[i] = 10.0 * float(np.log10(max(float(np.mean(psd_mean[mask])), 1e-18)))
    return bands.copy(), band_db


def design_match_eq(
    orig_db: np.ndarray,
    rend_db: np.ndarray,
    band_freqs: np.ndarray | None = None,
    strength: float = 0.7,
    max_boost_db: float = 3.0,
    max_cut_db: float = 24.0,
) -> np.ndarray:
    """Design smooth 1/3-octave match EQ gains (dB) aligned on 200-1000 Hz midrange."""
    o = np.asarray(orig_db, dtype=np.float64)
    r = np.asarray(rend_db, dtype=np.float64)
    n = len(o)
    if n == 0:
        return np.array([], dtype=np.float64)

    if band_freqs is not None and len(band_freqs) == n:
        bf = np.asarray(band_freqs, dtype=np.float64)
        mid_mask = (bf >= 200.0) & (bf <= 1000.0)
    else:
        bf = ONE_THIRD_OCT_BANDS_HZ[:n]
        mid_mask = (bf >= 200.0) & (bf <= 1000.0)

    diff = o - r
    if np.any(mid_mask):
        diff = diff - float(np.mean(diff[mid_mask]))

    raw_gains = np.clip(diff * float(strength), -float(max_cut_db), float(max_boost_db))
    if n >= 3:
        padded = np.pad(raw_gains, (1, 1), mode="edge")
        smoothed = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
    else:
        smoothed = raw_gains
    return np.clip(smoothed, -float(max_cut_db), float(max_boost_db))


def apply_eq_fft(
    x: np.ndarray,
    sr: int,
    band_freqs: np.ndarray,
    gains_db: np.ndarray,
) -> np.ndarray:
    """Apply zero-phase match EQ per channel by log-frequency interpolation in FFT domain."""
    was_1d = np.asarray(x).ndim == 1
    arr = ensure_2d(x)
    n_samples, n_ch = arr.shape
    if n_samples == 0 or len(band_freqs) == 0:
        return arr[:, 0].copy() if was_1d else arr.copy()

    n_fft = 1 << (n_samples - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    log_bands = np.log2(np.maximum(np.asarray(band_freqs, dtype=np.float64), 1.0))
    log_freqs = np.log2(np.maximum(freqs, 1.0))
    interp_db = np.interp(log_freqs, log_bands, np.asarray(gains_db, dtype=np.float64))
    interp_db[0] = 0.0  # preserve DC
    gain_lin = 10.0 ** (interp_db / 20.0)

    out = np.empty_like(arr)
    for ch in range(n_ch):
        sp = np.fft.rfft(arr[:, ch], n=n_fft)
        out[:, ch] = np.fft.irfft(sp * gain_lin, n=n_fft)[:n_samples]
    return out[:, 0] if was_1d else out


def estimate_decay_rate(
    x: np.ndarray,
    sr: int,
    t_start: float,
    t_end: float,
    band: tuple[float, float] = (400.0, 1500.0),
) -> tuple[float, float] | None:
    """Estimate reverb tail decay rate (dB/s) and R^2 via Schroeder backward integration.

    Returns `(slope_db_per_sec, r2)` where `slope_db_per_sec < 0`, or `None` if
    the tail is shorter than 1.0 s or linear fit R^2 < 0.8.
    """
    if (t_end - t_start) < 1.0:
        return None
    arr = ensure_2d(x)
    i0 = max(0, int(round(t_start * sr)))
    i1 = min(arr.shape[0], int(round(t_end * sr)))
    seg = arr[i0:i1]
    if seg.shape[0] < int(round(1.0 * sr)):
        return None

    # Bandpass filter in frequency domain per channel
    n_seg, n_ch = seg.shape
    n_fft = 1 << (n_seg - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))
    bp_mask = ((freqs >= band[0]) & (freqs <= band[1])).astype(np.float64)

    bp = np.empty_like(seg)
    for ch in range(n_ch):
        sp = np.fft.rfft(seg[:, ch], n=n_fft)
        bp[:, ch] = np.fft.irfft(sp * bp_mask, n=n_fft)[:n_seg]

    # 20 ms frame energy + Schroeder backward integration
    frame_len = max(1, int(round(0.020 * sr)))
    n_frames = n_seg // frame_len
    if n_frames < 15:
        return None
    frames = bp[: n_frames * frame_len].reshape(n_frames, frame_len, n_ch)
    frame_e = np.sum(frames ** 2, axis=(1, 2))
    edc = np.cumsum(frame_e[::-1])[::-1]
    if edc[0] <= 1e-18:
        return None
    edc_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-12))
    times = np.arange(n_frames, dtype=np.float64) * (frame_len / float(sr))

    # Linear fit on -5 dB to -25 dB decay region
    fit_mask = (edc_db <= -5.0) & (edc_db >= -25.0)
    if int(np.sum(fit_mask)) < 8:
        return None
    t_fit = times[fit_mask]
    y_fit = edc_db[fit_mask]
    t_mean = float(np.mean(t_fit))
    y_mean = float(np.mean(y_fit))
    denom = float(np.sum((t_fit - t_mean) ** 2))
    if denom <= 1e-12:
        return None
    slope = float(np.sum((t_fit - t_mean) * (y_fit - y_mean)) / denom)
    if slope >= -0.1:
        return None
    y_pred = y_mean + slope * (t_fit - t_mean)
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - y_mean) ** 2))
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-12))
    if r2 < 0.8:
        return None
    return float(slope), float(r2)
