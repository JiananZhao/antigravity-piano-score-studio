"""Stage 4.1: Analyze original audio in LH (45-350Hz) and RH (350-3500Hz) bands
and generate dynamics_plan.md + bar_lh_rh_envelope.csv BEFORE any MIDI/VST rendering.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from tempo_map import load_tempo_map


def suggest_dynamic(db: float, p15: float, p40: float, p70: float, p90: float) -> str:
    if db <= p15:
        return "pp"
    if db <= p40:
        return "p"
    if db <= p70:
        return "mp"
    if db <= p90:
        return "mf"
    return "f"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 4.1 two-band dynamics plan generator")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--beat-type", type=int, default=4, choices=(4, 8), help="Time signature denominator (4 or 8)")
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--stage1-cal", default=None, help="Optional Stage 1 stage1_calibration.json for rubato tempo-map alignment")
    ap.add_argument("--num-bars", type=int, required=True, help="Count of full bars 1..N")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data, sr = sf.read(args.audio, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)
    side = 0.5 * (data[:, 0] - data[:, 1]) if data.shape[1] >= 2 else np.zeros_like(mono)
    sm_ratio = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mono ** 2)) + 1e-12))

    fft_full = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    lh_sig = np.fft.irfft(np.where((freqs >= 45.0) & (freqs <= 350.0), fft_full, 0.0), n=len(mono))
    rh_sig = np.fft.irfft(np.where((freqs >= 350.0) & (freqs <= 3500.0), fft_full, 0.0), n=len(mono))

    tot_e = float(np.sum(np.abs(fft_full) ** 2)) + 1e-12
    bands = [(20, 150), (150, 500), (500, 2000), (2000, 8000), (8000, 20000)]
    band_pcts = [round(float(np.sum(np.abs(fft_full[(freqs >= lo) & (freqs < hi)]) ** 2) / tot_e * 100.0), 2) for lo, hi in bands]

    tmap = load_tempo_map(args.stage1_cal, args.bpm, args.beats_per_bar, args.beat_type, args.first_downbeat)
    rows = []
    for b in range(1, args.num_bars + 1):
        t0, t1 = tmap.bar_bounds_sec(b)
        i0, i1 = max(0, int(t0 * sr)), min(len(mono), int(t1 * sr))
        if i1 <= i0:
            full_db, lh_db, rh_db = -60.0, -60.0, -60.0
        else:
            full_db = 20.0 * np.log10(float(np.sqrt(np.mean(mono[i0:i1] ** 2))) + 1e-12)
            lh_db = 20.0 * np.log10(float(np.sqrt(np.mean(lh_sig[i0:i1] ** 2))) + 1e-12)
            rh_db = 20.0 * np.log10(float(np.sqrt(np.mean(rh_sig[i0:i1] ** 2))) + 1e-12)
        rows.append(
            {
                "bar": b,
                "t_start": round(t0, 3),
                "t_end": round(t1, 3),
                "full_rms_db": round(full_db, 2),
                "lh_45_350_db": round(lh_db, 2),
                "rh_350_3500_db": round(rh_db, 2),
                "lh_minus_rh_db": round(lh_db - rh_db, 2),
            }
        )

    full_dbs = np.array([r["full_rms_db"] for r in rows])
    lh_rh_diffs = np.array([r["lh_minus_rh_db"] for r in rows])
    p15, p40, p70, p90 = [float(np.percentile(full_dbs, q)) for q in (15, 40, 70, 90)]

    for r in rows:
        r["suggested_dynamic"] = suggest_dynamic(r["full_rms_db"], p15, p40, p70, p90)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bar_lh_rh_envelope.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Phrase summary (4-bar chunks)
    phrase_lines = []
    for i in range(0, len(rows), 4):
        ch = rows[i : i + 4]
        avg_db = float(np.mean([x["full_rms_db"] for x in ch]))
        avg_diff = float(np.mean([x["lh_minus_rh_db"] for x in ch]))
        dyn = suggest_dynamic(avg_db, p15, p40, p70, p90)
        phrase_lines.append(
            f"| Bars {ch[0]['bar']:>3d}-{ch[-1]['bar']:<3d} | {avg_db:6.2f} dB | {avg_diff:+6.2f} dB | `{dyn}` |"
        )

    med_diff = float(np.median(lh_rh_diffs))
    p10_diff = float(np.percentile(lh_rh_diffs, 10))
    plan_md = f"""# 力度分析与标记工作计划（Dynamics Plan）

- **参考原音**：`{args.audio}`
- **速度与拍号**：`{args.bpm} BPM`（`{tmap.mode}`），`{args.beats_per_bar}/{args.beat_type}`，首强拍 `{args.first_downbeat}s`，共 `{args.num_bars}` 小节
- **立体声 Side/Mid 比**：`{sm_ratio:.3f}`
- **5 频段能量分布（20-150 / 150-500 / 500-2k / 2k-8k / 8k-20k Hz）**：`{band_pcts}`

## 1. 双手能量平衡实测结论
- **LH (45–350 Hz) 相对 RH (350–3500 Hz) 差值**：中位数 **`{med_diff:+.2f} dB`**，P10 **`{p10_diff:+.2f} dB`**，最大值 **`{lh_rh_diffs.max():+.2f} dB`**。
- **执行规则**：
  1. 右手主旋律/和声目标 MIDI 力度区间：`[42, 94]`，多音和弦最高旋律音 `+4` 突出。
  2. 左手伴奏目标 MIDI 力度区间：`[26, 74]`（严禁触发 Salamander `v12+` 刺耳高力度层）。
  3. 大低音衰减：`<= B♭2 (46)` 减 `5`，`<= E2 (40)` 减 `8`（下限 `26`）。
  4. 当某小节原音 `LH - RH < -4.0 dB` 或处于轻柔段/挂留色彩和弦时，强制 `LH_vel <= RH_bar_mean - 14`。

## 2. 4 小节乐句力度候选总览
| 小节范围 | 平均全频 RMS | LH - RH 差值 | 建议段落记号 |
|---|---:|---:|:---:|
{chr(10).join(phrase_lines)}
"""
    md_path = out_dir / "dynamics_plan.md"
    md_path.write_text(plan_md, encoding="utf-8")
    print(f"Stage 4.1 Dynamics Plan written: {md_path} & {csv_path} (LH-RH median={med_diff:+.2f} dB)")


if __name__ == "__main__":
    main()