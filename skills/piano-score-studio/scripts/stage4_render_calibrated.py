"""Stage 4.2-4.4: Generalized two-band LH/RH velocity calibration, CC64 pedaling,
Salamander Grand Piano rendering (sfizz_render CLI or music-learning NativeHost),
and closed-loop two-band acoustic mastering.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import soundfile as sf

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from tempo_map import TempoMap, load_tempo_map
from style_analyzer import load_style_profile
from pedal_and_velocity import (
    align_microtiming_to_transkun,
    compute_dynamic_pedal_events,
    fuse_transkun_velocities,
    read_midi_pedal_events,
    read_midi_velocity_events,
)

STEP_MAP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FIFTHS_TO_KEY = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}


def find_plugin_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(plugin_root: Path, cfg_arg: str | None) -> dict:
    p = Path(cfg_arg) if cfg_arg else (plugin_root / "config.json")
    if not p.is_file():
        p = plugin_root / "config.example.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def parse_notes_from_xml(
    xml_path: Path,
    bpm: float,
    beats_per_bar: int,
    first_downbeat: float,
    tempo_map: TempoMap | None = None,
) -> tuple[list[dict], int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    divisions = int(root.findtext(".//attributes/divisions", "2"))
    beat_type = int(root.findtext(".//attributes/time/beat-type", "4"))
    quarter_s = (60.0 / bpm) * (beat_type / 4.0)
    unit_s = quarter_s / float(divisions)
    bar_units = (beats_per_bar * divisions * 4) // beat_type

    open_ties: dict[tuple[int, int], dict] = {}
    finished: list[dict] = []
    max_bar = 1

    for m in root.findall(".//part/measure"):
        b = int(m.attrib["number"])
        max_bar = max(max_bar, b)
        implicit = m.attrib.get("implicit") == "yes"
        cur, last_on, max_end = 0, 0, 0
        raw_items = []
        for el in m:
            if el.tag == "backup":
                cur -= int(el.findtext("duration"))
            elif el.tag == "forward":
                cur += int(el.findtext("duration"))
            elif el.tag == "note":
                is_ch = el.find("chord") is not None
                dur = int(el.findtext("duration"))
                on = last_on if is_ch else cur
                if not is_ch:
                    last_on = cur
                    cur += dur
                    max_end = max(max_end, cur)
                p = el.find("pitch")
                if p is None:
                    continue
                midi = (int(p.findtext("octave")) + 1) * 12 + STEP_MAP[p.findtext("step")] + int(p.findtext("alter", "0"))
                staff = int(el.findtext("staff", "1"))
                ts = any(t.attrib.get("type") == "stop" for t in el.findall("tie"))
                te = any(t.attrib.get("type") == "start" for t in el.findall("tie"))
                arp = el.find(".//notations/arpeggiate") is not None
                raw_items.append((on, dur, midi, staff, ts, te, arp))

        bar_slot_0 = (bar_units - max_end) if (b == 0 and implicit) else (b * bar_units)
        for on, dur, midi, staff, ts, te, arp in raw_items:
            gs, ge = bar_slot_0 + on, bar_slot_0 + on + dur
            key = (staff, midi)
            if ts and key in open_ties:
                obj = open_ties[key]
                obj["end_slot"] = max(obj["end_slot"], ge)
                if not te:
                    finished.append(open_ties.pop(key))
            else:
                obj = {"measure": b, "start_slot": gs, "end_slot": ge, "pitch": midi, "staff": staff, "arpeggiate": arp}
                if te:
                    if key in open_ties:
                        finished.append(open_ties.pop(key))
                    open_ties[key] = obj
                else:
                    finished.append(obj)

    for rem in open_ties.values():
        finished.append(rem)
    for n in finished:
        if tempo_map is not None and tempo_map.is_rubato:
            n["st"] = tempo_map.slot_to_sec(n["start_slot"] - bar_units, divisions)
            n["et"] = tempo_map.slot_to_sec(n["end_slot"] - bar_units, divisions)
        else:
            n["st"] = first_downbeat + (n["start_slot"] - bar_units) * unit_s
            n["et"] = first_downbeat + (n["end_slot"] - bar_units) * unit_s
    finished.sort(key=lambda x: (x["st"], x["staff"], x["pitch"]))
    return finished, max_bar, beat_type


def rh_harm_db(mono: np.ndarray, sr: int, st: float, pitch: int) -> float:
    f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    ch = mono[max(0, int((st - 0.005) * sr)) : min(len(mono), int((st + 0.095) * sr))]
    if len(ch) < 64:
        return -55.0
    sp = np.abs(np.fft.rfft(ch * np.hanning(len(ch)), n=8192)) / (len(ch) + 1e-12)
    fs = np.fft.rfftfreq(8192, 1.0 / sr)
    e = 0.0
    for h, w in ((1, 1.0), (2, 0.7), (3, 0.45), (4, 0.25)):
        ft = f0 * h
        if ft > sr / 2 - 100:
            break
        bw = max(ft * 0.05, 10.0)
        m = (fs >= ft - bw) & (fs <= ft + bw)
        if np.any(m):
            e += w * float(np.max(sp[m] ** 2))
    return 10.0 * np.log10(max(1e-12, e))


def lh_expr_db(mono: np.ndarray, lh_band: np.ndarray, sr: int, st: float, pitch: int) -> float:
    f0 = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    i0, i1 = max(0, int((st - 0.005) * sr)), min(len(mono), int((st + 0.150) * sr))
    lhc = lh_band[i0:i1]
    lh_rms = 20.0 * np.log10(float(np.sqrt(np.mean(lhc ** 2))) + 1e-12) if len(lhc) else -55.0
    ch = mono[i0:i1]
    if len(ch) < 64:
        return lh_rms
    sp = np.abs(np.fft.rfft(ch * np.hanning(len(ch)), n=16384)) / (len(ch) + 1e-12)
    fs = np.fft.rfftfreq(16384, 1.0 / sr)
    he = 0.0
    for h, w in ((1, 1.0), (2, 0.85), (3, 0.55), (4, 0.35)):
        ft = f0 * h
        if ft > sr / 2 - 100:
            break
        bw = max(ft * 0.06, 8.0)
        m = (fs >= ft - bw) & (fs <= ft + bw)
        if np.any(m):
            he += w * float(np.mean(sp[m] ** 2))
    return 0.65 * lh_rms + 0.35 * (10.0 * np.log10(max(1e-12, he)) + 26.0)


def export_midi(
    render_notes: list[dict],
    controllers: list[dict],
    rate: int,
    bpm: float,
    beats_per_bar: int,
    beat_type: int,
    key_fifths: int,
    out_mid: Path,
    tempo_map: TempoMap | None = None,
) -> None:
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.name = "Calibrated Salamander Edition"
    meta.append(mido.MetaMessage("time_signature", numerator=beats_per_bar, denominator=beat_type, time=0))
    meta.append(mido.MetaMessage("key_signature", key=FIFTHS_TO_KEY.get(key_fifths, "C"), time=0))
    quarter_bpm = bpm * (4.0 / beat_type)
    use_rubato = tempo_map is not None and tempo_map.is_rubato
    tempo_segs = tempo_map.midi_tempo_segments(tpb) if use_rubato and tempo_map is not None else None
    if use_rubato and tempo_segs:
        prev_tk = 0
        for tk, _, us in tempo_segs:
            meta.append(mido.MetaMessage("set_tempo", tempo=us, time=max(0, tk - prev_tk)))
            prev_tk = tk
    else:
        meta.append(mido.MetaMessage("set_tempo", tempo=int(round(60_000_000 / quarter_bpm)), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))

    ticks_per_sec = float(tpb) * (quarter_bpm / 60.0)
    for name, staff_num, inc_cc in (("Piano Right Hand", 1, False), ("Piano Left Hand & Pedal", 2, True)):
        tr = mido.MidiTrack()
        tr.name = name
        tr.append(mido.Message("program_change", channel=0, program=0, time=0))
        evs = []
        for n in render_notes:
            if n["staff"] != staff_num:
                continue
            v = max(1, min(127, int(round(n["velocity"] * 127))))
            evs.append((n["sample"], "on", n["note"], v))
            evs.append((n["sample"] + n["duration_samples"], "off", n["note"], 0))
        if inc_cc:
            for c in controllers:
                evs.append((c["sample"], "cc", c["controller"], c["value"]))
        evs.sort(key=lambda x: (x[0], 0 if x[1] == "cc" else (1 if x[1] == "off" else 2), x[2]))
        cur_tk = 0
        for smp, kind, p1, p2 in evs:
            t_sec = smp / float(rate)
            abs_tk = (
                tempo_map.sec_to_tick(t_sec, tpb, segments=tempo_segs)
                if (use_rubato and tempo_map is not None)
                else max(0, int(round(t_sec * ticks_per_sec)))
            )
            dt = max(0, abs_tk - cur_tk)
            cur_tk = abs_tk
            if kind == "on":
                tr.append(mido.Message("note_on", channel=0, note=p1, velocity=p2, time=dt))
            elif kind == "off":
                tr.append(mido.Message("note_off", channel=0, note=p1, velocity=0, time=dt))
            else:
                tr.append(mido.Message("control_change", channel=0, control=p1, value=p2, time=dt))
        tr.append(mido.MetaMessage("end_of_track", time=480))
        mid.tracks.append(tr)
    mid.save(str(out_mid))


def render_with_sfizz_or_native(
    plugin_root: Path, cfg: dict, render_notes: list[dict], controllers: list[dict],
    mid_path: Path, raw_wav: Path, rate: int, total_samples: int
) -> None:
    backend = cfg.get("renderer_backend", "auto")
    sfz_rel = cfg.get("salamander_sfz", "assets_local/salamander/Salamander Grand Piano V3.sfz")
    ml_root_str = cfg.get("music_learning_root") or os.environ.get("MUSIC_LEARNING_ROOT", "")
    sfz_candidates = [
        Path(sfz_rel) if Path(sfz_rel).is_absolute() else (plugin_root / sfz_rel),
        plugin_root / "assets_local" / "salamander" / "Salamander Grand Piano V3.sfz",
    ]
    if ml_root_str:
        sfz_candidates.append(Path(ml_root_str) / "data" / "instruments" / "render_tmp" / "resources" / "Salamander Grand Piano V3.sfz")
    sfz_path = next((p for p in sfz_candidates if p.is_file()), None)
    if not sfz_path:
        sal_dir = plugin_root / "assets_local" / "salamander"
        if sal_dir.is_dir():
            sfz_path = next((p for p in sorted(sal_dir.rglob("*Salamander*.sfz")) if p.is_file()), None)
    if not sfz_path:
        raise FileNotFoundError("Salamander Grand Piano V3.sfz not found. Run setup_assets.py --download-salamander")

    sfizz_exe = cfg.get("sfizz_render_exe") or shutil.which("sfizz_render")
    if not sfizz_exe:
        sfizz_dir = plugin_root / "assets_local" / "sfizz"
        if sfizz_dir.is_dir():
            for name in (("sfizz_render.exe", "sfizz_render") if sys.platform == "win32" else ("sfizz_render",)):
                hit = next((p for p in sfizz_dir.rglob(name) if p.is_file()), None)
                if hit:
                    sfizz_exe = str(hit)
                    break
    if backend in ("auto", "sfizz_render") and sfizz_exe:
        cmd = [
            str(sfizz_exe),
            "--sfz", str(sfz_path),
            "--midi", str(mid_path),
            "--wav", str(raw_wav),
            "--samplerate", str(rate),
            "--quality", "1",
            "--polyphony", "64",
            "--use-eot",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired as exc:
            if backend == "sfizz_render":
                raise RuntimeError(
                    f"sfizz_render timed out after {exc.timeout}s while rendering {mid_path.name}."
                ) from exc
            print(f"[WARN] sfizz_render timed out after {exc.timeout}s; checking fallback backend.")
            proc = None
        if proc is not None and proc.returncode == 0 and raw_wav.is_file():
            return

    ml_root = Path(ml_root_str) if ml_root_str else None
    if backend in ("auto", "native_host") and ml_root and (ml_root / "scripts" / "native_host.py").is_file():
        sys.path.insert(0, str(ml_root))
        sys.path.insert(0, str(ml_root / "src"))
        sys.path.insert(0, str(ml_root / "scripts"))
        from music_learning.instruments import materialize, preset_spec
        from native_host import NativeHost

        inst_root = ml_root / "data" / "instruments"
        cat = json.loads((inst_root / "catalog.json").read_text(encoding="utf-8"))
        preset = next(x for x in cat["presets"] if x["id"] == "salamander-grand")
        temp_dir = inst_root / "render_tmp"
        temp_dir.mkdir(exist_ok=True)
        local_entry = temp_dir / "resources" / preset["entrypoint"]
        if not local_entry.exists():
            local_entry = materialize(SimpleNamespace(instrument_library=inst_root), preset, temp_dir / "resources")
        spec = dict(id="instrument", path=preset["plugin_path"], binary_sha256=preset["plugin_sha256"], identifier=preset["identifier"])
        spec.update(preset_spec(preset, local_entry))
        host_notes = [{k: v for k, v in n.items() if k != "staff"} for n in render_notes]
        project = {
            "revision_id": "plugin-salamander-render",
            "sample_rate": rate,
            "block_size": 1024,
            "duration_samples": total_samples,
            "tail_samples": rate * 4,
            "tracks": [{"id": "piano", "gain_db": -1.0, "plugins": [spec], "notes": host_notes, "controllers": controllers}],
        }
        host = NativeHost(diagnostics=temp_dir / "native")
        host.call({"schema_version": 1, "action": "render", "project": project, "output_path": str(raw_wav)}, timeout=720)
        return

    raise RuntimeError("Neither sfizz_render nor music-learning NativeHost is available. Check setup_assets.py --check")


def apply_natural_hall_reverb(
    sig: np.ndarray,
    sr: int,
    wet_mix: float = 0.18,
    rt60_sec: float = 2.0,
) -> np.ndarray:
    """Apply warm, non-metallic stereo concert-hall reverb via pure NumPy FFT convolution.

    Uses decorrelated stereo early reflections + frequency-damped diffuse tail (highs decay
    faster than lows, wet bass high-passed at 120 Hz to keep low piano registers clean).
    """
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim == 1:
        sig = np.column_stack([sig, sig])
    if wet_mix <= 0.0 or len(sig) == 0:
        return sig

    ir_len = int(round(max(0.5, rt60_sec * 1.15) * sr))
    t = np.arange(ir_len, dtype=np.float64) / float(sr)
    rng = np.random.default_rng(202604)

    # Frequency-dependent exponential decay envelopes (RT60: -60 dB => exp(-6.9078 * t / rt60))
    tau_lo = max(0.15, rt60_sec / 6.9078)
    tau_hi = max(0.08, (rt60_sec * 0.42) / 6.9078)
    pre_s = int(round(0.018 * sr))
    fade_s = max(1, int(round(0.025 * sr)))
    gate = np.zeros(ir_len, dtype=np.float64)
    if pre_s < ir_len:
        gate[pre_s:] = 1.0
        f_end = min(ir_len, pre_s + fade_s)
        gate[pre_s:f_end] = np.linspace(0.0, 1.0, f_end - pre_s)

    env_lo = gate * np.exp(-np.maximum(0.0, t - 0.018) / tau_lo)
    env_hi = gate * np.exp(-np.maximum(0.0, t - 0.018) / tau_hi)

    freqs = np.fft.rfftfreq(ir_len, 1.0 / sr)
    hp_mask = np.clip((freqs - 90.0) / 80.0, 0.0, 1.0)
    lp_mask = 1.0 / (1.0 + (freqs / 2600.0) ** 2)
    hi_mask = (1.0 - lp_mask) * (1.0 / (1.0 + (freqs / 7500.0) ** 2))

    # Subtle stereo early reflections (14..62 ms)
    early_taps = (
        (0.014, 0.42, 0.25),
        (0.023, 0.22, 0.36),
        (0.034, 0.30, 0.18),
        (0.047, 0.15, 0.26),
        (0.061, 0.20, 0.14),
    )

    ir = np.zeros((ir_len, 2), dtype=np.float64)
    for ch in range(2):
        noise = rng.standard_normal(ir_len)
        sp = np.fft.rfft(noise)
        lo_tail = np.fft.irfft(sp * hp_mask * lp_mask, n=ir_len) * env_lo
        hi_tail = np.fft.irfft(sp * hp_mask * hi_mask, n=ir_len) * env_hi
        chan_ir = lo_tail + 0.55 * hi_tail
        for tap_t, g_l, g_r in early_taps:
            idx = int(round(tap_t * sr))
            if idx < ir_len:
                chan_ir[idx] += g_l if ch == 0 else g_r
        norm = float(np.sqrt(np.sum(chan_ir ** 2))) + 1e-12
        ir[:, ch] = chan_ir / norm

    n_out = len(sig)
    n_fft = 1 << (n_out + ir_len - 1).bit_length()
    wet = np.zeros_like(sig)
    for ch in range(2):
        conv = np.fft.irfft(
            np.fft.rfft(sig[:, ch], n=n_fft) * np.fft.rfft(ir[:, ch], n=n_fft),
            n=n_fft,
        )
        wet[:, ch] = conv[:n_out]

    dry_gain = float(np.sqrt(max(0.0, 1.0 - wet_mix * wet_mix)))
    return dry_gain * sig + wet_mix * wet


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 4.2-4.4 calibrated Salamander audio & MIDI renderer")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--orig-audio", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--stage1-cal", default=None, help="Optional Stage 1 stage1_calibration.json for rubato tempo-map alignment")
    ap.add_argument("--key-fifths", type=int, default=-2)
    ap.add_argument("--song-name", default="PianoPiece")
    ap.add_argument("--style-profile", default=None, help="Optional v3.0 style profile JSON or ID")
    ap.add_argument("--transkun-midi", default=None, help="Optional Stage 1 MIDI with real TransKun velocities for touch fusion")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    plugin_root = find_plugin_root()
    cfg = load_config(plugin_root, args.config)

    orig_d, sr_orig = sf.read(args.orig_audio, dtype="float64", always_2d=True)
    mono_orig = orig_d.mean(axis=1)
    target_rms_db = max(-26.0, min(-14.0, 20.0 * np.log10(float(np.sqrt(np.mean(mono_orig ** 2))) + 1e-12)))

    fft_full = np.fft.rfft(mono_orig)
    freqs_o = np.fft.rfftfreq(len(mono_orig), 1.0 / sr_orig)
    lh_band = np.fft.irfft(np.where((freqs_o >= 45.0) & (freqs_o <= 350.0), fft_full, 0.0), n=len(mono_orig))
    rh_band = np.fft.irfft(np.where((freqs_o >= 350.0) & (freqs_o <= 3500.0), fft_full, 0.0), n=len(mono_orig))

    xml_beat_type = int(ET.parse(args.xml).getroot().findtext(".//attributes/time/beat-type", "4"))
    tmap = load_tempo_map(args.stage1_cal, args.bpm, args.beats_per_bar, xml_beat_type, args.first_downbeat)
    notes, max_bar, beat_type = parse_notes_from_xml(Path(args.xml), args.bpm, args.beats_per_bar, args.first_downbeat, tempo_map=tmap)
    if not notes:
        raise ValueError(f"No playable notes found in MusicXML: {args.xml}")

    bar_lh_db, bar_rh_db = {}, {}
    for m in range(0, max_bar + 1):
        t0_raw, t1 = tmap.bar_bounds_sec(m)
        t0 = max(0.0, t0_raw)
        i0, i1 = int(t0 * sr_orig), min(len(mono_orig), int(t1 * sr_orig))
        bar_lh_db[m] = 20.0 * np.log10(float(np.sqrt(np.mean(lh_band[i0:i1] ** 2))) + 1e-12) if i1 > i0 else -55.0
        bar_rh_db[m] = 20.0 * np.log10(float(np.sqrt(np.mean(rh_band[i0:i1] ** 2))) + 1e-12) if i1 > i0 else -55.0

    rh_all = [n for n in notes if n["staff"] == 1]
    lh_all = [n for n in notes if n["staff"] == 2]
    if rh_all:
        for n in rh_all:
            n["raw_db"] = rh_harm_db(mono_orig, sr_orig, n["st"], n["pitch"])
        rh_dbs = [n["raw_db"] for n in rh_all]
        p10, p90 = np.percentile(rh_dbs, 10), np.percentile(rh_dbs, 90)
        rh_raw = [48.0 + np.clip((n["raw_db"] - p10) / (p90 - p10 + 1e-5), 0.0, 1.0) * 44.0 for n in rh_all]
        if len(rh_raw) >= 3:
            rh_sm = 0.60 * np.convolve(rh_raw, np.ones(3) / 3.0, mode="same") + 0.40 * np.array(rh_raw)
        else:
            rh_sm = np.array(rh_raw, dtype=np.float64)
        for n, v in zip(rh_all, rh_sm):
            n["calc_vel"] = int(np.clip(round(v), 42, 92))
        rh_slots: dict[int, list[dict]] = {}
        for n in rh_all:
            rh_slots.setdefault(n["start_slot"], []).append(n)
        for g in rh_slots.values():
            if len(g) >= 2:
                top = max(g, key=lambda x: x["pitch"])
                top["calc_vel"] = min(94, top["calc_vel"] + 4)

    bar_rh_mean, last_v = {}, 68.0
    for m in range(0, max_bar + 1):
        vals = [n["calc_vel"] for n in rh_all if n["measure"] == m]
        if vals:
            last_v = float(np.mean(vals))
        bar_rh_mean[m] = last_v

    if lh_all:
        for n in lh_all:
            nd = lh_expr_db(mono_orig, lh_band, sr_orig, n["st"], n["pitch"])
            n["raw_db"] = 0.50 * nd + 0.50 * bar_lh_db.get(n["measure"], nd)
        lh_p5, lh_p95 = np.percentile([n["raw_db"] for n in lh_all], 5), np.percentile([n["raw_db"] for n in lh_all], 95)
        for n in lh_all:
            m = n["measure"]
            norm = float(np.clip((n["raw_db"] - lh_p5) / (lh_p95 - lh_p5 + 1e-5), 0.0, 1.0))
            vel = 30.0 + (norm ** 1.15) * 42.0
            if n["pitch"] <= 40:
                vel -= 8.0
            elif n["pitch"] <= 46:
                vel -= 5.0
            if (bar_lh_db[m] - bar_rh_db[m]) < -4.0:
                vel = min(vel, bar_rh_mean[m] - 14.0)
            n["calc_vel"] = int(np.clip(round(vel), 26, 74))

    profile = load_style_profile(args.style_profile) if args.style_profile else None
    dyn_cfg = (profile or {}).get("dynamics", {})
    pedal_style = str(dyn_cfg.get("pedal_style", "legato_harmonic"))
    lh_ceiling_vel = int(dyn_cfg.get("lh_ceiling_vel", 90))

    if args.transkun_midi and Path(args.transkun_midi).is_file():
        src_vels = read_midi_velocity_events(Path(args.transkun_midi))
        fused_cnt = fuse_transkun_velocities(notes, src_vels, alpha=0.92, lh_ceiling_vel=lh_ceiling_vel)
        if fused_cnt > 0:
            print(f"Stage 4 TransKun velocity fusion applied to {fused_cnt}/{len(notes)} notes (alpha=0.92, lh_ceiling={lh_ceiling_vel})")
        aligned_cnt = align_microtiming_to_transkun(notes, src_vels, blend=0.80, max_shift_sec=0.10)
        if aligned_cnt > 0:
            print(f"Stage 4 TransKun micro-timing alignment applied to {aligned_cnt}/{len(notes)} notes")

    # Round 1 static dynamics check
    rh_med = float(np.median([n["calc_vel"] for n in rh_all])) if rh_all else 0.0
    lh_med = float(np.median([n["calc_vel"] for n in lh_all])) if lh_all else 0.0
    lh_max = int(max(n["calc_vel"] for n in lh_all)) if lh_all else 0
    print(f"Stage 4 Round 1 Static Check: RH_med={rh_med:.1f}, LH_med={lh_med:.1f}, LH_max={lh_max}")

    rate = 48000
    _, t_end = tmap.bar_bounds_sec(max_bar)
    real_pedals: list[dict] = []
    if pedal_style != "dry_articulated" and args.transkun_midi and Path(args.transkun_midi).is_file():
        real_pedals = read_midi_pedal_events(Path(args.transkun_midi), rate=rate)

    # P1-10: only stretch LH notes to bar end for legato styles when synthetic pedaling is used
    if pedal_style in ("legato_harmonic", "romantic_rubato") and len(real_pedals) < 2:
        for m in range(0, max_bar + 1):
            _, b_end = tmap.bar_bounds_sec(m)
            m_lh = sorted([n for n in lh_all if n["measure"] == m], key=lambda x: x["st"])
            for i, n in enumerate(m_lh):
                nxt = [o["st"] for o in m_lh[i + 1 :] if o["pitch"] == n["pitch"]]
                n["et"] = max(n["et"], (nxt[0] - 0.020) if nxt else (b_end - 0.015))

    render_notes, clusters = [], []
    for n in sorted(notes, key=lambda x: (x["st"], x["pitch"])):
        if not clusters or abs(n["st"] - clusters[-1][0]["st"]) > 0.025:
            clusters.append([n])
        else:
            clusters[-1].append(n)
    for cl in clusters:
        cl.sort(key=lambda x: x["pitch"])
        has_arp = len(cl) > 1 and any(x["arpeggiate"] for x in cl)
        all_human_timed = all(x.get("timing_aligned_transkun") for x in cl)
        step = 0.018 if has_arp else (0.0 if all_human_timed else 0.008)
        for idx, n in enumerate(cl):
            st = n["st"] + (idx * step if len(cl) > 1 else 0.0)
            dur = max(0.14, n["et"] - st) * (0.98 if n["staff"] == 1 else 1.0)
            render_notes.append({"sample": max(0, int(round(st * rate))), "duration_samples": int(round(dur * rate)), "note": n["pitch"], "velocity": round(n.get("calc_vel", 64) / 127.0, 4), "channel": 1, "staff": n["staff"]})
    if len(real_pedals) >= 2:
        controllers = real_pedals + [
            {"sample": max(0, int(round((t_end + 0.15) * rate))), "channel": 1, "controller": 64, "value": 0}
        ]
        print(f"Stage 4 Pedaling: using {len(real_pedals)} real CC64 events from {Path(args.transkun_midi).name}")
    else:
        controllers = compute_dynamic_pedal_events(
            notes,
            tempo_map=tmap,
            max_bar=max_bar,
            rate=rate,
            pedal_style=pedal_style,
            beats_per_bar=args.beats_per_bar,
            beat_type=beat_type,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_out = out_dir / f"{args.song_name}_Full_Calibrated_Salamander.mid"
    export_midi(render_notes, controllers, rate, args.bpm, args.beats_per_bar, beat_type, args.key_fifths, mid_out, tempo_map=tmap)

    with tempfile.TemporaryDirectory() as tmpd:
        raw_wav = Path(tmpd) / "raw.wav"
        render_with_sfizz_or_native(plugin_root, cfg, render_notes, controllers, mid_out, raw_wav, rate, int(round(t_end * rate)) + rate * 4)
        sig, sr = sf.read(str(raw_wav))

    # Preserve natural Salamander Grand Piano timbre & dynamics:
    # Add gentle stereo concert-hall reverb + transparent whole-song RMS loudness alignment
    # + -1.0 dBFS peak ceiling (no hard-coded EQ colorization and no 30ms envelope pumping).
    reverb_wet = 0.06 if pedal_style == "dry_articulated" else 0.18
    final_sig = apply_natural_hall_reverb(sig, sr, wet_mix=reverb_wet, rt60_sec=2.0)
    n_smp = len(final_sig)
    dar = min(n_smp, max(1, int(t_end * sr)))
    rms_cur = float(np.sqrt(np.mean(final_sig[:dar] ** 2)))
    if rms_cur > 1e-9:
        final_sig *= (10.0 ** (target_rms_db / 20.0)) / rms_cur
    ceil = 10.0 ** (-1.0 / 20.0)
    peak = float(np.max(np.abs(final_sig))) if n_smp else 0.0
    if peak > ceil:
        final_sig *= ceil / peak

    full_wav = out_dir / f"{args.song_name}_Full_Calibrated_Salamander.wav"
    prev_wav = out_dir / f"{args.song_name}_Bars0_20_Preview_Salamander.wav"
    sf.write(str(full_wav), final_sig, sr, subtype="PCM_24")

    _, b20_t_end = tmap.bar_bounds_sec(min(20, max_bar))
    b20_end = int((b20_t_end + 2.5) * sr)
    pv = final_sig[: min(len(final_sig), b20_end)].copy()
    fl = min(len(pv), int(1.5 * sr))
    pv[-fl:] *= np.linspace(1.0, 0.0, fl)[:, None]
    sf.write(str(prev_wav), pv, sr, subtype="PCM_24")
    print(f"Stage 4 Render Complete: {full_wav.name}, {prev_wav.name}, {mid_out.name}")


if __name__ == "__main__":
    main()
