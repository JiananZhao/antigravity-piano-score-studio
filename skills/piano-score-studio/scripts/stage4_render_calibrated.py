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


def parse_notes_from_xml(xml_path: Path, bpm: float, beats_per_bar: int, first_downbeat: float) -> tuple[list[dict], int, int]:
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


def export_midi(render_notes: list[dict], controllers: list[dict], rate: int, bpm: float, beats_per_bar: int, beat_type: int, key_fifths: int, out_mid: Path) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.name = "Calibrated Salamander Edition"
    meta.append(mido.MetaMessage("time_signature", numerator=beats_per_bar, denominator=beat_type, time=0))
    meta.append(mido.MetaMessage("key_signature", key=FIFTHS_TO_KEY.get(key_fifths, "C"), time=0))
    quarter_bpm = bpm * (4.0 / beat_type)
    meta.append(mido.MetaMessage("set_tempo", tempo=int(round(60_000_000 / quarter_bpm)), time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))

    ticks_per_sec = 480.0 * (quarter_bpm / 60.0)
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
        cur_s = 0
        for smp, kind, p1, p2 in evs:
            dt = max(0, int(round(((smp - cur_s) / rate) * ticks_per_sec)))
            cur_s = smp
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
        cmd = [str(sfizz_exe), "--sfz", str(sfz_path), "--midi", str(mid_path), "--wav", str(raw_wav), "--samplerate", str(rate)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and raw_wav.is_file():
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


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Stage 4.2-4.4 calibrated Salamander audio & MIDI renderer")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--orig-audio", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--beats-per-bar", type=int, required=True)
    ap.add_argument("--first-downbeat", type=float, required=True)
    ap.add_argument("--key-fifths", type=int, default=-2)
    ap.add_argument("--song-name", default="PianoPiece")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    plugin_root = find_plugin_root()
    cfg = load_config(plugin_root, args.config)

    orig_d, sr_orig = sf.read(args.orig_audio, dtype="float64", always_2d=True)
    mono_orig = orig_d.mean(axis=1)
    orig_side = 0.5 * (orig_d[:, 0] - orig_d[:, 1]) if orig_d.shape[1] >= 2 else np.zeros_like(mono_orig)
    target_sm = max(0.20, min(0.75, float(np.sqrt(np.mean(orig_side ** 2)) / (np.sqrt(np.mean(mono_orig ** 2)) + 1e-12))))
    target_rms_db = max(-26.0, min(-14.0, 20.0 * np.log10(float(np.sqrt(np.mean(mono_orig ** 2))) + 1e-12)))

    fft_full = np.fft.rfft(mono_orig)
    freqs_o = np.fft.rfftfreq(len(mono_orig), 1.0 / sr_orig)
    lh_band = np.fft.irfft(np.where((freqs_o >= 45.0) & (freqs_o <= 350.0), fft_full, 0.0), n=len(mono_orig))
    rh_band = np.fft.irfft(np.where((freqs_o >= 350.0) & (freqs_o <= 3500.0), fft_full, 0.0), n=len(mono_orig))

    notes, max_bar, beat_type = parse_notes_from_xml(Path(args.xml), args.bpm, args.beats_per_bar, args.first_downbeat)
    if not notes:
        raise ValueError(f"No playable notes found in MusicXML: {args.xml}")
    bar_s = args.beats_per_bar * (60.0 / args.bpm)

    bar_lh_db, bar_rh_db = {}, {}
    for m in range(0, max_bar + 1):
        t0 = max(0.0, args.first_downbeat + (m - 1) * bar_s)
        t1 = t0 + bar_s
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

    # Round 1 static dynamics check
    rh_med = float(np.median([n["calc_vel"] for n in rh_all])) if rh_all else 0.0
    lh_med = float(np.median([n["calc_vel"] for n in lh_all])) if lh_all else 0.0
    lh_max = int(max(n["calc_vel"] for n in lh_all)) if lh_all else 0
    print(f"Stage 4 Round 1 Static Check: RH_med={rh_med:.1f}, LH_med={lh_med:.1f}, LH_max={lh_max}")

    rate = 48000
    for m in range(1, max_bar + 1):
        b_end = args.first_downbeat + m * bar_s
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
        step = 0.018 if (len(cl) > 1 and any(x["arpeggiate"] for x in cl)) else 0.008
        for idx, n in enumerate(cl):
            st = n["st"] + (idx * step if len(cl) > 1 else 0.0)
            dur = max(0.14, n["et"] - st) * (0.98 if n["staff"] == 1 else 1.0)
            render_notes.append({"sample": max(0, int(round(st * rate))), "duration_samples": int(round(dur * rate)), "note": n["pitch"], "velocity": round(n.get("calc_vel", 64) / 127.0, 4), "channel": 1, "staff": n["staff"]})

    controllers = []
    for m in range(1, max_bar + 1):
        tb = args.first_downbeat + (m - 1) * bar_s
        controllers.append({"sample": max(0, int(round((tb - 0.018) * rate))), "channel": 1, "controller": 64, "value": 0})
        controllers.append({"sample": int(round((tb + 0.028) * rate)), "channel": 1, "controller": 64, "value": 127})
    t_end = args.first_downbeat + max_bar * bar_s
    controllers.append({"sample": int(round((t_end + 0.10) * rate)), "channel": 1, "controller": 64, "value": 0})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid_out = out_dir / f"{args.song_name}_Full_Calibrated_Salamander.mid"
    export_midi(render_notes, controllers, rate, args.bpm, args.beats_per_bar, beat_type, args.key_fifths, mid_out)

    with tempfile.TemporaryDirectory() as tmpd:
        raw_wav = Path(tmpd) / "raw.wav"
        render_with_sfizz_or_native(plugin_root, cfg, render_notes, controllers, mid_out, raw_wav, rate, int(round(t_end * rate)) + rate * 4)
        sig, sr = sf.read(str(raw_wav))

    mid, side = 0.5 * (sig[:, 0] + sig[:, 1]), 0.5 * (sig[:, 0] - sig[:, 1])
    cur_sm = float(np.sqrt(np.mean(side ** 2)) / (np.sqrt(np.mean(mid ** 2)) + 1e-12))
    wf = float(np.clip(target_sm / (cur_sm + 1e-6), 0.35, 1.35))
    sig_ms = np.column_stack([mid + side * wf, mid - side * wf])

    n_smp = len(sig_ms)
    freqs = np.fft.rfftfreq(n_smp, 1.0 / sr)
    eq = np.ones_like(freqs)
    sm = freqs < 130.0
    eq[sm] = 0.32 + 0.68 * (np.clip((freqs[sm] - 30.0) / 100.0, 0.0, 1.0) ** 1.4)
    eq += 0.45 * np.exp(-((freqs - 265.0) / 145.0) ** 2)
    hm = freqs > 1400.0
    eq[hm] *= 0.25 + 0.75 / (1.0 + ((freqs[hm] - 1400.0) / 900.0) ** 2)

    lh_sp = np.clip((360.0 - freqs) / 40.0, 0.0, 1.0)
    rh_sp = 1.0 - lh_sp
    sig_lh = np.column_stack([np.fft.irfft(np.fft.rfft(sig_ms[:, c]) * eq * lh_sp, n=n_smp) for c in range(2)])
    sig_rh = np.column_stack([np.fft.irfft(np.fft.rfft(sig_ms[:, c]) * eq * rh_sp, n=n_smp) for c in range(2)])

    m_lh_r, m_rh_r = sig_lh.mean(axis=1), sig_rh.mean(axis=1)
    dao, dar = min(len(mono_orig), int(t_end * sr_orig)), min(n_smp, int(t_end * sr))
    sig_lh *= np.sqrt(np.mean(lh_band[:dao] ** 2)) / (np.sqrt(np.mean(m_lh_r[:dar] ** 2)) + 1e-12)
    sig_rh *= np.sqrt(np.mean(rh_band[:dao] ** 2)) / (np.sqrt(np.mean(m_rh_r[:dar] ** 2)) + 1e-12)
    m_lh_r, m_rh_r = sig_lh.mean(axis=1), sig_rh.mean(axis=1)

    t_anc = np.arange(0.0, t_end + 0.5, 0.030)
    gla, gra = [], []
    for tc in t_anc:
        ro = float(np.sqrt(np.mean(lh_band[max(0, int((tc - 0.275) * sr_orig)) : min(len(mono_orig), int((tc + 0.275) * sr_orig))] ** 2) + 1e-10))
        rr = float(np.sqrt(np.mean(m_lh_r[max(0, int((tc - 0.275) * sr)) : min(n_smp, int((tc + 0.275) * sr))] ** 2) + 1e-10))
        gla.append(float(np.clip((ro / rr) ** 0.72, 0.45, 1.85)) if (tc <= t_end and rr > 1e-4) else 1.0)
        ro2 = float(np.sqrt(np.mean(rh_band[max(0, int((tc - 0.275) * sr_orig)) : min(len(mono_orig), int((tc + 0.275) * sr_orig))] ** 2) + 1e-10))
        rr2 = float(np.sqrt(np.mean(m_rh_r[max(0, int((tc - 0.275) * sr)) : min(n_smp, int((tc + 0.275) * sr))] ** 2) + 1e-10))
        gra.append(float(np.clip((ro2 / rr2) ** 0.65, 0.55, 1.80)) if (tc <= t_end and rr2 > 1e-4) else 1.0)

    k = np.exp(-0.5 * (np.arange(-12, 13) / 5.0) ** 2); k /= np.sum(k)
    tg = np.arange(n_smp) / sr
    el = np.interp(tg, t_anc, np.convolve(np.pad(gla, 12, mode="edge"), k, mode="valid"))
    er = np.interp(tg, t_anc, np.convolve(np.pad(gra, 12, mode="edge"), k, mode="valid"))
    final_sig = sig_lh * el[:, None] + sig_rh * er[:, None]

    rms_cur = float(np.sqrt(np.mean(final_sig[:dar] ** 2)))
    final_sig *= (10.0 ** (target_rms_db / 20.0)) / (rms_cur + 1e-12)
    ceil = 10.0 ** (-3.5 / 20.0)
    final_sig = ceil * np.tanh(final_sig / ceil)
    if np.max(np.abs(final_sig)) > ceil:
        final_sig *= ceil / np.max(np.abs(final_sig))

    full_wav = out_dir / f"{args.song_name}_Full_Calibrated_Salamander.wav"
    prev_wav = out_dir / f"{args.song_name}_Bars0_20_Preview_Salamander.wav"
    sf.write(str(full_wav), final_sig, sr, subtype="PCM_24")

    b20_end = int((args.first_downbeat + min(20, max_bar) * bar_s + 2.5) * sr)
    pv = final_sig[: min(len(final_sig), b20_end)].copy()
    fl = min(len(pv), int(1.5 * sr))
    pv[-fl:] *= np.linspace(1.0, 0.0, fl)[:, None]
    sf.write(str(prev_wav), pv, sr, subtype="PCM_24")
    print(f"Stage 4 Render Complete: {full_wav.name}, {prev_wav.name}, {mid_out.name}")


if __name__ == "__main__":
    main()
