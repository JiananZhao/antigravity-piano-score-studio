"""Shared TempoMap helper for fixed-tempo and rubato (tempo-map) alignment across Stages 1-5."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class TempoMap:
    """Maps between musical quarter-note positions (Bar 1 beat 1 == q 0.0) and wall-clock seconds."""

    bpm: float
    beats_per_bar: int
    beat_type: int
    first_downbeat: float
    mode: str = "fixed-tempo"
    q_grid: np.ndarray | None = None
    t_grid: np.ndarray | None = None

    @property
    def quarter_bpm(self) -> float:
        return self.bpm * (4.0 / float(self.beat_type))

    @property
    def quarter_sec(self) -> float:
        return 60.0 / self.quarter_bpm

    @property
    def bar_quarters(self) -> float:
        return float(self.beats_per_bar) * (4.0 / float(self.beat_type))

    @property
    def is_rubato(self) -> bool:
        return (
            self.mode == "tempo-map"
            and self.q_grid is not None
            and self.t_grid is not None
            and len(self.q_grid) >= 2
            and len(self.t_grid) == len(self.q_grid)
        )

    def _clamped_period(self, dt: float) -> float:
        q_s = self.quarter_sec
        return float(np.clip(dt, 0.5 * q_s, 2.0 * q_s))

    def q_to_sec(self, q: float) -> float:
        """Convert musical quarter-note offset (relative to Bar 1 beat 1 = 0.0) to seconds."""
        if not self.is_rubato:
            return float(self.first_downbeat + q * self.quarter_sec)
        assert self.q_grid is not None and self.t_grid is not None
        if q <= float(self.q_grid[0]):
            dt0 = self._clamped_period(float(self.t_grid[1] - self.t_grid[0]))
            return float(self.t_grid[0] + (q - float(self.q_grid[0])) * dt0)
        if q >= float(self.q_grid[-1]):
            dt1 = self._clamped_period(float(self.t_grid[-1] - self.t_grid[-2]))
            return float(self.t_grid[-1] + (q - float(self.q_grid[-1])) * dt1)
        return float(np.interp(q, self.q_grid, self.t_grid))

    def sec_to_q(self, t_sec: float) -> float:
        """Inverse mapping from wall-clock seconds to musical quarter-note offset."""
        if not self.is_rubato:
            return float((t_sec - self.first_downbeat) / self.quarter_sec)
        assert self.q_grid is not None and self.t_grid is not None
        if t_sec <= float(self.t_grid[0]):
            dt0 = self._clamped_period(float(self.t_grid[1] - self.t_grid[0]))
            return float(self.q_grid[0] + (t_sec - float(self.t_grid[0])) / dt0)
        if t_sec >= float(self.t_grid[-1]):
            dt1 = self._clamped_period(float(self.t_grid[-1] - self.t_grid[-2]))
            return float(self.q_grid[-1] + (t_sec - float(self.t_grid[-1])) / dt1)
        return float(np.interp(t_sec, self.t_grid, self.q_grid))

    def slot_to_sec(self, slot_rel_bar1: float, divisions: int) -> float:
        """Convert MusicXML slot offset relative to Bar 1 beat 1 into wall-clock seconds."""
        return self.q_to_sec(float(slot_rel_bar1) / float(divisions))

    def bar_bounds_sec(self, bar_num: int) -> tuple[float, float]:
        """Return (t_start, t_end) in seconds for 1-indexed full bar `bar_num` (or Bar 0)."""
        q0 = (bar_num - 1) * self.bar_quarters
        q1 = bar_num * self.bar_quarters
        return self.q_to_sec(q0), self.q_to_sec(q1)

    def midi_tempo_segments(
        self,
        ticks_per_beat: int,
        q_origin: float = 0.0,
        q_end: float | None = None,
    ) -> list[tuple[int, float, int]]:
        """Build piecewise-constant MIDI tempo segments: [(start_tick, start_sec, tempo_us_per_quarter)].

        MIDI tick 0 corresponds to wall-clock `t_origin = max(0.0, q_to_sec(q_origin))` when fixed,
        or to `t = 0.0` via a leading tempo segment so every event at `t_sec >= 0` maps accurately.
        """
        if not self.is_rubato:
            us = max(100_000, min(2_500_000, int(round(60_000_000.0 / self.quarter_bpm))))
            return [(0, 0.0, us)]

        assert self.q_grid is not None and self.t_grid is not None
        # Anchor tick 0 at wall-clock t = 0.0 so MIDI time == audio time
        q_at_zero = self.sec_to_q(0.0)
        q_hi = float(self.q_grid[-1]) if q_end is None else max(float(q_end), float(self.q_grid[-1]))

        # Collect knot points in quarter-notes starting strictly at q_at_zero
        knots_q = [q_at_zero]
        for qv in self.q_grid:
            if float(qv) > q_at_zero + 1e-6:
                knots_q.append(float(qv))
        if q_hi > knots_q[-1] + 1e-6:
            knots_q.append(q_hi)

        segments: list[tuple[int, float, int]] = []
        for i in range(len(knots_q) - 1):
            qa, qb = knots_q[i], knots_q[i + 1]
            dq = qb - qa
            if dq <= 1e-6:
                continue
            ta = max(0.0, self.q_to_sec(qa))
            tb = max(ta + 1e-4, self.q_to_sec(qb))
            dt_per_q = self._clamped_period((tb - ta) / dq)
            us = max(100_000, min(2_500_000, int(round(dt_per_q * 1_000_000.0))))
            tick_a = max(0, int(round((qa - q_at_zero) * ticks_per_beat)))
            if not segments or tick_a > segments[-1][0]:
                segments.append((tick_a, ta, us))
        if not segments:
            us = max(100_000, min(2_500_000, int(round(60_000_000.0 / self.quarter_bpm))))
            segments.append((0, 0.0, us))
        return segments

    def sec_to_tick(
        self,
        t_sec: float,
        ticks_per_beat: int,
        segments: list[tuple[int, float, int]] | None = None,
    ) -> int:
        """Convert wall-clock seconds to exact MIDI tick using piecewise tempo segments."""
        t_clamped = max(0.0, float(t_sec))
        if not self.is_rubato:
            ticks_per_sec = float(ticks_per_beat) * (self.quarter_bpm / 60.0)
            return max(0, int(round(t_clamped * ticks_per_sec)))

        segs = segments if segments is not None else self.midi_tempo_segments(ticks_per_beat)
        # Walk segments using their exact MIDI tempo durations so tick->sec round-trip is exact
        cur_t = 0.0
        for idx, (tk0, _, us) in enumerate(segs):
            sec_per_tick = (us / 1_000_000.0) / float(ticks_per_beat)
            if idx + 1 < len(segs):
                tk1 = segs[idx + 1][0]
                seg_dur = (tk1 - tk0) * sec_per_tick
                if t_clamped <= cur_t + seg_dur + 1e-9:
                    return max(0, int(round(tk0 + (t_clamped - cur_t) / sec_per_tick)))
                cur_t += seg_dur
            else:
                return max(0, int(round(tk0 + (t_clamped - cur_t) / sec_per_tick)))
        return 0


def load_tempo_map(
    stage1_cal_path: str | Path | None,
    bpm: float,
    beats_per_bar: int,
    beat_type: int,
    first_downbeat: float,
) -> TempoMap:
    """Load rubato TempoMap from stage1_calibration.json when tempo_mode == 'tempo-map', else fixed-tempo."""
    base = TempoMap(
        bpm=float(bpm),
        beats_per_bar=int(beats_per_bar),
        beat_type=int(beat_type),
        first_downbeat=float(first_downbeat),
        mode="fixed-tempo",
    )
    if not stage1_cal_path:
        return base
    p = Path(stage1_cal_path)
    if not p.is_file():
        return base
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return base

    cal = data.get("calibration") or {}
    tmap = data.get("tempo_map") or cal.get("tempo_map") or {}
    mode = str(tmap.get("mode") or cal.get("tempo_mode") or "fixed-tempo")
    raw_beats = tmap.get("quarter_beats")
    if mode != "tempo-map" or not isinstance(raw_beats, list) or len(raw_beats) < 2:
        return base

    t_arr = np.asarray(raw_beats, dtype=np.float64)
    # Enforce strictly increasing beat timestamps
    keep = [0]
    for i in range(1, len(t_arr)):
        if t_arr[i] > t_arr[keep[-1]] + 1e-4:
            keep.append(i)
    if len(keep) < 2:
        return base
    t_arr = t_arr[keep]

    ref_db = float(tmap.get("first_downbeat", first_downbeat))
    k0 = int(np.argmin(np.abs(t_arr - ref_db)))
    q_arr = np.arange(len(t_arr), dtype=np.float64) - float(k0)
    return TempoMap(
        bpm=float(bpm),
        beats_per_bar=int(beats_per_bar),
        beat_type=int(beat_type),
        first_downbeat=float(t_arr[k0]),
        mode="tempo-map",
        q_grid=q_arr,
        t_grid=t_arr,
    )