"""X-axis sources (sample index -> x value) and the time axis item.

Summary
    Each plotted channel has a map between its sample index and the x
    value. LinearMap covers waveform time and sample index. ArrayMap
    uses another channel (for example "Time") as the x values.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np
import pyqtgraph as pg

from .formatting import format_duration

# Axis label formats.
FMT_NUMBER = "number"
FMT_RELATIVE = "relative"
FMT_ABSOLUTE = "absolute"


class LinearMap:
    """x = x0 + index * dx."""

    monotonic = True

    def __init__(self, x0: float, dx: float):
        self.x0 = float(x0)
        self.dx = float(dx) if dx and math.isfinite(dx) and dx > 0 else 1.0

    def index_to_x(self, idx):
        return self.x0 + np.asarray(idx, dtype=np.float64) * self.dx

    def x_of(self, i: float) -> float:
        return self.x0 + i * self.dx

    def index_range(self, xa: float, xb: float, s: int, e: int) -> tuple[int, int]:
        """Sample range [i0, i1) that covers x in [xa, xb], plus one sample each side."""
        if e <= s:
            return s, s
        fa = (xa - self.x0) / self.dx
        fb = (xb - self.x0) / self.dx
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return s, e
        i0 = max(s, min(e, int(math.floor(fa)) - 1))
        i1 = max(s, min(e, int(math.ceil(fb)) + 2))
        return i0, i1

    def nearest(self, x: float, s: int, e: int) -> int:
        if e <= s:
            return -1
        return int(min(e - 1, max(s, round((x - self.x0) / self.dx))))


class ArrayMap:
    """x = X[index] from a channel. Monotonic X allows view decimation."""

    def __init__(self, x: np.ndarray, monotonic: bool):
        self.x = x
        self.monotonic = monotonic

    def index_to_x(self, idx):
        idx = np.asarray(idx, dtype=np.float64)
        n = self.x.size
        if n == 0:
            return np.full(idx.shape, np.nan)
        i = np.clip(np.floor(idx).astype(np.int64), 0, n - 1)
        j = np.minimum(i + 1, n - 1)
        f = np.clip(idx - i, 0.0, 1.0)
        xi = self.x[i]
        return xi + f * (self.x[j] - xi)

    def x_of(self, i: float) -> float:
        return float(self.index_to_x(np.array([i]))[0])

    def index_range(self, xa: float, xb: float, s: int, e: int) -> tuple[int, int]:
        e = min(e, self.x.size)
        if e <= s:
            return s, s
        if not self.monotonic:
            return s, e
        seg = self.x[s:e]
        i0 = s + int(np.searchsorted(seg, xa, side="left")) - 1
        i1 = s + int(np.searchsorted(seg, xb, side="right")) + 1
        return max(s, min(e, i0)), max(s, min(e, i1))

    def nearest(self, x: float, s: int, e: int) -> int:
        e = min(e, self.x.size)
        if e <= s:
            return -1
        seg = self.x[s:e]
        if self.monotonic:
            k = int(np.searchsorted(seg, x))
            cand = [c for c in (k - 1, k) if 0 <= c < seg.size]
        else:
            with np.errstate(invalid="ignore"):
                cand = [int(np.nanargmin(np.abs(seg - x)))] if np.isfinite(seg).any() else []
        if not cand:
            return -1
        best = min(cand, key=lambda c: abs(seg[c] - x) if np.isfinite(seg[c]) else math.inf)
        return s + best


def is_monotonic(x: np.ndarray) -> bool:
    """True if x is finite and non-decreasing."""
    if x.size < 2:
        return bool(np.isfinite(x).all())
    if not np.isfinite(x).all():
        return False
    return bool((np.diff(x) >= 0).all())


# -- axis item ----------------------------------------------------------------

_TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800,
               21600, 43200, 86400, 2 * 86400, 7 * 86400, 14 * 86400, 28 * 86400]
_MINOR = {1: 0.2, 2: 0.5, 5: 1, 10: 2, 15: 5, 30: 5, 60: 10, 120: 30, 300: 60, 600: 120,
          900: 300, 1800: 300, 3600: 600, 7200: 1800, 10800: 1800, 21600: 3600,
          43200: 7200, 86400: 21600}


class TimeAxisItem(pg.AxisItem):
    """Axis with number, relative time (HH:MM:SS.fff) or absolute time labels."""

    def __init__(self, orientation="bottom", **kw):
        super().__init__(orientation, **kw)
        self.fmt = FMT_NUMBER
        self.t_ref = None  # Unix seconds (UTC) of x == 0, for FMT_ABSOLUTE
        self.enableAutoSIPrefix(False)

    def set_format(self, fmt: str, t_ref: float | None = None) -> None:
        self.fmt = fmt if (fmt != FMT_ABSOLUTE or t_ref is not None) else FMT_RELATIVE
        self.t_ref = t_ref
        self.picture = None
        self.update()

    def _utc_offset(self) -> float:
        if self.t_ref is None:
            return 0.0
        try:
            off = _dt.datetime.fromtimestamp(self.t_ref).astimezone().utcoffset()
            return off.total_seconds() if off else 0.0
        except (OverflowError, OSError, ValueError):
            return 0.0

    def tickSpacing(self, minVal, maxVal, size):
        if self.fmt == FMT_NUMBER:
            return super().tickSpacing(minVal, maxVal, size)
        span = maxVal - minVal
        if not (span > 0 and math.isfinite(span)):
            return super().tickSpacing(minVal, maxVal, size)
        per_label = 115.0 if self.orientation in ("bottom", "top") else 30.0
        want = span / max(2.0, size / per_label)
        if want < 1.0:
            return super().tickSpacing(minVal, maxVal, size)
        major = next((s for s in _TIME_STEPS if s >= want), None)
        if major is None:
            days = 86400.0 * 10 ** math.ceil(math.log10(want / 86400.0))
            return [(days, 0), (days / 10, 0)]
        minor = _MINOR.get(major, major / 4)
        off = 0.0
        if self.fmt == FMT_ABSOLUTE:
            off = (-(self.t_ref + self._utc_offset())) % major
        return [(float(major), off), (float(minor), off % minor if minor else 0)]

    def tickValues(self, minVal, maxVal, size):
        if self.fmt == FMT_NUMBER:
            return super().tickValues(minVal, maxVal, size)
        out = []
        for spacing, offset in self.tickSpacing(minVal, maxVal, size):
            start = math.ceil((minVal - offset) / spacing) * spacing + offset
            n = int((maxVal - start) / spacing) + 1
            if n > 2000 or n < 0:
                continue
            vals = [start + k * spacing for k in range(n)]
            prev = [v for _, vs in out for v in vs]
            vals = [v for v in vals if all(abs(v - p) > spacing * 1e-6 for p in prev)]
            out.append((spacing, vals))
        return out

    def tickStrings(self, values, scale, spacing):
        if self.fmt == FMT_NUMBER:
            return super().tickStrings(values, scale, spacing)
        dec = 0 if spacing >= 1 else min(9, max(0, int(math.ceil(-math.log10(spacing) - 1e-9))))
        if self.fmt == FMT_RELATIVE:
            return [format_duration(v, dec) for v in values]
        out = []
        for v in values:
            try:
                t = _dt.datetime.fromtimestamp(self.t_ref + v)
            except (OverflowError, OSError, ValueError):
                out.append("")
                continue
            if spacing >= 86400:
                out.append(t.strftime("%Y-%m-%d"))
            elif spacing >= 1:
                out.append(t.strftime("%H:%M:%S"))
            else:
                frac = f"{t.microsecond / 1e6:.{dec}f}"[1:] if dec else ""
                out.append(t.strftime("%H:%M:%S") + frac)
        return out
