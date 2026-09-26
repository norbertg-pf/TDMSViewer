"""Text formatting for values, properties and time axes.

Rule: show the exact stored value. Floats use the shortest text that
reads back to the same bit pattern (Python repr). No rounding.
"""

from __future__ import annotations

import datetime as _dt
import math

import numpy as np

_EPOCH = np.datetime64(0, "us")


def format_float(value: float) -> str:
    """Return the exact, round-trip text of a float."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    return repr(float(value))


def format_datetime64(value: np.datetime64) -> str:
    """Return a timestamp as local time with UTC offset (TDMS stores UTC)."""
    if np.isnat(value):
        return "NaT"
    us = int((value.astype("datetime64[us]") - _EPOCH).astype(np.int64))
    try:
        utc = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(microseconds=us)
        return utc.astimezone().isoformat(sep=" ", timespec="microseconds")
    except (OverflowError, OSError, ValueError):
        return str(value)


def format_value(value: object) -> str:
    """Return display text for one data value or property value."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return format_float(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, np.datetime64):
        return format_datetime64(value)
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.astimezone().isoformat(sep=" ", timespec="microseconds")
    if isinstance(value, (complex, np.complexfloating)):
        c = complex(value)
        return f"{format_float(c.real)}{'+' if c.imag >= 0 or math.isnan(c.imag) else '-'}{format_float(abs(c.imag))}j"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def format_duration(seconds: float, decimals: int) -> str:
    """Return relative time as [-][HH:]MM:SS.fff (NI "relative time" style).

    decimals: number of digits after the decimal point (0..12).
    """
    if not math.isfinite(seconds):
        return format_float(seconds)
    sign = "-" if seconds < 0 else ""
    s = abs(seconds)
    scale = 10 ** decimals
    ticks = round(s * scale)
    whole, frac = divmod(ticks, scale)
    h, rem = divmod(whole, 3600)
    m, sec = divmod(rem, 60)
    text = f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
    if decimals > 0:
        text += f".{frac:0{decimals}d}"
    return sign + text


def format_si(value: float, digits: int = 6) -> str:
    """Return a compact number for statistics and cursor readouts."""
    if not math.isfinite(value):
        return format_float(value)
    return f"{value:.{digits}g}"
