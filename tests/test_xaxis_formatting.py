"""Tests for tdmsviewer.xaxis (index <-> x maps, time axis) and tdmsviewer.formatting."""

from __future__ import annotations

import datetime as dt
import math
import struct

import numpy as np
import pytest

from tdmsviewer import xaxis
from tdmsviewer.formatting import (format_datetime64, format_duration, format_float, format_si,
                                   format_value)
from tdmsviewer.xaxis import FMT_ABSOLUTE, FMT_NUMBER, FMT_RELATIVE, ArrayMap, LinearMap, TimeAxisItem, is_monotonic


# -- LinearMap -----------------------------------------------------------------------------

@pytest.mark.parametrize("dx", [0, -1.0, float("nan"), float("inf"), None])
def test_linear_map_bad_step_becomes_one(dx):
    m = LinearMap(2.0, dx)
    assert m.dx == 1.0
    assert m.x_of(3) == 5.0


def test_linear_map_index_to_x():
    m = LinearMap(-1.5, 0.25)
    assert m.x_of(0) == -1.5 and m.x_of(10) == 1.0
    np.testing.assert_array_equal(m.index_to_x([0, 1, 2.5]), [-1.5, -1.25, -0.875])
    assert m.index_to_x(np.arange(5)).dtype == np.float64
    assert m.index_to_x(4) == -0.5


def _needed(fa, fb):
    """Index range whose x is inside [xa, xb] (as floats of the index)."""
    return math.ceil(fa), math.floor(fb)


@pytest.mark.parametrize("x0,dx", [(0.0, 1.0), (10.0, 0.001), (-5.0, 3.0), (1e9, 1e-3)])
def test_linear_map_index_range_covers_view(x0, dx, rng):
    m = LinearMap(x0, dx)
    s, e = 3, 5003
    for _ in range(300):
        fa, fb = np.sort(rng.uniform(-100, 5200, 2))
        xa, xb = x0 + fa * dx, x0 + fb * dx
        i0, i1 = m.index_range(xa, xb, s, e)
        assert s <= i0 <= i1 <= e
        lo, hi = _needed((xa - x0) / dx, (xb - x0) / dx)
        lo, hi = max(s, lo - 1), min(e - 1, hi + 1)  # plus one sample each side
        if lo <= hi:
            assert i0 <= lo and i1 >= hi + 1, (fa, fb, i0, i1, lo, hi)
        # At most two extra samples on each side.
        assert i0 >= max(s, math.floor((xa - x0) / dx) - 2)
        assert i1 <= min(e, math.ceil((xb - x0) / dx) + 3)


def test_linear_map_index_range_edges():
    m = LinearMap(0.0, 1.0)
    assert m.index_range(0, 10, 5, 5) == (5, 5)
    assert m.index_range(0, 10, 7, 3) == (7, 7)
    assert m.index_range(float("-inf"), float("inf"), 2, 9) == (2, 9)
    assert m.index_range(float("nan"), 3.0, 2, 9) == (2, 9)
    assert m.index_range(-1e300, 1e300, 0, 100) == (0, 100)
    assert m.index_range(1000, 2000, 0, 100) == (100, 100)
    assert m.index_range(-50, -10, 0, 100) == (0, 0)
    assert m.index_range(3.0, 3.0, 0, 100) == (2, 5)


def test_linear_map_nearest():
    m = LinearMap(1.0, 0.5)
    assert m.nearest(1.0, 0, 10) == 0
    assert m.nearest(1.74, 0, 10) == 1
    assert m.nearest(1.76, 0, 10) == 2
    assert m.nearest(-100, 3, 10) == 3
    assert m.nearest(100, 3, 10) == 9
    assert m.nearest(2.0, 5, 5) == -1


# -- ArrayMap ------------------------------------------------------------------------------

def test_array_map_integer_indices_are_exact(rng):
    x = np.cumsum(rng.uniform(0.001, 2.0, 1000))
    m = ArrayMap(x, True)
    idx = np.arange(1000)
    np.testing.assert_array_equal(m.index_to_x(idx), x)
    assert m.x_of(17) == x[17]


def test_array_map_interpolation_and_clipping():
    m = ArrayMap(np.array([0.0, 10.0, 20.0, 40.0]), True)
    np.testing.assert_array_equal(m.index_to_x([0.5, 1.25, 2.5]), [5.0, 12.5, 30.0])
    np.testing.assert_array_equal(m.index_to_x([-3.0, -0.5, 3.0, 3.5, 99.0]), [0.0, 0.0, 40.0, 40.0, 40.0])
    assert np.isnan(ArrayMap(np.empty(0), True).index_to_x([0, 1])).all()


def test_array_map_integer_index_next_to_nan_or_inf():
    """x[k] must come back unchanged even if a neighbour is NaN or Inf (CSV export, cursors)."""
    x = np.array([0.0, 1.0, np.nan, 3.0, 4.0, np.inf, 6.0])
    m = ArrayMap(x, is_monotonic(x))
    with np.errstate(invalid="ignore"):
        got = m.index_to_x(np.arange(x.size))
        x1 = m.x_of(1)
    np.testing.assert_array_equal(got, x)
    assert x1 == 1.0


def test_array_map_index_range_monotonic(rng):
    x = np.cumsum(rng.uniform(0.0, 1.0, 2000))  # includes near-duplicates
    x[500:510] = x[500]  # flat part
    m = ArrayMap(x, True)
    for _ in range(300):
        xa, xb = np.sort(rng.uniform(x[0] - 5, x[-1] + 5, 2))
        s = int(rng.integers(0, 100))
        e = int(rng.integers(1900, 2100))
        i0, i1 = m.index_range(xa, xb, s, e)
        e2 = min(e, x.size)
        assert s <= i0 <= i1 <= e2
        inside = np.flatnonzero((x[s:e2] >= xa) & (x[s:e2] <= xb)) + s
        if inside.size:
            assert i0 <= max(s, inside[0] - 1) and i1 >= min(e2, inside[-1] + 2)
        # At most one extra sample each side.
        before = np.flatnonzero(x[s:e2] < xa)
        after = np.flatnonzero(x[s:e2] > xb)
        if before.size:
            assert i0 >= s + before[-1]
        if after.size:
            assert i1 <= s + after[0] + 1


def test_array_map_index_range_edges():
    x = np.array([3.0, 1.0, 2.0])
    assert ArrayMap(x, False).index_range(0, 1, 0, 3) == (0, 3)
    assert ArrayMap(x, False).index_range(0, 1, 0, 10) == (0, 3)
    assert ArrayMap(np.arange(5.0), True).index_range(1, 2, 4, 2) == (4, 4)
    assert ArrayMap(np.arange(5.0), True).index_range(1, 2, 7, 9) == (7, 7)


@pytest.mark.parametrize("monotonic", [True, False])
def test_array_map_nearest_vs_brute_force(monotonic, rng):
    x = np.cumsum(rng.uniform(0.0, 1.0, 500))
    if not monotonic:
        x = rng.permutation(x)
        x[::17] = np.nan
    m = ArrayMap(x, monotonic)
    for _ in range(300):
        q = float(rng.uniform(np.nanmin(x) - 3, np.nanmax(x) + 3))
        s = int(rng.integers(0, 50))
        e = int(rng.integers(s + 1, 520))
        k = m.nearest(q, s, e)
        seg = x[s:min(e, x.size)]
        best = np.nanmin(np.abs(seg - q))
        assert s <= k < min(e, x.size)
        assert abs(x[k] - q) == best


def test_array_map_nearest_edges():
    assert ArrayMap(np.full(5, np.nan), False).nearest(1.0, 0, 5) == -1
    assert ArrayMap(np.arange(5.0), True).nearest(1.0, 3, 3) == -1
    assert ArrayMap(np.arange(5.0), True).nearest(1.0, 6, 9) == -1
    m = ArrayMap(np.array([0.0, 1.0, 2.0]), True)
    assert m.nearest(0.5, 0, 3) == 0  # tie: lower index
    assert m.nearest(10.0, 0, 3) == 2
    assert m.nearest(-10.0, 1, 3) == 1


@pytest.mark.parametrize("x,expected", [
    ([], True), ([1.0], True), ([np.nan], False), ([np.inf], False), ([1.0, 1.0, 2.0], True),
    ([1.0, 0.5], False), ([0.0, np.nan, 1.0], False), ([0.0, 1.0, np.inf], False), ([-np.inf, 0.0], False),
    ([-0.0, 0.0], True), ([0.0, -0.0], True),
])
def test_is_monotonic(x, expected):
    assert is_monotonic(np.asarray(x, dtype=np.float64)) is expected


# -- TimeAxisItem -----------------------------------------------------------------------------

@pytest.fixture
def axis(qapp):
    return TimeAxisItem("bottom")


def test_axis_formats(axis):
    assert axis.fmt == FMT_NUMBER
    axis.set_format(FMT_ABSOLUTE, None)
    assert axis.fmt == FMT_RELATIVE
    axis.set_format(FMT_ABSOLUTE, 1.7e9)
    assert axis.fmt == FMT_ABSOLUTE and axis.t_ref == 1.7e9
    axis.set_format(FMT_NUMBER)
    assert axis.fmt == FMT_NUMBER


def test_number_format_is_pyqtgraph_default(axis, qapp):
    import pyqtgraph as pg

    ref = pg.AxisItem("bottom")
    for lo, hi, size in [(0, 1, 500), (-3.3, 1e4, 800), (1e9, 1e9 + 0.5, 300)]:
        assert axis.tickSpacing(lo, hi, size) == ref.tickSpacing(lo, hi, size)
        assert axis.tickStrings([lo, hi], 1.0, 0.1) == ref.tickStrings([lo, hi], 1.0, 0.1)


@pytest.mark.parametrize("span", [2.0, 7.0, 45.0, 100.0, 1000.0, 5000.0, 4 * 3600.0, 3 * 86400.0, 20 * 86400.0,
                                  400 * 86400.0])
@pytest.mark.parametrize("size", [120, 600, 1900])
def test_relative_tick_spacing(axis, span, size):
    axis.set_format(FMT_RELATIVE)
    levels = axis.tickSpacing(10.0, 10.0 + span, size)
    want = span / max(2.0, size / 115.0)
    (major, off1), (minor, off2) = levels[0], levels[1]
    if want < 1.0:
        assert major < 1.0 or major <= span
        return
    assert major >= want
    assert off1 == 0 and off2 == 0
    steps = xaxis._TIME_STEPS
    if major <= steps[-1]:
        assert major in steps
        assert all(s < want for s in steps if s < major)  # smallest step that is wide enough
    else:
        assert major % 86400 == 0 and major / 10 < want * 10
    ratio = major / minor
    assert abs(ratio - round(ratio)) < 1e-9 and ratio >= 2  # minor ticks line up with major ticks


def test_relative_spacing_fallback(axis):
    axis.set_format(FMT_RELATIVE)
    assert axis.tickSpacing(0.0, 0.5, 600)[0][0] < 1.0
    assert axis.tickSpacing(5.0, 5.0, 600) == []  # pyqtgraph: no ticks for an empty span


@pytest.mark.parametrize("tz", ["UTC", "Europe/Berlin", "America/New_York", "Asia/Kolkata"])
@pytest.mark.parametrize("span,size", [(100.0, 600), (4000.0, 900), (3 * 86400.0, 600), (40.0, 300)])
def test_absolute_ticks_on_round_local_times(axis, set_tz, tz, span, size):
    set_tz(tz)
    t_ref = 1753704336.123456  # 2025-07-28 12:05:36.123456 UTC
    axis.set_format(FMT_ABSOLUTE, t_ref)
    levels = axis.tickValues(0.0, span, size)
    assert levels
    for spacing, values in levels:
        assert values
        for v in values:
            assert -1e-6 <= v <= span + 1e-6
            local = dt.datetime.fromtimestamp(t_ref + v)
            secs = local.hour * 3600 + local.minute * 60 + local.second + local.microsecond / 1e6
            r = secs % spacing
            assert min(r, spacing - r) < 1e-3, (tz, spacing, v, local)
    major = levels[0][0]
    labels = axis.tickStrings(levels[0][1], 1.0, major)
    for v, text in zip(levels[0][1], labels):
        local = dt.datetime.fromtimestamp(round(t_ref + v))
        exp = local.strftime("%Y-%m-%d") if major >= 86400 else local.strftime("%H:%M:%S")
        assert text == exp


def test_relative_tick_strings(axis):
    axis.set_format(FMT_RELATIVE)
    assert axis.tickStrings([0, 30, 60, 90, 3600, 86400 + 61], 1.0, 30) == [
        "00:00", "00:30", "01:00", "01:30", "01:00:00", "24:01:01"]
    assert axis.tickStrings([0.1, 0.25, -1.5], 1.0, 0.05) == ["00:00.10", "00:00.25", "-00:01.50"]
    assert axis.tickStrings([0.2, 0.4], 1.0, 0.2) == ["00:00.2", "00:00.4"]
    assert axis.tickStrings([0.0001], 1.0, 0.0001) == ["00:00.0001"]
    assert axis.tickStrings([1e-9], 1.0, 1e-12) == ["00:00.000000001"]  # at most 9 decimals


def test_absolute_tick_strings(axis, set_tz):
    set_tz("Europe/Berlin")
    t_ref = 1753704336.0  # 2025-07-28 14:05:36 local (CEST)
    axis.set_format(FMT_ABSOLUTE, t_ref)
    assert axis.tickStrings([0.0, 24.0], 1.0, 1.0) == ["14:05:36", "14:06:00"]
    assert axis.tickStrings([0.1, 0.2, 0.3], 1.0, 0.1) == ["14:05:36.1", "14:05:36.2", "14:05:36.3"]
    assert axis.tickStrings([0.0], 1.0, 86400.0) == ["2025-07-28"]
    assert axis.tickStrings([0.0], 1.0, 7 * 86400.0) == ["2025-07-28"]
    set_tz("UTC")
    axis.set_format(FMT_ABSOLUTE, t_ref)
    assert axis.tickStrings([0.0], 1.0, 1.0) == ["12:05:36"]
    assert axis.tickStrings([1e30], 1.0, 1.0) == [""]


def test_absolute_day_ticks_across_dst(axis, set_tz):
    """Day ticks stay on local midnight after a DST change (no repeated date labels)."""
    set_tz("Europe/Berlin")
    t_ref = dt.datetime(2025, 10, 23, 0, 0).timestamp()  # local midnight, DST ends on 2025-10-26
    axis.set_format(FMT_ABSOLUTE, t_ref)
    spacing, values = axis.tickValues(0.0, 6 * 86400.0, 700)[0]
    assert spacing == 86400.0
    labels = axis.tickStrings(values, 1.0, spacing)
    assert len(set(labels)) == len(labels), labels
    hours = [dt.datetime.fromtimestamp(t_ref + v).strftime("%H:%M") for v in values]
    assert set(hours) == {"00:00"}, hours


def test_absolute_tick_string_rounds_into_next_second(axis, set_tz):
    """A fraction that rounds up (x.9996 s, 3 decimals) must carry into the seconds."""
    set_tz("UTC")
    axis.set_format(FMT_ABSOLUTE, 1000.0006)
    # 1000.9996 s after 1970-01-01 00:00:00 UTC is 00:16:40.9996, shown with 3 decimals.
    assert axis.tickStrings([0.999], 1.0, 0.001) == ["00:16:41.000"]


# -- formatting: floats and values --------------------------------------------------------------

def test_format_float_round_trip(rng):
    bits = rng.integers(0, 2**63, 20000, dtype=np.uint64) | (rng.integers(0, 2, 20000, dtype=np.uint64) << 63)
    vals = bits.view(np.float64)
    vals = vals[np.isfinite(vals)]
    extra = np.array([0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 1 / 3,
                      4400.000001, 1e16, 123456789012345678.0])
    for v in np.concatenate([vals, extra]):
        text = format_float(float(v))
        back = float(text)
        assert struct.pack("<d", back) == struct.pack("<d", float(v)), (v, text)
        assert text == repr(float(v))


def test_format_float_special():
    assert format_float(float("nan")) == "NaN"
    assert format_float(float("inf")) == "Inf"
    assert format_float(float("-inf")) == "-Inf"
    assert format_float(-0.0) == "-0.0"
    assert format_value(np.float64("nan")) == "NaN"
    assert format_value(np.float32("-inf")) == "-Inf"


def test_format_value_float32_round_trip(rng):
    vals = rng.integers(0, 2**32, 5000, dtype=np.uint64).astype(np.uint32).view(np.float32)
    vals = vals[np.isfinite(vals)]
    for v in vals:
        text = format_value(v)
        assert np.float32(text).tobytes() == v.tobytes(), (v, text)


def test_format_value_float32_shortest():
    """Shortest text that reads back to the same float32 bits (module rule), in Python repr style."""
    bad = []
    for v in (0.1, 1.1, 3.3, 1e-7, 123.456, 16777217.0, -2.5e38, 0.5, 3.0):
        f = np.float32(v)
        text = format_value(f)
        assert np.float32(text) == f
        if text != repr(float(str(f))):
            bad.append((str(f), text))
    assert bad == [], f"float32 shown with float64 digits: {bad}"


def test_format_value_complex64_shortest():
    assert format_value(np.complex64(0.1 + 0.2j)) == "0.1+0.2j"


def test_format_value_types():
    assert format_value(True) == "true" and format_value(False) == "false"
    assert format_value(np.bool_(True)) == "true" and format_value(np.bool_(False)) == "false"
    assert format_value(0) == "0" and format_value(-7) == "-7"
    assert format_value(np.uint64(2**64 - 1)) == "18446744073709551615"
    assert format_value(np.int64(-2**63)) == "-9223372036854775808"
    assert format_value(np.int8(-128)) == "-128"
    assert format_value(2**80) == str(2**80)
    assert format_value(0.1) == "0.1" and format_value(np.float64(1 / 3)) == repr(1 / 3)
    assert format_value("text, with comma") == "text, with comma"
    assert format_value(b"abc") == "abc"
    assert format_value(b"\xff\xfeok") == "��ok"
    assert format_value(None) == "None"


def test_format_value_complex():
    assert format_value(complex(1.5, -2.0)) == "1.5-2.0j"
    assert format_value(np.complex128(0.1 + 0.2j)) == "0.1+0.2j"
    assert format_value(complex(-1.0, float("nan"))) == "-1.0+NaNj"
    assert format_value(complex(float("inf"), float("-inf"))) == "Inf-Infj"
    assert format_value(complex(0.0, 0.0)) == "0.0+0.0j"


# -- formatting: time ------------------------------------------------------------------------------

def test_format_datetime64_local_with_offset(set_tz):
    t = np.datetime64("2024-01-02T03:04:05.123456", "us")
    set_tz("UTC")
    assert format_datetime64(t) == "2024-01-02 03:04:05.123456+00:00"
    set_tz("Europe/Berlin")
    assert format_datetime64(t) == "2024-01-02 04:04:05.123456+01:00"
    assert format_datetime64(np.datetime64("2024-07-01T12:00:00", "us")) == "2024-07-01 14:00:00.000000+02:00"
    set_tz("America/New_York")
    assert format_datetime64(t) == "2024-01-01 22:04:05.123456-05:00"
    set_tz("Asia/Kolkata")
    assert format_datetime64(t) == "2024-01-02 08:34:05.123456+05:30"


def test_format_datetime64_units_and_edges(set_tz):
    set_tz("UTC")
    assert format_datetime64(np.datetime64("NaT")) == "NaT"
    assert format_datetime64(np.datetime64("2024-01-02T03:04:05.123456789", "ns")) == "2024-01-02 03:04:05.123456+00:00"
    assert format_datetime64(np.datetime64("2024-01-02", "D")) == "2024-01-02 00:00:00.000000+00:00"
    assert format_datetime64(np.datetime64("1904-01-01T00:00:00", "us")) == "1904-01-01 00:00:00.000000+00:00"
    assert format_datetime64(np.datetime64("1969-12-31T23:59:59.5", "us")) == "1969-12-31 23:59:59.500000+00:00"
    far = np.datetime64("20000-01-01T00:00:00", "us")
    assert format_datetime64(far) == str(far)  # outside datetime range: plain numpy text
    assert format_value(np.datetime64("2024-01-02T03:04:05", "us")) == "2024-01-02 03:04:05.000000+00:00"


def test_format_python_datetime(set_tz):
    set_tz("Europe/Berlin")
    naive = dt.datetime(2024, 7, 1, 12, 0, 0)  # naive = UTC
    assert format_value(naive) == "2024-07-01 14:00:00.000000+02:00"
    aware = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=-3)))
    assert format_value(aware) == "2024-01-01 16:00:00.000000+01:00"


@pytest.mark.parametrize("seconds,dec,text", [
    (0.0, 0, "00:00"), (0.0, 3, "00:00.000"), (59.0, 0, "00:59"), (60.0, 0, "01:00"),
    (61.25, 2, "01:01.25"), (3599.0, 0, "59:59"), (3600.0, 0, "01:00:00"), (3661.5, 1, "01:01:01.5"),
    (59.9996, 3, "01:00.000"), (3599.99996, 4, "01:00:00.0000"), (-1.5, 1, "-00:01.5"), (-3600.0, 0, "-01:00:00"),
    (360000.0, 0, "100:00:00"), (1e-9, 9, "00:00.000000001"), (0.123456789012, 12, "00:00.123456789012"),
    (4427.45, 2, "01:13:47.45"), (0.05, 2, "00:00.05"),
])
def test_format_duration(seconds, dec, text):
    assert format_duration(seconds, dec) == text


def test_format_duration_non_finite():
    assert format_duration(float("nan"), 3) == "NaN"
    assert format_duration(float("inf"), 0) == "Inf"
    assert format_duration(float("-inf"), 2) == "-Inf"


def test_format_duration_matches_rounded_seconds(rng):
    for s in rng.uniform(-1e5, 1e5, 2000):
        for dec in (0, 1, 3, 6):
            text = format_duration(float(s), dec)
            neg = text.startswith("-")
            parts = text.lstrip("-").split(":")
            val = sum(float(p) * 60 ** k for k, p in enumerate(reversed(parts)))
            assert abs((-val if neg else val) - s) <= 0.5 * 10.0 ** -dec + 1e-9 * abs(s)
            assert len(parts[-1].split(".")[0]) == 2
            if dec:
                assert len(parts[-1].split(".")[1]) == dec


def test_format_si():
    assert format_si(1234567.891) == "1.23457e+06"
    assert format_si(0.000123456789, 3) == "0.000123"
    assert format_si(4400.0012345, 7) == "4400.001"
    assert format_si(float("nan")) == "NaN"
    assert format_si(float("-inf")) == "-Inf"
