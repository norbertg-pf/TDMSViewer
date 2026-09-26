"""Tests for tdmsviewer.pyramid (min/max decimation and range statistics).

All results are compared with brute force numpy (long double for
mean and M2).
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pytest

from tdmsviewer import pyramid as pyr
from tdmsviewer.pyramid import EMPTY, Pyramid, Stats, floor_pow2, interleave, merge_stats, raw_minmax, raw_stats

REL = 1e-12


# -- reference implementations ------------------------------------------------------------

def ref_stats(y):
    """(n, min, max, mean, m2) ignoring NaN, mean/m2 in long double."""
    y = np.asarray(y, dtype=np.float64)
    ok = y[~np.isnan(y)]
    if ok.size == 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    if not np.isfinite(ok).all():  # mean/m2 are not checked for Inf data
        return ok.size, float(ok.min()), float(ok.max()), math.nan, math.nan
    yl = ok.astype(np.longdouble)
    mean = yl.sum() / ok.size
    m2 = ((yl - mean) ** 2).sum()
    return ok.size, float(ok.min()), float(ok.max()), float(mean), float(m2)


def same_float(a, b) -> bool:
    """Equal, or both NaN."""
    return (math.isnan(a) and math.isnan(b)) or a == b


def assert_stats(s: Stats, y, rel=REL, msg=""):
    """Exact n/min/max; mean and std within `rel` of the long double reference."""
    n, mn, mx, mean, m2 = ref_stats(y)
    assert s is not None, msg
    assert s.n == n, msg
    assert same_float(s.min, mn) and same_float(s.max, mx), f"{msg}: min/max {s.min, s.max} != {mn, mx}"
    if n == 0:
        assert math.isnan(s.mean), msg
        return
    finite = np.isfinite(np.asarray(y, dtype=np.float64)[~np.isnan(np.asarray(y, dtype=np.float64))]).all()
    if not finite:
        with np.errstate(invalid="ignore"):
            exp_mean = float(np.nanmean(np.asarray(y, dtype=np.float64)))
        assert same_float(s.mean, exp_mean), f"{msg}: mean {s.mean} != {exp_mean}"
        return
    rms = math.sqrt(mean * mean + m2 / n)
    assert abs(s.mean - mean) <= rel * max(abs(mean), rms), f"{msg}: mean {s.mean!r} != {mean!r}"
    if n > 1:
        std = math.sqrt(m2 / (n - 1))
        assert abs(s.std - std) <= rel * std + 1e-300, f"{msg}: std {s.std!r} != {std!r} (rel {abs(s.std - std) / std:.2e})"
    assert abs(s.m2 - m2) <= rel * max(m2, 1e-300) * 2 + 1e-300, msg


def brute_minmax(y, w0, w1, bucket):
    """Buckets with edges at multiples of `bucket`, cut at w0 and w1."""
    edges = sorted({w0, w1, *range(-(-w0 // bucket) * bucket, w1, bucket)})
    centers, mins, maxs = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = y[lo:hi]
        centers.append((lo + hi) / 2.0)
        mins.append(np.fmin.reduce(seg))
        maxs.append(np.fmax.reduce(seg))
    return np.array(centers), np.array(mins, dtype=float), np.array(maxs, dtype=float)


def assert_same_arrays(got, exp, msg=""):
    """Compare (centers, mins, maxs) tuples; NaN == NaN."""
    for g, e, name in zip(got, exp, ("centers", "mins", "maxs")):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(e), err_msg=f"{msg} {name}")


class RawReader:
    """read_raw callback that counts samples per call."""

    def __init__(self, y):
        self.y = y
        self.calls = []

    def __call__(self, a, b):
        assert 0 <= a <= b <= self.y.size
        self.calls.append(b - a)
        return self.y[a:b]


def build(y, base, rng, max_block=None):
    """Complete pyramid of y, appended in random blocks (also empty ones)."""
    p = Pyramid(y.size, base)
    i = 0
    max_block = max_block or max(2, 3 * base)
    while i < y.size:
        k = int(rng.integers(0, max_block))
        p.append(y[i:i + k])
        i += k
    p.append(y[i:])
    assert p.complete and p.covered == y.size
    return p


def data(n, rng, kind="normal"):
    """Test signal: normal, offset (4400 +/- 1e-3), nan, inf or steps."""
    if kind == "normal":
        return rng.standard_normal(n) * 3.0 + 1.5
    if kind == "offset":
        return 4400.0 + 1e-3 * rng.standard_normal(n)
    if kind == "nan":
        y = rng.standard_normal(n)
        y[rng.random(n) < 0.1] = np.nan
        if n > 600:
            y[100:600] = np.nan  # whole NaN buckets
        return y
    if kind == "inf":
        y = rng.standard_normal(n)
        if n > 10:
            y[n // 3] = np.inf
            y[n // 2] = -np.inf if n > 50 else np.inf
        return y
    if kind == "steps":
        return np.repeat(rng.integers(-5, 5, max(1, n // 7 + 1)).astype(float), 7)[:n]
    raise ValueError(kind)


def random_ranges(n, rng, count=60, base=256):
    """Sample ranges [i0, i1): edges, bucket joints, out-of-range and random ones."""
    out = [(0, n), (0, 1), (max(0, n - 1), n), (0, 0), (n, n), (-5, n + 5), (n // 2, n // 2 + 1),
           (base - 1, base + 1), (base, 2 * base), (1, n - 1), (0, base * 8), (base * 8 - 1, base * 64 + 1)]
    for _ in range(count):
        a = int(rng.integers(0, max(1, n)))
        b = int(rng.integers(a, n + 1)) if rng.random() < 0.6 else a + int(rng.integers(0, 4 * base))
        out.append((a, b))
    return out


SHAPES = ["0", "1", "base-1", "base", "base+1", "2base+3", "multi"]


def length_for(shape: str, base: int) -> int:
    """Channel length for a shape name; "multi" gives 4 pyramid levels."""
    return {"0": 0, "1": 1, "base-1": base - 1, "base": base, "base+1": base + 1, "2base+3": 2 * base + 3,
            "multi": base * pyr.FACTOR ** 2 * 70 + 5}[shape]


# -- primitives -------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2, 7, 1000, 100_003])
@pytest.mark.parametrize("kind", ["normal", "offset", "nan", "inf", "steps"])
def test_raw_stats(n, kind, rng):
    y = data(n, rng, kind)
    assert_stats(raw_stats(y), y, msg=f"n={n} {kind}")


def test_raw_stats_all_nan():
    s = raw_stats(np.full(1000, np.nan))
    assert s.n == 0 and math.isnan(s.min) and math.isnan(s.max) and math.isnan(s.mean)
    assert raw_stats(np.empty(0)) is EMPTY


def test_stats_properties():
    s = raw_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s.n == 4 and s.min == 1.0 and s.max == 4.0 and s.p2p == 3.0
    assert s.mean == 2.5
    assert s.std == pytest.approx(np.std([1, 2, 3, 4], ddof=1), rel=1e-15)
    assert s.rms == pytest.approx(math.sqrt(7.5), rel=1e-15)
    one = raw_stats(np.array([5.0]))
    assert one.n == 1 and math.isnan(one.std) and one.rms == 5.0 and one.p2p == 0.0
    assert EMPTY.n == 0 and math.isnan(EMPTY.std) and math.isnan(EMPTY.rms) and math.isnan(EMPTY.p2p)


@pytest.mark.parametrize("kind", ["normal", "offset", "nan"])
def test_merge_stats_equals_whole(kind, rng):
    y = data(10_000, rng, kind)
    cuts = np.sort(rng.integers(0, y.size, 12))
    parts = [raw_stats(p) for p in np.split(y, cuts)]
    # Part means are rounded to float64 (ulp(4400) ~ 1e-12), so merging them
    # cannot reach 1e-12 on std for the offset data.
    assert_stats(merge_stats(parts), y, rel=1e-10 if kind == "offset" else REL, msg=kind)
    assert merge_stats([]) is EMPTY
    assert merge_stats([EMPTY, EMPTY]) is EMPTY
    single = raw_stats(y[:10])
    assert merge_stats([EMPTY, single, EMPTY]) is single


@pytest.mark.parametrize("bucket", [1, 2, 3, 8, 256, 1000])
@pytest.mark.parametrize("kind", ["normal", "nan", "inf"])
def test_raw_minmax_vs_brute_force(bucket, kind, rng):
    y = data(5000, rng, kind)
    for w0, w1 in [(0, 5000), (1, 4999), (0, 1), (17, 17 + bucket), (bucket, 3 * bucket), (4000, 4001),
                   (123, 3210)]:
        w1 = min(w1, y.size)
        got = raw_minmax(y[w0:w1], w0, bucket)
        assert_same_arrays(got, brute_minmax(y, w0, w1, bucket), f"[{w0},{w1}) bucket={bucket}")
    assert all(a.size == 0 for a in raw_minmax(np.empty(0), 10, bucket))


def test_interleave():
    x, y = interleave(np.array([0.5, 1.5]), np.array([1.0, -2.0]), np.array([3.0, 4.0]))
    assert x.tolist() == [0.5, 0.5, 1.5, 1.5]
    assert y.tolist() == [1.0, 3.0, -2.0, 4.0]


@pytest.mark.parametrize("v", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.999, 4.0, 1023.0, 1024.0, 1e6, 12345.678])
def test_floor_pow2(v):
    got = floor_pow2(v)
    if v < 1:
        assert got == 1
    else:
        assert got <= v < 2 * got and got & (got - 1) == 0


def test_floor_pow2_near_powers():
    """Largest power of two <= v, also just below a power of two (up to 2**62)."""
    bad = []
    for k in range(1, 63):
        for v in (2**k - 1, 2**k, 2**k + 1):
            fv = float(v)
            got = floor_pow2(fv)
            if got & (got - 1) or not got <= fv < 2 * got:
                bad.append((fv, got))
    assert bad == [], f"floor_pow2(v) > v for {len(bad)} values, first: {bad[:3]}"


# -- pyramid: build ---------------------------------------------------------------------------

def test_pyramid_rejects_bad_base():
    for base in (0, 1, 3, 100, -4):
        with pytest.raises(ValueError):
            Pyramid(10, base)


def test_empty_pyramid():
    p = Pyramid(0)
    assert p.complete and p.covered == 0
    p.append(np.arange(5.0))  # ignored
    assert p.stats(0, 10, lambda a, b: np.empty(0)) is EMPTY
    assert all(a.size == 0 for a in p.minmax(0, 10, 256, lambda a, b: np.empty(0)))


def test_append_too_much():
    p = Pyramid(100, 16)
    p.append(np.zeros(60))
    with pytest.raises(ValueError):
        p.append(np.zeros(41))


def test_append_after_complete_is_ignored():
    p = Pyramid(10, 4)
    p.append(np.arange(10.0))
    p.append(np.arange(100.0))
    assert p.stats(0, 10, lambda a, b: np.arange(10.0)[a:b]).max == 9.0


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("base", [2, 16, 256])
def test_levels_structure(shape, base, rng):
    n = length_for(shape, base)
    y = data(n, rng, "nan")
    p = build(y, base, rng)
    assert p.levels[0].size == base
    nb = -(-n // base)
    assert p.levels[0].n.size == nb
    for k, lv in enumerate(p.levels):
        assert lv.size == base * pyr.FACTOR ** k
        if k:
            prev = p.levels[k - 1]
            assert lv.n.size == -(-prev.n.size // pyr.FACTOR)
            assert lv.n.sum() == prev.n.sum()
    if len(p.levels) > 1:
        assert p.levels[-1].n.size <= pyr.TOP_BUCKETS < p.levels[-2].n.size
    if shape == "multi":
        assert len(p.levels) == 4
    assert p.levels[0].n.sum() == np.count_nonzero(~np.isnan(y))
    assert p.nbytes == sum(6 * lv.n.nbytes for lv in p.levels)


@pytest.mark.parametrize("block", [1, 7, 255, 256, 257, 10_000])
def test_block_sizes_give_same_pyramid(block, rng):
    y = data(40_000, rng, "nan")
    ref = Pyramid(y.size, 64)
    ref.append(y)
    p = Pyramid(y.size, 64)
    for i in range(0, y.size, block):
        p.append(y[i:i + block])
    assert len(p.levels) == len(ref.levels)
    for a, b in zip(p.levels, ref.levels):
        for name in ("n", "mn", "mx"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        np.testing.assert_allclose(a.mean[a.n > 0], b.mean[b.n > 0], rtol=1e-13)


def test_append_converts_to_float64(rng):
    y = rng.integers(-1000, 1000, 5000).astype(np.int16)
    p = Pyramid(y.size, 16)
    p.append(y[:1234])
    p.append(y[1234:].astype(np.float32))
    yf = y.astype(np.float64)
    assert_stats(p.stats(0, y.size, lambda a, b: yf[a:b]), yf)


# -- pyramid: statistics ---------------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("base", [2, 16, 256])
@pytest.mark.parametrize("kind", ["normal", "nan", "inf", "steps"])
def test_stats_vs_brute_force(shape, base, kind, rng):
    n = length_for(shape, base)
    y = data(n, rng, kind)
    p = build(y, base, rng, max_block=max(2, 3 * base) if n < 100_000 else 50_000)
    for i0, i1 in random_ranges(n, rng, base=base):
        rr = RawReader(y)
        s = p.stats(i0, i1, rr)
        lo, hi = max(0, i0), min(n, i1)
        if hi <= lo:
            assert s is EMPTY or s.n == 0
            continue
        assert_stats(s, y[lo:hi], msg=f"n={n} base={base} [{i0},{i1})")
        assert sum(rr.calls) <= 2 * (base - 1), rr.calls


@pytest.mark.parametrize("base", [2, 256])
@pytest.mark.parametrize("n", [5_000, 1_000_003])
def test_stats_small_noise_on_large_offset(n, base, rng):
    """4400 +/- 1e-3: mean and std within 1e-12 relative (raw_stats reaches ~1e-16)."""
    y = data(n, rng, "offset")
    p = build(y, base, rng, max_block=100_000)
    ranges = [(0, n), (1, n - 1), (5, 5 + 3 * base + 1), (7, 9 * base + 3), (n // 3, n // 2),
              (0, 64 * base), (base, base * 65)]
    for i0, i1 in ranges:
        i1 = min(i1, n)
        assert_stats(p.stats(i0, i1, lambda a, b: y[a:b]), y[i0:i1], msg=f"base={base} [{i0},{i1})")


def test_big_pyramid(rng):
    n = 3_000_017
    y = data(n, rng, "nan")
    p = build(y, 256, rng, max_block=1 << 20)
    assert len(p.levels) >= 4
    for i0, i1 in random_ranges(n, rng, count=25):
        rr = RawReader(y)
        assert_stats(p.stats(i0, i1, rr), y[max(0, i0):min(n, i1)], msg=f"[{i0},{i1})")
        assert sum(rr.calls) <= 2 * 255
    for bucket in (256, 1 << 12, 1 << 16, 1 << 22):
        for w0, w1 in [(0, n), (12345, 2_345_678), (n - 1000, n)]:
            got = p.minmax(w0, w1, bucket, lambda a, b: y[a:b])
            assert_same_arrays(got, raw_minmax(y[w0:w1], w0, bucket), f"bucket={bucket}")


def test_stats_exact_min_max_with_spikes(rng):
    y = rng.standard_normal(300_000) * 1e-6
    spikes = rng.integers(0, y.size, 25)
    y[spikes] = rng.standard_normal(25) * 1e6
    p = build(y, 256, rng, max_block=50_000)
    for i0, i1 in random_ranges(y.size, rng, 200):
        s = p.stats(i0, i1, lambda a, b: y[a:b])
        lo, hi = max(0, i0), min(y.size, i1)
        if hi > lo:
            assert s.min == y[lo:hi].min() and s.max == y[lo:hi].max()


# -- pyramid: min/max envelope ----------------------------------------------------------------

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("base", [2, 16, 256])
@pytest.mark.parametrize("kind", ["normal", "nan", "inf"])
def test_minmax_equals_raw_minmax(shape, base, kind, rng):
    n = length_for(shape, base)
    y = data(n, rng, kind)
    p = build(y, base, rng, max_block=max(2, 3 * base) if n < 100_000 else 50_000)
    buckets = [base * 2 ** k for k in range(0, 14) if base * 2 ** k <= max(base, 4 * n)]
    for w0, w1 in random_ranges(n, rng, count=15, base=base):
        for bucket in buckets:
            got = p.minmax(w0, w1, bucket, lambda a, b: y[a:b])
            lo, hi = max(0, w0), min(n, w1)
            if hi <= lo:
                assert all(a.size == 0 for a in got)
                continue
            exp = raw_minmax(y[lo:hi], lo, bucket)
            assert_same_arrays(got, exp, f"n={n} base={base} [{w0},{w1}) bucket={bucket}")
            if (hi - lo) // bucket < 300:
                assert_same_arrays(got, brute_minmax(y, lo, hi, bucket), "brute")


def test_minmax_keeps_single_spike(rng):
    y = np.zeros(2_000_000)
    y[1_234_567] = 1e-3
    y[777] = -5.0
    p = build(y, 256, rng, max_block=300_000)
    c, mn, mx = p.minmax(0, y.size, 1 << 16, lambda a, b: y[a:b])
    assert mx.max() == 1e-3 and mn.min() == -5.0
    k = int(np.argmax(mx))
    assert c[k] - (1 << 15) <= 1_234_567 < c[k] + (1 << 15)


# -- partial pyramid (still loading) -----------------------------------------------------------

@pytest.mark.parametrize("base", [2, 16, 256])
def test_partial_pyramid_covered_prefix(base, rng):
    n = 300 * base + 17
    y = data(n, rng, "nan")
    p = Pyramid(n, base)
    appended = 0
    while not p.complete:
        k = int(rng.integers(0, 5 * base))
        p.append(y[appended:appended + k])
        appended = min(n, appended + k)
        cov = p.covered
        if p.complete:
            assert cov == n
        else:
            assert cov == (appended // base) * base
        for i0, i1 in random_ranges(n, rng, count=6, base=base):
            s = p.stats(i0, i1, lambda a, b: y[a:b])
            lo, hi = max(0, i0), min(n, i1)
            if hi <= lo:
                assert s is EMPTY or s.n == 0
            elif hi > cov:
                assert s is None
            else:
                assert_stats(s, y[lo:hi], msg=f"covered={cov} [{i0},{i1})")
            bucket = base * 2 ** int(rng.integers(0, 6))
            got = p.minmax(i0, i1, bucket, lambda a, b: y[a:b])
            hi2 = min(hi, cov)
            if hi2 <= lo:
                assert all(a.size == 0 for a in got)
            else:
                assert_same_arrays(got, brute_minmax(y, lo, hi2, bucket), f"covered={cov}")


def test_concurrent_reader_sees_consistent_prefix(rng):
    n = 2_000_000
    y = data(n, rng, "normal")
    p = Pyramid(n, 256)
    errors = []
    done = threading.Event()

    def reader():
        r = np.random.default_rng(5)
        while not done.is_set():
            cov = p.covered
            if cov == 0:
                continue
            a = int(r.integers(0, cov))
            b = int(r.integers(a, cov + 1))
            s = p.stats(a, b, lambda u, v: y[u:v])
            if b > a and (s is None or s.n != b - a or s.min != y[a:b].min() or s.max != y[a:b].max()):
                errors.append((a, b, cov))
            c, mn, mx = p.minmax(0, n, 1 << 14, lambda u, v: y[u:v])
            if c.size and c[-1] > p.covered:
                errors.append(("minmax beyond covered", c[-1]))

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(0, n, 12_345):
            p.append(y[i:i + 12_345])
    finally:
        done.set()
        t.join()
    assert p.complete
    assert errors == []


def test_raw_stats_scratch_buffer_stays_small():
    """The work buffer is meant to stay small (4 MB); one big raw_stats call must not keep 100+ MB."""
    result = {}

    def work():
        raw_stats(np.ones(4_000_000))
        buf = getattr(pyr._SCRATCH, "buf", None)
        result["elems"] = 0 if buf is None else buf.size

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert result["elems"] <= pyr._SCRATCH_ELEMS
