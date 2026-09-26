"""Peak-preserving decimation and exact range statistics.

Summary
    The plot never hides a spike. Each drawn bucket keeps the exact
    minimum and maximum of its samples. A multi-level pyramid of bucket
    statistics (count, min, max, mean, M2) gives the envelope for any
    zoom level, and exact statistics for any sample range, in
    O(log n) time.

Numerics
    Mean and M2 (sum of squared deviations) are stored per bucket and
    merged with the Chan et al. parallel formula. This stays exact for a
    small noise on a large offset (for example 4400 A +/- 1 mA), where
    the naive sum-of-squares method fails. NaN samples are ignored.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

FACTOR = 8  # bucket size ratio between pyramid levels (power of 2)
TOP_BUCKETS = 64  # stop adding levels below this bucket count


@dataclass(slots=True)
class Stats:
    """Statistics of one sample range. Values are NaN if n == 0."""

    n: int
    min: float
    max: float
    mean: float
    m2: float

    @property
    def std(self) -> float:
        """Sample standard deviation (n - 1)."""
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else float("nan")

    @property
    def rms(self) -> float:
        """Root mean square of the samples."""
        if self.n == 0:
            return float("nan")
        return math.sqrt(max(0.0, self.mean * self.mean + self.m2 / self.n))

    @property
    def p2p(self) -> float:
        return self.max - self.min


EMPTY = Stats(0, math.nan, math.nan, math.nan, 0.0)


# -- primitive reductions ---------------------------------------------------

_SCRATCH = threading.local()
_SCRATCH_ELEMS = 1 << 19  # 4 MB work buffer: stays in cache, no page faults


def _scratch(n: int) -> np.ndarray:
    """Thread-local work buffer of at least n elements (n <= _SCRATCH_ELEMS)."""
    buf = getattr(_SCRATCH, "buf", None)
    if buf is None:
        buf = _SCRATCH.buf = np.empty(_SCRATCH_ELEMS)
    return buf[:n]


def _row_stats_nan(row: np.ndarray, ref: float):
    """(count, sum of (x - ref), M2) of one row, NaN ignored, in bounded chunks."""
    n = 0
    s = 0.0
    step = _SCRATCH_ELEMS
    with np.errstate(invalid="ignore"):
        for c in range(0, row.size, step):
            d = np.subtract(row[c:c + step], ref, out=_scratch(min(step, row.size - c)))
            ok = ~np.isnan(d)
            n += int(ok.sum())
            s += float(np.where(ok, d, 0.0).sum())
        if n == 0:
            return 0, math.nan, 0.0
        mu = s / n
        m2 = 0.0
        for c in range(0, row.size, step):
            d = np.subtract(row[c:c + step], ref, out=_scratch(min(step, row.size - c)))
            d -= mu
            m2 += float(np.where(np.isnan(d), 0.0, d * d).sum())
    return n, mu, m2


def _rows_stats(body: np.ndarray, ref: float):
    with np.errstate(invalid="ignore", over="ignore"):  # Inf data: Inf - Inf
        return _rows_stats_impl(body, ref)


def _rows_stats_impl(body: np.ndarray, ref: float):
    """Per-row (count, min, max, mean - ref, m2) of a 2-D float64 array.

    Means are relative to `ref` (one value per channel). This keeps full
    precision for a small signal on a large offset.
    """
    rows, k = body.shape
    mins = np.fmin.reduce(body, axis=1)
    maxs = np.fmax.reduce(body, axis=1)
    sums = np.empty(rows)
    m2 = np.empty(rows)
    if k <= _SCRATCH_ELEMS:
        step = _SCRATCH_ELEMS // k
        for r in range(0, rows, step):
            blk = body[r:r + step]
            d = _scratch(blk.size).reshape(blk.shape)
            np.subtract(blk, ref, out=d)
            sm = np.add.reduce(d, axis=1)
            sums[r:r + step] = sm
            np.subtract(d, (sm / k)[:, None], out=d)
            m2[r:r + step] = np.einsum("ij,ij->i", d, d)
    else:  # few very long rows (raw statistics): chunk the columns
        for r in range(rows):
            _, mu, m2[r] = _row_stats_nan(body[r], ref)
            sums[r] = mu * k
    means = sums / k
    counts = np.full(rows, float(k))
    bad = ~np.isfinite(sums)
    if bad.any():
        # NaN or Inf inside: recompute these rows without NaN.
        for r in np.flatnonzero(bad):
            c, mu, mm = _row_stats_nan(body[r], ref)
            counts[r], means[r], m2[r] = c, mu, mm
    return counts, mins, maxs, means, m2


def _first_finite(y: np.ndarray, default=0.0):
    """First finite value of y: the reference for relative means."""
    for c in range(0, y.size, 4096):
        seg = y[c:c + 4096]
        ok = np.flatnonzero(np.isfinite(seg))
        if ok.size:
            return float(seg[ok[0]])
    return default


def _raw_rel(y: np.ndarray, ref: float):
    """(count, min, max, mean - ref, m2) of a 1-D array as 1-element arrays."""
    if y.size == 0:
        z = np.zeros(1)
        return z, np.full(1, np.nan), np.full(1, np.nan), z.copy(), z.copy()
    return _rows_stats(y.reshape(1, -1), ref)


def raw_stats(y: np.ndarray) -> Stats:
    """Exact statistics of a 1-D float64 array (NaN ignored)."""
    if y.size == 0:
        return EMPTY
    ref = _first_finite(y)
    c, mn, mx, mu, m2 = _raw_rel(y, ref)
    n = int(c[0])
    if n == 0:
        return EMPTY
    return Stats(n, float(mn[0]), float(mx[0]), ref + float(mu[0]), float(m2[0]))


def merge_stats(parts: list[Stats]) -> Stats:
    """Combine statistics of disjoint ranges (Chan et al.)."""
    parts = [p for p in parts if p.n > 0]
    if not parts:
        return EMPTY
    if len(parts) == 1:
        return parts[0]
    n = np.array([p.n for p in parts], dtype=np.float64)
    mu = np.array([p.mean for p in parts])
    m2 = np.array([p.m2 for p in parts])
    return _merge_arrays(n, np.array([p.min for p in parts]), np.array([p.max for p in parts]), mu, m2)


def _merge_arrays(n, mn, mx, mu, m2) -> Stats:
    ok = n > 0
    if not ok.any():
        return EMPTY
    n, mn, mx, mu, m2 = n[ok], mn[ok], mx[ok], mu[ok], m2[ok]
    total = float(n.sum())
    with np.errstate(invalid="ignore", over="ignore"):  # Inf data
        mean = float((n * mu).sum() / total)
        d = mu - mean
        m2t = float(m2.sum() + (n * d * d).sum())
    return Stats(int(total), float(np.fmin.reduce(mn)), float(np.fmax.reduce(mx)), mean, m2t)


def _group_merge(n, mn, mx, mu, m2, g: int):
    """Merge each run of g consecutive buckets. Length must be a multiple of g."""
    k = n.size // g
    n2 = n.reshape(k, g)
    tot = n2.sum(axis=1)
    mu0 = np.where(n2 > 0, mu.reshape(k, g), 0.0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        mean = (n2 * mu0).sum(axis=1) / tot
        d = np.where(n2 > 0, mu0 - mean[:, None], 0.0)
        m2g = m2.reshape(k, g).sum(axis=1) + (n2 * d * d).sum(axis=1)
    return (tot,
            np.fmin.reduce(mn.reshape(k, g), axis=1),
            np.fmax.reduce(mx.reshape(k, g), axis=1),
            mean, np.where(tot > 0, m2g, 0.0))


def raw_minmax(y: np.ndarray, first: int, bucket: int):
    """Min/max per bucket of raw samples y (index of y[0] is `first`).

    Bucket edges are aligned to multiples of `bucket` so the picture does
    not shimmer while panning. Returns (centers, mins, maxs).
    """
    n = y.size
    if n == 0:
        e = np.empty(0)
        return e, e, e
    last = first + n
    a = min(last, -(-first // bucket) * bucket)  # first aligned edge
    b = max(a, (last // bucket) * bucket)  # last aligned edge
    centers, mins, maxs = [], [], []
    if a > first:
        seg = y[: a - first]
        centers.append(np.array([(first + a) / 2.0]))
        mins.append(np.array([np.fmin.reduce(seg)]))
        maxs.append(np.array([np.fmax.reduce(seg)]))
    if b > a:
        body = y[a - first: b - first].reshape(-1, bucket)
        k = body.shape[0]
        centers.append(a + bucket * (np.arange(k) + 0.5))
        mins.append(np.fmin.reduce(body, axis=1))
        maxs.append(np.fmax.reduce(body, axis=1))
    if last > b:
        seg = y[b - first:]
        centers.append(np.array([(b + last) / 2.0]))
        mins.append(np.array([np.fmin.reduce(seg)]))
        maxs.append(np.array([np.fmax.reduce(seg)]))
    return np.concatenate(centers), np.concatenate(mins), np.concatenate(maxs)


def interleave(centers: np.ndarray, mins: np.ndarray, maxs: np.ndarray):
    """Return (x_index, y) with a min and a max point per bucket."""
    x = np.repeat(centers, 2)
    y = np.empty(mins.size * 2)
    y[0::2] = mins
    y[1::2] = maxs
    return x, y


def floor_pow2(v: float) -> int:
    """Largest power of two <= v (1 for v < 2)."""
    if not v >= 2:
        return 1
    return 1 << (int(v).bit_length() - 1)


# -- pyramid ------------------------------------------------------------------

class _Level:
    __slots__ = ("size", "n", "mn", "mx", "mu", "m2")

    def __init__(self, size, n, mn, mx, mu, m2):
        self.size = size
        self.n, self.mn, self.mx, self.mu, self.m2 = n, mn, mx, mu, m2


class Pyramid:
    """Multi-level bucket statistics of one channel (float64 view).

    Build: call append() with consecutive blocks until all `length`
    samples are in. Readers may query while building; only the covered
    prefix [0, covered) is valid.

    Thread model: one writer thread. Readers in other threads see a
    consistent prefix because `covered` is updated after the data.
    """

    def __init__(self, length: int, base: int = 256):
        if base < 2 or base & (base - 1):
            raise ValueError("base must be a power of two")
        self.length = int(length)
        self.base = base
        nb = -(-self.length // base)
        self._l0 = _Level(base, np.zeros(nb), np.full(nb, np.nan), np.full(nb, np.nan),
                          np.zeros(nb), np.zeros(nb))
        self.levels: list[_Level] = [self._l0]
        self._carry = np.empty(0)
        self._appended = 0
        self.covered = 0  # samples covered by finished level-0 buckets
        self.complete = self.length == 0
        self.ref = None  # bucket means are stored relative to this value

    @property
    def nbytes(self) -> int:
        return sum(5 * lv.n.nbytes for lv in self.levels)

    def append(self, block: np.ndarray) -> None:
        """Add the next samples (float64, 1-D)."""
        if self.complete:
            return
        block = np.asarray(block, dtype=np.float64)
        if self._carry.size:
            block = np.concatenate((self._carry, block))
        self._appended += block.size - self._carry.size
        if self._appended > self.length:
            raise ValueError("more samples than channel length")
        if self.ref is None:
            self.ref = _first_finite(block, None)
        ref = 0.0 if self.ref is None else self.ref  # all-NaN so far: ref unused
        base = self.base
        j0 = self.covered // base
        nfull = block.size // base
        tail = block[nfull * base:]
        lv = self._l0
        if nfull:
            c, mn, mx, mu, m2 = _rows_stats(block[: nfull * base].reshape(nfull, base), ref)
            sl = slice(j0, j0 + nfull)
            lv.n[sl], lv.mn[sl], lv.mx[sl], lv.mu[sl], lv.m2[sl] = c, mn, mx, mu, m2
        if self._appended == self.length:
            if self.ref is None:
                self.ref = 0.0
            if tail.size:
                c, mn, mx, mu, m2 = _raw_rel(tail, self.ref)
                j = j0 + nfull
                lv.n[j], lv.mn[j], lv.mx[j], lv.mu[j], lv.m2[j] = c[0], mn[0], mx[0], mu[0], m2[0]
            self._carry = np.empty(0)
            self.covered = self.length
            self._build_levels()
            self.complete = True
        else:
            self._carry = tail.copy()
            self.covered = (j0 + nfull) * base

    def _build_levels(self) -> None:
        lv = self._l0
        levels = [lv]
        while lv.n.size > TOP_BUCKETS:
            k = -(-lv.n.size // FACTOR)
            pad = k * FACTOR - lv.n.size
            arrs = [lv.n, lv.mn, lv.mx, lv.mu, lv.m2]
            if pad:
                fills = (0.0, np.nan, np.nan, 0.0, 0.0)
                arrs = [np.concatenate((a, np.full(pad, f))) for a, f in zip(arrs, fills)]
            lv = _Level(lv.size * FACTOR, *_group_merge(*arrs, FACTOR))
            levels.append(lv)
        self.levels = levels

    # -- queries --------------------------------------------------------------

    def _level_pieces(self, j0: int, j1: int):
        """Split level-0 bucket range [j0, j1) into (level, a, b) pieces."""
        pieces = []
        levels = self.levels if self.complete else [self._l0]
        lvl = 0
        while True:
            if lvl + 1 >= len(levels):
                if j1 > j0:
                    pieces.append((lvl, j0, j1))
                break
            k0 = -(-j0 // FACTOR)
            k1 = j1 // FACTOR
            if k0 >= k1:
                if j1 > j0:
                    pieces.append((lvl, j0, j1))
                break
            if k0 * FACTOR > j0:
                pieces.append((lvl, j0, k0 * FACTOR))
            if j1 > k1 * FACTOR:
                pieces.append((lvl, k1 * FACTOR, j1))
            j0, j1 = k0, k1
            lvl += 1
        return pieces, levels

    def stats(self, i0: int, i1: int, read_raw) -> Stats | None:
        """Exact statistics of samples [i0, i1).

        read_raw(a, b) must return float64 samples [a, b). It is called for
        at most 2 * (base - 1) samples. Returns None if the range is not
        covered yet.
        """
        i0 = max(0, i0)
        i1 = min(self.length, i1)
        if i1 <= i0:
            return EMPTY
        if i1 > self.covered:
            return None
        base = self.base
        a = -(-i0 // base) * base
        b = (i1 // base) * base
        if b <= a:
            return raw_stats(read_raw(i0, i1))
        ref = self.ref if self.ref is not None else 0.0
        pieces, levels = self._level_pieces(a // base, b // base)
        cols = [[], [], [], [], []]  # n, min, max, mean - ref, m2
        for lvl, p, q in pieces:
            lv = levels[lvl]
            for col, arr in zip(cols, (lv.n, lv.mn, lv.mx, lv.mu, lv.m2)):
                col.append(arr[p:q])
        for p, q in ((i0, a), (b, i1)):
            if q > p:
                for col, arr in zip(cols, _raw_rel(read_raw(p, q), ref)):
                    col.append(arr)
        st = _merge_arrays(*(np.concatenate(c) for c in cols))
        if st.n == 0:
            return st
        return Stats(st.n, st.min, st.max, ref + st.mean, st.m2)

    def minmax(self, w0: int, w1: int, bucket: int, read_raw):
        """Envelope of [w0, w1) with aligned buckets of `bucket` samples.

        bucket must be a power of two >= base. Only the covered prefix is
        returned. Returns (centers, mins, maxs).
        """
        w0 = max(0, w0)
        w1 = min(self.covered, w1)
        e = np.empty(0)
        if w1 <= w0:
            return e, e, e
        a = min(w1, -(-w0 // bucket) * bucket)
        b = max(a, (w1 // bucket) * bucket)
        centers, mins, maxs = [], [], []

        def edge(p, q):
            s = self.stats(p, q, read_raw)
            centers.append(np.array([(p + q) / 2.0]))
            mins.append(np.array([s.min if s else np.nan]))
            maxs.append(np.array([s.max if s else np.nan]))

        if a > w0:
            edge(w0, a)
        if b > a:
            levels = self.levels if self.complete else [self._l0]
            lv = levels[0]
            for cand in levels:
                if cand.size <= bucket:
                    lv = cand
            g = bucket // lv.size
            p, q = a // lv.size, b // lv.size
            k = (q - p) // g
            mn = np.fmin.reduce(lv.mn[p:q].reshape(k, g), axis=1)
            mx = np.fmax.reduce(lv.mx[p:q].reshape(k, g), axis=1)
            centers.append(a + bucket * (np.arange(k) + 0.5))
            mins.append(mn)
            maxs.append(mx)
        if w1 > b:
            edge(b, w1)
        return np.concatenate(centers), np.concatenate(mins), np.concatenate(maxs)
