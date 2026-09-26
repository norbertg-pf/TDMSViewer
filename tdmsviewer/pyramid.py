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
        """Root mean square of the samples (Inf if a sample is +/-Inf)."""
        if self.n == 0:
            return float("nan")
        if math.isinf(self.min) or math.isinf(self.max):
            return math.inf
        v = self.mean * self.mean + self.m2 / self.n
        return math.nan if math.isnan(v) else math.sqrt(max(0.0, v))

    @property
    def p2p(self) -> float:
        return self.max - self.min


EMPTY = Stats(0, math.nan, math.nan, math.nan, 0.0)


# -- primitive reductions ---------------------------------------------------
#
# Each bucket stores (n, min, max, pivot, delta, m2):
#   pivot  first finite sample of the bucket (exact value)
#   delta  mean - pivot (small: exact to eps * |delta|)
#   m2     sum of squared deviations from the mean
# Buckets are merged relative to the pivot of the first bucket. Nearby
# pivots subtract exactly (Sterbenz), so a quiet signal keeps full
# precision anywhere in the channel.

_SCRATCH = threading.local()
_SCRATCH_ELEMS = 1 << 19  # 4 MB work buffer: stays in cache, no page faults


def _scratch(n: int) -> np.ndarray:
    """Thread-local work buffer of at least n elements (n <= _SCRATCH_ELEMS)."""
    buf = getattr(_SCRATCH, "buf", None)
    if buf is None:
        buf = _SCRATCH.buf = np.empty(_SCRATCH_ELEMS)
    return buf[:n]


def _first_finite(y: np.ndarray, default=0.0):
    """First finite value of y."""
    for c in range(0, y.size, 4096):
        seg = y[c:c + 4096]
        ok = np.flatnonzero(np.isfinite(seg))
        if ok.size:
            return float(seg[ok[0]])
    return default


def _row_stats_nan(row: np.ndarray):
    """(count, pivot, delta, m2) of one row, NaN ignored, in bounded chunks."""
    piv = _first_finite(row, None)
    if piv is None:
        if np.isnan(row).all():
            return 0, 0.0, math.nan, 0.0
        piv = 0.0  # only +/-Inf values: a pivot of 0 keeps Inf means correct
    n = 0
    s = 0.0
    step = _SCRATCH_ELEMS
    for c in range(0, row.size, step):
        seg = row[c:c + step]
        ok = ~np.isnan(seg)  # count on raw values: Inf - Inf would look like NaN
        d = np.subtract(seg, piv, out=_scratch(seg.size))
        n += int(ok.sum())
        s += float(np.where(ok, d, 0.0).sum())
    mu = s / n
    m2 = 0.0
    for c in range(0, row.size, step):
        seg = row[c:c + step]
        ok = ~np.isnan(seg)
        d = np.subtract(seg, piv, out=_scratch(seg.size))
        d -= mu
        m2 += float(np.where(ok, d * d, 0.0).sum())
    return n, piv, mu, m2


def _rows_stats(body: np.ndarray):
    """Per-row (count, min, max, pivot, delta, m2) of a 2-D float64 array."""
    with np.errstate(invalid="ignore", over="ignore"):  # Inf data: Inf - Inf
        rows, k = body.shape
        mins = np.fmin.reduce(body, axis=1)
        maxs = np.fmax.reduce(body, axis=1)
        piv = body[:, 0].copy()
        delta = np.empty(rows)
        m2 = np.empty(rows)
        if k <= _SCRATCH_ELEMS:
            step = _SCRATCH_ELEMS // k
            for r in range(0, rows, step):
                blk = body[r:r + step]
                d = _scratch(blk.size).reshape(blk.shape)
                np.subtract(blk, piv[r:r + step, None], out=d)
                dl = np.add.reduce(d, axis=1) / k
                delta[r:r + step] = dl
                np.subtract(d, dl[:, None], out=d)
                m2[r:r + step] = np.einsum("ij,ij->i", d, d)
            bad = ~(np.isfinite(delta) & np.isfinite(piv))
        else:  # few very long rows (raw statistics): chunk the columns
            bad = np.ones(rows, dtype=bool)
        counts = np.full(rows, float(k))
        for r in np.flatnonzero(bad):
            # NaN or Inf inside: recompute these rows without NaN.
            counts[r], piv[r], delta[r], m2[r] = _row_stats_nan(body[r])
        return counts, mins, maxs, piv, delta, m2


def _raw_parts(y: np.ndarray):
    """(count, min, max, pivot, delta, m2) of a 1-D array as 1-element arrays."""
    if y.size == 0:
        z = np.zeros(1)
        return z, np.full(1, np.nan), np.full(1, np.nan), z.copy(), z.copy(), z.copy()
    return _rows_stats(y.reshape(1, -1))


def raw_stats(y: np.ndarray) -> Stats:
    """Exact statistics of a 1-D float64 array (NaN ignored)."""
    if y.size == 0:
        return EMPTY
    c, mn, mx, pv, dl, m2 = _raw_parts(y)
    n = int(c[0])
    if n == 0:
        return EMPTY
    return Stats(n, float(mn[0]), float(mx[0]), float(pv[0] + dl[0]), float(m2[0]))


def merge_stats(parts: list[Stats]) -> Stats:
    """Combine statistics of disjoint ranges (Chan et al.)."""
    parts = [p for p in parts if p.n > 0]
    if not parts:
        return EMPTY
    if len(parts) == 1:
        return parts[0]
    col = lambda f: np.array([f(p) for p in parts], dtype=np.float64)  # noqa: E731
    return _merge_arrays(col(lambda p: p.n), col(lambda p: p.min), col(lambda p: p.max),
                         col(lambda p: p.mean), np.zeros(len(parts)), col(lambda p: p.m2))


def _merge_arrays(n, mn, mx, pv, dl, m2) -> Stats:
    """Merge bucket statistics into one Stats (relative to the first pivot)."""
    ok = n > 0
    if not ok.any():
        return EMPTY
    n, mn, mx, pv, dl, m2 = n[ok], mn[ok], mx[ok], pv[ok], dl[ok], m2[ok]
    total = float(n.sum())
    with np.errstate(invalid="ignore", over="ignore"):  # Inf data
        ref = pv[0]
        e = (pv - ref) + dl  # bucket means relative to ref
        em = float((n * e).sum() / total)
        d = e - em
        m2t = float(m2.sum() + (n * d * d).sum())
        mean = float(ref + em)
    return Stats(int(total), float(np.fmin.reduce(mn)), float(np.fmax.reduce(mx)), mean, m2t)


def _group_merge(n, mn, mx, pv, dl, m2, g: int):
    """Merge each run of g consecutive buckets. Length must be a multiple of g."""
    k = n.size // g
    n2 = n.reshape(k, g)
    pv2 = pv.reshape(k, g)
    tot = n2.sum(axis=1)
    first = np.argmax(n2 > 0, axis=1)  # first non-empty child of each group
    ref = np.where(tot > 0, pv2[np.arange(k), first], 0.0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        e = np.where(n2 > 0, (pv2 - ref[:, None]) + dl.reshape(k, g), 0.0)
        em = (n2 * e).sum(axis=1) / tot
        d = np.where(n2 > 0, e - em[:, None], 0.0)
        m2g = m2.reshape(k, g).sum(axis=1) + (n2 * d * d).sum(axis=1)
    return (tot,
            np.fmin.reduce(mn.reshape(k, g), axis=1),
            np.fmax.reduce(mx.reshape(k, g), axis=1),
            ref, np.where(tot > 0, em, np.nan), np.where(tot > 0, m2g, 0.0))


def _rows_minmax(body: np.ndarray):
    """NaN-ignoring min and max of each row.

    Short rows (<= 32, power of two) use pairwise halving: about 1 ns per
    sample instead of 6 to 11 ns for numpy's short-row reduce.
    """
    k = body.shape[1]
    if k > 32 or k & (k - 1) or not body.flags.c_contiguous:
        return np.fmin.reduce(body, axis=1), np.fmax.reduce(body, axis=1)
    mn = mx = body.reshape(-1)
    while k > 1:
        # Neighbours (2i, 2i+1) are always in the same bucket (power-of-two size).
        mn = np.fmin(mn[0::2], mn[1::2])
        mx = np.fmax(mx[0::2], mx[1::2])
        k //= 2
    return mn, mx


def raw_minmax(y: np.ndarray, first: int, bucket: int, with_missing: bool = False):
    """Min/max per bucket of raw samples y (index of y[0] is `first`).

    Bucket edges are aligned to multiples of `bucket` so the picture does
    not shimmer while panning. Returns (centers, mins, maxs), plus a bool
    array "bucket has NaN samples" if with_missing.
    """
    n = y.size
    if n == 0:
        e = np.empty(0)
        return (e, e, e, np.empty(0, dtype=bool)) if with_missing else (e, e, e)
    last = first + n
    a = min(last, -(-first // bucket) * bucket)  # first aligned edge
    b = max(a, (last // bucket) * bucket)  # last aligned edge
    centers, mins, maxs, miss = [], [], [], []
    if a > first:
        seg = y[: a - first]
        centers.append(np.array([(first + a) / 2.0]))
        mins.append(np.array([np.fmin.reduce(seg)]))
        maxs.append(np.array([np.fmax.reduce(seg)]))
        if with_missing:
            miss.append(np.array([bool(np.isnan(seg).any())]))
    if b > a:
        body = y[a - first: b - first].reshape(-1, bucket)
        k = body.shape[0]
        centers.append(a + bucket * (np.arange(k) + 0.5))
        mn, mx = _rows_minmax(body)
        mins.append(mn)
        maxs.append(mx)
        if with_missing:
            miss.append(np.isnan(body).any(axis=1))
    if last > b:
        seg = y[b - first:]
        centers.append(np.array([(b + last) / 2.0]))
        mins.append(np.array([np.fmin.reduce(seg)]))
        maxs.append(np.array([np.fmax.reduce(seg)]))
        if with_missing:
            miss.append(np.array([bool(np.isnan(seg).any())]))
    out = (np.concatenate(centers), np.concatenate(mins), np.concatenate(maxs))
    return out + (np.concatenate(miss),) if with_missing else out


def interleave(centers: np.ndarray, mins: np.ndarray, maxs: np.ndarray, missing=None):
    """Return (x_index, y) with a min and a max point per bucket.

    missing: bool per bucket (bucket has NaN samples). A NaN point after
    such a bucket breaks the line, so dropouts stay visible when zoomed out.
    """
    if missing is None or not np.any(missing):
        x = np.repeat(centers, 2)
        y = np.empty(mins.size * 2)
        y[0::2] = mins
        y[1::2] = maxs
        return x, y
    per = np.where(missing, 3, 2)
    start = np.concatenate(([0], np.cumsum(per)[:-1]))
    total = int(per.sum())
    x = np.repeat(centers, per)
    y = np.full(total, np.nan)
    y[start] = mins
    y[start + 1] = maxs
    return x, y


def floor_pow2(v: float) -> int:
    """Largest power of two <= v (1 for v < 2)."""
    if not v >= 2:
        return 1
    return 1 << (int(v).bit_length() - 1)


# -- pyramid ------------------------------------------------------------------

class _Level:
    __slots__ = ("size", "n", "mn", "mx", "pv", "dl", "m2")

    def __init__(self, size, n, mn, mx, pv, dl, m2):
        self.size = size
        self.n, self.mn, self.mx, self.pv, self.dl, self.m2 = n, mn, mx, pv, dl, m2

    @property
    def mean(self) -> np.ndarray:
        """Bucket means (NaN for empty buckets)."""
        with np.errstate(invalid="ignore"):
            return np.where(self.n > 0, self.pv + self.dl, np.nan)

    def columns(self):
        return (self.n, self.mn, self.mx, self.pv, self.dl, self.m2)


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
                          np.zeros(nb), np.zeros(nb), np.zeros(nb))
        self.levels: list[_Level] = [self._l0]
        self._carry = np.empty(0)
        self._appended = 0
        self.covered = 0  # samples covered by finished level-0 buckets
        self.complete = self.length == 0

    @property
    def nbytes(self) -> int:
        return sum(6 * lv.n.nbytes for lv in self.levels)

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
        base = self.base
        j0 = self.covered // base
        nfull = block.size // base
        tail = block[nfull * base:]
        lv = self._l0
        if nfull:
            sl = slice(j0, j0 + nfull)
            vals = _rows_stats(block[: nfull * base].reshape(nfull, base))
            for col, v in zip(lv.columns(), vals):
                col[sl] = v
        if self._appended == self.length:
            if tail.size:
                j = j0 + nfull
                for col, v in zip(lv.columns(), _raw_parts(tail)):
                    col[j] = v[0]
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
            arrs = list(lv.columns())
            if pad:
                fills = (0.0, np.nan, np.nan, 0.0, 0.0, 0.0)
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
        pieces, levels = self._level_pieces(a // base, b // base)
        cols = [[], [], [], [], [], []]  # n, min, max, pivot, delta, m2
        if a > i0:
            for col, arr in zip(cols, _raw_parts(read_raw(i0, a))):
                col.append(arr)
        for lvl, p, q in pieces:
            for col, arr in zip(cols, levels[lvl].columns()):
                col.append(arr[p:q])
        if i1 > b:
            for col, arr in zip(cols, _raw_parts(read_raw(b, i1))):
                col.append(arr)
        return _merge_arrays(*(np.concatenate(c) for c in cols))

    def minmax(self, w0: int, w1: int, bucket: int, read_raw, with_missing: bool = False):
        """Envelope of [w0, w1) with aligned buckets of `bucket` samples.

        bucket must be a power of two >= base. Only the covered prefix is
        returned. Returns (centers, mins, maxs), plus a bool array "bucket
        has NaN samples" if with_missing.
        """
        w0 = max(0, w0)
        w1 = min(self.covered, w1)
        e = np.empty(0)
        if w1 <= w0:
            return (e, e, e, np.empty(0, dtype=bool)) if with_missing else (e, e, e)
        a = min(w1, -(-w0 // bucket) * bucket)
        b = max(a, (w1 // bucket) * bucket)
        centers, mins, maxs, miss = [], [], [], []

        def edge(p, q):
            s = self.stats(p, q, read_raw)
            centers.append(np.array([(p + q) / 2.0]))
            mins.append(np.array([s.min if s else np.nan]))
            maxs.append(np.array([s.max if s else np.nan]))
            miss.append(np.array([bool(s is not None and s.n < q - p)]))

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
            if with_missing:
                miss.append(lv.n[p:q].reshape(k, g).sum(axis=1) < bucket)
        if w1 > b:
            edge(b, w1)
        out = (np.concatenate(centers), np.concatenate(mins), np.concatenate(maxs))
        return out + (np.concatenate(miss),) if with_missing else out
