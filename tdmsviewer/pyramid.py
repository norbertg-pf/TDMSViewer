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


def _rows_stats(body: np.ndarray):
    """Per-row (count, min, max, mean, m2) of a 2-D float64 array."""
    rows, k = body.shape
    mins = np.fmin.reduce(body, axis=1)
    maxs = np.fmax.reduce(body, axis=1)
    sums = np.add.reduce(body, axis=1)
    means = sums / k
    m2 = np.empty(rows)
    step = max(1, _SCRATCH_ELEMS // k)
    buf = getattr(_SCRATCH, "buf", None)
    if buf is None or buf.size < step * k:
        buf = _SCRATCH.buf = np.empty(max(_SCRATCH_ELEMS, step * k))
    for r in range(0, rows, step):
        blk = body[r:r + step]
        dev = buf[: blk.size].reshape(blk.shape)
        np.subtract(blk, means[r:r + step, None], out=dev)
        m2[r:r + step] = np.einsum("ij,ij->i", dev, dev)
    counts = np.full(rows, float(k))
    bad = ~np.isfinite(sums)
    if bad.any():
        # NaN or Inf inside: recompute these rows without NaN.
        rows = body[bad]
        ok = ~np.isnan(rows)
        c = ok.sum(axis=1).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(ok, rows, 0.0).sum(axis=1)
            mu = s / c
            d = np.where(ok, rows - mu[:, None], 0.0)
            m2b = np.einsum("ij,ij->i", d, d)
        counts[bad] = c
        means[bad] = mu
        m2[bad] = np.where(c > 0, m2b, 0.0)
    return counts, mins, maxs, means, m2


def raw_stats(y: np.ndarray) -> Stats:
    """Exact statistics of a 1-D float64 array (NaN ignored)."""
    if y.size == 0:
        return EMPTY
    c, mn, mx, mu, m2 = _rows_stats(y.reshape(1, -1))
    n = int(c[0])
    if n == 0:
        return EMPTY
    return Stats(n, float(mn[0]), float(mx[0]), float(mu[0]), float(m2[0]))


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
    with np.errstate(invalid="ignore", divide="ignore"):
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
    """Largest power of two <= v (v >= 1)."""
    return 1 << max(0, int(math.floor(math.log2(max(1.0, v)))))


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
        base = self.base
        j0 = self.covered // base
        nfull = block.size // base
        tail = block[nfull * base:]
        lv = self._l0
        if nfull:
            c, mn, mx, mu, m2 = _rows_stats(block[: nfull * base].reshape(nfull, base))
            sl = slice(j0, j0 + nfull)
            lv.n[sl], lv.mn[sl], lv.mx[sl], lv.mu[sl], lv.m2[sl] = c, mn, mx, mu, m2
        if self._appended == self.length:
            if tail.size:
                s = raw_stats(tail)
                j = j0 + nfull
                lv.n[j], lv.mn[j], lv.mx[j], lv.mu[j], lv.m2[j] = s.n, s.min, s.max, s.mean, s.m2
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
        parts = []
        if a > i0:
            parts.append(raw_stats(read_raw(i0, a)))
        if i1 > b:
            parts.append(raw_stats(read_raw(b, i1)))
        pieces, levels = self._level_pieces(a // base, b // base)
        ns, mns, mxs, mus, m2s = [], [], [], [], []
        for lvl, p, q in pieces:
            lv = levels[lvl]
            ns.append(lv.n[p:q]); mns.append(lv.mn[p:q]); mxs.append(lv.mx[p:q])
            mus.append(lv.mu[p:q]); m2s.append(lv.m2[p:q])
        for s in parts:
            ns.append(np.array([float(s.n)])); mns.append(np.array([s.min])); mxs.append(np.array([s.max]))
            mus.append(np.array([s.mean])); m2s.append(np.array([s.m2]))
        return _merge_arrays(np.concatenate(ns), np.concatenate(mns), np.concatenate(mxs),
                             np.concatenate(mus), np.concatenate(m2s))

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
