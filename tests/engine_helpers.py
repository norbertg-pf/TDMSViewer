"""Helpers for the DataEngine tests.

Contents
    - TDMS writers: segmented files (npTDMS writer) and interleaved files
      (small raw writer, npTDMS cannot write interleaved data).
    - Recorder / Driver: record engine signals and wait for answers.
    - Reference math: brute force envelope, statistics and float64 views.
"""

from __future__ import annotations

import math
import struct

import numpy as np
from nptdms import ChannelObject, GroupObject, RootObject, TdmsWriter

SIGNALS = ("opened", "openFailed", "progress", "channelsUpdated", "plotReady",
           "tableReady", "statsReady", "xReady", "exportDone", "message")

TIMEOUT_MS = 30_000


# -- TDMS writers ----------------------------------------------------------------

def write_segments(path, segments, file_props=None, group_props=None, index_file=False):
    """Write a TDMS file with npTDMS, one TDMS segment per list entry.

    segments: list of segments; each segment is a list of
    (group, channel, data, props) tuples. Props are written only the
    first time a channel appears.
    """
    group_props = group_props or {}
    seen = set()
    with TdmsWriter(str(path), index_file=index_file) as w:
        for k, seg in enumerate(segments):
            objs = []
            if k == 0:
                objs.append(RootObject(file_props or {}))
            for group, name, data, props in seg:
                if group not in seen:
                    seen.add(group)
                    objs.append(GroupObject(group, group_props.get(group, {})))
                key = (group, name)
                objs.append(ChannelObject(group, name, data, None if key in seen else props))
                seen.add(key)
            w.write_segment(objs)


def split_channels(channels, cuts):
    """Split channels into segments at the given sample indices.

    channels: list of (group, name, data, props). Returns a segment list
    for write_segments.
    """
    segs = []
    edges = [0, *cuts, None]
    for a, b in zip(edges[:-1], edges[1:]):
        segs.append([(g, n, d[a:b], p) for g, n, d, p in channels])
    return segs


_TYPE_CODES = {
    np.dtype("int8"): 1, np.dtype("int16"): 2, np.dtype("int32"): 3, np.dtype("int64"): 4,
    np.dtype("uint8"): 5, np.dtype("uint16"): 6, np.dtype("uint32"): 7, np.dtype("uint64"): 8,
    np.dtype("float32"): 9, np.dtype("float64"): 10, np.dtype("bool"): 0x21,
    np.dtype("complex64"): 0x08000C, np.dtype("complex128"): 0x10000D,
}
_TOC_META, _TOC_RAW, _TOC_NEWOBJ, _TOC_INTERLEAVED = 1 << 1, 1 << 3, 1 << 2, 1 << 5
_EPOCH_1904 = np.datetime64("1904-01-01T00:00:00", "us")


def _tdms_str(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _tdms_props(props: dict) -> bytes:
    out = [struct.pack("<I", len(props))]
    for key, val in props.items():
        out.append(_tdms_str(key))
        if isinstance(val, str):
            out.append(struct.pack("<I", 0x20) + _tdms_str(val))
        elif isinstance(val, bool):
            out.append(struct.pack("<IB", 0x21, int(val)))
        elif isinstance(val, int):
            out.append(struct.pack("<Ii", 3, val))
        elif isinstance(val, float):
            out.append(struct.pack("<Id", 10, val))
        else:
            raise TypeError(f"unsupported property {key}={val!r}")
    return b"".join(out)


def _column_bytes(a: np.ndarray) -> np.ndarray:
    """Return an (n, itemsize) uint8 view of little-endian TDMS values."""
    if a.dtype.kind == "M":
        us = (a.astype("datetime64[us]") - _EPOCH_1904).astype(np.int64)
        sec = np.floor_divide(us, 1_000_000)
        rem = (us - sec * 1_000_000).astype(np.uint64)
        # rem * 2**64 / 1e6 plus a tiny margin, so npTDMS reads back the same us.
        frac = rem * np.uint64(18446744073709) + (rem * np.uint64(551616)) // np.uint64(10**6)
        frac = frac + np.uint64(1 << 16)
        rec = np.empty(a.size, dtype=[("f", "<u8"), ("s", "<i8")])
        rec["f"], rec["s"] = frac, sec
        return rec.view(np.uint8).reshape(a.size, 16)
    le = a.astype(a.dtype.newbyteorder("<"))
    return le.view(np.uint8).reshape(a.size, a.dtype.itemsize)


def _type_code(dtype: np.dtype) -> int:
    if dtype.kind == "M":
        return 0x44
    return _TYPE_CODES[np.dtype(dtype).newbyteorder("=")]


def write_interleaved(path, group, channels, rows_per_segment, file_props=None, group_props=None):
    """Write interleaved TDMS segments (all channels have the same length).

    channels: list of (name, data, props). Data may be numeric or
    datetime64. Every segment has a new object list.
    """
    n = len(channels[0][1])
    assert all(len(d) == n for _, d, _ in channels)
    cols = [_column_bytes(np.asarray(d)) for _, d, _ in channels]
    with open(path, "wb") as fh:
        for k, a in enumerate(range(0, n, rows_per_segment)):
            b = min(n, a + rows_per_segment)
            meta = []
            objs = 0
            if k == 0:
                meta.append(_tdms_str("/") + b"\xff\xff\xff\xff" + _tdms_props(file_props or {}))
                meta.append(_tdms_str(f"/'{group}'") + b"\xff\xff\xff\xff" + _tdms_props(group_props or {}))
                objs += 2
            for (name, data, props), col in zip(channels, cols):
                index = struct.pack("<IIIQ", 20, _type_code(np.asarray(data).dtype), 1, b - a)
                meta.append(_tdms_str(f"/'{group}'/'{name}'") + index
                            + _tdms_props(props if k == 0 else {}))
                objs += 1
            meta_bytes = struct.pack("<I", objs) + b"".join(meta)
            raw = np.hstack([c[a:b] for c in cols]).tobytes()
            toc = _TOC_META | _TOC_RAW | _TOC_NEWOBJ | _TOC_INTERLEAVED
            lead = b"TDSm" + struct.pack("<IIQQ", toc, 4713, len(meta_bytes) + len(raw), len(meta_bytes))
            fh.write(lead + meta_bytes + raw)


# -- signal recording ----------------------------------------------------------

class Recorder:
    """Record every engine signal in arrival order (GUI thread)."""

    def __init__(self, eng):
        self.events: list[tuple[str, tuple]] = []
        for name in SIGNALS:
            getattr(eng, name).connect(lambda *a, _n=name: self.events.append((_n, a)))

    def of(self, name, gen=None):
        return [a for n, a in self.events if n == name and (gen is None or a[0] == gen)]

    def find(self, name, pred):
        for n, a in self.events:
            if n == name and pred(*a):
                return a
        return None

    def errors(self):
        """Messages that report an internal engine error."""
        return [a for a in self.of("message") if "Internal error" in a[1]]


class Driver:
    """Send engine requests and wait for the matching answers."""

    def __init__(self, qtbot, eng, timeout=TIMEOUT_MS):
        self.qtbot = qtbot
        self.eng = eng
        self.rec = Recorder(eng)
        self.timeout = timeout
        self._seq = 0

    def seq(self) -> int:
        self._seq += 1
        return self._seq

    def wait(self, name, pred):
        """Wait until a matching signal was recorded; return its args."""
        box = []

        def ok():
            a = self.rec.find(name, pred)
            if a is not None:
                box.append(a)
                return True
            return False

        self.qtbot.waitUntil(ok, timeout=self.timeout)
        return box[0]

    def open(self, path, loaded=True):
        """Open a file. Returns (gen, FileModel). Raises on openFailed."""
        gen = self.eng.open(str(path))

        def done():
            return bool(self.rec.find("opened", lambda g, *_: g == gen)
                        or self.rec.find("openFailed", lambda g, *_: g == gen))

        self.qtbot.waitUntil(done, timeout=self.timeout)
        bad = self.rec.find("openFailed", lambda g, *_: g == gen)
        if bad is not None:
            raise AssertionError(f"openFailed: {bad[1]}")
        model = self.rec.find("opened", lambda g, *_: g == gen)[1]
        if loaded:
            self.wait_loaded(gen)
        return gen, model

    def wait_loaded(self, gen):
        return self.wait("progress", lambda g, f, t: g == gen and f >= 1.0 and t.startswith("Loaded"))

    def plot(self, items, xa, xb, pixels, seq=None):
        from tdmsviewer.engine import PlotRequest

        seq = self.seq() if seq is None else seq
        gen = self.eng.generation
        self.eng.request_plot(PlotRequest(seq, list(items), xa, xb, pixels))
        return self.wait("plotReady", lambda g, s, o: g == gen and s == seq)[2]

    def stats(self, items, xa, xb, cursors=()):
        from tdmsviewer.engine import StatsRequest

        seq = self.seq()
        gen = self.eng.generation
        self.eng.request_stats(StatsRequest(seq, list(items), xa, xb, list(cursors)))
        return self.wait("statsReady", lambda g, s, o: g == gen and s == seq)[2]

    def table(self, cids, i0, i1):
        from tdmsviewer.engine import TableRequest

        seq = self.seq()
        gen = self.eng.generation
        self.eng.request_table(TableRequest(seq, list(cids), i0, i1))
        return self.wait("tableReady", lambda g, s, o: g == gen and s == seq)[2]

    def xmap(self, cid):
        from tdmsviewer.engine import XRequest

        seq = self.seq()
        gen = self.eng.generation
        self.eng.request_x(XRequest(seq, cid))
        return self.wait("xReady", lambda g, s, o: g == gen and s == seq)[2]

    def export(self, path, items, xa, xb, header):
        from tdmsviewer.engine import ExportRequest

        seq = self.seq()
        gen = self.eng.generation
        self.eng.request_export(ExportRequest(seq, str(path), list(items), xa, xb, list(header)))
        return self.wait("exportDone", lambda g, s, m: g == gen and s == seq)[2]


# -- reference math --------------------------------------------------------------

def time_to_seconds(a: np.ndarray) -> np.ndarray:
    """Timestamps as float seconds after the first valid timestamp."""
    a = a.astype("datetime64[us]")
    ok = a[~np.isnat(a)]
    t0 = ok[0] if ok.size else np.datetime64(0, "us")
    return (a - t0) / np.timedelta64(1, "us") / 1e6


def as_f64(a: np.ndarray) -> np.ndarray:
    """Plot values of native channel values (reference, independent of engine)."""
    if a.dtype.kind == "M":
        return time_to_seconds(a)
    if a.dtype.kind == "c":
        return np.abs(a).astype(np.float64)
    return a.astype(np.float64)


def nan_equal(a, b) -> bool:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def same_values(a, b) -> bool:
    """Exact equality of native channel values (NaN equal to NaN)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "OUS" or b.dtype.kind in "OUS":
        return list(a) == list(b)
    if a.dtype.kind == "M" or b.dtype.kind == "M":
        return a.dtype == b.dtype and bool(np.array_equal(a, b))
    if a.dtype.kind == "c" or b.dtype.kind == "c":
        return nan_equal(a.real, b.real) and nan_equal(a.imag, b.imag)
    return nan_equal(a, b)


def bucket_edges(centers_idx: np.ndarray, i0: int) -> np.ndarray:
    """Bucket edges from bucket centers (index units), first edge i0.

    Each bucket [p, q) has center (p + q) / 2, so q = 2 * center - p.
    """
    edges = [float(i0)]
    for c in centers_idx:
        edges.append(2.0 * c - edges[-1])
    return np.array(edges)


def check_envelope(x, y, samples, i0, i1, x_to_index, complete=True):
    """Assert that (x, y) is an exact min/max envelope of samples[i0:i1].

    x, y: plot output (two points per bucket: min then max).
    samples: float64 values of the whole channel.
    x_to_index: maps plot x back to fractional sample index.
    Returns the bucket edges (int array).
    """
    assert x.size == y.size and x.size % 2 == 0, (x.size, y.size)
    assert np.array_equal(x[0::2], x[1::2]), "min and max of a bucket share one x"
    ci = x_to_index(x[0::2])
    half = np.round(2.0 * ci) / 2.0
    assert np.allclose(ci, half, rtol=0, atol=1e-6), "bucket centers must be on half samples"
    edges = bucket_edges(half, i0)
    assert np.allclose(edges, np.round(edges), atol=1e-9)
    edges = np.round(edges).astype(np.int64)
    assert edges[0] == i0
    assert np.all(np.diff(edges) > 0), "buckets must be non-empty and ordered"
    if complete:
        assert edges[-1] == i1, (edges[-1], i1)
    else:
        assert edges[-1] <= i1
    seg = samples[edges[0]:edges[-1]]
    starts = edges[:-1] - edges[0]
    exp_min = np.fmin.reduceat(seg, starts)
    exp_max = np.fmax.reduceat(seg, starts)
    assert nan_equal(y[0::2], exp_min), "bucket minimum differs from brute force"
    assert nan_equal(y[1::2], exp_max), "bucket maximum differs from brute force"
    # Whole-window envelope equals brute force.
    if seg.size and not np.all(np.isnan(seg)):
        assert np.nanmin(y) == np.nanmin(seg)
        assert np.nanmax(y) == np.nanmax(seg)
    return edges


def ref_stats(v: np.ndarray) -> dict:
    """Exact statistics with numpy (NaN ignored)."""
    ok = v[~np.isnan(v)]
    n = ok.size
    if n == 0:
        return {"n": 0}
    mean = float(ok.mean())
    return {
        "n": n,
        "min": float(ok.min()),
        "max": float(ok.max()),
        "mean": mean,
        "std": float(ok.std(ddof=1)) if n > 1 else math.nan,
        "rms": float(np.sqrt(np.mean(ok * ok))),
    }


def assert_stats(st, ref: dict, rel=1e-12, std_rel=1e-9):
    """Compare an engine Stats object with ref_stats()."""
    assert st is not None
    assert st.n == ref["n"]
    if ref["n"] == 0:
        return
    assert st.min == ref["min"]
    assert st.max == ref["max"]
    scale = max(abs(ref["mean"]), abs(ref["max"]), abs(ref["min"]), 1e-300)
    assert abs(st.mean - ref["mean"]) <= rel * scale, (st.mean, ref["mean"])
    if ref["n"] > 1:
        assert math.isclose(st.std, ref["std"], rel_tol=std_rel, abs_tol=1e-300), (st.std, ref["std"])
    assert math.isclose(st.rms, ref["rms"], rel_tol=rel * 10), (st.rms, ref["rms"])
