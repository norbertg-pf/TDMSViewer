#!/usr/bin/env python3
"""Benchmark of the tdmsviewer data path.

Writes synthetic TDMS files with npTDMS (TdmsWriter), then measures the
real package code:
    - TdmsSource open (metadata, fast reader build and verification)
    - fast-path reads against npTDMS read_data (full channel, windows)
    - pyramid build throughput
    - DataEngine: open() -> opened, -> loaded (progress 1.0), plot,
      statistics and table latencies, in RAM mode and disk mode
A naive baseline (TdmsFile.read of the whole file, then numpy min/max
per view) is measured on the same views. Output is markdown on stdout;
progress text goes to stderr.

Example (default size: 3 files of 512 MB, about 2 minutes on 4 cores):
    QT_QPA_PLATFORM=offscreen python3 bench/bench_engine.py > bench.md
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import math
import os
import platform
import pstats
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import nptdms  # noqa: E402
import PySide6  # noqa: E402
from nptdms import ChannelObject, GroupObject, RootObject, TdmsFile, TdmsWriter  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

from tdmsviewer import engine as engmod  # noqa: E402
from tdmsviewer import pyramid as pyr  # noqa: E402
from tdmsviewer.engine import DataEngine, PlotItem, PlotRequest, StatsRequest, TableRequest  # noqa: E402
from tdmsviewer.tdmsfile import TdmsSource  # noqa: E402
from tdmsviewer.xaxis import LinearMap  # noqa: E402

perf = time.perf_counter

DT = 1e-4  # wf_increment of the synthetic channels (10 kHz)
GROUPS = 4  # synthetic channels are spread over this many groups
TABLE_ROWS = 1024  # rows of one table block request
RAW_WINDOW = 3000  # samples; below 2 * pixels, so the engine returns raw samples
SIGNALS = ("opened", "openFailed", "progress", "channelsUpdated", "plotReady",
           "tableReady", "statsReady", "xReady", "exportDone", "message")
VIEWS = (("full", 1.0), ("10", 0.10), ("1", 0.01), ("raw", None))
_APP = None  # the QCoreApplication (engine signals need an event loop)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# -- formatting ---------------------------------------------------------------

def fnum(v) -> str:
    """Short number text (3 significant digits, no exponent)."""
    if v is None or not math.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def lat(times, scale: float = 1e3) -> str:
    """'median / p90' of times in seconds, scaled (1e3: ms, 1e6: us)."""
    if not times:
        return "n/a"
    a = np.asarray(times) * scale
    med = fnum(float(np.median(a)))
    if a.size < 5:
        return med
    return f"{med} / {fnum(float(np.percentile(a, 90)))}"


def measure(fn, repeat: int, budget: float = 5.0) -> list[float]:
    """Run fn up to repeat times (at least once, stop after budget seconds)."""
    out = []
    end = perf() + budget
    for _ in range(max(1, repeat)):
        t0 = perf()
        fn()
        out.append(perf() - t0)
        if perf() > end:
            break
    return out


class Table:
    """Markdown table: rows are metrics, columns are files."""

    def __init__(self):
        self.rows: list[tuple[str, str]] = []  # (key, label); key "#" = section
        self.cols: list[tuple[str, str]] = []  # (key, header)
        self.cells: dict = {}

    def row(self, key: str, label: str) -> None:
        self.rows.append((key, label))

    def put(self, col: str, key: str, text: str) -> None:
        self.cells[(col, key)] = text

    def render(self) -> str:
        head = "| Metric | " + " | ".join(h for _, h in self.cols) + " |"
        out = [head, "|---" * (len(self.cols) + 1) + "|"]
        pending = None
        for key, label in self.rows:
            if key == "#":
                pending = label
                continue
            vals = [self.cells.get((c, key), "") for c, _ in self.cols]
            if not any(vals):
                continue
            if pending is not None:
                out.append(f"| **{pending}** |" + " |" * len(self.cols))
                pending = None
            out.append(f"| {label} | " + " | ".join(vals) + " |")
        return "\n".join(out)


# -- files --------------------------------------------------------------------

@dataclass
class Spec:
    key: str
    header: str
    path: str
    synthetic: bool
    nch: int = 0
    n: int = 0
    nseg: int = 0
    write_s: float | None = None
    notes: list = field(default_factory=list)


def _names(nch: int) -> list[tuple[str, str]]:
    per = max(1, -(-nch // GROUPS))
    return [(f"Group {c // per}", f"Channel {c:02d}") for c in range(nch)]


def _block(ch: int, a: int, b: int) -> np.ndarray:
    """Test signal of channel ch, samples [a, b): offset + sine + noise."""
    i = np.arange(a, b, dtype=np.float64)
    y = np.random.default_rng([ch, a]).standard_normal(b - a)
    y += 100.0 * ch + 10.0 * np.sin(i * (2 * np.pi / (4096.0 + 97 * ch)))
    return y


def write_synthetic(path: str, nch: int, n: int, nseg: int) -> float:
    """Write nch float64 channels of n samples in nseg segments. Returns seconds."""
    names = _names(nch)
    nseg = max(1, min(nseg, n))
    cuts = np.linspace(0, n, nseg + 1).round().astype(np.int64)
    span = max(1 << 20, int(cuts[1] - cuts[0]))  # generate data in large blocks
    ca, cb, cache = 0, 0, []
    t0 = perf()
    with TdmsWriter(path) as w:
        for k in range(nseg):
            a, b = int(cuts[k]), int(cuts[k + 1])
            if b > cb:
                ca, cb = a, min(n, max(b, a + span))
                cache = [_block(c, ca, cb) for c in range(nch)]
            objs = []
            if k == 0:
                objs.append(RootObject({"name": "tdmsviewer benchmark"}))
                objs += [GroupObject(g) for g in dict.fromkeys(g for g, _ in names)]
            for c, (g, name) in enumerate(names):
                props = {"wf_increment": DT, "wf_start_offset": 0.0, "unit_string": "V"} if k == 0 else None
                objs.append(ChannelObject(g, name, cache[c][a - ca:b - ca], props))
            w.write_segment(objs)
    del cache
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)  # no write-back during the measurements
    finally:
        os.close(fd)
    return perf() - t0


def evict(path: str) -> None:
    """Drop the file from the page cache (cold read)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def describe(spec: Spec) -> None:
    """Fill channel count, length and segment count of an existing file."""
    src = TdmsSource(spec.path)
    try:
        chans = [c for c in src.model.channels if c.plottable and c.length]
        spec.nch = len(chans)
        spec.n = max((c.length for c in chans), default=0)
        spec.nseg = src.model.n_segments
    finally:
        src.close()


def make_windows(n: int, repeat: int, rng) -> dict:
    """Sample windows [i0, i1) per view kind, the same for all runs."""
    wins = {}
    for kind, frac in VIEWS:
        w = n if frac == 1.0 else (min(n, RAW_WINDOW) if frac is None else max(1, int(n * frac)))
        if w >= n:
            wins[kind] = [(0, n)] * repeat
        else:
            starts = rng.integers(0, n - w, size=repeat)
            wins[kind] = [(int(s), int(s) + w) for s in starts]
    rows = min(n, TABLE_ROWS)
    wins["table"] = [(int(s), int(s) + rows) for s in rng.integers(0, max(1, n - rows), size=repeat)]
    return wins


# -- open, read, pyramid --------------------------------------------------------

def bench_open(spec: Spec, tab: Table, args) -> None:
    col = spec.key
    t_fast, nfast, nplot = [], 0, 0
    end = perf() + args.budget
    for _ in range(args.open_repeat):
        t0 = perf()
        src = TdmsSource(spec.path)
        t_fast.append(perf() - t0)
        nfast = sum(c.fast for c in src.model.channels)
        nplot = sum(c.plottable for c in src.model.channels)
        for w in src.model.warnings:
            spec.notes.append(f"open warning: {w}")
        src.close()
        if perf() > end:
            break
    t_slow = measure(lambda: TdmsSource(spec.path, use_fast_path=False).close(), args.open_repeat, args.budget)
    t_np = measure(lambda: TdmsFile.open(spec.path).close(), args.open_repeat, args.budget)
    tab.put(col, "open", lat(t_fast))
    tab.put(col, "open_nofast", lat(t_slow))
    tab.put(col, "open_np", lat(t_np))
    tab.put(col, "fastch", f"{nfast}/{nplot}")
    if nfast < nplot:
        spec.notes.append(f"{nplot - nfast} plottable channels are not on the fast path")


def bench_read(spec: Spec, tab: Table, args, rng) -> np.ndarray | None:
    """Read speed of one channel. Returns its values as float64 (or None)."""
    col = spec.key
    src = TdmsSource(spec.path)
    f = TdmsFile.open(spec.path)
    try:
        cands = [c for c in src.model.channels if c.fast and c.length]
        if not cands:
            spec.notes.append("no fast-path channel: read benchmark skipped")
            return None
        longest = max(c.length for c in cands)
        cands = [c for c in cands if c.length == longest]
        info = cands[len(cands) // 2]
        fast = src.fast_reader(info.id)
        ch = f[info.group][info.name]
        ch.read_data(0, 16)  # npTDMS builds its channel index here (not timed)
        n = info.length
        nbytes = n * info.dtype.itemsize

        t_fast = measure(lambda: fast.read(0, n), 3, args.budget)
        t_np = measure(lambda: ch.read_data(0, n), 3, args.budget)
        a, b = fast.read(0, n), np.asarray(ch.read_data(0, n))
        if a.dtype != b.dtype or a.tobytes() != b.tobytes():
            spec.notes.append(f"BUG? fast read of {info.label} differs from npTDMS")
        tab.put(col, "rd_parts", str(len(getattr(fast, "_parts", ())) or "?"))
        tab.put(col, "rd_full_fast", f"{fnum(1e3 * min(t_fast))} ({fnum(nbytes / min(t_fast) / 1e9)} GB/s)")
        tab.put(col, "rd_full_np", f"{fnum(1e3 * min(t_np))} ({fnum(nbytes / min(t_np) / 1e9)} GB/s)")
        del b

        for key, w, rep in (("4k", 4096, 30), ("100k", 100_000, 10)):
            w = min(w, n)
            starts = [int(s) for s in rng.integers(0, max(1, n - w), size=rep)]
            tf, tn = [], []
            for s in starts:
                t0 = perf()
                fast.read(s, s + w)
                tf.append(perf() - t0)
            end = perf() + args.budget
            for s in starts:
                t0 = perf()
                ch.read_data(s, w)
                tn.append(perf() - t0)
                if perf() > end:
                    break
            tab.put(col, f"rd_{key}_fast", lat(tf, 1e6))
            speed = float(np.median(tn)) / max(1e-12, float(np.median(tf)))
            tab.put(col, f"rd_{key}_np", f"{lat(tn, 1e6)} (x{fnum(speed)})")
        return a.astype(np.float64, copy=False)
    finally:
        f.close()
        src.close()


def bench_pyramid(spec: Spec, tab: Table, y: np.ndarray | None) -> None:
    if y is None or y.size < engmod.PYR_MIN_LEN:
        return
    n = y.size
    box = []

    def build():
        p = pyr.Pyramid(n, engmod.PYR_BASE_RAM)
        for i in range(0, n, engmod.BLOCK):
            p.append(y[i:i + engmod.BLOCK])
        box[:] = [p]

    t = min(measure(build, 3, 20.0))
    tab.put(spec.key, "pyr_build", f"{fnum(1e9 * t / n)} ({fnum(n * 8 / t / 1e6)} MB/s)")
    total = spec.nch * spec.n
    tab.put(spec.key, "pyr_mem", fnum(box[0].nbytes / n * total / 1e6))


# -- engine -------------------------------------------------------------------

class Probe:
    """Record engine signals (main thread) and wait for them."""

    def __init__(self, engine: DataEngine):
        self.engine = engine
        self.log: list = []  # (time, signal name, args)
        self._loop = None
        self._pred = None
        for name in SIGNALS:
            getattr(engine, name).connect(self._slot(name))

    def _slot(self, name):
        def slot(*args):
            self.log.append((perf(), name, args))
            if self._pred is not None and self._pred():
                self._loop.quit()
        return slot

    def find(self, names, test, since: int):
        for t, n, a in self.log[since:]:
            if n in names and test(a):
                return t, n, a
        return None

    timeout = 300.0  # seconds per wait

    def wait(self, names, test, since: int, timeout: float | None = None):
        timeout = timeout or self.timeout
        if isinstance(names, str):
            names = (names,)
        hit = self.find(names, test, since)
        if hit is None:
            loop = QEventLoop()
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            timer.start(int(timeout * 1000))
            self._loop = loop
            self._pred = lambda: self.find(names, test, since) is not None
            loop.exec()
            timer.stop()
            self._loop = self._pred = None
            hit = self.find(names, test, since)
        if hit is None:
            raise TimeoutError(f"no {names} signal within {timeout} s")
        return hit

    def request(self, kind: str, req):
        """Send one request, wait for its answer. Returns (seconds, result)."""
        sig = {"plot": "plotReady", "table": "tableReady", "stats": "statsReady"}[kind]
        since = len(self.log)
        t0 = perf()
        getattr(self.engine, "request_" + kind)(req)
        t, _, a = self.wait(sig, lambda a: a[1] == req.seq, since)
        return t - t0, a[2]


def plot_items(model) -> list[PlotItem]:
    """Plot items like the main window builds them (waveform time as x)."""
    return [PlotItem(c.id, LinearMap(model.start_seconds(c), c.wf_increment or 1.0), 0, c.length)
            for c in model.channels if c.plottable and c.length]


def _x_range(items, i0: int, i1: int) -> tuple[float, float]:
    m = items[0].xmap
    return m.x_of(i0), m.x_of(i1 - 1)


def _env(out: dict, model) -> dict:
    """Min and max of each plot result, keyed by (group, name)."""
    env = {}
    for cid, r in out.items():
        c = model.channels[cid]
        if r is None or r[1].size == 0:
            env[(c.group, c.name)] = None
        else:
            env[(c.group, c.name)] = (float(np.nanmin(r[1])), float(np.nanmax(r[1])))
    return env


def set_ram_mode(mode: str) -> None:
    if mode == "disk":
        os.environ["TDMSVIEWER_RAM_MB"] = "1"  # every channel stays on disk
    else:
        os.environ.pop("TDMSVIEWER_RAM_MB", None)


def run_engine(spec: Spec, mode: str, wins: dict, args, tab: Table | None, cold: bool = False,
               keep: bool = False) -> dict:
    """One engine session: open, first plot, full load, then request latencies.

    With tab None only open and load are measured (cold or profile runs).
    With keep the loaded engine is returned in res["eng"] (caller shuts it down).
    """
    set_ram_mode(mode)
    if cold:
        evict(spec.path)
    gc.collect()
    eng = DataEngine()
    probe = Probe(eng)
    res = {"checks": {}}
    col, m = spec.key, mode
    try:
        since = len(probe.log)
        t0 = perf()
        gen = eng.open(spec.path)
        t, name, a = probe.wait(("opened", "openFailed"), lambda a: a[0] == gen, since)
        if name == "openFailed":
            spec.notes.append(f"engine {mode}: open failed: {a[1]}")
            return res
        t_opened = t - t0
        model = a[1]
        items = plot_items(model)
        if not items:
            return res
        cids = [it.cid for it in items]
        px = args.pixels
        seq = 1
        xa, xb = _x_range(items, 0, spec.n)
        t_first, out = probe.request("plot", PlotRequest(seq, items, xa, xb, px))
        n_ok = sum(1 for r in out.values() if r is not None and r[2])
        t, _, _ = probe.wait("progress", lambda a: a[0] == gen and a[1] >= 1.0, since)
        t_loaded = t - t0
        res["loaded"] = t_loaded
        if tab is None:
            if keep:
                res.update(gen=gen, model=model, items=items, eng=eng, probe=probe)
            return res
        try:  # residency (private engine state, read only)
            stores = eng._stores
            n_ram = sum(st.ram is not None for st in stores if st.info.plottable)
            bases = sorted({st.pyr.base for st in stores if st.pyr is not None})
            tab.put(col, f"{m}_res", f"{n_ram}/{len(items)} RAM, base {'/'.join(map(str, bases)) or '-'}")
        except AttributeError:
            pass
        tab.put(col, f"{m}_opened", fnum(1e3 * t_opened))
        tab.put(col, f"{m}_first", f"{fnum(1e3 * (t_opened + t_first))} ({n_ok}/{len(items)} complete)")
        size = os.path.getsize(spec.path)
        tab.put(col, f"{m}_loaded", f"{fnum(1e3 * t_loaded)} ({fnum(size / t_loaded / 1e6)} MB/s)")

        # Steady state: all channels loaded.
        for kind, _ in VIEWS:
            times, bad = [], 0
            for k, (i0, i1) in enumerate(wins[kind]):
                seq += 1
                xa, xb = _x_range(items, i0, i1)
                dt, out = probe.request("plot", PlotRequest(seq, items, xa, xb, px))
                times.append(dt)
                bad += sum(1 for r in out.values() if r is None or not r[2])
                if k == 0:
                    ranges = {}
                    for it in items:
                        c = model.channels[it.cid]
                        ranges[(c.group, c.name)] = it.xmap.index_range(xa, xb, it.s, it.e)
                    res["checks"][f"plot_{kind}"] = (_env(out, model), ranges)
            tab.put(col, f"{m}_plot_{kind}", lat(times))
            if bad:
                spec.notes.append(f"engine {mode}: plot {kind}: {bad} incomplete or missing results")
        for kind in ("full", "1"):
            times = []
            for k, (i0, i1) in enumerate(wins[kind]):
                seq += 1
                xa, xb = _x_range(items, i0, i1)
                dt, out = probe.request("stats", StatsRequest(seq, items, xa, xb, []))
                times.append(dt)
                if k == 0:
                    st = {}
                    for cid, r in out.items():
                        c = model.channels[cid]
                        st[(c.group, c.name)] = (r["range"], r["stats"])
                    res["checks"][f"stats_{kind}"] = st
            tab.put(col, f"{m}_stats_{kind}", lat(times))
        times, sync = [], []
        for i0, i1 in wins["table"]:
            seq += 1
            dt, out = probe.request("table", TableRequest(seq, cids, i0, i1))
            times.append(dt)
            t1 = perf()
            got = [eng.try_read(gen, c, i0, i1) for c in cids]
            sync.append(perf() - t1)
            if any(g is None for g in got):
                sync.pop()
        tab.put(col, f"{m}_table", lat(times))
        tab.put(col, f"{m}_tryread", lat(sync, 1e6) if sync else "None (would block)")
        return res
    finally:
        for _, name, a in probe.log:
            if name == "message":
                spec.notes.append(f"engine {mode}: message: {a[1]}")
        if "eng" not in res:
            eng.shutdown()
            del probe, eng
            gc.collect()


# -- naive baseline -------------------------------------------------------------

def naive_plot(arrays: list, dts: list, i0: int, i1: int, px: int) -> list:
    """Min/max per pixel column with numpy, every sample of the view."""
    out = []
    for y, dt in zip(arrays, dts):
        seg = y[i0:i1]
        n = seg.size
        if n <= 2 * px:
            out.append((np.arange(i0, i0 + n) * dt, seg.copy()))
            continue
        k = n // px
        m = n // k
        body = seg[: m * k].reshape(m, k)
        yy = np.empty(2 * m)
        yy[0::2] = body.min(axis=1)
        yy[1::2] = body.max(axis=1)
        out.append((np.repeat(i0 + k * (np.arange(m) + 0.5), 2) * dt, yy))
    return out


def naive_stats(arrays: list, i0: int, i1: int) -> list:
    out = []
    for y in arrays:
        s = y[i0:i1]
        out.append((s.size, s.min(), s.max(), s.mean(), s.std(ddof=1)))
    return out


def run_naive(spec: Spec, wins: dict, args, tab: Table, eng_checks: dict) -> None:
    col = spec.key
    px = args.pixels
    gc.collect()
    if args.cold:
        evict(spec.path)
        t0 = perf()
        f = TdmsFile.read(spec.path)
        tab.put(col, "nv_read_cold", fnum(1e3 * (perf() - t0)))
        del f
        gc.collect()
    t0 = perf()
    f = TdmsFile.read(spec.path)
    t_read = perf() - t0
    keys, arrays, dts = [], [], []
    for g in f.groups():
        for ch in g.channels():
            a = ch[:]
            if len(a) and a.dtype.kind in "fiub":
                keys.append((g.name, ch.name))
                arrays.append(np.asarray(a, dtype=np.float64))
                dts.append(float(ch.properties.get("wf_increment", 1.0) or 1.0))
    del f
    if not arrays:
        return
    n = max(a.size for a in arrays)
    t1 = perf()
    naive_plot(arrays, dts, 0, n, px)
    tab.put(col, "nv_read", fnum(1e3 * t_read))
    tab.put(col, "nv_first", fnum(1e3 * (t_read + perf() - t1)))
    for kind, _ in VIEWS:
        times = []
        for i0, i1 in wins[kind]:
            t1 = perf()
            naive_plot(arrays, dts, i0, i1, px)
            times.append(perf() - t1)
        tab.put(col, f"nv_plot_{kind}", lat(times))
    tab.put(col, "nv_stats_full", lat(measure(lambda: naive_stats(arrays, 0, n), len(wins["full"]))))
    times = []
    for i0, i1 in wins["table"]:
        t1 = perf()
        [a[i0:i1].copy() for a in arrays]
        times.append(perf() - t1)
    tab.put(col, "nv_table", lat(times))
    check_results(spec, tab, dict(zip(keys, arrays)), eng_checks)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return bool(np.array_equal(np.array(a), np.array(b), equal_nan=True))


def _close(a: float, b: float, rel: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= rel * max(1e-300, abs(b))


def check_results(spec: Spec, tab: Table, data: dict, eng_checks: dict) -> None:
    """Compare engine envelopes and statistics with numpy on the same samples (NaN ignored)."""
    env_bad, env_n, st_bad, st_n = [], 0, [], 0
    with np.errstate(all="ignore"):
        for mode, checks in eng_checks.items():
            for name, val in checks.items():
                if name.startswith("plot_"):
                    env, ranges = val
                    for key, got in env.items():
                        i0, i1 = ranges[key]
                        y = data[key][i0:i1]
                        env_n += 1
                        want = (float(np.nanmin(y)), float(np.nanmax(y))) if y.size else None
                        if not _same(got, want):
                            env_bad.append(f"{mode} {name} {key[1]}: engine {got}, numpy {want} on [{i0}, {i1})")
                    continue
                for key, ((i0, i1), s) in val.items():
                    y = data[key][i0:i1]
                    y = y[~np.isnan(y)]
                    st_n += 1
                    if s is None:
                        st_bad.append(f"{mode} {name} {key[1]}: no statistics")
                        continue
                    if y.size == 0:
                        ok = s.n == 0
                    else:
                        ok = (s.n == y.size and s.min == y.min() and s.max == y.max()
                              and _close(s.mean, float(y.mean()), 1e-9)
                              and (y.size < 2 or _close(s.std, float(y.std(ddof=1)), 1e-7)))
                    if not ok:
                        ref = (float(y.mean()), float(y.std(ddof=1))) if y.size > 1 else (math.nan, math.nan)
                        st_bad.append(f"{mode} {name} {key[1]} on [{i0}, {i1}): engine n={s.n} mean={s.mean!r} "
                                      f"std={s.std!r}; numpy n={y.size} mean={ref[0]!r} std={ref[1]!r} "
                                      f"(rel. std error {abs(s.std - ref[1]) / ref[1]:.1e})")
    if env_n:
        tab.put(spec.key, "chk_env", "OK" if not env_bad else f"FAIL {len(env_bad)}/{env_n}")
    if st_n:
        tab.put(spec.key, "chk_stats", "OK" if not st_bad else f"FAIL {len(st_bad)}/{st_n}")
    spec.notes.extend(f"CHECK FAILED: {m}" for m in (env_bad + st_bad)[:10])


# -- profiling ------------------------------------------------------------------

class ThreadProfiler:
    """One cProfile.Profile per thread started while this is active."""

    def __init__(self):
        self.profiles: list = []  # (thread name, profile)

    def _hook(self, frame, event, arg):
        sys.setprofile(None)
        p = cProfile.Profile()
        self.profiles.append((threading.current_thread().name, p))
        p.enable()

    def __enter__(self):
        threading.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        threading.setprofile(None)

    def stats(self, prefix: str):
        sel = [p for name, p in self.profiles if name.startswith(prefix)]
        if not sel:
            return None
        st = pstats.Stats(sel[0])
        for p in sel[1:]:
            st.add(p)
        return st


_IDLE = ("acquire' of '_thread.lock", "acquire' of '_thread.RLock", "method 'exec' of",
         "'get' of '_queue.SimpleQueue'")


def _func_name(fn: str, line: int, name: str) -> str:
    if fn == "~":
        return name
    p = Path(fn)
    try:
        rel = p.resolve().relative_to(ROOT)
    except ValueError:
        rel = Path(p.parent.name) / p.name
    return f"{rel}:{line}({name})"


def profile_table(title: str, st, top: int) -> str:
    """Top functions by own time (idle lock waits removed) and package cumtime."""
    if st is None:
        return ""
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        if any(s in name for s in _IDLE):
            continue
        rows.append((tt, ct, nc, _func_name(fn, line, name), fn))
    busy = sum(r[0] for r in rows)
    out = [f"### {title} (busy {fnum(1e3 * busy)} ms, idle waits removed)", "",
           "| tottime ms | cumtime ms | calls | function |", "|---|---|---|---|"]
    for tt, ct, nc, name, _ in sorted(rows, key=lambda r: -r[0])[:top]:
        out.append(f"| {fnum(1e3 * tt)} | {fnum(1e3 * ct)} | {nc} | `{name}` |")
    pkg_dir = str(ROOT / "tdmsviewer")
    pkg = [r for r in rows if r[4].startswith(pkg_dir)]
    if pkg:
        out.append("| | | | *package functions by cumtime:* |")
        for tt, ct, nc, name, _ in sorted(pkg, key=lambda r: -r[1])[:5]:
            out.append(f"| {fnum(1e3 * tt)} | {fnum(1e3 * ct)} | {nc} | `{name}` |")
    return "\n".join(out) + "\n"


def run_profiles(spec: Spec, wins: dict, args) -> list[str]:
    """cProfile of open (main thread), load (engine threads) and requests."""
    out = []
    pr = cProfile.Profile()
    pr.enable()
    TdmsSource(spec.path).close()
    pr.disable()
    out.append(profile_table(f"{spec.header}: TdmsSource open", pstats.Stats(pr), args.profile_top))
    for mode in args.modes:
        with ThreadProfiler() as tp:
            res = run_engine(spec, mode, wins, args, None, keep=True)
        eng = res.get("eng")
        if eng is None:
            continue
        try:
            if mode == args.modes[0] and hasattr(eng, "_handle"):
                # Requests: DataEngine._handle called in this thread while the
                # worker is idle (same code the worker runs).
                time.sleep(0.1)
                gen, items = res["gen"], res["items"]
                cids = [it.cid for it in items]
                pr = cProfile.Profile()
                seq = 10_000
                pr.enable()
                for kind, _ in VIEWS:
                    for i0, i1 in wins[kind]:
                        seq += 1
                        xa, xb = _x_range(items, i0, i1)
                        eng._handle("plot", (gen, PlotRequest(seq, items, xa, xb, args.pixels)))
                for i0, i1 in wins["full"] + wins["1"]:
                    seq += 1
                    xa, xb = _x_range(items, i0, i1)
                    eng._handle("stats", (gen, StatsRequest(seq, items, xa, xb, [])))
                for i0, i1 in wins["table"]:
                    seq += 1
                    eng._handle("table", (gen, TableRequest(seq, cids, i0, i1)))
                pr.disable()
                out.append(profile_table(
                    f"{spec.header}: requests, {mode} mode ({len(wins['full'])} of each plot/stats/table kind)",
                    pstats.Stats(pr), args.profile_top))
        finally:
            eng.shutdown()
            pool = getattr(eng, "_pool", None)
            if pool is not None:
                pool.shutdown(wait=True)  # helper threads end before their profiles are read
        out.append(profile_table(f"{spec.header}: load, {mode} mode, engine worker thread",
                                 tp.stats("tdms-engine"), args.profile_top))
        out.append(profile_table(f"{spec.header}: load, {mode} mode, pyramid helper threads",
                                 tp.stats("tdms-pyramid"), args.profile_top))
        del res, eng
        gc.collect()
    return [o for o in out if o]


# -- main -----------------------------------------------------------------------

def build_rows(tab: Table, args) -> None:
    r = tab.row
    r("#", "File")
    r("size", "Size (MB), segments, channels x samples")
    r("write", "Write time, npTDMS TdmsWriter (s)")
    r("#", "Open (ms, median / p90)")
    r("open", "TdmsSource open")
    r("open_nofast", "TdmsSource open, fast path off")
    r("open_np", "npTDMS TdmsFile.open only")
    r("fastch", "Channels on the fast path")
    r("#", "Read one channel (warm page cache)")
    r("rd_parts", "Fast reader parts (layout runs) of this channel")
    r("rd_full_fast", "Full channel, fast path (ms best, GB/s)")
    r("rd_full_np", "Full channel, npTDMS read_data (ms best, GB/s)")
    r("rd_4k_fast", "Window 4096, fast path (us, median / p90)")
    r("rd_4k_np", "Window 4096, npTDMS (us), speedup")
    r("rd_100k_fast", "Window 100k, fast path (us)")
    r("rd_100k_np", "Window 100k, npTDMS (us), speedup")
    r("#", "Pyramid")
    r("pyr_build", "Build, 1 thread (ns/sample, MB/s)")
    r("pyr_mem", "Pyramid memory, all channels (MB)")
    for m in args.modes:
        r("#", f"Engine, {m} mode (all channels, {args.pixels} px; ms, median / p90)")
        r(f"{m}_res", "Residency")
        r(f"{m}_opened", "open() -> opened signal")
        r(f"{m}_first", "open() -> first full plot (sent at opened)")
        r(f"{m}_loaded", "open() -> loaded (progress 1.0)")
        r(f"{m}_cold", "open() -> loaded, cold page cache")
        for kind, label in (("full", "full view"), ("10", "zoom 10%"), ("1", "zoom 1%"),
                            ("raw", f"zoom to raw ({RAW_WINDOW} samples)")):
            r(f"{m}_plot_{kind}", f"Plot {label}")
        r(f"{m}_stats_full", "Stats, full range")
        r(f"{m}_stats_1", "Stats, 1% range")
        r(f"{m}_table", f"Table block, {TABLE_ROWS} rows x all channels")
        r(f"{m}_tryread", "try_read in GUI thread, same block (us)")
    r("#", "Naive baseline: TdmsFile.read + numpy min/max per view (ms)")
    r("nv_read", "TdmsFile.read whole file")
    r("nv_read_cold", "TdmsFile.read, cold page cache")
    r("nv_first", "First full plot (read + decimate)")
    for kind, label in (("full", "full view"), ("10", "zoom 10%"), ("1", "zoom 1%"),
                        ("raw", f"zoom to raw ({RAW_WINDOW} samples)")):
        r(f"nv_plot_{kind}", f"Plot {label}")
    r("nv_stats_full", "Stats, full range")
    r("nv_table", "Table block")
    r("#", "Checks (first window of each kind, engine vs numpy)")
    r("chk_env", "Plot envelope min/max")
    r("chk_stats", "Statistics n/min/max/mean/std")


def bench_file(spec: Spec, tab: Table, args, profiles: list) -> None:
    col = spec.key
    tab.cols.append((col, spec.header))
    size = os.path.getsize(spec.path)
    tab.put(col, "size", f"{fnum(size / 1e6)}, {spec.nseg}, {spec.nch} x {spec.n}")
    if spec.write_s is not None:
        tab.put(col, "write", fnum(spec.write_s))
    rng = np.random.default_rng(args.seed)
    wins = make_windows(spec.n, args.repeat, rng)
    log(f"[{spec.header}] open")
    bench_open(spec, tab, args)
    log(f"[{spec.header}] read")
    y = bench_read(spec, tab, args, rng)
    log(f"[{spec.header}] pyramid")
    bench_pyramid(spec, tab, y)
    del y
    checks = {}
    for mode in args.modes:
        log(f"[{spec.header}] engine {mode}")
        try:
            checks[mode] = run_engine(spec, mode, wins, args, tab)["checks"]
            if args.cold:
                res = run_engine(spec, mode, wins, args, None, cold=True)
                if "loaded" in res:
                    tab.put(col, f"{mode}_cold", fnum(1e3 * res["loaded"]))
        except TimeoutError as exc:
            spec.notes.append(f"engine {mode}: {exc}")
    if not args.no_naive:
        log(f"[{spec.header}] naive baseline")
        run_naive(spec, wins, args, tab, checks)
    if args.profile:
        log(f"[{spec.header}] profile")
        profiles.extend(run_profiles(spec, wins, args))
    set_ram_mode("ram")


def source_hash() -> str:
    """Short hash of the package sources (results belong to this code state)."""
    import hashlib

    h = hashlib.sha1()
    for f in sorted((ROOT / "tdmsviewer").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def fresh_memory_speed(mb: int = 256) -> float:
    """GB/s to allocate and first touch new memory (np.ones of mb MiB).

    In some VMs (for example Firecracker with free page reporting) this
    is slow, and it limits RAM-mode loads and TdmsFile.read.
    """
    t0 = perf()
    blocks = [np.ones(32 * 2**20 // 8) for _ in range(max(1, mb // 32))]
    dt = perf() - t0
    del blocks
    return max(1, mb // 32) * 32 * 2**20 / dt / 1e9


def _loadavg() -> str:
    try:
        with open("/proc/loadavg") as fh:
            return fh.read().split()[0]
    except OSError:
        return "?"


def environment_text(args) -> str:
    cpu = platform.processor() or "?"
    try:
        with open("/proc/cpuinfo") as fh:
            cpu = next(line.split(":", 1)[1].strip() for line in fh if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return (f"Python {platform.python_version()}, numpy {np.__version__}, npTDMS {nptdms.__version__}, "
            f"PySide6 {PySide6.__version__}; {os.cpu_count()} CPUs ({cpu}), {mem:.1f} GiB RAM. "
            f"Engine RAM budget: {engmod.ram_budget_bytes() / 2**20:.0f} MiB (RAM mode), "
            f"1 MiB (disk mode). Page cache is warm unless a row says cold. "
            f"Package source hash {source_hash()}. Load average at start {args.loadavg}, "
            f"at end {_loadavg()} (other processes add noise). New memory (allocate + first touch) "
            f"runs at {args.fresh_gbs:.2f} GB/s here; RAM-mode loads and TdmsFile.read pay this.")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--channels", type=int, default=16, help="channels per synthetic file")
    ap.add_argument("--samples", type=int, default=4_000_000, help="samples per channel")
    ap.add_argument("--segments", type=int, default=500, help="segments of the main file")
    ap.add_argument("--small-segments", type=int, default=20_000, help="segments of the many-segment file")
    ap.add_argument("--files", default="multi,one,small",
                    help="synthetic files: multi (--segments), one (1 segment), small (--small-segments); "
                         "'none' for no synthetic file")
    ap.add_argument("--extra", nargs="*", default=[], help="existing TDMS files to include (read only)")
    ap.add_argument("--modes", default="ram,disk", help="engine modes: ram, disk (TDMSVIEWER_RAM_MB=1)")
    ap.add_argument("--pixels", type=int, default=2000, help="plot width in pixels")
    ap.add_argument("--repeat", type=int, default=15, help="repeats per latency measurement")
    ap.add_argument("--open-repeat", type=int, default=3, help="repeats per open measurement")
    ap.add_argument("--budget", type=float, default=10.0, help="max seconds per repeated slow measurement")
    ap.add_argument("--cold", action="store_true", help="add cold page cache runs (posix_fadvise)")
    ap.add_argument("--no-naive", action="store_true", help="skip the naive baseline")
    ap.add_argument("--profile", action="store_true", help="print cProfile top functions per phase")
    ap.add_argument("--profile-top", type=int, default=10, help="functions per profile table")
    ap.add_argument("--workdir", default=None, help="folder for generated files (default: new temp folder)")
    ap.add_argument("--keep", action="store_true", help="keep generated files")
    ap.add_argument("--timeout", type=float, default=300.0, help="max seconds to wait for one engine signal")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    args.modes = [m for m in args.modes.split(",") if m]
    if any(m not in ("ram", "disk") for m in args.modes):
        ap.error("--modes: use ram and/or disk")
    args.files = [] if args.files == "none" else [f for f in args.files.split(",") if f]
    if any(f not in ("multi", "one", "small") for f in args.files):
        ap.error("--files: use multi, one, small or none")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    Probe.timeout = args.timeout
    args.loadavg = _loadavg()
    args.fresh_gbs = fresh_memory_speed()
    global _APP
    _APP = QCoreApplication.instance() or QCoreApplication(sys.argv[:1])
    made_dir = args.workdir is None
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="tdmsbench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    tab = Table()
    build_rows(tab, args)
    specs, profiles = [], []
    nseg = {"multi": args.segments, "one": 1, "small": args.small_segments}
    t_start = perf()
    try:
        for key in args.files:
            k = max(1, min(nseg[key], args.samples))
            specs.append(Spec(key, f"{k} segment{'s' if k > 1 else ''}", str(workdir / f"bench_{key}.tdms"),
                              True, args.channels, args.samples, k))
        for i, p in enumerate(args.extra):
            specs.append(Spec(f"extra{i}", Path(p).name[:28], p, False))
        for spec in specs:
            if spec.synthetic:
                log(f"[{spec.header}] writing {spec.nch} x {spec.n} samples ({spec.nch * spec.n * 8 / 1e6:.0f} MB)")
                spec.write_s = write_synthetic(spec.path, spec.nch, spec.n, spec.nseg)
            else:
                describe(spec)
            try:
                bench_file(spec, tab, args, profiles)
            finally:
                if spec.synthetic and not args.keep:
                    os.remove(spec.path)
    finally:
        if made_dir and not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
        set_ram_mode("ram")
    print("## tdmsviewer engine benchmark\n")
    print(environment_text(args) + "\n")
    print(tab.render())
    notes = [f"- {s.header}: {n}" for s in specs for n in dict.fromkeys(s.notes)]
    if notes:
        print("\nNotes:\n" + "\n".join(notes))
    print(f"\nTotal benchmark time: {perf() - t_start:.0f} s")
    if profiles:
        print("\n## Profiles (cProfile; its overhead inflates Python-heavy code)\n")
        print("\n".join(profiles))
    return 0


if __name__ == "__main__":
    sys.exit(main())
