"""Data engine: one worker thread owns the file and does all heavy work.

Summary
    The GUI sends requests (plot, table, statistics, x values). The
    worker answers with Qt signals. For each request kind, only the
    newest request is kept; older ones are dropped. When no request is
    waiting, the worker loads channels into RAM (if they fit) and builds
    the decimation pyramids.

Thread model
    - Worker thread: all npTDMS calls, loading, decimation.
    - GUI thread: try_read() for small synchronous reads. It reads only
      finished RAM arrays or the thread-safe fast reader.
    - Every result carries the file generation number. The GUI drops
      results of an older file.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent import futures
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Signal

from . import pyramid as pyr
from .tdmsfile import KIND_BOOL, KIND_COMPLEX, KIND_TIME, PLOTTABLE, ChannelInfo, TdmsSource
from .xaxis import ArrayMap, LinearMap

BLOCK = 1 << 22  # samples per load step
PYR_BASE_RAM = 256
PYR_MIN_LEN = 4096  # no pyramid below this length (raw decimation is cheap)
RAW_DECIMATE_MAX = 1 << 24  # max raw samples for statistics on demand
RAW_PLOT_MAX = 1 << 21  # max raw samples for one plot update
XY_MAX_POINTS = 1_000_000


def ram_budget_bytes() -> int:
    """RAM for channel data: 40 % of available memory, at most 16 GiB."""
    env = os.environ.get("TDMSVIEWER_RAM_MB")
    if env:
        return int(float(env) * 2**20)
    avail = None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if avail is None:
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            avail = 4 << 30
    return int(min(0.4 * avail, 16 << 30))


def pyramid_bucket_budget() -> int:
    """Level-0 buckets allowed for all disk channels (40 bytes each)."""
    return int(min(512 << 20, 0.05 * max(ram_budget_bytes() / 0.4, 1 << 30)) // 40)


def _itemsize(info: ChannelInfo) -> int:
    if info.kind in PLOTTABLE:
        return max(1, info.dtype.itemsize)
    return 64  # strings: estimate


# -- requests -------------------------------------------------------------------

@dataclass
class PlotItem:
    cid: int
    xmap: object  # LinearMap | ArrayMap
    s: int
    e: int


@dataclass
class PlotRequest:
    seq: int
    items: list
    xa: float
    xb: float
    pixels: int


@dataclass
class TableRequest:
    seq: int
    cids: list
    i0: int
    i1: int


@dataclass
class StatsRequest:
    seq: int
    items: list  # PlotItem
    xa: float
    xb: float
    cursors: list = field(default_factory=list)  # x positions


@dataclass
class XRequest:
    seq: int
    cid: int


@dataclass
class ExportRequest:
    seq: int
    path: str
    items: list  # PlotItem
    xa: float
    xb: float
    header: list


class _Store:
    """Engine-side state of one channel."""

    __slots__ = ("info", "ram", "pyr", "done", "to_ram", "fast", "t0", "base")

    def __init__(self, info: ChannelInfo, fast, to_ram: bool, base: int):
        self.info = info
        self.ram = None
        self.pyr = None
        self.done = False
        self.to_ram = to_ram
        self.fast = fast
        self.t0 = None
        self.base = base


class _Progress:
    """Load progress, emitted at most 10 times per second."""

    def __init__(self, engine, gen: int, total: int):
        self.engine, self.gen = engine, gen
        self.total = max(1, total)
        self.done = 0
        self._last = 0.0
        self._t0 = time.perf_counter()

    def add(self, nbytes: int, text: str) -> None:
        self.done += nbytes
        self.step(text)

    def step(self, text: str) -> None:
        now = time.perf_counter()
        if now - self._last > 0.1:
            self._last = now
            self.engine.progress.emit(self.gen, min(0.999, self.done / self.total), text)

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0


def _first_time(a: np.ndarray):
    """First valid timestamp of an array (the zero of time-channel plots)."""
    ok = a[~np.isnat(a)] if a.size else a
    return ok[0].astype("datetime64[us]") if ok.size else np.datetime64(0, "us")


def to_f64(a: np.ndarray, kind: str, t0=None) -> np.ndarray:
    """Plot values (float64) of native channel values."""
    if a.dtype == np.float64:
        return a
    if kind == KIND_TIME:
        if t0 is None:
            t0 = np.datetime64(0, "us")
        return (a.astype("datetime64[us]") - t0) / np.timedelta64(1, "us") / 1e6
    if kind == KIND_COMPLEX:
        return np.abs(a).astype(np.float64)
    if kind == KIND_BOOL:
        return a.astype(np.float64)
    return a.astype(np.float64)


class DataEngine(QObject):
    """Owns the worker thread. Create and use in the GUI thread."""

    opened = Signal(int, object)  # gen, FileModel
    openFailed = Signal(int, str)
    progress = Signal(int, float, str)  # gen, fraction 0..1 (1 = done), text
    channelsUpdated = Signal(int, object)  # gen, list of channel ids
    plotReady = Signal(int, int, object)  # gen, seq, {cid: (x, y, complete)}
    tableReady = Signal(int, int, object)  # gen, seq, {cid: (i0, values)}
    statsReady = Signal(int, int, object)  # gen, seq, {cid: dict}
    xReady = Signal(int, int, object)  # gen, seq, (cid, ArrayMap, t_ref_unix) or error str
    exportDone = Signal(int, int, str)  # gen, seq, message
    message = Signal(int, str)  # gen, warning text

    _ORDER = ("open", "close", "xarr", "plot", "table", "stats", "export")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cv = threading.Condition()
        self._slots: dict = {}
        self._quit = False
        self._gen = 0  # newest generation (GUI side)
        self._sync_lock = threading.Lock()
        # Worker-owned state.
        self._source: TdmsSource | None = None
        self._stores: list[_Store] = []
        self._stores_gen = -1
        self._bg = None
        self._priority: list[int] = []
        self._pool = futures.ThreadPoolExecutor(max_workers=max(1, min(4, (os.cpu_count() or 2) - 2)),
                                                thread_name_prefix="tdms-pyramid")
        self._thread = threading.Thread(target=self._run, name="tdms-engine", daemon=True)
        self._thread.start()

    # -- GUI thread API -----------------------------------------------------------

    @property
    def generation(self) -> int:
        return self._gen

    def _post(self, kind: str, payload) -> None:
        with self._cv:
            self._slots[kind] = payload
            self._cv.notify()

    def open(self, path: str) -> int:
        with self._cv:
            self._gen += 1
            gen = self._gen
            self._slots.clear()
            self._slots["open"] = (gen, path)
            self._cv.notify()
        return gen

    def close_file(self) -> None:
        with self._cv:
            self._gen += 1
            self._slots.clear()
            self._slots["close"] = self._gen
            self._cv.notify()

    def request_plot(self, req: PlotRequest) -> None:
        self._post("plot", (self._gen, req))

    def request_table(self, req: TableRequest) -> None:
        self._post("table", (self._gen, req))

    def request_stats(self, req: StatsRequest) -> None:
        self._post("stats", (self._gen, req))

    def request_x(self, req: XRequest) -> None:
        self._post("xarr", (self._gen, req))

    def request_export(self, req: ExportRequest) -> None:
        self._post("export", (self._gen, req))

    def set_priority(self, cids: list[int]) -> None:
        self._priority = list(cids)

    def try_read(self, gen: int, cid: int, i0: int, i1: int):
        """Synchronous read for the GUI. Returns None if it would block."""
        with self._sync_lock:
            if gen != self._gen or gen != self._stores_gen or not (0 <= cid < len(self._stores)):
                return None
            st = self._stores[cid]
            if st.ram is not None:
                return st.ram[max(0, i0):max(0, i1)]
            if st.fast is not None:
                try:
                    return st.fast.read(i0, i1)
                except Exception:
                    return None
        return None

    def shutdown(self) -> None:
        self._stores_gen = -1  # helper threads stop at their next block
        with self._cv:
            self._quit = True
            self._slots.clear()
            self._cv.notify()
        self._thread.join(timeout=3.0)
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- worker loop ----------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = None
            with self._cv:
                while not self._slots and self._bg is None and not self._quit:
                    self._cv.wait()
                if self._quit:
                    break
                for kind in self._ORDER:
                    if kind in self._slots:
                        job = (kind, self._slots.pop(kind))
                        break
            try:
                if job is not None:
                    self._handle(*job)
                elif self._bg is not None:
                    try:
                        next(self._bg)
                    except StopIteration:
                        self._bg = None
            except Exception as exc:  # never let the worker die
                self._bg = None if job is None else self._bg
                traceback.print_exc()
                self.message.emit(self._stores_gen, f"Internal error: {exc}")
        self._close_source()

    def _pending(self, kind: str) -> bool:
        return kind in self._slots or "open" in self._slots or "close" in self._slots

    def _handle(self, kind: str, payload) -> None:
        if kind == "open":
            gen, path = payload
            self._do_open(gen, path)
            return
        if kind == "close":
            self._close_source()
            self._stores_gen = payload
            return
        gen, req = payload
        if gen != self._stores_gen or self._source is None:
            return
        if kind == "plot":
            self._do_plot(gen, req)
        elif kind == "table":
            self._do_table(gen, req)
        elif kind == "stats":
            self._do_stats(gen, req)
        elif kind == "xarr":
            self._do_x(gen, req)
        elif kind == "export":
            self._do_export(gen, req)

    # -- open / load ----------------------------------------------------------------

    def _close_source(self) -> None:
        self._bg = None
        with self._sync_lock:
            src, self._source = self._source, None
            self._stores = []
            if src is not None:
                try:
                    src.close()
                except Exception:
                    pass

    def _do_open(self, gen: int, path: str) -> None:
        self._close_source()
        self.progress.emit(gen, 0.0, f"Opening {os.path.basename(path)} ...")
        t = time.perf_counter()
        try:
            src = TdmsSource(path)
        except Exception as exc:
            self._stores_gen = gen
            self.openFailed.emit(gen, f"{type(exc).__name__}: {exc}")
            return
        model = src.model
        # Residency: RAM for channels that fit the budget, in file order.
        budget = ram_budget_bytes()
        used = 0
        to_ram = []
        for c in model.channels:
            nbytes = c.length * _itemsize(c)
            ok = used + nbytes <= budget
            if ok:
                used += nbytes
            to_ram.append(ok)
        disk_samples = sum(c.length for c, r in zip(model.channels, to_ram) if not r and c.plottable)
        base_disk = PYR_BASE_RAM
        if disk_samples:
            need = disk_samples / max(1, pyramid_bucket_budget())
            while base_disk < need:
                base_disk *= 2
        stores = [
            _Store(c, src.fast_reader(c.id), r, PYR_BASE_RAM if r else base_disk)
            for c, r in zip(model.channels, to_ram)
        ]
        with self._sync_lock:
            self._source = src
            self._stores = stores
            self._stores_gen = gen
        model.open_ms = 1e3 * (time.perf_counter() - t)
        self.opened.emit(gen, model)
        self._bg = self._load(gen)

    def _load(self, gen: int):
        """Background loader (generator: one short step per next())."""
        src = self._source
        stores = self._stores
        prog = _Progress(self, gen, sum(s.info.length * _itemsize(s.info) for s in stores))
        mixed = src.has_mixed_layout()
        # npTDMS channels of an interleaved/DAQmx file: one data_chunks() pass.
        chunk_pass = [s for s in stores if mixed and s.fast is None and s.info.length]
        chunk_ids = {s.info.id for s in chunk_pass}
        builds: list = []
        remaining = dict.fromkeys(s.info.id for s in stores if s.info.id not in chunk_ids)
        while remaining:
            pick = next((c for c in self._priority if c in remaining), None)
            if pick is None:
                pick = next(iter(remaining))
            del remaining[pick]
            st = stores[pick]
            before = prog.done
            try:
                yield from self._load_channel(gen, src, st, builds, prog)
            except Exception as exc:  # one bad channel must not stop the others
                st.done = True
                self.message.emit(gen, f"{st.info.label}: cannot read ({type(exc).__name__}: {exc})")
            if gen != self._stores_gen:
                return
            prog.done = before + st.info.length * _itemsize(st.info)
            yield
        if chunk_pass:
            try:
                yield from self._load_chunk_pass(gen, src, stores, chunk_pass, prog)
            except Exception as exc:
                self.message.emit(gen, f"Reading interleaved data failed: {type(exc).__name__}: {exc}")
            for st in chunk_pass:
                st.done = True
            self.channelsUpdated.emit(gen, sorted(chunk_ids))
        while builds:
            if gen != self._stores_gen:
                return
            finished, rest = futures.wait(builds, timeout=0.02)
            for f in finished:
                if f.exception() is not None:
                    self.message.emit(gen, f"Overview build failed: {f.exception()}")
            builds = list(rest)
            prog.step("Building overview")
            yield
        for w in src.drain_warnings():
            self.message.emit(gen, w)
        self.progress.emit(gen, 1.0, f"Loaded {prog.total / 1e6:.1f} MB in {prog.elapsed():.2f} s")

    def _load_channel(self, gen: int, src: TdmsSource, st: _Store, builds: list, prog):
        """Load one channel: RAM copy and/or pyramid (generator)."""
        info = st.info
        n = info.length
        if n == 0:
            st.done = True
            return
        if info.kind == KIND_TIME and st.t0 is None:
            st.t0 = _first_time(src.read(info.id, 0, min(n, 1024)))
        want_pyr = info.plottable and n >= PYR_MIN_LEN
        label = f"Loading {info.label}"
        if st.to_ram:
            arr = np.empty(n, dtype=info.dtype)
            for i in range(0, n, BLOCK):
                if gen != self._stores_gen:
                    return
                k = min(BLOCK, n - i)
                src.read_into(info.id, arr[i:i + k], i)
                prog.add(k * _itemsize(info), label)
                yield
            st.ram = arr
            if want_pyr:
                # Pyramid on a helper thread; the worker reads the next channel.
                st.pyr = pyr.Pyramid(n, st.base)
                builds.append(self._pool.submit(self._build_pyramid, gen, st, arr))
            else:
                st.done = True
                self.channelsUpdated.emit(gen, [info.id])
            return
        p = pyr.Pyramid(n, st.base) if want_pyr else None
        st.pyr = p
        if p is not None and st.fast is not None:
            # Fast reads are thread-safe: stream on a helper thread.
            builds.append(self._pool.submit(self._stream_pyramid, gen, st))
            return
        if p is None:
            st.done = True  # read on demand only (strings, short channels)
            self.channelsUpdated.emit(gen, [info.id])
            return
        for i in range(0, n, BLOCK):
            if gen != self._stores_gen:
                return
            a = src.read(info.id, i, min(n, i + BLOCK))
            p.append(to_f64(a, info.kind, st.t0))
            prog.add(a.size * _itemsize(info), label)
            yield
        st.done = True
        self.channelsUpdated.emit(gen, [info.id])

    def _load_chunk_pass(self, gen: int, src: TdmsSource, stores, chunk_pass, prog):
        """Load all npTDMS channels of an interleaved/DAQmx file in one pass."""
        arrays = {}
        ids = sorted(st.info.id for st in chunk_pass)
        for st in chunk_pass:
            info = st.info
            arrays[info.id] = np.empty(info.length, dtype=info.dtype) if st.to_ram else None
            if info.plottable and info.length >= PYR_MIN_LEN:
                st.pyr = pyr.Pyramid(info.length, st.base)
        last_upd = time.perf_counter()
        for cid, off, a in src.data_chunks():
            if gen != self._stores_gen:
                return
            st = stores[cid]
            if cid not in arrays:
                continue
            info = st.info
            if info.kind == KIND_TIME and st.t0 is None:
                st.t0 = _first_time(a)
            arr = arrays[cid]
            if arr is not None:
                arr[off:off + a.size] = a
            if st.pyr is not None:
                st.pyr.append(to_f64(a, info.kind, st.t0))
            prog.add(a.size * _itemsize(info), "Loading (interleaved data)")
            now = time.perf_counter()
            if now - last_upd > 0.5:
                last_upd = now
                self.channelsUpdated.emit(gen, ids)
            yield
        for st in chunk_pass:
            if arrays[st.info.id] is not None:
                st.ram = arrays[st.info.id]

    def _build_pyramid(self, gen: int, st: _Store, arr: np.ndarray) -> None:
        """Helper thread: pyramid of a finished RAM array."""
        p = st.pyr
        info = st.info
        for i in range(0, arr.size, BLOCK):
            if gen != self._stores_gen:
                return
            p.append(to_f64(arr[i:i + BLOCK], info.kind, st.t0))
        st.done = True
        if gen == self._stores_gen:
            self.channelsUpdated.emit(gen, [info.id])

    def _stream_pyramid(self, gen: int, st: _Store) -> None:
        """Helper thread: pyramid of a disk channel via the fast reader."""
        p = st.pyr
        info = st.info
        n = info.length
        buf = np.empty(min(n, BLOCK), dtype=st.fast.dtype)
        for i in range(0, n, BLOCK):
            if gen != self._stores_gen:
                return
            blk = buf[: min(BLOCK, n - i)]
            st.fast.read_into(blk, i)
            p.append(to_f64(blk, info.kind, st.t0))
        st.done = True
        if gen == self._stores_gen:
            self.channelsUpdated.emit(gen, [info.id])

    # -- reading helpers (worker) ---------------------------------------------------

    def _store(self, cid) -> _Store | None:
        """Store of a channel id, or None for an unknown id (also negative ids)."""
        if isinstance(cid, (int, np.integer)) and 0 <= cid < len(self._stores):
            return self._stores[cid]
        return None

    def _read(self, st: _Store, i0: int, i1: int) -> np.ndarray:
        if st.ram is not None:
            return st.ram[max(0, i0):max(0, i1)]
        return self._source.read(st.info.id, i0, i1)

    def _read_f64(self, st: _Store, i0: int, i1: int) -> np.ndarray:
        if st.info.kind == KIND_TIME and st.t0 is None:
            st.t0 = _first_time(self._read(st, 0, min(st.info.length, 1024)))
        return to_f64(self._read(st, i0, i1), st.info.kind, st.t0)

    def _can_read_cheap(self, st: _Store, n: int) -> bool:
        if st.ram is not None or st.fast is not None:
            return n <= RAW_DECIMATE_MAX
        return n <= RAW_DECIMATE_MAX // 8

    # -- plot -----------------------------------------------------------------------

    def _do_plot(self, gen: int, req: PlotRequest) -> None:
        out = {}
        px = max(16, int(req.pixels))
        for it in req.items:
            if self._pending("plot"):
                if out:  # a newer view exists: deliver what is done
                    self.plotReady.emit(gen, req.seq, out)
                return
            st = self._store(it.cid)
            if st is None:
                continue
            if not st.info.plottable:
                continue
            try:
                out[it.cid] = self._plot_one(st, it, req.xa, req.xb, px)
            except Exception as exc:
                traceback.print_exc()
                self.message.emit(gen, f"{st.info.label}: {exc}")
        self.plotReady.emit(gen, req.seq, out)

    def _plot_one(self, st: _Store, it: PlotItem, xa: float, xb: float, px: int):
        xmap = it.xmap
        s, e = it.s, min(it.e, st.info.length)
        if isinstance(xmap, ArrayMap):
            e = min(e, xmap.x.size)
            if not xmap.monotonic:
                return self._plot_xy(st, xmap, s, e)
        i0, i1 = xmap.index_range(xa, xb, s, e)
        n = i1 - i0
        if n <= 0:
            return np.empty(0), np.empty(0), True
        if n <= 2 * px:
            y = self._read_f64(st, i0, i1)
            return xmap.index_to_x(np.arange(i0, i0 + y.size)), y, True
        b = pyr.floor_pow2(n / px)
        p = st.pyr
        rr = lambda a, z: self._read_f64(st, a, z)  # noqa: E731
        if p is not None and b >= p.base:
            if p.covered >= i1:
                c, mn, mx = p.minmax(i0, i1, b, rr)
                complete = True
            elif n <= self._raw_plot_max(st):
                c, mn, mx = pyr.raw_minmax(self._read_f64(st, i0, i1), i0, b)
                complete = True
            else:
                # Still loading: draw the finished part, refine later.
                c, mn, mx = p.minmax(i0, i1, b, rr)
                complete = False
        elif p is None and n > self._raw_plot_max(st) * 8:
            return None
        else:
            # Bucket smaller than the pyramid base: n < base * px samples.
            c, mn, mx = pyr.raw_minmax(self._read_f64(st, i0, i1), i0, b)
            complete = True
        x, yy = pyr.interleave(c, mn, mx)
        return xmap.index_to_x(x), yy, complete

    @staticmethod
    def _raw_plot_max(st: _Store) -> int:
        """Max raw samples for one plot update (keeps an update below ~10 ms)."""
        if st.ram is not None:
            return RAW_PLOT_MAX * 4  # RAM: compute only, no I/O
        return RAW_PLOT_MAX if st.fast is not None else RAW_PLOT_MAX // 8

    def _plot_xy(self, st: _Store, xmap: ArrayMap, s: int, e: int):
        n = e - s
        if n <= 0:
            return np.empty(0), np.empty(0), True
        if n > RAW_DECIMATE_MAX * 4 and st.ram is None and st.fast is None:
            return None
        y = self._read_f64(st, s, e)
        x = xmap.x[s:e]
        if n <= XY_MAX_POINTS:
            return x, y, True
        b = -(-n // (XY_MAX_POINTS // 2))
        k = n // b
        body = y[: k * b].reshape(k, b)
        lo = np.where(np.isnan(body), np.inf, body).argmin(axis=1)
        hi = np.where(np.isnan(body), -np.inf, body).argmax(axis=1)
        base = np.arange(k) * b
        i1 = base + np.minimum(lo, hi)
        i2 = base + np.maximum(lo, hi)
        idx = np.empty(2 * k, dtype=np.int64)
        idx[0::2], idx[1::2] = i1, i2
        tail = np.arange(k * b, n)
        idx = np.concatenate((idx, tail))
        return x[idx], y[idx], True

    # -- table ----------------------------------------------------------------------

    def _do_table(self, gen: int, req: TableRequest) -> None:
        out = {}
        for cid in req.cids:
            if self._pending("table"):
                return
            if not (0 <= cid < len(self._stores)):
                continue
            st = self._stores[cid]
            i0 = max(0, req.i0)
            i1 = min(st.info.length, req.i1)
            if i1 > i0:
                try:
                    out[cid] = (i0, self._read(st, i0, i1))
                except Exception as exc:  # one bad channel must not drop the answer
                    self.message.emit(gen, f"{st.info.label}: {exc}")
        self.tableReady.emit(gen, req.seq, out)

    # -- statistics -----------------------------------------------------------------

    def _do_stats(self, gen: int, req: StatsRequest) -> None:
        out = {}
        for it in req.items:
            if self._pending("stats"):
                return
            st = self._store(it.cid)
            if st is None or not st.info.plottable:
                continue
            try:
                out[it.cid] = self._stats_one(st, it, req)
            except Exception as exc:  # one bad channel must not drop the answer
                self.message.emit(gen, f"{st.info.label}: {exc}")
        self.statsReady.emit(gen, req.seq, out)

    def _stats_one(self, st: _Store, it: PlotItem, req: StatsRequest) -> dict:
        info = st.info
        s, e = it.s, min(it.e, info.length)
        xmap = it.xmap
        if isinstance(xmap, ArrayMap):
            e = min(e, xmap.x.size)
        i0, i1 = inner_range(xmap, req.xa, req.xb, s, e)
        res = {"range": (i0, i1), "stats": None, "cursors": []}
        n = i1 - i0
        if n > 0:
            p = st.pyr
            if p is not None and p.covered >= i1:
                res["stats"] = p.stats(i0, i1, lambda a, z: self._read_f64(st, a, z))
            elif self._can_read_cheap(st, n):
                res["stats"] = pyr.raw_stats(self._read_f64(st, i0, i1))
        else:
            res["stats"] = pyr.EMPTY
        for cx in req.cursors:
            k = xmap.nearest(cx, s, e)
            if k < 0:
                res["cursors"].append(None)
            else:
                v = self._read(st, k, k + 1)
                res["cursors"].append((k, float(xmap.x_of(k)), v[0] if v.size else None))
        return res

    # -- x channel ------------------------------------------------------------------

    def _do_x(self, gen: int, req: XRequest) -> None:
        if not (0 <= req.cid < len(self._stores)):
            self.xReady.emit(gen, req.seq, "Unknown channel")
            return
        st = self._stores[req.cid]
        info = st.info
        if not info.plottable:
            self.xReady.emit(gen, req.seq, f"{info.label} is not numeric")
            return
        if info.length * 8 > ram_budget_bytes():
            self.xReady.emit(gen, req.seq, f"{info.label} is too large to use as X axis")
            return
        x = self._read_f64(st, 0, info.length)
        if x.dtype != np.float64 or not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=np.float64)
        from .xaxis import is_monotonic

        mono = is_monotonic(x)
        t_ref = None
        if info.kind == KIND_TIME and st.t0 is not None:
            t_ref = float((st.t0 - np.datetime64(0, "us")) / np.timedelta64(1, "us")) / 1e6
        self.xReady.emit(gen, req.seq, (req.cid, ArrayMap(x, mono), t_ref))

    # -- export ---------------------------------------------------------------------

    def _do_export(self, gen: int, req: ExportRequest) -> None:
        from .formatting import format_export

        try:
            ranges = []
            for it in req.items:
                st = self._store(it.cid)
                if st is None:
                    continue
                e = min(it.e, st.info.length)
                if isinstance(it.xmap, ArrayMap):
                    e = min(e, it.xmap.x.size)
                ranges.append((st, it, *inner_range(it.xmap, req.xa, req.xb, it.s, e)))
            rows = max((i1 - i0 for _, _, i0, i1 in ranges), default=0)
            with open(req.path, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(_csv_cell(h) for h in req.header) + "\n")
                step = 1 << 16
                for r0 in range(0, rows, step):
                    if gen != self._stores_gen:
                        return
                    r1 = min(rows, r0 + step)
                    cols = []
                    for st, it, i0, i1 in ranges:
                        a0, a1 = i0 + r0, min(i1, i0 + r1)
                        idx = np.arange(a0, max(a0, a1))
                        xs = it.xmap.index_to_x(idx) if idx.size else np.empty(0)
                        vals = self._read(st, a0, a1) if a1 > a0 else np.empty(0)
                        cols.append((idx, xs, vals))
                    lines = []
                    for k in range(r1 - r0):
                        cells = []
                        for idx, xs, vals in cols:
                            if k < idx.size:
                                cells += [str(int(idx[k])), repr(float(xs[k])), _csv_cell(format_export(vals[k]))]
                            else:
                                cells += ["", "", ""]
                        lines.append(",".join(cells))
                    fh.write("\n".join(lines) + "\n")
                    self.progress.emit(gen, min(0.999, r1 / max(1, rows)), "Exporting CSV")
            self.progress.emit(gen, 1.0, f"Exported {rows} rows")
            self.exportDone.emit(gen, req.seq, f"Exported {rows} rows to {req.path}")
        except Exception as exc:
            self.exportDone.emit(gen, req.seq, f"Export failed: {exc}")


def _csv_cell(text: str) -> str:
    if any(ch in text for ch in ',"\n\r'):
        return '"' + text.replace('"', '""') + '"'
    return text


def inner_range(xmap, xa: float, xb: float, s: int, e: int) -> tuple[int, int]:
    """Samples [i0, i1) whose x is inside [xa, xb] (exact, no extra samples)."""
    if e <= s:
        return s, s
    if isinstance(xmap, LinearMap):
        fa = (xa - xmap.x0) / xmap.dx
        fb = (xb - xmap.x0) / xmap.dx
        if not (math.isfinite(fa) and math.isfinite(fb)):
            return s, e
        i0 = int(math.ceil(fa - 1e-9))
        i1 = int(math.floor(fb + 1e-9)) + 1
        return max(s, min(e, i0)), max(s, min(e, i1))
    if not xmap.monotonic:
        return s, e
    seg = xmap.x[s:e]
    i0 = s + int(np.searchsorted(seg, xa, side="left"))
    i1 = s + int(np.searchsorted(seg, xb, side="right"))
    return i0, max(i0, i1)
