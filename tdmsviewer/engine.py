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
from pyqtgraph.Qt.QtCore import QObject, Signal

from . import fastread
from . import pyramid as pyr
from .tdmsfile import KIND_BOOL, KIND_COMPLEX, KIND_INT, KIND_TIME, PLOTTABLE, ChannelInfo, TdmsSource
from .xaxis import ArrayMap, LinearMap, TimeRef, is_monotonic

BLOCK = 1 << 22  # samples per load step
FAMILY_STEP_BYTES = 32 << 20  # bytes per load step when channels load together
FAMILY_READ_MIN = 1 << 15  # plot: raw reads from this size read the channel family in one pass
FAMILY_READ_MAX = 1 << 25  # plot: max samples of one family read (all channels)
PYR_BASE_RAM = 256
PYR_MIN_LEN = 4096  # no pyramid below this length (raw decimation is cheap)
RAW_DECIMATE_MAX = 1 << 24  # max raw samples for statistics on demand
RAW_PLOT_MAX = 1 << 21  # max raw samples for one plot update
RAW_PLOT_BUDGET = 1 << 23  # max raw samples read from disk for one request (all channels)
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
            avail = _windows_available_ram() or 4 << 30
    return int(min(0.4 * avail, 16 << 30))


def _windows_available_ram() -> int | None:
    """Free physical memory in bytes (GlobalMemoryStatusEx), None if not Windows."""
    if os.name != "nt":
        return None
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    *((name, ctypes.c_ulonglong) for name in (
                        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual"))]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullAvailPhys)


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
    # Absolute time: x is seconds since time_ref (TimeRef). Each channel then
    # gets a UTC ISO 8601 column after x; empty for channels not in time_cids.
    time_ref: object = None
    time_cids: frozenset = frozenset()


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
    return ok[0].astype("datetime64[ns]") if ok.size else np.datetime64(0, "ns")


def _big_int(info: ChannelInfo) -> bool:
    """int64/uint64: float64 cannot hold every value (> 2**53)."""
    return info.kind == KIND_INT and info.dtype.itemsize == 8


def _int_zero(a: np.ndarray):
    """Integer offset of an int64/uint64 channel: its first sample (Python int)."""
    return int(a[0]) if a.size else 0


def to_f64(a: np.ndarray, kind: str, t0=None) -> np.ndarray:
    """Plot values (float64) of native channel values.

    t0: zero for timestamps (datetime64) and for int64/uint64 (Python int).
    Integers are made relative to t0 before the float conversion, so
    differences stay exact while the channel range is below 2**53.
    """
    if a.dtype == np.float64:
        return a
    if kind == KIND_INT and t0 is not None and a.dtype.itemsize == 8:
        if a.dtype.kind == "u":
            return (a.astype(np.uint64) - np.uint64(t0)).view(np.int64).astype(np.float64)
        return (a.astype(np.int64) - np.int64(t0)).astype(np.float64)
    if kind == KIND_TIME:
        if t0 is None:
            t0 = np.datetime64(0, "ns")
        return (a.astype("datetime64[ns]") - t0) / np.timedelta64(1, "ns") / 1e9
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

    _ORDER = ("open", "close", "xarr", "plot", "copy", "table", "stats", "export")

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
        self._tasks: list = []  # user tasks (generators): export, X values
        self._builds: list = []  # helper-thread futures of the current file
        self._raw_budget = RAW_PLOT_BUDGET
        self._xy_points = XY_MAX_POINTS
        self._plot_req = None  # (request, pixels) while a plot request runs
        self._prefetched: dict = {}  # cid -> (i0, i1, values), this plot request only
        self._file_stat = None  # (size, mtime, inode) at open: detect a rewrite
        self._file_warned = False
        self._stat_checked = 0.0
        self._priority: list[int] = []
        self._pool_workers = max(1, min(4, (os.cpu_count() or 2) - 2))
        self._pool = futures.ThreadPoolExecutor(max_workers=self._pool_workers, thread_name_prefix="tdms-pyramid")
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

    def request_copy(self, req: TableRequest) -> None:
        """Like request_table, in its own slot: table scrolling cannot drop it."""
        self._post("copy", (self._gen, req))

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
                while not self._slots and self._bg is None and not self._tasks and not self._quit:
                    self._cv.wait()
                if self._quit:
                    break
                for kind in self._ORDER:
                    if kind in self._slots:
                        job = (kind, self._slots.pop(kind))
                        break
            task = None
            try:
                if job is not None:
                    self._handle(*job)
                elif self._tasks:
                    # User tasks (export, X values) before the background load.
                    task = self._tasks.pop(0)
                    try:
                        next(task)
                        self._tasks.append(task)
                    except StopIteration:
                        pass
                elif self._bg is not None:
                    try:
                        next(self._bg)
                    except StopIteration:
                        self._bg = None
            except Exception as exc:  # never let the worker die
                if job is None and task is None:
                    self._bg = None
                traceback.print_exc()
                gen = self._stores_gen
                if job is not None and isinstance(job[1], tuple) and job[1] and isinstance(job[1][0], int):
                    gen = job[1][0]  # report to the file the job belongs to
                self.message.emit(gen, f"Internal error: {exc}")
        self._close_source()

    def _pending(self, kind: str) -> bool:
        return kind in self._slots or "open" in self._slots or "close" in self._slots

    def _cancelled(self, gen: int) -> bool:
        """True if a user task of file `gen` must stop (new file, close, quit)."""
        return (gen != self._gen or gen != self._stores_gen or self._quit
                or "open" in self._slots or "close" in self._slots)

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
            self._do_table(gen, req, "table")
        elif kind == "copy":
            self._do_table(gen, req, "copy")
        elif kind == "stats":
            self._do_stats(gen, req)
        elif kind == "xarr":
            self._tasks.append(self._x_task(gen, req))
        elif kind == "export":
            self._tasks.append(self._export_task(gen, req))

    # -- open / load ----------------------------------------------------------------

    def _close_source(self) -> None:
        self._bg = None
        for task in self._tasks:
            task.close()  # runs the task's cleanup (for example: delete .part file)
        self._tasks = []
        # Helper threads stop at their next block; wait so no fd is used after close.
        if self._builds:
            futures.wait(self._builds, timeout=10.0)
            self._builds = []
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
        try:
            self._setup_file(gen, src, t)
        except Exception as exc:  # never leave the GUI at "Opening ..."
            traceback.print_exc()
            self._close_source()
            try:
                src.close()
            except Exception:
                pass
            self._stores_gen = gen
            self.openFailed.emit(gen, f"{type(exc).__name__}: {exc}")

    def _setup_file(self, gen: int, src: TdmsSource, t: float) -> None:
        """Residency, stores and background load of a new file."""
        model = src.model
        try:
            st = os.stat(src.path)
            self._file_stat = (st.st_size, st.st_mtime_ns, st.st_ino)
        except OSError:
            self._file_stat = None
        self._file_warned = False
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
        builds = self._builds
        remaining = dict.fromkeys(s.info.id for s in stores if s.info.id not in chunk_ids)
        # Fragmented fast channels of one family load together: one file pass, not one per channel.
        families: dict = {}
        for s in stores:
            if s.info.id in remaining and s.fast is not None and s.fast.fragmented and s.info.length:
                families.setdefault((s.fast.family, s.to_ram), []).append(s)
        while remaining:
            pick = next((c for c in self._priority if c in remaining), None)
            if pick is None:
                pick = next(iter(remaining))
            st = stores[pick]
            group = [st]
            if st.fast is not None and st.fast.fragmented and st.info.length:
                group = [s for s in families.get((st.fast.family, st.to_ram), ()) if s.info.id in remaining] or [st]
            for s in group:
                del remaining[s.info.id]
            before = prog.done
            try:
                if len(group) > 1:
                    yield from self._load_family(gen, src, group, builds, prog)
                else:
                    yield from self._load_channel(gen, src, st, builds, prog)
            except Exception as exc:  # one bad channel must not stop the others
                for s in group:
                    s.done = True
                self.message.emit(gen, f"{st.info.label}: cannot read ({type(exc).__name__}: {exc})")
            if gen != self._stores_gen:
                return
            prog.done = before + sum(s.info.length * _itemsize(s.info) for s in group)
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
            finished, _rest = futures.wait(builds, timeout=0.02)
            for f in finished:
                builds.remove(f)
                if f.exception() is not None:
                    self.message.emit(gen, f"Overview build failed: {f.exception()}")
            prog.step("Building overview")
            yield
        for w in src.drain_warnings():
            self.message.emit(gen, w)
        self.progress.emit(gen, 1.0, f"Loaded {prog.total / 1e6:.1f} MB in {prog.elapsed():.2f} s")

    @staticmethod
    def _set_t0(src: TdmsSource, st: _Store) -> None:
        """Integer origin of time and int64/uint64 channels (exact float64 offsets)."""
        info = st.info
        if info.kind == KIND_TIME and st.t0 is None:
            st.t0 = _first_time(src.read(info.id, 0, min(info.length, 1024)))
        elif _big_int(info) and st.t0 is None:
            st.t0 = _int_zero(src.read(info.id, 0, 1))

    def _load_family(self, gen: int, src: TdmsSource, group: list, builds: list, prog):
        """Load fragmented fast channels of one family in one pass (generator).

        RAM mode: the worker fills all arrays block by block. Disk mode:
        helper threads stream the pyramids, each for a share of the channels.
        """
        n = group[0].info.length
        for st in group:
            self._set_t0(src, st)
        readers = [st.fast for st in group]
        row = sum(r.dtype.itemsize for r in readers)
        base = max(st.base for st in group)
        step = max(base, FAMILY_STEP_BYTES // row // base * base)
        if group[0].to_ram:
            arrs = [np.empty(n, dtype=st.info.dtype) for st in group]
            label = f"Loading {len(group)} channels"
            for i in range(0, n, step):
                if gen != self._stores_gen:
                    return
                k = min(step, n - i)
                fastread.read_many(readers, i, i + k, [a[i:i + k] for a in arrs])
                prog.add(k * row, label)
                yield
            for st, arr in zip(group, arrs):
                st.ram = arr
                if st.info.plottable and n >= PYR_MIN_LEN:
                    st.pyr = pyr.Pyramid(n, st.base)
                    builds.append(self._pool.submit(self._build_pyramid, gen, st, arr))
                else:
                    st.done = True
            self.channelsUpdated.emit(gen, [st.info.id for st in group if st.done])
            return
        streamed = [st for st in group if st.info.plottable and n >= PYR_MIN_LEN]
        for st in group:
            if st in streamed:
                st.pyr = pyr.Pyramid(n, st.base)
            else:
                st.done = True  # read on demand only
        threads = max(1, min(len(streamed), self._pool_workers))
        for k in range(threads):
            share = streamed[k::threads]
            builds.append(self._pool.submit(self._stream_family, gen, share, step))
        yield

    def _stream_family(self, gen: int, sts: list, step: int) -> None:
        """Helper thread: pyramids of several disk channels from one pass."""
        readers = [st.fast for st in sts]
        n = sts[0].info.length
        bufs = [np.empty(min(n, step), dtype=r.dtype) for r in readers]
        for i in range(0, n, step):
            if gen != self._stores_gen or gen != self._gen:
                return
            k = min(step, n - i)
            outs = fastread.read_many(readers, i, i + k, [b[:k] for b in bufs])
            for st, blk in zip(sts, outs):
                st.pyr.append(to_f64(blk, st.info.kind, st.t0))
        for st in sts:
            st.done = True
        if gen == self._stores_gen:
            self.channelsUpdated.emit(gen, [st.info.id for st in sts])

    def _load_channel(self, gen: int, src: TdmsSource, st: _Store, builds: list, prog):
        """Load one channel: RAM copy and/or pyramid (generator)."""
        info = st.info
        n = info.length
        if n == 0:
            st.done = True
            return
        self._set_t0(src, st)
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
            elif _big_int(info) and st.t0 is None:
                st.t0 = _int_zero(a) if off == 0 else _int_zero(src.read(cid, 0, 1))
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
            if gen != self._stores_gen or gen != self._gen:
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
            if gen != self._stores_gen or gen != self._gen:
                return
            blk = buf[: min(BLOCK, n - i)]
            st.fast.read_into(blk, i)
            p.append(to_f64(blk, info.kind, st.t0))
        st.done = True
        if gen == self._stores_gen:
            self.channelsUpdated.emit(gen, [info.id])

    # -- reading helpers (worker) ---------------------------------------------------

    def _check_file_changed(self, gen: int) -> None:
        """Warn once if the open file was rewritten, replaced or grew (at most 2x per second)."""
        if self._file_stat is None or self._file_warned or self._source is None:
            return
        now = time.monotonic()
        if now - self._stat_checked < 0.5:
            return
        self._stat_checked = now
        try:
            st = os.stat(self._source.path)
            cur = (st.st_size, st.st_mtime_ns, st.st_ino)
        except OSError:
            cur = None
        if cur != self._file_stat:
            self._file_warned = True
            self.message.emit(gen, "The file was changed on disk after it was opened. The display can mix "
                                   "old and new data. Press F5 to reload.")

    def _store(self, cid) -> _Store | None:
        """Store of a channel id, or None for an unknown id (also negative ids)."""
        if isinstance(cid, (int, np.integer)) and 0 <= cid < len(self._stores):
            return self._stores[cid]
        return None

    def _read(self, st: _Store, i0: int, i1: int) -> np.ndarray:
        if st.ram is not None:
            return st.ram[max(0, i0):max(0, i1)]
        pf = self._prefetched.get(st.info.id)
        if pf is None and self._plot_req is not None and i1 - i0 >= FAMILY_READ_MIN:
            self._read_family(st, i0, i1)
            pf = self._prefetched.get(st.info.id)
        if pf is not None and pf[0] <= max(0, i0) and min(i1, st.info.length) <= pf[1]:
            return pf[2][max(0, i0) - pf[0]:min(i1, st.info.length) - pf[0]]
        return self._source.read(st.info.id, i0, i1)

    def _read_family(self, st: _Store, i0: int, i1: int) -> None:
        """Plot of a fragmented disk channel: read [i0, i1) of the channels of its
        family that this request needs in the same range, in one file pass.

        Only loaded channels (each takes the same raw path). No read if no
        other channel needs this range.
        """
        f = st.fast
        if f is None or not f.fragmented or not st.done:
            return
        req, _px = self._plot_req
        group = [st]
        for it in req.items:
            o = self._store(it.cid)
            if (o is None or o in group or o.ram is not None or o.fast is None or not o.done
                    or o.fast.family != f.family or not o.info.plottable or it.cid in self._prefetched
                    or (isinstance(it.xmap, ArrayMap) and not it.xmap.monotonic)):
                continue
            e = min(it.e, o.info.length)
            if isinstance(it.xmap, ArrayMap):
                e = min(e, it.xmap.x.size)
            if it.xmap.index_range(req.xa, req.xb, it.s, e) == (i0, i1):
                if (len(group) + 1) * (i1 - i0) > FAMILY_READ_MAX:
                    break
                group.append(o)
        if len(group) < 2:
            return
        a, b = max(0, i0), min(i1, f.length)
        for o, arr in zip(group, fastread.read_many([o.fast for o in group], a, b)):
            self._prefetched[o.info.id] = (a, b, arr)

    def _read_f64(self, st: _Store, i0: int, i1: int) -> np.ndarray:
        """float64 values; timestamps and int64/uint64 relative to st.t0."""
        if st.t0 is None:
            if st.info.kind == KIND_TIME:
                st.t0 = _first_time(self._read(st, 0, min(st.info.length, 1024)))
            elif _big_int(st.info):
                st.t0 = _int_zero(self._read(st, 0, 1))
        return to_f64(self._read(st, i0, i1), st.info.kind, st.t0)

    @staticmethod
    def _y_offset(st: _Store) -> float:
        """Value to add back to plotted int64/uint64 values (0 for other kinds)."""
        return float(st.t0) if _big_int(st.info) and st.t0 is not None else 0.0

    def _can_read_cheap(self, st: _Store, n: int) -> bool:
        if st.ram is not None or st.fast is not None:
            return n <= RAW_DECIMATE_MAX
        return n <= RAW_DECIMATE_MAX // 8

    # -- plot -----------------------------------------------------------------------

    def _do_plot(self, gen: int, req: PlotRequest) -> None:
        px = max(16, int(req.pixels))
        self._plot_req = (req, px)
        try:
            self._do_plot_items(gen, req, px)
        finally:
            self._plot_req = None
            self._prefetched = {}

    def _do_plot_items(self, gen: int, req: PlotRequest, px: int) -> None:
        out = {}
        self._raw_budget = RAW_PLOT_BUDGET
        # X-Y plots: one point budget for the whole request, not per channel.
        n_xy = sum(1 for it in req.items if isinstance(it.xmap, ArrayMap) and not it.xmap.monotonic)
        self._xy_points = max(min(2_000, XY_MAX_POINTS), XY_MAX_POINTS // max(1, n_xy))
        self._check_file_changed(gen)
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
                res = self._plot_one(st, it, req.xa, req.xb, px)
                off = self._y_offset(st)
                if res is not None and off:
                    res = (res[0], res[1] + off, res[2])  # absolute values on the y axis
                out[it.cid] = res
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
                return self._plot_xy(st, xmap, s, e, self._xy_points)
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
                c, mn, mx, miss = p.minmax(i0, i1, b, rr, with_missing=True)
                complete = True
            elif n <= self._raw_plot_max(st) and self._take_raw_budget(st, n):
                c, mn, mx, miss = pyr.raw_minmax(self._read_f64(st, i0, i1), i0, b, with_missing=True)
                complete = True
            else:
                # Still loading: draw the finished part, refine later.
                c, mn, mx, miss = p.minmax(i0, i1, b, rr, with_missing=True)
                complete = False
        elif p is None and (n > self._raw_plot_max(st) * 8 or not self._take_raw_budget(st, n)):
            return None  # not loaded yet: drawn when channelsUpdated arrives
        elif (p is not None and p.covered >= i1 and st.ram is None and st.fast is None
              and n > self._raw_plot_max(st)):
            # npTDMS-only channel on disk (DAQmx, scaled): a raw read would decode every
            # channel of each segment. Draw the pyramid at its own resolution instead
            # (exact min/max per bucket, a little coarser than one bucket per pixel).
            c, mn, mx, miss = p.minmax(i0, i1, p.base, rr, with_missing=True)
            complete = True
        else:
            # Bucket smaller than the pyramid base: n < base * px samples.
            c, mn, mx, miss = pyr.raw_minmax(self._read_f64(st, i0, i1), i0, b, with_missing=True)
            complete = True
        x, yy = pyr.interleave(c, mn, mx, miss)
        return xmap.index_to_x(x), yy, complete

    def _take_raw_budget(self, st: _Store, n: int) -> bool:
        """Charge n raw samples of a channel without RAM copy to this plot request.

        While a file loads, this stops one view update from reading whole channels.
        """
        if st.ram is not None:
            return True
        if n > self._raw_budget:
            return False
        self._raw_budget -= n
        return True

    @staticmethod
    def _raw_plot_max(st: _Store) -> int:
        """Max raw samples for one plot update (keeps an update below ~10 ms)."""
        if st.ram is not None:
            return RAW_PLOT_MAX * 4  # RAM: compute only, no I/O
        return RAW_PLOT_MAX if st.fast is not None else RAW_PLOT_MAX // 8

    def _plot_xy(self, st: _Store, xmap: ArrayMap, s: int, e: int, max_points: int = XY_MAX_POINTS):
        n = e - s
        if n <= 0:
            return np.empty(0), np.empty(0), True
        if n > RAW_DECIMATE_MAX * 4 and st.ram is None and st.fast is None:
            return None
        y = self._read_f64(st, s, e)
        x = xmap.x[s:e]
        if n <= max_points:
            return x, y, True
        b = -(-n // (max_points // 2))
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

    def _do_table(self, gen: int, req: TableRequest, kind: str = "table") -> None:
        self._check_file_changed(gen)
        out = {}
        for cid in req.cids:
            if self._pending(kind):
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
        self._check_file_changed(gen)
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
        res = {"range": (s, s), "stats": None, "cursors": []}
        if _big_int(info):
            self._read_f64(st, 0, 0)  # sets st.t0
            res["offset"] = st.t0  # statistics are relative to this integer
        if isinstance(xmap, ArrayMap) and not xmap.monotonic:
            # X-Y plot: the samples whose x is inside [xa, xb], in any order.
            res["range"] = (s, e)
            if e <= s:
                res["stats"] = pyr.EMPTY
                res["total"] = 0
            elif self._can_read_cheap(st, e - s):
                xs = xmap.x[s:e]
                sel = (xs >= req.xa) & (xs <= req.xb)
                res["stats"] = pyr.raw_stats(self._read_f64(st, s, e)[sel])
                res["total"] = int(sel.sum())
        else:
            i0, i1 = inner_range(xmap, req.xa, req.xb, s, e)
            res["range"] = (i0, i1)
            n = i1 - i0
            res["total"] = max(0, n)
            if n <= 0:
                res["stats"] = pyr.EMPTY
            else:
                p = st.pyr
                if p is not None and p.covered >= i1:
                    res["stats"] = p.stats(i0, i1, lambda a, z: self._read_f64(st, a, z))
                elif self._can_read_cheap(st, n):
                    res["stats"] = pyr.raw_stats(self._read_f64(st, i0, i1))
        for cx in req.cursors:
            k = xmap.nearest(cx, s, e)
            if k < 0 or not covers(xmap, cx, s, e):
                res["cursors"].append(None)  # no sample of this channel at the cursor
            else:
                v = self._read(st, k, k + 1)
                res["cursors"].append((k, float(xmap.x_of(k)), v[0] if v.size else None))
        return res

    # -- x channel ------------------------------------------------------------------

    def _x_task(self, gen: int, req: XRequest):
        """Load a channel as X values; any error is answered (the GUI falls back)."""
        try:
            yield from self._x_task_body(gen, req)
        except Exception as exc:
            traceback.print_exc()
            self.xReady.emit(gen, req.seq, f"Cannot read the X values: {type(exc).__name__}: {exc}")

    def _x_task_body(self, gen: int, req: XRequest):
        """Load a channel as X values (task: one block per step, cancellable)."""
        st = self._store(req.cid)
        if st is None:
            self.xReady.emit(gen, req.seq, "Unknown channel")
            return
        info = st.info
        if not info.plottable:
            self.xReady.emit(gen, req.seq, f"{info.label} is not numeric")
            return
        if info.length * 8 > ram_budget_bytes():
            self.xReady.emit(gen, req.seq, f"{info.label} is too large to use as X axis")
            return
        x = np.empty(info.length)
        for i in range(0, info.length, BLOCK):
            if self._cancelled(gen):
                return
            j = min(info.length, i + BLOCK)
            x[i:j] = self._read_f64(st, i, j)
            if info.length > BLOCK:
                self.progress.emit(gen, min(0.999, j / info.length), f"Loading X values of {info.label}")
                yield
        if _big_int(info) and st.t0:
            x += float(st.t0)  # absolute values (rounded above 2**53)
        mono = is_monotonic(x)
        t_ref = None
        if info.kind == KIND_TIME and st.t0 is not None:
            t_ref = TimeRef.from_datetime64(st.t0)
        if info.length > BLOCK:
            self.progress.emit(gen, 1.0, f"X values of {info.label} loaded")
        self.xReady.emit(gen, req.seq, (req.cid, ArrayMap(x, mono), t_ref))

    # -- export ---------------------------------------------------------------------

    def _export_task(self, gen: int, req: ExportRequest):
        """Write the samples inside [xa, xb] as CSV (task, cancellable).

        The file is written as <path>.part and renamed only when complete,
        so a cancelled or failed export never leaves a file that looks complete.
        """
        from .formatting import format_export

        part = req.path + ".part"
        done = False
        try:
            cols = []
            for it in req.items:
                st = self._store(it.cid)
                if st is None:
                    continue
                e = min(it.e, st.info.length)
                if isinstance(it.xmap, ArrayMap):
                    e = min(e, it.xmap.x.size)
                if isinstance(it.xmap, ArrayMap) and not it.xmap.monotonic:
                    xs = it.xmap.x[it.s:e]
                    idx = it.s + np.flatnonzero((xs >= req.xa) & (xs <= req.xb))
                    cols.append((st, it, idx, idx.size))
                else:
                    i0, i1 = inner_range(it.xmap, req.xa, req.xb, it.s, e)
                    cols.append((st, it, (i0, i1), max(0, i1 - i0)))
            rows = max((c[3] for c in cols), default=0)
            timed = req.time_ref is not None
            step = 1 << 16
            with open(part, "w", encoding="utf-8", newline="") as fh:
                fh.write(",".join(_csv_cell(h) for h in req.header) + "\n")
                for r0 in range(0, rows, step):
                    if self._cancelled(gen):
                        self.exportDone.emit(gen, req.seq, "Export cancelled. No file was written.")
                        return
                    r1 = min(rows, r0 + step)
                    blocks = []
                    for st, it, sel, count in cols:
                        a1 = min(r1, count)
                        if a1 <= r0:
                            blocks.append(None)
                            continue
                        if isinstance(sel, tuple):
                            idx = np.arange(sel[0] + r0, sel[0] + a1)
                        else:
                            idx = sel[r0:a1]
                        raw = self._read(st, int(idx[0]), int(idx[-1]) + 1)
                        xs = it.xmap.index_to_x(idx)
                        times = iso_times(req.time_ref, xs) if timed and it.cid in req.time_cids else None
                        blocks.append((idx, xs, raw[idx - idx[0]], times))
                    lines = []
                    empty = ["", "", "", ""] if timed else ["", "", ""]
                    for k in range(r1 - r0):
                        cells = []
                        for blk in blocks:
                            if blk is not None and k < blk[0].size:
                                cells += [str(int(blk[0][k])), repr(float(blk[1][k]))]
                                if timed:
                                    cells.append(blk[3][k] if blk[3] is not None else "")
                                cells.append(_csv_cell(format_export(blk[2][k])))
                            else:
                                cells += empty
                        lines.append(",".join(cells))
                    fh.write("\n".join(lines) + "\n")
                    self.progress.emit(gen, min(0.999, r1 / max(1, rows)), "Exporting CSV")
                    yield
            os.replace(part, req.path)
            done = True
            self.progress.emit(gen, 1.0, f"Exported {rows} rows")
            self.exportDone.emit(gen, req.seq, f"Exported {rows} rows to {req.path}")
        except Exception as exc:
            self.exportDone.emit(gen, req.seq, f"Export failed: {exc}. No file was written.")
        finally:
            if not done:
                try:
                    os.remove(part)
                except OSError:
                    pass


_NS_SAFE_S = 9_000_000_000  # |Unix seconds| that fit datetime64[ns] with margin


def iso_times(t_ref, x) -> list[str]:
    """UTC ISO 8601 texts of t_ref + x (x: seconds, float64), exact to 1 ns.

    t_ref is a TimeRef (whole Unix seconds + fraction), so the sum keeps
    ns resolution. Non-finite x or dates outside 1685..2255 give "".
    """
    t_ref = TimeRef.of(t_ref)
    f = t_ref.frac + np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        w = np.floor(f)
        ok = np.isfinite(f) & (np.abs(w + t_ref.sec) < _NS_SAFE_S)
    f = np.where(ok, f, 0.0)
    w = np.where(ok, w, 0.0)
    ns = np.rint((f - w) * 1e9).astype(np.int64)  # 1e9 carries into the seconds below
    t = ((t_ref.sec + w.astype(np.int64)) * 1_000_000_000 + ns).astype("datetime64[ns]")
    text = np.datetime_as_string(t, unit="ns")
    return [s + "Z" if good else "" for s, good in zip(text.tolist(), ok.tolist())]


def absolute_stats(res: dict):
    """Stats of a stats result with the int64/uint64 offset added back (float).

    Min/max above 2**53 are rounded here; use res["offset"] + stats.min for
    the exact integer.
    """
    st = res.get("stats")
    off = res.get("offset")
    if st is None or off is None or st.n == 0:
        return st
    o = float(off)
    return pyr.Stats(st.n, o + st.min, o + st.max, o + st.mean, st.m2)


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
        # Decide the edges in x space with the same formula as the plot (x_of),
        # so a sample drawn exactly on the view edge is always inside.
        x_of = xmap.x_of
        i0 = max(s, min(e, int(math.ceil(fa))))
        while i0 > s and x_of(i0 - 1) >= xa:
            i0 -= 1
        while i0 < e and x_of(i0) < xa:
            i0 += 1
        i1 = max(s, min(e, int(math.floor(fb)) + 1))
        while i1 < e and x_of(i1) <= xb:
            i1 += 1
        while i1 > s and x_of(i1 - 1) > xb:
            i1 -= 1
        return i0, max(i0, i1)
    if not xmap.monotonic:
        return s, e
    seg = xmap.x[s:e]
    i0 = s + int(np.searchsorted(seg, xa, side="left"))
    i1 = s + int(np.searchsorted(seg, xb, side="right"))
    return i0, max(i0, i1)


def covers(xmap, x: float, s: int, e: int) -> bool:
    """True if x is inside the channel's samples [s, e) plus half a sample."""
    if e <= s:
        return False
    if isinstance(xmap, LinearMap):
        lo, hi = xmap.x_of(s), xmap.x_of(e - 1)
        half = 0.5 * xmap.dx
        return lo - half <= x <= hi + half
    if not xmap.monotonic:
        return True
    lo, hi = float(xmap.x[s]), float(xmap.x[e - 1])
    half0 = 0.5 * float(xmap.x[s + 1] - xmap.x[s]) if e - s > 1 else 0.0
    half1 = 0.5 * float(xmap.x[e - 1] - xmap.x[e - 2]) if e - s > 1 else 0.0
    return lo - half0 <= x <= hi + half1
