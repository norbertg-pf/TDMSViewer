"""Read-only TDMS file access: metadata model and channel readers.

Summary
    TdmsSource opens a file with npTDMS (metadata only) and gives one
    reader per channel. A verified fast reader (fastread) is used where
    possible; npTDMS is the fallback. The file is never written.

Thread model
    npTDMS objects are not thread-safe. Only the engine worker thread
    calls TdmsSource. The GUI uses only the plain FileModel data.
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass, field

import numpy as np
from nptdms import TdmsFile
from nptdms.timestamp import TdmsTimestamp

from . import fastread
from .fastread import nptdms_read
from .xaxis import TimeRef

_log = logging.getLogger(__name__)

# The TDMS epoch (1904-01-01) as wf_start_time means "relative time".
_TDMS_EPOCH = np.datetime64("1904-01-01T00:00:00", "ns")
_EPOCH_OFFSET_S = 2_082_844_800  # seconds from 1904-01-01 to 1970-01-01 (UTC)


def tdms_time_to_ns(seconds, fractions):
    """TDMS timestamps (seconds since 1904, 2**-64 s fractions) -> datetime64[ns].

    Exact integer arithmetic, rounded to the nearest ns (npTDMS itself
    truncates to 1 us). ns = round(fractions * 1e9 / 2**64).
    """
    sec = np.asarray(seconds, dtype=np.int64)
    fr = np.asarray(fractions, dtype=np.uint64)
    g = np.uint64(1_000_000_000)
    s32 = np.uint64(32)
    total = (fr >> s32) * g + (((fr & np.uint64(0xFFFFFFFF)) * g) >> s32)  # floor(fr * 1e9 / 2**32)
    ns = ((total + np.uint64(1 << 31)) >> s32).astype(np.int64)  # round(total / 2**32)
    return ((sec - _EPOCH_OFFSET_S) * 1_000_000_000 + ns).astype("datetime64[ns]")


def _to_datetime64(v):
    """npTDMS raw timestamp (scalar or TimestampArray) -> datetime64[ns]; other values unchanged."""
    if isinstance(v, TdmsTimestamp):
        try:
            return tdms_time_to_ns(v.seconds, v.second_fractions)[()]
        except (OverflowError, ValueError):
            return v.as_datetime64("us")
    if isinstance(v, np.ndarray) and v.dtype.names and "second_fractions" in v.dtype.names:
        return tdms_time_to_ns(v["seconds"], v["second_fractions"])
    return v


# -- npTDMS log capture ------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(record.getMessage())


_capture = _Capture()


def _setup_nptdms_logging() -> None:
    try:
        from nptdms.log import log_manager

        log_manager.console_handler.setLevel(logging.CRITICAL + 1)  # no stderr noise
    except Exception:  # pragma: no cover
        pass
    lg = logging.getLogger("nptdms")
    if _capture not in lg.handlers:
        lg.addHandler(_capture)


_setup_nptdms_logging()


# -- model ----------------------------------------------------------------------

KIND_FLOAT, KIND_INT, KIND_BOOL, KIND_TIME = "float", "int", "bool", "time"
KIND_COMPLEX, KIND_STRING, KIND_EMPTY = "complex", "string", "empty"
PLOTTABLE = {KIND_FLOAT, KIND_INT, KIND_BOOL, KIND_TIME, KIND_COMPLEX}


@dataclass(eq=False)
class ChannelInfo:
    """Metadata of one channel. Plain data, safe to read in any thread."""

    id: int
    group: str
    name: str
    path: str
    length: int
    kind: str
    dtype: np.dtype
    type_code: int | None
    unit: str
    properties: dict
    wf_increment: float | None
    wf_start_offset: float | None
    wf_start_time: np.datetime64 | None
    fast: bool = False

    @property
    def plottable(self) -> bool:
        return self.kind in PLOTTABLE

    @property
    def label(self) -> str:
        return f"{self.group}/{self.name}"

    def display_properties(self) -> dict:
        """Properties plus the NI pseudo-properties NI_ChannelLength and NI_DataType."""
        props = dict(self.properties)
        props.setdefault("NI_ChannelLength", self.length)
        if self.type_code is not None:
            props.setdefault("NI_DataType", self.type_code)
        return props


@dataclass(eq=False)
class GroupInfo:
    name: str
    properties: dict
    channels: list[ChannelInfo]


@dataclass(eq=False)
class FileModel:
    """Metadata of an open file."""

    path: str
    size: int
    properties: dict
    groups: list[GroupInfo]
    channels: list[ChannelInfo]
    t_ref: np.datetime64 | None  # earliest wf_start_time (absolute time origin)
    n_segments: int
    index_used: bool
    open_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def t_ref_unix(self):
        """t_ref as an exact TimeRef (Unix seconds + fraction), or None."""
        if self.t_ref is None:
            return None
        return TimeRef.from_datetime64(self.t_ref)

    def start_seconds(self, ch: ChannelInfo) -> float:
        """Channel start time relative to t_ref, in seconds (0 if unknown), 1 ns resolution."""
        off = ch.wf_start_offset or 0.0
        if ch.wf_start_time is not None and self.t_ref is not None:
            off += int((ch.wf_start_time - self.t_ref) / np.timedelta64(1, "ns")) / 1e9
        return off


def _kind(dtype: np.dtype, length: int) -> str:
    k = dtype.kind
    if k == "f":
        return KIND_FLOAT
    if k in "iu":
        return KIND_INT
    if k == "b":
        return KIND_BOOL
    if k == "M":
        return KIND_TIME
    if k == "c":
        return KIND_COMPLEX
    if k in "OUS":
        return KIND_STRING
    return KIND_EMPTY


def _props(props) -> dict:
    """Properties with raw timestamps converted to datetime64[ns]."""
    return {k: _to_datetime64(v) for k, v in props.items()}


def _float_prop(props: dict, key: str) -> float | None:
    v = props.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _time_prop(props: dict, key: str) -> np.datetime64 | None:
    v = props.get(key)
    if isinstance(v, np.datetime64):
        v = v.astype("datetime64[ns]")
        if np.isnat(v) or v == _TDMS_EPOCH:
            return None
        return v
    return None


# -- source ---------------------------------------------------------------------

class TdmsSource:
    """An open TDMS file. Worker-thread use only."""

    def __init__(self, path: str, use_fast_path: bool = True):
        self.path = os.path.abspath(path)
        self.warnings: list[str] = []
        _capture.records.clear()
        self._fd = -1
        self._fh = None
        size = os.path.getsize(self.path)
        has_index = os.path.isfile(self.path + "_index")
        try:
            self._file = TdmsFile.open(self.path, raw_timestamps=True)
        except Exception as exc:
            if not has_index:
                raise
            # A damaged .tdms_index must not block a good data file.
            self._open_without_index()
            self.warnings.append(f"The .tdms_index file cannot be read ({type(exc).__name__}: {exc}). "
                                 "It was ignored.")
        index_used = bool(getattr(self._file._reader, "_index_file_path", None))
        if index_used and not self._index_matches(size):
            # A stale or foreign .tdms_index gives wrong data. Read the data file alone.
            self._file.close()
            self._open_without_index()
            index_used = False
            self.warnings.append("The .tdms_index file does not match the data file. It was ignored.")
        try:
            status = self._file.file_status
            if status.incomplete_final_segment:
                self.warnings.append("The last segment of the file is incomplete (the write was not finished).")
        except Exception:
            pass
        self._channels = []  # npTDMS channel objects, same order as model.channels
        self.readers: list = []
        self._fast: list = []
        if use_fast_path:
            try:
                self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            except OSError:
                self._fd = -1
        self.model = self._build_model(size, index_used)
        self.warnings.extend(_dedupe(_capture.records))
        _capture.records.clear()
        self.model.warnings = list(self.warnings)

    def _open_without_index(self) -> None:
        self._fh = open(self.path, "rb")  # a file object makes npTDMS skip the index
        self._file = TdmsFile.open(self._fh, raw_timestamps=True)

    def _index_matches(self, size: int, samples: int = 256) -> bool:
        """True if the index segments agree with the data file lead-ins.

        Checks the end of the last segment and the 28-byte lead-in of the
        first, the last and up to `samples` spread segments.
        """
        try:
            segs = self._file._reader._segments
        except Exception:
            return True
        if not segs:
            return True
        if segs[-1].next_segment_pos != size:
            return False
        n = len(segs)
        picks = sorted({0, n - 1, *range(0, n, max(1, n // samples))})
        try:
            with open(self.path, "rb") as fh:
                for i in picks:
                    seg = segs[i]
                    fh.seek(seg.position)
                    lead = fh.read(28)
                    if len(lead) < 28 or lead[:4] != b"TDSm":
                        return False
                    toc = struct.unpack("<l", lead[4:8])[0]
                    if toc != seg.toc_mask:
                        return False
                    order = ">" if toc & (1 << 6) else "<"  # kTocBigEndian
                    _ver, nxt, raw = struct.unpack(order + "lQQ", lead[8:28])
                    if seg.position + 28 + raw != seg.data_position:
                        return False
                    if nxt != 0xFFFFFFFFFFFFFFFF and min(size, seg.position + 28 + nxt) != seg.next_segment_pos:
                        return False
        except (OSError, struct.error, AttributeError):
            return False
        return True

    def _build_model(self, size: int, index_used: bool) -> FileModel:
        groups: list[GroupInfo] = []
        channels: list[ChannelInfo] = []
        starts = []
        for g in self._file.groups():
            ginfo = GroupInfo(g.name, _props(g.properties), [])
            for c in g.channels():
                props = _props(c.properties)
                length = len(c)
                try:
                    dtype = np.dtype(c.dtype)
                except Exception:
                    dtype = np.dtype("V8")
                if dtype.kind == "M":
                    dtype = np.dtype("datetime64[ns]")  # converted exactly from raw timestamps
                dt = c.data_type
                code = getattr(dt, "enum_value", None) if dt is not None else None
                unit = props.get("unit_string") or props.get("Unit") or props.get("unit") or ""
                info = ChannelInfo(
                    id=len(channels), group=g.name, name=c.name, path=c.path, length=length,
                    kind=_kind(dtype, length) if length else KIND_EMPTY, dtype=dtype,
                    type_code=code, unit=str(unit), properties=props,
                    wf_increment=_float_prop(props, "wf_increment"),
                    wf_start_offset=_float_prop(props, "wf_start_offset"),
                    wf_start_time=_time_prop(props, "wf_start_time"),
                )
                fast = None
                if self._fd >= 0 and length and info.kind in (KIND_FLOAT, KIND_INT, KIND_BOOL, KIND_COMPLEX):
                    try:
                        fast = fastread.make_fast_reader(self._file, c, self._fd, nptdms_read)
                    except Exception as exc:  # any doubt -> npTDMS
                        _log.debug("fast path off for %s: %s", c.path, exc)
                        fast = None
                info.fast = fast is not None
                if info.wf_start_time is not None:
                    starts.append(info.wf_start_time)
                self._channels.append(c)
                self._fast.append(fast)
                ginfo.channels.append(info)
                channels.append(info)
            groups.append(ginfo)
        try:
            nseg = len(self._file._reader._segments)
        except Exception:
            nseg = 0
        return FileModel(
            path=self.path, size=size, properties=_props(self._file.properties), groups=groups,
            channels=channels, t_ref=min(starts) if starts else None, n_segments=nseg,
            index_used=index_used,
        )

    # -- data access (worker thread) --------------------------------------------

    def is_fast(self, cid: int) -> bool:
        return self._fast[cid] is not None

    def fast_reader(self, cid: int):
        return self._fast[cid]

    def read(self, cid: int, start: int, stop: int) -> np.ndarray:
        """Native values [start, stop) of channel cid."""
        info = self.model.channels[cid]
        start = max(0, start)
        stop = min(info.length, stop)
        if stop <= start:
            return np.empty(0, dtype=info.dtype if info.kind != KIND_EMPTY else np.float64)
        fast = self._fast[cid]
        if fast is not None:
            return fast.read(start, stop)
        # npTDMS seeks the shared file handle. Restore the position, so that a
        # data_chunks() pass that runs between requests reads on correctly.
        fh = getattr(self._file._reader, "_file", None)
        pos = fh.tell() if fh is not None else None
        try:
            a = _to_datetime64(nptdms_read(self._channels[cid], start, stop - start))
            a = np.asarray(a)
        finally:
            if pos is not None:
                fh.seek(pos)
        if not a.dtype.isnative:
            a = a.astype(a.dtype.newbyteorder("="))
        return a

    def read_into(self, cid: int, out: np.ndarray, start: int) -> None:
        """Fill out with native values of channel cid from start."""
        fast = self._fast[cid]
        if fast is not None and out.dtype == fast.dtype and out.flags.c_contiguous:
            fast.read_into(out, start)
            return
        a = self.read(cid, start, start + out.size)
        if a.size != out.size:
            raise IOError(f"short read: {a.size} of {out.size} values")
        out[:] = a

    def data_chunks(self):
        """Yield (channel id, offset, values) for all channels in file order."""
        by_key = {(c.group_name, c.name): i for i, c in enumerate(self._channels)}
        for chunk in self._file.data_chunks():
            for g in chunk.groups():
                for cc in g.channels():
                    cid = by_key.get((g.name, cc.name))
                    if cid is not None and len(cc):
                        a = np.asarray(_to_datetime64(cc[:]))
                        if not a.dtype.isnative:
                            a = a.astype(a.dtype.newbyteorder("="))
                        yield cid, cc.offset, a

    def has_mixed_layout(self) -> bool:
        """True if segments are interleaved or DAQmx (per-channel reads are costly)."""
        try:
            from nptdms.common import toc_properties as toc

            for seg in self._file._reader._segments:
                if seg.toc_mask & toc["kTocInterleavedData"]:
                    return True
                if seg.ordered_objects and seg._have_daqmx_objects():
                    return True
        except Exception:
            return False
        return False

    def drain_warnings(self) -> list[str]:
        out = _dedupe(_capture.records)
        _capture.records.clear()
        return out

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


def _dedupe(msgs: list[str]) -> list[str]:
    seen = []
    for m in msgs:
        if m not in seen:
            seen.append(m)
    return seen[:20]
