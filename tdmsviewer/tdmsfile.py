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
from dataclasses import dataclass, field

import numpy as np
from nptdms import TdmsFile

from . import fastread

_log = logging.getLogger(__name__)

# The TDMS epoch (1904-01-01) as wf_start_time means "relative time".
_TDMS_EPOCH = np.datetime64("1904-01-01T00:00:00", "us")
_UNIX_EPOCH = np.datetime64(0, "us")


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
    def t_ref_unix(self) -> float | None:
        """t_ref as Unix seconds (float), or None."""
        if self.t_ref is None:
            return None
        return float((self.t_ref - _UNIX_EPOCH) / np.timedelta64(1, "us")) / 1e6

    def start_seconds(self, ch: ChannelInfo) -> float:
        """Channel start time relative to t_ref, in seconds (0 if unknown)."""
        off = ch.wf_start_offset or 0.0
        if ch.wf_start_time is not None and self.t_ref is not None:
            off += float((ch.wf_start_time - self.t_ref) / np.timedelta64(1, "us")) / 1e6
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
        v = v.astype("datetime64[us]")
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
        self._file = TdmsFile.open(self.path)
        index_used = bool(getattr(self._file._reader, "_index_file_path", None))
        size = os.path.getsize(self.path)
        if index_used and not self._index_matches(size):
            # A stale .tdms_index gives wrong data. Read the data file alone.
            self._file.close()
            self._fh = open(self.path, "rb")
            self._file = TdmsFile.open(self._fh)
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

    def _index_matches(self, size: int) -> bool:
        try:
            segs = self._file._reader._segments
            return not segs or segs[-1].next_segment_pos == size
        except Exception:
            return True

    def _build_model(self, size: int, index_used: bool) -> FileModel:
        groups: list[GroupInfo] = []
        channels: list[ChannelInfo] = []
        starts = []
        for g in self._file.groups():
            ginfo = GroupInfo(g.name, dict(g.properties), [])
            for c in g.channels():
                props = dict(c.properties)
                length = len(c)
                try:
                    dtype = np.dtype(c.dtype)
                except Exception:
                    dtype = np.dtype("V8")
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
                        fast = fastread.make_fast_reader(self._file, c, self._fd)
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
            path=self.path, size=size, properties=dict(self._file.properties), groups=groups,
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
        a = np.asarray(self._channels[cid].read_data(start, stop - start))
        if not a.dtype.isnative:
            a = a.astype(a.dtype.newbyteorder("="))
        return a

    def data_chunks(self):
        """Yield (channel id, offset, values) for all channels in file order."""
        by_key = {(c.group_name, c.name): i for i, c in enumerate(self._channels)}
        for chunk in self._file.data_chunks():
            for g in chunk.groups():
                for cc in g.channels():
                    cid = by_key.get((g.name, cc.name))
                    if cid is not None and len(cc):
                        a = np.asarray(cc[:])
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
