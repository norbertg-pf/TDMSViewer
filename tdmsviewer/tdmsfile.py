"""Read-only TDMS file access: metadata model and channel readers.

Summary
    TdmsSource opens a file with npTDMS (metadata only) and gives one
    reader per channel. A verified fast reader (fastread) is used where
    possible; npTDMS is the fallback. The file is never written.

Damaged and unusual files
    - Bytes after the last valid segment (zeros or garbage from a crashed
      writer) are ignored: npTDMS reads a size-limited view of the file.
    - A .tdms_index is used only if its lead-ins and metadata are the
      same bytes as in the data file.
    - Names and strings that are not UTF-8 are decoded as Windows-1252
      (npTDMS would replace the bad bytes and join different channels).
    - EXT values (80-bit extended precision) are read as float64.
    - Each problem gives one warning in FileModel.warnings (or later
      from drain_warnings), never a silent change.

Thread model
    npTDMS objects are not thread-safe. Only the engine worker thread
    calls TdmsSource. The GUI uses only the plain FileModel data.
"""

from __future__ import annotations

import io
import logging
import os
import struct
from dataclasses import dataclass, field

import numpy as np
from nptdms import TdmsFile
from nptdms import types as _nt
from nptdms.timestamp import TdmsTimestamp

from . import fastread
from .fastread import nptdms_read
from .xaxis import TimeRef

_log = logging.getLogger(__name__)

# The TDMS epoch (1904-01-01) as wf_start_time means "relative time".
_TDMS_EPOCH = np.datetime64("1904-01-01T00:00:00", "ns")
_EPOCH_OFFSET_S = 2_082_844_800  # seconds from 1904-01-01 to 1970-01-01 (UTC)
# wf_start_time before this date is a relative time (LabVIEW: epoch + offset).
_RELATIVE_BEFORE = np.datetime64("1905-01-01T00:00:00", "ns")
# Start times that spread more than this are from a wrong clock.
_MAX_START_SPREAD_NS = 365 * 86_400 * 10**9

# datetime64[ns] holds 1677-09-21 .. 2262-04-11. Whole TDMS seconds (since 1904) in that range:
_NS_MIN_TDMS_S = -9_223_372_036 + _EPOCH_OFFSET_S
_NS_MAX_TDMS_S = 9_223_372_035 + _EPOCH_OFFSET_S

# TDMS segment lead-in.
_LEAD_IN = 28
_INCOMPLETE = 0xFFFFFFFFFFFFFFFF  # next segment offset of an unfinished segment
_TOC_BIG_ENDIAN = 1 << 6
_TOC_KNOWN = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7)


# -- decode notes -----------------------------------------------------------------

class _Notes:
    """Problems found while values are decoded (worker thread only).

    The npTDMS patches below set these flags. TdmsSource resets them when
    it opens a file and turns them into one warning each.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.non_utf8 = False
        self.bad_time = False
        self.bad_ext = False
        self.paths: set | None = None  # UTF-8 object paths, while metadata is read
        self.fallback: dict | None = None  # non-UTF-8 object path -> raw bytes, same time

    def begin_parse(self) -> None:
        self.paths = set()
        self.fallback = {}

    def end_parse(self) -> None:
        self.paths = None
        self.fallback = None

    def same_paths(self) -> list[str]:
        """Paths that come from different raw bytes (npTDMS joins these objects)."""
        if not self.fallback or not self.paths:
            return []
        return sorted(s for s in self.fallback if s in self.paths)


_notes = _Notes()

_NOTE_TEXT = {
    "non_utf8": "Some names or strings are not UTF-8; they were decoded as Windows-1252.",
    "bad_time": ("Some timestamps are outside the datetime64[ns] range (1677-09-21 to 2262-04-11); "
                 "they are shown as NaT."),
    "bad_ext": "Some EXT (extended precision) values are not in the 80-bit x87 format; they are shown as NaN.",
}


# -- timestamps ---------------------------------------------------------------------

def tdms_time_to_ns(seconds, fractions):
    """TDMS timestamps (seconds since 1904, 2**-64 s fractions) -> datetime64[ns].

    Exact integer arithmetic, rounded to the nearest ns (npTDMS itself
    truncates to 1 us). ns = round(fractions * 1e9 / 2**64).
    Values outside the datetime64[ns] range become NaT (not a wrapped date).
    """
    sec = np.asarray(seconds, dtype=np.int64)
    fr = np.asarray(fractions, dtype=np.uint64)
    ok = (sec >= _NS_MIN_TDMS_S) & (sec <= _NS_MAX_TDMS_S)
    all_ok = bool(ok.all())
    if not all_ok:
        _notes.bad_time = True
        sec = np.where(ok, sec, _EPOCH_OFFSET_S)
    g = np.uint64(1_000_000_000)
    s32 = np.uint64(32)
    total = (fr >> s32) * g + (((fr & np.uint64(0xFFFFFFFF)) * g) >> s32)  # floor(fr * 1e9 / 2**32)
    ns = ((total + np.uint64(1 << 31)) >> s32).astype(np.int64)  # round(total / 2**32)
    out = ((sec - _EPOCH_OFFSET_S) * 1_000_000_000 + ns).astype("datetime64[ns]")
    if not all_ok:
        out = np.where(ok, out, np.datetime64("NaT", "ns"))
    return out


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


# -- npTDMS patches (applied once at import) --------------------------------------

def _decode_string(raw: bytes) -> str:
    """Bytes -> str: UTF-8, else Windows-1252, else Latin-1.

    Each decoder is one-to-one, so different names stay different.
    (npTDMS uses errors='replace': 'I_\\xb5' and 'I_\\xb0' both become
    'I_\\ufffd' and two channels are joined.) No log record per value.
    """
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            s = raw.decode("cp1252")
        except UnicodeDecodeError:
            s = raw.decode("latin-1")
        _notes.non_utf8 = True
        if _notes.fallback is not None and s[:1] == "/":
            _notes.fallback[s] = raw
        return s
    if _notes.paths is not None and s[:1] == "/":
        _notes.paths.add(s)
    return s


def ext_to_float64(raw, endianness: str = "<") -> np.ndarray:
    """TDMS EXT values (16 bytes each) -> float64.

    Each value is an 80-bit x87 extended float in the first 10 bytes of
    16 (little endian: 8 bytes mantissa with the explicit integer bit,
    then 2 bytes sign and exponent; 6 bytes padding). Big-endian segments
    have the same 16 bytes in reverse order. A value with padding that is
    not zero, or without the integer bit, is not x87 data: it becomes NaN
    and sets a warning.
    """
    if isinstance(raw, np.ndarray):
        b = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    else:
        b = np.frombuffer(raw, dtype=np.uint8)
    n = b.size // 16
    rows = b[:n * 16].reshape(n, 16)
    if endianness == ">":
        rows = rows[:, ::-1]
    rows = np.ascontiguousarray(rows)
    mant = rows[:, :8].copy().view("<u8").reshape(n)
    se = rows[:, 8:10].copy().view("<u2").reshape(n).astype(np.int64)
    exp = se & 0x7FFF
    top = (mant >> np.uint64(63)) != 0
    bad = rows[:, 10:].any(axis=1) | ((exp != 0) & ~top)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        v = np.ldexp(mant.astype(np.float64), (np.maximum(exp, 1) - 16383 - 63).astype(np.int32))
    special = exp == 0x7FFF
    if special.any():
        frac_zero = (mant & np.uint64(0x7FFFFFFFFFFFFFFF)) == 0
        v[special] = np.where(frac_zero[special], np.inf, np.nan)
    v = np.where((se & 0x8000) != 0, -v, v)
    if bad.any():
        v[bad] = np.nan
        _notes.bad_ext = True
    return v


def _ext_read(cls, file, endianness="<"):
    """One EXT property value -> float."""
    raw = file.read(16)
    if len(raw) != 16:
        raise ValueError("EXT value is incomplete")
    return float(ext_to_float64(raw, endianness)[0])


def _ext_from_bytes(cls, byte_array, endianness="<"):
    """EXT raw data -> float64 values, returned as 8-byte void items.

    npTDMS keeps data of types without a numpy type in lists and makes a
    'V8' array of them. Void items keep their bytes exactly; TdmsSource
    views them as float64.
    """
    return ext_to_float64(byte_array, endianness).view("V8")


class _UnknownType(KeyError):
    """A TDMS data type code that npTDMS does not know."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code

    def __str__(self):
        try:
            return f"unknown TDMS data type 0x{int(self.code):X}"
        except (TypeError, ValueError):
            return f"unknown TDMS data type {self.code!r}"


class _TypeTable(dict):
    """npTDMS type table that names the unknown code in the KeyError."""

    def __missing__(self, code):
        raise _UnknownType(code)


def _patch_nptdms() -> tuple:
    """Make npTDMS 1.11 safe for the files above. Returns the patched EXT types."""
    try:
        _nt.String._decode = staticmethod(_decode_string)
    except Exception:  # pragma: no cover - depends on npTDMS version
        _log.warning("npTDMS string decoder not patched")
    ext = []
    for name in ("ExtendedFloat", "ExtendedFloatWithUnit"):
        t = getattr(_nt, name, None)
        if t is None:  # pragma: no cover
            continue
        if getattr(t, "size", None) is None and getattr(t, "nptype", None) is None:
            t.size = 16
            t.read = classmethod(_ext_read)
            t.from_bytes = classmethod(_ext_from_bytes)
        if getattr(t, "read", None) is not None and getattr(t.read, "__func__", None) is _ext_read:
            ext.append(t)
    table = getattr(_nt, "tds_data_types", None)
    if isinstance(table, dict) and not isinstance(table, _TypeTable):
        _nt.tds_data_types = _TypeTable(table)
    return tuple(ext)


# npTDMS types whose values this module decodes (EXT, read as float64).
_EXT_TYPES = _patch_nptdms()


# -- npTDMS log capture ------------------------------------------------------

class _Capture(logging.Handler):
    """Keeps npTDMS warnings: at most MAX messages, then only a count."""

    MAX = 200

    def __init__(self):
        super().__init__(logging.WARNING)
        self.records: list[str] = []
        self.dropped = 0

    def emit(self, record):
        if len(self.records) < self.MAX:
            self.records.append(record.getMessage())
        else:
            self.dropped += 1

    def clear(self) -> None:
        self.acquire()
        try:
            self.records = []
            self.dropped = 0
        finally:
            self.release()

    def take(self) -> list[str]:
        """Unique messages (at most 20, then a count of the others); clears the store."""
        self.acquire()
        try:
            records, dropped = self.records, self.dropped
            self.records = []
            self.dropped = 0
        finally:
            self.release()
        return _dedupe(records, dropped)


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


def _dedupe(msgs: list[str], dropped: int = 0, limit: int = 20) -> list[str]:
    """Unique messages in order (at most `limit`), then "... and N more warnings"."""
    seen: set[str] = set()
    out: list[str] = []
    more = dropped
    for m in msgs:
        if m in seen:
            continue
        seen.add(m)
        if len(out) < limit:
            out.append(m)
        else:
            more += 1
    if more:
        out.append(f"... and {more} more warnings")
    return out


# -- file layout helpers --------------------------------------------------------------

class _LimitedFile(io.RawIOBase):
    """Read-only view of the first `limit` bytes of a file.

    npTDMS gets the data size with seek(0, SEEK_END) and tell()
    (nptdms.reader._get_file_size), so it sees `limit` bytes. There is
    no fileno(): code that asks the OS for the size (os.fstat) gets an
    error, not the full size.
    """

    def __init__(self, path: str, limit: int):
        super().__init__()
        self._f = open(path, "rb", buffering=0)
        self._limit = limit

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_END:
            return self._f.seek(self._limit + offset)
        return self._f.seek(offset, whence)

    def tell(self) -> int:
        return self._f.tell()

    def readinto(self, b) -> int:
        mv = memoryview(b).cast("B")
        n = min(mv.nbytes, self._limit - self._f.tell())
        if n <= 0:
            return 0
        return self._f.readinto(mv[:n])

    def close(self) -> None:
        try:
            if not self.closed:
                self._f.close()
        finally:
            super().close()


def _valid_prefix(fd: int, size: int) -> int:
    """End offset of the valid TDMS segments at the start of a file.

    Walks the 28-byte lead-ins from offset 0. Stops at a tag that is not
    'TDSm' or at a lead-in that is not possible (unknown ToC bits,
    metadata larger than the segment). A last segment with an unknown
    size (0xFFFFFFFFFFFFFFFF) or a size past the end of the file goes to
    the end of the file.
    """
    pos = 0
    while pos + _LEAD_IN <= size:
        lead = fastread.pread(fd, _LEAD_IN, pos)
        if len(lead) < _LEAD_IN or lead[:4] != b"TDSm":
            break
        toc = struct.unpack_from("<l", lead, 4)[0]
        if toc & ~_TOC_KNOWN:
            break
        order = ">" if toc & _TOC_BIG_ENDIAN else "<"
        _ver, nxt, raw = struct.unpack_from(order + "lQQ", lead, 8)
        if nxt == _INCOMPLETE:
            return size
        if raw > nxt:
            break
        end = pos + _LEAD_IN + nxt
        if end >= size:
            return size
        pos = end
    return pos


def _parsed_prefix(path: str, limit: int) -> int:
    """End of the segments that npTDMS can parse in the first `limit` bytes.

    npTDMS keeps the segments before the one that fails. Returns `limit`
    if all segments can be parsed, 0 if none.
    """
    from nptdms.reader import TdmsReader

    fh = io.BufferedReader(_LimitedFile(path, limit), buffer_size=1 << 16)
    try:
        reader = TdmsReader(fh)
        try:
            reader.read_metadata()
            return limit
        except Exception:
            segs = reader._segments or []
            return segs[-1].next_segment_pos if segs else 0
    except Exception:
        return 0
    finally:
        fh.close()


def _gap_cuts(channel) -> np.ndarray:
    """Channel offsets where a read must restart (npTDMS 1.11 bug).

    read_raw_data_for_channel loses count of the segments when a segment
    without values of this channel is inside the window. It then reads
    wrong chunks; for strings and EXT values without an error. A read
    that does not cross such a gap is correct.
    """
    try:
        rd = channel._reader
        path = channel.path
        if path not in rd._segment_channel_offsets:
            rd._build_index(path)
        _first, offs = rd._segment_channel_offsets[path]
        offs = np.asarray(offs, dtype=np.int64)
    except Exception:
        return np.empty(0, dtype=np.int64)
    if offs.size < 2:
        return np.empty(0, dtype=np.int64)
    gaps = np.flatnonzero(np.diff(offs) == 0) + 1
    return np.unique(offs[gaps])


def _unknown_type(exc: BaseException) -> _UnknownType | None:
    """The _UnknownType error in the cause chain of exc, or None."""
    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, _UnknownType):
            return e
        e = e.__cause__ or e.__context__
    return None


def _open_error(exc: BaseException) -> Exception | None:
    """A clearer exception for an open error, or None."""
    u = _unknown_type(exc)
    if u is not None:
        return ValueError(f"The file uses a data type that is not a known TDMS type ({u}). "
                          "The file cannot be read.")
    return None


def _error_text(exc: BaseException) -> str:
    u = _unknown_type(exc)
    return str(u) if u is not None else f"{type(exc).__name__}: {exc}"


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
    t_ref: np.datetime64 | None  # absolute time of x == 0: earliest wf_start_time (rules: _time_reference)
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


def _time_prop(props: dict, key: str) -> tuple[np.datetime64 | None, float]:
    """(absolute time or None, relative seconds) of a timestamp property.

    A time before 1905-01-01 is a relative time (LabVIEW writes relative
    waveforms as 1904-01-01 + t0). It gives no absolute time; its
    distance to 1904-01-01 is returned as relative seconds.
    """
    v = props.get(key)
    if not isinstance(v, np.datetime64):
        return None, 0.0
    v = v.astype("datetime64[ns]")
    if np.isnat(v):
        return None, 0.0
    if v < _RELATIVE_BEFORE:
        return None, int((v - _TDMS_EPOCH) / np.timedelta64(1, "ns")) / 1e9
    return v, 0.0


def _time_reference(channels: list[ChannelInfo]) -> tuple[np.datetime64 | None, list[ChannelInfo]]:
    """(t_ref, channels far from t_ref).

    Rule: t_ref is the earliest wf_start_time of all channels. Relative
    times (before 1905, see _time_prop) are not used. If the start times
    spread over more than 1 year (for example one device with a wrong
    clock), t_ref is the earliest start within 1 year of the median start
    (weighted by channel length, so that the long, fast channels keep
    their sample resolution). The channels outside are returned for a
    warning.
    """
    timed = [c for c in channels if c.wf_start_time is not None]
    if not timed:
        return None, []
    ns = [int(c.wf_start_time.astype("datetime64[ns]").astype(np.int64)) for c in timed]
    lo = min(ns)
    if max(ns) - lo <= _MAX_START_SPREAD_NS:
        return np.datetime64(lo, "ns"), []
    order = sorted(range(len(timed)), key=lambda i: ns[i])
    half = sum(max(1, c.length) for c in timed) / 2
    acc = 0
    med = ns[order[-1]]
    for i in order:
        acc += max(1, timed[i].length)
        if acc >= half:
            med = ns[i]
            break
    near = [abs(v - med) <= _MAX_START_SPREAD_NS for v in ns]
    t_ref = min(v for v, ok in zip(ns, near) if ok)
    far = [c for c, ok in zip(timed, near) if not ok]
    return np.datetime64(t_ref, "ns"), far


# -- source ---------------------------------------------------------------------

class TdmsSource:
    """An open TDMS file. Worker-thread use only."""

    def __init__(self, path: str, use_fast_path: bool = True):
        self.path = os.path.abspath(path)
        self.warnings: list[str] = []
        _capture.clear()
        _notes.reset()
        self._reported: set[str] = set()
        self._fd = -1
        self._fh = None
        self._file = None
        self._tail_note = None  # why bytes at the end were ignored (if not the default text)
        size = os.path.getsize(self.path)
        try:
            index_used = self._open(size)
        except Exception as exc:
            _notes.end_parse()
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            better = _open_error(exc)
            if better is not None:
                raise better from exc
            raise
        for s in _notes.same_paths()[:5]:
            self.warnings.append(f"Different objects have the same path {s} after decoding. "
                                 "Their data and properties are joined.")
        _notes.end_parse()
        try:
            segs = self._file._reader._segments
            used = segs[-1].next_segment_pos if segs else 0
        except Exception:
            used = size
        if used < size:
            self.warnings.append(self._tail_note or
                                 f"{size - used} bytes after offset {used} are not TDMS data and were ignored.")
        try:
            status = self._file.file_status
            if status.incomplete_final_segment:
                self.warnings.append("The last segment of the file is incomplete (the write was not finished).")
        except Exception:
            pass
        self._channels = []  # npTDMS channel objects, same order as model.channels
        self._ext: set[int] = set()  # channel ids with EXT values (read as V8, viewed as float64)
        self._cuts: dict[int, np.ndarray] = {}
        self.readers: list = []
        self._fast: list = []
        if use_fast_path:
            try:
                self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0))
            except OSError:
                self._fd = -1
        self.model = self._build_model(size, index_used)
        self.warnings.extend(_capture.take())
        self.warnings.extend(self._note_warnings())
        self.model.warnings = list(self.warnings)

    # -- open -----------------------------------------------------------------

    def _open(self, size: int) -> bool:
        """Open self._file. Returns True if the .tdms_index is used."""
        has_index = os.path.isfile(self.path + "_index")
        _notes.begin_parse()
        error = None
        try:
            self._file = TdmsFile.open(self.path, raw_timestamps=True)
        except Exception as exc:
            error = exc
        if error is not None:
            if not has_index:
                self._open_limited(size, error)
                return False
            # A damaged .tdms_index must not block a good data file.
            self._open_without_index(size)
            self.warnings.append(f"The .tdms_index file cannot be read ({type(error).__name__}: {error}). "
                                 "It was ignored.")
            return False
        index_used = bool(getattr(self._file._reader, "_index_file_path", None))
        if index_used and not self._index_matches(size):
            # A stale or foreign .tdms_index gives wrong data. Read the data file alone.
            self._file.close()
            self._open_without_index(size)
            self.warnings.append("The .tdms_index file does not match the data file. It was ignored.")
            return False
        return index_used

    def _open_without_index(self, size: int) -> None:
        _notes.begin_parse()
        fh = open(self.path, "rb")  # a file object makes npTDMS skip the index
        try:
            self._file = TdmsFile.open(fh, raw_timestamps=True)
            self._fh = fh
            return
        except Exception as exc:
            fh.close()
            error = exc
        self._open_limited(size, error)

    def _open_limited(self, size: int, error: Exception) -> None:
        """Open the valid start of a file with bytes after its last segment.

        1. Walk the lead-ins (_valid_prefix) and open the part before the
           first bad lead-in.
        2. If that fails too (a lead-in that looks correct, with bad
           metadata), open the segments that npTDMS can parse.
        Raises `error` (the first error) if this does not help.
        """
        try:
            with open(self.path, "rb", buffering=0) as fh:
                limit = _valid_prefix(fh.fileno(), size)
        except OSError:
            limit = 0
        if 0 < limit < size and self._try_limited(limit):
            return
        if limit > 0:
            good = _parsed_prefix(self.path, limit)
            if 0 < good < limit and self._try_limited(good):
                self._tail_note = (f"{size - good} bytes after offset {good} cannot be read "
                                   f"({_error_text(error)}) and were ignored.")
                return
        raise error

    def _try_limited(self, limit: int) -> bool:
        """Open the first `limit` bytes with npTDMS. False if this fails."""
        _notes.begin_parse()
        fh = io.BufferedReader(_LimitedFile(self.path, limit), buffer_size=1 << 16)
        try:
            self._file = TdmsFile.open(fh, raw_timestamps=True)
        except Exception as exc:
            _log.debug("open of the first %d bytes failed: %s", limit, exc)
            fh.close()
            return False
        self._fh = fh
        return True

    def _index_matches(self, size: int, samples: int = 256) -> bool:
        """True if the .tdms_index describes this data file.

        The index repeats the lead-in (tag TDSh) and the metadata of each
        segment. For the first, the last and up to `samples` spread
        segments, these bytes must be the same in the data file, and the
        lead-in must agree with the segment table. Bytes after the last
        indexed segment must not be a segment (a stale index); other
        bytes there are not TDMS data and are ignored.
        """
        try:
            segs = self._file._reader._segments
        except Exception:
            return True
        if not segs:
            return size == 0
        end = segs[-1].next_segment_pos
        if end > size:
            return False
        n = len(segs)
        picks = {0, n - 1, *range(0, n, max(1, n // samples))}
        try:
            with open(self.path, "rb", buffering=0) as fh, open(self.path + "_index", "rb", buffering=0) as ih:
                fd, ifd = fh.fileno(), ih.fileno()
                if end < size and (fastread.pread(fd, 4, end) == b"TDSm" or _valid_prefix(fd, size) != end):
                    return False
                off = 0  # position of the segment in the index file
                for i, seg in enumerate(segs):
                    head = seg.data_position - seg.position  # lead-in + metadata
                    if i in picks and not _segment_matches(seg, fd, ifd, off, head, end):
                        return False
                    off += head
        except (OSError, struct.error, AttributeError):
            return False
        return True

    # -- model ----------------------------------------------------------------

    def _build_model(self, size: int, index_used: bool) -> FileModel:
        groups: list[GroupInfo] = []
        channels: list[ChannelInfo] = []
        bad_dx = []
        fast_map = self._make_fast_readers()
        for g in self._file.groups():
            ginfo = GroupInfo(g.name, _props(g.properties), [])
            for c in g.channels():
                props = _props(c.properties)
                length = len(c)
                dt = c.data_type
                ext = dt is not None and dt in _EXT_TYPES
                try:
                    dtype = np.dtype(c.dtype)
                except Exception:
                    dtype = np.dtype("V8")
                if dtype.kind == "M":
                    dtype = np.dtype("datetime64[ns]")  # converted exactly from raw timestamps
                if ext:
                    dtype = np.dtype(np.float64)  # decoded from 80 bits
                code = getattr(dt, "enum_value", None) if dt is not None else None
                unit = props.get("unit_string") or props.get("Unit") or props.get("unit") or ""
                label = f"{g.name}/{c.name}"
                start, rel = _time_prop(props, "wf_start_time")
                offset = _float_prop(props, "wf_start_offset")
                if rel:
                    offset = (offset or 0.0) + rel
                dx = _float_prop(props, "wf_increment")
                if "wf_increment" in props and not (dx is not None and dx > 0):
                    bad_dx.append((label, props["wf_increment"]))
                    dx = None
                info = ChannelInfo(
                    id=len(channels), group=g.name, name=c.name, path=c.path, length=length,
                    kind=_kind(dtype, length) if length else KIND_EMPTY, dtype=dtype,
                    type_code=code, unit=str(unit), properties=props,
                    wf_increment=dx, wf_start_offset=offset, wf_start_time=start,
                )
                if ext:
                    self._ext.add(info.id)
                fast = None
                if length and not ext and info.kind in (KIND_FLOAT, KIND_INT, KIND_BOOL, KIND_COMPLEX):
                    fast = fast_map.get(c.path)
                    if not isinstance(fast, fastread.FastChannelReader):  # any doubt -> npTDMS
                        if fast is not None:
                            _log.debug("fast path off for %s: %s", c.path, fast)
                        fast = None
                info.fast = fast is not None
                self._channels.append(c)
                self._fast.append(fast)
                ginfo.channels.append(info)
                channels.append(info)
            groups.append(ginfo)
        for label, v in bad_dx[:5]:
            shown = repr(v) if isinstance(v, str) else v
            self.warnings.append(f"{label}: invalid wf_increment {shown} ignored (1 sample = 1 s)")
        if len(bad_dx) > 5:
            self.warnings.append(f"... and {len(bad_dx) - 5} more channels with an invalid wf_increment")
        t_ref, far = _time_reference(channels)
        if far:
            c = far[0]
            more = f" (and {len(far) - 1} more channels)" if len(far) > 1 else ""
            self.warnings.append(
                f"{c.label}{more}: wf_start_time {c.wf_start_time} is more than 1 year away from the "
                f"other channels. The time reference is {t_ref}. Check the clock of the data source.")
        try:
            nseg = len(self._file._reader._segments)
        except Exception:
            nseg = 0
        return FileModel(
            path=self.path, size=size, properties=_props(self._file.properties), groups=groups,
            channels=channels, t_ref=t_ref, n_segments=nseg, index_used=index_used,
        )

    def _make_fast_readers(self) -> dict:
        """Fast readers of all channels (one layout scan): path -> reader or exception."""
        if self._fd < 0:
            return {}
        chans = [c for g in self._file.groups() for c in g.channels() if len(c)]
        try:
            made = fastread.make_fast_readers(self._file, chans, self._fd, nptdms_read)
        except Exception as exc:  # any doubt -> npTDMS
            _log.debug("fast path off: %s", exc)
            return {}
        return {c.path: r for c, r in zip(chans, made)}

    def _note_warnings(self) -> list[str]:
        """One warning per decode problem and file (see _Notes)."""
        out = []
        for key, text in _NOTE_TEXT.items():
            if getattr(_notes, key) and key not in self._reported:
                self._reported.add(key)
                out.append(text)
            setattr(_notes, key, False)
        return out

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
            a = self._nptdms_values(cid, start, stop)
        finally:
            if pos is not None:
                fh.seek(pos)
        if not a.dtype.isnative:
            a = a.astype(a.dtype.newbyteorder("="))
        return a

    def _nptdms_values(self, cid: int, start: int, stop: int) -> np.ndarray:
        """Values [start, stop) from npTDMS; one read per run without gaps (see _gap_cuts)."""
        ch = self._channels[cid]
        cuts = self._cuts.get(cid)
        if cuts is None:
            cuts = self._cuts[cid] = _gap_cuts(ch)
        bounds = [int(v) for v in cuts[(cuts > start) & (cuts < stop)]] + [stop]
        parts = []
        a = start
        for b in bounds:
            parts.append(np.asarray(_to_datetime64(nptdms_read(ch, a, b - a))))
            a = b
        out = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if out.shape[0] > stop - start:
            raise IOError(f"{ch.path}: npTDMS gave {out.shape[0]} values for a window of {stop - start}")
        if cid in self._ext:
            out = _ext_view(out)
        return out

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
                        if cid in self._ext:
                            a = _ext_view(a)
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
        return _capture.take() + self._note_warnings()

    def close(self) -> None:
        try:
            if self._file is not None:
                self._file.close()
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


def _ext_view(a: np.ndarray) -> np.ndarray:
    """EXT channel data from npTDMS (8-byte void items) as float64."""
    if a.dtype.kind == "V" and a.dtype.itemsize == 8:
        return np.ascontiguousarray(a).view(np.float64)
    return a.astype(np.float64)


def _segment_matches(seg, fd: int, ifd: int, off: int, head: int, end: int) -> bool:
    """True if lead-in and metadata of one segment are the same in data and index file."""
    data = fastread.pread(fd, head, seg.position)
    idx = fastread.pread(ifd, head, off)
    if len(data) != head or len(idx) != head or head < _LEAD_IN:
        return False
    if data[:4] != b"TDSm" or idx[:4] != b"TDSh" or data[4:] != idx[4:]:
        return False
    toc = struct.unpack_from("<l", data, 4)[0]
    if toc != seg.toc_mask:
        return False
    order = ">" if toc & _TOC_BIG_ENDIAN else "<"
    _ver, nxt, raw = struct.unpack_from(order + "lQQ", data, 8)
    if _LEAD_IN + raw != head:
        return False
    if nxt != _INCOMPLETE and min(end, seg.position + _LEAD_IN + nxt) != seg.next_segment_pos:
        return False
    return True
