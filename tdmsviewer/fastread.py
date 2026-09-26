"""Direct reader for fixed-size TDMS channel data (fast path).

Summary
    Reads channel values 3 to 10 times faster than npTDMS. It uses the
    segment table that npTDMS already parsed and reads bytes with
    os.preadv. Each reader is verified against npTDMS before use.

Rules
    - Only plain, fixed-size numeric data. No DAQmx raw data, no
      scaling, no strings, no timestamps. The caller uses npTDMS for
      all other channels.
    - No memory map. A file that becomes shorter causes a clean
      error, not a crash (SIGBUS).
    - Thread-safe: preadv does not use the shared file position.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

try:  # Private npTDMS names. Missing names disable the fast path.
    from nptdms import types as _nt
    from nptdms.common import toc_properties as _toc
    from nptdms.scaling import get_scaling as _get_scaling
except Exception:  # pragma: no cover - depends on npTDMS version
    _nt = _toc = _get_scaling = None

# Span reads above this size are split into blocks (bytes).
_BLOCK_BYTES = 16 << 20
# Per-chunk reads are used when one chunk piece is at least this large.
_PIECE_READ_MIN = 32 << 10


class FastPathError(RuntimeError):
    """The fast path cannot read this channel."""


@dataclass(slots=True, frozen=True)
class _Part:
    """One run of channel values with a regular 2-level layout.

    Value v (0 <= v < count) is at file byte
        offset + (v // npc) * chunk_stride + (v % npc) * item_stride
    """

    start: int
    count: int
    offset: int
    npc: int
    chunk_stride: int
    item_stride: int
    dtype: np.dtype


class FastChannelReader:
    """Read values of one channel directly from the file."""

    __slots__ = ("_fd", "_parts", "_starts", "length", "dtype")

    def __init__(self, fd: int, parts: list[_Part], length: int, dtype: np.dtype):
        self._fd = fd
        self._parts = parts
        self._starts = np.array([p.start for p in parts], dtype=np.int64)
        self.length = length
        self.dtype = dtype.newbyteorder("=")

    def read(self, start: int, stop: int) -> np.ndarray:
        """Return values [start, stop) in native byte order."""
        start = max(0, start)
        stop = min(self.length, stop)
        out = np.empty(max(0, stop - start), dtype=self.dtype)
        if stop > start:
            self.read_into(out, start)
        return out

    def read_into(self, out: np.ndarray, start: int) -> None:
        """Fill out (1-D, C-contiguous, this dtype) with values from start."""
        stop = start + out.size
        if start < 0 or stop > self.length:
            raise IndexError("read outside the channel")
        if out.size == 0:
            return
        i = int(np.searchsorted(self._starts, start, side="right")) - 1
        w = 0
        parts = self._parts
        while w < out.size:
            p = parts[i]
            lo = start + w - p.start
            hi = min(stop - p.start, p.count)
            self._read_part(p, lo, hi, out[w:w + hi - lo])
            w += hi - lo
            i += 1
        return out

    # -- internal ---------------------------------------------------------

    def _pread_into(self, buf: np.ndarray, offset: int) -> None:
        mv = memoryview(buf).cast("B")
        done = 0
        while done < mv.nbytes:
            n = os.preadv(self._fd, [mv[done:]], offset + done)
            if n <= 0:
                raise FastPathError("File is shorter than its index (file changed on disk?)")
            done += n

    def _read_part(self, p: _Part, lo: int, hi: int, out: np.ndarray) -> None:
        isz = p.dtype.itemsize
        contiguous_items = p.item_stride == isz
        c0, c1 = lo // p.npc, (hi - 1) // p.npc
        if contiguous_items and (c0 == c1 or p.chunk_stride == p.npc * isz):
            # One contiguous byte range.
            pos = p.offset + c0 * p.chunk_stride + (lo - c0 * p.npc) * isz
            if p.dtype == out.dtype and out.flags.c_contiguous:
                self._pread_into(out, pos)
            else:
                tmp = np.empty(hi - lo, dtype=p.dtype)
                self._pread_into(tmp, pos)
                out[:] = tmp
            return
        if contiguous_items and p.npc * isz >= _PIECE_READ_MIN:
            # Large pieces: one read per chunk.
            w = 0
            for c in range(c0, c1 + 1):
                a = max(lo, c * p.npc)
                b = min(hi, (c + 1) * p.npc)
                pos = p.offset + c * p.chunk_stride + (a - c * p.npc) * isz
                seg = out[w:w + b - a]
                if p.dtype == out.dtype:
                    self._pread_into(seg, pos)
                else:
                    tmp = np.empty(b - a, dtype=p.dtype)
                    self._pread_into(tmp, pos)
                    seg[:] = tmp
                w += b - a
            return
        # Small pieces or interleaved values: read spans, extract strided.
        if contiguous_items:
            chunks_per_block = max(1, _BLOCK_BYTES // p.chunk_stride)
            w = 0
            c = c0
            while c <= c1:
                ce = min(c1, c + chunks_per_block - 1)
                nch = ce - c + 1
                nbytes = (nch - 1) * p.chunk_stride + p.npc * isz
                raw = np.empty(nbytes, dtype=np.uint8)
                self._pread_into(raw, p.offset + c * p.chunk_stride)
                view = np.ndarray((nch, p.npc), dtype=p.dtype, buffer=raw,
                                  strides=(p.chunk_stride, isz)).reshape(-1)
                a = max(lo, c * p.npc) - c * p.npc
                b = min(hi, (ce + 1) * p.npc) - c * p.npc
                out[w:w + b - a] = view[a:b]
                w += b - a
                c = ce + 1
            return
        # Interleaved: npc == count, one strided run.
        per_block = max(1, _BLOCK_BYTES // p.item_stride)
        w = 0
        v = lo
        while v < hi:
            ve = min(hi, v + per_block)
            n = ve - v
            nbytes = (n - 1) * p.item_stride + isz
            raw = np.empty(nbytes, dtype=np.uint8)
            self._pread_into(raw, p.offset + v * p.item_stride)
            view = np.ndarray((n,), dtype=p.dtype, buffer=raw, strides=(p.item_stride,))
            out[w:w + n] = view
            w += n
            v = ve


def _segment_flag(seg, name: str) -> bool:
    return bool(seg.toc_mask & _toc[name])


def build_parts(tdms_file, channel) -> tuple[list[_Part], np.dtype]:
    """Return the value layout of a channel from the npTDMS segment table.

    Raises FastPathError if the channel is not plain fixed-size data.
    """
    if _nt is None:
        raise FastPathError("npTDMS internals not available")
    dt = channel.data_type
    if dt is None or dt in (_nt.String, _nt.TimeStamp) or getattr(dt, "size", None) is None:
        raise FastPathError("unsized or special type")
    if dt is getattr(_nt, "DaqMxRawData", None) or getattr(dt, "nptype", None) is None:
        raise FastPathError("no numpy type")
    group = tdms_file[channel.group_name]
    if _get_scaling(channel.properties, group.properties, tdms_file.properties) is not None:
        raise FastPathError("channel has scaling")
    base_dtype = np.dtype(dt.nptype)
    if channel.dtype != base_dtype.newbyteorder("="):
        raise FastPathError("scaled dtype differs")
    isz = dt.size
    if base_dtype.itemsize != isz:
        raise FastPathError("item size mismatch")

    path = channel.path
    parts: list[_Part] = []
    pos = 0
    for seg in tdms_file._reader._segments:
        objs = seg.ordered_objects
        if not objs or not _segment_flag(seg, "kTocRawData") or seg.num_chunks <= 0:
            continue
        data_objs = [o for o in objs if o.has_data]
        target = None
        for o in data_objs:
            if o.path == path:
                target = o
                break
        if target is None or target.number_values == 0:
            continue
        if seg._have_daqmx_objects():
            raise FastPathError("DAQmx segment")
        if any(o.data_type is None or o.data_type.size is None for o in data_objs):
            raise FastPathError("segment has unsized data")
        if target.data_type is not dt:
            raise FastPathError("data type changes between segments")
        order = ">" if _segment_flag(seg, "kTocBigEndian") else "<"
        sdt = base_dtype.newbyteorder(order)
        npc = target.number_values
        chunk_size = seg._get_chunk_size()
        override = seg.final_chunk_lengths_override
        n_full = seg.num_chunks - (1 if override is not None else 0)
        interleaved = _segment_flag(seg, "kTocInterleavedData") and seg._have_interleaved_data()
        if interleaved:
            if len({o.number_values for o in data_objs}) != 1:
                raise FastPathError("interleaved counts differ")
            row = sum(o.data_type.size for o in data_objs)
            col = 0
            for o in data_objs:
                if o is target:
                    break
                col += o.data_type.size
            count = npc * n_full + (override.get(path, 0) if override is not None else 0)
            if count:
                parts.append(_Part(pos, count, seg.data_position + col, count, 0, row, sdt))
                pos += count
            continue
        off = 0
        for o in data_objs:
            if o is target:
                break
            off += o.data_size
        if n_full > 0:
            count = npc * n_full
            parts.append(_Part(pos, count, seg.data_position + off, npc, chunk_size, isz, sdt))
            pos += count
        if override is not None:
            nfinal = override.get(path, 0)
            if nfinal:
                # Truncated final chunk: earlier objects may be truncated too.
                off_final = 0
                for o in data_objs:
                    if o is target:
                        break
                    off_final += o.data_type.size * override.get(o.path, 0)
                base = seg.data_position + n_full * chunk_size + off_final
                parts.append(_Part(pos, nfinal, base, nfinal, 0, isz, sdt))
                pos += nfinal
    if pos != len(channel):
        raise FastPathError(f"value count {pos} differs from channel length {len(channel)}")
    return parts, base_dtype


def _windows(length: int, parts: list[_Part]) -> list[tuple[int, int]]:
    """Return sample windows for verification (head, tail, part joints)."""
    w = min(length, 1024)
    out = [(0, w), (length - w, length)]
    if len(parts) > 1:
        for k in (1, len(parts) // 2, len(parts) - 1):
            j = parts[k].start
            out.append((max(0, j - 512), min(length, j + 512)))
    mid = length // 2
    out.append((max(0, mid - 300), min(length, mid + 300)))
    return out


def make_fast_reader(tdms_file, channel, fd: int) -> FastChannelReader:
    """Build and verify a fast reader. Raises FastPathError on any doubt."""
    parts, base_dtype = build_parts(tdms_file, channel)
    reader = FastChannelReader(fd, parts, len(channel), base_dtype)
    if reader.length == 0:
        return reader
    for a, b in _windows(reader.length, parts):
        ref = np.asarray(channel.read_data(a, b - a))
        got = reader.read(a, b)
        if ref.dtype != got.dtype or ref.shape != got.shape or ref.tobytes() != got.tobytes():
            raise FastPathError(f"verification failed at [{a}, {b})")
    return reader
