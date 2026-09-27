"""Robustness tests for tdmsviewer.tdmsfile: damaged and unusual files.

Each test is a regression test for one review finding:
trailing bytes, foreign .tdms_index, time reference, non-UTF-8 names,
warning flood, EXT and unknown types, out-of-range timestamps, bad
wf_increment, and the npTDMS over-read of channels with gaps.
"""

from __future__ import annotations

import gc
import io
import logging
import math
import os
import shutil
import struct
import time

import numpy as np
import pytest

import tdms_builders as tb
from tdms_builders import Obj, Prop, TdmsBuilder, obj_path
from tdmsviewer import tdmsfile
from tdmsviewer.tdmsfile import KIND_FLOAT, KIND_TIME, TdmsSource, tdms_time_to_ns

pytestmark = pytest.mark.usefixtures("read_path")  # POSIX and Windows read paths

NON_UTF8 = "Some names or strings are not UTF-8; they were decoded as Windows-1252."


def _fds_on(path) -> int:
    targets = {os.path.realpath(path), os.path.realpath(str(path) + "_index")}
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")) in targets:
                n += 1
        except OSError:
            pass
    return n


def _same_as_nptdms(src: TdmsSource, ref_path: str, rng, count: int = 20) -> None:
    """Every channel of src, many windows: same bytes as npTDMS on ref_path (data file only)."""
    ref = tb.nptdms_full(ref_path, use_index=False)
    ref_channels = [c for g in ref.groups() for c in g.channels()]
    assert [c.path for c in ref_channels] == [c.path for c in src.model.channels]
    for info, rc in zip(src.model.channels, ref_channels):
        n = len(rc)
        assert info.length == n, info.path
        for a, b in tb.windows(n, rng, count=count):
            lo, hi = tb.clamp_window(n, a, b)
            got = src.read(info.id, a, b)
            if n == 0:
                assert got.size == 0
                continue
            tb.assert_same_values(got, np.asarray(rc[lo:hi]), f"{info.path} [{a}, {b})")


def _three_segments(rng) -> TdmsBuilder:
    b = TdmsBuilder()
    p, q = obj_path("G", "x"), obj_path("G", "y")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 300, rng), props=tb.wf_props(0.001)),
                                          Obj(q, tb.random_values("i2", 300, rng))])
    b.raw_segment({p: tb.random_values("f8", 300, rng), q: tb.random_values("i2", 300, rng)})
    b.segment([Obj(p, tb.random_values("f8", 100, rng), props={"late": 1}),
               Obj(q, tb.random_values("i2", 100, rng))], new_obj_list=False)
    return b


# -- 1. bytes after the last segment ---------------------------------------------------------

TAILS = {
    "zeros": b"\0" * 65536,  # preallocated file of a crashed writer
    "garbage": bytes(np.random.default_rng(7).integers(0, 256, 1000, dtype=np.uint8)),
    "short": b"\0" * 5,  # shorter than a lead-in
    "tdsm_garbage": b"TDSm" + b"\xff" * 1000,  # tag, then an impossible lead-in
}


@pytest.mark.parametrize("index", [False, True])
@pytest.mark.parametrize("tail", list(TAILS))
def test_bytes_after_last_segment_are_ignored(tail, index, tmp_path, open_source, rng):
    b = _three_segments(rng)
    clean = b.write(tmp_path / "clean.tdms")
    path = b.write(tmp_path / "tail.tdms", index=index)
    with open(path, "ab") as fh:
        fh.write(TAILS[tail])
    src = open_source(path)
    m = src.model
    assert m.size == os.path.getsize(path)
    msg = f"{len(TAILS[tail])} bytes after offset {b.size} are not TDMS data and were ignored."
    assert msg in m.warnings, m.warnings
    assert m.n_segments == 3
    assert [c.length for c in m.channels] == [700, 700]
    assert all(c.fast for c in m.channels)
    # 'TDSm' after the indexed segments looks like a newer segment: the index is not used.
    assert m.index_used is (index and tail != "tdsm_garbage")
    for c in m.channels:
        tb.assert_same_values(src.read(c.id, 0, c.length), b.expected(c.path), c.path)
    _same_as_nptdms(src, clean, rng)
    assert src.model.channels[0].properties["late"] == 1


def test_bytes_after_last_segment_npTDMS_path(tmp_path, open_source, rng):
    """Channels that npTDMS reads (no fast path) read the limited file correctly."""
    b = _three_segments(rng)
    clean = b.write(tmp_path / "clean.tdms")
    path = b.write(tmp_path / "tail.tdms")
    with open(path, "ab") as fh:
        fh.write(b"\0" * 4096)
    src = open_source(path, use_fast_path=False)
    assert not any(c.fast for c in src.model.channels)
    _same_as_nptdms(src, clean, rng)
    parts = {c.id: [] for c in src.model.channels}
    for cid, off, a in src.data_chunks():
        parts[cid].append(a)
    for c in src.model.channels:
        tb.assert_same_values(np.concatenate(parts[c.id]), b.expected(c.path), c.path)


def test_bytes_after_last_segment_fds_released(tmp_path, rng):
    path = _three_segments(rng).write(tmp_path / "tail.tdms")
    with open(path, "ab") as fh:
        fh.write(b"\0" * 4096)
    gc.collect()
    src = TdmsSource(path)
    src.close()
    gc.collect()
    assert _fds_on(path) == 0


def test_plausible_lead_in_with_bad_metadata_after_last_segment(tmp_path, open_source, rng):
    """A 'TDSm' lead-in that looks correct, with metadata that npTDMS cannot parse."""
    b = _three_segments(rng)
    path = b.write(tmp_path / "bogus.tdms")
    meta = (struct.pack("<I", 1) + tb._string("/'G'/'z'", "<")  # one object
            + struct.pack("<IIIQ", 20, 0x99, 1, 10))  # raw data index with an unknown type code
    lead = b"TDSm" + struct.pack("<iiQQ", 0x0E, 4713, len(meta) + 80, len(meta))
    with open(path, "ab") as fh:
        fh.write(lead + meta + b"\0" * 80 + b"\x5a" * 300)
    src = open_source(path)
    assert [c.length for c in src.model.channels] == [700, 700]
    w = [m for m in src.model.warnings if f"after offset {b.size} cannot be read" in m]
    assert len(w) == 1 and "0x99" in w[0], src.model.warnings
    for c in src.model.channels:
        tb.assert_same_values(src.read(c.id, 0, c.length), b.expected(c.path), c.path)


def test_bad_metadata_in_a_middle_segment_opens_the_start(tmp_path, open_source, rng):
    """Segment 2 of 3 has an unknown data type: segment 1 is shown, with a warning."""
    b = TdmsBuilder()
    p = obj_path("G", "x")
    first = b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 100, rng))])
    b.segment([Obj(obj_path("G", "odd"), np.zeros(4, np.float32), type_code=tb.T_F32)], new_obj_list=False,
              data={p: tb.random_values("f8", 100, rng)})
    b.raw_segment({p: tb.random_values("f8", 100, rng), obj_path("G", "odd"): np.zeros(4, np.float32)})
    raw = bytearray(b.data_bytes())
    odd = tb._string(obj_path("G", "odd"), "<")
    code_at = b.segments[1].position + 28 + 4 + len(odd) + 4  # object count, path, index length
    assert raw[code_at:code_at + 4] == struct.pack("<I", tb.T_F32)
    raw[code_at:code_at + 4] = struct.pack("<I", 0x77)
    path = tmp_path / "middle.tdms"
    path.write_bytes(bytes(raw))
    src = open_source(path)
    end = len(first.lead_in + first.meta + first.raw)
    assert [(c.name, c.length) for c in src.model.channels] == [("x", 100)]
    assert any(f"after offset {end} cannot be read (unknown TDMS data type 0x77)" in m
               for m in src.model.warnings), src.model.warnings
    tb.assert_same_values(src.read(0, 0, 100), b.expected(p)[:100], "x")


def test_file_without_tdms_data_still_fails(tmp_path):
    p = tmp_path / "zeros.tdms"
    p.write_bytes(b"\0" * 4096)
    with pytest.raises(ValueError):
        TdmsSource(str(p))


def test_limited_file_object(tmp_path):
    """npTDMS sees only `limit` bytes; there is no fileno() that shows more."""
    from nptdms.reader import _get_file_size

    p = tmp_path / "f.bin"
    p.write_bytes(bytes(range(256)) * 4)
    fh = io.BufferedReader(tdmsfile._LimitedFile(str(p), 300))
    try:
        assert _get_file_size(fh) == 300
        assert fh.tell() == 0
        assert fh.read(10) == bytes(range(10))
        fh.seek(290)
        assert fh.read(100) == bytes(range(34, 44))
        assert fh.read(1) == b""
        fh.seek(-4, os.SEEK_END)
        assert fh.tell() == 296 and fh.read() == bytes(range(40, 44))
        fh.seek(500)
        assert fh.read(5) == b""
        buf = bytearray(8)
        fh.seek(296)
        assert fh.readinto(buf) == 4 and bytes(buf[:4]) == bytes(range(40, 44))
        with pytest.raises(OSError):
            fh.fileno()
    finally:
        fh.close()
    assert fh.closed


def test_valid_prefix_walk(tmp_path, rng):
    b = _three_segments(rng)
    path = b.write(tmp_path / "w.tdms")
    size = os.path.getsize(path)
    with open(path, "ab") as fh:
        fh.write(b"\0" * 100)
    with open(path, "rb") as fh:
        assert tdmsfile._valid_prefix(fh.fileno(), size + 100) == size
        assert tdmsfile._valid_prefix(fh.fileno(), size) == size
        # A cut last segment runs to the end of the file.
        assert tdmsfile._valid_prefix(fh.fileno(), size - 10) == size - 10
    inc = TdmsBuilder()
    inc.segment(tb.header_objects(["G"]) + [Obj(obj_path("G", "x"), np.arange(10.0))], incomplete=True)
    p2 = inc.write(tmp_path / "inc.tdms")
    with open(p2, "ab") as fh:
        fh.write(b"\0" * 50)
    with open(p2, "rb") as fh:
        n = os.path.getsize(p2)
        assert tdmsfile._valid_prefix(fh.fileno(), n) == n  # size unknown: all of the file


# -- 2. .tdms_index of another run ----------------------------------------------------------------

def _run(path, start, dt, values, test_no):
    b = TdmsBuilder()
    for k in range(4):
        objs = tb.header_objects(["g"], file_props={"test_no": test_no}) if k == 0 else []
        objs.append(Obj(obj_path("g", "I"), values[k * 1000:(k + 1) * 1000],
                        props=tb.wf_props(dt, start_time=start) if k == 0 else {}))
        b.segment(objs)
    b.write(path, index=True)
    return b


def test_index_of_other_run_with_same_layout_is_ignored(tmp_path, open_source, rng):
    a_path, b_path = str(tmp_path / "test_011.tdms"), str(tmp_path / "test_012.tdms")
    _run(a_path, "2026-07-27T09:00:00", 1e-3, rng.standard_normal(4000), "011")
    b = _run(b_path, "2026-07-28T12:05:36", 1e-4, rng.standard_normal(4000) + 100, "012")
    shutil.copyfile(b_path, a_path)  # data of run 012, index of run 011
    src = open_source(a_path)
    m = src.model
    assert m.index_used is False
    assert "The .tdms_index file does not match the data file. It was ignored." in m.warnings
    assert m.properties["test_no"] == "012"
    c = m.channels[0]
    assert c.wf_increment == 1e-4
    assert c.wf_start_time == np.datetime64("2026-07-28T12:05:36.000000500", "ns")
    tb.assert_same_values(src.read(0, 0, c.length), b.expected(c.path), c.path)
    assert open_source(b_path).model.index_used is True  # its own index is fine


def test_index_metadata_difference_in_last_segment(tmp_path, open_source, rng):
    """Same sizes, one changed property byte in the last segment's metadata."""
    b = TdmsBuilder()
    p = obj_path("G", "x")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 50, rng))])
    for k in range(300):
        b.segment([Obj(p, tb.random_values("f8", 50, rng), props={"step": k % 7})], new_obj_list=False)
    path = b.write(tmp_path / "many.tdms", index=True)
    assert open_source(path).model.index_used is True
    idx = bytearray(open(path + "_index", "rb").read())
    idx[-1] ^= 0x01  # last property value of the last segment
    open(path + "_index", "wb").write(bytes(idx))
    src = open_source(path)
    assert src.model.index_used is False
    assert src.model.channels[0].properties["step"] == 299 % 7


def test_empty_index_is_ignored(tmp_path, open_source, rng):
    b = _three_segments(rng)
    path = b.write(tmp_path / "e.tdms")
    open(path + "_index", "wb").close()
    src = open_source(path)
    assert src.model.index_used is False
    assert [c.length for c in src.model.channels] == [700, 700]


# -- 3. time reference -----------------------------------------------------------------------------

HALF_SECOND = 1 << 63  # 0.5 s in 2**-64 fractions


def test_relative_start_time_does_not_move_t_ref(write_tdms, open_source):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["daq", "calc"]) + [
        Obj(obj_path("daq", "U"), np.arange(100000.0), props=tb.wf_props(1e-7, start_time="2026-07-28T12:05:36")),
        Obj(obj_path("calc", "dU"), np.arange(1000.0),
            props=tb.wf_props(1e-3, start_time=tb.RawTime(0, HALF_SECOND))),
        Obj(obj_path("calc", "rel2"), np.arange(10.0),
            props=tb.wf_props(1.0, 1.5, start_time=tb.RawTime(3600, HALF_SECOND))),
    ])
    src = open_source(write_tdms(b))
    m = src.model
    u, du, rel2 = m.channels
    assert m.t_ref == u.wf_start_time == np.datetime64("2026-07-28T12:05:36.000000500", "ns")
    assert du.wf_start_time is None and du.wf_start_offset == 0.5
    assert rel2.wf_start_time is None and rel2.wf_start_offset == pytest.approx(1.5 + 3600.5)
    assert m.start_seconds(u) == 0.0
    assert m.start_seconds(du) == 0.5
    # Raw properties are unchanged.
    assert du.properties["wf_start_time"] == np.datetime64("1904-01-01T00:00:00.5", "ns")
    x = m.start_seconds(u) + np.arange(20) * u.wf_increment
    assert np.unique(x).size == 20
    assert m.warnings == []


def test_start_time_of_a_wrong_clock(write_tdms, open_source):
    """One slow channel with a 1970 clock must not cost the fast channel its resolution."""
    b = TdmsBuilder()
    b.segment(tb.header_objects(["fast", "slow"]) + [
        Obj(obj_path("fast", "U_quench"), np.arange(100000.0),
            props=tb.wf_props(1e-7, start_time="2026-07-28T12:05:36")),
        Obj(obj_path("fast", "I"), np.arange(100000.0),
            props=tb.wf_props(1e-7, start_time="2026-07-28T12:05:37")),
        Obj(obj_path("slow", "T"), np.arange(10.0), props=tb.wf_props(1.0, start_time="1970-01-01T00:00:01")),
    ])
    src = open_source(write_tdms(b))
    m = src.model
    fast, i_ch, slow = m.channels
    assert m.t_ref == fast.wf_start_time
    assert m.start_seconds(fast) == 0.0 and m.start_seconds(i_ch) == 1.0
    assert slow.wf_start_time == np.datetime64("1970-01-01T00:00:01.000000500", "ns")  # kept
    assert m.start_seconds(slow) < -1e9
    w = [s for s in m.warnings if "slow/T" in s]
    assert len(w) == 1 and "more than 1 year" in w[0], m.warnings
    x = m.start_seconds(fast) + np.arange(20) * fast.wf_increment
    assert np.unique(x).size == 20


def test_start_times_within_a_year_use_the_earliest(write_tdms, open_source):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["G"]) + [
        Obj(obj_path("G", "a"), np.arange(10.0), props=tb.wf_props(1.0, start_time="2026-07-28T12:00:00")),
        Obj(obj_path("G", "b"), np.arange(1000.0), props=tb.wf_props(1.0, start_time="2026-12-28T12:00:00")),
    ])
    m = open_source(write_tdms(b)).model
    assert m.t_ref == m.channels[0].wf_start_time
    assert m.warnings == []


# -- 4. names and strings that are not UTF-8 -----------------------------------------------------------

@pytest.fixture
def raw_names(monkeypatch):
    """Object paths and string properties 'RAW:<latin-1 text>' are written as those raw bytes."""
    orig = tb._string

    def raw_string(s, e):
        if s.startswith("RAW:"):
            bb = s[4:].encode("latin-1")
            return struct.pack(e + "I", len(bb)) + bb
        return orig(s, e)

    monkeypatch.setattr(tb, "_string", raw_string)


def test_cp1252_names_stay_apart(raw_names, write_tdms, open_source):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["g"]) + [
        Obj("RAW:/'g'/'I_\xb5'", np.full(100, 1.0), props={"RAW:Einheit": "RAW:\xb5A"}),
        Obj("RAW:/'g'/'I_\xb0'", np.full(100, 2.0)),
        Obj("RAW:/'g'/'x\x81'", np.full(100, 3.0)),  # 0x81 is not in cp1252: Latin-1
    ])
    src = open_source(write_tdms(b))
    m = src.model
    assert [c.name for c in m.channels] == ["I_µ", "I_°", "x\x81"]
    for c, v in zip(m.channels, (1.0, 2.0, 3.0)):
        assert c.length == 100 and c.fast
        assert np.all(src.read(c.id, 0, 100) == v)
    assert m.channels[0].properties == {"Einheit": "µA"}
    assert m.warnings.count(NON_UTF8) == 1
    assert not any("Error decoding" in w for w in m.warnings)


def test_names_that_join_after_decoding_give_a_warning(raw_names, write_tdms, open_source):
    """UTF-8 'é' (C3 A9) and cp1252 'é' (E9) are the same text: npTDMS joins the objects."""
    b = TdmsBuilder()
    b.segment(tb.header_objects(["g"]) + [Obj(obj_path("g", "é"), np.full(10, 1.0)),
                                          Obj("RAW:/'g'/'\xe9'", np.full(10, 2.0))])
    m = open_source(write_tdms(b)).model
    assert [c.name for c in m.channels] == ["é"]
    assert any("same path /'g'/'é'" in w for w in m.warnings), m.warnings


@pytest.fixture
def cp1252_values(monkeypatch):
    """String channel values are written in Windows-1252."""
    orig = tb.encode_values

    def enc(code, values, e="<"):
        if code == tb.T_STRING:
            raw = [str(s).encode("cp1252") for s in values]
            ends = np.cumsum([len(x) for x in raw], dtype=np.int64)
            return ends.astype(e + "u4").tobytes() + b"".join(raw)
        return orig(code, values, e)

    monkeypatch.setattr(tb, "encode_values", enc)


def test_cp1252_string_values(cp1252_values, write_tdms, open_source):
    n = 20000
    words = [f"Kälte {i}" for i in range(n)]
    b = TdmsBuilder()
    b.segment(tb.header_objects(["g"]) + [Obj(obj_path("g", "log"), words), Obj(obj_path("g", "I"), np.arange(10.0))])
    src = open_source(write_tdms(b))
    assert src.model.warnings == []  # values are decoded later
    log = src.model.channels[0]
    t = time.perf_counter()
    got = src.read(log.id, 0, log.length)
    assert list(got) == words
    w = src.drain_warnings()
    assert time.perf_counter() - t < 5.0
    assert w == [NON_UTF8]
    src.read(log.id, 5, 10)
    assert src.drain_warnings() == []  # once per file


# -- 5. warning flood --------------------------------------------------------------------------

def test_dedupe_is_linear_and_limited():
    msgs = [f"msg {i}" for i in range(100_000)]
    t = time.perf_counter()
    out = tdmsfile._dedupe(msgs + msgs)
    assert time.perf_counter() - t < 1.0
    assert out[:20] == msgs[:20]
    assert out[20:] == ["... and 99980 more warnings"]
    assert tdmsfile._dedupe(["a", "b", "a"]) == ["a", "b"]
    assert tdmsfile._dedupe(["a"], dropped=3) == ["a", "... and 3 more warnings"]


def test_capture_is_bounded():
    cap = tdmsfile._capture
    cap.clear()
    lg = logging.getLogger("nptdms.tests_flood")
    for i in range(5000):
        lg.warning("value %d is bad", i)
    assert len(cap.records) <= cap.MAX
    out = cap.take()
    assert out[:2] == ["value 0 is bad", "value 1 is bad"] and len(out) == 21
    assert out[-1] == "... and 4980 more warnings"
    assert cap.take() == [] and cap.records == [] and cap.dropped == 0


def test_many_npTDMS_warnings_in_one_file(write_tdms, open_source, rng):
    """Many npTDMS warnings (one per segment) end in at most 21 entries."""
    b = TdmsBuilder()
    p = obj_path("G", "x")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 10, rng))], version=4713)
    for k in range(400):
        b.raw_segment({p: tb.random_values("f8", 10, rng)}, version=5000 + k)  # "Segment version mismatch"
    src = open_source(write_tdms(b))
    w = src.model.warnings
    assert len(w) <= 21 and w[-1].startswith("... and ") and w[-1].endswith(" more warnings"), w[-3:]
    assert src.model.channels[0].length == 4010


# -- 6. EXT (extended float) and unknown type codes ----------------------------------------------------

def x87(v: float, big_endian: bool = False, pad: bytes = b"\0" * 6) -> bytes:
    """Reference encoder: float -> 16-byte TDMS EXT value (80-bit x87 + padding)."""
    if math.isnan(v):
        se, m = 0x7FFF, 0xC000000000000000
    elif math.isinf(v):
        se, m = 0x7FFF | (0x8000 if v < 0 else 0), 1 << 63
    elif v == 0:
        se, m = (0x8000 if math.copysign(1.0, v) < 0 else 0), 0
    else:
        mant, exp = math.frexp(abs(v))
        m = int(mant * 2**64)
        se = (exp - 1 + 16383) | (0x8000 if v < 0 else 0)
    raw = struct.pack("<QH", m, se) + pad
    return raw[::-1] if big_endian else raw


EXT_VALUES = [0.0, -0.0, 1.0, -2.5, 1e-300, 1e300, 5e-324, np.finfo(np.float64).max, math.pi, -1 / 3,
              math.inf, -math.inf, 123456.789]


@pytest.mark.parametrize("big_endian", [False, True])
def test_ext_decoder(big_endian):
    raw = b"".join(x87(v, big_endian) for v in EXT_VALUES + [math.nan])
    tdmsfile._notes.bad_ext = False
    got = tdmsfile.ext_to_float64(np.frombuffer(raw, np.uint8), ">" if big_endian else "<")
    exp = np.array(EXT_VALUES + [math.nan])
    assert got.dtype == np.float64
    assert got[:-1].tobytes() == exp[:-1].tobytes()  # also the sign of -0.0
    assert np.isnan(got[-1])
    assert not tdmsfile._notes.bad_ext
    # 64-bit mantissa: rounded to the nearest float64.
    one_plus = struct.pack("<QH", (1 << 63) | 1, 16383) + b"\0" * 6
    huge = struct.pack("<QH", 1 << 63, 0x7FFE) + b"\0" * 6
    tiny = struct.pack("<QH", 1 << 63, 1) + b"\0" * 6
    assert list(tdmsfile.ext_to_float64(one_plus + huge + tiny)) == [1.0, math.inf, 0.0]


def test_ext_decoder_rejects_other_formats():
    tdmsfile._notes.bad_ext = False
    quad_one = b"\0" * 14 + struct.pack("<H", 0x3FFF)  # IEEE binary128 1.0: not x87
    unnormal = struct.pack("<QH", 1 << 62, 16383) + b"\0" * 6  # no integer bit
    got = tdmsfile.ext_to_float64(quad_one + unnormal + x87(2.0))
    assert np.isnan(got[0]) and np.isnan(got[1]) and got[2] == 2.0
    assert tdmsfile._notes.bad_ext
    tdmsfile._notes.bad_ext = False


@pytest.fixture
def ext_code(monkeypatch):
    """The builder writes type code 11 (EXT) with 16-byte values (V16 arrays)."""
    monkeypatch.setitem(tb.CODE_DTYPE, 11, np.dtype("V16"))
    monkeypatch.setitem(tb.CODE_SIZE, 11, 16)


def _ext_array(values, big_endian=False) -> np.ndarray:
    return np.frombuffer(b"".join(x87(float(v), big_endian) for v in values), dtype="V16")


def test_ext_channel_and_property(ext_code, write_tdms, open_source, rng):
    n = 700
    vals = rng.standard_normal(3 * n) * 1e3
    ok = tb.random_values("f8", 3 * n, rng)
    pe, po = obj_path("g", "ext"), obj_path("g", "ok")
    b = TdmsBuilder()
    b.segment([Obj("/", props={"gain": Prop(11, _ext_array([2.5])[0])}), Obj(obj_path("g"))]
              + [Obj(pe, _ext_array(vals[:n]), type_code=11, props={"cal": Prop(11, _ext_array([-0.125])[0])}),
                 Obj(po, ok[:n])])
    b.segment([Obj(pe, index="same"), Obj(po, index="same")], interleaved=True,
              data={pe: _ext_array(vals[n:2 * n]), po: ok[n:2 * n]})
    # Big endian: the builder does not swap V16 values, so give them reversed.
    b.segment([Obj(pe, index="same"), Obj(po, index="same")], big_endian=True,
              data={pe: _ext_array(vals[2 * n:], big_endian=True), po: ok[2 * n:]})
    src = open_source(write_tdms(b))
    m = src.model
    ext, okc = m.channels
    assert m.properties == {"gain": 2.5}
    assert ext.properties == {"cal": -0.125}
    assert ext.kind == KIND_FLOAT and ext.dtype == np.float64 and ext.plottable
    assert ext.type_code == 11 and ext.length == 3 * n and not ext.fast
    assert okc.fast
    tb.assert_same_values(src.read(ext.id, 0, ext.length), vals, "ext")
    for a, z in tb.windows(ext.length, rng, count=30, joints=(n, 2 * n)):
        lo, hi = tb.clamp_window(ext.length, a, z)
        tb.assert_same_values(src.read(ext.id, a, z), vals[lo:hi], f"ext [{a}, {z})")
    out = np.empty(100, np.float64)
    src.read_into(ext.id, out, n - 50)
    assert np.array_equal(out, vals[n - 50:n + 50])
    tb.assert_same_values(src.read(okc.id, 0, okc.length), ok, "ok")
    got = {}
    for cid, off, a in src.data_chunks():
        got.setdefault(cid, []).append(a)
    tb.assert_same_values(np.concatenate(got[ext.id]), vals, "ext chunks")
    assert m.warnings == [] and src.drain_warnings() == []


@pytest.mark.parametrize("where", ["channel", "property"])
def test_unknown_type_code_is_named(where, monkeypatch, write_tdms):
    monkeypatch.setitem(tb.CODE_DTYPE, 0x99, np.dtype("V4"))
    monkeypatch.setitem(tb.CODE_SIZE, 0x99, 4)
    b = TdmsBuilder()
    objs = tb.header_objects(["g"]) + [Obj(obj_path("g", "ok"), np.arange(10.0))]
    if where == "channel":
        objs.append(Obj(obj_path("g", "unk"), np.zeros(10, "V4"), type_code=0x99))
    else:
        objs[0] = Obj("/", props={"odd": Prop(0x99, np.zeros(1, "V4")[0])})
    b.segment(objs)
    path = write_tdms(b)
    with pytest.raises(ValueError) as ei:
        TdmsSource(path)
    assert "0x99" in str(ei.value) and "not a known TDMS type" in str(ei.value)


# -- 7. timestamps outside datetime64[ns] ----------------------------------------------------------------

def test_time_to_ns_out_of_range_is_nat():
    off = 2_082_844_800
    lo, hi = -9_223_372_036 + off, 9_223_372_035 + off
    secs = np.array([0, off, lo, hi, lo - 1, hi + 1, -10**10, 2**40, np.iinfo(np.int64).min,
                     np.iinfo(np.int64).max], dtype=np.int64)
    fr = np.full(secs.size, 1 << 63, dtype=np.uint64)
    tdmsfile._notes.bad_time = False
    got = tdms_time_to_ns(secs, fr)
    assert tdmsfile._notes.bad_time
    assert got[0] == np.datetime64("1904-01-01T00:00:00.5", "ns")
    assert got[1] == np.datetime64("1970-01-01T00:00:00.5", "ns")
    assert got[2] == np.datetime64(-9_223_372_036 * 10**9 + 5 * 10**8, "ns")
    assert got[3] == np.datetime64(9_223_372_035 * 10**9 + 5 * 10**8, "ns")
    assert np.isnat(got[4:]).all()
    tdmsfile._notes.bad_time = False
    assert np.isnat(tdms_time_to_ns(2**40, 0))  # scalar
    assert not np.isnat(tdms_time_to_ns(off, 0)) and tdmsfile._notes.bad_time
    tdmsfile._notes.bad_time = False


def test_out_of_range_timestamps_in_a_file(write_tdms, open_source, rng):
    t = tb.random_times(50, rng)
    raw = [tb.RawTime(*tb.timestamp_parts(v)) for v in t]
    raw[3] = tb.RawTime(2**40, 0)  # year ~36800
    raw[7] = tb.RawTime(-10**10, 0)  # year ~1587
    b = TdmsBuilder()
    b.segment(tb.header_objects(["g"]) + [
        Obj(obj_path("g", "t"), raw, type_code=tb.T_TIME),
        Obj(obj_path("g", "x"), np.arange(10.0), props={"wf_increment": 1.0, "wf_start_time": tb.RawTime(2**40, 0)}),
    ])
    src = open_source(write_tdms(b))
    m = src.model
    tch, x = m.channels
    assert tch.kind == KIND_TIME
    assert x.wf_start_time is None and np.isnat(x.properties["wf_start_time"])
    assert m.t_ref is None
    text = [w for w in m.warnings if "datetime64[ns] range" in w]
    assert len(text) == 1
    a = src.read(tch.id, 0, tch.length)
    assert np.isnat(a[3]) and np.isnat(a[7])
    keep = [i for i in range(50) if i not in (3, 7)]
    exact = t.astype("datetime64[ns]") + np.timedelta64(500, "ns")  # the builder writes the middle of the us
    assert np.array_equal(a[keep], exact[keep])
    assert src.drain_warnings() == []  # once per file


# -- 8. wf_increment that is not a positive number ----------------------------------------------------

@pytest.mark.parametrize("value,shown", [(-1e-3, "-0.001"), (0.0, "0.0"), (float("nan"), "nan"),
                                         (float("inf"), "inf"), ("fast", "'fast'")])
def test_invalid_wf_increment(value, shown, write_tdms, open_source):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["g"]) + [Obj(obj_path("g", "c"), np.arange(10.0), props={"wf_increment": value}),
                                          Obj(obj_path("g", "ok"), np.arange(10.0), props={"wf_increment": 0.5})])
    m = open_source(write_tdms(b)).model
    c, ok = m.channels
    assert c.wf_increment is None
    raw = c.properties["wf_increment"]
    assert raw == value or (isinstance(value, float) and math.isnan(value) and math.isnan(raw))
    assert ok.wf_increment == 0.5
    assert m.warnings == [f"g/c: invalid wf_increment {shown} ignored (1 sample = 1 s)"]


def test_many_invalid_wf_increments_give_few_warnings(write_tdms, open_source):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["g"]) + [Obj(obj_path("g", f"c{i}"), np.arange(4.0), props={"wf_increment": -1.0})
                                          for i in range(12)])
    w = open_source(write_tdms(b)).model.warnings
    assert len(w) == 6 and w[-1] == "... and 7 more channels with an invalid wf_increment"


# -- npTDMS over-read of channels with gaps ----------------------------------------------------------------

def _gap_file(rng, ext=False):
    """Channel s: 10 values, no values in segment 2, 30 values in 3 chunks in segment 3."""
    b = TdmsBuilder()
    s, x = obj_path("G", "s"), obj_path("G", "x")

    def vals(n, k):
        if ext:
            return rng.standard_normal(n)
        return [f"w{k}_{i:03d}" for i in range(n)]

    v0, v2 = vals(10, 0), vals(30, 2)
    enc = (lambda v: _ext_array(v)) if ext else (lambda v: v)
    code = {"type_code": 11} if ext else {}
    b.segment(tb.header_objects(["G"]) + [Obj(s, enc(v0), **code), Obj(x, np.arange(10.0))])
    b.segment([Obj(s, index="none"), Obj(x, np.arange(10.0))], new_obj_list=False)
    b.segment([Obj(s, enc(v2), **code), Obj(x, np.arange(30.0))], new_obj_list=False, chunks=3)
    exp = (np.concatenate([v0, v2]) if ext else list(v0) + list(v2))
    return b, exp


@pytest.mark.parametrize("kind", ["string", "ext"])
def test_npTDMS_read_across_a_gap(kind, ext_code, write_tdms, open_source, rng):
    b, exp = _gap_file(rng, ext=kind == "ext")
    src = open_source(write_tdms(b))
    info = src.model.channels[0]
    assert info.length == 40 and not info.fast
    for a in range(0, 41):
        for z in range(a, 42):
            got = src.read(info.id, a, z)
            if kind == "ext":
                tb.assert_same_values(got, exp[a:z], f"[{a}, {z})")
            else:
                assert list(got) == exp[a:z], (a, z)


# -- open time ----------------------------------------------------------------------------------------------

def test_sample_file_opens_without_warnings(sample_tdms, open_source):
    for _ in range(2):
        open_source(sample_tdms)
    t = time.perf_counter()
    src = open_source(sample_tdms)
    dt = time.perf_counter() - t
    assert src.model.warnings == [] and not src.model.index_used
    assert dt < 0.5  # about 8 ms warm; a loose bound for slow CI machines
