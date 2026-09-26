"""Tests for tdmsviewer.tdmsfile (TdmsSource and the metadata model).

TdmsSource.read(cid, a, b) must give the same bytes as
TdmsFile.read(path)[group][channel][a:b] for every channel.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pytest
from nptdms import TdmsFile

import tdms_builders as tb
from tdms_builders import Obj, Prop, TdmsBuilder, obj_path
from tdmsviewer.tdmsfile import (KIND_BOOL, KIND_COMPLEX, KIND_EMPTY, KIND_FLOAT, KIND_INT, KIND_STRING,
                                 KIND_TIME, TdmsSource)

G = tb.G

DTYPE_KIND = {"f": KIND_FLOAT, "i": KIND_INT, "u": KIND_INT, "b": KIND_BOOL, "c": KIND_COMPLEX,
              "M": KIND_TIME, "O": KIND_STRING}


def check_source_against_nptdms(src: TdmsSource, ref: TdmsFile, rng, count: int = 40):
    """Every channel, many windows: byte-identical to npTDMS."""
    ref_channels = [c for g in ref.groups() for c in g.channels()]
    assert [c.path for c in ref_channels] == [c.path for c in src.model.channels]
    for info, rc in zip(src.model.channels, ref_channels):
        full = np.asarray(rc[:])
        n = len(full)
        assert info.length == n, info.path
        for a, b in tb.windows(n, rng, count=count):
            got = src.read(info.id, a, b)
            lo, hi = tb.clamp_window(n, a, b)
            exp = np.asarray(rc[lo:hi])
            if n == 0:
                assert got.size == 0
                continue
            tb.assert_same_values(got, exp, f"{info.path} [{a}, {b})")


# -- cross check -------------------------------------------------------------------

@pytest.mark.parametrize("name", tb.ALL_SCENARIOS)
def test_read_matches_nptdms(name, scenario, open_source, rng):
    path, builder, notes = scenario(name)
    src = open_source(path)
    check_source_against_nptdms(src, tb.nptdms_full(path), rng)
    by_path = {c.path: c for c in src.model.channels}
    for p, want in notes["fast"].items():
        assert by_path[p].fast is want, p
        assert src.is_fast(by_path[p].id) is want
        assert (src.fast_reader(by_path[p].id) is not None) is want


@pytest.mark.parametrize("name", ["contiguous", "mixed", "interleaved_be", "all_dtypes", "truncated_mid_chunk"])
def test_read_without_fast_path(name, scenario, open_source, rng):
    path, builder, notes = scenario(name)
    src = open_source(path, use_fast_path=False)
    assert not any(c.fast for c in src.model.channels)
    check_source_against_nptdms(src, tb.nptdms_full(path), rng, count=20)


@pytest.mark.parametrize("name", ["contiguous", "mixed", "interleaved", "big_endian", "raw_only", "growing"])
def test_read_matches_written_values(name, scenario, open_source):
    path, builder, notes = scenario(name)
    src = open_source(path)
    for info in src.model.channels:
        exp = builder.expected(info.path)
        if len(exp) == 0:
            continue
        exp = np.asarray(exp)
        tb.assert_same_values(src.read(info.id, 0, info.length), exp.astype(exp.dtype.newbyteorder("=")),
                              info.path)


def test_sample_file(sample_tdms, open_source, rng):
    src = open_source(sample_tdms)
    m = src.model
    assert m.name.endswith(".tdms")
    assert m.size == os.path.getsize(sample_tdms)
    assert m.n_segments == 1
    assert len(m.groups) == 4
    assert len(m.channels) == 16
    assert m.warnings == []
    assert m.t_ref is None and m.t_ref_unix is None
    for c in m.channels:
        assert c.fast and c.kind == KIND_FLOAT and c.length == 44275 and c.dtype == np.float64
        assert c.wf_increment is None and c.wf_start_time is None
        assert m.start_seconds(c) == 0.0
    time_ch = [c for c in m.channels if c.label == "Time/Time"]
    assert len(time_ch) == 1
    t = src.read(time_ch[0].id, 0, time_ch[0].length)
    assert abs(t[0] - 0.05) < 1e-3 and abs(t[-1] - 4427.45) < 1e-3
    assert time_ch[0].unit == "s"
    check_source_against_nptdms(src, tb.nptdms_full(sample_tdms), rng, count=20)


# -- metadata model --------------------------------------------------------------------

def test_model_matches_nptdms_metadata(scenario, open_source):
    path, builder, notes = scenario("all_dtypes")
    src = open_source(path)
    ref = tb.nptdms_full(path)
    m = src.model
    assert m.path == os.path.abspath(path)
    assert m.properties == dict(ref.properties)
    assert [g.name for g in m.groups] == [g.name for g in ref.groups()]
    for gi, rg in zip(m.groups, ref.groups()):
        assert gi.properties == dict(rg.properties)
        assert [c.name for c in gi.channels] == [c.name for c in rg.channels()]
        for ci, rc in zip(gi.channels, rg.channels()):
            assert m.channels[ci.id] is ci
            assert ci.group == rg.name and ci.path == rc.path and ci.label == f"{rg.name}/{rc.name}"
            assert ci.properties == dict(rc.properties)
            if len(rc):
                assert ci.dtype == rc.dtype
                assert ci.kind == DTYPE_KIND[rc.dtype.kind]
            else:
                assert ci.kind == KIND_EMPTY and not ci.plottable
            code = rc.data_type.enum_value if rc.data_type is not None else None
            assert ci.type_code == code
            dp = ci.display_properties()
            assert dp["NI_ChannelLength"] == len(rc)
            if code is None:
                assert "NI_DataType" not in dp
            else:
                assert dp["NI_DataType"] == code
            assert "NI_ChannelLength" not in ci.properties


def test_kinds_and_type_codes(scenario, open_source):
    path, builder, notes = scenario("all_dtypes")
    src = open_source(path)
    by_name = {c.name: c for c in src.model.channels}
    codes = {"t_i1": 1, "t_i2": 2, "t_i4": 3, "t_i8": 4, "t_u1": 5, "t_u2": 6, "t_u4": 7, "t_u8": 8,
             "t_f4": 9, "t_f8": 10, "t_?": 0x21, "t_c8": 0x08000C, "t_c16": 0x10000D, "text": 0x20,
             "time": 0x44, "empty_i32": 3}
    for name, code in codes.items():
        assert by_name[name].type_code == code, name
    assert by_name["t_?"].kind == KIND_BOOL
    assert by_name["t_c16"].kind == KIND_COMPLEX and by_name["t_c16"].plottable
    assert by_name["text"].kind == KIND_STRING and not by_name["text"].plottable
    assert by_name["time"].kind == KIND_TIME and by_name["time"].plottable
    assert by_name["time"].dtype == np.dtype("datetime64[us]")
    assert by_name["empty_i32"].kind == KIND_EMPTY and by_name["empty_i32"].length == 0
    assert by_name["no_data"].kind == KIND_EMPTY and by_name["no_data"].type_code is None
    assert by_name["no_data"].properties == {"note": "never has data"}
    for name in ("empty_i32", "no_data"):
        a = src.read(by_name[name].id, 0, 10)
        assert a.shape == (0,)
    s = src.read(by_name["text"].id, 3, 9)
    assert s.dtype == object and all(isinstance(v, str) for v in s)


def test_waveform_properties_and_time_reference(scenario, open_source):
    path, builder, notes = scenario("waveform")
    src = open_source(path)
    m = src.model
    by_name = {c.name: c for c in m.channels}
    sine, later, rel = by_name["sine"], by_name["later"], by_name["relative"]
    assert sine.wf_increment == 0.001 and sine.wf_start_offset == 0.5
    assert sine.wf_start_time == np.datetime64("2026-07-28T12:05:36.250000", "us")
    assert later.wf_start_time == np.datetime64("2026-07-28T12:05:40", "us")
    assert rel.wf_start_time is None  # TDMS epoch means relative time
    assert rel.wf_start_offset == 1.5 and rel.wf_increment == 0.01
    assert sine.unit == "V" and later.unit == "A" and rel.unit == ""
    assert m.t_ref == sine.wf_start_time
    exp_unix = (np.datetime64("2026-07-28T12:05:36.250000", "us") - np.datetime64(0, "us")) / np.timedelta64(1, "s")
    assert m.t_ref_unix == pytest.approx(float(exp_unix), abs=1e-6)
    assert m.start_seconds(sine) == pytest.approx(0.5, abs=1e-12)
    assert m.start_seconds(later) == pytest.approx(3.75, abs=1e-12)
    assert m.start_seconds(rel) == 1.5
    assert m.properties == {"name": "wave test", "author": "tests"}


def test_bad_waveform_properties_are_ignored(write_tdms, open_source, rng):
    b = TdmsBuilder()
    props = {"wf_increment": "fast", "wf_start_offset": float("nan"), "wf_start_time": 12.5, "unit": "degC"}
    b.segment(tb.header_objects(["G"]) + [Obj(obj_path("G", "x"), tb.random_values("f8", 10, rng), props=props),
                                          Obj(obj_path("G", "y"), tb.random_values("f8", 10, rng),
                                              props={"wf_increment": Prop(tb.T_I32, 2), "Unit": ""})])
    src = open_source(write_tdms(b))
    x, y = src.model.channels
    assert x.wf_increment is None and x.wf_start_offset is None and x.wf_start_time is None
    assert x.unit == "degC"
    assert y.wf_increment == 2.0 and y.unit == ""
    assert src.model.t_ref is None


def test_scaled_values(scenario, open_source):
    path, builder, notes = scenario("scaled")
    src = open_source(path)
    by_path = {c.path: c for c in src.model.channels}
    raw = builder.expected(obj_path(G, "scaled_i16")).astype(np.float64)
    got = src.read(by_path[obj_path(G, "scaled_i16")].id, 0, 500)
    assert got.dtype == np.float64
    assert np.array_equal(got, raw * 0.001 - 3.0)
    raw = builder.expected(obj_path("GroupScaled", "by_group")).astype(np.float64)
    assert np.array_equal(src.read(by_path[obj_path("GroupScaled", "by_group")].id, 0, 500), raw * 0.5 + 1.0)
    raw = builder.expected(obj_path(G, "already_scaled"))
    assert np.array_equal(src.read(by_path[obj_path(G, "already_scaled")].id, 0, 500), raw, equal_nan=True)


def test_daqmx_values(scenario, open_source):
    path, builder, notes = scenario("daqmx")
    src = open_source(path)
    by_path = {c.path: c for c in src.model.channels}
    info = by_path[obj_path(G, "dq_raw")]
    raw = builder.expected(info.path).astype(np.float64)
    assert info.kind == KIND_FLOAT and info.length == raw.size
    assert np.array_equal(src.read(info.id, 0, info.length), raw * 0.01 + 0.5)
    info = by_path[obj_path(G, "dq_i32")]
    assert info.kind == KIND_INT
    assert np.array_equal(src.read(info.id, 0, info.length), builder.expected(info.path))
    assert src.has_mixed_layout()


@pytest.mark.parametrize("name,mixed", [("contiguous", False), ("big_endian", False), ("interleaved", True),
                                        ("mixed", True), ("daqmx", True), ("all_dtypes", False)])
def test_has_mixed_layout(name, mixed, scenario, open_source):
    path, builder, notes = scenario(name)
    assert open_source(path).has_mixed_layout() is mixed


def test_void_channel(write_tdms, open_source, rng):
    b = TdmsBuilder()
    b.segment(tb.header_objects(["G"]) + [Obj(obj_path("G", "v"), [], type_code=tb.T_VOID),
                                          Obj(obj_path("G", "x"), tb.random_values("f8", 10, rng))])
    src = open_source(write_tdms(b))
    v, x = src.model.channels
    assert v.kind == KIND_EMPTY and v.length == 0 and v.type_code == 0
    assert src.read(v.id, 0, 5).shape == (0,)
    # x may use npTDMS (the segment has a Void object); values must be right.
    assert np.array_equal(src.read(x.id, 0, 10), b.expected(x.path), equal_nan=True)


def test_special_names(write_tdms, open_source, rng):
    b = TdmsBuilder()
    g = "it's / a group"
    names = ["µV 'quoted'", "日本語", "a/b", " spaces "]
    b.segment([Obj("/"), Obj(obj_path(g))] + [Obj(obj_path(g, n), tb.random_values("f8", 20, rng))
                                              for n in names])
    path = write_tdms(b)
    src = open_source(path)
    assert [c.name for c in src.model.channels] == names
    assert all(c.group == g and c.fast for c in src.model.channels)
    check_source_against_nptdms(src, tb.nptdms_full(path), rng, count=5)


def test_metadata_only_segment(write_tdms, open_source, rng):
    """A segment with only metadata (no kTocRawData) updates properties, no data."""
    b = TdmsBuilder()
    p = obj_path("G", "x")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 64, rng), props={"step": 1})])
    b.segment([Obj(p, index="same", props={"step": 2})], chunks=0, data={p: []})
    b.raw_segment({p: tb.random_values("f8", 128, rng)}, chunks=2)
    path = write_tdms(b)
    src = open_source(path)
    info = src.model.channels[0]
    assert info.length == 192 and info.fast
    assert info.properties["step"] == 2
    check_source_against_nptdms(src, tb.nptdms_full(path), rng)
    tb.assert_same_values(src.read(0, 0, 192), b.expected(p), p)


def _truncated_mixed_types(rng) -> tuple[TdmsBuilder, int]:
    """Last segment cut inside the first object of the last chunk.

    The later objects (timestamp, scaled i16) have no values in the last
    chunk. They do not use the fast path, so npTDMS reads them.
    """
    b = TdmsBuilder()
    names = [("value", "f8"), ("stamp", "time"), ("scaled", "i2")]

    def objs(n):
        out = []
        for name, kind in names:
            vals = tb.random_times(n, rng) if kind == "time" else tb.random_values(kind, n, rng)
            props = tb.linear_scale_props(0.5, 2.0) if name == "scaled" else {}
            out.append(Obj(obj_path("G", name), vals, props=props))
        return out

    b.segment(tb.header_objects(["G"]) + objs(40))
    last = b.segment(objs(40 * 5), chunks=5)
    chunk = 40 * (8 + 16 + 2)
    keep = 4 * chunk + 8 * 13 + 5
    return b, len(last.raw) - keep


def test_truncated_file_npTDMS_fallback_channels(write_tdms, open_source, rng):
    """Scaled and timestamp channels of a crashed (truncated) file must be readable.

    These channels do not use the fast path, so TdmsSource.read calls
    npTDMS channel.read_data for them.
    """
    b, cut = _truncated_mixed_types(rng)
    path = write_tdms(b, cut=cut)
    src = open_source(path)
    by_name = {c.name: c for c in src.model.channels}
    assert by_name["value"].fast
    assert not by_name["stamp"].fast and not by_name["scaled"].fast
    assert by_name["value"].length == 40 + 4 * 40 + 13
    assert by_name["stamp"].length == by_name["scaled"].length == 40 + 4 * 40
    check_source_against_nptdms(src, tb.nptdms_full(path), rng)


def _crashed_two_channels(rng) -> tuple[TdmsBuilder, int]:
    """One segment, 4 chunks of [a, b] (600 f8 each), cut inside a of chunk 4."""
    b = TdmsBuilder()
    objs = tb.header_objects(["G"]) + [Obj(obj_path("G", "a"), tb.random_values("f8", 2400, rng)),
                                       Obj(obj_path("G", "b"), tb.random_values("f8", 2400, rng))]
    last = b.segment(objs, chunks=4)
    return b, len(last.raw) - (3 * 600 * 16 + 8 * 100)


def test_truncated_file_plain_channel_with_empty_last_chunk(write_tdms, open_source, rng):
    """Plain float64 channel b has no values in the cut chunk: it must stay fast and readable."""
    b, cut = _crashed_two_channels(rng)
    path = write_tdms(b, cut=cut)
    src = open_source(path)
    a_info, b_info = src.model.channels
    assert (a_info.length, b_info.length) == (1900, 1800)
    for a, z in [(0, 1024), (500, 700), (0, 1800), (1799, 1800)]:
        tb.assert_same_values(src.read(b_info.id, a, z), b.expected(b_info.path)[a:z], f"b [{a}, {z})")
    assert a_info.fast and b_info.fast


# -- warnings ------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", tb.TRUNCATED_MODES)
def test_truncated_file_warnings(mode, scenario, open_source):
    path, builder, notes = scenario("truncated_" + mode)
    src = open_source(path)
    w = src.model.warnings
    assert any("last segment" in m.lower() and "incomplete" in m.lower() for m in w), w
    assert any("less data than expected" in m for m in w), w
    assert src.warnings == w
    assert len(w) == len(set(w))


def test_warnings_do_not_leak_between_files(scenario, open_source):
    bad, _, _ = scenario("truncated_mid_chunk")
    good, _, _ = scenario("contiguous")
    open_source(bad)
    src = open_source(good)
    assert src.model.warnings == []
    assert src.drain_warnings() == []


def test_drain_warnings_after_reads(scenario, open_source):
    path, builder, notes = scenario("truncated_interleaved")
    src = open_source(path)
    src.drain_warnings()
    for c in src.model.channels:
        src.read(c.id, 0, c.length)
    list(src.data_chunks())
    w = src.drain_warnings()
    assert all(isinstance(m, str) for m in w)
    assert src.drain_warnings() == []


def test_nptdms_messages_are_not_printed(scenario, open_source, capfd):
    path, builder, notes = scenario("truncated_mid_chunk")
    open_source(path)
    out, err = capfd.readouterr()
    assert "less data than expected" not in err


# -- .tdms_index handling ---------------------------------------------------------------------

def _three_segments(rng) -> TdmsBuilder:
    b = TdmsBuilder()
    p, q = obj_path("G", "x"), obj_path("G", "y")
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 100, rng)),
                                          Obj(q, tb.random_values("i2", 100, rng))])
    b.raw_segment({p: tb.random_values("f8", 100, rng), q: tb.random_values("i2", 100, rng)})
    b.segment([Obj(p, tb.random_values("f8", 50, rng), props={"late": 1}),
               Obj(q, tb.random_values("i2", 50, rng))], new_obj_list=False)
    return b


def test_valid_index_is_used(write_tdms, open_source, rng):
    b = _three_segments(rng)
    path = write_tdms(b, index=True)
    src = open_source(path)
    assert src.model.index_used is True
    assert src.model.warnings == []
    assert src.model.n_segments == 3
    check_source_against_nptdms(src, tb.nptdms_full(path, use_index=False), rng)


def test_stale_index_is_ignored(write_tdms, open_source, rng):
    b = _three_segments(rng)
    path = write_tdms(b, index=2)  # index knows only the first 2 segments
    assert len(tb.nptdms_full(path)["G"]["x"]) == 200  # npTDMS alone trusts the index
    src = open_source(path)
    assert src.model.index_used is False
    assert any(".tdms_index" in m for m in src.model.warnings)
    assert src.model.channels[0].length == 250
    assert src.model.channels[0].properties.get("late") == 1
    check_source_against_nptdms(src, tb.nptdms_full(path, use_index=False), rng)
    assert all(c.fast for c in src.model.channels)


def test_index_longer_than_data_file(write_tdms, open_source, rng):
    """Data file lost its last segment, the index still lists it."""
    b = _three_segments(rng)
    last = b.segments[-1]
    path = write_tdms(b, index=True, cut=len(last.lead_in) + len(last.meta) + len(last.raw))
    src = open_source(path)
    assert src.model.channels[0].length == 200
    check_source_against_nptdms(src, tb.nptdms_full(path, use_index=False), rng)


def test_index_of_same_size_but_other_layout(tmp_path, open_source, rng):
    """An index of another file with the same size must not give wrong data.

    Data file: 2 segments of 100 values. Index: 1 segment of 200 values,
    padded (string property) to the same total file size.
    """
    p = obj_path("G", "x")
    b = TdmsBuilder()
    b.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 100, rng), props={"tag": "B"})])
    b.raw_segment({p: tb.random_values("f8", 100, rng)})

    def other(tag):
        a = TdmsBuilder()
        a.segment(tb.header_objects(["G"]) + [Obj(p, tb.random_values("f8", 200, rng), props={"tag": tag})])
        return a

    a = other("A" * (b.size - other("").size))
    assert a.size == b.size
    path = str(tmp_path / "same_size.tdms")
    b.write(path)
    with open(path + "_index", "wb") as fh:
        fh.write(a.index_bytes())
    src = open_source(path)
    got = src.read(0, 0, src.model.channels[0].length)
    tb.assert_same_values(got, np.asarray(b.expected(p)), "values with a foreign index")
    assert src.model.channels[0].properties["tag"] == "B"


@pytest.mark.parametrize("damage", ["garbage", "truncated_meta"])
def test_damaged_index_falls_back_to_data_file(damage, write_tdms, open_source, rng):
    """A broken .tdms_index must not make a good data file unreadable."""
    b = _three_segments(rng)
    path = write_tdms(b, index=True)
    idx = path + "_index"
    raw = open(idx, "rb").read()
    if damage == "garbage":
        raw = b"XXXX" + raw[4:]
    else:
        last_meta = len(b.segments[-1].meta)
        raw = raw[: len(raw) - last_meta // 2]
    with open(idx, "wb") as fh:
        fh.write(raw)
    src = open_source(path)
    assert src.model.index_used is False
    assert src.model.warnings
    check_source_against_nptdms(src, tb.nptdms_full(path, use_index=False), rng)


# -- chunk iteration -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["contiguous", "mixed", "interleaved_be", "all_dtypes", "daqmx",
                                  "truncated_interleaved", "growing"])
def test_data_chunks_rebuild_channels(name, scenario, open_source):
    path, builder, notes = scenario(name)
    src = open_source(path)
    ref = tb.nptdms_full(path)
    parts = {c.id: [] for c in src.model.channels}
    next_off = {c.id: 0 for c in src.model.channels}
    for cid, off, a in src.data_chunks():
        assert off == next_off[cid], (cid, off)
        assert a.size > 0
        assert a.dtype.isnative or a.dtype.kind in "OU"
        next_off[cid] += a.size
        parts[cid].append(a)
    for c in src.model.channels:
        exp = np.asarray(ref[c.group][c.name][:])
        assert next_off[c.id] == c.length
        if not c.length:
            continue
        got = np.concatenate(parts[c.id])
        if exp.dtype.kind == "O":
            assert list(got) == list(exp)
        else:
            tb.assert_same_values(got, exp, c.path)


# -- open / close -------------------------------------------------------------------------------

def _fds_on(path) -> int:
    """Number of open file descriptors of this process on `path` (or its index)."""
    targets = {os.path.realpath(path), os.path.realpath(str(path) + "_index")}
    n = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(os.readlink(f"/proc/self/fd/{fd}")) in targets:
                n += 1
        except OSError:
            pass
    return n


def test_close_releases_files(scenario):
    path, builder, notes = scenario("contiguous", index=True)
    gc.collect()
    assert _fds_on(path) == 0
    src = TdmsSource(path)
    assert _fds_on(path) >= 1
    src.close()
    src.close()  # twice is fine
    assert _fds_on(path) == 0


def test_close_after_stale_index(write_tdms, rng):
    path = write_tdms(_three_segments(rng), index=2)
    gc.collect()
    src = TdmsSource(path)
    assert src.model.index_used is False
    src.close()
    assert _fds_on(path) == 0


def test_not_a_tdms_file(tmp_path):
    p = tmp_path / "bad.tdms"
    p.write_bytes(b"this is not a TDMS file at all" * 10)
    with pytest.raises(Exception):
        TdmsSource(str(p))
    gc.collect()  # npTDMS keeps its file open until the failed TdmsFile is collected
    assert _fds_on(p) == 0


def test_source_does_not_modify_file(scenario, tmp_path):
    path, builder, notes = scenario("mixed", index=True)
    before = open(path, "rb").read(), open(path + "_index", "rb").read()
    src = TdmsSource(path)
    for c in src.model.channels:
        src.read(c.id, 0, c.length)
    list(src.data_chunks())
    src.close()
    assert (open(path, "rb").read(), open(path + "_index", "rb").read()) == before
    assert sorted(os.listdir(os.path.dirname(path))) == sorted([os.path.basename(path),
                                                                os.path.basename(path) + "_index"])


def test_two_sources_on_one_file(scenario, open_source, rng):
    path, builder, notes = scenario("chunks")
    s1, s2 = open_source(path), open_source(path)
    for c in s1.model.channels:
        assert s1.read(c.id, 0, c.length).tobytes() == s2.read(c.id, 0, c.length).tobytes()


def test_relative_path(scenario, open_source, monkeypatch):
    path, builder, notes = scenario("contiguous")
    monkeypatch.chdir(os.path.dirname(path))
    src = open_source(os.path.basename(path))
    assert src.model.path == path and os.path.isabs(src.path)


# -- files written by nptdms.TdmsWriter --------------------------------------------------------

def _writer_segments(rng):
    from nptdms import ChannelObject, GroupObject, RootObject

    x, y = tb.random_values("f8", 50, rng), tb.random_values("i4", 50, rng)
    x2, y2 = tb.random_values("f8", 30, rng), tb.random_values("i4", 30, rng)
    props = {"unit_string": "V", "gain": 2.5, "count": 7, "ok": True}
    ours = TdmsBuilder(version=4712)
    ours.segment([Obj("/", props={"title": "writer test"}), Obj(obj_path("G"), props={"flag": False}),
                  Obj(obj_path("G", "x"), x, props=props), Obj(obj_path("G", "y"), y)])
    ours.segment([Obj(obj_path("G", "x"), x2), Obj(obj_path("G", "y"), y2)])
    theirs = [[RootObject({"title": "writer test"}), GroupObject("G", {"flag": False}),
               ChannelObject("G", "x", x, props), ChannelObject("G", "y", y)],
              [ChannelObject("G", "x", x2), ChannelObject("G", "y", y2)]]
    return ours, theirs


def test_builder_matches_nptdms_writer(tmp_path, rng):
    """Self check of the raw builder: same bytes as nptdms.TdmsWriter (data and index)."""
    ours, theirs = _writer_segments(rng)
    path = tb.write_nptdms(tmp_path / "writer.tdms", theirs, index_file=True)
    assert open(path, "rb").read() == ours.data_bytes()
    assert open(path + "_index", "rb").read() == ours.index_bytes()


def test_nptdms_writer_file(tmp_path, open_source, rng):
    from nptdms import ChannelObject, GroupObject, RootObject

    t = tb.random_times(40, rng)
    words = tb.random_strings(40, rng)
    start = np.datetime64("2026-07-28T12:05:36.5", "us")
    segs = [[RootObject({"started": start, "operator": "tests"}), GroupObject("Log"),
             ChannelObject("Log", "time", t), ChannelObject("Log", "words", words),
             ChannelObject("Log", "value", tb.random_values("f4", 40, rng),
                           {"wf_start_time": start, "wf_increment": 0.5, "wf_start_offset": 0.0})],
            [ChannelObject("Log", "value", tb.random_values("f4", 25, rng))]]
    path = tb.write_nptdms(tmp_path / "log.tdms", segs, index_file=True)
    src = open_source(path)
    assert src.model.index_used and src.model.warnings == []
    by_name = {c.name: c for c in src.model.channels}
    assert by_name["time"].kind == KIND_TIME and by_name["words"].kind == KIND_STRING
    assert by_name["value"].length == 65  # shares a segment with strings: may use npTDMS
    assert by_name["value"].wf_start_time is not None and src.model.t_ref == by_name["value"].wf_start_time
    assert src.model.properties["operator"] == "tests"
    check_source_against_nptdms(src, tb.nptdms_full(path), rng)
