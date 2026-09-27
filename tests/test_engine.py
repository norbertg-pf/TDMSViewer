"""End-to-end tests of tdmsviewer.engine.DataEngine.

The engine runs a worker thread and answers with Qt signals. Each test
posts requests like the GUI does and compares the answers with brute
force numpy results on the same data (read back with npTDMS).
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from nptdms import TdmsFile  # noqa: E402
from tdms_builders import exact_ns  # noqa: E402

from engine_helpers import split_breaks  # noqa: E402
from engine_helpers import (  # noqa: E402
    Driver, as_f64, assert_stats, check_envelope, nan_equal, ref_stats, same_values,
    split_channels, time_to_seconds, write_interleaved, write_segments,
)
from tdmsviewer import engine as eng_mod  # noqa: E402
from tdmsviewer.engine import absolute_stats  # noqa: E402
from tdmsviewer.engine import (  # noqa: E402
    DataEngine, PlotItem, PlotRequest, StatsRequest, TableRequest,
)
from tdmsviewer.formatting import format_value  # noqa: E402
from tdmsviewer.xaxis import ArrayMap, LinearMap  # noqa: E402

REAL_FILE = Path("/root/.claude/uploads/5899cc9d-0cc2-5e06-94ba-ed80568becb4/"
                 "24399bc7-CBL-1252_test_012_2026-07-28_12-05-36.tdms")

# -- main test file layout ------------------------------------------------------

N = 150_001  # not a multiple of any pyramid bucket size
CUTS = (50_000, 100_000)  # segment boundaries of the numeric channels
UP = (5, 1234, 77_777, 100_000, N - 1)  # single-sample positive spikes
DOWN = (3, 50_000, 149_990)  # single-sample negative spikes
X0, DX = 2.5, 1e-3  # wf_start_offset, wf_increment of Wave/sig and Wave/offset
T_START = np.datetime64("2024-05-01T12:00:00", "us")
N_STAMP, N_FLAG, N_CPLX, N_TEXT = 6000, 3000, 5000, 40


@dataclass
class DataFile:
    path: Path
    ids: dict  # label -> channel id
    data: dict  # label -> native values (npTDMS read back)

    def f64(self, label):
        return as_f64(self.data[label])


def _read_back(path):
    ids, data = {}, {}
    f = TdmsFile.read(str(path), raw_timestamps=True)
    for g in f.groups():
        for c in g.channels():
            label = f"{g.name}/{c.name}"
            ids[label] = len(ids)
            data[label] = np.asarray(exact_ns(c[:]))  # timestamps exact to 1 ns, like the viewer
    return ids, data


def _build_main(path) -> dict:
    rng = np.random.default_rng(20240501)
    i = np.arange(N)
    sig = np.sin(i / 3000.0) + 0.1 * rng.standard_normal(N)
    sig[list(UP)] = 100.0 + np.arange(len(UP))
    sig[list(DOWN)] = -100.0 - np.arange(len(DOWN))
    sig[60_000:60_600] = np.nan  # holds one full all-NaN level-0 bucket
    sig[90_001] = np.nan
    offset = 4400.0 + 1e-3 * rng.standard_normal(N)  # small noise on large offset
    i32 = rng.integers(-1000, 1000, N).astype(np.int32)
    i32[[10, 120_000]] = [2**30, -(2**30)]
    f32 = rng.standard_normal(N).astype(np.float32)
    xy = np.sin(i / 700.0) * (1.0 + 0.05 * rng.standard_normal(N))  # not monotonic
    t = 0.05 + np.cumsum(rng.uniform(0.05, 0.15, N))  # monotonic, irregular
    wave = {"wf_increment": DX, "wf_start_offset": X0, "wf_start_time": T_START, "unit_string": "V"}
    chans = [
        ("Wave", "sig", sig, wave),
        ("Wave", "offset", offset, dict(wave, unit_string="A")),
        ("Wave", "i32", i32, {"unit_string": "counts"}),
        ("Wave", "f32", f32, {}),
        ("Wave", "XY", xy, {}),
        ("Time", "Time", t, {"unit_string": "s"}),
    ]
    segs = split_channels(chans, CUTS)
    stamp = (T_START + np.timedelta64(250_000, "us")
             + np.cumsum(rng.integers(1_000, 20_000, N_STAMP)).astype("timedelta64[us]"))
    flag = rng.random(N_FLAG) > 0.5
    cplx = rng.standard_normal(N_CPLX) + 1j * rng.standard_normal(N_CPLX)
    segs.append([("Misc", "stamp", stamp, {}), ("Misc", "flag", flag, {}),
                 ("Misc", "cplx", cplx, {}), ("Misc", "empty", np.zeros(0), {})])
    text = np.array([f'row {k}, "q{k}"' + ("\nline 2" if k % 7 == 0 else "") + (" Ω" if k % 5 == 0 else "")
                     for k in range(N_TEXT)])
    segs.append([("Misc", "text", text, {"note": "strings"})])
    write_segments(path, segs, file_props={"title": "engine test", "count": 3},
                   group_props={"Wave": {"rack": "A1"}})
    return {"Wave/sig": sig, "Wave/offset": offset, "Wave/i32": i32, "Wave/f32": f32, "Wave/XY": xy,
            "Time/Time": t, "Misc/stamp": stamp, "Misc/flag": flag, "Misc/cplx": cplx,
            "Misc/text": text}


@pytest.fixture(scope="module")
def main_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("engine_main") / "main.tdms"
    design = _build_main(path)
    ids, data = _read_back(path)
    for label, want in design.items():  # the writer must round-trip
        if want.dtype.kind == "M":  # npTDMS writer may lose 1 us; npTDMS read back is the reference
            assert np.abs((data[label] - want).astype(np.int64)).max() <= 1
        else:
            assert same_values(data[label], want), label
    yield DataFile(path, ids, data)
    path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def small_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("engine_small") / "small.tdms"
    z = np.arange(1000, dtype=np.float64) * 0.5 - 7.0
    write_segments(path, [[("B", "z", z, {"wf_increment": 0.1})]])
    ids, data = _read_back(path)
    yield DataFile(path, ids, data)
    path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def big_file(tmp_path_factory):
    """24 channels x 100k samples: loading takes many steps with a small BLOCK."""
    path = tmp_path_factory.mktemp("engine_big") / "big.tdms"
    rng = np.random.default_rng(7)
    chans = [("Big", f"c{k:02d}", rng.standard_normal(100_000), {}) for k in range(24)]
    write_segments(path, [chans])
    ids, _ = _read_back(path)
    yield DataFile(path, ids, {})
    path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _ram_mode(monkeypatch):
    """Default: everything fits in RAM (independent of the machine)."""
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", "2048")


@pytest.fixture
def new_driver(qtbot):
    made = []

    def make():
        d = Driver(qtbot, DataEngine())
        made.append(d)
        return d

    yield make
    for d in made:
        d.eng.shutdown()
        assert not d.eng._thread.is_alive()
        assert d.rec.errors() == [], d.rec.errors()


@pytest.fixture
def drv(new_driver):
    return new_driver()


# -- local helpers -----------------------------------------------------------------

def lin_to_index(xm: LinearMap):
    return lambda x: (np.asarray(x, dtype=np.float64) - xm.x0) / xm.dx


def arr_to_index(xs: np.ndarray):
    idx = np.arange(xs.size, dtype=np.float64)
    return lambda x: np.interp(x, xs, idx)


def sample_x(xm, i):
    """x of integer sample indices (reference, not engine code)."""
    i = np.asarray(i)
    if isinstance(xm, LinearMap):
        return xm.x0 + i.astype(np.float64) * xm.dx
    return xm.x[i]


def check_plot(res, samples, xm, xa, xb, px, s=0, e=None, to_index=None):
    """Check one plot answer (x, y, complete). Returns 'empty', 'raw' or 'decimated'."""
    assert res is not None
    x, y, complete = res
    e = samples.size if e is None else min(e, samples.size)
    if isinstance(xm, ArrayMap):
        e = min(e, xm.x.size)
    i0, i1 = xm.index_range(xa, xb, s, e)
    xs = sample_x(xm, np.arange(s, e))
    # The window covers the view: first sample at/left of xa, last at/right of xb.
    if i1 > i0 and i0 > s:
        assert xs[i0 - s] <= xa
    if i1 > i0 and i1 < e:
        assert xs[i1 - 1 - s] >= xb
    n = i1 - i0
    if n <= 0:
        assert x.size == 0 and y.size == 0 and complete
        return "empty"
    if n <= 2 * max(16, px):
        assert complete
        np.testing.assert_array_equal(x, sample_x(xm, np.arange(i0, i1)))
        assert nan_equal(y, samples[i0:i1])
        return "raw"
    if to_index is None:
        to_index = lin_to_index(xm) if isinstance(xm, LinearMap) else arr_to_index(xm.x)
    check_envelope(x, y, samples, i0, i1, to_index, complete=complete)
    assert y.size <= 4 * max(16, px) + 4, "output must be decimated"
    return "decimated"


def brute_inner(xm, xa, xb, s, e):
    """Samples [i0, i1) with xa <= x <= xb (brute force); None if empty."""
    if isinstance(xm, ArrayMap):
        e = min(e, xm.x.size)
        if not xm.monotonic:
            return (s, e) if e > s else None
    xs = sample_x(xm, np.arange(s, e))
    hit = np.nonzero((xs >= xa) & (xs <= xb))[0]
    if hit.size == 0:
        return None
    assert np.all(np.diff(hit) == 1)
    return s + int(hit[0]), s + int(hit[-1]) + 1


def _cursor_on_channel(xm, xs, cx) -> bool:
    """A cursor has a value only inside the channel's x range plus half a sample."""
    if xs.size == 0:
        return False
    if isinstance(xm, ArrayMap) and not xm.monotonic:
        return True
    h0 = 0.5 * (xs[1] - xs[0]) if xs.size > 1 else 0.0
    h1 = 0.5 * (xs[-1] - xs[-2]) if xs.size > 1 else 0.0
    if isinstance(xm, LinearMap):
        h0 = h1 = 0.5 * xm.dx
    return xs[0] - h0 <= cx <= xs[-1] + h1


def check_stats(res, samples, native, xm, xa, xb, s, e, cursors=()):
    e = min(e, samples.size)
    rng_ = brute_inner(xm, xa, xb, s, e)
    i0, i1 = res["range"]
    if rng_ is None:
        assert i1 - i0 <= 0
        assert res["stats"].n == 0
    else:
        assert (i0, i1) == rng_
        assert_stats(absolute_stats(res), ref_stats(samples[i0:i1]))
        if res.get("offset") is not None and res["stats"].n:  # int64/uint64: exact integers
            nat = native[i0:i1]
            assert res["offset"] + int(res["stats"].min) == int(nat.min())
            assert res["offset"] + int(res["stats"].max) == int(nat.max())
    assert len(res["cursors"]) == len(cursors)
    if isinstance(xm, ArrayMap):
        e = min(e, xm.x.size)
    xs = sample_x(xm, np.arange(s, e))
    for cx, cur in zip(cursors, res["cursors"]):
        if not _cursor_on_channel(xm, xs, cx):
            assert cur is None, f"cursor {cx} is outside the channel: no value expected"
            continue
        k = s + int(np.argmin(np.abs(xs - cx)))
        kk, xk, v = cur
        assert kk == k
        assert math.isclose(xk, float(xs[k - s]), rel_tol=1e-15, abs_tol=1e-12)
        want = native[k]
        if isinstance(want, float) and math.isnan(want):
            assert math.isnan(v)
        else:
            assert v == want
            assert type(v) is type(want)


# -- open ------------------------------------------------------------------------

def test_open_emits_model(drv, main_file):
    path = main_file.path
    gen = drv.eng.open(str(path))
    assert gen == drv.eng.generation == 1
    g, model = drv.wait("opened", lambda g, m: g == gen)
    first = drv.rec.events[0]
    assert first == ("progress", (gen, 0.0, f"Opening {path.name} ..."))

    assert model.path == os.path.abspath(path)
    assert model.name == path.name
    assert model.size == os.path.getsize(path)
    assert model.n_segments == 5
    assert model.index_used is False
    assert model.warnings == []
    assert model.properties == {"title": "engine test", "count": 3}
    assert model.open_ms > 0
    assert [gr.name for gr in model.groups] == ["Wave", "Time", "Misc"]
    assert model.groups[0].properties == {"rack": "A1"}
    flat = [c for gr in model.groups for c in gr.channels]
    assert flat == model.channels
    assert [c.id for c in model.channels] == list(range(len(model.channels)))
    assert {c.label: c.id for c in model.channels} == main_file.ids

    by = {c.label: c for c in model.channels}
    kinds = {"Wave/sig": "float", "Wave/offset": "float", "Wave/i32": "int", "Wave/f32": "float",
             "Wave/XY": "float", "Time/Time": "float", "Misc/stamp": "time", "Misc/flag": "bool",
             "Misc/cplx": "complex", "Misc/empty": "empty", "Misc/text": "string"}
    assert {k: c.kind for k, c in by.items()} == kinds
    lengths = {k: v.size for k, v in main_file.data.items()}
    assert {k: c.length for k, c in by.items()} == lengths
    assert by["Wave/i32"].dtype == np.int32
    assert by["Wave/f32"].dtype == np.float32
    assert by["Misc/stamp"].dtype == np.dtype("datetime64[ns]")
    assert by["Wave/sig"].type_code == 10 and by["Misc/text"].type_code == 0x20
    assert by["Wave/sig"].unit == "V" and by["Wave/offset"].unit == "A" and by["Time/Time"].unit == "s"
    sig = by["Wave/sig"]
    assert sig.wf_increment == DX and sig.wf_start_offset == X0
    assert sig.wf_start_time == T_START
    assert model.t_ref == T_START
    assert model.start_seconds(sig) == X0
    assert by["Wave/i32"].wf_increment is None and by["Wave/i32"].wf_start_offset is None
    assert by["Misc/text"].properties == {"note": "strings"}
    assert sig.display_properties()["NI_ChannelLength"] == N
    # Numeric channels use the verified fast reader; strings/timestamps do not.
    for label in ("Wave/sig", "Wave/offset", "Wave/i32", "Wave/f32", "Wave/XY", "Time/Time",
                  "Misc/flag", "Misc/cplx"):
        assert by[label].fast, label
    assert not by["Misc/stamp"].fast and not by["Misc/text"].fast
    assert drv.rec.of("openFailed") == []


def test_open_failed_not_tdms(drv, tmp_path, small_file):
    bad = tmp_path / "notes.tdms"
    bad.write_bytes(b"this is not a TDMS file, just text\n" * 20)
    gen = drv.eng.open(str(bad))
    g, msg = drv.wait("openFailed", lambda g, m: g == gen)
    assert msg and ":" in msg
    assert drv.rec.of("opened") == []
    assert drv.eng.try_read(gen, 0, 0, 10) is None
    # A request for the failed file is dropped, not answered with data.
    drv.eng.request_table(TableRequest(99, [0], 0, 10))
    drv.qtbot.wait(200)
    assert drv.rec.of("tableReady") == []
    # The engine still works after the failure.
    gen2, model = drv.open(small_file.path)
    assert gen2 == gen + 1 and model.channels[0].label == "B/z"


def test_open_failed_missing_file(drv, tmp_path):
    gen = drv.eng.open(str(tmp_path / "does_not_exist.tdms"))
    g, msg = drv.wait("openFailed", lambda g, m: g == gen)
    assert msg.startswith("FileNotFoundError")
    assert drv.rec.of("opened") == []


def test_open_failed_directory(drv, tmp_path):
    gen = drv.eng.open(str(tmp_path))
    g, msg = drv.wait("openFailed", lambda g, m: g == gen)
    assert msg.split(":")[0] in ("IsADirectoryError", "PermissionError", "OSError")


def test_file_without_channels(drv, tmp_path):
    empty = tmp_path / "zero.tdms"
    empty.write_bytes(b"")
    gen, model = drv.open(empty)
    assert model.channels == [] and model.groups == [] and model.n_segments == 0
    meta = tmp_path / "meta.tdms"
    from nptdms import GroupObject, RootObject, TdmsWriter

    with TdmsWriter(str(meta)) as w:
        w.write_segment([RootObject({"a": 1}), GroupObject("only group", {"b": "x"})])
    gen2, model2 = drv.open(meta)
    assert [g.name for g in model2.groups] == ["only group"] and model2.channels == []
    assert model2.properties == {"a": 1} and model2.groups[0].properties == {"b": "x"}
    assert drv.plot([], 0.0, 1.0, 100) == {}
    assert drv.table([0], 0, 10) == {}


# -- background load -------------------------------------------------------------

def test_background_load_progress_and_updates(drv, main_file):
    gen, model = drv.open(main_file.path)
    prog = [(f, t) for g, f, t in drv.rec.of("progress", gen)]
    fr = [f for f, _ in prog]
    assert fr[0] == 0.0 and fr[-1] == 1.0
    assert all(b >= a for a, b in zip(fr, fr[1:])), "progress must not go back"
    assert all(f < 1.0 for f in fr[:-1])
    assert prog[-1][1].startswith("Loaded ")
    # The 1.0 progress is the last load signal.
    names = [n for n, a in drv.rec.events if a[0] == gen]
    assert names[-1] == "progress"
    updated = [c for g, cids in drv.rec.of("channelsUpdated", gen) for c in cids]
    non_empty = sorted(c.id for c in model.channels if c.length)
    assert sorted(updated) == non_empty, "each loaded channel is reported once"
    stores = drv.eng._stores
    assert all(st.done for st in stores)
    assert all(st.ram is not None for st in stores if st.info.length)
    for label, cid in main_file.ids.items():
        if main_file.data[label].size:
            assert same_values(stores[cid].ram, main_file.data[label]), label
    assert drv.rec.of("message", gen) == []


def test_priority_channel_loads_first(drv, main_file, monkeypatch):
    from tdmsviewer import tdmsfile

    order = []  # channel ids in the order the loader reads them

    def spy(name):
        real = getattr(tdmsfile.TdmsSource, name)

        def wrapper(self, cid, *args):
            if cid not in order:
                order.append(cid)
            return real(self, cid, *args)

        return wrapper

    for name in ("read", "read_into"):
        if hasattr(tdmsfile.TdmsSource, name):
            monkeypatch.setattr(tdmsfile.TdmsSource, name, spy(name))
    first = [main_file.ids["Misc/cplx"], main_file.ids["Time/Time"]]
    drv.eng.set_priority(first)
    gen, _ = drv.open(main_file.path)
    assert order[:2] == first
    assert sorted(order) == sorted(c for k, c in main_file.ids.items() if main_file.data[k].size)


# -- plot: envelope, spikes, raw ---------------------------------------------------

VIEWS = [  # (first index, last index, pixels) in sample units of the LinearMap
    (0, N - 1, 16), (0, N - 1, 20), (0, N - 1, 100), (0, N - 1, 300), (0, N - 1, 1000),
    (12_345.3, 98_765.7, 50), (12_345.3, 98_765.7, 200), (12_345.3, 98_765.7, 700),
    (49_000.5, 51_000.5, 30), (99_000.0, 101_000.0, 64),  # across segment joints
    (-5_000, 3_000, 100), (N - 2_000, N + 5_000, 100),  # beyond the ends
    (59_000, 61_000, 40),  # NaN block
    (77_770.2, 77_790.9, 500), (N + 10, N + 100, 100), (-100, -10, 100),
]


@pytest.mark.parametrize("label", ["Wave/sig", "Wave/offset", "Wave/i32", "Wave/f32"])
def test_plot_envelope_equals_bruteforce(drv, main_file, label):
    gen, model = drv.open(main_file.path)
    cid = main_file.ids[label]
    ch = model.channels[cid]
    xm = LinearMap(model.start_seconds(ch), ch.wf_increment or 1.0)
    y_all = main_file.f64(label)
    kinds = set()
    for ia, ib, px in VIEWS:
        xa, xb = xm.x0 + ia * xm.dx, xm.x0 + ib * xm.dx
        out = drv.plot([PlotItem(cid, xm, 0, ch.length)], xa, xb, px)
        assert set(out) == {cid}
        kinds.add(check_plot(out[cid], y_all, xm, xa, xb, px))
    assert kinds == {"empty", "raw", "decimated"}


def test_plot_spikes_are_preserved(drv, main_file):
    gen, model = drv.open(main_file.path)
    cid = main_file.ids["Wave/sig"]
    xm = LinearMap(X0, DX)
    sig = main_file.f64("Wave/sig")
    xa, xb = X0, X0 + (N - 1) * DX
    i0, i1 = xm.index_range(xa, xb, 0, N)
    for px in (16, 37, 100, 256, 300, 640, 1000, 3000):
        x, y, complete = drv.plot([PlotItem(cid, xm, 0, N)], xa, xb, px)[cid]
        assert complete
        assert y.size < N // 10
        edges = check_envelope(x, y, sig, i0, i1, lin_to_index(xm))
        x, y, _ = split_breaks(x, y)  # 2 points per bucket from here on
        for i in (*UP, *DOWN):
            k = int(np.searchsorted(edges, i, side="right")) - 1
            lo, hi = y[2 * k], y[2 * k + 1]
            if i in UP:
                assert hi >= sig[i], f"spike at {i} lost with {px} px"
            else:
                assert lo <= sig[i], f"spike at {i} lost with {px} px"
            blk = sig[edges[k]:edges[k + 1]]
            if sig[i] in (np.nanmax(blk), np.nanmin(blk)):
                # The spike is the extreme of its bucket: drawn with its exact value, near its x.
                hit = np.nonzero(y == sig[i])[0]
                assert hit.size >= 1
                assert np.min(np.abs(x[hit] - (X0 + i * DX))) <= (edges[k + 1] - edges[k]) * DX
        if px >= 300:  # bucket <= 256 samples: every positive spike is alone in its bucket
            for i in UP:
                assert sig[i] in y
        assert np.nanmax(y) == np.nanmax(sig) and np.nanmin(y) == np.nanmin(sig)


def test_plot_multi_item_request(drv, main_file):
    gen, model = drv.open(main_file.path)
    labels = ["Wave/sig", "Wave/offset", "Wave/i32", "Wave/f32", "Wave/XY", "Time/Time"]
    xm = LinearMap(X0, DX)
    items = [PlotItem(main_file.ids[k], xm, 0, N) for k in labels]
    xa, xb = X0 + 1000 * DX, X0 + 140_000 * DX
    out = drv.plot(items, xa, xb, 400)
    assert set(out) == {main_file.ids[k] for k in labels}
    for k in labels:
        assert check_plot(out[main_file.ids[k]], main_file.f64(k), xm, xa, xb, 400) == "decimated"


def test_plot_raw_zoom_exact_samples_and_x(drv, main_file):
    gen, model = drv.open(main_file.path)
    sig = model.channels[main_file.ids["Wave/sig"]]
    xm = LinearMap(model.start_seconds(sig), sig.wf_increment)
    for label in ("Wave/sig", "Wave/offset"):
        cid = main_file.ids[label]
        for ia, ib in ((1000.0, 1100.0), (0.0, 50.0), (N - 60.0, N - 1.0), (77_777.2, 77_777.4)):
            xa, xb = X0 + ia * DX, X0 + ib * DX
            x, y, complete = drv.plot([PlotItem(cid, xm, 0, N)], xa, xb, 400)[cid]
            i0 = max(0, math.floor(ia) - 1)
            i1 = min(N, math.ceil(ib) + 2)
            assert complete
            idx = np.arange(i0, i1)
            # x = wf_start_offset + i * wf_increment
            np.testing.assert_array_equal(x, sig.wf_start_offset + idx.astype(np.float64) * sig.wf_increment)
            assert nan_equal(y, main_file.data[label][i0:i1])
            assert y.dtype == np.float64
            assert x.size >= 2
    # Integer, float32, bool and complex channels: y is the float64 value.
    for label in ("Wave/i32", "Wave/f32", "Misc/flag", "Misc/cplx"):
        cid = main_file.ids[label]
        xm1 = LinearMap(0.0, 1.0)
        # index_range adds one sample on each side: [floor(100) - 1, ceil(300) + 2).
        x, y, complete = drv.plot([PlotItem(cid, xm1, 0, model.channels[cid].length)], 100.0, 300.0, 200)[cid]
        np.testing.assert_array_equal(x, np.arange(99, 302, dtype=np.float64))
        want = main_file.data[label][99:302]
        want = np.abs(want) if want.dtype.kind == "c" else want.astype(np.float64)
        np.testing.assert_array_equal(y, want)


def test_plot_item_sample_window(drv, main_file):
    """PlotItem.s / .e limit the samples that are drawn."""
    gen, model = drv.open(main_file.path)
    cid = main_file.ids["Wave/sig"]
    xm = LinearMap(X0, DX)
    sig = main_file.f64("Wave/sig")
    s, e = 20_000, 70_000
    for ia, ib, px in ((0, N - 1, 100), (19_990, 20_050, 200), (69_950, 70_300, 300), (80_000, 90_000, 50)):
        xa, xb = X0 + ia * DX, X0 + ib * DX
        res = drv.plot([PlotItem(cid, xm, s, e)], xa, xb, px)[cid]
        check_plot(res, sig, xm, xa, xb, px, s=s, e=e)
        x = res[0]
        if x.size:
            assert x.min() >= X0 + (s - 0.5) * DX and x.max() <= X0 + (e - 0.5) * DX


def test_plot_and_stats_while_pyramid_builds(drv, main_file, monkeypatch):
    """Answers during the pyramid build are exact for the part that is covered."""
    import time

    from tdmsviewer import pyramid

    monkeypatch.setattr(eng_mod, "BLOCK", 4096)
    monkeypatch.setattr(eng_mod, "RAW_PLOT_MAX", 256)  # raw plot limit: 1024 RAM samples
    real_append = pyramid.Pyramid.append

    def slow_append(self, block):
        time.sleep(0.003)
        return real_append(self, block)

    monkeypatch.setattr(pyramid.Pyramid, "append", slow_append)
    cid = main_file.ids["Wave/sig"]
    drv.eng.set_priority([cid])
    gen, model = drv.open(main_file.path, loaded=False)
    xm = LinearMap(X0, DX)
    sig = main_file.f64("Wave/sig")
    xa, xb = X0, X0 + (N - 1) * DX
    i0, i1 = xm.index_range(xa, xb, 0, N)
    seen = {"none": 0, "partial": 0, "complete": 0}
    for _ in range(2000):
        res = drv.plot([PlotItem(cid, xm, 0, N)], xa, xb, 100)[cid]
        if res is None:
            seen["none"] += 1
        else:
            x, y, complete = res
            if x.size:
                check_envelope(x, y, sig, i0, i1, lin_to_index(xm), complete=complete)
            else:
                assert not complete
            seen["complete" if complete else "partial"] += 1
        st = drv.stats([PlotItem(cid, xm, 0, N)], X0 + 10.5 * DX, X0 + 140_000.5 * DX)[cid]["stats"]
        assert_stats(st, ref_stats(sig[11:140_001]))
        if res is not None and res[2]:
            break
    assert seen["partial"] >= 1, seen
    assert seen["complete"] == 1
    drv.wait_loaded(gen)
    x, y, complete = drv.plot([PlotItem(cid, xm, 0, N)], xa, xb, 100)[cid]
    assert complete
    check_envelope(x, y, sig, i0, i1, lin_to_index(xm))


# -- plot: X from a channel ------------------------------------------------------------

def test_x_channel_monotonic_arraymap(drv, main_file):
    gen, model = drv.open(main_file.path)
    tid = main_file.ids["Time/Time"]
    cid = main_file.ids["Wave/sig"]
    T = main_file.data["Time/Time"]
    sig = main_file.f64("Wave/sig")
    got = drv.xmap(tid)
    assert isinstance(got, tuple)
    xcid, amap, t_ref = got
    assert xcid == tid and t_ref is None
    assert isinstance(amap, ArrayMap) and amap.monotonic
    assert amap.x.dtype == np.float64
    np.testing.assert_array_equal(amap.x, T)
    kinds = set()
    for ia, ib, px in ((0, N - 1, 100), (0, N - 1, 1000), (1000, 120_000, 200), (5000, 5050, 300),
                       (77_700, 77_900, 64), (N - 30, N - 1, 64)):
        xa, xb = T[ia] - 1e-7, T[ib] + 1e-7
        res = drv.plot([PlotItem(cid, amap, 0, N)], xa, xb, px)[cid]
        kinds.add(check_plot(res, sig, amap, xa, xb, px))
    assert kinds == {"raw", "decimated"}
    # View between two samples: both neighbours are returned.
    xa, xb = T[500] + 0.3 * (T[501] - T[500]), T[500] + 0.6 * (T[501] - T[500])
    x, y, _ = drv.plot([PlotItem(cid, amap, 0, N)], xa, xb, 100)[cid]
    np.testing.assert_array_equal(x, T[500:502])
    np.testing.assert_array_equal(y, sig[500:502])


def test_x_channel_non_monotonic_xy(drv, main_file):
    gen, model = drv.open(main_file.path)
    xid = main_file.ids["Wave/XY"]
    X = main_file.data["Wave/XY"]
    xcid, amap, t_ref = drv.xmap(xid)
    assert xcid == xid and not amap.monotonic
    for label in ("Wave/sig", "Wave/f32"):
        cid = main_file.ids[label]
        y_all = main_file.f64(label)
        # XY mode ignores the view and returns all samples in [s, e).
        x, y, complete = drv.plot([PlotItem(cid, amap, 0, N)], -0.1, 0.1, 300)[cid]
        assert complete
        np.testing.assert_array_equal(x, X)
        assert nan_equal(y, y_all)
        x, y, complete = drv.plot([PlotItem(cid, amap, 1000, 90_000)], -0.1, 0.1, 300)[cid]
        np.testing.assert_array_equal(x, X[1000:90_000])
        assert nan_equal(y, y_all[1000:90_000])


def test_x_channel_xy_decimated(drv, main_file, monkeypatch):
    monkeypatch.setattr(eng_mod, "XY_MAX_POINTS", 2000)
    gen, model = drv.open(main_file.path)
    X = main_file.data["Wave/XY"]
    _, amap, _ = drv.xmap(main_file.ids["Wave/XY"])
    where = {float(v): k for k, v in enumerate(X)}
    assert len(where) == N  # X values are unique
    for label, s, e in (("Wave/sig", 0, N), ("Wave/f32", 0, N), ("Wave/sig", 777, 123_456)):
        cid = main_file.ids[label]
        Y = main_file.f64(label)
        x, y, complete = drv.plot([PlotItem(cid, amap, s, e)], 0.0, 1.0, 300)[cid]
        assert complete and x.size == y.size
        idx = np.array([where[float(v)] for v in x])
        # Output points are real (x, y) samples, in sample order.
        assert nan_equal(y, Y[idx])
        assert np.all(np.diff(idx) >= 0)
        n = e - s
        b = -(-n // 1000)  # bucket: ceil(n / (XY_MAX_POINTS / 2)), min and max per bucket
        k = n // b
        assert x.size == 2 * k + (n - k * b) <= 2000 + b
        for j in range(k):
            a0 = s + j * b
            blk = Y[a0:a0 + b]
            pair = idx[2 * j:2 * j + 2]
            assert np.all((pair >= a0) & (pair < a0 + b))
            if not np.all(np.isnan(blk)):
                assert {float(Y[pair[0]]), float(Y[pair[1]])} == {float(np.nanmin(blk)), float(np.nanmax(blk))}
        np.testing.assert_array_equal(idx[2 * k:], np.arange(s + k * b, e))
        # Spikes and global extremes survive.
        for sp in (*UP, *DOWN):
            if s <= sp < e and label == "Wave/sig":
                assert sp in set(idx.tolist())


def test_x_request_errors(drv, main_file, monkeypatch):
    gen, model = drv.open(main_file.path)
    assert drv.xmap(999) == "Unknown channel"
    msg = drv.xmap(main_file.ids["Misc/text"])
    assert isinstance(msg, str) and "not numeric" in msg
    msg = drv.xmap(main_file.ids["Misc/empty"])
    assert isinstance(msg, str) and "not numeric" in msg
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", "0.5")  # 150001 * 8 bytes > 0.5 MB
    msg = drv.xmap(main_file.ids["Time/Time"])
    assert isinstance(msg, str) and "too large" in msg


def test_unknown_channel_ids_are_skipped(drv, main_file):
    """Plot and statistics requests skip unknown channel ids, like table requests do."""
    gen, model = drv.open(main_file.path)
    ok = main_file.ids["Wave/i32"]
    bad = len(model.channels)
    xm = LinearMap(0.0, 1.0)
    items = [PlotItem(ok, xm, 0, N), PlotItem(bad, xm, 0, N), PlotItem(-1, xm, 0, N)]
    posts = [("tableReady", lambda q: drv.eng.request_table(TableRequest(q, [ok, bad, -1], 0, 10))),
             ("plotReady", lambda q: drv.eng.request_plot(PlotRequest(q, items, 0.0, 100.0, 100))),
             ("statsReady", lambda q: drv.eng.request_stats(StatsRequest(q, items, 0.0, 100.0)))]
    got, errors = {}, []
    for name, post in posts:  # one at a time: an answer or an engine error
        seq = drv.seq()
        n_err = len(drv.rec.errors())
        post(seq)
        drv.qtbot.waitUntil(lambda: drv.rec.find(name, lambda g, s, o: g == gen and s == seq) is not None
                            or len(drv.rec.errors()) > n_err, timeout=10_000)
        a = drv.rec.find(name, lambda g, s, o: g == gen and s == seq)
        if a is None:
            errors.append((name, drv.rec.errors()[-1][1]))
        else:
            got[name] = a[2]
    # Keep the fixture teardown quiet: this test reports the errors itself.
    drv.rec.events = [e for e in drv.rec.events if not (e[0] == "message" and "Internal error" in e[1][1])]
    assert set(got["tableReady"]) == {ok}
    assert errors == [], f"no answer, engine error for an unknown channel id: {errors}"
    assert set(got["plotReady"]) == {ok}, "cid -1 must not map to the last channel"
    assert set(got["statsReady"]) == {ok}


# -- statistics ------------------------------------------------------------------

STAT_RANGES = [  # sample index ranges (first, last)
    (0, N - 1), (12_345.5, 98_765.2), (12_345, 98_765), (100.0, 300.0), (255.0, 257.0),
    (60_000, 60_600), (59_990.2, 60_610.8), (-100, 50.5), (N - 10.5, N + 100), (N + 5, N + 10),
    (49_999.5, 100_000.5), (1.2, 1.8), (500.0, 400.0),
]


@pytest.mark.parametrize("label", ["Wave/sig", "Wave/offset", "Wave/i32", "Wave/f32", "Time/Time"])
def test_stats_exact_linear(drv, main_file, label):
    gen, model = drv.open(main_file.path)
    cid = main_file.ids[label]
    ch = model.channels[cid]
    xm = LinearMap(model.start_seconds(ch), ch.wf_increment or 1.0)
    y = main_file.f64(label)
    native = main_file.data[label]
    for ia, ib in STAT_RANGES:
        xa, xb = xm.x0 + ia * xm.dx, xm.x0 + ib * xm.dx
        cursors = [xm.x0 + 10.4 * xm.dx, xm.x0 + 10.6 * xm.dx, xm.x0 - 5.0, xm.x0 + (N + 50) * xm.dx,
                   xm.x0 + 77_777.3 * xm.dx]
        out = drv.stats([PlotItem(cid, xm, 0, ch.length)], xa, xb, cursors)
        assert set(out) == {cid}
        check_stats(out[cid], y, native, xm, xa, xb, 0, ch.length, cursors)


def test_stats_large_offset_small_noise(drv, main_file):
    """4400 A +/- 1 mA: std must stay exact (no sum-of-squares cancellation)."""
    gen, model = drv.open(main_file.path)
    cid = main_file.ids["Wave/offset"]
    xm = LinearMap(X0, DX)
    y = main_file.f64("Wave/offset")
    for ia, ib in ((0, N - 1), (1234.5, 145_678.5), (7, 70_007)):
        res = drv.stats([PlotItem(cid, xm, 0, N)], X0 + ia * DX, X0 + ib * DX)[cid]
        i0, i1 = res["range"]
        want = y[i0:i1]
        st = res["stats"]
        assert math.isclose(st.std, float(want.std(ddof=1)), rel_tol=1e-9)
        assert math.isclose(st.std, 1e-3, rel_tol=0.05)
        assert math.isclose(st.mean, float(want.mean()), rel_tol=1e-14)


def test_stats_other_kinds_and_maps(drv, main_file):
    gen, model = drv.open(main_file.path)
    # Timestamps in seconds, complex as magnitude, bool as 0/1 (flag is short: no pyramid, raw path).
    for label in ("Misc/stamp", "Misc/cplx", "Misc/flag"):
        cid = main_file.ids[label]
        n = model.channels[cid].length
        xm = LinearMap(0.0, 1.0)
        for ia, ib in ((0, n - 1), (17.5, n - 100.5), (4000, 4096)):
            out = drv.stats([PlotItem(cid, xm, 0, n)], ia, ib, [3.3, n + 5.0])
            check_stats(out[cid], main_file.f64(label), main_file.data[label], xm, ia, ib, 0, n, [3.3, n + 5.0])
    # Monotonic ArrayMap: range by searchsorted on the X channel.
    T = main_file.data["Time/Time"]
    _, amap, _ = drv.xmap(main_file.ids["Time/Time"])
    cid = main_file.ids["Wave/sig"]
    for xa, xb in ((T[0], T[-1]), (T[100] + 1e-9, T[99_000] - 1e-9), ((T[10] + T[11]) / 2, (T[10] + T[11]) / 2),
                   (T[-1] + 1.0, T[-1] + 2.0), (T[500], T[500])):
        cur = [T[42] + 0.2 * (T[43] - T[42]), T[42] + 0.8 * (T[43] - T[42]), -1e9, 1e9]
        out = drv.stats([PlotItem(cid, amap, 0, N)], xa, xb, cur)
        check_stats(out[cid], main_file.f64("Wave/sig"), main_file.data["Wave/sig"], amap, xa, xb, 0, N, cur)
    # Non-monotonic ArrayMap (XY): statistics of the samples in [s, e) whose x is in [xa, xb].
    _, xy, _ = drv.xmap(main_file.ids["Wave/XY"])
    res = drv.stats([PlotItem(cid, xy, 10, 5000)], -0.5, 0.5)[cid]
    assert res["range"] == (10, 5000)
    xs = xy.x[10:5000]
    sel = (xs >= -0.5) & (xs <= 0.5)
    assert 0 < sel.sum() < sel.size  # the range really selects a part
    assert_stats(res["stats"], ref_stats(main_file.f64("Wave/sig")[10:5000][sel]))
    # Several items in one request; strings and empty channels are skipped.
    xm = LinearMap(X0, DX)
    items = [PlotItem(main_file.ids[k], xm, 0, main_file.data[k].size)
             for k in ("Wave/sig", "Wave/offset", "Misc/text", "Misc/empty")]
    out = drv.stats(items, X0 + 5 * DX, X0 + 50 * DX)
    assert set(out) == {main_file.ids["Wave/sig"], main_file.ids["Wave/offset"]}


# -- table / try_read ------------------------------------------------------------

def test_table_blocks(drv, main_file):
    gen, model = drv.open(main_file.path)
    ids = main_file.ids
    cids = [ids[k] for k in ("Wave/sig", "Wave/i32", "Misc/text", "Misc/stamp", "Misc/cplx",
                             "Misc/flag", "Misc/empty")] + [999, -1]
    out = drv.table(cids, 5, 45)
    assert set(out) == {ids[k] for k in ("Wave/sig", "Wave/i32", "Misc/text", "Misc/stamp",
                                         "Misc/cplx", "Misc/flag")}
    for k in ("Wave/sig", "Wave/i32", "Misc/stamp", "Misc/cplx", "Misc/flag"):
        i0, vals = out[ids[k]]
        assert i0 == 5
        assert vals.dtype == main_file.data[k].dtype
        assert same_values(vals, main_file.data[k][5:45]), k
    i0, vals = out[ids["Misc/text"]]
    assert i0 == 5 and list(vals) == list(main_file.data["Misc/text"][5:40])
    # Clamping at both ends; blocks across segment joints.
    out = drv.table([ids["Wave/sig"]], -10, 20)
    assert out[ids["Wave/sig"]][0] == 0
    assert nan_equal(out[ids["Wave/sig"]][1], main_file.data["Wave/sig"][:20])
    out = drv.table([ids["Wave/i32"]], 49_990, 100_010)
    np.testing.assert_array_equal(out[ids["Wave/i32"]][1], main_file.data["Wave/i32"][49_990:100_010])
    out = drv.table([ids["Wave/f32"]], N - 5, N + 100)
    assert out[ids["Wave/f32"]][0] == N - 5
    np.testing.assert_array_equal(out[ids["Wave/f32"]][1], main_file.data["Wave/f32"][N - 5:])
    assert drv.table([ids["Wave/sig"]], N + 1, N + 10) == {}


def test_try_read_generation(drv, main_file, small_file):
    gen, model = drv.open(main_file.path)
    cid = main_file.ids["Wave/i32"]
    got = drv.eng.try_read(gen, cid, 10, 20)
    np.testing.assert_array_equal(got, main_file.data["Wave/i32"][10:20])
    np.testing.assert_array_equal(drv.eng.try_read(gen, cid, N - 3, N + 10), main_file.data["Wave/i32"][N - 3:])
    np.testing.assert_array_equal(drv.eng.try_read(gen, cid, -5, 2), main_file.data["Wave/i32"][:2])
    assert drv.eng.try_read(gen - 1, cid, 10, 20) is None
    assert drv.eng.try_read(gen + 1, cid, 10, 20) is None
    assert drv.eng.try_read(gen, 999, 0, 10) is None
    assert drv.eng.try_read(gen, -1, 0, 10) is None
    # A second file: the old generation gives nothing, the new one its own data.
    gen2 = drv.eng.open(str(small_file.path))
    assert drv.eng.try_read(gen, cid, 10, 20) is None  # at once, before the worker ran
    drv.wait("opened", lambda g, m: g == gen2)
    assert drv.eng.try_read(gen, cid, 10, 20) is None
    np.testing.assert_array_equal(drv.eng.try_read(gen2, 0, 0, 5), small_file.data["B/z"][:5])
    drv.eng.close_file()
    assert drv.eng.try_read(gen2, 0, 0, 5) is None
    assert drv.eng.try_read(drv.eng.generation, 0, 0, 5) is None


def test_close_file_drops_requests(drv, main_file, small_file):
    gen, model = drv.open(main_file.path)
    drv.eng.close_file()
    gen_closed = drv.eng.generation
    assert gen_closed == gen + 1
    drv.eng.request_plot(PlotRequest(1, [PlotItem(0, LinearMap(X0, DX), 0, N)], X0, X0 + 1, 100))
    drv.eng.request_table(TableRequest(1, [0], 0, 10))
    drv.qtbot.waitUntil(lambda: drv.eng._source is None, timeout=5000)
    drv.qtbot.wait(200)
    assert drv.rec.of("plotReady", gen_closed) == [] and drv.rec.of("tableReady", gen_closed) == []
    gen3, m3 = drv.open(small_file.path)
    assert gen3 == gen_closed + 1
    out = drv.table([0], 0, 3)
    np.testing.assert_array_equal(out[0][1], small_file.data["B/z"][:3])


# -- generations and request dropping ------------------------------------------------

def test_second_open_drops_first_file(drv, big_file, small_file, monkeypatch):
    monkeypatch.setattr(eng_mod, "BLOCK", 4096)  # many load steps for file A
    gen_a, model_a = drv.open(big_file.path, loaded=False)
    xm = LinearMap(0.0, 1.0)
    drv.eng.request_plot(PlotRequest(1, [PlotItem(k, xm, 0, 100_000) for k in range(24)], 0, 99_999, 500))
    drv.eng.request_table(TableRequest(1, list(range(24)), 0, 1000))
    gen_b = drv.eng.open(str(small_file.path))
    assert gen_b == gen_a + 1
    # Posted before the worker opened B: must be answered for B.
    drv.eng.request_plot(PlotRequest(2, [PlotItem(0, LinearMap(0.0, 0.1), 0, 1000)], 1.0, 5.0, 300))
    res = drv.wait("plotReady", lambda g, s, o: g == gen_b and s == 2)[2]
    assert set(res) == {0}
    check_plot(res[0], small_file.f64("B/z"), LinearMap(0.0, 0.1), 1.0, 5.0, 300)
    drv.wait_loaded(gen_b)
    ev = drv.rec.events
    k_open_b = ev.index(("opened", drv.rec.find("opened", lambda g, m: g == gen_b)))
    # After B is open the worker sends nothing for A. Pyramid helper threads may
    # still report a channel of A; it carries gen A, so the GUI drops it.
    late = [(n, a[0]) for n, a in ev[k_open_b:] if a[0] == gen_a and n != "channelsUpdated"]
    assert late == [], "no answer or progress of the old file after the new file was opened"
    assert all(a[0] in (gen_a, gen_b) for _, a in ev)
    assert not any(n in ("plotReady", "tableReady") and a[0] == gen_b and a[1] == 1 for n, a in ev)
    assert drv.eng.try_read(gen_a, 0, 0, 10) is None
    assert drv.rec.of("plotReady", gen_b)[0][1] == 2


def test_latest_request_wins(drv, main_file):
    gen, model = drv.open(main_file.path)
    eng = drv.eng
    cid = main_file.ids["Wave/sig"]
    i32 = main_file.ids["Wave/i32"]
    xm = LinearMap(X0, DX)
    sig = main_file.f64("Wave/sig")
    rng = np.random.default_rng(3)
    count = 150
    views = []
    for k in range(1, count + 1):
        ia = float(rng.uniform(-1000, N / 2))
        ib = float(rng.uniform(ia + 1, N + 1000))
        px = int(rng.integers(1, 3000))
        views.append((ia, ib, px))
        xa, xb = X0 + ia * DX, X0 + ib * DX
        eng.request_plot(PlotRequest(k, [PlotItem(cid, xm, 0, N)], xa, xb, px))
        eng.request_table(TableRequest(k, [cid, i32], int(ia), int(ia) + 100))
        eng.request_stats(StatsRequest(k, [PlotItem(cid, xm, 0, N)], xa, xb, [xa]))
        got = eng.try_read(gen, i32, k, k + 7)  # GUI reads while the worker is busy
        np.testing.assert_array_equal(got, main_file.data["Wave/i32"][k:k + 7])
    for sig_name in ("plotReady", "tableReady", "statsReady"):
        drv.wait(sig_name, lambda g, s, o: g == gen and s == count)
    drv.qtbot.wait(300)
    for sig_name in ("plotReady", "tableReady", "statsReady"):
        seqs = [s for g, s, o in drv.rec.of(sig_name, gen)]
        assert seqs[-1] == count, sig_name
        assert seqs == sorted(set(seqs)), f"{sig_name}: answers must be in order, each once"
    # Every answer that was sent belongs to its own request.
    for g, s, out in drv.rec.of("plotReady", gen):
        ia, ib, px = views[s - 1]
        check_plot(out[cid], sig, xm, X0 + ia * DX, X0 + ib * DX, px)
    for g, s, out in drv.rec.of("tableReady", gen):
        ia = int(views[s - 1][0])
        if cid in out:
            i0, vals = out[cid]
            assert i0 == max(0, ia)
            assert nan_equal(vals, main_file.data["Wave/sig"][i0:ia + 100])
    for g, s, out in drv.rec.of("statsReady", gen):
        ia, ib, _ = views[s - 1]
        check_stats(out[cid], sig, main_file.data["Wave/sig"], xm, X0 + ia * DX, X0 + ib * DX, 0, N,
                    [X0 + ia * DX])
    # The newest answer equals a fresh request with the same view.
    ia, ib, px = views[-1]
    last = drv.rec.of("plotReady", gen)[-1][2][cid]
    again = drv.plot([PlotItem(cid, xm, 0, N)], X0 + ia * DX, X0 + ib * DX, px)[cid]
    assert nan_equal(last[0], again[0]) and nan_equal(last[1], again[1])


def test_latest_request_wins_multi_item(drv, main_file):
    """Multi-item requests: an old answer may hold only some items; each is correct."""
    gen, model = drv.open(main_file.path)
    labels = ["Wave/sig", "Wave/offset", "Wave/i32", "Wave/f32", "Wave/XY", "Time/Time"]
    cids = [main_file.ids[k] for k in labels]
    xm = LinearMap(X0, DX)
    rng = np.random.default_rng(9)
    views = []
    for k in range(1, 61):
        ia = float(rng.uniform(0, N / 2))
        ib = float(rng.uniform(ia + 100, N))
        px = int(rng.integers(16, 2000))
        views.append((ia, ib, px))
        drv.eng.request_plot(PlotRequest(k, [PlotItem(c, xm, 0, N) for c in cids], X0 + ia * DX, X0 + ib * DX, px))
    drv.wait("plotReady", lambda g, s, o: g == gen and s == 60)
    drv.qtbot.wait(200)
    answers = drv.rec.of("plotReady", gen)
    assert [s for _, s, _ in answers] == sorted({s for _, s, _ in answers})
    assert answers[-1][1] == 60 and set(answers[-1][2]) == set(cids)
    for _, s, out in answers:
        assert out, "an answer is never empty"
        assert list(out) == cids[:len(out)], "items are answered in request order"
        ia, ib, px = views[s - 1]
        for c in out:
            check_plot(out[c], main_file.f64(labels[cids.index(c)]), xm, X0 + ia * DX, X0 + ib * DX, px)


# -- export ------------------------------------------------------------------------

def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_export_csv_rows(drv, main_file, tmp_path):
    gen, model = drv.open(main_file.path)
    ids = main_file.ids
    xm = LinearMap(X0, DX)
    cols = ["Wave/sig", "Wave/i32", "Misc/text", "Misc/stamp"]
    items = [PlotItem(ids[k], xm, 0, main_file.data[k].size) for k in cols]
    header = ["i", "t", 'sig, "V"', "i", "t", "i32", "i", "t", "text", "i", "t", "stamp"]
    out = tmp_path / "export.csv"
    xa, xb = X0 + 10 * DX, X0 + 250 * DX
    msg = drv.export(out, items, xa, xb, header)
    assert msg == f"Exported 241 rows to {out}"
    assert drv.rec.find("progress", lambda g, f, t: g == gen and f == 1.0 and t == "Exported 241 rows")
    rows = _read_csv(out)
    assert rows[0] == header
    body = rows[1:]
    assert len(body) == 241
    for r, row in enumerate(body):
        assert len(row) == 12
        i = 10 + r
        for j, k in enumerate(cols):
            native = main_file.data[k]
            cell = row[3 * j:3 * j + 3]
            if i >= native.size:
                assert cell == ["", "", ""]
                continue
            assert cell[0] == str(i)
            assert float(cell[1]) == X0 + i * DX
            if k == "Misc/text":
                assert cell[2] == native[i]
            elif k == "Misc/stamp":
                assert cell[2] == format_value(native[i])
            elif k == "Wave/i32":
                assert int(cell[2]) == native[i]
            else:
                assert float(cell[2]) == native[i]
    # NaN values and exact float text.
    out2 = tmp_path / "nan.csv"
    msg = drv.export(out2, [PlotItem(ids["Wave/sig"], xm, 0, N)], X0 + 90_000 * DX, X0 + 90_003 * DX,
                     ["i", "t", "sig"])
    rows = _read_csv(out2)[1:]
    assert [r[0] for r in rows] == ["90000", "90001", "90002", "90003"]
    assert rows[1][2] == "NaN"
    assert rows[0][2] == repr(float(main_file.data["Wave/sig"][90_000]))


def test_export_csv_arraymap_and_errors(drv, main_file, tmp_path):
    gen, model = drv.open(main_file.path)
    T = main_file.data["Time/Time"]
    _, amap, _ = drv.xmap(main_file.ids["Time/Time"])
    cid = main_file.ids["Wave/f32"]
    out = tmp_path / "time.csv"
    msg = drv.export(out, [PlotItem(cid, amap, 0, N)], (T[100] + T[101]) / 2, T[110], ["i", "t", "f32"])
    assert msg.startswith("Exported 10 rows")
    rows = _read_csv(out)[1:]
    assert [int(r[0]) for r in rows] == list(range(101, 111))
    assert [float(r[1]) for r in rows] == [float(v) for v in T[101:111]]
    assert [float(r[2]) for r in rows] == [float(v) for v in main_file.data["Wave/f32"][101:111]]
    # Empty range: header only.
    out3 = tmp_path / "none.csv"
    msg = drv.export(out3, [PlotItem(cid, amap, 0, N)], T[-1] + 5, T[-1] + 6, ["a"])
    assert msg.startswith("Exported 0 rows")
    assert _read_csv(out3) == [["a"]]
    # Bad target path: a clear error message, no crash.
    msg = drv.export(tmp_path / "no_such_dir" / "x.csv", [PlotItem(cid, amap, 0, N)], T[0], T[5], ["a"])
    assert msg.startswith("Export failed")


X_NAN = np.array([0.0, 1.0, np.nan, 3.0, 2.0, 5.0, np.inf, 7.0])  # X channel with NaN and Inf


@pytest.fixture
def xnan_file(tmp_path):
    path = tmp_path / "xnan.tdms"
    write_segments(path, [[("g", "X", X_NAN, {}), ("g", "Y", np.arange(X_NAN.size, dtype=np.float64) * 10.0, {})]])
    return path


def test_xy_plot_with_nan_in_x(drv, xnan_file):
    gen, model = drv.open(xnan_file)
    _, amap, _ = drv.xmap(0)
    assert not amap.monotonic
    x, y, _ = drv.plot([PlotItem(1, amap, 0, X_NAN.size)], -10, 10, 100)[1]
    assert nan_equal(x, X_NAN) and nan_equal(y, np.arange(X_NAN.size) * 10.0)


def test_export_x_column_next_to_nan_x(drv, xnan_file, tmp_path):
    """Export: x of sample i is X[i], also when X[i + 1] is NaN or Inf."""
    gen, model = drv.open(xnan_file)
    _, amap, _ = drv.xmap(0)
    out = tmp_path / "xnan.csv"
    msg = drv.export(out, [PlotItem(1, amap, 0, X_NAN.size)], -10, 10, ["i", "x", "y"])
    # X-Y export: only samples whose x is inside [-10, 10] (NaN and Inf are not).
    keep = [i for i, v in enumerate(X_NAN) if -10 <= v <= 10]
    assert msg.startswith(f"Exported {len(keep)} rows")
    rows = _read_csv(out)[1:]
    assert [r[0] for r in rows] == [str(i) for i in keep]
    got_x = [float(r[1]) for r in rows]
    assert got_x == [float(X_NAN[i]) for i in keep], f"export x column {got_x}"


def test_cursor_x_next_to_nan_x(drv, xnan_file):
    """Cursor readout: x of the nearest sample is X[k], also when X[k + 1] is NaN."""
    gen, model = drv.open(xnan_file)
    _, amap, _ = drv.xmap(0)
    res = drv.stats([PlotItem(1, amap, 0, X_NAN.size)], -10, 10, [0.9, 4.9])[1]
    assert res["range"] == (0, 8)
    (k1, x1, v1), (k2, x2, v2) = res["cursors"]
    assert (k1, v1, k2, v2) == (1, 10.0, 5, 50.0)
    assert (x1, x2) == (1.0, 5.0), f"cursor x values {(x1, x2)}, want X[1] = 1.0 and X[5] = 5.0"


# -- interleaved files (npTDMS data_chunks pass) -------------------------------------

N_IL = 30_011
SCALE = {"NI_Number_Of_Scales": 1, "NI_Scale[0]_Scale_Type": "Linear", "NI_Scale[0]_Linear_Slope": 0.5,
         "NI_Scale[0]_Linear_Y_Intercept": -3.0, "NI_Scaling_Status": "unscaled", "unit_string": "mV"}


@pytest.fixture(scope="module")
def interleaved_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("engine_il") / "interleaved.tdms"
    rng = np.random.default_rng(11)
    a = rng.standard_normal(N_IL)
    a[[17, 20_000]] = [50.0, -50.0]
    raw = rng.integers(-2000, 2000, N_IL).astype(np.int32)
    raw[[5, 14_000, N_IL - 1]] = [100_000, -100_000, 90_000]
    ts = (np.datetime64("2025-02-03T04:05:06.5", "us")
          + np.cumsum(rng.integers(100, 5000, N_IL)).astype("timedelta64[us]"))
    f32 = rng.standard_normal(N_IL).astype(np.float32)
    write_interleaved(path, "IL", [("a", a, {"wf_increment": 0.01}), ("raw", raw, SCALE), ("ts", ts, {}),
                                   ("f32", f32, {})], rows_per_segment=7_000, file_props={"kind": "il"})
    ids, data = _read_back(path)
    np.testing.assert_array_equal(data["IL/a"], a)
    np.testing.assert_array_equal(data["IL/raw"], raw * 0.5 - 3.0)
    np.testing.assert_array_equal(data["IL/ts"], ts)
    yield DataFile(path, ids, data)
    path.unlink(missing_ok=True)


def _check_file(drv, f, labels=None):
    """Plot, stats and table answers of every channel equal brute force."""
    xm = LinearMap(0.0, 1.0)
    for label in labels or list(f.ids):
        cid = f.ids[label]
        y = f.f64(label)
        n = y.size
        for ia, ib, px in ((0, n - 1, 16), (0, n - 1, 100), (0, n - 1, 1000), (n * 0.233 + 0.5, n * 0.234, 200),
                           (n * 0.03, n * 0.97, 50)):
            res = drv.plot([PlotItem(cid, xm, 0, n)], ia, ib, px)[cid]
            check_plot(res, y, xm, ia, ib, px)
        out = drv.stats([PlotItem(cid, xm, 0, n)], 3.5, n - 7.5, [n * 0.2333])
        check_stats(out[cid], y, f.data[label], xm, 3.5, n - 7.5, 0, n, [n * 0.2333])
        a = int(n * 0.23)
        i0, vals = drv.table([cid], a, a + 40)[cid]
        assert i0 == a
        assert same_values(vals, f.data[label][a:a + 40])
    assert drv.rec.of("message") == []


def _check_interleaved(drv, f):
    _check_file(drv, f, ["IL/a", "IL/raw", "IL/ts", "IL/f32"])


def test_interleaved_file_chunk_pass(drv, interleaved_file):
    f = interleaved_file
    gen, model = drv.open(f.path)
    by = {c.label: c for c in model.channels}
    assert model.n_segments == 5
    assert by["IL/a"].fast and by["IL/f32"].fast  # interleaved fast reader
    assert not by["IL/raw"].fast and not by["IL/ts"].fast  # scaled / timestamp: npTDMS
    assert by["IL/raw"].dtype == np.float64 and by["IL/raw"].unit == "mV"
    assert by["IL/a"].wf_increment == 0.01
    chunk_ids = sorted([f.ids["IL/raw"], f.ids["IL/ts"]])
    upd = [cids for g, cids in drv.rec.of("channelsUpdated", gen)]
    assert upd[-1] == chunk_ids
    flat = {c for cids in upd for c in cids}
    assert flat == set(range(4))
    assert all(st.done and st.ram is not None for st in drv.eng._stores)
    assert all(st.pyr is not None and st.pyr.complete for st in drv.eng._stores)
    _check_interleaved(drv, f)
    # Timestamp channel as X: seconds from the first sample, t_ref in Unix seconds.
    _, amap, t_ref = drv.xmap(f.ids["IL/ts"])
    assert amap.monotonic
    np.testing.assert_array_equal(amap.x, time_to_seconds(f.data["IL/ts"]))
    t0 = f.data["IL/ts"][0]
    assert float(t_ref) == pytest.approx((t0 - np.datetime64(0, "us")) / np.timedelta64(1, "s"), abs=1e-6)
    assert t_ref.sec * 10**9 + round(t_ref.frac * 1e9) == int(np.datetime64(t0, "ns").astype(np.int64))


def test_interleaved_file_disk_mode(drv, interleaved_file, monkeypatch):
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", "0")
    monkeypatch.setattr(eng_mod, "pyramid_bucket_budget", lambda: 50)
    gen, model = drv.open(interleaved_file.path)
    stores = drv.eng._stores
    assert all(st.ram is None for st in stores)
    assert all(st.base == 4096 for st in stores)  # 4 * 30011 samples / 50 buckets
    _check_interleaved(drv, interleaved_file)


@pytest.mark.parametrize("interleaved", [True, False], ids=["interleaved", "contiguous"])
@pytest.mark.parametrize("big_endian", [False, True], ids=["little", "big"])
def test_layouts_and_byte_order(drv, tmp_path, interleaved, big_endian):
    from engine_helpers import write_raw

    n = 12_007
    rng = np.random.default_rng(21)
    f64 = rng.standard_normal(n)
    f64[[1, 6_000, n - 1]] = [30.0, -30.0, 31.0]
    i16 = rng.integers(-3000, 3000, n).astype(np.int16)
    u32 = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)
    ts = (np.datetime64("2023-01-01T00:00:00", "us")
          + np.cumsum(rng.integers(10, 900, n)).astype("timedelta64[us]"))
    raw = rng.integers(-500, 500, n).astype(np.int32)
    chans = [("f64 'q'/x", f64, {"wf_increment": 0.5}), ("i16", i16, {}), ("u32", u32, {}), ("ts", ts, {}),
             ("scaled", raw, SCALE)]
    group = "G 'x'"
    path = tmp_path / "layout.tdms"
    write_raw(path, group, chans, 5_000, interleaved=interleaved, big_endian=big_endian,
              file_props={"a": 1.5}, group_props={"b": "c"})
    ids, data = _read_back(path)
    f = DataFile(path, ids, data)
    gen, model = drv.open(path)
    assert [c.label for c in model.channels] == [f"{group}/{c[0]}" for c in chans]
    assert model.properties == {"a": 1.5} and model.groups[0].properties == {"b": "c"}
    by = {c.name: c for c in model.channels}
    assert by["f64 'q'/x"].fast and by["i16"].fast and by["u32"].fast
    assert not by["ts"].fast and not by["scaled"].fast
    np.testing.assert_array_equal(data[f"{group}/scaled"], raw * 0.5 - 3.0)
    assert same_values(data[f"{group}/i16"], i16) and same_values(data[f"{group}/ts"], ts.astype("datetime64[ns]"))
    _check_file(drv, f)


def test_channels_in_some_segments(drv, tmp_path):
    """Channels that are missing from some segments or change order."""
    rng = np.random.default_rng(4)
    a = [rng.standard_normal(k) for k in (5000, 3000, 2000, 1000)]
    b = [rng.integers(0, 100, k).astype(np.int64) for k in (5000, 7000)]
    c = [rng.standard_normal(k).astype(np.float32) for k in (6000, 50)]
    segs = [
        [("G", "a", a[0], {"unit_string": "V"}), ("G", "b", b[0], {})],
        [("G", "a", a[1], {})],
        [("G", "b", b[1], {}), ("G", "a", a[2], {})],
        [("H", "c", c[0], {})],
        [("G", "a", a[3], {}), ("H", "c", c[1], {})],
    ]
    path = tmp_path / "some.tdms"
    write_segments(path, segs)
    ids, data = _read_back(path)
    np.testing.assert_array_equal(data["G/a"], np.concatenate(a))
    np.testing.assert_array_equal(data["G/b"], np.concatenate(b))
    gen, model = drv.open(path)
    assert model.n_segments == 5
    assert [(ch.label, ch.length) for ch in model.channels] == [("G/a", 11_000), ("G/b", 12_000), ("H/c", 6_050)]
    assert all(ch.fast for ch in model.channels)
    _check_file(drv, DataFile(path, ids, data))


@pytest.fixture
def main_copy(main_file, tmp_path):
    """A private copy of the main file (deleted after the test)."""
    import shutil

    path = tmp_path / "copy.tdms"
    shutil.copyfile(main_file.path, path)
    yield path
    path.unlink(missing_ok=True)


def test_file_truncated_after_open(drv, main_file, main_copy, monkeypatch, small_file):
    """The file becomes shorter on disk: errors are reported, the worker keeps running."""
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", "0")  # all data read from disk on demand
    path = main_copy
    gen, model = drv.open(path)
    cid = main_file.ids["Wave/sig"]
    xm = LinearMap(X0, DX)
    good = drv.plot([PlotItem(cid, xm, 0, N)], X0 + 10 * DX, X0 + 90 * DX, 500)[cid]
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) // 3)
    n_msg = len(drv.rec.of("message"))
    drv.eng.request_plot(PlotRequest(900, [PlotItem(cid, xm, 0, N)], X0 + 140_000 * DX, X0 + 140_100 * DX, 500))
    drv.qtbot.waitUntil(lambda: len(drv.rec.of("message")) > n_msg
                        or drv.rec.find("plotReady", lambda g, s, o: s == 900) is not None, timeout=10_000)
    ans = drv.rec.find("plotReady", lambda g, s, o: s == 900)
    assert ans is None or cid not in ans[2], "no data for samples beyond the end of the file"
    assert drv.eng._thread.is_alive()
    assert drv.eng.try_read(gen, cid, 140_000, 140_010) is None
    # Data before the cut still reads; other files still open.
    again = drv.plot([PlotItem(cid, xm, 0, N)], X0 + 10 * DX, X0 + 90 * DX, 500)[cid]
    assert nan_equal(again[1], good[1])
    drv.rec.events = [e for e in drv.rec.events if e[0] != "message"]  # expected read errors
    gen2, m2 = drv.open(small_file.path)
    assert drv.table([0], 0, 3)[0][0] == 0


def test_table_and_stats_answered_after_read_error(drv, main_file, main_copy, monkeypatch):
    """A read error in one channel: table and stats still answer (like plot does)."""
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", "0")
    path = main_copy
    gen, model = drv.open(path)
    sig, i32 = main_file.ids["Wave/sig"], main_file.ids["Wave/i32"]
    flag = main_file.ids["Misc/flag"]  # in a later segment: cut away below
    xm = LinearMap(0.0, 1.0)
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) // 3)
    requests = [
        ("tableReady", lambda q: drv.eng.request_table(TableRequest(q, [sig, i32], 0, 10))),
        ("tableReady", lambda q: drv.eng.request_table(TableRequest(q, [flag, i32], 0, 10))),
        ("statsReady", lambda q: drv.eng.request_stats(
            StatsRequest(q, [PlotItem(flag, xm, 0, N_FLAG), PlotItem(sig, xm, 0, N)], 0.0, 100.0))),
    ]
    got = []
    for name, post in requests:
        seq = drv.seq()
        n_msg = len(drv.rec.of("message"))
        post(seq)
        drv.qtbot.waitUntil(lambda: drv.rec.find(name, lambda g, s, o: s == seq) is not None
                            or len(drv.rec.of("message")) > n_msg, timeout=10_000)
        drv.qtbot.wait(100)
        a = drv.rec.find(name, lambda g, s, o: s == seq)
        got.append(None if a is None else a[2])
    msgs = [m for _, m in drv.rec.of("message")]
    drv.rec.events = [e for e in drv.rec.events if e[0] != "message"]  # reported below
    assert got[0] is not None and set(got[0]) == {sig, i32}
    assert got[1] is not None, f"table request not answered after a read error; messages: {msgs}"
    assert set(got[1]) == {i32}
    np.testing.assert_array_equal(got[1][i32][1], main_file.data["Wave/i32"][:10])
    assert got[2] is not None, f"stats request not answered after a read error; messages: {msgs}"
    assert sig in got[2] and (flag not in got[2] or got[2][flag]["stats"] is None)
    assert_stats(got[2][sig]["stats"], ref_stats(main_file.f64("Wave/sig")[0:101]))


# -- RAM mode vs disk mode -------------------------------------------------------------

def _query_suite(drv, f, model, amap, export_dir):
    """Run a fixed list of requests; return all answers in order."""
    ids = f.ids
    out = []
    lin = LinearMap(X0, DX)
    idx = LinearMap(0.0, 1.0)
    chans = [("Wave/sig", lin), ("Wave/offset", lin), ("Wave/i32", idx), ("Wave/f32", idx),
             ("Misc/stamp", idx), ("Misc/flag", idx), ("Misc/cplx", idx)]
    for label, xm in chans:
        cid = ids[label]
        n = f.data[label].size
        for ia, ib, px in ((0, n - 1, 16), (0, n - 1, 64), (0, n - 1, 300), (0, n - 1, 1000),
                           (123.3, n * 0.66, 40), (123.3, n * 0.66, 200), (n * 0.51, n * 0.51 + 20.9, 500)):
            xa, xb = xm.x0 + ia * xm.dx, xm.x0 + ib * xm.dx
            res = drv.plot([PlotItem(cid, xm, 0, n)], xa, xb, px)
            check_plot(res[cid], f.f64(label), xm, xa, xb, px)
            out.append(("plot", label, ia, ib, px, res))
        for ia, ib in ((0, n - 1), (17.5, n * 0.7), (100.0, 300.0)):
            xa, xb = xm.x0 + ia * xm.dx, xm.x0 + ib * xm.dx
            cur = [xm.x0 + 42.4 * xm.dx, xm.x0 + (n - 3) * xm.dx]
            res = drv.stats([PlotItem(cid, xm, 0, n)], xa, xb, cur)
            check_stats(res[cid], f.f64(label), f.data[label], xm, xa, xb, 0, n, cur)
            out.append(("stats", label, ia, ib, res))
    T = f.data["Time/Time"]
    for a, b, px in ((0, N - 1, 100), (2000, 100_000, 300), (70_000, 70_100, 400)):
        res = drv.plot([PlotItem(ids["Wave/sig"], amap, 0, N)], T[a], T[b], px)
        check_plot(res[ids["Wave/sig"]], f.f64("Wave/sig"), amap, T[a], T[b], px)
        out.append(("plot-x", a, b, px, res))
    xy = ArrayMap(np.ascontiguousarray(f.data["Wave/XY"]), False)
    for s, e in ((0, N), (1234, 99_999)):
        res = drv.plot([PlotItem(ids["Wave/sig"], xy, s, e)], -1.0, 1.0, 300)
        np.testing.assert_array_equal(res[ids["Wave/sig"]][0], f.data["Wave/XY"][s:e])
        out.append(("plot-xy", s, e, res))
    for i0, i1 in ((0, 50), (49_990, 50_010), (N - 20, N + 20)):
        out.append(("table", i0, i1, drv.table(list(ids.values()), i0, i1)))
    items = [PlotItem(ids[k], LinearMap(0.0, 1.0), 0, f.data[k].size)
             for k in ("Wave/sig", "Wave/i32", "Misc/stamp", "Misc/text", "Misc/cplx", "Misc/flag")]
    csv_path = export_dir / f"suite_{id(drv)}.csv"
    msg = drv.export(csv_path, items, 20.0, 3020.0, ["h"] * 18)
    assert msg.startswith("Exported 3001 rows")
    out.append(("export", csv_path.read_text(encoding="utf-8")))
    return out


def _same(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for p, q in zip(a, b):
            _same(p, q)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        if a.dtype.kind in "fc":
            assert nan_equal(a.real, b.real) and nan_equal(a.imag, b.imag)
        else:
            assert np.array_equal(a, b)
    elif hasattr(a, "m2"):  # Stats: pyramid merge order may differ in the last bits
        assert (a.n, a.min, a.max) == (b.n, b.min, b.max) or (a.n == b.n == 0)
        if a.n:
            # Bucket sizes differ (256 vs 1024), so the summation order differs:
            # allow rounding relative to the data scale, not to a near-zero mean.
            scale = max(abs(a.min), abs(a.max), 1e-300)
            assert math.isclose(a.mean, b.mean, rel_tol=1e-13, abs_tol=1e-14 * scale)
            assert math.isclose(a.m2, b.m2, rel_tol=1e-9, abs_tol=1e-300)
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b)
    elif isinstance(a, str) and a.startswith("Exported "):
        assert a.split(" to ")[0] == b.split(" to ")[0]
    else:
        assert a == b


def test_ram_and_disk_mode_identical(new_driver, main_file, monkeypatch, tmp_path):
    T = main_file.data["Time/Time"]
    amap = ArrayMap(np.ascontiguousarray(T, dtype=np.float64), True)
    ram = new_driver()
    gen, model = ram.open(main_file.path)
    assert all(st.ram is not None for st in ram.eng._stores if st.info.length)
    assert all(st.base == 256 for st in ram.eng._stores)
    res_ram = _query_suite(ram, main_file, model, amap, tmp_path)

    # Disk mode: tiny RAM budget, small pyramid budget (larger base), small load blocks.
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", "0.01")
    monkeypatch.setattr(eng_mod, "pyramid_bucket_budget", lambda: 1000)
    monkeypatch.setattr(eng_mod, "BLOCK", 10_007)
    disk = new_driver()
    gen_d, model_d = disk.open(main_file.path)
    stores = disk.eng._stores
    for label in ("Wave/sig", "Wave/offset", "Wave/i32", "Wave/f32", "Wave/XY", "Time/Time", "Misc/stamp",
                  "Misc/cplx"):
        st = stores[main_file.ids[label]]
        assert st.ram is None, label
        assert st.base == 1024, label
        assert st.pyr is not None and st.pyr.complete and st.pyr.base == 1024
    res_disk = _query_suite(disk, main_file, model_d, amap, tmp_path)
    assert len(res_ram) == len(res_disk)
    for a, b in zip(res_ram, res_disk):
        _same(a, b)
    # try_read uses the fast reader for disk channels; npTDMS-only channels would block.
    cid = main_file.ids["Wave/sig"]
    assert nan_equal(disk.eng.try_read(gen_d, cid, 59_990, 60_010), main_file.data["Wave/sig"][59_990:60_010])
    assert disk.eng.try_read(gen_d, main_file.ids["Misc/stamp"], 0, 10) is None


# -- channel kinds ---------------------------------------------------------------------

def test_timestamp_channel_plotted_as_seconds(drv, main_file):
    gen, model = drv.open(main_file.path)
    cid = main_file.ids["Misc/stamp"]
    ts = main_file.data["Misc/stamp"]
    sec = time_to_seconds(ts)
    assert sec[0] == 0.0 and np.all(np.diff(sec) > 0)
    xm = LinearMap(0.0, 1.0)
    kinds = set()
    for ia, ib, px in ((0, N_STAMP - 1, 16), (0, N_STAMP - 1, 50), (100, 180, 100), (4000.5, 5999.5, 20)):
        res = drv.plot([PlotItem(cid, xm, 0, N_STAMP)], ia, ib, px)[cid]
        kinds.add(check_plot(res, sec, xm, ia, ib, px))
    assert kinds == {"raw", "decimated"}
    x, y, _ = drv.plot([PlotItem(cid, xm, 0, N_STAMP)], 0, 10, 100)[cid]
    np.testing.assert_array_equal(y, (ts[:12] - ts[0]) / np.timedelta64(1, "us") / 1e6)
    _, amap, t_ref = drv.xmap(cid)
    assert amap.monotonic
    np.testing.assert_array_equal(amap.x, sec)
    assert float(t_ref) == pytest.approx(float((ts[0] - np.datetime64(0, "us")) / np.timedelta64(1, "us")) / 1e6,
                                         abs=1e-6)
    # Another channel against the timestamp channel as X.
    fid = main_file.ids["Misc/flag"]
    res = drv.plot([PlotItem(fid, amap, 0, N_FLAG)], sec[100], sec[2000], 30)[fid]
    check_plot(res, main_file.f64("Misc/flag"), amap, sec[100], sec[2000], 30)


def test_string_channel_skipped_by_plot_returned_by_table(drv, main_file):
    gen, model = drv.open(main_file.path)
    tid = main_file.ids["Misc/text"]
    xm = LinearMap(0.0, 1.0)
    assert drv.plot([PlotItem(tid, xm, 0, N_TEXT)], 0, 39, 100) == {}
    out = drv.plot([PlotItem(tid, xm, 0, N_TEXT), PlotItem(main_file.ids["Wave/i32"], xm, 0, N)], 0, 39, 100)
    assert set(out) == {main_file.ids["Wave/i32"]}
    i0, vals = drv.table([tid], 0, 100)[tid]
    assert i0 == 0 and list(vals) == list(main_file.data["Misc/text"])
    assert "\n" in vals[0] and "Ω" in vals[0] and '"q0"' in vals[0]
    assert drv.stats([PlotItem(tid, xm, 0, N_TEXT)], 0, 39) == {}


def test_empty_channel(drv, main_file):
    gen, model = drv.open(main_file.path)
    eid = main_file.ids["Misc/empty"]
    ch = model.channels[eid]
    assert ch.length == 0 and ch.kind == "empty" and not ch.plottable
    xm = LinearMap(0.0, 1.0)
    assert drv.plot([PlotItem(eid, xm, 0, 0)], 0, 10, 100) == {}
    assert drv.table([eid], 0, 10) == {}
    assert drv.stats([PlotItem(eid, xm, 0, 0)], 0, 10) == {}
    r = drv.eng.try_read(gen, eid, 0, 10)
    assert r is None or r.size == 0


def test_short_channel_plot_and_stats_without_pyramid(drv, tmp_path):
    """Channels shorter than PYR_MIN_LEN have no pyramid; results are still exact."""
    y = np.random.default_rng(5).standard_normal(4000)
    y[[0, 1999, 3999]] = [9.0, -9.0, 8.0]
    path = tmp_path / "short.tdms"
    write_segments(path, [[("S", "y", y, {"wf_increment": 0.25, "wf_start_offset": -1.0})]])
    gen, model = drv.open(path)
    assert drv.eng._stores[0].pyr is None
    xm = LinearMap(-1.0, 0.25)
    for ia, ib, px in ((0, 3999, 16), (0, 3999, 100), (10.5, 3000.5, 60), (1990, 2010, 60)):
        xa, xb = -1.0 + ia * 0.25, -1.0 + ib * 0.25
        check_plot(drv.plot([PlotItem(0, xm, 0, 4000)], xa, xb, px)[0], y, xm, xa, xb, px)
        res = drv.stats([PlotItem(0, xm, 0, 4000)], xa, xb, [xa])[0]
        check_stats(res, y, y, xm, xa, xb, 0, 4000, [xa])


def test_index_file_and_stale_index(drv, tmp_path):
    y = np.arange(5000, dtype=np.float64)
    path = tmp_path / "idx.tdms"
    write_segments(path, [[("g", "y", y[:3000], {})], [("g", "y", y[3000:], {})]], index_file=True)
    assert Path(str(path) + "_index").exists()
    gen, model = drv.open(path)
    assert model.index_used and model.channels[0].length == 5000 and model.warnings == []
    np.testing.assert_array_equal(drv.table([0], 0, 5000)[0][1], y)
    # Append data without updating the index: the stale index must be ignored.
    more = np.arange(5000, 6000, dtype=np.float64)
    from nptdms import ChannelObject, TdmsWriter

    with TdmsWriter(str(path), mode="a") as w:
        w.write_segment([ChannelObject("g", "y", more)])
    gen2, model2 = drv.open(path)
    assert not model2.index_used
    assert any("tdms_index" in w for w in model2.warnings)
    assert model2.channels[0].length == 6000
    np.testing.assert_array_equal(drv.table([0], 0, 6000)[0][1], np.arange(6000, dtype=np.float64))
    xm = LinearMap(0.0, 1.0)
    check_plot(drv.plot([PlotItem(0, xm, 0, 6000)], 0, 5999, 50)[0], np.arange(6000.0), xm, 0, 5999, 50)


def test_truncated_file_warnings(drv, tmp_path):
    y = np.arange(2000, dtype=np.float64)
    path = tmp_path / "cut.tdms"
    write_segments(path, [[("g", "y", y[:1000], {})], [("g", "y", y[1000:], {})]])
    data = path.read_bytes()
    path.write_bytes(data[:-100])
    gen, model = drv.open(path)
    assert any("incomplete" in w for w in model.warnings)
    n = model.channels[0].length
    assert 1000 <= n < 2000
    np.testing.assert_array_equal(drv.table([0], 0, n)[0][1], y[:n])


# -- shutdown ------------------------------------------------------------------------

def test_shutdown_joins_thread(qtbot, main_file):
    eng = DataEngine()
    d = Driver(qtbot, eng)
    gen, model = d.open(main_file.path)
    src = eng._source
    assert eng._thread.is_alive()
    eng.shutdown()
    assert not eng._thread.is_alive()
    assert eng._source is None and src._fd == -1
    # Posting after shutdown does not raise.
    eng.request_plot(PlotRequest(1, [], 0, 1, 100))
    assert eng.try_read(gen, 0, 0, 10) is None
    eng.shutdown()  # twice is fine


def test_shutdown_during_load(qtbot, big_file, monkeypatch):
    import time

    monkeypatch.setattr(eng_mod, "BLOCK", 1024)
    eng = DataEngine()
    d = Driver(qtbot, eng)
    gen, model = d.open(big_file.path, loaded=False)
    qtbot.waitUntil(lambda: bool(d.rec.of("progress", gen)[1:]), timeout=10_000)  # loading
    t = time.perf_counter()
    eng.shutdown()
    assert time.perf_counter() - t < 3.0
    assert not eng._thread.is_alive()
    assert eng._source is None
    # Pyramid helper threads (if any) stop at their next block.
    pool = getattr(eng, "_pool", None)
    helpers = list(pool._threads) if pool is not None else []
    qtbot.waitUntil(lambda: not any(h.is_alive() for h in helpers), timeout=5_000)


def test_shutdown_idle_engine(qtbot):
    eng = DataEngine()
    eng.shutdown()
    assert not eng._thread.is_alive()


# -- real FlexLogger file ----------------------------------------------------------------

@pytest.mark.skipif(not REAL_FILE.exists(), reason="sample file not available")
@pytest.mark.parametrize("ram_mb", ["2048", "0.5"], ids=["ram", "mixed"])
def test_real_flexlogger_file(drv, tmp_path, monkeypatch, ram_mb):
    monkeypatch.setenv("TDMSVIEWER_RAM_MB", ram_mb)  # 0.5 MB: one channel in RAM, 15 on disk
    ids, data = _read_back(REAL_FILE)
    gen, model = drv.open(REAL_FILE)
    in_ram = sum(st.ram is not None for st in drv.eng._stores)
    assert in_ram == (16 if ram_mb == "2048" else 1)
    assert [g.name for g in model.groups] == ["PXIe-4303 (PXI2Slot3)", "PXIe-4303 (PXI2Slot5)", "PXIe-6363", "Time"]
    assert len(model.channels) == 16
    assert all(c.length == 44_275 and c.kind == "float" and c.dtype == np.float64 for c in model.channels)
    assert all(c.fast for c in model.channels)
    assert model.n_segments == 1 and model.t_ref is None
    assert all(c.wf_increment is None for c in model.channels)
    upd = sorted(c for g, cids in drv.rec.of("channelsUpdated", gen) for c in cids)
    assert upd == list(range(16))
    tid = ids["Time/Time"]
    by = {c.label: c for c in model.channels}
    assert by["Time/Time"].unit == "s"
    _, amap, t_ref = drv.xmap(tid)
    T = data["Time/Time"]
    assert amap.monotonic and t_ref is None
    np.testing.assert_array_equal(amap.x, T)
    assert T[0] == pytest.approx(0.05, abs=1e-4) and T[-1] == pytest.approx(4427.45, abs=1e-4)
    n = 44_275
    for label, cid in ids.items():
        y = data[label].astype(np.float64)
        for xa, xb, px in ((T[0], T[-1], 800), (T[0], T[-1], 100), (100.0, 105.0, 300), (1234.5, 3999.9, 64)):
            res = drv.plot([PlotItem(cid, amap, 0, n)], xa, xb, px)[cid]
            check_plot(res, y, amap, xa, xb, px)
        res = drv.stats([PlotItem(cid, amap, 0, n)], 100.0, 3000.0, [2000.02])[cid]
        check_stats(res, y, data[label], amap, 100.0, 3000.0, 0, n, [2000.02])
    i0, vals = drv.table([ids["PXIe-4303 (PXI2Slot5)/Imon"]], 44_000, 50_000)[ids["PXIe-4303 (PXI2Slot5)/Imon"]]
    assert i0 == 44_000
    np.testing.assert_array_equal(vals, data["PXIe-4303 (PXI2Slot5)/Imon"][44_000:])
    out = tmp_path / "real.csv"
    v0 = ids["PXIe-4303 (PXI2Slot3)/V0"]
    msg = drv.export(out, [PlotItem(v0, amap, 0, n), PlotItem(tid, amap, 0, n)], 10.0, 12.0,
                     ["i", "t", "V0", "i", "t", "Time"])
    k0, k1 = int(np.searchsorted(T, 10.0)), int(np.searchsorted(T, 12.0, side="right"))
    assert msg.startswith(f"Exported {k1 - k0} rows")
    rows = _read_csv(out)[1:]
    assert [int(r[0]) for r in rows] == list(range(k0, k1))
    assert [float(r[2]) for r in rows] == data["PXIe-4303 (PXI2Slot3)/V0"][k0:k1].tolist()
    assert [float(r[5]) for r in rows] == T[k0:k1].tolist()
    assert drv.rec.of("message", gen) == []


# -- robustness of answers ---------------------------------------------------------

def test_x_task_error_is_answered(drv, small_file, monkeypatch):
    """A read error while loading X values gives an xReady error text (GUI falls back)."""
    gen, model = drv.open(small_file.path)

    def broken(self, st, i0, i1):
        raise OSError("disk gone")

    monkeypatch.setattr(DataEngine, "_read_f64", broken)
    res = drv.xmap(0)
    assert isinstance(res, str) and "disk gone" in res


def test_open_error_after_metadata_gives_open_failed(new_driver, small_file, monkeypatch):
    """An error after the metadata is read still ends in openFailed (never stuck at 'Opening')."""
    d = new_driver()

    def bad_budget():
        raise ValueError("could not convert string to float: '1,5'")

    monkeypatch.setattr(eng_mod, "ram_budget_bytes", bad_budget)
    gen = d.eng.open(str(small_file.path))
    g, msg = d.wait("openFailed", lambda g, m: g == gen)
    assert "1,5" in msg
    assert d.eng._source is None  # the file is closed again


def test_file_changed_on_disk_is_reported(drv, tmp_path):
    """A rewrite of the open file gives one clear warning (display can mix versions)."""
    path = tmp_path / "live.tdms"
    write_segments(path, [[("G", "a", np.zeros(5000), {})]])
    gen, model = drv.open(path)
    time.sleep(0.6)
    write_segments(path, [[("G", "a", np.ones(5000), {})]])  # same name, new content
    items = [PlotItem(0, LinearMap(0.0, 1.0), 0, 5000)]
    drv.plot(items, 0, 5000, 100)
    msgs = [m for g, m in drv.rec.of("message") if g == gen]
    assert any("changed on disk" in m for m in msgs), msgs
    drv.plot(items, 0, 2500, 100)
    time.sleep(0.6)
    drv.plot(items, 0, 5000, 100)
    msgs = [m for g, m in drv.rec.of("message") if g == gen and "changed on disk" in m]
    assert len(msgs) == 1  # reported once


def test_xy_point_budget_is_per_request(drv, main_file, monkeypatch):
    """Many X-Y curves share one point budget (no memory explosion)."""
    monkeypatch.setattr(eng_mod, "XY_MAX_POINTS", 40_000)
    gen, model = drv.open(main_file.path)
    _, amap, _ = drv.xmap(main_file.ids["Wave/XY"])
    cids = [main_file.ids[k] for k in ("Wave/sig", "Wave/offset", "Wave/f32", "Wave/i32")]
    out = drv.plot([PlotItem(c, amap, 0, N) for c in cids], 0.0, 1.0, 300)
    total = sum(r[0].size for r in out.values())
    assert total <= 40_000 + 4 * 2 * (N // 10_000 + 1)


# -- fragmented files: channels of one family load in one pass -------------------------------

@pytest.mark.parametrize("ram_mb", ["2048", "0"], ids=["ram", "disk"])
def test_fragmented_family_loads_in_one_pass(new_driver, tmp_path, monkeypatch, ram_mb):
    """300 small segments: 3 channels load together; values, pyramids and stats stay exact."""
    from nptdms import ChannelObject, TdmsWriter

    monkeypatch.setenv("TDMSVIEWER_RAM_MB", ram_mb)
    rng = np.random.default_rng(7)
    n_seg, npc = 300, 100
    data = {"a": rng.normal(size=n_seg * npc),
            "b": rng.integers(-1000, 1000, n_seg * npc).astype(np.int32),
            "c": rng.normal(size=n_seg * npc).astype(np.float32)}
    data["a"][12_345] = 99.0  # spike
    path = tmp_path / "frag.tdms"
    with TdmsWriter(str(path)) as w:
        for k in range(n_seg):
            w.write_segment([ChannelObject("G", nm, v[k * npc:(k + 1) * npc]) for nm, v in data.items()])
    calls = []
    orig = eng_mod.DataEngine._load_family

    def spy(self, gen, src, group, builds, prog):
        calls.append(sorted(st.info.name for st in group))
        yield from orig(self, gen, src, group, builds, prog)

    monkeypatch.setattr(eng_mod.DataEngine, "_load_family", spy)
    drv = new_driver()
    gen, model = drv.open(path)
    assert calls == [["a", "b", "c"]]
    n = n_seg * npc
    for st in drv.eng._stores:
        ref = data[st.info.name]
        assert st.done and st.fast is not None and st.fast.fragmented
        assert st.pyr is not None and st.pyr.complete
        if ram_mb == "0":
            assert st.ram is None
        else:
            np.testing.assert_array_equal(st.ram, ref)
        xm = LinearMap(0.0, 1.0)
        res = drv.plot([PlotItem(st.info.id, xm, 0, n)], 0.0, n - 1.0, 300)
        check_plot(res[st.info.id], ref.astype(np.float64), xm, 0.0, n - 1.0, 300)
        s = absolute_stats(drv.stats([PlotItem(st.info.id, xm, 0, n)], 0.0, n - 1.0)[st.info.id])
        f = ref.astype(np.float64)
        assert (s.n, s.min, s.max) == (n, f.min(), f.max())
        assert math.isclose(s.mean, f.mean(), rel_tol=1e-12, abs_tol=1e-12)
