"""GUI tests: main window with the real engine (offscreen Qt)."""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from nptdms import ChannelObject, GroupObject, TdmsWriter
from pyqtgraph.Qt.QtCore import Qt

pytest.importorskip("pytestqt")

from tdmsviewer.engine import DataEngine  # noqa: E402
from tdmsviewer.mainwindow import MainWindow  # noqa: E402

SAMPLE = "/root/.claude/uploads/5899cc9d-0cc2-5e06-94ba-ed80568becb4/24399bc7-CBL-1252_test_012_2026-07-28_12-05-36.tdms"


@pytest.fixture
def gui_file(tmp_path):
    """Two groups, waveform properties, one spike, a string channel."""
    path = str(tmp_path / "gui.tdms")
    n = 50_000
    t = np.arange(n)
    y = np.sin(t / 500.0)
    y[31_337] = 7.5  # spike
    props = {"wf_increment": 1e-3, "wf_start_offset": 0.0, "unit_string": "V"}
    with TdmsWriter(path) as w:
        w.write_segment([
            GroupObject("A", properties={"gp": 1}),
            ChannelObject("A", "sine", y, properties=props),
            ChannelObject("A", "ramp", t.astype(np.float64) * 0.1, properties=props),
            ChannelObject("B", "count", t.astype(np.int32)),
            ChannelObject("B", "text", np.array([f"s{i}" for i in range(100)])),
        ])
    return path


@pytest.fixture
def win(qtbot, tmp_path):
    from pyqtgraph.Qt.QtCore import QSettings

    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path / "cfg"))
    QSettings("tdmsviewer", "tdmsviewer").clear()
    engine = DataEngine()
    w = MainWindow(engine)
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    yield w
    engine.shutdown()


def _open(qtbot, w, path):
    with qtbot.waitSignal(w.engine.opened, timeout=10_000):
        w.open_file(path)
    qtbot.waitUntil(lambda: all(s.done for s in w.engine._stores), timeout=20_000)
    qtbot.wait(300)


def _select(w, item):
    w.tree.clearSelection()
    w.tree.setCurrentItem(item)
    item.setSelected(True)


def test_open_shows_all_channels(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    root = win.tree.topLevelItem(0)
    assert root.text(0) == "gui.tdms"
    assert root.childCount() == 2
    assert [root.child(0).child(i).text(0) for i in range(2)] == ["sine", "ramp"]
    # File selected: all channels in table, plottable ones in the graph.
    assert win.values_model.columnCount() == 4
    assert win.plot.legend.count() == 4
    assert len(win.plot.plotted_cids()) == 3  # string channel is table only
    assert win.values_model.headerData(0, Qt.Horizontal) == "A\nsine [00]"


def test_group_and_channel_selection_and_properties(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    root = win.tree.topLevelItem(0)
    _select(win, root.child(0))
    qtbot.wait(100)
    assert [c.name for c in win.current] == ["sine", "ramp"]
    names = [win.prop_model.items[i][0] for i in range(win.prop_model.rowCount())]
    assert names == ["gp"]
    _select(win, root.child(0).child(0))
    qtbot.wait(100)
    props = {k: v for k, v, _ in win.prop_model.items}
    assert props["NI_ChannelLength"] == "50000"
    assert props["NI_DataType"] == "10"
    assert props["wf_increment"] == "0.001"
    keys = [k for k, _, _ in win.prop_model.items]
    assert keys == sorted(keys)  # NI order: ASCII sort


def test_plot_keeps_spike_and_fit_range(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(0))
    qtbot.wait(400)
    x0, x1 = win.plot.view_x_range()
    assert x0 == pytest.approx(0.0, abs=1e-9)
    assert x1 == pytest.approx(49.999, abs=1e-9)
    cid = win.current[0].id
    curve = win.plot._curves[cid]
    xs, ys = curve.getData()
    assert np.nanmax(ys) == 7.5  # peak preserved at full view
    assert xs.size < 20_000  # decimated


def test_zoom_to_raw_samples(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(0))
    qtbot.wait(300)
    win.plot.set_view(31.3, 31.4)
    qtbot.wait(400)
    xs, ys = win.plot._curves[win.current[0].id].getData()
    k = np.searchsorted(xs, 31.337 - 1e-9)
    assert xs[k] == pytest.approx(31.337)
    assert ys[k] == 7.5
    # Previous view restores the full range.
    win.plot.push_history()
    win.plot.set_view(10, 11)
    win.plot.back()
    assert win.plot.view_x_range()[0] == pytest.approx(31.3)


def test_range_controls(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(0))
    win.all_check.setChecked(False)
    win.start_spin.setValue(1000)
    win.samples_spin.setValue(500)
    qtbot.wait(300)
    assert win.values_model.rowCount() == 500
    assert win.values_model.headerData(0, Qt.Vertical) == "1000"
    x0, x1 = win.plot.view_x_range()
    assert x0 == pytest.approx(1.0)
    assert x1 == pytest.approx(1.499)


def test_values_table_and_copy(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(1))  # group B: count, text
    qtbot.wait(300)
    m = win.values_model
    assert m.rowCount() == 50_000
    qtbot.waitUntil(lambda: m.data(m.index(5, 0)) is not None, timeout=5000)
    assert m.data(m.index(5, 0)) == "5"
    assert m.data(m.index(5, 1)) == "s5"
    assert m.data(m.index(200, 1)) is None  # string channel has 100 values
    sel = win.values_view.selectionModel()
    from pyqtgraph.Qt.QtCore import QItemSelection, QItemSelectionModel

    sel.select(QItemSelection(m.index(2, 0), m.index(3, 1)), QItemSelectionModel.Select)
    win._copy_values()
    from pyqtgraph.Qt.QtGui import QGuiApplication

    qtbot.waitUntil(lambda: QGuiApplication.clipboard().text() == "2\ts2\n3\ts3", timeout=5000)


def test_time_channel_as_x_and_stats(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0))
    i = win.x_combo.findText("Channel: A/ramp")
    assert i > 0
    win.x_combo.setCurrentIndex(i)
    with qtbot.waitSignal(win.engine.xReady, timeout=5000):
        win._x_source_changed(i)
    qtbot.wait(400)
    x0, x1 = win.plot.view_x_range()
    assert x0 == pytest.approx(0.0)
    assert x1 == pytest.approx(4999.9)
    win.tabs.setCurrentIndex(1)
    win.plot.set_cursors_visible(True)
    win.plot.cursors[0].setValue(100.0)
    win.plot.cursors[1].setValue(200.0)
    with qtbot.waitSignal(win.engine.statsReady, timeout=5000):
        win._request_stats()
    sine = win.current[0].id
    res = win.stats_model.values[sine]
    s = res["stats"]
    assert s.n == 1001  # ramp 100.0 .. 200.0 in steps of 0.1
    assert res["cursors"][0][0] == 1000


def test_export_csv(qtbot, win, gui_file, tmp_path, monkeypatch):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(1))  # ramp
    qtbot.wait(300)
    win.plot.set_view(1.0, 1.004)
    qtbot.wait(300)
    out = str(tmp_path / "out.csv")
    monkeypatch.setattr("tdmsviewer.mainwindow.QFileDialog.getSaveFileName", lambda *a, **k: (out, ""))
    with qtbot.waitSignal(win.engine.exportDone, timeout=5000):
        win.export_csv()
    rows = list(csv.reader(open(out)))
    assert rows[0][0].startswith("A/ramp")
    assert [r[0] for r in rows[1:]] == ["1000", "1001", "1002", "1003", "1004"]
    assert float(rows[1][2]) == 100.0


@pytest.mark.skipif(not os.path.exists(SAMPLE), reason="sample file not present")
def test_real_sample_file(qtbot, win):
    _open(qtbot, win, SAMPLE)
    assert len(win.model.channels) == 16
    assert all(c.fast for c in win.model.channels)
    assert win.plot.legend.count() == 16
    assert win.x_combo.itemText(0).startswith("Waveform time")
    # NI shows 44275 samples at 1 s per sample: 0 .. 12:17:54.
    x0, x1 = win.plot.view_x_range()
    assert (x0, x1) == (0.0, 44274.0)


def test_legend_styles_and_zoom_about_point(qtbot, win, gui_file):
    from pyqtgraph.Qt.QtCore import QPointF
    from pyqtgraph.Qt.QtGui import QColor

    from tdmsviewer.plotpanel import STYLE_BOTH, STYLE_POINTS, TOOL_ZOOMPT

    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0))
    qtbot.wait(300)
    cid = win.current[0].id
    curve = win.plot._curves[cid]
    win.plot.set_style([cid], color=QColor("#123456"), width=3)
    assert curve.opts["pen"].width() == 3 and curve.opts["pen"].color().name() == "#123456"
    win.plot.set_style([cid], mode=STYLE_POINTS)
    assert curve.opts["pen"].style() == Qt.NoPen and curve.opts["symbol"] == "o"  # points only
    win.plot.set_style([cid], mode=STYLE_BOTH)
    assert curve.opts["pen"].style() != Qt.NoPen and curve.opts["symbol"] == "o"
    # The style stays when the channel is shown again (same session).
    _select(win, win.tree.topLevelItem(0).child(0).child(0))
    qtbot.wait(300)
    assert win.plot._curves[cid].opts["pen"].width() == 3
    # Zoom about point: click zooms in 2x around the point.
    win.plot.set_tool(TOOL_ZOOMPT)
    x0, x1 = win.plot.view_x_range()
    vb = win.plot.vb

    class Ev:
        def __init__(self, shift):
            self._shift = shift

        def button(self):
            return Qt.LeftButton

        def double(self):
            return False

        def modifiers(self):
            return Qt.ShiftModifier if self._shift else Qt.NoModifier

        def pos(self):
            return vb.mapFromView(QPointF((x0 + x1) / 2, 0.0))

        def accept(self):
            pass

    vb.mouseClickEvent(Ev(False))
    a, b = win.plot.view_x_range()
    assert (b - a) == pytest.approx((x1 - x0) / 2, rel=1e-6)
    vb.mouseClickEvent(Ev(True))
    a, b = win.plot.view_x_range()
    assert (b - a) == pytest.approx(x1 - x0, rel=1e-6)


def test_export_csv_absolute_time_column(qtbot, win, tmp_path, monkeypatch):
    """wf_start_time + wf_increment: a UTC ISO time column, exact to 1 ns."""
    path = str(tmp_path / "timed.tdms")
    t0 = np.datetime64("2026-07-28T10:05:36.000000500", "ns")
    props = {"wf_increment": 1e-3, "wf_start_offset": 0.0, "wf_start_time": t0}
    with TdmsWriter(path) as w:
        w.write_segment([ChannelObject("A", "sig", np.arange(20_000, dtype=np.float64), properties=props),
                         ChannelObject("A", "plain", np.arange(20_000, dtype=np.float64))])
    _open(qtbot, win, path)
    _select(win, win.tree.topLevelItem(0).child(0))
    qtbot.wait(300)
    win.plot.set_view(1.0, 1.002)
    qtbot.wait(300)
    out = str(tmp_path / "t.csv")
    monkeypatch.setattr("tdmsviewer.mainwindow.QFileDialog.getSaveFileName", lambda *a, **k: (out, ""))
    with qtbot.waitSignal(win.engine.exportDone, timeout=5000):
        win.export_csv()
    rows = list(csv.reader(open(out)))
    assert rows[0][2] == "A/sig time [UTC]" and rows[0][6] == "A/plain time [UTC]"
    assert [r[0] for r in rows[1:]] == ["1000", "1001", "1002"]
    # The npTDMS writer stores whole us (the 500 ns are lost): 10:05:36.000000 + 1.000 s
    assert rows[1][2] == "2026-07-28T10:05:37.000000000Z" and rows[2][2] == "2026-07-28T10:05:37.001000000Z"
    assert rows[1][6] == ""  # no wf_start_time: no absolute time
    assert float(rows[1][3]) == 1000.0
