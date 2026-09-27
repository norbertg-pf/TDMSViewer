"""GUI regression tests for the review findings (table limits, copy, plot speed, labels, reload)."""

from __future__ import annotations

import csv
import os
import shutil
import time

import numpy as np
import pytest
from nptdms import ChannelObject, TdmsWriter
from pyqtgraph.Qt.QtCore import QItemSelection, QItemSelectionModel, QMimeData, QPointF, Qt, QUrl
from pyqtgraph.Qt.QtGui import QDropEvent, QFontMetrics, QGuiApplication

pytest.importorskip("pytestqt")

import test_gui  # noqa: E402

from tdmsviewer import theme  # noqa: E402
from tdmsviewer.plotpanel import MAX_MARKER_PLOTS, LegendEntry, PlotPanel, _line_icon  # noqa: E402
from tdmsviewer.tables import WIDEST_FLOAT, ValuesModel, ValuesView, text_width  # noqa: E402
from tdmsviewer.tdmsfile import ChannelInfo  # noqa: E402

T0 = np.datetime64("2026-07-28T10:05:36", "us")

# Fixtures and helpers of test_gui.py.
win = test_gui.win
gui_file = test_gui.gui_file
_open = test_gui._open
_select = test_gui._select


def _fake_channel(n: int, cid: int = 0, name: str = "big") -> ChannelInfo:
    return ChannelInfo(id=cid, group="G", name=name, path=f"/'G'/'{name}'", length=n, kind="float",
                       dtype=np.dtype("f8"), type_code=10, unit="", properties={}, wf_increment=None,
                       wf_start_offset=None, wf_start_time=None)


def _wait_x(qtbot, w):
    qtbot.waitUntil(lambda: w.x_array is not None, timeout=5000)
    qtbot.wait(300)


@pytest.fixture
def types_file(tmp_path):
    """Timestamp, complex and bool channels with wf_start_time."""
    path = str(tmp_path / "types.tdms")
    n = 5000
    ts = T0 + (np.arange(n) * 1000).astype("timedelta64[us]")
    z = (np.arange(n) + 1j * -np.arange(n)).astype(np.complex128)
    props = {"wf_increment": 1e-3, "wf_start_offset": 0.0, "wf_start_time": T0}
    with TdmsWriter(path) as w:
        w.write_segment([ChannelObject("T", "stamp", ts, properties=props),
                         ChannelObject("T", "cplx", z, properties=dict(props, unit_string="V")),
                         ChannelObject("T", "flag", (np.arange(n) % 2).astype(bool), properties=props)])
    return path


@pytest.fixture
def same_names_file(tmp_path):
    path = str(tmp_path / "samename.tdms")
    with TdmsWriter(path) as w:
        w.write_segment([ChannelObject(g, c, np.arange(1000.0) + k)
                         for k, (g, c) in enumerate((g, c) for g in ("Card1", "Card2") for c in ("U", "I"))])
    return path


# -- 1: values table row limit -------------------------------------------------------------

def test_values_table_rows_clamped_to_header_pixel_limit(qtbot):
    """Above (2**31 - 1) / row height rows QHeaderView hangs: the model shows the first rows only."""
    m = ValuesModel()
    m.reader = lambda cid, a, b: np.arange(a, b, dtype=np.float64)
    v = ValuesView()
    qtbot.addWidget(v)
    v.setModel(m)
    limit = v.max_rows()
    assert limit == (2**31 - 1) // v.verticalHeader().defaultSectionSize() - 1
    m.set_channels([_fake_channel(3_000_000_000)], 1000, None)
    assert m.rowCount() == limit and m.clamped and m.total_rows == 3_000_000_000 - 1000
    v.resize(600, 400)
    v.show()
    t = time.perf_counter()
    qtbot.wait(100)
    sb = v.verticalScrollBar()
    sb.setValue(sb.maximum())
    qtbot.wait(100)
    assert time.perf_counter() - t < 5.0
    assert v.verticalHeader().length() <= 2**31 - 1
    # Headers keep the absolute sample index.
    assert m.headerData(m.rowCount() - 1, Qt.Vertical) == str(1000 + limit - 1)
    last = m.rowCount() - 1
    qtbot.waitUntil(lambda: m.data(m.index(last, 0)) is not None, timeout=5000)
    assert m.data(m.index(last, 0)) == repr(float(1000 + last))


def test_values_note_when_clamped(qtbot, win):
    m = win.values_model
    m.set_channels([_fake_channel(200_000_000)], 5, None)
    win._update_values_note()
    assert m.clamped and not win.values_note.isHidden()
    assert "Start index" in win.values_note.text() and f"{m.rows:,}" in win.values_note.text()
    m.set_channels([_fake_channel(1000)], 0, None)
    win._update_values_note()
    assert win.values_note.isHidden()


# -- 2: numbers are never cut on the right -----------------------------------------------

def test_statistics_columns_fit_numbers(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0))  # sine, ramp
    win.tabs.setCurrentIndex(1)
    win.plot.set_cursors_visible(True)
    win.plot.set_cursor_positions(10.0, 30.123)
    with qtbot.waitSignal(win.engine.statsReady, timeout=5000):
        win._request_stats()
    qtbot.wait(100)
    sm, sv = win.stats_model, win.stats_view
    assert sv.textElideMode() == Qt.ElideLeft
    fm = QFontMetrics(theme.mono_font())
    checked = 0
    for r in range(sm.rowCount()):
        for c in range(3, sm.columnCount()):
            txt = sm._cell(r, c)
            if txt:
                checked += 1
                assert fm.horizontalAdvance(txt) + 6 <= sv.columnWidth(c), (sm.headerData(c, Qt.Horizontal), txt)
    assert checked > 10


def test_values_column_width_fits_widest_float(qtbot, win):
    v = win.values_view
    assert v.textElideMode() == Qt.ElideLeft
    need = QFontMetrics(theme.mono_font()).horizontalAdvance(WIDEST_FLOAT)
    assert v.horizontalHeader().defaultSectionSize() >= need + 6
    assert text_width(WIDEST_FLOAT) == v.horizontalHeader().defaultSectionSize()


# -- 3: Ctrl+A / Ctrl+C on huge tables ----------------------------------------------------

def test_select_all_and_copy_huge_table_is_fast(qtbot, win, monkeypatch):
    m = win.values_model
    m.reader = lambda cid, a, b: np.arange(a, b, dtype=np.float64)
    m.set_channels([_fake_channel(5_000_000, 0, "a"), _fake_channel(5_000_000, 1, "b")], 0, None)
    qtbot.wait(50)
    t = time.perf_counter()
    win.values_view.selectAll()
    qtbot.wait(50)
    assert time.perf_counter() - t < 2.0  # was 16 s: header asked flags() of every row
    boxes = []
    monkeypatch.setattr("tdmsviewer.mainwindow.QMessageBox.information", lambda *a, **k: boxes.append(a[2]))
    t = time.perf_counter()
    win._copy_values()
    assert time.perf_counter() - t < 1.0  # was 66 s and 2.9 GB
    assert boxes and "too large" in boxes[0]


def test_copy_two_ranges(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(1))  # group B: count, text
    qtbot.wait(300)
    m = win.values_model
    qtbot.waitUntil(lambda: m.data(m.index(5, 0)) is not None, timeout=5000)
    sel = win.values_view.selectionModel()
    sel.select(QItemSelection(m.index(2, 0), m.index(3, 0)), QItemSelectionModel.ClearAndSelect)
    sel.select(QItemSelection(m.index(3, 1), m.index(4, 1)), QItemSelectionModel.Select)
    win._copy_values()
    qtbot.waitUntil(lambda: QGuiApplication.clipboard().text() == "2\t\n3\ts3\n\ts4", timeout=5000)


# -- 4: no Plot Options (transforms), harmless Mouse Mode ---------------------------------

def test_plot_options_menu_disabled(qtbot, win):
    pi = win.plot.plot.getPlotItem()
    vb = win.plot.vb
    assert not pi.menuEnabled() and vb.menuEnabled()
    visible = [a.text() for a in vb.menu.actions() if a.isVisible() and a.text()]
    assert "View All" in visible and "Mouse Mode" not in visible
    vb.setMouseMode(vb.RectMode)
    assert vb.state["mouseMode"] == vb.PanMode


# -- 5: cursors ------------------------------------------------------------------------------

def test_cursors_placed_in_view_and_cleared_on_close(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(0))
    qtbot.wait(300)
    win.plot.btn_cursors.click()
    x0, x1 = win.plot.view_x_range()
    a, b = win.plot.cursor_positions()
    assert a == pytest.approx(x0 + (x1 - x0) / 3) and b == pytest.approx(x0 + 2 * (x1 - x0) / 3)
    win.tabs.setCurrentIndex(1)
    # X source change: positions in samples (30000, 40000) are outside the new view (0 .. 50 s).
    i = win.x_combo.findText("Sample index")
    win.x_combo.setCurrentIndex(i)
    win._x_source_changed(i)
    qtbot.wait(300)
    win.plot.set_cursor_positions(30000.0, 40000.0)
    win.x_combo.setCurrentIndex(0)
    win._x_source_changed(0)
    qtbot.wait(300)
    x0, x1 = win.plot.view_x_range()
    assert x1 < 100 and all(x0 <= c <= x1 for c in win.plot.cursor_positions())
    with qtbot.waitSignal(win.engine.statsReady, timeout=5000):
        win._request_stats()
    res = win.stats_model.values[win.current[0].id]
    assert res["stats"].n > 10_000  # not N = 0 of cursors outside the data
    win.close_file()
    assert win.plot.readout.text() == "" and win.stats_label.text() == ""


# -- 6: complex and timestamp channels are labeled -----------------------------------------

def test_complex_and_timestamp_labels(qtbot, win, types_file, set_tz):
    set_tz("UTC")
    _open(qtbot, win, types_file)
    g = win.tree.topLevelItem(0).child(0)
    _select(win, g.child(0))  # stamp
    qtbot.wait(300)
    assert win.plot.legend.item(0).text() == "00  stamp (s since 2026-07-28 10:05:36 UTC)"
    assert "2026-07-28 10:05:36" in win.plot.legend.item(0).toolTip()
    assert win.stats_model._cell(0, 2) == "s since 2026-07-28 10:05:36 UTC"
    assert win.plot.plot.getPlotItem().getAxis("left").labelText == "s since 2026-07-28 10:05:36 UTC"
    _select(win, g.child(1))  # cplx
    qtbot.wait(300)
    assert win.plot.legend.item(0).text() == "00  cplx |z|"
    assert "|z|" in win.plot.legend.item(0).toolTip()
    assert win.stats_model._cell(0, 2) == "|z| [V]"
    assert win.plot.plot.getPlotItem().getAxis("left").labelText == "|z| [V]"
    _select(win, win.tree.topLevelItem(0))  # all: mixed kinds
    qtbot.wait(300)
    label = win.plot.plot.getPlotItem().getAxis("left").labelText
    assert "complex as |z|" in label and "timestamps as s since first sample" in label


def test_timestamp_zero_from_async_table_read(qtbot, win, types_file, set_tz, monkeypatch):
    """Timestamp channel not in RAM: the zero comes from an engine table request."""
    set_tz("UTC")
    _open(qtbot, win, types_file)
    monkeypatch.setattr(win, "_try_read", lambda *a: None)
    win._t0.clear()
    with qtbot.waitSignal(win.engine.tableReady, timeout=5000):
        _select(win, win.tree.topLevelItem(0).child(0).child(0))
    qtbot.wait(100)
    assert win.plot.legend.item(0).text() == "00  stamp (s since 2026-07-28 10:05:36 UTC)"


# -- 7: X channel only for the same file ---------------------------------------------------

def test_x_channel_stored_per_file(qtbot, win, gui_file, tmp_path):
    other = str(tmp_path / "other.tdms")
    shutil.copy(gui_file, other)
    _open(qtbot, win, gui_file)
    i = win.x_combo.findText("Channel: A/ramp")
    win.x_combo.setCurrentIndex(i)
    win._x_source_changed(i)
    _wait_x(qtbot, win)
    _open(qtbot, win, other)
    assert win.x_combo.currentIndex() == 0 and win.x_source == ("wave", None)
    assert all(win.plot.legend.item(k).flags() & Qt.ItemIsEnabled for k in range(3))
    _open(qtbot, win, gui_file)
    _wait_x(qtbot, win)
    assert win.x_combo.currentText() == "Channel: A/ramp"


# -- 8: reload and range changes keep the view -------------------------------------------

def test_reload_keeps_selection_hidden_zoom_cursors(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0))  # sine, ramp
    qtbot.wait(300)
    win.plot.legend.item(0).setCheckState(Qt.Unchecked)  # hide sine
    win.plot.set_view(10.0, 20.0)
    win.plot.set_cursors_visible(True)
    win.plot.set_cursor_positions(12.0, 15.0)
    qtbot.wait(300)
    with qtbot.waitSignal(win.engine.opened, timeout=10_000):
        win.reload()
    qtbot.waitUntil(lambda: all(s.done for s in win.engine._stores), timeout=20_000)
    qtbot.wait(400)
    assert [c.name for c in win.current] == ["sine", "ramp"]
    assert win.tree.currentItem().text(0) == "A"
    hidden = {win.model.channels[c].name for c in win.plot.hidden_cids()}
    assert hidden == {"sine"}
    assert win.plot.view_x_range() == pytest.approx((10.0, 20.0))
    assert win.plot.cursor_positions() == pytest.approx([12.0, 15.0])


def test_range_change_keeps_hidden_plots(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0))
    qtbot.wait(300)
    pool = [id(c) for c in win.plot._pool]
    win.plot.legend.item(1).setCheckState(Qt.Unchecked)
    ramp = win.current[1].id
    win.start_spin.setValue(100)
    qtbot.wait(300)
    assert win.plot.hidden_cids() == {ramp}
    assert win.plot.legend.item(1).checkState() == Qt.Unchecked
    assert [id(c) for c in win.plot._pool] == pool  # curves used again, not created
    # A new tree selection shows all plots again (NI).
    _select(win, win.tree.topLevelItem(0).child(0))
    _select(win, win.tree.topLevelItem(0))
    qtbot.wait(300)
    assert win.plot.hidden_cids() == set()


def test_reload_after_failed_open_uses_path_box(qtbot, win, gui_file, tmp_path, monkeypatch):
    path = str(tmp_path / "later.tdms")
    with open(path, "wb") as fh:
        fh.write(b"not a tdms file" * 10)
    monkeypatch.setattr("tdmsviewer.mainwindow.QMessageBox.critical", lambda *a, **k: None)
    with qtbot.waitSignal(win.engine.openFailed, timeout=10_000):
        win.open_file(path)
    assert win.model is None and win.path_edit.text() == path
    shutil.copy(gui_file, path)  # the file is complete now
    with qtbot.waitSignal(win.engine.opened, timeout=10_000):
        win.reload()
    assert win.model is not None and win.model.path == path


# -- 9: CSV header states the x unit and reference -------------------------------------------

def test_export_header_has_x_reference(qtbot, win, types_file, tmp_path, monkeypatch):
    _open(qtbot, win, types_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(2))  # flag
    qtbot.wait(300)
    win.plot.set_view(1.0, 1.002)
    qtbot.wait(300)
    out = str(tmp_path / "x.csv")
    monkeypatch.setattr("tdmsviewer.mainwindow.QFileDialog.getSaveFileName", lambda *a, **k: (out, ""))
    with qtbot.waitSignal(win.engine.exportDone, timeout=5000):
        win.export_csv()
    head = next(csv.reader(open(out)))
    assert head[:2] == ["T/flag index", "T/flag x [s since 2026-07-28T10:05:36.000000Z]"]
    i = win.x_combo.findText("Sample index")
    win.x_combo.setCurrentIndex(i)
    win._x_source_changed(i)
    qtbot.wait(300)
    win.plot.set_view(1000, 1002)
    qtbot.wait(300)
    with qtbot.waitSignal(win.engine.exportDone, timeout=5000):
        win.export_csv()
    assert next(csv.reader(open(out)))[1] == "T/flag x [sample index]"


# -- 10: file drops on line edits open the file ----------------------------------------------

def test_drop_on_line_edits_opens_file(qtbot, win, gui_file):
    for le in (win.path_edit, win.tree_filter, win.prop_filter):
        assert not le.acceptDrops()
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(gui_file)])
    ev = QDropEvent(QPointF(10, 10), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    with qtbot.waitSignal(win.engine.opened, timeout=10_000):
        win.dropEvent(ev)
    assert win.model.path == gui_file


# -- 11: same channel names in different groups -----------------------------------------------

def test_legend_uses_group_for_same_names(qtbot, win, same_names_file):
    _open(qtbot, win, same_names_file)
    texts = [win.plot.legend.item(i).text() for i in range(win.plot.legend.count())]
    assert texts == ["00  Card1/U", "01  Card1/I", "02  Card2/U", "03  Card2/I"]
    _select(win, win.tree.topLevelItem(0).child(0))  # Card1 only: names are unique
    qtbot.wait(200)
    assert [win.plot.legend.item(i).text() for i in range(2)] == ["00  U", "01  I"]


# -- 12: graph keys only in the graph --------------------------------------------------------

def test_graph_keys_do_not_fire_in_combo_boxes(qtbot, win, gui_file):
    from pyqtgraph.Qt import QtTest

    QTest = QtTest.QTest

    _open(qtbot, win, gui_file)
    keys = {a.shortcut().toString() for a in win.plot.plot.actions()}
    assert {"C", "X", "Y", "Z", "P"} <= keys
    assert not any(a.shortcut().toString() == "C" for a in win.plot.actions())
    win.activateWindow()
    win.fmt_combo.setFocus()
    qtbot.wait(50)
    tool = win.plot.vb.tool
    QTest.keyClick(win.fmt_combo, Qt.Key_C)
    QTest.keyClick(win.fmt_combo, Qt.Key_X)
    qtbot.wait(50)
    assert not win.plot.cursors_on() and win.plot.vb.tool == tool


# -- 13: stale plot results, point markers ---------------------------------------------------

def test_results_of_older_selection_are_dropped(qtbot, win, gui_file):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(0))
    qtbot.wait(300)
    cid = win.current[0].id
    old = win._plot_seq
    win.set_selection([cid], keep_hidden=True)  # for example a range change
    assert win._min_plot_seq == old + 1
    x = np.linspace(1e6, 2e6, 5000)
    win._on_plot_ready(win.gen, old, {cid: (x, np.zeros_like(x), True)})
    xs, _ = win.plot._curves[cid].getData()
    assert xs is None or not np.any(xs >= 1e6)


def test_markers_from_total_points_not_from_visible_points(qtbot):
    p = PlotPanel()
    qtbot.addWidget(p)
    p.resize(1200, 600)
    p.show()
    p.set_channels([LegendEntry(0, 0, "a", theme.plot_color(0))])
    p.set_view(0.0, 10.0)
    x = np.linspace(100.0, 200.0, 5000)  # all outside the view
    p.set_data_many([(0, x, np.sin(x))])
    assert p._curves[0].opts["symbol"] is None
    x = np.linspace(0.0, 10.0, 20)  # few points: markers
    p.set_data_many([(0, x, np.sin(x))])
    assert p._curves[0].opts["symbol"] == "o"
    n = MAX_MARKER_PLOTS + 1  # many plots: no automatic markers
    p.set_channels([LegendEntry(i, i, f"c{i}", theme.plot_color(i)) for i in range(n)])
    p.set_data_many([(i, x, np.sin(x) + i) for i in range(n)])
    assert all(p._curves[i].opts["symbol"] is None for i in range(n))


# -- 14: curve pool and legend -----------------------------------------------------------

def test_curve_pool_and_icon_cache(qtbot):
    p = PlotPanel()
    qtbot.addWidget(p)
    entries = [LegendEntry(i, i, f"c{i}", theme.plot_color(i)) for i in range(50)]
    p.set_channels(entries)
    pool = [id(c) for c in p._pool]
    assert len(pool) == 50
    p.set_channels(entries[:3])
    assert [id(c) for c in p._pool] == pool and len(p.visible_cids()) == 3
    assert sum(c.isVisible() for c in p._pool) == 3
    p.set_channels(entries[::-1])
    assert [id(c) for c in p._pool] == pool and len(p.visible_cids()) == 50
    assert _line_icon(theme.plot_color(3)) is _line_icon(theme.plot_color(3))


# -- 15: own auto Y, one computation per data update -----------------------------------------

def test_auto_y_from_visible_data(qtbot, win, gui_file, monkeypatch):
    import pyqtgraph as pg

    calls = []
    orig = pg.ViewBox.childrenBounds
    monkeypatch.setattr(pg.ViewBox, "childrenBounds", lambda self, *a, **k: (calls.append(1), orig(self, *a, **k))[1])
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(0))  # sine with a 7.5 spike at 31.337 s
    qtbot.wait(400)
    y0, y1 = win.plot.vb.viewRange()[1]
    assert y1 > 7.5 and y0 < -1.0
    win.plot.set_view(10.0, 11.0)  # sin(t / 500) only: about 0.16 .. 0.26 here (t in ms)
    qtbot.wait(400)
    y0, y1 = win.plot.vb.viewRange()[1]
    lo, hi = np.sin(10_000 / 500.0), np.sin(11_000 / 500.0)
    lo, hi = min(lo, hi), max(lo, hi)
    assert y0 < lo and y1 > hi and (y1 - y0) < 2 * (hi - lo)
    assert not calls  # pyqtgraph's slow bounds of all curves are not used
    # A Y zoom switches auto Y off; X zoom then keeps Y.
    win.plot.set_y_view(-5.0, 5.0)
    assert not win.plot.auto_y_on()
    yr = win.plot.vb.viewRange()[1]
    win.plot.vb.setXRange(20.0, 21.0, padding=0)
    qtbot.wait(400)
    assert win.plot.vb.viewRange()[1] == pytest.approx(yr)


# -- 16: export never overwrites the TDMS file -----------------------------------------------

def test_export_refuses_tdms_paths(qtbot, win, gui_file, tmp_path, monkeypatch):
    _open(qtbot, win, gui_file)
    _select(win, win.tree.topLevelItem(0).child(0).child(1))
    qtbot.wait(300)
    before = open(gui_file, "rb").read()
    link = str(tmp_path / "link.csv")
    os.symlink(gui_file, link)
    warnings = []
    monkeypatch.setattr("tdmsviewer.mainwindow.QMessageBox.warning", lambda *a, **k: warnings.append(a[2]))
    requests = []
    monkeypatch.setattr(win.engine, "request_export", requests.append)
    for target in (gui_file, str(tmp_path / "copy.TDMS"), gui_file + "_index", link):
        monkeypatch.setattr("tdmsviewer.mainwindow.QFileDialog.getSaveFileName", lambda *a, t=target, **k: (t, ""))
        win.export_csv()
    assert len(warnings) == 4 and not requests
    assert open(gui_file, "rb").read() == before
    # No suffix: ".csv" is added (as QFileDialog.setDefaultSuffix).
    monkeypatch.setattr("tdmsviewer.mainwindow.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "plain"), ""))
    win.export_csv()
    assert requests and requests[0].path == str(tmp_path / "plain.csv")
