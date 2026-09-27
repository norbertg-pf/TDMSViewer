"""Main window: NI TDMS File Viewer layout and the viewer logic.

Layout (as NI)
    Left:  file path + "...", "File contents" tree, property table,
           Start index / Samples / All.
    Right: waveform graph with palette and legend; below it the values
           table (and a statistics tab).

Selection
    File -> all channels, group -> its channels, channel -> one channel.
    Ctrl/Shift-click selects more items (union, in file order).
"""

from __future__ import annotations

import collections
import datetime as _dt
import json
import math
import os

import numpy as np
from pyqtgraph.Qt.QtCore import QSettings, Qt, QTimer
from pyqtgraph.Qt.QtGui import QAction, QKeySequence
from pyqtgraph.Qt.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSplitter,
    QTableView, QTabWidget, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from . import __version__, theme
from .engine import DataEngine, ExportRequest, PlotItem, PlotRequest, StatsRequest, TableRequest, XRequest
from .formatting import format_datetime64, format_float, format_value
from .plotpanel import LegendEntry, PlotPanel
from .tables import (
    COPY_MAX_CELLS, PropertyFilter, PropertyModel, StatsModel, ValuesModel, ValuesView, copy_selection,
    selection_ranges, set_clipboard,
)
from .tdmsfile import KIND_COMPLEX, KIND_TIME, ChannelInfo, FileModel
from .xaxis import FMT_ABSOLUTE, FMT_NUMBER, FMT_RELATIVE, ArrayMap, LinearMap, TimeRef

ROLE_KIND = Qt.UserRole
ROLE_ID = Qt.UserRole + 1
SRC_WAVE, SRC_INDEX, SRC_CHAN = "wave", "index", "chan"
MAX_RECENT = 10
MAX_X_FILES = 50  # files with a stored X channel
T0_SAMPLES = 1024  # the engine takes the first valid timestamp of these samples as zero


def _human_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return str(n)


def _first_time(a: np.ndarray):
    """First valid timestamp (the zero of a timestamp plot), as the engine takes it."""
    ok = a[~np.isnat(a)] if a.size else a
    return ok[0].astype("datetime64[ns]") if ok.size else np.datetime64(0, "ns")


def _short_local(t: np.datetime64) -> str:
    """Local time 'YYYY-MM-DD HH:MM:SS[.fraction] TZ' (fraction only if not zero)."""
    ns = int(np.datetime64(t, "ns").astype(np.int64))
    sec, rem = divmod(ns, 1_000_000_000)
    try:
        d = _dt.datetime.fromtimestamp(sec, _dt.timezone.utc).astimezone()
    except (OverflowError, OSError, ValueError):
        return format_datetime64(t)
    text = d.strftime("%Y-%m-%d %H:%M:%S")
    if rem:
        text += "." + f"{rem:09d}".rstrip("0")  # exact, no rounding
    return f"{text} {d.strftime('%Z')}".strip()


def _iso_utc(t_ref) -> str:
    """ISO 8601 UTC text of a TimeRef, for example 2026-07-28T10:05:36.000000Z."""
    t = TimeRef.of(t_ref)
    ns = round(t.frac * 1e9)
    sec = t.sec + ns // 1_000_000_000
    ns %= 1_000_000_000
    try:
        d = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc) + _dt.timedelta(seconds=sec)
    except OverflowError:
        return f"{float(t)!r} s Unix time"
    us, rem = divmod(ns, 1000)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + f".{us:06d}" + (f"{rem:03d}" if rem else "") + "Z"


class MainWindow(QMainWindow):
    """TDMS viewer main window."""

    def __init__(self, engine: DataEngine):
        super().__init__()
        self.engine = engine
        self.settings = QSettings("tdmsviewer", "tdmsviewer")
        self.model: FileModel | None = None
        self.gen = -1
        self.current: list[ChannelInfo] = []
        self.maps: dict[int, object] = {}
        self.ranges: dict[int, tuple[int, int]] = {}
        self.x_source = (SRC_WAVE, None)
        self.x_array: ArrayMap | None = None
        self.x_array_tref = None
        self._plot_seq = 0
        self._min_plot_seq = 0  # plot results of older requests belong to an old selection
        self._applied: dict[int, int] = {}
        self._table_seq = 0
        self._stats_seq = 0
        self._x_seq = 0
        self._export_seq = 0
        self._copy_job = None
        self._xy_key = None
        self._incomplete = False
        self._restore = None  # view state to restore after a reload (F5)
        self._t0: dict[int, np.datetime64] = {}  # zero of timestamp plots (first valid sample)
        self._t0_seq = 0  # table requests for timestamp zeros use negative seq numbers
        self._t0_pending = False
        self._stats_resize = False

        self.setWindowTitle("TDMS Viewer")
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_menus()
        self._connect_engine()
        self._restore_settings()

    # ======================================================================== UI

    def _build_ui(self) -> None:
        # -- left column --------------------------------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)

        path_row = QHBoxLayout()
        self.btn_browse = QToolButton()
        self.btn_browse.setText("...")
        self.btn_browse.setToolTip("Open a TDMS file (Ctrl+O)")
        self.btn_browse.clicked.connect(self.browse)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a TDMS file to inspect")
        self.path_edit.returnPressed.connect(lambda: self.open_file(self.path_edit.text().strip()))
        # A dropped file opens (MainWindow.dropEvent); it is not inserted as URL text.
        self.path_edit.setAcceptDrops(False)
        path_row.addWidget(self.btn_browse)
        path_row.addWidget(self.path_edit, 1)
        ll.addLayout(path_row)

        vsplit = QSplitter(Qt.Vertical)
        tree_box = QWidget()
        tl = QVBoxLayout(tree_box)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(2)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File contents"])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.setUniformRowHeights(True)
        self.tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self.tree.currentItemChanged.connect(self._tree_current_changed)
        self.tree_filter = QLineEdit()
        self.tree_filter.setPlaceholderText("Filter channels")
        self.tree_filter.setClearButtonEnabled(True)
        self.tree_filter.setAcceptDrops(False)
        self.tree_filter.textChanged.connect(self._filter_tree)
        tl.addWidget(self.tree, 1)
        tl.addWidget(self.tree_filter)
        vsplit.addWidget(tree_box)

        prop_box = QWidget()
        pl = QVBoxLayout(prop_box)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(2)
        self.prop_model = PropertyModel(self)
        self.prop_proxy = PropertyFilter(self)
        self.prop_proxy.setSourceModel(self.prop_model)
        self.prop_view = QTableView()
        self.prop_view.setModel(self.prop_proxy)
        self.prop_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.prop_view.verticalHeader().hide()
        self.prop_view.verticalHeader().setDefaultSectionSize(max(18, self.fontMetrics().height() + 4))
        self.prop_view.horizontalHeader().setStretchLastSection(True)
        self.prop_view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.prop_view.setColumnWidth(0, 160)
        self.prop_view.setWordWrap(False)
        self.prop_view.setAlternatingRowColors(True)
        copy_prop = QAction("Copy", self.prop_view)
        copy_prop.setShortcut(QKeySequence.Copy)
        copy_prop.setShortcutContext(Qt.WidgetShortcut)
        copy_prop.triggered.connect(self._copy_properties)
        self.prop_view.addAction(copy_prop)
        self.prop_filter = QLineEdit()
        self.prop_filter.setPlaceholderText("Filter properties")
        self.prop_filter.setClearButtonEnabled(True)
        self.prop_filter.setAcceptDrops(False)
        self.prop_filter.textChanged.connect(self.prop_proxy.setFilterFixedString)
        pl.addWidget(self.prop_view, 1)
        pl.addWidget(self.prop_filter)
        vsplit.addWidget(prop_box)
        vsplit.setSizes([420, 420])
        ll.addWidget(vsplit, 1)
        self.left_split = vsplit

        rng = QHBoxLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setDecimals(0)
        self.start_spin.setRange(0, 9e15)
        self.start_spin.setToolTip("Index of the first sample to show")
        self.samples_spin = QDoubleSpinBox()
        self.samples_spin.setDecimals(0)
        self.samples_spin.setRange(1, 9e15)
        self.samples_spin.setValue(100000)
        self.samples_spin.setToolTip("Number of samples to show")
        self.all_check = QCheckBox("All")
        self.all_check.setToolTip("Show all samples (the fast engine handles any file size)")
        self.all_check.setChecked(True)
        for w in (self.start_spin, self.samples_spin):
            w.setKeyboardTracking(False)
            w.valueChanged.connect(self._range_changed)
        self.all_check.toggled.connect(self._range_changed)
        for lbl, w in (("Start index", self.start_spin), ("Samples", self.samples_spin)):
            box = QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(QLabel(lbl))
            box.addWidget(w)
            rng.addLayout(box, 1)
        box = QVBoxLayout()
        box.setSpacing(0)
        box.addWidget(QLabel(""))
        box.addWidget(self.all_check)
        rng.addLayout(box)
        ll.addLayout(rng)

        # -- right column -------------------------------------------------------
        self.plot = PlotPanel()
        self.x_combo = QComboBox()
        self.x_combo.setToolTip("X axis source")
        self.x_combo.setMinimumContentsLength(18)
        self.x_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.x_combo.activated.connect(self._x_source_changed)
        self.fmt_combo = QComboBox()
        self.fmt_combo.setToolTip("X axis label format")
        for text, fmt in (("Number", FMT_NUMBER), ("Relative time", FMT_RELATIVE), ("Absolute time", FMT_ABSOLUTE)):
            self.fmt_combo.addItem(text, fmt)
        self.fmt_combo.activated.connect(self._x_format_changed)
        self.plot.top_bar.insertWidget(0, QLabel("X:"))
        self.plot.top_bar.insertWidget(1, self.x_combo)
        self.plot.top_bar.insertWidget(2, self.fmt_combo)
        self.plot.viewChanged.connect(self._view_changed)
        self.plot.fitRequested.connect(self.fit_view)
        self.plot.visibilityChanged.connect(self._visibility_changed)
        self.plot.cursorsChanged.connect(self._request_stats_soon)
        self.plot.cursorMoved.connect(self._cursor_to_table)
        self.plot.hoverChanged.connect(lambda t: self.hover_label.setText(t))

        self.values_model = ValuesModel(self)
        self.values_model.reader = self._try_read
        self.values_model.fetchRequested.connect(self._fetch_table)
        self.values_view = ValuesView()
        self.values_view.setModel(self.values_model)
        self.values_view.copyRequested.connect(self._copy_values)
        self.values_note = QLabel("")
        self.values_note.setWordWrap(True)
        self.values_note.setStyleSheet("QLabel { background: #fff4ce; padding: 2px 4px; }")
        self.values_note.hide()
        values_box = QWidget()
        vl = QVBoxLayout(values_box)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        vl.addWidget(self.values_note)
        vl.addWidget(self.values_view, 1)

        self.stats_model = StatsModel(self)
        self.stats_view = QTableView()
        self.stats_view.setModel(self.stats_model)
        self.stats_view.setTextElideMode(Qt.ElideLeft)  # a cut number keeps its exponent: "…708e-05"
        self.stats_view.setWordWrap(False)
        self.stats_view.verticalHeader().hide()
        self.stats_view.verticalHeader().setDefaultSectionSize(max(18, self.fontMetrics().height() + 4))
        self.stats_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.stats_view.horizontalHeader().setStretchLastSection(True)
        self.stats_view.setColumnWidth(0, 36)
        self.stats_view.setColumnWidth(1, 220)
        self.stats_view.setColumnWidth(2, 50)
        self.stats_view.setColumnWidth(4, 50)
        self.stats_view.setAlternatingRowColors(True)
        self.stats_label = QLabel("")
        self.stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        copy_stats = QPushButton("Copy")
        copy_stats.setToolTip("Copy the statistics as tab separated text (full precision)")
        copy_stats.clicked.connect(lambda: set_clipboard(self.stats_model.as_text()))
        stats_box = QWidget()
        sl = QVBoxLayout(stats_box)
        sl.setContentsMargins(0, 2, 0, 0)
        top = QHBoxLayout()
        top.addWidget(self.stats_label, 1)
        top.addWidget(copy_stats)
        sl.addLayout(top)
        sl.addWidget(self.stats_view, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(values_box, "Values")
        self.tabs.addTab(stats_box, "Statistics")
        self.tabs.currentChanged.connect(self._tab_changed)

        rsplit = QSplitter(Qt.Vertical)
        rsplit.addWidget(self.plot)
        rsplit.addWidget(self.tabs)
        rsplit.setSizes([560, 440])
        self.right_split = rsplit

        main = QSplitter(Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(rsplit)
        main.setStretchFactor(0, 0)
        main.setStretchFactor(1, 1)
        main.setSizes([330, 1270])
        self.main_split = main
        self.setCentralWidget(main)

        # -- status bar ------------------------------------------------------------
        sb = self.statusBar()
        self.status_label = QLabel("Open a TDMS file: Ctrl+O, drag and drop, or the \"...\" button.")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.warn_btn = QToolButton()
        self.warn_btn.setText("⚠ 0")
        self.warn_btn.setToolTip("File warnings")
        self.warn_btn.setAutoRaise(True)
        self.warn_btn.clicked.connect(self._show_warnings)
        self.warn_btn.hide()
        self.hover_label = QLabel("")
        sb.addWidget(self.status_label, 1)
        sb.addPermanentWidget(self.warn_btn)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.hover_label)
        self.warnings: list[str] = []

        self._stats_timer = QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.setInterval(120)
        self._stats_timer.timeout.connect(self._request_stats)
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(150)
        self._update_timer.timeout.connect(self._data_arrived)
        self.resize(1600, 1000)

    def _build_menus(self) -> None:
        m = self.menuBar().addMenu("&File")
        self.act_open = m.addAction("&Open...", self.browse, QKeySequence.Open)
        self.recent_menu = m.addMenu("Open &Recent")
        self.act_reload = m.addAction("&Reload", self.reload, QKeySequence("F5"))
        m.addSeparator()
        self.act_export = m.addAction("&Export visible range as CSV...", self.export_csv, QKeySequence("Ctrl+E"))
        m.addSeparator()
        m.addAction("&Close file", self.close_file, QKeySequence.Close)
        m.addAction("&Quit", self.close, QKeySequence.Quit)
        v = self.menuBar().addMenu("&View")
        v.addAction("Zoom to &fit", self.fit_view, QKeySequence("Ctrl+0"))
        v.addAction("&Previous view", self.plot.back, QKeySequence("Alt+Left"))
        act_c = QAction("&Cursors", self, checkable=True)
        act_c.toggled.connect(self.plot.set_cursors_visible)
        self.plot.btn_cursors.toggled.connect(act_c.setChecked)
        v.addAction(act_c)
        v.addSeparator()
        v.addAction("&Show all plots", lambda: self.plot.set_all_visible(True))
        v.addAction("&Hide all plots", lambda: self.plot.set_all_visible(False))
        v.addSeparator()
        v.addAction("Reset &layout", self._reset_layout)
        h = self.menuBar().addMenu("&Help")
        h.addAction("&Mouse and keys", self._show_help, QKeySequence.HelpContents)
        h.addAction("&About", self._show_about)
        self._update_recent_menu()

    def _connect_engine(self) -> None:
        e = self.engine
        e.opened.connect(self._on_opened)
        e.openFailed.connect(self._on_open_failed)
        e.progress.connect(self._on_progress)
        e.channelsUpdated.connect(self._on_channels_updated)
        e.plotReady.connect(self._on_plot_ready)
        e.tableReady.connect(self._on_table_ready)
        e.statsReady.connect(self._on_stats_ready)
        e.xReady.connect(self._on_x_ready)
        e.exportDone.connect(self._on_export_done)
        e.message.connect(self._on_message)

    # ================================================================ settings

    def _restore_settings(self) -> None:
        s = self.settings
        geo = s.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        for key, sp in (("split/main", self.main_split), ("split/left", self.left_split),
                        ("split/right", self.right_split), ("split/plot", self.plot.split)):
            st = s.value(key)
            if st is not None:
                sp.restoreState(st)
        self.all_check.blockSignals(True)
        self.samples_spin.blockSignals(True)
        self.all_check.setChecked(s.value("range/all", True, type=bool))
        self.samples_spin.setValue(float(s.value("range/samples", 100000)))
        self.all_check.blockSignals(False)
        self.samples_spin.blockSignals(False)
        self.samples_spin.setEnabled(not self.all_check.isChecked())

    def closeEvent(self, ev) -> None:
        s = self.settings
        s.setValue("geometry", self.saveGeometry())
        s.setValue("split/main", self.main_split.saveState())
        s.setValue("split/left", self.left_split.saveState())
        s.setValue("split/right", self.right_split.saveState())
        s.setValue("split/plot", self.plot.split.saveState())
        s.setValue("range/all", self.all_check.isChecked())
        s.setValue("range/samples", int(self.samples_spin.value()))
        super().closeEvent(ev)

    def _reset_layout(self) -> None:
        self.main_split.setSizes([330, max(400, self.width() - 330)])
        self.left_split.setSizes([420, 420])
        self.right_split.setSizes([560, 440])
        self.plot.split.setSizes([max(300, self.plot.width() - 180), 180])

    def _x_choices(self) -> list[list[str]]:
        """Stored X channels: [[file path, channel path], ...], newest last."""
        try:
            v = json.loads(self.settings.value("x_channel_by_file", "[]") or "[]")
        except (TypeError, ValueError):
            return []
        return [p for p in v if isinstance(p, list) and len(p) == 2 and all(isinstance(s, str) for s in p)]

    def _stored_x_channel(self, file_path: str) -> str:
        """X channel path chosen last time for this file ("" = Waveform time)."""
        return next((c for f, c in reversed(self._x_choices()) if f == file_path), "")

    def _store_x_channel(self, file_path: str, chan_path: str) -> None:
        """Store the X channel of one file only (another file starts with Waveform time, as NI)."""
        v = [p for p in self._x_choices() if p[0] != file_path]
        if chan_path:
            v.append([file_path, chan_path])
        self.settings.setValue("x_channel_by_file", json.dumps(v[-MAX_X_FILES:]))

    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if isinstance(p, str)]

    def _add_recent(self, path: str) -> None:
        r = [p for p in self._recent() if p != path]
        r.insert(0, path)
        self.settings.setValue("recent", r[:MAX_RECENT])
        self._update_recent_menu()

    def _update_recent_menu(self) -> None:
        self.recent_menu.clear()
        rec = self._recent()
        for p in rec:
            self.recent_menu.addAction(p, lambda p=p: self.open_file(p))
        self.recent_menu.setEnabled(bool(rec))

    # ================================================================ open / close

    def browse(self) -> None:
        start = self.settings.value("last_dir", os.path.expanduser("~"))
        path, _ = QFileDialog.getOpenFileName(self, "Open TDMS file", start, "TDMS files (*.tdms);;All files (*)")
        if path:
            self.open_file(path)

    def open_file(self, path: str, restore: dict | None = None) -> None:
        """Open a file. restore: view state of reload() (selection, hidden plots, zoom, cursors)."""
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Open file", f"File not found:\n{path}")
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)
        self._clear_view()
        self._restore = restore
        self.gen = self.engine.open(path)
        self._set_status(f"Opening {os.path.basename(path)} ...")
        self.progress.setValue(0)
        self.progress.show()

    def reload(self) -> None:
        """Open the file again (F5). Keeps selection, hidden plots, zoom and cursors if still valid.

        After a failed open there is no model: use the path in the path box.
        """
        if self.model is not None:
            path, restore = self.model.path, self._view_state()
        else:
            path, restore = self.path_edit.text().strip(), self._restore
        self.open_file(path, restore)

    def close_file(self) -> None:
        self._clear_view()
        self.engine.close_file()
        self.gen = self.engine.generation
        self.path_edit.clear()
        self.setWindowTitle("TDMS Viewer")
        self._set_status("No file open.")

    def _clear_view(self) -> None:
        self._copy_job = None
        self._restore = None
        self.x_combo.clear()
        self.x_source = (SRC_WAVE, None)
        self.model = None
        self.current = []
        self.maps = {}
        self.ranges = {}
        self.x_array = None
        self._xy_key = None
        self._applied.clear()
        self._t0.clear()
        self._t0_pending = False
        self.tree.clear()
        self.prop_model.set_properties({})
        self.plot.set_channels([])
        self.plot.clear_history()
        self.values_model.set_channels([], 0, None)
        self._update_values_note()
        self.stats_model.set_channels([])
        self.stats_label.setText("")
        self.plot.refresh_readout()  # no plots: empty readout
        self.hover_label.setText("")
        self.warnings = []
        self.warn_btn.hide()
        self.progress.hide()

    def _on_opened(self, gen: int, model: FileModel) -> None:
        if gen != self.gen:
            return
        self.model = model
        self._add_recent(model.path)
        self.setWindowTitle(f"{model.name} - TDMS Viewer")
        self._populate_tree(model)
        self._populate_x_combo(model)
        n_fast = sum(c.fast for c in model.channels)
        n_ch = len(model.channels)
        info = (f"{model.name}: {len(model.groups)} groups, {n_ch} channels, {_human_bytes(model.size)}, "
                f"{model.n_segments} segments, fast read {n_fast}/{n_ch}, opened in {model.open_ms:.0f} ms")
        if model.index_used:
            info += ", index file used"
        self._set_status(info)
        self.file_info = info
        for w in model.warnings:
            self._add_warning(w)
        if self._restore is not None and self._restore_selection(self._restore):
            return
        # NI default: the file is selected, so all channels are shown.
        root = self.tree.topLevelItem(0)
        if root is not None:
            self.tree.setCurrentItem(root)
            root.setSelected(True)

    # ================================================================ reload state

    def _item_key(self, it):
        """Tree item as a key that stays valid in a reloaded file."""
        if it is None or self.model is None:
            return None
        kind = it.data(0, ROLE_KIND)
        if kind == "group":
            return ("group", self.model.groups[it.data(0, ROLE_ID)].name)
        if kind == "chan":
            return ("chan", self.model.channels[it.data(0, ROLE_ID)].path)
        return ("file", "")

    def _item_of_key(self, key):
        root = self.tree.topLevelItem(0)
        if root is None or key is None:
            return None
        kind, name = key
        if kind == "file":
            return root
        for gi in range(root.childCount()):
            g = root.child(gi)
            if kind == "group" and self.model.groups[g.data(0, ROLE_ID)].name == name:
                return g
            if kind == "chan":
                for ci in range(g.childCount()):
                    c = g.child(ci)
                    if self.model.channels[c.data(0, ROLE_ID)].path == name:
                        return c
        return None

    def _view_state(self) -> dict:
        """What reload() keeps: tree selection, X source, hidden plots, zoom, cursors."""
        (x0, x1), (y0, y1) = self.plot.vb.viewRange()
        src, cid = self.x_source
        return {
            "selected": [self._item_key(it) for it in self.tree.selectedItems()],
            "current": self._item_key(self.tree.currentItem()),
            "x": (src, self.model.channels[cid].path if src == SRC_CHAN else ""),
            "format": self.fmt_combo.currentData(),
            "hidden": [self.model.channels[cid].path for cid in self.plot.hidden_cids()],
            "view": (float(x0), float(x1), float(y0), float(y1), self.plot.auto_y_on()),
            "cursors": self.plot.cursor_positions(),
        }

    def _restore_selection(self, st: dict) -> bool:
        """Select the stored tree items and X source again. False if none of the items exists."""
        src, xpath = st.get("x", (SRC_WAVE, ""))
        want = next((i for i in range(self.x_combo.count())
                     if (self.x_combo.itemData(i) or (None,))[0] == src
                     and (src != SRC_CHAN or self.model.channels[self.x_combo.itemData(i)[1]].path == xpath)), 0)
        if want != self.x_combo.currentIndex():
            self.x_combo.setCurrentIndex(want)
            self._x_source_changed(want)
        items = [it for it in (self._item_of_key(k) for k in st.get("selected", [])) if it is not None]
        if not items:
            self._restore = None
            return False
        cur = self._item_of_key(st.get("current")) or items[0]
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(cur)
        self.tree.clearSelection()
        for it in items:
            it.setSelected(True)
        self.tree.blockSignals(False)
        self._tree_current_changed(cur, None)
        self._tree_selection_changed()
        return True

    def _apply_restore(self) -> None:
        """Hidden plots, zoom and cursors of the reload state (when the X values are ready)."""
        st = self._restore
        if st is None or self.model is None or (self.x_source[0] == SRC_CHAN and self.x_array is None):
            return
        self._restore = None
        fmt = st.get("format")
        i = self.fmt_combo.findData(fmt)
        item = self.fmt_combo.model().item(i) if i >= 0 else None
        if item is not None and item.isEnabled() and i != self.fmt_combo.currentIndex():
            self.fmt_combo.setCurrentIndex(i)
            self._x_format_changed(i)
        hidden = set(st.get("hidden", []))
        plotted = self.plot.plotted_cids()
        if hidden and plotted:
            show = {cid for cid in plotted if self.model.channels[cid].path not in hidden}
            self.plot.set_all_visible(True, only=show)
        x0, x1, y0, y1, auto_y = st.get("view", (0.0, 0.0, 0.0, 0.0, True))
        ext = self._x_extent()
        if ext is not None and all(map(math.isfinite, (x0, x1, y0, y1))) and x1 > x0 and x1 >= ext[0] and x0 <= ext[1]:
            self.plot.vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
            if auto_y:
                self.plot.vb.enableAutoRange(y=True)
        cur = st.get("cursors") or []
        if len(cur) == 2 and all(map(math.isfinite, cur)):
            self.plot.set_cursors_visible(True)
            self.plot.set_cursor_positions(*cur)

    def _on_open_failed(self, gen: int, msg: str) -> None:
        if gen != self.gen:
            return
        self.progress.hide()
        self._set_status("Open failed.")
        QMessageBox.critical(self, "Open file", f"The file cannot be read as TDMS.\n\n{msg}")

    def _on_progress(self, gen: int, frac: float, text: str) -> None:
        if gen != self.gen:
            return
        if frac >= 1.0:
            self.progress.hide()
            base = getattr(self, "file_info", "")
            self._set_status(f"{base}. {text}." if base else text)
        else:
            self.progress.show()
            self.progress.setValue(int(frac * 1000))
            self.progress.setToolTip(text)

    def _on_message(self, gen: int, text: str) -> None:
        if gen == self.gen:
            self._add_warning(text)

    def _add_warning(self, text: str) -> None:
        if text in self.warnings:
            return
        self.warnings.append(text)
        self.warn_btn.setText(f"⚠ {len(self.warnings)}")
        self.warn_btn.show()

    def _show_warnings(self) -> None:
        QMessageBox.warning(self, "File warnings", "\n\n".join(self.warnings) or "No warnings.")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # ================================================================ tree

    def _populate_tree(self, model: FileModel) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        root = QTreeWidgetItem([model.name])
        root.setData(0, ROLE_KIND, "file")
        root.setToolTip(0, model.path)
        for gi, g in enumerate(model.groups):
            gitem = QTreeWidgetItem([g.name])
            gitem.setData(0, ROLE_KIND, "group")
            gitem.setData(0, ROLE_ID, gi)
            gitem.setToolTip(0, f"{g.name}\n{len(g.channels)} channels")
            for c in g.channels:
                citem = QTreeWidgetItem([c.name])
                citem.setData(0, ROLE_KIND, "chan")
                citem.setData(0, ROLE_ID, c.id)
                unit = f"\nunit: {c.unit}" if c.unit else ""
                citem.setToolTip(0, f"{c.label}\n{c.length} samples, {c.kind} ({c.dtype}){unit}"
                                    f"\nread: {'fast path' if c.fast else 'npTDMS'}")
                if not c.plottable:
                    citem.setForeground(0, Qt.gray)
                gitem.addChild(citem)
            root.addChild(gitem)
        self.tree.addTopLevelItem(root)
        root.setExpanded(True)
        for i in range(root.childCount()):
            root.child(i).setExpanded(True)
        self.tree.blockSignals(False)
        self._filter_tree(self.tree_filter.text())

    def _filter_tree(self, text: str) -> None:
        root = self.tree.topLevelItem(0)
        if root is None:
            return
        t = text.strip().lower()
        for gi in range(root.childCount()):
            g = root.child(gi)
            gmatch = not t or t in g.text(0).lower()
            anyc = False
            for ci in range(g.childCount()):
                c = g.child(ci)
                show = gmatch or t in c.text(0).lower()
                c.setHidden(not show)
                anyc = anyc or show
            g.setHidden(not (gmatch or anyc))

    def _tree_current_changed(self, cur, prev) -> None:
        if cur is None or self.model is None:
            self.prop_model.set_properties({})
            return
        kind = cur.data(0, ROLE_KIND)
        if kind == "file":
            self.prop_model.set_properties(self.model.properties)
        elif kind == "group":
            self.prop_model.set_properties(self.model.groups[cur.data(0, ROLE_ID)].properties)
        elif kind == "chan":
            self.prop_model.set_properties(self.model.channels[cur.data(0, ROLE_ID)].display_properties())

    def _tree_selection_changed(self) -> None:
        if self.model is None:
            return
        ids: set[int] = set()
        for it in self.tree.selectedItems():
            kind = it.data(0, ROLE_KIND)
            if kind == "file":
                ids.update(c.id for c in self.model.channels)
            elif kind == "group":
                ids.update(c.id for c in self.model.groups[it.data(0, ROLE_ID)].channels)
            elif kind == "chan":
                ids.add(it.data(0, ROLE_ID))
        self.set_selection(sorted(ids))

    def _copy_properties(self) -> None:
        rows = sorted({i.row() for i in self.prop_view.selectionModel().selectedIndexes()})
        lines = []
        for r in rows:
            src = self.prop_proxy.mapToSource(self.prop_proxy.index(r, 0)).row()
            name, val, _ = self.prop_model.items[src]
            lines.append(f"{name}\t{val}")
        set_clipboard("\n".join(lines))

    # ================================================================ selection and range

    def _range(self, c: ChannelInfo) -> tuple[int, int]:
        start = int(self.start_spin.value())
        s = min(start, c.length)
        if self.all_check.isChecked():
            return s, c.length
        return s, min(c.length, start + int(self.samples_spin.value()))

    def _range_changed(self, *args) -> None:
        self.samples_spin.setEnabled(not self.all_check.isChecked())
        if self.model is not None:
            self.set_selection([c.id for c in self.current], keep_hidden=True)

    def set_selection(self, cids: list[int], keep_hidden: bool = False) -> None:
        """Show these channels. keep_hidden: hidden plots stay hidden (range or X change)."""
        model = self.model
        if model is None:
            return
        self.current = [model.channels[i] for i in cids]
        self.ranges = {c.id: self._range(c) for c in self.current}
        self._build_maps()
        # Results of older requests have the old maps and ranges: never draw them.
        self._min_plot_seq = self._plot_seq + 1
        self._applied.clear()
        self._read_time_zeros()
        self.plot.set_channels(self._legend_entries(), keep_hidden=keep_hidden)
        self._update_axis_labels()
        start = int(self.start_spin.value())
        count = None if self.all_check.isChecked() else int(self.samples_spin.value())
        self.values_model.set_channels(self.current, start, count)
        self._update_values_note()
        self.stats_model.set_channels(list(enumerate(self.current)))
        self.stats_model.set_units(self._unit_texts())
        self.engine.set_priority([c.id for c in self.current])
        self._xy_key = None
        self.plot.clear_history()
        self.fit_view(push=False)
        if self.plot.cursors_on():
            self.plot.cursors_to_view()  # after a file, X or selection change they can be off-screen
        if len(self.current) > 200:
            self._set_status(f"{len(self.current)} channels selected. Large selections redraw slower.")
        self._apply_restore()

    # -- legend texts: same names, complex and timestamp channels --------------------

    def _legend_entries(self) -> list[LegendEntry]:
        names = collections.Counter(c.name for c in self.current)
        entries = []
        for n, c in enumerate(self.current):
            enabled = c.plottable and self.maps.get(c.id) is not None
            # Same name in two groups: show group/name, or the plots look identical.
            label = c.label if names[c.name] > 1 else c.name
            tip = f"{c.label}\n{c.length} samples, {c.kind}" + (f", unit {c.unit}" if c.unit else "")
            if c.kind == KIND_COMPLEX:
                label += " |z|"
                tip += "\nComplex values: plotted and summarized as the magnitude |z|"
            elif c.kind == KIND_TIME:
                t0 = self._t0.get(c.id)
                label += f" (s since {_short_local(t0)})" if t0 is not None else " (s since first sample)"
                tip += ("\nTimestamps: plotted and summarized as seconds since the first valid sample"
                        + (f"\n{format_datetime64(t0)}" if t0 is not None else ""))
            if not c.plottable:
                tip += "\nNot plottable (table only)"
            elif self.x_source[0] == SRC_WAVE and not c.wf_increment:
                tip += "\nNo wf_increment: drawn at 1 s per sample"
            elif self.maps.get(c.id) is None:
                tip += "\nLength differs from the X channel: not plotted"
            entries.append(LegendEntry(c.id, n, label, theme.plot_color(n), enabled, tip))
        return entries

    def _unit_text(self, c: ChannelInfo) -> str:
        """Unit of the plotted and summarized values of a channel."""
        if c.kind == KIND_COMPLEX:
            return f"|z| [{c.unit}]" if c.unit else "|z|"
        if c.kind == KIND_TIME:
            t0 = self._t0.get(c.id)
            return f"s since {_short_local(t0)}" if t0 is not None else "s since first sample"
        return c.unit

    def _unit_texts(self) -> dict[int, str]:
        return {c.id: self._unit_text(c) for c in self.current if c.kind in (KIND_COMPLEX, KIND_TIME)}

    def _read_time_zeros(self) -> bool:
        """Zero (first valid sample) of the timestamp channels of the selection.

        Sync read if the channel is in RAM, else one table request (copy slot:
        table scrolling cannot drop it). The labels are updated when it arrives.
        Returns True if a new zero is known now.
        """
        missing, found = [], False
        for c in self.current:
            if c.kind != KIND_TIME or c.id in self._t0 or c.length == 0:
                continue
            a = self._try_read(c.id, 0, min(c.length, T0_SAMPLES))
            if a is None:
                missing.append(c.id)
            else:
                self._t0[c.id] = _first_time(a)
                found = True
        if missing and not self._t0_pending and self._copy_job is None:
            self._t0_pending = True
            self._t0_seq -= 1
            self.engine.request_copy(TableRequest(self._t0_seq, missing, 0, T0_SAMPLES))
        return found

    def _refresh_labels(self) -> None:
        """Legend, unit and axis texts again (for example: a timestamp zero is known now)."""
        self.plot.update_entries(self._legend_entries())
        self.stats_model.set_units(self._unit_texts())
        self._update_axis_labels()

    def _build_maps(self) -> None:
        model = self.model
        self.maps = {}
        src, cid = self.x_source
        for c in self.current:
            if not c.plottable:
                continue
            if src == SRC_WAVE:
                self.maps[c.id] = LinearMap(model.start_seconds(c), c.wf_increment or 1.0)
            elif src == SRC_INDEX:
                self.maps[c.id] = LinearMap(0.0, 1.0)
            elif self.x_array is not None and c.length == self.x_array.x.size:
                self.maps[c.id] = self.x_array

    def _items(self, visible_only: bool) -> list[PlotItem]:
        vis = set(self.plot.visible_cids()) if visible_only else None
        out = []
        for c in self.current:
            m = self.maps.get(c.id)
            if m is None or (vis is not None and c.id not in vis):
                continue
            s, e = self.ranges[c.id]
            out.append(PlotItem(c.id, m, s, e))
        return out

    def _x_extent(self) -> tuple[float, float] | None:
        lo, hi = math.inf, -math.inf
        for it in self._items(visible_only=False):
            if it.e <= it.s:
                continue
            m = it.xmap
            if isinstance(m, ArrayMap) and not m.monotonic:
                seg = m.x[it.s:it.e]
                fin = seg[np.isfinite(seg)]
                if fin.size:
                    lo, hi = min(lo, float(fin.min())), max(hi, float(fin.max()))
                continue
            a, b = m.x_of(it.s), m.x_of(it.e - 1)
            if math.isfinite(a) and math.isfinite(b):
                lo, hi = min(lo, a, b), max(hi, a, b)
        return (lo, hi) if lo <= hi else None

    def fit_view(self, push: bool = True) -> None:
        if push:
            self.plot.push_history()
        ext = self._x_extent()
        if ext is None:
            self.plot.clear_data()
            self._request_stats_soon()
            return
        self.plot.set_view(ext[0], ext[1], auto_y=True)
        self._view_changed()

    # ================================================================ x axis

    def _populate_x_combo(self, model: FileModel) -> None:
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        has_dt = any(c.wf_increment for c in model.channels)
        self.x_combo.addItem("Waveform time" if has_dt else "Waveform time (dt = 1 s)",
                             (SRC_WAVE, None))
        self.x_combo.addItem("Sample index", (SRC_INDEX, None))
        self.x_combo.insertSeparator(2)
        want = self._stored_x_channel(model.path)  # same file only
        pick = 0
        for c in model.channels:
            if c.plottable and c.length > 1:
                self.x_combo.addItem(f"Channel: {c.label}", (SRC_CHAN, c.id))
                if c.path == want:
                    pick = self.x_combo.count() - 1
        self.x_combo.blockSignals(False)
        self.x_array = None
        if pick:
            self.x_combo.setCurrentIndex(pick)
            self._x_source_changed(pick)
        else:
            self.x_combo.setCurrentIndex(0)
            self.x_source = (SRC_WAVE, None)
            self._set_default_format()

    def _x_source_changed(self, index: int) -> None:
        data = self.x_combo.itemData(index)
        if not data or self.model is None:
            return
        src, cid = data
        if src == SRC_CHAN:
            self._store_x_channel(self.model.path, self.model.channels[cid].path)
            self._x_seq += 1
            self.x_source = (SRC_CHAN, cid)
            self.x_array = None
            self._set_status(f"Loading X values of {self.model.channels[cid].label} ...")
            self.engine.request_x(XRequest(self._x_seq, cid))
            if self.current:
                self.set_selection([c.id for c in self.current], keep_hidden=True)
            return
        self._store_x_channel(self.model.path, "")
        self.x_source = (src, None)
        self.x_array = None
        self._set_default_format()
        if self.current:
            self.set_selection([c.id for c in self.current], keep_hidden=True)

    def _on_x_ready(self, gen: int, seq: int, res) -> None:
        if gen != self.gen or seq != self._x_seq:
            return
        if isinstance(res, str):
            QMessageBox.warning(self, "X axis", res)
            self.x_combo.setCurrentIndex(0)
            self._x_source_changed(0)
            return
        cid, amap, t_ref = res
        self.x_array = amap
        self.x_array_tref = t_ref
        self._set_default_format()
        note = "" if amap.monotonic else " (not monotonic: X-Y plot of the full range)"
        self._set_status(f"X axis: {self.model.channels[cid].label}{note}")
        if self.current:
            self.set_selection([c.id for c in self.current], keep_hidden=True)

    def _t_ref(self):
        src, _ = self.x_source
        if src == SRC_WAVE and self.model is not None:
            return self.model.t_ref_unix
        if src == SRC_CHAN:
            return self.x_array_tref
        return None

    def _set_default_format(self) -> None:
        src, cid = self.x_source
        if src == SRC_WAVE:
            fmt = FMT_ABSOLUTE if self._t_ref() is not None else FMT_RELATIVE
        elif src == SRC_CHAN and self.model is not None and self.model.channels[cid].kind == KIND_TIME:
            fmt = FMT_ABSOLUTE
        else:
            fmt = FMT_NUMBER
        t_ref = self._t_ref()
        item = self.fmt_combo.model().item(2)
        if item is not None:
            item.setEnabled(t_ref is not None)
        if fmt == FMT_ABSOLUTE and t_ref is None:
            fmt = FMT_RELATIVE
        self.fmt_combo.setCurrentIndex(self.fmt_combo.findData(fmt))
        self._update_axis_labels()

    def _x_format_changed(self, index: int) -> None:
        self._update_axis_labels()
        self.plot.refresh_readout()

    def _update_axis_labels(self) -> None:
        fmt = self.fmt_combo.currentData() or FMT_NUMBER
        src, cid = self.x_source
        t_ref = self._t_ref()
        if src == SRC_WAVE:
            label = "Time"
            plotted = [c for c in self.current if c.plottable]
            missing = [c for c in plotted if not c.wf_increment]
            if missing and len(missing) == len(plotted):
                label += " (no wf_increment: 1 sample = 1 s)"
            elif missing:
                label += f" ({len(missing)} channels have no wf_increment: 1 sample = 1 s for them)"
            elif fmt == FMT_NUMBER:
                label += " [s]"
        elif src == SRC_INDEX:
            label = "Sample index"
        else:
            c = self.model.channels[cid] if self.model else None
            label = c.label + (f" [{c.unit}]" if c and c.unit else "") if c else ""
        if fmt == FMT_ABSOLUTE and t_ref is not None:
            import datetime as _dt

            try:
                d = _dt.datetime.fromtimestamp(float(t_ref)).astimezone()
                label += f"  (local time, start {d.strftime('%Y-%m-%d %H:%M:%S %Z')})"
            except (OverflowError, OSError, ValueError):
                pass
        self.plot.set_x_axis(fmt, t_ref, label)
        self.plot.set_y_label(self._y_label())

    def _y_label(self) -> str:
        """Y axis label; says how complex and timestamp values are drawn."""
        plotted = [c for c in self.current if c.plottable and self.maps.get(c.id) is not None]
        labels = set()
        for c in plotted:
            if c.kind in (KIND_COMPLEX, KIND_TIME):
                labels.add(self._unit_text(c))
            else:
                labels.add(f"Value [{c.unit}]" if c.unit else "")
        if len(labels) == 1:
            return labels.pop()
        notes = []
        if any(c.kind == KIND_COMPLEX for c in plotted):
            notes.append("complex as |z|")
        if any(c.kind == KIND_TIME for c in plotted):
            notes.append("timestamps as s since first sample")
        return f"Value ({', '.join(notes)})" if notes else ""

    # ================================================================ plot data

    def _view_changed(self) -> None:
        if self.model is None:
            return
        items = self._items(visible_only=True)
        if not items:
            return
        xa, xb, px = self.plot.fetch_window()
        xy = any(isinstance(i.xmap, ArrayMap) and not i.xmap.monotonic for i in items)
        if xy:
            key = (tuple((i.cid, i.s, i.e) for i in items), id(self.x_array))
            if key == self._xy_key:
                self.plot.update_auto_y()  # same data, new view
                self._request_stats_soon()
                return
            self._xy_key = key
        self._plot_seq += 1
        self.engine.request_plot(PlotRequest(self._plot_seq, items, xa, xb, px))
        self._request_stats_soon()

    def _on_plot_ready(self, gen: int, seq: int, results: dict) -> None:
        if gen != self.gen or seq < self._min_plot_seq:
            return  # other file, or a request of an older selection / X source / range
        incomplete = False
        batch = []
        for cid, res in results.items():
            if res is None:
                incomplete = True
                continue
            if seq < self._applied.get(cid, -1):
                continue
            x, y, complete = res
            batch.append((cid, x, y))
            self._applied[cid] = seq
            incomplete = incomplete or not complete
        self.plot.set_data_many(batch)  # one repaint and one auto Y for all plots
        self._incomplete = incomplete

    def _on_channels_updated(self, gen: int, cids) -> None:
        if gen != self.gen or not self.current:
            return
        cur = {c.id for c in self.current}
        if cur.intersection(cids):
            self._update_timer.start()

    def _data_arrived(self) -> None:
        self._xy_key = None
        self._view_changed()
        self.values_model.invalidate()
        self.values_view._schedule()
        if self._read_time_zeros():
            self._refresh_labels()

    def _visibility_changed(self) -> None:
        self._xy_key = None
        self._view_changed()

    # ================================================================ table

    def _try_read(self, cid: int, i0: int, i1: int):
        return self.engine.try_read(self.gen, cid, i0, i1)

    def _fetch_table(self, cids: list, i0: int, i1: int) -> None:
        self._table_seq += 1
        self.engine.request_table(TableRequest(self._table_seq, list(cids), i0, i1))

    def _on_table_ready(self, gen: int, seq: int, blocks: dict) -> None:
        if gen != self.gen:
            return
        if seq < 0:  # zero of timestamp plots (_read_time_zeros)
            if seq == self._t0_seq:
                self._t0_pending = False
                for cid, (_i0, arr) in blocks.items():
                    self._t0[cid] = _first_time(arr)
                if blocks:
                    self._refresh_labels()
            return
        job = self._copy_job
        if job is not None and seq == job[0]:
            self._copy_job = None
            self._finish_copy(job, blocks)
            if self._read_time_zeros():
                self._refresh_labels()
            return
        if seq == self._table_seq:
            self.values_model.put(blocks)

    def _copy_values(self) -> None:
        # Size the selection from its ranges: a Ctrl+A of millions of rows is one range.
        ranges = selection_ranges(self.values_view)
        if not ranges:
            return
        r0, r1 = min(r[0] for r in ranges), max(r[1] for r in ranges) + 1
        cols = sorted({c for r in ranges for c in range(r[2], r[3] + 1)})
        if (r1 - r0) * len(cols) > COPY_MAX_CELLS:  # rows r0..r1 are read: limit the box, not the cells
            QMessageBox.information(self, "Copy", f"The selection is too large to copy (max. {COPY_MAX_CELLS:,} "
                                                  "cells). Use File > Export visible range as CSV.")
            return
        m = self.values_model
        start = m.start
        blocks = {}
        missing = []
        for col in cols:
            c = m.channels[col]
            a = self._try_read(c.id, start + r0, start + r1)
            if a is None:
                missing.append(c.id)
            else:
                blocks[c.id] = (start + r0, a)
        job = (None, r0, r1, cols, blocks, ranges)
        if missing:
            self._table_seq += 1
            self._copy_job = (self._table_seq, r0, r1, cols, blocks, ranges)
            set_clipboard("")  # never paste old data if the copy cannot finish
            self._t0_pending = False  # the copy request replaces a pending timestamp-zero request
            self.engine.request_copy(TableRequest(self._table_seq, missing, start + r0, start + r1))
            self._set_status("Copying ...")
            return
        self._finish_copy(job, {})

    def _finish_copy(self, job, extra: dict) -> None:
        _, r0, r1, cols, blocks, ranges = job
        blocks = dict(blocks)
        blocks.update(extra)
        m = self.values_model

        def value_of(r, col):
            c = m.channels[col]
            hit = blocks.get(c.id)
            if hit is None:
                return ""
            i0, arr = hit
            k = m.start + r - i0
            if m.start + r >= m.stops[col] or not (0 <= k < arr.size):
                return ""
            return format_value(arr[k])

        def column_texts(col, a, b):
            """Texts of rows [a, b) of one column (one list comprehension, not one call per cell)."""
            c = m.channels[col]
            hit = blocks.get(c.id)
            if hit is None:
                return [""] * (b - a)
            i0, arr = hit
            k0 = max(0, m.start + a - i0)
            k1 = max(k0, min(arr.size, m.stops[col] - i0, m.start + b - i0))
            seg = arr[k0:k1]
            # Same texts as format_value; tolist() makes no numpy scalar per value.
            if seg.dtype == np.float64:
                texts = [repr(v) if math.isfinite(v) else format_float(v) for v in seg.tolist()]
            elif seg.dtype.kind in "iu":
                texts = [str(v) for v in seg.tolist()]
            else:
                texts = [format_value(v) for v in seg]
            lead = max(0, i0 + k0 - (m.start + a))
            return [""] * lead + texts + [""] * (b - a - lead - len(texts))

        text = copy_selection(self.values_view, value_of, ranges=ranges, column_texts=column_texts)
        set_clipboard(text)
        self._set_status(f"Copied {text.count(chr(10)) + 1 if text else 0} rows.")

    def _cursor_to_table(self, x: float) -> None:
        for c in self.current:
            m = self.maps.get(c.id)
            if m is None:
                continue
            s, e = self.ranges[c.id]
            k = m.nearest(x, s, e)
            if k >= 0:
                row = k - self.values_model.start
                if row >= self.values_model.rowCount() and self.values_model.clamped:
                    self._set_status(f"Sample {k} is after the last table row. "
                                     "Set Start index to see it in the table.")
                self.values_view.scroll_to_row(row)
            return

    def _update_values_note(self) -> None:
        """Note above the table when the Qt row limit cuts the range."""
        m = self.values_model
        if not m.clamped:
            self.values_note.hide()
            return
        nxt = m.start + m.rows
        text = (f"The table shows the first {m.rows:,} of {m.total_rows:,} samples from Start index {m.start:,} "
                f"(Qt row limit). Use Start index to page: set it to {nxt:,} for the next samples. "
                "The graph, statistics and export use all samples.")
        self.values_note.setText(text)
        self.values_note.show()
        self._set_status(f"Table shows the first {m.rows:,} rows from Start index; use Start index to page.")

    # ================================================================ statistics

    def _request_stats_soon(self) -> None:
        self.plot.refresh_readout()
        if self.tabs.currentIndex() == 1 or self.plot.cursors_on():
            self._stats_timer.start()

    def _request_stats(self) -> None:
        if self.model is None or not self.current:
            return
        items = self._items(visible_only=False)
        cur = self.plot.cursor_positions()
        if len(cur) == 2:
            xa, xb = min(cur), max(cur)
            what = "between the cursors"
        else:
            xa, xb = self.plot.view_x_range()
            what = "in the visible X range"
        self.stats_label.setText(f"Statistics {what}: {self.plot.format_x(xa)}  …  {self.plot.format_x(xb)}")
        self._stats_seq += 1
        self.engine.request_stats(StatsRequest(self._stats_seq, items, xa, xb, cur))

    def _on_stats_ready(self, gen: int, seq: int, values: dict) -> None:
        if gen != self.gen or seq != self._stats_seq:
            return
        self.stats_model.put(values)
        self._stats_resize = True
        if self.tabs.currentIndex() == 1:
            self._fit_stats_columns()

    def _fit_stats_columns(self) -> None:
        """Unit and number columns as wide as their texts, on every result: a cut number
        could be read as another value. Only while the tab is shown (few rows: cheap)."""
        self._stats_resize = False
        for col in range(2, self.stats_model.columnCount()):
            self.stats_view.resizeColumnToContents(col)  # last column: stretches only into free space

    def _tab_changed(self, index: int) -> None:
        if index == 1 and self._stats_resize:
            self._fit_stats_columns()
        self._request_stats_soon()

    # ================================================================ export

    def export_csv(self) -> None:
        if self.model is None:
            return
        items = self._items(visible_only=True)
        if not items:
            QMessageBox.information(self, "Export", "No visible plots to export.")
            return
        base = os.path.splitext(self.model.path)[0] + "_export.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export visible range as CSV", base, "CSV files (*.csv)")
        if not path:
            return
        if not os.path.splitext(os.path.basename(path))[1]:
            # Default suffix (as QFileDialog.setDefaultSuffix("csv")); ask before replacing a file.
            path += ".csv"
            if os.path.exists(path) and QMessageBox.question(
                    self, "Export", f"{path} exists. Replace it?") != QMessageBox.Yes:
                return
        refuse = self._export_refused(path)
        if refuse:
            QMessageBox.warning(self, "Export", f"{refuse}\n\n{path}\n\nNo file was written.")
            return
        xa, xb = self.plot.view_x_range()
        t_ref, timed = self._export_time(items)
        header = []
        for it in items:
            c = self.model.channels[it.cid]
            header += [f"{c.label} index", self._x_header(c)]
            if t_ref is not None:
                header.append(f"{c.label} time [UTC]")
            header.append(f"{c.label}" + (f" [{c.unit}]" if c.unit else ""))
        self._export_seq += 1
        self.engine.request_export(ExportRequest(self._export_seq, path, items, xa, xb, header,
                                                 t_ref, frozenset(timed)))
        self._set_status("Exporting ...")

    def _export_refused(self, path: str) -> str:
        """Reason not to write the CSV to this path ("" = OK). The export never replaces TDMS data."""
        if path.lower().endswith((".tdms", ".tdms_index")):
            return "The export cannot write a .tdms or .tdms_index file. Use a .csv file name."
        if self.model is not None and os.path.exists(path):
            for p in (self.model.path, self.model.path + "_index"):
                try:
                    if os.path.exists(p) and os.path.samefile(path, p):
                        return "This is the open TDMS file (or its index). Use another file name."
                except OSError:
                    pass
        return ""

    def _export_time(self, items) -> tuple:
        """(TimeRef of x == 0, channel ids with an absolute time) for the CSV time column.

        Waveform time: channels with wf_start_time and wf_increment. Time
        channel as X: all channels. Otherwise no time column.
        """
        src, xcid = self.x_source
        if src == SRC_WAVE and self.model.t_ref_unix is not None:
            timed = {it.cid for it in items
                     if self.model.channels[it.cid].wf_start_time is not None
                     and self.model.channels[it.cid].wf_increment}
            return (self.model.t_ref_unix, timed) if timed else (None, set())
        if src == SRC_CHAN and self.model.channels[xcid].kind == KIND_TIME and self.x_array_tref is not None:
            return TimeRef.of(self.x_array_tref), {it.cid for it in items}
        return None, set()

    def _x_header(self, c: ChannelInfo) -> str:
        """CSV header of the x column of a channel: unit and time reference of x."""
        src, xcid = self.x_source
        if src == SRC_INDEX:
            return f"{c.label} x [sample index]"
        if src == SRC_WAVE:
            t_ref = self.model.t_ref_unix
            unit = f"s since {_iso_utc(t_ref)}" if t_ref is not None else "s"
            if not c.wf_increment:
                unit += ", no wf_increment: 1 s per sample"
            return f"{c.label} x [{unit}]"
        xc = self.model.channels[xcid]
        if xc.kind == KIND_TIME and self.x_array_tref is not None:
            return f"{c.label} x = {xc.label} [s since {_iso_utc(self.x_array_tref)}]"
        return f"{c.label} x = {xc.label}" + (f" [{xc.unit}]" if xc.unit else "")

    def _on_export_done(self, gen: int, seq: int, msg: str) -> None:
        if gen == self.gen:
            self._set_status(msg)

    # ================================================================ drag and drop, help

    def dragEnterEvent(self, ev) -> None:
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev) -> None:
        for url in ev.mimeData().urls():
            if url.isLocalFile():
                self.open_file(url.toLocalFile())
                break

    def _show_help(self) -> None:
        QMessageBox.information(self, "Mouse and keys", (
            "Graph\n"
            "  Left drag: tool of the palette (zoom box, zoom X, zoom Y, pan)\n"
            "  Zoom-about-point tool: click zooms in 2x, Shift+click zooms out\n"
            "  Middle drag: pan.  Right drag: zoom about the start point\n"
            "  Wheel: zoom.  Shift+wheel: X only.  Ctrl+wheel: Y only\n"
            "  Wheel or drag on an axis: that axis only\n"
            "  Double-click or Home: zoom to fit.  Backspace: previous view\n"
            "  Z / X / Y / P: select tool.  C: cursors on/off (click the graph first)\n\n"
            "Legend\n"
            "  Checkbox: show/hide.  Right-click: color, line width, line/points\n\n"
            "Tree\n"
            "  File: all channels.  Group: its channels.  Ctrl/Shift-click: more items\n\n"
            "Table\n"
            "  Ctrl+C: copy the selected cells (full precision)\n\n"
            "File\n"
            "  Ctrl+O open, F5 reload, Ctrl+E export CSV, drag and drop a file"))

    def _show_about(self) -> None:
        import nptdms
        import pyqtgraph
        from pyqtgraph.Qt import VERSION_INFO

        QMessageBox.about(self, "About TDMS Viewer", (
            f"<b>TDMS Viewer {__version__}</b><br>"
            "Fast, read-only viewer for NI TDMS files.<br><br>"
            f"npTDMS {nptdms.__version__}, pyqtgraph {pyqtgraph.__version__}, "
            f"{VERSION_INFO}, NumPy {np.__version__}<br>"
            "All components are free and open source."))
