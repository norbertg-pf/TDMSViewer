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

import math
import os

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSplitter,
    QTableView, QTabWidget, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from . import __version__, theme
from .engine import DataEngine, ExportRequest, PlotItem, PlotRequest, StatsRequest, TableRequest, XRequest
from .plotpanel import LegendEntry, PlotPanel
from .tables import (
    PropertyFilter, PropertyModel, StatsModel, ValuesModel, ValuesView, copy_selection, set_clipboard,
)
from .tdmsfile import KIND_TIME, ChannelInfo, FileModel
from .xaxis import FMT_ABSOLUTE, FMT_NUMBER, FMT_RELATIVE, ArrayMap, LinearMap

ROLE_KIND = Qt.UserRole
ROLE_ID = Qt.UserRole + 1
SRC_WAVE, SRC_INDEX, SRC_CHAN = "wave", "index", "chan"
MAX_RECENT = 10


def _human_bytes(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return str(n)


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
        self._applied: dict[int, int] = {}
        self._table_seq = 0
        self._stats_seq = 0
        self._x_seq = 0
        self._export_seq = 0
        self._copy_job = None
        self._xy_key = None
        self._incomplete = False

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

        self.stats_model = StatsModel(self)
        self.stats_view = QTableView()
        self.stats_view.setModel(self.stats_model)
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
        self.tabs.addTab(self.values_view, "Values")
        self.tabs.addTab(stats_box, "Statistics")
        self.tabs.currentChanged.connect(lambda _: self._request_stats_soon())

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

    def open_file(self, path: str) -> None:
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
        self.gen = self.engine.open(path)
        self._set_status(f"Opening {os.path.basename(path)} ...")
        self.progress.setValue(0)
        self.progress.show()

    def reload(self) -> None:
        if self.model is not None:
            self.open_file(self.model.path)

    def close_file(self) -> None:
        self._clear_view()
        self.engine.close_file()
        self.gen = self.engine.generation
        self.path_edit.clear()
        self.setWindowTitle("TDMS Viewer")
        self._set_status("No file open.")

    def _clear_view(self) -> None:
        self._copy_job = None
        self.x_combo.clear()
        self.x_source = (SRC_WAVE, None)
        self.model = None
        self.current = []
        self.maps = {}
        self.ranges = {}
        self.x_array = None
        self._applied.clear()
        self.tree.clear()
        self.prop_model.set_properties({})
        self.plot.set_channels([])
        self.plot.clear_history()
        self.values_model.set_channels([], 0, None)
        self.stats_model.set_channels([])
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
        # NI default: the file is selected, so all channels are shown.
        root = self.tree.topLevelItem(0)
        if root is not None:
            self.tree.setCurrentItem(root)
            root.setSelected(True)

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
            self.set_selection([c.id for c in self.current])

    def set_selection(self, cids: list[int]) -> None:
        model = self.model
        if model is None:
            return
        self.current = [model.channels[i] for i in cids]
        self.ranges = {c.id: self._range(c) for c in self.current}
        self._build_maps()
        entries = []
        for n, c in enumerate(self.current):
            enabled = c.plottable and self.maps.get(c.id) is not None
            tip = f"{c.label}\n{c.length} samples, {c.kind}" + (f", unit {c.unit}" if c.unit else "")
            if not c.plottable:
                tip += "\nNot plottable (table only)"
            elif self.x_source[0] == SRC_WAVE and not c.wf_increment:
                tip += "\nNo wf_increment: drawn at 1 s per sample"
            elif self.maps.get(c.id) is None:
                tip += "\nLength differs from the X channel: not plotted"
            entries.append(LegendEntry(c.id, n, c.name, theme.plot_color(n), enabled, tip))
        self._applied.clear()
        self.plot.set_channels(entries)
        self._update_axis_labels()
        start = int(self.start_spin.value())
        count = None if self.all_check.isChecked() else int(self.samples_spin.value())
        self.values_model.set_channels(self.current, start, count)
        self.stats_model.set_channels(list(enumerate(self.current)))
        self.engine.set_priority([c.id for c in self.current])
        self._xy_key = None
        self.plot.clear_history()
        self.fit_view(push=False)
        if len(self.current) > 200:
            self._set_status(f"{len(self.current)} channels selected. Large selections redraw slower.")

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
        want = self.settings.value("x_channel_path", "")
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
            self.settings.setValue("x_channel_path", self.model.channels[cid].path if self.model else "")
            self._x_seq += 1
            self.x_source = (SRC_CHAN, cid)
            self.x_array = None
            self._set_status(f"Loading X values of {self.model.channels[cid].label} ...")
            self.engine.request_x(XRequest(self._x_seq, cid))
            if self.current:
                self.set_selection([c.id for c in self.current])
            return
        self.settings.setValue("x_channel_path", "")
        self.x_source = (src, None)
        self.x_array = None
        self._set_default_format()
        if self.current:
            self.set_selection([c.id for c in self.current])

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
            self.set_selection([c.id for c in self.current])

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
        units = {c.unit for c in self.current if c.plottable and self.maps.get(c.id) is not None}
        unit = units.pop() if len(units) == 1 else ""
        self.plot.set_y_label(f"Value [{unit}]" if unit else "")

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
                self._request_stats_soon()
                return
            self._xy_key = key
        self._plot_seq += 1
        self.engine.request_plot(PlotRequest(self._plot_seq, items, xa, xb, px))
        self._request_stats_soon()

    def _on_plot_ready(self, gen: int, seq: int, results: dict) -> None:
        if gen != self.gen:
            return
        incomplete = False
        for cid, res in results.items():
            if res is None:
                incomplete = True
                continue
            if seq < self._applied.get(cid, -1):
                continue
            x, y, complete = res
            self.plot.set_data(cid, x, y)
            self._applied[cid] = seq
            incomplete = incomplete or not complete
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
        job = self._copy_job
        if job is not None and seq == job[0]:
            self._copy_job = None
            self._finish_copy(job, blocks)
            return
        if seq == self._table_seq:
            self.values_model.put(blocks)

    def _copy_values(self) -> None:
        sel = self.values_view.selectionModel()
        if sel is None or not sel.selectedIndexes():
            return
        idx = sel.selectedIndexes()
        rows = [i.row() for i in idx]
        cols = sorted({i.column() for i in idx})
        r0, r1 = min(rows), max(rows) + 1
        if (r1 - r0) * len(cols) > 2_000_000:
            QMessageBox.information(self, "Copy", "The selection is too large to copy (max. 2 million cells). "
                                                  "Use File > Export visible range as CSV.")
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
        job = (None, r0, r1, cols, blocks)
        if missing:
            self._table_seq += 1
            self._copy_job = (self._table_seq, r0, r1, cols, blocks)
            set_clipboard("")  # never paste old data if the copy cannot finish
            self.engine.request_copy(TableRequest(self._table_seq, missing, start + r0, start + r1))
            self._set_status("Copying ...")
            return
        self._finish_copy(job, {})

    def _finish_copy(self, job, extra: dict) -> None:
        _, r0, r1, cols, blocks = job
        blocks = dict(blocks)
        blocks.update(extra)
        m = self.values_model
        from .formatting import format_value

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

        text = copy_selection(self.values_view, value_of)
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
                self.values_view.scroll_to_row(k - self.values_model.start)
            return

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
        first = not self.stats_model.values
        self.stats_model.put(values)
        if first:
            for col in range(3, self.stats_model.columnCount()):
                self.stats_view.resizeColumnToContents(col)

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
        xa, xb = self.plot.view_x_range()
        header = []
        for it in items:
            c = self.model.channels[it.cid]
            header += [f"{c.label} index", f"{c.label} x", f"{c.label}" + (f" [{c.unit}]" if c.unit else "")]
        self._export_seq += 1
        self.engine.request_export(ExportRequest(self._export_seq, path, items, xa, xb, header))
        self._set_status("Exporting ...")

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
            "  Z / X / Y / P: select tool.  C: cursors on/off\n\n"
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
        import PySide6
        import pyqtgraph

        QMessageBox.about(self, "About TDMS Viewer", (
            f"<b>TDMS Viewer {__version__}</b><br>"
            "Fast, read-only viewer for NI TDMS files.<br><br>"
            f"npTDMS {nptdms.__version__}, pyqtgraph {pyqtgraph.__version__}, "
            f"PySide6 {PySide6.__version__}, NumPy {np.__version__}<br>"
            "All components are free and open source."))
