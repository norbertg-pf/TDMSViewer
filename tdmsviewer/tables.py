"""Values table, statistics table and property table.

Summary
    ValuesModel is lazy. It keeps only a window of rows around the
    visible area. RAM and fast-path channels are read at once; other
    channels are read by the engine and filled in when ready. Qt handles
    up to 2**31 - 1 rows with this model at ~10 MB memory.
"""

from __future__ import annotations


import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView

from . import theme
from .formatting import format_si, format_value
from .pyramid import Stats
from .tdmsfile import ChannelInfo

MAX_ROWS = 2**31 - 1
WINDOW_PAD = 256  # extra rows cached above and below the visible rows
COPY_MAX_CELLS = 2_000_000


class ValuesModel(QAbstractTableModel):
    """Channel values in columns, sample index in rows (lazy)."""

    fetchRequested = Signal(list, int, int)  # channel ids, i0, i1 (async read needed)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels: list[ChannelInfo] = []
        self.start = 0
        self.stops: list[int] = []
        self.rows = 0
        self.reader = None  # callable(cid, i0, i1) -> array or None
        self._cache: dict[int, tuple[int, np.ndarray]] = {}
        self._align = Qt.AlignRight | Qt.AlignVCenter
        self._font = theme.mono_font()
        self._missing = QColor("#f4f4f4")

    def set_channels(self, channels: list[ChannelInfo], start: int, count: int | None) -> None:
        self.beginResetModel()
        self.channels = list(channels)
        self.start = max(0, int(start))
        self.stops = []
        rows = 0
        for c in self.channels:
            stop = c.length if count is None else min(c.length, self.start + count)
            stop = max(self.start, stop)
            self.stops.append(stop)
            rows = max(rows, stop - self.start)
        self.rows = min(MAX_ROWS, rows)
        self._cache.clear()
        self.endResetModel()

    def invalidate(self) -> None:
        self._cache.clear()
        if self.rows and self.channels:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rows - 1, len(self.channels) - 1))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.rows

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.channels)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if 0 <= section < len(self.channels):
                    c = self.channels[section]
                    return f"{c.group}\n{c.name} [{section:02d}]"
                return None
            return str(self.start + section)
        if role == Qt.ToolTipRole and orientation == Qt.Horizontal and 0 <= section < len(self.channels):
            c = self.channels[section]
            unit = f", unit {c.unit}" if c.unit else ""
            return f"{c.label}\n{c.length} samples, {c.dtype}{unit}"
        if role == Qt.TextAlignmentRole and orientation == Qt.Vertical:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def value(self, row: int, col: int):
        """Native value at a cell, or None if not loaded / outside the channel."""
        c = self.channels[col]
        idx = self.start + row
        if idx >= self.stops[col]:
            return None
        hit = self._cache.get(c.id)
        if hit is not None:
            i0, arr = hit
            k = idx - i0
            if 0 <= k < arr.size:
                return arr[k]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            v = self.value(index.row(), index.column())
            return None if v is None else format_value(v)
        if role == Qt.TextAlignmentRole:
            return int(self._align)
        if role == Qt.FontRole:
            return self._font
        if role == Qt.BackgroundRole:
            col = index.column()
            if self.start + index.row() >= self.stops[col]:
                return self._missing
        return None

    # -- window management -----------------------------------------------------------

    def ensure_rows(self, top: int, bottom: int) -> None:
        """Make rows [top, bottom] available (sync read or async request)."""
        if not self.channels or self.rows == 0:
            return
        top = max(0, top)
        bottom = min(self.rows - 1, bottom)
        want0 = self.start + max(0, top - WINDOW_PAD)
        want1 = self.start + min(self.rows, bottom + 1 + WINDOW_PAD)
        need0 = self.start + top
        need1 = self.start + bottom + 1
        missing = []
        changed = False
        for col, c in enumerate(self.channels):
            s1 = self.stops[col]
            a, b = need0, min(need1, s1)
            if b <= a:
                continue
            hit = self._cache.get(c.id)
            if hit is not None and hit[0] <= a and hit[0] + hit[1].size >= b:
                continue
            w0, w1 = want0, min(want1, s1)
            arr = self.reader(c.id, w0, w1) if self.reader else None
            if arr is None:
                missing.append(c.id)
            else:
                self._cache[c.id] = (w0, arr)
                changed = True
        if changed:
            self.dataChanged.emit(self.index(top, 0), self.index(bottom, len(self.channels) - 1))
        if missing:
            self.fetchRequested.emit(missing, want0, want1)

    def put(self, blocks: dict) -> None:
        """Store async results {cid: (i0, values)}."""
        if not blocks:
            return
        for cid, (i0, arr) in blocks.items():
            self._cache[cid] = (i0, arr)
        if self.rows and self.channels:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rows - 1, len(self.channels) - 1))


class ValuesView(QTableView):
    """Table view that loads rows while scrolling."""

    copyRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)
        vh = self.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(max(18, self.fontMetrics().height() + 4))
        vh.setMinimumWidth(56)
        hh = self.horizontalHeader()
        hh.setDefaultSectionSize(190)
        hh.setMinimumSectionSize(60)
        hh.setDefaultAlignment(Qt.AlignCenter)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._fill)
        self.verticalScrollBar().valueChanged.connect(self._schedule)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def setModel(self, model):
        super().setModel(model)
        model.modelReset.connect(self._schedule)

    def _schedule(self, *a):
        self._timer.start()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._schedule()

    def showEvent(self, ev):
        super().showEvent(ev)
        self._schedule()

    def _fill(self) -> None:
        m = self.model()
        if m is None or m.rowCount() == 0:
            return
        top = self.rowAt(0)
        bottom = self.rowAt(self.viewport().height() - 1)
        if top < 0:
            top = 0
        if bottom < 0:
            bottom = min(m.rowCount() - 1, top + self.viewport().height() // max(1, self.verticalHeader().defaultSectionSize()) + 1)
        m.ensure_rows(top, bottom)

    def keyPressEvent(self, ev):
        if ev.matches(QKeySequence.Copy):
            self.copyRequested.emit()
            return
        super().keyPressEvent(ev)

    def _menu(self, pos):
        m = QMenu(self)
        m.addAction("Copy (tab separated)", self.copyRequested.emit)
        m.exec(self.viewport().mapToGlobal(pos))

    def scroll_to_row(self, row: int) -> None:
        m = self.model()
        if m is None or not (0 <= row < m.rowCount()):
            return
        self.scrollTo(m.index(row, 0), QAbstractItemView.PositionAtCenter)
        self.selectRow(row)


# -- statistics --------------------------------------------------------------------

STATS_COLUMNS = ["#", "Channel", "Unit", "N", "Min", "Max", "Peak-peak", "Mean", "Std dev", "RMS",
                 "C1", "C2", "C2 − C1"]


class StatsModel(QAbstractTableModel):
    """Per-channel statistics of the visible range or of the cursor range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[tuple[int, ChannelInfo]] = []
        self.values: dict[int, dict] = {}
        self._font = theme.mono_font()

    def set_channels(self, rows: list[tuple[int, ChannelInfo]]) -> None:
        self.beginResetModel()
        self.rows = [(n, c) for n, c in rows if c.plottable]
        self.values = {}
        self.endResetModel()

    def put(self, values: dict) -> None:
        self.values = values
        if self.rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, len(STATS_COLUMNS) - 1))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(STATS_COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return STATS_COLUMNS[section]
        return None

    def _cell(self, row: int, col: int, exact: bool = False):
        num, c = self.rows[row]
        if col == 0:
            return f"{num:02d}"
        if col == 1:
            return c.label
        if col == 2:
            return c.unit
        res = self.values.get(c.id)
        if res is None:
            return "" if col < 10 else ""
        s: Stats | None = res.get("stats")
        fmt = (lambda v: format_value(float(v))) if exact else (lambda v: format_si(v, 7))
        if 3 <= col <= 9:
            if s is None:
                return "…"
            if col == 3:
                return str(s.n)
            if s.n == 0:
                return ""
            v = (s.min, s.max, s.p2p, s.mean, s.std, s.rms)[col - 4]
            return fmt(v)
        cur = res.get("cursors") or []
        vals = []
        for k in range(2):
            item = cur[k] if k < len(cur) else None
            vals.append(None if item is None else item[2])
        if col in (10, 11):
            v = vals[col - 10]
            if v is None:
                return ""
            return format_value(v) if exact or not isinstance(v, (float, np.floating)) else format_si(float(v), 7)
        if col == 12:
            a, b = vals
            try:
                d = float(b) - float(a)
            except (TypeError, ValueError):
                return ""
            return fmt(d)
        return ""

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._cell(index.row(), index.column())
        if role == Qt.ToolTipRole and index.column() >= 3:
            return self._cell(index.row(), index.column(), exact=True)
        if role == Qt.TextAlignmentRole:
            return int((Qt.AlignLeft if index.column() in (1, 2) else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.FontRole and index.column() >= 3:
            return self._font
        if role == Qt.ForegroundRole and index.column() == 0:
            return theme.plot_color(self.rows[index.row()][0])
        return None

    def as_text(self) -> str:
        lines = ["\t".join(STATS_COLUMNS)]
        for r in range(len(self.rows)):
            lines.append("\t".join(self._cell(r, c, exact=True) for c in range(len(STATS_COLUMNS))))
        return "\n".join(lines)


# -- properties --------------------------------------------------------------------

class PropertyModel(QAbstractTableModel):
    """Two columns: property name, property value (NI order: ASCII sort)."""

    HEAD = ("Property name", "Property value")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[tuple[str, str, str]] = []

    def set_properties(self, props: dict) -> None:
        self.beginResetModel()
        self.items = [(str(k), format_value(v), type(v).__name__) for k, v in sorted(props.items(), key=lambda kv: str(kv[0]))]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.items)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEAD[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if role in (Qt.DisplayRole, Qt.EditRole):
            return self.items[index.row()][index.column()]
        if role == Qt.ToolTipRole:
            name, val, typ = self.items[index.row()]
            return f"{name} ({typ})\n{val}"
        return None


class PropertyFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterKeyColumn(-1)


def copy_selection(view: QTableView, value_of) -> str:
    """TSV text of the selected cells. value_of(row, col) -> text."""
    sel = view.selectionModel()
    if sel is None:
        return ""
    idx = sel.selectedIndexes()
    if not idx:
        return ""
    rows = sorted({i.row() for i in idx})
    cols = sorted({i.column() for i in idx})
    if len(rows) * len(cols) > COPY_MAX_CELLS:
        return ""
    chosen = {(i.row(), i.column()) for i in idx}
    lines = []
    for r in rows:
        lines.append("\t".join(value_of(r, c) if (r, c) in chosen else "" for c in cols))
    return "\n".join(lines)


def set_clipboard(text: str) -> None:
    QGuiApplication.clipboard().setText(text)
