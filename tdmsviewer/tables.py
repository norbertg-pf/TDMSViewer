"""Values table, statistics table and property table.

Summary
    ValuesModel is lazy. It keeps only a window of rows around the
    visible area. RAM and fast-path channels are read at once; other
    channels are read by the engine and filled in when ready.

Row limit
    Qt keeps the length of the vertical header in pixels as a 32-bit
    int. Above (2**31 - 1) / row height rows (about 119M rows at 18 px)
    the GUI thread hangs inside QHeaderView. ValuesView gives the model
    this limit; the model shows only the first rows from Start index.

Numbers
    Cells elide on the left ("…708e-05"), never on the right: a cut
    number must not look like another value ("-2…" for -2.1e-05).
"""

from __future__ import annotations


import math

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel, QItemSelection, QItemSelectionModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFontMetrics, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView

from . import theme
from .formatting import format_si, format_value
from .pyramid import Stats
from .tdmsfile import KIND_COMPLEX, KIND_TIME, ChannelInfo

MAX_ROWS = 2**31 - 1
WINDOW_PAD = 256  # extra rows cached above and below the visible rows
COPY_MAX_CELLS = 2_000_000
# Widest texts of a full-precision value, per kind (for the column width).
WIDEST_FLOAT = "-2.2250738585072014e-308"
WIDEST_TEXT = {
    KIND_TIME: "2026-07-28 10:05:36.000000123+14:00",
    KIND_COMPLEX: "-2.2250738585072014e-308-2.2250738585072014e-308j",
}
CELL_PAD = 16  # cell margins, grid line and a small reserve (pixels)


def max_table_rows(row_height: int) -> int:
    """Largest row count whose header length in pixels fits a 32-bit int."""
    return (2**31 - 1) // max(1, int(row_height)) - 1


def text_width(text: str, font=None) -> int:
    """Column width (pixels) that shows `text` without eliding (default: value font)."""
    return QFontMetrics(font if font is not None else theme.mono_font()).horizontalAdvance(text) + CELL_PAD


class ValuesModel(QAbstractTableModel):
    """Channel values in columns, sample index in rows (lazy)."""

    fetchRequested = Signal(list, int, int)  # channel ids, i0, i1 (async read needed)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.channels: list[ChannelInfo] = []
        self.start = 0
        self.stops: list[int] = []
        self.rows = 0
        self.total_rows = 0  # rows before the row limit
        self.row_limit = MAX_ROWS  # set by ValuesView from its row height
        self.reader = None  # callable(cid, i0, i1) -> array or None
        self._cache: dict[int, tuple[int, np.ndarray]] = {}
        self._align = Qt.AlignRight | Qt.AlignVCenter
        self._font = theme.mono_font()
        self._missing = QColor("#f4f4f4")

    @property
    def clamped(self) -> bool:
        """True if the table shows fewer rows than the range has."""
        return self.total_rows > self.rows

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
        self.total_rows = rows
        self.rows = min(MAX_ROWS, self.row_limit, rows)
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
        if role in (Qt.DisplayRole, Qt.ToolTipRole):  # tooltip: full text of a narrow cell
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
        self.setTextElideMode(Qt.ElideLeft)
        vh = self.verticalHeader()
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(max(18, self.fontMetrics().height() + 4))
        vh.setMinimumWidth(56)
        hh = self.horizontalHeader()
        hh.setDefaultSectionSize(text_width(WIDEST_FLOAT))
        hh.setMinimumSectionSize(60)
        hh.setDefaultAlignment(Qt.AlignCenter)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._fill)
        self.verticalScrollBar().valueChanged.connect(self._schedule)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def max_rows(self) -> int:
        """Row limit of this view (header length in pixels must fit an int32)."""
        return max_table_rows(self.verticalHeader().defaultSectionSize())

    def setModel(self, model):
        super().setModel(model)
        model.row_limit = self.max_rows()
        model.modelReset.connect(self._schedule)
        model.modelReset.connect(self._set_widths)
        self._detach_headers()

    def setSelectionModel(self, sel):
        super().setSelectionModel(sel)
        self._detach_headers()

    def _detach_headers(self) -> None:
        """Give the headers their own empty selection model.

        A header asks isColumnSelected() for every painted section. Qt
        answers it with one flags() call per selected row, so a selection
        of millions of rows (Ctrl+A) blocks each repaint for many seconds.
        Without the table selection the headers do not highlight
        selected sections; selecting by header click still works.
        """
        m, sel = self.model(), self.selectionModel()
        if m is None:
            return
        for h in (self.horizontalHeader(), self.verticalHeader()):
            hs = h.selectionModel()
            if hs is None or hs is sel or hs.model() is not m:
                h.setSelectionModel(QItemSelectionModel(m, h))

    def _set_widths(self) -> None:
        """Column widths that show a full-precision value of the column kind."""
        m = self.model()
        for col, c in enumerate(getattr(m, "channels", [])):
            txt = WIDEST_TEXT.get(c.kind)
            if txt is not None:
                self.setColumnWidth(col, text_width(txt))

    def selectAll(self):
        """Select all cells as one range (cheap for any row count)."""
        m, sel = self.model(), self.selectionModel()
        if m is None or sel is None or m.rowCount() == 0 or m.columnCount() == 0:
            return
        sel.select(QItemSelection(m.index(0, 0), m.index(m.rowCount() - 1, m.columnCount() - 1)),
                   QItemSelectionModel.ClearAndSelect)

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

STATS_COLUMNS = ["#", "Channel", "Unit", "N", "NaN", "Min", "Max", "Peak-peak", "Mean", "Std dev", "RMS",
                 "C1", "C2", "C2 − C1"]
_C_N, _C_NAN, _C_FIRST_VAL, _C_LAST_VAL, _C_C1, _C_C2, _C_DIFF = 3, 4, 5, 10, 11, 12, 13


class StatsModel(QAbstractTableModel):
    """Per-channel statistics of the visible range or of the cursor range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[tuple[int, ChannelInfo]] = []
        self.values: dict[int, dict] = {}
        self.units: dict[int, str] = {}  # unit text per channel id (default: channel unit)
        self._font = theme.mono_font()

    def set_channels(self, rows: list[tuple[int, ChannelInfo]]) -> None:
        self.beginResetModel()
        self.rows = [(n, c) for n, c in rows if c.plottable]
        self.values = {}
        self.endResetModel()

    def set_units(self, units: dict[int, str]) -> None:
        """Unit column text, for example "|z|" for complex values."""
        self.units = dict(units)
        if self.rows:
            self.dataChanged.emit(self.index(0, 2), self.index(len(self.rows) - 1, 2))

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
            return self.units.get(c.id, c.unit)
        res = self.values.get(c.id)
        if res is None:
            return ""
        s: Stats | None = res.get("stats")
        fmt = (lambda v: format_value(float(v))) if exact else (lambda v: format_si(v, 7))
        if _C_N <= col <= _C_LAST_VAL:
            if s is None:
                return "\u2026"
            if col == _C_N:
                return str(s.n)
            if col == _C_NAN:
                total = res.get("total")
                return "" if total is None else str(max(0, total - s.n))
            if s.n == 0:
                return ""
            k = col - _C_FIRST_VAL
            off = res.get("offset")
            if off is not None and k in (0, 1):  # int64/uint64 min/max: exact integers
                v = off + int(round((s.min, s.max)[k]))
                return str(v) if exact else format_si(float(v), 7)
            if off is not None and k in (3, 5):  # mean, RMS of absolute values
                mean = float(off) + s.mean
                if k == 3:
                    return fmt(mean)
                return fmt(math.sqrt(mean * mean + s.m2 / s.n))
            v = (s.min, s.max, s.p2p, s.mean, s.std, s.rms)[k]
            return fmt(v)
        cur = res.get("cursors") or []
        vals = [None if k >= len(cur) or cur[k] is None else cur[k][2] for k in range(2)]
        if col in (_C_C1, _C_C2):
            v = vals[col - _C_C1]
            if v is None:
                return ""
            return format_value(v) if exact or not isinstance(v, (float, np.floating)) else format_si(float(v), 7)
        if col == _C_DIFF:
            return _difference(vals[0], vals[1], exact, fmt)
        return ""

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return self._cell(index.row(), index.column())
        if role == Qt.ToolTipRole and index.column() >= _C_N:
            return self._cell(index.row(), index.column(), exact=True)
        if role == Qt.ToolTipRole and index.column() in (1, 2):
            return self._cell(index.row(), index.column())
        if role == Qt.TextAlignmentRole:
            return int((Qt.AlignLeft if index.column() in (1, 2) else Qt.AlignRight) | Qt.AlignVCenter)
        if role == Qt.FontRole and index.column() >= _C_N:
            return self._font
        if role == Qt.ForegroundRole and index.column() == 0:
            return theme.plot_color(self.rows[index.row()][0])
        return None

    def as_text(self) -> str:
        lines = ["\t".join(STATS_COLUMNS)]
        for r in range(len(self.rows)):
            lines.append("\t".join(self._cell(r, c, exact=True) for c in range(len(STATS_COLUMNS))))
        return "\n".join(lines)


def _difference(a, b, exact: bool, fmt) -> str:
    """C2 - C1 without loss: integers as integers, timestamps in seconds."""
    if a is None or b is None:
        return ""
    if isinstance(a, np.datetime64) and isinstance(b, np.datetime64):
        ns = int((b.astype("datetime64[ns]") - a.astype("datetime64[ns]")) / np.timedelta64(1, "ns"))
        return f"{ns / 1e9!r} s" if exact else f"{format_si(ns / 1e9, 9)} s"
    ints = (int, np.integer, bool, np.bool_)
    if isinstance(a, ints) and isinstance(b, ints):
        return str(int(b) - int(a))  # exact for int64/uint64
    try:
        return fmt(float(b) - float(a))
    except (TypeError, ValueError):
        return ""


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


def selection_ranges(view: QTableView) -> list[tuple[int, int, int, int]]:
    """Selected ranges (top, bottom, left, right), inclusive.

    Uses the selection ranges only. selectedIndexes() makes one Python
    object per cell (GBs of RAM and minutes for a Ctrl+A of millions of rows).
    """
    sel = view.selectionModel()
    if sel is None:
        return []
    return [(r.top(), r.bottom(), r.left(), r.right()) for r in sel.selection() if r.isValid()]


def selection_grid(ranges) -> tuple[list[tuple[int, int]], list[int]]:
    """Merged row intervals [r0, r1) and the sorted columns of the ranges."""
    rows: list[tuple[int, int]] = []
    for a, b in sorted((t, bt + 1) for t, bt, _, _ in ranges):
        if rows and a <= rows[-1][1]:
            rows[-1] = (rows[-1][0], max(rows[-1][1], b))
        else:
            rows.append((a, b))
    cols = sorted({c for _, _, left, right in ranges for c in range(left, right + 1)})
    return rows, cols


def copy_selection(view: QTableView, value_of, ranges=None, column_texts=None) -> str:
    """TSV text of the selected cells. value_of(row, col) -> text.

    column_texts(col, r0, r1) -> list of texts: optional fast path for
    one rectangular selection.
    """
    if ranges is None:
        ranges = selection_ranges(view)
    if not ranges:
        return ""
    rows, cols = selection_grid(ranges)
    if sum(b - a for a, b in rows) * len(cols) > COPY_MAX_CELLS:
        return ""
    if len(ranges) == 1 and column_texts is not None:
        (r0, r1), = rows
        return "\n".join("\t".join(t) for t in zip(*(column_texts(c, r0, r1) for c in cols)))
    lines = []
    for a, b in rows:
        for r in range(a, b):
            hit = [(left, right) for t, bt, left, right in ranges if t <= r <= bt]
            lines.append("\t".join(value_of(r, c) if any(left <= c <= right for left, right in hit) else ""
                                   for c in cols))
    return "\n".join(lines)


def set_clipboard(text: str) -> None:
    QGuiApplication.clipboard().setText(text)
