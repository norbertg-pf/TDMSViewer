"""Graph panel: waveform graph, graph palette, plot legend and cursors.

Summary
    The graph shows only the samples that the engine sends for the
    visible range (peak-preserving). The panel reports view changes;
    the main window requests new data. X autorange on the drawn data is
    blocked, because the drawn data is only a window of the channel.

Mouse
    Left drag   zoom box / zoom X / zoom Y / pan (palette tool)
    Middle drag pan          Right drag  zoom about the start point
    Wheel       zoom (Shift: X only, Ctrl: Y only, on an axis: that axis)
    Double click zoom to fit
"""

from __future__ import annotations

import datetime as _dt
import math
import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
    QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from . import theme
from .formatting import format_duration, format_si
from .xaxis import FMT_ABSOLUTE, FMT_NUMBER, FMT_RELATIVE, TimeAxisItem

TOOL_ZOOM, TOOL_ZOOMX, TOOL_ZOOMY, TOOL_PAN = "zoom", "zoomx", "zoomy", "pan"
_ZOOM_TOOLS = (TOOL_ZOOM, TOOL_ZOOMX, TOOL_ZOOMY)
MARGIN = 0.25  # extra data fetched on each side of the view (fraction of width)
SPARSE_PX_PER_POINT = 8.0  # show point markers when points are this far apart


def _keep_image_exporters() -> None:
    """Remove data exporters: they would export the decimated data only."""
    try:
        import pyqtgraph.exporters as ex

        keep = [e for e in ex.Exporter.Exporters if e.__name__ in ("ImageExporter", "SVGExporter", "PrintExporter")]
        ex.Exporter.Exporters[:] = keep
    except Exception:  # pragma: no cover
        pass


_keep_image_exporters()


class GraphViewBox(pg.ViewBox):
    """ViewBox with LabVIEW-like tools and no X autorange."""

    sigFit = Signal()
    sigBeforeChange = Signal()

    def __init__(self):
        super().__init__(enableMenu=True)
        self.tool = TOOL_ZOOM
        self.setMouseMode(self.PanMode)
        super().enableAutoRange(self.XAxis, False)
        super().enableAutoRange(self.YAxis, True)
        self.setAutoVisible(y=True)

    # X autorange would fit the drawn window only. Route it to "fit".
    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        if x is not None or y is not None:
            if x is not None:
                self.enableAutoRange(self.XAxis, x)
            if y is not None:
                self.enableAutoRange(self.YAxis, y)
            return
        if axis is None or axis == self.XYAxes:
            self.enableAutoRange(self.XAxis, enable)
            self.enableAutoRange(self.YAxis, enable)
            return
        if axis == self.XAxis and enable is not False and enable != 0:
            super().enableAutoRange(self.XAxis, False)
            QTimer.singleShot(0, self.sigFit.emit)
            return
        super().enableAutoRange(axis, enable)

    def autoRange(self, padding=None, items=None, item=None):
        self.sigFit.emit()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and ev.double():
            ev.accept()
            self.sigFit.emit()
            return
        super().mouseClickEvent(ev)

    def wheelEvent(self, ev, axis=None):
        if axis is None:
            mods = ev.modifiers()
            if mods & Qt.ShiftModifier:
                axis = 0
            elif mods & Qt.ControlModifier:
                axis = 1
        self.sigBeforeChange.emit()
        super().wheelEvent(ev, axis)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.LeftButton and axis is None and self.tool in _ZOOM_TOOLS:
            ev.accept()
            p1, p2 = ev.buttonDownPos(), ev.pos()
            if ev.isStart():
                self.sigBeforeChange.emit()
            full = self.boundingRect()
            if self.tool == TOOL_ZOOMX:
                p1, p2 = QPointF(p1.x(), full.top()), QPointF(p2.x(), full.bottom())
            elif self.tool == TOOL_ZOOMY:
                p1, p2 = QPointF(full.left(), p1.y()), QPointF(full.right(), p2.y())
            if ev.isFinish():
                self.rbScaleBox.hide()
                r = QRectF(p1, p2).normalized()
                if r.width() < 3 and self.tool != TOOL_ZOOMY or r.height() < 3 and self.tool != TOOL_ZOOMX:
                    return
                vr = self.childGroup.mapRectFromParent(r)
                if self.tool == TOOL_ZOOM:
                    self.setRange(xRange=(vr.left(), vr.right()), yRange=(vr.top(), vr.bottom()), padding=0)
                elif self.tool == TOOL_ZOOMX:
                    self.setXRange(vr.left(), vr.right(), padding=0)
                else:
                    self.setYRange(vr.top(), vr.bottom(), padding=0)
            else:
                self.updateScaleBox(p1, p2)
            return
        if ev.isStart():
            self.sigBeforeChange.emit()
        super().mouseDragEvent(ev, axis)


@dataclass
class LegendEntry:
    cid: int
    number: int
    label: str
    color: QColor
    enabled: bool = True
    tooltip: str = ""


def _line_icon(color: QColor, enabled: bool = True) -> QIcon:
    pm = QPixmap(28, 14)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color) if enabled else QColor("#b0b0b0")
    p.setPen(QPen(c, 2))
    path = QPainterPath(QPointF(1, 11))
    path.lineTo(9, 4)
    path.lineTo(17, 9)
    path.lineTo(27, 2)
    p.drawPath(path)
    p.end()
    return QIcon(pm)


def _tool_icon(kind: str) -> QIcon:
    pm = QPixmap(20, 20)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor("#303030"), 1.6)
    p.setPen(pen)

    def arrow(x1, y1, x2, y2):
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        ang = math.atan2(y2 - y1, x2 - x1)
        for d in (2.5, -2.5):
            p.drawLine(QPointF(x2, y2), QPointF(x2 - 4 * math.cos(ang + d / 3), y2 - 4 * math.sin(ang + d / 3)))

    if kind == "fit":
        p.drawRect(QRectF(5, 5, 10, 10))
        for (a, b, c, d) in ((5, 5, 1, 1), (15, 5, 19, 1), (5, 15, 1, 19), (15, 15, 19, 19)):
            arrow(a, b, c, d)
    elif kind == TOOL_ZOOM:
        p.setPen(QPen(QColor("#303030"), 1.2, Qt.DashLine))
        p.drawRect(QRectF(2, 2, 11, 9))
        p.setPen(pen)
        p.drawEllipse(QRectF(9, 9, 7, 7))
        p.drawLine(QPointF(15, 15), QPointF(19, 19))
    elif kind == TOOL_ZOOMX:
        arrow(10, 10, 1.5, 10)
        arrow(10, 10, 18.5, 10)
        p.drawLine(QPointF(1, 4), QPointF(1, 16))
        p.drawLine(QPointF(19, 4), QPointF(19, 16))
    elif kind == TOOL_ZOOMY:
        arrow(10, 10, 10, 1.5)
        arrow(10, 10, 10, 18.5)
        p.drawLine(QPointF(4, 1), QPointF(16, 1))
        p.drawLine(QPointF(4, 19), QPointF(16, 19))
    elif kind == TOOL_PAN:
        for (x2, y2) in ((10, 1.5), (10, 18.5), (1.5, 10), (18.5, 10)):
            arrow(10, 10, x2, y2)
    elif kind == "back":
        path = QPainterPath(QPointF(15, 15))
        path.cubicTo(QPointF(18, 6), QPointF(10, 2), QPointF(4, 7))
        p.drawPath(path)
        p.drawLine(QPointF(4, 7), QPointF(4, 2))
        p.drawLine(QPointF(4, 7), QPointF(9, 8))
    elif kind == "cursors":
        p.setPen(QPen(QColor(theme.CURSOR_COLORS[0]), 2))
        p.drawLine(QPointF(6, 2), QPointF(6, 18))
        p.setPen(QPen(QColor(theme.CURSOR_COLORS[1]), 2))
        p.drawLine(QPointF(14, 2), QPointF(14, 18))
    p.end()
    return QIcon(pm)


class PlotPanel(QWidget):
    """Waveform graph with palette, legend and two measurement cursors."""

    viewChanged = Signal()  # x range or size changed (debounced)
    fitRequested = Signal()
    cursorsChanged = Signal()
    visibilityChanged = Signal()
    hoverChanged = Signal(str)
    cursorMoved = Signal(float)  # x of cursor 1 after a drag

    def __init__(self, parent=None):
        super().__init__(parent)
        pg.setConfigOptions(antialias=False, background=theme.PLOT_BG, foreground=theme.PLOT_FG)
        self._curves: dict[int, pg.PlotDataItem] = {}
        self._entries: dict[int, LegendEntry] = {}
        self._history: list = []
        self._last_push = 0.0
        self._restoring = False
        self.x_format = FMT_NUMBER
        self.t_ref = None

        self.vb = GraphViewBox()
        self.xaxis = TimeAxisItem("bottom")
        self.plot = pg.PlotWidget(viewBox=self.vb, axisItems={"bottom": self.xaxis})
        self.plot.setMenuEnabled(True)
        pi = self.plot.getPlotItem()
        pi.hideButtons()
        pi.showGrid(x=True, y=True, alpha=theme.GRID_ALPHA)
        pi.getAxis("left").enableAutoSIPrefix(False)
        pi.getAxis("left").setWidth(72)
        self.plot.setMinimumSize(300, 150)

        # Palette (top right, like the NI graph palette).
        self.palette_bar = QWidget()
        pb = QHBoxLayout(self.palette_bar)
        pb.setContentsMargins(0, 0, 0, 0)
        pb.setSpacing(1)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.btn_fit = self._tool_button("fit", "Zoom to fit (Home, double-click)")
        self.btn_fit.clicked.connect(self.fitRequested)
        pb.addWidget(self.btn_fit)
        self._tool_buttons = {}
        for tool, tip in ((TOOL_ZOOM, "Zoom to rectangle (Z)"), (TOOL_ZOOMX, "Zoom X only (X)"),
                          (TOOL_ZOOMY, "Zoom Y only (Y)"), (TOOL_PAN, "Pan (P). Middle drag pans in all tools")):
            b = self._tool_button(tool, tip, checkable=True)
            b.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            self.tool_group.addButton(b)
            self._tool_buttons[tool] = b
            pb.addWidget(b)
        self.btn_back = self._tool_button("back", "Previous view (Backspace)")
        self.btn_back.clicked.connect(self.back)
        self.btn_back.setEnabled(False)
        pb.addWidget(self.btn_back)
        self.btn_cursors = self._tool_button("cursors", "Measurement cursors (C)", checkable=True)
        self.btn_cursors.toggled.connect(self.set_cursors_visible)
        pb.addWidget(self.btn_cursors)
        self._tool_buttons[TOOL_ZOOM].setChecked(True)

        # Legend (right of the graph, like the NI plot legend).
        self.legend = QListWidget()
        self.legend.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.legend.setIconSize(QSize(28, 14))
        self.legend.setUniformItemSizes(True)
        self.legend.setContextMenuPolicy(Qt.CustomContextMenu)
        self.legend.customContextMenuRequested.connect(self._legend_menu)
        self.legend.itemChanged.connect(self._legend_changed)
        self.legend.setToolTip("Check to show a plot. Right-click for more.")

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(2)
        sl.addWidget(self.palette_bar)
        sl.addWidget(self.legend, 1)

        # Top bar: controls from the main window (left), readout (right).
        self.top_bar = QHBoxLayout()
        self.top_bar.setContentsMargins(4, 2, 4, 0)
        self.readout = QLabel("")
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.top_bar.addStretch(1)
        self.top_bar.addWidget(self.readout)

        self.split = QSplitter(Qt.Horizontal)
        self.split.addWidget(self.plot)
        self.split.addWidget(side)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([1000, 180])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(self.top_bar)
        lay.addWidget(self.split, 1)

        # Cursors.
        self.cursors = []
        for i, col in enumerate(theme.CURSOR_COLORS):
            line = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(col, width=1.5),
                                   hoverPen=pg.mkPen(col, width=3), label=f"C{i + 1}",
                                   labelOpts={"position": 0.97, "color": col, "fill": (255, 255, 255, 200)})
            line.setZValue(1000)
            line.sigPositionChanged.connect(self._cursor_moving)
            line.sigPositionChangeFinished.connect(self._cursor_finished)
            line.hide()
            self.vb.addItem(line, ignoreBounds=True)
            self.cursors.append(line)

        self._view_timer = QTimer(self)
        self._view_timer.setSingleShot(True)
        self._view_timer.setInterval(8)
        self._view_timer.timeout.connect(self.viewChanged)
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(40)
        self._cursor_timer.timeout.connect(self.cursorsChanged)
        self.vb.sigXRangeChanged.connect(self._schedule_view)
        self.vb.sigResized.connect(self._schedule_view)
        self.vb.sigFit.connect(self.fitRequested)
        self.vb.sigBeforeChange.connect(self._push_history)
        self._hover_proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._hover)

        for key, fn in (("Z", lambda: self.set_tool(TOOL_ZOOM)), ("X", lambda: self.set_tool(TOOL_ZOOMX)),
                        ("Y", lambda: self.set_tool(TOOL_ZOOMY)), ("P", lambda: self.set_tool(TOOL_PAN)),
                        ("C", self.btn_cursors.toggle), ("Home", self.fitRequested.emit),
                        ("Backspace", self.back)):
            act = QAction(self)
            act.setShortcut(key)
            act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            act.triggered.connect(fn)
            self.addAction(act)

    # -- construction helpers ---------------------------------------------------------

    def _tool_button(self, kind: str, tip: str, checkable: bool = False) -> QToolButton:
        b = QToolButton()
        b.setIcon(_tool_icon(kind))
        b.setIconSize(QSize(20, 20))
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setAutoRaise(True)
        return b

    # -- channels and data ------------------------------------------------------------

    def set_channels(self, entries: list[LegendEntry]) -> None:
        """Replace the plots. Keeps nothing from the previous selection."""
        for c in self._curves.values():
            self.vb.removeItem(c)
        self._curves.clear()
        self._entries = {e.cid: e for e in entries}
        self.legend.blockSignals(True)
        self.legend.clear()
        for e in entries:
            it = QListWidgetItem(_line_icon(e.color, e.enabled), f"{e.number:02d}  {e.label}")
            it.setData(Qt.UserRole, e.cid)
            it.setToolTip(e.tooltip)
            if e.enabled:
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked)
                curve = pg.PlotDataItem(pen=pg.mkPen(e.color, width=1), connect="finite",
                                        antialias=False, autoDownsample=False, clipToView=False)
                curve.setZValue(10 + len(entries) - e.number)
                self.vb.addItem(curve)
                self._curves[e.cid] = curve
            else:
                it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsEnabled)
            self.legend.addItem(it)
        self.legend.blockSignals(False)

    def set_data(self, cid: int, x: np.ndarray, y: np.ndarray) -> None:
        curve = self._curves.get(cid)
        if curve is None:
            return
        sparse = False
        if x.size >= 1:
            (x0, x1), w = self.vb.viewRange()[0], max(1.0, self.vb.width())
            span = x1 - x0
            if span > 0:
                visible = int(np.count_nonzero((x >= x0) & (x <= x1))) if x.size < 200000 else x.size
                sparse = visible * SPARSE_PX_PER_POINT <= w
        if sparse:
            e = self._entries[cid]
            curve.setData(x, y, connect="finite", symbol="o", symbolSize=5,
                          symbolPen=pg.mkPen(e.color), symbolBrush=pg.mkBrush(e.color))
        else:
            curve.setData(x, y, connect="finite", symbol=None)

    def clear_data(self) -> None:
        for c in self._curves.values():
            c.setData([], [])

    def visible_cids(self) -> list[int]:
        return [cid for cid, c in self._curves.items() if c.isVisible()]

    def plotted_cids(self) -> list[int]:
        return list(self._curves)

    # -- axes -------------------------------------------------------------------------

    def set_x_axis(self, fmt: str, t_ref: float | None, label: str) -> None:
        self.x_format = fmt if (fmt != FMT_ABSOLUTE or t_ref is not None) else FMT_RELATIVE
        self.t_ref = t_ref
        self.xaxis.set_format(self.x_format, t_ref)
        self.plot.getPlotItem().setLabel("bottom", label)

    def set_y_label(self, text: str) -> None:
        self.plot.getPlotItem().setLabel("left", text)

    def format_x(self, v: float) -> str:
        """Readout text of an x value (full resolution)."""
        if not math.isfinite(v):
            return str(v)
        if self.x_format == FMT_RELATIVE:
            return format_duration(v, 6)
        if self.x_format == FMT_ABSOLUTE and self.t_ref is not None:
            try:
                t = _dt.datetime.fromtimestamp(self.t_ref + v)
                return t.strftime("%Y-%m-%d %H:%M:%S.%f")
            except (OverflowError, OSError, ValueError):
                return format_si(v, 10)
        return format_si(v, 10)

    def format_dx(self, dv: float) -> str:
        if self.x_format in (FMT_RELATIVE, FMT_ABSOLUTE):
            return format_duration(dv, 6)
        return format_si(dv, 10)

    # -- view -------------------------------------------------------------------------

    def view_x_range(self) -> tuple[float, float]:
        (x0, x1), _ = self.vb.viewRange()
        return float(x0), float(x1)

    def fetch_window(self) -> tuple[float, float, int]:
        """X window to fetch (view plus margins) and its width in pixels."""
        x0, x1 = self.view_x_range()
        w = x1 - x0
        px = max(64.0, float(self.vb.width()))
        return x0 - MARGIN * w, x1 + MARGIN * w, int(px * (1 + 2 * MARGIN))

    def set_view(self, x0: float, x1: float, auto_y: bool = True) -> None:
        if not (math.isfinite(x0) and math.isfinite(x1)):
            return
        if x1 <= x0:
            pad = abs(x0) * 1e-6 or 1.0
            x0, x1 = x0 - pad, x1 + pad
        self.vb.setXRange(x0, x1, padding=0.0)
        if auto_y:
            self.vb.enableAutoRange(y=True)

    def set_y_view(self, y0: float, y1: float) -> None:
        self.vb.setYRange(y0, y1, padding=0.02)

    def _schedule_view(self, *args) -> None:
        if not self._view_timer.isActive():
            self._view_timer.start()

    # -- tools and history ------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        self.vb.tool = tool
        b = self._tool_buttons.get(tool)
        if b is not None and not b.isChecked():
            b.setChecked(True)

    def _push_history(self) -> None:
        if self._restoring:
            return
        now = time.monotonic()
        if now - self._last_push < 0.4:
            self._last_push = now
            return
        self._last_push = now
        xr, yr = self.vb.viewRange()
        auto_y = bool(self.vb.state["autoRange"][1])
        self._history.append((tuple(xr), tuple(yr), auto_y))
        del self._history[:-50]
        self.btn_back.setEnabled(True)

    def push_history(self) -> None:
        self._last_push = 0.0
        self._push_history()

    def back(self) -> None:
        if not self._history:
            return
        xr, yr, auto_y = self._history.pop()
        self._restoring = True
        try:
            self.vb.setRange(xRange=xr, yRange=yr, padding=0)
            if auto_y:
                self.vb.enableAutoRange(y=True)
        finally:
            self._restoring = False
        self.btn_back.setEnabled(bool(self._history))

    def clear_history(self) -> None:
        self._history.clear()
        self.btn_back.setEnabled(False)

    # -- cursors ----------------------------------------------------------------------

    def set_cursors_visible(self, on: bool) -> None:
        if self.btn_cursors.isChecked() != on:
            self.btn_cursors.setChecked(on)
            return
        if on:
            x0, x1 = self.view_x_range()
            for i, line in enumerate(self.cursors):
                v = line.value()
                if not (x0 <= v <= x1):
                    line.setValue(x0 + (x1 - x0) * (i + 1) / 3.0)
                line.show()
        else:
            for line in self.cursors:
                line.hide()
        self._update_readout()
        self.cursorsChanged.emit()

    def cursors_on(self) -> bool:
        return self.btn_cursors.isChecked()

    def cursor_positions(self) -> list[float]:
        return [float(c.value()) for c in self.cursors] if self.cursors_on() else []

    def _cursor_moving(self) -> None:
        self._update_readout()
        self._cursor_timer.start()

    def _cursor_finished(self, line) -> None:
        if line is self.cursors[0]:
            self.cursorMoved.emit(float(line.value()))

    def _update_readout(self) -> None:
        if not self.cursors_on():
            self.readout.setText("")
            return
        a, b = (float(c.value()) for c in self.cursors)
        d = b - a
        txt = f"C1: {self.format_x(a)}   C2: {self.format_x(b)}   Δ: {self.format_dx(d)}"
        if d != 0 and self.x_format != FMT_NUMBER:
            txt += f"   1/Δ: {format_si(1.0 / d, 7)} Hz"
        self.readout.setText(txt)

    def refresh_readout(self) -> None:
        self._update_readout()

    # -- legend -----------------------------------------------------------------------

    def _legend_changed(self, item: QListWidgetItem) -> None:
        cid = item.data(Qt.UserRole)
        curve = self._curves.get(cid)
        if curve is not None:
            curve.setVisible(item.checkState() == Qt.Checked)
            self.visibilityChanged.emit()

    def set_all_visible(self, on: bool, only: set | None = None) -> None:
        self.legend.blockSignals(True)
        for i in range(self.legend.count()):
            it = self.legend.item(i)
            cid = it.data(Qt.UserRole)
            if cid not in self._curves:
                continue
            vis = (cid in only) if only is not None else on
            it.setCheckState(Qt.Checked if vis else Qt.Unchecked)
            self._curves[cid].setVisible(vis)
        self.legend.blockSignals(False)
        self.visibilityChanged.emit()

    def _legend_menu(self, pos) -> None:
        m = QMenu(self)
        sel = {it.data(Qt.UserRole) for it in self.legend.selectedItems()}
        m.addAction("Show all", lambda: self.set_all_visible(True))
        m.addAction("Hide all", lambda: self.set_all_visible(False))
        if sel:
            m.addAction("Show only selected", lambda: self.set_all_visible(True, only=sel))
        m.exec(self.legend.mapToGlobal(pos))

    # -- hover ------------------------------------------------------------------------

    def _hover(self, evt) -> None:
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        self.hoverChanged.emit(f"x = {self.format_x(p.x())}    y = {format_si(p.y(), 8)}")
