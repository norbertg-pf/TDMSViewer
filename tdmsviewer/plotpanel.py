"""Graph panel: waveform graph, graph palette, plot legend and cursors.

Summary
    The graph shows only the samples that the engine sends for the
    visible range (peak-preserving). The panel reports view changes;
    the main window requests new data. X autorange on the drawn data is
    blocked, because the drawn data is only a window of the channel.

Speed with many plots
    Curves are kept in a pool and used again (creating and deleting
    2000 PlotDataItems takes seconds). Auto Y is computed here from the
    drawn data inside the X view, once per data update (set_data_many);
    the pyqtgraph autorange computes the bounds of every curve on every
    repaint and is not used. A range change is applied at once (not in
    the next paint), so each change paints one time. Automatic point
    markers (zoomed in to few samples) only for up to MAX_MARKER_PLOTS
    visible plots.

Mouse
    Left drag   zoom box / zoom X / zoom Y / pan (palette tool)
    Middle drag pan          Right drag  zoom about the start point
    Wheel       zoom (Shift: X only, Ctrl: Y only, on an axis: that axis)
    Double click zoom to fit
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from pyqtgraph.Qt.QtGui import QAction, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from pyqtgraph.Qt.QtWidgets import (
    QAbstractItemView, QButtonGroup, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu,
    QSplitter, QToolButton, QVBoxLayout, QWidget,
)

from . import theme
from .formatting import format_duration, format_si
from .xaxis import FMT_ABSOLUTE, FMT_NUMBER, FMT_RELATIVE, GridAxisItem, TimeAxisItem, TimeRef

TOOL_ZOOM, TOOL_ZOOMX, TOOL_ZOOMY, TOOL_PAN = "zoom", "zoomx", "zoomy", "pan"
TOOL_ZOOMPT = "zoompt"  # click: zoom in 2x about the point, Shift+click: zoom out
STYLE_LINE, STYLE_POINTS, STYLE_BOTH = "line", "points", "both"
_ZOOM_TOOLS = (TOOL_ZOOM, TOOL_ZOOMX, TOOL_ZOOMY)
MARGIN = 0.25  # extra data fetched on each side of the view (fraction of width)
SPARSE_PX_PER_POINT = 8.0  # show point markers when points are this far apart
MAX_MARKER_PLOTS = 100  # no automatic point markers above this number of visible plots


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
    """ViewBox with LabVIEW-like tools and no X autorange.

    Auto Y: state["autoRange"][1] is only the on/off flag (pyqtgraph sets
    it off when the user changes Y). The panel sets the Y range itself.
    """

    sigFit = Signal()
    sigBeforeChange = Signal()
    sigAutoY = Signal()  # auto Y was switched on: set the Y range from the drawn data

    def __init__(self):
        super().__init__(enableMenu=True)
        self.tool = TOOL_ZOOM
        self.setMouseMode(self.PanMode)
        super().enableAutoRange(self.XAxis, False)
        super().enableAutoRange(self.YAxis, True)

    def setMouseMode(self, mode):
        # The palette sets the left-drag tool. The ViewBox "1 button" mode
        # would turn the pan tool into a zoom box: always keep PanMode.
        super().setMouseMode(self.PanMode)

    # X autorange would fit the drawn window only. Route it to "fit".
    def enableAutoRange(self, axis=None, enable=True, x=None, y=None):
        if x is not None or y is not None:
            if x is not None:
                self.enableAutoRange(self.XAxis, x)
            if y is not None:
                self.enableAutoRange(self.YAxis, y)
            return
        if axis is None or axis in (self.XYAxes, "xy"):
            self.enableAutoRange(self.XAxis, enable)
            self.enableAutoRange(self.YAxis, enable)
            return
        if axis in (self.XAxis, "x") and enable is not False and enable != 0:
            super().enableAutoRange(self.XAxis, False)
            QTimer.singleShot(0, self.sigFit.emit)
            return
        was_on = self.state["autoRange"][1] is not False
        super().enableAutoRange(axis, enable)
        if axis in (self.YAxis, "y") and enable is not False and not was_on:
            self.sigAutoY.emit()

    def updateAutoRange(self):
        # Do not compute the bounds of all curves (slow): PlotPanel.update_auto_y does it.
        self._autoRangeNeedsUpdate = False

    def updateViewRange(self, forceX=False, forceY=False):
        super().updateViewRange(forceX, forceY)
        # pyqtgraph applies a new range in the next paint; the curves then change
        # their geometry during that paint and Qt paints all again. Apply it now:
        # one paint per change.
        r = self.rect()
        if self._matrixNeedsUpdate and r.width() > 0 and r.height() > 0:
            self.updateMatrix()

    def autoRange(self, padding=None, items=None, item=None):
        self.sigFit.emit()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.LeftButton and ev.double():
            ev.accept()
            self.sigFit.emit()
            return
        if ev.button() == Qt.LeftButton and self.tool == TOOL_ZOOMPT:
            ev.accept()
            self.sigBeforeChange.emit()
            f = 2.0 if ev.modifiers() & Qt.ShiftModifier else 0.5
            self.scaleBy((f, f), center=self.mapToView(ev.pos()))
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


_ICONS: dict[tuple[int, bool], QIcon] = {}


def _line_icon(color: QColor, enabled: bool = True) -> QIcon:
    """Legend icon (cached: a selection has few colors but can have 1000s of plots)."""
    key = (QColor(color).rgba(), bool(enabled))
    icon = _ICONS.get(key)
    if icon is None:
        icon = _ICONS[key] = _draw_line_icon(color, enabled)
    return icon


def _draw_line_icon(color: QColor, enabled: bool) -> QIcon:
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
    elif kind == TOOL_ZOOMPT:
        p.drawEllipse(QRectF(2, 2, 11, 11))
        p.drawLine(QPointF(12, 12), QPointF(18, 18))
        p.drawLine(QPointF(5, 7.5), QPointF(10, 7.5))
        p.drawLine(QPointF(7.5, 5), QPointF(7.5, 10))
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
        self._curves: dict[int, pg.PlotDataItem] = {}  # current plots (curves from the pool)
        self._pool: list[pg.PlotDataItem] = []  # all curves, created once, used again
        self._n_used = 0  # curves of the pool in use
        self._inf_marks: dict[int, pg.ScatterPlotItem] = {}
        self._inf_free: list[pg.ScatterPlotItem] = []
        self._styles: dict[int, dict] = {}  # current plots: color, width, mode
        self._user_styles: dict[int, dict] = {}  # styles set in the legend (this session)
        self._sparse: dict[int, bool] = {}
        self._sorted: dict[int, bool] = {}  # x of the drawn data is sorted (not an X-Y plot)
        self._hidden: set[int] = set()  # plots hidden in the legend
        self._entries: dict[int, LegendEntry] = {}
        self._legend_key: list = []
        self._history: list = []
        self._last_push = 0.0
        self._restoring = False
        self.x_format = FMT_NUMBER
        self.t_ref = None

        self.vb = GraphViewBox()
        self.xaxis = TimeAxisItem("bottom")
        self.plot = pg.PlotWidget(viewBox=self.vb, axisItems={"bottom": self.xaxis, "left": GridAxisItem("left")})
        pi = self.plot.getPlotItem()
        # No "Plot Options" menu: its transforms (log, FFT, ...) change the axes
        # but not the decimated data, so values read off the axes would be wrong.
        pi.setMenuEnabled(False, enableViewBoxMenu=True)
        self._hide_mouse_mode_menu()
        pi.hideButtons()
        pi.showGrid(x=True, y=True, alpha=theme.GRID_ALPHA)
        for name in ("bottom", "left"):
            pi.getAxis(name).behind_curves()
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
                          (TOOL_ZOOMY, "Zoom Y only (Y)"),
                          (TOOL_ZOOMPT, "Zoom about point: click zooms in, Shift+click zooms out"),
                          (TOOL_PAN, "Pan (P). Middle drag pans in all tools")):
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
        self._auto_y_timer = QTimer(self)
        self._auto_y_timer.setSingleShot(True)
        self._auto_y_timer.setInterval(0)
        self._auto_y_timer.timeout.connect(self.update_auto_y)
        self.vb.sigXRangeChanged.connect(self._schedule_view)
        self.vb.sigResized.connect(self._schedule_view)
        self.vb.sigFit.connect(self.fitRequested)
        self.vb.sigBeforeChange.connect(self._push_history)
        self.vb.sigAutoY.connect(self._auto_y_timer.start)
        self._hover_proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._hover)

        # Keys of the graph only: in the X combo boxes (same panel) C, S, ... select items.
        for key, fn in (("Z", lambda: self.set_tool(TOOL_ZOOM)), ("X", lambda: self.set_tool(TOOL_ZOOMX)),
                        ("Y", lambda: self.set_tool(TOOL_ZOOMY)), ("P", lambda: self.set_tool(TOOL_PAN)),
                        ("C", self.btn_cursors.toggle), ("Home", self.fitRequested.emit),
                        ("Backspace", self.back)):
            act = QAction(self.plot)
            act.setShortcut(key)
            act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            act.triggered.connect(fn)
            self.plot.addAction(act)

    def _hide_mouse_mode_menu(self) -> None:
        """Hide "Mouse Mode" of the ViewBox menu (the palette sets the tools)."""
        menu = self.vb.menu
        modes = getattr(menu, "mouseModes", None)
        if menu is None or not modes:
            return
        for a in menu.actions():
            if a.menu() is not None and modes[0] in a.menu().actions():
                a.setVisible(False)

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

    def set_channels(self, entries: list[LegendEntry], keep_hidden: bool = False) -> None:
        """Show these plots, without data (the data comes with set_data_many).

        keep_hidden: plots hidden in the legend stay hidden (same channel id),
        for example after a Start index or X source change.
        Curves come from the pool. The legend is built again only if the
        plots (id, number, color, enabled) change; texts are updated in place.
        """
        if not keep_hidden:
            self._hidden = set()
        self._release_inf_marks()
        self._styles.clear()
        self._sparse.clear()
        self._sorted.clear()
        self._entries = {e.cid: e for e in entries}
        enabled = [e for e in entries if e.enabled]
        while len(self._pool) < len(enabled):
            curve = pg.PlotDataItem(connect="finite", antialias=False, autoDownsample=False, clipToView=False)
            curve.setVisible(False)
            self.vb.addItem(curve)
            self._pool.append(curve)
        self._curves = {}
        n = len(entries)
        for curve, e in zip(self._pool, enabled):
            st = dict(self._user_styles.get(e.cid) or {"color": QColor(e.color), "width": 1, "mode": STYLE_LINE})
            self._styles[e.cid] = st
            self._curves[e.cid] = curve
            curve.setData([], [], connect="finite", **self._style_args(e.cid, False))
            self._sparse[e.cid] = False
            curve.setZValue(10 + n - e.number)
            curve.setVisible(e.cid not in self._hidden)
        for curve in self._pool[len(enabled):self._n_used]:
            curve.setData([], [])
            curve.setVisible(False)
        self._n_used = len(enabled)
        key = [(e.cid, e.number, QColor(e.color).rgba(), e.enabled) for e in entries]
        self.legend.blockSignals(True)
        if key == self._legend_key:
            for i, e in enumerate(entries):
                it = self.legend.item(i)
                self._set_item_text(it, e)
                if e.enabled:
                    it.setCheckState(Qt.Unchecked if e.cid in self._hidden else Qt.Checked)
                    it.setIcon(_line_icon(self._styles[e.cid]["color"], True))
        else:
            self.legend.clear()
            for e in entries:
                it = QListWidgetItem(_line_icon(e.color, e.enabled), "")
                it.setData(Qt.UserRole, e.cid)
                self._set_item_text(it, e)
                if e.enabled:
                    it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                    it.setCheckState(Qt.Unchecked if e.cid in self._hidden else Qt.Checked)
                    it.setIcon(_line_icon(self._styles[e.cid]["color"], True))
                else:
                    it.setFlags(it.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsEnabled)
                self.legend.addItem(it)
            self._legend_key = key
        self.legend.blockSignals(False)
        self._update_readout()

    @staticmethod
    def _set_item_text(it: QListWidgetItem, e: LegendEntry) -> None:
        text = f"{e.number:02d}  {e.label}"
        if it.text() != text:
            it.setText(text)
        if it.toolTip() != e.tooltip:
            it.setToolTip(e.tooltip)

    def update_entries(self, entries: list[LegendEntry]) -> None:
        """New legend texts and tooltips of the current plots (no other change)."""
        by_cid = {e.cid: e for e in entries}
        for i in range(self.legend.count()):
            it = self.legend.item(i)
            e = by_cid.get(it.data(Qt.UserRole))
            if e is not None:
                self._entries[e.cid] = e
                self._set_item_text(it, e)

    def set_data(self, cid: int, x: np.ndarray, y: np.ndarray) -> None:
        """Data of one plot. Use set_data_many for more plots (one repaint, one auto Y)."""
        curve = self._curves.get(cid)
        if curve is None:
            return
        self._draw(cid, curve, x, self._prepare(cid, x, y), self._markers_allowed())

    def _prepare(self, cid: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Inf samples at the edge of the finite data; x sorted flag. Returns the y to draw."""
        if y.size and np.isinf(y).any():
            y = self._clamp_inf(cid, x, y)
        elif cid in self._inf_marks:
            self._inf_marks[cid].setData([], [])
        self._sorted[cid] = bool(x.size < 2 or np.all(x[1:] >= x[:-1]))
        return y

    def _markers_allowed(self) -> bool:
        """Point markers only for few visible plots (thousands of symbol sets repaint slowly)."""
        n = 0
        for c in self._curves.values():
            n += c.isVisible()
            if n > MAX_MARKER_PLOTS:
                return False
        return True

    def _draw(self, cid: int, curve, x: np.ndarray, y: np.ndarray, markers: bool) -> None:
        # Point markers only if all points together are sparse on the screen. The engine
        # sends the view plus a margin on each side (1 + 2 * MARGIN view widths).
        # Never count the points inside the view: data outside the view would count 0.
        w = max(1.0, float(self.vb.width())) * (1 + 2 * MARGIN)
        sparse = markers and 0 < x.size and x.size * SPARSE_PX_PER_POINT <= w
        if self._sparse.get(cid) == sparse:
            curve.setData(x, y)  # same style: no new pens
        else:
            curve.setData(x, y, connect="finite", **self._style_args(cid, sparse))
        self._sparse[cid] = sparse

    def set_data_many(self, items) -> None:
        """Data of many plots: items = [(cid, x, y), ...]. One Y update, one repaint."""
        markers = self._markers_allowed()
        todo = []
        for cid, x, y in items:
            curve = self._curves.get(cid)
            if curve is not None:
                todo.append((cid, curve, x, self._prepare(cid, x, y)))
        # Y range first: then each curve makes its display data once, for the new view.
        self._auto_y_timer.stop()
        rng = self._auto_y_range({cid: (x, y) for cid, _, x, y in todo})
        if rng is not None:
            self._set_y_range(rng, [curve for _, curve, _, _ in todo])
        for cid, curve, x, y in todo:
            self._draw(cid, curve, x, y, markers)

    def update_auto_y(self) -> None:
        """If auto Y is on: Y range of the drawn data inside the X view (NaN ignored)."""
        self._auto_y_timer.stop()
        rng = self._auto_y_range({})
        if rng is not None:
            self._set_y_range(rng, [])

    def _set_y_range(self, rng, new_data_curves) -> None:
        """Set the Y range. Curves that get new data next skip their redraw for this change
        (pyqtgraph dynamicRangeLimit: one updateItems per curve and Y change)."""
        saved = []
        for c in new_data_curves:
            lim = c.opts.get("dynamicRangeLimit")
            if lim is not None:
                c.opts["dynamicRangeLimit"] = None
                saved.append((c, lim))
        try:
            # Same padding as the pyqtgraph autorange; a flat line keeps the Y scale.
            self.vb.setRange(yRange=rng, padding=None, disableAutoRange=False)
        finally:
            for c, lim in saved:
                c.opts["dynamicRangeLimit"] = lim

    def _auto_y_range(self, new: dict):
        """(lo, hi) of the visible plots inside the X view, or None (auto Y off or no data).

        new: {cid: (x, y)} data that is not in the curves yet.
        """
        if self.vb.state["autoRange"][1] is False:
            return None
        (x0, x1), _ = self.vb.viewRange()
        lo, hi = math.inf, -math.inf
        for cid, curve in self._curves.items():
            if not curve.isVisible():
                continue
            x, y = new[cid] if cid in new else (curve.xData, curve.yData)
            if x is None or y is None or not x.size:
                continue
            if self._sorted.get(cid, False):
                i0 = int(np.searchsorted(x, x0, "left"))
                i1 = int(np.searchsorted(x, x1, "right"))
                if i1 - i0 < 2:  # view between two samples: use the line that crosses it
                    i0, i1 = max(0, i0 - 1), min(x.size, i1 + 1)
                seg = y[i0:i1]
            else:
                with np.errstate(invalid="ignore"):
                    seg = y[(x >= x0) & (x <= x1)]
            if not seg.size:
                continue
            a, b = np.fmin.reduce(seg), np.fmax.reduce(seg)  # NaN ignored; all NaN: NaN (skipped)
            if a <= b:
                lo, hi = min(lo, float(a)), max(hi, float(b))
        if lo <= hi and math.isfinite(lo) and math.isfinite(hi):
            return lo, hi
        return None

    def _style_args(self, cid: int, sparse: bool) -> dict:
        """pyqtgraph pen/symbol arguments of a plot's style."""
        st = self._styles.get(cid) or {"color": QColor("#000"), "width": 1, "mode": STYLE_LINE}
        col = st["color"]
        pen = pg.mkPen(col, width=st["width"]) if st["mode"] != STYLE_POINTS else None
        points = st["mode"] in (STYLE_POINTS, STYLE_BOTH) or sparse
        if points:
            return {"pen": pen, "symbol": "o", "symbolSize": 4 + st["width"],
                    "symbolPen": pg.mkPen(col), "symbolBrush": pg.mkBrush(col)}
        return {"pen": pen, "symbol": None}

    def _apply_style(self, cid: int) -> None:
        curve = self._curves.get(cid)
        if curve is None:
            return
        args = self._style_args(cid, self._sparse.get(cid, False))
        curve.setPen(args["pen"])
        curve.setSymbol(args["symbol"])
        if args["symbol"]:
            curve.setSymbolSize(args["symbolSize"])
            curve.setSymbolPen(args["symbolPen"])
            curve.setSymbolBrush(args["symbolBrush"])
        for i in range(self.legend.count()):
            it = self.legend.item(i)
            if it.data(Qt.UserRole) == cid:
                it.setIcon(_line_icon(self._styles[cid]["color"], True))
                break

    def set_style(self, cids, **changes) -> None:
        """Change color / width / mode of plots (view only; the file is not changed)."""
        for cid in cids:
            if cid not in self._styles:
                continue
            self._styles[cid].update(changes)
            self._user_styles[cid] = dict(self._styles[cid])
            self._apply_style(cid)

    def _clamp_inf(self, cid: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Draw +/-Inf samples at the edge of the finite data, with red triangles.

        pyqtgraph cannot draw Inf; without this an Inf spike would be invisible.
        """
        fin = y[np.isfinite(y)]
        lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
        pad = 0.08 * (hi - lo) or 0.08 * max(abs(hi), 1.0)
        pos, neg = y == np.inf, y == -np.inf
        y = y.copy()
        y[pos] = hi + pad
        y[neg] = lo - pad
        marks = self._inf_marks.get(cid)
        if marks is None:
            if self._inf_free:
                marks = self._inf_free.pop()
            else:
                marks = pg.ScatterPlotItem(size=11, pen=pg.mkPen("#b00020"), brush=pg.mkBrush("#ff4d6d"))
                marks.setZValue(900)
                marks.setToolTip("Inf sample (drawn at the edge of the finite data)")
                self.vb.addItem(marks, ignoreBounds=True)
            self._inf_marks[cid] = marks
        sel = pos | neg
        marks.setData(x[sel], y[sel], symbol=np.where(pos[sel], "t1", "t").tolist())
        marks.setVisible(self._curves[cid].isVisible())
        return y

    def _release_inf_marks(self) -> None:
        for m in self._inf_marks.values():
            m.setData([], [])
            m.setVisible(False)
            self._inf_free.append(m)
        self._inf_marks.clear()

    def clear_data(self) -> None:
        for c in self._curves.values():
            c.setData([], [])
        for m in self._inf_marks.values():
            m.setData([], [])

    def visible_cids(self) -> list[int]:
        return [cid for cid, c in self._curves.items() if c.isVisible()]

    def plotted_cids(self) -> list[int]:
        return list(self._curves)

    # -- axes -------------------------------------------------------------------------

    def set_x_axis(self, fmt: str, t_ref, label: str) -> None:
        """t_ref: TimeRef (or Unix seconds) of x == 0 for absolute labels."""
        self.x_format = fmt if (fmt != FMT_ABSOLUTE or t_ref is not None) else FMT_RELATIVE
        self.t_ref = TimeRef.of(t_ref)
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
                return self.t_ref.label(v, 9, "%Y-%m-%d %H:%M:%S")  # ns resolution
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

    def auto_y_on(self) -> bool:
        return self.vb.state["autoRange"][1] is not False

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
            self.cursors_to_view()
            for line in self.cursors:
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

    def cursors_to_view(self) -> None:
        """Put C1, C2 at 1/3 and 2/3 of the view if one is outside it or both are at one x."""
        x0, x1 = self.view_x_range()
        a, b = (float(c.value()) for c in self.cursors)
        if a != b and x0 <= a <= x1 and x0 <= b <= x1:
            return
        for i, line in enumerate(self.cursors):
            line.setValue(x0 + (x1 - x0) * (i + 1) / 3.0)

    def set_cursor_positions(self, a: float, b: float) -> None:
        for line, v in zip(self.cursors, (a, b)):
            line.setValue(v)

    def _cursor_moving(self) -> None:
        self._update_readout()
        self._cursor_timer.start()

    def _cursor_finished(self, line) -> None:
        if line is self.cursors[0]:
            self.cursorMoved.emit(float(line.value()))

    def _update_readout(self) -> None:
        if not self.cursors_on() or not self._entries:
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
            vis = item.checkState() == Qt.Checked
            curve.setVisible(vis)
            (self._hidden.discard if vis else self._hidden.add)(cid)
            if cid in self._inf_marks:
                self._inf_marks[cid].setVisible(curve.isVisible())
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
            (self._hidden.discard if vis else self._hidden.add)(cid)
            if cid in self._inf_marks:
                self._inf_marks[cid].setVisible(vis)
        self.legend.blockSignals(False)
        self.visibilityChanged.emit()

    def hidden_cids(self) -> set[int]:
        """Plots hidden in the legend."""
        return {cid for cid in self._hidden if cid in self._curves}

    def _legend_menu(self, pos) -> None:
        m = QMenu(self)
        item = self.legend.itemAt(pos)
        if item is not None and not item.isSelected():
            self.legend.clearSelection()
            item.setSelected(True)
        sel = [it.data(Qt.UserRole) for it in self.legend.selectedItems() if it.data(Qt.UserRole) in self._curves]
        m.addAction("Show all", lambda: self.set_all_visible(True))
        m.addAction("Hide all", lambda: self.set_all_visible(False))
        if sel:
            m.addAction("Show only selected", lambda: self.set_all_visible(True, only=set(sel)))
            m.addSeparator()
            m.addAction("Color...", lambda: self._pick_color(sel))
            wm = m.addMenu("Line width")
            for w in (1, 2, 3, 4):
                wm.addAction(str(w), lambda w=w: self.set_style(sel, width=w))
            sm = m.addMenu("Plot style")
            for label, mode in (("Line", STYLE_LINE), ("Points", STYLE_POINTS), ("Line and points", STYLE_BOTH)):
                sm.addAction(label, lambda mode=mode: self.set_style(sel, mode=mode))
            m.addAction("Reset style", lambda: self._reset_style(sel))
        m.exec(self.legend.mapToGlobal(pos))

    def _pick_color(self, cids) -> None:
        from pyqtgraph.Qt.QtWidgets import QColorDialog

        start = self._styles[cids[0]]["color"] if cids and cids[0] in self._styles else QColor("#000")
        col = QColorDialog.getColor(start, self, "Plot color")
        if col.isValid():
            self.set_style(cids, color=col)

    def _reset_style(self, cids) -> None:
        for cid in cids:
            self._user_styles.pop(cid, None)
            e = self._entries.get(cid)
            if e is not None and cid in self._styles:
                self._styles[cid] = {"color": QColor(e.color), "width": 1, "mode": STYLE_LINE}
                self._apply_style(cid)

    # -- hover ------------------------------------------------------------------------

    def _hover(self, evt) -> None:
        pos = evt[0]
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        p = self.vb.mapSceneToView(pos)
        self.hoverChanged.emit(f"x = {self.format_x(p.x())}    y = {format_si(p.y(), 8)}")
