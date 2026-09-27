#!/usr/bin/env python3
"""Benchmark of the GUI paths: plot update, plot paint, values table.

Compares Qt bindings. Run once per binding and compare the lines:
    QT_QPA_PLATFORM=offscreen PYQTGRAPH_QT_LIB=PySide6 python3 bench/bench_gui.py
    QT_QPA_PLATFORM=offscreen PYQTGRAPH_QT_LIB=PyQt5 python3 bench/bench_gui.py

Rows (median of --repeat runs, ms):
    plot update  view change -> engine -> plotReady -> curves set (all channels)
    plot paint   render the graph (grab) with all curves
    zoom in      same as plot update, 1000x zoom (raw samples, markers)
    table page   scroll the values table by one page and render it
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from nptdms import ChannelObject, TdmsWriter  # noqa: E402
from pyqtgraph.Qt import VERSION_INFO  # noqa: E402
from pyqtgraph.Qt.QtCore import QCoreApplication, QSettings  # noqa: E402
from pyqtgraph.Qt.QtWidgets import QApplication  # noqa: E402

from tdmsviewer.engine import DataEngine  # noqa: E402
from tdmsviewer.mainwindow import MainWindow  # noqa: E402

perf = time.perf_counter


def write_file(path: str, channels: int, samples: int, segments: int) -> None:
    """Noisy sines, `segments` segments, 1 kHz waveform channels."""
    rng = np.random.default_rng(1)
    per = samples // segments
    props = {"wf_increment": 1e-3, "wf_start_offset": 0.0}
    with TdmsWriter(path) as w:
        for s in range(segments):
            t = np.arange(s * per, (s + 1) * per)
            w.write_segment([ChannelObject("G", f"ch{c:02d}", np.sin(t / (500.0 + c)) + 0.01 * rng.standard_normal(per),
                                           properties=props) for c in range(channels)])


def wait_until(pred, timeout: float = 60.0) -> None:
    end = perf() + timeout
    while not pred():
        QCoreApplication.processEvents()
        if perf() > end:
            raise TimeoutError("benchmark step did not finish")
        time.sleep(0.0005)


def median_ms(fn, repeat: int) -> float:
    fn()  # warm up
    times = []
    for _ in range(repeat):
        t0 = perf()
        fn()
        times.append((perf() - t0) * 1e3)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--samples", type=int, default=2_000_000, help="samples per channel")
    ap.add_argument("--segments", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    app = QApplication([sys.argv[0]])
    tmp = tempfile.mkdtemp(prefix="tdmsviewer_bench_")
    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, tmp)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tmp)
    path = os.path.join(tmp, "bench.tdms")
    write_file(path, args.channels, args.samples, args.segments)

    engine = DataEngine()
    win = MainWindow(engine)
    win.resize(1600, 1000)
    win.show()
    t0 = perf()
    win.open_file(path)
    wait_until(lambda: win.model is not None and engine._stores and all(s.done for s in engine._stores))
    load_s = perf() - t0
    for _ in range(50):
        QCoreApplication.processEvents()

    done = {"seq": -1}
    handler = win._on_plot_ready

    def counting(gen, seq, results):
        handler(gen, seq, results)
        done["seq"] = seq

    engine.plotReady.disconnect(handler)
    engine.plotReady.connect(counting)
    x_end = (args.samples - 1) * 1e-3
    step = {"i": 0}

    def plot_update(width: float) -> None:
        step["i"] += 1
        x0 = (step["i"] % 7) / 7 * (x_end - width)
        win.plot.set_view(x0, x0 + width, auto_y=False)
        win.plot._view_timer.stop()  # the benchmark sends the request itself
        win._view_changed()
        want = win._plot_seq
        wait_until(lambda: done["seq"] >= want)

    rows = [("load (open to all channels loaded)", load_s * 1e3)]
    rows.append(("plot update, full view", median_ms(lambda: plot_update(x_end), args.repeat)))
    rows.append(("plot paint, full view", median_ms(lambda: win.plot.plot.grab(), args.repeat)))
    rows.append(("plot update, 1000x zoom", median_ms(lambda: plot_update(x_end / 1000), args.repeat)))
    rows.append(("plot paint, 1000x zoom", median_ms(lambda: win.plot.plot.grab(), args.repeat)))

    view = win.values_view
    bar = view.verticalScrollBar()
    page = {"row": 0}

    def table_page() -> None:
        page["row"] = (page["row"] + 7919) % max(1, bar.maximum())
        bar.setValue(page["row"])
        QCoreApplication.processEvents()
        view.viewport().grab()

    rows.append(("table page (scroll + render)", median_ms(table_page, args.repeat)))

    print(f"{VERSION_INFO}; {args.channels} channels x {args.samples} samples, {args.segments} segments")
    for name, ms in rows:
        print(f"  {name:38s} {ms:9.2f} ms")
    engine.shutdown()
    win.close()
    os.remove(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
