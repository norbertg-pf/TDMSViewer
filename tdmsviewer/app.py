"""Program entry: `tdmsviewer [file.tdms]` or `python -m tdmsviewer [file.tdms]`."""

from __future__ import annotations

import argparse
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tdmsviewer", description="Fast, read-only viewer for NI TDMS files.")
    parser.add_argument("file", nargs="?", help="TDMS file to open")
    args, qt_args = parser.parse_known_args(argv)

    from pyqtgraph.Qt import QT_LIB
    from pyqtgraph.Qt.QtCore import Qt
    from pyqtgraph.Qt.QtWidgets import QApplication

    # GUI and engine threads share the GIL: switch more often (default 5 ms).
    sys.setswitchinterval(0.002)

    if QT_LIB in ("PyQt5", "PySide2"):  # Qt 6 does this by default
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("TDMS Viewer")
    app.setOrganizationName("tdmsviewer")
    app.setDesktopFileName("tdmsviewer")
    if app.style().objectName().lower() not in ("fusion", "breeze", "adwaita"):
        app.setStyle("Fusion")
    import os

    from pyqtgraph.Qt.QtGui import QIcon

    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "icon.svg")))

    from .engine import DataEngine
    from .mainwindow import MainWindow

    engine = DataEngine()
    win = MainWindow(engine)
    app.aboutToQuit.connect(engine.shutdown)
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in the terminal quits
    win.show()
    if args.file:
        win.open_file(args.file)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
