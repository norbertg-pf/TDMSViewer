#!/usr/bin/env python3
"""Start TDMS Viewer from the source folder.

    python3 main.py [file.tdms]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_missing = []
for _mod, _pkg in (("numpy", "numpy"), ("nptdms", "npTDMS"), ("PySide6", "PySide6"), ("pyqtgraph", "pyqtgraph")):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    sys.exit("Missing Python packages: " + ", ".join(_missing) + "\n"
             "Install them with:\n    python3 -m pip install " + " ".join(_missing))

from tdmsviewer.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
