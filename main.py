#!/usr/bin/env python3
"""Start TDMS Viewer from the source folder.

    python3 main.py [file.tdms]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Use the project virtual environment (.venv, made by run.sh) if it exists.
_venv_python = os.path.join(HERE, ".venv", "bin", "python")
if (os.path.exists(_venv_python) and sys.prefix == sys.base_prefix
        and not os.environ.get("TDMSVIEWER_NO_VENV")):
    os.execv(_venv_python, [_venv_python, os.path.abspath(__file__), *sys.argv[1:]])

sys.path.insert(0, HERE)

_missing = []
for _mod, _pkg in (("numpy", "numpy"), ("nptdms", "npTDMS"), ("PySide6", "PySide6"), ("pyqtgraph", "pyqtgraph")):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    sys.exit("Missing Python packages: " + ", ".join(_missing) + "\n"
             "Start with ./run.sh (it makes a virtual environment), or install them:\n"
             "    python3 -m pip install " + " ".join(_missing))

from tdmsviewer.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
