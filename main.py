#!/usr/bin/env python3
"""Start TDMS Viewer from the source folder.

    python3 main.py [file.tdms]
"""

import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Use the project virtual environment (.venv, made by run.sh) if it exists.
if os.name == "nt":
    _venv_python = os.path.join(HERE, ".venv", "Scripts", "python.exe")
else:
    _venv_python = os.path.join(HERE, ".venv", "bin", "python")
if (os.path.exists(_venv_python) and sys.prefix == sys.base_prefix
        and not os.environ.get("TDMSVIEWER_NO_VENV")):
    _argv = [_venv_python, os.path.abspath(__file__), *sys.argv[1:]]
    if os.name == "nt":  # Windows has no real exec: run it and pass on the exit code
        sys.exit(subprocess.call(_argv))
    os.execv(_venv_python, _argv)

sys.path.insert(0, HERE)

_missing = [_pkg for _mod, _pkg in (("numpy", "numpy"), ("nptdms", "npTDMS"), ("pyqtgraph", "pyqtgraph"))
            if importlib.util.find_spec(_mod) is None]
if not any(importlib.util.find_spec(_qt) for _qt in ("PySide6", "PyQt6", "PyQt5", "PySide2")):
    _missing.append("PySide6")  # any Qt binding that pyqtgraph supports works
if _missing:
    sys.exit("Missing Python packages: " + ", ".join(_missing) + "\n"
             "Start with ./run.sh (it makes a virtual environment), or install them:\n"
             "    python3 -m pip install " + " ".join(_missing))

from tdmsviewer.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
