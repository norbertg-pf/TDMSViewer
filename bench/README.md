# Engine benchmark

`bench_engine.py` writes three synthetic TDMS files, one at a time (16 x 4M float64 = 512 MB each; `--segments` 500, 1 segment, `--small-segments` 20000), measures the real package code (TdmsSource, fast reader, pyramid, DataEngine in RAM and disk mode) and a naive baseline (npTDMS `TdmsFile.read` + numpy min/max), prints one markdown table on stdout and deletes the files.

```bash
QT_QPA_PLATFORM=offscreen python3 bench/bench_engine.py > bench.md                  # default size, about 2 min on 4 cores
QT_QPA_PLATFORM=offscreen python3 bench/bench_engine.py --samples 1000000 --cold    # smaller files, plus cold page cache rows
QT_QPA_PLATFORM=offscreen python3 bench/bench_engine.py --profile > bench.md       # plus cProfile tables (about 2 min more)
QT_QPA_PLATFORM=offscreen python3 bench/bench_engine.py --files none --extra my.tdms  # existing files only (read only)
```

`--workdir DIR` selects the disk (needs one file size free), `--keep` keeps the files. Timings are warm page cache unless a row says cold. Other processes add noise: the output shows the load average.

`bench_gui.py` measures the GUI paths (plot update and paint, full view and 1000x zoom, values table page) with a 16 x 2M sample file. Run it once per Qt binding:

```bash
QT_QPA_PLATFORM=offscreen PYQTGRAPH_QT_LIB=PySide6 python3 bench/bench_gui.py
QT_QPA_PLATFORM=offscreen PYQTGRAPH_QT_LIB=PyQt5 python3 bench/bench_gui.py
```
