# TDMS Viewer

Fast, read-only viewer for NI TDMS files on Ubuntu. It has the functions
of the NI "TDMS File Viewer" and adds tools for fast analysis. It is
free and open source (Python, Qt).

Run from the source folder, in its own virtual environment (`.venv`):

```bash
./run.sh measurement.tdms    # first start: makes .venv and installs (~1 min, ~760 MB)
python3 main.py              # later: main.py uses .venv automatically
```

Without a virtual environment: `python3 -m pip install -r requirements.txt`,
then `TDMSVIEWER_NO_VENV=1 python3 main.py`. An active virtual environment
of your own is always used as it is.

Or install it for your user (command `tdmsviewer`, menu entry, double-click
on `.tdms` files):

```bash
packaging/install-ubuntu.sh
tdmsviewer measurement.tdms
```

If Qt reports a missing `xcb` plugin, install the system libraries:
`sudo apt-get install libxcb-cursor0 libxkbcommon-x11-0 libegl1`.

## Main points

- **Same layout as NI.** File path, "File contents" tree, property table,
  graph with palette and legend, values table, Start index / Samples / All.
- **Any file size.** The graph shows the exact minimum and maximum of each
  screen pixel column. A one-sample spike is never hidden. Zoom and pan stay
  fast on files larger than RAM.
- **Exact values.** Tables show the stored value with full precision.
  Statistics are exact for any range (relative error ~1e-16). Timestamps
  are exact to 1 ns (npTDMS alone rounds to 1 µs). int64/uint64 values stay
  exact integers.
- **Nothing is hidden.** ±Inf samples are drawn as red triangles. Samples
  with NaN (dropouts) break the line, also when zoomed out.
- **Read-only.** The viewer never writes to a TDMS file.

## Functions

| Function | How to use |
|---|---|
| Open | `...` button, Ctrl+O, drag and drop, recent files, F5 reload |
| Select data | Tree: file = all channels, group = its channels, channel = one. Ctrl/Shift-click for more |
| Range | Start index, Samples, All (All is on by default) |
| Graph palette | Zoom to fit, zoom box, zoom X, zoom Y, zoom about point, pan, previous view |
| Mouse | Wheel zoom (Shift: X, Ctrl: Y), middle drag pan, double-click fit |
| Legend | Checkbox per plot. Right-click: show all / hide all / show only, color, line width, line / points |
| X axis | Waveform time (`wf_increment`), sample index, or any channel (for example `Time`). Labels: number, relative time (HH:MM:SS.fff), absolute local time |
| X-Y plot | Select a channel that is not monotonic as X (for example current) |
| Cursors | Two cursors (C key). Readout of x, Δx, 1/Δx. Table follows cursor 1 |
| Statistics | Tab "Statistics": N, NaN count, min, max, peak-peak, mean, std dev, RMS, cursor values, C2 − C1, for the visible range or between the cursors |
| Values table | Lazy, up to 2^31 rows. Ctrl+C copies with full precision |
| Properties | NI order (ASCII sort), NI_ChannelLength, NI_DataType, filter box |
| Export | Ctrl+E: visible range as CSV (full resolution); graph image: right-click > Export |

## How it is fast

1. **Metadata only at open.** Channel data loads in the background.
2. **Direct reads.** A reader uses the npTDMS segment table and reads with
   `preadv`, 3 to 10 times faster than npTDMS. It is compared bit by bit
   with npTDMS for each channel before use; if not equal, npTDMS is used.
3. **RAM when it fits.** Channels go to RAM up to 40 % of free memory. Larger
   files stay on disk.
4. **Min/max pyramid.** Per 256 samples: count, min, max, mean and M2. Levels
   of 8x. A view update touches about 2 values per pixel, not all samples.
5. **One worker thread** owns the file. The newest request wins. The window
   never waits for the disk.

## Downsides and limits

- A channel as X axis must fit in RAM (8 bytes per sample).
- DAQmx raw, scaled, string and timestamp channels use npTDMS (slower reads).
- Interleaved and DAQmx files load in one pass of npTDMS `data_chunks()`.
- A file that is still being written is shown as it was at open. Use F5.
- The graph shows the envelope (min/max) when zoomed out. Zoom in to see
  single samples (markers appear when points are far apart).

## Develop

```bash
python3 -m pip install -e ".[test]"
QT_QPA_PLATFORM=offscreen python3 -m pytest
python3 -m tdmsviewer file.tdms
```

Dependencies: NumPy (BSD), npTDMS (LGPL-3.0), PySide6 (LGPL-3.0),
pyqtgraph (MIT).
