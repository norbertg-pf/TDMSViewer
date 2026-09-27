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
- **Damaged files open.** Garbage or zeros after the last segment (writer
  crash), a `.tdms_index` from another run, names that are not UTF-8, EXT
  (80-bit) values and out-of-range timestamps give one clear warning each,
  never silently wrong data.
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
   `preadv`, 3 to 30 times faster than npTDMS. One pass over the segment
   table builds all channels; equal segments merge into one part.
3. **Checked before use.** npTDMS decodes the first values of the first
   and last segment of each layout kind (and spread segments). The direct
   reader must give the same bytes; if not, npTDMS is used for that channel.
4. **One pass for many channels.** Channels written in small pieces (many
   small segments, interleaved data) load together in one pass over the file.
5. **RAM when it fits.** Channels go to RAM up to 40 % of free memory. Larger
   files stay on disk.
6. **Min/max pyramid.** Per 256 samples: count, min, max, mean and M2. Levels
   of 8x. A view update touches about 2 values per pixel, not all samples.
7. **One worker thread** owns the file. The newest request wins. The window
   never waits for the disk.

## Downsides and limits

- A channel as X axis must fit in RAM (8 bytes per sample).
- DAQmx raw, scaled, string, timestamp and EXT channels use npTDMS (slower
  reads). In files with interleaved or DAQmx data they load in one pass of
  npTDMS `data_chunks()`.
- Opening a file with very many segments is limited by npTDMS: about 65 µs
  per segment (20 000 segments: 1.4 s; with a `.tdms_index` about 0.4 s).
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
pyqtgraph (MIT). The test suite passes with the oldest and the newest
allowed versions: NumPy 1.26 and 2.4, npTDMS 1.11, PySide6 6.8 and 6.11,
pyqtgraph 0.13.7 and 0.14. npTDMS and pyqtgraph are capped below their next
minor version, because the viewer uses their internals.
