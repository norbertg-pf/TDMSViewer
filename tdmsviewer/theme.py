"""Colors and fonts. The plot palette follows the NI graph order."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase

# NI plot order (00..09, then repeats). Light colors are a little darker
# than NI so that they stay readable on a white background.
PLOT_COLORS = [
    "#0f5aa6",  # 00 dark blue
    "#e3262d",  # 01 red
    "#1fb31f",  # 02 green
    "#27a6de",  # 03 sky blue
    "#9fc93a",  # 04 yellow green
    "#c64fe0",  # 05 violet
    "#f39c12",  # 06 orange
    "#2f4fd8",  # 07 blue
    "#ec4f9a",  # 08 pink
    "#35c3c9",  # 09 cyan
]

PLOT_BG = "#ffffff"
PLOT_FG = "#202020"
GRID_ALPHA = 0.18
CURSOR_COLORS = ("#d4380d", "#08979c")


def plot_color(i: int) -> QColor:
    return QColor(PLOT_COLORS[i % len(PLOT_COLORS)])


def mono_font() -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    return f
