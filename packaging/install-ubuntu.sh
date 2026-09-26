#!/usr/bin/env bash
# Install TDMS Viewer for the current user on Ubuntu (22.04 or later).
#
# What it does:
#   1. Installs the Qt system libraries (needs sudo, skip with --no-apt).
#   2. Creates a virtual environment in ~/.local/share/tdmsviewer/venv.
#   3. Installs TDMS Viewer and its Python packages into it.
#   4. Adds the "tdmsviewer" command, a menu entry, and opens .tdms files
#      with a double-click.
# Remove: packaging/install-ubuntu.sh --uninstall
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(dirname "$HERE")"
PREFIX="${HOME}/.local"
VENV="${PREFIX}/share/tdmsviewer/venv"
BIN="${PREFIX}/bin"
APPS="${PREFIX}/share/applications"
ICONS="${PREFIX}/share/icons/hicolor/scalable/apps"
MIME="${PREFIX}/share/mime"

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -rf "${PREFIX}/share/tdmsviewer" "${BIN}/tdmsviewer" "${APPS}/tdmsviewer.desktop" \
           "${ICONS}/tdmsviewer.svg" "${MIME}/packages/tdmsviewer-mime.xml"
    update-mime-database "${MIME}" >/dev/null 2>&1 || true
    update-desktop-database "${APPS}" >/dev/null 2>&1 || true
    echo "TDMS Viewer removed."
    exit 0
fi

if [[ "${1:-}" != "--no-apt" ]]; then
    sudo apt-get install -y python3-venv libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0 \
        libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0 libxcb-randr0 \
        libxcb-render-util0 libxcb-image0 libfontconfig1 libdbus-1-3
fi

python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV}/bin/python" -m pip install "${SRC}"

mkdir -p "${BIN}" "${APPS}" "${ICONS}" "${MIME}/packages"
ln -sf "${VENV}/bin/tdmsviewer" "${BIN}/tdmsviewer"
install -m 644 "${HERE}/tdmsviewer.desktop" "${APPS}/tdmsviewer.desktop"
install -m 644 "${HERE}/tdmsviewer.svg" "${ICONS}/tdmsviewer.svg"
install -m 644 "${HERE}/tdmsviewer-mime.xml" "${MIME}/packages/tdmsviewer-mime.xml"
update-mime-database "${MIME}" >/dev/null 2>&1 || true
update-desktop-database "${APPS}" >/dev/null 2>&1 || true
xdg-mime default tdmsviewer.desktop application/x-tdms >/dev/null 2>&1 || true

echo "TDMS Viewer installed. Start it with: tdmsviewer [file.tdms]"
case ":${PATH}:" in *":${BIN}:"*) ;; *) echo "Note: add ${BIN} to PATH (log out and in on Ubuntu).";; esac
