#!/usr/bin/env bash
# Start TDMS Viewer in its own virtual environment (.venv in this folder).
#
#   ./run.sh [file.tdms]
#
# First start: creates .venv and installs the packages (about 1 minute).
# Later starts: installs again only if requirements.txt changed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HERE}/.venv"
STAMP="${VENV}/.requirements.sha256"

if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "Creating virtual environment in ${VENV} ..."
    if ! python3 -m venv "${VENV}"; then
        echo "Cannot create the virtual environment. On Ubuntu run:" >&2
        echo "    sudo apt-get install python3-venv" >&2
        exit 1
    fi
fi

WANT="$(sha256sum "${HERE}/requirements.txt" | cut -d' ' -f1)"
if [[ ! -f "${STAMP}" || "$(cat "${STAMP}")" != "${WANT}" ]]; then
    echo "Installing packages into ${VENV} ..."
    "${VENV}/bin/python" -m pip install --quiet --upgrade pip
    "${VENV}/bin/python" -m pip install --quiet -r "${HERE}/requirements.txt"
    echo "${WANT}" > "${STAMP}"
fi

exec "${VENV}/bin/python" "${HERE}/main.py" "$@"
