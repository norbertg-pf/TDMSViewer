"""Shared fixtures for the tdmsviewer tests.

Synthetic TDMS file builders are in tdms_builders.py (import it
directly: ``import tdms_builders as tb``). This file only has fixtures.
"""

from __future__ import annotations

import os
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(TESTS_DIR), str(TESTS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tdms_builders as tb  # noqa: E402

# Real FlexLogger file (16 float64 channels x 44275 samples, 4 groups).
SAMPLE_TDMS = os.environ.get(
    "TDMSVIEWER_SAMPLE",
    "/root/.claude/uploads/5899cc9d-0cc2-5e06-94ba-ed80568becb4/"
    "24399bc7-CBL-1252_test_012_2026-07-28_12-05-36.tdms",
)


@pytest.fixture
def rng(request) -> np.random.Generator:
    """Random generator with a stable seed per test."""
    return np.random.default_rng(zlib.crc32(request.node.nodeid.encode()))


@pytest.fixture(scope="session")
def sample_tdms() -> str:
    """Path of the real sample file (test is skipped if it is missing)."""
    if not os.path.isfile(SAMPLE_TDMS):
        pytest.skip(f"sample file not found: {SAMPLE_TDMS}")
    return SAMPLE_TDMS


@pytest.fixture
def write_tdms(tmp_path):
    """write_tdms(builder, name="test.tdms", index=False, cut=0) -> path."""

    def _write(builder: tb.TdmsBuilder, name: str = "test.tdms", index=False, cut: int = 0) -> str:
        return builder.write(tmp_path / name, index=index, cut=cut)

    return _write


@pytest.fixture
def scenario(tmp_path):
    """scenario(name, seed=1234, index=False) -> (path, builder, notes).

    Names: tdms_builders.ALL_SCENARIOS.
    """

    def _build(name: str, seed: int = 1234, index=False):
        d = tmp_path / f"{name}_{seed}"
        d.mkdir(exist_ok=True)
        return tb.build_scenario(name, d, seed=seed, index=index)

    return _build


@pytest.fixture
def open_source():
    """open_source(path, **kw) -> TdmsSource. All sources are closed at teardown."""
    from tdmsviewer.tdmsfile import TdmsSource

    opened = []

    def _open(path, **kw):
        src = TdmsSource(os.fspath(path), **kw)
        opened.append(src)
        return src

    yield _open
    for src in opened:
        try:
            src.close()
        except Exception:
            pass


@pytest.fixture
def set_tz():
    """set_tz("Europe/Berlin") sets the local time zone; restored at teardown."""
    old = os.environ.get("TZ")

    def _set(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield _set
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()
