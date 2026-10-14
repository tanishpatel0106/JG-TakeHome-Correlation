"""Processed artifacts: write them atomically, load them cheaply.

Layout of the artifact directory (default ``data/processed``)::

    returns.npy          float32 [T x N]  single source for every correlation (memmapped)
    tickers.json         [N] column labels (sorted)
    dates.json           [T] row labels (ISO dates)
    prices.parquet       long (date, ticker, price) cleaned prices, zstd; audit trail only
    daily_stats.parquet  date, n_valid, mean_corr, std_corr
    hist.npy             int64 [T x B] per-date histogram of pairwise rho
    hist_edges.npy       float64 [B + 1]
    manifest.json        provenance + sha256 of every artifact

Full N x N matrices are never stored: ~44 GB for all dates versus 18 MB of
returns plus ~50 ms to recompute one date on demand.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import (
    ARTIFACT_FORMAT_VERSION,
    DAILY_STATS_FILE,
    DATES_FILE,
    HIST_EDGES_FILE,
    HIST_FILE,
    MANIFEST_FILE,
    PRICES_FILE,
    RETURNS_FILE,
    TICKERS_FILE,
)
from .io import sha256_file

log = logging.getLogger(__name__)

#: Artifacts listed (and checksummed) in the manifest, in write order.
ARTIFACT_FILES: tuple[str, ...] = (
    RETURNS_FILE,
    TICKERS_FILE,
    DATES_FILE,
    PRICES_FILE,
    DAILY_STATS_FILE,
    HIST_FILE,
    HIST_EDGES_FILE,
)


class StoreError(RuntimeError):
    """The artifact directory is missing, incomplete or inconsistent."""


# --- writing -----------------------------------------------------------------


@contextmanager
def staging_dir(out_dir: str | Path) -> Iterator[Path]:
    """Yield an empty staging directory and publish it as ``out_dir`` on success.

    The staging directory is a sibling of ``out_dir`` (same filesystem), so the
    publish is two ``os.replace`` renames: old -> backup, staging -> out_dir.
    If the body raises, the staging directory is deleted and ``out_dir`` is
    untouched, so a failed build never leaves half-written artifacts behind.
    """
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    _publish(tmp, out_dir)


def _publish(tmp: Path, out_dir: Path) -> None:
    backup = None
    if out_dir.exists():
        backup = out_dir.parent / f".{out_dir.name}.old-{uuid.uuid4().hex[:8]}"
        os.replace(out_dir, backup)
    try:
        os.replace(tmp, out_dir)
    except BaseException:
        if backup is not None:
            os.replace(backup, out_dir)
        raise
    if backup is not None:
        # On Windows a file still memory-mapped by a running API cannot be
        # deleted; leave the backup behind rather than fail the build.
        shutil.rmtree(backup, ignore_errors=True)
        if backup.exists():
            log.warning("could not remove previous artifacts at %s", backup)


def write_returns(directory: Path, returns: np.ndarray) -> None:
    """Save returns as float32 ``[T x N]`` ``.npy`` (C order, memmappable)."""
    np.save(directory / RETURNS_FILE, np.ascontiguousarray(returns, dtype=np.float32))


def write_axes(directory: Path, tickers: Sequence[str], dates: Sequence[date]) -> None:
    """Save the fixed axis labels: tickers (sorted) and ISO dates."""
    iso = []
    for t in range(len(dates)):
        iso.append(dates[t].isoformat())
    _write_json(directory / TICKERS_FILE, list(tickers))
    _write_json(directory / DATES_FILE, iso)


def write_prices(
    directory: Path, prices: np.ndarray, tickers: Sequence[str], dates: Sequence[date]
) -> None:
    """Save cleaned prices in long format ``(date, ticker, price)`` as zstd Parquet.

    One row per (date, ticker) cell of the ``[T x N]`` matrix; ``price`` is null
    where the cell is missing or invalid. Built from numpy arrays (dictionary
    encoded tickers), so no per-row Python objects are created.
    """
    n_dates, n_tickers = prices.shape
    day = np.array(dates, dtype="datetime64[D]")
    table = pa.table(
        {
            "date": pa.array(np.repeat(day, n_tickers)).cast(pa.date32()),
            "ticker": pa.DictionaryArray.from_arrays(
                pa.array(np.tile(np.arange(n_tickers, dtype=np.int32), n_dates)),
                pa.array(list(tickers), type=pa.string()),
            ),
            "price": pa.array(prices.reshape(-1), type=pa.float64(), from_pandas=True),
        }
    )
    pq.write_table(table, directory / PRICES_FILE, compression="zstd", row_group_size=1 << 20)


def write_daily_stats(
    directory: Path,
    dates: Sequence[date],
    n_valid: np.ndarray,
    mean_corr: np.ndarray,
    std_corr: np.ndarray,
) -> None:
    """Save the per-date timeline. NaN mean/std are stored as nulls."""
    table = pa.table(
        {
            "date": pa.array(np.array(dates, dtype="datetime64[D]")).cast(pa.date32()),
            "n_valid": pa.array(np.asarray(n_valid, dtype=np.int32)),
            "mean_corr": pa.array(np.asarray(mean_corr, dtype=np.float64), from_pandas=True),
            "std_corr": pa.array(np.asarray(std_corr, dtype=np.float64), from_pandas=True),
        }
    )
    pq.write_table(table, directory / DAILY_STATS_FILE, compression="zstd")


def write_hist(directory: Path, counts: np.ndarray, edges: np.ndarray) -> None:
    """Save per-date histogram counts int64 ``[T x B]`` and edges float64 ``[B + 1]``."""
    np.save(directory / HIST_FILE, np.asarray(counts, dtype=np.int64))
    np.save(directory / HIST_EDGES_FILE, np.asarray(edges, dtype=np.float64))


def write_manifest(directory: Path, manifest: dict) -> dict:
    """Add format version and artifact checksums to ``manifest`` and save it.

    Every file in :data:`ARTIFACT_FILES` must already exist in ``directory``.
    Returns the completed manifest.
    """
    artifacts: dict[str, dict] = {}
    for k in range(len(ARTIFACT_FILES)):
        path = directory / ARTIFACT_FILES[k]
        artifacts[ARTIFACT_FILES[k]] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    full = {"format_version": ARTIFACT_FORMAT_VERSION, **manifest, "artifacts": artifacts}
    _write_json(directory / MANIFEST_FILE, full, indent=2)
    return full


def _write_json(path: Path, obj: object, indent: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, allow_nan=False)
        fh.write("\n")


# --- reading -----------------------------------------------------------------


@dataclass
class Store:
    """Loaded artifacts. ``returns`` is a read-only memmap; the rest is small.

    Attributes
    ----------
    returns:
        float32 ``[T x N]`` (memmap).
    tickers, dates:
        Axis labels; ``dates`` are ISO strings.
    ticker_index, date_index:
        ``ticker -> column`` and ``ISO date -> row`` lookups.
    n_valid, mean_corr, std_corr:
        Per-date aggregates ``[T]`` (mean/std NaN where undefined).
    hist, hist_edges:
        int64 ``[T x B]`` and float64 ``[B + 1]``.
    window:
        Window length W used by the build.
    first_valid_index:
        First date index with at least one valid pair (``-1`` if none).
    """

    data_dir: Path
    returns: np.ndarray
    tickers: list[str]
    dates: list[str]
    ticker_index: dict[str, int]
    date_index: dict[str, int]
    n_valid: np.ndarray
    mean_corr: np.ndarray
    std_corr: np.ndarray
    hist: np.ndarray
    hist_edges: np.ndarray
    manifest: dict
    window: int
    first_valid_index: int

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)


def load_store(data_dir: str | Path) -> Store:
    """Load an artifact directory written by the build.

    ``returns.npy`` is opened with ``mmap_mode="r"``: pages are read on demand
    and shared between processes. Shapes are cross-checked against the axes.

    Raises
    ------
    StoreError
        If a file is missing or the artifacts are inconsistent.
    """
    data_dir = Path(data_dir)
    missing = []
    required = (*ARTIFACT_FILES, MANIFEST_FILE)
    for k in range(len(required)):
        if not (data_dir / required[k]).is_file():
            missing.append(required[k])
    if missing:
        raise StoreError(
            f"{data_dir}: missing artifacts {missing}; run `python -m corrlib.build` first"
        )

    manifest = _read_json(data_dir / MANIFEST_FILE)
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise StoreError(
            f"{data_dir}: artifact format {manifest.get('format_version')!r}, "
            f"expected {ARTIFACT_FORMAT_VERSION}; rebuild the artifacts"
        )
    tickers = _read_json(data_dir / TICKERS_FILE)
    dates = _read_json(data_dir / DATES_FILE)
    returns = np.load(data_dir / RETURNS_FILE, mmap_mode="r")
    stats = pq.read_table(data_dir / DAILY_STATS_FILE)
    hist = np.load(data_dir / HIST_FILE)
    hist_edges = np.load(data_dir / HIST_EDGES_FILE)

    n_dates, n_tickers = len(dates), len(tickers)
    if returns.shape != (n_dates, n_tickers) or returns.dtype != np.float32:
        raise StoreError(f"returns.npy is {returns.dtype}{returns.shape}, expected float32{(n_dates, n_tickers)}")
    if stats.num_rows != n_dates or hist.shape != (n_dates, hist_edges.size - 1):
        raise StoreError("daily_stats / hist do not match the date axis")

    ticker_index: dict[str, int] = {}
    for j in range(n_tickers):
        ticker_index[tickers[j]] = j
    date_index: dict[str, int] = {}
    for t in range(n_dates):
        date_index[dates[t]] = t

    n_valid = stats.column("n_valid").to_numpy().astype(np.int32)
    first = np.flatnonzero(n_valid >= 2)
    return Store(
        data_dir=data_dir,
        returns=returns,
        tickers=tickers,
        dates=dates,
        ticker_index=ticker_index,
        date_index=date_index,
        n_valid=n_valid,
        mean_corr=_float_column(stats, "mean_corr"),
        std_corr=_float_column(stats, "std_corr"),
        hist=hist,
        hist_edges=hist_edges,
        manifest=manifest,
        window=int(manifest["window"]),
        first_valid_index=int(first[0]) if first.size else -1,
    )


def verify_manifest(data_dir: str | Path) -> list[str]:
    """Recompute artifact checksums. Returns a list of problems (empty if all match)."""
    data_dir = Path(data_dir)
    manifest = _read_json(data_dir / MANIFEST_FILE)
    problems = []
    names = list(manifest.get("artifacts", {}))
    for k in range(len(names)):
        path = data_dir / names[k]
        if not path.is_file():
            problems.append(f"{names[k]}: missing")
        elif sha256_file(path) != manifest["artifacts"][names[k]]["sha256"]:
            problems.append(f"{names[k]}: checksum mismatch")
    for k in range(len(ARTIFACT_FILES)):
        if ARTIFACT_FILES[k] not in manifest.get("artifacts", {}):
            problems.append(f"{ARTIFACT_FILES[k]}: not listed in manifest")
    return problems


def _float_column(table: pa.Table, name: str) -> np.ndarray:
    """Parquet float column -> float64 numpy with NaN for nulls."""
    return table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)


def _read_json(path: Path) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
