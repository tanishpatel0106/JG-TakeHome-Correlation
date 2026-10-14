"""Processed artifacts: write one snapshot directory, load it cheaply.

Layout of one snapshot directory (artifact format 2)::

    returns.npy             float32 [T x N]  single source for every correlation (memmapped)
    tickers.json            [N] column labels (sorted)
    dates.json              [T] row labels (ISO dates)
    prices.parquet          long (date, ticker, price_raw, price), zstd, grid order (date-major,
                            ticker-minor); price_raw = after per-file cleaning only (the source
                            evidence), price = the values actually used (anomaly handling applied)
    daily_stats.parquet     date, n_valid, mean_corr, std_corr
    hist.npy                int64 [T x B] per-date histogram of pairwise rho
    hist_edges.npy          float64 [B + 1]
    anomalies.json          {"rule", "view", "records": [...]}: one record per uniform-price day
    issues_by_date.parquet  date + every IssueCounts counter, one row per file date
    inputs.json             base zip identity + sha256 of every daily file's bytes (+ overlays)
    overrides.json          (optional) verbatim copy of the overrides file used
    manifest.json           provenance, policy, view, dataset, sha256 of every artifact

:func:`load_store` accepts such a directory, a **store root** (a directory
with a ``CURRENT`` pointer file naming ``snapshots/<id>``; see
:mod:`corrlib.snapshots`, which writes and publishes snapshots), or a legacy
format-1 directory (no anomalies/inputs/issues files; still served). The
``read_*`` helpers and :func:`verify_manifest` accept a root as well.

Full N x N matrices are never stored: ~44 GB for all dates versus 18 MB of
returns plus ~50 ms to recompute one date on demand.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .clean import IssueCounts, count_fields
from .config import (
    ANOMALIES_FILE,
    ANOMALY_RULE,
    ARTIFACT_FORMAT_VERSION,
    CURRENT_FILE,
    DAILY_STATS_FILE,
    DATES_FILE,
    HIST_EDGES_FILE,
    HIST_FILE,
    INPUTS_FILE,
    ISSUES_BY_DATE_FILE,
    LEGACY_FORMAT_VERSION,
    MANIFEST_FILE,
    OVERRIDES_FILE,
    PRICES_FILE,
    RETURNS_FILE,
    SNAPSHOTS_DIR,
    TICKERS_FILE,
    VALIDITY_EPS,
    VIEW,
    WINDOW,
    build_policy,
)
from .io import sha256_file
from .returns import (
    STATUS_INFERRED,
    action_for,
    affected_outputs,
    anomaly_masks,
    nulled_counts,
    summarize_anomalies,
)

log = logging.getLogger(__name__)

#: Artifacts of the legacy format 1 (all required there).
V1_ARTIFACT_FILES: tuple[str, ...] = (
    RETURNS_FILE,
    TICKERS_FILE,
    DATES_FILE,
    PRICES_FILE,
    DAILY_STATS_FILE,
    HIST_FILE,
    HIST_EDGES_FILE,
)

#: Artifacts every format-2 snapshot has, listed (and checksummed) in the manifest.
ARTIFACT_FILES: tuple[str, ...] = (
    *V1_ARTIFACT_FILES,
    ANOMALIES_FILE,
    ISSUES_BY_DATE_FILE,
    INPUTS_FILE,
)

#: Artifacts a format-2 snapshot may have; checksummed when present.
OPTIONAL_ARTIFACT_FILES: tuple[str, ...] = (OVERRIDES_FILE,)


class StoreError(RuntimeError):
    """The artifact directory is missing, incomplete or inconsistent."""


# --- writing -----------------------------------------------------------------


def write_returns(directory: Path, returns: np.ndarray) -> None:
    """Save returns as float32 ``[T x N]`` ``.npy`` (C order, memmappable)."""
    np.save(directory / RETURNS_FILE, np.ascontiguousarray(returns, dtype=np.float32))


def write_axes(directory: Path, tickers: Sequence[str], dates: Sequence[date]) -> None:
    """Save the fixed axis labels: tickers (sorted) and ISO dates."""
    _write_json(directory / TICKERS_FILE, list(tickers))
    _write_json(directory / DATES_FILE, _iso_list(dates))


def write_prices(
    directory: Path,
    prices_raw: np.ndarray,
    prices: np.ndarray,
    tickers: Sequence[str],
    dates: Sequence[date],
) -> None:
    """Save raw and used prices in long format ``(date, ticker, price_raw, price)``, zstd Parquet.

    One row per (date, ticker) cell of the ``[T x N]`` grid, date-major and
    ticker-minor (so ``column.reshape(T, N)`` restores the matrix). Values are
    null where the cell is missing or invalid. ``price_raw`` is after
    per-file cleaning only; ``price`` has the anomaly handling applied.
    Built from numpy arrays (dictionary-encoded tickers): no per-row Python objects.
    """
    n_dates, n_tickers = prices.shape
    if prices_raw.shape != prices.shape:
        raise ValueError(f"price_raw {prices_raw.shape} and price {prices.shape} differ")
    day = np.array(dates, dtype="datetime64[D]")
    table = pa.table(
        {
            "date": pa.array(np.repeat(day, n_tickers)).cast(pa.date32()),
            "ticker": pa.DictionaryArray.from_arrays(
                pa.array(np.tile(np.arange(n_tickers, dtype=np.int32), n_dates)),
                pa.array(list(tickers), type=pa.string()),
            ),
            "price_raw": pa.array(np.asarray(prices_raw, dtype=np.float64).reshape(-1), from_pandas=True),
            "price": pa.array(np.asarray(prices, dtype=np.float64).reshape(-1), from_pandas=True),
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


def write_anomalies(directory: Path, records: Sequence[dict[str, Any]]) -> None:
    """Save the anomaly records as ``{"rule", "view", "records": [...]}``."""
    _write_json(
        directory / ANOMALIES_FILE,
        {"rule": ANOMALY_RULE, "view": VIEW, "records": list(records)},
        indent=2,
    )


def write_issues_by_date(
    directory: Path, dates: Sequence[date], issues: Sequence[IssueCounts]
) -> None:
    """Save one row per file date: ``date`` + every :class:`IssueCounts` counter (int64)."""
    if len(dates) != len(issues):
        raise ValueError(f"{len(issues)} issue rows for {len(dates)} dates")
    names = count_fields()
    columns: dict[str, pa.Array] = {
        "date": pa.array(np.array(dates, dtype="datetime64[D]")).cast(pa.date32())
    }
    for k in range(len(names)):
        values = np.zeros(len(issues), dtype=np.int64)
        for t in range(len(issues)):
            values[t] = getattr(issues[t], names[k])
        columns[names[k]] = pa.array(values)
    pq.write_table(pa.table(columns), directory / ISSUES_BY_DATE_FILE, compression="zstd")


def write_inputs(directory: Path, inputs: dict[str, Any]) -> None:
    """Save the input identity document (``inputs.json``)."""
    _write_json(directory / INPUTS_FILE, inputs, indent=1)


def write_overrides(directory: Path, data: bytes) -> str:
    """Copy the overrides file bytes verbatim to ``overrides.json``; returns their sha256."""
    (directory / OVERRIDES_FILE).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def artifact_checksums(directory: Path) -> dict[str, dict[str, Any]]:
    """``{name: {sha256, size_bytes}}`` for every format-2 artifact (optional ones if present).

    Raises
    ------
    StoreError
        If a required artifact is missing.
    """
    names = list(ARTIFACT_FILES)
    for k in range(len(OPTIONAL_ARTIFACT_FILES)):
        if (directory / OPTIONAL_ARTIFACT_FILES[k]).is_file():
            names.append(OPTIONAL_ARTIFACT_FILES[k])
    artifacts: dict[str, dict[str, Any]] = {}
    for k in range(len(names)):
        path = directory / names[k]
        if not path.is_file():
            raise StoreError(f"{directory}: artifact {names[k]} was not written")
        artifacts[names[k]] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return artifacts


def write_manifest(
    directory: Path,
    manifest: dict[str, Any],
    snapshot: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add format version and artifact checksums to ``manifest`` and save it.

    Every file in :data:`ARTIFACT_FILES` must already exist in ``directory``.
    ``snapshot``, if given, is called with the artifact checksums and its
    result is stored as ``manifest["snapshot"]`` (so a snapshot id can be
    derived from the content). Returns the completed manifest.
    """
    artifacts = artifact_checksums(directory)
    full: dict[str, Any] = {"format_version": ARTIFACT_FORMAT_VERSION, **manifest}
    if snapshot is not None:
        full["snapshot"] = snapshot(artifacts)
    full["artifacts"] = artifacts
    _write_json(directory / MANIFEST_FILE, full, indent=2)
    return full


@dataclass
class SnapshotContent:
    """Everything a snapshot directory holds, in memory (input of :func:`write_snapshot`).

    Attributes
    ----------
    dates, tickers:
        The axes: ``T`` file dates and ``N`` sorted tickers.
    prices_raw:
        float64 ``[T x N]`` after per-file cleaning only (NaN = missing/invalid).
    prices:
        float64 ``[T x N]`` the prices actually used (anomaly handling applied).
    returns:
        float32 ``[T x N]`` the returns stored and served.
    n_valid, mean_corr, std_corr:
        Per-date aggregates: int32 / float64 / float64 ``[T]`` (NaN where undefined).
    hist, hist_edges:
        int64 ``[T x B]`` and float64 ``[B + 1]``.
    anomalies:
        Anomaly records (see :func:`corrlib.returns.anomaly_record`), date order.
    issues_by_date:
        One :class:`IssueCounts` per date.
    inputs:
        The ``inputs.json`` document.
    overrides_data:
        Bytes of the overrides file used, or None.
    window, eps:
        Window length W and validity threshold used for the aggregates.
    """

    dates: list[date]
    tickers: list[str]
    prices_raw: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    n_valid: np.ndarray
    mean_corr: np.ndarray
    std_corr: np.ndarray
    hist: np.ndarray
    hist_edges: np.ndarray
    anomalies: list[dict[str, Any]]
    issues_by_date: list[IssueCounts]
    inputs: dict[str, Any]
    overrides_data: bytes | None = None
    window: int = WINDOW
    eps: float = VALIDITY_EPS

    def check(self) -> None:
        """Raise :class:`StoreError` if the shapes or axes are inconsistent."""
        n_dates, n_tickers = len(self.dates), len(self.tickers)
        problems = []
        for name in ("prices_raw", "prices", "returns"):
            if getattr(self, name).shape != (n_dates, n_tickers):
                problems.append(f"{name} {getattr(self, name).shape} != {(n_dates, n_tickers)}")
        for name in ("n_valid", "mean_corr", "std_corr"):
            if getattr(self, name).shape != (n_dates,):
                problems.append(f"{name} {getattr(self, name).shape} != {(n_dates,)}")
        if self.hist.shape != (n_dates, self.hist_edges.size - 1):
            problems.append(f"hist {self.hist.shape} != {(n_dates, self.hist_edges.size - 1)}")
        if len(self.issues_by_date) != n_dates:
            problems.append(f"{len(self.issues_by_date)} issue rows for {n_dates} dates")
        for t in range(1, n_dates):
            if not self.dates[t - 1] < self.dates[t]:
                problems.append(f"dates not strictly increasing at index {t}")
                break
        if problems:
            raise StoreError("inconsistent snapshot content: " + "; ".join(problems))


def manifest_fields(content: SnapshotContent) -> dict[str, Any]:
    """Manifest entries derived from the content (axes, policy, issues, anomalies, overrides)."""
    first_valid = np.flatnonzero(np.asarray(content.n_valid) >= 2)
    total = IssueCounts()
    for t in range(len(content.issues_by_date)):
        total = total + content.issues_by_date[t]
    rebase, placeholder, unresolved = anomaly_masks(content.anomalies, len(content.dates))
    returns_nulled, placeholder_nulled, pending_nulled = nulled_counts(
        content.prices_raw, content.prices, rebase, placeholder, unresolved
    )
    overrides = None
    if content.overrides_data is not None:
        n_overrides = 0
        for k in range(len(content.anomalies)):
            n_overrides += content.anomalies[k].get("override") is not None
        overrides = {
            "file": OVERRIDES_FILE,
            "sha256": hashlib.sha256(content.overrides_data).hexdigest(),
            "count": n_overrides,
        }
    return {
        "view": VIEW,
        "policy": build_policy(content.window, content.eps),
        "window": content.window,
        "validity_eps": content.eps,
        "n_dates": len(content.dates),
        "n_tickers": len(content.tickers),
        "first_date": content.dates[0].isoformat(),
        "last_date": content.dates[-1].isoformat(),
        "first_valid_date": content.dates[int(first_valid[0])].isoformat() if first_valid.size else None,
        "hist": {
            "bins": int(content.hist_edges.size - 1),
            "range": [float(content.hist_edges[0]), float(content.hist_edges[-1])],
        },
        "issue_counts": total.as_dict(),
        "structural": {
            "rebase_dates": _iso_where(content.dates, rebase),
            "placeholder_dates": _iso_where(content.dates, placeholder),
            "unresolved_dates": _iso_where(content.dates, unresolved),
            "returns_nulled_at_rebase": returns_nulled,
            "prices_nulled_on_placeholder_days": placeholder_nulled,
            "prices_nulled_pending_next_day": pending_nulled,
        },
        "anomalies_summary": summarize_anomalies(content.anomalies),
        "overrides": overrides,
    }


def write_snapshot(
    directory: str | Path,
    content: SnapshotContent,
    extra: dict[str, Any],
    snapshot: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write every artifact of ``content`` plus the manifest into an empty ``directory``.

    Parameters
    ----------
    directory:
        Existing, empty directory (e.g. a staging directory).
    content:
        The snapshot data; checked with :meth:`SnapshotContent.check` first.
    extra:
        Manifest entries that cannot be derived from the content (e.g.
        ``corrlib_version``, ``built_at``, ``source``, ``dataset``, timings,
        ``changes``). Merged before the derived entries of :func:`manifest_fields`.
    snapshot:
        Optional callable receiving the artifact checksums and returning the
        manifest's ``snapshot`` block (see :func:`write_manifest`).

    Returns
    -------
    dict
        The manifest as written.
    """
    directory = Path(directory)
    content.check()
    if any(directory.iterdir()):
        raise StoreError(f"{directory}: snapshot directory is not empty")
    write_returns(directory, content.returns)
    write_axes(directory, content.tickers, content.dates)
    write_prices(directory, content.prices_raw, content.prices, content.tickers, content.dates)
    write_daily_stats(directory, content.dates, content.n_valid, content.mean_corr, content.std_corr)
    write_hist(directory, content.hist, content.hist_edges)
    write_anomalies(directory, content.anomalies)
    write_issues_by_date(directory, content.dates, content.issues_by_date)
    write_inputs(directory, content.inputs)
    if content.overrides_data is not None:
        write_overrides(directory, content.overrides_data)
    return write_manifest(directory, {**extra, **manifest_fields(content)}, snapshot=snapshot)


def _write_json(path: Path, obj: object, indent: int | None = None) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, allow_nan=False)
        fh.write("\n")


def write_json(path: str | Path, obj: object, indent: int | None = 2) -> None:
    """Write ``obj`` as UTF-8 JSON (NaN not allowed), with a trailing newline."""
    _write_json(Path(path), obj, indent=indent)


# --- reading -----------------------------------------------------------------


@dataclass
class Store:
    """Loaded artifacts. ``returns`` is a read-only memmap; the rest is small.

    Attributes
    ----------
    data_dir:
        The snapshot directory actually loaded.
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
    manifest:
        The parsed ``manifest.json``.
    window:
        Window length W used by the build.
    first_valid_index:
        First date index with at least one valid pair (``-1`` if none).
    snapshot_id:
        ``manifest.snapshot.id`` when present, else ``"legacy-"`` + the first
        12 hex digits of the manifest file's sha256.
    snapshot:
        ``{id, seq, parent, created_at, kind}`` (``seq`` is None for an
        unversioned directory; ``kind`` "legacy" for format 1).
    anomalies:
        Anomaly records (format 1: reconstructed from ``manifest.structural``,
        without evidence).
    policy:
        The data policy (``config.build_policy``) the artifacts were built with.
    view:
        ``"retrospective"``: history is cleaned with hindsight (see the policy).
    dataset:
        ``{kind: "source" | "demo", label}``.
    format_version:
        Artifact format of ``data_dir`` (1 or 2).
    root:
        The store root when loaded through a ``CURRENT`` pointer, else None.
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
    snapshot_id: str = "unversioned"
    snapshot: dict[str, Any] = field(default_factory=dict)
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=build_policy)
    view: str = VIEW
    dataset: dict[str, Any] = field(default_factory=lambda: {"kind": "source", "label": "Source data"})
    format_version: int = ARTIFACT_FORMAT_VERSION
    root: Path | None = None

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_tickers(self) -> int:
        return len(self.tickers)


def resolve_snapshot_dir(path: str | Path) -> tuple[Path, Path | None]:
    """Locate the snapshot directory for ``path``.

    * ``path/CURRENT`` exists -> ``path`` is a store root; returns
      ``path/snapshots/<CURRENT.snapshot>``. This wins over a ``manifest.json``
      in the same directory (legacy flat artifacts left in a directory that was
      later turned into a root by ``corrlib.build``).
    * ``path/manifest.json`` exists -> ``path`` is a snapshot (or legacy) directory.

    Returns
    -------
    (snapshot_dir, root)
        ``root`` is ``path`` when resolved through ``CURRENT``, else None.

    Raises
    ------
    StoreError
        If neither exists, or ``CURRENT`` is malformed or names a missing snapshot.
    """
    path = Path(path)
    if (path / CURRENT_FILE).is_file():
        try:
            pointer = _read_json(path / CURRENT_FILE)
        except (OSError, ValueError) as exc:
            raise StoreError(f"{path / CURRENT_FILE}: unreadable pointer ({exc})") from exc
        snapshot_id = pointer.get("snapshot") if isinstance(pointer, dict) else None
        if (
            not isinstance(snapshot_id, str)
            or not snapshot_id
            or snapshot_id.startswith(".")
            or snapshot_id != Path(snapshot_id).name
        ):
            raise StoreError(f"{path / CURRENT_FILE}: does not name a snapshot id")
        target = path / SNAPSHOTS_DIR / snapshot_id
        if not (target / MANIFEST_FILE).is_file():
            raise StoreError(f"{path}: CURRENT names {snapshot_id!r} but {target} has no manifest")
        return target, path
    if (path / MANIFEST_FILE).is_file():
        return path, None
    raise StoreError(
        f"{path}: missing artifacts (no {MANIFEST_FILE} and no {CURRENT_FILE}); "
        "run `python -m corrlib.build` first"
    )


def load_store(data_dir: str | Path) -> Store:
    """Load a snapshot directory, a store root (via ``CURRENT``) or a legacy v1 directory.

    ``returns.npy`` is opened with ``mmap_mode="r"``: pages are read on demand
    and shared between processes. Shapes are cross-checked against the axes.

    Raises
    ------
    StoreError
        If a file is missing or the artifacts are inconsistent.
    """
    directory, root = resolve_snapshot_dir(data_dir)

    manifest_bytes = (directory / MANIFEST_FILE).read_bytes()
    manifest = json.loads(manifest_bytes)
    version = manifest.get("format_version")
    if version not in (LEGACY_FORMAT_VERSION, ARTIFACT_FORMAT_VERSION):
        raise StoreError(
            f"{directory}: artifact format {version!r}, expected {LEGACY_FORMAT_VERSION} or "
            f"{ARTIFACT_FORMAT_VERSION}; rebuild the artifacts"
        )
    required = V1_ARTIFACT_FILES if version == LEGACY_FORMAT_VERSION else ARTIFACT_FILES
    missing = []
    for k in range(len(required)):
        if not (directory / required[k]).is_file():
            missing.append(required[k])
    if missing:
        raise StoreError(f"{directory}: missing artifacts {missing}; run `python -m corrlib.build` first")

    tickers = _read_json(directory / TICKERS_FILE)
    dates = _read_json(directory / DATES_FILE)
    returns = np.load(directory / RETURNS_FILE, mmap_mode="r")
    stats = pq.read_table(directory / DAILY_STATS_FILE)
    hist = np.load(directory / HIST_FILE)
    hist_edges = np.load(directory / HIST_EDGES_FILE)

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

    window = int(manifest["window"])
    eps = float(manifest.get("validity_eps", VALIDITY_EPS))
    if version == LEGACY_FORMAT_VERSION:
        anomalies = legacy_anomalies(manifest, dates, window)
        policy = build_policy(window, eps)
    else:
        document = _read_json(directory / ANOMALIES_FILE)
        anomalies = list(document["records"]) if isinstance(document, dict) else list(document)
        policy = manifest.get("policy") or build_policy(window, eps)

    snapshot = _snapshot_info(manifest, manifest_bytes, version)
    n_valid = stats.column("n_valid").to_numpy().astype(np.int32)
    first = np.flatnonzero(n_valid >= 2)
    return Store(
        data_dir=directory,
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
        window=window,
        first_valid_index=int(first[0]) if first.size else -1,
        snapshot_id=snapshot["id"],
        snapshot=snapshot,
        anomalies=anomalies,
        policy=policy,
        view=manifest.get("view", VIEW),
        dataset=_dataset_info(manifest),
        format_version=int(version),
        root=root,
    )


def _snapshot_info(manifest: dict, manifest_bytes: bytes, version: int) -> dict[str, Any]:
    """``manifest.snapshot`` when present, else an unversioned / legacy description."""
    block = manifest.get("snapshot")
    if isinstance(block, dict) and isinstance(block.get("id"), str):
        return {
            "id": block["id"],
            "seq": block.get("seq"),
            "parent": block.get("parent"),
            "created_at": block.get("created_at", manifest.get("built_at")),
            "kind": block.get("kind", "full_build"),
        }
    return {
        "id": "legacy-" + hashlib.sha256(manifest_bytes).hexdigest()[:12],
        "seq": None,
        "parent": None,
        "created_at": manifest.get("built_at"),
        "kind": "legacy" if version == LEGACY_FORMAT_VERSION else "full_build",
    }


def _dataset_info(manifest: dict) -> dict[str, Any]:
    dataset = manifest.get("dataset")
    if isinstance(dataset, dict) and dataset.get("kind"):
        return dict(dataset)
    name = manifest.get("source", {}).get("file_name", "unknown")
    return {"kind": "source", "label": f"Source data: {name}"}


def legacy_anomalies(manifest: dict, dates: Sequence[str], window: int) -> list[dict[str, Any]]:
    """Anomaly records for a format-1 directory, from ``manifest.structural`` (no evidence).

    The classification and the affected outputs are exact; the continuity
    numbers were not recorded by format 1, so ``evidence`` says so.
    """
    structural = manifest.get("structural", {})
    position: dict[str, int] = {}
    for t in range(len(dates)):
        position[dates[t]] = t
    found: list[tuple[int, str]] = []
    for kind, key in (("rebase", "rebase_dates"), ("placeholder", "placeholder_dates")):
        listed = structural.get(key, [])
        for k in range(len(listed)):
            if listed[k] in position:
                found.append((position[listed[k]], kind))
    found.sort()
    records = []
    n_dates = len(dates)
    for k in range(len(found)):
        t, classification = found[k]
        interior = 0 < t < n_dates - 1
        observed = [dates[t - 1], dates[t], dates[t + 1]] if interior else [dates[t]]
        action = action_for(classification, t)
        records.append(
            {
                "date": dates[t],
                "index": t,
                "uniform_value": None,
                "n_prices": None,
                "classification": classification,
                "status": STATUS_INFERRED,
                "rule": ANOMALY_RULE,
                "evidence": {
                    "previous_date": dates[t - 1] if t > 0 else None,
                    "next_date": dates[t + 1] if t < n_dates - 1 else None,
                    "median_abs_log_from_uniform": None,
                    "median_abs_log_from_previous": None,
                    "n_compared": None,
                    "rule_classification": classification,
                    "note": "legacy format-1 artifacts record only the classification; rebuild for the evidence",
                },
                "observation_dates_used": observed,
                "look_ahead": interior,
                "override": None,
                "action": action,
                "affected": affected_outputs(action, t, dates, window),
            }
        )
    return records


def snapshot_path(path: str | Path) -> Path:
    """The snapshot directory of ``path``: itself, or the CURRENT snapshot of a store root.

    Same resolution as :func:`resolve_snapshot_dir`; a directory with neither
    ``CURRENT`` nor ``manifest.json`` is returned unchanged (so a writer can
    read artifacts of a directory whose manifest is not written yet).
    """
    path = Path(path)
    if (path / CURRENT_FILE).is_file() or (path / MANIFEST_FILE).is_file():
        return resolve_snapshot_dir(path)[0]
    return path


def read_prices(directory: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Raw and used price grids ``(price_raw, price)``, float64 ``[T x N]`` each (NaN for null).

    Raises
    ------
    StoreError
        For a format-1 directory (it has no ``price_raw``) or a grid mismatch.
    """
    directory = snapshot_path(directory)
    n_dates = len(_read_json(directory / DATES_FILE))
    n_tickers = len(_read_json(directory / TICKERS_FILE))
    table = pq.read_table(directory / PRICES_FILE, columns=None)
    if "price_raw" not in table.column_names:
        raise StoreError(f"{directory}: prices.parquet has no price_raw column (format 1); rebuild")
    if table.num_rows != n_dates * n_tickers:
        raise StoreError(f"{directory}: prices.parquet has {table.num_rows} rows, expected {n_dates * n_tickers}")
    raw = _float_column(table, "price_raw").reshape(n_dates, n_tickers)
    used = _float_column(table, "price").reshape(n_dates, n_tickers)
    return raw, used


def read_issues_by_date(directory: str | Path) -> tuple[list[date], list[IssueCounts]]:
    """``(dates, counts)`` from ``issues_by_date.parquet``: one :class:`IssueCounts` per file date."""
    table = pq.read_table(snapshot_path(directory) / ISSUES_BY_DATE_FILE)
    days = table.column("date").to_pylist()
    names = count_fields()
    columns = {}
    for k in range(len(names)):
        columns[names[k]] = table.column(names[k]).to_numpy()
    counts = []
    for t in range(len(days)):
        row = IssueCounts()
        for k in range(len(names)):
            setattr(row, names[k], int(columns[names[k]][t]))
        counts.append(row)
    return days, counts


def read_anomalies(directory: str | Path) -> list[dict[str, Any]]:
    """The anomaly records of a format-2 snapshot directory."""
    document = _read_json(snapshot_path(directory) / ANOMALIES_FILE)
    return list(document["records"]) if isinstance(document, dict) else list(document)


def read_inputs(directory: str | Path) -> dict[str, Any]:
    """The ``inputs.json`` document of a format-2 snapshot directory."""
    return _read_json(snapshot_path(directory) / INPUTS_FILE)


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """The parsed ``manifest.json`` of a snapshot directory."""
    return _read_json(snapshot_path(directory) / MANIFEST_FILE)


def verify_manifest(data_dir: str | Path) -> list[str]:
    """Recompute artifact checksums (of a snapshot dir or a root's CURRENT one). Returns the problems (empty if all match)."""
    data_dir = snapshot_path(data_dir)
    manifest = _read_json(data_dir / MANIFEST_FILE)
    listed = manifest.get("artifacts", {})
    problems = []
    names = list(listed)
    for k in range(len(names)):
        path = data_dir / names[k]
        if not path.is_file():
            problems.append(f"{names[k]}: missing")
        elif sha256_file(path) != listed[names[k]]["sha256"]:
            problems.append(f"{names[k]}: checksum mismatch")
    required = V1_ARTIFACT_FILES if manifest.get("format_version") == LEGACY_FORMAT_VERSION else ARTIFACT_FILES
    for k in range(len(required)):
        if required[k] not in listed:
            problems.append(f"{required[k]}: not listed in manifest")
    return problems


def _iso_list(dates: Sequence[date]) -> list[str]:
    out = []
    for t in range(len(dates)):
        out.append(dates[t].isoformat())
    return out


def _iso_where(dates: Sequence[date], mask: np.ndarray) -> list[str]:
    out = []
    idx = np.flatnonzero(mask)
    for k in range(len(idx)):
        out.append(dates[int(idx[k])].isoformat())
    return out


def _float_column(table: pa.Table, name: str) -> np.ndarray:
    """Parquet float column -> float64 numpy with NaN for nulls."""
    return table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)


def read_json(path: str | Path) -> Any:
    """Parse a UTF-8 JSON file."""
    return _read_json(Path(path))


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
