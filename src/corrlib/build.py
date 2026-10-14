"""End-to-end build: zip -> cleaned panel -> returns -> per-date aggregates -> a published snapshot.

Usage::

    python -m corrlib.build --zip data/raw/stock_data.zip --out data/processed \\
        [--window 20] [--overrides overrides.json] [--demo]

``--out`` is a **store root** (see :mod:`corrlib.snapshots`). Every build
creates a new immutable snapshot (kind ``full_build``, parent = the CURRENT
snapshot if any) from the zip alone and publishes it by swapping ``CURRENT``;
earlier snapshots stay available for rollback. The build holds the root's
single-writer lock and records its attempt in ``ingest_log.jsonl``.

Stages (each logged with its wall time):

1. hash the source zip (provenance)
2. ingest + clean every daily file (streamed from the zip; sha256 of each member)
3. wide raw price matrix, uniform-day records (+ overrides), used prices, returns
4. per-date ``n_valid``, mean/std of pairwise rho and the rho histogram
5. write the artifacts (format 2) to a staging directory, verify, publish

The pieces are reusable: :func:`prepare_build` returns the in-memory
:class:`corrlib.store.SnapshotContent` plus the manifest entries that are not
derivable from it, :func:`daily_aggregates` computes the aggregates of any
subset of dates (used by :mod:`corrlib.ingest`), and
:func:`corrlib.snapshots.publish` writes, verifies and publishes a snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, snapshots, store
from .clean import IssueCounts, clean_frame
from .config import (
    CURRENT_FILE,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_ZIP,
    DEMO_LABEL,
    DEMO_ZIP_COMMENT,
    HIST_BINS,
    HIST_BLOCK_ROWS,
    HIST_RANGE,
    MANIFEST_FILE,
    VALIDITY_EPS,
    WINDOW,
)
from .corr import histogram_pairwise, mean_and_std_pairwise, prepare_window
from .io import iter_daily_files, scan_members, sha256_file, zip_comment
from .returns import (
    OverrideError,
    derive_returns,
    parse_overrides,
    ticker_universe,
    to_wide,
)

log = logging.getLogger("corrlib.build")


@dataclass
class Panel:
    """Everything derived from the zip before any correlation work.

    Attributes
    ----------
    dates, tickers:
        The axes (``T`` file dates, ``N`` sorted tickers: the union over files).
    prices_raw:
        float64 ``[T x N]`` after per-file cleaning only (the source evidence).
    prices:
        ``prices_raw`` with placeholder and unresolved uniform days set to NaN:
        the prices actually used.
    returns:
        float64 ``[T x N]`` simple returns of ``prices`` with re-base breaks applied.
    rebase, placeholder, unresolved:
        bool ``[T]`` final uniform-day classification (overrides applied).
    anomalies:
        One record per uniform-price day (see :func:`corrlib.returns.anomaly_record`).
    issues:
        Summed per-file :class:`IssueCounts`.
    issues_by_date:
        One :class:`IssueCounts` per file date (``T`` entries).
    date_sources:
        ``{ISO date: {"source": "base", "sha256": <member bytes>, "file": <member name>}}``.
    rows_per_file:
        Raw row count of each daily file.
    skipped_members:
        ``(name, reason)`` of zip entries that are not daily files.
    """

    dates: list[date]
    tickers: list[str]
    prices_raw: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    rebase: np.ndarray
    placeholder: np.ndarray
    unresolved: np.ndarray
    anomalies: list[dict[str, Any]]
    issues: IssueCounts
    issues_by_date: list[IssueCounts] = field(default_factory=list)
    date_sources: dict[str, dict[str, str]] = field(default_factory=dict)
    rows_per_file: list[int] = field(default_factory=list)
    skipped_members: list[tuple[str, str]] = field(default_factory=list)
    #: ``(file date, {counter: n})`` for files with any non-zero issue counter
    files_with_issues: list[tuple[date, dict[str, int]]] = field(default_factory=list)
    #: finite returns set to NaN because they end on a re-base day
    returns_nulled_at_rebase: int = 0
    #: finite prices set to NaN because the day is a placeholder
    prices_nulled_on_placeholder_days: int = 0
    #: finite prices set to NaN because the (last) day is unresolved
    prices_nulled_pending: int = 0


@dataclass(frozen=True)
class Overrides:
    """An overrides file: its exact bytes (copied into the artifacts) and validated entries."""

    path: Path
    data: bytes
    sha256: str
    entries: list[dict[str, str]]


@dataclass
class PreparedBuild:
    """A computed build, ready to be written by :func:`corrlib.store.write_snapshot`.

    Attributes
    ----------
    content:
        Every artifact's data.
    extra:
        Manifest entries not derivable from ``content`` (version, build time,
        source zip identity, dataset label, stage timings).
    panel:
        The ingested panel (for reporting).
    """

    content: store.SnapshotContent
    extra: dict[str, Any]
    panel: Panel


@contextmanager
def stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    """Log the start and wall time of a build stage."""
    log.info("%s ...", name)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings[name] = elapsed
    log.info("%s done in %.2fs", name, elapsed)


def read_overrides(path: str | Path) -> Overrides:
    """Read and validate an overrides JSON file (see :func:`corrlib.returns.parse_overrides`).

    Raises
    ------
    OverrideError
        If the file is not valid JSON or not a valid overrides document.
    """
    path = Path(path)
    data = path.read_bytes()
    try:
        document = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OverrideError(f"{path.name}: not valid JSON ({exc})") from exc
    try:
        entries = parse_overrides(document)
    except OverrideError as exc:
        raise OverrideError(f"{path.name}: {exc}") from exc
    return Overrides(path=path, data=data, sha256=hashlib.sha256(data).hexdigest(), entries=entries)


def ingest(
    zip_path: str | Path,
    timings: dict[str, float] | None = None,
    window: int = WINDOW,
    overrides: Sequence[Mapping[str, Any]] | None = None,
) -> Panel:
    """Read, clean and assemble the zip into raw prices, records, used prices and returns.

    Parameters
    ----------
    zip_path:
        Source zip of ``YYYYMMDD.csv`` daily files.
    timings:
        Stage name -> seconds, filled in place.
    window:
        Window length W (for the anomaly records' affected windows).
    overrides:
        Validated override entries (``Overrides.entries``), or None.

    Raises
    ------
    OverrideError
        If an override targets a day that is not a detected uniform-price day.
    """
    timings = {} if timings is None else timings

    with stage("ingest + clean", timings):
        scan = scan_members(zip_path)
        dates: list[date] = []
        frames: list[pd.DataFrame] = []
        rows_per_file: list[int] = []
        issues = IssueCounts()
        issues_by_date: list[IssueCounts] = []
        date_sources: dict[str, dict[str, str]] = {}
        files_with_issues: list[tuple[date, dict[str, int]]] = []
        for daily in iter_daily_files(zip_path, scan.members):
            cleaned, counts = clean_frame(daily.file_date, daily.frame)
            dates.append(daily.file_date)
            frames.append(cleaned)
            rows_per_file.append(len(daily.frame))
            issues = issues + counts
            issues_by_date.append(counts)
            date_sources[daily.file_date.isoformat()] = {
                "source": "base",
                "sha256": daily.sha256,
                "file": daily.name,
            }
            found = _nonzero_issues(counts)
            if found:
                files_with_issues.append((daily.file_date, found))
        if not dates:
            raise ValueError(f"{zip_path}: no YYYYMMDD.csv files found")
        log.info("%d files, %d rows in, %d rows out", issues.files, issues.rows_in, issues.rows_out)

    with stage("wide prices + returns", timings):
        tickers = ticker_universe(frames)
        prices_raw = to_wide(frames, tickers, dates)
        del frames
        derived = derive_returns(prices_raw, dates, window, overrides)
        for k in range(len(derived.records)):
            record = derived.records[k]
            log.info(
                "uniform-price day %s: %s (%s, action %s)",
                record["date"], record["classification"], record["status"], record["action"],
            )
        log.info("T=%d dates, N=%d tickers", len(dates), len(tickers))

    return Panel(
        dates=dates,
        tickers=tickers,
        prices_raw=prices_raw,
        prices=derived.prices,
        returns=derived.returns,
        rebase=derived.rebase,
        placeholder=derived.placeholder,
        unresolved=derived.unresolved,
        anomalies=derived.records,
        issues=issues,
        issues_by_date=issues_by_date,
        date_sources=date_sources,
        rows_per_file=rows_per_file,
        skipped_members=scan.skipped,
        files_with_issues=files_with_issues,
        returns_nulled_at_rebase=derived.returns_nulled_at_rebase,
        prices_nulled_on_placeholder_days=derived.prices_nulled_on_placeholder_days,
        prices_nulled_pending=derived.prices_nulled_pending,
    )


def _nonzero_issues(counts: IssueCounts) -> dict[str, int]:
    """Issue counters of one file that are non-zero (bookkeeping counters excluded)."""
    found: dict[str, int] = {}
    values = counts.as_dict()
    names = list(values)
    for k in range(len(names)):
        if names[k] not in ("files", "rows_in", "rows_out") and values[names[k]]:
            found[names[k]] = values[names[k]]
    return found


def hist_edges() -> np.ndarray:
    """The histogram bin edges, float64 ``[HIST_BINS + 1]`` on ``HIST_RANGE``.

    Rounded so the stored edges read as -1.0, -0.95, ... rather than linspace noise.
    """
    return np.round(np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1), 12)


def daily_aggregates(
    returns: np.ndarray,
    window: int,
    edges: np.ndarray,
    block: int = HIST_BLOCK_ROWS,
    indices: np.ndarray | None = None,
    eps: float = VALIDITY_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-date ``n_valid``, mean rho, std rho and rho histogram.

    Parameters
    ----------
    returns:
        ``[T x N]`` returns exactly as stored (float32), so these aggregates
        match what the API computes on demand.
    indices:
        Date indices to compute (others keep the "no window" values below);
        None computes every date ``t >= window``. Indices below ``window``
        are skipped (they have no window).
    eps:
        Validity threshold of the centred window norm.

    Returns
    -------
    (n_valid int32 [T], mean float64 [T], std float64 [T], hist int64 [T x B])
        mean/std are NaN and the histogram row is zero where ``n_valid < 2``
        or the date was not computed.
    """
    n_dates = returns.shape[0]
    n_valid = np.zeros(n_dates, dtype=np.int32)
    mean = np.full(n_dates, np.nan)
    std = np.full(n_dates, np.nan)
    hist = np.zeros((n_dates, edges.size - 1), dtype=np.int64)
    todo = np.arange(window, n_dates) if indices is None else np.asarray(indices, dtype=np.int64)
    todo = todo[todo >= window]
    for k in range(todo.size):
        t = int(todo[k])
        Z, valid = prepare_window(returns, t, window, eps)
        mean[t], std[t], n_valid[t] = mean_and_std_pairwise(Z, valid)
        if n_valid[t] >= 2:
            hist[t] = histogram_pairwise(Z, valid, edges, block=block)
        if (k + 1) % 100 == 0:
            log.info("  aggregates: %d/%d dates (date index %d, n_valid=%d)", k + 1, todo.size, t, n_valid[t])
    return n_valid, mean, std, hist


def detect_dataset(zip_path: str | Path, demo: bool | None = None) -> dict[str, str]:
    """``{kind, label}``: ``"demo"`` if ``demo`` is True or (None and) the zip carries the demo comment."""
    zip_path = Path(zip_path)
    if demo is None:
        demo = zip_comment(zip_path).strip() == DEMO_ZIP_COMMENT
    if demo:
        return {"kind": "demo", "label": DEMO_LABEL}
    return {"kind": "source", "label": f"Source data: {zip_path.name}"}


def source_inputs(
    zip_path: Path, zip_sha: str, panel: Panel, overrides: Overrides | None
) -> dict[str, Any]:
    """The ``inputs.json`` document of a full build: the zip, every member's sha256, overrides."""
    skipped = []
    for k in range(len(panel.skipped_members)):
        skipped.append({"name": panel.skipped_members[k][0], "reason": panel.skipped_members[k][1]})
    return {
        "base": {
            "file_name": zip_path.name,
            "sha256": zip_sha,
            "size_bytes": zip_path.stat().st_size,
            "skipped_members": skipped,
        },
        "date_sources": panel.date_sources,
        "overlays": [],
        "overrides_sha256": overrides.sha256 if overrides is not None else None,
    }


def prepare_build(
    zip_path: str | Path,
    window: int = WINDOW,
    block: int = HIST_BLOCK_ROWS,
    overrides: str | Path | Overrides | None = None,
    demo: bool | None = None,
    timings: dict[str, float] | None = None,
) -> PreparedBuild:
    """Run every compute stage of a full build from the zip (nothing is written).

    Parameters
    ----------
    zip_path:
        Source zip.
    window, block:
        Window length and histogram block rows.
    overrides:
        Path of an overrides file (or an already read :class:`Overrides`), or None.
    demo:
        Force the dataset kind (True demo / False source); None auto-detects
        the demo zip comment.
    timings:
        Stage name -> seconds, filled in place.

    Raises
    ------
    OverrideError
        If the overrides file is invalid or targets a non-uniform day.
    """
    zip_path = Path(zip_path)
    timings = {} if timings is None else timings
    started = time.perf_counter()
    if overrides is not None and not isinstance(overrides, Overrides):
        overrides = read_overrides(overrides)

    with stage("hash source zip", timings):
        zip_sha = sha256_file(zip_path)

    panel = ingest(zip_path, timings, window=window, overrides=overrides.entries if overrides else None)

    with stage("per-date aggregates", timings):
        returns32 = panel.returns.astype(np.float32)
        edges = hist_edges()
        n_valid, mean, std, hist = daily_aggregates(returns32, window, edges, block)

    content = store.SnapshotContent(
        dates=panel.dates,
        tickers=panel.tickers,
        prices_raw=panel.prices_raw,
        prices=panel.prices,
        returns=returns32,
        n_valid=n_valid,
        mean_corr=mean,
        std_corr=std,
        hist=hist,
        hist_edges=edges,
        anomalies=panel.anomalies,
        issues_by_date=panel.issues_by_date,
        inputs=source_inputs(zip_path, zip_sha, panel, overrides),
        overrides_data=overrides.data if overrides is not None else None,
        window=window,
        eps=VALIDITY_EPS,
    )
    extra = {
        "corrlib_version": __version__,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": detect_dataset(zip_path, demo),
        "source": {
            "file_name": zip_path.name,
            "sha256": zip_sha,
            "size_bytes": zip_path.stat().st_size,
        },
        # Histogram block rows: an update recomputes dates with the same blocking.
        "hist_block_rows": block,
        "changes": None,
        # Stages completed before the artifacts were written.
        "timings_seconds": _rounded(timings),
        "build_seconds_before_write": round(time.perf_counter() - started, 2),
    }
    return PreparedBuild(content=content, extra=extra, panel=panel)


def run_build(
    zip_path: str | Path,
    out_dir: str | Path,
    window: int = WINDOW,
    block: int = HIST_BLOCK_ROWS,
    overrides: str | Path | Overrides | None = None,
    demo: bool | None = None,
) -> tuple[dict, dict[str, float]]:
    """Build a new snapshot from the zip and publish it in the store root ``out_dir``.

    Holds the root's lock for the whole build (another writer gets a
    :class:`corrlib.snapshots.LockError` naming this one), creates a
    ``full_build`` snapshot whose parent is the CURRENT snapshot (if any),
    publishes it (:func:`corrlib.snapshots.publish`) and logs the attempt
    (``published`` or ``failed`` with the error) in ``ingest_log.jsonl``. On
    any failure CURRENT is untouched and no staging directory is left behind.

    Returns ``(manifest, timings)``; ``timings`` maps stage name -> seconds.
    """
    root = Path(out_dir)
    zip_path = Path(zip_path)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    attempt = snapshots.new_attempt(
        root, "build", files=[{"name": zip_path.name, "date": None, "sha256": None, "kind": "base"}]
    )
    try:
        with snapshots.locked(root, "build") as lock:
            parent = snapshots.current_id(root)
            attempt["parent"] = parent
            if (root / MANIFEST_FILE).is_file() and not (root / CURRENT_FILE).is_file():
                log.warning(
                    "%s holds legacy flat (format 1) artifacts; they are left in place for running "
                    "readers, and the new CURRENT pointer takes precedence over them",
                    root,
                )
            prepared = prepare_build(zip_path, window, block, overrides=overrides, demo=demo, timings=timings)
            attempt["files"][0]["sha256"] = prepared.extra["source"]["sha256"]
            with stage("write + verify + publish snapshot", timings):
                manifest = snapshots.publish(
                    root, prepared.content, prepared.extra, kind="full_build", parent=parent, lock=lock
                )
    except BaseException as exc:
        attempt.update(status="failed", reason=snapshots.describe_error(exc))
        snapshots.append_log(root, attempt)
        raise
    timings["total"] = time.perf_counter() - started
    attempt.update(
        status="published",
        reason=f"full build from {zip_path.name}",
        snapshot=manifest["snapshot"]["id"],
        duration_seconds=round(timings["total"], 3),
    )
    snapshots.append_log(root, attempt)
    return manifest, timings


def _rounded(timings: dict[str, float]) -> dict[str, float]:
    out = {}
    names = list(timings)
    for k in range(len(names)):
        out[names[k]] = round(timings[names[k]], 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corrlib.build", description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="source zip (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_PROCESSED_DIR, help="store root to publish the new snapshot in (default: %(default)s)")
    parser.add_argument("--window", type=int, default=WINDOW, help="window length in returns (default: %(default)s)")
    parser.add_argument("--block", type=int, default=HIST_BLOCK_ROWS, help="rows per histogram block (default: %(default)s)")
    parser.add_argument("--overrides", type=Path, default=None, help="JSON overrides of uniform-day classifications")
    parser.add_argument("--demo", action="store_true", default=None, help="label the dataset as synthetic demo data (auto-detected from the demo zip comment)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.zip.is_file():
        log.error("source zip not found: %s (pass --zip PATH)", args.zip)
        return 2
    if args.window < 2:
        log.error("--window must be >= 2")
        return 2
    if args.overrides is not None and not args.overrides.is_file():
        log.error("overrides file not found: %s", args.overrides)
        return 2

    try:
        manifest, timings = run_build(
            args.zip, args.out, window=args.window, block=args.block, overrides=args.overrides, demo=args.demo
        )
    except OverrideError as exc:
        log.error("invalid overrides: %s", exc)
        return 2
    except snapshots.LockError as exc:
        log.error("%s", exc)
        return 3
    except snapshots.SnapshotError as exc:
        log.error("publish failed (CURRENT unchanged): %s", exc)
        return 1

    summary = manifest["anomalies_summary"]
    snapshot = manifest["snapshot"]
    print()
    print(f"Build complete -> {args.out}")
    print(f"  snapshot        {snapshot['id']} (full_build, parent {snapshot['parent']}) now CURRENT")
    print(f"  dataset         {manifest['dataset']['label']} ({manifest['dataset']['kind']})")
    print(f"  source          {manifest['source']['file_name']} (sha256 {manifest['source']['sha256'][:12]}...)")
    print(f"  dates x tickers {manifest['n_dates']} x {manifest['n_tickers']} (window {manifest['window']})")
    print(f"  date range      {manifest['first_date']} .. {manifest['last_date']}")
    print(f"  first valid     {manifest['first_valid_date']}")
    print(f"  view            {manifest['view']} (uniform-day decisions may use the next day)")
    print(f"  anomalies       {summary['count']} uniform-price day(s): {summary['by_classification']}, {summary['by_status']}")
    records = store.read_anomalies(snapshots.snapshots_path(args.out) / snapshot["id"])
    for k in range(len(records)):
        r = records[k]
        print(f"    {r['date']}  {r['classification']:<11} {r['status']:<10} look-ahead={str(r['look_ahead']).lower():<5} {r['action']}")
    if manifest.get("overrides"):
        print(f"  overrides       {manifest['overrides']['count']} applied (sha256 {manifest['overrides']['sha256'][:12]}...)")
    names = list(manifest["artifacts"])
    for k in range(len(names)):
        size = manifest["artifacts"][names[k]]["size_bytes"]
        print(f"  {names[k]:<24} {size / 1e6:8.2f} MB")
    stages = list(timings)
    for k in range(len(stages)):
        print(f"  [{timings[stages[k]]:7.2f}s] {stages[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
