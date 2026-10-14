"""Ingest new dates and corrections into a versioned store root, as a new snapshot.

Usage::

    python -m corrlib.ingest --root data/processed 20240909.csv [20240910.csv ...] \\
        [--overrides overrides.json] [--reason "vendor correction"] [--full]

Every ``FILE`` must be named ``YYYYMMDD.csv`` (the name is the date) with the
header ``Ticker,Date,Price``; it is parsed and cleaned exactly like a zip
member (:func:`corrlib.io.parse_daily_csv`, :func:`corrlib.clean.clean_frame`).

One invocation, holding the root's single-writer lock:

1. **Parent** = the CURRENT snapshot: its raw price panel (``prices.parquet``
   ``price_raw``; the zip is not re-read), axes, ``issues_by_date``,
   ``inputs.json``, aggregates and overrides.
2. **Identity** of each file = the sha256 of its bytes, with the date from its
   name. Same date and same sha256 as the parent's source of that date
   (``inputs.date_sources``, which covers the base zip members too) ->
   ``duplicate`` (no-op). Same date, other bytes -> ``correction`` (the file
   replaces the whole day: tickers absent from it become missing). New date ->
   ``append`` (after the last date) or ``insert`` (before it).
3. **Validation**: bad name, unreadable/unparseable file, wrong header, no data
   rows, no valid price, or two different files for one date -> ``rejected``.
   The invocation is all-or-nothing: one rejected file publishes nothing.
4. Nothing effective (only duplicates, overrides unchanged in effect) -> logged,
   exit 0, no new snapshot.
5. Otherwise the new raw panel is the parent's with the files applied (the
   universe is the sorted union: a new ticker has an all-missing history),
   the uniform-day rule is re-run on the **whole** panel (a uniform last date
   can be resolved now; the change is recorded), returns are recomputed and
   **diffed against the parent by date label** (NaN-aware). Aggregates
   (``n_valid``, mean/std, histogram) are recomputed only for window-end
   dates whose window ``[t-W+1, t]`` contains a changed return row; every
   other date's aggregates are copied from the parent, where they were
   computed from identical returns. ``--full`` recomputes every date (the
   result is the same; tests use it to prove that).
6. The new snapshot (kind ``update``) records ``changes = {parent, reason,
   inputs, dates_added, returns_changed_dates, recomputed_window_dates,
   anomaly_changes, tickers_added, ...}``; each ingested file is copied to
   ``overlays/<YYYYMMDD>-<sha12>.csv`` (content-addressed, idempotent); the
   snapshot is verified and published by swapping ``CURRENT``
   (:func:`corrlib.snapshots.publish`).

Every attempt is appended to ``ingest_log.jsonl`` with its status
(``published``, ``duplicate``, ``rejected`` or ``failed``) and reason. A
failure leaves CURRENT untouched and removes the staging directory, so the
same command can simply be re-run (files already published are then
duplicates). Overrides persist: without ``--overrides`` the parent's
overrides are re-applied; pass ``{"overrides": []}`` to drop them.

Exit codes: 0 published or nothing to do, 1 failed, 2 rejected input or bad
usage, 3 the root is locked by another writer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__, snapshots, store
from .build import Overrides, daily_aggregates, read_overrides, stage
from .clean import IssueCounts, clean_frame
from .config import (
    DEFAULT_PROCESSED_DIR,
    HIST_BLOCK_ROWS,
    OVERLAYS_DIR,
    OVERRIDES_FILE,
    VALIDITY_EPS,
)
from .io import SourceFormatError, parse_daily_csv, parse_file_date
from .returns import OverrideError, derive_returns, diff_anomalies

log = logging.getLogger("corrlib.ingest")

KIND_APPEND = "append"
KIND_INSERT = "insert"
KIND_CORRECTION = "correction"
KIND_DUPLICATE = "duplicate"
KIND_REJECTED = "rejected"

#: Kinds that change the data.
EFFECTIVE_KINDS: tuple[str, ...] = (KIND_APPEND, KIND_INSERT, KIND_CORRECTION)


class IngestRejected(ValueError):
    """At least one input failed validation; nothing was published. ``files`` has the reasons."""

    def __init__(self, message: str, files: list[IngestFile]) -> None:
        super().__init__(message)
        self.files = files


@dataclass
class IngestFile:
    """One input file: identity, classification and (for effective kinds) its cleaned rows.

    Attributes
    ----------
    path:
        The file as given.
    date:
        Date from the file name (None if the name is invalid).
    sha256, size_bytes:
        Identity of the file's bytes (None / 0 if unreadable).
    kind:
        ``append`` | ``insert`` | ``correction`` | ``duplicate`` | ``rejected``.
    reason:
        Why a file is a duplicate or was rejected.
    frame:
        Cleaned frame (``date, ticker, price``) for effective kinds.
    counts:
        Its :class:`IssueCounts` (one ``issues_by_date`` row).
    previous_source:
        The parent's ``date_sources`` entry for this date (duplicate / correction).
    """

    path: Path
    date: date | None
    sha256: str | None = None
    size_bytes: int = 0
    kind: str = KIND_REJECTED
    reason: str | None = None
    data: bytes = field(default=b"", repr=False)
    frame: pd.DataFrame | None = field(default=None, repr=False)
    counts: IssueCounts | None = None
    previous_source: dict[str, str] | None = None

    @property
    def iso(self) -> str | None:
        return self.date.isoformat() if self.date is not None else None

    def overlay_file(self) -> str:
        """Root-relative path of this file's content-addressed copy: ``overlays/<YYYYMMDD>-<sha12>.csv``."""
        assert self.date is not None and self.sha256 is not None
        return f"{OVERLAYS_DIR}/{self.date.strftime('%Y%m%d')}-{self.sha256[:12]}.csv"

    def log_entry(self) -> dict[str, Any]:
        """``{name, date, sha256, kind}`` (+ ``reason`` when set) for the log and the manifest."""
        entry: dict[str, Any] = {"name": self.path.name, "date": self.iso, "sha256": self.sha256, "kind": self.kind}
        if self.reason:
            entry["reason"] = self.reason
        return entry


@dataclass
class IngestResult:
    """Outcome of :func:`run_ingest`.

    ``status`` is ``"published"`` (``snapshot`` is the new CURRENT id) or
    ``"duplicate"`` (nothing effective; ``snapshot`` is the unchanged CURRENT).
    """

    status: str
    snapshot: str
    parent: str
    files: list[IngestFile]
    record: dict[str, Any]
    manifest: dict[str, Any] | None = None
    changes: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)


# --- inspection --------------------------------------------------------------------------


def inspect_file(path: str | Path, calendar: dict[str, int], last: date, date_sources: dict[str, Any]) -> IngestFile:
    """Identify, validate and classify one file against the parent.

    Parameters
    ----------
    calendar:
        ISO date -> index of the parent's dates.
    last:
        The parent's last date (``append`` vs ``insert``).
    date_sources:
        The parent's ``inputs.date_sources``.
    """
    path = Path(path)
    day = parse_file_date(path.name)
    item = IngestFile(path=path, date=day)
    if day is None:
        item.reason = "file name must be YYYYMMDD.csv with a real calendar date"
        return item
    try:
        item.data = path.read_bytes()
    except OSError as exc:
        item.reason = f"cannot read the file ({exc.strerror or exc})"
        return item
    item.sha256 = hashlib.sha256(item.data).hexdigest()
    item.size_bytes = len(item.data)
    iso = day.isoformat()
    if iso in calendar:
        item.previous_source = dict(date_sources.get(iso, {})) or None
        if item.previous_source is not None and item.previous_source.get("sha256") == item.sha256:
            item.kind = KIND_DUPLICATE
            item.reason = f"identical to the published source of {iso} ({item.previous_source.get('file')})"
            return item
    try:
        raw = parse_daily_csv(item.data, path.name)
    except SourceFormatError as exc:
        item.reason = str(exc)
        return item
    if len(raw) == 0:
        item.reason = f"{path.name}: no data rows"
        return item
    cleaned, counts = clean_frame(day, raw)
    if not np.isfinite(cleaned["price"].to_numpy(dtype=np.float64)).any():
        item.reason = f"{path.name}: no valid price (every price is missing or invalid)"
        return item
    item.frame, item.counts = cleaned, counts
    if iso in calendar:
        item.kind = KIND_CORRECTION
    else:
        item.kind = KIND_APPEND if day > last else KIND_INSERT
    return item


def inspect_files(
    paths: list[str | Path], dates: list[date], date_sources: dict[str, Any]
) -> list[IngestFile]:
    """:func:`inspect_file` for every path, then reject conflicting files for one date.

    Two files for the same date with identical bytes: the second is a
    duplicate. With different bytes: both are rejected (ambiguous).
    """
    calendar: dict[str, int] = {}
    for t in range(len(dates)):
        calendar[dates[t].isoformat()] = t
    items = []
    for k in range(len(paths)):
        items.append(inspect_file(paths[k], calendar, dates[-1], date_sources))
    first_by_date: dict[str, int] = {}
    for k in range(len(items)):
        iso = items[k].iso
        if iso is None or items[k].sha256 is None:
            continue
        if iso not in first_by_date:
            first_by_date[iso] = k
            continue
        first = items[first_by_date[iso]]
        if first.sha256 == items[k].sha256:
            if items[k].kind != KIND_REJECTED:
                items[k].kind = KIND_DUPLICATE
                items[k].reason = f"same file as {first.path.name} in this invocation"
        else:
            message = f"two different files for {iso} in one invocation ({first.path.name}, {items[k].path.name})"
            first.kind, first.reason = KIND_REJECTED, message
            items[k].kind, items[k].reason = KIND_REJECTED, message
    return items


# --- panel assembly and diff -------------------------------------------------------------------


def _positions(labels: list[Any]) -> dict[Any, int]:
    out: dict[Any, int] = {}
    for k in range(len(labels)):
        out[labels[k]] = k
    return out


def _index_map(labels: list[Any], position: dict[Any, int]) -> np.ndarray:
    """int64 ``[len(labels)]``: the new position of each old label."""
    out = np.empty(len(labels), dtype=np.int64)
    for k in range(len(labels)):
        out[k] = position[labels[k]]
    return out


@dataclass
class Panel:
    """The new raw panel and how it maps onto the parent's axes.

    ``rows_old[t]`` / ``cols_old[j]`` are the new positions of the parent's
    date ``t`` / ticker ``j``. ``raw`` is float64 ``[T' x N']`` (NaN = missing).
    """

    dates: list[date]
    tickers: list[str]
    raw: np.ndarray
    rows_old: np.ndarray
    cols_old: np.ndarray
    issues: list[IssueCounts]


def assemble_panel(
    parent_raw: np.ndarray,
    parent_dates: list[date],
    parent_tickers: list[str],
    parent_issues: list[IssueCounts],
    files: list[IngestFile],
) -> Panel:
    """Apply the effective files to the parent's raw panel (never forward-fills).

    Dates: sorted union. Tickers: sorted union (the universe only grows; a new
    ticker is NaN on every other date). A file replaces its whole date row: a
    ticker absent from it is NaN that day. ``issues`` gets the file's counts
    on its date and keeps the parent's elsewhere.
    """
    day_set = set(parent_dates)
    ticker_set = set(parent_tickers)
    applied: list[tuple[date, pd.DataFrame, IssueCounts]] = []
    for k in range(len(files)):
        day, frame, counts = files[k].date, files[k].frame, files[k].counts
        if day is None or frame is None or counts is None:
            raise ValueError(f"{files[k].path.name}: not a validated file (kind {files[k].kind})")
        applied.append((day, frame, counts))
        day_set.add(day)
        ticker_set.update(frame["ticker"].tolist())
    dates = sorted(day_set)
    tickers = sorted(ticker_set)
    date_pos, ticker_pos = _positions(dates), _positions(tickers)
    rows_old = _index_map(parent_dates, date_pos)
    cols_old = _index_map(parent_tickers, ticker_pos)

    raw = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
    raw[np.ix_(rows_old, cols_old)] = parent_raw
    issues: list[IssueCounts] = []
    for t in range(len(dates)):
        issues.append(IssueCounts())
    for t in range(len(parent_dates)):
        issues[int(rows_old[t])] = parent_issues[t]
    for k in range(len(applied)):
        day, frame, counts = applied[k]
        t = date_pos[day]
        cols = frame["ticker"].map(ticker_pos).to_numpy(dtype=np.int64)
        raw[t, :] = np.nan
        raw[t, cols] = frame["price"].to_numpy(dtype=np.float64)
        issues[t] = counts
    return Panel(dates=dates, tickers=tickers, raw=raw, rows_old=rows_old, cols_old=cols_old, issues=issues)


def changed_return_rows(returns: np.ndarray, parent_returns: np.ndarray, panel: Panel) -> np.ndarray:
    """bool ``[T']``: return rows that differ from the parent's row of the same date label.

    The parent's rows and columns are aligned onto the new axes (a new ticker
    is NaN in the parent), then compared NaN-aware (NaN == NaN). Dates the
    parent did not have are always changed.
    """
    aligned = np.full(returns.shape, np.nan, dtype=returns.dtype)
    aligned[np.ix_(panel.rows_old, panel.cols_old)] = parent_returns
    same = (returns == aligned) | (np.isnan(returns) & np.isnan(aligned))
    changed = ~same.all(axis=1)
    changed[np.setdiff1d(np.arange(len(panel.dates)), panel.rows_old)] = True
    return changed


@dataclass(frozen=True)
class RecomputePlan:
    """Which window-end dates to recompute and which to copy from the parent.

    All masks are bool ``[T']``; ``parent_index[t]`` is the parent's index of
    date ``t`` (-1 for a new date).
    """

    affected: np.ndarray
    recompute: np.ndarray
    copy: np.ndarray
    parent_index: np.ndarray


def plan_recompute(changed: np.ndarray, panel: Panel, window: int, full: bool = False) -> RecomputePlan:
    """Window ends ``t >= W`` whose window ``[t-W+1, t]`` holds a changed row are recomputed.

    The rest are copied from the parent, which computed them from identical
    return rows: a date is copyable only if the parent had it at an index
    ``>= W`` too (otherwise the parent stored no window for it). ``full``
    recomputes every ``t >= W``. Vectorised with a cumulative sum.
    """
    n_dates = changed.size
    index = np.arange(n_dates)
    parent_index = np.full(n_dates, -1, dtype=np.int64)
    parent_index[panel.rows_old] = np.arange(panel.rows_old.size)
    counts = np.concatenate(([0], np.cumsum(changed.astype(np.int64))))
    start = np.maximum(index - window + 1, 0)
    has_window = index >= window
    affected = has_window & ((counts[index + 1] - counts[start]) > 0)
    copy_mask = has_window & ~affected & (parent_index >= window)
    recompute = has_window if full else has_window & ~copy_mask
    if full:
        copy_mask = np.zeros(n_dates, dtype=bool)
    return RecomputePlan(affected=affected, recompute=recompute, copy=copy_mask, parent_index=parent_index)


def date_ranges(mask: np.ndarray, dates: list[date]) -> dict[str, Any]:
    """``{count, ranges: [[first ISO, last ISO], ...]}`` of the contiguous runs of ``mask``."""
    idx = np.flatnonzero(mask)
    ranges: list[list[str]] = []
    if idx.size:
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[breaks + 1]))
        ends = np.concatenate((idx[breaks], [idx[-1]]))
        for k in range(starts.size):
            ranges.append([dates[int(starts[k])].isoformat(), dates[int(ends[k])].isoformat()])
    return {"count": int(idx.size), "ranges": ranges}


def _iso_where(mask: np.ndarray, dates: list[date]) -> list[str]:
    idx = np.flatnonzero(mask)
    out = []
    for k in range(idx.size):
        out.append(dates[int(idx[k])].isoformat())
    return out


# --- overlays and overrides ----------------------------------------------------------------


def copy_overlay(root: Path, item: IngestFile) -> Path:
    """Store the file's bytes at ``<root>/overlays/<YYYYMMDD>-<sha12>.csv`` (idempotent, atomic)."""
    target = root / item.overlay_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == item.sha256:
        return target
    tmp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(item.data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def effective_overrides(
    parent_dir: Path, parent_inputs: dict[str, Any], path: str | Path | None
) -> tuple[Overrides | None, bool]:
    """The overrides to apply and whether they differ (by sha256) from the parent's.

    With ``path``: that file. Without: the parent's ``overrides.json`` (so
    confirmed classifications persist), or None.

    Raises
    ------
    OverrideError
        If the file is not a valid overrides document.
    """
    previous = parent_inputs.get("overrides_sha256")
    if path is not None:
        overrides = read_overrides(path)
        return overrides, overrides.sha256 != previous
    if (parent_dir / OVERRIDES_FILE).is_file():
        return read_overrides(parent_dir / OVERRIDES_FILE), False
    return None, False


# --- the ingest --------------------------------------------------------------------------------


def run_ingest(
    root: str | Path,
    files: list[str | Path],
    overrides: str | Path | None = None,
    reason: str | None = None,
    full: bool = False,
    block: int | None = None,
) -> IngestResult:
    """Ingest ``files`` (and/or an overrides change) into the store root as one new snapshot.

    See the module docstring for the protocol. ``block`` defaults to the
    parent's ``hist_block_rows`` so recomputed dates match a full rebuild.

    Raises
    ------
    IngestRejected
        If any input failed validation (logged as ``rejected``; nothing published).
    corrlib.snapshots.LockError
        If another writer holds the lock (logged as ``failed``).
    Exception
        Anything else (logged as ``failed``; CURRENT untouched, staging removed).
    """
    root = Path(root)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    attempt = snapshots.new_attempt(root, "ingest")
    attempt["update_reason"] = reason
    attempt["mode"] = "full" if full else "incremental"
    if overrides is not None:
        attempt["overrides_file"] = str(overrides)
    items: list[IngestFile] = []
    try:
        with snapshots.locked(root, "ingest") as lock:
            result = _ingest_locked(root, lock, files, overrides, reason, full, block, attempt, items, timings)
    except IngestRejected as exc:
        attempt.update(status="rejected", reason=str(exc), files=_log_entries(exc.files or items))
        snapshots.append_log(root, attempt)
        raise
    except BaseException as exc:
        attempt.update(status="failed", reason=snapshots.describe_error(exc), files=_log_entries(items))
        snapshots.append_log(root, attempt)
        raise
    timings["total"] = time.perf_counter() - started
    attempt["duration_seconds"] = round(timings["total"], 3)
    result.timings = timings
    result.record = snapshots.append_log(root, attempt)
    return result


def _log_entries(items: list[IngestFile]) -> list[dict[str, Any]]:
    out = []
    for k in range(len(items)):
        out.append(items[k].log_entry())
    return out


def _ingest_locked(
    root: Path,
    lock: snapshots.WriteLock,
    paths: list[str | Path],
    overrides_path: str | Path | None,
    reason: str | None,
    full: bool,
    block: int | None,
    attempt: dict[str, Any],
    items: list[IngestFile],
    timings: dict[str, float],
) -> IngestResult:
    """The body of :func:`run_ingest` (the lock is held; ``attempt`` is filled in)."""
    parent_id = snapshots.current_id(root)
    if parent_id is None:
        raise snapshots.SnapshotError(
            f"{root}: no published snapshot to ingest into; run `python -m corrlib.build --out {root}` first"
        )
    attempt["parent"] = parent_id
    parent_dir = snapshots.snapshots_path(root) / parent_id

    with stage("read parent snapshot", timings):
        problems = store.verify_manifest(parent_dir)
        if problems:
            raise snapshots.SnapshotError(f"parent snapshot {parent_id} fails verification: {'; '.join(problems)}")
        parent = store.load_store(parent_dir)
        parent_inputs = store.read_inputs(parent_dir)
        parent_dates = []
        for t in range(parent.n_dates):
            parent_dates.append(date.fromisoformat(parent.dates[t]))
        window = parent.window
        eps = float(parent.manifest.get("validity_eps", VALIDITY_EPS))
        if block is None:
            block = int(parent.manifest.get("hist_block_rows", HIST_BLOCK_ROWS))

    with stage("inspect files", timings):
        items.extend(inspect_files(list(paths), parent_dates, parent_inputs.get("date_sources", {})))
        attempt["files"] = _log_entries(items)
        rejected = []
        for k in range(len(items)):
            if items[k].kind == KIND_REJECTED:
                rejected.append(f"{items[k].path.name}: {items[k].reason}")
        if rejected:
            raise IngestRejected("rejected input (nothing published): " + " | ".join(rejected), items)
        try:
            chosen, overrides_changed = effective_overrides(parent_dir, parent_inputs, overrides_path)
        except OverrideError as exc:
            raise IngestRejected(f"rejected overrides (nothing published): {exc}", items) from exc
        effective: list[IngestFile] = []
        for k in range(len(items)):
            if items[k].kind in EFFECTIVE_KINDS:
                effective.append(items[k])

    if not effective and not overrides_changed:
        return _no_op(attempt, parent_id, items, "every file is already published (identical sha256)" if items else "nothing to ingest")

    with stage("assemble panel + returns", timings):
        parent_raw, _ = store.read_prices(parent_dir)
        parent_issue_dates, parent_issues = store.read_issues_by_date(parent_dir)
        if list(parent_issue_dates) != parent_dates:
            raise snapshots.SnapshotError(f"parent snapshot {parent_id}: issues_by_date does not match its dates")
        panel = assemble_panel(parent_raw, parent_dates, parent.tickers, parent_issues, effective)
        del parent_raw
        try:
            derived = derive_returns(panel.raw, panel.dates, window, chosen.entries if chosen else None)
        except OverrideError as exc:
            raise IngestRejected(f"rejected overrides for the new data (nothing published): {exc}", items) from exc
        returns32 = derived.returns.astype(np.float32)

    if not effective and derived.records == parent.anomalies:
        return _no_op(attempt, parent_id, items, "the overrides do not change any anomaly record")

    with stage("diff returns + plan", timings):
        changed = changed_return_rows(returns32, np.asarray(parent.returns), panel)
        plan = plan_recompute(changed, panel, window, full=full)
        log.info(
            "%d return row(s) changed; %d window date(s) affected; recomputing %d, copying %d",
            int(changed.sum()), int(plan.affected.sum()), int(plan.recompute.sum()), int(plan.copy.sum()),
        )

    n_recompute = int(plan.recompute.sum())
    with stage(f"aggregates ({n_recompute} dates)", timings):
        n_valid, mean, std, hist = daily_aggregates(
            returns32, window, parent.hist_edges, block, indices=np.flatnonzero(plan.recompute), eps=eps
        )
        source = plan.parent_index[plan.copy]
        n_valid[plan.copy] = parent.n_valid[source]
        mean[plan.copy] = parent.mean_corr[source]
        std[plan.copy] = parent.std_corr[source]
        hist[plan.copy] = parent.hist[source]

    changes = _changes(parent_id, reason, full, items, panel, parent, changed, plan, derived.records, chosen, overrides_changed)
    inputs = _updated_inputs(parent_inputs, panel.dates, effective, chosen, parent_id)
    content = store.SnapshotContent(
        dates=panel.dates,
        tickers=panel.tickers,
        prices_raw=panel.raw,
        prices=derived.prices,
        returns=returns32,
        n_valid=n_valid,
        mean_corr=mean,
        std_corr=std,
        hist=hist,
        hist_edges=np.asarray(parent.hist_edges),
        anomalies=derived.records,
        issues_by_date=panel.issues,
        inputs=inputs,
        overrides_data=chosen.data if chosen is not None else None,
        window=window,
        eps=eps,
    )
    extra = {
        "corrlib_version": __version__,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": parent.dataset,
        "source": parent.manifest.get("source"),
        "hist_block_rows": block,
        "changes": changes,
        "timings_seconds": _rounded(timings),
    }
    with stage("copy overlays + write + verify + publish", timings):
        for k in range(len(effective)):
            copy_overlay(root, effective[k])
        manifest = snapshots.publish(root, content, extra, kind="update", parent=parent_id, lock=lock)

    snapshot_id = manifest["snapshot"]["id"]
    attempt.update(
        status="published",
        snapshot=snapshot_id,
        summary={
            "dates_added": len(changes["dates_added"]),
            "dates_corrected": len(changes["dates_corrected"]),
            "returns_changed_dates": len(changes["returns_changed_dates"]),
            "recomputed_window_dates": changes["recomputed_window_dates"]["count"],
            "anomaly_changes": len(changes["anomaly_changes"]),
            "tickers_added": len(changes["tickers_added"]),
        },
    )
    attempt["reason"] = reason or "update"
    return IngestResult(
        status="published",
        snapshot=snapshot_id,
        parent=parent_id,
        files=items,
        record=attempt,
        manifest=manifest,
        changes=changes,
    )


def _no_op(attempt: dict[str, Any], parent_id: str, items: list[IngestFile], why: str) -> IngestResult:
    attempt.update(status="duplicate", reason=why, snapshot=parent_id)
    log.info("nothing to publish: %s (CURRENT stays %s)", why, parent_id)
    return IngestResult(status="duplicate", snapshot=parent_id, parent=parent_id, files=items, record=attempt)


def _changes(
    parent_id: str,
    reason: str | None,
    full: bool,
    items: list[IngestFile],
    panel: Panel,
    parent: store.Store,
    changed: np.ndarray,
    plan: RecomputePlan,
    records: list[dict[str, Any]],
    chosen: Overrides | None,
    overrides_changed: bool,
) -> dict[str, Any]:
    """The manifest's ``changes`` block of an update."""
    inputs = []
    added: list[str] = []
    corrected: list[str] = []
    for k in range(len(items)):
        item = items[k]
        source: str | None = None
        if item.kind in EFFECTIVE_KINDS:
            source = item.overlay_file()
        elif item.previous_source is not None:
            source = item.previous_source.get("file")
        inputs.append(
            {"date": item.iso, "kind": item.kind, "sha256": item.sha256, "file": source, "original_name": item.path.name}
        )
        if item.kind in (KIND_APPEND, KIND_INSERT):
            added.append(str(item.iso))
        elif item.kind == KIND_CORRECTION:
            corrected.append(str(item.iso))
    tickers_added = sorted(set(panel.tickers) - set(parent.tickers))
    return {
        "parent": parent_id,
        "reason": reason,
        "mode": "full" if full else "incremental",
        "inputs": inputs,
        "dates_added": sorted(added),
        "dates_corrected": sorted(corrected),
        "returns_changed_dates": _iso_where(changed, panel.dates),
        "affected_window_dates": date_ranges(plan.affected, panel.dates),
        "recomputed_window_dates": date_ranges(plan.recompute, panel.dates),
        "copied_window_dates": int(plan.copy.sum()),
        "anomaly_changes": diff_anomalies(parent.anomalies, records),
        "tickers_added": tickers_added,
        "overrides": {
            "changed": overrides_changed,
            "sha256": chosen.sha256 if chosen is not None else None,
            "count": len(chosen.entries) if chosen is not None else 0,
        },
    }


def _updated_inputs(
    parent_inputs: dict[str, Any],
    dates: list[date],
    effective: list[IngestFile],
    chosen: Overrides | None,
    parent_id: str,
) -> dict[str, Any]:
    """The parent's ``inputs.json`` with the overlay sources applied (date order kept)."""
    inputs = copy.deepcopy(parent_inputs)
    previous = inputs.get("date_sources", {})
    overlays = list(inputs.get("overlays", []))
    for k in range(len(effective)):
        item = effective[k]
        previous[item.iso] = {"source": "overlay", "sha256": item.sha256, "file": item.overlay_file()}
        overlays.append(
            {
                "date": item.iso,
                "kind": item.kind,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
                "file": item.overlay_file(),
                "original_name": item.path.name,
                "parent": parent_id,
            }
        )
    sources: dict[str, Any] = {}
    for t in range(len(dates)):
        sources[dates[t].isoformat()] = previous[dates[t].isoformat()]
    inputs["date_sources"] = sources
    inputs["overlays"] = overlays
    inputs["overrides_sha256"] = chosen.sha256 if chosen is not None else None
    return inputs


def _rounded(timings: dict[str, float]) -> dict[str, float]:
    out = {}
    names = list(timings)
    for k in range(len(names)):
        out[names[k]] = round(timings[names[k]], 3)
    return out


# --- CLI ------------------------------------------------------------------------------------


def _print_result(result: IngestResult) -> None:
    print()
    for k in range(len(result.files)):
        item = result.files[k]
        print(f"  {item.kind:<10} {item.path.name:<14} {item.iso or '-':<10}  sha256 {str(item.sha256)[:12]}"
              + (f"  ({item.reason})" if item.reason and item.kind == KIND_DUPLICATE else ""))
    if result.status != "published":
        print(f"Nothing published: {result.record.get('reason')}. CURRENT stays {result.snapshot}.")
        return
    changes = result.changes or {}
    recomputed = changes["recomputed_window_dates"]
    print(f"Published {result.snapshot} (update, parent {result.parent}) -> now CURRENT")
    print(f"  dates added        {', '.join(changes['dates_added']) or '-'}")
    print(f"  dates corrected    {', '.join(changes['dates_corrected']) or '-'}")
    print(f"  tickers added      {len(changes['tickers_added'])} {changes['tickers_added'][:10]}")
    print(f"  return rows changed {len(changes['returns_changed_dates'])}")
    ranges = []
    for k in range(len(recomputed["ranges"])):
        ranges.append(f"{recomputed['ranges'][k][0]}..{recomputed['ranges'][k][1]}")
    print(f"  windows recomputed {recomputed['count']} ({changes['mode']}) {', '.join(ranges)}")
    print(f"  windows copied     {changes['copied_window_dates']}")
    anomaly_changes = changes["anomaly_changes"]
    for k in range(len(anomaly_changes)):
        c = anomaly_changes[k]
        print(f"  anomaly change     {c['date']}: {c['from']} -> {c['to']}")
    names = list(result.timings)
    for k in range(len(names)):
        print(f"  [{result.timings[names[k]]:7.3f}s] {names[k]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corrlib.ingest", description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_PROCESSED_DIR, help="store root (default: %(default)s)")
    parser.add_argument("files", nargs="*", type=Path, help="daily files named YYYYMMDD.csv (header Ticker,Date,Price)")
    parser.add_argument("--overrides", type=Path, default=None, help="overrides JSON to apply from now on")
    parser.add_argument("--reason", default=None, help="why this update is made (recorded in the snapshot and the log)")
    parser.add_argument("--full", action="store_true", help="recompute the aggregates of every date (default: only affected windows)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.files and args.overrides is None:
        parser.error("give at least one FILE.csv or --overrides")
    try:
        result = run_ingest(args.root, args.files, overrides=args.overrides, reason=args.reason, full=args.full)
    except IngestRejected as exc:
        log.error("%s", exc)
        return 2
    except snapshots.LockError as exc:
        log.error("%s", exc)
        return 3
    except (snapshots.SnapshotError, store.StoreError) as exc:
        log.error("ingest failed (CURRENT unchanged): %s", exc)
        return 1 if store_has_current(args.root) else 2
    _print_result(result)
    return 0


def store_has_current(root: Path) -> bool:
    """Whether ``root`` has a published snapshot (distinguishes a failure from bad usage)."""
    try:
        return snapshots.current_id(root) is not None
    except snapshots.SnapshotError:
        return False


if __name__ == "__main__":
    sys.exit(main())
