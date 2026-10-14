"""Versioned snapshot store: immutable snapshots, one atomic pointer, one writer at a time.

Layout of a **store root** (default ``data/processed``; the API reads ``CORR_DATA_DIR``)::

    <root>/CURRENT                     {"snapshot": "<id>", "published_at": "...Z"}: the published snapshot
    <root>/snapshots/<id>/             one immutable artifact directory per snapshot (see corrlib.store)
    <root>/snapshots/.staging-<uuid>/  a build in progress (ignored by readers; removed on failure / by prune)
    <root>/overlays/<YYYYMMDD>-<sha12>.csv   content-addressed copy of every ingested daily file
    <root>/ingest_log.jsonl            append-only audit log: one JSON object per write attempt
    <root>/.lock                       single-writer lock {pid, host, started_at, operation, token}

Snapshot ids are ``v{seq:04d}-{h12}``: ``seq`` is one more than the highest
sequence number in the root (so ids sort in creation order, also after a
rollback), ``h12`` the first 12 hex digits of the sha256 over the sorted
``"<artifact> <sha256>"`` lines of the snapshot's artifacts (the id is derived
from the content).

**Publication protocol** (:func:`publish`), always under the lock:

1. write every artifact and the manifest into ``snapshots/.staging-<uuid>/``
   and fsync them;
2. verify the staged directory (:func:`verify_snapshot`: checksums, shapes,
   axes, per-date invariants such as ``sum(hist[t]) == n_valid (n_valid - 1) / 2``);
3. ``os.rename`` it to ``snapshots/<id>`` (same filesystem: the directory is
   complete before it has its final name);
4. replace ``CURRENT`` atomically (temp file in the root, fsync, ``os.replace``,
   fsync of the root directory).

Step 4 is the only publish step. A reader resolves ``CURRENT`` once and then
reads one complete, immutable directory, so it sees the old snapshot or the new
one, never a mix. An interruption anywhere before step 4 leaves the previous
snapshot published and readable. Publishing never deletes anything; only
``prune`` does, and it never deletes the ``CURRENT`` snapshot.

Tooling::

    python -m corrlib.snapshots --root ROOT list
    python -m corrlib.snapshots --root ROOT show ID|current
    python -m corrlib.snapshots --root ROOT verify [ID] [--all]
    python -m corrlib.snapshots --root ROOT rollback ID
    python -m corrlib.snapshots --root ROOT prune --keep N [--dry-run]
    python -m corrlib.snapshots --root ROOT unlock [--force]

``unlock`` is the documented recovery after a writer was killed (the lock file
outlives a hard kill; a writer that fails normally always releases it). It
refuses to remove the lock of a process that is still running on this host
unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import sys
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from . import store
from .config import (
    ANOMALIES_FILE,
    ARTIFACT_FORMAT_VERSION,
    CURRENT_FILE,
    DAILY_STATS_FILE,
    DATES_FILE,
    DEFAULT_PROCESSED_DIR,
    HIST_EDGES_FILE,
    HIST_FILE,
    INGEST_LOG_FILE,
    INPUTS_FILE,
    ISSUES_BY_DATE_FILE,
    LOCK_FILE,
    MANIFEST_FILE,
    PRICES_FILE,
    RETURNS_FILE,
    SNAPSHOT_ID_PATTERN,
    SNAPSHOTS_DIR,
    STAGING_PREFIX,
    TICKERS_FILE,
)

log = logging.getLogger("corrlib.snapshots")

_ID_RE = re.compile(SNAPSHOT_ID_PATTERN)

#: Tolerance of the per-date invariants checked by :func:`verify_snapshot` (|mean| <= 1, std >= 0).
_STAT_TOL = 1e-9


class SnapshotError(RuntimeError):
    """A store-root operation failed (missing snapshot, failed verification, bad pointer...)."""


class LockError(SnapshotError):
    """Another writer holds the store root's lock. ``holder`` is the lock file's content (or None)."""

    def __init__(self, root: Path, holder: dict[str, Any] | None, message: str | None = None) -> None:
        self.root = root
        self.holder = holder
        super().__init__(message or lock_message(root, holder))


def now_utc() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SS.mmmZ``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --- ids and directories --------------------------------------------------------------


def parse_snapshot_id(name: str) -> tuple[int, str] | None:
    """``(seq, h12)`` of a snapshot id, or None if ``name`` is not one."""
    match = _ID_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def content_hash(artifacts: dict[str, dict[str, Any]]) -> str:
    """First 12 hex digits of sha256 over the sorted ``"<name> <sha256>"`` lines of ``artifacts``."""
    names = sorted(artifacts)
    lines = []
    for k in range(len(names)):
        lines.append(f"{names[k]} {artifacts[names[k]]['sha256']}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def make_snapshot_id(seq: int, artifacts: dict[str, dict[str, Any]]) -> str:
    """``v{seq:04d}-{content_hash(artifacts)}``."""
    if seq < 1:
        raise ValueError(f"snapshot seq must be >= 1, got {seq}")
    return f"v{seq:04d}-{content_hash(artifacts)}"


def snapshots_path(root: str | Path) -> Path:
    """``<root>/snapshots``."""
    return Path(root) / SNAPSHOTS_DIR


def list_snapshot_ids(root: str | Path) -> list[str]:
    """Ids of the snapshot directories in ``root`` (with a manifest), in ``seq`` order.

    Staging directories and anything whose name is not a snapshot id are ignored.
    """
    base = snapshots_path(root)
    if not base.is_dir():
        return []
    found: list[tuple[int, str]] = []
    entries = sorted(base.iterdir())
    for k in range(len(entries)):
        parsed = parse_snapshot_id(entries[k].name)
        if parsed is not None and (entries[k] / MANIFEST_FILE).is_file():
            found.append((parsed[0], entries[k].name))
    found.sort()
    ids = []
    for k in range(len(found)):
        ids.append(found[k][1])
    return ids


def snapshot_dir(root: str | Path, snapshot_id: str) -> Path:
    """The directory of snapshot ``snapshot_id`` (``"current"`` names the CURRENT one).

    Raises
    ------
    SnapshotError
        If the id is malformed or the snapshot does not exist.
    """
    if snapshot_id == "current":
        current = current_id(root)
        if current is None:
            raise SnapshotError(f"{root}: no published snapshot (no {CURRENT_FILE})")
        snapshot_id = current
    if parse_snapshot_id(snapshot_id) is None:
        raise SnapshotError(f"{snapshot_id!r} is not a snapshot id (expected v0001-<12 hex>)")
    path = snapshots_path(root) / snapshot_id
    if not (path / MANIFEST_FILE).is_file():
        raise SnapshotError(f"{root}: no snapshot {snapshot_id} (known: {', '.join(list_snapshot_ids(root)) or 'none'})")
    return path


def next_seq(root: str | Path, parent: str | None = None) -> int:
    """One more than the highest ``seq`` among the root's snapshots and ``parent`` (1 for an empty root)."""
    highest = 0
    ids = list_snapshot_ids(root)
    if parent is not None:
        ids.append(parent)
    for k in range(len(ids)):
        parsed = parse_snapshot_id(ids[k])
        if parsed is not None:
            highest = max(highest, parsed[0])
    return highest + 1


# --- the CURRENT pointer ------------------------------------------------------------------


def read_current(root: str | Path) -> dict[str, Any] | None:
    """The parsed ``CURRENT`` pointer, or None if the root has none.

    Raises
    ------
    SnapshotError
        If the file exists but does not name a snapshot id.
    """
    path = Path(root) / CURRENT_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        pointer = json.loads(text)
    except ValueError as exc:
        raise SnapshotError(f"{path}: unreadable pointer ({exc})") from exc
    if not isinstance(pointer, dict) or parse_snapshot_id(str(pointer.get("snapshot", ""))) is None:
        raise SnapshotError(f"{path}: does not name a snapshot id")
    return pointer


def current_id(root: str | Path) -> str | None:
    """The id ``CURRENT`` points at, or None if nothing is published yet."""
    pointer = read_current(root)
    return None if pointer is None else str(pointer["snapshot"])


def write_current(root: str | Path, snapshot_id: str) -> dict[str, Any]:
    """Point ``CURRENT`` at ``snapshot_id`` atomically (the publish step).

    Writes a temp file in the root, fsyncs it, ``os.replace``-s it over
    ``CURRENT`` and fsyncs the root directory. A reader sees the old pointer or
    the new one, never a partial file. Returns the pointer written.
    """
    root = Path(root)
    pointer = {"snapshot": snapshot_id, "published_at": now_utc()}
    tmp = root / f".{CURRENT_FILE}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(pointer, fh)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, root / CURRENT_FILE)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(root)
    return pointer


def _fsync_dir(path: Path) -> None:
    """fsync a directory so renames inside it are durable (best effort; POSIX only)."""
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _fsync_tree(directory: Path) -> None:
    """fsync every file directly in ``directory`` and the directory itself (best effort)."""
    entries = sorted(directory.iterdir())
    for k in range(len(entries)):
        if entries[k].is_file():
            fd = os.open(entries[k], os.O_RDONLY)
            try:
                os.fsync(fd)
            except OSError:
                pass
            finally:
                os.close(fd)
    _fsync_dir(directory)


# --- single-writer lock -------------------------------------------------------------------


@dataclass(frozen=True)
class WriteLock:
    """A held store-root lock. ``holder`` is what was written into ``<root>/.lock``."""

    root: Path
    path: Path
    holder: dict[str, Any]


def read_lock(root: str | Path) -> dict[str, Any] | None:
    """Content of ``<root>/.lock`` (``{"unreadable": True}`` if it is not JSON yet), or None."""
    path = Path(root) / LOCK_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        holder = json.loads(text)
    except ValueError:
        return {"unreadable": True}
    return holder if isinstance(holder, dict) else {"unreadable": True}


def holder_alive(holder: dict[str, Any] | None) -> bool | None:
    """Whether the lock holder's process is running: True / False, or None when it cannot be told.

    Only a holder on this host can be checked, and only on POSIX
    (``os.kill(pid, 0)``; on Windows that call would terminate the process).
    """
    if not holder or holder.get("host") != socket.gethostname() or os.name != "posix":
        return None
    pid = holder.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def lock_message(root: Path, holder: dict[str, Any] | None) -> str:
    """Human-readable explanation of a held lock, naming the holder and the recovery command."""
    recovery = f"python -m corrlib.snapshots --root {root} unlock"
    if not holder or holder.get("unreadable"):
        who = "another writer (lock file being written or unreadable)"
    else:
        who = (
            f"pid {holder.get('pid')} on {holder.get('host')} since {holder.get('started_at')} "
            f"({holder.get('operation')})"
        )
    alive = holder_alive(holder)
    if alive is False:
        state = " That process is not running: the lock is stale."
    elif alive:
        state = " That process is still running; wait for it to finish."
    else:
        state = ""
    return (
        f"store root {root} is locked by {who}.{state} If the writer is gone (killed or crashed), "
        f"recover with `{recovery}`."
    )


def acquire_lock(root: str | Path, operation: str) -> WriteLock:
    """Take the single-writer lock of ``root`` (created with ``O_CREAT | O_EXCL``).

    Raises
    ------
    LockError
        If the lock file already exists; the message names the holder.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / LOCK_FILE
    holder = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": now_utc(),
        "operation": operation,
        "token": uuid.uuid4().hex,
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise LockError(root, read_lock(root)) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(holder, fh)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    log.debug("lock taken: %s", holder)
    return WriteLock(root=root, path=path, holder=holder)


def release_lock(lock: WriteLock) -> None:
    """Remove the lock file if it is still ours (same token); otherwise leave it and warn."""
    current = read_lock(lock.root)
    if current is None or current.get("token") != lock.holder["token"]:
        log.warning("lock %s was removed or replaced while held; leaving it alone", lock.path)
        return
    lock.path.unlink(missing_ok=True)


@contextmanager
def locked(root: str | Path, operation: str) -> Iterator[WriteLock]:
    """Hold the store root's single-writer lock for the body (released on any exit)."""
    lock = acquire_lock(root, operation)
    try:
        yield lock
    finally:
        release_lock(lock)


def unlock(root: str | Path, force: bool = False) -> dict[str, Any] | None:
    """Remove a leftover lock (recovery after a killed writer). Returns the removed holder, or None.

    Raises
    ------
    LockError
        If the holder is still running on this host and ``force`` is False.
    """
    root = Path(root)
    holder = read_lock(root)
    if holder is None:
        return None
    if holder_alive(holder) and not force:
        raise LockError(
            root,
            holder,
            f"refusing to unlock {root}: pid {holder.get('pid')} ({holder.get('operation')}) is still "
            "running on this host; wait for it, or pass --force if you are sure it is not writing",
        )
    (root / LOCK_FILE).unlink(missing_ok=True)
    append_log(
        root,
        {
            **new_attempt(root, "unlock"),
            "status": "unlocked",
            "reason": f"removed lock of {holder}" + (" (forced)" if force else ""),
        },
    )
    return holder


# --- audit log ----------------------------------------------------------------------------


def new_attempt(
    root: str | Path, operation: str, files: Sequence[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A fresh log record ``{attempt_id, at, operation, root, files, status, reason, parent, snapshot}``."""
    return {
        "attempt_id": uuid.uuid4().hex,
        "at": now_utc(),
        "operation": operation,
        "root": str(Path(root).resolve()),
        "files": list(files) if files is not None else [],
        "status": None,
        "reason": None,
        "parent": None,
        "snapshot": None,
    }


def append_log(root: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """Append one JSON line to ``<root>/ingest_log.jsonl`` (``O_APPEND``, one write, fsync)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, allow_nan=False, default=str) + "\n").encode("utf-8")
    fd = os.open(root / INGEST_LOG_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return record


def read_log(root: str | Path) -> list[dict[str, Any]]:
    """Every record of ``<root>/ingest_log.jsonl`` in order (empty if there is no log)."""
    path = Path(root) / INGEST_LOG_FILE
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for k in range(len(lines)):
        if lines[k].strip():
            records.append(json.loads(lines[k]))
    return records


def describe_error(exc: BaseException) -> str:
    """``"<ExceptionType>: <message>"`` (``"interrupted"`` for KeyboardInterrupt)."""
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


# --- publish ------------------------------------------------------------------------------


def new_staging(root: str | Path) -> Path:
    """Create and return an empty ``<root>/snapshots/.staging-<uuid>/``."""
    path = snapshots_path(root) / f"{STAGING_PREFIX}{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def publish(
    root: str | Path,
    content: store.SnapshotContent,
    extra: dict[str, Any],
    kind: str,
    parent: str | None,
    lock: WriteLock | None = None,
) -> dict[str, Any]:
    """Write, verify and publish a new snapshot of ``content``; returns its manifest.

    The caller must hold the root's lock (pass it to have that checked).
    Steps: staging write + fsync -> :func:`verify_snapshot` -> ``os.rename`` to
    ``snapshots/<id>`` -> :func:`write_current`. On any exception the staging
    directory (or the renamed, never-pointed-at snapshot) is removed and
    ``CURRENT`` is untouched, then the exception propagates.

    Parameters
    ----------
    content, extra:
        As for :func:`corrlib.store.write_snapshot`.
    kind:
        ``"full_build"`` or ``"update"`` (``manifest.snapshot.kind``).
    parent:
        Id of the snapshot this one derives from (the CURRENT one), or None.
    """
    root = Path(root)
    if lock is not None and read_lock(root) != lock.holder:
        raise LockError(root, read_lock(root), f"{root}: the lock is no longer held by this writer")
    seq = next_seq(root, parent)
    created_at = now_utc()

    def snapshot_block(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": make_snapshot_id(seq, artifacts),
            "seq": seq,
            "parent": parent,
            "created_at": created_at,
            "kind": kind,
        }

    staging = new_staging(root)
    target: Path | None = None
    snapshot_id = None
    try:
        manifest = store.write_snapshot(staging, content, extra, snapshot=snapshot_block)
        snapshot_id = manifest["snapshot"]["id"]
        _fsync_tree(staging)
        problems = verify_snapshot(staging, expected_id=snapshot_id)
        if problems:
            raise SnapshotError(f"snapshot {snapshot_id} failed verification: " + "; ".join(problems))
        destination = snapshots_path(root) / snapshot_id
        if destination.exists():
            raise SnapshotError(f"{destination} already exists")
        os.rename(staging, destination)
        target = destination
        _fsync_dir(snapshots_path(root))
        write_current(root, snapshot_id)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if target is not None and _points_elsewhere(root, snapshot_id):
            shutil.rmtree(target, ignore_errors=True)
        raise
    log.info("published %s (%s, parent %s)", snapshot_id, kind, parent)
    return manifest


def _points_elsewhere(root: Path, snapshot_id: str | None) -> bool:
    """True unless CURRENT (readable) names ``snapshot_id`` (then the snapshot must be kept)."""
    try:
        return current_id(root) != snapshot_id
    except SnapshotError:
        return False


# --- verification -------------------------------------------------------------------------


def verify_snapshot(directory: str | Path, expected_id: str | None = None) -> list[str]:
    """Check one snapshot directory thoroughly. Returns the problems found (empty = OK).

    Checks: every artifact's sha256 against the manifest; the snapshot block
    (id format, ``seq`` matches the id, the id's hash matches the artifacts,
    the id matches ``expected_id`` and the directory name when that is an id);
    the axes (ISO dates strictly increasing, tickers sorted and unique,
    manifest ``n_dates/n_tickers/first_date/last_date`` agree); the shapes of
    ``returns.npy`` (float32 ``[T x N]``), ``daily_stats`` (``T`` rows, same
    dates), ``hist`` (``[T x B]``, ``B`` = manifest bins), ``hist_edges``,
    ``prices.parquet`` (``T*N`` rows), ``issues_by_date`` (same dates),
    ``inputs.date_sources`` (same dates) and the anomaly records (date and
    index on the axis); and per date: ``0 <= n_valid <= N``, ``hist`` row sum
    ``== n_valid (n_valid - 1) / 2``, mean/std finite exactly where
    ``n_valid >= 2``, ``|mean| <= 1``, ``std >= 0``, nothing valid before ``W``,
    and ``first_valid_date`` consistent with ``n_valid``.
    """
    directory = Path(directory)
    try:
        manifest = store.read_manifest(directory)
    except (OSError, ValueError) as exc:
        return [f"{MANIFEST_FILE}: unreadable ({exc})"]
    problems = store.verify_manifest(directory)
    if problems:
        return problems  # nothing else can be trusted
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        return [f"format_version {manifest.get('format_version')!r}: a store snapshot must be format {ARTIFACT_FORMAT_VERSION}"]
    problems.extend(_check_snapshot_block(directory, manifest, expected_id))
    try:
        problems.extend(_check_content(directory, manifest))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"cannot read the artifacts: {describe_error(exc)}")
    return problems


def _check_snapshot_block(directory: Path, manifest: dict[str, Any], expected_id: str | None) -> list[str]:
    problems: list[str] = []
    block = manifest.get("snapshot")
    if not isinstance(block, dict):
        return ["manifest has no snapshot block"]
    sid = str(block.get("id"))
    parsed = parse_snapshot_id(sid)
    if parsed is None:
        return [f"snapshot id {sid!r} is malformed"]
    if block.get("seq") != parsed[0]:
        problems.append(f"snapshot seq {block.get('seq')!r} does not match id {sid}")
    if content_hash(manifest["artifacts"]) != parsed[1]:
        problems.append(f"snapshot id {sid} does not match the artifact checksums")
    if block.get("kind") not in ("full_build", "update"):
        problems.append(f"snapshot kind {block.get('kind')!r} is not full_build or update")
    if expected_id is not None and sid != expected_id:
        problems.append(f"snapshot id {sid} != expected {expected_id}")
    if parse_snapshot_id(directory.name) is not None and directory.name != sid:
        problems.append(f"directory {directory.name} holds snapshot {sid}")
    return problems


def _check_content(directory: Path, manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    dates = store.read_json(directory / DATES_FILE)
    tickers = store.read_json(directory / TICKERS_FILE)
    n_dates, n_tickers = len(dates), len(tickers)
    if n_dates == 0 or n_tickers == 0:
        return [f"empty axes: {n_dates} dates x {n_tickers} tickers"]
    for t in range(n_dates):
        if not isinstance(dates[t], str) or not _is_iso_date(dates[t]):
            problems.append(f"dates[{t}] = {dates[t]!r} is not an ISO date")
            break
        if t > 0 and not dates[t - 1] < dates[t]:
            problems.append(f"dates not strictly increasing at index {t} ({dates[t - 1]} -> {dates[t]})")
            break
    for j in range(1, n_tickers):
        if not tickers[j - 1] < tickers[j]:
            problems.append(f"tickers not sorted and unique at index {j}")
            break
    expected_axes = {"n_dates": n_dates, "n_tickers": n_tickers, "first_date": dates[0], "last_date": dates[-1]}
    names = list(expected_axes)
    for k in range(len(names)):
        if manifest.get(names[k]) != expected_axes[names[k]]:
            problems.append(f"manifest {names[k]} {manifest.get(names[k])!r} != axes {expected_axes[names[k]]!r}")

    returns = np.load(directory / RETURNS_FILE, mmap_mode="r")
    if returns.shape != (n_dates, n_tickers) or returns.dtype != np.float32:
        problems.append(f"returns.npy is {returns.dtype}{returns.shape}, expected float32{(n_dates, n_tickers)}")
    del returns

    stats = pq.read_table(directory / DAILY_STATS_FILE)
    hist = np.load(directory / HIST_FILE)
    edges = np.load(directory / HIST_EDGES_FILE)
    bins = int(manifest.get("hist", {}).get("bins", -1))
    if edges.shape != (bins + 1,) or not (np.diff(edges) > 0).all():
        problems.append(f"hist_edges {edges.shape} is not {bins + 1} increasing edges")
    if hist.shape != (n_dates, bins):
        problems.append(f"hist {hist.shape} != {(n_dates, bins)}")
    if stats.num_rows != n_dates or _iso_column(stats, "date") != dates:
        problems.append("daily_stats dates do not match dates.json")
    if problems:
        return problems

    window = int(manifest["window"])
    n_valid = stats.column("n_valid").to_numpy().astype(np.int64)
    mean = stats.column("mean_corr").to_numpy(zero_copy_only=False).astype(np.float64)
    std = stats.column("std_corr").to_numpy(zero_copy_only=False).astype(np.float64)
    _check_per_date(problems, dates, n_valid, mean, std, hist, n_tickers, window)
    first_valid = np.flatnonzero(n_valid >= 2)
    expected_first = dates[int(first_valid[0])] if first_valid.size else None
    if manifest.get("first_valid_date") != expected_first:
        problems.append(f"manifest first_valid_date {manifest.get('first_valid_date')!r} != {expected_first!r}")
    if manifest.get("policy", {}).get("window") != window:
        problems.append("manifest policy window != manifest window")

    n_price_rows = pq.read_metadata(directory / PRICES_FILE).num_rows
    if n_price_rows != n_dates * n_tickers:
        problems.append(f"prices.parquet has {n_price_rows} rows, expected {n_dates * n_tickers}")
    issues = pq.read_table(directory / ISSUES_BY_DATE_FILE, columns=["date"])
    if _iso_column(issues, "date") != dates:
        problems.append("issues_by_date dates do not match dates.json")
    sources = store.read_json(directory / INPUTS_FILE).get("date_sources", {})
    if list(sources) != dates:
        problems.append("inputs.json date_sources do not match dates.json")
    document = store.read_json(directory / ANOMALIES_FILE)
    records = document.get("records", []) if isinstance(document, dict) else document
    for k in range(len(records)):
        t = records[k].get("index")
        if not isinstance(t, int) or not 0 <= t < n_dates or dates[t] != records[k].get("date"):
            problems.append(f"anomaly record {records[k].get('date')!r} index {t!r} is not on the date axis")
            break
    return problems


def _check_per_date(
    problems: list[str],
    dates: Sequence[str],
    n_valid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    hist: np.ndarray,
    n_tickers: int,
    window: int,
) -> None:
    """Vectorised per-date invariants of the aggregates (appends to ``problems``)."""

    def first(mask: np.ndarray, what: str) -> None:
        bad = np.flatnonzero(mask)
        if bad.size:
            problems.append(f"{what} on {bad.size} date(s), first {dates[int(bad[0])]}")

    first((n_valid < 0) | (n_valid > n_tickers), f"n_valid outside [0, {n_tickers}]")
    first(n_valid[: min(window, n_valid.size)] != 0, "n_valid > 0 before the first window")
    first(hist.sum(axis=1) != n_valid * (n_valid - 1) // 2, "hist row sum != n_valid (n_valid - 1) / 2")
    first((hist < 0).any(axis=1), "negative histogram counts")
    defined = n_valid >= 2
    first(np.isfinite(mean) != defined, "mean_corr finite where n_valid < 2 (or NaN where >= 2)")
    first(np.isfinite(std) != defined, "std_corr finite where n_valid < 2 (or NaN where >= 2)")
    with np.errstate(invalid="ignore"):
        first(np.abs(mean) > 1 + _STAT_TOL, "|mean_corr| > 1")
        first(std < -_STAT_TOL, "negative std_corr")


def _iso_column(table: Any, name: str) -> list[str]:
    values = table.column(name).to_pylist()
    out = []
    for k in range(len(values)):
        value = values[k]
        out.append(value.isoformat() if isinstance(value, date) else str(value))
    return out


def _is_iso_date(text: str) -> bool:
    if len(text) != 10:
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def verify_root(root: str | Path, snapshot_id: str | None = None, all_snapshots: bool = False) -> dict[str, list[str]]:
    """Verify snapshots of a root: ``snapshot_id``, every snapshot, or (default) the CURRENT one.

    Returns ``{id: problems}``; a root-level problem (unreadable or dangling
    ``CURRENT``) is reported under the key ``"CURRENT"``.
    """
    results: dict[str, list[str]] = {}
    try:
        current = current_id(root)
    except SnapshotError as exc:
        results[CURRENT_FILE] = [str(exc)]
        current = None
    if current is not None and current not in list_snapshot_ids(root):
        results[CURRENT_FILE] = [f"CURRENT names {current}, which does not exist"]
    if all_snapshots:
        ids = list_snapshot_ids(root)
    elif snapshot_id is not None:
        ids = [snapshot_dir(root, snapshot_id).name]
    elif current is not None and CURRENT_FILE not in results:
        ids = [current]
    else:
        ids = []
        results.setdefault(CURRENT_FILE, ["no published snapshot"])
    for k in range(len(ids)):
        results[ids[k]] = verify_snapshot(snapshots_path(root) / ids[k])
    return results


# --- listing ------------------------------------------------------------------------------


def changes_summary(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Counts from ``manifest.changes`` of an update (None for a full build).

    ``{reason, inputs: {kind: n}, dates_added, returns_changed_dates,
    recomputed_window_dates, anomaly_changes, tickers_added}`` (all counts).
    """
    changes = manifest.get("changes")
    if not isinstance(changes, dict):
        return None
    by_kind: dict[str, int] = {}
    inputs = changes.get("inputs", [])
    for k in range(len(inputs)):
        by_kind[inputs[k]["kind"]] = by_kind.get(inputs[k]["kind"], 0) + 1
    return {
        "reason": changes.get("reason"),
        "inputs": by_kind,
        "dates_added": len(changes.get("dates_added", [])),
        "returns_changed_dates": len(changes.get("returns_changed_dates", [])),
        "recomputed_window_dates": int(changes.get("recomputed_window_dates", {}).get("count", 0)),
        "anomaly_changes": len(changes.get("anomaly_changes", [])),
        "tickers_added": len(changes.get("tickers_added", [])),
    }


def describe_snapshot(directory: str | Path, current: str | None = None) -> dict[str, Any]:
    """Summary of one snapshot from its manifest (cheap: reads ``manifest.json`` only).

    ``{id, seq, parent, created_at, kind, first_date, data_as_of, n_dates,
    n_tickers, window, dataset, is_current, changes_summary}``.
    """
    manifest = store.read_manifest(directory)
    block = manifest.get("snapshot", {})
    return {
        "id": block.get("id"),
        "seq": block.get("seq"),
        "parent": block.get("parent"),
        "created_at": block.get("created_at"),
        "kind": block.get("kind"),
        "first_date": manifest.get("first_date"),
        "data_as_of": manifest.get("last_date"),
        "n_dates": manifest.get("n_dates"),
        "n_tickers": manifest.get("n_tickers"),
        "window": manifest.get("window"),
        "dataset": manifest.get("dataset"),
        "is_current": block.get("id") == current,
        "changes_summary": changes_summary(manifest),
    }


def list_snapshots(root: str | Path) -> list[dict[str, Any]]:
    """:func:`describe_snapshot` of every snapshot in the root, in ``seq`` order."""
    current = current_id(root)
    ids = list_snapshot_ids(root)
    out = []
    for k in range(len(ids)):
        out.append(describe_snapshot(snapshots_path(root) / ids[k], current))
    return out


def staging_dirs(root: str | Path) -> list[Path]:
    """Leftover ``snapshots/.staging-*`` directories (a writer in progress, or a crashed one)."""
    base = snapshots_path(root)
    if not base.is_dir():
        return []
    return sorted(base.glob(f"{STAGING_PREFIX}*"))


# --- operator actions ---------------------------------------------------------------------


def rollback(root: str | Path, snapshot_id: str) -> dict[str, Any]:
    """Point CURRENT at an existing snapshot (after verifying it). Returns the log record.

    Raises
    ------
    SnapshotError
        If the snapshot does not exist or fails :func:`verify_snapshot`.
    LockError
        If another writer holds the lock.
    """
    root = Path(root)
    attempt = new_attempt(root, "rollback")
    try:
        with locked(root, "rollback"):
            target = snapshot_dir(root, snapshot_id).name
            previous = current_id(root)
            attempt["parent"], attempt["snapshot"] = previous, target
            if previous == target:
                attempt.update(status="duplicate", reason=f"{target} is already current")
                return append_log(root, attempt)
            problems = verify_snapshot(snapshots_path(root) / target)
            if problems:
                raise SnapshotError(f"refusing to roll back to {target}: " + "; ".join(problems))
            write_current(root, target)
    except BaseException as exc:
        attempt.update(status="failed", reason=describe_error(exc))
        append_log(root, attempt)
        raise
    attempt.update(status="rolled_back", reason=f"CURRENT {previous} -> {target}")
    log.info("rolled back: CURRENT %s -> %s", previous, target)
    return append_log(root, attempt)


def prune(root: str | Path, keep: int, dry_run: bool = False) -> dict[str, Any]:
    """Delete old snapshots and stale staging directories. Returns the log record.

    Keeps the ``keep`` snapshots with the highest ``seq`` plus the CURRENT
    one (never deleted). Every ``.staging-*`` directory is removed: the lock
    is held, so no build is in progress. A running API that already loaded a
    pruned snapshot keeps serving it from open files on POSIX.
    """
    if keep < 1:
        raise ValueError("--keep must be >= 1")
    root = Path(root)
    attempt = new_attempt(root, "prune")
    try:
        with locked(root, "prune"):
            current = current_id(root)
            ids = list_snapshot_ids(root)
            kept = set(ids[max(len(ids) - keep, 0) :])
            if current is not None:
                kept.add(current)
            removed = []
            for k in range(len(ids)):
                if ids[k] not in kept:
                    removed.append(ids[k])
            stale = staging_dirs(root)
            stale_names = []
            for k in range(len(stale)):
                stale_names.append(stale[k].name)
            attempt.update(
                parent=current,
                snapshot=current,
                removed=removed,
                staging_removed=stale_names,
                kept=sorted(kept),
                dry_run=dry_run,
            )
            if not dry_run:
                for k in range(len(removed)):
                    shutil.rmtree(snapshots_path(root) / removed[k], ignore_errors=True)
                for k in range(len(stale)):
                    shutil.rmtree(stale[k], ignore_errors=True)
    except BaseException as exc:
        attempt.update(status="failed", reason=describe_error(exc))
        append_log(root, attempt)
        raise
    attempt.update(
        status="pruned",
        reason=f"keep {keep}: removed {len(removed)} snapshot(s) and {len(stale_names)} staging dir(s)"
        + (" (dry run)" if dry_run else ""),
    )
    if dry_run:
        return attempt
    return append_log(root, attempt)


# --- CLI ----------------------------------------------------------------------------------


def _print_list(root: Path) -> int:
    rows = list_snapshots(root)
    if not rows:
        print(f"{root}: no snapshots")
        return 0
    print(f"{'':2}{'id':<20} {'kind':<10} {'parent':<20} {'created_at':<25} {'data_as_of':<11} {'T x N':>12}  changes")
    for k in range(len(rows)):
        r = rows[k]
        mark = "* " if r["is_current"] else "  "
        summary = r["changes_summary"]
        if summary is None:
            text = "full build"
        else:
            kinds = []
            names = sorted(summary["inputs"])
            for j in range(len(names)):
                kinds.append(f"{summary['inputs'][names[j]]} {names[j]}")
            text = (
                f"{', '.join(kinds) or 'overrides'}: +{summary['dates_added']} date(s), "
                f"{summary['returns_changed_dates']} return row(s), "
                f"{summary['recomputed_window_dates']} window(s) recomputed"
            )
            if summary["anomaly_changes"]:
                text += f", {summary['anomaly_changes']} anomaly change(s)"
            if summary["reason"]:
                text += f" - {summary['reason']}"
        shape = f"{r['n_dates']} x {r['n_tickers']}"
        print(f"{mark}{r['id']:<20} {r['kind']:<10} {r['parent']!s:<20} {r['created_at']:<25} {r['data_as_of']:<11} {shape:>12}  {text}")
    stale = staging_dirs(root)
    if stale:
        print(f"({len(stale)} staging dir(s) present: a build in progress, or leftovers removed by `prune`)")
    holder = read_lock(root)
    if holder is not None:
        print(lock_message(root, holder))
    return 0


def _print_show(root: Path, snapshot_id: str) -> int:
    directory = snapshot_dir(root, snapshot_id)
    manifest = store.read_manifest(directory)
    artifacts = {}
    names = list(manifest.get("artifacts", {}))
    for k in range(len(names)):
        artifacts[names[k]] = manifest["artifacts"][names[k]]["size_bytes"]
    summary = {
        "snapshot": manifest.get("snapshot"),
        "is_current": directory.name == current_id(root),
        "path": str(directory),
        "dataset": manifest.get("dataset"),
        "view": manifest.get("view"),
        "data": {
            "n_dates": manifest.get("n_dates"),
            "n_tickers": manifest.get("n_tickers"),
            "first_date": manifest.get("first_date"),
            "last_date": manifest.get("last_date"),
            "first_valid_date": manifest.get("first_valid_date"),
            "window": manifest.get("window"),
        },
        "source": manifest.get("source"),
        "anomalies_summary": manifest.get("anomalies_summary"),
        "overrides": manifest.get("overrides"),
        "changes": manifest.get("changes"),
        "artifacts_bytes": artifacts,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _print_verify(root: Path, snapshot_id: str | None, all_snapshots: bool) -> int:
    results = verify_root(root, snapshot_id, all_snapshots)
    failed = 0
    names = list(results)
    for k in range(len(names)):
        problems = results[names[k]]
        if problems:
            failed += 1
            print(f"FAIL {names[k]}")
            for p in range(len(problems)):
                print(f"     - {problems[p]}")
        else:
            print(f"OK   {names[k]}")
    stale = staging_dirs(root)
    if stale:
        print(f"note: {len(stale)} staging dir(s) present (ignored by readers; `prune` removes them)")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corrlib.snapshots", description="Inspect and manage a versioned store root.")
    parser.add_argument("--root", type=Path, default=DEFAULT_PROCESSED_DIR, help="store root (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the snapshots (* = CURRENT)")
    show = commands.add_parser("show", help="show one snapshot's manifest summary and changes")
    show.add_argument("id", help="snapshot id, or 'current'")
    verify = commands.add_parser("verify", help="verify checksums and internal consistency (default: CURRENT)")
    verify.add_argument("id", nargs="?", default=None, help="snapshot id (default: CURRENT)")
    verify.add_argument("--all", action="store_true", help="verify every snapshot")
    back = commands.add_parser("rollback", help="point CURRENT at an existing snapshot")
    back.add_argument("id", help="snapshot id")
    trim = commands.add_parser("prune", help="delete old snapshots (never CURRENT) and stale staging dirs")
    trim.add_argument("--keep", type=int, required=True, help="number of newest snapshots to keep (CURRENT is kept in addition)")
    trim.add_argument("--dry-run", action="store_true", help="only print what would be removed")
    release = commands.add_parser("unlock", help="remove a leftover lock after a writer was killed")
    release.add_argument("--force", action="store_true", help="also when the holder still runs on this host")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    root: Path = args.root
    try:
        if args.command == "list":
            return _print_list(root)
        if args.command == "show":
            return _print_show(root, args.id)
        if args.command == "verify":
            return _print_verify(root, args.id, args.all)
        if args.command == "rollback":
            record = rollback(root, args.id)
            print(f"{record['status']}: {record['reason']}")
            return 0
        if args.command == "prune":
            record = prune(root, args.keep, dry_run=args.dry_run)
            print(f"{record['reason']}")
            print(f"  kept     {', '.join(record['kept']) or '-'}")
            print(f"  removed  {', '.join(record['removed']) or '-'}")
            if record["staging_removed"]:
                print(f"  staging  {', '.join(record['staging_removed'])}")
            return 0
        holder = unlock(root, force=args.force)
        print("no lock present" if holder is None else f"removed lock of pid {holder.get('pid')} on {holder.get('host')} ({holder.get('operation')})")
        return 0
    except LockError as exc:
        log.error("%s", exc)
        return 3
    except (SnapshotError, store.StoreError, ValueError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
