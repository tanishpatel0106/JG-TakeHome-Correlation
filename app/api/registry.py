"""Snapshot registry: which published snapshot each request reads.

A store root (``CURRENT`` + ``snapshots/<id>/``) is served as a set of
immutable snapshots. The registry resolves ``CURRENT``, loads :class:`Store`
objects lazily and caches up to ``MAX_CACHED`` of them by id (the current one
is never evicted). A background task (see ``main.lifespan``) calls
:meth:`SnapshotRegistry.reload` every ``CORR_RELOAD_SECONDS`` so a new publish
is picked up without a restart. A plain snapshot directory or a legacy v1
directory is a single snapshot and is never reloaded.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from corrlib import snapshots
from corrlib.store import Store, load_store, resolve_snapshot_dir

log = logging.getLogger("app.api.registry")

#: Env var: seconds between polls of CURRENT (0 = never poll).
RELOAD_ENV = "CORR_RELOAD_SECONDS"
DEFAULT_RELOAD_SECONDS = 2.0
#: Stores kept in memory (each is a memmap + small per-date arrays).
MAX_CACHED = 4


class SnapshotNotFound(LookupError):
    """``snapshot_id`` is not a snapshot of this registry."""

    def __init__(self, snapshot_id: str, available: list[str]) -> None:
        super().__init__(snapshot_id)
        self.snapshot_id = snapshot_id
        self.available = available


class SnapshotRegistry:
    """Lazily loaded, cached :class:`Store` objects keyed by snapshot id (thread-safe)."""

    def __init__(self, path: str | Path, max_cached: int = MAX_CACHED) -> None:
        snapshot_path, root = resolve_snapshot_dir(path)
        self.root: Path | None = root
        self.max_cached = max_cached
        self._lock = threading.Lock()
        self._stores: OrderedDict[str, Store] = OrderedDict()
        self._meta: dict[str, Any] = {}
        store = load_store(snapshot_path)
        self.current_id: str = store.snapshot_id
        self._stores[store.snapshot_id] = store

    def available(self) -> list[str]:
        """Every snapshot id that :meth:`get` can serve."""
        if self.root is None:
            return [self.current_id]
        return snapshots.list_snapshot_ids(self.root)

    def get(self, snapshot_id: str | None = None) -> Store:
        """The Store for ``snapshot_id`` (None or ``"current"`` = the current snapshot).

        Raises
        ------
        SnapshotNotFound
            If the id is not a published snapshot of this root.
        """
        wanted = self.current_id if snapshot_id in (None, "", "current") else snapshot_id
        with self._lock:
            store = self._stores.get(wanted)
            if store is not None:
                self._stores.move_to_end(wanted)
                return store
        if self.root is None or snapshots.parse_snapshot_id(wanted) is None:
            raise SnapshotNotFound(wanted, self.available())
        directory = snapshots.snapshots_path(self.root) / wanted
        if not (directory / "manifest.json").is_file():
            raise SnapshotNotFound(wanted, self.available())
        store = load_store(directory)  # outside the lock: requests for cached ids keep flowing
        with self._lock:
            existing = self._stores.get(wanted)
            if existing is not None:
                return existing
            self._stores[wanted] = store
            self._evict()
        log.info("loaded snapshot %s", wanted)
        return store

    def _evict(self) -> None:
        """Drop least recently used stores beyond ``max_cached`` (never the current one). Lock held."""
        ids = list(self._stores)
        k = 0
        while len(self._stores) > self.max_cached and k < len(ids):
            if ids[k] != self.current_id:
                del self._stores[ids[k]]
                self._meta.pop(ids[k], None)
            k += 1

    def reload(self) -> tuple[str, str]:
        """Re-read ``CURRENT``; load and switch to the snapshot it names. Returns ``(previous, current)``."""
        previous = self.current_id
        if self.root is None:
            return previous, previous
        new_id = snapshots.current_id(self.root)
        if new_id is None or new_id == previous:
            return previous, previous
        self.get(new_id)
        with self._lock:
            self.current_id = new_id
        log.info("CURRENT changed: %s -> %s", previous, new_id)
        return previous, new_id

    def meta(self, store: Store, build: Any) -> Any:
        """``build(store)`` computed once per snapshot."""
        cached = self._meta.get(store.snapshot_id)
        if cached is None:
            cached = build(store)
            self._meta[store.snapshot_id] = cached
        return cached

    def describe_all(self) -> list[dict[str, Any]]:
        """One summary per snapshot (``corrlib.snapshots.describe_snapshot`` fields), in seq order."""
        if self.root is None:
            return [describe_store(self.get(None), True)]
        out = snapshots.list_snapshots(self.root)
        for k in range(len(out)):
            out[k]["is_current"] = out[k]["id"] == self.current_id
        return out


def describe_store(store: Store, is_current: bool) -> dict[str, Any]:
    """Snapshot summary built from a loaded Store (for single-snapshot / legacy directories)."""
    return {
        "id": store.snapshot_id,
        "seq": store.snapshot.get("seq"),
        "parent": store.snapshot.get("parent"),
        "created_at": store.snapshot.get("created_at"),
        "kind": store.snapshot.get("kind"),
        "data_as_of": store.dates[-1] if store.dates else None,
        "n_dates": store.n_dates,
        "n_tickers": store.n_tickers,
        "is_current": is_current,
        "changes_summary": snapshots.changes_summary(store.manifest),
    }
