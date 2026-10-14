"""Versioned snapshot store (corrlib.snapshots): atomic publish, interruption, rollback, prune, lock, readers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURE_WINDOW, FixtureData, daily_csv_text

from corrlib import snapshots, store
from corrlib.build import run_build
from corrlib.ingest import run_ingest


@pytest.fixture
def root(built_dir: Path, tmp_path: Path) -> Path:
    """A private copy of the shared fixture root (one snapshot v0001, CURRENT -> it)."""
    copy = tmp_path / "root"
    shutil.copytree(built_dir, copy)
    return copy


def rebuild(fixture_data: FixtureData, root: Path) -> str:
    manifest, _ = run_build(fixture_data.zip_path, root, window=FIXTURE_WINDOW, block=4)
    return manifest["snapshot"]["id"]


def state(root: Path) -> tuple[str | None, list[str]]:
    return snapshots.current_id(root), snapshots.list_snapshot_ids(root)


# --- ids and publication ---------------------------------------------------------------------


def test_snapshot_ids_are_sequence_plus_content_hash(built_dir: Path) -> None:
    sid = snapshots.current_id(built_dir)
    assert sid is not None
    seq, h12 = snapshots.parse_snapshot_id(sid)
    artifacts = store.read_manifest(built_dir)["artifacts"]
    assert seq == 1 and h12 == snapshots.content_hash(artifacts)
    assert snapshots.make_snapshot_id(12, artifacts) == f"v0012-{h12}"
    for bad in ("v1-0123456789ab", "v0001-0123456789AB", ".staging-0123", "current", "v0001-0123456789ab/x"):
        assert snapshots.parse_snapshot_id(bad) is None, bad


def test_publish_swaps_current_and_keeps_the_previous_snapshot(root: Path, fixture_data: FixtureData) -> None:
    first = snapshots.current_id(root)
    old = store.load_store(root)
    second = rebuild(fixture_data, root)
    assert second.startswith("v0002-") and second[6:] == first[6:]  # same zip -> same content hash
    assert snapshots.current_id(root) == second
    assert set(json.loads((root / "CURRENT").read_text())) == {"snapshot", "published_at"}
    s = store.load_store(root)
    assert s.snapshot == {**s.snapshot, "id": second, "seq": 2, "parent": first, "kind": "full_build"}
    pinned = store.load_store(snapshots.snapshot_dir(root, first))  # still there, unchanged
    np.testing.assert_array_equal(np.asarray(pinned.returns), np.asarray(old.returns))
    assert snapshots.verify_root(root, all_snapshots=True) == {first: [], second: []}
    assert sorted(p.name for p in root.iterdir()) == ["CURRENT", "ingest_log.jsonl", "snapshots"]
    assert snapshots.staging_dirs(root) == [] and not (root / ".lock").exists()
    listed = snapshots.list_snapshots(root)
    assert [(r["id"], r["is_current"], r["data_as_of"]) for r in listed] == [
        (first, False, fixture_data.dates[-1].isoformat()),
        (second, True, fixture_data.dates[-1].isoformat()),
    ]


@pytest.mark.parametrize("where", ["while_writing", "verification", "pointer_swap"])
def test_interrupted_build_leaves_the_previous_snapshot_published(
    where: str, root: Path, fixture_data: FixtureData, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = state(root)
    old_id = store.load_store(root).snapshot_id

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError(f"simulated crash ({where})")

    if where == "while_writing":
        monkeypatch.setattr(store, "write_hist", boom)  # staging holds some artifacts, not all
    elif where == "verification":
        monkeypatch.setattr(snapshots, "verify_snapshot", lambda *a, **k: ["simulated inconsistency"])
    else:
        monkeypatch.setattr(snapshots, "write_current", boom)  # staged, verified, renamed, never pointed at
    with pytest.raises((RuntimeError, snapshots.SnapshotError), match="simulated"):
        rebuild(fixture_data, root)
    assert state(root) == before
    assert store.load_store(root).snapshot_id == old_id
    assert snapshots.staging_dirs(root) == [] and not (root / ".lock").exists()
    assert snapshots.read_log(root)[-1]["status"] == "failed"
    monkeypatch.undo()
    assert rebuild(fixture_data, root).startswith("v0002-")


@pytest.mark.parametrize("point", ["verify_snapshot", "write_current"])
def test_hard_kill_before_the_swap_is_invisible_and_recoverable(point: str, root: Path, fixture_data: FixtureData) -> None:
    """A writer killed (no cleanup at all) after staging: readers keep the old snapshot; unlock + prune recover."""
    before = state(root)
    old = store.load_store(root)
    code = textwrap.dedent(
        f"""
        import os, sys
        from corrlib import snapshots
        from corrlib.build import run_build
        snapshots.{point} = lambda *a, **k: os._exit(17)
        run_build(sys.argv[1], sys.argv[2], window={FIXTURE_WINDOW}, block=4)
        """
    )
    proc = subprocess.run([sys.executable, "-c", code, str(fixture_data.zip_path), str(root)], capture_output=True, timeout=300)
    assert proc.returncode == 17, proc.stderr.decode()

    assert snapshots.current_id(root) == before[0]
    s = store.load_store(root)
    assert s.snapshot_id == old.snapshot_id
    np.testing.assert_array_equal(np.asarray(s.returns), np.asarray(old.returns))
    leftovers = snapshots.staging_dirs(root)
    if point == "verify_snapshot":
        assert len(leftovers) == 1 and snapshots.list_snapshot_ids(root) == before[1]
    else:  # complete and verified, but never published: not what readers get
        assert leftovers == [] and len(snapshots.list_snapshot_ids(root)) == 2

    with pytest.raises(snapshots.LockError, match="not running: the lock is stale") as err:
        rebuild(fixture_data, root)
    assert "corrlib.snapshots --root" in str(err.value) and "unlock" in str(err.value)
    assert snapshots.unlock(root)["operation"] == "build"
    snapshots.prune(root, keep=10)
    assert snapshots.staging_dirs(root) == []
    new_id = rebuild(fixture_data, root)
    assert snapshots.current_id(root) == new_id and store.load_store(root).snapshot_id == new_id


# --- verification -------------------------------------------------------------------------------


def rehash(directory: Path) -> None:
    """Rewrite the manifest checksums after tampering (so the consistency checks are reached)."""
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["artifacts"] = store.artifact_checksums(directory)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))


def test_verify_detects_tampering_and_inconsistency(built_snapshot_dir: Path, tmp_path: Path) -> None:
    assert snapshots.verify_snapshot(built_snapshot_dir) == []

    tampered = tmp_path / built_snapshot_dir.name
    shutil.copytree(built_snapshot_dir, tampered)
    hist = np.load(tampered / "hist.npy")
    t = int(np.flatnonzero(hist.sum(axis=1))[-1])
    hist[t, 0] += 1
    np.save(tampered / "hist.npy", hist)
    assert snapshots.verify_snapshot(tampered) == ["hist.npy: checksum mismatch"]
    rehash(tampered)
    problems = snapshots.verify_snapshot(tampered)
    assert any("does not match the artifact checksums" in p for p in problems)
    assert any("hist row sum != n_valid (n_valid - 1) / 2" in p for p in problems)

    shuffled = tmp_path / "shuffled"
    shutil.copytree(built_snapshot_dir, shuffled)
    dates = json.loads((shuffled / "dates.json").read_text())
    dates[3], dates[4] = dates[4], dates[3]
    (shuffled / "dates.json").write_text(json.dumps(dates))
    rehash(shuffled)
    problems = snapshots.verify_snapshot(shuffled, expected_id="v0001-000000000000")
    assert any("not strictly increasing" in p for p in problems)
    assert any("!= expected v0001-000000000000" in p for p in problems)


# --- rollback, prune, lock ------------------------------------------------------------------------


def test_rollback_points_current_at_an_existing_verified_snapshot(root: Path, fixture_data: FixtureData) -> None:
    first = snapshots.current_id(root)
    second = rebuild(fixture_data, root)
    record = snapshots.rollback(root, first)
    assert record["status"] == "rolled_back" and record["parent"] == second and record["snapshot"] == first
    assert snapshots.current_id(root) == first and store.load_store(root).snapshot_id == first
    assert snapshots.list_snapshot_ids(root) == [first, second]  # nothing deleted
    assert snapshots.rollback(root, first)["status"] == "duplicate"
    with pytest.raises(snapshots.SnapshotError, match="no snapshot v0009"):
        snapshots.rollback(root, "v0009-000000000000")

    data = bytearray((root / "snapshots" / second / "returns.npy").read_bytes())
    data[-1] ^= 0xFF
    (root / "snapshots" / second / "returns.npy").write_bytes(bytes(data))
    with pytest.raises(snapshots.SnapshotError, match="refusing to roll back"):
        snapshots.rollback(root, second)
    assert snapshots.current_id(root) == first
    assert [r["status"] for r in snapshots.read_log(root)[-4:]] == ["rolled_back", "duplicate", "failed", "failed"]
    assert snapshots.main(["--root", str(root), "rollback", "current"]) == 0  # already current: harmless


def test_prune_keeps_current_and_removes_stale_staging(root: Path, fixture_data: FixtureData) -> None:
    first = snapshots.current_id(root)
    ids = [first, rebuild(fixture_data, root), rebuild(fixture_data, root), rebuild(fixture_data, root)]
    snapshots.rollback(root, first)
    stale = root / "snapshots" / ".staging-deadbeef"
    stale.mkdir()
    (stale / "returns.npy").write_bytes(b"half written")

    dry = snapshots.prune(root, keep=1, dry_run=True)
    assert dry["removed"] == ids[1:3] and dry["kept"] == sorted([ids[0], ids[3]]) and dry["staging_removed"] == [stale.name]
    assert snapshots.list_snapshot_ids(root) == ids and stale.exists()

    record = snapshots.prune(root, keep=1)
    assert record["status"] == "pruned"
    assert snapshots.list_snapshot_ids(root) == [ids[0], ids[3]]  # CURRENT (oldest) is never deleted
    assert not stale.exists() and snapshots.current_id(root) == first
    assert store.load_store(root).snapshot_id == first
    with pytest.raises(ValueError):
        snapshots.prune(root, keep=0)


def test_single_writer_lock(root: Path, fixture_data: FixtureData, tmp_path: Path) -> None:
    with snapshots.locked(root, "ingest"):
        holder = snapshots.read_lock(root)
        assert holder["pid"] == os.getpid() and holder["host"] == socket.gethostname()
        assert holder["operation"] == "ingest" and holder["started_at"].endswith("Z")
        with pytest.raises(snapshots.LockError) as err:
            rebuild(fixture_data, root)
        message = str(err.value)
        assert f"pid {os.getpid()} on {socket.gethostname()}" in message and "still running" in message
        assert "(ingest)" in message and "unlock" in message
        with pytest.raises(snapshots.LockError, match="refusing to unlock"):
            snapshots.unlock(root)
        assert snapshots.main(["--root", str(root), "unlock"]) == 3
        assert snapshots.main(["--root", str(root), "prune", "--keep", "1"]) == 3
    assert not (root / ".lock").exists()
    assert snapshots.unlock(root) is None

    # A lock left by a process that no longer exists is reported as stale and removable.
    done = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True)
    dead = {"pid": int(done.stdout), "host": socket.gethostname(), "started_at": "2026-01-01T00:00:00.000Z", "operation": "build", "token": "x"}
    (root / ".lock").write_text(json.dumps(dead))
    with pytest.raises(snapshots.LockError, match="not running: the lock is stale"):
        rebuild(fixture_data, root)
    assert snapshots.main(["--root", str(root), "unlock"]) == 0
    assert snapshots.read_log(root)[-1]["status"] == "unlocked"
    assert rebuild(fixture_data, root).startswith("v0002-")

    # Releasing never removes a lock that someone else replaced in the meantime.
    lock = snapshots.acquire_lock(root, "a")
    (root / ".lock").write_text(json.dumps({**lock.holder, "token": "other"}))
    snapshots.release_lock(lock)
    assert (root / ".lock").exists()
    assert snapshots.unlock(root, force=True)["token"] == "other"


# --- readers during publication -------------------------------------------------------------------


def assert_consistent(s: store.Store) -> None:
    """Everything a reader got comes from one complete snapshot."""
    assert s.data_dir.name == s.snapshot_id == s.manifest["snapshot"]["id"]
    assert s.returns.shape == (s.n_dates, s.n_tickers) == (s.manifest["n_dates"], s.manifest["n_tickers"])
    assert s.dates[-1] == s.manifest["last_date"] and len(s.n_valid) == len(s.hist) == s.n_dates
    np.testing.assert_array_equal(s.hist.sum(axis=1), s.n_valid.astype(np.int64) * (s.n_valid - 1) // 2)
    returns_sha = hashlib.sha256((s.data_dir / "returns.npy").read_bytes()).hexdigest()
    assert returns_sha == s.manifest["artifacts"]["returns.npy"]["sha256"]


def test_concurrent_readers_never_see_a_mixed_or_partial_snapshot(root: Path, tmp_path: Path) -> None:
    stop = threading.Event()
    errors: list[str] = []
    seen: dict[str, set[int]] = {}
    reads = [0]

    def reader() -> None:
        while not stop.is_set():
            try:
                s = store.load_store(root)
                assert_consistent(s)
            except Exception as exc:  # noqa: BLE001 - any failure is the finding
                errors.append(repr(exc))
                return
            seen.setdefault(s.snapshot_id, set()).add(s.n_dates)
            reads[0] += 1

    base = store.load_store(root)
    raw, _ = store.read_prices(root)
    last = raw[-1].copy()
    days = list(pd.bdate_range(base.dates[-1], periods=8).date)[1:]
    rng = np.random.default_rng(5)
    threads = [threading.Thread(target=reader) for _ in range(3)]
    for thread in threads:
        thread.start()
    try:
        for k in range(len(days)):
            last = np.round(last * np.exp(rng.normal(0.0, 0.02, last.size)), 6)
            last[base.ticker_index["FLAT"]] = 20.0
            path = tmp_path / "in" / f"{days[k].strftime('%Y%m%d')}.csv"
            path.parent.mkdir(exist_ok=True)
            path.write_text(daily_csv_text(days[k], base.tickers, last))
            assert run_ingest(root, [path]).status == "published"
            time.sleep(0.02)
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    assert errors == []
    published = snapshots.list_snapshot_ids(root)
    assert len(published) == 1 + len(days) and len(seen) >= 2 and reads[0] > len(days)
    for sid, n_dates in seen.items():
        assert sid in published
        assert n_dates == {store.read_manifest(root / "snapshots" / sid)["n_dates"]}


# --- CLI ------------------------------------------------------------------------------------------


def test_cli_list_show_verify(root: Path, fixture_data: FixtureData, capsys: pytest.CaptureFixture[str]) -> None:
    first = snapshots.current_id(root)
    second = rebuild(fixture_data, root)
    assert snapshots.main(["--root", str(root), "list"]) == 0
    out = capsys.readouterr().out
    assert f"* {second}" in out and f"  {first}" in out and "full build" in out
    assert snapshots.main(["--root", str(root), "show", "current"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["snapshot"]["id"] == second and shown["is_current"] is True and shown["data"]["n_dates"] == len(fixture_data.dates)
    assert snapshots.main(["--root", str(root), "verify", "--all"]) == 0
    assert capsys.readouterr().out.count("OK") == 2
    assert snapshots.main(["--root", str(root), "show", "v0042-000000000000"]) == 2

    (root / "snapshots" / first / "tickers.json").write_text('["tampered"]')
    assert snapshots.main(["--root", str(root), "verify", first]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert snapshots.main(["--root", str(root), "verify"]) == 0  # CURRENT is still fine
