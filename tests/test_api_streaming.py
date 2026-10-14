"""Full-matrix streaming over a real HTTP server: overlap, cancellation, cleanup, publication.

A live uvicorn server runs in a thread. Block production is gated by a
semaphore the test controls (``Gate``), so "two downloads in flight", "client
disconnects mid-stream" and "a snapshot is published while a download is
running" are reproduced deterministically: no sleeps, every wait is a bounded
poll on an observable condition.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn

import app.api.download as dl
from app.api.main import create_app
from corrlib.corr import iter_full_blocks, window_status
from corrlib.ingest import run_ingest
from corrlib.store import load_store

TIMEOUT = 15.0


def wait_for(condition: Callable[[], bool], what: str, timeout: float = TIMEOUT) -> None:
    """Poll ``condition`` until true; fail after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for: {what}")
        time.sleep(0.005)


class Gate:
    """Wraps ``iter_full_blocks``: every block waits for one permit from the test."""

    def __init__(self) -> None:
        self.permits = threading.Semaphore(0)
        self.produced = 0
        self._lock = threading.Lock()

    def wrap(self, *args, **kwargs) -> Iterator:
        for item in iter_full_blocks(*args, **kwargs):
            if not self.permits.acquire(timeout=TIMEOUT):
                raise TimeoutError("gate permit not released")
            with self._lock:
                self.produced += 1
            yield item

    def release(self, n: int = 1) -> None:
        for _ in range(n):
            self.permits.release()


@contextmanager
def live_server(root: Path) -> Iterator[tuple[str, object]]:
    app = create_app(root, reload_seconds=0)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on", log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_for(lambda: server.started, "server start")
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        thread.join(TIMEOUT)


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Gate:
    g = Gate()
    monkeypatch.setattr(dl, "iter_full_blocks", g.wrap)
    monkeypatch.setattr(dl, "DOWNLOAD_BLOCK_ROWS", 4)  # many small blocks: 300 tickers -> 75 blocks
    return g


@pytest.fixture
def root(demo_store_dir: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "store"
    shutil.copytree(demo_store_dir, copy)
    return copy


def last_date(root: Path) -> str:
    return load_store(root).dates[-1]


def test_overlapping_downloads_are_capped_and_slots_freed(root: Path, gate: Gate) -> None:
    date = last_date(root)
    url = f"/api/corr/{date}/download?format=npy"
    with live_server(root) as (base, app):
        c1, c2, c3 = httpx.Client(base_url=base), httpx.Client(base_url=base), httpx.Client(base_url=base)
        with c1.stream("GET", url) as r1, c2.stream("GET", url) as r2:
            # Both responses have started (headers received) and are blocked in the gate.
            assert r1.status_code == r2.status_code == 200
            assert app.state.active_downloads == 2
            third = c3.get(url)
            assert third.status_code == 429
            assert third.json()["detail"]["code"] == "too_many_downloads" and third.headers["retry-after"]
            # Finish the first download completely; its slot comes back.
            gate.release(10_000)
            body = r1.read()
            wait_for(lambda: app.state.active_downloads <= 1, "first slot released")
            r2.read()
        wait_for(lambda: app.state.active_downloads == 0, "all slots released")
        assert c3.get(url).status_code == 200
        store = load_store(root)
        n = store.n_tickers
        matrix = np.load(io.BytesIO(body))
        assert matrix.shape == (n, n) and matrix.dtype == np.float32
        for c in (c1, c2, c3):
            c.close()


@pytest.mark.parametrize("fmt", ["npy", "npz"])
def test_client_disconnect_stops_computation_and_frees_the_slot(
    root: Path, gate: Gate, fmt: str, caplog: pytest.LogCaptureFixture
) -> None:
    date = last_date(root)
    total_blocks = -(-load_store(root).n_tickers // dl.DOWNLOAD_BLOCK_ROWS)
    with live_server(root) as (base, app):
        client = httpx.Client(base_url=base)
        with client.stream("GET", f"/api/corr/{date}/download?format={fmt}") as response:
            assert response.status_code == 200
            gate.release(2)
            got = 0
            for chunk in response.iter_raw():
                got += len(chunk)
                if gate.produced >= 2:
                    break
            assert got > 0
        client.close()  # the connection is gone mid-stream
        # Keep feeding permits one at a time: the server must notice the disconnect and stop.
        deadline = time.monotonic() + TIMEOUT
        while app.state.active_downloads and time.monotonic() < deadline:
            gate.release(1)
            time.sleep(0.005)
        assert app.state.active_downloads == 0, "download slot leaked after disconnect"
        assert gate.produced < total_blocks, f"computed all {total_blocks} blocks after the client left"
        # The server is healthy and a fresh download still succeeds end to end.
        gate.release(10_000)
        ok = httpx.get(f"{base}/api/corr/{date}/download?format={fmt}", timeout=TIMEOUT)
        assert ok.status_code == 200
        wait_for(lambda: app.state.active_downloads == 0, "slot released after a complete download")
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("dtype,fmt,atol", [("float32", "npz", 1e-6), ("float32", "npy", 1e-6), ("float16", "npz", 1e-3), ("float16", "npy", 1e-3)])
def test_successful_download_contract(root: Path, dtype: str, fmt: str, atol: float) -> None:
    store = load_store(root)
    date = store.dates[-1]
    n = store.n_tickers
    with live_server(root) as (base, _app):
        with httpx.Client(base_url=base, timeout=TIMEOUT) as client:
            info = client.get(f"/api/corr/{date}/download/info", params={"dtype": dtype, "format": fmt}).json()
            r = client.get(f"/api/corr/{date}/download", params={"dtype": dtype, "format": fmt})
            row = client.get(f"/api/corr/{date}/row/{store.tickers[3]}").json()
    assert r.status_code == 200
    assert r.headers["x-snapshot-id"] == store.snapshot_id
    assert f"_{dtype}_" in r.headers["content-disposition"] and r.headers["content-disposition"].endswith(f'.{fmt}"')
    assert len(r.content) == info["bytes"] == int(r.headers["content-length"])
    if fmt == "npz":
        archive = np.load(io.BytesIO(r.content))
        matrix = archive["corr"]
        assert archive["tickers"].tolist() == store.tickers
        meta = json.loads(str(archive["meta"]))
        assert meta["dtype"] == dtype and meta["shape"] == [n, n] and meta["snapshot_id"] == store.snapshot_id
        assert zipfile.ZipFile(io.BytesIO(r.content)).testzip() is None
    else:
        matrix = np.load(io.BytesIO(r.content))
    assert matrix.dtype == np.dtype(dtype) and matrix.shape == (n, n)
    api_row = np.array([np.nan if v is None else v for v in row["corr"]])
    np.testing.assert_allclose(matrix[3].astype(np.float64), api_row, atol=atol, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(matrix), np.isnan(matrix.T))


def test_publication_during_an_active_download(root: Path, demo_files, gate: Gate) -> None:
    """An in-flight request keeps its snapshot; requests after the publish see the new one."""
    date = demo_files.correction_date.isoformat()
    old = load_store(root)
    t = old.dates.index(date)
    col = old.tickers.index("TEC07")
    with live_server(root) as (base, app):
        client = httpx.Client(base_url=base, timeout=TIMEOUT)
        with client.stream("GET", f"/api/corr/{date}/download", params={"format": "npy"}) as response:
            assert response.headers["x-snapshot-id"] == old.snapshot_id
            gate.release(1)
            wait_for(lambda: gate.produced >= 1, "first block computed")
            # Publish a historical correction while the download is half done, then reload.
            new_id = run_ingest(root, [demo_files.correction_csv], reason="test correction").snapshot
            assert client.post("/api/snapshots/reload").json()["current"] == new_id
            assert client.get("/api/health").headers["x-snapshot-id"] == new_id
            gate.release(10_000)
            body = response.read()
        # The in-flight download is entirely from the OLD snapshot.
        Z, valid, _ = window_status(old.returns, t, old.window)
        expected = np.vstack([blk for _, blk in iter_full_blocks(Z, valid, dtype=np.float64)]).astype(np.float32)
        got = np.load(io.BytesIO(body))
        np.testing.assert_array_equal(got, expected)
        # New requests see the corrected data, and it differs where the correction applies.
        new_row = client.get(f"/api/corr/{date}/row/TEC07").json()
        assert client.get(f"/api/corr/{date}/row/TEC07").headers["x-snapshot-id"] == new_id
        new_vals = np.array([np.nan if v is None else v for v in new_row["corr"]])
        assert not np.allclose(new_vals, got[col].astype(np.float64), atol=1e-6, equal_nan=True)
        client.close()
        wait_for(lambda: app.state.active_downloads == 0, "slot released")
