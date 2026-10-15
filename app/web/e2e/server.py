"""Test server for the browser regressions in this folder (``npx playwright test``).

Builds the synthetic demo store in a temporary directory and serves the real API
plus the built UI (``app/web/dist``) with uvicorn. The only additions are test
controls under ``/__e2e`` (never part of the production app):

- ``POST /__e2e/downloads/hold`` / ``release``: matrix blocks wait until released,
  so downloads stay in flight and occupy their slots (a real HTTP 429 for the next one).
- ``POST /__e2e/downloads/drop``: the next download raises after its first block, so
  the server aborts the connection mid-stream (a dropped stream with a short body).
- ``GET /__e2e/downloads/active``: the server's in-flight download count.
- ``POST /__e2e/publish``: ingest the next demo update (append, then correction),
  publishing a new snapshot exactly as ``make demo-update`` does.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
from pathlib import Path

import uvicorn
from fastapi import APIRouter, Request

import app.api.download as dl
from app.api.main import create_app
from corrlib.build import run_build
from corrlib.corr import iter_full_blocks
from corrlib.demo import generate_demo
from corrlib.ingest import run_ingest

HOLD_TIMEOUT = 60.0


class Gate:
    """Wraps ``iter_full_blocks`` so tests can hold or break a download in flight."""

    def __init__(self) -> None:
        self.open = threading.Event()
        self.open.set()
        self.drop_next = False
        self._lock = threading.Lock()

    def wrap(self, *args, **kwargs):
        with self._lock:
            drop, self.drop_next = self.drop_next, False
        for i, item in enumerate(iter_full_blocks(*args, **kwargs)):
            if drop and i == 1:
                raise ConnectionAbortedError("e2e: simulated dropped stream")
            if not self.open.wait(HOLD_TIMEOUT):
                raise TimeoutError("e2e: download held too long")
            yield item


def build_app(work: Path):
    demo = generate_demo(work / "demo")
    store = work / "store"
    run_build(demo.zip_path, store)
    updates = [("append", demo.append_csv), ("correction", demo.correction_csv)]

    gate = Gate()
    dl.iter_full_blocks = gate.wrap
    dl.DOWNLOAD_BLOCK_ROWS = 4  # 300 tickers -> 75 blocks: long enough to hold or break mid-stream

    app = create_app(store, reload_seconds=0.5)
    router = APIRouter(prefix="/__e2e")

    @router.post("/downloads/hold")
    def hold() -> dict:
        gate.open.clear()
        return {"held": True}

    @router.post("/downloads/release")
    def release() -> dict:
        gate.open.set()
        return {"held": False}

    @router.post("/downloads/drop")
    def drop() -> dict:
        gate.drop_next = True
        return {"drop_next": True}

    @router.get("/downloads/active")
    def active(request: Request) -> dict:
        return {"active": request.app.state.active_downloads}

    @router.post("/publish")
    def publish() -> dict:
        if not updates:
            return {"published": None}
        kind, csv = updates.pop(0)
        return {"published": run_ingest(store, [csv], reason=f"e2e: {kind}").snapshot, "kind": kind}

    app.include_router(router)
    # Keep the static UI mount ("/") last so it does not shadow the test routes.
    web = [r for r in app.router.routes if getattr(r, "name", None) == "web"]
    for r in web:
        app.router.routes.remove(r)
        app.router.routes.append(r)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=4179)
    args = parser.parse_args()
    work = Path(tempfile.mkdtemp(prefix="corr-e2e-"))
    try:
        uvicorn.run(build_app(work), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
