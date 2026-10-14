"""FastAPI application factory.

Run with ``uvicorn app.api.main:app --port 8000``. The artifact directory is
``CORR_DATA_DIR`` (default ``data/processed``): a store root (served through
its ``CURRENT`` pointer, re-read every ``CORR_RELOAD_SECONDS``), a snapshot
directory or a legacy v1 directory.
Once ``app/web/dist`` exists (``npm run build``), the built frontend is served at ``/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from corrlib import __version__
from corrlib.config import CORS_ORIGINS, DATA_DIR_ENV, DEFAULT_PROCESSED_DIR

from .registry import DEFAULT_RELOAD_SECONDS, RELOAD_ENV, SnapshotRegistry
from .routes import router
from .schemas import ORJSONResponse

log = logging.getLogger("app.api")

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


class FrontendFiles(StaticFiles):
    """The built UI. Until ``npm run build`` has produced the directory, a JSON 404."""

    async def check_config(self) -> None:
        if self.directory is None or not Path(self.directory).is_dir():
            raise StarletteHTTPException(404, "Frontend not built: run `npm run build` in app/web")
        await super().check_config()


class SnapshotHeaderMiddleware:
    """Adds ``X-Snapshot-Id`` (set by the store dependency in ``request.state``) to every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                snapshot_id = scope.get("state", {}).get("snapshot_id")
                if snapshot_id:
                    MutableHeaders(scope=message)["X-Snapshot-Id"] = snapshot_id
            await send(message)

        await self.app(scope, receive, send_with_header)


async def _watch(registry: SnapshotRegistry, interval: float) -> None:
    """Poll CURRENT forever; errors are logged and retried on the next tick."""
    while True:
        await asyncio.sleep(interval)
        try:
            await asyncio.to_thread(registry.reload)
        except Exception:
            log.exception("snapshot reload failed")


def _error(status: int, detail: dict, headers: dict[str, str] | None = None) -> ORJSONResponse:
    return ORJSONResponse({"detail": detail}, status_code=status, headers=headers)


def create_app(data_dir: Path | str | None = None, reload_seconds: float | None = None) -> FastAPI:
    """Build the API app.

    ``data_dir`` defaults to ``$CORR_DATA_DIR`` or ``data/processed``;
    ``reload_seconds`` to ``$CORR_RELOAD_SECONDS`` or 2 (0 disables the watcher).
    """
    resolved = Path(data_dir) if data_dir is not None else Path(os.environ.get(DATA_DIR_ENV, DEFAULT_PROCESSED_DIR))
    if reload_seconds is None:
        reload_seconds = float(os.environ.get(RELOAD_ENV, DEFAULT_RELOAD_SECONDS))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = SnapshotRegistry(resolved)
        app.state.registry = registry
        app.state.active_downloads = 0
        store = registry.get(None)
        log.info("loaded %s (%s): %d dates x %d tickers", resolved, store.snapshot_id, store.n_dates, store.n_tickers)
        watcher = None
        if registry.root is not None and reload_seconds > 0:
            watcher = asyncio.create_task(_watch(registry, reload_seconds))
        yield
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    app = FastAPI(
        title="Stock Correlation Explorer API",
        version=__version__,
        description="Trailing-window return correlations computed on demand from memmapped returns.",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.add_middleware(SnapshotHeaderMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Snapshot-Id", "Content-Disposition", "Content-Length"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> ORJSONResponse:
        detail = exc.detail
        if not (isinstance(detail, dict) and "code" in detail):
            code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
            detail = {"code": code, "message": str(detail)}
        return _error(exc.status_code, detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> ORJSONResponse:
        errors = jsonable_encoder(exc.errors())
        parts = []
        for k in range(len(errors)):
            loc = ".".join(str(p) for p in errors[k].get("loc", []))
            parts.append(f"{loc}: {errors[k].get('msg', 'invalid')}")
        return _error(422, {"code": "invalid_input", "message": "; ".join(parts) or "Invalid input", "errors": errors})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> ORJSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return _error(500, {"code": "internal_error", "message": "Internal server error"})

    app.include_router(router)
    # Checked per request, so the UI appears as soon as app/web/dist is built,
    # even if the API started first.
    app.mount("/", FrontendFiles(directory=WEB_DIST, html=True, check_dir=False), name="web")
    return app


app = create_app()
