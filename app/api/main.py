"""FastAPI application factory.

Run with ``uvicorn app.api.main:app --port 8000``. The artifact directory is
``CORR_DATA_DIR`` (default ``data/processed``); it is loaded once at startup.
If ``app/web/dist`` exists (``npm run build``), the built frontend is served at ``/``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from corrlib import __version__
from corrlib.config import CORS_ORIGINS, DATA_DIR_ENV, DEFAULT_PROCESSED_DIR
from corrlib.store import load_store

from .routes import build_meta, router
from .schemas import ORJSONResponse

log = logging.getLogger("app.api")

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    """Build the API app. ``data_dir`` defaults to ``$CORR_DATA_DIR`` or ``data/processed``."""
    resolved = Path(data_dir) if data_dir is not None else Path(os.environ.get(DATA_DIR_ENV, DEFAULT_PROCESSED_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = load_store(resolved)
        app.state.store = store
        app.state.meta = build_meta(store)
        log.info("loaded %s: %d dates x %d tickers", resolved, store.n_dates, store.n_tickers)
        yield

    app = FastAPI(
        title="Stock Correlation Explorer API",
        version=__version__,
        description="Trailing-window return correlations computed on demand from memmapped returns.",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_methods=["GET"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "Content-Length"],
    )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> ORJSONResponse:
        log.exception("unhandled error on %s", request.url.path)
        return ORJSONResponse({"detail": "Internal server error"}, status_code=500)

    app.include_router(router)
    if (WEB_DIST / "index.html").is_file():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()
