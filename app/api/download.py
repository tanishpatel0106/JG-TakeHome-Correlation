"""Full-matrix downloads: streamed ``.npz`` (corr + tickers + meta) or raw ``.npy``.

The matrix is computed block by block (``DOWNLOAD_BLOCK_ROWS`` rows of float64,
cast to the requested dtype), so memory stays O(block x N). Between blocks the
stream checks ``request.is_disconnected()`` and stops computing once the
client is gone. At most ``MAX_CONCURRENT_DOWNLOADS`` run at once (429 beyond).

The npz is a ZIP_STORED archive written through :mod:`zipfile` to a non-seekable
sink. Its layout does not depend on the matrix values, so its exact size is
known up front (a dry run writes everything except the matrix body).
"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import numpy as np
from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from corrlib import config
from corrlib.corr import iter_full_blocks, window_status
from corrlib.store import Store

from .deps import api_error

DOWNLOAD_BLOCK_ROWS: int = config.DOWNLOAD_BLOCK_ROWS
MAX_CONCURRENT_DOWNLOADS: int = getattr(config, "MAX_CONCURRENT_DOWNLOADS", 2)
RETRY_AFTER_SECONDS = 5
#: Declared max abs error of each export dtype vs the float64 values of the JSON API.
TOLERANCE = {"float32": 1e-6, "float16": 1e-3}
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # fixed, so the archive layout is reproducible


@dataclass
class DownloadPlan:
    """Everything about a download except the matrix body."""

    dtype: str
    fmt: str
    filename: str
    meta: dict[str, Any]
    #: ``.npy`` header of the [N x N] matrix
    head: bytes
    #: bytes of the matrix body (N * N * itemsize)
    body_bytes: int
    tickers_npy: bytes
    meta_npy: bytes
    size: int
    size_exact: bool


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _download_meta(store: Store, t: int, dtype: str) -> dict[str, Any]:
    n = store.n_tickers
    policy = store.policy
    return {
        "date": store.dates[t],
        "window": store.window,
        "snapshot_id": store.snapshot_id,
        "dtype": dtype,
        "shape": [n, n],
        "ticker_order": "rows and columns follow tickers.npy (= /api/meta tickers)",
        "invalid_semantics": "a stock without a valid window has an all-NaN row and column; never 0",
        "diagonal_semantics": "1.0 for valid stocks, NaN for invalid ones",
        "tolerance": {"abs": TOLERANCE[dtype], "versus": "float64 values of the JSON API"},
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "view": store.view,
        "policy": {
            "window": policy.get("window"),
            "min_valid_returns": policy.get("min_valid_returns"),
            "eps": policy.get("eps"),
            "reason_codes": policy.get("reason_codes"),
        },
    }


class _Sink:
    """Write-only, non-seekable byte collector that ``zipfile`` streams into."""

    def __init__(self, keep: bool = True) -> None:
        self.keep = keep
        self.chunks: list[bytes] = []
        self.count = 0

    def write(self, data: bytes) -> int:
        if self.keep:
            self.chunks.append(bytes(data))
        self.count += len(data)
        return len(data)

    def flush(self) -> None:
        pass

    def take(self) -> bytes:
        out = b"".join(self.chunks)
        self.chunks = []
        return out


def _zinfo(name: str, size: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.file_size = size  # decides zip64 up front, identically in the dry run and the real stream
    return info


def plan_download(store: Store, t: int, dtype: str, fmt: str) -> DownloadPlan:
    """Headers, metadata, filename and exact size of the ``(date t, dtype, fmt)`` download."""
    n = store.n_tickers
    np_dtype = np.dtype("<f4" if dtype == "float32" else "<f2")
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": np.lib.format.dtype_to_descr(np_dtype), "fortran_order": False, "shape": (n, n)}
    )
    head = header.getvalue()
    body_bytes = n * n * np_dtype.itemsize
    meta = _download_meta(store, t, dtype)
    tickers_npy = _npy_bytes(np.array(store.tickers, dtype=str))
    meta_npy = _npy_bytes(np.array(json.dumps(meta, sort_keys=True)))
    filename = f"corr_{store.dates[t]}_w{store.window}_{dtype}_{store.snapshot_id}.{fmt}"
    if fmt == "npy":
        size, exact = len(head) + body_bytes, True
    else:
        # Dry run without the matrix body: a stored zip's layout is independent of the content.
        sink = _Sink(keep=False)
        with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED) as zf:  # type: ignore[arg-type]
            with zf.open(_zinfo("corr.npy", len(head) + body_bytes), "w") as f:
                f.write(head)
            zf.writestr(_zinfo("tickers.npy", len(tickers_npy)), tickers_npy)
            zf.writestr(_zinfo("meta.npy", len(meta_npy)), meta_npy)
        size = sink.count + body_bytes
        exact = size < zipfile.ZIP64_LIMIT  # beyond it, central-directory zip64 extras add a few bytes
    return DownloadPlan(dtype, fmt, filename, meta, head, body_bytes, tickers_npy, meta_npy, size, exact)


async def _blocks(request: Request, Z: np.ndarray, valid: np.ndarray, dtype: str) -> AsyncIterator[bytes]:
    """Matrix body, block by block; stops when the client disconnects."""
    blocks = iter_full_blocks(Z, valid, block=DOWNLOAD_BLOCK_ROWS, dtype=np.float64)
    np_dtype = "<f4" if dtype == "float32" else "<f2"
    while True:
        if await request.is_disconnected():
            return
        item = await asyncio.to_thread(next, blocks, None)
        if item is None:
            return
        yield item[1].astype(np_dtype).tobytes()


async def _npy_stream(request: Request, plan: DownloadPlan, Z: np.ndarray, valid: np.ndarray) -> AsyncIterator[bytes]:
    yield plan.head
    async for chunk in _blocks(request, Z, valid, plan.dtype):
        yield chunk


async def _npz_stream(request: Request, plan: DownloadPlan, Z: np.ndarray, valid: np.ndarray) -> AsyncIterator[bytes]:
    sink = _Sink()
    zf = zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED)  # type: ignore[arg-type]
    with zf.open(_zinfo("corr.npy", len(plan.head) + plan.body_bytes), "w") as f:
        f.write(plan.head)
        async for chunk in _blocks(request, Z, valid, plan.dtype):
            f.write(chunk)
            yield sink.take()
        if await request.is_disconnected():
            return
    zf.writestr(_zinfo("tickers.npy", len(plan.tickers_npy)), plan.tickers_npy)
    zf.writestr(_zinfo("meta.npy", len(plan.meta_npy)), plan.meta_npy)
    zf.close()
    yield sink.take()


class _SlotResponse(StreamingResponse):
    """StreamingResponse that frees its download slot however the response ends."""

    def __init__(self, release: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


def start_download(request: Request, store: Store, t: int, plan: DownloadPlan) -> StreamingResponse:
    """Take a download slot (429 if none is free) and stream the planned file."""
    state = request.app.state
    if state.active_downloads >= MAX_CONCURRENT_DOWNLOADS:
        raise api_error(
            429,
            "too_many_downloads",
            f"At most {MAX_CONCURRENT_DOWNLOADS} downloads at a time; retry shortly",
            headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            limit=MAX_CONCURRENT_DOWNLOADS,
        )
    Z, valid, _ = window_status(store.returns, t, store.window)
    state.active_downloads += 1
    released = [False]

    def release() -> None:
        if not released[0]:
            released[0] = True
            state.active_downloads -= 1

    headers = {
        "Content-Disposition": f'attachment; filename="{plan.filename}"',
        "X-Row-Order": "meta.tickers",
    }
    if plan.size_exact:
        headers["Content-Length"] = str(plan.size)
    if plan.fmt == "npy":
        body = _npy_stream(request, plan, Z, valid)
        media_type = "application/octet-stream"
    else:
        body = _npz_stream(request, plan, Z, valid)
        media_type = "application/zip"
    return _SlotResponse(release, body, media_type=media_type, headers=headers)
