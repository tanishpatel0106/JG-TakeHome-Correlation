"""HTTP endpoints. Every correlation is computed on demand from the memmapped
returns; nothing here ever holds an N x N matrix."""

from __future__ import annotations

import bisect
import io
from collections.abc import Iterator

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from corrlib.config import (
    DEFAULT_K,
    DOWNLOAD_BLOCK_ROWS,
    MATRIX_MAX_TICKERS,
    MATRIX_MIN_TICKERS,
    MAX_K,
)
from corrlib.corr import iter_full_blocks, one_vs_all, pair_series, prepare_window, subset_matrix, top_bottom_k
from corrlib.store import Store

from .deps import (
    StoreDep,
    api_error,
    check_iso_date,
    require_window,
    resolve_date,
    resolve_ticker,
    suggest_tickers,
)
from .schemas import (
    Breaks,
    ErrorResponse,
    HealthResponse,
    MatrixResponse,
    MetaResponse,
    Neighbor,
    PairResponse,
    RowResponse,
    StockResponse,
    SummaryResponse,
    TimelineResponse,
    finite_or_none,
    floats_or_none,
    matrix_or_none,
)

router = APIRouter(prefix="/api")

_ERRORS = {
    404: {"model": ErrorResponse, "description": "Unknown date or ticker"},
    422: {"model": ErrorResponse, "description": "Invalid input, or date before the first valid window"},
}


def build_meta(store: Store) -> MetaResponse:
    """Static metadata, computed once at startup."""
    default_index = -1
    usable = np.flatnonzero(store.n_valid >= 2)
    if usable.size:
        default_index = int(usable[-1])
    default_ticker = None
    if default_index >= 0:
        _, valid = prepare_window(store.returns, default_index, store.window)
        default_ticker = store.tickers[int(np.flatnonzero(valid)[0])]
    structural = store.manifest.get("structural", {})
    return MetaResponse(
        window=store.window,
        tickers=store.tickers,
        dates=store.dates,
        first_valid_date=store.dates[store.first_valid_index] if store.first_valid_index >= 0 else None,
        last_date=store.dates[-1],
        default_date=store.dates[default_index] if default_index >= 0 else None,
        default_ticker=default_ticker,
        breaks=Breaks(
            rebase_dates=structural.get("rebase_dates", []),
            placeholder_dates=structural.get("placeholder_dates", []),
        ),
        manifest=store.manifest,
    )


@router.get("/health", response_model=HealthResponse)
def health(store: StoreDep) -> HealthResponse:
    return HealthResponse(status="ok", n_tickers=store.n_tickers, n_dates=store.n_dates)


@router.get("/meta", response_model=MetaResponse)
def meta(request: Request) -> MetaResponse:
    return request.app.state.meta


@router.get("/stats/timeline", response_model=TimelineResponse)
def timeline(store: StoreDep) -> TimelineResponse:
    return TimelineResponse(
        dates=store.dates,
        mean_corr=floats_or_none(store.mean_corr),
        std_corr=floats_or_none(store.std_corr),
        n_valid=store.n_valid.tolist(),
    )


@router.get("/dates/{date}/summary", response_model=SummaryResponse, responses=_ERRORS)
def date_summary(date: str, store: StoreDep) -> SummaryResponse:
    t = resolve_date(store, date)
    require_window(store, t)
    n_valid = int(store.n_valid[t])
    return SummaryResponse(
        date=store.dates[t],
        n_valid=n_valid,
        n_pairs=n_valid * (n_valid - 1) // 2,
        mean=finite_or_none(store.mean_corr[t]),
        std=finite_or_none(store.std_corr[t]),
        bin_edges=store.hist_edges.tolist(),
        counts=store.hist[t].tolist(),
    )


@router.get("/corr/pair", response_model=PairResponse, responses=_ERRORS)
def pair(
    store: StoreDep,
    a: str = Query(..., description="First ticker"),
    b: str = Query(..., description="Second ticker"),
    start: str | None = Query(None, description="Inclusive start date YYYY-MM-DD"),
    end: str | None = Query(None, description="Inclusive end date YYYY-MM-DD"),
) -> PairResponse:
    i = resolve_ticker(store, a)
    j = resolve_ticker(store, b)
    lo, hi = 0, store.n_dates
    if start is not None:
        lo = bisect.bisect_left(store.dates, check_iso_date(start, "start"))
    if end is not None:
        hi = bisect.bisect_right(store.dates, check_iso_date(end, "end"))
    if start is not None and end is not None and start > end:
        raise api_error(422, "start must not be after end", start=start, end=end)
    series = pair_series(store.returns, i, j, store.window)
    return PairResponse(
        a=store.tickers[i],
        b=store.tickers[j],
        window=store.window,
        dates=store.dates[lo:hi],
        corr=floats_or_none(series[lo:hi]),
    )


@router.get("/corr/{date}/stock/{ticker}", response_model=StockResponse, responses=_ERRORS)
def stock_neighbors(
    date: str,
    ticker: str,
    store: StoreDep,
    k: int = Query(DEFAULT_K, ge=1, le=MAX_K, description="Neighbours on each side"),
) -> StockResponse:
    t = resolve_date(store, date)
    require_window(store, t)
    i = resolve_ticker(store, ticker)
    Z, valid = prepare_window(store.returns, t, store.window)
    row = one_vs_all(Z, valid, i)
    top, bottom = top_bottom_k(row, k, exclude=i)
    return StockResponse(
        date=store.dates[t],
        ticker=store.tickers[i],
        valid=bool(valid[i]),
        n_valid=int(valid.sum()),
        k=k,
        top=_neighbors(store, row, top),
        bottom=_neighbors(store, row, bottom),
    )


def _neighbors(store: Store, row: np.ndarray, idx: np.ndarray) -> list[Neighbor]:
    out = []
    for k in range(len(idx)):
        out.append(Neighbor(ticker=store.tickers[int(idx[k])], corr=float(row[idx[k]])))
    return out


@router.get("/corr/{date}/matrix", response_model=MatrixResponse, responses=_ERRORS)
def matrix(
    date: str,
    store: StoreDep,
    tickers: str = Query(..., description=f"Comma-separated, {MATRIX_MIN_TICKERS}-{MATRIX_MAX_TICKERS} unique tickers"),
) -> MatrixResponse:
    t = resolve_date(store, date)
    require_window(store, t)
    requested = _unique_tickers(tickers)
    if not MATRIX_MIN_TICKERS <= len(requested) <= MATRIX_MAX_TICKERS:
        raise api_error(
            422,
            f"Need between {MATRIX_MIN_TICKERS} and {MATRIX_MAX_TICKERS} unique tickers, got {len(requested)}",
            count=len(requested),
        )
    idx = np.empty(len(requested), dtype=np.intp)
    unknown: dict[str, list[str]] = {}  # every unknown ticker, not just the first
    for k in range(len(requested)):
        j = store.ticker_index.get(requested[k])
        if j is None:
            unknown[requested[k]] = suggest_tickers(store, requested[k])
        else:
            idx[k] = j
    if unknown:
        raise api_error(404, f"Unknown tickers: {', '.join(unknown)}", unknown=unknown)
    Z, valid = prepare_window(store.returns, t, store.window)
    C = subset_matrix(Z, valid, idx)
    return MatrixResponse(
        date=store.dates[t],
        tickers=requested,
        valid=valid[idx].tolist(),
        matrix=matrix_or_none(C),
    )


def _unique_tickers(raw: str) -> list[str]:
    """Split, trim, upper-case and de-duplicate (keeping first-seen order)."""
    parts = raw.split(",")
    seen: set[str] = set()
    out = []
    for k in range(len(parts)):
        ticker = parts[k].strip().upper()
        if ticker and ticker not in seen:
            seen.add(ticker)
            out.append(ticker)
    return out


@router.get("/corr/{date}/row/{ticker}", response_model=RowResponse, responses=_ERRORS)
def row(date: str, ticker: str, store: StoreDep) -> RowResponse:
    t = resolve_date(store, date)
    require_window(store, t)
    i = resolve_ticker(store, ticker)
    Z, valid = prepare_window(store.returns, t, store.window)
    return RowResponse(date=store.dates[t], ticker=store.tickers[i], corr=floats_or_none(one_vs_all(Z, valid, i)))


@router.get(
    "/corr/{date}/download",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/octet-stream": {}},
            "description": "float16 [N x N] .npy; row/column order is meta.tickers; NaN for invalid stocks",
        },
        **_ERRORS,
    },
)
def download(date: str, store: StoreDep) -> StreamingResponse:
    t = resolve_date(store, date)
    require_window(store, t)
    Z, valid = prepare_window(store.returns, t, store.window)
    n = store.n_tickers
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, {"descr": np.lib.format.dtype_to_descr(np.dtype("<f2")), "fortran_order": False, "shape": (n, n)}
    )
    head = header.getvalue()
    filename = f"corr_{store.dates[t]}_w{store.window}.npy"
    return StreamingResponse(
        _npy_body(head, Z, valid),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(head) + n * n * 2),
            "X-Row-Order": "meta.tickers",
        },
    )


def _npy_body(head: bytes, Z: np.ndarray, valid: np.ndarray) -> Iterator[bytes]:
    """Header, then the matrix row block by row block (O(block * N) memory)."""
    yield head
    for _, rows in iter_full_blocks(Z, valid, block=DOWNLOAD_BLOCK_ROWS):
        yield rows.astype("<f2").tobytes()
