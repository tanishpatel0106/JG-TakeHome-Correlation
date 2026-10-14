"""HTTP endpoints. Every correlation is computed on demand from the memmapped
returns; nothing here ever holds an N x N matrix."""

from __future__ import annotations

import bisect
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from corrlib import snapshots
from corrlib.config import (
    DEFAULT_K,
    MATRIX_MAX_TICKERS,
    MATRIX_MIN_TICKERS,
    MAX_K,
    REASON_NAMES,
)
from corrlib.corr import one_vs_all, pair_series, subset_matrix, top_bottom_k, window_status
from corrlib.store import Store

from . import download as dl
from .deps import (
    StoreDep,
    get_registry,
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
    404: {"model": ErrorResponse, "description": "Unknown date, ticker or snapshot"},
    422: {"model": ErrorResponse, "description": "Invalid input, or date before the first valid window"},
}


def _reason_names(codes: np.ndarray) -> list[str]:
    out = []
    for k in range(len(codes)):
        out.append(REASON_NAMES[int(codes[k])])
    return out


def build_meta(store: Store) -> MetaResponse:
    """Static metadata, computed once at startup."""
    default_index = -1
    usable = np.flatnonzero(store.n_valid >= 2)
    if usable.size:
        default_index = int(usable[-1])
    default_ticker = None
    if default_index >= 0:
        _, valid, _ = window_status(store.returns, default_index, store.window)
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
        snapshot=store.snapshot,
        view=store.view,
        dataset=store.dataset,
        policy=store.policy,
        anomalies=store.anomalies,
    )


@router.get("/health", response_model=HealthResponse)
def health(store: StoreDep) -> HealthResponse:
    return HealthResponse(status="ok", snapshot_id=store.snapshot_id, n_tickers=store.n_tickers, n_dates=store.n_dates)


@router.get("/snapshots")
def list_snapshots(request: Request) -> dict[str, Any]:
    """Every published snapshot (seq order) and which one is current."""
    registry = get_registry(request)
    request.state.snapshot_id = registry.current_id
    return {"current": registry.current_id, "snapshots": registry.describe_all()}


@router.get("/snapshots/current")
def current_snapshot(request: Request) -> dict[str, Any]:
    """The current snapshot (cheap; the UI polls it)."""
    store = get_registry(request).get(None)
    request.state.snapshot_id = store.snapshot_id
    return {
        "id": store.snapshot_id,
        "seq": store.snapshot.get("seq"),
        "created_at": store.snapshot.get("created_at"),
        "data_as_of": store.dates[-1] if store.dates else None,
    }


@router.post("/snapshots/reload")
def reload_snapshots(request: Request) -> dict[str, Any]:
    """Re-read CURRENT now (the background watcher does this every CORR_RELOAD_SECONDS)."""
    try:
        previous, current = get_registry(request).reload()
    except snapshots.SnapshotError as exc:
        raise api_error(500, "internal_error", f"Cannot reload: {exc}") from None
    request.state.snapshot_id = current
    return {"previous": previous, "current": current, "changed": previous != current}


@router.get("/meta", response_model=MetaResponse, responses=_ERRORS)
def meta(request: Request, store: StoreDep) -> MetaResponse:
    return get_registry(request).meta(store, build_meta)


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
        status="ok" if n_valid >= 2 else "no_valid_pairs",
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
        raise api_error(422, "invalid_input", "start must not be after end", start=start, end=end)
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
    Z, valid, reason = window_status(store.returns, t, store.window)
    row = one_vs_all(Z, valid, i)
    top, bottom = top_bottom_k(row, k, exclude=i)
    return StockResponse(
        date=store.dates[t],
        ticker=store.tickers[i],
        valid=bool(valid[i]),
        reason=REASON_NAMES[int(reason[i])],
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
            "invalid_input",
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
        raise api_error(404, "unknown_tickers", f"Unknown tickers: {', '.join(unknown)}", unknown=unknown)
    Z, valid, reason = window_status(store.returns, t, store.window)
    C = subset_matrix(Z, valid, idx)
    return MatrixResponse(
        date=store.dates[t],
        tickers=requested,
        valid=valid[idx].tolist(),
        reasons=_reason_names(reason[idx]),
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
    Z, valid, reason = window_status(store.returns, t, store.window)
    return RowResponse(
        date=store.dates[t],
        ticker=store.tickers[i],
        valid=bool(valid[i]),
        reason=REASON_NAMES[int(reason[i])],
        n_valid=int(valid.sum()),
        corr=floats_or_none(one_vs_all(Z, valid, i)),
    )


DtypeQuery = Query("float32", description="float32 (abs error <= 1e-6) or float16 (abs error <= 1e-3)")
FormatQuery = Query("npz", alias="format", description="npz (corr + tickers + meta) or npy (matrix only)")


@router.get("/corr/{date}/download/info", responses=_ERRORS)
def download_info(
    date: str,
    store: StoreDep,
    dtype: Literal["float32", "float16"] = DtypeQuery,
    fmt: Literal["npz", "npy"] = FormatQuery,
) -> dict[str, Any]:
    """Size, filename and metadata of a download, without computing it."""
    t = resolve_date(store, date)
    require_window(store, t)
    plan = dl.plan_download(store, t, dtype, fmt)
    return {
        "date": store.dates[t],
        "window": store.window,
        "snapshot_id": store.snapshot_id,
        "dtype": dtype,
        "format": fmt,
        "shape": [store.n_tickers, store.n_tickers],
        "bytes": plan.size,
        "bytes_exact": plan.size_exact,
        "filename": plan.filename,
        "ticker_order": plan.meta["ticker_order"],
        "invalid_semantics": plan.meta["invalid_semantics"],
        "diagonal_semantics": plan.meta["diagonal_semantics"],
        "tolerance": plan.meta["tolerance"],
    }


@router.get(
    "/corr/{date}/download",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"application/zip": {}, "application/octet-stream": {}},
            "description": "[N x N] matrix in meta.tickers order (npz: corr.npy + tickers.npy + meta.npy); "
            "NaN for invalid stocks",
        },
        429: {"model": ErrorResponse, "description": "Too many concurrent downloads"},
        **_ERRORS,
    },
)
async def download(
    date: str,
    request: Request,
    store: StoreDep,
    dtype: Literal["float32", "float16"] = DtypeQuery,
    fmt: Literal["npz", "npy"] = FormatQuery,
) -> StreamingResponse:
    t = resolve_date(store, date)
    require_window(store, t)
    plan = dl.plan_download(store, t, dtype, fmt)
    return dl.start_download(request, store, t, plan)
