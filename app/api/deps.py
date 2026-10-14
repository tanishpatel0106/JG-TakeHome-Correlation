"""Request dependencies: the loaded store, and date/ticker resolution with useful errors."""

from __future__ import annotations

import bisect
import difflib
import re
from datetime import date
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request

from corrlib.config import MAX_SUGGESTIONS
from corrlib.store import Store

from .registry import SnapshotNotFound, SnapshotRegistry

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_registry(request: Request) -> SnapshotRegistry:
    """The snapshot registry created by the app's lifespan handler."""
    return request.app.state.registry


def get_store(
    request: Request,
    snapshot: str | None = Query(None, description="Pin a snapshot id (default: the current one)"),
) -> Store:
    """The one Store this request reads (resolved exactly once; also sets ``X-Snapshot-Id``)."""
    try:
        store = get_registry(request).get(snapshot)
    except SnapshotNotFound as exc:
        raise api_error(
            404,
            "snapshot_not_found",
            f"Unknown snapshot {exc.snapshot_id!r}",
            snapshot=exc.snapshot_id,
            available=exc.available,
        ) from None
    request.state.snapshot_id = store.snapshot_id
    return store


StoreDep = Annotated[Store, Depends(get_store)]


def api_error(
    status: int, code: str, message: str, headers: dict[str, str] | None = None, **context: object
) -> HTTPException:
    """HTTPException whose JSON body is ``{"detail": {"code": ..., "message": ..., **context}}``."""
    return HTTPException(status_code=status, detail={"code": code, "message": message, **context}, headers=headers)


def check_iso_date(value: str, field: str = "date") -> str:
    """Validate ``YYYY-MM-DD`` (a real calendar date). 422 otherwise."""
    if not _ISO_DATE.match(value):
        raise api_error(422, "invalid_date", f"Invalid {field} {value!r}: expected YYYY-MM-DD", **{field: value})
    try:
        date.fromisoformat(value)
    except ValueError:
        raise api_error(422, "invalid_date", f"Invalid {field} {value!r}: not a calendar date", **{field: value}) from None
    return value


def resolve_date(store: Store, value: str) -> int:
    """Row index of a trading date. 422 if malformed, 404 (with neighbours) if not traded."""
    check_iso_date(value)
    t = store.date_index.get(value)
    if t is None:
        pos = bisect.bisect_left(store.dates, value)
        nearest = []
        if pos > 0:
            nearest.append(store.dates[pos - 1])
        if pos < store.n_dates:
            nearest.append(store.dates[pos])
        raise api_error(404, "date_not_traded", f"{value} is not a trading date in the data", date=value, nearest=nearest)
    return t


def require_window(store: Store, t: int) -> None:
    """422 if date index ``t`` is before the first date with a valid window."""
    if store.first_valid_index < 0 or t < store.first_valid_index:
        first = store.dates[store.first_valid_index] if store.first_valid_index >= 0 else None
        raise api_error(
            422,
            "date_before_first_window",
            f"{store.dates[t]} is before the first date with a full {store.window}-return window",
            date=store.dates[t],
            first_valid_date=first,
        )


def suggest_tickers(store: Store, query: str) -> list[str]:
    """Up to MAX_SUGGESTIONS tickers that look like ``query``.

    Up to half are tickers sharing the longest prefix with the query (typos at
    the end, e.g. ``MSFX`` -> ``MSFT``); the rest are ``difflib`` close matches
    (transpositions, e.g. ``APPL`` -> ``AAPL``).
    """
    found: list[str] = []
    prefix_quota = MAX_SUGGESTIONS // 2
    for length in range(len(query), 0, -1):
        prefix = query[:length]
        j = bisect.bisect_left(store.tickers, prefix)
        while j < store.n_tickers and store.tickers[j].startswith(prefix) and len(found) < prefix_quota:
            if store.tickers[j] != query:
                found.append(store.tickers[j])
            j += 1
        if found:
            break
    close = difflib.get_close_matches(query, store.tickers, n=MAX_SUGGESTIONS, cutoff=0.6)
    for k in range(len(close)):
        if len(found) >= MAX_SUGGESTIONS:
            break
        if close[k] not in found:
            found.append(close[k])
    return found


def resolve_ticker(store: Store, value: str) -> int:
    """Column index of a ticker (case-insensitive).

    404 ``unknown_ticker`` with suggestions if unknown; 409 ``ambiguous_ticker``
    if several tickers differ only by case and none matches exactly.
    """
    exact = store.ticker_index.get(value.strip())
    if exact is not None:
        return exact
    key = value.strip().upper()
    j = store.ticker_index.get(key)
    if j is not None:
        return j
    candidates = []
    for k in range(store.n_tickers):
        if store.tickers[k].upper() == key:
            candidates.append(store.tickers[k])
    if len(candidates) == 1:
        return store.ticker_index[candidates[0]]
    if candidates:
        raise api_error(409, "ambiguous_ticker", f"Ticker {value!r} is ambiguous", ticker=value, candidates=candidates)
    raise api_error(
        404, "unknown_ticker", f"Unknown ticker {value!r}", ticker=value, suggestions=suggest_tickers(store, key)
    )
