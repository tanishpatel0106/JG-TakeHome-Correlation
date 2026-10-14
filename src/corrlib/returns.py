"""Long prices -> wide [T x N] price matrix -> simple returns.

Pure functions only: arrays and frames in, arrays out. No file I/O, no global
state. Missing data stays NaN; nothing is forward-filled or invented.

Besides the textbook steps this module detects *uniform-price days*: dates on
which every ticker has exactly the same price (the data contains 20.0 "seed"
days and a day with every price at 1.0). Real prices are never identical
across thousands of stocks, so such a day is either

* a **re-base**: every price path restarts at the seed value and the next day
  continues from it. The seed is a valid base for the next return, but the
  return *into* the seed is not a market move, so it is a structural break; or
* a **placeholder**: the next day continues from the day *before*, so the
  uniform values are fill-ins and the day's prices are invalid.

See :func:`classify_uniform_days` for the test used.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd


def ticker_universe(frames: Sequence[pd.DataFrame]) -> list[str]:
    """Sorted union of the ``ticker`` column over all cleaned daily frames."""
    seen: set[str] = set()
    for t in range(len(frames)):
        seen.update(frames[t]["ticker"].tolist())
    return sorted(seen)


def to_wide(
    frames: Sequence[pd.DataFrame],
    tickers: Sequence[str],
    dates: Sequence[date],
) -> np.ndarray:
    """Assemble cleaned daily frames into a wide price matrix.

    Parameters
    ----------
    frames:
        ``frames[t]`` is the cleaned frame for ``dates[t]`` with columns
        ``ticker`` (unique) and ``price`` (float64, NaN for invalid).
    tickers:
        Column order (the fixed ticker universe). Every frame ticker must be in it.
    dates:
        Row labels; only the length is used (and in error messages).

    Returns
    -------
    np.ndarray
        float64 ``[T x N]``. A cell is NaN when the ticker has no row that day
        or its price was invalid. Filled day by day into a preallocated array.
    """
    if len(frames) != len(dates):
        raise ValueError(f"{len(frames)} frames but {len(dates)} dates")
    column: dict[str, int] = {}
    for j in range(len(tickers)):
        column[tickers[j]] = j

    prices = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
    for t in range(len(frames)):
        frame = frames[t]
        if not frame["ticker"].is_unique:
            raise ValueError(f"{dates[t]}: duplicate tickers in a cleaned frame")
        cols = frame["ticker"].map(column)
        if cols.isna().any():
            unknown = frame["ticker"][cols.isna()].tolist()[:5]
            raise ValueError(f"{dates[t]}: tickers not in the universe: {unknown}")
        prices[t, cols.to_numpy(dtype=np.int64)] = frame["price"].to_numpy(dtype=np.float64)
    return prices


def find_uniform_days(prices: np.ndarray, min_coverage: float = 0.5) -> np.ndarray:
    """Flag dates on which every finite price is identical.

    Parameters
    ----------
    prices:
        float ``[T x N]``, NaN for missing.
    min_coverage:
        Fraction of tickers that must have a finite price for the day to be
        judged (and at least 2), so a nearly empty day is never flagged.

    Returns
    -------
    np.ndarray
        bool ``[T]``.
    """
    n_tickers = prices.shape[1]
    finite = np.isfinite(prices)
    n_finite = finite.sum(axis=1)
    high = np.where(finite, prices, -np.inf).max(axis=1)
    low = np.where(finite, prices, np.inf).min(axis=1)
    enough = n_finite >= max(2, math.ceil(min_coverage * n_tickers))
    return enough & (high == low)


def classify_uniform_days(
    prices: np.ndarray, uniform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split uniform-price days into re-bases and placeholders.

    For a uniform day ``t`` with neighbours on both sides, compare the
    cross-sectional median of ``|log(P[t+1] / P[t])|`` (next day continues from
    the uniform value) with that of ``|log(P[t+1] / P[t-1])|`` (next day
    continues from the previous day). The smaller one says where the price
    paths really continue from.

    * smaller from ``P[t]`` -> **re-base** (keep prices; break the return into ``t``)
    * smaller from ``P[t-1]`` -> **placeholder** (prices on ``t`` are invalid)

    Edge cases: ``t == 0`` is a re-base (the series starts there). The last
    date cannot be tested and is conservatively a placeholder. A day whose
    neighbours share no finite prices is also a placeholder.

    Returns
    -------
    (rebase, placeholder)
        Two disjoint bool ``[T]`` masks whose union is ``uniform``.
    """
    n_dates = prices.shape[0]
    rebase = np.zeros(n_dates, dtype=bool)
    placeholder = np.zeros(n_dates, dtype=bool)
    days = np.flatnonzero(uniform)
    for k in range(len(days)):
        t = int(days[k])
        if t == 0:
            rebase[t] = True
            continue
        if t == n_dates - 1:
            placeholder[t] = True
            continue
        both = np.isfinite(prices[t - 1]) & np.isfinite(prices[t]) & np.isfinite(prices[t + 1])
        if not both.any():
            placeholder[t] = True
            continue
        from_uniform = np.median(np.abs(np.log(prices[t + 1, both] / prices[t, both])))
        from_previous = np.median(np.abs(np.log(prices[t + 1, both] / prices[t - 1, both])))
        if from_uniform <= from_previous:
            rebase[t] = True
        else:
            placeholder[t] = True
    return rebase, placeholder


def simple_returns(prices: np.ndarray, breaks: np.ndarray | None = None) -> np.ndarray:
    """Simple returns ``R[t] = P[t] / P[t-1] - 1``.

    Parameters
    ----------
    prices:
        float ``[T x N]``; positive or NaN.
    breaks:
        Optional bool ``[T]``. Where ``breaks[t]`` is True, ``R[t]`` is NaN for
        every ticker (the move from ``t-1`` to ``t`` is not a market return,
        e.g. a re-base to a seed price).

    Returns
    -------
    np.ndarray
        float64 ``[T x N]``. ``R[0]`` is all NaN. ``R[t]`` is NaN if either price
        is NaN (no forward fill across gaps) or the result is not finite.
    """
    p = np.asarray(prices, dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = p[1:] / p[:-1] - 1.0
    out[~np.isfinite(out)] = np.nan
    if breaks is not None:
        out[np.asarray(breaks, dtype=bool)] = np.nan
    return out
