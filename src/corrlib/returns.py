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

See :func:`classify_uniform_days` for the test used. The test looks at the
*next* day, so it is a retrospective decision; a uniform *last* date cannot be
tested and stays **unresolved** (its prices are excluded until a later
snapshot adds the next day). :func:`anomaly_records` turns every decision into
an explicit record (evidence, observation dates, look-ahead, action, affected
windows), :func:`apply_overrides` lets an operator confirm or change a
classification, and :func:`derive_returns` applies the records to the raw
prices. The raw prices are never modified.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import ANOMALY_RULE

#: Classifications of a uniform-price day.
REBASE = "rebase"
PLACEHOLDER = "placeholder"
UNRESOLVED = "unresolved"

#: Status of a classification.
STATUS_INFERRED = "inferred"
STATUS_CONFIRMED = "confirmed"
STATUS_UNRESOLVED = "unresolved"

#: What was done to the data because of a record.
ACTION_NONE = "none"
ACTION_RETURN_NULLED = "return_into_day_nulled"
ACTION_PRICES_NULLED = "prices_nulled"
ACTION_PENDING = "excluded_pending_next_day"

#: Classifications an override may set.
OVERRIDE_CLASSIFICATIONS: tuple[str, ...] = (REBASE, PLACEHOLDER)

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class OverrideError(ValueError):
    """An overrides document is malformed or targets a day that is not a detected uniform day."""


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


def continuity_evidence(prices: np.ndarray, t: int) -> tuple[float, float, int]:
    """The continuity test numbers for an interior day ``t`` (``0 < t < T - 1``).

    Over the tickers with finite prices on ``t - 1``, ``t`` and ``t + 1``:

    * ``from_uniform``: median of ``|log(P[t+1] / P[t])|``
    * ``from_previous``: median of ``|log(P[t+1] / P[t-1])|``

    Returns
    -------
    (from_uniform, from_previous, n_compared)
        Both medians are NaN when ``n_compared == 0`` or ``t`` is the first or
        last index (the test needs a day on each side).
    """
    n_dates = prices.shape[0]
    if t <= 0 or t >= n_dates - 1:
        return float("nan"), float("nan"), 0
    both = np.isfinite(prices[t - 1]) & np.isfinite(prices[t]) & np.isfinite(prices[t + 1])
    n_compared = int(both.sum())
    if n_compared == 0:
        return float("nan"), float("nan"), 0
    from_uniform = float(np.median(np.abs(np.log(prices[t + 1, both] / prices[t, both]))))
    from_previous = float(np.median(np.abs(np.log(prices[t + 1, both] / prices[t - 1, both]))))
    return from_uniform, from_previous, n_compared


def infer_uniform_day(prices: np.ndarray, t: int) -> str:
    """Rule classification of one uniform day: ``"rebase"``, ``"placeholder"`` or ``"unresolved"``.

    * ``t == 0`` -> re-base (the series starts there; ``R[0]`` is undefined anyway).
    * ``t == T - 1`` (and ``T > 1``) -> unresolved: the next day is needed.
    * otherwise the continuity test (:func:`continuity_evidence`): the smaller
      median says where the price paths continue from; ``from_uniform <=
      from_previous`` is a re-base, else a placeholder. A day whose neighbours
      share no finite prices is conservatively a placeholder.
    """
    n_dates = prices.shape[0]
    if t == 0:
        return REBASE
    if t == n_dates - 1:
        return UNRESOLVED
    from_uniform, from_previous, n_compared = continuity_evidence(prices, t)
    if n_compared == 0:
        return PLACEHOLDER
    return REBASE if from_uniform <= from_previous else PLACEHOLDER


def classify_uniform_days(
    prices: np.ndarray, uniform: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split uniform-price days into re-bases, placeholders and unresolved days.

    Applies :func:`infer_uniform_day` to every flagged day (the rule only; no
    overrides).

    * re-base -> keep prices; break the return into ``t``
    * placeholder -> the prices on ``t`` are invalid
    * unresolved (uniform last date) -> excluded until the next day is known

    Returns
    -------
    (rebase, placeholder, unresolved)
        Three disjoint bool ``[T]`` masks whose union is ``uniform``.
    """
    n_dates = prices.shape[0]
    rebase = np.zeros(n_dates, dtype=bool)
    placeholder = np.zeros(n_dates, dtype=bool)
    unresolved = np.zeros(n_dates, dtype=bool)
    days = np.flatnonzero(uniform)
    for k in range(len(days)):
        t = int(days[k])
        verdict = infer_uniform_day(prices, t)
        if verdict == REBASE:
            rebase[t] = True
        elif verdict == PLACEHOLDER:
            placeholder[t] = True
        else:
            unresolved[t] = True
    return rebase, placeholder, unresolved


# --- anomaly records ----------------------------------------------------------


def _iso(day: date | str) -> str:
    return day if isinstance(day, str) else day.isoformat()


def _round(value: float) -> float | None:
    return round(float(value), 6) if math.isfinite(value) else None


def action_for(classification: str, t: int) -> str:
    """The data action implied by a final classification of the uniform day at index ``t``."""
    if classification == REBASE:
        return ACTION_NONE if t == 0 else ACTION_RETURN_NULLED
    if classification == PLACEHOLDER:
        return ACTION_PRICES_NULLED
    return ACTION_PENDING


def affected_outputs(
    action: str, t: int, dates: Sequence[date | str], window: int
) -> dict[str, Any]:
    """Dates whose returns / prices / windows an action changes.

    * ``return_into_day_nulled``: ``R[t]`` -> window ends ``t .. t+W-1``.
    * ``prices_nulled`` / ``excluded_pending_next_day``: ``P[t]`` -> ``R[t]`` and
      ``R[t+1]`` -> window ends ``t .. t+W``.
    * ``none``: nothing.

    Window ends are clipped to ``[W, T-1]`` (earlier dates have no window).

    Returns
    -------
    dict
        ``{"returns_nulled_dates": [...], "prices_nulled_date": str | None,
        "window_end_dates": {"first", "last", "count"}}``.
    """
    n_dates = len(dates)
    returns_nulled: list[str] = []
    prices_nulled = None
    last_return = -1
    if action == ACTION_RETURN_NULLED:
        last_return = t
    elif action in (ACTION_PRICES_NULLED, ACTION_PENDING):
        prices_nulled = _iso(dates[t])
        last_return = min(t + 1, n_dates - 1)
    for s in range(t, last_return + 1):
        if s >= 1:
            returns_nulled.append(_iso(dates[s]))
    first_end, last_end = max(t, window), min(last_return + window - 1, n_dates - 1)
    if last_return < 0 or first_end > last_end:
        ends: dict[str, Any] = {"first": None, "last": None, "count": 0}
    else:
        ends = {"first": _iso(dates[first_end]), "last": _iso(dates[last_end]), "count": last_end - first_end + 1}
    return {"returns_nulled_dates": returns_nulled, "prices_nulled_date": prices_nulled, "window_end_dates": ends}


def anomaly_record(
    prices_raw: np.ndarray, dates: Sequence[date | str], t: int, window: int
) -> dict[str, Any]:
    """The record of the uniform day at index ``t`` as decided by the rule (no override).

    Parameters
    ----------
    prices_raw:
        float ``[T x N]`` per-file-cleaned prices, NaN for missing (never modified).
    dates:
        ``T`` dates (``date`` or ISO strings).
    t:
        Index of a day flagged by :func:`find_uniform_days`.
    window:
        Window length W (for the affected window ends).

    Returns
    -------
    dict
        JSON-ready record: ``date, index, uniform_value, n_prices,
        classification, status, rule, evidence, observation_dates_used,
        look_ahead, override, action, affected``.
        ``evidence`` holds ``previous_date, next_date,
        median_abs_log_from_uniform, median_abs_log_from_previous, n_compared,
        rule_classification, note`` (medians null when not computed).
    """
    n_dates = prices_raw.shape[0]
    row = prices_raw[t]
    finite = row[np.isfinite(row)]
    classification = infer_uniform_day(prices_raw, t)
    from_uniform, from_previous, n_compared = continuity_evidence(prices_raw, t)
    previous_date = _iso(dates[t - 1]) if t > 0 else None
    next_date = _iso(dates[t + 1]) if t < n_dates - 1 else None
    if t == 0:
        note = "first date: the price paths start here and R[0] is undefined, so there is nothing to break"
        observed = [_iso(dates[t])]
        look_ahead = False
    elif t == n_dates - 1:
        note = (
            "last date: the next day's prices are needed to tell a re-base from a placeholder, "
            "so the day's prices are excluded until a later snapshot adds the next day"
        )
        observed = [_iso(dates[t])]
        look_ahead = False
    else:
        observed = [_iso(dates[t - 1]), _iso(dates[t]), _iso(dates[t + 1])]
        look_ahead = True
        if n_compared == 0:
            note = "no ticker is priced on the previous, uniform and next day: conservatively a placeholder"
        elif classification == REBASE:
            note = (
                f"the next day continues from the uniform value (median |log P[t+1]/P[t]| "
                f"{from_uniform:.4f} <= {from_previous:.4f} from the previous day): re-base"
            )
        else:
            note = (
                f"the next day continues from the previous day (median |log P[t+1]/P[t-1]| "
                f"{from_previous:.4f} < {from_uniform:.4f} from the uniform value): placeholder"
            )
    status = STATUS_UNRESOLVED if classification == UNRESOLVED else STATUS_INFERRED
    action = action_for(classification, t)
    return {
        "date": _iso(dates[t]),
        "index": int(t),
        "uniform_value": float(finite[0]) if finite.size else None,
        "n_prices": int(finite.size),
        "classification": classification,
        "status": status,
        "rule": ANOMALY_RULE,
        "evidence": {
            "previous_date": previous_date,
            "next_date": next_date,
            "median_abs_log_from_uniform": _round(from_uniform),
            "median_abs_log_from_previous": _round(from_previous),
            "n_compared": n_compared,
            "rule_classification": classification,
            "note": note,
        },
        "observation_dates_used": observed,
        "look_ahead": look_ahead,
        "override": None,
        "action": action,
        "affected": affected_outputs(action, t, dates, window),
    }


def anomaly_records(
    prices_raw: np.ndarray,
    dates: Sequence[date | str],
    window: int,
    overrides: Sequence[Mapping[str, Any]] | None = None,
    uniform: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Records for every uniform-price day, with overrides applied.

    Parameters
    ----------
    prices_raw:
        float ``[T x N]`` per-file-cleaned prices (never modified).
    dates:
        ``T`` dates (``date`` or ISO strings).
    window:
        Window length W.
    overrides:
        Validated override entries (see :func:`parse_overrides`), or None.
    uniform:
        bool ``[T]`` from :func:`find_uniform_days`; computed when omitted.

    Returns
    -------
    list[dict]
        One record per uniform day in date order (see :func:`anomaly_record`).

    Raises
    ------
    OverrideError
        If an override targets a date that is not a detected uniform day.
    """
    if uniform is None:
        uniform = find_uniform_days(prices_raw)
    days = np.flatnonzero(uniform)
    records = []
    for k in range(len(days)):
        records.append(anomaly_record(prices_raw, dates, int(days[k]), window))
    if overrides:
        records = apply_overrides(records, overrides, dates, window)
    return records


def parse_overrides(document: object) -> list[dict[str, str]]:
    """Validate an overrides document and return its entries.

    Expected shape: ``{"overrides": [{"date": "YYYY-MM-DD", "classification":
    "rebase" | "placeholder", "reason": "...", "author": "..."}, ...]}``.
    Dates must be unique; ``reason`` and ``author`` must be non-empty.

    Raises
    ------
    OverrideError
        On any structural problem (the message names the entry).
    """
    if not isinstance(document, dict) or not isinstance(document.get("overrides"), list):
        raise OverrideError('overrides must be a JSON object {"overrides": [...]}')
    entries = document["overrides"]
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for k in range(len(entries)):
        entry = entries[k]
        if not isinstance(entry, dict):
            raise OverrideError(f"overrides[{k}] is not an object")
        day = entry.get("date")
        if not isinstance(day, str) or not _ISO_DATE.match(day) or not _is_calendar_date(day):
            raise OverrideError(f"overrides[{k}].date {day!r} is not a YYYY-MM-DD date")
        if day in seen:
            raise OverrideError(f"overrides[{k}]: {day} is overridden twice")
        seen.add(day)
        classification = entry.get("classification")
        if classification not in OVERRIDE_CLASSIFICATIONS:
            raise OverrideError(
                f"overrides[{k}].classification {classification!r} must be one of {list(OVERRIDE_CLASSIFICATIONS)}"
            )
        texts = {}
        for name in ("reason", "author"):
            value = entry.get(name)
            if not isinstance(value, str) or not value.strip():
                raise OverrideError(f"overrides[{k}].{name} must be a non-empty string")
            texts[name] = value.strip()
        out.append({"date": day, "classification": classification, **texts})
    return out


def _is_calendar_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def apply_overrides(
    records: Sequence[Mapping[str, Any]],
    overrides: Sequence[Mapping[str, Any]],
    dates: Sequence[date | str],
    window: int,
) -> list[dict[str, Any]]:
    """Apply validated overrides to rule records (returns new records).

    Only a detected uniform day may be overridden. An overridden record keeps
    its rule evidence (``evidence.rule_classification``, ``look_ahead``) and
    gets ``classification`` from the override, ``status = "confirmed"``,
    ``override = {classification, reason, author}`` and the matching
    ``action`` / ``affected``.

    Raises
    ------
    OverrideError
        If an override's date is not one of the records' dates.
    """
    by_date: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for k in range(len(records)):
        out.append(copy.deepcopy(dict(records[k])))
        by_date[out[k]["date"]] = k
    for k in range(len(overrides)):
        entry = overrides[k]
        position = by_date.get(entry["date"])
        if position is None:
            raise OverrideError(
                f"override for {entry['date']}: not a detected uniform-price day "
                f"(only these can be overridden: {', '.join(by_date) or 'none'})"
            )
        record = out[position]
        t = int(record["index"])
        record["classification"] = entry["classification"]
        record["status"] = STATUS_CONFIRMED
        record["override"] = {
            "classification": entry["classification"],
            "reason": entry["reason"],
            "author": entry["author"],
        }
        record["action"] = action_for(entry["classification"], t)
        record["affected"] = affected_outputs(record["action"], t, dates, window)
    return out


def diff_anomalies(
    parent: Sequence[Mapping[str, Any]], child: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Classification/status changes between two record lists, by date.

    Returns ``[{date, from: {classification, status} | None, to: {...} | None}]``
    in date order for every date whose classification or status differs (a
    date present on one side only has ``None`` on the other).
    """
    def index(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for k in range(len(records)):
            out[records[k]["date"]] = {
                "classification": records[k]["classification"],
                "status": records[k]["status"],
            }
        return out

    before, after = index(parent), index(child)
    changes = []
    days = sorted(set(before) | set(after))
    for k in range(len(days)):
        old, new = before.get(days[k]), after.get(days[k])
        if old != new:
            changes.append({"date": days[k], "from": old, "to": new})
    return changes


def summarize_anomalies(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts of records by classification and status, plus look-ahead and unresolved dates."""
    by_classification = {REBASE: 0, PLACEHOLDER: 0, UNRESOLVED: 0}
    by_status = {STATUS_INFERRED: 0, STATUS_CONFIRMED: 0, STATUS_UNRESOLVED: 0}
    look_ahead = 0
    unresolved_dates = []
    overridden = 0
    for k in range(len(records)):
        record = records[k]
        by_classification[record["classification"]] = by_classification.get(record["classification"], 0) + 1
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
        look_ahead += bool(record["look_ahead"])
        overridden += record.get("override") is not None
        if record["classification"] == UNRESOLVED:
            unresolved_dates.append(record["date"])
    return {
        "rule": ANOMALY_RULE,
        "count": len(records),
        "by_classification": by_classification,
        "by_status": by_status,
        "look_ahead_count": look_ahead,
        "overridden_count": overridden,
        "unresolved_dates": unresolved_dates,
    }


def anomaly_masks(
    records: Sequence[Mapping[str, Any]], n_dates: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final classification masks ``(rebase, placeholder, unresolved)``, bool ``[T]`` each."""
    rebase = np.zeros(n_dates, dtype=bool)
    placeholder = np.zeros(n_dates, dtype=bool)
    unresolved = np.zeros(n_dates, dtype=bool)
    for k in range(len(records)):
        t = int(records[k]["index"])
        classification = records[k]["classification"]
        if classification == REBASE:
            rebase[t] = True
        elif classification == PLACEHOLDER:
            placeholder[t] = True
        else:
            unresolved[t] = True
    return rebase, placeholder, unresolved


@dataclass(frozen=True)
class Derived:
    """Prices and returns actually used, derived from the raw prices and the records.

    Attributes
    ----------
    records:
        Anomaly records (overrides applied).
    prices:
        float64 ``[T x N]``: ``prices_raw`` with placeholder and unresolved
        days set to NaN (a copy; the raw array is untouched).
    returns:
        float64 ``[T x N]`` simple returns of ``prices`` with the return into
        every re-base day set to NaN.
    rebase, placeholder, unresolved:
        bool ``[T]`` final classification masks.
    returns_nulled_at_rebase:
        Finite returns set to NaN because they end on a re-base day.
    prices_nulled_on_placeholder_days:
        Finite prices set to NaN on placeholder days.
    prices_nulled_pending:
        Finite prices set to NaN on an unresolved (last) day.
    """

    records: list[dict[str, Any]]
    prices: np.ndarray
    returns: np.ndarray
    rebase: np.ndarray
    placeholder: np.ndarray
    unresolved: np.ndarray
    returns_nulled_at_rebase: int
    prices_nulled_on_placeholder_days: int
    prices_nulled_pending: int


def derive_returns(
    prices_raw: np.ndarray,
    dates: Sequence[date | str],
    window: int,
    overrides: Sequence[Mapping[str, Any]] | None = None,
) -> Derived:
    """Raw prices -> anomaly records -> used prices -> returns (pure; raw kept intact).

    Parameters
    ----------
    prices_raw:
        float ``[T x N]`` per-file-cleaned prices, NaN for missing.
    dates:
        ``T`` dates.
    window:
        Window length W (for the records' affected windows).
    overrides:
        Validated override entries (:func:`parse_overrides`), or None.
    """
    prices_raw = np.asarray(prices_raw, dtype=np.float64)
    records = anomaly_records(prices_raw, dates, window, overrides)
    rebase, placeholder, unresolved = anomaly_masks(records, prices_raw.shape[0])
    prices = prices_raw.copy()
    prices[placeholder | unresolved] = np.nan
    returns = simple_returns(prices, breaks=rebase)
    returns_nulled, placeholder_nulled, pending_nulled = nulled_counts(
        prices_raw, prices, rebase, placeholder, unresolved
    )
    return Derived(
        records=records,
        prices=prices,
        returns=returns,
        rebase=rebase,
        placeholder=placeholder,
        unresolved=unresolved,
        returns_nulled_at_rebase=returns_nulled,
        prices_nulled_on_placeholder_days=placeholder_nulled,
        prices_nulled_pending=pending_nulled,
    )


def nulled_counts(
    prices_raw: np.ndarray,
    prices: np.ndarray,
    rebase: np.ndarray,
    placeholder: np.ndarray,
    unresolved: np.ndarray,
) -> tuple[int, int, int]:
    """How many finite values the anomaly handling removed.

    Returns
    -------
    (returns_nulled_at_rebase, prices_nulled_on_placeholder_days, prices_nulled_pending)
        Finite returns into re-base days that were set to NaN (both used
        prices finite), and finite raw prices set to NaN on placeholder /
        unresolved days.
    """
    rebase_rows = np.flatnonzero(np.asarray(rebase, dtype=bool)[1:]) + 1
    into = np.isfinite(prices[rebase_rows]) & np.isfinite(prices[rebase_rows - 1])
    return (
        int(into.sum()),
        int(np.isfinite(prices_raw[np.asarray(placeholder, dtype=bool)]).sum()),
        int(np.isfinite(prices_raw[np.asarray(unresolved, dtype=bool)]).sum()),
    )


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
