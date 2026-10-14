"""Validation and normalisation of one daily file.

Rules (each one counted in :class:`IssueCounts`, never applied silently):

* **Ticker**: strip whitespace, upper-case. Empty tickers are dropped.
* **Date**: the *file name* date is authoritative and every row gets it. The
  ``Date`` column is only checked: ISO ``YYYY-MM-DD``, ``M/D/YYYY`` and
  ``YYYYMMDD`` are tried in that order with explicit patterns (no guessing).
* **Price**: parsed as a number. Empty, non-numeric, non-finite and <= 0
  values become NaN.
* **Duplicate tickers in one file**: identical prices are collapsed to one row.
  Conflicting prices become NaN (we do not pick one).
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: How many example mismatching dates an IssueCounts keeps (across files).
MAX_EXAMPLES = 10

# (format name, shape check, strptime format). The shapes are mutually exclusive.
_DATE_FORMATS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("iso", re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    ("mdy", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"), "%m/%d/%Y"),
    ("compact", re.compile(r"^\d{8}$"), "%Y%m%d"),
)

_NAN_LITERALS = ("nan", "+nan", "-nan")


@dataclass
class IssueCounts:
    """Counts of every data issue found while cleaning. Summable across files.

    All fields are row counts unless stated otherwise.
    """

    files: int = 0
    rows_in: int = 0
    rows_out: int = 0
    #: rows dropped because the ticker was empty after stripping
    ticker_empty: int = 0
    #: tickers changed by strip/upper-case (row kept)
    ticker_normalized: int = 0
    #: Date parsed to the file date but was not written as ISO
    date_format_variant: int = 0
    #: Date parsed to a different date than the file name (file date used)
    date_mismatch: int = 0
    #: Date empty or in none of the accepted formats (file date used)
    date_unparseable: int = 0
    #: empty price -> NaN
    price_missing: int = 0
    #: price text is not a number -> NaN
    price_non_numeric: int = 0
    #: price is nan / inf -> NaN
    price_non_finite: int = 0
    #: price <= 0 -> NaN
    price_non_positive: int = 0
    #: tickers (not rows) repeated in a file with one consistent price; one row kept
    duplicate_identical: int = 0
    #: tickers (not rows) repeated in a file with conflicting prices; price -> NaN
    duplicate_conflicting: int = 0
    #: extra rows removed while collapsing duplicates
    duplicate_rows_dropped: int = 0
    #: a capped sample of "file date: Date value" for date mismatches
    mismatch_examples: list[str] = field(default_factory=list)

    def __add__(self, other: IssueCounts) -> IssueCounts:
        total = IssueCounts()
        names = count_fields()
        for k in range(len(names)):
            name = names[k]
            setattr(total, name, getattr(self, name) + getattr(other, name))
        total.mismatch_examples = (self.mismatch_examples + other.mismatch_examples)[
            :MAX_EXAMPLES
        ]
        return total

    def as_dict(self) -> dict[str, int]:
        """Numeric counters only (no examples), in declaration order."""
        out: dict[str, int] = {}
        names = count_fields()
        for k in range(len(names)):
            out[names[k]] = int(getattr(self, names[k]))
        return out


def count_fields() -> list[str]:
    """Names of the integer counters of :class:`IssueCounts`."""
    names = []
    all_fields = dataclasses.fields(IssueCounts)
    for k in range(len(all_fields)):
        if all_fields[k].name != "mismatch_examples":
            names.append(all_fields[k].name)
    return names


def parse_date_value(text: str) -> tuple[date | None, str | None]:
    """Parse one ``Date`` cell with the accepted explicit formats.

    Returns ``(date, format_name)`` where format_name is ``"iso"``, ``"mdy"`` or
    ``"compact"``. Returns ``(None, None)`` if the text matches no format or is
    not a real calendar date (e.g. ``2021-02-30``).
    """
    value = text.strip()
    for k in range(len(_DATE_FORMATS)):
        name, shape, fmt = _DATE_FORMATS[k]
        if shape.match(value):
            try:
                return datetime.strptime(value, fmt).date(), name
            except ValueError:
                return None, None
    return None, None


def clean_frame(file_date: date, df: pd.DataFrame) -> tuple[pd.DataFrame, IssueCounts]:
    """Validate and normalise one raw daily frame.

    Parameters
    ----------
    file_date:
        Date from the file name. It is assigned to every row.
    df:
        Raw frame with string columns ``Ticker, Date, Price`` (see
        :func:`corrlib.io.read_member`).

    Returns
    -------
    (frame, counts)
        ``frame`` has columns ``date`` (the file date on every row),
        ``ticker`` (str, upper-case, unique, non-empty) and ``price`` (float64,
        > 0, NaN where missing/invalid/conflicting). ``counts`` holds this
        file's issues.
    """
    counts = IssueCounts(files=1, rows_in=len(df))

    raw_ticker = df["Ticker"].fillna("").astype(str)
    ticker = raw_ticker.str.strip().str.upper()
    keep = (ticker != "").to_numpy()
    counts.ticker_empty = int((~keep).sum())
    counts.ticker_normalized = int(((ticker != raw_ticker).to_numpy() & keep).sum())

    kept = df.loc[keep]
    _check_dates(file_date, kept["Date"], counts)
    price = _parse_prices(kept["Price"], counts)

    # Keep pandas' string dtype (Arrow-backed when available): far smaller than
    # Python str objects when hundreds of daily frames are held in memory.
    out = pd.DataFrame({"ticker": ticker.loc[keep].reset_index(drop=True), "price": price})
    out.insert(0, "date", pd.Timestamp(file_date))
    out = _resolve_duplicates(file_date, out, counts)
    counts.rows_out = len(out)
    return out, counts


def _check_dates(file_date: date, values: pd.Series, counts: IssueCounts) -> None:
    """Count date-column issues. Works on the distinct values (usually 1-2 per file)."""
    distinct = values.fillna("").astype(str).value_counts(sort=False)
    texts = distinct.index.tolist()
    sizes = distinct.to_numpy()
    for k in range(len(texts)):
        text = texts[k]
        n = int(sizes[k])
        parsed, fmt = parse_date_value(text)
        if parsed is None:
            counts.date_unparseable += n
        elif parsed != file_date:
            counts.date_mismatch += n
            if len(counts.mismatch_examples) < MAX_EXAMPLES:
                counts.mismatch_examples.append(f"{file_date.isoformat()}: {text!r}")
            log.warning(
                "%s: %d row(s) have Date %r (= %s); using the file-name date",
                file_date, n, text, parsed,
            )
        elif fmt != "iso":
            counts.date_format_variant += n


def _parse_prices(values: pd.Series, counts: IssueCounts) -> np.ndarray:
    """Parse price text to float64; invalid values become NaN and are counted."""
    text = values.fillna("").astype(str).str.strip()
    raw = text.to_numpy(dtype=object)
    number = np.asarray(pd.to_numeric(raw, errors="coerce"), dtype=np.float64)

    empty = (text == "").to_numpy()
    nan_literal = text.str.lower().isin(_NAN_LITERALS).to_numpy()
    unparsed = np.isnan(number) & ~nan_literal
    infinite = np.isinf(number)
    non_positive = np.isfinite(number) & (number <= 0)

    counts.price_missing += int(empty.sum())
    counts.price_non_numeric += int((unparsed & ~empty).sum())
    counts.price_non_finite += int((nan_literal | infinite).sum())
    counts.price_non_positive += int(non_positive.sum())

    valid = np.isfinite(number) & (number > 0)
    return np.where(valid, number, np.nan)


def _resolve_duplicates(file_date: date, frame: pd.DataFrame, counts: IssueCounts) -> pd.DataFrame:
    """Collapse repeated tickers: identical prices -> one row, conflicts -> NaN."""
    repeated = frame["ticker"].duplicated(keep=False).to_numpy()
    if not repeated.any():
        return frame.reset_index(drop=True)

    groups = frame.loc[repeated].groupby("ticker", sort=False)["price"]
    distinct = groups.nunique(dropna=False)
    sizes = groups.size()
    conflicting = distinct.index[distinct.to_numpy() > 1]

    counts.duplicate_identical += int((distinct.to_numpy() == 1).sum())
    counts.duplicate_conflicting += len(conflicting)
    counts.duplicate_rows_dropped += int(sizes.sum() - len(sizes))
    if len(conflicting):
        log.warning(
            "%s: conflicting duplicate prices for %s; set to NaN",
            file_date, ", ".join(conflicting.tolist()[:5]),
        )

    out = frame.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    out.loc[out["ticker"].isin(conflicting), "price"] = np.nan
    return out
