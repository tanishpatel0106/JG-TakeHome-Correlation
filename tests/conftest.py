"""Shared fixtures: a tiny synthetic source zip with every data issue we handle.

The zip has 40 business days x 6 tickers (``A, AAPL, FLAT, MSFT, NA, ZZZ``),
fixed-seed random-walk prices starting from a uniform 20.0 seed day, and these
deliberate defects (day index -> defect):

====  ==========================================================  =====================
day   defect                                                      expected counter
====  ==========================================================  =====================
3     duplicate MSFT row, same price                              duplicate_identical
4     duplicate AAPL row, different price (-> NaN)                duplicate_conflicting
5     every Date written as M/D/YYYY                              date_format_variant x6
6     ZZZ price "abc" (-> NaN)                                    price_non_numeric
7     A price "0" (-> NaN)                                        price_non_positive
8     NA row missing (-> NaN)                                     (no counter: absent)
9     MSFT Date is the previous day (file date wins)              date_mismatch
10    AAPL Date "not-a-date"                                      date_unparseable
11    MSFT written as " msft "                                    ticker_normalized
12    extra row with empty ticker; ZZZ price "" (-> NaN)          ticker_empty, price_missing
13    AAPL price "inf" (-> NaN); A Date as YYYYMMDD               price_non_finite, date_format_variant
all   FLAT has a constant price (zero variance, always invalid)
====  ==========================================================  =====================

Plus stray zip entries: a directory, ``__MACOSX/`` metadata and ``notes.txt``.
The members are written in shuffled order to test sorting.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TICKERS = ["A", "AAPL", "FLAT", "MSFT", "NA", "ZZZ"]
N_DATES = 40
FIXTURE_WINDOW = 20

EXPECTED_COUNTS = {
    "files": 40,
    "rows_in": 40 * 6 + 1 + 1 - 1 + 1,
    "rows_out": 40 * 6 - 1,
    "ticker_empty": 1,
    "ticker_normalized": 1,
    "date_format_variant": 7,
    "date_mismatch": 1,
    "date_unparseable": 1,
    "price_missing": 1,
    "price_non_numeric": 1,
    "price_non_finite": 1,
    "price_non_positive": 1,
    "duplicate_identical": 1,
    "duplicate_conflicting": 1,
    "duplicate_rows_dropped": 2,
}

STRAY_ENTRIES = ["stock_data/", "__MACOSX/stock_data/._20210104.csv", "stock_data/notes.txt"]


@dataclass(frozen=True)
class FixtureData:
    """The fixture zip and the ground truth it was generated from."""

    zip_path: Path
    dates: list[date]
    tickers: list[str]
    #: float64 [T x N] prices the pipeline must produce (NaN where invalid/missing)
    expected_prices: np.ndarray


def _iso(d: date) -> str:
    return d.isoformat()


def make_fixture_zip(path: Path, seed: int = 7) -> FixtureData:
    """Write the synthetic zip described in the module docstring."""
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2021-01-04", periods=N_DATES).date)
    n = len(TICKERS)
    col: dict[str, int] = {}
    for j in range(n):
        col[TICKERS[j]] = j

    log_ret = rng.normal(0.0, 0.02, size=(N_DATES, n))
    log_ret[0] = 0.0
    prices = np.round(20.0 * np.exp(np.cumsum(log_ret, axis=0)), 6)
    prices[:, col["FLAT"]] = 20.0  # day 0 is a uniform 20.0 seed day
    expected = prices.copy()

    files: dict[str, str] = {}
    for t in range(len(dates)):
        d = dates[t]
        rows = []
        by_ticker = {}
        for j in range(n):
            row = [TICKERS[j], _iso(d), f"{prices[t, j]:.6f}"]
            rows.append(row)
            by_ticker[TICKERS[j]] = row
        if t == 3:
            rows.append(list(by_ticker["MSFT"]))
        if t == 4:
            rows.append(["AAPL", _iso(d), f"{prices[t, col['AAPL']] + 1:.6f}"])
            expected[t, col["AAPL"]] = np.nan
        if t == 5:
            for k in range(len(rows)):
                rows[k][1] = f"{d.month}/{d.day}/{d.year}"
        if t == 6:
            by_ticker["ZZZ"][2] = "abc"
            expected[t, col["ZZZ"]] = np.nan
        if t == 7:
            by_ticker["A"][2] = "0"
            expected[t, col["A"]] = np.nan
        if t == 8:
            rows.remove(by_ticker["NA"])
            expected[t, col["NA"]] = np.nan
        if t == 9:
            by_ticker["MSFT"][1] = _iso(dates[t - 1])
        if t == 10:
            by_ticker["AAPL"][1] = "not-a-date"
        if t == 11:
            by_ticker["MSFT"][0] = " msft "
        if t == 12:
            rows.append(["  ", _iso(d), "12.5"])
            by_ticker["ZZZ"][2] = ""
            expected[t, col["ZZZ"]] = np.nan
        if t == 13:
            by_ticker["AAPL"][2] = "inf"
            expected[t, col["AAPL"]] = np.nan
            by_ticker["A"][1] = d.strftime("%Y%m%d")
        lines = ["Ticker,Date,Price"]
        for k in range(len(rows)):
            lines.append(",".join(rows[k]))
        files[f"stock_data/{d.strftime('%Y%m%d')}.csv"] = "\n".join(lines) + "\n"

    names = list(files)
    order = rng.permutation(len(names))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("stock_data/", "")
        for k in range(len(order)):
            zf.writestr(names[order[k]], files[names[order[k]]])
        zf.writestr("__MACOSX/stock_data/._20210104.csv", b"\x00\x05\x16\x07 mac metadata")
        zf.writestr("stock_data/notes.txt", "not a daily file")
    return FixtureData(zip_path=path, dates=dates, tickers=list(TICKERS), expected_prices=expected)


def write_zip(path: Path, members: dict[str, str]) -> Path:
    """Write a zip with the given ``name -> text`` members (helper for edge-case tests)."""
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def daily_csv_text(day: date, tickers: list[str], prices: np.ndarray) -> str:
    """``Ticker,Date,Price`` text for one day; NaN prices are left out (ticker absent)."""
    lines = ["Ticker,Date,Price"]
    for j in range(len(tickers)):
        if np.isfinite(prices[j]):
            lines.append(f"{tickers[j]},{day.isoformat()},{prices[j]:.6f}")
    return "\n".join(lines) + "\n"


def write_price_zip(
    path: Path, prices: np.ndarray, tickers: list[str], dates: list[date], comment: str | None = None
) -> Path:
    """Write a source zip from a ``[T x N]`` price matrix (NaN = row absent that day)."""
    with zipfile.ZipFile(path, "w") as zf:
        if comment is not None:
            zf.comment = comment.encode("utf-8")
        for t in range(len(dates)):
            zf.writestr(f"d/{dates[t].strftime('%Y%m%d')}.csv", daily_csv_text(dates[t], tickers, prices[t]))
    return path


def random_walk_prices(n_dates: int, n_tickers: int, seed: int = 0, vol: float = 0.02) -> np.ndarray:
    """float64 ``[T x N]`` positive prices starting at 20.0 (a uniform seed day at t = 0)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, vol, size=(n_dates, n_tickers))
    steps[0] = 0.0
    return np.round(20.0 * np.exp(np.cumsum(steps, axis=0)), 6)


@pytest.fixture(scope="session")
def fixture_data(tmp_path_factory: pytest.TempPathFactory) -> FixtureData:
    return make_fixture_zip(tmp_path_factory.mktemp("raw") / "stock_data.zip")


@pytest.fixture(scope="session")
def built_dir(fixture_data: FixtureData, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A store ROOT built from the fixture zip (shared by store and API tests; never mutate it).

    ``run_build`` publishes one ``full_build`` snapshot (``v0001-<h12>``) and
    points ``CURRENT`` at it; ``load_store(built_dir)`` resolves it. Tests that
    ingest, roll back or prune work on a copy (``shutil.copytree``).
    """
    from corrlib.build import run_build

    out = tmp_path_factory.mktemp("processed") / "processed"
    run_build(fixture_data.zip_path, out, window=FIXTURE_WINDOW, block=4)
    return out


@pytest.fixture(scope="session")
def built_snapshot_dir(built_dir: Path) -> Path:
    """The snapshot directory ``built_dir/snapshots/<id>`` that CURRENT names."""
    from corrlib.store import resolve_snapshot_dir

    return resolve_snapshot_dir(built_dir)[0]


@pytest.fixture(scope="session")
def demo_files(tmp_path_factory: pytest.TempPathFactory):
    """The synthetic demo dataset (``corrlib.demo.generate_demo``) in a temp directory."""
    from corrlib.demo import generate_demo

    return generate_demo(tmp_path_factory.mktemp("demo") / "demo")


@pytest.fixture(scope="session")
def demo_store_dir(demo_files, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A store ROOT built from the demo zip (dataset kind auto-detected as "demo"; never mutate it)."""
    from corrlib.build import run_build

    out = tmp_path_factory.mktemp("demo_store") / "store"
    run_build(demo_files.zip_path, out)
    return out
