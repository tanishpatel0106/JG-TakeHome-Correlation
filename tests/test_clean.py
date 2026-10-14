from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from conftest import EXPECTED_COUNTS, FixtureData
from corrlib.clean import IssueCounts, clean_frame, parse_date_value
from corrlib.io import iter_daily_frames

DAY = date(2021, 2, 22)


def raw(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Ticker", "Date", "Price"], dtype=str)


def test_every_issue_counter_matches_the_fixture(fixture_data: FixtureData) -> None:
    total = IssueCounts()
    for file_date, frame in iter_daily_frames(fixture_data.zip_path):
        _, counts = clean_frame(file_date, frame)
        total = total + counts
    assert total.as_dict() == EXPECTED_COUNTS


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2021-02-22", (DAY, "iso")),
        ("2/22/2021", (DAY, "mdy")),
        ("02/22/2021", (DAY, "mdy")),
        (" 20210222 ", (DAY, "compact")),
        ("2021-02-30", (None, None)),  # right shape, impossible date
        ("22/2/2021", (None, None)),  # D/M/Y is not accepted
        ("2021-2-22", (None, None)),
        ("", (None, None)),
        ("yesterday", (None, None)),
    ],
)
def test_parse_date_value(text: str, expected: tuple) -> None:
    assert parse_date_value(text) == expected


def test_filename_date_wins_over_date_column() -> None:
    frame, counts = clean_frame(
        DAY,
        raw([("A", "2021-02-19", "10"), ("B", "2/22/2021", "11"), ("C", "garbage", "12"), ("D", "2021-02-22", "13")]),
    )
    assert (frame["date"] == pd.Timestamp(DAY)).all()
    assert frame["price"].tolist() == [10.0, 11.0, 12.0, 13.0]
    assert counts.date_mismatch == 1
    assert counts.date_format_variant == 1
    assert counts.date_unparseable == 1
    assert counts.mismatch_examples == ["2021-02-22: '2021-02-19'"]


def test_conflicting_duplicates_become_nan_identical_collapse() -> None:
    frame, counts = clean_frame(
        DAY,
        raw([("A", "2021-02-22", "10"), ("A", "2021-02-22", "10.0"), ("B", "2021-02-22", "5"), ("b ", "2021-02-22", "6")]),
    )
    prices = frame.set_index("ticker")["price"]
    assert prices["A"] == 10.0
    assert np.isnan(prices["B"])
    assert len(frame) == 2
    assert (counts.duplicate_identical, counts.duplicate_conflicting, counts.duplicate_rows_dropped) == (1, 1, 2)
    assert counts.ticker_normalized == 1


def test_duplicate_with_one_invalid_copy_is_a_conflict() -> None:
    frame, counts = clean_frame(DAY, raw([("A", "2021-02-22", "10"), ("A", "2021-02-22", "")]))
    assert np.isnan(frame["price"].iloc[0])
    assert counts.duplicate_conflicting == 1
    assert counts.price_missing == 1


def test_price_categories() -> None:
    frame, counts = clean_frame(
        DAY,
        raw([
            ("A", "2021-02-22", ""),
            ("B", "2021-02-22", "n/a"),
            ("C", "2021-02-22", "inf"),
            ("D", "2021-02-22", "NaN"),
            ("E", "2021-02-22", "0"),
            ("F", "2021-02-22", "-3.5"),
            ("G", "2021-02-22", " 1e2 "),
        ]),
    )
    assert np.isnan(frame["price"].iloc[:6]).all()
    assert frame["price"].iloc[6] == 100.0
    assert (counts.price_missing, counts.price_non_numeric, counts.price_non_finite, counts.price_non_positive) == (1, 1, 2, 2)


def test_na_ticker_is_kept_and_empty_ticker_dropped() -> None:
    frame, counts = clean_frame(DAY, raw([("NA", "2021-02-22", "1"), ("   ", "2021-02-22", "2")]))
    assert frame["ticker"].tolist() == ["NA"]
    assert counts.ticker_empty == 1
    assert (counts.rows_in, counts.rows_out) == (2, 1)


def test_issue_counts_sum_and_cap_examples() -> None:
    a = IssueCounts(files=1, price_missing=2, mismatch_examples=["x"] * 8)
    b = IssueCounts(files=1, price_missing=3, mismatch_examples=["y"] * 8)
    total = a + b
    assert total.files == 2 and total.price_missing == 5
    assert len(total.mismatch_examples) == 10
