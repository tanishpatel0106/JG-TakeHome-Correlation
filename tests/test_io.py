from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from conftest import N_DATES, STRAY_ENTRIES, FixtureData, write_zip
from corrlib.io import SourceFormatError, iter_daily_frames, list_members, scan_members


def test_na_ticker_survives_as_string(fixture_data: FixtureData) -> None:
    for _, frame in iter_daily_frames(fixture_data.zip_path):
        assert "NA" in frame["Ticker"].tolist()
        assert not frame["Ticker"].isna().any()
        break


def test_stray_entries_are_skipped(fixture_data: FixtureData) -> None:
    scan = scan_members(fixture_data.zip_path)
    skipped = [name for name, _ in scan.skipped]
    assert sorted(skipped) == sorted(STRAY_ENTRIES)
    assert len(scan.members) == N_DATES


def test_members_sorted_by_date(fixture_data: FixtureData) -> None:
    members = list_members(fixture_data.zip_path)
    dates = [d for d, _ in members]
    assert dates == sorted(dates)
    assert dates == fixture_data.dates
    # The zip was written in shuffled order; the names follow the dates.
    assert members[0][1] == "stock_data/20210104.csv"


def test_bad_header_raises(tmp_path: Path) -> None:
    path = write_zip(tmp_path / "bad.zip", {"d/20210104.csv": "Symbol,Date,Price\nA,2021-01-04,1\n"})
    with pytest.raises(SourceFormatError, match="unexpected header"):
        list(iter_daily_frames(path))


def test_header_tolerates_bom_and_whitespace(tmp_path: Path) -> None:
    path = write_zip(tmp_path / "bom.zip", {"20210104.csv": "﻿ Ticker , Date,Price \nNA,2021-01-04,1.5\n"})
    ((file_date, frame),) = list(iter_daily_frames(path))
    assert file_date == date(2021, 1, 4)
    assert list(frame.columns) == ["Ticker", "Date", "Price"]
    assert frame["Ticker"].tolist() == ["NA"]


def test_two_members_for_one_date_raise(tmp_path: Path) -> None:
    text = "Ticker,Date,Price\nA,2021-01-04,1\n"
    path = write_zip(tmp_path / "dup.zip", {"a/20210104.csv": text, "b/20210104.csv": text})
    with pytest.raises(SourceFormatError, match="map to 2021-01-04"):
        list_members(path)


def test_invalid_date_in_name_is_skipped(tmp_path: Path) -> None:
    text = "Ticker,Date,Price\nA,2021-01-04,1\n"
    path = write_zip(tmp_path / "x.zip", {"20210104.csv": text, "20211340.csv": text})
    scan = scan_members(path)
    assert [d for d, _ in scan.members] == [date(2021, 1, 4)]
    assert scan.skipped == [("20211340.csv", "file name is not a valid date")]
