"""The synthetic demo dataset: deterministic, clearly labelled, and exercising every rule."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURE_WINDOW, FixtureData

from corrlib import store
from corrlib.build import read_overrides, run_build
from corrlib.config import (
    DEMO_LABEL,
    DEMO_ZIP_COMMENT,
    REASON_CONSTANT_RETURNS,
    REASON_VALID,
)
from corrlib.corr import one_vs_all, pair_series, reason_series, window_status
from corrlib.demo import (
    CORRECTION_DAY,
    DELISTING_DAY,
    LISTING_DAY,
    N_DAYS,
    PLACEHOLDER_DAY,
    REBASE_DAY,
    generate_demo,
)
from corrlib.io import iter_daily_files, read_daily_file


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_is_deterministic(demo_files, tmp_path: Path) -> None:
    again = generate_demo(tmp_path / "again")
    for name in ("zip_path", "append_csv", "correction_csv", "overrides_path"):
        assert sha(getattr(again, name)) == sha(getattr(demo_files, name)), name
    assert again.dates == demo_files.dates


def test_zip_is_marked_synthetic(demo_files) -> None:
    with zipfile.ZipFile(demo_files.zip_path) as zf:
        assert zf.comment.decode() == DEMO_ZIP_COMMENT
        readme = [n for n in zf.namelist() if n.endswith("README.txt")]
        assert readme and b"NOT real market data" in zf.read(readme[0])
    assert len(demo_files.dates) == N_DAYS
    assert len(demo_files.tickers) == 300
    assert "NA" in demo_files.tickers


def test_demo_build_is_labelled_and_planted_features_survive(demo_store_dir: Path, demo_files) -> None:
    s = store.load_store(demo_store_dir)
    assert s.dataset == {"kind": "demo", "label": DEMO_LABEL}
    assert (s.n_dates, s.n_tickers) == (N_DAYS, 300)
    iso = [d.isoformat() for d in demo_files.dates]
    assert [(r["date"], r["classification"], r["action"], r["look_ahead"]) for r in s.anomalies] == [
        (iso[0], "rebase", "none", False),
        (iso[PLACEHOLDER_DAY], "placeholder", "prices_nulled", True),
        (iso[REBASE_DAY], "rebase", "return_into_day_nulled", True),
    ]
    gap = (demo_files.dates[REBASE_DAY] - demo_files.dates[REBASE_DAY - 1]).days
    assert gap > 14  # the re-base follows a calendar gap

    counts = s.manifest["issue_counts"]
    assert counts["date_format_variant"] == 299  # one M/D/YYYY file (LATE not listed yet)
    assert counts["price_missing"] == 2 and counts["price_non_numeric"] == 2
    assert counts["duplicate_identical"] == 1 and counts["duplicate_rows_dropped"] == 1
    assert counts["rows_in"] == counts["rows_out"] + 1


def test_demo_correlation_structure(demo_store_dir: Path) -> None:
    s = store.load_store(demo_store_dir)
    t = s.n_dates - 1
    Z, valid, reason = window_status(s.returns, t, s.window)
    col = s.ticker_index
    assert reason[col["FLAT"]] == REASON_CONSTANT_RETURNS
    assert reason[col["NA"]] == REASON_VALID
    tec = [col[f"TEC{k:02d}"] for k in range(2, 41)]
    other = [col[f"IDI{k:02d}"] for k in range(1, 56)]
    row = one_vs_all(Z, valid, col["TEC01"])
    assert np.nanmean(row[tec]) > 0.5  # clearly positive intra-sector
    assert abs(np.nanmean(row[other])) < 0.15  # near zero against idiosyncratic names
    inverse = pair_series(s.returns, col["INVA"], col["INVB"], s.window)
    assert np.nanmean(inverse) < -0.8


def test_listing_and_delisting(demo_store_dir: Path) -> None:
    s = store.load_store(demo_store_dir)
    late = reason_series(s.returns[:, s.ticker_index["LATE"]], s.window)
    gone = reason_series(s.returns[:, s.ticker_index["GONE"]], s.window)
    assert (late[: LISTING_DAY + s.window] != REASON_VALID).all()
    assert late[LISTING_DAY + s.window] == REASON_VALID
    assert gone[DELISTING_DAY] == REASON_VALID
    assert (gone[DELISTING_DAY + 1 :] != REASON_VALID).all()


def test_update_files(demo_files) -> None:
    append = read_daily_file(demo_files.append_csv)
    assert append.file_date == demo_files.append_date > demo_files.dates[-1]
    assert pd.Timestamp(demo_files.append_date).dayofweek < 5
    assert "GONE" not in append.frame["Ticker"].tolist()

    correction = read_daily_file(demo_files.correction_csv)
    assert correction.file_date == demo_files.dates[CORRECTION_DAY]
    original = None
    for daily in iter_daily_files(demo_files.zip_path):
        if daily.file_date == correction.file_date:
            original = daily
    assert original is not None and original.sha256 != correction.sha256
    merged = original.frame.merge(correction.frame, on=["Ticker", "Date"], suffixes=("_zip", "_fix"))
    changed = merged[merged["Price_zip"] != merged["Price_fix"]]
    assert len(merged) == len(original.frame) == len(correction.frame)
    assert changed["Ticker"].tolist() == [demo_files.correction_ticker]
    assert float(changed["Price_fix"].iloc[0]) == demo_files.correct_price
    assert float(changed["Price_zip"].iloc[0]) == demo_files.zip_price


def test_example_overrides_confirm_the_detected_days(demo_files, tmp_path: Path) -> None:
    overrides = read_overrides(demo_files.overrides_path)
    assert {e["date"] for e in overrides.entries} == {
        demo_files.placeholder_date.isoformat(),
        demo_files.rebase_date.isoformat(),
    }
    out = tmp_path / "store"
    manifest, timings = run_build(demo_files.zip_path, out, overrides=demo_files.overrides_path)
    assert manifest["anomalies_summary"]["by_status"] == {"inferred": 1, "confirmed": 2, "unresolved": 0}
    assert manifest["overrides"]["sha256"] == overrides.sha256
    assert timings["total"] < 60  # "a few seconds" in practice; generous bound for slow CI


def test_demo_flag_and_auto_detection(fixture_data: FixtureData, built_dir: Path, tmp_path: Path) -> None:
    assert store.load_store(built_dir).dataset["kind"] == "source"
    manifest, _ = run_build(fixture_data.zip_path, tmp_path / "forced", window=FIXTURE_WINDOW, demo=True)
    assert manifest["dataset"] == {"kind": "demo", "label": DEMO_LABEL}


@pytest.mark.parametrize("seed", [1])
def test_other_seeds_still_plant_every_feature(seed: int, tmp_path: Path) -> None:
    files = generate_demo(tmp_path / "d", seed=seed)
    manifest, _ = run_build(files.zip_path, tmp_path / "s")
    assert manifest["anomalies_summary"]["by_classification"] == {"rebase": 2, "placeholder": 1, "unresolved": 0}
