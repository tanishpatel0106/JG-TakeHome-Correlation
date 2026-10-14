from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import FIXTURE_WINDOW, FixtureData
from corrlib import store
from corrlib.build import ingest
from corrlib.config import RETURNS_FILE


def test_round_trip_preserves_shapes_dtypes_axes(built_dir: Path, fixture_data: FixtureData) -> None:
    s = store.load_store(built_dir)
    T, N = len(fixture_data.dates), len(fixture_data.tickers)
    assert isinstance(s.returns, np.memmap)
    assert s.returns.shape == (T, N) and s.returns.dtype == np.float32
    assert s.tickers == fixture_data.tickers
    assert s.dates == [d.isoformat() for d in fixture_data.dates]
    assert s.ticker_index["NA"] == fixture_data.tickers.index("NA")
    assert s.hist.shape == (T, 40) and s.hist.dtype == np.int64
    assert s.hist_edges.shape == (41,)
    assert s.n_valid.shape == s.mean_corr.shape == s.std_corr.shape == (T,)
    assert s.window == FIXTURE_WINDOW
    # Windows exist from index W = 20, but the fixture's defects leave fewer than
    # two valid stocks until A's zero price (day 7) leaves the window at t = 28.
    assert s.n_valid[FIXTURE_WINDOW] == 1  # only MSFT
    assert s.first_valid_index == 28

    expected = ingest(fixture_data.zip_path).returns.astype(np.float32)
    np.testing.assert_array_equal(np.asarray(s.returns), expected)


def test_daily_stats_and_hist_are_consistent(built_dir: Path) -> None:
    s = store.load_store(built_dir)
    early = np.arange(s.n_dates) < s.window
    assert (s.n_valid[early] == 0).all() and np.isnan(s.mean_corr[early]).all()
    late = s.n_valid >= 2
    assert np.isfinite(s.mean_corr[late]).all()
    pairs = s.n_valid.astype(np.int64) * (s.n_valid - 1) // 2
    np.testing.assert_array_equal(s.hist.sum(axis=1), pairs)
    # FLAT never has a valid window; the 5 others do once their defects leave the window.
    assert s.n_valid.max() == 5


def test_prices_parquet_is_long_and_nullable(built_dir: Path, fixture_data: FixtureData) -> None:
    table = pq.read_table(built_dir / "prices.parquet")
    assert table.column_names == ["date", "ticker", "price"]
    T, N = fixture_data.expected_prices.shape
    assert table.num_rows == T * N
    prices = table.column("price").to_numpy(zero_copy_only=False).reshape(T, N)
    np.testing.assert_allclose(prices, fixture_data.expected_prices, rtol=1e-12, equal_nan=True)
    assert table.column("ticker").to_pylist()[:N] == fixture_data.tickers


def test_manifest_checksums_verify(built_dir: Path) -> None:
    assert store.verify_manifest(built_dir) == []
    manifest = json.loads((built_dir / "manifest.json").read_text())
    assert set(manifest["artifacts"]) == set(store.ARTIFACT_FILES)
    assert manifest["window"] == FIXTURE_WINDOW
    assert manifest["issue_counts"]["price_non_numeric"] == 1
    assert len(manifest["source"]["sha256"]) == 64


def test_tampering_is_detected(built_dir: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    copy.mkdir()
    for path in built_dir.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    data = bytearray((copy / RETURNS_FILE).read_bytes())
    data[-1] ^= 0xFF
    (copy / RETURNS_FILE).write_bytes(bytes(data))
    assert store.verify_manifest(copy) == [f"{RETURNS_FILE}: checksum mismatch"]


def test_failed_build_leaves_previous_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "processed"
    out.mkdir()
    (out / "marker").write_text("old")
    with pytest.raises(RuntimeError):
        with store.staging_dir(out) as tmp:
            (tmp / "half-written").write_text("x")
            raise RuntimeError("boom")
    assert [p.name for p in out.iterdir()] == ["marker"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed"]


def test_publish_replaces_directory(tmp_path: Path) -> None:
    out = tmp_path / "processed"
    out.mkdir()
    (out / "old").write_text("old")
    with store.staging_dir(out) as tmp:
        (tmp / "new").write_text("new")
    assert [p.name for p in out.iterdir()] == ["new"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["processed"]


def test_load_store_reports_missing_artifacts(tmp_path: Path) -> None:
    with pytest.raises(store.StoreError, match="missing artifacts"):
        store.load_store(tmp_path)
