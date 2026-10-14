"""API v2: snapshot pinning, npz download, error envelope (download concurrency: test_api_streaming.py)."""

from __future__ import annotations

import io
import json
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from conftest import FIXTURE_WINDOW, FixtureData, daily_csv_text
from corrlib.corr import prepare_window
from corrlib.ingest import run_ingest
from corrlib.store import load_store


@pytest.fixture(scope="module")
def client(built_dir: Path) -> TestClient:
    with TestClient(create_app(built_dir, reload_seconds=0)) as c:
        yield c


def test_reload_and_pinning_after_an_ingest(built_dir: Path, fixture_data: FixtureData, tmp_path: Path) -> None:
    root = tmp_path / "root"
    shutil.copytree(built_dir, root)
    with TestClient(create_app(root, reload_seconds=0)) as c:
        old = c.get("/api/snapshots/current").json()["id"]
        last = fixture_data.dates[-1]
        day = last + timedelta(days=3 if last.weekday() == 4 else 1)
        prices = np.nan_to_num(fixture_data.expected_prices[-1], nan=21.0) * 1.01
        path = tmp_path / "in" / f"{day.strftime('%Y%m%d')}.csv"
        path.parent.mkdir()
        path.write_text(daily_csv_text(day, fixture_data.tickers, prices))
        new = run_ingest(root, [path], reason="append").snapshot

        assert c.get("/api/health").json()["snapshot_id"] == old  # not reloaded yet
        assert c.post("/api/snapshots/reload").json() == {"previous": old, "current": new, "changed": True}
        r = c.get("/api/health")
        assert r.json()["n_dates"] == 41 and r.headers["x-snapshot-id"] == new
        pinned = c.get("/api/health", params={"snapshot": old})
        assert pinned.json()["n_dates"] == 40 and pinned.headers["x-snapshot-id"] == old
        assert c.get("/api/meta", params={"snapshot": old}).json()["last_date"] == last.isoformat()
        listing = c.get("/api/snapshots").json()
        assert listing["current"] == new and [s["id"] for s in listing["snapshots"]] == [old, new]
        missing = c.get("/api/health", params={"snapshot": "v0099-000000000000"})
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "snapshot_not_found"
        assert set(missing.json()["detail"]["available"]) == {old, new}


@pytest.mark.parametrize("dtype,atol", [("float32", 1e-6), ("float16", 1e-3)])
def test_npz_download_matches_the_matrix(
    client: TestClient, built_dir: Path, fixture_data: FixtureData, dtype: str, atol: float
) -> None:
    day = fixture_data.dates[-1].isoformat()
    info = client.get(f"/api/corr/{day}/download/info", params={"dtype": dtype}).json()
    r = client.get(f"/api/corr/{day}/download", params={"dtype": dtype})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert int(r.headers["content-length"]) == len(r.content) == info["bytes"] and info["bytes_exact"]
    assert info["filename"] in r.headers["content-disposition"] and info["filename"].endswith(".npz")
    with np.load(io.BytesIO(r.content), allow_pickle=False) as npz:
        corr, tickers, meta = npz["corr"], npz["tickers"], json.loads(str(npz["meta"]))
    assert corr.dtype == np.dtype(dtype) and corr.shape == (6, 6)
    assert tickers.tolist() == fixture_data.tickers
    assert meta["date"] == day and meta["window"] == FIXTURE_WINDOW and meta["dtype"] == dtype
    assert meta["snapshot_id"] == r.headers["x-snapshot-id"] and meta["tolerance"]["abs"] == atol
    store = load_store(built_dir)
    Z, valid = prepare_window(store.returns, 39, FIXTURE_WINDOW)
    exact = Z.T @ Z
    exact[~valid] = np.nan
    exact[:, ~valid] = np.nan
    np.testing.assert_allclose(corr.astype(np.float64), np.clip(exact, -1, 1), atol=atol)
    assert np.isnan(corr[2]).all()  # FLAT: NaN, never 0


def test_error_envelope_codes(client: TestClient, fixture_data: FixtureData) -> None:
    dates = [d.isoformat() for d in fixture_data.dates]
    cases = [
        ("/api/dates/2021-1-9/summary", 422, "invalid_date"),
        ("/api/dates/2021-01-09/summary", 404, "date_not_traded"),
        (f"/api/dates/{dates[5]}/summary", 422, "date_before_first_window"),
        (f"/api/corr/{dates[-1]}/stock/MSFTT", 404, "unknown_ticker"),
        (f"/api/corr/{dates[-1]}/stock/A?k=0", 422, "invalid_input"),
        (f"/api/corr/{dates[-1]}/download?dtype=float64", 422, "invalid_input"),
        ("/api/nope", 404, "not_found"),
    ]
    for path, status, code in cases:
        r = client.get(path)
        assert r.status_code == status, path
        detail = r.json()["detail"]
        assert detail["code"] == code and isinstance(detail["message"], str), path
