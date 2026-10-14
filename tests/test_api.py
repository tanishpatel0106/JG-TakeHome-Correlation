from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from conftest import FIXTURE_WINDOW, FixtureData
from corrlib.corr import iter_full_blocks, pair_series, prepare_window
from corrlib.store import load_store

FIRST_VALID = 28  # see test_store: first date index with >= 2 valid stocks


@pytest.fixture(scope="module")
def client(built_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(built_dir, reload_seconds=0)) as c:
        yield c


@pytest.fixture(scope="module")
def dates(fixture_data: FixtureData) -> list[str]:
    return [d.isoformat() for d in fixture_data.dates]


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "snapshot_id": body["snapshot_id"], "n_tickers": 6, "n_dates": 40}
    assert body["snapshot_id"].startswith("v0001-") and r.headers["x-snapshot-id"] == body["snapshot_id"]


def test_meta(client: TestClient, dates: list[str], fixture_data: FixtureData) -> None:
    body = client.get("/api/meta").json()
    assert body["window"] == FIXTURE_WINDOW
    assert body["tickers"] == fixture_data.tickers
    assert body["dates"] == dates
    assert body["first_valid_date"] == dates[FIRST_VALID]
    assert body["last_date"] == dates[-1] == body["default_date"]
    assert body["default_ticker"] == "A"
    assert body["breaks"] == {"rebase_dates": [dates[0]], "placeholder_dates": []}
    assert body["manifest"]["n_tickers"] == 6 and "artifacts" in body["manifest"]
    assert body["snapshot"]["id"].startswith("v0001-") and body["view"] == "retrospective"
    assert body["dataset"]["kind"] == "source" and body["policy"]["window"] == FIXTURE_WINDOW
    assert isinstance(body["anomalies"], list)


def test_timeline_has_nulls_before_first_window(client: TestClient) -> None:
    r = client.get("/api/stats/timeline")
    body = r.json()
    assert len(body["dates"]) == len(body["mean_corr"]) == len(body["std_corr"]) == len(body["n_valid"]) == 40
    assert body["mean_corr"][:FIRST_VALID] == [None] * FIRST_VALID
    assert all(isinstance(v, float) for v in body["mean_corr"][FIRST_VALID:])
    assert body["n_valid"][-1] == 5
    assert "NaN" not in r.text and "null" in r.text


def test_summary(client: TestClient, built_dir: Path, dates: list[str]) -> None:
    body = client.get(f"/api/dates/{dates[-1]}/summary").json()
    assert body["n_valid"] == 5 and body["n_pairs"] == 10 and body["status"] == "ok"
    assert len(body["bin_edges"]) == 41 and len(body["counts"]) == 40
    assert sum(body["counts"]) == 10
    store = load_store(built_dir)
    Z, valid = prepare_window(store.returns, 39, FIXTURE_WINDOW)
    C = (Z.T @ Z)[np.ix_(valid, valid)]
    pairs = C[np.triu_indices(5, k=1)]
    assert body["mean"] == pytest.approx(pairs.mean(), abs=1e-12)
    assert body["std"] == pytest.approx(pairs.std(), abs=1e-12)


def test_stock_neighbors(client: TestClient, dates: list[str]) -> None:
    body = client.get(f"/api/corr/{dates[-1]}/stock/na", params={"k": 3}).json()
    assert body["ticker"] == "NA" and body["valid"] is True and body["n_valid"] == 5
    assert len(body["top"]) == 3 and len(body["bottom"]) == 3
    tops = [n["corr"] for n in body["top"]]
    assert tops == sorted(tops, reverse=True)
    names = {n["ticker"] for n in body["top"] + body["bottom"]}
    assert "NA" not in names and "FLAT" not in names  # self and invalid excluded


def test_invalid_stock_is_an_empty_200(client: TestClient, dates: list[str]) -> None:
    body = client.get(f"/api/corr/{dates[-1]}/stock/FLAT").json()
    assert body["valid"] is False and body["top"] == [] and body["bottom"] == []
    assert body["reason"] == "constant_returns"


def test_matrix_with_nulls_for_invalid(client: TestClient, dates: list[str]) -> None:
    r = client.get(f"/api/corr/{dates[-1]}/matrix", params={"tickers": "a, FLAT,MSFT,A"})
    body = r.json()
    assert body["tickers"] == ["A", "FLAT", "MSFT"]
    assert body["valid"] == [True, False, True]
    assert body["reasons"] == ["valid", "constant_returns", "valid"]
    m = body["matrix"]
    assert m[0][0] == 1.0 and m[2][2] == 1.0
    assert m[1] == [None, None, None] and [row[1] for row in m] == [None, None, None]
    assert m[0][2] == m[2][0] and -1 <= m[0][2] <= 1


def test_matrix_ticker_count_limits(client: TestClient, dates: list[str]) -> None:
    r = client.get(f"/api/corr/{dates[-1]}/matrix", params={"tickers": "A"})
    assert r.status_code == 422
    many = ",".join(f"T{k}" for k in range(151))
    r = client.get(f"/api/corr/{dates[-1]}/matrix", params={"tickers": many})
    assert r.status_code == 422
    assert "between 2 and 150" in r.json()["detail"]["message"]


def test_matrix_unknown_tickers_listed(client: TestClient, dates: list[str]) -> None:
    r = client.get(f"/api/corr/{dates[-1]}/matrix", params={"tickers": "A,MSFX,ZZQ"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "unknown_tickers"
    unknown = r.json()["detail"]["unknown"]
    assert set(unknown) == {"MSFX", "ZZQ"}
    assert "MSFT" in unknown["MSFX"]


def test_pair_series(client: TestClient, built_dir: Path, dates: list[str]) -> None:
    body = client.get("/api/corr/pair", params={"a": "A", "b": "msft"}).json()
    assert (body["a"], body["b"], body["window"]) == ("A", "MSFT", FIXTURE_WINDOW)
    assert body["dates"] == dates
    store = load_store(built_dir)
    expected = pair_series(store.returns, 0, 3, FIXTURE_WINDOW)
    for t in range(40):
        if np.isfinite(expected[t]):
            assert body["corr"][t] == pytest.approx(expected[t], abs=1e-12)
        else:
            assert body["corr"][t] is None
    assert body["corr"][:FIXTURE_WINDOW] == [None] * FIXTURE_WINDOW

    ranged = client.get("/api/corr/pair", params={"a": "A", "b": "MSFT", "start": dates[30], "end": dates[35]}).json()
    assert ranged["dates"] == dates[30:36]
    assert ranged["corr"] == body["corr"][30:36]
    bad = client.get("/api/corr/pair", params={"a": "A", "b": "MSFT", "start": dates[35], "end": dates[30]})
    assert bad.status_code == 422


def test_row_aligned_with_meta(client: TestClient, dates: list[str]) -> None:
    body = client.get(f"/api/corr/{dates[-1]}/row/MSFT").json()
    assert len(body["corr"]) == 6
    assert body["corr"][3] == 1.0
    assert body["corr"][2] is None  # FLAT
    assert (body["valid"], body["reason"], body["n_valid"]) == (True, "valid", 5)


def test_unknown_ticker_404_with_suggestions(client: TestClient, dates: list[str]) -> None:
    r = client.get(f"/api/corr/{dates[-1]}/stock/MSFTT")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["ticker"] == "MSFTT"
    assert "MSFT" in detail["suggestions"]


def test_early_date_422_with_first_valid_date(client: TestClient, dates: list[str]) -> None:
    for path in (f"/api/dates/{dates[5]}/summary", f"/api/corr/{dates[5]}/stock/A", f"/api/corr/{dates[5]}/download"):
        r = client.get(path)
        assert r.status_code == 422, path
        assert r.json()["detail"]["first_valid_date"] == dates[FIRST_VALID]


def test_date_errors(client: TestClient) -> None:
    r = client.get("/api/dates/2021-01-09/summary")  # a Saturday
    assert r.status_code == 404
    assert r.json()["detail"]["nearest"] == ["2021-01-08", "2021-01-11"]
    assert client.get("/api/dates/2021-1-9/summary").status_code == 422
    assert client.get("/api/dates/2021-02-30/summary").status_code == 422


def test_k_out_of_range(client: TestClient, dates: list[str]) -> None:
    r = client.get(f"/api/corr/{dates[-1]}/stock/A", params={"k": 201})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "invalid_input" and "k" in detail["message"] and detail["errors"]


def test_download_float16_npy_full_matrix(client: TestClient, built_dir: Path, dates: list[str]) -> None:
    r = client.get(f"/api/corr/{dates[-1]}/download", params={"dtype": "float16", "format": "npy"})
    assert r.status_code == 200
    sid = r.headers["x-snapshot-id"]
    name = f"corr_{dates[-1]}_w{FIXTURE_WINDOW}_float16_{sid}.npy"
    assert r.headers["content-disposition"] == f'attachment; filename="{name}"'
    assert int(r.headers["content-length"]) == len(r.content)
    got = np.load(io.BytesIO(r.content))
    assert got.dtype == np.float16 and got.shape == (6, 6)

    store = load_store(built_dir)
    Z, valid = prepare_window(store.returns, 39, FIXTURE_WINDOW)
    rows = []
    for _, block in iter_full_blocks(Z, valid):
        rows.append(block)
    expected = np.vstack(rows).astype(np.float16)
    np.testing.assert_array_equal(got, expected)
    assert np.isnan(got[2]).all() and np.isnan(got[:, 2]).all()
    brute = Z.T @ Z
    finite = np.isfinite(brute)
    np.testing.assert_allclose(got[finite], brute[finite], atol=1e-3)


def test_unknown_route_is_json_404(client: TestClient) -> None:
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "not_found"


def test_frontend_is_served_once_built(built_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.main as main_module

    dist = tmp_path / "dist"
    monkeypatch.setattr(main_module, "WEB_DIST", dist)
    with TestClient(create_app(built_dir, reload_seconds=0)) as c:
        r = c.get("/")
        assert r.status_code == 404 and "npm run build" in r.json()["detail"]["message"]
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html><title>ok</title>")
        r = c.get("/")
        assert r.status_code == 200 and "ok" in r.text
        assert c.get("/api/health").status_code == 200
