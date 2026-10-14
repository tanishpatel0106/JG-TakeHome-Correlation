"""Uniform-price days: explicit records, evidence, look-ahead, overrides, raw prices kept."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from conftest import random_walk_prices, write_price_zip

from corrlib import snapshots, store
from corrlib.build import main as build_main
from corrlib.build import read_overrides, run_build
from corrlib.returns import (
    OverrideError,
    anomaly_records,
    apply_overrides,
    derive_returns,
    diff_anomalies,
    find_uniform_days,
    parse_overrides,
    summarize_anomalies,
)

W = 5
T = 40
TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
PLACEHOLDER, REBASE, LAST = 12, 25, T - 1


def planted_prices() -> np.ndarray:
    """Random walks with a seed day (t=0), a placeholder (t=12), a re-base (t=25) and a uniform last date."""
    path = random_walk_prices(T, len(TICKERS), seed=9)
    P = path.copy()
    P[PLACEHOLDER] = 1.0  # t=13 continues from t=11
    P[REBASE:] = 20.0 * path[REBASE:] / path[REBASE]  # every path restarts at 20.0
    P[LAST] = 7.0  # cannot be classified: no next day
    return np.round(P, 6)  # what the CSV files carry


@pytest.fixture(scope="module")
def dates() -> list[date]:
    return list(pd.bdate_range("2022-03-01", periods=T).date)


@pytest.fixture(scope="module")
def iso(dates: list[date]) -> list[str]:
    return [d.isoformat() for d in dates]


@pytest.fixture(scope="module")
def records(dates: list[date]) -> list[dict]:
    return anomaly_records(planted_prices(), dates, W)


def by_date(records: list[dict]) -> dict[str, dict]:
    return {r["date"]: r for r in records}


def test_every_uniform_day_gets_one_record(records: list[dict], iso: list[str]) -> None:
    assert [r["date"] for r in records] == [iso[0], iso[PLACEHOLDER], iso[REBASE], iso[LAST]]
    assert [r["index"] for r in records] == [0, PLACEHOLDER, REBASE, LAST]
    for r in records:
        assert r["rule"] == "uniform-continuity/v1"
        assert r["n_prices"] == len(TICKERS)
        assert r["override"] is None
        json.dumps(r, allow_nan=False)  # JSON-ready, no NaN
    assert [r["uniform_value"] for r in records] == [20.0, 1.0, 20.0, 7.0]


def test_first_day_is_a_rebase_without_look_ahead(records: list[dict], iso: list[str]) -> None:
    r = records[0]
    assert (r["classification"], r["status"], r["action"], r["look_ahead"]) == ("rebase", "inferred", "none", False)
    assert r["observation_dates_used"] == [iso[0]]
    assert r["evidence"]["previous_date"] is None and r["evidence"]["next_date"] == iso[1]
    assert r["evidence"]["median_abs_log_from_uniform"] is None
    assert r["affected"] == {
        "returns_nulled_dates": [],
        "prices_nulled_date": None,
        "window_end_dates": {"first": None, "last": None, "count": 0},
    }


def test_potential_rebase_is_inferred_with_evidence_and_look_ahead(records: list[dict], iso: list[str]) -> None:
    r = by_date(records)[iso[REBASE]]
    assert (r["classification"], r["status"], r["action"]) == ("rebase", "inferred", "return_into_day_nulled")
    assert r["look_ahead"] is True
    assert r["observation_dates_used"] == [iso[REBASE - 1], iso[REBASE], iso[REBASE + 1]]
    ev = r["evidence"]
    assert (ev["previous_date"], ev["next_date"]) == (iso[REBASE - 1], iso[REBASE + 1])
    assert ev["n_compared"] == len(TICKERS)
    assert ev["median_abs_log_from_uniform"] < ev["median_abs_log_from_previous"]
    assert ev["rule_classification"] == "rebase" and "re-base" in ev["note"]
    P = planted_prices()
    expected = np.median(np.abs(np.log(P[REBASE + 1] / P[REBASE])))
    assert ev["median_abs_log_from_uniform"] == pytest.approx(expected, abs=1e-6)
    assert r["affected"] == {
        "returns_nulled_dates": [iso[REBASE]],
        "prices_nulled_date": None,
        "window_end_dates": {"first": iso[REBASE], "last": iso[REBASE + W - 1], "count": W},
    }


def test_placeholder_nulls_prices_and_two_returns(records: list[dict], iso: list[str]) -> None:
    r = by_date(records)[iso[PLACEHOLDER]]
    assert (r["classification"], r["status"], r["action"]) == ("placeholder", "inferred", "prices_nulled")
    assert r["look_ahead"] is True
    ev = r["evidence"]
    assert ev["median_abs_log_from_previous"] < ev["median_abs_log_from_uniform"]
    assert r["affected"] == {
        "returns_nulled_dates": [iso[PLACEHOLDER], iso[PLACEHOLDER + 1]],
        "prices_nulled_date": iso[PLACEHOLDER],
        "window_end_dates": {"first": iso[PLACEHOLDER], "last": iso[PLACEHOLDER + W], "count": W + 1},
    }


def test_uniform_last_date_is_unresolved_and_excluded(records: list[dict], iso: list[str], dates: list[date]) -> None:
    r = by_date(records)[iso[LAST]]
    assert (r["classification"], r["status"], r["action"]) == ("unresolved", "unresolved", "excluded_pending_next_day")
    assert r["look_ahead"] is False
    assert r["observation_dates_used"] == [iso[LAST]]
    assert r["evidence"]["next_date"] is None and "next day" in r["evidence"]["note"]
    assert r["affected"]["prices_nulled_date"] == iso[LAST]
    assert r["affected"]["window_end_dates"] == {"first": iso[LAST], "last": iso[LAST], "count": 1}

    raw = planted_prices()
    derived = derive_returns(raw, dates, W)
    assert np.isnan(derived.prices[LAST]).all()
    assert (raw[LAST] == 7.0).all()  # raw evidence untouched
    assert np.isnan(derived.returns[LAST]).all()
    assert derived.prices_nulled_pending == len(TICKERS)
    assert np.flatnonzero(derived.unresolved).tolist() == [LAST]


def test_next_day_reclassifies_the_last_date_and_the_change_is_visible(dates: list[date], iso: list[str]) -> None:
    raw = planted_prices()
    before = anomaly_records(raw, dates, W)
    # The next day continues from the day before the uniform day: a placeholder after all.
    next_day = raw[LAST - 1] * 1.01
    extended = np.vstack([raw, next_day])
    later_dates = [*dates, date(2022, 4, 26)]
    after = anomaly_records(extended, later_dates, W)
    r = by_date(after)[iso[LAST]]
    assert (r["classification"], r["status"], r["look_ahead"]) == ("placeholder", "inferred", True)
    assert diff_anomalies(before, after) == [
        {
            "date": iso[LAST],
            "from": {"classification": "unresolved", "status": "unresolved"},
            "to": {"classification": "placeholder", "status": "inferred"},
        }
    ]
    assert diff_anomalies(before, before) == []


def test_derived_prices_and_returns(dates: list[date]) -> None:
    raw = planted_prices()
    copy = raw.copy()
    d = derive_returns(raw, dates, W)
    np.testing.assert_array_equal(raw, copy)  # pure: the input is not modified
    assert np.isnan(d.prices[PLACEHOLDER]).all() and np.isnan(d.prices[LAST]).all()
    assert np.isfinite(d.prices[REBASE]).all()
    assert np.isnan(d.returns[[0, PLACEHOLDER, PLACEHOLDER + 1, REBASE, LAST]]).all()
    finite_rows = np.setdiff1d(np.arange(T), [0, PLACEHOLDER, PLACEHOLDER + 1, REBASE, LAST])
    assert np.isfinite(d.returns[finite_rows]).all()
    assert d.returns_nulled_at_rebase == len(TICKERS)
    assert d.prices_nulled_on_placeholder_days == len(TICKERS)
    assert (np.flatnonzero(d.rebase).tolist(), np.flatnonzero(d.placeholder).tolist()) == ([0, REBASE], [PLACEHOLDER])


def test_confirmed_placeholder_override(dates: list[date], iso: list[str]) -> None:
    overrides = parse_overrides(
        {"overrides": [{"date": iso[PLACEHOLDER], "classification": "placeholder", "reason": "vendor confirmed", "author": "ops"}]}
    )
    records = anomaly_records(planted_prices(), dates, W, overrides)
    r = by_date(records)[iso[PLACEHOLDER]]
    assert (r["classification"], r["status"], r["action"]) == ("placeholder", "confirmed", "prices_nulled")
    assert r["override"] == {"classification": "placeholder", "reason": "vendor confirmed", "author": "ops"}
    assert r["evidence"]["rule_classification"] == "placeholder"
    summary = summarize_anomalies(records)
    assert summary["by_status"] == {"inferred": 2, "confirmed": 1, "unresolved": 1}
    assert summary["overridden_count"] == 1 and summary["unresolved_dates"] == [iso[LAST]]


def test_override_can_change_and_resolve_a_classification(dates: list[date], iso: list[str]) -> None:
    overrides = parse_overrides(
        {
            "overrides": [
                {"date": iso[PLACEHOLDER], "classification": "rebase", "reason": "r1", "author": "a"},
                {"date": iso[LAST], "classification": "placeholder", "reason": "r2", "author": "b"},
            ]
        }
    )
    d = derive_returns(planted_prices(), dates, W, overrides)
    changed = by_date(d.records)
    r = changed[iso[PLACEHOLDER]]
    assert (r["classification"], r["status"], r["action"]) == ("rebase", "confirmed", "return_into_day_nulled")
    assert r["evidence"]["rule_classification"] == "placeholder"  # the rule's verdict stays visible
    assert r["affected"]["window_end_dates"]["count"] == W
    assert np.isfinite(d.prices[PLACEHOLDER]).all()  # kept as a base now
    last = changed[iso[LAST]]
    assert (last["classification"], last["status"], last["action"]) == ("placeholder", "confirmed", "prices_nulled")
    assert diff_anomalies(anomaly_records(planted_prices(), dates, W), d.records)[0]["date"] == iso[PLACEHOLDER]


def test_override_of_a_non_uniform_day_is_rejected(dates: list[date], iso: list[str]) -> None:
    for day in (iso[5], "2030-01-01"):
        overrides = parse_overrides({"overrides": [{"date": day, "classification": "placeholder", "reason": "x", "author": "y"}]})
        with pytest.raises(OverrideError, match="not a detected uniform-price day"):
            anomaly_records(planted_prices(), dates, W, overrides)
    with pytest.raises(OverrideError):
        apply_overrides([], [{"date": iso[1], "classification": "rebase", "reason": "x", "author": "y"}], dates, W)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "JSON object"),
        ({"overrides": {}}, "JSON object"),
        ({"overrides": ["x"]}, "not an object"),
        ({"overrides": [{"date": "2021-W08-1", "classification": "rebase", "reason": "r", "author": "a"}]}, "YYYY-MM-DD"),
        ({"overrides": [{"date": "2021-02-30", "classification": "rebase", "reason": "r", "author": "a"}]}, "YYYY-MM-DD"),
        ({"overrides": [{"date": "2021-02-22", "classification": "unresolved", "reason": "r", "author": "a"}]}, "classification"),
        ({"overrides": [{"date": "2021-02-22", "classification": "rebase", "reason": " ", "author": "a"}]}, "reason"),
        ({"overrides": [{"date": "2021-02-22", "classification": "rebase", "reason": "r"}]}, "author"),
        (
            {"overrides": [{"date": "2021-02-22", "classification": "rebase", "reason": "r", "author": "a"}] * 2},
            "twice",
        ),
    ],
)
def test_malformed_overrides_are_rejected(document: object, message: str) -> None:
    with pytest.raises(OverrideError, match=message):
        parse_overrides(document)


def test_find_uniform_days_ignores_sparse_days() -> None:
    P = random_walk_prices(6, 10, seed=1)
    P[3, :4] = 5.0
    P[3, 4:] = np.nan  # only 4 of 10 priced: not judged
    assert np.flatnonzero(find_uniform_days(P)).tolist() == [0]


# --- through the build ------------------------------------------------------------


@pytest.fixture(scope="module")
def planted_zip(tmp_path_factory: pytest.TempPathFactory, dates: list[date]) -> Path:
    return write_price_zip(tmp_path_factory.mktemp("planted") / "planted.zip", planted_prices(), TICKERS, dates)


def test_build_writes_records_and_keeps_raw_prices(planted_zip: Path, tmp_path: Path, iso: list[str]) -> None:
    out = tmp_path / "out"
    manifest, _ = run_build(planted_zip, out, window=W)
    records = store.read_anomalies(out)
    assert [r["classification"] for r in records] == ["rebase", "placeholder", "rebase", "unresolved"]
    assert manifest["anomalies_summary"]["by_classification"] == {"rebase": 2, "placeholder": 1, "unresolved": 1}
    assert manifest["anomalies_summary"]["look_ahead_count"] == 2
    assert manifest["structural"]["unresolved_dates"] == [iso[LAST]]
    assert manifest["structural"]["placeholder_dates"] == [iso[PLACEHOLDER]]
    assert manifest["overrides"] is None
    assert manifest["view"] == "retrospective"

    table = pq.read_table(store.snapshot_path(out) / "prices.parquet")
    raw = table.column("price_raw").to_numpy(zero_copy_only=False).reshape(T, len(TICKERS))
    used = table.column("price").to_numpy(zero_copy_only=False).reshape(T, len(TICKERS))
    np.testing.assert_allclose(raw, planted_prices(), rtol=1e-12)
    assert (raw[PLACEHOLDER] == 1.0).all() and np.isnan(used[PLACEHOLDER]).all()
    assert (raw[LAST] == 7.0).all() and np.isnan(used[LAST]).all()

    s = store.load_store(out)
    assert s.anomalies == records
    early = s.n_valid[PLACEHOLDER : PLACEHOLDER + W + 1]
    assert (early == 0).all()
    assert s.n_valid[LAST] == 0  # its window contains the excluded last date


def test_build_with_overrides_copies_and_hashes_the_file(planted_zip: Path, tmp_path: Path, iso: list[str]) -> None:
    path = tmp_path / "overrides.json"
    text = json.dumps(
        {"overrides": [{"date": iso[PLACEHOLDER], "classification": "placeholder", "reason": "confirmed by vendor", "author": "qa"}]},
        indent=2,
    )
    path.write_text(text + "\n", encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "out"
    manifest, _ = run_build(planted_zip, out, window=W, overrides=path)

    assert (store.snapshot_path(out) / "overrides.json").read_bytes() == path.read_bytes()
    assert manifest["overrides"] == {"file": "overrides.json", "sha256": sha, "count": 1}
    assert manifest["artifacts"]["overrides.json"]["sha256"] == sha
    assert store.read_inputs(out)["overrides_sha256"] == sha
    assert store.verify_manifest(out) == []
    r = by_date(store.read_anomalies(out))[iso[PLACEHOLDER]]
    assert (r["status"], r["override"]["author"]) == ("confirmed", "qa")
    assert read_overrides(path).sha256 == sha


def test_build_rejects_override_of_non_uniform_day_and_leaves_output_alone(
    planted_zip: Path, tmp_path: Path, iso: list[str]
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"overrides": [{"date": iso[3], "classification": "rebase", "reason": "r", "author": "a"}]}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "marker").write_text("previous")
    with pytest.raises(OverrideError):
        run_build(planted_zip, out, window=W, overrides=path)
    assert build_main(["--zip", str(planted_zip), "--out", str(out), "--window", str(W), "--overrides", str(path)]) == 2
    # Nothing published, nothing half-written, the lock released; only the audit log is new.
    assert sorted(p.name for p in out.iterdir()) == ["ingest_log.jsonl", "marker"]
    assert (out / "marker").read_text() == "previous"
    attempts = snapshots.read_log(out)
    assert [(a["operation"], a["status"]) for a in attempts] == [("build", "failed"), ("build", "failed")]
    assert "OverrideError" in attempts[0]["reason"] and "not a detected uniform-price day" in attempts[0]["reason"]
    path.write_text("{not json")
    with pytest.raises(OverrideError, match="not valid JSON"):
        read_overrides(path)
