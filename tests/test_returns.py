from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from conftest import FixtureData

from corrlib.build import ingest
from corrlib.returns import (
    classify_uniform_days,
    find_uniform_days,
    simple_returns,
    ticker_universe,
    to_wide,
)

NAN = np.nan


def test_hand_computed_returns() -> None:
    P = np.array([[10.0, 20.0], [11.0, 18.0], [12.1, 18.0]])
    R = simple_returns(P)
    expected = np.array([[NAN, NAN], [0.1, -0.1], [0.1, 0.0]])
    np.testing.assert_allclose(R, expected, rtol=0, atol=1e-15, equal_nan=True)


def test_first_row_is_nan() -> None:
    R = simple_returns(np.full((5, 3), 7.0))
    assert np.isnan(R[0]).all()
    assert (R[1:] == 0).all()


def test_gap_gives_nan_in_both_affected_returns() -> None:
    P = np.array([[10.0], [11.0], [NAN], [12.0], [13.0]])
    R = simple_returns(P)[:, 0]
    assert np.isnan(R[2]) and np.isnan(R[3])  # into and out of the gap
    np.testing.assert_allclose(R[[1, 4]], [0.1, 13.0 / 12.0 - 1.0])


def test_breaks_null_whole_rows() -> None:
    P = np.array([[10.0, 5.0], [11.0, 6.0], [20.0, 20.0], [21.0, 19.0]])
    breaks = np.array([False, False, True, False])
    R = simple_returns(P, breaks=breaks)
    assert np.isnan(R[2]).all()
    np.testing.assert_allclose(R[3], [0.05, -0.05])


def test_non_finite_results_become_nan() -> None:
    R = simple_returns(np.array([[0.0], [1.0]]))
    assert np.isnan(R[1, 0])


def test_to_wide_places_values_and_leaves_missing_nan() -> None:
    dates = [date(2021, 1, 4), date(2021, 1, 5)]
    frames = [
        pd.DataFrame({"ticker": ["B", "A"], "price": [2.0, 1.0]}),
        pd.DataFrame({"ticker": ["A"], "price": [NAN]}),
    ]
    tickers = ticker_universe(frames)
    P = to_wide(frames, tickers, dates)
    assert tickers == ["A", "B"]
    np.testing.assert_array_equal(P, [[1.0, 2.0], [NAN, NAN]])


def test_uniform_days_are_classified_by_continuity() -> None:
    rng = np.random.default_rng(0)
    n = 50
    path = 20.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(12, n)), axis=0))
    P = path.copy()
    P[0] = 20.0  # seed at the start
    P[4] = 1.0  # placeholder: day 5 continues from day 3
    P[8] = 20.0  # re-base: day 9 continues from 20.0
    P[9:] = 20.0 * path[9:] / path[8]
    uniform = find_uniform_days(P)
    assert np.flatnonzero(uniform).tolist() == [0, 4, 8]
    rebase, placeholder, unresolved = classify_uniform_days(P, uniform)
    assert np.flatnonzero(rebase).tolist() == [0, 8]
    assert np.flatnonzero(placeholder).tolist() == [4]
    assert not unresolved.any()


def test_uniform_last_date_is_unresolved() -> None:
    P = np.full((4, 3), 20.0)
    P[1:3] = [[21.0, 19.0, 20.5], [22.0, 18.0, 20.1]]
    P[3] = 1.0
    uniform = find_uniform_days(P)
    assert np.flatnonzero(uniform).tolist() == [0, 3]
    rebase, placeholder, unresolved = classify_uniform_days(P, uniform)
    assert np.flatnonzero(rebase).tolist() == [0]
    assert not placeholder.any()
    assert np.flatnonzero(unresolved).tolist() == [3]


def test_uniform_day_needs_coverage() -> None:
    P = np.full((3, 10), NAN)
    P[1, :2] = 5.0  # only 2 of 10 tickers present: not judged
    P[2, :] = 5.0
    assert find_uniform_days(P).tolist() == [False, False, True]


def test_fixture_panel_matches_ground_truth(fixture_data: FixtureData) -> None:
    panel = ingest(fixture_data.zip_path)
    assert panel.tickers == fixture_data.tickers
    assert panel.dates == fixture_data.dates
    np.testing.assert_allclose(panel.prices, fixture_data.expected_prices, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(panel.prices_raw, fixture_data.expected_prices, rtol=1e-12, equal_nan=True)
    # Day 0 is the uniform 20.0 seed: a re-base at t=0, which changes nothing.
    assert np.flatnonzero(panel.rebase).tolist() == [0]
    assert not panel.placeholder.any() and not panel.unresolved.any()
    assert [(r["date"], r["classification"], r["status"], r["action"]) for r in panel.anomalies] == [
        ("2021-01-04", "rebase", "inferred", "none")
    ]
    assert len(panel.issues_by_date) == len(panel.dates)
    assert sum(c.files for c in panel.issues_by_date) == len(panel.dates)
    np.testing.assert_allclose(
        panel.returns, simple_returns(fixture_data.expected_prices), rtol=1e-12, equal_nan=True
    )
