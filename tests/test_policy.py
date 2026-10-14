"""The validity policy on a hand-made reference dataset: one rule, every corr function agrees.

Reference panel (W = 4, T = 16), one column per edge case:

=======  ================================================  ===========================================
column   prices                                            expected reason at window end t >= W
=======  ================================================  ===========================================
A        full random walk                                  valid everywhere
B        price missing at p = 8 (R[8], R[9] NaN)           missing_returns for t in 8 .. 12 (W+1 ends)
LIST     listed at t = 6 (no price before)                 missing_returns until t = 9, valid from 10
DELIST   delisted after t = 9 (no price after)             valid until t = 9, missing_returns after
FLAT     constant price                                    constant_returns everywhere
STEP     constant until t = 7, then moves                  constant_returns for t <= 7, valid after
C        full random walk                                  valid everywhere (the unaffected neighbour)
=======  ================================================  ===========================================

Every column is ``insufficient_history`` for ``t < W``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corrlib.build import daily_aggregates, hist_edges
from corrlib.config import (
    POLICY,
    REASON_CONSTANT_RETURNS,
    REASON_INSUFFICIENT_HISTORY,
    REASON_MISSING_RETURNS,
    REASON_NAMES,
    REASON_VALID,
    VALIDITY_EPS,
    WINDOW,
    build_policy,
)
from corrlib.corr import (
    histogram_pairwise,
    iter_full_blocks,
    mean_and_std_pairwise,
    one_vs_all,
    pair_series,
    prepare_window,
    reason_series,
    subset_matrix,
    top_bottom_k,
    window_status,
)
from corrlib.returns import simple_returns

W = 4
T = 16
COLUMNS = ["A", "B", "LIST", "DELIST", "FLAT", "STEP", "C"]
COL = {name: j for j, name in enumerate(COLUMNS)}
MISSING_AT, LISTED_AT, DELISTED_AFTER, STEP_UNTIL = 8, 6, 9, 7


def reference_prices() -> np.ndarray:
    rng = np.random.default_rng(42)
    P = 20.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, size=(T, len(COLUMNS))), axis=0))
    P[MISSING_AT, COL["B"]] = np.nan
    P[:LISTED_AT, COL["LIST"]] = np.nan
    P[DELISTED_AFTER + 1 :, COL["DELIST"]] = np.nan
    P[:, COL["FLAT"]] = 12.5
    P[: STEP_UNTIL + 1, COL["STEP"]] = 30.0
    return P


def expected_reasons() -> np.ndarray:
    """int8 [T x N] reason codes worked out by hand from the table in the module docstring."""
    E = np.full((T, len(COLUMNS)), REASON_VALID, dtype=np.int8)
    for t in range(T):
        if t < W:
            E[t] = REASON_INSUFFICIENT_HISTORY
            continue
        if MISSING_AT <= t <= MISSING_AT + W:
            E[t, COL["B"]] = REASON_MISSING_RETURNS
        if t < LISTED_AT + W:
            E[t, COL["LIST"]] = REASON_MISSING_RETURNS
        if t > DELISTED_AFTER:
            E[t, COL["DELIST"]] = REASON_MISSING_RETURNS
        E[t, COL["FLAT"]] = REASON_CONSTANT_RETURNS
        if t <= STEP_UNTIL:
            E[t, COL["STEP"]] = REASON_CONSTANT_RETURNS
    return E


@pytest.fixture(scope="module")
def R() -> np.ndarray:
    return simple_returns(reference_prices())


def test_missing_price_nulls_both_adjacent_returns(R: np.ndarray) -> None:
    b = R[:, COL["B"]]
    assert np.isnan(b[MISSING_AT]) and np.isnan(b[MISSING_AT + 1])
    assert np.isfinite(np.delete(b, [0, MISSING_AT, MISSING_AT + 1])).all()


def test_window_status_matches_hand_worked_reasons(R: np.ndarray) -> None:
    E = expected_reasons()
    for t in range(T):
        Z, valid, reason = window_status(R, t, W)
        assert reason.dtype == np.int8
        np.testing.assert_array_equal(reason, E[t], err_msg=f"t={t}")
        np.testing.assert_array_equal(valid, E[t] == REASON_VALID)
        assert np.isnan(Z[:, ~valid]).all() and np.isfinite(Z[:, valid]).all()
        Z2, valid2 = prepare_window(R, t, W)
        np.testing.assert_array_equal(valid2, valid)
        np.testing.assert_array_equal(Z2, Z)


def test_missing_price_invalidates_exactly_w_plus_one_window_ends(R: np.ndarray) -> None:
    invalid = []
    for t in range(W, T):
        if not window_status(R, t, W)[1][COL["B"]]:
            invalid.append(t)
    assert invalid == list(range(MISSING_AT, MISSING_AT + W + 1))


def test_production_window_missing_price_vs_missing_return() -> None:
    """With W = 20: a missing price kills 21 window ends, a single missing return 20."""
    rng = np.random.default_rng(3)
    P = 20.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(90, 2)), axis=0))
    P[40, 0] = np.nan
    R = simple_returns(P)
    R[40, 1] = np.nan  # e.g. the return into a re-base day
    reasons = np.stack([reason_series(R[:, 0], WINDOW), reason_series(R[:, 1], WINDOW)], axis=1)
    bad_price = np.flatnonzero(reasons[WINDOW:, 0] != REASON_VALID) + WINDOW
    bad_return = np.flatnonzero(reasons[WINDOW:, 1] != REASON_VALID) + WINDOW
    assert bad_price.tolist() == list(range(40, 40 + WINDOW + 1))
    assert bad_return.tolist() == list(range(40, 40 + WINDOW))


def test_reason_series_equals_window_status_per_column(R: np.ndarray) -> None:
    E = expected_reasons()
    for j in range(len(COLUMNS)):
        np.testing.assert_array_equal(reason_series(R[:, j], W), E[:, j], err_msg=COLUMNS[j])


def test_reason_series_agrees_on_random_data_with_gaps() -> None:
    rng = np.random.default_rng(11)
    R = rng.normal(0.0, 0.02, size=(120, 30))
    R[0] = np.nan
    R[rng.random(R.shape) < 0.01] = np.nan
    R[:, 5] = 0.003  # constant, non-zero returns: zero variance
    R[0, 5] = np.nan
    for t in range(R.shape[0]):
        _, _, reason = window_status(R, t, WINDOW)
        for j in (0, 5, 17, 29):
            assert reason_series(R[:, j], WINDOW)[t] == reason[j]


def test_every_corr_function_uses_the_same_validity(R: np.ndarray) -> None:
    edges = np.linspace(-1.0, 1.0, 41)
    all_idx = np.arange(len(COLUMNS))
    for t in range(T):
        Z, valid, _ = window_status(R, t, W)
        n_valid = int(valid.sum())
        # one-vs-all: invalid stock -> all NaN (self too); valid -> NaN exactly at invalid columns
        for i in range(len(COLUMNS)):
            row = one_vs_all(Z, valid, i)
            if valid[i]:
                assert row[i] == 1.0
                np.testing.assert_array_equal(np.isfinite(row), valid)
                top, bottom = top_bottom_k(row, 10, exclude=i)
                assert valid[top].all() and valid[bottom].all() and i not in top
                assert len(top) == n_valid - 1
            else:
                assert np.isnan(row).all()
        # subset matrix and streamed full matrix: NaN rows/cols (diagonal too) for invalid stocks
        C = subset_matrix(Z, valid, all_idx)
        np.testing.assert_array_equal(np.isfinite(np.diag(C)), valid)
        np.testing.assert_array_equal(np.isfinite(C), np.outer(valid, valid))
        full = np.vstack([rows for _, rows in iter_full_blocks(Z, valid, block=3)])
        np.testing.assert_array_equal(np.isfinite(full), np.outer(valid, valid))
        np.testing.assert_allclose(full, C, atol=1e-6, equal_nan=True)
        # aggregates count exactly the valid stocks
        mean, _std, n = mean_and_std_pairwise(Z, valid)
        assert n == n_valid
        assert np.isfinite(mean) == (n_valid >= 2)
        if n_valid >= 2:
            assert histogram_pairwise(Z, valid, edges).sum() == n_valid * (n_valid - 1) // 2
    # pair series: finite iff both valid; self-correlation 1.0 where valid, NaN where not
    for i in range(len(COLUMNS)):
        valid_i = reason_series(R[:, i], W) == REASON_VALID
        self_corr = pair_series(R, i, i, W)
        np.testing.assert_array_equal(np.isfinite(self_corr), valid_i, err_msg=COLUMNS[i])
        np.testing.assert_allclose(self_corr[valid_i], 1.0, atol=1e-12)
        for j in range(len(COLUMNS)):
            valid_j = reason_series(R[:, j], W) == REASON_VALID
            np.testing.assert_array_equal(np.isfinite(pair_series(R, i, j, W)), valid_i & valid_j)


def test_daily_aggregates_follow_the_policy(R: np.ndarray) -> None:
    n_valid, mean, _std, hist = daily_aggregates(R.astype(np.float32), W, hist_edges(), block=2)
    expected = (expected_reasons() == REASON_VALID).sum(axis=1)
    np.testing.assert_array_equal(n_valid, expected)
    np.testing.assert_array_equal(hist.sum(axis=1), expected * (expected - 1) // 2)
    assert np.isnan(mean[:W]).all() and (hist[:W] == 0).all()


def test_constant_series_has_nan_self_correlation(R: np.ndarray) -> None:
    t = T - 1
    Z, valid, reason = window_status(R, t, W)
    flat = COL["FLAT"]
    assert reason[flat] == REASON_CONSTANT_RETURNS and not valid[flat]
    assert np.isnan(one_vs_all(Z, valid, flat)[flat])
    assert np.isnan(subset_matrix(Z, valid, np.array([flat, COL["A"]]))[0, 0])
    assert np.isnan(pair_series(R, flat, flat, W)).all()


def test_valid_neighbour_is_unaffected_by_invalid_columns(R: np.ndarray) -> None:
    a, c = COL["A"], COL["C"]
    ref = pd.Series(R[:, a]).rolling(W).corr(pd.Series(R[:, c])).to_numpy()
    for t in range(W, T):
        Z, valid, _ = window_status(R, t, W)
        alone_Z, alone_valid, _ = window_status(R[:, [a, c]], t, W)
        assert one_vs_all(Z, valid, a)[c] == pytest.approx(ref[t], abs=1e-12)
        assert one_vs_all(Z, valid, a)[c] == pytest.approx(one_vs_all(alone_Z, alone_valid, 0)[1], abs=1e-15)
        assert pair_series(R, a, c, W)[t] == pytest.approx(ref[t], abs=1e-12)


def test_insufficient_history_applies_to_every_stock() -> None:
    R = np.random.default_rng(5).normal(size=(30, 4))  # no NaN at all, even R[0]
    for t in range(WINDOW):
        _, valid, reason = window_status(R, t, WINDOW)
        assert not valid.any() and (reason == REASON_INSUFFICIENT_HISTORY).all()
    _, valid, _ = window_status(R, WINDOW, WINDOW)
    assert valid.all()


def test_eps_boundary_is_strict() -> None:
    x = np.zeros((6, 1))
    x[0] = np.nan
    x[1:, 0] = [0.0, 0.0, 0.0, 0.0, 1e-13]  # centred norm ~ 8e-14 <= eps
    assert window_status(x, 5, 5, VALIDITY_EPS)[2][0] == REASON_CONSTANT_RETURNS
    x[5, 0] = 1e-6
    assert window_status(x, 5, 5, VALIDITY_EPS)[2][0] == REASON_VALID


def test_policy_document() -> None:
    assert POLICY == build_policy(WINDOW, VALIDITY_EPS)
    assert POLICY["window"] == POLICY["min_valid_returns"] == WINDOW
    assert POLICY["eps"] == VALIDITY_EPS
    codes = POLICY["reason_codes"]
    assert [c["code"] for c in codes] == [0, 1, 2, 3]
    assert [c["name"] for c in codes] == list(REASON_NAMES)
    assert REASON_NAMES == ("valid", "insufficient_history", "missing_returns", "constant_returns")
    assert POLICY["view"] == "retrospective"
    assert "symbol renames are not linked" in POLICY["universe"]
    assert "never 0" in POLICY["invalid_semantics"]
    assert build_policy(10)["window_definition"].startswith("the window as of date index t is R[t-10+1")
