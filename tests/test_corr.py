from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from corrlib.corr import (
    histogram_pairwise,
    iter_full_blocks,
    mean_and_std_pairwise,
    one_vs_all,
    pair_series,
    prepare_window,
    subset_matrix,
    top_bottom_k,
)
from corrlib.returns import simple_returns

W = 20
EDGES = np.linspace(-1.0, 1.0, 41)


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    """60 dates x 40 stocks of correlated returns (factor model), R[0] NaN."""
    rng = np.random.default_rng(123)
    factor = rng.normal(0, 0.01, size=(60, 1))
    beta = rng.uniform(-1.5, 1.5, size=(1, 40))
    noise = rng.normal(0, 0.01, size=(60, 40))
    prices = 50.0 * np.exp(np.cumsum(factor @ beta + noise, axis=0))
    return simple_returns(prices)


def full_matrix(Z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    C = Z.T @ Z
    C[~valid, :] = np.nan
    C[:, ~valid] = np.nan
    return C


def upper(C: np.ndarray, valid: np.ndarray) -> np.ndarray:
    Cv = C[np.ix_(valid, valid)]
    return Cv[np.triu_indices(Cv.shape[0], k=1)]


def test_one_vs_all_matches_corrcoef(returns: np.ndarray) -> None:
    t = 45
    Z, valid = prepare_window(returns, t, W)
    assert valid.all()
    expected = np.corrcoef(returns[t - W + 1 : t + 1].T)
    for i in (0, 7, 39):
        np.testing.assert_allclose(one_vs_all(Z, valid, i), expected[i], rtol=0, atol=1e-10)


def test_pair_series_matches_pandas_rolling(returns: np.ndarray) -> None:
    R = returns.copy()
    R[30, 3] = np.nan  # a missing return knocks out 20 windows
    series = pair_series(R, 3, 5, W)
    ref = pd.Series(R[:, 3]).rolling(W).corr(pd.Series(R[:, 5])).to_numpy()
    finite = np.isfinite(ref)
    assert finite.sum() > 10
    np.testing.assert_allclose(series[finite], ref[finite], rtol=0, atol=1e-10)
    assert np.isnan(series[:W]).all()
    assert np.isnan(series[30 : 30 + W]).all()
    assert np.isfinite(series[30 + W :]).all()


def test_pair_series_agrees_with_window_math(returns: np.ndarray) -> None:
    series = pair_series(returns, 2, 9, W)
    for t in (W, 33, 59):
        Z, valid = prepare_window(returns, t, W)
        assert series[t] == pytest.approx(one_vs_all(Z, valid, 2)[9], abs=1e-12)


def test_subset_matrix_symmetric_unit_diagonal_and_matches_full(returns: np.ndarray) -> None:
    Z, valid = prepare_window(returns, 50, W)
    idx = np.array([5, 0, 17, 33, 2])
    C = subset_matrix(Z, valid, idx)
    assert C.shape == (5, 5)
    np.testing.assert_array_equal(C, C.T)
    np.testing.assert_array_equal(np.diag(C), 1.0)
    np.testing.assert_allclose(C, full_matrix(Z, valid)[np.ix_(idx, idx)], rtol=0, atol=1e-12)


def test_iter_full_blocks_reassembles_full_matrix(returns: np.ndarray) -> None:
    R = returns.copy()
    R[40, 11] = np.nan
    Z, valid = prepare_window(R, 50, W)
    assert not valid[11]
    blocks = []
    starts = []
    for r0, rows in iter_full_blocks(Z, valid, block=7):
        assert rows.shape[0] <= 7 and rows.dtype == np.float32
        starts.append(r0)
        blocks.append(rows)
    assert starts == list(range(0, 40, 7))
    got = np.vstack(blocks)
    ref = full_matrix(Z, valid)
    np.fill_diagonal(ref, np.where(valid, 1.0, np.nan))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6, equal_nan=True)


def test_mean_std_aggregates_match_brute_force(returns: np.ndarray) -> None:
    R = returns.copy()
    R[45, [1, 2]] = np.nan
    Z, valid = prepare_window(R, 50, W)
    mean, std, n_valid = mean_and_std_pairwise(Z, valid)
    pairs = upper(Z.T @ Z, valid)
    assert n_valid == 38
    assert mean == pytest.approx(pairs.mean(), abs=1e-12)
    assert std == pytest.approx(pairs.std(), abs=1e-12)


def test_mean_std_need_two_valid() -> None:
    R = simple_returns(np.vstack([np.full((30, 1), 5.0), np.full((1, 1), 5.0)]))
    Z, valid = prepare_window(R, 25, W)
    mean, std, n_valid = mean_and_std_pairwise(Z, valid)
    assert n_valid == 0 and np.isnan(mean) and np.isnan(std)


def test_histogram_counts_sum_to_pairs_and_match_numpy(returns: np.ndarray) -> None:
    R = returns.copy()
    R[42, 0] = np.nan
    Z, valid = prepare_window(R, 50, W)
    n_valid = int(valid.sum())
    counts = histogram_pairwise(Z, valid, EDGES, block=6)
    assert counts.dtype == np.int64
    assert counts.sum() == n_valid * (n_valid - 1) // 2
    ref, _ = np.histogram(upper(Z.T @ Z, valid), bins=EDGES)
    exact = histogram_pairwise(Z, valid, EDGES, block=6, dtype=np.float64)
    np.testing.assert_array_equal(exact, ref)
    assert np.abs(counts - ref).sum() <= 2  # float32 may move a value across a bin edge


def test_histogram_puts_perfect_correlation_in_last_bin() -> None:
    x = np.linspace(-1, 1, W)
    Z = np.column_stack([x, x, -x]) / np.linalg.norm(x)
    counts = histogram_pairwise(Z, np.ones(3, dtype=bool), EDGES)
    assert counts[-1] == 1 and counts[0] == 2


def test_zero_variance_stock_is_nan_everywhere(returns: np.ndarray) -> None:
    R = returns.copy()
    R[:, 4] = 0.0  # flat price
    R[0, 4] = np.nan
    Z, valid = prepare_window(R, 50, W)
    assert not valid[4]
    assert np.isnan(Z[:, 4]).all()
    assert np.isnan(one_vs_all(Z, valid, 4)).all()
    row = one_vs_all(Z, valid, 0)
    assert np.isnan(row[4]) and np.isfinite(np.delete(row, 4)).all()
    C = subset_matrix(Z, valid, np.array([0, 4]))
    assert np.isnan(C[1]).all() and np.isnan(C[:, 1]).all()  # diagonal too
    assert C[0, 0] == 1.0


def test_one_missing_return_in_window_invalidates(returns: np.ndarray) -> None:
    R = returns.copy()
    R[31, 6] = np.nan
    for t in (31, 40, 50):
        _, valid = prepare_window(R, t, W)
        assert not valid[6]
    _, valid = prepare_window(R, 51, W)  # window 32..51 no longer contains it
    assert valid[6]


def test_early_dates_are_all_invalid(returns: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    finite = rng.normal(size=returns.shape)  # even with no NaN anywhere
    for t in (0, 5, W - 1):
        Z, valid = prepare_window(finite, t, W)
        assert not valid.any() and np.isnan(Z).all()
    _, valid = prepare_window(finite, W, W)
    assert valid.all()


def test_prepare_window_rejects_out_of_range(returns: np.ndarray) -> None:
    with pytest.raises(IndexError):
        prepare_window(returns, returns.shape[0], W)


def test_top_bottom_k_ignores_nan_and_self() -> None:
    row = np.array([1.0, 0.2, np.nan, -0.7, 0.9, -0.1, 0.5])
    top, bottom = top_bottom_k(row, 3, exclude=0)
    assert top.tolist() == [4, 6, 1]
    assert bottom.tolist() == [3, 5, 1]
    top, bottom = top_bottom_k(row, 50, exclude=0)
    assert len(top) == 5 and 2 not in top and 0 not in top
    empty_top, empty_bottom = top_bottom_k(np.full(4, np.nan), 3)
    assert empty_top.size == 0 and empty_bottom.size == 0
