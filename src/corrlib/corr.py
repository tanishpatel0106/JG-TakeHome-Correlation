"""Correlation math. Pure numpy: arrays in, arrays out, no I/O, no state.

Contract (see README, "Approach & math")
-----------------------------------------
* ``R`` is ``[T x N]`` simple returns, ``R[0]`` is NaN.
* The window *as of* date index ``t`` is ``R[t-W+1 .. t]`` (W returns, W+1
  prices). The first date with a window is ``t = W``. For ``t < W`` nothing
  is valid, because the window would need ``R[0]``.
* Stock ``i`` is **valid** at ``t`` iff ``t >= W``, its W returns are all
  finite and its centred window norm is > ``eps`` (complete observations
  only). Every invalid stock gets a reason code (``config.REASON_*``):
  1 insufficient_history, 2 missing_returns, 3 constant_returns.
  :func:`window_status` is the one implementation of this rule;
  :func:`reason_series` is its vectorised form over all dates for one column.
* For valid ``i``: ``z_i = (x - mean x) / ||x - mean x||``. Then Pearson
  ``rho_ij = z_i . z_j`` exactly, i.e. ``C = Z^T Z``. Invalid stocks give NaN
  correlations with everything, including themselves.

Correlations returned to callers are clipped to [-1, 1] (rounding can push a
dot product of unit vectors a few ulp outside).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import (
    BLOCK_ROWS,
    HIST_BLOCK_ROWS,
    REASON_CONSTANT_RETURNS,
    REASON_INSUFFICIENT_HISTORY,
    REASON_MISSING_RETURNS,
    REASON_VALID,
    VALIDITY_EPS,
    WINDOW,
)


def window_status(
    R: np.ndarray, t: int, W: int = WINDOW, eps: float = VALIDITY_EPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised window matrix for date index ``t``, with a reason per stock.

    This is the single implementation of the validity policy
    (``config.build_policy``): every other function here either calls it or
    applies the same three tests.

    Parameters
    ----------
    R:
        Returns ``[T x N]`` (any float dtype, may be a read-only memmap).
    t:
        Date index, ``0 <= t < T``.
    W:
        Window length in returns.
    eps:
        Minimum centred norm for a stock to be valid.

    Returns
    -------
    (Z, valid, reason)
        ``Z`` float64 ``[W x N]`` with unit-norm, zero-mean columns for valid
        stocks and NaN columns otherwise. ``valid`` bool ``[N]``.
        ``reason`` int8 ``[N]``: ``REASON_VALID`` (0) where valid, else
        ``REASON_INSUFFICIENT_HISTORY`` (1, every stock when ``t < W``),
        ``REASON_MISSING_RETURNS`` (2, any return in the window is NaN/inf) or
        ``REASON_CONSTANT_RETURNS`` (3, complete window with norm <= eps).
    """
    n_dates, n_stocks = R.shape
    if not 0 <= t < n_dates:
        raise IndexError(f"date index {t} out of range [0, {n_dates})")
    Z = np.full((W, n_stocks), np.nan, dtype=np.float64)
    if t < W:
        reason = np.full(n_stocks, REASON_INSUFFICIENT_HISTORY, dtype=np.int8)
        return Z, np.zeros(n_stocks, dtype=bool), reason

    X = np.asarray(R[t - W + 1 : t + 1], dtype=np.float64)
    finite = np.isfinite(X).all(axis=0)
    centred = X - X.mean(axis=0)
    norm = np.sqrt(np.einsum("ij,ij->j", centred, centred))
    reason = _reasons(finite, norm, eps)
    valid = reason == REASON_VALID
    Z[:, valid] = centred[:, valid] / norm[valid]
    return Z, valid, reason


def _reasons(finite: np.ndarray, norm: np.ndarray, eps: float) -> np.ndarray:
    """Reason codes (int8) from "all returns finite" and the centred norm (same shape)."""
    reason = np.full(finite.shape, REASON_VALID, dtype=np.int8)
    reason[finite & ~(norm > eps)] = REASON_CONSTANT_RETURNS
    reason[~finite] = REASON_MISSING_RETURNS
    return reason


def prepare_window(
    R: np.ndarray, t: int, W: int = WINDOW, eps: float = VALIDITY_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised window matrix for date index ``t`` (no reasons).

    Same as :func:`window_status` without the reason codes: returns ``(Z,
    valid)`` with ``Z`` float64 ``[W x N]`` (NaN columns for invalid stocks)
    and ``valid`` bool ``[N]``. For ``t < W`` every stock is invalid.
    """
    Z, valid, _ = window_status(R, t, W, eps)
    return Z, valid


def reason_series(x: np.ndarray, W: int = WINDOW, eps: float = VALIDITY_EPS) -> np.ndarray:
    """Reason code of one stock at every date: the vectorised :func:`window_status`.

    Parameters
    ----------
    x:
        One return column ``[T]`` (any float dtype).

    Returns
    -------
    np.ndarray
        int8 ``[T]``; ``out[t] == window_status(R, t, W, eps)[2][i]`` for the
        column ``x = R[:, i]``. The first ``W`` values are
        ``REASON_INSUFFICIENT_HISTORY``.
    """
    reason, _, _ = _column_windows(np.asarray(x, dtype=np.float64), W, eps)
    return reason


def _column_windows(
    x: np.ndarray, W: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trailing windows of one float64 column.

    Returns ``(reason int8 [T], centred float64 [T-W x W], norm float64 [T-W])``
    where row ``k`` of ``centred``/``norm`` is the window ending at ``t = W + k``.
    Empty arrays (0 rows) when ``T <= W``.
    """
    n_dates = x.shape[0]
    reason = np.full(n_dates, REASON_INSUFFICIENT_HISTORY, dtype=np.int8)
    if n_dates <= W:
        return reason, np.empty((0, W)), np.empty(0)
    # Window k covers indices k .. k+W-1 and ends at t = k + W - 1. Windows
    # that may be valid end at t >= W, i.e. k >= 1.
    xw = sliding_window_view(x, W)[1:]
    finite = np.isfinite(xw).all(axis=1)
    centred = xw - xw.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->i", centred, centred))
    reason[W:] = _reasons(finite, norm, eps)
    return reason, centred, norm


def one_vs_all(Z: np.ndarray, valid: np.ndarray, i: int) -> np.ndarray:
    """Correlation of stock ``i`` with every stock: ``Z^T z_i``. Cost O(N W).

    Returns
    -------
    np.ndarray
        float64 ``[N]`` in [-1, 1]. All NaN if ``i`` is invalid. NaN for every
        invalid stock. ``out[i] == 1.0`` when ``i`` is valid.
    """
    out = np.full(Z.shape[1], np.nan, dtype=np.float64)
    if not valid[i]:
        return out
    out[valid] = Z[:, valid].T @ Z[:, i]
    np.clip(out, -1.0, 1.0, out=out)
    out[i] = 1.0
    return out


def subset_matrix(Z: np.ndarray, valid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Correlation matrix of the stocks ``idx`` (k of them). Cost O(k^2 W).

    Returns
    -------
    np.ndarray
        float64 ``[k x k]``, exactly symmetric, in [-1, 1], diagonal 1.0 for
        valid stocks. Rows and columns of invalid stocks are NaN (diagonal too).
    """
    idx = np.asarray(idx, dtype=np.intp)
    ok = valid[idx]
    # Zero the NaN columns only so the GEMM is defined; they are reset to NaN below.
    sub = np.where(ok, Z[:, idx], 0.0)
    C = sub.T @ sub
    C = 0.5 * (C + C.T)
    np.clip(C, -1.0, 1.0, out=C)
    C[~ok, :] = np.nan
    C[:, ~ok] = np.nan
    diag = np.flatnonzero(ok)
    C[diag, diag] = 1.0
    return C


def iter_full_blocks(
    Z: np.ndarray,
    valid: np.ndarray,
    block: int = BLOCK_ROWS,
    dtype: type[np.floating] = np.float32,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream the full ``[N x N]`` correlation matrix as row blocks.

    Yields ``(r0, rows)`` where ``rows`` is ``[b x N]`` (``b <= block``) holding
    matrix rows ``r0 .. r0+b-1`` in ``dtype``. Same NaN / clip / diagonal rules
    as :func:`subset_matrix`. Peak memory is O(block * N), never O(N^2).
    """
    n_stocks = Z.shape[1]
    # Zero the NaN columns only so the GEMM is defined; they are reset to NaN below.
    Zd = np.where(valid, Z, 0.0).astype(dtype)
    invalid = ~valid
    any_invalid = bool(invalid.any())
    for r0 in range(0, n_stocks, block):
        r1 = min(r0 + block, n_stocks)
        rows = Zd[:, r0:r1].T @ Zd
        np.clip(rows, -1.0, 1.0, out=rows)
        if any_invalid:
            rows[:, invalid] = np.nan
            rows[invalid[r0:r1], :] = np.nan
        local = np.flatnonzero(valid[r0:r1])
        rows[local, local + r0] = 1.0
        yield r0, rows


def mean_and_std_pairwise(Z: np.ndarray, valid: np.ndarray) -> tuple[float, float, int]:
    """Mean and std of ``rho_ij`` over all pairs ``i < j`` of valid stocks.

    Uses no N x N matrix. With ``s = sum_i z_i`` over the ``N_v`` valid columns:

    * ``sum_{i != j} rho_ij = ||s||^2 - N_v``
    * ``sum_{i != j} rho_ij^2 = ||Z Z^T||_F^2 - N_v`` (``Z Z^T`` is only W x W)

    Dividing both by ``N_v (N_v - 1)`` gives the mean and the mean square over
    the unique pairs. The std is the population std (ddof=0) over the pairs:
    the pairs are the whole population, not a sample.

    Returns
    -------
    (mean, std, n_valid)
        ``mean`` and ``std`` are NaN when ``n_valid < 2``.
    """
    Zv = Z[:, valid]
    n_valid = int(Zv.shape[1])
    if n_valid < 2:
        return float("nan"), float("nan"), n_valid
    n_ordered = n_valid * (n_valid - 1)
    s = Zv.sum(axis=1)
    mean = (float(s @ s) - n_valid) / n_ordered
    gram = Zv @ Zv.T
    mean_sq = (float(np.einsum("ij,ij->", gram, gram)) - n_valid) / n_ordered
    variance = max(mean_sq - mean * mean, 0.0)
    return mean, variance**0.5, n_valid


def histogram_pairwise(
    Z: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    block: int = HIST_BLOCK_ROWS,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """Histogram of ``rho_ij`` over the upper triangle (``i < j``) of valid stocks.

    Built block by block: each block of rows is multiplied only against the
    columns to its right, so roughly half of the N x N products are formed and
    peak memory is O(block * N). Small blocks keep each product in cache while
    it is binned.

    Parameters
    ----------
    edges:
        ``B + 1`` increasing, uniformly spaced bin edges. Values are clipped to
        ``[edges[0], edges[-1]]``; bins are half-open except the last, which is
        closed (as in ``np.histogram``), up to rounding at the edges.

    Returns
    -------
    np.ndarray
        int64 ``[B]``; sums to ``N_v (N_v - 1) / 2``.
    """
    edges = np.asarray(edges, dtype=np.float64)
    n_bins = edges.size - 1
    lo, hi = float(edges[0]), float(edges[-1])
    if n_bins < 1 or not np.allclose(np.diff(edges), (hi - lo) / n_bins, rtol=1e-9, atol=0):
        raise ValueError("histogram_pairwise needs uniformly spaced, increasing edges")

    counts = np.zeros(n_bins, dtype=np.int64)
    Zv = np.ascontiguousarray(Z[:, valid], dtype=dtype)
    n_valid = Zv.shape[1]
    for r0 in range(0, n_valid, block):
        r1 = min(r0 + block, n_valid)
        b = r1 - r0
        # rows[a, c] = rho(r0 + a, r0 + c); keep c > a only.
        rows = Zv[:, r0:r1].T @ Zv[:, r0:]
        upper = np.triu_indices(b, k=1)
        counts += _uniform_bin_counts(rows[:, :b][upper], lo, hi, n_bins)
        if r1 < n_valid:
            counts += _uniform_bin_counts(rows[:, b:], lo, hi, n_bins)
    return counts


def _uniform_bin_counts(values: np.ndarray, lo: float, hi: float, n_bins: int) -> np.ndarray:
    """Counts of ``values`` in ``n_bins`` equal bins on [lo, hi] (values clipped)."""
    scale = n_bins / (hi - lo)
    position = values * values.dtype.type(scale)
    position -= values.dtype.type(lo * scale)
    np.clip(position, 0, n_bins, out=position)
    # position >= 0, so truncation is floor. Values equal to `hi` land in index
    # n_bins and are folded into the last (closed) bin.
    index = position.astype(np.uint8 if n_bins < 255 else np.intp)
    counts = np.bincount(index.ravel(), minlength=n_bins + 1)
    counts[n_bins - 1] += counts[n_bins]
    return counts[:n_bins].astype(np.int64)


def pair_series(
    R: np.ndarray, i: int, j: int, W: int = WINDOW, eps: float = VALIDITY_EPS
) -> np.ndarray:
    """Trailing-window correlation of stocks ``i`` and ``j`` at every date. Cost O(T W).

    Computed directly from the two return columns with
    ``sliding_window_view``; validity is :func:`reason_series` of each column,
    i.e. exactly the rule of :func:`window_status`.

    Returns
    -------
    np.ndarray
        float64 ``[T]`` in [-1, 1]. The first ``W`` values are NaN. NaN wherever
        either stock is invalid in that window (so ``pair_series(R, i, i)`` is
        1.0 where ``i`` is valid and NaN elsewhere).
    """
    x = np.asarray(R[:, i], dtype=np.float64)
    y = np.asarray(R[:, j], dtype=np.float64)
    n_dates = x.shape[0]
    out = np.full(n_dates, np.nan, dtype=np.float64)
    if n_dates <= W:
        return out

    reason_x, xc, nx = _column_windows(x, W, eps)
    reason_y, yc, ny = _column_windows(y, W, eps)
    ok = (reason_x[W:] == REASON_VALID) & (reason_y[W:] == REASON_VALID)

    rho = np.full(ok.shape, np.nan, dtype=np.float64)
    rho[ok] = np.einsum("ij,ij->i", xc[ok], yc[ok]) / (nx[ok] * ny[ok])
    np.clip(rho, -1.0, 1.0, out=rho)
    out[W:] = rho
    return out


def top_bottom_k(
    row: np.ndarray, k: int, exclude: int | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the ``k`` largest and ``k`` smallest finite values of ``row``.

    NaN entries and ``exclude`` (e.g. the query stock itself) are ignored, so
    invalid stocks are never ranked. The order is a total order, so the result
    is fully deterministic, including exact ties at the k-th boundary:

    * ``top``: by value descending, then column index ascending;
    * ``bottom``: by value ascending, then column index ascending.

    ``-0.0`` and ``0.0`` compare equal (ties broken by index). Uses
    ``np.lexsort`` over the finite candidates: O(N log N), ~0.3 ms at N = 5,000.

    Returns
    -------
    (top, bottom)
        intp arrays of length ``min(k, n_candidates)``. They may overlap when
        there are fewer than ``2k`` candidates.
    """
    candidates = np.flatnonzero(np.isfinite(row))
    if exclude is not None:
        candidates = candidates[~np.isin(candidates, np.atleast_1d(exclude))]
    k = min(int(k), candidates.size)
    if k <= 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy()
    values = np.asarray(row[candidates], dtype=np.float64)
    # np.lexsort sorts by the last key first; candidates are column indices.
    top = np.lexsort((candidates, -values))[:k]
    bottom = np.lexsort((candidates, values))[:k]
    return candidates[top].astype(np.intp), candidates[bottom].astype(np.intp)
