"""Correlation math. Pure numpy: arrays in, arrays out, no I/O, no state.

Contract (see README, "Approach & math")
-----------------------------------------
* ``R`` is ``[T x N]`` simple returns, ``R[0]`` is NaN.
* The window *as of* date index ``t`` is ``R[t-W+1 .. t]`` (W returns, W+1
  prices). The first date with a window is ``t = W``. For ``t < W`` nothing
  is valid, because the window would need ``R[0]``.
* Stock ``i`` is **valid** at ``t`` iff its W returns are all finite and its
  centred window norm is > ``eps`` (complete observations only).
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

from .config import BLOCK_ROWS, VALIDITY_EPS, WINDOW


def prepare_window(
    R: np.ndarray, t: int, W: int = WINDOW, eps: float = VALIDITY_EPS
) -> tuple[np.ndarray, np.ndarray]:
    """Standardised window matrix for date index ``t``.

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
    (Z, valid)
        ``Z`` float64 ``[W x N]`` with unit-norm, zero-mean columns for valid
        stocks and NaN columns otherwise. ``valid`` bool ``[N]``.
        For ``t < W`` every stock is invalid.
    """
    n_dates, n_stocks = R.shape
    if not 0 <= t < n_dates:
        raise IndexError(f"date index {t} out of range [0, {n_dates})")
    Z = np.full((W, n_stocks), np.nan, dtype=np.float64)
    if t < W:
        return Z, np.zeros(n_stocks, dtype=bool)

    X = np.asarray(R[t - W + 1 : t + 1], dtype=np.float64)
    finite = np.isfinite(X).all(axis=0)
    centred = X - X.mean(axis=0)
    norm = np.sqrt(np.einsum("ij,ij->j", centred, centred))
    valid = finite & (norm > eps)
    Z[:, valid] = centred[:, valid] / norm[valid]
    return Z, valid


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
    block: int = BLOCK_ROWS,
    dtype: type[np.floating] = np.float32,
) -> np.ndarray:
    """Histogram of ``rho_ij`` over the upper triangle (``i < j``) of valid stocks.

    Built block by block: each block of rows is multiplied only against the
    columns to its right, so roughly half of the N x N products are formed and
    peak memory is O(block * N).

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
    scaled = np.clip(values, lo, hi)
    scaled -= lo
    scaled *= n_bins / (hi - lo)
    # scaled >= 0, so truncation is floor. Values equal to `hi` land in index
    # n_bins and are folded into the last (closed) bin.
    counts = np.bincount(scaled.astype(np.intp).ravel(), minlength=n_bins + 1)
    counts[n_bins - 1] += counts[n_bins]
    return counts[:n_bins].astype(np.int64)


def pair_series(
    R: np.ndarray, i: int, j: int, W: int = WINDOW, eps: float = VALIDITY_EPS
) -> np.ndarray:
    """Trailing-window correlation of stocks ``i`` and ``j`` at every date. Cost O(T W).

    Computed directly from the two return columns with
    ``sliding_window_view``, applying the same validity rule as
    :func:`prepare_window` to both stocks.

    Returns
    -------
    np.ndarray
        float64 ``[T]`` in [-1, 1]. The first ``W`` values are NaN. NaN wherever
        either stock is invalid in that window.
    """
    x = np.asarray(R[:, i], dtype=np.float64)
    y = np.asarray(R[:, j], dtype=np.float64)
    n_dates = x.shape[0]
    out = np.full(n_dates, np.nan, dtype=np.float64)
    if n_dates <= W:
        return out

    # Window k covers indices k .. k+W-1 and ends at t = k + W - 1. Valid
    # windows end at t >= W, i.e. k >= 1.
    xw = sliding_window_view(x, W)[1:]
    yw = sliding_window_view(y, W)[1:]
    finite = np.isfinite(xw).all(axis=1) & np.isfinite(yw).all(axis=1)
    xc = xw - xw.mean(axis=1, keepdims=True)
    yc = yw - yw.mean(axis=1, keepdims=True)
    nx = np.sqrt(np.einsum("ij,ij->i", xc, xc))
    ny = np.sqrt(np.einsum("ij,ij->i", yc, yc))
    ok = finite & (nx > eps) & (ny > eps)

    rho = np.full(ok.shape, np.nan, dtype=np.float64)
    rho[ok] = np.einsum("ij,ij->i", xc[ok], yc[ok]) / (nx[ok] * ny[ok])
    np.clip(rho, -1.0, 1.0, out=rho)
    out[W:] = rho
    return out


def top_bottom_k(
    row: np.ndarray, k: int, exclude: int | np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the ``k`` largest and ``k`` smallest finite values of ``row``.

    NaN entries and ``exclude`` (e.g. the query stock itself) are ignored.
    Selection uses ``argpartition`` (O(N)); only the selected k are sorted.

    Returns
    -------
    (top, bottom)
        intp arrays of length ``min(k, n_candidates)``: ``top`` in descending
        order of value, ``bottom`` in ascending order. They may overlap when
        there are fewer than ``2k`` candidates.
    """
    candidates = np.flatnonzero(np.isfinite(row))
    if exclude is not None:
        candidates = candidates[~np.isin(candidates, np.atleast_1d(exclude))]
    k = min(int(k), candidates.size)
    if k <= 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy()
    values = row[candidates]
    top = np.argpartition(-values, k - 1)[:k]
    top = top[np.argsort(-values[top], kind="stable")]
    bottom = np.argpartition(values, k - 1)[:k]
    bottom = bottom[np.argsort(values[bottom], kind="stable")]
    return candidates[top], candidates[bottom]
