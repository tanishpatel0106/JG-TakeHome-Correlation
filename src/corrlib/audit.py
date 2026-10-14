"""Data-audit report: ``python -m corrlib.audit --zip PATH [--out reports/data_audit.md]``.

Runs the same ingest/clean pipeline as the build (see :func:`corrlib.build.ingest`)
and writes a Markdown report with real numbers: calendar, tickers, every
cleaning-issue counter, uniform-price days and how they were classified,
coverage, the return distribution, window validity, the cross-section compared
with a null benchmark, and the anomaly records (status, evidence, look-ahead).

Statistical wording is deliberately limited to what the numbers show: a match
with the null benchmark of independent series is reported as *consistent
with* that benchmark, never as proof of independence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .build import Panel, ingest, read_overrides
from .config import DEFAULT_ZIP, REPORTS_DIR, VALIDITY_EPS, VIEW, WINDOW
from .io import sha256_file
from .returns import OverrideError

log = logging.getLogger("corrlib.audit")

#: Calendar gaps longer than this many days are listed.
GAP_DAYS = 4

#: Tolerances for calling the per-date rho distribution "consistent with the
#: null benchmark": |median mean rho| and |median std / (1/sqrt(W-1)) - 1|.
NULL_MEAN_TOL = 0.01
NULL_STD_REL_TOL = 0.02

_ISSUE_DESCRIPTIONS = {
    "files": "daily files read",
    "rows_in": "rows read",
    "rows_out": "rows kept (one per ticker per file)",
    "ticker_empty": "rows dropped: empty ticker",
    "ticker_normalized": "tickers changed by strip/upper-case (kept)",
    "date_format_variant": "`Date` is the file date but not ISO (e.g. `M/D/YYYY`)",
    "date_mismatch": "`Date` parses to a different day than the file name (file date used)",
    "date_unparseable": "`Date` empty or unparseable (file date used)",
    "price_missing": "empty price -> NaN",
    "price_non_numeric": "non-numeric price -> NaN",
    "price_non_finite": "nan/inf price -> NaN",
    "price_non_positive": "price <= 0 -> NaN",
    "duplicate_identical": "tickers repeated in a file with one price (collapsed)",
    "duplicate_conflicting": "tickers repeated in a file with conflicting prices -> NaN",
    "duplicate_rows_dropped": "extra rows removed by de-duplication",
}


def build_report(zip_path: Path, panel: Panel, window: int = WINDOW) -> str:
    """Render the audit Markdown for an ingested panel."""
    dates = panel.dates
    tickers = panel.tickers
    n_dates, n_tickers = len(dates), len(tickers)
    lines: list[str] = []
    add = lines.append

    add("# Data audit")
    add("")
    add(f"- Source: `{zip_path.name}` ({zip_path.stat().st_size / 1e6:.1f} MB, sha256 `{sha256_file(zip_path)}`)")
    add(f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')} by corrlib {__version__} (`python -m corrlib.audit`)")
    add(f"- Window: W = {window} returns; validity threshold ||x - mean|| > {VALIDITY_EPS:g}")
    add(f"- View: **{VIEW}** (uniform-day decisions may use the next day's prices; see sections 4 and 9)")
    add("")

    # 1. Files and calendar ---------------------------------------------------
    add("## 1. Files and calendar")
    add("")
    add(f"- Daily files: **{n_dates}**, from **{dates[0]}** to **{dates[-1]}** (dates taken from file names)")
    skipped = panel.skipped_members
    if skipped:
        parts = []
        for k in range(len(skipped)):
            parts.append(f"`{skipped[k][0]}` ({skipped[k][1]})")
        add(f"- Skipped zip entries: {len(skipped)}: " + ", ".join(parts))
    else:
        add("- Skipped zip entries: none")
    weekend = []
    for t in range(n_dates):
        if dates[t].weekday() >= 5:
            weekend.append(dates[t].isoformat())
    add(f"- Weekend files: {len(weekend)}" + (f" ({', '.join(weekend)})" if weekend else ""))
    rows = np.asarray(panel.rows_per_file)
    add(f"- Rows per file: min {rows.min():,}, max {rows.max():,}; total {rows.sum():,}")
    per_year: dict[int, int] = {}
    for t in range(n_dates):
        per_year[dates[t].year] = per_year.get(dates[t].year, 0) + 1
    years = sorted(per_year)
    year_parts = []
    for k in range(len(years)):
        year_parts.append(f"{years[k]}: {per_year[years[k]]}")
    add(f"- Files per year: {', '.join(year_parts)}")
    add("")
    gaps = []
    for t in range(1, n_dates):
        days = (dates[t] - dates[t - 1]).days
        if days > GAP_DAYS:
            gaps.append((dates[t - 1], dates[t], days))
    add(f"Calendar gaps longer than {GAP_DAYS} days ({len(gaps)}):")
    add("")
    if gaps:
        add("| last date before | next file | calendar days |")
        add("|---|---|---:|")
        for k in range(len(gaps)):
            add(f"| {gaps[k][0]} | {gaps[k][1]} | {gaps[k][2]} |")
    else:
        add("none")
    add("")

    # 2. Tickers ----------------------------------------------------------------
    add("## 2. Tickers")
    add("")
    finite_prices = np.isfinite(panel.prices_raw)
    na_present = "NA" in tickers
    add(f"- Distinct tickers (union over files): **{n_tickers:,}**")
    if na_present:
        na_col = tickers.index("NA")
        add(f"- Ticker `NA` present: **yes**, with a valid price on {int(finite_prices[:, na_col].sum())} of {n_dates} dates (read as text, not as missing)")
    else:
        add("- Ticker `NA` present: **no**")
    per_file = np.asarray(panel.rows_per_file)
    always = int(np.isfinite(panel.prices_raw).all(axis=0).sum())
    add(f"- Tickers with a valid price on every date (before placeholder handling): {always:,} of {n_tickers:,}")
    add(f"- Tickers per file: min {per_file.min():,}, max {per_file.max():,}")
    add("")

    # 3. Cleaning issues ------------------------------------------------------
    add("## 3. Cleaning issues (per-file rules)")
    add("")
    add("| counter | rows | meaning |")
    add("|---|---:|---|")
    counts = panel.issues.as_dict()
    names = list(counts)
    for k in range(len(names)):
        add(f"| `{names[k]}` | {counts[names[k]]:,} | {_ISSUE_DESCRIPTIONS.get(names[k], '')} |")
    add("")
    if panel.issues.mismatch_examples:
        add("Date mismatch examples: " + "; ".join(panel.issues.mismatch_examples))
        add("")
    issue_files = panel.files_with_issues
    add(f"Files with at least one issue: {len(issue_files)} of {n_dates}")
    add("")
    if issue_files:
        add("| file date | issues |")
        add("|---|---|")
        for k in range(min(len(issue_files), 30)):
            found = issue_files[k][1]
            parts = []
            keys = list(found)
            for m in range(len(keys)):
                parts.append(f"`{keys[m]}` {found[keys[m]]:,}")
            add(f"| {issue_files[k][0]} | {', '.join(parts)} |")
        if len(issue_files) > 30:
            add(f"| ... | {len(issue_files) - 30} more |")
        add("")

    # 4. Uniform-price days -----------------------------------------------------
    add("## 4. Uniform-price days (every ticker has the same price)")
    add("")
    uniform = np.flatnonzero(panel.rebase | panel.placeholder | panel.unresolved)
    if uniform.size == 0:
        add("None.")
    else:
        add("A real cross-section never has identical prices. For each such day `t` the test compares the")
        add("cross-sectional median of `|log P[t+1]/P[t]|` (next day continues from the uniform value) with")
        add("`|log P[t+1]/P[t-1]|` (next day continues from the day before). For reference, the median")
        add(f"`|log return|` over all ordinary days is **{_typical_abs_log_return(panel):.4f}**.")
        add("")
        add("| date | index | price | days since previous file | median abs log, from uniform day | from previous day | verdict | action |")
        add("|---|---:|---:|---:|---:|---:|---|---|")
        for k in range(len(uniform)):
            t = int(uniform[k])
            value = panel.prices_raw[t][np.isfinite(panel.prices_raw[t])][0]
            gap = (dates[t] - dates[t - 1]).days if t > 0 else None
            from_u, from_p = _continuity(panel.prices_raw, t)
            if panel.rebase[t]:
                verdict = "re-base"
                action = "keep prices; R[t] := NaN" if t > 0 else "keep (R[0] is NaN anyway)"
            elif panel.placeholder[t]:
                verdict = "placeholder"
                action = "prices := NaN"
            else:
                verdict = "unresolved (last date)"
                action = "prices := NaN until the next day is known"
            add(
                f"| {dates[t]} | {t} | {value:g} | {gap if gap is not None else '-'} | "
                f"{_fmt(from_u)} | {_fmt(from_p)} | {verdict} | {action} |"
            )
        add("")
        add(f"- Returns set to NaN because they end on a re-base day: {panel.returns_nulled_at_rebase:,}")
        add(f"- Prices set to NaN on placeholder days: {panel.prices_nulled_on_placeholder_days:,}")
        if panel.unresolved.any():
            add(f"- Prices set to NaN on an unresolved last date: {panel.prices_nulled_pending:,}")
        add("- The raw values are kept: `prices.parquet` stores `price_raw` (per-file cleaning only) next to `price` (used).")
    add("")

    # 5. Coverage ---------------------------------------------------------------
    add("## 5. Price coverage per ticker (after cleaning and placeholder handling)")
    add("")
    missing = (~np.isfinite(panel.prices)).sum(axis=0)
    coverage = 1.0 - missing / n_dates
    add(f"- Coverage (share of dates with a valid price): min {coverage.min():.2%}, median {np.median(coverage):.2%}, max {coverage.max():.2%}")
    worst = np.argsort(-missing, kind="stable")[:10]
    add("")
    add("10 tickers with the most missing prices:")
    add("")
    add("| ticker | missing dates | coverage |")
    add("|---|---:|---:|")
    for k in range(len(worst)):
        j = int(worst[k])
        add(f"| `{tickers[j]}` | {int(missing[j])} | {coverage[j]:.2%} |")
    add("")

    # 6. Returns ----------------------------------------------------------------
    add("## 6. Returns")
    add("")
    r = panel.returns[np.isfinite(panel.returns)]
    naive = _naive_returns(panel.prices_raw)
    rn = naive[np.isfinite(naive)]
    qs = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
    add("Simple returns `R[t] = P[t]/P[t-1] - 1`. *Used* is after the uniform-day handling in section 4;")
    add("*naive* is the same formula on the per-file-cleaned prices with no uniform-day handling.")
    add("")
    add("| quantile | used | naive |")
    add("|---:|---:|---:|")
    q_used = np.quantile(r, qs)
    q_naive = np.quantile(rn, qs)
    for k in range(len(qs)):
        add(f"| {qs[k]:g} | {q_used[k]:.4f} | {q_naive[k]:.4f} |")
    add("")
    add("| | used | naive |")
    add("|---|---:|---:|")
    add(f"| finite returns | {r.size:,} | {rn.size:,} |")
    add(f"| abs(r) > 0.25 | {int((np.abs(r) > 0.25).sum()):,} | {int((np.abs(rn) > 0.25).sum()):,} |")
    add(f"| abs(r) > 0.5 | {int((np.abs(r) > 0.5).sum()):,} | {int((np.abs(rn) > 0.5).sum()):,} |")
    add(f"| std of returns | {r.std():.4f} | {rn.std():.4f} |")
    add("")
    add("Largest absolute returns used (after handling):")
    add("")
    add("| date | ticker | P[t-1] | P[t] | return |")
    add("|---|---|---:|---:|---:|")
    flat = np.where(np.isfinite(panel.returns), np.abs(panel.returns), -1.0).ravel()
    top = np.argsort(-flat, kind="stable")[:10]
    for k in range(len(top)):
        t, j = divmod(int(top[k]), n_tickers)
        add(
            f"| {dates[t]} | `{tickers[j]}` | {panel.prices[t - 1, j]:.4f} | "
            f"{panel.prices[t, j]:.4f} | {panel.returns[t, j]:+.4f} |"
        )
    add("")
    if n_dates > 2:
        first = panel.returns[1][np.isfinite(panel.returns[1])]
        rest = panel.returns[2:][np.isfinite(panel.returns[2:])]
        add(f"Seed-day check: returns from {dates[0]} to {dates[1]} (first return, based on the {dates[0]} prices)")
        add(f"vs all later returns. Quantiles 1% / 50% / 99%: first {_q3(first)}; later {_q3(rest)}.")
        add(f"Std: first {first.std():.4f}, later {rest.std():.4f}.")
        add("")

    # 7. Windows and validity ---------------------------------------------------
    add("## 7. Windows and validity")
    add("")
    returns32 = panel.returns.astype(np.float32)
    n_valid, n_constant, mean_rho, std_rho = _window_validity(returns32, window)
    usable = np.arange(n_dates) >= window
    first_valid = np.flatnonzero(n_valid >= 2)
    add(f"- First date with a full window (index W = {window}): **{dates[window] if n_dates > window else '-'}**")
    add(f"- First date with a valid pair: **{dates[int(first_valid[0])] if first_valid.size else '-'}**")
    total = int(usable.sum()) * n_tickers
    add(f"- Valid stock-dates (complete window and non-zero variance), over dates t >= W: **{int(n_valid[usable].sum()):,} of {total:,} ({n_valid[usable].sum() / max(total, 1):.2%})**")
    add(f"- Stock-dates with a complete window but zero variance (`constant_returns`, flat prices): {int(n_constant[usable].sum()):,}")
    nonempty = n_valid[usable & (n_valid > 0)]
    if nonempty.size:
        add(f"- n_valid on dates that have any valid stock: min {nonempty.min():,}, median {int(np.median(nonempty)):,}, max {nonempty.max():,}")
    empty_runs = _runs(np.flatnonzero(usable & (n_valid < 2)))
    add(f"- Dates at or after index W with fewer than 2 valid stocks: {int((usable & (n_valid < 2)).sum())}")
    if empty_runs:
        add("")
        add("| from | to | dates | cause |")
        add("|---|---|---:|---|")
        for k in range(len(empty_runs)):
            a, b = empty_runs[k]
            add(f"| {dates[a]} | {dates[b]} | {b - a + 1} | {_cause(panel, a, window)} |")
    add("")

    # 8. Cross-sectional dependence --------------------------------------------
    add("## 8. Cross-sectional dependence compared with a null benchmark")
    add("")
    has = np.isfinite(mean_rho)
    null_w = 1.0 / np.sqrt(window - 1)
    add(f"**Observation.** Per-date pairwise rho over the {int(has.sum())} dates with valid pairs (W = {window}):")
    add("")
    if has.any():
        add(f"- mean rho: min {np.nanmin(mean_rho):+.4f}, median {np.nanmedian(mean_rho):+.4f}, max {np.nanmax(mean_rho):+.4f}")
        add(f"- std of rho: min {np.nanmin(std_rho):.4f}, median {np.nanmedian(std_rho):.4f}, max {np.nanmax(std_rho):.4f}")
    add(
        f"- **Null benchmark** (an assumption, not a fact about the data): for independent normal series the sample "
        f"correlation of {window} returns has mean 0 and std ~ 1/sqrt(W - 1) = {null_w:.4f}"
    )
    full = _full_sample_dependence(returns32)
    if full is not None:
        n_rows, mean_all, std_all, max_abs = full
        add("")
        add(f"Full-sample correlation over the {n_rows} dates on which every ticker has a return:")
        add(f"mean rho {mean_all:+.5f}, std {std_all:.4f} (null benchmark 1/sqrt(n - 1) = {1 / np.sqrt(n_rows - 1):.4f}),")
        add(f"max abs rho off the diagonal {max_abs:.3f}.")
    add("")
    if has.any():
        median_mean = float(np.nanmedian(mean_rho))
        median_std = float(np.nanmedian(std_rho))
        consistent = abs(median_mean) <= NULL_MEAN_TOL and abs(median_std / null_w - 1.0) <= NULL_STD_REL_TOL
        add(
            f"**Inference.** Criterion: |median mean rho| <= {NULL_MEAN_TOL} and the median std within "
            f"{NULL_STD_REL_TOL:.0%} of {null_w:.4f}."
        )
        add("")
        if consistent:
            add(
                "The per-date distribution of pairwise correlations is **consistent with the null benchmark of "
                "independent series**: these summary statistics show no detectable common factor at this window length."
            )
        else:
            add(
                f"The per-date distribution **departs from the null benchmark of independent series** "
                f"(median mean rho {median_mean:+.4f}, median std {median_std:.4f} vs {null_w:.4f}): "
                "a positive mean and a wider spread are what common (market or sector) factors produce."
            )
        add("")
    n_peers = int(np.nanmax(n_valid)) - 1 if n_valid.size and np.nanmax(n_valid) > 1 else 0
    add("**Limitations.**")
    add("")
    add(
        "- A benchmark match is not a proof: it does **not** show that the stocks are independent, that no "
        "common factor exists, or that every individual correlation is noise. Weak or intermittent dependence, "
        "or dependence confined to a few pairs, can hide inside a distribution that matches the benchmark."
    )
    add(
        f"- A {window}-return correlation is a noisy estimate: around rho = 0 its 95% interval (Fisher z) is about "
        f"+/-{np.tanh(1.96 / np.sqrt(window - 3)):.2f}, so measured rho, its uncertainty and any economic "
        "usefulness are different things."
    )
    if n_peers > 0:
        add(
            f"- Multiple comparisons: among {n_peers:,} peers the largest |rho| of independent series is typically "
            f"about {_null_max_abs_rho(window, n_peers):.2f} (median of the maximum under the benchmark), so an "
            "extreme top/bottom neighbour is expected by chance and does not establish a stable relationship."
        )
    add("")

    # 9. Anomaly records ----------------------------------------------------------
    add("## 9. Anomaly records (status, evidence, look-ahead)")
    add("")
    records = panel.anomalies
    if not records:
        add("No uniform-price days, so no records.")
    else:
        add(
            "One record per uniform-price day, exactly as stored in `anomalies.json`. *Status* is `inferred` "
            "(decided by the rule), `confirmed` (an override file set it) or `unresolved` (cannot be decided yet). "
            "*Look-ahead* means the rule used prices from a later date, so the cleaned history is a "
            f"**{VIEW}** view, not a point-in-time one."
        )
        add("")
        add("| date | classification | status | rule | look-ahead | observation dates | from uniform / from previous (n) | action | affected window ends | override |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for k in range(len(records)):
            rec = records[k]
            ev = rec["evidence"]
            numbers = "-"
            if ev.get("median_abs_log_from_uniform") is not None:
                numbers = (
                    f"{ev['median_abs_log_from_uniform']:.4f} / {ev['median_abs_log_from_previous']:.4f} "
                    f"({ev['n_compared']:,})"
                )
            ends = rec["affected"]["window_end_dates"]
            span = f"{ends['first']} .. {ends['last']} ({ends['count']})" if ends["count"] else "none"
            override = "-"
            if rec.get("override"):
                ov = rec["override"]
                override = f"{ov['classification']} by {ov['author']}: {ov['reason']}"
            add(
                f"| {rec['date']} | {rec['classification']} | {rec['status']} | `{rec['rule']}` | "
                f"{'yes' if rec['look_ahead'] else 'no'} | {', '.join(rec['observation_dates_used'])} | {numbers} | "
                f"`{rec['action']}` | {span} | {override} |"
            )
        add("")
        add("Evidence notes:")
        add("")
        for k in range(len(records)):
            add(f"- {records[k]['date']}: {records[k]['evidence']['note']}")
        add("")
    return "\n".join(lines) + "\n"


def _full_sample_dependence(returns32: np.ndarray) -> tuple[int, float, float, float] | None:
    """Pearson correlation over all dates with complete rows: (n_rows, mean, std, max |rho|)."""
    from .corr import iter_full_blocks, mean_and_std_pairwise

    complete = np.isfinite(returns32).all(axis=1)
    if complete.sum() < 3:
        return None
    X = returns32[complete].astype(np.float64)
    centred = X - X.mean(axis=0)
    norm = np.sqrt((centred * centred).sum(axis=0))
    valid = norm > VALIDITY_EPS
    Z = np.full(X.shape, np.nan)
    Z[:, valid] = centred[:, valid] / norm[valid]
    mean, std, _ = mean_and_std_pairwise(Z, valid)
    max_abs = 0.0
    for r0, rows in iter_full_blocks(Z, valid):
        local = np.arange(rows.shape[0])
        rows[local, local + r0] = np.nan
        max_abs = max(max_abs, float(np.nanmax(np.abs(rows))))
    return int(complete.sum()), mean, std, max_abs


def _null_max_abs_rho(window: int, n_peers: int) -> float:
    """Median of the largest |rho| among ``n_peers`` independent null draws (W = ``window``).

    Under the benchmark (independent normal series, n = W points) the sample
    correlation has density proportional to ``(1 - r^2)^((n - 4) / 2)`` on
    [-1, 1]. The median of the maximum of ``m`` i.i.d. |r| is the
    ``0.5^(1/m)`` quantile of |r|, found on a fine grid (no scipy needed).
    """
    grid = np.linspace(0.0, 1.0, 200_001)
    density = np.power(np.clip(1.0 - grid * grid, 0.0, None), (window - 4) / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return float(np.interp(0.5 ** (1.0 / n_peers), cdf, grid))


def _typical_abs_log_return(panel: Panel) -> float:
    """Median |log return| over ordinary days (excluding returns into/out of uniform days)."""
    logp = np.log(panel.prices_raw)
    diffs = np.diff(logp, axis=0)
    special = panel.rebase | panel.placeholder | panel.unresolved
    keep = ~(special[1:] | special[:-1])
    values = np.abs(diffs[keep])
    return float(np.nanmedian(values))


def _continuity(prices: np.ndarray, t: int) -> tuple[float, float]:
    if t == 0 or t == prices.shape[0] - 1:
        return float("nan"), float("nan")
    both = np.isfinite(prices[t - 1]) & np.isfinite(prices[t]) & np.isfinite(prices[t + 1])
    if not both.any():
        return float("nan"), float("nan")
    from_u = float(np.median(np.abs(np.log(prices[t + 1, both] / prices[t, both]))))
    from_p = float(np.median(np.abs(np.log(prices[t + 1, both] / prices[t - 1, both]))))
    return from_u, from_p


def _naive_returns(prices: np.ndarray) -> np.ndarray:
    out = np.full(prices.shape, np.nan)
    out[1:] = prices[1:] / prices[:-1] - 1.0
    return out


def _window_validity(
    returns32: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per date: valid stocks, ``constant_returns`` stocks, mean and std of rho (via ``window_status``)."""
    from .config import REASON_CONSTANT_RETURNS
    from .corr import mean_and_std_pairwise, window_status

    n_dates = returns32.shape[0]
    n_valid = np.zeros(n_dates, dtype=np.int64)
    n_constant = np.zeros(n_dates, dtype=np.int64)
    mean_rho = np.full(n_dates, np.nan)
    std_rho = np.full(n_dates, np.nan)
    for t in range(window, n_dates):
        Z, valid, reason = window_status(returns32, t, window)
        mean_rho[t], std_rho[t], n_valid[t] = mean_and_std_pairwise(Z, valid)
        n_constant[t] = int((reason == REASON_CONSTANT_RETURNS).sum())
    return n_valid, n_constant, mean_rho, std_rho


def _runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted indices into (start, end) runs of consecutive values."""
    runs: list[tuple[int, int]] = []
    for k in range(len(indices)):
        i = int(indices[k])
        if runs and runs[-1][1] == i - 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def _cause(panel: Panel, start: int, window: int) -> str:
    """Explain an empty-window run by the nearest uniform day at or before its start."""
    special = np.flatnonzero((panel.rebase | panel.placeholder | panel.unresolved)[: start + 1])
    if special.size == 0:
        return "missing returns"
    t = int(special[-1])
    if start - t > window:
        return "missing returns"
    kind = "re-base" if panel.rebase[t] else ("placeholder day" if panel.placeholder[t] else "unresolved last date")
    return f"window spans the {kind} on {panel.dates[t]}"


def _q3(values: np.ndarray) -> str:
    if values.size == 0:
        return "-"
    q = np.quantile(values, [0.01, 0.5, 0.99])
    return f"{q[0]:+.4f} / {q[1]:+.4f} / {q[2]:+.4f}"


def _fmt(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corrlib.audit", description="Write the data-audit report.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="source zip (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "data_audit.md", help="report path (default: %(default)s)")
    parser.add_argument("--window", type=int, default=WINDOW, help="window length (default: %(default)s)")
    parser.add_argument("--overrides", type=Path, default=None, help="JSON overrides of uniform-day classifications")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not args.zip.is_file():
        log.error("source zip not found: %s (pass --zip PATH)", args.zip)
        return 2
    try:
        entries = read_overrides(args.overrides).entries if args.overrides is not None else None
        panel = ingest(args.zip, window=args.window, overrides=entries)
    except OverrideError as exc:
        log.error("invalid overrides: %s", exc)
        return 2
    report = build_report(args.zip, panel, args.window)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"Audit written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
