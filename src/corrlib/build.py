"""End-to-end build: zip -> cleaned panel -> returns -> per-date aggregates -> artifacts.

Usage::

    python -m corrlib.build --zip data/raw/stock_data.zip --out data/processed [--window 20]

Stages (each logged with its wall time):

1. hash the source zip (provenance)
2. ingest + clean every daily file (streamed from the zip)
3. wide price matrix, uniform-day detection, returns
4. per-date ``n_valid``, mean/std of pairwise rho and the rho histogram
5. write artifacts to a staging directory, checksum them, publish atomically
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, store
from .clean import IssueCounts, clean_frame
from .config import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_ZIP,
    HIST_BINS,
    HIST_BLOCK_ROWS,
    HIST_RANGE,
    VALIDITY_EPS,
    WINDOW,
)
from .corr import histogram_pairwise, mean_and_std_pairwise, prepare_window
from .io import iter_daily_frames, scan_members, sha256_file
from .returns import (
    classify_uniform_days,
    find_uniform_days,
    simple_returns,
    ticker_universe,
    to_wide,
)

log = logging.getLogger("corrlib.build")


@dataclass
class Panel:
    """Everything derived from the zip before any correlation work.

    Attributes
    ----------
    dates, tickers:
        The fixed axes (``T`` file dates, ``N`` sorted tickers).
    prices_raw:
        float64 ``[T x N]`` after per-file cleaning only.
    prices:
        ``prices_raw`` with placeholder days set to NaN. The prices actually used.
    returns:
        float64 ``[T x N]`` simple returns of ``prices`` with re-base breaks applied.
    rebase, placeholder:
        bool ``[T]`` uniform-price day classification.
    issues:
        Summed per-file :class:`IssueCounts`.
    rows_per_file:
        Raw row count of each daily file.
    skipped_members:
        ``(name, reason)`` of zip entries that are not daily files.
    """

    dates: list[date]
    tickers: list[str]
    prices_raw: np.ndarray
    prices: np.ndarray
    returns: np.ndarray
    rebase: np.ndarray
    placeholder: np.ndarray
    issues: IssueCounts
    rows_per_file: list[int] = field(default_factory=list)
    skipped_members: list[tuple[str, str]] = field(default_factory=list)
    #: ``(file date, {counter: n})`` for files with any non-zero issue counter
    files_with_issues: list[tuple[date, dict[str, int]]] = field(default_factory=list)
    #: finite returns set to NaN because they end on a re-base day
    returns_nulled_at_rebase: int = 0
    #: finite prices set to NaN because the day is a placeholder
    prices_nulled_on_placeholder_days: int = 0


@contextmanager
def stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    """Log the start and wall time of a build stage."""
    log.info("%s ...", name)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings[name] = elapsed
    log.info("%s done in %.2fs", name, elapsed)


def ingest(zip_path: str | Path, timings: dict[str, float] | None = None) -> Panel:
    """Read, clean and assemble the zip into prices and returns (no correlations)."""
    timings = {} if timings is None else timings

    with stage("ingest + clean", timings):
        scan = scan_members(zip_path)
        dates: list[date] = []
        frames: list[pd.DataFrame] = []
        rows_per_file: list[int] = []
        issues = IssueCounts()
        files_with_issues: list[tuple[date, dict[str, int]]] = []
        for file_date, raw in iter_daily_frames(zip_path, scan.members):
            cleaned, counts = clean_frame(file_date, raw)
            dates.append(file_date)
            frames.append(cleaned)
            rows_per_file.append(len(raw))
            issues = issues + counts
            found = _nonzero_issues(counts)
            if found:
                files_with_issues.append((file_date, found))
        if not dates:
            raise ValueError(f"{zip_path}: no YYYYMMDD.csv files found")
        log.info("%d files, %d rows in, %d rows out", issues.files, issues.rows_in, issues.rows_out)

    with stage("wide prices + returns", timings):
        tickers = ticker_universe(frames)
        prices_raw = to_wide(frames, tickers, dates)
        del frames
        uniform = find_uniform_days(prices_raw)
        rebase, placeholder = classify_uniform_days(prices_raw, uniform)
        prices = prices_raw.copy()
        prices[placeholder] = np.nan
        returns = simple_returns(prices, breaks=rebase)
        unbroken = simple_returns(prices)
        days = np.flatnonzero(uniform)
        for k in range(len(days)):
            t = int(days[k])
            kind = "re-base" if rebase[t] else "placeholder"
            log.info("uniform-price day %s classified as %s", dates[t], kind)
        log.info("T=%d dates, N=%d tickers", len(dates), len(tickers))

    return Panel(
        dates=dates,
        tickers=tickers,
        prices_raw=prices_raw,
        prices=prices,
        returns=returns,
        rebase=rebase,
        placeholder=placeholder,
        issues=issues,
        rows_per_file=rows_per_file,
        skipped_members=scan.skipped,
        files_with_issues=files_with_issues,
        returns_nulled_at_rebase=int(np.isfinite(unbroken[rebase]).sum()),
        prices_nulled_on_placeholder_days=int(np.isfinite(prices_raw[placeholder]).sum()),
    )


def _nonzero_issues(counts: IssueCounts) -> dict[str, int]:
    """Issue counters of one file that are non-zero (bookkeeping counters excluded)."""
    found: dict[str, int] = {}
    values = counts.as_dict()
    names = list(values)
    for k in range(len(names)):
        if names[k] not in ("files", "rows_in", "rows_out") and values[names[k]]:
            found[names[k]] = values[names[k]]
    return found


def daily_aggregates(
    returns: np.ndarray, window: int, edges: np.ndarray, block: int = HIST_BLOCK_ROWS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-date ``n_valid``, mean rho, std rho and rho histogram.

    Parameters
    ----------
    returns:
        ``[T x N]`` returns exactly as stored (float32), so these aggregates
        match what the API computes on demand.

    Returns
    -------
    (n_valid int32 [T], mean float64 [T], std float64 [T], hist int64 [T x B])
        mean/std are NaN and the histogram row is zero where ``n_valid < 2``.
    """
    n_dates = returns.shape[0]
    n_valid = np.zeros(n_dates, dtype=np.int32)
    mean = np.full(n_dates, np.nan)
    std = np.full(n_dates, np.nan)
    hist = np.zeros((n_dates, edges.size - 1), dtype=np.int64)
    for t in range(window, n_dates):
        Z, valid = prepare_window(returns, t, window)
        mean[t], std[t], n_valid[t] = mean_and_std_pairwise(Z, valid)
        if n_valid[t] >= 2:
            hist[t] = histogram_pairwise(Z, valid, edges, block=block)
        if t % 100 == 0:
            log.info("  aggregates: date %d/%d (n_valid=%d)", t, n_dates, n_valid[t])
    return n_valid, mean, std, hist


def run_build(
    zip_path: str | Path,
    out_dir: str | Path,
    window: int = WINDOW,
    block: int = HIST_BLOCK_ROWS,
) -> tuple[dict, dict[str, float]]:
    """Run the full build and publish artifacts to ``out_dir``.

    Returns ``(manifest, timings)``; ``timings`` maps stage name -> seconds.
    """
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    with stage("hash source zip", timings):
        zip_sha = sha256_file(zip_path)

    panel = ingest(zip_path, timings)

    with stage("per-date aggregates", timings):
        returns32 = panel.returns.astype(np.float32)
        edges = np.linspace(HIST_RANGE[0], HIST_RANGE[1], HIST_BINS + 1)
        n_valid, mean, std, hist = daily_aggregates(returns32, window, edges, block)

    first_valid = np.flatnonzero(n_valid >= 2)
    first_valid_date = panel.dates[int(first_valid[0])].isoformat() if first_valid.size else None

    with stage("write artifacts", timings):
        with store.staging_dir(out_dir) as tmp:
            store.write_returns(tmp, returns32)
            store.write_axes(tmp, panel.tickers, panel.dates)
            store.write_prices(tmp, panel.prices, panel.tickers, panel.dates)
            store.write_daily_stats(tmp, panel.dates, n_valid, mean, std)
            store.write_hist(tmp, hist, edges)
            manifest = store.write_manifest(
                tmp,
                {
                    "corrlib_version": __version__,
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "source": {
                        "file_name": zip_path.name,
                        "sha256": zip_sha,
                        "size_bytes": zip_path.stat().st_size,
                    },
                    "window": window,
                    "validity_eps": VALIDITY_EPS,
                    "n_dates": len(panel.dates),
                    "n_tickers": len(panel.tickers),
                    "first_date": panel.dates[0].isoformat(),
                    "last_date": panel.dates[-1].isoformat(),
                    "first_valid_date": first_valid_date,
                    "hist": {"bins": HIST_BINS, "range": list(HIST_RANGE)},
                    "issue_counts": panel.issues.as_dict(),
                    "structural": {
                        "rebase_dates": _iso_where(panel.dates, panel.rebase),
                        "placeholder_dates": _iso_where(panel.dates, panel.placeholder),
                        "returns_nulled_at_rebase": panel.returns_nulled_at_rebase,
                        "prices_nulled_on_placeholder_days": panel.prices_nulled_on_placeholder_days,
                    },
                    # Stages completed before the artifacts were written.
                    "timings_seconds": _rounded(timings),
                    "build_seconds_before_write": round(time.perf_counter() - started, 2),
                },
            )
    timings["total"] = time.perf_counter() - started
    return manifest, timings


def _iso_where(dates: list[date], mask: np.ndarray) -> list[str]:
    out = []
    idx = np.flatnonzero(mask)
    for k in range(len(idx)):
        out.append(dates[int(idx[k])].isoformat())
    return out


def _rounded(timings: dict[str, float]) -> dict[str, float]:
    out = {}
    names = list(timings)
    for k in range(len(names)):
        out[names[k]] = round(timings[names[k]], 2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m corrlib.build", description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="source zip (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_PROCESSED_DIR, help="artifact directory (default: %(default)s)")
    parser.add_argument("--window", type=int, default=WINDOW, help="window length in returns (default: %(default)s)")
    parser.add_argument("--block", type=int, default=HIST_BLOCK_ROWS, help="rows per histogram block (default: %(default)s)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.zip.is_file():
        log.error("source zip not found: %s (pass --zip PATH)", args.zip)
        return 2
    if args.window < 2:
        log.error("--window must be >= 2")
        return 2

    manifest, timings = run_build(args.zip, args.out, window=args.window, block=args.block)

    structural = manifest["structural"]
    print()
    print(f"Build complete -> {args.out}")
    print(f"  source          {manifest['source']['file_name']} (sha256 {manifest['source']['sha256'][:12]}...)")
    print(f"  dates x tickers {manifest['n_dates']} x {manifest['n_tickers']} (window {manifest['window']})")
    print(f"  date range      {manifest['first_date']} .. {manifest['last_date']}")
    print(f"  first valid     {manifest['first_valid_date']}")
    print(f"  re-base days    {', '.join(structural['rebase_dates']) or '-'}")
    print(f"  placeholder     {', '.join(structural['placeholder_dates']) or '-'}")
    names = list(manifest["artifacts"])
    for k in range(len(names)):
        size = manifest["artifacts"][names[k]]["size_bytes"]
        print(f"  {names[k]:<20} {size / 1e6:8.2f} MB")
    stages = list(timings)
    for k in range(len(stages)):
        print(f"  [{timings[stages[k]]:7.2f}s] {stages[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
