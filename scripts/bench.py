"""Latency and memory benchmark: ``python scripts/bench.py [--data data/processed] [--out reports/bench.md]``.

Everything runs in-process. Endpoints go through FastAPI's TestClient (routing,
validation and JSON serialisation included, no network); the core kernels
are timed directly as well. Each case runs ``--calls`` random requests
(fixed seed) after a few warm-up calls and reports p50 / p95 / max in ms.
The full-matrix download is streamed once and timed, with the Python heap
peak (tracemalloc) and the process RSS sampled while it streams.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import threading
import time
import tracemalloc
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `app` importable when run as a script

import httpx  # noqa: E402

warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import create_app  # noqa: E402
from app.api.routes import download  # noqa: E402
from corrlib import __version__  # noqa: E402
from corrlib.corr import (  # noqa: E402
    histogram_pairwise,
    mean_and_std_pairwise,
    one_vs_all,
    pair_series,
    prepare_window,
    subset_matrix,
)
from corrlib.store import Store, load_store  # noqa: E402

WARMUP = 3


def percentiles(samples_ms: list[float]) -> tuple[float, float, float]:
    values = np.asarray(samples_ms)
    return float(np.percentile(values, 50)), float(np.percentile(values, 95)), float(values.max())


def time_calls(make_call: Callable[[], Callable[[], object]], calls: int) -> list[float]:
    """Run ``calls`` timed calls; ``make_call`` picks random inputs outside the timer."""
    for _ in range(WARMUP):
        make_call()()
    samples = []
    for _ in range(calls):
        call = make_call()
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def rss_of(pid: int) -> int | None:
    """Resident set size of a process in bytes (Linux /proc), or None elsewhere."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def bench_download_http(data_dir: Path, date: str) -> dict[str, float | None]:
    """Stream one download from a real uvicorn process over localhost, sampling the server's RSS."""
    port = free_port()
    env = {**os.environ, "CORR_DATA_DIR": str(data_dir.resolve())}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 30
        while True:
            try:
                if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.1)
        url = f"{base}/api/corr/{date}/download"
        with httpx.stream("GET", url, timeout=60) as response:  # warm-up
            for _ in response.iter_bytes():
                pass
        before = rss_of(server.pid)
        peak = [before or 0]
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                current = rss_of(server.pid)
                if current is not None and current > peak[0]:
                    peak[0] = current
                time.sleep(0.002)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        start = time.perf_counter()
        n_bytes = 0
        with httpx.stream("GET", url, timeout=60) as response:
            for chunk in response.iter_bytes():
                n_bytes += len(chunk)
        seconds = time.perf_counter() - start
        stop.set()
        sampler.join()
        return {
            "bytes": n_bytes,
            "seconds": seconds,
            "rss_before": before,
            "rss_peak": peak[0] if before is not None else None,
        }
    finally:
        server.terminate()
        server.wait(timeout=10)


def bench_download_generator(store: Store, date: str) -> tuple[float, int]:
    """Consume the endpoint's body iterator in-process: (seconds, Python heap peak bytes)."""
    response = download(date, store)

    async def consume() -> int:
        total = 0
        async for chunk in response.body_iterator:
            total += len(chunk)
        return total

    tracemalloc.start()
    start = time.perf_counter()
    asyncio.run(consume())
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return seconds, peak


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("reports/bench.md"))
    parser.add_argument("--calls", type=int, default=50, help="timed calls per case (>= 30)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if args.calls < 30:
        parser.error("--calls must be >= 30")

    rng = random.Random(args.seed)
    store = load_store(args.data)
    usable = [store.dates[t] for t in np.flatnonzero(store.n_valid >= 2)]
    usable_idx = [int(t) for t in np.flatnonzero(store.n_valid >= 2)]
    tickers = store.tickers
    rows: list[tuple[str, str, list[float]]] = []

    with TestClient(create_app(args.data)) as client:

        def get(path: str, params: dict | None = None) -> Callable[[], object]:
            def call() -> object:
                response = client.get(path, params=params)
                if response.status_code != 200:
                    raise RuntimeError(f"{path}: {response.status_code} {response.text[:200]}")
                return response.content

            return call

        http_cases: list[tuple[str, str, Callable[[], Callable[[], object]]]] = [
            ("one-vs-all (top/bottom 20)", "GET /api/corr/{date}/stock/{ticker}?k=20",
             lambda: get(f"/api/corr/{rng.choice(usable)}/stock/{rng.choice(tickers)}", {"k": 20})),
            ("full row (N values)", "GET /api/corr/{date}/row/{ticker}",
             lambda: get(f"/api/corr/{rng.choice(usable)}/row/{rng.choice(tickers)}")),
            ("100-ticker matrix", "GET /api/corr/{date}/matrix?tickers=<100>",
             lambda: get(f"/api/corr/{rng.choice(usable)}/matrix", {"tickers": ",".join(rng.sample(tickers, 100))})),
            ("150-ticker matrix (max)", "GET /api/corr/{date}/matrix?tickers=<150>",
             lambda: get(f"/api/corr/{rng.choice(usable)}/matrix", {"tickers": ",".join(rng.sample(tickers, 150))})),
            ("pair series (all dates)", "GET /api/corr/pair?a=&b=",
             lambda: get("/api/corr/pair", {"a": rng.choice(tickers), "b": rng.choice(tickers)})),
            ("date summary + histogram", "GET /api/dates/{date}/summary",
             lambda: get(f"/api/dates/{rng.choice(usable)}/summary")),
            ("timeline", "GET /api/stats/timeline", lambda: get("/api/stats/timeline")),
            ("meta (5,000 tickers)", "GET /api/meta", lambda: get("/api/meta")),
        ]
        for k in range(len(http_cases)):
            name, path, make = http_cases[k]
            rows.append((name, path, time_calls(make, args.calls)))
            print(f"  {name}: done", file=sys.stderr)


    download_date = usable[-1]
    http_dl = bench_download_http(args.data, download_date)
    gen_seconds, gen_heap = bench_download_generator(store, download_date)
    print("  download: done", file=sys.stderr)

    # Core kernels without HTTP.
    W = store.window
    R = store.returns

    def kernel(fn: Callable[[int, list[int]], object]) -> Callable[[], Callable[[], object]]:
        def make() -> Callable[[], object]:
            t = rng.choice(usable_idx)
            cols = rng.sample(range(store.n_tickers), 100)
            return lambda: fn(t, cols)

        return make

    def one_vs_all_kernel(t: int, cols: list[int]) -> object:
        Z, valid = prepare_window(R, t, W)
        return one_vs_all(Z, valid, cols[0])

    def subset_kernel(t: int, cols: list[int]) -> object:
        Z, valid = prepare_window(R, t, W)
        return subset_matrix(Z, valid, np.asarray(cols))

    def pair_kernel(t: int, cols: list[int]) -> object:
        return pair_series(R, cols[0], cols[1], W)

    def aggregates_kernel(t: int, cols: list[int]) -> object:
        Z, valid = prepare_window(R, t, W)
        return mean_and_std_pairwise(Z, valid)

    edges = store.hist_edges

    def histogram_kernel(t: int, cols: list[int]) -> object:
        Z, valid = prepare_window(R, t, W)
        return histogram_pairwise(Z, valid, edges)

    kernel_cases = [
        ("prepare_window + one_vs_all", one_vs_all_kernel),
        ("prepare_window + subset_matrix (k=100)", subset_kernel),
        ("pair_series (T dates)", pair_kernel),
        ("prepare_window + mean/std aggregates", aggregates_kernel),
        ("prepare_window + histogram (build, per date)", histogram_kernel),
    ]
    kernel_rows = []
    for k in range(len(kernel_cases)):
        name, fn = kernel_cases[k]
        kernel_rows.append((name, time_calls(kernel(fn), args.calls)))

    manifest = store.manifest
    lines = [
        "# Benchmarks",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `python scripts/bench.py` (corrlib {__version__})",
        f"- Data: `{manifest['source']['file_name']}` -> {store.n_dates} dates x {store.n_tickers:,} tickers, W = {W}",
        f"- Machine: {platform.platform()}, {os.cpu_count()} CPUs, Python {platform.python_version()}, numpy {np.__version__}",
        f"- Method: in-process, {args.calls} random calls per case after {WARMUP} warm-up calls (seed {args.seed}); "
        "returns memmapped and in the OS page cache",
        "",
        "## API endpoints (FastAPI TestClient: routing, validation and JSON included)",
        "",
        "| case | request | p50 ms | p95 ms | max ms |",
        "|---|---|---:|---:|---:|",
    ]
    for k in range(len(rows)):
        name, path, samples = rows[k]
        p50, p95, worst = percentiles(samples)
        lines.append(f"| {name} | `{path}` | {p50:.1f} | {p95:.1f} | {worst:.1f} |")
    lines += [
        "",
        "## Full-matrix download (streamed float16 `.npy`)",
        "",
        f"One date ({download_date}), N x N = {store.n_tickers:,} x {store.n_tickers:,}.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| bytes | {http_dl['bytes']:,} ({http_dl['bytes'] / 1e6:.1f} MB) |",
        f"| wall time, real uvicorn over localhost (httpx client) | {http_dl['seconds']:.2f} s ({http_dl['bytes'] / 1e6 / http_dl['seconds']:.0f} MB/s) |",
    ]
    if http_dl["rss_before"] is not None:
        grow = (http_dl["rss_peak"] - http_dl["rss_before"]) / 1e6
        lines.append(
            f"| server RSS before -> peak while streaming (sampled every 2 ms) | "
            f"{http_dl['rss_before'] / 1e6:.0f} MB -> {http_dl['rss_peak'] / 1e6:.0f} MB (+{grow:.0f} MB) |"
        )
    lines += [
        f"| body generator alone, in-process | {gen_seconds:.2f} s |",
        f"| Python heap peak of the body generator (tracemalloc) | {gen_heap / 1e6:.1f} MB |",
        "",
        "Server memory stays O(block x N): one row block of 256 x N float32 (5 MB) plus its float16",
        "copy, never the 100 MB float32 / 50 MB float16 matrix.",
        "",
        "## Core kernels (no HTTP)",
        "",
        "| kernel | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|",
    ]
    for k in range(len(kernel_rows)):
        name, samples = kernel_rows[k]
        p50, p95, worst = percentiles(samples)
        lines.append(f"| {name} | {p50:.2f} | {p95:.2f} | {worst:.2f} |")
    timings = manifest.get("timings_seconds", {})
    if timings:
        lines += [
            "",
            "## Build (from manifest.json of these artifacts)",
            "",
            "| stage | seconds |",
            "|---|---:|",
        ]
        names = list(timings)
        for k in range(len(names)):
            lines.append(f"| {names[k]} | {timings[names[k]]:.1f} |")
        lines.append(f"| total before writing artifacts | {manifest.get('build_seconds_before_write', float('nan')):.1f} |")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
