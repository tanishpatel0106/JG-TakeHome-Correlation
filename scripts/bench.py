"""Benchmark the v2 API against a published store root.

    python scripts/bench.py --data data/processed --out reports/bench.md

What it measures (every figure in the report comes from this run; nothing is copied):

1. In-process ASGI latency (FastAPI TestClient: routing, validation, JSON) for the
   JSON endpoints, warm page cache.
2. The same endpoints over real HTTP (a uvicorn subprocess on 127.0.0.1), warm.
3. Cold first requests: a fresh uvicorn process after ``returns.npy`` was evicted
   from the OS page cache (``posix_fadvise(DONTNEED)``, residency verified with
   ``mincore``; Linux only, reported as not measured elsewhere).
4. Full-matrix downloads over HTTP for every supported (dtype, format): wall time
   from request to last byte, bytes, server RSS before / sampled peak.
5. A concurrent mixed workload over HTTP (several client threads, fixed duration).
6. Build/update stage timings as recorded in the snapshot manifest by corrlib.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import mmap
import os
import platform
import random
import socket
import subprocess
import sys
import threading
import time
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
from corrlib import __version__  # noqa: E402
from corrlib.store import load_store, resolve_snapshot_dir  # noqa: E402

WARMUP = 3
DOWNLOAD_MODES = [("float32", "npz"), ("float32", "npy"), ("float16", "npz"), ("float16", "npy")]


# --- helpers ---------------------------------------------------------------------


def stats(samples_ms: list[float]) -> dict[str, float]:
    a = np.asarray(samples_ms)
    return {"n": len(a), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def versions() -> dict[str, str]:
    out = {}
    for name in ("corrlib", "numpy", "pandas", "pyarrow", "fastapi", "starlette", "uvicorn", "httpx"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    out["corrlib (__version__)"] = __version__
    return out


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True)
        return sha.stdout.strip() + (" (uncommitted changes)" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rss_of(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def evict(path: Path) -> bool:
    if not hasattr(os, "posix_fadvise"):
        return False
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return True


def resident_fraction(path: Path) -> float | None:
    """Share of the file's pages in the page cache (mincore); None if unsupported."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        size = path.stat().st_size
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), size, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        buf = ctypes.c_char.from_buffer(mm)
        try:
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size), vec) != 0:
                return None
            return sum(v & 1 for v in vec) / pages
        finally:
            del buf
            mm.close()
    except (OSError, AttributeError, ValueError):
        return None


class Server:
    """uvicorn subprocess on a free localhost port (one worker, no reload)."""

    def __init__(self, data: Path) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        env = {**os.environ, "CORR_DATA_DIR": str(data.resolve()), "CORR_RELOAD_SECONDS": "0"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=ROOT,
            env=env,
        )
        self.base = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 60
        while True:
            try:
                if httpx.get(f"{self.base}/api/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None or time.time() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.05)

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=15)


# --- workloads -------------------------------------------------------------------


def make_requests(store, rng: random.Random) -> dict[str, Callable[[], tuple[str, dict]]]:
    dates = [store.dates[t] for t in np.flatnonzero(store.n_valid >= 2)]
    tickers = store.tickers
    k = min(100, len(tickers))
    return {
        "one stock vs all, top/bottom 20": lambda: (f"/api/corr/{rng.choice(dates)}/stock/{rng.choice(tickers)}", {"k": 20}),
        f"full row ({len(tickers):,} values)": lambda: (f"/api/corr/{rng.choice(dates)}/row/{rng.choice(tickers)}", {}),
        f"{k}-ticker matrix": lambda: (f"/api/corr/{rng.choice(dates)}/matrix", {"tickers": ",".join(rng.sample(tickers, k))}),
        f"pair series ({len(store.dates)} dates)": lambda: ("/api/corr/pair", {"a": rng.choice(tickers), "b": rng.choice(tickers)}),
        "date summary + histogram": lambda: (f"/api/dates/{rng.choice(dates)}/summary", {}),
        "timeline": lambda: ("/api/stats/timeline", {}),
        "meta": lambda: ("/api/meta", {}),
    }


def time_get(get: Callable[[str, dict], object], make: Callable[[], tuple[str, dict]], calls: int) -> list[float]:
    for _ in range(WARMUP):
        get(*make())
    out = []
    for _ in range(calls):
        path, params = make()
        start = time.perf_counter()
        get(path, params)
        out.append((time.perf_counter() - start) * 1000)
    return out


def checked(response: httpx.Response) -> httpx.Response:
    if response.status_code != 200:
        raise RuntimeError(f"{response.request.url}: {response.status_code} {response.text[:200]}")
    return response


def sample_rss(pid: int, stop: threading.Event, peak: list[int]) -> None:
    """Record the highest RSS of ``pid`` into ``peak[0]`` every 2 ms until ``stop`` is set."""
    while not stop.is_set():
        cur = rss_of(pid)
        if cur is not None and cur > peak[0]:
            peak[0] = cur
        time.sleep(0.002)


def bench_downloads(server: Server, date: str, repeats: int) -> list[dict]:
    rows = []
    for dtype, fmt in DOWNLOAD_MODES:
        params = {"dtype": dtype, "format": fmt}
        info = checked(httpx.get(f"{server.base}/api/corr/{date}/download/info", params=params, timeout=30)).json()
        times, peaks, before_all, sizes = [], [], [], []
        for r in range(repeats + 1):  # first run is a warm-up, not reported
            before = rss_of(server.proc.pid)
            peak = [before or 0]
            stop = threading.Event()
            sampler = threading.Thread(target=sample_rss, args=(server.proc.pid, stop, peak), daemon=True)
            sampler.start()
            start = time.perf_counter()
            n_bytes = 0
            with httpx.stream("GET", f"{server.base}/api/corr/{date}/download", params=params, timeout=120) as resp:
                checked(resp)
                for chunk in resp.iter_raw():
                    n_bytes += len(chunk)
            elapsed = time.perf_counter() - start
            stop.set()
            sampler.join()
            if r == 0:
                continue
            times.append(elapsed)
            sizes.append(n_bytes)
            if before is not None:
                before_all.append(before)
                peaks.append(peak[0])
        if any(s != info["bytes"] for s in sizes):
            raise RuntimeError(f"{dtype}/{fmt}: received {sizes} bytes, /download/info says {info['bytes']}")
        rows.append(
            {
                "dtype": dtype,
                "format": fmt,
                "bytes": info["bytes"],
                "size_exact": info.get("bytes_exact", True),
                "times": times,
                "rss_before": max(before_all) if before_all else None,
                "rss_peak": max(peaks) if peaks else None,
            }
        )
    return rows


def bench_cold(data: Path, returns_path: Path, store, samples: int, rng: random.Random) -> dict | None:
    """First requests in a fresh server after evicting returns.npy. None if eviction is unsupported."""
    if not hasattr(os, "posix_fadvise"):
        return None
    dates = [store.dates[t] for t in np.flatnonzero(store.n_valid >= 2)]
    stock_ms, pair_ms, residency = [], [], []
    for _ in range(samples):
        if not evict(returns_path):
            return None
        residency.append(resident_fraction(returns_path))
        server = Server(data)
        try:
            start = time.perf_counter()
            checked(httpx.get(f"{server.base}/api/corr/{rng.choice(dates)}/stock/{rng.choice(store.tickers)}", timeout=30))
            stock_ms.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            checked(httpx.get(f"{server.base}/api/corr/pair", params={"a": rng.choice(store.tickers), "b": rng.choice(store.tickers)}, timeout=30))
            pair_ms.append((time.perf_counter() - start) * 1000)
        finally:
            server.close()
    measured = [r for r in residency if r is not None]
    return {"stock": stock_ms, "pair": pair_ms, "residency_after_evict": max(measured) if measured else None}


def bench_concurrent(server: Server, store, threads: int, seconds: float, seed: int) -> dict:
    latencies: list[float] = []
    errors = [0]
    lock = threading.Lock()
    stop_at = time.perf_counter() + seconds

    def worker(i: int) -> None:
        rng = random.Random(seed + i)
        reqs = list(make_requests(store, rng).values())[:5]  # stock, row, matrix, pair, summary
        with httpx.Client(base_url=server.base, timeout=30) as client:
            while time.perf_counter() < stop_at:
                path, params = rng.choice(reqs)()
                start = time.perf_counter()
                r = client.get(path, params=params)
                ms = (time.perf_counter() - start) * 1000
                with lock:
                    if r.status_code == 200:
                        latencies.append(ms)
                    else:
                        errors[0] += 1

    started = time.perf_counter()
    pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    elapsed = time.perf_counter() - started
    a = np.asarray(latencies)
    return {
        "threads": threads,
        "seconds": elapsed,
        "requests": len(a),
        "errors": errors[0],
        "rps": len(a) / elapsed,
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


# --- main ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/processed"), help="store root or snapshot dir")
    parser.add_argument("--out", type=Path, default=Path("reports/bench.md"))
    parser.add_argument("--calls", type=int, default=50, help="timed calls per endpoint (>= 30)")
    parser.add_argument("--http-calls", type=int, default=30, help="timed calls per endpoint over HTTP")
    parser.add_argument("--download-repeats", type=int, default=3, help="timed downloads per (dtype, format)")
    parser.add_argument("--cold-samples", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--concurrent-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.calls < 30:
        parser.error("--calls must be >= 30")
    try:
        snapshot_dir = resolve_snapshot_dir(args.data)[0]
    except Exception as exc:  # noqa: BLE001 - one clear message for any missing/invalid store
        print(
            f"No published store at {args.data} ({exc}).\n"
            "Build one first: `make build-data` (real data at data/raw/stock_data.zip) or run `make bench-demo`.",
            file=sys.stderr,
        )
        return 2
    returns_path = Path(snapshot_dir) / "returns.npy"
    rng = random.Random(args.seed)
    started = datetime.now(timezone.utc)

    # The probe memmaps returns.npy but reads no data pages before the cold runs;
    # the residency check after each eviction verifies the cache really is cold.
    probe = load_store(args.data)
    manifest = probe.manifest
    dataset = probe.dataset or {}
    n_dates, n_tickers, snapshot_id = probe.n_dates, probe.n_tickers, probe.snapshot_id
    download_date = probe.dates[int(np.flatnonzero(probe.n_valid >= 2)[-1])]
    store_for_requests = probe

    print("cold first requests ...", file=sys.stderr)
    cold = bench_cold(args.data, returns_path, store_for_requests, args.cold_samples, rng)

    print("in-process ASGI ...", file=sys.stderr)
    inproc = {}
    with TestClient(create_app(args.data, reload_seconds=0)) as client:
        def get(path: str, params: dict) -> object:
            r = client.get(path, params=params)
            if r.status_code != 200:
                raise RuntimeError(f"{path}: {r.status_code} {r.text[:200]}")
            return r.content

        reqs = make_requests(store_for_requests, rng)
        for name in reqs:
            inproc[name] = stats(time_get(get, reqs[name], args.calls))

    print("HTTP warm, downloads, concurrency ...", file=sys.stderr)
    server = Server(args.data)
    try:
        with httpx.Client(base_url=server.base, timeout=30) as client:
            def hget(path: str, params: dict) -> object:
                return checked(client.get(path, params=params)).content

            reqs = make_requests(store_for_requests, rng)
            http = {}
            for name in list(reqs)[:5]:
                http[name] = stats(time_get(hget, reqs[name], args.http_calls))
        downloads = bench_downloads(server, download_date, args.download_repeats)
        concurrent = bench_concurrent(server, store_for_requests, args.threads, args.concurrent_seconds, args.seed)
    finally:
        server.close()

    # --- report ---
    v = versions()
    L: list[str] = []
    add = L.append
    add("# Benchmarks")
    add("")
    add(f"Generated {started.isoformat(timespec='seconds')} by `python scripts/bench.py --data {args.data} --out {args.out}` "
        "(what `make bench` runs). Every number below was measured in that run; nothing is carried over from earlier reports.")
    add("")
    add("## Environment and data")
    add("")
    add(f"- Machine: {platform.platform()}, {os.cpu_count()} logical CPUs, Python {platform.python_version()}")
    add("- Packages: " + ", ".join(f"{k} {v[k]}" for k in v))
    add(f"- Code: git {git_commit()}")
    add(f"- Store: `{args.data}` -> snapshot `{snapshot_id}` ({dataset.get('label', 'unknown dataset')}), "
        f"{n_dates} dates x {n_tickers:,} tickers, window {manifest.get('window')}")
    add(f"- Seed {args.seed}; {WARMUP} untimed warm-up calls before every timed series")
    add("")
    add("## 1. JSON endpoints, in-process ASGI (warm page cache)")
    add("")
    add("FastAPI `TestClient`: routing, validation, computation and JSON serialisation; no network. "
        f"Random dates/tickers, {args.calls} timed calls each.")
    add("")
    add("| request | n | p50 ms | p95 ms | max ms |")
    add("|---|---:|---:|---:|---:|")
    for name in inproc:
        s = inproc[name]
        add(f"| {name} | {s['n']} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |")
    add("")
    add("## 2. JSON endpoints over HTTP (warm)")
    add("")
    add(f"uvicorn subprocess (1 worker) on 127.0.0.1, httpx client with keep-alive, {args.http_calls} timed calls each.")
    add("")
    add("| request | n | p50 ms | p95 ms | max ms |")
    add("|---|---:|---:|---:|---:|")
    for name in http:
        s = http[name]
        add(f"| {name} | {s['n']} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |")
    add("")
    add("## 3. Cold first requests")
    add("")
    if cold is None:
        add("Not measured: `posix_fadvise` is unavailable on this platform.")
    else:
        res = cold["residency_after_evict"]
        add(f"{args.cold_samples} samples. Each: evict `returns.npy` from the page cache "
            f"(`posix_fadvise(DONTNEED)`; resident share afterwards, by `mincore`: "
            f"{'not verified' if res is None else f'{res:.1%} at most'}), start a fresh uvicorn process, "
            "then time its first one-stock request and first pair request over HTTP.")
        add("")
        add("| request | n | p50 ms | p95 ms | max ms |")
        add("|---|---:|---:|---:|---:|")
        for name, key in (("first one-stock-vs-all request", "stock"), ("first pair-series request", "pair")):
            s = stats(cold[key])
            add(f"| {name} | {s['n']} | {s['p50']:.1f} | {s['p95']:.1f} | {s['max']:.1f} |")
    add("")
    add("## 4. Full-matrix downloads over HTTP")
    add("")
    add(f"Date {download_date}, matrix {n_tickers:,} x {n_tickers:,}. Each mode: 1 untimed warm-up, then "
        f"{args.download_repeats} timed downloads; time = request sent to last byte received by the client. "
        "Server RSS sampled every 2 ms from /proc; peak is the highest sample across the timed runs. "
        "Bytes received were checked against `/download/info`.")
    add("")
    add("| dtype | format | bytes | median s | max s | MB/s (median) | server RSS before -> peak |")
    add("|---|---|---:|---:|---:|---:|---|")
    for row in downloads:
        med = float(np.median(row["times"]))
        rss = "not measured"
        if row["rss_before"] is not None:
            rss = f"{row['rss_before'] / 1e6:.0f} MB -> {row['rss_peak'] / 1e6:.0f} MB (+{(row['rss_peak'] - row['rss_before']) / 1e6:.0f} MB)"
        exact = "" if row["size_exact"] else " (estimated)"
        add(f"| {row['dtype']} | {row['format']} | {row['bytes']:,}{exact} | {med:.2f} | {max(row['times']):.2f} | "
            f"{row['bytes'] / 1e6 / med:.0f} | {rss} |")
    add("")
    add("## 5. Concurrent mixed workload over HTTP")
    add("")
    add(f"{concurrent['threads']} client threads for {concurrent['seconds']:.1f} s against one uvicorn worker; each request "
        "is a random one-stock, row, 100-ticker matrix, pair or summary call.")
    add("")
    add(f"- completed requests: {concurrent['requests']:,} ({concurrent['rps']:.0f} req/s), errors: {concurrent['errors']}")
    add(f"- latency p50 {concurrent['p50']:.1f} ms, p95 {concurrent['p95']:.1f} ms, p99 {concurrent['p99']:.1f} ms")
    add("")
    add("## 6. Build / update timings (recorded by corrlib in the snapshot manifest)")
    add("")
    timings = manifest.get("timings_seconds") or {}
    if timings:
        add(f"Snapshot `{snapshot_id}` (kind {manifest.get('snapshot', {}).get('kind', 'n/a')}); seconds per stage as measured by "
            "`corrlib.build` / `corrlib.ingest` when it was built (not re-run by this benchmark).")
        add("")
        add("| stage | seconds |")
        add("|---|---:|")
        for name in timings:
            add(f"| {name} | {timings[name]:.2f} |")
    else:
        add("No stage timings recorded in this snapshot's manifest.")
    add("")
    add("## Not measured")
    add("")
    add("- Multi-worker or multi-host deployments, remote networks, or other machines.")
    add("- Downloads in parallel (the server allows 2 at a time; extra requests get HTTP 429).")
    add("- Cold-cache behaviour beyond the first request of a fresh process.")
    add("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nWritten to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
