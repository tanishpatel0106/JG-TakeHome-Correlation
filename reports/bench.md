# Benchmarks

Generated 2026-10-14T22:31:50+00:00 by `python scripts/bench.py --data data/processed --out reports/bench.md` (what `make bench` runs). Every number below was measured in that run; nothing is carried over from earlier reports.

## Environment and data

- Machine: Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, 4 logical CPUs, Python 3.13.16
- Packages: corrlib 0.2.0, numpy 2.5.4, pandas 3.0.6, pyarrow 26.0.0, fastapi 0.143.1, starlette 1.8.0, uvicorn 0.54.0, httpx 0.28.1, corrlib (__version__) 0.2.0
- Code: git f90ecfc
- Store: `data/processed` -> snapshot `v0001-e724ede614dd` (Source data: stock_data.zip), 900 dates x 5,000 tickers, window 20
- Seed 0; 3 untimed warm-up calls before every timed series

## 1. JSON endpoints, in-process ASGI (warm page cache)

FastAPI `TestClient`: routing, validation, computation and JSON serialisation; no network. Random dates/tickers, 50 timed calls each.

| request | n | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|
| one stock vs all, top/bottom 20 | 50 | 4.9 | 5.8 | 6.9 |
| full row (5,000 values) | 50 | 4.9 | 6.4 | 12.8 |
| 100-ticker matrix | 50 | 6.2 | 7.8 | 8.8 |
| pair series (900 dates) | 50 | 2.6 | 3.6 | 4.5 |
| date summary + histogram | 50 | 1.5 | 1.9 | 3.1 |
| timeline | 50 | 2.0 | 2.7 | 3.0 |
| meta | 50 | 2.0 | 2.3 | 2.4 |

## 2. JSON endpoints over HTTP (warm)

uvicorn subprocess (1 worker) on 127.0.0.1, httpx client with keep-alive, 30 timed calls each.

| request | n | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|
| one stock vs all, top/bottom 20 | 30 | 5.1 | 7.1 | 9.3 |
| full row (5,000 values) | 30 | 4.6 | 6.2 | 7.9 |
| 100-ticker matrix | 30 | 5.8 | 6.4 | 7.7 |
| pair series (900 dates) | 30 | 2.8 | 3.4 | 4.3 |
| date summary + histogram | 30 | 1.5 | 1.8 | 2.2 |

## 3. Cold first requests

5 samples. Each: evict `returns.npy` from the page cache (`posix_fadvise(DONTNEED)`; resident share afterwards, by `mincore`: 0.0% at most), start a fresh uvicorn process, then time its first one-stock request and first pair request over HTTP.

| request | n | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|---:|
| first one-stock-vs-all request | 5 | 44.3 | 51.1 | 51.3 |
| first pair-series request | 5 | 59.4 | 73.5 | 75.8 |

## 4. Full-matrix downloads over HTTP

Date 2024-09-06, matrix 5,000 x 5,000. Each mode: 1 untimed warm-up, then 3 timed downloads; time = request sent to last byte received by the client. Server RSS sampled every 2 ms from /proc; peak is the highest sample across the timed runs. Bytes received were checked against `/download/info`.

| dtype | format | bytes | median s | max s | MB/s (median) | server RSS before -> peak |
|---|---|---:|---:|---:|---:|---|
| float32 | npz | 100,085,580 | 0.21 | 0.31 | 483 | 191 MB -> 218 MB (+27 MB) |
| float32 | npy | 100,000,128 | 0.19 | 0.23 | 523 | 191 MB -> 213 MB (+22 MB) |
| float16 | npz | 50,085,580 | 0.30 | 0.33 | 169 | 191 MB -> 213 MB (+22 MB) |
| float16 | npy | 50,000,128 | 0.20 | 0.26 | 253 | 213 MB -> 229 MB (+16 MB) |

## 5. Concurrent mixed workload over HTTP

8 client threads for 10.0 s against one uvicorn worker; each request is a random one-stock, row, 100-ticker matrix, pair or summary call.

- completed requests: 2,991 (299 req/s), errors: 0
- latency p50 26.9 ms, p95 38.5 ms, p99 43.9 ms

## 6. Build / update timings (recorded by corrlib in the snapshot manifest)

Snapshot `v0001-e724ede614dd` (kind full_build); seconds per stage as measured by `corrlib.build` / `corrlib.ingest` when it was built (not re-run by this benchmark).

| stage | seconds |
|---|---:|
| hash source zip | 0.16 |
| ingest + clean | 10.95 |
| wide prices + returns | 3.74 |
| per-date aggregates | 46.48 |

## Not measured

- Multi-worker or multi-host deployments, remote networks, or other machines.
- Downloads in parallel (the server allows 2 at a time; extra requests get HTTP 429).
- Cold-cache behaviour beyond the first request of a fresh process.
