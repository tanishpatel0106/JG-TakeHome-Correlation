# Benchmarks

- Generated: 2026-10-14T18:12:51+00:00 by `python scripts/bench.py` (corrlib 0.1.0)
- Data: `stock_data.zip` -> 900 dates x 5,000 tickers, W = 20
- Machine: Linux-6.18.44-fc-v114-x86_64-with-glibc2.39, 4 CPUs, Python 3.13.16, numpy 2.5.4
- Method: in-process, 50 random calls per case after 3 warm-up calls (seed 0); returns memmapped and in the OS page cache

## API endpoints (FastAPI TestClient: routing, validation and JSON included)

| case | request | p50 ms | p95 ms | max ms |
|---|---|---:|---:|---:|
| one-vs-all (top/bottom 20) | `GET /api/corr/{date}/stock/{ticker}?k=20` | 5.5 | 8.3 | 16.2 |
| full row (N values) | `GET /api/corr/{date}/row/{ticker}` | 5.8 | 9.6 | 12.4 |
| 100-ticker matrix | `GET /api/corr/{date}/matrix?tickers=<100>` | 8.6 | 11.3 | 13.3 |
| 150-ticker matrix (max) | `GET /api/corr/{date}/matrix?tickers=<150>` | 11.8 | 15.6 | 18.5 |
| pair series (all dates) | `GET /api/corr/pair?a=&b=` | 4.0 | 7.0 | 10.3 |
| date summary + histogram | `GET /api/dates/{date}/summary` | 1.8 | 4.7 | 7.8 |
| timeline | `GET /api/stats/timeline` | 2.2 | 3.6 | 5.1 |
| meta (5,000 tickers) | `GET /api/meta` | 1.8 | 2.8 | 10.8 |

## Full-matrix download (streamed float16 `.npy`)

One date (2024-09-06), N x N = 5,000 x 5,000.

| metric | value |
|---|---:|
| bytes | 50,000,128 (50.0 MB) |
| wall time, real uvicorn over localhost (httpx client) | 0.25 s (204 MB/s) |
| server RSS before -> peak while streaming (sampled every 2 ms) | 215 MB -> 215 MB (+0 MB) |
| body generator alone, in-process | 0.20 s |
| Python heap peak of the body generator (tracemalloc) | 13.2 MB |

Server memory stays O(block x N): one row block of 256 x N float32 (5 MB) plus its float16
copy, never the 100 MB float32 / 50 MB float16 matrix.

## Core kernels (no HTTP)

| kernel | p50 ms | p95 ms | max ms |
|---|---:|---:|---:|
| prepare_window + one_vs_all | 1.00 | 1.30 | 3.58 |
| prepare_window + subset_matrix (k=100) | 1.09 | 1.49 | 1.61 |
| pair_series (T dates) | 0.44 | 0.64 | 0.71 |
| prepare_window + mean/std aggregates | 1.60 | 1.92 | 2.14 |
| prepare_window + histogram (build, per date) | 59.71 | 86.25 | 108.79 |

## Build (from manifest.json of these artifacts)

| stage | seconds |
|---|---:|
| hash source zip | 0.2 |
| ingest + clean | 14.3 |
| wide prices + returns | 4.8 |
| per-date aggregates | 60.7 |
| total before writing artifacts | 81.2 |
