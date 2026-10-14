# Data audit

- Source: `stock_data.zip` (60.3 MB, sha256 `ecab8196c97b44affa6c41985fd9fc50a571bcf2e74d19f45f3f6e2d7d09ffac`)
- Generated: 2026-10-14T18:12:38+00:00 by corrlib 0.1.0 (`python -m corrlib.audit`)
- Window: W = 20 returns; validity threshold ||x - mean|| > 1e-12

## 1. Files and calendar

- Daily files: **900**, from **2020-01-01** to **2024-09-06** (dates taken from file names)
- Skipped zip entries: 1: `stock_data/` (directory)
- Weekend files: 0
- Rows per file: min 5,000, max 5,000; total 4,500,000
- Files per year: 2020: 181, 2021: 180, 2022: 179, 2023: 180, 2024: 180

Calendar gaps longer than 4 days (4):

| last date before | next file | calendar days |
|---|---|---:|
| 2020-09-08 | 2020-12-31 | 114 |
| 2021-09-08 | 2021-12-31 | 114 |
| 2022-09-08 | 2023-01-02 | 116 |
| 2023-09-08 | 2024-01-01 | 115 |

## 2. Tickers

- Distinct tickers (union over files): **5,000**
- Ticker `NA` present: **yes**, with a valid price on 900 of 900 dates (read as text, not as missing)
- Tickers with a valid price on every date (before placeholder handling): 4,994 of 5,000
- Tickers per file: min 5,000, max 5,000

## 3. Cleaning issues (per-file rules)

| counter | rows | meaning |
|---|---:|---|
| `files` | 900 | daily files read |
| `rows_in` | 4,500,000 | rows read |
| `rows_out` | 4,500,000 | rows kept (one per ticker per file) |
| `ticker_empty` | 0 | rows dropped: empty ticker |
| `ticker_normalized` | 0 | tickers changed by strip/upper-case (kept) |
| `date_format_variant` | 10,000 | `Date` is the file date but not ISO (e.g. `M/D/YYYY`) |
| `date_mismatch` | 0 | `Date` parses to a different day than the file name (file date used) |
| `date_unparseable` | 0 | `Date` empty or unparseable (file date used) |
| `price_missing` | 6 | empty price -> NaN |
| `price_non_numeric` | 0 | non-numeric price -> NaN |
| `price_non_finite` | 0 | nan/inf price -> NaN |
| `price_non_positive` | 0 | price <= 0 -> NaN |
| `duplicate_identical` | 0 | tickers repeated in a file with one price (collapsed) |
| `duplicate_conflicting` | 0 | tickers repeated in a file with conflicting prices -> NaN |
| `duplicate_rows_dropped` | 0 | extra rows removed by de-duplication |

Files with at least one issue: 2 of 900

| file date | issues |
|---|---|
| 2021-02-22 | `date_format_variant` 5,000 |
| 2022-05-30 | `date_format_variant` 5,000, `price_missing` 6 |

## 4. Uniform-price days (every ticker has the same price)

A real cross-section never has identical prices. For each such day `t` the test compares the
cross-sectional median of `|log P[t+1]/P[t]|` (next day continues from the uniform value) with
`|log P[t+1]/P[t-1]|` (next day continues from the day before). For reference, the median
`|log return|` over all ordinary days is **0.0306**.

| date | index | price | days since previous file | median abs log, from uniform day | from previous day | verdict | action |
|---|---:|---:|---:|---:|---:|---|---|
| 2020-01-01 | 0 | 20 | - | - | - | re-base | keep (R[0] is NaN anyway) |
| 2020-12-31 | 180 | 20 | 114 | 0.0308 | 0.4409 | re-base | keep prices; R[t] := NaN |
| 2021-02-22 | 217 | 1 | 3 | 2.9578 | 0.0430 | placeholder | prices := NaN |
| 2021-12-31 | 360 | 20 | 114 | 0.0314 | 0.4245 | re-base | keep prices; R[t] := NaN |
| 2023-01-02 | 540 | 20 | 116 | 0.0307 | 0.4391 | re-base | keep prices; R[t] := NaN |
| 2024-01-01 | 720 | 20 | 115 | 0.0305 | 0.4397 | re-base | keep prices; R[t] := NaN |

- Returns set to NaN because they end on a re-base day: 20,000
- Prices set to NaN on placeholder days: 5,000

## 5. Price coverage per ticker (after cleaning and placeholder handling)

- Coverage (share of dates with a valid price): min 99.78%, median 99.89%, max 99.89%

10 tickers with the most missing prices:

| ticker | missing dates | coverage |
|---|---:|---:|
| `BKRP` | 2 | 99.78% |
| `DAF` | 2 | 99.78% |
| `PDY` | 2 | 99.78% |
| `TVT` | 2 | 99.78% |
| `UJK` | 2 | 99.78% |
| `WJX` | 2 | 99.78% |
| `A` | 1 | 99.89% |
| `AA` | 1 | 99.89% |
| `AAE` | 1 | 99.89% |
| `AAFB` | 1 | 99.89% |

## 6. Returns

Simple returns `R[t] = P[t]/P[t-1] - 1`. *Used* is after the uniform-day handling in section 4;
*naive* is the same formula on the per-file-cleaned prices with no uniform-day handling.

| quantile | used | naive |
|---:|---:|---:|
| 0 | -0.4249 | -0.9910 |
| 0.001 | -0.2401 | -0.9169 |
| 0.01 | -0.1648 | -0.1736 |
| 0.05 | -0.1017 | -0.1035 |
| 0.25 | -0.0304 | -0.0306 |
| 0.5 | 0.0002 | 0.0002 |
| 0.75 | 0.0308 | 0.0312 |
| 0.95 | 0.1022 | 0.1050 |
| 0.99 | 0.1650 | 0.1794 |
| 0.999 | 0.2409 | 12.5329 |
| 1 | 0.4428 | 747.5025 |

| | used | naive |
|---|---:|---:|
| finite returns | 4,464,988 | 4,494,988 |
| abs(r) > 0.25 | 6,515 | 29,829 |
| abs(r) > 0.5 | 0 | 19,101 |
| std of returns | 0.0609 | 0.8239 |

Largest absolute returns used (after handling):

| date | ticker | P[t-1] | P[t] | return |
|---|---|---:|---:|---:|
| 2022-04-06 | `FNO` | 15.1897 | 21.9156 | +0.4428 |
| 2022-06-15 | `XNA` | 27.2064 | 38.8918 | +0.4295 |
| 2023-05-03 | `QSJ` | 29.1846 | 41.6367 | +0.4267 |
| 2022-04-25 | `LTYD` | 28.7673 | 16.5434 | -0.4249 |
| 2024-08-19 | `CHYC` | 25.6466 | 14.8002 | -0.4229 |
| 2022-01-11 | `QLX` | 22.4174 | 31.6335 | +0.4111 |
| 2024-02-13 | `XYIS` | 12.1010 | 17.0149 | +0.4061 |
| 2020-02-17 | `WJS` | 12.4112 | 7.3769 | -0.4056 |
| 2022-08-30 | `EU` | 12.1665 | 17.0724 | +0.4032 |
| 2020-01-16 | `MCGQ` | 12.7645 | 7.6399 | -0.4015 |

Seed-day check: returns from 2020-01-01 to 2020-01-02 (first return, based on the 2020-01-01 prices)
vs all later returns. Quantiles 1% / 50% / 99%: first -0.1614 / -0.0005 / +0.1698; later -0.1648 / +0.0002 / +0.1650.
Std: first 0.0609, later 0.0609.

## 7. Windows and validity

- First date with a full window (index W = 20): **2020-01-29**
- First date with a valid pair: **2020-01-29**
- Valid stock-dates (complete window and non-zero variance), over dates t >= W: **3,894,874 of 4,400,000 (88.52%)**
- Stock-dates with a complete window but zero variance (flat prices): 0
- n_valid on dates that have any valid stock: min 4,994, median 5,000, max 5,000
- Dates at or after index W with fewer than 2 valid stocks: 101

| from | to | dates | cause |
|---|---|---:|---|
| 2020-12-31 | 2021-01-27 | 20 | window spans the re-base on 2020-12-31 |
| 2021-02-22 | 2021-03-22 | 21 | window spans the placeholder day on 2021-02-22 |
| 2021-12-31 | 2022-01-27 | 20 | window spans the re-base on 2021-12-31 |
| 2023-01-02 | 2023-01-27 | 20 | window spans the re-base on 2023-01-02 |
| 2024-01-01 | 2024-01-26 | 20 | window spans the re-base on 2024-01-01 |

## 8. Cross-sectional dependence

Per-date pairwise rho over the 779 dates with valid pairs (W = 20):

- mean rho: min -0.0001, median +0.0000, max +0.0002
- std of rho: min 0.2293, median 0.2294, max 0.2296
- for independent series the sample correlation of 20 points has mean 0 and std ~ 1/sqrt(W - 1) = 0.2294

Full-sample correlation over the 891 dates on which every ticker has a return:
mean rho +0.00001, std 0.0335 (independent-series value 1/sqrt(n - 1) = 0.0335),
max abs rho off the diagonal 0.177.

Both are what independent random walks produce: the cross-section has no common factor, so
trailing-window correlations in this data set are sampling noise around zero.

