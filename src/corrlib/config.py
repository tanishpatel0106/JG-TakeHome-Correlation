"""Project-wide constants.

Everything that defines the *contract* of the derived data (window length,
validity threshold, histogram layout) lives here so the build, the API and the
tests all agree on it.
"""

from __future__ import annotations

from pathlib import Path

# --- Correlation contract -------------------------------------------------

#: Number of returns in a trailing correlation window (spans WINDOW + 1 prices).
WINDOW: int = 20

#: A stock is valid in a window only if its centred window norm exceeds this.
#: Guards against zero / numerically-zero variance (flat prices).
VALIDITY_EPS: float = 1e-12

#: Per-date histogram of pairwise correlations: uniform bins on HIST_RANGE.
HIST_BINS: int = 40
HIST_RANGE: tuple[float, float] = (-1.0, 1.0)

#: Rows per block when streaming the full N x N matrix (download).
BLOCK_ROWS: int = 1024

#: Rows per block for the per-date histogram. Small, so each [b x N] float32
#: block stays in CPU cache while it is binned (~2x faster than 1024).
HIST_BLOCK_ROWS: int = 128

# --- Input format ---------------------------------------------------------

#: Zip members we treat as daily files: optional directories, then YYYYMMDD.csv.
MEMBER_PATTERN: str = r"^(?:.*/)?(\d{8})\.csv$"

#: Required CSV header (after trimming whitespace and a UTF-8 BOM).
EXPECTED_COLUMNS: tuple[str, str, str] = ("Ticker", "Date", "Price")

# --- Paths (relative to the working directory; every CLI can override) ----

DEFAULT_ZIP: Path = Path("data/raw/stock_data.zip")
DEFAULT_PROCESSED_DIR: Path = Path("data/processed")
REPORTS_DIR: Path = Path("reports")

#: Environment variable the API reads to locate the processed artifacts.
DATA_DIR_ENV: str = "CORR_DATA_DIR"

# --- Artifact file names ---------------------------------------------------

RETURNS_FILE = "returns.npy"
TICKERS_FILE = "tickers.json"
DATES_FILE = "dates.json"
PRICES_FILE = "prices.parquet"
DAILY_STATS_FILE = "daily_stats.parquet"
HIST_FILE = "hist.npy"
HIST_EDGES_FILE = "hist_edges.npy"
MANIFEST_FILE = "manifest.json"

#: Version of the on-disk layout; bump when artifacts change incompatibly.
ARTIFACT_FORMAT_VERSION: int = 1

# --- API limits ------------------------------------------------------------

DEFAULT_K: int = 20
MAX_K: int = 200
MATRIX_MIN_TICKERS: int = 2
MATRIX_MAX_TICKERS: int = 150
MAX_SUGGESTIONS: int = 10
CORS_ORIGINS: tuple[str, ...] = ("http://localhost:5173",)
