"""Project-wide constants.

Everything that defines the *contract* of the derived data (window length,
validity threshold, reason codes, histogram layout, the data policy) lives here
so the build, the API and the tests all agree on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# --- Correlation contract -------------------------------------------------

#: Number of returns in a trailing correlation window (spans WINDOW + 1 prices).
WINDOW: int = 20

#: A stock is valid in a window only if its centred window norm exceeds this.
#: Guards against zero / numerically-zero variance (flat prices).
VALIDITY_EPS: float = 1e-12

# --- Validity reason codes (one int8 per stock, see corr.window_status) ----

REASON_VALID: int = 0
REASON_INSUFFICIENT_HISTORY: int = 1
REASON_MISSING_RETURNS: int = 2
REASON_CONSTANT_RETURNS: int = 3

#: API string of each reason code, indexed by the code.
REASON_NAMES: tuple[str, ...] = (
    "valid",
    "insufficient_history",
    "missing_returns",
    "constant_returns",
)

#: Human-readable meaning of each reason code, indexed by the code.
REASON_DESCRIPTIONS: tuple[str, ...] = (
    "complete window of W finite returns with non-zero variance",
    "the date index is below W: the window would need R[0] (undefined) or earlier; applies to every stock",
    (
        "at least one of the W returns in the window is missing (missing or invalid price, not yet listed, "
        "delisted, or a structural break such as a re-base)"
    ),
    "complete window but zero variance: Pearson correlation is undefined, including the self-correlation",
)

# --- Uniform-price days (see returns.anomaly_records) ----------------------

#: Identifier of the detection + continuity rule recorded in every anomaly record.
ANOMALY_RULE: str = "uniform-continuity/v1"

#: The product serves retrospectively cleaned history (decisions may use later days).
VIEW: str = "retrospective"

#: Per-date histogram of pairwise correlations: uniform bins on HIST_RANGE.
HIST_BINS: int = 40
HIST_RANGE: tuple[float, float] = (-1.0, 1.0)

#: Rows per block when streaming the full N x N matrix (download).
BLOCK_ROWS: int = 1024

#: Rows per block when streaming the float16 download (~2.5 MB per chunk at N = 5000).
DOWNLOAD_BLOCK_ROWS: int = 256

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
ANOMALIES_FILE = "anomalies.json"
ISSUES_BY_DATE_FILE = "issues_by_date.parquet"
INPUTS_FILE = "inputs.json"
OVERRIDES_FILE = "overrides.json"
MANIFEST_FILE = "manifest.json"

#: Version of the on-disk layout written by this code; bump when artifacts change incompatibly.
ARTIFACT_FORMAT_VERSION: int = 2

#: The original flat layout (no anomalies / inputs / issues_by_date). Still loadable.
LEGACY_FORMAT_VERSION: int = 1

# --- Versioned store root (a directory holding several snapshots) -----------

#: Pointer file naming the published snapshot: ``{"snapshot": "<id>", "published_at": ...}``.
CURRENT_FILE = "CURRENT"

#: Sub-directory of a store root holding one directory per snapshot id.
SNAPSHOTS_DIR = "snapshots"

#: Prefix of an in-progress build directory inside ``snapshots/`` (ignored by readers).
STAGING_PREFIX = ".staging-"

#: Sub-directory of a store root holding a content-addressed copy of every ingested daily file.
OVERLAYS_DIR = "overlays"

#: Append-only audit log of every write attempt on a store root (one JSON object per line).
INGEST_LOG_FILE = "ingest_log.jsonl"

#: Single-writer lock file of a store root (created with O_CREAT | O_EXCL).
LOCK_FILE = ".lock"

#: Snapshot ids: ``v{seq:04d}-{first 12 hex of sha256 over the sorted artifact checksums}``.
SNAPSHOT_ID_PATTERN: str = r"^v(\d{4,})-([0-9a-f]{12})$"

# --- Datasets ----------------------------------------------------------------

#: Zip comment that marks the synthetic demo dataset (``python -m corrlib.demo``).
DEMO_ZIP_COMMENT: str = "corrlib synthetic demo dataset"

#: ``manifest.dataset.label`` of a demo build.
DEMO_LABEL: str = "Synthetic demo data"

# --- API limits ------------------------------------------------------------

DEFAULT_K: int = 20
MAX_K: int = 200
MATRIX_MIN_TICKERS: int = 2
MATRIX_MAX_TICKERS: int = 150
MAX_SUGGESTIONS: int = 10
CORS_ORIGINS: tuple[str, ...] = ("http://localhost:5173",)


# --- Data policy (single source of truth, exposed as manifest.policy / meta.policy) ---


def build_policy(window: int = WINDOW, eps: float = VALIDITY_EPS) -> dict[str, Any]:
    """The missing-data, validity and cleaning policy as a JSON-ready dict.

    Every view (stock, row, matrix, pair series, summary, histogram, download)
    follows this one rule; :func:`corrlib.corr.window_status` implements it.
    """
    reason_codes = []
    for code in range(len(REASON_NAMES)):
        reason_codes.append(
            {"code": code, "name": REASON_NAMES[code], "description": REASON_DESCRIPTIONS[code]}
        )
    return {
        "window": window,
        "min_valid_returns": window,
        "eps": eps,
        "estimator": (
            "Pearson correlation of simple returns over the trailing window, complete "
            "observations only (no pairwise deletion)"
        ),
        "return_definition": (
            "R[t] = P[t] / P[t-1] - 1 between consecutive file dates; R[0] is undefined; "
            "NaN if either price is missing or invalid; never forward-filled"
        ),
        "window_definition": (
            f"the window as of date index t is R[t-{window}+1 .. t] ({window} returns spanning "
            f"{window + 1} prices); the first possible window ends at t = {window}"
        ),
        "validity_rule": (
            f"a stock is valid at t iff t >= {window}, all {window} returns in the window are "
            f"finite and the centred window norm exceeds {eps:g}"
        ),
        "invalid_semantics": (
            "invalid => NaN in arrays and null in JSON, never 0; invalid stocks are never "
            "ranked; the self-correlation of an invalid stock is NaN"
        ),
        "reason_codes": reason_codes,
        "missing_price_effect": (
            f"a missing price at index p makes R[p] and R[p+1] NaN, so the stock is invalid "
            f"in the {window + 1} windows ending at p .. p+{window}"
        ),
        "uniform_price_days": {
            "rule": ANOMALY_RULE,
            "detection": "every finite price of the day is identical (at least half the tickers priced)",
            "rebase": (
                "next day continues from the uniform value: prices kept as a new base, "
                "the return into the day is set to NaN (structural break)"
            ),
            "placeholder": (
                "next day continues from the previous day: the day's prices are set to NaN "
                "(raw values kept in prices.parquet price_raw)"
            ),
            "unresolved": (
                "uniform last date: cannot be tested without the next day, so its prices are "
                "excluded (NaN) until a later snapshot adds the next day and re-classifies it"
            ),
            "look_ahead": (
                "an interior classification uses the next day's prices, so the history is a "
                "retrospectively cleaned view, not a point-in-time one"
            ),
            "overrides": (
                "an overrides file may confirm or change the classification of a detected "
                "uniform day (status 'confirmed'); it is copied into the artifacts"
            ),
        },
        "universe": "growing union; a ticker absent on a date is missing; symbol renames are not linked",
        "calendar": (
            "the sorted set of file dates (the file name is authoritative); dates without a "
            "file do not exist; a return spans consecutive file dates, including calendar gaps"
        ),
        "view": VIEW,
    }


#: The policy for the default window (see :func:`build_policy`).
POLICY: dict[str, Any] = build_policy()
