"""Response models and JSON serialization helpers.

Every float that can be undefined is typed ``float | None``. NaN and inf are
converted to ``None`` explicitly by :func:`floats_or_none` before a model is
built, and the response class (orjson) and the model config would both emit
``null`` for any NaN that slipped through.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import orjson
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


class ORJSONResponse(JSONResponse):
    """JSON response rendered with orjson (NaN/inf -> null, numpy supported).

    FastAPI's own ``ORJSONResponse`` is deprecated in recent releases; this is
    the same few lines kept local so the default response class stays orjson.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)


def finite_or_none(value: float) -> float | None:
    """``float(value)`` if finite, else ``None``."""
    value = float(value)
    return value if np.isfinite(value) else None


def floats_or_none(values: np.ndarray) -> list[float | None]:
    """1-D array -> list of Python floats with ``None`` for NaN/inf."""
    array = np.asarray(values, dtype=np.float64)
    out = array.tolist()
    bad = np.flatnonzero(~np.isfinite(array))
    for k in range(len(bad)):
        out[bad[k]] = None
    return out


def matrix_or_none(matrix: np.ndarray) -> list[list[float | None]]:
    """2-D array -> nested lists with ``None`` for NaN/inf."""
    rows = []
    for r in range(matrix.shape[0]):
        rows.append(floats_or_none(matrix[r]))
    return rows


class _Model(BaseModel):
    model_config = ConfigDict(ser_json_inf_nan="null")


class HealthResponse(_Model):
    status: str
    snapshot_id: str
    n_tickers: int
    n_dates: int


class Breaks(_Model):
    rebase_dates: list[str] = Field(description="Uniform-price re-base days: the return into each is NaN")
    placeholder_dates: list[str] = Field(description="Uniform-price placeholder days: prices are NaN")


class MetaResponse(_Model):
    window: int
    tickers: list[str] = Field(description="Column order of every vector/matrix/download")
    dates: list[str]
    first_valid_date: str | None
    last_date: str
    default_date: str | None = Field(description="Last date with at least one valid pair")
    default_ticker: str | None = Field(description="First valid ticker (alphabetically) on default_date")
    breaks: Breaks
    manifest: dict[str, Any]
    snapshot: dict[str, Any] = Field(description="id, seq, parent, created_at, kind of the snapshot served")
    view: str = Field(description="retrospective: every date is computed with all data of this snapshot")
    dataset: dict[str, Any] = Field(description="kind (source|demo) and label")
    policy: dict[str, Any] = Field(description="Validity / missing-data policy (reason codes etc.)")
    anomalies: list[dict[str, Any]] = Field(description="Uniform-price days with evidence and classification")


class TimelineResponse(_Model):
    dates: list[str]
    mean_corr: list[float | None]
    std_corr: list[float | None]
    n_valid: list[int]


class SummaryResponse(_Model):
    date: str
    status: str = Field(description="ok, or no_valid_pairs when fewer than 2 stocks are valid")
    n_valid: int
    n_pairs: int
    mean: float | None
    std: float | None
    bin_edges: list[float]
    counts: list[int]


class Neighbor(_Model):
    ticker: str
    corr: float


class StockResponse(_Model):
    date: str
    ticker: str
    valid: bool = Field(description="False when the stock has missing or constant returns in the window")
    reason: str = Field(description="valid | insufficient_history | missing_returns | constant_returns")
    n_valid: int
    k: int
    top: list[Neighbor]
    bottom: list[Neighbor]


class MatrixResponse(_Model):
    date: str
    tickers: list[str]
    valid: list[bool]
    reasons: list[str] = Field(description="Validity reason per requested ticker")
    matrix: list[list[float | None]]


class PairResponse(_Model):
    a: str
    b: str
    window: int
    dates: list[str]
    corr: list[float | None]


class RowResponse(_Model):
    date: str
    ticker: str
    valid: bool
    reason: str
    n_valid: int
    corr: list[float | None] = Field(description="Aligned with meta.tickers")


class ErrorResponse(_Model):
    detail: dict[str, Any] = Field(description="`code` (machine-readable), `message`, plus context fields")
