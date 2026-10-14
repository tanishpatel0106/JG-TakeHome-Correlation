// Typed client for the FastAPI backend. Types mirror app/api/schemas.py.

export interface Breaks {
  rebase_dates: string[];
  placeholder_dates: string[];
}

export interface Meta {
  window: number;
  tickers: string[];
  dates: string[];
  first_valid_date: string | null;
  last_date: string;
  default_date: string | null;
  default_ticker: string | null;
  breaks: Breaks;
  manifest: Record<string, unknown>;
}

export interface Timeline {
  dates: string[];
  mean_corr: (number | null)[];
  std_corr: (number | null)[];
  n_valid: number[];
}

export interface Summary {
  date: string;
  n_valid: number;
  n_pairs: number;
  mean: number | null;
  std: number | null;
  bin_edges: number[];
  counts: number[];
}

export interface Neighbor {
  ticker: string;
  corr: number;
}

export interface StockNeighbors {
  date: string;
  ticker: string;
  valid: boolean;
  n_valid: number;
  k: number;
  top: Neighbor[];
  bottom: Neighbor[];
}

export interface MatrixResult {
  date: string;
  tickers: string[];
  valid: boolean[];
  matrix: (number | null)[][];
}

export interface PairSeries {
  a: string;
  b: string;
  window: number;
  dates: string[];
  corr: (number | null)[];
}

/** An HTTP error with a human-readable message taken from the API's `detail`. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** Turn FastAPI's `detail` (string, {message}, or validation list) into one sentence. */
export function detailMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts: string[] = [];
    for (let i = 0; i < detail.length; i++) {
      const item = detail[i] as { loc?: unknown[]; msg?: string };
      const where = Array.isArray(item.loc) ? item.loc[item.loc.length - 1] : undefined;
      parts.push(where !== undefined ? `${String(where)}: ${item.msg ?? "invalid"}` : (item.msg ?? "invalid"));
    }
    return parts.join("; ") || fallback;
  }
  if (detail && typeof detail === "object") {
    const d = detail as { message?: unknown; suggestions?: unknown };
    let message = typeof d.message === "string" ? d.message : fallback;
    if (Array.isArray(d.suggestions) && d.suggestions.length > 0) {
      message += `. Did you mean ${d.suggestions.slice(0, 5).join(", ")}?`;
    }
    return message;
  }
  return fallback;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, { signal, headers: { Accept: "application/json" } });
  } catch (err) {
    if ((err as Error).name === "AbortError") throw err;
    throw new ApiError(0, "Cannot reach the API. Is it running on port 8000?", null);
  }
  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = ((await response.json()) as { detail?: unknown }).detail;
    } catch {
      // non-JSON error body
    }
    throw new ApiError(response.status, detailMessage(detail, `Request failed (${response.status})`), detail);
  }
  return (await response.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  meta: (signal?: AbortSignal) => getJson<Meta>("/api/meta", signal),
  timeline: (signal?: AbortSignal) => getJson<Timeline>("/api/stats/timeline", signal),
  summary: (date: string, signal?: AbortSignal) => getJson<Summary>(`/api/dates/${enc(date)}/summary`, signal),
  neighbors: (date: string, ticker: string, k: number, signal?: AbortSignal) =>
    getJson<StockNeighbors>(`/api/corr/${enc(date)}/stock/${enc(ticker)}?k=${k}`, signal),
  matrix: (date: string, tickers: string[], signal?: AbortSignal) =>
    getJson<MatrixResult>(`/api/corr/${enc(date)}/matrix?tickers=${enc(tickers.join(","))}`, signal),
  pair: (a: string, b: string, signal?: AbortSignal) =>
    getJson<PairSeries>(`/api/corr/pair?a=${enc(a)}&b=${enc(b)}`, signal),
  downloadUrl: (date: string) => `/api/corr/${enc(date)}/download`,
};
