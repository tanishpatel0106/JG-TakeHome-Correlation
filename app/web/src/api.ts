// Typed client for the FastAPI backend. Types mirror app/api/schemas.py.

export interface Breaks {
  rebase_dates: string[];
  placeholder_dates: string[];
}

export interface SnapshotInfo {
  id: string;
  seq?: number;
  parent?: string | null;
  created_at?: string;
  kind?: string;
  data_as_of?: string;
}

export interface DatasetInfo {
  kind?: string; // "demo" | "real" | ...
  label?: string;
}

/** One detected uniform-price day (corrlib anomalies.json); every field optional for older APIs. */
export interface Anomaly {
  date: string;
  classification?: string;
  status?: string;
  look_ahead?: boolean;
  action?: string;
  rule?: string;
  evidence?: { note?: string; median_abs_log_from_uniform?: number; median_abs_log_from_previous?: number };
  override?: { classification?: string; reason?: string; author?: string } | null;
  affected?: { window_end_dates?: { first?: string; last?: string; count?: number } };
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
  // v2 additions (optional: older APIs omit them)
  snapshot?: SnapshotInfo;
  dataset?: DatasetInfo;
  view?: string;
  policy?: Record<string, unknown>;
  anomalies?: Anomaly[];
}

export interface DownloadInfo {
  date: string;
  window: number;
  snapshot_id?: string;
  dtype: string;
  format: string;
  shape: number[];
  bytes: number;
  bytes_exact?: boolean;
  bytes_estimated?: boolean;
  filename?: string;
  ticker_order?: string;
  invalid_semantics?: string;
  tolerance?: number;
}

export type DownloadDtype = "float32" | "float16";
export type DownloadFormat = "npz" | "npy";

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
  /** Machine-readable `detail.code` (e.g. "date_not_traded"), or a client-side code. */
  readonly code: string;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    const code = detail && typeof detail === "object" && !Array.isArray(detail) ? (detail as { code?: unknown }).code : undefined;
    this.code = typeof code === "string" ? code : status === 0 ? "network_error" : `http_${status}`;
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
    throw new ApiError(0, "Cannot reach the API. Is the server running?", null);
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

/** Append query parameters, skipping null/undefined (so `snapshot` is only sent when known). */
export function withQuery(path: string, params: Record<string, string | number | null | undefined>): string {
  const parts: string[] = [];
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === "") continue;
    parts.push(`${key}=${enc(String(value))}`);
  }
  if (parts.length === 0) return path;
  return `${path}${path.includes("?") ? "&" : "?"}${parts.join("&")}`;
}

/** The snapshot a Meta response describes (v2 `meta.snapshot`, else the manifest's record). */
export function metaSnapshot(meta: Meta): SnapshotInfo | null {
  if (meta.snapshot?.id) return meta.snapshot;
  const fromManifest = meta.manifest?.snapshot as SnapshotInfo | undefined;
  return fromManifest?.id ? fromManifest : null;
}

export function metaDataset(meta: Meta): DatasetInfo | null {
  return meta.dataset ?? (meta.manifest?.dataset as DatasetInfo | undefined) ?? null;
}

type Snap = string | null | undefined;

export const api = {
  meta: (signal?: AbortSignal) => getJson<Meta>("/api/meta", signal),
  current: (signal?: AbortSignal) => getJson<SnapshotInfo>("/api/snapshots/current", signal),
  timeline: (snapshot: Snap, signal?: AbortSignal) => getJson<Timeline>(withQuery("/api/stats/timeline", { snapshot }), signal),
  summary: (date: string, snapshot: Snap, signal?: AbortSignal) =>
    getJson<Summary>(withQuery(`/api/dates/${enc(date)}/summary`, { snapshot }), signal),
  neighbors: (date: string, ticker: string, k: number, snapshot: Snap, signal?: AbortSignal) =>
    getJson<StockNeighbors>(withQuery(`/api/corr/${enc(date)}/stock/${enc(ticker)}`, { k, snapshot }), signal),
  matrix: (date: string, tickers: string[], snapshot: Snap, signal?: AbortSignal) =>
    getJson<MatrixResult>(withQuery(`/api/corr/${enc(date)}/matrix`, { tickers: tickers.join(","), snapshot }), signal),
  pair: (a: string, b: string, snapshot: Snap, signal?: AbortSignal) =>
    getJson<PairSeries>(withQuery("/api/corr/pair", { a, b, snapshot }), signal),
  downloadInfo: (date: string, dtype: DownloadDtype, format: DownloadFormat, snapshot: Snap, signal?: AbortSignal) =>
    getJson<DownloadInfo>(withQuery(`/api/corr/${enc(date)}/download/info`, { dtype, format, snapshot }), signal),
  downloadUrl: (date: string, dtype: DownloadDtype, format: DownloadFormat, snapshot: Snap) =>
    withQuery(`/api/corr/${enc(date)}/download`, { dtype, format, snapshot }),
};
