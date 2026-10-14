// Small pure helpers shared by components.

const DAY_MS = 86_400_000;

function toTime(iso: string): number {
  return Date.parse(`${iso}T00:00:00Z`);
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2024-09-06" -> "Fri 6 Sep 2024" (calendar date, no time-zone shift). */
export function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return `${WEEKDAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** "2024-09-06" -> "6 Sep 2024". */
export function formatShortDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00Z`);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** Escape text for ECharts tooltip HTML (tickers come from the source data). */
export function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

export function formatCorr(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "n/a";
  if (Math.abs(value) < 0.0005) return "0.000";
  const s = value.toFixed(3);
  return value > 0 ? `+${s}` : s.replace("-", "−");
}

export function formatInt(value: number): string {
  return value.toLocaleString("en-US");
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e3).toFixed(0)} kB`;
}

/**
 * Nearest selectable date to `target` (an ISO date, not necessarily traded).
 * `selectable` must be sorted. Ties go to the earlier date. Returns null if empty.
 */
export function snapDate(target: string, selectable: string[]): string | null {
  if (selectable.length === 0) return null;
  let lo = 0;
  let hi = selectable.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (selectable[mid]! < target) lo = mid + 1;
    else hi = mid;
  }
  if (lo < selectable.length && selectable[lo] === target) return target;
  const after = lo < selectable.length ? selectable[lo]! : null;
  const before = lo > 0 ? selectable[lo - 1]! : null;
  if (before === null) return after;
  if (after === null) return before;
  const t = toTime(target);
  return t - toTime(before) <= toTime(after) - t ? before : after;
}

export function daysBetween(a: string, b: string): number {
  return Math.round((toTime(b) - toTime(a)) / DAY_MS);
}

/** log Beta function via Lanczos log-gamma (enough precision for a reference curve). */
function logGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  x -= 1;
  let a = c[0]!;
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i]! / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Expected histogram counts of Pearson r for independent normal series of
 * length n: density (1 - r^2)^((n - 4) / 2) / B(1/2, (n - 2) / 2), integrated
 * over each bin with Simpson's rule, times the number of pairs.
 */
export function nullCorrCounts(edges: number[], n: number, pairs: number): number[] {
  const logB = logGamma(0.5) + logGamma((n - 2) / 2) - logGamma((n - 1) / 2);
  const density = (r: number) => (Math.abs(r) >= 1 ? 0 : Math.exp(((n - 4) / 2) * Math.log(1 - r * r) - logB));
  const out: number[] = [];
  for (let i = 0; i + 1 < edges.length; i++) {
    const a = edges[i]!;
    const b = edges[i + 1]!;
    const mass = ((b - a) / 6) * (density(a) + 4 * density((a + b) / 2) + density(b));
    out.push(mass * pairs);
  }
  return out;
}

// --- sampling uncertainty ---------------------------------------------------------

/**
 * 95% confidence interval for a Pearson correlation from n paired observations
 * (Fisher z: atanh(r) ± 1.96 / sqrt(n - 3), mapped back with tanh). Null for
 * undefined r or n <= 3.
 */
export function fisherCI(r: number | null | undefined, n: number, zCrit = 1.959964): [number, number] | null {
  if (r === null || r === undefined || !Number.isFinite(r) || n <= 3) return null;
  const clipped = Math.max(-0.999999, Math.min(0.999999, r));
  const z = Math.atanh(clipped);
  const se = 1 / Math.sqrt(n - 3);
  return [Math.tanh(z - zCrit * se), Math.tanh(z + zCrit * se)];
}

/**
 * Null benchmark for the extremes of a ranking: the expected largest |r| among
 * `m` independent draws of the sample correlation of two independent normal
 * series of length n. E[max] = ∫_0^1 (1 - F(x)^m) dx with F the CDF of |r|,
 * integrated numerically from the null density. A benchmark, not a test: the
 * candidates in a real cross-section are not independent of one another.
 */
export function nullMaxAbsCorr(m: number, n: number, steps = 2000): number {
  if (m < 1 || n <= 3) return NaN;
  const logB = logGamma(0.5) + logGamma((n - 2) / 2) - logGamma((n - 1) / 2);
  const density = (r: number) => (Math.abs(r) >= 1 ? 0 : Math.exp(((n - 4) / 2) * Math.log(1 - r * r) - logB));
  const h = 1 / steps;
  let cdf = 0; // P(|r| <= x), density of |r| is 2 f(x)
  let expected = 0;
  let prev = 2 * density(0);
  for (let i = 1; i <= steps; i++) {
    const x = i * h;
    const cur = 2 * density(x);
    const next = Math.min(1, cdf + (h * (prev + cur)) / 2);
    // trapezoid on 1 - F^m over [x - h, x]
    expected += (h * (1 - Math.pow(cdf, m) + (1 - Math.pow(next, m)))) / 2;
    cdf = next;
    prev = cur;
  }
  return expected;
}

// --- stale-result guard -----------------------------------------------------------

export interface FreshnessExpectation {
  date: string | null;
  ticker?: string | null;
  k?: number;
  tickers?: string[];
}

/**
 * True only when fetched data can safely drive an action: it is not a
 * keepPreviousData placeholder and it describes exactly the current selection
 * (date, ticker, k, basket). Anything else is "Updating…".
 */
export function isFresh(
  data: { date: string; ticker?: string; k?: number; tickers?: string[] } | undefined,
  isPlaceholderData: boolean,
  expected: FreshnessExpectation,
): boolean {
  if (!data || isPlaceholderData || expected.date === null) return false;
  if (data.date !== expected.date) return false;
  if (expected.ticker !== undefined && data.ticker !== expected.ticker) return false;
  if (expected.k !== undefined && data.k !== expected.k) return false;
  if (expected.tickers !== undefined) {
    const got = data.tickers ?? [];
    if (got.length !== expected.tickers.length) return false;
    for (let i = 0; i < got.length; i++) if (got[i] !== expected.tickers[i]) return false;
  }
  return true;
}
