import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  api,
  fetchDownload,
  type DownloadDtype,
  type DownloadFormat,
  type Meta,
  type MatrixResult,
  metaDataset,
  metaSnapshot,
  type Timeline as TimelineData,
} from "./api";
import { DateControls } from "./components/DateControls";
import { HEATMAP_HEIGHT, Heatmap } from "./components/Heatmap";
import { HISTOGRAM_HEIGHT, Histogram } from "./components/Histogram";
import { NeighborsBody } from "./components/NeighborsPanel";
import { PAIR_HEIGHT, PairChart } from "./components/PairChart";
import { Message, Panel, Skeleton, UpdatingBadge } from "./components/Panel";
import { TickerSearch } from "./components/TickerSearch";
import { Timeline } from "./components/Timeline";
import { Modal } from "./components/ui/animated-modal";
import { DotBackground } from "./components/ui/dot-background";
import { HoverBorderGradient } from "./components/ui/hover-border-gradient";
import { Tooltip } from "./components/ui/tooltip";
import {
  type ColorMode,
  type ThemeChoice,
  type ViewState,
  readUrlState,
  useColorMode,
  useMatrix,
  useMeta,
  useNeighbors,
  usePair,
  useSummary,
  useTimeline,
  useUrlSync,
} from "./hooks";
import {
  fisherCI,
  formatBytes,
  formatCorr,
  formatDate,
  formatInt,
  isFresh,
  nullMaxAbsCorr,
  snapDate,
} from "./util";

const MAX_BASKET = 150;
const SEED_NEIGHBORS = 14;
const K_OPTIONS = [10, 20, 50, 100];
const NEIGHBORS_HEIGHT = 560;
const SEARCH_ID = "focus-search";

type DialogName = "download" | "about" | "shortcuts" | null;

export function App() {
  const meta = useMeta();
  const color = useColorMode();
  const snap = meta.data ? (metaSnapshot(meta.data)?.id ?? null) : null;
  const timeline = useTimeline(snap, meta.data !== undefined);

  const error = meta.error ?? timeline.error;
  if (error) {
    return (
      <Shell choice={color.choice} onCycleTheme={color.cycle}>
        <Message height={240} tone="error" onRetry={() => void (meta.error ? meta.refetch() : timeline.refetch())}>
          <strong>Could not load the data.</strong> {error.message}
        </Message>
      </Shell>
    );
  }
  if (!meta.data || !timeline.data) {
    return (
      <Shell choice={color.choice} onCycleTheme={color.cycle}>
        <div className="layout" aria-busy="true">
          <div className="panel area-timeline">
            <Skeleton height={230} />
          </div>
          <div className="panel area-left">
            <Skeleton height={NEIGHBORS_HEIGHT} />
          </div>
          <div className="panel area-center">
            <Skeleton height={HEATMAP_HEIGHT} />
          </div>
          <div className="panel area-right">
            <Skeleton height={HISTOGRAM_HEIGHT} />
          </div>
        </div>
      </Shell>
    );
  }
  return (
    <Explorer
      key={snap ?? "legacy"}
      meta={meta.data}
      timeline={timeline.data}
      snap={snap}
      mode={color.mode}
      choice={color.choice}
      onCycleTheme={color.cycle}
    />
  );
}

function Shell({
  choice,
  onCycleTheme,
  badges,
  actions,
  controls,
  children,
}: {
  choice: ThemeChoice;
  onCycleTheme: () => void;
  badges?: React.ReactNode;
  actions?: React.ReactNode;
  controls?: React.ReactNode;
  children: React.ReactNode;
}) {
  const label = choice === "system" ? "Theme: system" : choice === "light" ? "Theme: light" : "Theme: dark";
  return (
    <div className="relative min-h-screen">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <DotBackground className="pointer-events-none absolute inset-x-0 top-0 h-56" />
      <div className="relative mx-auto max-w-[1680px] px-4 pb-10 pt-5 sm:px-6">
        <header className="mb-4 flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="bg-gradient-to-br from-ink to-ink-2 bg-clip-text text-[22px] font-semibold tracking-tight text-transparent">
                Stock Correlation Explorer
              </h1>
              {badges}
            </div>
            <p className="text-[13px] text-muted">Trailing 20-day return correlations, computed on demand from versioned snapshots</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {actions}
            <button type="button" className="button ghost" onClick={onCycleTheme} title="Cycle theme: system → light → dark">
              {label}
            </button>
          </div>
        </header>
        {controls ? <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-4">{controls}</div> : null}
        <main id="main" tabIndex={-1}>
          {children}
        </main>
      </div>
    </div>
  );
}

/** Dates on which at least one pair has a valid window: the only selectable dates. */
function selectableDates(timeline: TimelineData): string[] {
  const out: string[] = [];
  for (let i = 0; i < timeline.dates.length; i++) {
    if ((timeline.n_valid[i] ?? 0) >= 2) out.push(timeline.dates[i]!);
  }
  return out;
}

function initialView(meta: Meta, selectable: string[], known: Set<string>): ViewState {
  const url = readUrlState();
  const fallbackDate = meta.default_date ?? selectable[selectable.length - 1] ?? null;
  const date = url.date ? (snapDate(url.date, selectable) ?? fallbackDate) : fallbackDate;
  const focus = url.focus && known.has(url.focus) ? url.focus : meta.default_ticker;
  const basket = url.basket.filter((t) => known.has(t)).slice(0, MAX_BASKET);
  const pair = url.pair && known.has(url.pair[0]) && known.has(url.pair[1]) ? url.pair : null;
  return { date, focus, basket, pair };
}

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

interface ExplorerProps {
  meta: Meta;
  timeline: TimelineData;
  snap: string | null;
  mode: ColorMode;
  choice: ThemeChoice;
  onCycleTheme: () => void;
}

function Explorer({ meta, timeline, snap, mode, choice, onCycleTheme }: ExplorerProps) {
  const queryClient = useQueryClient();
  const selectable = useMemo(() => selectableDates(timeline), [timeline]);
  const known = useMemo(() => new Set(meta.tickers), [meta.tickers]);
  const [view, setView] = useState<ViewState>(() => initialView(meta, selectable, known));
  const [k, setK] = useState(20);
  const [dialog, setDialog] = useState<DialogName>(null);
  const [tableView, setTableView] = useState(false);
  const autoSeed = useRef(view.basket.length === 0);
  useUrlSync(view, true);

  const date = view.date;
  const dateIndex = date ? timeline.dates.indexOf(date) : -1;
  const nValid = dateIndex >= 0 ? (timeline.n_valid[dateIndex] ?? 0) : 0;
  const snapshot = metaSnapshot(meta);
  const dataset = metaDataset(meta);

  const summary = useSummary(date, snap);
  const neighbors = useNeighbors(date, view.focus, k, snap);
  const matrix = useMatrix(date, view.basket, snap);
  const pairSeries = usePair(view.pair, snap);

  // Newer published snapshot? The server reloads CURRENT; the UI stays on its version until the user switches.
  const current = useQuery({
    queryKey: ["snapshot-current"],
    queryFn: ({ signal }) => api.current(signal),
    refetchInterval: 10_000,
    retry: false,
    enabled: snap !== null,
  });
  const newer = current.data && snap && current.data.id !== snap ? current.data : null;

  const neighborsFresh = isFresh(neighbors.data, neighbors.isPlaceholderData, { date, ticker: view.focus, k });
  const matrixFresh = isFresh(matrix.data, matrix.isPlaceholderData, { date, tickers: view.basket });

  const setDate = useCallback(
    (value: string) => {
      const snapped = snapDate(value, selectable);
      if (snapped) setView((v) => (v.date === snapped ? v : { ...v, date: snapped }));
    },
    [selectable],
  );
  const stepDate = useCallback(
    (delta: number) => {
      if (!date) return;
      const i = selectable.indexOf(date);
      const next = selectable[i + delta];
      if (i >= 0 && next) setView((v) => ({ ...v, date: next }));
    },
    [date, selectable],
  );
  const setFocus = useCallback((ticker: string) => setView((v) => ({ ...v, focus: ticker })), []);
  const setPair = useCallback((a: string, b: string) => setView((v) => ({ ...v, pair: [a, b] })), []);
  const addToBasket = useCallback((tickers: string[]) => {
    setView((v) => {
      const next = v.basket.slice();
      for (let i = 0; i < tickers.length && next.length < MAX_BASKET; i++) {
        if (!next.includes(tickers[i]!)) next.push(tickers[i]!);
      }
      return { ...v, basket: next };
    });
  }, []);
  const removeFromBasket = useCallback(
    (ticker: string) => setView((v) => ({ ...v, basket: v.basket.filter((t) => t !== ticker) })),
    [],
  );

  // First load without a basket in the URL: seed it from FRESH neighbours of the focus stock only.
  useEffect(() => {
    const data = neighbors.data;
    if (!autoSeed.current || !data || !neighborsFresh || !data.valid) return;
    autoSeed.current = false;
    const top: string[] = [];
    for (let i = 0; i < data.top.length && i < SEED_NEIGHBORS; i++) top.push(data.top[i]!.ticker);
    setView((v) => ({
      ...v,
      basket: v.basket.length ? v.basket : [data.ticker, ...top],
      pair: v.pair ?? (top[0] ? [data.ticker, top[0]] : null),
    }));
  }, [neighbors.data, neighborsFresh]);

  // Seeding fetches neighbours on demand; failures are shown next to the button and leave the basket as it was.
  const [seed, setSeed] = useState<{ phase: "idle" | "loading" | "error"; message?: string; date?: string; focus?: string }>({
    phase: "idle",
  });
  const seedFromFocus = useCallback(async () => {
    if (!date || !view.focus) return;
    const focus = view.focus;
    setSeed({ phase: "loading", date, focus });
    try {
      const data = await queryClient.fetchQuery({
        queryKey: ["neighbors", snap, date, focus, SEED_NEIGHBORS],
        queryFn: ({ signal }) => api.neighbors(date, focus, SEED_NEIGHBORS, snap, signal),
        staleTime: Infinity,
      });
      const tickers = [data.ticker];
      for (let i = 0; i < data.top.length; i++) tickers.push(data.top[i]!.ticker);
      // Only apply if the selection did not move on while the request was in flight.
      setView((v) => (v.date === date && v.focus === focus ? { ...v, basket: tickers } : v));
      setSeed({ phase: "idle" });
    } catch (err) {
      setSeed({ phase: "error", message: (err as Error).message, date, focus });
    }
  }, [date, view.focus, queryClient, snap]);
  // A seed result or error only describes the selection it was made for.
  const seedState = seed.date === date && seed.focus === view.focus ? seed.phase : "idle";

  // Keyboard shortcuts: ←/→ dates, "/" search, "?" help, "d" download.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (dialog || e.metaKey || e.ctrlKey || e.altKey || isTyping(e.target)) return;
      if (e.key === "ArrowLeft" || e.key === "[") {
        e.preventDefault();
        stepDate(-1);
      } else if (e.key === "ArrowRight" || e.key === "]") {
        e.preventDefault();
        stepDate(1);
      } else if (e.key === "/") {
        e.preventDefault();
        document.getElementById(SEARCH_ID)?.focus();
      } else if (e.key === "?") {
        e.preventDefault();
        setDialog("shortcuts");
      } else if (e.key === "d") {
        e.preventDefault();
        setDialog("download");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dialog, stepDate]);

  const nullMax = useMemo(() => (nValid >= 3 ? nullMaxAbsCorr(nValid - 1, meta.window) : NaN), [nValid, meta.window]);
  const announcement = date
    ? `${formatDate(date)}. ${formatInt(nValid)} valid stocks. Focus ${view.focus ?? "none"}.`
    : "No date selected.";

  if (!date) {
    return (
      <Shell choice={choice} onCycleTheme={onCycleTheme}>
        <Message height={240} tone="empty">
          No date in this data set has a complete {meta.window}-return window.
        </Message>
      </Shell>
    );
  }

  const topTen = neighborsFresh && neighbors.data?.valid ? neighbors.data.top.slice(0, 10).map((n) => n.ticker) : [];

  const badges = (
    <>
      {dataset?.kind === "demo" ? (
        <span className="rounded-full border border-warn/40 bg-warn-wash px-2.5 py-0.5 text-[11.5px] font-medium text-warn">
          {dataset.label ?? "Synthetic demo data"}
        </span>
      ) : null}
      {snapshot ? (
        <Tooltip
          content={
            <span>
              Dataset version {snapshot.id}
              {snapshot.created_at ? ` · published ${snapshot.created_at}` : ""}. Every request is pinned to this version, so
              views never mix versions.
            </span>
          }
        >
          <span
            tabIndex={0}
            className="rounded-full border border-line-strong bg-surface-2 px-2.5 py-0.5 font-mono text-[11.5px] text-ink-2"
          >
            {snapshot.id} · data as of {snapshot.data_as_of ?? meta.last_date}
          </span>
        </Tooltip>
      ) : null}
      <span className="rounded-full border border-line px-2.5 py-0.5 text-[11.5px] text-muted" title="Uniform-price days are classified using the next day's prices">
        {meta.view === "retrospective" ? "retrospective view" : "view"}
      </span>
    </>
  );

  const actions = (
    <>
      <button type="button" className="button ghost" onClick={() => setDialog("about")}>
        About these statistics
      </button>
      <button type="button" className="button ghost" onClick={() => setDialog("shortcuts")} aria-label="Keyboard shortcuts">
        <span className="kbd">?</span>
      </button>
    </>
  );

  const controls = (
    <>
      <div className="flex flex-col leading-tight">
        <span className="text-[15px] font-semibold">{formatDate(date)}</span>
        <span className="text-[12.5px] text-muted">{formatInt(nValid)} valid stocks</span>
      </div>
      <DateControls date={date} selectable={selectable} onChange={setDate} />
      <HoverBorderGradient
        as="button"
        type="button"
        onClick={() => setDialog("download")}
        className="px-4 py-1.5 text-[13px] font-semibold"
      >
        Download ρ matrix
      </HoverBorderGradient>
    </>
  );

  return (
    <Shell choice={choice} onCycleTheme={onCycleTheme} badges={badges} actions={actions} controls={controls}>
      <div className="sr-only" aria-live="polite">
        {announcement}
      </div>
      {newer ? (
        <div className="mb-4 flex flex-wrap items-center gap-3 rounded-xl border border-accent/40 bg-accent-wash px-4 py-2.5 text-[13px]" role="status">
          <span>
            A newer dataset version <strong className="font-mono">{newer.id}</strong> is published
            {newer.data_as_of ? ` (data as of ${newer.data_as_of})` : ""}. You are viewing {snap}.
          </span>
          <button type="button" className="button small" onClick={() => void queryClient.invalidateQueries({ queryKey: ["meta"] })}>
            Switch to latest
          </button>
        </div>
      ) : null}

      <div className="layout">
        <Panel
          id="timeline"
          className="area-timeline"
          title="Mean pairwise correlation"
          subtitle={`Mean sample ρ over all valid pairs per date (line) ± 1 std across pairs (band), ${meta.window}-return window. Click to select a date; vertical marks are uniform-price days, gaps are windows that span them.`}
        >
          <Timeline data={timeline} breaks={meta.breaks} selected={date} mode={mode} onSelect={setDate} />
        </Panel>

        <Panel
          id="neighbours"
          className="area-left"
          title={view.focus ? `Neighbours of ${view.focus}` : "Neighbours"}
          subtitle="Sample correlations on the selected date. Even with no real relationship, the most extreme of thousands of candidates reaches a high |ρ|; see the dashed guide."
        >
          <div className="mb-2 flex items-center gap-2">
            <div className="min-w-0 flex-1">
              <TickerSearch
                inputId={SEARCH_ID}
                tickers={meta.tickers}
                onSelect={setFocus}
                label="Focus stock"
                placeholder={`Search ${formatInt(meta.tickers.length)} tickers  ( / )`}
              />
            </div>
            <label className="flex items-center gap-1.5 text-[12.5px] text-muted">
              k
              <select className="input" value={k} onChange={(e) => setK(Number(e.target.value))}>
                {K_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="mb-2">
            <button type="button" className="button" disabled={topTen.length === 0} onClick={() => addToBasket(topTen)}>
              Add top 10 to heatmap
            </button>
          </div>
          <div className={neighborsFresh || !neighbors.data ? "panel-body" : "panel-body is-stale"} style={{ height: NEIGHBORS_HEIGHT }}>
            <UpdatingBadge show={neighbors.data !== undefined && !neighborsFresh && !neighbors.error} />
            {neighbors.error ? (
              <Message height={NEIGHBORS_HEIGHT} tone="error" onRetry={() => void neighbors.refetch()}>
                {neighbors.error.message}
              </Message>
            ) : neighbors.data ? (
              <NeighborsBody
                data={neighbors.data}
                window={meta.window}
                nullMax={nullMax}
                pair={view.pair}
                disabled={!neighborsFresh}
                onPair={setPair}
                onFocus={setFocus}
              />
            ) : view.focus ? (
              <Skeleton height={NEIGHBORS_HEIGHT} />
            ) : (
              <Message height={NEIGHBORS_HEIGHT} tone="empty">
                Pick a focus stock.
              </Message>
            )}
          </div>
        </Panel>

        <Panel
          id="heatmap"
          className="area-center"
          title={`Correlation heatmap · ${view.basket.length} stocks`}
          subtitle="Fixed diverging scale −1 … +1. Hatched cells are undefined (n/a), never 0. Click a cell to chart the pair."
          actions={
            <button type="button" className="button small" aria-pressed={tableView} onClick={() => setTableView((x) => !x)}>
              {tableView ? "Chart view" : "Table view"}
            </button>
          }
        >
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <div className="min-w-[160px] flex-1">
              <TickerSearch
                tickers={meta.tickers}
                onSelect={(t) => addToBasket([t])}
                label="Add a ticker to the heatmap"
                placeholder="Add ticker…"
                exclude={view.basket}
                disabled={view.basket.length >= MAX_BASKET}
              />
            </div>
            <button
              type="button"
              className="button"
              onClick={() => void seedFromFocus()}
              disabled={!view.focus || seedState === "loading"}
              aria-busy={seedState === "loading"}
            >
              {seedState === "loading" ? "Seeding…" : `Seed: focus + top ${SEED_NEIGHBORS}`}
            </button>
            <button
              type="button"
              className="button ghost"
              onClick={() => setView((v) => ({ ...v, basket: [] }))}
              disabled={view.basket.length === 0}
            >
              Clear
            </button>
          </div>
          {seedState === "error" ? (
            <div className="mb-2 flex flex-wrap items-center gap-2 rounded-lg bg-warn-wash px-3 py-1.5 text-[12.5px] text-warn" role="alert">
              <span>
                <strong>Could not seed from {view.focus}.</strong> {seed.message} The heatmap tickers were left unchanged.
              </span>
              <button type="button" className="button small" onClick={() => void seedFromFocus()}>
                Retry
              </button>
            </div>
          ) : null}
          <ul className="mb-2 flex h-[58px] flex-wrap content-start gap-1 overflow-y-auto" aria-label="Heatmap tickers">
            {view.basket.map((t) => (
              <li key={t} className="chip">
                <span>{t}</span>
                <button type="button" onClick={() => removeFromBasket(t)} aria-label={`Remove ${t}`} title={`Remove ${t}`}>
                  ×
                </button>
              </li>
            ))}
            {view.basket.length === 0 ? <li className="pt-1 text-[12.5px] text-muted">No tickers yet</li> : null}
          </ul>
          <div className={matrixFresh || !matrix.data ? "panel-body" : "panel-body is-stale"} style={{ height: HEATMAP_HEIGHT }}>
            <UpdatingBadge show={view.basket.length >= 2 && matrix.data !== undefined && !matrixFresh && !matrix.error} />
            {view.basket.length < 2 ? (
              <Message height={HEATMAP_HEIGHT} tone="empty">
                Add at least two tickers to draw the heatmap (up to {MAX_BASKET}).
              </Message>
            ) : matrix.error ? (
              <Message height={HEATMAP_HEIGHT} tone="error" onRetry={() => void matrix.refetch()}>
                {matrix.error.message}
              </Message>
            ) : matrix.data ? (
              tableView ? (
                <MatrixTable data={matrix.data} />
              ) : (
                <Heatmap data={matrix.data} mode={mode} onPair={matrixFresh ? setPair : () => undefined} />
              )
            ) : (
              <Skeleton height={HEATMAP_HEIGHT} />
            )}
          </div>
        </Panel>

        <Panel
          id="distribution"
          className="area-right"
          title="Distribution of ρ"
          subtitle={`All ${summary.data ? formatInt(summary.data.n_pairs) : "…"} pairs on ${formatDate(date)}`}
        >
          <div className={summary.isPlaceholderData ? "panel-body is-stale" : "panel-body"} style={{ minHeight: HISTOGRAM_HEIGHT }}>
            <UpdatingBadge show={summary.isPlaceholderData} />
            {summary.error ? (
              <Message height={HISTOGRAM_HEIGHT} tone="error" onRetry={() => void summary.refetch()}>
                {summary.error.message}
              </Message>
            ) : summary.data ? (
              <Histogram summary={summary.data} window={meta.window} mode={mode} />
            ) : (
              <Skeleton height={HISTOGRAM_HEIGHT} />
            )}
          </div>
          <dl className="my-3 grid grid-cols-2 gap-x-4 gap-y-2">
            <Stat label="Mean ρ" value={summary.data ? formatCorr(summary.data.mean) : "…"} />
            <Stat label="Std ρ" value={summary.data?.std != null ? summary.data.std.toFixed(3) : "…"} />
            <Stat label="Valid stocks" value={summary.data ? formatInt(summary.data.n_valid) : "…"} />
            <Stat label="Null-benchmark std" value={(1 / Math.sqrt(meta.window - 1)).toFixed(3)} />
          </dl>
          <p className="text-[12px] text-muted">
            The gray curve is a benchmark: the distribution expected if the series were independent. Bars that track it are{" "}
            <em>consistent with</em> that benchmark; this is not proof of independence.
          </p>
        </Panel>

        <Panel
          id="pair"
          className="area-pair"
          title={view.pair ? `ρ(${view.pair[0]}, ${view.pair[1]}) over time` : "Pair over time"}
          subtitle={`Trailing ${meta.window}-return sample correlation with its 95% Fisher-z band; the vertical line marks the selected date. Click to select a date.`}
        >
          <div className={pairSeries.isPlaceholderData ? "panel-body is-stale" : "panel-body"} style={{ height: PAIR_HEIGHT }}>
            <UpdatingBadge show={pairSeries.isPlaceholderData} />
            {!view.pair ? (
              <Message height={PAIR_HEIGHT} tone="empty">
                Click a neighbour or a heatmap cell to chart a pair.
              </Message>
            ) : pairSeries.error ? (
              <Message height={PAIR_HEIGHT} tone="error" onRetry={() => void pairSeries.refetch()}>
                {pairSeries.error.message}
              </Message>
            ) : pairSeries.data ? (
              <PairChart data={pairSeries.data} selected={date} mode={mode} onSelectDate={setDate} />
            ) : (
              <Skeleton height={PAIR_HEIGHT} />
            )}
          </div>
        </Panel>
      </div>

      <footer className="mt-5 text-[12px] text-muted">
        {dataset?.label ?? "Source data"} · {formatInt(meta.tickers.length)} tickers × {formatInt(meta.dates.length)} dates · window{" "}
        {meta.window} returns · first valid window {meta.first_valid_date ? formatDate(meta.first_valid_date) : "n/a"}
      </footer>

      <DownloadDialog open={dialog === "download"} onClose={() => setDialog(null)} date={date} snap={snap} n={meta.tickers.length} />
      <AboutDialog open={dialog === "about"} onClose={() => setDialog(null)} meta={meta} nValid={nValid} nullMax={nullMax} />
      <ShortcutsDialog open={dialog === "shortcuts"} onClose={() => setDialog(null)} />
    </Shell>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col">
      <dt className="text-[12px] text-muted">{label}</dt>
      <dd className="m-0 font-mono text-[18px] font-semibold tabular-nums">{value}</dd>
    </div>
  );
}

/** Accessible table alternative to the heatmap canvas. */
function MatrixTable({ data }: { data: MatrixResult }) {
  return (
    <div className="h-full overflow-auto rounded-lg border border-line" tabIndex={0} aria-label="Correlation matrix table">
      <table className="matrix-table">
        <caption className="sr-only">Correlation matrix on {data.date}; n/a means undefined</caption>
        <thead>
          <tr>
            <th scope="col">
              <span className="sr-only">Ticker</span>
            </th>
            {data.tickers.map((t) => (
              <th key={t} scope="col">
                {t}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.tickers.map((row, i) => (
            <tr key={row}>
              <th scope="row">{row}</th>
              {data.tickers.map((col, j) => {
                const v = data.matrix[i]?.[j];
                return (
                  <td key={col} className={v === null || v === undefined ? "is-na" : undefined}>
                    {formatCorr(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type DownloadStatus =
  | { phase: "idle" }
  | { phase: "downloading"; received: number; total: number | null }
  | { phase: "done"; bytes: number; filename: string }
  | { phase: "error"; message: string };

/** Hand a finished download to the browser as a file. */
function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function DownloadProgress({ status }: { status: DownloadStatus }) {
  if (status.phase === "downloading") {
    const pct = status.total ? Math.round((100 * status.received) / status.total) : null;
    return (
      <div role="status" className="space-y-1 text-[12.5px] text-ink-2">
        <progress className="w-full" max={status.total ?? undefined} value={status.received} aria-label="Download progress" />
        <div>
          Downloading… {formatBytes(status.received)}
          {status.total ? ` of ${formatBytes(status.total)} (${pct}%)` : ""}. The file is saved once it has fully arrived; closing
          this dialog cancels it.
        </div>
      </div>
    );
  }
  if (status.phase === "done") {
    return (
      <p role="status" className="text-[12.5px] text-ink-2">
        Saved {status.filename} ({formatBytes(status.bytes)}, complete).
      </p>
    );
  }
  if (status.phase === "error") {
    return (
      <p role="alert" className="rounded-lg bg-warn-wash px-3 py-2 text-[12.5px] text-warn">
        <strong>Download failed.</strong> {status.message}
      </p>
    );
  }
  return null;
}

function DownloadDialog({
  open,
  onClose,
  date,
  snap,
  n,
}: {
  open: boolean;
  onClose: () => void;
  date: string;
  snap: string | null;
  n: number;
}) {
  const [dtype, setDtype] = useState<DownloadDtype>("float32");
  const [format, setFormat] = useState<DownloadFormat>("npz");
  const [status, setStatus] = useState<DownloadStatus>({ phase: "idle" });
  const controller = useRef<AbortController | null>(null);
  const info = useQuery({
    queryKey: ["download-info", snap, date, dtype, format],
    queryFn: ({ signal }) => api.downloadInfo(date, dtype, format, snap, signal),
    enabled: open,
    retry: false,
    staleTime: Infinity,
  });
  const estimate = n * n * (dtype === "float32" ? 4 : 2);
  const bytes = info.data?.bytes ?? estimate;
  const filename = info.data?.filename ?? `corr_${date}.${format}`;
  const busy = status.phase === "downloading";

  // A new request (other options, date or version) or closing the dialog cancels any download in flight.
  useEffect(() => {
    controller.current?.abort();
    controller.current = null;
    setStatus({ phase: "idle" });
  }, [open, date, snap, dtype, format]);
  useEffect(() => () => controller.current?.abort(), []);

  async function start() {
    controller.current?.abort();
    const ctrl = new AbortController();
    controller.current = ctrl;
    let shown = -1;
    setStatus({ phase: "downloading", received: 0, total: info.data?.bytes ?? null });
    try {
      const blob = await fetchDownload(
        api.downloadUrl(date, dtype, format, snap),
        (received, total) => {
          const pct = total ? Math.floor((100 * received) / total) : Math.floor(received / 1e6);
          if (pct === shown) return;
          shown = pct;
          setStatus({ phase: "downloading", received, total });
        },
        ctrl.signal,
      );
      if (ctrl.signal.aborted) return;
      saveBlob(blob, filename);
      setStatus({ phase: "done", bytes: blob.size, filename });
    } catch (err) {
      if (ctrl.signal.aborted) return;
      setStatus({ phase: "error", message: (err as Error).message });
    } finally {
      if (controller.current === ctrl) controller.current = null;
    }
  }

  const blocked = info.isError || busy;
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Download the full correlation matrix"
      description={`${formatInt(n)} × ${formatInt(n)} on ${formatDate(date)}, rows and columns in /api/meta ticker order.`}
      footer={
        <>
          <button type="button" className="button ghost" onClick={onClose}>
            {busy ? "Cancel download" : status.phase === "done" ? "Close" : "Cancel"}
          </button>
          <button type="button" className="button primary" onClick={() => void start()} disabled={blocked}>
            {busy ? "Downloading…" : status.phase === "error" ? "Retry download" : `Download ${formatBytes(bytes)}`}
          </button>
        </>
      }
    >
      <div className="prose-dialog space-y-4">
        <fieldset className="space-y-1.5">
          <legend className="mb-1 text-[12.5px] font-semibold text-ink">Precision</legend>
          <label className="flex items-center gap-2">
            <input type="radio" name="dtype" checked={dtype === "float32"} onChange={() => setDtype("float32")} />
            float32: matches the API values within 1e-6 (recommended)
          </label>
          <label className="flex items-center gap-2">
            <input type="radio" name="dtype" checked={dtype === "float16"} onChange={() => setDtype("float16")} />
            float16: half the size, values rounded to about ±0.001
          </label>
          {dtype === "float16" ? (
            <p className="rounded-lg bg-warn-wash px-3 py-2 text-[12.5px] text-warn" role="note">
              float16 loses precision (≈3 decimals). Use it for visual work only, not for re-ranking or statistics.
            </p>
          ) : null}
        </fieldset>
        <fieldset className="space-y-1.5">
          <legend className="mb-1 text-[12.5px] font-semibold text-ink">Format</legend>
          <label className="flex items-center gap-2">
            <input type="radio" name="format" checked={format === "npz"} onChange={() => setFormat("npz")} />
            .npz: self-describing (corr, tickers, metadata)
          </label>
          <label className="flex items-center gap-2">
            <input type="radio" name="format" checked={format === "npy"} onChange={() => setFormat("npy")} />
            .npy: matrix only
          </label>
        </fieldset>
        <p className="text-[12.5px] text-muted">
          {info.data ? `Exact size ${formatBytes(info.data.bytes)}` : `About ${formatBytes(estimate)}`} · invalid stocks are NaN rows and
          columns · snapshot {snap ?? "n/a"}
        </p>
        {info.isError ? (
          <Message height={0} tone="error" onRetry={() => void info.refetch()}>
            <strong>This file cannot be prepared.</strong> {info.error.message}
          </Message>
        ) : null}
        <DownloadProgress status={status} />
        <pre className="overflow-x-auto rounded-lg bg-surface-2 p-3 font-mono text-[12px]">
          {format === "npz"
            ? `import numpy as np, json\nz = np.load("${filename}")\ncorr, tickers = z["corr"], z["tickers"]\nmeta = json.loads(str(z["meta"]))`
            : `import numpy as np\ncorr = np.load("${filename}")  # order: /api/meta tickers`}
        </pre>
      </div>
    </Modal>
  );
}

function AboutDialog({
  open,
  onClose,
  meta,
  nValid,
  nullMax,
}: {
  open: boolean;
  onClose: () => void;
  meta: Meta;
  nValid: number;
  nullMax: number;
}) {
  const W = meta.window;
  const ci = fisherCI(0.5, W);
  const anomalies = meta.anomalies ?? [];
  return (
    <Modal open={open} onClose={onClose} title="About these statistics" description="What the numbers mean, and what they do not.">
      <div className="prose-dialog">
        <h3>Short-window uncertainty</h3>
        <p>
          Each ρ is a sample correlation over {W} daily returns, so it is noisy. Near 0 its standard error is about{" "}
          {(1 / Math.sqrt(W - 1)).toFixed(2)}; a measured ρ = 0.50 has a 95% Fisher-z interval of roughly{" "}
          {ci ? `${ci[0].toFixed(2)} to ${ci[1].toFixed(2)}` : "n/a"}.
        </p>
        <h3>Multiple comparisons</h3>
        <p>
          Ranking one stock against {formatInt(Math.max(nValid - 1, 0))} others picks extremes. Among that many independent series the
          largest |ρ| is typically about {Number.isFinite(nullMax) ? nullMax.toFixed(2) : "n/a"}, so top or bottom neighbours do not by
          themselves establish a stable relationship.
        </p>
        <h3>The null benchmark is not a proof</h3>
        <p>
          The histogram's gray curve is the distribution expected for independent series. Matching it is <em>consistent with</em> that
          benchmark; it does not prove the stocks are independent or that every correlation is noise.
        </p>
        <h3>Validity</h3>
        <p>
          A stock needs all {W} returns in its window and non-constant returns; otherwise its correlations (including with itself) are
          undefined, shown as n/a, never 0, and never ranked.
        </p>
        <h3>Retrospective cleaning</h3>
        <p>
          Days on which every stock has the same price are classified using the <em>next</em> day&apos;s prices (look-ahead), so history is a
          retrospectively cleaned view.
        </p>
        {anomalies.length ? (
          <ul>
            {anomalies.map((a) => (
              <li key={a.date}>
                <code>{a.date}</code> {a.classification} ({a.status}
                {a.look_ahead ? ", look-ahead" : ""}) — {a.action?.replaceAll("_", " ")}
                {a.affected?.window_end_dates?.count ? `; ${a.affected.window_end_dates.count} window ends affected` : ""}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </Modal>
  );
}

function ShortcutsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const rows: [string, string][] = [
    ["← / →", "Previous / next valid date"],
    ["/", "Search the focus ticker"],
    ["d", "Download dialog"],
    ["?", "This help"],
    ["Esc", "Close dialogs and lists"],
  ];
  return (
    <Modal open={open} onClose={onClose} title="Keyboard shortcuts">
      <table className="w-full text-[13px]">
        <tbody>
          {rows.map(([key, what]) => (
            <tr key={key} className="border-b border-line">
              <td className="py-1.5 pr-4">
                <span className="kbd">{key}</span>
              </td>
              <td className="py-1.5">{what}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Modal>
  );
}
