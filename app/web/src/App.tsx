import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, type Meta, type Timeline as TimelineData } from "./api";
import { DateControls } from "./components/DateControls";
import { HEATMAP_HEIGHT, Heatmap } from "./components/Heatmap";
import { HISTOGRAM_HEIGHT, Histogram } from "./components/Histogram";
import { NeighborsBody } from "./components/NeighborsPanel";
import { PAIR_HEIGHT, PairChart } from "./components/PairChart";
import { Message, Panel, Skeleton } from "./components/Panel";
import { TickerSearch } from "./components/TickerSearch";
import { Timeline } from "./components/Timeline";
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
import { formatBytes, formatCorr, formatDate, formatInt, snapDate } from "./util";

const MAX_BASKET = 150;
const SEED_NEIGHBORS = 14;
const K_OPTIONS = [10, 20, 50, 100];
const NEIGHBORS_HEIGHT = 520;

export function App() {
  const meta = useMeta();
  const timeline = useTimeline();
  const color = useColorMode();

  const error = meta.error ?? timeline.error;
  if (error) {
    return (
      <Shell choice={color.choice} onCycleTheme={color.cycle}>
        <Message height={240} tone="error">
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
  return <Explorer meta={meta.data} timeline={timeline.data} mode={color.mode} choice={color.choice} onCycleTheme={color.cycle} />;
}

function Shell({
  choice,
  onCycleTheme,
  controls,
  children,
}: {
  choice: ThemeChoice;
  onCycleTheme: () => void;
  controls?: React.ReactNode;
  children: React.ReactNode;
}) {
  const label = choice === "system" ? "Theme: system" : choice === "light" ? "Theme: light" : "Theme: dark";
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Stock Correlation Explorer</h1>
          <p className="muted">Trailing-window return correlations, computed on demand</p>
        </div>
        <div className="topbar-controls">
          {controls}
          <button type="button" className="button ghost" onClick={onCycleTheme} title="Cycle theme: system → light → dark">
            {label}
          </button>
        </div>
      </header>
      <main>{children}</main>
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

interface ExplorerProps {
  meta: Meta;
  timeline: TimelineData;
  mode: ColorMode;
  choice: ThemeChoice;
  onCycleTheme: () => void;
}

function Explorer({ meta, timeline, mode, choice, onCycleTheme }: ExplorerProps) {
  const queryClient = useQueryClient();
  const selectable = useMemo(() => selectableDates(timeline), [timeline]);
  const known = useMemo(() => new Set(meta.tickers), [meta.tickers]);
  const [view, setView] = useState<ViewState>(() => initialView(meta, selectable, known));
  const [k, setK] = useState(20);
  const autoSeed = useRef(view.basket.length === 0);
  useUrlSync(view, true);

  const date = view.date;
  const dateIndex = date ? timeline.dates.indexOf(date) : -1;
  const nValid = dateIndex >= 0 ? (timeline.n_valid[dateIndex] ?? 0) : 0;

  const summary = useSummary(date);
  const neighbors = useNeighbors(date, view.focus, k);
  const matrix = useMatrix(date, view.basket);
  const pairSeries = usePair(view.pair);

  const setDate = useCallback(
    (value: string) => {
      const snapped = snapDate(value, selectable);
      if (snapped) setView((v) => (v.date === snapped ? v : { ...v, date: snapped }));
    },
    [selectable],
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

  // First load without a basket in the URL: seed it with the focus and its top
  // neighbours, and chart the focus against its closest neighbour.
  useEffect(() => {
    const data = neighbors.data;
    if (!autoSeed.current || !data || neighbors.isPlaceholderData || !data.valid) return;
    autoSeed.current = false;
    const top: string[] = [];
    for (let i = 0; i < data.top.length && i < SEED_NEIGHBORS; i++) top.push(data.top[i]!.ticker);
    setView((v) => ({
      ...v,
      basket: v.basket.length ? v.basket : [data.ticker, ...top],
      pair: v.pair ?? (top[0] ? [data.ticker, top[0]] : null),
    }));
  }, [neighbors.data, neighbors.isPlaceholderData]);

  const seedFromFocus = useCallback(async () => {
    if (!date || !view.focus) return;
    const data = await queryClient.fetchQuery({
      queryKey: ["neighbors", date, view.focus, SEED_NEIGHBORS],
      queryFn: ({ signal }) => api.neighbors(date, view.focus!, SEED_NEIGHBORS, signal),
      staleTime: Infinity,
    });
    const tickers = [data.ticker];
    for (let i = 0; i < data.top.length; i++) tickers.push(data.top[i]!.ticker);
    setView((v) => ({ ...v, basket: tickers }));
  }, [date, view.focus, queryClient]);

  const downloadBytes = 128 + meta.tickers.length * meta.tickers.length * 2;

  const controls = date ? (
    <>
      <div className="date-summary">
        <span className="date-label">{formatDate(date)}</span>
        <span className="muted">{formatInt(nValid)} valid stocks</span>
      </div>
      <DateControls date={date} selectable={selectable} onChange={setDate} />
      <a
        className="button primary"
        href={api.downloadUrl(date)}
        download
        title={`float16 .npy, ${formatInt(meta.tickers.length)} × ${formatInt(meta.tickers.length)}; rows and columns follow /api/meta tickers (alphabetical)`}
      >
        Download ρ matrix · {formatBytes(downloadBytes)}
      </a>
    </>
  ) : null;

  if (!date) {
    return (
      <Shell choice={choice} onCycleTheme={onCycleTheme}>
        <Message height={240} tone="empty">
          No date in this data set has a complete {meta.window}-return window.
        </Message>
      </Shell>
    );
  }

  const topTen = neighbors.data?.valid ? neighbors.data.top.slice(0, 10).map((n) => n.ticker) : [];

  return (
    <Shell choice={choice} onCycleTheme={onCycleTheme} controls={controls}>
      <div className="layout">
        <Panel
          className="area-timeline"
          title="Mean pairwise correlation"
          subtitle={`Mean ρ over all valid pairs per date (line) ± 1 std across pairs (band), ${meta.window}-day window. Click to select a date; gaps are windows that span a re-base or placeholder day.`}
        >
          <Timeline data={timeline} breaks={meta.breaks} selected={date} mode={mode} onSelect={setDate} />
        </Panel>

        <Panel
          className="area-left"
          title={view.focus ? `Neighbours of ${view.focus}` : "Neighbours"}
          subtitle="Top and bottom k by ρ on the selected date. Click a row to chart the pair."
        >
          <div className="toolbar">
            <TickerSearch tickers={meta.tickers} onSelect={setFocus} label="Focus stock" placeholder={`Search ${formatInt(meta.tickers.length)} tickers…`} />
            <label className="k-select">
              <span className="muted">k</span>
              <select className="input" value={k} onChange={(e) => setK(Number(e.target.value))}>
                {K_OPTIONS.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="toolbar">
            <button type="button" className="button" disabled={topTen.length === 0} onClick={() => addToBasket(topTen)}>
              Add top 10 to heatmap
            </button>
          </div>
          <div className={neighbors.isPlaceholderData ? "panel-body is-stale" : "panel-body"} style={{ height: NEIGHBORS_HEIGHT }}>
            {neighbors.error ? (
              <Message height={NEIGHBORS_HEIGHT} tone="error">
                {neighbors.error.message}
              </Message>
            ) : neighbors.data ? (
              <NeighborsBody data={neighbors.data} window={meta.window} pair={view.pair} onPair={setPair} onFocus={setFocus} />
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
          className="area-center"
          title={`Correlation heatmap · ${view.basket.length} stocks`}
          subtitle="Fixed diverging scale from −1 to +1. Hatched gray cells are undefined. Click a cell to chart the pair."
        >
          <div className="toolbar">
            <TickerSearch
              tickers={meta.tickers}
              onSelect={(t) => addToBasket([t])}
              label="Add a ticker to the heatmap"
              placeholder="Add ticker…"
              exclude={view.basket}
              disabled={view.basket.length >= MAX_BASKET}
            />
            <button type="button" className="button" onClick={() => void seedFromFocus()} disabled={!view.focus}>
              Seed: focus + top {SEED_NEIGHBORS}
            </button>
            <button type="button" className="button ghost" onClick={() => setView((v) => ({ ...v, basket: [] }))} disabled={view.basket.length === 0}>
              Clear
            </button>
          </div>
          <ul className="chips" aria-label="Heatmap tickers">
            {view.basket.map((t) => (
              <li key={t} className="chip">
                <span>{t}</span>
                <button type="button" onClick={() => removeFromBasket(t)} aria-label={`Remove ${t}`} title={`Remove ${t}`}>
                  ×
                </button>
              </li>
            ))}
            {view.basket.length === 0 ? <li className="muted chips-empty">No tickers yet</li> : null}
          </ul>
          <div className={matrix.isPlaceholderData ? "panel-body is-stale" : "panel-body"} style={{ height: HEATMAP_HEIGHT }}>
            {view.basket.length < 2 ? (
              <Message height={HEATMAP_HEIGHT} tone="empty">
                Add at least two tickers to draw the heatmap (up to {MAX_BASKET}).
              </Message>
            ) : matrix.error ? (
              <Message height={HEATMAP_HEIGHT} tone="error">
                {matrix.error.message}
              </Message>
            ) : matrix.data ? (
              <Heatmap data={matrix.data} mode={mode} onPair={setPair} />
            ) : (
              <Skeleton height={HEATMAP_HEIGHT} />
            )}
          </div>
        </Panel>

        <Panel className="area-right" title="Distribution of ρ" subtitle={`All ${summary.data ? formatInt(summary.data.n_pairs) : "…"} pairs on ${formatDate(date)}`}>
          <div className={summary.isPlaceholderData ? "panel-body is-stale" : "panel-body"} style={{ minHeight: HISTOGRAM_HEIGHT }}>
            {summary.error ? (
              <Message height={HISTOGRAM_HEIGHT} tone="error">
                {summary.error.message}
              </Message>
            ) : summary.data ? (
              <Histogram summary={summary.data} window={meta.window} mode={mode} />
            ) : (
              <Skeleton height={HISTOGRAM_HEIGHT} />
            )}
          </div>
          <dl className="stats">
            <div>
              <dt>Mean ρ</dt>
              <dd>{summary.data ? formatCorr(summary.data.mean) : "…"}</dd>
            </div>
            <div>
              <dt>Std ρ</dt>
              <dd>{summary.data?.std != null ? summary.data.std.toFixed(3) : "…"}</dd>
            </div>
            <div>
              <dt>Valid stocks</dt>
              <dd>{summary.data ? formatInt(summary.data.n_valid) : "…"}</dd>
            </div>
            <div>
              <dt>Independent-series std</dt>
              <dd>{(1 / Math.sqrt(meta.window - 1)).toFixed(3)}</dd>
            </div>
          </dl>
          <p className="note muted">
            The gray curve is the distribution expected if all stocks were independent. Bars that track it mean the cross-section has no
            common factor on this date.
          </p>
        </Panel>

        <Panel
          className="area-pair"
          title={view.pair ? `ρ(${view.pair[0]}, ${view.pair[1]}) over time` : "Pair over time"}
          subtitle="Trailing-window correlation of the selected pair; the vertical line marks the selected date. Click to select a date."
        >
          <div className={pairSeries.isPlaceholderData ? "panel-body is-stale" : "panel-body"} style={{ height: PAIR_HEIGHT }}>
            {!view.pair ? (
              <Message height={PAIR_HEIGHT} tone="empty">
                Click a neighbour or a heatmap cell to chart a pair.
              </Message>
            ) : pairSeries.error ? (
              <Message height={PAIR_HEIGHT} tone="error">
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
      <footer className="footer muted">
        Source {String((meta.manifest.source as { file_name?: string } | undefined)?.file_name ?? "")} · built{" "}
        {String(meta.manifest.built_at ?? "")} · {formatInt(meta.tickers.length)} tickers × {formatInt(meta.dates.length)} dates ·
        window {meta.window} returns · first valid window {meta.first_valid_date ? formatDate(meta.first_valid_date) : "n/a"}
      </footer>
    </Shell>
  );
}
