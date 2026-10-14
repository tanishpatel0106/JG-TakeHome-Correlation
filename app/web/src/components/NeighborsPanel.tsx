import { useState } from "react";

import type { Neighbor, StockNeighbors } from "../api";
import { fisherCI, formatCorr, formatDate } from "../util";
import { AnimatedTabs } from "./ui/animated-tabs";

const REASON_TEXT: Record<string, string> = {
  insufficient_history: "the date is too early for a full window",
  missing_returns: "at least one return in its window is missing (a missing price, listing, delisting or a structural break)",
  constant_returns: "its returns are constant over the window, so Pearson correlation is undefined",
};

/** Bar position for a correlation in [-1, 1] on a 0–100% track centred at 0. */
function pct(value: number): number {
  return 50 + Math.max(-1, Math.min(1, value)) * 50;
}

/** Diverging bar (red +, blue −) with its 95% Fisher-z interval and the null-benchmark guide. */
function CorrBar({ value, window, nullMax }: { value: number; window: number; nullMax: number }) {
  const width = `${Math.min(1, Math.abs(value)) * 50}%`;
  const fill = value >= 0 ? { left: "50%", width } : { right: "50%", width };
  const ci = fisherCI(value, window);
  return (
    <span className="corr-bar" aria-hidden="true">
      <span className={value >= 0 ? "corr-bar-fill is-pos" : "corr-bar-fill is-neg"} style={fill} />
      {ci ? <span className="corr-bar-ci" style={{ left: `${pct(ci[0])}%`, width: `${pct(ci[1]) - pct(ci[0])}%` }} /> : null}
      {Number.isFinite(nullMax) ? (
        <>
          <span className="corr-bar-null" style={{ left: `${pct(nullMax)}%` }} />
          <span className="corr-bar-null" style={{ left: `${pct(-nullMax)}%` }} />
        </>
      ) : null}
    </span>
  );
}

interface NeighborListProps {
  rows: Neighbor[];
  focus: string;
  window: number;
  nullMax: number;
  pair: [string, string] | null;
  disabled: boolean;
  caption: string;
  onPair: (a: string, b: string) => void;
  onFocus: (ticker: string) => void;
}

function NeighborList({ rows, focus, window, nullMax, pair, disabled, caption, onPair, onFocus }: NeighborListProps) {
  return (
    <table className="neighbors">
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr>
          <th scope="col">Ticker</th>
          <th scope="col" className="num">
            ρ
          </th>
          <th scope="col">
            <span className="sr-only">Bar</span>
          </th>
          <th scope="col">
            <span className="sr-only">Actions</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const selected = pair !== null && pair.includes(focus) && pair.includes(row.ticker);
          const ci = fisherCI(row.corr, window);
          return (
            <tr
              key={row.ticker}
              className={selected ? "is-selected" : undefined}
              onClick={() => !disabled && onPair(focus, row.ticker)}
              title={`Chart ρ(${focus}, ${row.ticker}) over time`}
            >
              <td>
                <button
                  type="button"
                  className="font-semibold text-ink hover:underline disabled:opacity-50"
                  disabled={disabled}
                  onClick={(e) => {
                    e.stopPropagation();
                    onPair(focus, row.ticker);
                  }}
                >
                  {row.ticker}
                </button>
              </td>
              <td className="num" title={ci ? `95% CI ${ci[0].toFixed(2)} … ${ci[1].toFixed(2)}` : undefined}>
                {formatCorr(row.corr)}
                {ci ? <span className="sr-only">, 95% confidence interval {ci[0].toFixed(2)} to {ci[1].toFixed(2)}</span> : null}
              </td>
              <td>
                <CorrBar value={row.corr} window={window} nullMax={nullMax} />
              </td>
              <td className="text-right">
                <button
                  type="button"
                  className="button ghost small"
                  disabled={disabled}
                  onClick={(e) => {
                    e.stopPropagation();
                    onFocus(row.ticker);
                  }}
                  aria-label={`Set ${row.ticker} as focus`}
                  title="Set as focus"
                >
                  Focus
                </button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

interface NeighborsBodyProps {
  data: StockNeighbors & { reason?: string };
  window: number;
  nullMax: number;
  pair: [string, string] | null;
  disabled: boolean;
  onPair: (a: string, b: string) => void;
  onFocus: (ticker: string) => void;
}

/** Top / bottom k neighbours in animated tabs, or the empty state when the focus stock has no valid window. */
export function NeighborsBody({ data, window, nullMax, pair, disabled, onPair, onFocus }: NeighborsBodyProps) {
  const [tab, setTab] = useState("top");
  if (!data.valid) {
    const reason = data.reason ? REASON_TEXT[data.reason] : undefined;
    return (
      <div className="message message-empty" role="status">
        <p>
          <strong>{data.ticker}</strong> has no correlations on {formatDate(data.date)}.
        </p>
        <p>
          Insufficient or constant history in this window: {reason ?? `a stock needs all ${window} returns up to this date, and they must not all be equal`}.
          Try another date or stock.
        </p>
      </div>
    );
  }
  const rows = tab === "top" ? data.top : data.bottom;
  return (
    <div className="flex h-full flex-col gap-2">
      <AnimatedTabs
        label="Neighbour ranking"
        value={tab}
        onChange={setTab}
        tabs={[
          { value: "top", title: "Most correlated" },
          { value: "bottom", title: "Most anti-correlated" },
        ]}
      >
        <div className="overflow-y-auto pr-1" style={{ maxHeight: 400 }}>
          <NeighborList
            rows={rows}
            focus={data.ticker}
            window={window}
            nullMax={nullMax}
            pair={pair}
            disabled={disabled}
            caption={`${tab === "top" ? "Most correlated with" : "Most anti-correlated with"} ${data.ticker}`}
            onPair={onPair}
            onFocus={onFocus}
          />
        </div>
      </AnimatedTabs>
      {Number.isFinite(nullMax) ? (
        <p className="text-[11.5px] text-muted">
          Dashed lines: |ρ| ≈ {nullMax.toFixed(2)}, the typical largest value among {data.n_valid - 1} independent series of{" "}
          {window} returns under an independent-series null. Values near or inside them are common under that null; whiskers are 95% Fisher-z intervals.
        </p>
      ) : null}
    </div>
  );
}
