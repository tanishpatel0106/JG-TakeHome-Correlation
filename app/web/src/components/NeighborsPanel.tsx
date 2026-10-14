import type { Neighbor, StockNeighbors } from "../api";
import { formatCorr, formatDate } from "../util";

interface NeighborListProps {
  title: string;
  rows: Neighbor[];
  focus: string;
  pair: [string, string] | null;
  onPair: (a: string, b: string) => void;
  onFocus: (ticker: string) => void;
}

/** Inline diverging bar: centred at 0, extends right for +rho (red), left for -rho (blue). */
function CorrBar({ value }: { value: number }) {
  const width = `${Math.min(1, Math.abs(value)) * 50}%`;
  const style = value >= 0 ? { left: "50%", width } : { right: "50%", width };
  return (
    <span className="corr-bar" aria-hidden="true">
      <span className={value >= 0 ? "corr-bar-fill is-pos" : "corr-bar-fill is-neg"} style={style} />
    </span>
  );
}

function NeighborList({ title, rows, focus, pair, onPair, onFocus }: NeighborListProps) {
  return (
    <table className="neighbors">
      <caption>{title}</caption>
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
          return (
            <tr
              key={row.ticker}
              className={selected ? "is-selected" : undefined}
              onClick={() => onPair(focus, row.ticker)}
              title={`Chart ρ(${focus}, ${row.ticker}) over time`}
            >
              <td>
                <button
                  type="button"
                  className="link"
                  onClick={(e) => {
                    e.stopPropagation();
                    onPair(focus, row.ticker);
                  }}
                >
                  {row.ticker}
                </button>
              </td>
              <td className="num">{formatCorr(row.corr)}</td>
              <td className="bar-cell">
                <CorrBar value={row.corr} />
              </td>
              <td className="row-actions">
                <button
                  type="button"
                  className="button ghost small"
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

interface NeighborsPanelBodyProps {
  data: StockNeighbors;
  window: number;
  pair: [string, string] | null;
  onPair: (a: string, b: string) => void;
  onFocus: (ticker: string) => void;
}

/** Top and bottom k neighbours, or the empty state when the focus stock has no valid window. */
export function NeighborsBody({ data, window, pair, onPair, onFocus }: NeighborsPanelBodyProps) {
  if (!data.valid) {
    return (
      <div className="message message-empty" role="status">
        <p>
          <strong>{data.ticker}</strong> has no correlations on {formatDate(data.date)}.
        </p>
        <p>
          Insufficient or constant history in this window: a stock needs all {window} returns up to this date, and they must
          not all be equal. Try another date or stock.
        </p>
      </div>
    );
  }
  return (
    <div className="neighbors-scroll">
      <NeighborList
        title={`Most correlated with ${data.ticker}`}
        rows={data.top}
        focus={data.ticker}
        pair={pair}
        onPair={onPair}
        onFocus={onFocus}
      />
      <NeighborList
        title={`Most anti-correlated with ${data.ticker}`}
        rows={data.bottom}
        focus={data.ticker}
        pair={pair}
        onPair={onPair}
        onFocus={onFocus}
      />
    </div>
  );
}
