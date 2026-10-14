import { useId, useMemo, useRef, useState } from "react";

const MAX_OPTIONS = 50;

interface TickerSearchProps {
  tickers: string[];
  onSelect: (ticker: string) => void;
  label: string;
  placeholder?: string;
  disabled?: boolean;
  /** Tickers to leave out of the list (e.g. already in the basket). */
  exclude?: string[];
}

/** Exact match first, then prefix matches, then substring matches; at most 50. */
export function filterTickers(tickers: string[], query: string, exclude: string[] = []): string[] {
  const q = query.trim().toUpperCase();
  if (!q) return [];
  const skip = new Set(exclude);
  const exact: string[] = [];
  const prefix: string[] = [];
  const contains: string[] = [];
  for (let i = 0; i < tickers.length; i++) {
    const t = tickers[i]!;
    if (skip.has(t)) continue;
    if (t === q) exact.push(t);
    else if (t.startsWith(q)) prefix.push(t);
    else if (contains.length < MAX_OPTIONS && t.includes(q)) contains.push(t);
  }
  return exact.concat(prefix, contains).slice(0, MAX_OPTIONS);
}

/** Accessible combobox (ARIA 1.2 pattern): type to filter, arrows to move, Enter to pick, Esc to close. */
export function TickerSearch({ tickers, onSelect, label, placeholder, disabled, exclude }: TickerSearchProps) {
  const id = useId();
  const listId = `${id}-list`;
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);

  const options = useMemo(() => filterTickers(tickers, query, exclude), [tickers, query, exclude]);
  const showList = open && options.length > 0;
  const noMatch = open && query.trim() !== "" && options.length === 0;

  function choose(ticker: string) {
    onSelect(ticker);
    setQuery("");
    setOpen(false);
    setActive(0);
  }

  function move(delta: number) {
    if (options.length === 0) return;
    const next = (active + delta + options.length) % options.length;
    setActive(next);
    const item = listRef.current?.children[next] as HTMLElement | undefined;
    item?.scrollIntoView({ block: "nearest" });
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      if (!open) setOpen(true);
      else move(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      move(-1);
    } else if (event.key === "Enter") {
      const pick = options[active];
      if (open && pick) {
        event.preventDefault();
        choose(pick);
      }
    } else if (event.key === "Escape") {
      if (open) {
        event.preventDefault();
        setOpen(false);
      } else {
        setQuery("");
      }
    }
  }

  return (
    <div className="combo">
      <label className="sr-only" htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className="input"
        type="text"
        role="combobox"
        aria-expanded={showList}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={showList ? `${id}-opt-${active}` : undefined}
        autoComplete="off"
        spellCheck={false}
        placeholder={placeholder ?? label}
        value={query}
        disabled={disabled}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
          setActive(0);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={onKeyDown}
      />
      <ul id={listId} ref={listRef} role="listbox" className="combo-list" hidden={!showList} aria-label={label}>
        {options.map((t, i) => (
          <li
            key={t}
            id={`${id}-opt-${i}`}
            role="option"
            aria-selected={i === active}
            className={i === active ? "combo-option is-active" : "combo-option"}
            onMouseDown={(e) => {
              e.preventDefault(); // keep focus in the input
              choose(t);
            }}
            onMouseEnter={() => setActive(i)}
          >
            {t}
          </li>
        ))}
      </ul>
      {noMatch ? <div className="combo-empty">No ticker matches “{query.trim()}”</div> : null}
    </div>
  );
}
