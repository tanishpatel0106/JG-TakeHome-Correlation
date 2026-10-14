import { useState } from "react";

import { formatDate, snapDate } from "../util";

interface DateControlsProps {
  date: string;
  /** Sorted dates that have a valid correlation window. */
  selectable: string[];
  onChange: (date: string) => void;
}

/** Date input limited to [first valid, last] plus previous / next trading-day buttons. */
export function DateControls({ date, selectable, onChange }: DateControlsProps) {
  // A snapping note belongs to the date it produced; it disappears once the date moves on.
  const [note, setNote] = useState<{ date: string; text: string } | null>(null);
  const index = selectable.indexOf(date);
  const first = selectable[0] ?? date;
  const last = selectable[selectable.length - 1] ?? date;

  function step(delta: number) {
    const next = selectable[index + delta];
    if (next) {
      setNote(null);
      onChange(next);
    }
  }

  function pick(value: string) {
    if (!value) return;
    const snapped = snapDate(value, selectable);
    if (!snapped) return;
    setNote(
      snapped !== value
        ? { date: snapped, text: `No valid window on ${formatDate(value)}; showing the nearest date` }
        : null,
    );
    onChange(snapped);
  }

  return (
    <div className="date-controls">
      <button
        type="button"
        className="button icon"
        onClick={() => step(-1)}
        disabled={index <= 0}
        aria-label="Previous trading day"
        title="Previous trading day"
      >
        ◀
      </button>
      <label className="sr-only" htmlFor="date-input">
        Date
      </label>
      <input
        id="date-input"
        className="input date-input"
        type="date"
        min={first}
        max={last}
        value={date}
        required
        onChange={(e) => pick(e.target.value)}
      />
      <button
        type="button"
        className="button icon"
        onClick={() => step(1)}
        disabled={index < 0 || index >= selectable.length - 1}
        aria-label="Next trading day"
        title="Next trading day"
      >
        ▶
      </button>
      <span className="date-note" role="status">
        {note && note.date === date ? note.text : null}
      </span>
    </div>
  );
}
