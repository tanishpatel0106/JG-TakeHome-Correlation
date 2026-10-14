import type { ReactNode } from "react";

import { CardSpotlight } from "./ui/card-spotlight";

interface PanelProps {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
  id?: string;
  children: ReactNode;
}

/** A titled card with the Aceternity spotlight hover glow. Body content decides its own fixed height. */
export function Panel({ title, subtitle, actions, className, id, children }: PanelProps) {
  return (
    <CardSpotlight className={className} aria-labelledby={id ? `${id}-title` : undefined}>
      <header className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id={id ? `${id}-title` : undefined} className="text-[15px] font-semibold tracking-tight text-ink">
            {title}
          </h2>
          {subtitle ? <p className="mt-0.5 max-w-[80ch] text-[12.5px] text-muted">{subtitle}</p> : null}
        </div>
        {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
      </header>
      {children}
    </CardSpotlight>
  );
}

/** Placeholder of a given height while data loads. */
export function Skeleton({ height }: { height: number }) {
  return <div className="skeleton w-full" style={{ height }} aria-busy="true" aria-label="Loading" />;
}

/** Error or empty message occupying the panel body at a fixed height. */
export function Message({
  height,
  tone,
  children,
  onRetry,
}: {
  height: number;
  tone: "error" | "empty";
  children: ReactNode;
  onRetry?: () => void;
}) {
  return (
    <div className={`message message-${tone}`} style={{ minHeight: height }} role={tone === "error" ? "alert" : "status"}>
      <div>{children}</div>
      {onRetry ? (
        <div>
          <button type="button" className="button small" onClick={onRetry}>
            Retry
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** Small "Updating…" pill shown while placeholder (previous) data is on screen. */
export function UpdatingBadge({ show }: { show: boolean }) {
  if (!show) return null;
  return (
    <span className="updating-badge rounded-full border border-line-strong bg-surface-2 px-2.5 py-0.5 text-[11.5px] text-ink-2" role="status">
      Updating…
    </span>
  );
}
