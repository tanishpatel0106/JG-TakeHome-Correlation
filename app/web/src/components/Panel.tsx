import type { ReactNode } from "react";

interface PanelProps {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}

/** A titled card. Body content decides its own fixed height. */
export function Panel({ title, subtitle, actions, className, children }: PanelProps) {
  return (
    <section className={`panel ${className ?? ""}`}>
      <header className="panel-head">
        <div>
          <h2 className="panel-title">{title}</h2>
          {subtitle ? <p className="panel-subtitle">{subtitle}</p> : null}
        </div>
        {actions ? <div className="panel-actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}

/** Placeholder of a given height while data loads. */
export function Skeleton({ height }: { height: number }) {
  return <div className="skeleton" style={{ height }} aria-busy="true" aria-label="Loading" />;
}

/** Error or empty message occupying the panel body at a fixed height. */
export function Message({ height, tone, children }: { height: number; tone: "error" | "empty"; children: ReactNode }) {
  return (
    <div className={`message message-${tone}`} style={{ minHeight: height }} role={tone === "error" ? "alert" : "status"}>
      {children}
    </div>
  );
}
