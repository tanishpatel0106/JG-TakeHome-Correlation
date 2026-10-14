// Adapted from Aceternity UI "Animated Tabs" (https://ui.aceternity.com/components/tabs): the sliding
// pill (motion layoutId) is kept; the 3-D stacked content is replaced by a single tab panel, and the
// WAI-ARIA tabs pattern is added (roving tabindex, Left/Right/Home/End, aria-selected/controls).
import { motion } from "motion/react";
import { type KeyboardEvent, type ReactNode, useId, useRef } from "react";

import { cn } from "../../lib/cn";

export interface TabItem {
  value: string;
  title: ReactNode;
}

interface AnimatedTabsProps {
  tabs: TabItem[];
  value: string;
  onChange: (value: string) => void;
  label: string;
  className?: string;
  children: ReactNode;
}

export function AnimatedTabs({ tabs, value, onChange, label, className, children }: AnimatedTabsProps) {
  const id = useId();
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const activeIndex = Math.max(0, tabs.findIndex((t) => t.value === value));

  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    let next = -1;
    if (event.key === "ArrowRight") next = (activeIndex + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (activeIndex - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    if (next < 0) return;
    event.preventDefault();
    event.stopPropagation();
    onChange(tabs[next]!.value);
    refs.current[next]?.focus();
  }

  return (
    <div className={className}>
      <div role="tablist" aria-label={label} className="inline-flex rounded-full border border-line bg-surface-2 p-1" onKeyDown={onKeyDown}>
        {tabs.map((tab, i) => {
          const selected = i === activeIndex;
          return (
            <button
              key={tab.value}
              ref={(el) => {
                refs.current[i] = el;
              }}
              id={`${id}-tab-${tab.value}`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`${id}-panel`}
              tabIndex={selected ? 0 : -1}
              onClick={() => onChange(tab.value)}
              className={cn(
                "relative rounded-full px-3.5 py-1 text-[12.5px] font-medium transition-colors",
                selected ? "text-ink" : "text-muted hover:text-ink",
              )}
            >
              {selected ? (
                <motion.span
                  layoutId={`${id}-pill`}
                  transition={{ type: "spring", bounce: 0.25, duration: 0.5 }}
                  className="absolute inset-0 rounded-full border border-line-strong bg-surface-3"
                />
              ) : null}
              <span className="relative">{tab.title}</span>
            </button>
          );
        })}
      </div>
      <div role="tabpanel" id={`${id}-panel`} aria-labelledby={`${id}-tab-${tabs[activeIndex]?.value}`} className="mt-3">
        {children}
      </div>
    </div>
  );
}
