// Tooltip in the style of Aceternity UI "Animated Tooltip" / "Tooltip Card"
// (https://ui.aceternity.com/components/animated-tooltip): spring fade-in, theme tokens.
// Opens on hover AND keyboard focus, closes on Esc; content is linked with aria-describedby.
import { AnimatePresence, motion } from "motion/react";
import { type ReactElement, type ReactNode, cloneElement, useId, useState } from "react";

import { cn } from "../../lib/cn";

interface TooltipProps {
  content: ReactNode;
  children: ReactElement<Record<string, unknown>>;
  side?: "top" | "bottom";
  className?: string;
}

export function Tooltip({ content, children, side = "bottom", className }: TooltipProps) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const trigger = cloneElement(children, {
    "aria-describedby": open ? id : undefined,
    onFocus: () => setOpen(true),
    onBlur: () => setOpen(false),
    onKeyDown: (e: React.KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    },
  });
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      {trigger}
      <AnimatePresence>
        {open ? (
          <motion.span
            id={id}
            role="tooltip"
            initial={{ opacity: 0, y: side === "top" ? 6 : -6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", stiffness: 300, damping: 22 }}
            className={cn(
              "absolute left-1/2 z-50 w-max max-w-[300px] -translate-x-1/2 rounded-xl border border-line-strong bg-surface-2 px-3 py-2 text-left text-[12px] leading-snug text-ink-2 shadow-2xl",
              side === "top" ? "bottom-[calc(100%+8px)]" : "top-[calc(100%+8px)]",
              className,
            )}
          >
            {content}
          </motion.span>
        ) : null}
      </AnimatePresence>
    </span>
  );
}
