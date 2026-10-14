// Adapted from Aceternity UI "Hover Border Gradient" (https://ui.aceternity.com/components/hover-border-gradient).
// Changes: theme-token colours, polymorphic `as` typed for <a>/<button>, rotation pauses for
// prefers-reduced-motion (useReducedMotion) and while hovered.
import { motion, useReducedMotion } from "motion/react";
import { type ElementType, type ReactNode, useEffect, useState } from "react";

import { cn } from "../../lib/cn";

type Direction = "TOP" | "LEFT" | "BOTTOM" | "RIGHT";
const ORDER: Direction[] = ["TOP", "LEFT", "BOTTOM", "RIGHT"];

const MOVING: Record<Direction, string> = {
  TOP: "radial-gradient(20.7% 50% at 50% 0%, var(--accent-strong) 0%, transparent 100%)",
  LEFT: "radial-gradient(16.6% 43.1% at 0% 50%, var(--accent-strong) 0%, transparent 100%)",
  BOTTOM: "radial-gradient(20.7% 50% at 50% 100%, var(--accent-strong) 0%, transparent 100%)",
  RIGHT: "radial-gradient(16.2% 41.2% at 100% 50%, var(--accent-strong) 0%, transparent 100%)",
};
const HIGHLIGHT = "radial-gradient(75% 181% at 50% 50%, var(--accent) 0%, transparent 100%)";

type Props = {
  as?: ElementType;
  children: ReactNode;
  containerClassName?: string;
  className?: string;
  duration?: number;
} & Record<string, unknown>;

export function HoverBorderGradient({ as: Tag = "button", children, containerClassName, className, duration = 1, ...props }: Props) {
  const [hovered, setHovered] = useState(false);
  const [direction, setDirection] = useState<Direction>("TOP");
  const reduce = useReducedMotion();

  useEffect(() => {
    if (hovered || reduce) return;
    const id = window.setInterval(() => {
      setDirection((d) => ORDER[(ORDER.indexOf(d) - 1 + ORDER.length) % ORDER.length]!);
    }, duration * 1000);
    return () => window.clearInterval(id);
  }, [hovered, reduce, duration]);

  return (
    <Tag
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={cn(
        "relative inline-flex h-min w-fit shrink-0 items-center justify-center overflow-visible rounded-full border border-line-strong p-px no-underline transition duration-500",
        containerClassName,
      )}
      {...props}
    >
      <span className={cn("relative z-10 inline-flex items-center gap-2 rounded-[inherit] bg-surface-2 px-4 py-1.5 text-[13px] font-semibold text-ink", className)}>
        {children}
      </span>
      <motion.span
        aria-hidden="true"
        className="absolute inset-0 z-0 flex-none overflow-hidden rounded-[inherit]"
        style={{ filter: "blur(2px)" }}
        initial={{ background: MOVING[direction] }}
        animate={{ background: hovered ? [MOVING[direction], HIGHLIGHT] : MOVING[direction] }}
        transition={{ ease: "linear", duration }}
      />
      <span aria-hidden="true" className="absolute inset-[2px] z-[1] flex-none rounded-full bg-surface-2" />
    </Tag>
  );
}
