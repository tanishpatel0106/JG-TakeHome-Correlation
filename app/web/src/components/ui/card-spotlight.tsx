// Adapted from Aceternity UI "Card Spotlight" (https://ui.aceternity.com/components/card-spotlight).
// Changes: semantic <section>, theme tokens instead of fixed neutral colours, the three.js
// CanvasRevealEffect dropped (too heavy for a data dashboard): the hover glow is a CSS radial mask.
import { motion, useMotionTemplate, useMotionValue } from "motion/react";
import { type MouseEvent as ReactMouseEvent, type ReactNode, useState } from "react";

import { cn } from "../../lib/cn";

export function CardSpotlight({
  children,
  radius = 320,
  className,
  ...props
}: { radius?: number; children: ReactNode } & React.HTMLAttributes<HTMLElement>) {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const [hovering, setHovering] = useState(false);
  const mask = useMotionTemplate`radial-gradient(${radius}px circle at ${mouseX}px ${mouseY}px, white, transparent 80%)`;

  function onMouseMove({ currentTarget, clientX, clientY }: ReactMouseEvent<HTMLElement>) {
    const { left, top } = currentTarget.getBoundingClientRect();
    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  return (
    <section
      className={cn("panel", className)}
      onMouseMove={onMouseMove}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
      {...props}
    >
      <motion.div
        aria-hidden="true"
        className="pointer-events-none absolute -inset-px z-0 rounded-[inherit]"
        style={{ background: "var(--glow)", maskImage: mask, WebkitMaskImage: mask }}
        initial={{ opacity: 0 }}
        animate={{ opacity: hovering ? 1 : 0 }}
        transition={{ duration: 0.3 }}
      />
      <div className="relative z-[1]">{children}</div>
    </section>
  );
}
