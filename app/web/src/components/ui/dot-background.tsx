// Adapted from Aceternity UI "Grid and Dot Backgrounds" (https://ui.aceternity.com/components/grid-and-dot-backgrounds):
// a dotted backdrop with a radial fade mask. Decorative only (aria-hidden), colours from theme tokens.
import { cn } from "../../lib/cn";

export function DotBackground({ className }: { className?: string }) {
  return (
    <div aria-hidden="true" className={cn("pointer-events-none absolute inset-0 overflow-hidden", className)}>
      <div className="absolute inset-0 [background-image:radial-gradient(var(--dot)_1px,transparent_1px)] [background-size:18px_18px]" />
      <div className="absolute inset-0 bg-page [mask-image:radial-gradient(ellipse_at_top,transparent_15%,black_75%)]" />
      <div className="absolute -top-24 left-1/2 h-48 w-[60%] -translate-x-1/2 rounded-full bg-accent opacity-[0.10] blur-3xl" />
    </div>
  );
}
