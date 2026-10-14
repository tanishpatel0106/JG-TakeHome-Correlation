import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** Aceternity / shadcn convention: merge conditional class lists, last Tailwind class wins. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
