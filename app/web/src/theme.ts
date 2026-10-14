// Chart colour tokens. CSS (styles.css) carries the same values for the page
// chrome; charts are canvas-rendered, so they take them from here per mode.

import type { ColorMode } from "./hooks";

export interface ChartTheme {
  surface: string;
  ink: string;
  inkSecondary: string;
  muted: string;
  grid: string;
  axis: string;
  accent: string;
  accentWash: string;
  /** Diverging ramp for rho in [-1, 1]: blue (negative) -> neutral gray -> red (positive). */
  diverging: string[];
  /** Fill for undefined (NaN) cells: a darker gray than the midpoint, hatched. */
  missing: string;
  missingHatch: string;
  positive: string;
  negative: string;
}

const LIGHT: ChartTheme = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  inkSecondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  accent: "#2a78d6",
  accentWash: "rgba(42, 120, 214, 0.12)",
  diverging: ["#0d366b", "#3987e5", "#b7d3f6", "#f0efec", "#f8cdcb", "#e34948", "#7a1616"],
  missing: "#b9b8b0",
  missingHatch: "rgba(11, 11, 11, 0.28)",
  positive: "#e34948",
  negative: "#2a78d6",
};

const DARK: ChartTheme = {
  surface: "#1a1a19",
  ink: "#f0efec",
  inkSecondary: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
  accent: "#3987e5",
  accentWash: "rgba(57, 135, 229, 0.16)",
  diverging: ["#9ec5f4", "#3987e5", "#1f4d86", "#383835", "#7e2f2e", "#e66767", "#f4aaa8"],
  missing: "#5d5d58",
  missingHatch: "rgba(240, 239, 236, 0.30)",
  positive: "#e66767",
  negative: "#3987e5",
};

export function chartTheme(mode: ColorMode): ChartTheme {
  return mode === "dark" ? DARK : LIGHT;
}

/** Shared axis / tooltip styling so every chart reads as one system. */
export function baseOption(t: ChartTheme) {
  return {
    backgroundColor: "transparent",
    animation: false,
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: t.inkSecondary },
    tooltip: {
      backgroundColor: t.surface,
      borderColor: t.axis,
      borderWidth: 1,
      textStyle: { color: t.ink, fontSize: 12 },
      extraCssText: "box-shadow: 0 4px 16px rgba(0,0,0,0.12); border-radius: 6px;",
    },
  };
}

export function axisStyle(t: ChartTheme) {
  return {
    axisLine: { lineStyle: { color: t.axis } },
    axisTick: { show: false },
    axisLabel: { color: t.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: t.grid, width: 1, type: "solid" as const } },
  };
}
