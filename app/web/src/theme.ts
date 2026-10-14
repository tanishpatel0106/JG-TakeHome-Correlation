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
  /** Anomaly / warning markers. */
  warn: string;
  /** Confidence-band fill. */
  band: string;
}

const LIGHT: ChartTheme = {
  surface: "#ffffff",
  ink: "#0a0a0c",
  inkSecondary: "#3c3c44",
  muted: "#64646e",
  grid: "#ececf0",
  axis: "#d4d4dc",
  accent: "#5b4fe0",
  accentWash: "rgba(91, 79, 224, 0.12)",
  diverging: ["#0d366b", "#2468c9", "#b7d3f6", "#f0f0f3", "#f8cdcb", "#d23c3b", "#7a1616"],
  missing: "#c4c4cc",
  missingHatch: "rgba(10, 10, 12, 0.28)",
  positive: "#d23c3b",
  negative: "#2468c9",
  warn: "#b26a00",
  band: "rgba(91, 79, 224, 0.14)",
};

const DARK: ChartTheme = {
  surface: "#111114",
  ink: "#f4f4f5",
  inkSecondary: "#c4c4cc",
  muted: "#9a9aa5",
  grid: "#1f1f24",
  axis: "#2e2e35",
  accent: "#8b8cf8",
  accentWash: "rgba(139, 140, 248, 0.16)",
  diverging: ["#a8ccf7", "#4f8fe6", "#1f3f6e", "#2a2a30", "#6e2a2a", "#e66767", "#f6b3b1"],
  missing: "#4a4a52",
  missingHatch: "rgba(244, 244, 245, 0.30)",
  positive: "#f07272",
  negative: "#5b9cf0",
  warn: "#f5b84d",
  band: "rgba(139, 140, 248, 0.20)",
};

export function chartTheme(mode: ColorMode): ChartTheme {
  return mode === "dark" ? DARK : LIGHT;
}

/** Shared axis / tooltip styling so every chart reads as one system. */
export function baseOption(t: ChartTheme) {
  return {
    backgroundColor: "transparent",
    animation: false,
    textStyle: { fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif', color: t.inkSecondary },
    tooltip: {
      backgroundColor: t.surface,
      borderColor: t.axis,
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { color: t.ink, fontSize: 12 },
      extraCssText: "box-shadow: 0 12px 32px rgba(0,0,0,0.28); border-radius: 12px;",
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
