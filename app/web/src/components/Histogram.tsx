import type { EChartsOption } from "echarts";
import { useMemo } from "react";

import type { Summary } from "../api";
import type { ColorMode } from "../hooks";
import { axisStyle, baseOption, chartTheme } from "../theme";
import { formatCorr, formatInt, nullCorrCounts } from "../util";
import { Chart } from "./Chart";

export const HISTOGRAM_HEIGHT = 260;

interface HistogramProps {
  summary: Summary;
  window: number;
  mode: ColorMode;
}

/** Distribution of all pairwise rho on one date, with the independent-series reference. */
export function Histogram({ summary, window, mode }: HistogramProps) {
  const option = useMemo<EChartsOption>(() => {
    const t = chartTheme(mode);
    const edges = summary.bin_edges;
    const reference = nullCorrCounts(edges, window, summary.n_pairs);
    // Bars sit on bin centres of a value axis, so ticks line up with real rho values.
    const bars: [number, number][] = [];
    const curve: [number, number][] = [];
    for (let i = 0; i + 1 < edges.length; i++) {
      const centre = (edges[i]! + edges[i + 1]!) / 2;
      bars.push([centre, summary.counts[i] ?? 0]);
      curve.push([centre, reference[i] ?? 0]);
    }
    return {
      ...baseOption(t),
      grid: { left: 52, right: 12, top: 30, bottom: 28 },
      legend: {
        top: 0,
        right: 0,
        itemWidth: 14,
        itemHeight: 8,
        textStyle: { color: t.inkSecondary, fontSize: 11 },
        data: [
          { name: "pairs", icon: "roundRect" },
          { name: `Null benchmark: independent series (n = ${window})`, icon: "path://M0,4 L14,4 L14,6 L0,6 Z" },
        ],
      },
      tooltip: {
        ...baseOption(t).tooltip,
        trigger: "axis",
        axisPointer: { type: "shadow", shadowStyle: { color: t.accentWash } },
        formatter: (params: unknown) => {
          const rows = params as { dataIndex: number }[];
          const i = rows[0]?.dataIndex ?? 0;
          const lo = edges[i]!;
          const hi = edges[i + 1]!;
          const count = summary.counts[i] ?? 0;
          const share = summary.n_pairs > 0 ? (100 * count) / summary.n_pairs : 0;
          return (
            `ρ in [${formatCorr(lo)}, ${formatCorr(hi)}${i === edges.length - 2 ? "]" : ")"}<br/>` +
            `<b>${formatInt(count)}</b> pairs (${share.toFixed(2)}%)<br/>` +
            `<span style="color:${t.muted}">null benchmark: ${formatInt(Math.round(reference[i] ?? 0))}</span>`
          );
        },
      },
      xAxis: {
        type: "value",
        min: edges[0],
        max: edges[edges.length - 1],
        interval: 0.5,
        ...axisStyle(t),
        splitLine: { show: false },
        axisLabel: {
          color: t.muted,
          fontSize: 11,
          formatter: (v: number) => (v === 0 ? "0" : `${v > 0 ? "+" : "\u2212"}${Math.abs(v)}`),
        },
      },
      yAxis: {
        type: "value",
        ...axisStyle(t),
        axisLabel: {
          color: t.muted,
          fontSize: 11,
          formatter: (v: number) => (v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}K` : `${v}`),
        },
      },
      series: [
        {
          name: "pairs",
          type: "bar",
          data: bars,
          barWidth: "80%",
          itemStyle: { color: t.accent, borderRadius: [3, 3, 0, 0] },
          emphasis: { itemStyle: { color: t.accent, opacity: 0.8 } },
        },
        {
          name: `Null benchmark: independent series (n = ${window})`,
          type: "line",
          data: curve,
          symbol: "none",
          smooth: true,
          lineStyle: { width: 2, color: t.inkSecondary },
          itemStyle: { color: t.inkSecondary },
          z: 3,
        },
      ],
    };
  }, [summary, window, mode]);

  return <Chart option={option} height={HISTOGRAM_HEIGHT} label={`Histogram of pairwise correlations on ${summary.date}`} />;
}
