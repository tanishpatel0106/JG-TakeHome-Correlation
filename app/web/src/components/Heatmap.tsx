import type { EChartsOption } from "echarts";
import { useMemo } from "react";

import type { MatrixResult } from "../api";
import type { ColorMode } from "../hooks";
import { baseOption, chartTheme } from "../theme";
import { escapeHtml, formatCorr } from "../util";
import { Chart } from "./Chart";

export const HEATMAP_HEIGHT = 500;

interface HeatmapProps {
  data: MatrixResult;
  mode: ColorMode;
  onPair: (a: string, b: string) => void;
}

/** k x k correlation heatmap on a fixed diverging scale [-1, 1]; undefined cells are hatched gray. */
export function Heatmap({ data, mode, onPair }: HeatmapProps) {
  const option = useMemo<EChartsOption>(() => {
    const t = chartTheme(mode);
    const n = data.tickers.length;
    const values: [number, number, number][] = [];
    const missing: [number, number, number][] = [];
    for (let y = 0; y < n; y++) {
      const row = data.matrix[y]!;
      for (let x = 0; x < n; x++) {
        const v = row[x];
        if (v === null || v === undefined) missing.push([x, y, 0]);
        else values.push([x, y, v]);
      }
    }
    const showAll = n <= 40;
    const axisLabel = { color: t.muted, fontSize: n > 60 ? 9 : 11, interval: showAll ? 0 : ("auto" as const) };
    const gap = n <= 40 ? 1 : 0;
    return {
      ...baseOption(t),
      grid: { left: 58, right: 12, top: 8, bottom: 96 },
      tooltip: {
        ...baseOption(t).tooltip,
        trigger: "item",
        formatter: (params: unknown) => {
          const p = params as { seriesIndex: number; value: [number, number, number] };
          const a = escapeHtml(data.tickers[p.value[1]] ?? "");
          const b = escapeHtml(data.tickers[p.value[0]] ?? "");
          if (p.seriesIndex === 1) {
            return `${a} × ${b}<br/><b>n/a</b><br/><span style="color:${t.muted}">insufficient or constant history in this window</span>`;
          }
          const hint = a !== b ? `<br/><span style="color:${t.muted}">click to chart this pair</span>` : "";
          return `${a} × ${b}<br/>ρ <b>${formatCorr(p.value[2])}</b>${hint}`;
        },
      },
      xAxis: {
        type: "category",
        data: data.tickers,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { ...axisLabel, rotate: 90 },
        splitArea: { show: false },
      },
      yAxis: {
        type: "category",
        data: data.tickers,
        inverse: true,
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel,
      },
      visualMap: [
        {
        type: "continuous",
        min: -1,
        max: 1,
        seriesIndex: 0,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemWidth: 12,
        itemHeight: 220,
        text: ["+1", "−1"],
        textGap: 8,
        textStyle: { color: t.inkSecondary, fontSize: 11 },
        inRange: { color: t.diverging },
        },
        // ECharts colours every heatmap through a visualMap; undefined cells get a constant gray.
        { type: "piecewise", show: false, seriesIndex: 1, pieces: [{ value: 0, color: t.missing }] },
      ],
      series: [
        {
          name: "ρ",
          type: "heatmap",
          data: values,
          progressive: 0,
          itemStyle: { borderColor: t.surface, borderWidth: gap },
          emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 1 } },
        },
        {
          name: "n/a",
          type: "heatmap",
          data: missing,
          progressive: 0,
          itemStyle: {
            color: t.missing,
            borderColor: t.surface,
            borderWidth: gap,
            decal: {
              symbol: "rect",
              symbolSize: 1,
              dashArrayX: [1, 0],
              dashArrayY: [2, 3],
              rotation: Math.PI / 4,
              color: t.missingHatch,
            },
          },
          emphasis: { itemStyle: { borderColor: t.ink, borderWidth: 1 } },
        },
      ],
    };
  }, [data, mode]);

  const events = useMemo(
    () => ({
      click: (params: unknown) => {
        const p = params as { value?: [number, number, number] };
        if (!p.value) return;
        const a = data.tickers[p.value[1]];
        const b = data.tickers[p.value[0]];
        if (a && b && a !== b) onPair(a, b);
      },
    }),
    [data.tickers, onPair],
  );

  return (
    <Chart
      option={option}
      height={HEATMAP_HEIGHT}
      label={`Correlation heatmap of ${data.tickers.length} stocks on ${data.date}`}
      events={events}
    />
  );
}
