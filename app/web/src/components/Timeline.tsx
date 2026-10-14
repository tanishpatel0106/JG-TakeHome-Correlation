import type { ECharts, EChartsOption } from "echarts";
import { useEffect, useMemo, useRef } from "react";

import type { Breaks, Timeline as TimelineData } from "../api";
import type { ColorMode } from "../hooks";
import { axisStyle, baseOption, chartTheme } from "../theme";
import { escapeHtml, formatCorr, formatDate, formatInt, formatShortDate } from "../util";
import { Chart } from "./Chart";

const HEIGHT = 230;

interface TimelineProps {
  data: TimelineData;
  breaks: Breaks;
  selected: string;
  mode: ColorMode;
  onSelect: (date: string) => void;
}

/** Mean pairwise rho per date with a +/- 1 std band; click a date to select it. */
export function Timeline({ data, breaks, selected, mode, onSelect }: TimelineProps) {
  // The zrender click handler is bound once; it reads the latest props from this ref.
  const latest = useRef({ dates: data.dates, onSelect });
  const chartRef = useRef<ECharts | null>(null);
  useEffect(() => {
    latest.current = { dates: data.dates, onSelect };
  }, [data.dates, onSelect]);

  useEffect(() => {
    return () => {
      chartRef.current?.getZr().off("click");
    };
  }, []);

  const option = useMemo<EChartsOption>(() => {
    const t = chartTheme(mode);
    const lower: (number | null)[] = [];
    const band: (number | null)[] = [];
    for (let i = 0; i < data.dates.length; i++) {
      const m = data.mean_corr[i];
      const s = data.std_corr[i];
      const ok = m !== null && m !== undefined && s !== null && s !== undefined;
      lower.push(ok ? m - s : null);
      band.push(ok ? 2 * s : null);
    }
    const breakLines = [];
    for (let i = 0; i < breaks.rebase_dates.length; i++) {
      // A re-base on the very first date breaks nothing (R[0] is undefined anyway).
      if (breaks.rebase_dates[i] === data.dates[0]) continue;
      breakLines.push({ xAxis: breaks.rebase_dates[i], label: { formatter: "re-base" } });
    }
    for (let i = 0; i < breaks.placeholder_dates.length; i++) {
      breakLines.push({ xAxis: breaks.placeholder_dates[i], label: { formatter: "placeholder" } });
    }
    return {
      ...baseOption(t),
      grid: { left: 48, right: 56, top: 24, bottom: 56 },
      tooltip: {
        ...baseOption(t).tooltip,
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: t.muted, width: 1 } },
        formatter: (params: unknown) => {
          const first = (params as { dataIndex: number }[])[0];
          if (!first) return "";
          const i = first.dataIndex;
          const date = data.dates[i]!;
          const nValid = data.n_valid[i] ?? 0;
          if (nValid < 2) {
            return `<b>${formatDate(date)}</b><br/>No valid window (${formatInt(nValid)} valid stocks)`;
          }
          return (
            `<b>${escapeHtml(formatDate(date))}</b><br/>` +
            `mean ρ <b>${formatCorr(data.mean_corr[i])}</b><br/>` +
            `std ρ ${data.std_corr[i]?.toFixed(3) ?? "n/a"}<br/>` +
            `${formatInt(nValid)} valid stocks<br/><span style="color:${t.muted}">click to select</span>`
          );
        },
      },
      xAxis: {
        type: "category",
        data: data.dates,
        boundaryGap: false,
        ...axisStyle(t),
        splitLine: { show: false },
        axisLabel: { color: t.muted, fontSize: 11, hideOverlap: true },
      },
      yAxis: {
        type: "value",
        ...axisStyle(t),
        axisLabel: { color: t.muted, fontSize: 11, formatter: (v: number) => v.toFixed(2) },
      },
      dataZoom: [
        { type: "inside", filterMode: "none" },
        {
          type: "slider",
          height: 18,
          bottom: 8,
          filterMode: "none",
          borderColor: t.grid,
          fillerColor: t.accentWash,
          handleStyle: { color: t.surface, borderColor: t.muted },
          textStyle: { color: t.muted },
          dataBackground: { lineStyle: { color: t.axis }, areaStyle: { color: t.grid } },
        },
      ],
      series: [
        {
          name: "lower",
          type: "line",
          data: lower,
          stack: "band",
          stackStrategy: "all", // lower bound is negative, band height positive
          symbol: "none",
          lineStyle: { opacity: 0 },
          silent: true,
        },
        {
          name: "±1 std",
          type: "line",
          data: band,
          stack: "band",
          stackStrategy: "all",
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: t.accentWash },
          silent: true,
        },
        {
          name: "mean ρ",
          type: "line",
          data: data.mean_corr,
          symbol: "none",
          lineStyle: { width: 2, color: t.accent, cap: "round", join: "round" },
          itemStyle: { color: t.accent },
          markLine: {
            silent: true,
            symbol: "none",
            animation: false,
            data: [
              ...breakLines.map((b) => ({
                ...b,
                lineStyle: { color: t.axis, width: 1, type: "solid" as const },
                label: { ...b.label, color: t.muted, fontSize: 10, position: "insideEndTop" as const },
              })),
              {
                xAxis: selected,
                lineStyle: { color: t.ink, width: 1.5, type: "solid" as const },
                label: { formatter: formatShortDate(selected), color: t.ink, fontSize: 11, position: "end" as const },
              },
            ],
          },
        },
      ],
    };
  }, [data, breaks, selected, mode]);

  function onReady(chart: ECharts) {
    chartRef.current = chart;
    chart.getZr().off("click");
    chart.getZr().on("click", (event: { offsetX: number; offsetY: number }) => {
      const point = [event.offsetX, event.offsetY];
      if (!chart.containPixel({ gridIndex: 0 }, point)) return;
      const index = chart.convertFromPixel({ xAxisIndex: 0 }, point[0]!);
      const i = Math.round(Array.isArray(index) ? (index[0] as number) : (index as number));
      const { dates, onSelect: select } = latest.current;
      const date = dates[Math.max(0, Math.min(dates.length - 1, i))];
      if (date) select(date);
    });
  }

  return <Chart option={option} height={HEIGHT} label="Mean pairwise correlation over time" onReady={onReady} />;
}
