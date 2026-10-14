import type { ECharts, EChartsOption } from "echarts";
import { useEffect, useMemo, useRef } from "react";

import type { PairSeries } from "../api";
import type { ColorMode } from "../hooks";
import { axisStyle, baseOption, chartTheme } from "../theme";
import { escapeHtml, formatCorr, formatDate, formatShortDate } from "../util";
import { Chart } from "./Chart";

export const PAIR_HEIGHT = 240;

interface PairChartProps {
  data: PairSeries;
  selected: string;
  mode: ColorMode;
  onSelectDate: (date: string) => void;
}

/** rho(a, b) over time on a fixed [-1, 1] axis, with the selected date marked. */
export function PairChart({ data, selected, mode, onSelectDate }: PairChartProps) {
  // The zrender click handler is bound once; it reads the latest props from this ref.
  const latest = useRef({ dates: data.dates, onSelect: onSelectDate });
  useEffect(() => {
    latest.current = { dates: data.dates, onSelect: onSelectDate };
  }, [data.dates, onSelectDate]);

  const option = useMemo<EChartsOption>(() => {
    const t = chartTheme(mode);
    const name = `ρ(${data.a}, ${data.b})`;
    return {
      ...baseOption(t),
      grid: { left: 44, right: 56, top: 24, bottom: 56 },
      tooltip: {
        ...baseOption(t).tooltip,
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: t.muted, width: 1 } },
        formatter: (params: unknown) => {
          const first = (params as { dataIndex: number }[])[0];
          if (!first) return "";
          const i = first.dataIndex;
          const value = data.corr[i];
          return (
            `<b>${escapeHtml(formatDate(data.dates[i]!))}</b><br/>` +
            `${escapeHtml(name)} <b>${formatCorr(value)}</b>` +
            (value === null || value === undefined
              ? `<br/><span style="color:${t.muted}">no complete ${data.window}-return window</span>`
              : "")
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
        min: -1,
        max: 1,
        interval: 0.5,
        ...axisStyle(t),
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
          name,
          type: "line",
          data: data.corr,
          symbol: "none",
          connectNulls: false,
          lineStyle: { width: 2, color: t.accent, cap: "round", join: "round" },
          itemStyle: { color: t.accent },
          markLine: {
            silent: true,
            symbol: "none",
            animation: false,
            data: [
              { yAxis: 0, lineStyle: { color: t.axis, width: 1, type: "solid" as const }, label: { show: false } },
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
  }, [data, selected, mode]);

  function onReady(chart: ECharts) {
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

  return (
    <Chart option={option} height={PAIR_HEIGHT} label={`Correlation of ${data.a} and ${data.b} over time`} onReady={onReady} />
  );
}
