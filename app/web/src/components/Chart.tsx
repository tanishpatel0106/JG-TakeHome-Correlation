import { BarChart, HeatmapChart, LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
  AriaComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/lib/core";
import type { EChartsOption, ECharts } from "echarts";
import { useMemo } from "react";

echarts.use([
  BarChart,
  HeatmapChart,
  LineChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  VisualMapComponent,
  AriaComponent,
  CanvasRenderer,
]);

export type ChartEvents = Record<string, (params: unknown, chart: ECharts) => void>;

interface ChartProps {
  option: EChartsOption;
  height: number;
  label: string;
  events?: ChartEvents;
  onReady?: (chart: ECharts) => void;
}

/** ECharts canvas with a fixed height (no layout shift); auto-resizes with its container. */
export function Chart({ option, height, label, events, onReady }: ChartProps) {
  const style = useMemo(() => ({ height, width: "100%" }), [height]);
  return (
    <div role="img" aria-label={label} className="chart">
      <ReactEChartsCore
        echarts={echarts}
        option={option}
        style={style}
        notMerge
        lazyUpdate
        onEvents={events}
        onChartReady={onReady}
        opts={{ renderer: "canvas" }}
      />
    </div>
  );
}
