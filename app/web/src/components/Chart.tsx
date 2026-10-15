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
import { useCallback, useEffect, useMemo, useRef } from "react";

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
  /** Click anywhere on the canvas (not only on a mark), in canvas pixels. */
  onCanvasClick?: (chart: ECharts, x: number, y: number) => void;
}

interface ZrClick {
  chart: ECharts;
  zr: ReturnType<ECharts["getZr"]>;
  handler: (event: { offsetX: number; offsetY: number }) => void;
}

/** ECharts canvas with a fixed height (no layout shift); auto-resizes with its container. */
export function Chart({ option, height, label, events, onCanvasClick }: ChartProps) {
  const style = useMemo(() => ({ height, width: "100%" }), [height]);
  // The zrender listener is registered once per chart instance and reads the latest callback from a ref.
  const clickRef = useRef(onCanvasClick);
  useEffect(() => {
    clickRef.current = onCanvasClick;
  }, [onCanvasClick]);
  const bound = useRef<ZrClick | null>(null);
  // Remove only our own listener, and only while the chart is alive: echarts-for-react disposes the
  // instance (and its zrender, whose getZr() then returns null) before this component's effects clean up.
  const unbind = useCallback(() => {
    const b = bound.current;
    bound.current = null;
    if (b && !b.chart.isDisposed()) b.zr.off("click", b.handler);
  }, []);
  useEffect(() => unbind, [unbind]);
  const onReady = useCallback(
    (chart: ECharts) => {
      unbind();
      const zr = chart.getZr();
      if (!zr) return;
      const handler = (event: { offsetX: number; offsetY: number }) => {
        if (!chart.isDisposed()) clickRef.current?.(chart, event.offsetX, event.offsetY);
      };
      zr.on("click", handler);
      bound.current = { chart, zr, handler };
    },
    [unbind],
  );
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

/** The category (x axis) index under a canvas click, or null outside the plot area. */
export function categoryAt(chart: ECharts, x: number, y: number, count: number): number | null {
  if (count === 0 || !chart.containPixel({ gridIndex: 0 }, [x, y])) return null;
  const index = chart.convertFromPixel({ xAxisIndex: 0 }, x);
  const i = Math.round(Array.isArray(index) ? (index[0] as number) : (index as number));
  return Number.isFinite(i) ? Math.max(0, Math.min(count - 1, i)) : null;
}
