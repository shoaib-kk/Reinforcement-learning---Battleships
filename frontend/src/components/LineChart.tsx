// Thin Chart.js wrapper styled to the design spec: 2px lines, hairline solid
// gridlines, recessive axes, index-mode tooltip (the hover layer), legend
// only when there are two or more series.

import { useEffect, useRef } from "react";
import {
  Chart,
  LineController,
  LineElement,
  PointElement,
  LinearScale,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { C } from "../colors";

Chart.register(LineController, LineElement, PointElement, LinearScale, Tooltip, Legend, Filler);

Chart.defaults.color = C.muted;
Chart.defaults.borderColor = C.grid;
Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", sans-serif';
Chart.defaults.font.size = 11;

export interface Series {
  label: string;
  color: string;
  points: { x: number; y: number }[];
}

interface Props {
  series: Series[];
  yMin?: number;
  yMax?: number;
  xLabel?: string;
}

export default function LineChart({ series, yMin, yMax, xLabel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<Chart<"line"> | null>(null);

  useEffect(() => {
    const chart = new Chart(canvasRef.current!, {
      type: "line",
      data: { datasets: [] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        normalized: true,
        parsing: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        plugins: {
          legend: {
            display: false,
            labels: { boxWidth: 8, boxHeight: 8, usePointStyle: true, color: C.inkSecondary },
          },
          tooltip: {
            backgroundColor: C.surfaceRaised,
            titleColor: C.inkPrimary,
            bodyColor: C.inkSecondary,
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1,
            padding: 8,
            displayColors: true,
            boxWidth: 8,
            boxHeight: 8,
            usePointStyle: true,
          },
        },
        scales: {
          x: {
            type: "linear",
            grid: { color: C.grid, lineWidth: 1 },
            border: { color: C.axis },
            ticks: { maxTicksLimit: 6, color: C.muted },
            title: xLabel
              ? { display: true, text: xLabel, color: C.muted, font: { size: 10 } }
              : undefined,
          },
          y: {
            grid: { color: C.grid, lineWidth: 1 },
            border: { color: C.axis },
            ticks: { maxTicksLimit: 5, color: C.muted },
            min: yMin,
            max: yMax,
          },
        },
        elements: {
          line: { borderWidth: 2, tension: 0, borderJoinStyle: "round", borderCapStyle: "round" },
          point: { radius: 0, hoverRadius: 4, hitRadius: 12 },
        },
      },
    });
    chartRef.current = chart;
    return () => chart.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.data.datasets = series.map((s) => ({
      label: s.label,
      data: s.points,
      borderColor: s.color,
      backgroundColor: s.color,
      pointBackgroundColor: s.color,
      // 2px surface ring keeps hover markers legible where lines cross
      pointBorderColor: C.surface,
      pointBorderWidth: 2,
      fill: false,
    }));
    // Legend only for >= 2 series; a single series is named by the card title.
    chart.options.plugins!.legend!.display = series.length >= 2;
    chart.update("none");
  }, [series]);

  return (
    <div className="chart-box">
      <canvas ref={canvasRef} />
    </div>
  );
}
