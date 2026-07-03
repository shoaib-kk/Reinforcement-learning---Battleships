// Training dashboard: stat tiles + live-updating charts + a table twin so
// every charted value is also readable as text.

import { useMemo } from "react";
import LineChart, { Series } from "./LineChart";
import { C, OPPONENT_COLOR } from "../colors";
import type { EpisodeRecord, TrainStatus } from "../types";

const MAX_POINTS = 800; // stride-downsample beyond this so charts stay fluid

function downsample(records: EpisodeRecord[]): EpisodeRecord[] {
  if (records.length <= MAX_POINTS) return records;
  const stride = Math.ceil(records.length / MAX_POINTS);
  return records.filter((_, i) => i % stride === 0 || i === records.length - 1);
}

function movingAvg(points: { x: number; y: number }[], w: number) {
  const out: { x: number; y: number }[] = [];
  let sum = 0;
  const q: number[] = [];
  for (const p of points) {
    q.push(p.y);
    sum += p.y;
    if (q.length > w) sum -= q.shift()!;
    out.push({ x: p.x, y: sum / q.length });
  }
  return out;
}

export function StatTiles({ status, latest }: { status: TrainStatus | null; latest: EpisodeRecord | null }) {
  const tiles = [
    { label: "Episode", value: latest ? String(latest.episode) : "—" },
    {
      label: "Win rate (last 100)",
      value: latest ? `${Math.round(latest.win_rate * 100)}%` : "—",
      sub: latest ? `vs ${latest.opponent}` : undefined,
    },
    {
      label: "Avg shots to win",
      value: latest?.avg_shots_to_win != null ? latest.avg_shots_to_win.toFixed(1) : "—",
    },
    { label: "Epsilon", value: latest ? latest.epsilon.toFixed(3) : "—" },
    {
      label: "Buffer",
      value: latest ? latest.buffer_size.toLocaleString() : "—",
      sub: status?.running ? "training" : "idle",
    },
  ];
  return (
    <div className="tile-row">
      {tiles.map((t) => (
        <div className="tile" key={t.label}>
          <div className="label">{t.label}</div>
          <div className="value">{t.value}</div>
          {t.sub && <div className="sub">{t.sub}</div>}
        </div>
      ))}
    </div>
  );
}

export default function MetricsDashboard({
  records,
  status,
}: {
  records: EpisodeRecord[];
  status: TrainStatus | null;
}) {
  const data = useMemo(() => downsample(records), [records]);
  const latest = records.length ? records[records.length - 1] : null;

  // Win rate: one series per opponent, hue fixed per opponent (never cycled).
  const winSeries: Series[] = useMemo(() => {
    const byOpp = new Map<string, { x: number; y: number }[]>();
    data.forEach((r, i) => {
      if (!byOpp.has(r.opponent)) byOpp.set(r.opponent, []);
      byOpp.get(r.opponent)!.push({ x: i, y: r.win_rate });
    });
    return [...byOpp.entries()].map(([opp, points]) => ({
      label: `vs ${opp}`,
      color: OPPONENT_COLOR[opp] ?? C.series.red,
      points,
    }));
  }, [data]);

  const single = (
    pick: (r: EpisodeRecord) => number | null | undefined,
    color: string,
    label: string,
    smooth = 0
  ): Series[] => {
    const pts = data
      .map((r, i) => ({ x: i, y: pick(r) }))
      .filter((p): p is { x: number; y: number } => p.y != null);
    return [{ label, color, points: smooth ? movingAvg(pts, smooth) : pts }];
  };

  return (
    <>
      <StatTiles status={status} latest={latest} />
      <div className="chart-grid">
        <div className="chart-card">
          <div className="title">Win rate (rolling 100 episodes)</div>
          <LineChart series={winSeries} yMin={0} yMax={1} xLabel="episode" />
        </div>
        <div className="chart-card">
          <div className="title">Shaped return (moving avg, 25 ep)</div>
          <LineChart
            series={single((r) => r.shaped_return, C.series.blue, "shaped return", 25)}
            xLabel="episode"
          />
        </div>
        <div className="chart-card">
          <div className="title">TD loss (per-episode mean)</div>
          <LineChart series={single((r) => r.loss, C.series.blue, "loss")} xLabel="episode" />
        </div>
        <div className="chart-card">
          <div className="title">Epsilon (exploration rate)</div>
          <LineChart
            series={single((r) => r.epsilon, C.series.blue, "epsilon")}
            yMin={0}
            yMax={1}
            xLabel="episode"
          />
        </div>
        <div className="chart-card">
          <div className="title">Avg shots to win (rolling 100 wins)</div>
          <LineChart
            series={single((r) => r.avg_shots_to_win, C.series.blue, "avg shots")}
            xLabel="episode"
          />
        </div>
        <div className="chart-card">
          <div className="title">Recent episodes</div>
          <RecentTable records={records.slice(-9)} />
        </div>
      </div>
    </>
  );
}

function RecentTable({ records }: { records: EpisodeRecord[] }) {
  return (
    <table className="metrics">
      <thead>
        <tr>
          <th>ep</th>
          <th>opponent</th>
          <th>won</th>
          <th>shots</th>
          <th>return</th>
          <th>loss</th>
          <th>ε</th>
        </tr>
      </thead>
      <tbody>
        {[...records].reverse().map((r, i) => (
          <tr key={`${r.episode}-${i}`}>
            <td>{r.episode}</td>
            <td>{r.opponent}</td>
            <td>{r.won ? "✓" : "✗"}</td>
            <td>{r.shots}</td>
            <td>{r.shaped_return.toFixed(1)}</td>
            <td>{r.loss?.toFixed(4) ?? "—"}</td>
            <td>{r.epsilon.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
