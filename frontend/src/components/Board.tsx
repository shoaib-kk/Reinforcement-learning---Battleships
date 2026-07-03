// The 10x10 attack board on canvas: hits / misses / sunk ships, the cell the
// agent just targeted, and an optional heatmap overlay (Q-values in blue,
// saliency in aqua). Hovering any cell shows its coordinate and overlay value.

import { useEffect, useRef, useState } from "react";
import { C, heatAqua, heatBlue } from "../colors";

const N = 10;
const CELL = 34;
const PAD = 18; // row/col label gutter
const SIZE = PAD + N * CELL + 2;

export type OverlayMode = "none" | "q" | "saliency";

interface Props {
  view: number[][]; // 0 unknown 1 hit 2 miss 3 sunk
  qValues?: (number | null)[];
  saliency?: number[][];
  lastAction?: number;
  overlay: OverlayMode;
}

function normalize(qs: (number | null)[]): (number | null)[] {
  const vals = qs.filter((v): v is number => v !== null);
  if (!vals.length) return qs;
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  return qs.map((v) => (v === null ? null : (v - lo) / span));
}

export default function Board({ view, qValues, saliency, lastAction, overlay }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = SIZE * dpr;
    cv.height = SIZE * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, SIZE, SIZE);

    // gutter labels
    ctx.fillStyle = C.muted;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let i = 0; i < N; i++) {
      ctx.fillText("ABCDEFGHIJ"[i], PAD + i * CELL + CELL / 2, PAD / 2 + 2);
      ctx.fillText(String(i + 1), PAD / 2, PAD + i * CELL + CELL / 2);
    }

    // heatmap overlay under the state marks
    const qNorm = overlay === "q" && qValues ? normalize(qValues) : null;
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const x = PAD + c * CELL;
        const y = PAD + r * CELL;
        if (qNorm) {
          const t = qNorm[r * N + c];
          if (t !== null) {
            ctx.fillStyle = heatBlue(t);
            ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
          }
        } else if (overlay === "saliency" && saliency) {
          ctx.fillStyle = heatAqua(saliency[r][c]);
          ctx.fillRect(x + 1, y + 1, CELL - 2, CELL - 2);
        }
      }
    }

    // hairline grid
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= N; i++) {
      ctx.beginPath();
      ctx.moveTo(PAD + i * CELL + 0.5, PAD);
      ctx.lineTo(PAD + i * CELL + 0.5, PAD + N * CELL);
      ctx.moveTo(PAD, PAD + i * CELL + 0.5);
      ctx.lineTo(PAD + N * CELL, PAD + i * CELL + 0.5);
      ctx.stroke();
    }

    // shot results
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const x = PAD + c * CELL + CELL / 2;
        const y = PAD + r * CELL + CELL / 2;
        const s = view[r][c];
        if (s === 2) {
          // miss: muted dot with a surface ring so it survives the overlay
          ctx.beginPath();
          ctx.arc(x, y, 3.5, 0, Math.PI * 2);
          ctx.fillStyle = C.muted;
          ctx.strokeStyle = C.surface;
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.fill();
        } else if (s === 1 || s === 3) {
          if (s === 3) {
            // sunk: filled cell behind the X
            ctx.fillStyle = "rgba(208, 59, 59, 0.35)";
            ctx.fillRect(PAD + c * CELL + 2, PAD + r * CELL + 2, CELL - 4, CELL - 4);
          }
          ctx.strokeStyle = C.status.critical;
          ctx.lineWidth = 2.5;
          ctx.lineCap = "round";
          const d = 6;
          ctx.beginPath();
          ctx.moveTo(x - d, y - d);
          ctx.lineTo(x + d, y + d);
          ctx.moveTo(x + d, y - d);
          ctx.lineTo(x - d, y + d);
          ctx.stroke();
        }
      }
    }

    // last targeted cell
    if (lastAction !== undefined && lastAction >= 0) {
      const r = Math.floor(lastAction / N);
      const c = lastAction % N;
      ctx.strokeStyle = C.inkPrimary;
      ctx.lineWidth = 2;
      ctx.strokeRect(PAD + c * CELL + 1.5, PAD + r * CELL + 1.5, CELL - 3, CELL - 3);
    }
  }, [view, qValues, saliency, lastAction, overlay]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const c = Math.floor((px - PAD) / CELL);
    const r = Math.floor((py - PAD) / CELL);
    if (r < 0 || r >= N || c < 0 || c >= N) {
      setTip(null);
      return;
    }
    const coord = `${"ABCDEFGHIJ"[c]}${r + 1}`;
    let text = coord;
    if (overlay === "q" && qValues) {
      const q = qValues[r * N + c];
      text += q === null ? " — already fired" : ` · Q = ${q.toFixed(3)}`;
    } else if (overlay === "saliency" && saliency) {
      text += ` · saliency = ${saliency[r][c].toFixed(3)}`;
    }
    setTip({ x: px + 12, y: py - 8, text });
  };

  return (
    <div className="board-wrap">
      <canvas
        ref={ref}
        style={{ width: SIZE, height: SIZE }}
        onMouseMove={onMove}
        onMouseLeave={() => setTip(null)}
      />
      {tip && (
        <div className="board-tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.text}
        </div>
      )}
    </div>
  );
}

/** The agent's own fleet (being shot at by the opponent), smaller scale. */
export function OwnBoard({ grid }: { grid: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const cell = 18;
  const size = N * cell + 2;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size * dpr;
    cv.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, size, size);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const s = grid[r][c];
        const x = c * cell;
        const y = r * cell;
        if (s === 1) ctx.fillStyle = C.axis; // intact ship
        else if (s === 2) ctx.fillStyle = C.status.serious; // hit
        else if (s === 4) ctx.fillStyle = C.status.critical; // sunk
        else ctx.fillStyle = C.surface;
        ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);
        if (s === 3) {
          ctx.beginPath();
          ctx.arc(x + cell / 2, y + cell / 2, 2, 0, Math.PI * 2);
          ctx.fillStyle = C.muted;
          ctx.fill();
        }
      }
    }
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= N; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cell + 0.5, 0);
      ctx.lineTo(i * cell + 0.5, N * cell);
      ctx.moveTo(0, i * cell + 0.5);
      ctx.lineTo(N * cell, i * cell + 0.5);
      ctx.stroke();
    }
  }, [grid, size]);

  return <canvas ref={ref} style={{ width: size, height: size }} />;
}
