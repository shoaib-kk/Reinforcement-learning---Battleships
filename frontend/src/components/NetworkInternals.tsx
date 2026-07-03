// The "network thinking" panel: architecture diagram, live per-layer
// activations (conv feature maps as heatmap tiles, dense layers as strip
// charts), weight/gradient histograms, and a 2D PCA of replay-buffer state
// embeddings colored by episode outcome.

import { useEffect, useRef, useState } from "react";
import { C } from "../colors";
import { api } from "../api";
import type {
  Activations,
  ArchLayer,
  ConvActivation,
  DenseActivation,
  EmbeddingData,
  LayerHistograms,
} from "../types";

// ---------------------------------------------------------------------------
// Architecture diagram (auto-generated from /api/model/architecture)
// ---------------------------------------------------------------------------

export function NetworkDiagram() {
  const [layers, setLayers] = useState<ArchLayer[]>([]);
  useEffect(() => {
    api.architecture().then((r) => setLayers(r.layers)).catch(() => {});
  }, []);
  if (!layers.length) return <div className="hint">loading architecture…</div>;
  const total = layers.reduce((s, l) => s + l.params, 0);
  return (
    <div>
      <div className="arch-flow">
        {layers.map((l, i) => (
          <span key={l.name} style={{ display: "contents" }}>
            {i > 0 && <span className="arch-arrow">→</span>}
            <div className="arch-box">
              <div className="name">{l.name}</div>
              <div className="shape">{l.out_shape.join("×")}</div>
              {l.params > 0 && <div className="params">{l.params.toLocaleString()} params</div>}
            </div>
          </span>
        ))}
      </div>
      <div className="hint" style={{ marginTop: 6 }}>
        {total.toLocaleString()} trainable parameters · ReLU between layers · output = one
        Q-value per board cell
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/** All feature maps of one conv layer drawn into a single canvas grid,
 *  normalized by the layer's max so maps are comparable within the layer. */
function ConvMaps({ data }: { data: ConvActivation }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const tile = 40;
  const gap = 4;
  const cols = 8;
  const rows = Math.ceil(data.maps.length / cols);
  const w = cols * (tile + gap) - gap;
  const h = rows * (tile + gap) - gap;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    cv.width = w;
    cv.height = h;
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, w, h);
    let peak = 0;
    for (const m of data.maps) for (const row of m) for (const v of row) peak = Math.max(peak, v);
    if (peak === 0) peak = 1;
    const px = tile / 10;
    data.maps.forEach((m, i) => {
      const ox = (i % cols) * (tile + gap);
      const oy = Math.floor(i / cols) * (tile + gap);
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const t = m[r][c] / peak;
          ctx.fillStyle = `rgba(57, 135, 229, ${(0.06 + 0.9 * t).toFixed(3)})`;
          ctx.fillRect(ox + c * px, oy + r * px, Math.ceil(px), Math.ceil(px));
        }
      }
    });
  }, [data, w, h]);

  return <canvas ref={ref} className="pixelated" style={{ width: w, height: h, maxWidth: "100%" }} />;
}

/** Dense layer as a magnitude strip: one thin bar per unit. */
function DenseStrip({ data }: { data: DenseActivation }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const w = 512;
  const h = 46;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    cv.width = w;
    cv.height = h;
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(0, h - 0.5);
    ctx.lineTo(w, h - 0.5);
    ctx.stroke();
    const vals = data.values;
    const peak = Math.max(...vals.map((v) => Math.abs(v))) || 1;
    const bw = w / vals.length;
    ctx.fillStyle = C.series.blue;
    vals.forEach((v, i) => {
      const bh = (Math.abs(v) / peak) * (h - 4);
      ctx.fillRect(i * bw, h - bh, Math.max(bw - 0.4, 0.6), bh);
    });
  }, [data]);

  return <canvas ref={ref} style={{ width: "100%", maxWidth: w, height: h }} />;
}

export function ActivationsView({ activations }: { activations: Activations | null }) {
  if (!activations)
    return <div className="hint">No activations yet — start training or step a demo game.</div>;
  return (
    <div>
      {Object.entries(activations).map(([name, act]) => (
        <div className="layer-block" key={name}>
          <div className="layer-title">
            {name}{" "}
            <span className="dim">
              {act.kind === "conv"
                ? `· first ${act.maps.length} of ${act.n_total} feature maps`
                : `· ${act.values.length} units, |activation|`}
            </span>
          </div>
          {act.kind === "conv" ? <ConvMaps data={act} /> : <DenseStrip data={act} />}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Weight / gradient histograms
// ---------------------------------------------------------------------------

function Histogram({ data, color }: { data: { counts: number[]; edges: number[] }; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const w = 210;
  const h = 64;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    cv.width = w;
    cv.height = h;
    ctx.fillStyle = C.surface;
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = C.axis;
    ctx.beginPath();
    ctx.moveTo(0, h - 0.5);
    ctx.lineTo(w, h - 0.5);
    ctx.stroke();
    const peak = Math.max(...data.counts) || 1;
    const n = data.counts.length;
    const bw = w / n;
    ctx.fillStyle = color;
    data.counts.forEach((cnt, i) => {
      const bh = (cnt / peak) * (h - 6);
      // 2px surface gap between adjacent bars
      ctx.fillRect(i * bw + 1, h - 1 - bh, Math.max(bw - 2, 1), bh);
    });
  }, [data, color]);

  const fmt = (v: number) => (v === 0 ? "0" : v.toPrecision(2));
  const lo = data.edges[0];
  const hi = data.edges[data.edges.length - 1];
  return (
    <div>
      <canvas ref={ref} style={{ width: w, height: h }} />
      <div className="hist-range" style={{ width: w }}>
        <span>{fmt(lo)}</span>
        <span>{fmt(hi)}</span>
      </div>
    </div>
  );
}

export function WeightHists({
  layers,
  onRefresh,
}: {
  layers: LayerHistograms[] | null;
  onRefresh: () => void;
}) {
  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="hint">
          Sampled during training (every N episodes) — weights in blue, latest gradients in aqua.
        </span>
        <button onClick={onRefresh}>Refresh now</button>
      </div>
      {!layers?.length ? (
        <div className="hint">No histograms yet.</div>
      ) : (
        <div className="hist-grid">
          {layers.map((l) => (
            <div className="hist-cell" key={l.layer}>
              <div className="layer-title">{l.layer} · weights</div>
              <Histogram data={l.weights} color={C.series.blue} />
              {l.grads && (
                <>
                  <div className="layer-title" style={{ marginTop: 6 }}>
                    {l.layer} · grads
                  </div>
                  <Histogram data={l.grads} color={C.series.aqua} />
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Embedding projection (PCA of replay-buffer states)
// ---------------------------------------------------------------------------

const OUTCOME_META = [
  { label: "in progress", color: C.muted },
  { label: "won episode", color: C.status.good },
  { label: "lost episode", color: C.status.critical },
];

export function EmbeddingView() {
  const ref = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<EmbeddingData | null>(null);
  const [busy, setBusy] = useState(false);
  const size = 340;

  const refresh = async () => {
    setBusy(true);
    try {
      setData(await api.embeddings(500));
    } catch {
      /* trainer may not exist yet */
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(size / 2 + 0.5, 0);
    ctx.lineTo(size / 2 + 0.5, size);
    ctx.moveTo(0, size / 2 + 0.5);
    ctx.lineTo(size, size / 2 + 0.5);
    ctx.stroke();
    if (!data) return;
    for (const [x, y, outcome] of data.points) {
      const px = size / 2 + x * (size / 2 - 14);
      const py = size / 2 - y * (size / 2 - 14);
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = OUTCOME_META[outcome]?.color ?? C.muted;
      ctx.strokeStyle = C.surface; // 2px surface ring on overlapping dots
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fill();
    }
  }, [data]);

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <span className="hint">
          PCA of fc1 embeddings for {data?.n ?? 0} replay-buffer states
          {data && data.explained.length === 2 && (
            <> · PC1 {Math.round(data.explained[0] * 100)}% / PC2 {Math.round(data.explained[1] * 100)}% var</>
          )}
        </span>
        <button onClick={refresh} disabled={busy}>
          {busy ? "…" : "Refresh"}
        </button>
      </div>
      <canvas ref={ref} style={{ width: size, height: size }} />
      <div className="legend-row">
        {OUTCOME_META.map((o) => (
          <span className="legend-key" key={o.label}>
            <span className="legend-dot" style={{ background: o.color }} />
            {o.label}
          </span>
        ))}
      </div>
    </div>
  );
}
