// 3D network readout panel. Consumes the same step payloads that drive the
// 2D board (so board / Q-heatmap / 3D scene stay in lockstep per step) plus
// the one-time static payload (topology + head weights) from REST.

import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { StepPayload } from "../types";
import { DEFAULT_OPTIONS, NetScene, SceneOptions } from "./scene";

interface Props {
  payload: StepPayload | null; // latest train_step or demo_step
  mode: "train" | "demo";
  staticKey: string; // changes when the underlying model may have changed
}

function cellName(action: number | undefined): string {
  if (action === undefined || action < 0) return "—";
  return `${"ABCDEFGHIJ"[action % 10]}${Math.floor(action / 10) + 1}`;
}

const OPTS_KEY = "battleship.viz3d.options.v2";

function loadOptions(): SceneOptions {
  try {
    const raw = localStorage.getItem(OPTS_KEY);
    if (raw) return { ...DEFAULT_OPTIONS, ...(JSON.parse(raw) as Partial<SceneOptions>) };
  } catch {
    /* corrupted or unavailable storage -> defaults */
  }
  return { ...DEFAULT_OPTIONS };
}

export default function NetworkViz3D({ payload, mode, staticKey }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<NetScene | null>(null);
  const skipCounter = useRef(0);
  const [opts, setOpts] = useState<SceneOptions>(loadOptions);
  const [clientEvery, setClientEvery] = useState(1);
  const [backendEvery, setBackendEvery] = useState(1);
  const [lastBound, setLastBound] = useState<string>("");
  const [edgeCount, setEdgeCount] = useState(0);

  // scene lifecycle
  useEffect(() => {
    const scene = new NetScene(mountRef.current!);
    scene.onStats = setEdgeCount;
    sceneRef.current = scene;
    return () => {
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // the tuned panel settings survive a reload
  useEffect(() => {
    try {
      localStorage.setItem(OPTS_KEY, JSON.stringify(opts));
    } catch {
      /* private mode / storage full — settings just won't persist */
    }
  }, [opts]);

  // static payload (topology + head weights): once per model change
  useEffect(() => {
    api
      .viz3dStatic(mode)
      .then((st) => sceneRef.current?.setStatic(st))
      .catch(() => {});
  }, [mode, staticKey]);

  // per-step data binding, with a client-side throttle
  useEffect(() => {
    if (!payload?.viz3d || !sceneRef.current) return;
    skipCounter.current += 1;
    if (skipCounter.current % clientEvery !== 0) return;
    sceneRef.current.setStep(payload.viz3d, payload.action);
    const what = payload.type === "demo_step" ? "demo" : "training";
    setLastBound(
      `${what} step ${payload.step} — fired at ${cellName(payload.action)} (${payload.result})` +
        (payload.viz3d.grads ? " · gradients available" : "")
    );
  }, [payload, clientEvery]);

  // option changes -> re-filter with the already-bound data
  useEffect(() => {
    sceneRef.current?.setOptions(opts);
  }, [opts]);

  const gradsAvailable = !!payload?.viz3d?.grads;
  const set = (patch: Partial<SceneOptions>) => setOpts((o) => ({ ...o, ...patch }));

  return (
    <div className="card">
      <h2>3D network — live readout</h2>
      <div className="row viz3d-controls">
        <div className="mode-toggle">
          <button
            className={opts.mode === "inference" ? "active" : ""}
            onClick={() => set({ mode: "inference" })}
          >
            Inference (forward)
          </button>
          <button
            className={opts.mode === "gradients" ? "active" : ""}
            disabled={!gradsAvailable}
            title={gradsAvailable ? "" : "gradients only exist on training steps"}
            onClick={() => set({ mode: "gradients" })}
          >
            Gradients (backward)
          </button>
        </div>
        <label style={{ minWidth: 0 }}>focus</label>
        <select
          className="focus-select"
          value={opts.focus}
          onChange={(e) => set({ focus: e.target.value })}
          style={{ width: 90 }}
        >
          {["all", "input", "conv1", "conv2", "conv3", "dense"].map((f) => (
            <option key={f} value={f}>{f}</option>
          ))}
        </select>
      </div>
      <div className="row viz3d-controls">
        <label style={{ minWidth: 0 }}>top-k / neuron</label>
        <input
          type="range" min={1} max={12} value={opts.k} disabled={opts.showAll}
          onChange={(e) => set({ k: Number(e.target.value) })} style={{ width: 90 }}
        />
        <span className="hint">{opts.showAll ? "—" : opts.k}</span>
        <label style={{ minWidth: 0 }}>edge budget</label>
        <input
          type="range" min={32} max={1024} step={32} value={opts.topN} disabled={opts.showAll}
          onChange={(e) => set({ topN: Number(e.target.value) })} style={{ width: 90 }}
        />
        <span className="hint">{opts.showAll ? "—" : opts.topN}</span>
        <span className="hint">· {edgeCount} shown</span>
        <label className="hint">
          <input
            type="checkbox" checked={opts.showAll}
            onChange={(e) => set({ showAll: e.target.checked })}
          />{" "}
          all connections
        </label>
        <label className="hint">
          <input
            type="checkbox" checked={opts.weightWireframe}
            onChange={(e) => set({ weightWireframe: e.target.checked })}
          />{" "}
          weight wireframe
        </label>
      </div>
      <div className="row viz3d-controls">
        <label style={{ minWidth: 0 }}>apply every</label>
        <select
          value={clientEvery}
          onChange={(e) => setClientEvery(Number(e.target.value))}
          style={{ width: 110 }}
        >
          {[1, 2, 5, 10].map((n) => (
            <option key={n} value={n}>{n === 1 ? "every update" : `${n}th update`}</option>
          ))}
        </select>
        <label style={{ minWidth: 0 }}>stream 3D every</label>
        <select
          value={backendEvery}
          onChange={(e) => {
            const n = Number(e.target.value);
            setBackendEvery(n);
            api.streamConfig({ viz3d_every: n }).catch(() => {});
          }}
          style={{ width: 120 }}
        >
          {[1, 2, 5, 10].map((n) => (
            <option key={n} value={n}>{n === 1 ? "every message" : `${n}th message`}</option>
          ))}
        </select>
        <span className="hint">(backend throttle, training stream only)</span>
      </div>

      <div ref={mountRef} className="viz3d-canvas" />

      <div className="hint" style={{ marginTop: 8 }}>
        {lastBound || "Waiting for a step — start training or step a demo game."}
      </div>
      <div className="legend-row" style={{ marginTop: 6 }}>
        <span className="legend-key"><span className="legend-swatch" style={{ background: "#ff7a29" }} /> positive (fired)</span>
        <span className="legend-key"><span className="legend-swatch" style={{ background: "#3987e5" }} /> negative (ReLU-suppressed / −Q / −w)</span>
        <span className="legend-key"><span className="legend-swatch" style={{ background: "#212126" }} /> ≈ 0 (didn't fire — stays dark)</span>
        <span className="legend-key"><span className="legend-swatch" style={{ border: "2px solid #fff", background: "transparent" }} /> chosen action</span>
      </div>
      <div className="hint" style={{ marginTop: 4 }}>
        Node size/brightness = |activation| of this pass · line width &amp; particle size/speed =
        |activation×weight| contribution (gradient mode: |∂L/∂W|, reversed direction) · hover any
        node, cell or connection for its exact value · click a Q node to isolate its inputs ·
        drag to orbit, wheel to zoom, right-drag to pan.
      </div>

      <div className="viz3d-explain">
        <div className="viz3d-explain-title">How to read this network (left → right)</div>
        <div>
          <b>input</b> — what the agent knows: one 10×10 plane per channel — the cells it
          hasn&apos;t fired at yet, its hits, and its misses.
        </div>
        <div>
          <b>conv1 → conv3</b> — stacks of 3×3 convolution maps. Each small square is one learned
          detector, lighting up where its pattern (say, a line of hits) appears on the board.
          Orange = fired; blue = negative pre-ReLU, so ReLU zeroes it and it contributes nothing
          downstream.
        </div>
        <div>
          <b>fc1</b> — conv3&apos;s 64 maps × 100 cells are flattened and mixed into 512
          whole-board features; one sphere per unit.
        </div>
        <div>
          <b>Q head</b> — one node per board cell, laid out exactly like the board. Each value is
          the net&apos;s predicted return for firing at that cell; the white ring marks the argmax
          — the move actually taken.
        </div>
        <div>
          <b>edges &amp; pulses</b> — the strongest activation×weight products flowing in this
          pass (width and pulse speed scale with the contribution). Hover anything for its exact
          number; click a Q node to trace where its value came from.
        </div>
      </div>
    </div>
  );
}
