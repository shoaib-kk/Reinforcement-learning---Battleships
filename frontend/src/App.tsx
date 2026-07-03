import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { useStream } from "./useStream";
import Board, { OwnBoard, OverlayMode } from "./components/Board";
import ControlPanel from "./components/ControlPanel";
import MetricsDashboard from "./components/MetricsDashboard";
import {
  ActivationsView,
  EmbeddingView,
  NetworkDiagram,
  WeightHists,
} from "./components/NetworkInternals";
import type {
  BoardState,
  EpisodeRecord,
  LayerHistograms,
  StepPayload,
  TrainStatus,
  WsMessage,
} from "./types";

type Mode = "train" | "demo";
type Tab = "dashboard" | "network";

const EMPTY_GRID: number[][] = Array.from({ length: 10 }, () => Array(10).fill(0));

function cellName(action: number | undefined): string {
  if (action === undefined || action < 0) return "—";
  return `${"ABCDEFGHIJ"[action % 10]}${Math.floor(action / 10) + 1}`;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("train");
  // deep-linkable tab: /#network opens the internals view directly
  const [tab, setTab] = useState<Tab>(() =>
    location.hash === "#network" ? "network" : "dashboard"
  );
  const [overlay, setOverlay] = useState<OverlayMode>("q");
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [records, setRecords] = useState<EpisodeRecord[]>([]);
  const [lastTrain, setLastTrain] = useState<StepPayload | null>(null);
  const [lastDemo, setLastDemo] = useState<StepPayload | null>(null);
  const [demoBoard, setDemoBoard] = useState<BoardState | null>(null);
  const [demoSource, setDemoSource] = useState<string>("");
  const [weights, setWeights] = useState<LayerHistograms[] | null>(null);

  const onMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case "train_step":
        setLastTrain(msg);
        break;
      case "demo_step":
        if (!msg.error) setLastDemo(msg);
        break;
      case "demo_reset":
        setLastDemo(null);
        setDemoBoard(msg.board);
        setDemoSource(msg.source);
        break;
      case "episode":
        setRecords((prev) => {
          const next = [...prev, msg];
          return next.length > 6000 ? next.slice(-5000) : next;
        });
        break;
      case "weights":
        setWeights(msg.layers);
        break;
      case "status":
        setStatus(msg);
        break;
    }
  }, []);
  const connected = useStream(onMessage);

  useEffect(() => {
    api.metrics().then((r) => setRecords(r.records)).catch(() => {});
    api.trainStatus().then(setStatus).catch(() => {});
    api.weights().then((r) => setWeights(r.layers)).catch(() => {});
  }, []);

  const refreshWeights = () =>
    api.weights().then((r) => setWeights(r.layers)).catch(() => {});

  const onDemoPayload = useCallback((p: StepPayload) => setLastDemo(p), []);
  const onDemoReset = useCallback((board: BoardState, source: string) => {
    setLastDemo(null);
    setDemoBoard(board);
    setDemoSource(source);
    setMode("demo");
  }, []);

  // What the board panel shows depends on the mode.
  const current = mode === "train" ? lastTrain : lastDemo;
  const board = current?.board ?? (mode === "demo" ? demoBoard : null);
  const saliencyAvailable = mode === "demo" && !!lastDemo?.saliency;

  return (
    <>
      <header className="app-header">
        <h1>Battleship RL</h1>
        <span className="hint">
          <span className={`conn-dot ${connected ? "on" : "off"}`} />
          {connected ? "stream connected" : "reconnecting…"}
        </span>
        <span className="spacer" />
        <div className="mode-toggle">
          <button className={mode === "train" ? "active" : ""} onClick={() => setMode("train")}>
            Watch live training
          </button>
          <button className={mode === "demo" ? "active" : ""} onClick={() => setMode("demo")}>
            Demo a checkpoint
          </button>
        </div>
      </header>

      <div className="layout">
        <div>
          <ControlPanel
            status={status}
            onStatus={setStatus}
            onDemoPayload={onDemoPayload}
            onDemoReset={onDemoReset}
          />

          <div className="card">
            <h2>{mode === "train" ? "Live training game" : `Demo game${demoSource ? ` — ${demoSource}` : ""}`}</h2>
            <div className="row">
              <label>Overlay</label>
              <div className="mode-toggle">
                {(["none", "q", "saliency"] as OverlayMode[]).map((m) => (
                  <button
                    key={m}
                    className={overlay === m ? "active" : ""}
                    disabled={m === "saliency" && !saliencyAvailable}
                    onClick={() => setOverlay(m)}
                  >
                    {m === "q" ? "Q-values" : m === "saliency" ? "saliency" : "none"}
                  </button>
                ))}
              </div>
            </div>
            <Board
              view={board?.view ?? EMPTY_GRID}
              qValues={current?.q_values}
              saliency={mode === "demo" ? lastDemo?.saliency : undefined}
              lastAction={current?.action}
              overlay={overlay === "saliency" && !saliencyAvailable ? "q" : overlay}
            />
            <div className="legend-row">
              <span className="legend-key">
                <span className="legend-swatch" style={{ background: "#d03b3b" }} /> hit / sunk
              </span>
              <span className="legend-key">
                <span className="legend-dot" style={{ background: "#898781" }} /> miss
              </span>
              <span className="legend-key">
                <span className="legend-swatch" style={{ border: "2px solid #fff", background: "transparent" }} /> target
              </span>
              {overlay === "q" && (
                <span className="legend-key">
                  <span className="legend-swatch" style={{ background: "rgba(57,135,229,0.7)" }} /> higher Q
                </span>
              )}
              {overlay === "saliency" && (
                <span className="legend-key">
                  <span className="legend-swatch" style={{ background: "rgba(35,196,143,0.7)" }} /> input influence
                </span>
              )}
            </div>
            <div className="hint" style={{ marginTop: 8 }}>
              {current
                ? `shot ${board?.shots ?? 0} · fired at ${cellName(current.action)} → ${current.result}` +
                  (current.sunk_ship ? ` (${current.sunk_ship} down!)` : "") +
                  (mode === "train" && current.epsilon !== undefined
                    ? ` · ε ${current.epsilon.toFixed(3)} · ep ${current.episode}`
                    : "") +
                  (current.done ? (current.raw?.win ? " · WON" : " · game over") : "")
                : mode === "train"
                  ? "Start training to watch the agent play live."
                  : "Pick weights and start a new game."}
            </div>

            <div className="boards-flex" style={{ marginTop: 12 }}>
              <div>
                <h3>Agent's fleet (under fire)</h3>
                <OwnBoard grid={board?.own_board ?? EMPTY_GRID} />
              </div>
              <div>
                <h3>Enemy fleet</h3>
                <div className="fleet-list">
                  {(board?.enemy_ships ?? []).map((s) => (
                    <div key={s.name} className={s.sunk ? "sunk" : ""}>
                      {s.name} ({s.size}) — {s.hits}/{s.size}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <div className="tabs">
            <button
              className={tab === "dashboard" ? "active" : ""}
              onClick={() => {
                setTab("dashboard");
                history.replaceState(null, "", "#");
              }}
            >
              Training dashboard
            </button>
            <button
              className={tab === "network" ? "active" : ""}
              onClick={() => {
                setTab("network");
                history.replaceState(null, "", "#network");
              }}
            >
              Network internals
            </button>
          </div>

          {tab === "dashboard" ? (
            <MetricsDashboard records={records} status={status} />
          ) : (
            <>
              <div className="card">
                <h2>Architecture</h2>
                <NetworkDiagram />
              </div>
              <div className="card">
                <h2>
                  Layer activations{" "}
                  {current ? `— ${mode} step, firing at ${cellName(current.action)}` : ""}
                </h2>
                <ActivationsView activations={current?.activations ?? null} />
              </div>
              <div className="card">
                <h2>Weight & gradient distributions</h2>
                <WeightHists layers={weights} onRefresh={refreshWeights} />
              </div>
              <div className="card">
                <h2>State-embedding projection</h2>
                <EmbeddingView />
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
