// Control plane: start/stop training (with live hyperparameters), throttle
// the live-training stream, and drive the demo game (reset / step /
// play-pause / speed / export).

import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { StepPayload, TrainStatus } from "../types";

interface Props {
  status: TrainStatus | null;
  onStatus: (s: TrainStatus) => void;
  onDemoPayload: (p: StepPayload) => void;
  onDemoReset: (board: StepPayload["board"], source: string) => void;
}

export default function ControlPanel({ status, onStatus, onDemoPayload, onDemoReset }: Props) {
  // training hyperparameters (stretch goal: exposed in the UI)
  const [opponent, setOpponent] = useState("random");
  const [lr, setLr] = useState(0.0001);
  const [epsDecay, setEpsDecay] = useState(30000);
  const [maxEpisodes, setMaxEpisodes] = useState(10000);
  const [resume, setResume] = useState(true);
  const [vizEvery, setVizEvery] = useState(20);
  const [error, setError] = useState<string | null>(null);

  // demo controls
  const [demoSource, setDemoSource] = useState("latest");
  const [demoOpponent, setDemoOpponent] = useState("random");
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(400); // ms between demo steps
  const [demoActive, setDemoActive] = useState(false);
  const stepBusy = useRef(false);

  const running = !!status?.running;

  const refreshCheckpoints = () =>
    api
      .checkpoints()
      .then((r) => setCheckpoints(r.checkpoints.map((c) => c.file)))
      .catch(() => {});

  useEffect(() => {
    refreshCheckpoints();
  }, [running]);

  const start = async () => {
    setError(null);
    try {
      const s = await api.trainStart({
        opponent,
        lr,
        eps_decay_steps: epsDecay,
        max_episodes: maxEpisodes,
        viz_every: vizEvery,
        resume,
      });
      onStatus({ ...s, running: true });
    } catch (e) {
      setError(String(e));
    }
  };

  const stop = async () => {
    setError(null);
    try {
      const s = await api.trainStop();
      onStatus({ ...s, running: false });
      refreshCheckpoints();
    } catch (e) {
      setError(String(e));
    }
  };

  const demoReset = async () => {
    setError(null);
    setPlaying(false);
    try {
      const r = await api.demoReset(demoSource, demoOpponent);
      setDemoActive(true);
      onDemoReset(r.board, r.source);
    } catch (e) {
      setError(String(e));
    }
  };

  const demoStep = async (): Promise<boolean> => {
    if (stepBusy.current) return true;
    stepBusy.current = true;
    try {
      const p = await api.demoStep();
      if (p.error) {
        setError(p.error);
        return false;
      }
      onDemoPayload(p);
      return !p.done;
    } catch (e) {
      setError(String(e));
      return false;
    } finally {
      stepBusy.current = false;
    }
  };

  // play loop
  useEffect(() => {
    if (!playing) return;
    let alive = true;
    const id = window.setInterval(async () => {
      const more = await demoStep();
      if (!more && alive) setPlaying(false);
    }, speed);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed]);

  return (
    <div className="card">
      <h2>Training</h2>
      <div className="row">
        <label>Opponent</label>
        <select value={opponent} onChange={(e) => setOpponent(e.target.value)} disabled={running}>
          <option value="random">random shooter</option>
          <option value="hunt_target">hunt/target bot</option>
          <option value="self">self-play</option>
          <option value="none">none (solo board)</option>
        </select>
      </div>
      <div className="row">
        <label>Learning rate</label>
        <input
          type="number"
          step="0.00005"
          value={lr}
          onChange={(e) => setLr(Number(e.target.value))}
          disabled={running}
        />
      </div>
      <div className="row">
        <label>ε decay steps</label>
        <input
          type="number"
          step="1000"
          value={epsDecay}
          onChange={(e) => setEpsDecay(Number(e.target.value))}
          disabled={running}
        />
      </div>
      <div className="row">
        <label>Max episodes</label>
        <input
          type="number"
          step="500"
          value={maxEpisodes}
          onChange={(e) => setMaxEpisodes(Number(e.target.value))}
          disabled={running}
        />
      </div>
      <div className="row">
        <label>Resume latest</label>
        <input
          type="checkbox"
          checked={resume}
          onChange={(e) => setResume(e.target.checked)}
          disabled={running}
        />
        <span className="hint">continue from latest checkpoint</span>
      </div>
      <div className="row">
        {!running ? (
          <button className="primary" onClick={start}>
            Start training
          </button>
        ) : (
          <button className="danger" onClick={stop}>
            Stop training
          </button>
        )}
        <span className="hint">
          {running
            ? `running · ep ${status?.episode ?? 0} · ${status?.total_steps?.toLocaleString() ?? 0} steps`
            : "idle"}
        </span>
      </div>
      <div className="row">
        <label>Stream every</label>
        <input
          type="range"
          min={1}
          max={100}
          value={vizEvery}
          onChange={(e) => {
            const n = Number(e.target.value);
            setVizEvery(n);
            api.streamConfig({ viz_every: n }).catch(() => {});
          }}
        />
        <span className="hint">{vizEvery} steps</span>
      </div>

      <h2 style={{ marginTop: 18 }}>Demo game</h2>
      <div className="row">
        <label>Weights</label>
        <select value={demoSource} onChange={(e) => setDemoSource(e.target.value)}>
          <option value="latest">latest checkpoint</option>
          <option value="live">live trainer (snapshot)</option>
          <option value="fresh">fresh / untrained</option>
          {checkpoints
            .filter((c) => c !== "latest.pt")
            .map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
        </select>
      </div>
      <div className="row">
        <label>Opponent</label>
        <select value={demoOpponent} onChange={(e) => setDemoOpponent(e.target.value)}>
          <option value="random">random shooter</option>
          <option value="hunt_target">hunt/target bot</option>
          <option value="self">self (same net)</option>
          <option value="none">none (solo board)</option>
        </select>
      </div>
      <div className="row">
        <button onClick={demoReset}>New game</button>
        <button onClick={() => demoStep()} disabled={!demoActive || playing}>
          Step
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={!demoActive}>
          {playing ? "Pause" : "Play"}
        </button>
        <a href="/api/demo/export" download>
          <button disabled={!demoActive}>Export replay</button>
        </a>
      </div>
      <div className="row">
        <label>Step delay</label>
        <input
          type="range"
          min={60}
          max={1500}
          step={20}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        />
        <span className="hint">{speed} ms</span>
      </div>
      {error && <div className="error-text">{error}</div>}
    </div>
  );
}
