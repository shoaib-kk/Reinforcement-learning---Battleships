"""DQN training loop, instrumented for live visualization.

The Trainer is designed to run inside a background thread owned by the
FastAPI server (see server/main.py): it never touches asyncio itself, it
just calls the `emit(event_dict)` callback for anything worth streaming and
guards its network with a plain threading.Lock so the serving thread can run
introspection forwards (embeddings, weight histograms) without racing the
optimizer.

Events emitted (all JSON-safe dicts with a "type" key):
    train_step — throttled to every `viz_every` env steps; carries the board,
                 chosen action, masked Q-values and layer activations.
    episode    — once per episode; the metrics record (also persisted).
    weights    — every `hist_every` episodes; per-layer weight/grad histograms.
    status     — lifecycle changes (running / stopped / finished).

Algorithm notes: vanilla DQN with a target network, masked epsilon-greedy
exploration and masked max-Q bootstrapping. Double-DQN would be a two-line
change in `_optimize` (argmax from online net, value from target net). For a
PPO alternative see the note in agent/model.py.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from env.battleship_env import BattleshipEnv, BOARD_SIZE, N_CELLS
from .model import DQN
from .opponents import make_opponent
from .replay_buffer import ReplayBuffer, OUTCOME_WIN, OUTCOME_LOSS
from . import viz3d


@dataclass
class TrainConfig:
    opponent: str = "random"          # none | random | hunt_target | self
    include_own_board: bool = False   # adds the own-ships input channel
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_capacity: int = 50_000
    warmup: int = 1_000               # steps before learning starts
    train_every: int = 1              # env steps per optimize() call
    target_sync: int = 1_000          # steps between target-network syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: float = 30_000   # exponential decay time constant
    max_episodes: int = 10_000
    checkpoint_every: int = 200       # episodes
    hist_every: int = 25              # episodes between weight-histogram events
    viz_every: int = 20               # steps between live train_step events
    viz3d: bool = True                # attach 3D-scene tensors to the stream
    viz3d_every: int = 1              # every Nth train_step also carries 3D data
    self_play_sync: int = 200         # episodes between frozen-opponent refreshes
    grad_clip: float = 10.0
    seed: int | None = None
    device: str = "cpu"


def round_list(arr: np.ndarray, nd: int = 3) -> list:
    return np.round(np.asarray(arr, dtype=np.float64), nd).tolist()


def activation_payload(acts: dict[str, torch.Tensor], max_maps: int = 16) -> dict:
    """Shrink captured activations into a JSON payload the frontend can draw.

    Conv layers: the first `max_maps` feature maps as 10x10 grids.
    Dense layers: the full activation vector.
    """
    out = {}
    for name, t in acts.items():
        t = t[0]
        if t.dim() == 3:
            out[name] = {
                "kind": "conv",
                "n_total": int(t.shape[0]),
                "maps": [round_list(m) for m in t[:max_maps].cpu().numpy()],
            }
        else:
            out[name] = {"kind": "dense", "values": round_list(t.cpu().numpy())}
    return out


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        emit=None,
        checkpoint_dir: str = "checkpoints",
        metrics_path: str = "runs/metrics.jsonl",
    ):
        self.cfg = config
        self.emit = emit or (lambda event: None)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self.device = config.device
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed if config.seed is not None else int(time.time()))

        n_ch = 4 if config.include_own_board else 3
        self.model = DQN(in_channels=n_ch).to(self.device)
        self.target = DQN(in_channels=n_ch).to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(
            config.buffer_capacity, (n_ch, BOARD_SIZE, BOARD_SIZE), seed=config.seed
        )

        # Self-play uses a frozen copy so the opponent doesn't shift mid-game.
        self._frozen = None
        if config.opponent == "self":
            self._frozen = copy.deepcopy(self.model).eval()
        opponent = make_opponent(config.opponent, model=self._frozen, device=self.device)
        self.env = BattleshipEnv(
            opponent=opponent,
            include_own_board=config.include_own_board,
            seed=config.seed,
        )

        # Cross-thread coordination with the server.
        self.lock = threading.Lock()      # guards self.model / buffer reads
        self.stop_event = threading.Event()
        self.running = False

        self.episode = 0
        self.total_steps = 0
        self._viz_msg_count = 0
        self.metrics: list[dict] = []
        self._win_hist: dict[str, deque] = {}
        self._shots_hist: deque = deque(maxlen=100)
        self._load_metrics_history()

    # -- public control ----------------------------------------------------

    def set_viz_every(self, n: int) -> None:
        """Live stream speed control (called from the server thread)."""
        self.cfg.viz_every = max(1, int(n))

    def set_viz3d(self, enabled: bool | None = None, every: int | None = None) -> None:
        """Separate throttle for the (heavier) 3D tensor stream."""
        if enabled is not None:
            self.cfg.viz3d = bool(enabled)
        if every is not None:
            self.cfg.viz3d_every = max(1, int(every))

    def status(self) -> dict:
        return {
            "running": self.running,
            "episode": self.episode,
            "total_steps": self.total_steps,
            "epsilon": round(self._epsilon(), 4),
            "buffer_size": len(self.buffer),
            "opponent": self.cfg.opponent,
            "config": asdict(self.cfg),
        }

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episode = ckpt.get("episode", 0)
        self.total_steps = ckpt.get("total_steps", 0)
        if self._frozen is not None:
            self._frozen.load_state_dict(ckpt["model"])

    # -- main loop -----------------------------------------------------------

    def run(self) -> None:
        """Blocking training loop; run inside a thread. Stops when
        stop_event is set or max_episodes is reached."""
        self.running = True
        self.stop_event.clear()
        self.emit({"type": "status", "training": True, **self.status()})
        try:
            while not self.stop_event.is_set() and self.episode < self.cfg.max_episodes:
                self._run_episode()
                self.episode += 1
                if self.episode % self.cfg.checkpoint_every == 0:
                    self.save_checkpoint()
                if self.episode % self.cfg.hist_every == 0:
                    self._emit_weight_histograms()
                if (
                    self._frozen is not None
                    and self.episode % self.cfg.self_play_sync == 0
                ):
                    with self.lock:
                        self._frozen.load_state_dict(self.model.state_dict())
        finally:
            self.running = False
            self.save_checkpoint()
            self.emit({"type": "status", "training": False, **self.status()})

    def _run_episode(self) -> None:
        cfg = self.cfg
        obs = self.env.reset()
        mask = self.env.action_mask()
        ep_indices: list[int] = []
        shaped_return = 0.0
        losses: list[float] = []
        won = False
        shots = 0

        done = False
        while not done:
            eps = self._epsilon()
            action, q_np = self._select_action(obs, mask, eps)
            next_obs, reward, done, info = self.env.step(action)
            next_mask = info["action_mask"]

            idx = self.buffer.add(obs, action, reward, next_obs, done, next_mask)
            ep_indices.append(idx)
            shaped_return += reward
            shots = info["shots"]
            won = info["raw"]["win"]

            self.total_steps += 1
            if (
                self.total_steps % cfg.train_every == 0
                and len(self.buffer) >= cfg.warmup
            ):
                losses.append(self._optimize())
            if self.total_steps % cfg.target_sync == 0:
                self.target.load_state_dict(self.model.state_dict())

            if self.total_steps % cfg.viz_every == 0:
                self._emit_train_step(obs, mask, action, q_np, info, eps)

            obs, mask = next_obs, next_mask

        # Back-fill episode outcome so the embedding view can color states.
        self.buffer.set_outcomes(ep_indices, OUTCOME_WIN if won else OUTCOME_LOSS)
        self._record_episode(won, shots, shaped_return, losses)

    # -- pieces ------------------------------------------------------------

    def _epsilon(self) -> float:
        cfg = self.cfg
        return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(
            -self.total_steps / cfg.eps_decay_steps
        )

    def _select_action(self, obs: np.ndarray, mask: np.ndarray, eps: float):
        """Masked epsilon-greedy. Returns (action, q_values or None)."""
        q_np = None
        with self.lock, torch.no_grad():
            q = self.model(torch.from_numpy(obs).unsqueeze(0).to(self.device))[0]
            q_np = q.cpu().numpy()
        if self.rng.random() < eps:
            action = int(self.rng.choice(np.flatnonzero(mask)))
        else:
            masked = np.where(mask, q_np, -np.inf)
            action = int(np.argmax(masked))
        return action, q_np

    def _optimize(self) -> float:
        cfg = self.cfg
        s, a, r, s2, d, m2 = self.buffer.sample(cfg.batch_size, self.device)
        with self.lock:
            q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = self.target(s2)
                q_next = q_next.masked_fill(~m2, float("-inf")).max(dim=1).values
                # Terminal / fully-fired states have no legal action: zero them.
                q_next = torch.where(torch.isinf(q_next), torch.zeros_like(q_next), q_next)
                target = r + cfg.gamma * (1.0 - d) * q_next
            loss = self.loss_fn(q, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
        return float(loss.item())

    def _record_episode(self, won: bool, shots: int, shaped_return: float, losses: list):
        opp = self.cfg.opponent
        hist = self._win_hist.setdefault(opp, deque(maxlen=100))
        hist.append(1.0 if won else 0.0)
        if won:
            self._shots_hist.append(shots)
        record = {
            "type": "episode",
            "episode": self.episode,
            "total_steps": self.total_steps,
            "opponent": opp,
            "won": won,
            "shots": shots,
            "shots_to_win": shots if won else None,
            "shaped_return": round(shaped_return, 3),
            "raw_return": 1.0 if won else -1.0,   # unshaped win/loss signal
            "loss": round(float(np.mean(losses)), 5) if losses else None,
            "epsilon": round(self._epsilon(), 4),
            "win_rate": round(float(np.mean(hist)), 3),
            "avg_shots_to_win": round(float(np.mean(self._shots_hist)), 2)
            if self._shots_hist
            else None,
            "buffer_size": len(self.buffer),
            "time": time.time(),
        }
        self.metrics.append(record)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.emit(record)

    def _emit_train_step(self, obs, mask, action, q_np, info, eps) -> None:
        # Q-values at the decision point; cells that were illegal then -> null.
        q_json = [
            round(float(v), 3) if legal else None for v, legal in zip(q_np, mask)
        ]
        payload_3d = None
        self._viz_msg_count += 1
        want_3d = self.cfg.viz3d and self._viz_msg_count % self.cfg.viz3d_every == 0
        with self.lock:
            _, acts = self.model.forward_with_activations(
                torch.from_numpy(obs).unsqueeze(0).to(self.device)
            )
            if want_3d:
                # Same captured pass that produced `acts`; contributions are
                # computed against the current weights, so nothing is stale.
                payload_3d = viz3d.forward_payload(self.model, obs, acts)
                grads = viz3d.grad_payload(self.model)  # latest optimize step
                if grads is not None:
                    payload_3d["grads"] = grads
        msg = {
            "type": "train_step",
            "episode": self.episode,
            "step": self.total_steps,
            "epsilon": round(eps, 4),
            "action": int(action),
            "result": info["result"],
            "q_values": q_json,
            "board": self.env.get_render_state(),
            "activations": activation_payload(acts),
        }
        if payload_3d is not None:
            msg["viz3d"] = payload_3d
        self.emit(msg)

    def _emit_weight_histograms(self) -> None:
        with self.lock:
            layers = self.model.weight_histograms()
        self.emit({"type": "weights", "episode": self.episode, "layers": layers})

    # -- persistence ---------------------------------------------------------

    def save_checkpoint(self) -> str:
        path = self.checkpoint_dir / f"ckpt_ep{self.episode}.pt"
        with self.lock:
            payload = {
                "model": self.model.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.cfg),
                "episode": self.episode,
                "total_steps": self.total_steps,
            }
        torch.save(payload, path)
        torch.save(payload, self.checkpoint_dir / "latest.pt")
        return str(path)

    def _load_metrics_history(self) -> None:
        if self.metrics_path.exists():
            with self.metrics_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
