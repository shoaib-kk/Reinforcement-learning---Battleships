"""Step-by-step demo game, decoupled from training.

DemoGame owns its own DQN instance, so the frontend can step through a game
(from a checkpoint, a snapshot of the live trainer's weights, or fresh
random weights) without ever blocking the training loop. Every step returns
the full introspection payload — board, masked Q-values, per-layer
activations, and input saliency — and is recorded for JSON export.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from agent import viz3d
from agent.model import DQN
from agent.opponents import make_opponent
from agent.trainer import activation_payload, round_list
from env.battleship_env import BattleshipEnv, N_CELLS


class DemoGame:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model: DQN | None = None
        self.env: BattleshipEnv | None = None
        self.obs = None
        self.mask = None
        self.source = "none"     # what the current weights came from
        self.history: list[dict] = []
        self.step_idx = 0

    # -- weight loading ------------------------------------------------------

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        in_ch = 4 if ckpt.get("config", {}).get("include_own_board") else 3
        self.model = DQN(in_channels=in_ch).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.source = f"checkpoint:{path} (ep {ckpt.get('episode', '?')})"

    def load_live(self, state_dict: dict, in_channels: int) -> None:
        self.model = DQN(in_channels=in_channels).to(self.device).eval()
        self.model.load_state_dict(state_dict)
        self.source = "live trainer weights"

    def ensure_model(self) -> None:
        if self.model is None:
            self.model = DQN(in_channels=3).to(self.device).eval()
            self.source = "fresh (untrained) weights"

    # -- game control ---------------------------------------------------------

    def reset(self, opponent: str = "random", seed: int | None = None) -> dict:
        self.ensure_model()
        opp = make_opponent(
            opponent, model=self.model if opponent == "self" else None, device=self.device
        )
        in_ch = self.model.in_channels
        self.env = BattleshipEnv(
            opponent=opp, include_own_board=(in_ch == 4), seed=seed
        )
        self.obs = self.env.reset()
        self.mask = self.env.action_mask()
        self.history = []
        self.step_idx = 0
        return {
            "type": "demo_reset",
            "board": self.env.get_render_state(),
            "source": self.source,
            "opponent": opponent,
        }

    def step(self) -> dict:
        """Greedy step with full introspection. Returns the step payload."""
        if self.env is None or self.env.done:
            return {"type": "demo_step", "error": "no active game — reset first"}

        x = torch.from_numpy(self.obs).unsqueeze(0).to(self.device)
        q, acts = self.model.forward_with_activations(x)
        q_np = q[0].cpu().numpy()
        masked = np.where(self.mask, q_np, -np.inf)
        action = int(np.argmax(masked))
        saliency = self.model.saliency(x, action)
        # 3D scene data for this exact forward pass (demo = inference mode,
        # so there are never gradients here).
        payload_3d = viz3d.forward_payload(self.model, self.obs, acts)

        obs2, reward, done, info = self.env.step(action)
        payload = {
            "type": "demo_step",
            "step": self.step_idx,
            "action": action,
            "result": info["result"],
            "sunk_ship": info["sunk_ship"],
            "reward": round(float(reward), 3),
            "done": bool(done),
            "raw": info["raw"],
            "q_values": [
                round(float(v), 3) if legal else None
                for v, legal in zip(q_np, self.mask)
            ],
            "saliency": [round_list(row) for row in saliency],
            "activations": activation_payload(acts),
            "viz3d": payload_3d,
            "board": self.env.get_render_state(),
        }
        self.history.append(payload)
        self.step_idx += 1
        self.obs, self.mask = obs2, info["action_mask"]
        return payload

    def export(self) -> dict:
        """Full game replay (state/action/Q/activations per step) as JSON."""
        return {
            "exported_at": time.time(),
            "source": self.source,
            "n_steps": len(self.history),
            "steps": self.history,
        }
