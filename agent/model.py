"""DQN model for Battleship, built to be watched.

A small convolutional encoder treats the 10x10 board like an image and a
dense head maps it to one Q-value per cell. Everything the frontend
visualizes — per-layer activations, gradients, weight histograms, input
saliency — comes out of the hooks and helpers defined here.

Swapping in a policy-gradient method (e.g. PPO): keep `encoder` as-is and
replace the single `head` with two heads — an actor (Linear(512, 100) +
masked softmax over legal cells) and a critic (Linear(512, 1)). The trainer
would then optimize the clipped-surrogate + value loss on on-policy rollouts
instead of TD error on replayed transitions; the introspection helpers below
work unchanged since they only depend on layer names.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from env.battleship_env import BOARD_SIZE, N_CELLS


class DQN(nn.Module):
    def __init__(self, in_channels: int = 3, n_actions: int = N_CELLS):
        super().__init__()
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * N_CELLS, 512)
        self.head = nn.Linear(512, n_actions)
        self.act = nn.ReLU()

        # Populated by forward hooks whenever capture is enabled.
        self._captured: dict[str, torch.Tensor] = {}
        self._capture = False
        for name in ("conv1", "conv2", "conv3", "fc1", "head"):
            getattr(self, name).register_forward_hook(self._make_hook(name))

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = h.flatten(1)
        h = self.act(self.fc1(h))
        return self.head(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate (fc1) embedding — used for the 2D state projection."""
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        return self.act(self.fc1(h.flatten(1)))

    # -- introspection ------------------------------------------------------

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            if self._capture:
                self._captured[name] = out.detach()
        return hook

    @torch.no_grad()
    def forward_with_activations(self, x: torch.Tensor):
        """Returns (q_values, {layer: activation tensor}) for one batch."""
        self._captured = {}
        self._capture = True
        try:
            q = self.forward(x)
        finally:
            self._capture = False
        return q, dict(self._captured)

    def saliency(self, x: torch.Tensor, action: int) -> np.ndarray:
        """Gradient-based saliency: |d Q[action] / d input|, summed over
        channels and normalized to [0, 1]. Shape (10, 10).

        (An occlusion-sensitivity variant would re-run forward 100 times with
        each cell zeroed and diff the Q-value — sturdier but 100x the cost.)
        """
        x = x.clone().requires_grad_(True)
        q = self.forward(x)
        q[0, action].backward()
        sal = x.grad.abs().sum(dim=1)[0]
        peak = sal.max()
        if peak > 0:
            sal = sal / peak
        return sal.detach().cpu().numpy()

    def architecture_summary(self) -> list[dict]:
        """Layer list with shapes for the frontend's network diagram."""
        dummy = torch.zeros(1, self.in_channels, BOARD_SIZE, BOARD_SIZE)
        _, acts = self.forward_with_activations(dummy)
        layers = [
            {
                "name": "input",
                "type": "Input",
                "out_shape": [self.in_channels, BOARD_SIZE, BOARD_SIZE],
                "params": 0,
            }
        ]
        for name in ("conv1", "conv2", "conv3", "fc1", "head"):
            mod = getattr(self, name)
            layers.append(
                {
                    "name": name,
                    "type": type(mod).__name__,
                    "out_shape": list(acts[name].shape[1:]),
                    "params": sum(p.numel() for p in mod.parameters()),
                }
            )
        return layers

    def weight_histograms(self, bins: int = 33) -> list[dict]:
        """Per-layer weight (and grad, when present) histograms."""
        out = []
        for name in ("conv1", "conv2", "conv3", "fc1", "head"):
            w = getattr(self, name).weight
            counts, edges = np.histogram(w.detach().cpu().numpy(), bins=bins)
            entry = {
                "layer": name,
                "weights": {"counts": counts.tolist(), "edges": np.round(edges, 4).tolist()},
            }
            if w.grad is not None:
                gcounts, gedges = np.histogram(w.grad.detach().cpu().numpy(), bins=bins)
                entry["grads"] = {"counts": gcounts.tolist(), "edges": np.round(gedges, 5).tolist()}
            out.append(entry)
        return out
