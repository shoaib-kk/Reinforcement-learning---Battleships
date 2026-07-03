"""Uniform experience-replay ring buffer backed by preallocated numpy arrays.

Beyond the usual (s, a, r, s', done) it stores the next-state legal-action
mask (needed for masked max-Q targets) and a per-transition episode outcome
that the trainer back-fills when the episode ends — the outcome label colors
the 2D embedding projection in the frontend.
"""

from __future__ import annotations

import numpy as np
import torch

OUTCOME_UNKNOWN, OUTCOME_WIN, OUTCOME_LOSS = 0, 1, 2


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int | None = None):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, obs_shape[-1] * obs_shape[-2]), dtype=bool)
        self.outcomes = np.zeros(capacity, dtype=np.int8)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, s, a, r, s2, done, next_mask) -> int:
        """Store one transition; returns its slot index (for outcome back-fill)."""
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = float(done)
        self.next_masks[i] = next_mask
        self.outcomes[i] = OUTCOME_UNKNOWN
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True
        return i

    def set_outcomes(self, indices: list[int], outcome: int) -> None:
        self.outcomes[indices] = outcome

    def sample(self, batch_size: int, device: str = "cpu"):
        idx = self.rng.integers(0, len(self), size=batch_size)
        to = lambda arr, dtype: torch.as_tensor(arr[idx], dtype=dtype, device=device)
        return (
            to(self.states, torch.float32),
            to(self.actions, torch.int64),
            to(self.rewards, torch.float32),
            to(self.next_states, torch.float32),
            to(self.dones, torch.float32),
            to(self.next_masks, torch.bool),
        )

    def sample_states(self, n: int):
        """(states, outcomes) for the embedding-projection view."""
        n = min(n, len(self))
        idx = self.rng.choice(len(self), size=n, replace=False)
        return self.states[idx], self.outcomes[idx]

    def snapshot(self, n: int = 8) -> list[dict]:
        """A few raw transitions, JSON-friendly — surfaced for debugging."""
        n = min(n, len(self))
        idx = self.rng.choice(len(self), size=n, replace=False)
        return [
            {
                "action": int(self.actions[i]),
                "reward": round(float(self.rewards[i]), 3),
                "done": bool(self.dones[i]),
                "outcome": int(self.outcomes[i]),
            }
            for i in idx
        ]
