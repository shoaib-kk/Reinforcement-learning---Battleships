"""Opponent policies the agent trains against.

All opponents share a tiny interface driven by the env:
    reset()                      — new game
    act(mask) -> int             — pick a cell to fire at (mask: legal cells)
    observe(cell, result)        — told what happened ("hit"/"miss"/"sunk")
"""

from __future__ import annotations

import numpy as np

from env.battleship_env import BOARD_SIZE, N_CELLS


class RandomShooter:
    """Uniform random over unfired cells — the easy baseline."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, mask: np.ndarray) -> int:
        return int(self.rng.choice(np.flatnonzero(mask)))

    def observe(self, cell: int, result: str) -> None:
        pass


class HuntTargetBot:
    """Classic hunt/target heuristic — the strong fixed baseline.

    Hunt mode fires on a checkerboard parity (every ship covers at least one
    parity cell). After a hit it switches to target mode: try the 4
    neighbours, and once a second hit establishes a line, extend that line in
    both directions until the ship sinks.
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.targets: list[int] = []        # stack of cells to try next
        self.cluster: list[int] = []        # hits on the ship being chased

    def act(self, mask: np.ndarray) -> int:
        while self.targets:
            cell = self.targets.pop()
            if mask[cell]:
                return cell
        parity = [c for c in np.flatnonzero(mask) if (c // BOARD_SIZE + c % BOARD_SIZE) % 2 == 0]
        pool = parity if parity else list(np.flatnonzero(mask))
        return int(self.rng.choice(pool))

    def observe(self, cell: int, result: str) -> None:
        if result == "sunk":
            self.targets.clear()
            self.cluster.clear()
            return
        if result != "hit":
            return
        self.cluster.append(cell)
        r, c = divmod(cell, BOARD_SIZE)
        if len(self.cluster) >= 2:
            # Line established: only extend along it.
            rows = {t // BOARD_SIZE for t in self.cluster}
            self.targets.clear()
            if len(rows) == 1:  # horizontal
                cols = sorted(t % BOARD_SIZE for t in self.cluster)
                row = r
                if cols[0] - 1 >= 0:
                    self.targets.append(row * BOARD_SIZE + cols[0] - 1)
                if cols[-1] + 1 < BOARD_SIZE:
                    self.targets.append(row * BOARD_SIZE + cols[-1] + 1)
            else:  # vertical
                rs = sorted(t // BOARD_SIZE for t in self.cluster)
                col = c
                if rs[0] - 1 >= 0:
                    self.targets.append((rs[0] - 1) * BOARD_SIZE + col)
                if rs[-1] + 1 < BOARD_SIZE:
                    self.targets.append((rs[-1] + 1) * BOARD_SIZE + col)
        else:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                    self.targets.append(rr * BOARD_SIZE + cc)


class PolicyOpponent:
    """Self-play opponent: greedy play from a (frozen) copy of the agent's net.

    It keeps its own view of the board it is firing at and feeds that through
    the network, so both sides literally run the same policy.
    """

    def __init__(self, model, device="cpu"):
        import torch  # local import so opponents.py stays torch-free otherwise

        self.torch = torch
        self.model = model
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.view = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)  # 0 unk 1 hit 2 miss

    def act(self, mask: np.ndarray) -> int:
        torch = self.torch
        n_ch = getattr(self.model, "in_channels", 3)
        obs = np.zeros((n_ch, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.view == 0).astype(np.float32)
        obs[1] = (self.view == 1).astype(np.float32)
        obs[2] = (self.view == 2).astype(np.float32)
        with torch.no_grad():
            q = self.model(torch.from_numpy(obs).unsqueeze(0).to(self.device))[0]
        q = q.cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def observe(self, cell: int, result: str) -> None:
        r, c = divmod(cell, BOARD_SIZE)
        self.view[r, c] = 1 if result in ("hit", "sunk") else 2


def make_opponent(kind: str, model=None, device: str = "cpu", seed: int | None = None):
    """Factory used by the trainer and the demo runner."""
    if kind in ("none", None, ""):
        return None
    if kind == "random":
        return RandomShooter(seed)
    if kind == "hunt_target":
        return HuntTargetBot(seed)
    if kind == "self":
        if model is None:
            raise ValueError("self-play opponent needs a model")
        return PolicyOpponent(model, device)
    raise ValueError(f"unknown opponent kind: {kind!r}")
