"""Battleship as a Gym-style environment.

Standard rules: 10x10 grid, fleet of carrier(5), battleship(4), cruiser(3),
submarine(3), destroyer(2). The agent fires at a hidden opponent board; an
optional opponent policy fires back at the agent's own board each turn.

API follows the classic Gym convention (no gym dependency required):
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.render()

Observation: float32 tensor of shape (C, 10, 10).
    channel 0: unknown (1 where the agent has not fired)
    channel 1: hit     (1 where the agent hit a ship, incl. sunk cells)
    channel 2: miss    (1 where the agent fired and missed)
    channel 3: own ship placement (only if include_own_board=True) — an
               auxiliary input so the net can learn a "hunt" prior from
               plausible ship layouts.

Action space: Discrete(100), one per cell (row-major). Cells already fired
at are illegal; the current legal-action mask is in info["action_mask"] and
via env.action_mask().

Reward shaping (see constants below): +hit, +sink bonus, small per-step
penalty, terminal win/loss reward. Raw performance (win/loss, shots-to-win)
is reported separately in info["raw"] so shaping never obscures the real
metric.
"""

from __future__ import annotations

import numpy as np

BOARD_SIZE = 10
N_CELLS = BOARD_SIZE * BOARD_SIZE

# name -> length, standard 1990 Milton Bradley fleet
FLEET = {
    "carrier": 5,
    "battleship": 4,
    "cruiser": 3,
    "submarine": 3,
    "destroyer": 2,
}

# --- reward shaping ---------------------------------------------------------
REWARD_HIT = 1.0        # any shot that hits a ship
REWARD_SINK_BONUS = 2.0  # extra on the shot that sinks a ship
REWARD_WIN = 10.0       # sank the whole enemy fleet
REWARD_LOSS = -10.0     # opponent sank ours first
STEP_PENALTY = -0.1     # per shot, encourages short games

# view-grid cell codes (agent's view of the opponent board)
UNKNOWN, HIT, MISS, SUNK = 0, 1, 2, 3


class Board:
    """One player's board: ship placement plus incoming-shot bookkeeping."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # 0 = water, i+1 = index of ship i in self.ship_names
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.ship_names: list[str] = list(FLEET.keys())
        self.ship_cells: list[set[tuple[int, int]]] = []
        self.ship_hits: list[set[tuple[int, int]]] = []
        self.shots = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        self._place_fleet()

    def _place_fleet(self) -> None:
        for idx, name in enumerate(self.ship_names):
            length = FLEET[name]
            while True:
                horiz = self.rng.random() < 0.5
                if horiz:
                    r = int(self.rng.integers(0, BOARD_SIZE))
                    c = int(self.rng.integers(0, BOARD_SIZE - length + 1))
                    cells = [(r, c + i) for i in range(length)]
                else:
                    r = int(self.rng.integers(0, BOARD_SIZE - length + 1))
                    c = int(self.rng.integers(0, BOARD_SIZE))
                    cells = [(r + i, c) for i in range(length)]
                if all(self.grid[rr, cc] == 0 for rr, cc in cells):
                    for rr, cc in cells:
                        self.grid[rr, cc] = idx + 1
                    self.ship_cells.append(set(cells))
                    self.ship_hits.append(set())
                    break

    def receive_shot(self, r: int, c: int) -> tuple[str, str | None]:
        """Apply a shot. Returns (result, sunk_ship_name).

        result is one of "hit", "miss", "sunk", "repeat".
        """
        if self.shots[r, c]:
            return "repeat", None
        self.shots[r, c] = True
        ship_id = int(self.grid[r, c]) - 1
        if ship_id < 0:
            return "miss", None
        self.ship_hits[ship_id].add((r, c))
        if self.ship_hits[ship_id] == self.ship_cells[ship_id]:
            return "sunk", self.ship_names[ship_id]
        return "hit", None

    def all_sunk(self) -> bool:
        return all(h == c for h, c in zip(self.ship_hits, self.ship_cells))

    def ships_status(self) -> list[dict]:
        return [
            {
                "name": name,
                "size": FLEET[name],
                "hits": len(self.ship_hits[i]),
                "sunk": self.ship_hits[i] == self.ship_cells[i],
            }
            for i, name in enumerate(self.ship_names)
        ]


class BattleshipEnv:
    """Gym-style Battleship. See module docstring for conventions.

    opponent: None for a pure "sink the fleet fast" task, or an object with
        reset() / act(mask, view) / observe(cell, result) — see
        agent/opponents.py. When set, the opponent fires back after every
        agent shot and the game becomes win/lose.
    """

    def __init__(
        self,
        opponent=None,
        include_own_board: bool = False,
        max_shots: int = N_CELLS,
        seed: int | None = None,
    ):
        self.opponent = opponent
        self.include_own_board = include_own_board
        self.max_shots = max_shots
        self.rng = np.random.default_rng(seed)
        self.n_channels = 4 if include_own_board else 3
        self.enemy_board: Board | None = None
        self.own_board: Board | None = None
        self.view = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.shots_taken = 0
        self.done = True

    # -- core API -------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.enemy_board = Board(self.rng)
        self.own_board = Board(self.rng)
        self.view = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.shots_taken = 0
        self.done = False
        if self.opponent is not None:
            self.opponent.reset()
        return self._obs()

    def step(self, action: int):
        assert not self.done, "step() called on a finished episode; call reset()"
        r, c = divmod(int(action), BOARD_SIZE)
        reward = STEP_PENALTY
        self.shots_taken += 1

        result, sunk_name = self.enemy_board.receive_shot(r, c)
        if result == "repeat":
            # Masked action selection should prevent this; treat as a wasted
            # shot so a buggy caller can't crash training.
            pass
        elif result == "miss":
            self.view[r, c] = MISS
        else:
            self.view[r, c] = HIT
            reward += REWARD_HIT
            if result == "sunk":
                reward += REWARD_SINK_BONUS
                ship_id = self.enemy_board.ship_names.index(sunk_name)
                for rr, cc in self.enemy_board.ship_cells[ship_id]:
                    self.view[rr, cc] = SUNK

        win = self.enemy_board.all_sunk()
        loss = False

        if win:
            reward += REWARD_WIN
            self.done = True
        elif self.opponent is not None:
            loss = self._opponent_turn()
            if loss:
                reward += REWARD_LOSS
                self.done = True

        if self.shots_taken >= self.max_shots and not self.done:
            self.done = True  # out of ammo — counts as a loss in raw stats

        info = {
            "result": result,
            "sunk_ship": sunk_name,
            "action_mask": self.action_mask(),
            "shots": self.shots_taken,
            # Raw, unshaped outcome — the honest performance signal.
            "raw": {
                "win": bool(win),
                "loss": bool(loss or (self.done and not win)),
                "shots_to_win": self.shots_taken if win else None,
            },
        }
        return self._obs(), float(reward), self.done, info

    def render(self) -> str:
        chars = {UNKNOWN: "·", HIT: "X", MISS: "o", SUNK: "#"}
        lines = ["  " + " ".join(str(i) for i in range(BOARD_SIZE))]
        for r in range(BOARD_SIZE):
            lines.append(f"{r} " + " ".join(chars[int(v)] for v in self.view[r]))
        out = "\n".join(lines)
        print(out)
        return out

    # -- helpers ----------------------------------------------------------

    def action_mask(self) -> np.ndarray:
        """Boolean (100,) — True where the agent may still fire."""
        return ~self.enemy_board.shots.reshape(-1)

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.n_channels, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.view == UNKNOWN).astype(np.float32)
        obs[1] = ((self.view == HIT) | (self.view == SUNK)).astype(np.float32)
        obs[2] = (self.view == MISS).astype(np.float32)
        if self.include_own_board:
            obs[3] = (self.own_board.grid > 0).astype(np.float32)
        return obs

    def _opponent_turn(self) -> bool:
        """Opponent fires once at our board. Returns True if our fleet died."""
        mask = ~self.own_board.shots.reshape(-1)
        if not mask.any():
            return False
        cell = int(self.opponent.act(mask))
        r, c = divmod(cell, BOARD_SIZE)
        result, _ = self.own_board.receive_shot(r, c)
        self.opponent.observe(cell, result)
        return self.own_board.all_sunk()

    def get_render_state(self) -> dict:
        """JSON-friendly snapshot for the frontend."""
        own = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        if self.own_board is not None:
            ship = self.own_board.grid > 0
            shot = self.own_board.shots
            own[ship] = 1
            own[ship & shot] = 2                     # our ship, hit
            own[(~ship) & shot] = 3                  # opponent missed
            for i, cells in enumerate(self.own_board.ship_cells):
                if self.own_board.ship_hits[i] == cells:
                    for rr, cc in cells:
                        own[rr, cc] = 4              # our ship, sunk
        return {
            "view": self.view.tolist(),
            "own_board": own.tolist(),
            "enemy_ships": self.enemy_board.ships_status() if self.enemy_board else [],
            "own_ships": self.own_board.ships_status() if self.own_board else [],
            "shots": self.shots_taken,
            "done": self.done,
        }
