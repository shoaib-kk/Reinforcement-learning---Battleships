# Battleship RL — watch a DQN learn to play

A reinforcement-learning system that trains a Deep Q-Network to play
Battleship, with a web frontend that shows both the live gameplay **and the
network's internals** while it learns: Q-value heatmaps over the board,
per-layer activations, input saliency, weight/gradient distributions, a
2D projection of how the agent represents board states, and a 3D scene of
the network itself with neurons firing per decision.

## Layout

```
env/       Battleship as a Gym-style environment (reset/step/render,
           3-channel board observation, action masking, reward shaping)
agent/     DQN model with introspection hooks, replay buffer, opponents
           (random, hunt/target heuristic, self-play), training loop with
           metrics + checkpointing, quantized 3D-viz payloads (viz3d.py)
server/    FastAPI app — WebSocket stream of steps/metrics/activations,
           REST control plane, step-by-step demo runner, PCA embeddings
frontend/  React + TypeScript app — board with heatmap overlays, playback
           controls, training dashboard (Chart.js), network-internals views,
           Three.js 3D network scene (src/NetworkViz3D/)
scripts/   smoke_test.py — end-to-end backend test without the server
```

## Setup

Requires Python 3.10+ and Node 18+.

```bash
# backend (CPU torch is fine; see pytorch.org for CUDA builds)
pip install -r requirements.txt

# frontend
cd frontend
npm install
npm run build        # produces frontend/dist, served by the API server
cd ..
```

## Launch

```bash
python -m uvicorn server.main:app --port 8000
```

Open **http://localhost:8000** — the built frontend is served there.
Training, metrics, and checkpoints all run through this one process
(training itself runs in a background thread, so the UI never blocks it).

For frontend development with hot reload, run `npm run dev` inside
`frontend/` and open http://localhost:5173 (API/WebSocket calls are proxied
to port 8000).

Everything can also be driven headlessly over REST — see the endpoint list
in `server/main.py`.

## The two modes

**Watch live training** — press *Start training* in the left panel. The
board shows the game the agent is playing right now (throttled by the
"stream every N steps" slider), the dashboard charts update per episode, and
the Network internals tab updates activations and weight histograms as they
stream. Hyperparameters (opponent, learning rate, epsilon decay, episode
budget) are set from the same panel; "resume latest" continues from the last
checkpoint.

**Load checkpoint and demo** — pick weights (latest checkpoint, any specific
checkpoint, a snapshot of the live trainer, or fresh untrained weights),
pick an opponent, press *New game*, then *Step* through shots one at a time
or *Play* with a speed slider. Demo steps include the full introspection
payload — Q-values, activations, gradient saliency, and the 3D scene data.
*Export replay* downloads the whole game (state, action, Q-values,
activations per step) as JSON. The demo runs on its own model instance and
never touches the training loop.

## What you're looking at (2D views)

- **Board heatmap (blue)** — the agent's Q-value for every legal cell at the
  current decision point; the white outline is the cell it chose.
- **Saliency (aqua, demo mode)** — |∂Q(chosen)/∂input| per cell: which board
  cells most influenced this shot.
- **Layer activations** — feature maps of each conv layer as 10×10 tiles,
  dense layers as magnitude strips, updating every streamed step.
- **Weight & gradient histograms** — per-layer distributions, sampled every
  few episodes during training.
- **State-embedding projection** — PCA of fc1 embeddings for sampled replay
  states, colored by episode outcome; watch win/loss states separate as the
  net learns.
- **Dashboard** — rolling win rate per opponent, moving-average shaped
  return, TD loss, epsilon decay, and average shots-to-win. Raw outcomes
  (win/loss, shots-to-win) are logged separately from the shaped reward in
  `runs/metrics.jsonl`.

## 3D network visualization (the "3D network" tab)

A Three.js scene that renders the network as the agent plays: conv layers as
stacked feature-map planes, the dense head as a 3D node-link graph with
particles traveling the strongest connections. Orbit with the mouse, zoom
with the wheel, pan with right-drag. It updates in lockstep with the 2D
board — both are driven by the same WebSocket step message, so pausing a
demo game freezes the exact forward pass that chose that move.

**Accuracy contract — every visual is bound to a real tensor value from the
step being shown; nothing is a decorative loop:**

| Visual property | Bound value (exact source) |
|---|---|
| Feature-map plane texel color | that cell's **pre-ReLU activation** from this forward pass, normalized by the layer's max abs (warm = positive/fired, blue = negative → zeroed by ReLU, dark ≈ 0) |
| Plane border color | inference: per-map mean of positive (post-ReLU) activation this step; gradient mode: per-out-channel Σ\|∂L/∂W\| of that conv kernel from the latest optimizer step |
| Dense node hue (fc1 / head) | **sign of the pre-activation** (fc1: fired vs ReLU-suppressed; head: sign of the Q-value) |
| Dense node brightness & size | \|pre-activation\| / layer max for this pass — a unit at 0 stays small and dark |
| Edge existence, line width & brightness | the **exact contribution** flowing on that connection: conv3→fc1 edge (m→j) = Σ_p W1[j,m,p]·relu(conv3[m,p]); fc1→head edge (i→j) = relu(fc1[i])·W2[j,i]. These summands literally add up (plus bias) to the target's pre-activation — verified in `scripts/smoke_test.py::test_viz3d` |
| Hover tooltip | the raw decoded value behind whatever is under the pointer — a Q-value, an fc1 pre-activation, a feature-map cell, or a single edge's contribution/gradient |
| Particle size & speed | \|contribution\| of that edge, normalized by the step's max |
| Particle color | sign of the contribution (warm +, cool −) |
| Particle direction | forward (source→target) in inference mode; **reversed** in gradient mode |
| Gradient-mode edge intensity | fc1→head: the actual ∂L/∂W2[j,i] from the most recent optimizer step; conv3→fc1: Σ_p\|∂L/∂W1[j,m,p]\| per connection. (Gradients come from a replay **batch**, not the on-screen board — that's what training actually updates on) |
| White torus marker | the argmax head node = the cell fired at this step (the head layer is laid out 10×10 exactly like the board) |
| "Weight wireframe" toggle | top 1,200 \|W2\| entries as static wires (color = sign of the weight, brightness = \|w\|/max) — the network's wiring, fetched once per model load from `/api/viz3d/static` |
| A neuron that didn't fire | pre-ReLU ≤ 0 ⇒ contribution is **exactly 0** on every outgoing edge ⇒ no particles, dark node — by construction, not by styling |

The only time-driven animation is a particle's *position along its edge*;
its existence, brightness, size and speed all come from the numbers above.
The de-sync phase offset per particle is a deterministic constant (golden
ratio × index), not data.

**Data feed:** activations and both contribution matrices are int8-quantized
(symmetric, per-array scale = max|v|; ≤ 0.8 % error) and base64-packed by
`agent/viz3d.py` — a full snapshot is ~100 KB (~280 KB with gradients)
instead of several MB of float JSON. Contributions are recomputed against
the *current* weights every step, so nothing goes stale while training
mutates the network. The 3D feed has its own throttle (`viz3d_every` in
`POST /api/stream/config`, or the "stream 3D every" control), independent of
the 2D board stream, plus a client-side "apply every Nth update" control and
a per-layer focus/LOD selector. Connection filtering keeps the top-k (k=4,
configurable 1–12) strongest active connections per target neuron,
intersected with a global "edge budget" of the N strongest edges in the
matrix (default 256) — rank-based, so visual density stays stable whether
the weights are flat (early training) or spiky (one dominant unit).
Selection has a 3-step hysteresis so borderline edges don't flicker in and
out (values shown are always the current ones; an edge whose contribution
hits exactly 0 still vanishes instantly). "All connections" lifts both
limits (still capped at the 4,096-edge pool). Hover any node, feature-map
cell or connection to read the exact value it renders; click a Q-head node
to isolate the edge paths that produced that Q-value (click empty space to
clear). Panel settings persist in localStorage.

**Modes:** *Inference* shows the forward pass that chose the current move
(works for live training and demo). *Gradients* replays the same geometry
with per-connection weight gradients flowing backward — only available on
training steps, since demo games never run the optimizer.

**Why not TensorSpace.js:** it's built to load a TF/Keras model and run its
own in-browser inference, then visualize that; it has no supported path for
binding externally streamed per-step activations from a PyTorch training
loop. A from-scratch Three.js scene (pooled geometries + typed-array
updates + GPU-side particle motion, ~one render pass) binds directly to the
stream and stays cheap at this network's scale.

## Training notes

- Reward shaping: +1 per hit, +2 sink bonus, +10 win, −10 loss, −0.1 per
  shot (constants at the top of `env/battleship_env.py`).
- Opponents: `random` (easy), `hunt_target` (strong fixed heuristic with
  parity search — beat this and the agent is genuinely good), `self`
  (frozen copy of the policy, refreshed periodically), `none` (solo
  board-clearing; performance = shots to clear).
- The DQN is vanilla (target network + masked epsilon-greedy + masked max-Q
  bootstrapping). `agent/model.py` notes where a PPO actor-critic head could
  be swapped in; `agent/trainer.py` notes the two-line double-DQN change.
- Checkpoints land in `checkpoints/` (`ckpt_ep*.pt` + `latest.pt`) every N
  episodes and on stop. Metrics persist to `runs/metrics.jsonl` and reload
  into the dashboard on restart.
- CPU training is roughly a few episodes per second early on; expect the
  win rate vs `random` to move meaningfully after a few thousand episodes.

## Tests

```bash
python scripts/smoke_test.py
```

covers env rules, the hunt/target baseline, model introspection shapes, the
3D-feed contract (quantization error bounds; contribution matrices summing
exactly to the next layer's pre-activations; ReLU-dead units having zero
outgoing edges; real gradients), a short real training run, checkpoint→demo
round-trip, and the embedding projection.
