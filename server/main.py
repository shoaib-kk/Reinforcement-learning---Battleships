"""FastAPI app: REST control plane + WebSocket stream.

Run with:  uvicorn server.main:app --port 8000

Layout:
  WS  /ws                          — live stream (train_step / episode /
                                     weights / status / demo events)
  POST /api/train/start            — start training in a background thread
  POST /api/train/stop             — stop it (checkpoint is saved on stop)
  GET  /api/train/status
  POST /api/stream/config          — {"viz_every": N} live-viz throttle
  GET  /api/metrics?after=N        — episode metrics history (for charts)
  GET  /api/checkpoints            — list saved checkpoints
  GET  /api/model/architecture     — layer/shape list for the network diagram
  GET  /api/introspection/weights  — per-layer weight/grad histograms (live)
  GET  /api/introspection/embeddings — 2D PCA of replay-buffer embeddings
  GET  /api/replay-buffer/sample   — a few raw transitions
  POST /api/demo/reset             — {"source": "latest"|"live"|"fresh"|<file>,
                                      "opponent": "random"|...}
  POST /api/demo/step              — advance the demo game one shot
  GET  /api/demo/export            — full replay JSON of the demo game

Training runs in a plain thread and pushes events through a thread-safe
queue into the asyncio broadcaster, so serving (including demo stepping)
never blocks the training loop and vice versa.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agent import viz3d
from agent.model import DQN
from agent.trainer import Trainer, TrainConfig
from server.demo import DemoGame
from server.introspection import embedding_projection

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
METRICS_PATH = ROOT / "runs" / "metrics.jsonl"
FRONTEND_DIST = ROOT / "frontend" / "dist"


class Hub:
    """Fan-out of trainer/demo events to every connected WebSocket.

    emit_threadsafe() may be called from any thread (the trainer thread
    uses it); messages funnel through an asyncio queue drained by a single
    broadcaster task. When the queue backs up, oldest messages are dropped —
    a slow viewer should never stall training.
    """

    def __init__(self, maxsize: int = 500):
        self.clients: set[WebSocket] = set()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.loop.create_task(self._broadcast_loop())

    def emit_threadsafe(self, msg: dict) -> None:
        if self.loop is None:
            return
        def _put() -> None:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self.queue.put_nowait(msg)
        self.loop.call_soon_threadsafe(_put)

    async def _broadcast_loop(self) -> None:
        while True:
            msg = await self.queue.get()
            if not self.clients:
                continue
            data = json.dumps(msg)
            dead = []
            for ws in list(self.clients):
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)


class AppState:
    def __init__(self):
        self.hub = Hub()
        self.trainer: Trainer | None = None
        self.trainer_thread: threading.Thread | None = None
        self.demo = DemoGame()
        self.demo_lock = threading.Lock()


state = AppState()
app = FastAPI(title="Battleship RL")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev convenience; frontend runs on another port
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    state.hub.start()


# --------------------------------------------------------------------------
# WebSocket stream
# --------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    state.hub.clients.add(ws)
    try:
        if state.trainer:
            await ws.send_text(json.dumps({"type": "status", **state.trainer.status()}))
        while True:
            await ws.receive_text()  # we don't expect client messages; keepalive
    except WebSocketDisconnect:
        pass
    finally:
        state.hub.clients.discard(ws)


# --------------------------------------------------------------------------
# Training control
# --------------------------------------------------------------------------

def _training_active() -> bool:
    return state.trainer_thread is not None and state.trainer_thread.is_alive()


@app.post("/api/train/start")
async def train_start(body: dict | None = None) -> dict:
    if _training_active():
        raise HTTPException(409, "training is already running")
    body = body or {}
    resume = bool(body.pop("resume", False))
    known = {f for f in TrainConfig.__dataclass_fields__}
    unknown = set(body) - known
    if unknown:
        raise HTTPException(422, f"unknown config fields: {sorted(unknown)}")
    cfg = TrainConfig(**body)
    trainer = Trainer(
        cfg,
        emit=state.hub.emit_threadsafe,
        checkpoint_dir=str(CHECKPOINT_DIR),
        metrics_path=str(METRICS_PATH),
    )
    latest = CHECKPOINT_DIR / "latest.pt"
    if resume and latest.exists():
        trainer.load_checkpoint(str(latest))
    state.trainer = trainer
    state.trainer_thread = threading.Thread(target=trainer.run, daemon=True)
    state.trainer_thread.start()
    return {"ok": True, "resumed": resume and latest.exists(), **trainer.status()}


@app.post("/api/train/stop")
async def train_stop() -> dict:
    if not state.trainer or not _training_active():
        return {"ok": True, "note": "training was not running"}
    state.trainer.stop_event.set()
    state.trainer_thread.join(timeout=30)
    return {"ok": True, **state.trainer.status()}


@app.get("/api/train/status")
async def train_status() -> dict:
    if state.trainer:
        return {"active": _training_active(), **state.trainer.status()}
    return {"active": False, "running": False, "episode": 0, "total_steps": 0}


@app.post("/api/stream/config")
async def stream_config(body: dict) -> dict:
    """Stream throttles. viz_every gates board/2D messages (in env steps);
    viz3d/viz3d_every gate the heavier 3D tensors independently (in emitted
    messages), so the 3D feed can be thinned without slowing the board."""
    out: dict = {"ok": True}
    if "viz_every" in body:
        n = max(1, int(body["viz_every"]))
        if state.trainer:
            state.trainer.set_viz_every(n)
        out["viz_every"] = n
    if "viz3d" in body or "viz3d_every" in body:
        enabled = body.get("viz3d")
        every = body.get("viz3d_every")
        if state.trainer:
            state.trainer.set_viz3d(enabled, every)
        if enabled is not None:
            out["viz3d"] = bool(enabled)
        if every is not None:
            out["viz3d_every"] = max(1, int(every))
    return out


# --------------------------------------------------------------------------
# Metrics & checkpoints
# --------------------------------------------------------------------------

def _metrics_from_disk() -> list[dict]:
    out: list[dict] = []
    if METRICS_PATH.exists():
        with METRICS_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return out


@app.get("/api/metrics")
async def get_metrics(after: int = 0, limit: int = 5000) -> dict:
    records = state.trainer.metrics if state.trainer else _metrics_from_disk()
    chunk = records[after : after + limit]
    return {"total": len(records), "after": after, "records": chunk}


@app.get("/api/checkpoints")
async def list_checkpoints() -> dict:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    items = []
    for p in sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime):
        items.append({"file": p.name, "mtime": p.stat().st_mtime, "bytes": p.stat().st_size})
    return {"checkpoints": items}


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

def _some_model() -> DQN:
    """Best available model: live trainer's, else demo's, else fresh."""
    if state.trainer:
        return state.trainer.model
    state.demo.ensure_model()
    return state.demo.model


@app.get("/api/model/architecture")
async def architecture() -> dict:
    model = _some_model()
    lock = state.trainer.lock if state.trainer else threading.Lock()
    with lock:
        return {"layers": model.architecture_summary()}


@app.get("/api/introspection/weights")
async def weights() -> dict:
    model = _some_model()
    lock = state.trainer.lock if state.trainer else threading.Lock()
    with lock:
        return {"layers": model.weight_histograms()}


@app.get("/api/introspection/embeddings")
async def embeddings(n: int = 400) -> dict:
    if not state.trainer:
        return {"points": [], "explained": [], "n": 0, "note": "no trainer yet"}
    with state.trainer.lock:
        return embedding_projection(
            state.trainer.model, state.trainer.buffer, n=n, device=state.trainer.device
        )


@app.get("/api/viz3d/static")
async def viz3d_static(model: str = "train") -> dict:
    """Per-model static data for the 3D scene: layer topology + the head
    weight matrix. Sent once per model load / checkpoint change — the
    frontend refetches when model_version changes, not per step."""
    if model == "demo":
        state.demo.ensure_model()
        with state.demo_lock:
            return viz3d.static_payload(state.demo.model, f"demo:{state.demo.source}")
    if not state.trainer:
        state.demo.ensure_model()
        with state.demo_lock:
            return viz3d.static_payload(state.demo.model, "fresh")
    with state.trainer.lock:
        return viz3d.static_payload(
            state.trainer.model, f"trainer:step{state.trainer.total_steps}"
        )


@app.get("/api/replay-buffer/sample")
async def buffer_sample(n: int = 8) -> dict:
    if not state.trainer:
        return {"transitions": []}
    return {"transitions": state.trainer.buffer.snapshot(n)}


# --------------------------------------------------------------------------
# Demo game (checkpoint / live-weights playback)
# --------------------------------------------------------------------------

def _resolve_checkpoint(name: str) -> Path:
    p = (CHECKPOINT_DIR / name).resolve()
    if CHECKPOINT_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, f"checkpoint not found: {name}")
    return p


@app.post("/api/demo/reset")
async def demo_reset(body: dict | None = None) -> dict:
    body = body or {}
    source = body.get("source", "keep")   # keep current weights by default
    opponent = body.get("opponent", "random")
    seed = body.get("seed")
    with state.demo_lock:
        if source == "latest":
            latest = CHECKPOINT_DIR / "latest.pt"
            if not latest.exists():
                raise HTTPException(404, "no checkpoint saved yet")
            state.demo.load_checkpoint(str(latest))
        elif source == "live":
            if not state.trainer:
                raise HTTPException(409, "no live trainer to copy weights from")
            with state.trainer.lock:
                sd = {k: v.clone() for k, v in state.trainer.model.state_dict().items()}
            state.demo.load_live(sd, state.trainer.model.in_channels)
        elif source == "fresh":
            state.demo.model = None
            state.demo.ensure_model()
        elif source != "keep":
            state.demo.load_checkpoint(str(_resolve_checkpoint(source)))
        payload = state.demo.reset(opponent=opponent, seed=seed)
    state.hub.emit_threadsafe(payload)
    return payload


@app.post("/api/demo/step")
async def demo_step(body: dict | None = None) -> dict:
    n = int((body or {}).get("n", 1))
    with state.demo_lock:
        payload = {}
        for _ in range(max(1, n)):
            payload = state.demo.step()
            if payload.get("error") or payload.get("done"):
                break
    state.hub.emit_threadsafe(payload)
    return payload


@app.get("/api/demo/export")
async def demo_export() -> JSONResponse:
    with state.demo_lock:
        data = state.demo.export()
    return JSONResponse(
        data,
        headers={"Content-Disposition": "attachment; filename=battleship_replay.json"},
    )


# --------------------------------------------------------------------------
# Static frontend (after `npm run build`)
# --------------------------------------------------------------------------

if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
