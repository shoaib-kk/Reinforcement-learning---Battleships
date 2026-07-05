"""End-to-end backend smoke test (no server): env rules, opponents, model
introspection, a short training run, checkpoint save/load, demo stepping,
and the embedding projection. Run from the repo root:

    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_env():
    from env import BattleshipEnv, FLEET
    from agent.opponents import HuntTargetBot

    env = BattleshipEnv(opponent=None, seed=0)
    obs = env.reset()
    assert obs.shape == (3, 10, 10) and obs[0].sum() == 100
    total_ship_cells = sum(FLEET.values())
    hits = 0
    shots = 0
    rng = np.random.default_rng(1)
    done = False
    while not done:
        mask = env.action_mask()
        a = int(rng.choice(np.flatnonzero(mask)))
        obs, r, done, info = env.step(a)
        shots += 1
        if info["result"] in ("hit", "sunk"):
            hits += 1
    assert hits == total_ship_cells, f"expected {total_ship_cells} hits, got {hits}"
    assert info["raw"]["win"] and info["raw"]["shots_to_win"] == shots
    assert not info["action_mask"].all()

    # vs hunt/target opponent: someone must win
    env2 = BattleshipEnv(opponent=HuntTargetBot(seed=2), seed=3)
    env2.reset()
    done = False
    while not done:
        mask = env2.action_mask()
        a = int(rng.choice(np.flatnonzero(mask)))
        _, _, done, info = env2.step(a)
    assert info["raw"]["win"] or info["raw"]["loss"]
    print("  env ok — random play wins solo in", shots, "shots")


def test_hunt_target_efficiency():
    from env import BattleshipEnv
    from agent.opponents import HuntTargetBot

    # The heuristic bot should finish a solo board far faster than random.
    shots = []
    for seed in range(10):
        env = BattleshipEnv(opponent=None, seed=seed)
        env.reset()
        bot = HuntTargetBot(seed=seed)
        bot.reset()
        done = False
        n = 0
        while not done:
            a = bot.act(env.action_mask())
            _, _, done, info = env.step(a)
            bot.observe(a, info["result"])
            n += 1
        shots.append(n)
    avg = float(np.mean(shots))
    assert avg < 75, f"hunt/target too weak: avg {avg} shots"
    print(f"  hunt/target ok — avg {avg:.1f} shots to clear a board")


def test_model_and_introspection():
    import torch
    from agent.model import DQN

    m = DQN(in_channels=3)
    x = torch.rand(2, 3, 10, 10)
    q = m(x)
    assert q.shape == (2, 100)
    q1, acts = m.forward_with_activations(x[:1])
    assert set(acts) == {"conv1", "conv2", "conv3", "fc1", "head"}
    assert acts["conv1"].shape == (1, 32, 10, 10)
    sal = m.saliency(x[:1], 42)
    assert sal.shape == (10, 10) and 0 <= sal.min() and sal.max() <= 1.0
    arch = m.architecture_summary()
    assert arch[0]["name"] == "input" and arch[-1]["out_shape"] == [100]
    hists = m.weight_histograms()
    assert len(hists) == 5 and len(hists[0]["weights"]["counts"]) == 33
    print("  model introspection ok —", sum(l["params"] for l in arch), "params")


def test_viz3d():
    """The 3D feed must be exact: quantization error bounded, and the edge
    'contribution' matrices must literally sum to the next layer's
    pre-activation (minus bias) — that's what makes the viz verifiable."""
    import base64

    import torch
    from agent import viz3d
    from agent.model import DQN

    def dec(t):
        raw = np.frombuffer(base64.b64decode(t["b64"]), dtype=np.int8)
        return (raw.astype(np.float32) * t["scale"] / 127).reshape(t["shape"])

    m = DQN(in_channels=3)
    x = torch.rand(1, 3, 10, 10)
    q, acts = m.forward_with_activations(x)
    p = viz3d.forward_payload(m, x[0].numpy(), acts)

    # quantization round-trip within one int8 step
    head = dec(p["layers"]["head"])
    assert head.shape == (100,)
    assert np.max(np.abs(head - q[0].numpy())) <= p["layers"]["head"]["scale"] / 127
    assert dec(p["contrib"]["conv3_fc1"]).shape == (64, 512)
    assert dec(p["contrib"]["fc1_head"]).shape == (100, 512)

    # exactness (float, pre-quantization): contributions + bias == pre-act
    a3 = torch.relu(acts["conv3"][0]).reshape(64, -1)
    w1 = m.fc1.weight.view(512, 64, -1)
    contrib = torch.einsum("jmp,mp->mj", w1, a3)
    assert torch.allclose(contrib.sum(0) + m.fc1.bias, acts["fc1"][0], atol=1e-4)
    contrib2 = torch.relu(acts["fc1"][0]).unsqueeze(0) * m.head.weight
    assert torch.allclose(contrib2.sum(1) + m.head.bias, acts["head"][0], atol=1e-4)

    # a ReLU-dead fc1 unit must produce an all-zero outgoing edge row
    dead = (acts["fc1"][0] <= 0).nonzero()
    if len(dead):
        assert contrib2[:, dead[0, 0]].abs().max() == 0

    # gradients from a real backward pass
    m.zero_grad()
    m(x).sum().backward()
    g = viz3d.grad_payload(m)
    assert g is not None and dec(g["fc1_head"]).shape == (100, 512)
    assert dec(g["conv3_fc1"]).shape == (64, 512)
    assert set(g["conv_mags"]) == {"conv1", "conv2", "conv3"}

    st = viz3d.static_payload(m, "test")
    assert dec(st["head_weights"]).shape == (100, 512)
    assert st["topology"][0]["name"] == "input"
    print("  viz3d ok — contributions reconstruct pre-activations exactly")


def test_training_and_demo():
    import json
    import tempfile

    from agent.trainer import Trainer, TrainConfig
    from server.demo import DemoGame
    from server.introspection import embedding_projection

    events = []
    with tempfile.TemporaryDirectory() as td:
        cfg = TrainConfig(
            opponent="random",
            max_episodes=3,
            warmup=50,
            viz_every=25,
            hist_every=1,
            checkpoint_every=2,
            eps_decay_steps=500,
            seed=0,
        )
        tr = Trainer(
            cfg,
            emit=events.append,
            checkpoint_dir=f"{td}/ckpt",
            metrics_path=f"{td}/metrics.jsonl",
        )
        tr.run()
        kinds = {e["type"] for e in events}
        assert {"status", "episode", "train_step", "weights"} <= kinds, kinds
        ep = [e for e in events if e["type"] == "episode"]
        assert len(ep) == 3 and ep[-1]["win_rate"] is not None
        assert ep[-1]["loss"] is not None, "optimizer never ran"
        step = next(e for e in events if e["type"] == "train_step")
        json.dumps(step)  # payload must be JSON-serializable
        assert len(step["q_values"]) == 100
        assert step["activations"]["conv1"]["kind"] == "conv"
        assert "viz3d" in step and "conv3_fc1" in step["viz3d"]["contrib"]
        # gradients appear once the optimizer has run
        grad_steps = [e for e in events if e["type"] == "train_step" and "grads" in e.get("viz3d", {})]
        assert grad_steps, "no train_step carried gradients"

        # checkpoint -> demo game
        path = tr.save_checkpoint()
        demo = DemoGame()
        demo.load_checkpoint(path)
        demo.reset(opponent="random", seed=1)
        for _ in range(5):
            p = demo.step()
        json.dumps(p)
        assert len(p["saliency"]) == 10 and len(p["q_values"]) == 100
        assert "viz3d" in p and "grads" not in p["viz3d"]  # demo = inference only
        json.dumps(p)
        assert demo.export()["n_steps"] == 5

        # embeddings from the real buffer
        emb = embedding_projection(tr.model, tr.buffer, n=100)
        assert emb["n"] > 0 and len(emb["points"][0]) == 3

        # self-play trainer constructs and runs one episode
        cfg2 = TrainConfig(opponent="self", max_episodes=1, warmup=10, seed=0)
        tr2 = Trainer(cfg2, checkpoint_dir=f"{td}/ckpt2", metrics_path=f"{td}/m2.jsonl")
        tr2.run()
        assert tr2.episode == 1
    print("  trainer + demo + embeddings + self-play ok")


if __name__ == "__main__":
    test_env()
    test_hunt_target_efficiency()
    test_model_and_introspection()
    test_viz3d()
    test_training_and_demo()
    print("ALL SMOKE TESTS PASSED")
