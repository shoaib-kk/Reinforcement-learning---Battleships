"""Data feed for the 3D network visualization.

Everything here is a real number read out of one specific forward pass (or,
for gradients, one specific optimizer step) — the frontend binds visual
properties directly to these arrays and adds nothing decorative. The README
documents the exact visual→tensor mapping.

Encoding: arrays are symmetrically quantized to int8 (value ≈ q / 127 *
scale, scale = max|value| of that array) and base64-encoded. That keeps a
full per-step snapshot (all layer activations + both edge-contribution
matrices, ~50k values) around 100 KB instead of several MB of float JSON.

Contribution matrices — why they're exact, not aesthetic:
  fc1's pre-activation for unit j is  Σ_m Σ_p W1[j, m, p] · relu(conv3[m, p]) + b_j.
  We ship  contrib_c3_fc1[m, j] = Σ_p W1[j, m, p] · relu(conv3[m, p]) — the
  exact summand contributed by feature map m to unit j in THIS forward pass.
  Likewise  contrib_fc1_head[j, i] = relu(fc1[i]) · W2[j, i]  is the exact
  term i contributes to Q-value j. An edge/particle in the 3D scene renders
  precisely one of these numbers. They are recomputed against the *current*
  weights every step, so nothing goes stale while training updates weights.

Note: the model's forward hooks capture module outputs, i.e. PRE-ReLU
values. Sign therefore encodes fired-vs-suppressed: a negative conv/fc1
value is exactly a neuron ReLU zeroed out — it contributes nothing
downstream, and the scene must show it dark with no outgoing particles.
"""

from __future__ import annotations

import base64

import numpy as np
import torch

LAYER_ORDER = ("conv1", "conv2", "conv3", "fc1", "head")


def q8(arr) -> dict:
    """Symmetric int8 quantization, base64-packed. Decode: q/127*scale."""
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    a = np.asarray(arr, dtype=np.float32)
    scale = float(np.abs(a).max())
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    q = np.clip(np.round(a / scale * 127.0), -127, 127).astype(np.int8)
    return {
        "shape": list(a.shape),
        "scale": round(scale, 6),
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
    }


@torch.no_grad()
def forward_payload(model, obs: np.ndarray, acts: dict[str, torch.Tensor]) -> dict:
    """Quantized snapshot of one forward pass.

    obs:  the exact input the decision was made on, (C, 10, 10).
    acts: pre-activation module outputs captured by the model's hooks,
          each shaped (1, ...).
    """
    layers = {"input": q8(obs)}
    for name in LAYER_ORDER:
        layers[name] = q8(acts[name][0])

    # Exact per-edge contributions (see module docstring).
    a3 = torch.relu(acts["conv3"][0])                 # (C, 10, 10) as it flowed on
    n_maps = a3.shape[0]
    a3 = a3.reshape(n_maps, -1)                       # (C, P), flatten matches model
    w1 = model.fc1.weight.view(model.fc1.out_features, n_maps, -1)  # (512, C, P)
    contrib_c3_fc1 = torch.einsum("jmp,mp->mj", w1, a3)             # (C, 512)

    f1 = torch.relu(acts["fc1"][0])                   # (512,)
    contrib_fc1_head = f1.unsqueeze(0) * model.head.weight          # (100, 512)

    return {
        "layers": layers,
        "contrib": {
            "conv3_fc1": q8(contrib_c3_fc1),
            "fc1_head": q8(contrib_fc1_head),
        },
    }


def grad_payload(model) -> dict | None:
    """Quantized gradients from the most recent optimizer step (a replay
    batch, not the on-screen board — that distinction is documented).

    Edge matrices keep the same orientation as the forward contributions so
    the frontend reuses the same geometry with reversed particle direction:
      conv3_fc1[m, j] = Σ_p |∂L/∂W1[j, m, p]|   (per-connection grad mass)
      fc1_head[j, i]  = ∂L/∂W2[j, i]            (the actual per-edge grad)
    """
    gh = model.head.weight.grad
    gf = model.fc1.weight.grad
    if gh is None or gf is None:
        return None
    n_maps = model.conv3.out_channels
    gf_map = gf.abs().view(gf.shape[0], n_maps, -1).sum(dim=-1).T  # (C, 512)
    conv_mags = {}
    for name in ("conv1", "conv2", "conv3"):
        g = getattr(model, name).weight.grad
        if g is not None:
            conv_mags[name] = q8(g.abs().sum(dim=(1, 2, 3)))  # per out-channel
    return {
        "conv3_fc1": q8(gf_map),
        "fc1_head": q8(gh),
        "conv_mags": conv_mags,
    }


@torch.no_grad()
def static_payload(model, model_version: str) -> dict:
    """Sent once per model load / checkpoint change (REST, not per step):
    layer topology for scene layout plus the head weight matrix, which the
    frontend's optional 'weight wireframe' renders (edge color = sign of
    W2[j, i], brightness = |W2[j, i]|)."""
    return {
        "model_version": model_version,
        "topology": model.architecture_summary(),
        "head_weights": q8(model.head.weight),
    }
