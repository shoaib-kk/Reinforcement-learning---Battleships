"""Network-introspection helpers that don't belong to the model itself."""

from __future__ import annotations

import numpy as np
import torch


def embedding_projection(model, buffer, n: int = 400, device: str = "cpu") -> dict:
    """2D PCA of replay-buffer state embeddings (fc1 features), colored by
    episode outcome. Shows whether the net's internal representation
    separates "winning" board states from "losing" ones as training goes on.

    PCA is done with a plain centered SVD — no sklearn dependency. t-SNE
    would give prettier clusters but is far slower; PCA is fine for a live
    view.
    """
    if len(buffer) < 10:
        return {"points": [], "explained": [], "n": 0}
    states, outcomes = buffer.sample_states(n)
    with torch.no_grad():
        feats = model.features(torch.as_tensor(states, device=device)).cpu().numpy()
    centered = feats - feats.mean(axis=0, keepdims=True)
    # Economy SVD: components are rows of Vt.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    pts = centered @ vt[:2].T
    scale = np.abs(pts).max() or 1.0
    pts = pts / scale
    var = s**2
    explained = (var[:2] / var.sum()).tolist() if var.sum() > 0 else [0.0, 0.0]
    return {
        "points": [
            [round(float(x), 4), round(float(y), 4), int(o)]
            for (x, y), o in zip(pts, outcomes)
        ],
        "explained": [round(float(e), 4) for e in explained],
        "n": int(len(pts)),
    }
