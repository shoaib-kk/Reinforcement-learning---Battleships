// Decoding for the backend's symmetric int8 quantization (agent/viz3d.py):
// original value ≈ int8 / 127 * scale, where scale = max|value| of the array.

import type { Q8Tensor } from "../types";

export function decodeQ8(t: Q8Tensor): Float32Array {
  const bin = atob(t.b64);
  const n = bin.length;
  const out = new Float32Array(n);
  const k = t.scale / 127;
  for (let i = 0; i < n; i++) {
    let v = bin.charCodeAt(i);
    if (v > 127) v -= 256; // bytes are the two's-complement int8
    out[i] = v * k;
  }
  return out;
}
