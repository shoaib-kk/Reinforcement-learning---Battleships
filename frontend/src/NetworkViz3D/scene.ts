// Three.js scene for the live network readout.
//
// Every visual property binds to a real number from one specific forward
// (or backward) pass — see the mapping table in the README. Nothing here
// runs a decorative loop: the only time-driven animation is particle
// *position along its edge*, and each particle's existence, speed, size and
// color all come from the actual (activation × weight) contribution (or
// weight gradient, in gradient mode) it represents. A ReLU-dead neuron has
// contribution exactly 0 on every outgoing edge, so it gets no particles
// and stays dark by construction.
//
// Performance: all geometry is pooled and preallocated once per model
// topology (plane textures, node InstancedMeshes, edge/particle buffers
// with drawRange) — per step we only rewrite typed arrays and flag them
// dirty. Particle motion is computed in the vertex shader from a time
// uniform, so a paused stream costs ~zero CPU.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { decodeQ8 } from "./decode";
import type { ArchLayer, Q8Tensor, Viz3DStatic, Viz3DStep } from "../types";

// -- palette (matches the app's dark theme; diverging warm/cool per spec) --
const BG = 0x0d0d0d;
const WARM = { r: 1.0, g: 0.48, b: 0.16 };  // positive activation / weight
const COOL = { r: 0.22, g: 0.53, b: 0.9 };  // negative (pre-ReLU => suppressed)
const DIM = { r: 0.13, g: 0.13, b: 0.15 };  // ~zero: visibly dark, not pulsing

/** t in [-1, 1] -> rgb. Magnitude drives brightness away from DIM. */
function diverge(t: number, out: { r: number; g: number; b: number }) {
  const m = Math.min(1, Math.abs(t));
  const hue = t >= 0 ? WARM : COOL;
  out.r = DIM.r + (hue.r - DIM.r) * m;
  out.g = DIM.g + (hue.g - DIM.g) * m;
  out.b = DIM.b + (hue.b - DIM.b) * m;
  return out;
}

export interface SceneOptions {
  mode: "inference" | "gradients";
  k: number;           // top-k strongest connections per target neuron
  threshold: number;   // min |value| as a fraction of the matrix max
  showAll: boolean;    // ignore k, keep threshold (still capped by pool)
  weightWireframe: boolean; // static head-weight wires (from /api/viz3d/static)
  focus: string;       // all | input | conv1 | conv2 | conv3 | dense
}

export const DEFAULT_OPTIONS: SceneOptions = {
  mode: "inference",
  k: 6,
  threshold: 0.05,
  showAll: false,
  weightWireframe: false,
  focus: "all",
};

const MAX_EDGES = 4096; // per edge set (pool size)
const PLANE = 2.4;      // feature-map plane size
const LAYER_Z: Record<string, number> = {
  input: 0, conv1: 9, conv2: 18, conv3: 27, fc1: 38, head: 50,
};

interface PlaneLayer {
  name: string;
  n: number;
  meshes: THREE.Mesh[];
  textures: THREE.DataTexture[];
  texData: Uint8Array[];
  centers: Float32Array; // (n*3) plane centers, used as edge endpoints
  frame: THREE.LineSegments; // per-map border, color = real per-map aggregate
  frameColors: Float32Array;
  group: THREE.Group;
}

interface NodeLayer {
  name: string;
  n: number;
  mesh: THREE.InstancedMesh;
  positions: Float32Array; // (n*3)
  group: THREE.Group;
}

interface EdgeSet {
  lines: THREE.LineSegments;
  linePos: Float32Array;
  lineCol: Float32Array;
  points: THREE.Points;
  pStart: Float32Array;
  pEnd: Float32Array;
  pCol: Float32Array;
  pSize: Float32Array;
  pSpeed: Float32Array;
  pPhase: Float32Array;
  count: number;
}

const PARTICLE_VERT = `
  uniform float uTime;
  attribute vec3 aEnd;
  attribute vec3 aColor;
  attribute float aSize;
  attribute float aSpeed;
  attribute float aPhase;
  varying vec3 vColor;
  void main() {
    float t = fract(aPhase + uTime * aSpeed);
    vec3 pos = mix(position, aEnd, t);
    vColor = aColor;
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = aSize * (160.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }
`;
const PARTICLE_FRAG = `
  varying vec3 vColor;
  void main() {
    float r = length(gl_PointCoord - vec2(0.5));
    if (r > 0.5) discard;
    float a = smoothstep(0.5, 0.1, r);
    gl_FragColor = vec4(vColor, a);
  }
`;

export class NetScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private clock = new THREE.Clock();
  private timeUniform = { value: 0 };
  private raf = 0;
  private resizeObs: ResizeObserver;
  private disposed = false;

  private planes: PlaneLayer[] = [];
  private fc1?: NodeLayer;
  private head?: NodeLayer;
  private edgeC3F?: EdgeSet;   // conv3 maps -> fc1 units
  private edgeF1H?: EdgeSet;   // fc1 units -> head units
  private wireframe?: THREE.LineSegments; // static head weights
  private actionMarker?: THREE.Mesh;
  private denseGroup = new THREE.Group();
  private built = false;

  private opts: SceneOptions = { ...DEFAULT_OPTIONS };
  private lastStep: Viz3DStep | null = null;
  private lastAction = -1;
  private tmp = { r: 0, g: 0, b: 0 };
  private dummy = new THREE.Object3D();

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(BG);
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500);
    // oblique default so the layer stack reads in depth instead of overlapping
    this.camera.position.set(58, 24, 92);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 25);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.scene.add(this.denseGroup);
    this.resize();
    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.animate();
  }

  // -- lifecycle -----------------------------------------------------------

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
      else if (mat) mat.dispose();
    });
    this.planes.forEach((p) => p.textures.forEach((t) => t.dispose()));
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }

  private resize() {
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 560;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private animate = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.animate);
    this.timeUniform.value += this.clock.getDelta();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  // -- build (once per topology) -------------------------------------------

  setStatic(st: Viz3DStatic) {
    if (!this.built) this.build(st.topology);
    this.buildWireframe(st.head_weights);
    if (this.lastStep) this.applyStep(); // re-render pending data
  }

  private build(topology: ArchLayer[]) {
    const shape = (name: string) =>
      topology.find((l) => l.name === name)?.out_shape ?? [];

    const grids: Record<string, [number, number]> = {
      input: [1, shape("input")[0] ?? 3],
      conv1: [4, 8],
      conv2: [8, 8],
      conv3: [8, 8],
    };
    for (const name of ["input", "conv1", "conv2", "conv3"]) {
      const n = shape(name)[0] ?? 0;
      if (n > 0) this.planes.push(this.buildPlaneLayer(name, n, grids[name]));
    }
    this.fc1 = this.buildNodeLayer("fc1", shape("fc1")[0] ?? 512, 16, 32, 0.85, 0.3);
    // head laid out exactly like the board: 10x10, row 1 at top
    this.head = this.buildNodeLayer("head", shape("head")[0] ?? 100, 10, 10, 1.7, 0.5);
    this.denseGroup.add(this.fc1.group, this.head.group);

    this.edgeC3F = this.buildEdgeSet();
    this.edgeF1H = this.buildEdgeSet();

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.95, 0.09, 8, 32),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    ring.visible = false;
    this.actionMarker = ring;
    this.denseGroup.add(ring);

    this.addLabels(topology);
    this.built = true;
    this.applyFocus();
  }

  private buildPlaneLayer(name: string, n: number, [rows, cols]: [number, number]): PlaneLayer {
    const group = new THREE.Group();
    const z = LAYER_Z[name];
    const pitch = PLANE + 0.55;
    const meshes: THREE.Mesh[] = [];
    const textures: THREE.DataTexture[] = [];
    const texData: Uint8Array[] = [];
    const centers = new Float32Array(n * 3);
    const geo = new THREE.PlaneGeometry(PLANE, PLANE);
    const framePos = new Float32Array(n * 8 * 3);
    const frameColors = new Float32Array(n * 8 * 3);

    for (let i = 0; i < n; i++) {
      const data = new Uint8Array(10 * 10 * 4);
      const tex = new THREE.DataTexture(data, 10, 10, THREE.RGBAFormat);
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.NearestFilter;
      const mat = new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geo, mat);
      const r = Math.floor(i / cols);
      const c = i % cols;
      const x = (c - (cols - 1) / 2) * pitch;
      const y = ((rows - 1) / 2 - r) * pitch;
      mesh.position.set(x, y, z);
      group.add(mesh);
      meshes.push(mesh);
      textures.push(tex);
      texData.push(data);
      centers.set([x, y, z], i * 3);

      // border frame: 4 segments per plane
      const hp = PLANE / 2 + 0.06;
      const corners = [
        [x - hp, y - hp], [x + hp, y - hp],
        [x + hp, y - hp], [x + hp, y + hp],
        [x + hp, y + hp], [x - hp, y + hp],
        [x - hp, y + hp], [x - hp, y - hp],
      ];
      corners.forEach(([cx, cy], j) => framePos.set([cx, cy, z], (i * 8 + j) * 3));
    }

    const frameGeo = new THREE.BufferGeometry();
    frameGeo.setAttribute("position", new THREE.BufferAttribute(framePos, 3));
    frameGeo.setAttribute("color", new THREE.BufferAttribute(frameColors, 3));
    const frame = new THREE.LineSegments(
      frameGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })
    );
    frame.frustumCulled = false;
    group.add(frame);
    this.scene.add(group);
    return { name, n, meshes, textures, texData, centers, frame, frameColors, group };
  }

  private buildNodeLayer(
    name: string, n: number, rows: number, cols: number, pitch: number, radius: number
  ): NodeLayer {
    const group = new THREE.Group();
    const z = LAYER_Z[name];
    const geo = new THREE.SphereGeometry(radius, 10, 8);
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial(), n);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const x = (c - (cols - 1) / 2) * pitch;
      const y = ((rows - 1) / 2 - r) * pitch;
      positions.set([x, y, z], i * 3);
      this.dummy.position.set(x, y, z);
      this.dummy.scale.setScalar(0.4);
      this.dummy.updateMatrix();
      mesh.setMatrixAt(i, this.dummy.matrix);
      mesh.setColorAt(i, new THREE.Color(DIM.r, DIM.g, DIM.b));
    }
    group.add(mesh);
    return { name, n, mesh, positions, group };
  }

  private buildEdgeSet(): EdgeSet {
    const linePos = new Float32Array(MAX_EDGES * 2 * 3);
    const lineCol = new Float32Array(MAX_EDGES * 2 * 3);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
    lineGeo.setAttribute("color", new THREE.BufferAttribute(lineCol, 3));
    lineGeo.setDrawRange(0, 0);
    const lines = new THREE.LineSegments(
      lineGeo,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.42,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    lines.frustumCulled = false;

    const pStart = new Float32Array(MAX_EDGES * 3);
    const pEnd = new Float32Array(MAX_EDGES * 3);
    const pCol = new Float32Array(MAX_EDGES * 3);
    const pSize = new Float32Array(MAX_EDGES);
    const pSpeed = new Float32Array(MAX_EDGES);
    const pPhase = new Float32Array(MAX_EDGES);
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pStart, 3));
    pGeo.setAttribute("aEnd", new THREE.BufferAttribute(pEnd, 3));
    pGeo.setAttribute("aColor", new THREE.BufferAttribute(pCol, 3));
    pGeo.setAttribute("aSize", new THREE.BufferAttribute(pSize, 1));
    pGeo.setAttribute("aSpeed", new THREE.BufferAttribute(pSpeed, 1));
    pGeo.setAttribute("aPhase", new THREE.BufferAttribute(pPhase, 1));
    pGeo.setDrawRange(0, 0);
    const points = new THREE.Points(
      pGeo,
      new THREE.ShaderMaterial({
        uniforms: { uTime: this.timeUniform },
        vertexShader: PARTICLE_VERT,
        fragmentShader: PARTICLE_FRAG,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    points.frustumCulled = false;
    this.denseGroup.add(lines, points);
    return { lines, linePos, lineCol, points, pStart, pEnd, pCol, pSize, pSpeed, pPhase, count: 0 };
  }

  private buildWireframe(weights: Q8Tensor) {
    if (this.wireframe) {
      this.denseGroup.remove(this.wireframe);
      this.wireframe.geometry.dispose();
      (this.wireframe.material as THREE.Material).dispose();
      this.wireframe = undefined;
    }
    if (!this.fc1 || !this.head) return;
    // Top |W2| connections as a dim static wireframe: color = sign of the
    // weight, brightness = |w| / max|w|. This is the network's wiring, not
    // this step's signal — off by default, toggle in the panel.
    const w = decodeQ8(weights); // (100, 512)
    const nH = weights.shape[0];
    const nF = weights.shape[1];
    const N = 1200;
    const idx = new Uint32Array(nH * nF);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    const byMag = Array.from(idx).sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).slice(0, N);
    const pos = new Float32Array(N * 2 * 3);
    const col = new Float32Array(N * 2 * 3);
    byMag.forEach((flat, e) => {
      const j = Math.floor(flat / nF);
      const i = flat % nF;
      pos.set(this.fc1!.positions.subarray(i * 3, i * 3 + 3), e * 6);
      pos.set(this.head!.positions.subarray(j * 3, j * 3 + 3), e * 6 + 3);
      const t = w[flat] / weights.scale;
      const c = diverge(t * 0.8, this.tmp);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], e * 6);
    });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    this.wireframe = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.16,
        blending: THREE.AdditiveBlending, depthWrite: false,
      })
    );
    this.wireframe.frustumCulled = false;
    this.wireframe.visible = this.opts.weightWireframe;
    this.denseGroup.add(this.wireframe);
  }

  private addLabels(topology: ArchLayer[]) {
    for (const l of topology) {
      if (!(l.name in LAYER_Z)) continue;
      const text =
        l.name === "head" ? "Q head 10×10" : `${l.name} ${l.out_shape.join("×")}`;
      const cv = document.createElement("canvas");
      cv.width = 256;
      cv.height = 48;
      const ctx = cv.getContext("2d")!;
      ctx.font = "24px system-ui, sans-serif";
      ctx.fillStyle = "#898781";
      ctx.textAlign = "center";
      ctx.fillText(text, 128, 32);
      const tex = new THREE.CanvasTexture(cv);
      const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })
      );
      sprite.scale.set(9, 1.7, 1);
      const yTop = l.name === "fc1" ? 8.5 : l.name === "head" ? 10 : 8;
      sprite.position.set(0, -yTop - 2, LAYER_Z[l.name]);
      this.scene.add(sprite);
    }
  }

  // -- per-step data binding -------------------------------------------------

  setStep(step: Viz3DStep, action: number) {
    this.lastStep = step;
    this.lastAction = action;
    if (this.built) this.applyStep();
  }

  setOptions(o: SceneOptions) {
    const focusChanged = o.focus !== this.opts.focus;
    this.opts = { ...o };
    if (this.wireframe) this.wireframe.visible = o.weightWireframe;
    if (focusChanged) this.applyFocus();
    if (this.lastStep && this.built) this.applyStep(); // re-filter edges
  }

  private applyStep() {
    const step = this.lastStep!;
    const gradMode = this.opts.mode === "gradients" && !!step.grads;

    for (const layer of this.planes) {
      const t = step.layers[layer.name];
      if (t) this.updatePlanes(layer, t, gradMode ? step.grads!.conv_mags[layer.name] : undefined);
    }
    if (this.fc1 && step.layers.fc1) this.updateNodes(this.fc1, step.layers.fc1);
    if (this.head && step.layers.head) this.updateNodes(this.head, step.layers.head);

    // action marker sits on the chosen head node — argmax of this pass
    if (this.actionMarker && this.head && this.lastAction >= 0) {
      const p = this.head.positions;
      const a = this.lastAction;
      this.actionMarker.position.set(p[a * 3], p[a * 3 + 1], p[a * 3 + 2]);
      this.actionMarker.visible = true;
    }

    const conv3 = this.planes.find((p) => p.name === "conv3");
    if (conv3 && this.fc1 && this.edgeC3F) {
      const m = gradMode ? step.grads!.conv3_fc1 : step.contrib.conv3_fc1;
      this.updateEdges(this.edgeC3F, m, conv3.centers, this.fc1.positions, gradMode);
    }
    if (this.fc1 && this.head && this.edgeF1H) {
      const m = gradMode ? step.grads!.fc1_head : step.contrib.fc1_head;
      // matrix is (target=head j, source=fc1 i) — flag transposed layout
      this.updateEdges(this.edgeF1H, m, this.fc1.positions, this.head.positions, gradMode, true);
    }
  }

  /** Feature-map plane: each texel = that cell's pre-ReLU activation,
   *  normalized by the layer max (the q8 scale). Frame color = per-map mean
   *  |post-ReLU activation| (inference) or per-map |grad| mass (gradients). */
  private updatePlanes(layer: PlaneLayer, t: Q8Tensor, gradMags?: Q8Tensor) {
    const vals = decodeQ8(t); // (n, 10, 10), values in [-scale, scale]
    const inv = t.scale > 0 ? 1 / t.scale : 1;
    const frameVals = new Float32Array(layer.n);
    for (let m = 0; m < layer.n; m++) {
      const data = layer.texData[m];
      let acc = 0;
      for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 10; c++) {
          const v = vals[m * 100 + r * 10 + c] * inv;
          if (v > 0) acc += v;
          const col = diverge(v, this.tmp);
          // DataTexture row 0 renders at the bottom; board row 0 is the top
          const o = ((9 - r) * 10 + c) * 4;
          data[o] = col.r * 255;
          data[o + 1] = col.g * 255;
          data[o + 2] = col.b * 255;
          data[o + 3] = 255;
        }
      }
      frameVals[m] = acc / 100;
      layer.textures[m].needsUpdate = true;
    }
    let frameScale = 0;
    let source: Float32Array = frameVals;
    if (gradMags) {
      source = decodeQ8(gradMags);
      frameScale = gradMags.scale;
    } else {
      for (let i = 0; i < layer.n; i++) frameScale = Math.max(frameScale, frameVals[i]);
    }
    const fInv = frameScale > 0 ? 1 / frameScale : 1;
    for (let m = 0; m < layer.n; m++) {
      const c = diverge(Math.abs(source[m]) * fInv * (gradMags ? -1 : 1), this.tmp);
      for (let j = 0; j < 8; j++) layer.frameColors.set([c.r, c.g, c.b], (m * 8 + j) * 3);
    }
    (layer.frame.geometry.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
  }

  /** Node color = sign of pre-activation (warm fired / cool ReLU-suppressed;
   *  for the head, sign of Q). Brightness and size = |value| / layer max. */
  private updateNodes(layer: NodeLayer, t: Q8Tensor) {
    const vals = decodeQ8(t);
    const inv = t.scale > 0 ? 1 / t.scale : 1;
    const color = new THREE.Color();
    for (let i = 0; i < layer.n; i++) {
      const v = vals[i] * inv; // [-1, 1]
      const c = diverge(v, this.tmp);
      color.setRGB(c.r, c.g, c.b);
      layer.mesh.setColorAt(i, color);
      this.dummy.position.set(
        layer.positions[i * 3], layer.positions[i * 3 + 1], layer.positions[i * 3 + 2]
      );
      this.dummy.scale.setScalar(0.4 + 1.15 * Math.min(1, Math.abs(v)));
      this.dummy.updateMatrix();
      layer.mesh.setMatrixAt(i, this.dummy.matrix);
    }
    layer.mesh.instanceMatrix.needsUpdate = true;
    if (layer.mesh.instanceColor) layer.mesh.instanceColor.needsUpdate = true;
  }

  /** Select edges (top-k per target above threshold, or threshold-only in
   *  show-all) and bind: line/particle color = sign×magnitude of the actual
   *  contribution, particle size & speed = its magnitude, direction
   *  source→target in inference and reversed in gradient mode. */
  private updateEdges(
    set: EdgeSet,
    t: Q8Tensor,
    srcPos: Float32Array,
    dstPos: Float32Array,
    reverse: boolean,
    targetMajor = false // true when matrix rows are targets (fc1_head)
  ) {
    const m = decodeQ8(t);
    const [d0, d1] = t.shape; // src-major: (nS, nT); target-major: (nT, nS)
    const nS = targetMajor ? d1 : d0;
    const nT = targetMajor ? d0 : d1;
    const val = targetMajor
      ? (s: number, tt: number) => m[tt * nS + s]
      : (s: number, tt: number) => m[s * nT + tt];
    const thrAbs = this.opts.threshold * (t.scale || 1);
    const k = this.opts.showAll ? Infinity : this.opts.k;

    // top-k per target neuron by |contribution|
    type Pick = { s: number; t: number; v: number };
    let picks: Pick[] = [];
    const kb: Pick[] = [];
    for (let tt = 0; tt < nT; tt++) {
      kb.length = 0;
      for (let s = 0; s < nS; s++) {
        const v = val(s, tt);
        const a = Math.abs(v);
        if (a < thrAbs || a === 0) continue;
        if (kb.length < k || k === Infinity) {
          kb.push({ s, t: tt, v });
        } else {
          let minI = 0;
          for (let i = 1; i < kb.length; i++)
            if (Math.abs(kb[i].v) < Math.abs(kb[minI].v)) minI = i;
          if (Math.abs(kb[minI].v) < a) kb[minI] = { s, t: tt, v };
        }
      }
      picks.push(...kb);
    }
    if (picks.length > MAX_EDGES) {
      picks.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      picks = picks.slice(0, MAX_EDGES);
    }

    const inv = t.scale > 0 ? 1 / t.scale : 1;
    picks.forEach((p, e) => {
      const a = reverse ? dstPos.subarray(p.t * 3, p.t * 3 + 3) : srcPos.subarray(p.s * 3, p.s * 3 + 3);
      const b = reverse ? srcPos.subarray(p.s * 3, p.s * 3 + 3) : dstPos.subarray(p.t * 3, p.t * 3 + 3);
      const mn = Math.min(1, Math.abs(p.v) * inv);
      const c = diverge(Math.sign(p.v) * Math.max(mn, 0.2), this.tmp);
      // line: dimmed by magnitude
      set.linePos.set(a, e * 6);
      set.linePos.set(b, e * 6 + 3);
      const lr = c.r * (0.2 + 0.6 * mn), lg = c.g * (0.2 + 0.6 * mn), lb = c.b * (0.2 + 0.6 * mn);
      set.lineCol.set([lr, lg, lb, lr, lg, lb], e * 6);
      // particle: size & speed proportional to the real contribution
      set.pStart.set(a, e * 3);
      set.pEnd.set(b, e * 3);
      set.pCol.set([c.r, c.g, c.b], e * 3);
      set.pSize[e] = 2.2 + 8.5 * mn;
      set.pSpeed[e] = 0.15 + 0.85 * mn;
      set.pPhase[e] = (e * 0.618034) % 1; // deterministic de-sync, not data
    });
    set.count = picks.length;

    const lg = set.lines.geometry;
    (lg.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
    (lg.getAttribute("color") as THREE.BufferAttribute).needsUpdate = true;
    lg.setDrawRange(0, picks.length * 2);
    const pg = set.points.geometry;
    for (const name of ["position", "aEnd", "aColor", "aSize", "aSpeed", "aPhase"]) {
      (pg.getAttribute(name) as THREE.BufferAttribute).needsUpdate = true;
    }
    pg.setDrawRange(0, picks.length);
  }

  // -- focus / level of detail ------------------------------------------------

  private applyFocus() {
    const f = this.opts.focus;
    for (const p of this.planes) p.group.visible = f === "all" || f === p.name;
    this.denseGroup.visible = f === "all" || f === "dense";
    const targets: Record<string, [number, number]> = {
      all: [25, 78], input: [LAYER_Z.input, 22], conv1: [LAYER_Z.conv1, 30],
      conv2: [LAYER_Z.conv2, 34], conv3: [LAYER_Z.conv3, 34], dense: [44, 46],
    };
    const [z, dist] = targets[f] ?? targets.all;
    this.controls.target.set(0, 0, z);
    this.camera.position.set(f === "all" ? 58 : 8, f === "all" ? 24 : 7, z + dist);
  }
}
