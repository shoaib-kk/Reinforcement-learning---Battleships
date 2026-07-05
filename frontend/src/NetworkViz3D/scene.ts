// Three.js scene for the live network readout.
//
// Every visual property binds to a real number from one specific forward
// (or backward) pass — see the mapping table in the README. Nothing here
// runs a decorative loop: the only time-driven animation is particle
// *position along its edge*, and each particle's existence, speed, size and
// color all come from the actual (activation × weight) contribution (or
// weight gradient, in gradient mode) it represents. A ReLU-dead neuron has
// contribution exactly 0 on every outgoing edge, so it gets no particles
// and stays dark by construction. Hovering any node, feature-map cell or
// connection shows the exact number it renders; clicking a head node
// isolates the edge paths that produced that Q-value. Edge *selection* has
// a small hysteresis to stop top-k churn from flickering, but a kept edge
// always renders its current true value.
//
// Performance: all geometry is pooled and preallocated once per model
// topology (plane textures, node InstancedMeshes, edge/particle buffers
// with drawRange) — per step we only rewrite typed arrays and flag them
// dirty. Particle motion is computed in the vertex shader from a time
// uniform, so a paused stream costs ~zero CPU.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
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
  topN: number;        // global budget: the N strongest edges kept per matrix
  showAll: boolean;    // ignore k and topN (still capped by pool)
  weightWireframe: boolean; // static head-weight wires (from /api/viz3d/static)
  focus: string;       // all | input | conv1 | conv2 | conv3 | dense
}

export const DEFAULT_OPTIONS: SceneOptions = {
  mode: "inference",
  k: 4,
  topN: 256,
  showAll: false,
  weightWireframe: false,
  focus: "all",
};

const MAX_EDGES = 4096; // per edge set (pool size)
const NOISE_FRAC = 0.02; // below ~2.5 q8 LSB of the matrix scale: quantization noise, never drawn
const HYST_MISSES = 3;   // steps an edge must miss the selection before dropping (anti-flicker)
const FOCUS_K = 32;      // per-target k while a head node is focused (one target => show more)

/** Board cell name for a 0-99 head/board index (column letter + 1-based row). */
function cellLabel(i: number): string {
  return `${"ABCDEFGHIJ"[i % 10]}${Math.floor(i / 10) + 1}`;
}

// what each input plane encodes (env/battleship_env.py `_obs`)
const CHANNEL_NAMES = ["not fired yet", "hits", "misses", "own ships"];

// one-line role of each layer, shown under its name in the scene
const LAYER_ROLES: Record<string, string> = {
  input: "the agent's view of the board",
  conv1: "3×3 detectors scan for local patterns",
  conv2: "combines patterns into shapes",
  conv3: "high-level board features",
  fc1: "mixes all positions into 512 features",
  head: "expected value of firing each cell",
};

function fmt(v: number): string {
  return Math.abs(v) >= 1e-4 || v === 0 ? v.toFixed(3) : v.toExponential(2);
}

interface EdgePick {
  s: number; // source neuron index
  t: number; // target neuron index
  v: number; // the actual contribution (or gradient) on that edge
}
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
  values: Float32Array; // decoded (n*100) values of the bound step, for hover
}

interface NodeLayer {
  name: string;
  n: number;
  mesh: THREE.InstancedMesh;
  positions: Float32Array; // (n*3)
  group: THREE.Group;
  values: Float32Array; // decoded per-unit values of the bound step, for hover
}

interface EdgeSet {
  lines: LineSegments2;
  linePos: Float32Array;   // interleaved (start.xyz, end.xyz) per edge
  lineCol: Float32Array;   // interleaved (startColor, endColor) per edge
  lineWidth: Float32Array; // per-edge world-units width = |contribution|
  linePosBuf: THREE.InstancedInterleavedBuffer;
  lineColBuf: THREE.InstancedInterleavedBuffer;
  lineWidthAttr: THREE.InstancedBufferAttribute;
  points: THREE.Points;
  pStart: Float32Array;
  pEnd: Float32Array;
  pCol: Float32Array;
  pSize: Float32Array;
  pSpeed: Float32Array;
  pPhase: Float32Array;
  count: number;
  meta: EdgePick[];            // per rendered edge, for the hover readout
  sticky: Map<number, number>; // edge key (t*nS+s) -> consecutive misses
  grad: boolean;               // whether the current binding is gradient data
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

// Per-pixel brightness cap for the additive particles: identity below the
// luminance knee, Reinhard-compressed above it. Overshooting channels are
// rescaled (not clamped) so dense pile-ups saturate at the palette hue at
// full brightness instead of fusing to white.
const TONEMAP_SHADER = {
  uniforms: { tDiffuse: { value: null as THREE.Texture | null } },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    varying vec2 vUv;
    void main() {
      vec4 c = texture2D(tDiffuse, vUv);
      float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      float knee = 0.7;
      float room = 0.3; // luminance asymptotes at knee + room
      float cl = l <= knee ? l : knee + (l - knee) / (1.0 + (l - knee) / room);
      vec3 o = c.rgb * (cl / max(l, 1e-5));
      float m = max(o.r, max(o.g, o.b));
      gl_FragColor = vec4(m > 1.0 ? o / m : o, c.a);
    }
  `,
};

export class NetScene {
  private renderer: THREE.WebGLRenderer;
  private composer: EffectComposer;
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

  // hover readout & click-to-trace
  private raycaster = new THREE.Raycaster();
  private pointerNdc = new THREE.Vector2();
  private pointerPx = { x: 0, y: 0 };
  private pointerMoved = false;
  private downAt: { x: number; y: number } | null = null;
  private tooltip: HTMLDivElement;
  private focusLabel: HTMLDivElement;
  private focusHead = -1; // head unit whose input edges are isolated (click)

  /** Edge count readout for the panel, called after every (re)bind. */
  onStats?: (edges: number) => void;

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

    // HDR target (MSAA to keep line antialiasing) -> luminance cap -> sRGB out
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const target = new THREE.WebGLRenderTarget(size.x, size.y, {
      type: THREE.HalfFloatType,
      samples: 4,
    });
    this.composer = new EffectComposer(this.renderer, target);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new ShaderPass(TONEMAP_SHADER));
    this.composer.addPass(new OutputPass());

    // depth cue: distant layers fade toward the background. Particles are a
    // fog-less ShaderMaterial on purpose — the pulses stay readable at depth.
    this.scene.fog = new THREE.Fog(BG, 90, 260);

    this.tooltip = document.createElement("div");
    this.tooltip.className = "viz3d-tooltip";
    container.appendChild(this.tooltip);
    this.focusLabel = document.createElement("div");
    this.focusLabel.className = "viz3d-focus";
    container.appendChild(this.focusLabel);

    (this.raycaster.params as { Line2?: { threshold: number } }).Line2 = { threshold: 0.12 };
    const el = this.renderer.domElement;
    el.addEventListener("pointermove", (e) => {
      const rect = el.getBoundingClientRect();
      this.pointerPx.x = e.clientX - rect.left;
      this.pointerPx.y = e.clientY - rect.top;
      this.pointerNdc.set(
        (this.pointerPx.x / rect.width) * 2 - 1,
        -(this.pointerPx.y / rect.height) * 2 + 1
      );
      this.pointerMoved = true;
    });
    el.addEventListener("pointerleave", () => {
      this.pointerMoved = false;
      this.tooltip.style.display = "none";
    });
    el.addEventListener("pointerdown", (e) => {
      this.downAt = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("pointerup", (e) => {
      const d = this.downAt;
      this.downAt = null;
      // a click, not an orbit/pan drag
      if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) < 6) this.handleClick();
    });

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
    this.composer.dispose();
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
    this.container.removeChild(this.tooltip);
    this.container.removeChild(this.focusLabel);
  }

  private resize() {
    const w = this.container.clientWidth || 800;
    const h = this.container.clientHeight || 560;
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private animate = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.animate);
    this.timeUniform.value += this.clock.getDelta();
    this.controls.update();
    if (this.pointerMoved && this.built) {
      this.pointerMoved = false; // raycast at most once per rendered frame
      this.updateHover();
    }
    this.composer.render();
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
      mesh.userData = { plane: name, map: i }; // hover readout looks this up
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
    // the input planes get a caption naming what each channel encodes
    if (name === "input") {
      for (let i = 0; i < n; i++) {
        const cap = this.textSprite(
          [{ text: CHANNEL_NAMES[i] ?? `channel ${i}`, font: "18px system-ui, sans-serif", fill: "#898781", y: 24 }],
          160, 36, 2.6
        );
        cap.position.set(centers[i * 3], centers[i * 3 + 1] - PLANE / 2 - 0.5, z);
        group.add(cap);
      }
    }
    this.scene.add(group);
    return {
      name, n, meshes, textures, texData, centers, frame, frameColors, group,
      values: new Float32Array(n * 100),
    };
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
    return { name, n, mesh, positions, group, values: new Float32Array(n) };
  }

  private buildEdgeSet(): EdgeSet {
    const linePos = new Float32Array(MAX_EDGES * 6);
    const lineCol = new Float32Array(MAX_EDGES * 6);
    const lineWidth = new Float32Array(MAX_EDGES);
    const lineGeo = new LineSegmentsGeometry();
    lineGeo.setPositions(linePos); // wraps (not copies) — rewritten in place per step
    lineGeo.setColors(lineCol);
    lineGeo.setAttribute("linewidth", new THREE.InstancedBufferAttribute(lineWidth, 1));
    lineGeo.instanceCount = 0;
    // constant generous bound (raycast early-out only) — content never leaves it
    lineGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 25), 250);
    const linePosBuf = (lineGeo.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute)
      .data as THREE.InstancedInterleavedBuffer;
    const lineColBuf = (lineGeo.getAttribute("instanceColorStart") as THREE.InterleavedBufferAttribute)
      .data as THREE.InstancedInterleavedBuffer;
    const lineWidthAttr = lineGeo.getAttribute("linewidth") as THREE.InstancedBufferAttribute;
    linePosBuf.setUsage(THREE.DynamicDrawUsage);
    lineColBuf.setUsage(THREE.DynamicDrawUsage);
    lineWidthAttr.setUsage(THREE.DynamicDrawUsage);

    // Fat lines so |contribution| maps to *width* (native GPU line width is
    // 1px on most Windows drivers, so LineBasicMaterial can't encode it).
    // Normal alpha blending on purpose: overlapping lines converge toward
    // their own color and can never stack past this fixed opacity. Additive
    // glow is reserved for the traveling pulse particles below.
    const lineMat = new LineMaterial({
      vertexColors: true,
      worldUnits: true,
      linewidth: 0.2, // raycast tolerance only; drawing reads the per-edge attribute
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      fog: true,
    });
    // Per-edge width: swap the linewidth uniform for our instanced attribute.
    // The fragment stage (WORLD_UNITS path) also reads it, via a varying.
    lineMat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace(
          "uniform float linewidth;",
          "attribute float linewidth;\nvarying float vLinewidth;"
        )
        .replace("void main() {", "void main() {\n\tvLinewidth = linewidth;");
      shader.fragmentShader = shader.fragmentShader
        .replace(/\blinewidth\b/g, "vLinewidth")
        .replace("uniform float vLinewidth;", "varying float vLinewidth;");
    };
    const lines = new LineSegments2(lineGeo, lineMat);
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
    return {
      lines, linePos, lineCol, lineWidth, linePosBuf, lineColBuf, lineWidthAttr,
      points, pStart, pEnd, pCol, pSize, pSpeed, pPhase,
      count: 0, meta: [], sticky: new Map(), grad: false,
    };
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
        vertexColors: true, transparent: true, opacity: 0.22, depthWrite: false,
      })
    );
    this.wireframe.frustumCulled = false;
    this.wireframe.visible = this.opts.weightWireframe;
    this.denseGroup.add(this.wireframe);
  }

  /** Canvas-backed text sprite; `lines` are drawn centered at their `y`. */
  private textSprite(
    lines: { text: string; font: string; fill: string; y: number }[],
    w: number,
    h: number,
    scaleX: number
  ): THREE.Sprite {
    const cv = document.createElement("canvas");
    cv.width = w;
    cv.height = h;
    const ctx = cv.getContext("2d")!;
    ctx.textAlign = "center";
    for (const l of lines) {
      ctx.font = l.font;
      ctx.fillStyle = l.fill;
      ctx.fillText(l.text, w / 2, l.y);
    }
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: new THREE.CanvasTexture(cv),
        transparent: true,
        depthWrite: false,
      })
    );
    sprite.scale.set(scaleX, (scaleX * h) / w, 1);
    return sprite;
  }

  private addLabels(topology: ArchLayer[]) {
    for (const l of topology) {
      if (!(l.name in LAYER_Z)) continue;
      const title =
        l.name === "head" ? "Q head 10×10" : `${l.name} ${l.out_shape.join("×")}`;
      const sprite = this.textSprite(
        [
          { text: title, font: "26px system-ui, sans-serif", fill: "#c3c2b7", y: 30 },
          { text: LAYER_ROLES[l.name] ?? "", font: "19px system-ui, sans-serif", fill: "#898781", y: 60 },
        ],
        512, 80, 12
      );
      const yTop = l.name === "fc1" ? 8.5 : l.name === "head" ? 10 : 8;
      sprite.position.set(0, -yTop - 2.4, LAYER_Z[l.name]);
      this.scene.add(sprite);
    }
  }

  // -- per-step data binding -------------------------------------------------

  setStep(step: Viz3DStep, action: number) {
    this.lastStep = step;
    this.lastAction = action;
    if (this.built) this.applyStep(true);
  }

  setOptions(o: SceneOptions) {
    const focusChanged = o.focus !== this.opts.focus;
    this.opts = { ...o };
    if (this.wireframe) this.wireframe.visible = o.weightWireframe;
    if (focusChanged) this.applyFocus();
    this.clearSticky(); // selection params changed — old grace would fight the new filter
    if (this.lastStep && this.built) this.applyStep(false); // re-filter edges
  }

  /** advance=true on a new step (hysteresis clock ticks); false on re-filters
   *  of the same data (option changes, focus clicks, late static payload). */
  private applyStep(advance = false) {
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

    // fc1->head first: when a head node is focused, its surviving edges
    // decide which fc1 units the conv3->fc1 set is traced back to
    let fc1Allowed: Set<number> | null = null;
    if (this.fc1 && this.head && this.edgeF1H) {
      const m = gradMode ? step.grads!.fc1_head : step.contrib.fc1_head;
      // matrix is (target=head j, source=fc1 i) — flag transposed layout
      const picks = this.updateEdges(
        this.edgeF1H, m, this.fc1.positions, this.head.positions,
        gradMode, true, this.focusHead, null, advance
      );
      if (this.focusHead >= 0) fc1Allowed = new Set(picks.map((p) => p.s));
    }
    const conv3 = this.planes.find((p) => p.name === "conv3");
    if (conv3 && this.fc1 && this.edgeC3F) {
      const m = gradMode ? step.grads!.conv3_fc1 : step.contrib.conv3_fc1;
      this.updateEdges(
        this.edgeC3F, m, conv3.centers, this.fc1.positions,
        gradMode, false, -1, fc1Allowed, advance
      );
    }
    this.updateFocusLabel();
    this.onStats?.((this.edgeC3F?.count ?? 0) + (this.edgeF1H?.count ?? 0));
  }

  /** Feature-map plane: each texel = that cell's pre-ReLU activation,
   *  normalized by the layer max (the q8 scale). Frame color = per-map mean
   *  |post-ReLU activation| (inference) or per-map |grad| mass (gradients). */
  private updatePlanes(layer: PlaneLayer, t: Q8Tensor, gradMags?: Q8Tensor) {
    const vals = decodeQ8(t); // (n, 10, 10), values in [-scale, scale]
    layer.values = vals; // kept for the hover readout
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
    layer.values = vals; // kept for the hover readout
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

  /** Select edges — per-target top-k intersected with a global strongest-N
   *  budget (rank-based, so density stays stable whether the matrix is flat
   *  or spiky) — and bind: line width & brightness = the actual
   *  |contribution|, particle size & speed = its magnitude, direction
   *  source→target in inference and reversed in gradient mode.
   *
   *  Selection (never values) has hysteresis: an edge must miss the cut
   *  HYST_MISSES consecutive steps before it drops, and while in grace it
   *  renders its *current* true value — a contribution that hits zero
   *  (ReLU-dead source) still vanishes instantly. */
  private updateEdges(
    set: EdgeSet,
    t: Q8Tensor,
    srcPos: Float32Array,
    dstPos: Float32Array,
    reverse: boolean,       // gradient mode: particles flow target→source
    targetMajor: boolean,   // true when matrix rows are targets (fc1_head)
    focusTarget: number,    // >= 0: render only edges into this target
    targetSet: Set<number> | null, // restrict targets (focused path tracing)
    advance: boolean        // new step (hysteresis ticks) vs re-filter
  ): EdgePick[] {
    const m = decodeQ8(t);
    const [d0, d1] = t.shape; // src-major: (nS, nT); target-major: (nT, nS)
    const nS = targetMajor ? d1 : d0;
    const nT = targetMajor ? d0 : d1;
    const val = targetMajor
      ? (s: number, tt: number) => m[tt * nS + s]
      : (s: number, tt: number) => m[s * nT + tt];
    const noiseAbs = NOISE_FRAC * (t.scale || 1);
    const k = this.opts.showAll ? Infinity : focusTarget >= 0 ? FOCUS_K : this.opts.k;
    const budget = this.opts.showAll ? MAX_EDGES : Math.min(this.opts.topN, MAX_EDGES);

    // top-k per target neuron by |contribution|
    let picks: EdgePick[] = [];
    const kb: EdgePick[] = [];
    for (let tt = 0; tt < nT; tt++) {
      if (focusTarget >= 0 && tt !== focusTarget) continue;
      if (targetSet && !targetSet.has(tt)) continue;
      kb.length = 0;
      for (let s = 0; s < nS; s++) {
        const v = val(s, tt);
        const a = Math.abs(v);
        if (a < noiseAbs) continue;
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
    // global budget: keep only the strongest edges across the whole matrix
    if (picks.length > budget) {
      picks.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
      picks = picks.slice(0, budget);
    }

    if (advance) {
      const fresh = new Set<number>();
      for (const p of picks) fresh.add(p.t * nS + p.s);
      for (const [key, misses] of set.sticky) {
        if (fresh.has(key)) continue;
        const s = key % nS;
        const tt = (key - s) / nS;
        const filtered =
          (focusTarget >= 0 && tt !== focusTarget) ||
          (targetSet !== null && !targetSet.has(tt));
        const v = filtered ? 0 : val(s, tt);
        if (filtered || misses + 1 >= HYST_MISSES || Math.abs(v) < noiseAbs) {
          set.sticky.delete(key);
          continue;
        }
        set.sticky.set(key, misses + 1);
        picks.push({ s, t: tt, v });
      }
      for (const key of fresh) set.sticky.set(key, 0);
      if (picks.length > MAX_EDGES) {
        picks.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
        picks = picks.slice(0, MAX_EDGES);
      }
    }

    const inv = t.scale > 0 ? 1 / t.scale : 1;
    picks.forEach((p, e) => {
      const a = reverse ? dstPos.subarray(p.t * 3, p.t * 3 + 3) : srcPos.subarray(p.s * 3, p.s * 3 + 3);
      const b = reverse ? srcPos.subarray(p.s * 3, p.s * 3 + 3) : dstPos.subarray(p.t * 3, p.t * 3 + 3);
      const mn = Math.min(1, Math.abs(p.v) * inv);
      const c = diverge(Math.sign(p.v) * Math.max(mn, 0.2), this.tmp);
      // line: width and brightness follow the magnitude
      set.linePos.set(a, e * 6);
      set.linePos.set(b, e * 6 + 3);
      const lr = c.r * (0.3 + 0.7 * mn), lg = c.g * (0.3 + 0.7 * mn), lb = c.b * (0.3 + 0.7 * mn);
      set.lineCol.set([lr, lg, lb, lr, lg, lb], e * 6);
      set.lineWidth[e] = 0.03 + 0.25 * mn; // world units
      // particle: size & speed proportional to the real contribution
      set.pStart.set(a, e * 3);
      set.pEnd.set(b, e * 3);
      set.pCol.set([c.r, c.g, c.b], e * 3);
      set.pSize[e] = 1.8 + 3.2 * mn;
      set.pSpeed[e] = 0.15 + 0.85 * mn;
      set.pPhase[e] = (e * 0.618034) % 1; // deterministic de-sync, not data
    });
    // zero the stale tail so the raycaster can't hit ghost segments
    if (picks.length < set.count) set.linePos.fill(0, picks.length * 6, set.count * 6);
    set.count = picks.length;
    set.meta = picks;
    set.grad = reverse;

    set.linePosBuf.needsUpdate = true;
    set.lineColBuf.needsUpdate = true;
    set.lineWidthAttr.needsUpdate = true;
    (set.lines.geometry as LineSegmentsGeometry).instanceCount = picks.length;
    const pg = set.points.geometry;
    for (const name of ["position", "aEnd", "aColor", "aSize", "aSpeed", "aPhase"]) {
      (pg.getAttribute(name) as THREE.BufferAttribute).needsUpdate = true;
    }
    pg.setDrawRange(0, picks.length);
    return picks;
  }

  // -- hover readout & click-to-trace ----------------------------------------

  private clearSticky() {
    this.edgeC3F?.sticky.clear();
    this.edgeF1H?.sticky.clear();
  }

  private pickAt(): THREE.Intersection | null {
    this.raycaster.setFromCamera(this.pointerNdc, this.camera);
    const targets: THREE.Object3D[] = [];
    if (this.denseGroup.visible) {
      if (this.fc1) targets.push(this.fc1.mesh);
      if (this.head) targets.push(this.head.mesh);
      if (this.edgeC3F && this.edgeC3F.count > 0) targets.push(this.edgeC3F.lines);
      if (this.edgeF1H && this.edgeF1H.count > 0) targets.push(this.edgeF1H.lines);
    }
    for (const p of this.planes) if (p.group.visible) targets.push(...p.meshes);
    return this.raycaster.intersectObjects(targets, false)[0] ?? null;
  }

  private updateHover() {
    const hit = this.lastStep ? this.pickAt() : null;
    const text = hit ? this.describe(hit) : null;
    if (!text) {
      this.tooltip.style.display = "none";
      return;
    }
    this.tooltip.textContent = text;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = `${this.pointerPx.x + 14}px`;
    this.tooltip.style.top = `${this.pointerPx.y + 12}px`;
  }

  /** The exact number behind whatever the pointer is on. */
  private describe(hit: THREE.Intersection): string | null {
    if (this.head && hit.object === this.head.mesh && hit.instanceId !== undefined) {
      const i = hit.instanceId;
      const chosen = i === this.lastAction ? " · chosen action" : "";
      return `Q(${cellLabel(i)}) = ${fmt(this.head.values[i])}${chosen}`;
    }
    if (this.fc1 && hit.object === this.fc1.mesh && hit.instanceId !== undefined) {
      const i = hit.instanceId;
      const v = this.fc1.values[i];
      return `fc1 #${i} · pre-act ${fmt(v)} ${v > 0 ? "(fired)" : "(ReLU-suppressed)"}`;
    }
    const ud = hit.object.userData as { plane?: string; map?: number };
    if (ud.plane !== undefined && ud.map !== undefined && hit.uv) {
      const layer = this.planes.find((p) => p.name === ud.plane);
      if (!layer) return null;
      const col = Math.min(9, Math.floor(hit.uv.x * 10));
      const row = 9 - Math.min(9, Math.floor(hit.uv.y * 10)); // texture row 0 = board bottom
      const v = layer.values[ud.map * 100 + row * 10 + col];
      return ud.plane === "input"
        ? `input · ${CHANNEL_NAMES[ud.map] ?? `ch ${ud.map}`} · ${cellLabel(row * 10 + col)} = ${fmt(v)}`
        : `${ud.plane} map ${ud.map} · ${cellLabel(row * 10 + col)} · pre-ReLU ${fmt(v)}`;
    }
    for (const set of [this.edgeC3F, this.edgeF1H]) {
      if (!set || hit.object !== set.lines) continue;
      const idx = hit.faceIndex ?? -1;
      const p = idx >= 0 && idx < set.count ? set.meta[idx] : undefined;
      if (!p) return null;
      const c3f = set === this.edgeC3F;
      const src = c3f ? `conv3 map ${p.s}` : `fc1 #${p.s}`;
      const dst = c3f ? `fc1 #${p.t}` : `Q(${cellLabel(p.t)})`;
      return set.grad
        ? `${src} ← ${dst} · ${c3f ? "Σ|∂L/∂W1|" : "∂L/∂W2"} = ${fmt(p.v)}`
        : `${src} → ${dst} · ${c3f ? "Σ relu(a)·w" : "relu(a)·w"} = ${fmt(p.v)}`;
    }
    return null;
  }

  private handleClick() {
    if (!this.built) return;
    const hit = this.pickAt();
    let next = -1; // clicking anything but a head node clears the focus
    if (this.head && hit && hit.object === this.head.mesh && hit.instanceId !== undefined) {
      next = hit.instanceId === this.focusHead ? -1 : hit.instanceId; // re-click toggles off
    }
    if (next === this.focusHead) return;
    this.focusHead = next;
    this.clearSticky();
    if (this.lastStep) this.applyStep(false);
    else this.updateFocusLabel();
  }

  private updateFocusLabel() {
    if (this.focusHead < 0 || !this.head) {
      this.focusLabel.style.display = "none";
      return;
    }
    this.focusLabel.textContent =
      `Q(${cellLabel(this.focusHead)}) = ${fmt(this.head.values[this.focusHead])}` +
      ` — showing only its inputs · click empty space to clear`;
    this.focusLabel.style.display = "block";
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
