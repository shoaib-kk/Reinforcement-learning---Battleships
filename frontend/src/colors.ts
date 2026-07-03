// Dark-surface design tokens (validated palette — see dataviz reference).
// The UI deliberately commits to a dark look; every hue below was checked
// against the dark surface #1a1a19 (contrast >= 3:1, adjacent CVD dE >= 12).

export const C = {
  page: "#0d0d0d",
  surface: "#1a1a19",
  surfaceRaised: "#222221",
  inkPrimary: "#ffffff",
  inkSecondary: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  axis: "#383835",
  border: "rgba(255,255,255,0.10)",
  series: {
    blue: "#3987e5",
    aqua: "#199e70",
    yellow: "#c98500",
    violet: "#9085e9",
    red: "#e66767",
  },
  status: {
    good: "#0ca30c",
    warning: "#fab219",
    serious: "#ec835a",
    critical: "#d03b3b",
  },
};

// Color follows the entity: each opponent keeps its hue forever,
// regardless of which opponents are currently on the chart.
export const OPPONENT_COLOR: Record<string, string> = {
  random: C.series.blue,
  hunt_target: C.series.yellow,
  self: C.series.violet,
  none: C.series.aqua,
};

/** Sequential single-hue blue for magnitude on the dark surface:
 *  t=0 recedes into the surface, t=1 is bright. Returns an rgba fill. */
export function heatBlue(t: number): string {
  const k = Math.max(0, Math.min(1, t));
  return `rgba(57, 135, 229, ${(0.08 + 0.84 * k).toFixed(3)})`;
}

/** Second sequential context (saliency) takes the next slot's hue: aqua. */
export function heatAqua(t: number): string {
  const k = Math.max(0, Math.min(1, t));
  return `rgba(35, 196, 143, ${(0.08 + 0.84 * k).toFixed(3)})`;
}
