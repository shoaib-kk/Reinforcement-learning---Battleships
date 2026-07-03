// Payload shapes mirrored from the Python backend (agent/trainer.py,
// server/demo.py, server/main.py).

export interface ShipStatus {
  name: string;
  size: number;
  hits: number;
  sunk: boolean;
}

export interface BoardState {
  view: number[][];       // 0 unknown, 1 hit, 2 miss, 3 sunk
  own_board: number[][];  // 0 water, 1 ship, 2 ship-hit, 3 opp miss, 4 ship-sunk
  enemy_ships: ShipStatus[];
  own_ships: ShipStatus[];
  shots: number;
  done: boolean;
}

export interface ConvActivation {
  kind: "conv";
  n_total: number;
  maps: number[][][]; // [map][row][col]
}
export interface DenseActivation {
  kind: "dense";
  values: number[];
}
export type Activations = Record<string, ConvActivation | DenseActivation>;

export interface StepPayload {
  type: "train_step" | "demo_step";
  step: number;
  episode?: number;
  epsilon?: number;
  action: number;
  result: string;
  sunk_ship?: string | null;
  reward?: number;
  done?: boolean;
  raw?: { win: boolean; loss: boolean; shots_to_win: number | null };
  q_values: (number | null)[];
  saliency?: number[][];
  activations: Activations;
  board: BoardState;
  error?: string;
}

export interface EpisodeRecord {
  type: "episode";
  episode: number;
  total_steps: number;
  opponent: string;
  won: boolean;
  shots: number;
  shots_to_win: number | null;
  shaped_return: number;
  raw_return: number;
  loss: number | null;
  epsilon: number;
  win_rate: number;
  avg_shots_to_win: number | null;
  buffer_size: number;
}

export interface HistogramData {
  counts: number[];
  edges: number[];
}
export interface LayerHistograms {
  layer: string;
  weights: HistogramData;
  grads?: HistogramData;
}

export interface ArchLayer {
  name: string;
  type: string;
  out_shape: number[];
  params: number;
}

export interface TrainStatus {
  active?: boolean;
  running: boolean;
  episode: number;
  total_steps: number;
  epsilon?: number;
  buffer_size?: number;
  opponent?: string;
}

export interface EmbeddingData {
  points: [number, number, number][]; // x, y, outcome (0 unk, 1 win, 2 loss)
  explained: number[];
  n: number;
}

export type WsMessage =
  | StepPayload
  | EpisodeRecord
  | ({ type: "status" } & TrainStatus)
  | { type: "weights"; episode: number; layers: LayerHistograms[] }
  | { type: "demo_reset"; board: BoardState; source: string; opponent: string };
