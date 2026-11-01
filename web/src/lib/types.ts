export type EmotionKey = "anger" | "frustration" | "happiness" | "sadness" | "neutral";

export interface Chunk {
  start: number;
  end: number;
}

export interface Alert {
  start: number;
  end: number;
  peak: number;
  peak_time: number;
  kind: "threshold" | "rise";
}

export interface Summary {
  mean: number;
  peak: number;
  peak_time: number;
  trend: "escalating" | "de-escalating" | "stable";
  n_alerts: number;
  fraction_above: number;
  negative: number[];
}

export interface Note {
  id: number;
  t: number;
  text: string;
  created: string;
}

export interface EscalationSettings {
  alert_threshold: number;
  rise_delta: number;
  rise_window_s: number;
  rise_floor: number;
  smooth_window: number;
  merge_gap_s: number;
}

export interface SessionDetail {
  id: string;
  title: string;
  tag: string;
  created: string;
  filename: string;
  is_sample: boolean;
  duration: number;
  emotions: EmotionKey[];
  chunks: Chunk[];
  scores: Record<EmotionKey, number[]>;
  waveform: number[];
  summary: Summary;
  alerts: Alert[];
  escalation_config: EscalationSettings;
  notes: Note[];
  model?: { id: string; kind: string };
  warning?: string;
}

export interface SessionCard {
  id: string;
  title: string;
  tag: string;
  created: string;
  duration: number;
  filename: string;
  is_sample: boolean;
  notes: number;
  summary: {
    mean: number;
    peak: number;
    peak_time: number;
    trend: "escalating" | "de-escalating" | "stable";
    n_alerts: number;
    fraction_above: number;
    emotion_means: Record<EmotionKey, number>;
    sparkline: number[];
  };
}

export interface JobState {
  id: string;
  session_id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  detail: string;
  error: string | null;
}

export interface Health {
  status: string;
  model: "ready" | "loading" | "error" | "absent";
  model_error: string | null;
  timeline_model?: string | null;
  formats: string[];
}

export interface ModelStats {
  ccc?: { mean: number; std: number };
  rmse?: number;
  mae?: number;
  alert_f1?: number;
  trend_accuracy?: number;
  per_emotion_ccc?: Record<EmotionKey, number>;
  folds?: { name: string; wa: number; ua: number }[];
  mean?: { wa: number; ua: number };
  std?: { wa: number; ua: number };
  recipe_comparison?: { label: string; baseline: { wa: number; ua: number }; improved: { wa: number; ua: number } };
}

export interface ModelCard {
  id: string;
  name: string;
  role: "timeline" | "encoder" | "classifier";
  kind: string;
  tagline: string;
  description: string;
  strengths: string[];
  limits: string[];
  stats: ModelStats | null;
  selectable: boolean;
  available: boolean;
  active: boolean;
  file?: string;
  size_mb?: number;
  params?: number;
  params_m?: number;
  details: [string, string][];
  files?: { file: string; fold: number | null; size_mb: number }[];
}

export interface ModelsResponse {
  active: string | null;
  timeline: ModelCard[];
  encoder: ModelCard;
  classifier: ModelCard;
  glossary: { term: string; plain: string }[];
  supported: { name: string; note: string }[];
  data_note: string;
}
