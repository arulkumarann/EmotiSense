/**
 * Tension and alert detection. A direct port of src/emotisense/timeline/escalation.py so the Settings page can
 * preview thresholds instantly in the browser; the backend uses the Python version for saved results.
 */
import type { Alert, EscalationSettings, Summary } from "./types";

export const DEFAULT_SETTINGS: EscalationSettings = {
  alert_threshold: 0.6,
  rise_delta: 0.3,
  rise_window_s: 20,
  rise_floor: 0.35,
  smooth_window: 3,
  merge_gap_s: 5,
};

const TREND_DELTA = 0.2;

export function smooth(x: number[], window: number): number[] {
  if (window <= 1 || x.length === 0) return [...x];
  const half = Math.floor(window / 2);
  return x.map((_, i) => {
    const lo = Math.max(0, i - half);
    const hi = Math.min(x.length, i + half + 1);
    let sum = 0;
    for (let k = lo; k < hi; k++) sum += x[k];
    return sum / (hi - lo);
  });
}

/** Tension = the stronger of anger and frustration at each moment. */
export function tensionSeries(anger: number[], frustration: number[]): number[] {
  return anger.map((a, i) => Math.max(a, frustration[i] ?? 0));
}

export function detectAlerts(times: number[], score: number[], cfg: EscalationSettings, ends?: number[]): Alert[] {
  const stops = ends ?? times;
  const n = score.length;
  const flag = new Array<boolean>(n).fill(false);
  const kind = new Array<"threshold" | "rise" | "">(n).fill("");
  for (let k = 0; k < n; k++) {
    if (score[k] >= cfg.alert_threshold) {
      flag[k] = true;
      kind[k] = "threshold";
      continue;
    }
    let lowest = Infinity;
    for (let j = 0; j < n; j++) {
      if (times[j] < times[k] && times[j] >= times[k] - cfg.rise_window_s) lowest = Math.min(lowest, score[j]);
    }
    if (lowest !== Infinity && score[k] >= cfg.rise_floor && score[k] - lowest >= cfg.rise_delta) {
      flag[k] = true;
      kind[k] = "rise";
    }
  }
  const events: number[][] = [];
  for (let k = 0; k < n; k++) {
    if (!flag[k]) continue;
    const last = events[events.length - 1];
    if (last && times[k] - stops[last[last.length - 1]] <= cfg.merge_gap_s) last.push(k);
    else events.push([k]);
  }
  return events.map((idx) => {
    let peakIdx = idx[0];
    for (const i of idx) if (score[i] > score[peakIdx]) peakIdx = i;
    return {
      start: times[idx[0]],
      end: stops[idx[idx.length - 1]],
      peak: score[peakIdx],
      peak_time: times[peakIdx],
      kind: idx.some((i) => kind[i] === "threshold") ? "threshold" : "rise",
    };
  });
}

export function classifyTrend(times: number[], score: number[]): Summary["trend"] {
  const n = score.length;
  if (n < 3) return "stable";
  const span = Math.max(...times) - Math.min(...times);
  if (span <= 0) return "stable";
  const tMean = times.reduce((a, b) => a + b, 0) / n;
  const sMean = score.reduce((a, b) => a + b, 0) / n;
  let cov = 0;
  let varT = 0;
  for (let i = 0; i < n; i++) {
    cov += (times[i] - tMean) * (score[i] - sMean);
    varT += (times[i] - tMean) ** 2;
  }
  const change = (cov / varT) * span;
  if (change >= TREND_DELTA) return "escalating";
  if (change <= -TREND_DELTA) return "de-escalating";
  return "stable";
}

export function analyzeTension(
  times: number[],
  ends: number[],
  anger: number[],
  frustration: number[],
  cfg: EscalationSettings,
): Summary & { alerts: Alert[] } {
  if (times.length === 0) {
    return { mean: 0, peak: 0, peak_time: 0, trend: "stable", n_alerts: 0, fraction_above: 0, negative: [], alerts: [] };
  }
  const negative = smooth(tensionSeries(anger, frustration), cfg.smooth_window);
  const alerts = detectAlerts(times, negative, cfg, ends);
  let peakIdx = 0;
  negative.forEach((v, i) => {
    if (v > negative[peakIdx]) peakIdx = i;
  });
  return {
    mean: negative.reduce((a, b) => a + b, 0) / negative.length,
    peak: negative[peakIdx],
    peak_time: times[peakIdx],
    trend: classifyTrend(times, negative),
    n_alerts: alerts.length,
    fraction_above: negative.filter((v) => v >= cfg.alert_threshold).length / negative.length,
    negative,
    alerts,
  };
}

export const TREND_LABEL: Record<Summary["trend"], string> = {
  escalating: "Rising",
  "de-escalating": "Easing",
  stable: "Steady",
};
