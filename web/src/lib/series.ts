/**
 * Smooth curves through the per-passage scores: monotone cubic (Fritsch-Carlson) interpolation never overshoots
 * the data, so a curve can't swing above 100% or below 0% between two points.
 */
import type { Chunk, EmotionKey } from "./types";

export interface Series {
  xs: number[];
  ys: number[];
  ms: number[]; // tangents
}

export function buildSeries(xs: number[], ys: number[]): Series {
  const n = xs.length;
  const ms = new Array<number>(n).fill(0);
  if (n < 2) return { xs: [...xs], ys: [...ys], ms };
  const d: number[] = [];
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / Math.max(xs[i + 1] - xs[i], 1e-9));
  ms[0] = d[0];
  ms[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) ms[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) {
      ms[i] = 0;
      ms[i + 1] = 0;
      continue;
    }
    const a = ms[i] / d[i];
    const b = ms[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) {
      const t = 3 / Math.sqrt(s);
      ms[i] = t * a * d[i];
      ms[i + 1] = t * b * d[i];
    }
  }
  return { xs: [...xs], ys: [...ys], ms };
}

export function evalSeries(s: Series, x: number): number {
  const n = s.xs.length;
  if (n === 0) return 0;
  if (n === 1 || x <= s.xs[0]) return s.ys[0];
  if (x >= s.xs[n - 1]) return s.ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (s.xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const h = s.xs[hi] - s.xs[lo];
  const t = (x - s.xs[lo]) / h;
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    (2 * t3 - 3 * t2 + 1) * s.ys[lo] +
    (t3 - 2 * t2 + t) * h * s.ms[lo] +
    (-2 * t3 + 3 * t2) * s.ys[hi] +
    (t3 - t2) * h * s.ms[hi]
  );
}

/** SVG path through the series; `px`/`py` map data coordinates to pixels. */
export function seriesPath(s: Series, px: (x: number) => number, py: (y: number) => number, close?: number): string {
  const n = s.xs.length;
  if (n === 0) return "";
  if (n === 1) return `M${px(s.xs[0])},${py(s.ys[0])}`;
  let d = `M${px(s.xs[0]).toFixed(2)},${py(s.ys[0]).toFixed(2)}`;
  for (let i = 0; i < n - 1; i++) {
    const h = s.xs[i + 1] - s.xs[i];
    const c1x = s.xs[i] + h / 3;
    const c1y = s.ys[i] + (s.ms[i] * h) / 3;
    const c2x = s.xs[i + 1] - h / 3;
    const c2y = s.ys[i + 1] - (s.ms[i + 1] * h) / 3;
    d += ` C${px(c1x).toFixed(2)},${py(c1y).toFixed(2)} ${px(c2x).toFixed(2)},${py(c2y).toFixed(2)} ${px(s.xs[i + 1]).toFixed(2)},${py(s.ys[i + 1]).toFixed(2)}`;
  }
  if (close !== undefined) d += ` L${px(s.xs[n - 1]).toFixed(2)},${close} L${px(s.xs[0]).toFixed(2)},${close} Z`;
  return d;
}

export interface Curves {
  times: number[];
  series: Record<EmotionKey, Series>;
}

/** One series per emotion, positioned at the middle of each analysed passage. */
export function movingAverage(values: number[], window: number): number[] {
  if (window <= 1) return values;
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    const lo = Math.max(0, i - half);
    const hi = Math.min(values.length, i + half + 1);
    let sum = 0;
    for (let k = lo; k < hi; k++) sum += values[k];
    return sum / (hi - lo);
  });
}

/** `smoothWindow` > 1 averages neighbouring passages for display, which calms speaker-to-speaker flicker. */
export function buildCurves(chunks: Chunk[], scores: Record<EmotionKey, number[]>, smoothWindow = 1): Curves {
  const times = chunks.map((c) => (c.start + c.end) / 2);
  const series = {} as Record<EmotionKey, Series>;
  (Object.keys(scores) as EmotionKey[]).forEach((k) => {
    const clean = scores[k].map((v) => Math.min(1, Math.max(0, v)));
    series[k] = buildSeries(times, movingAverage(clean, smoothWindow));
  });
  return { times, series };
}

export function sampleAll(curves: Curves, t: number): Record<EmotionKey, number> {
  const out = {} as Record<EmotionKey, number>;
  (Object.keys(curves.series) as EmotionKey[]).forEach((k) => {
    out[k] = Math.min(1, Math.max(0, evalSeries(curves.series[k], t)));
  });
  return out;
}

/** How "present" speech is at time t: 1 inside a passage, easing to 0 over a second of silence. */
export function speechPresence(chunks: Chunk[], t: number): number {
  let best = 0;
  for (const c of chunks) {
    const gap = t < c.start ? c.start - t : t > c.end ? t - c.end : 0;
    best = Math.max(best, Math.max(0, 1 - gap / 1.2));
    if (best === 1) break;
  }
  return best;
}

export function peakAt(peaks: number[], t: number, duration: number): number {
  if (!peaks.length || duration <= 0) return 0;
  const i = Math.min(peaks.length - 1, Math.max(0, Math.floor((t / duration) * peaks.length)));
  return peaks[i];
}
