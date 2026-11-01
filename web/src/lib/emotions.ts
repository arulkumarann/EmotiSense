import type { EmotionKey } from "./types";

export interface EmotionMeta {
  key: EmotionKey;
  label: string;
  /** CSS custom property holding this emotion's colour (defined per theme). */
  cssVar: string;
  blurb: string;
}

export const EMOTIONS: EmotionMeta[] = [
  { key: "anger", label: "Anger", cssVar: "--c-anger", blurb: "Sharp, forceful, raised" },
  { key: "frustration", label: "Frustration", cssVar: "--c-frustration", blurb: "Strained, impatient, blocked" },
  { key: "happiness", label: "Joy", cssVar: "--c-joy", blurb: "Bright, warm, light" },
  { key: "sadness", label: "Sadness", cssVar: "--c-sadness", blurb: "Low, slow, heavy" },
  { key: "neutral", label: "Calm", cssVar: "--c-calm", blurb: "Even, steady, settled" },
];

export const EMOTION_BY_KEY = Object.fromEntries(EMOTIONS.map((e) => [e.key, e])) as Record<EmotionKey, EmotionMeta>;
export const EMOTION_KEYS = EMOTIONS.map((e) => e.key);

export type RGB = [number, number, number];

/** Read the current theme's emotion colours as RGB triples (for canvas and blended gradients). */
export function readEmotionColors(): Record<EmotionKey, RGB> {
  const style = getComputedStyle(document.documentElement);
  const out = {} as Record<EmotionKey, RGB>;
  for (const e of EMOTIONS) out[e.key] = parseHex(style.getPropertyValue(e.cssVar).trim());
  return out;
}

export function parseHex(hex: string): RGB {
  const h = hex.replace("#", "");
  const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h.padEnd(6, "0");
  return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)];
}

/** Weighted blend of the emotion colours; sharper weights make the dominant emotion win. */
export function blendColor(values: Record<EmotionKey, number>, colors: Record<EmotionKey, RGB>, power = 2.2): RGB {
  let total = 0;
  const acc: RGB = [0, 0, 0];
  for (const k of EMOTION_KEYS) {
    const w = Math.pow(Math.max(values[k] ?? 0, 0), power);
    total += w;
    acc[0] += colors[k][0] * w;
    acc[1] += colors[k][1] * w;
    acc[2] += colors[k][2] * w;
  }
  if (total <= 1e-9) return colors.neutral;
  return [Math.round(acc[0] / total), Math.round(acc[1] / total), Math.round(acc[2] / total)];
}

export function dominantEmotion(values: Record<EmotionKey, number>): EmotionKey {
  let best: EmotionKey = "neutral";
  let bestValue = -1;
  for (const k of EMOTION_KEYS) {
    if ((values[k] ?? 0) > bestValue) {
      best = k;
      bestValue = values[k] ?? 0;
    }
  }
  return best;
}

export const rgb = (c: RGB, a = 1) => `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;
