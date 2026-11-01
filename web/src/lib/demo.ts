import { analyzeTension, DEFAULT_SETTINGS } from "./escalation";
import type { EmotionKey, SessionDetail } from "./types";

/** A deterministic, made-up session used only to demonstrate the interface on the landing page. */
export function makeDemoSession(): SessionDetail {
  const duration = 64;
  const step = 3.2;
  const n = 19;
  const chunks = Array.from({ length: n }, (_, i) => ({ start: i * step + 0.4, end: i * step + 0.4 + 2.8 }));
  const bump = (i: number, center: number, width: number) => Math.exp(-(((i - center) / width) ** 2));
  const noise = (i: number, seed: number) => 0.04 * Math.sin(i * 2.3 + seed) + 0.03 * Math.sin(i * 5.1 + seed * 2);
  const clamp = (v: number) => Math.min(0.98, Math.max(0.02, v));
  const scores = {} as Record<EmotionKey, number[]>;
  scores.anger = chunks.map((_, i) => clamp(0.08 + 0.7 * bump(i, 11.5, 2.6) + noise(i, 1)));
  scores.frustration = chunks.map((_, i) => clamp(0.12 + 0.55 * bump(i, 8.5, 3.2) + 0.2 * bump(i, 12.5, 2) + noise(i, 2)));
  scores.sadness = chunks.map((_, i) => clamp(0.1 + 0.14 * bump(i, 3, 3) + 0.1 * bump(i, 16, 2) + noise(i, 3)));
  scores.happiness = chunks.map((_, i) => clamp(0.18 + 0.5 * bump(i, 17.5, 2.4) + 0.2 * bump(i, 1.5, 2) + noise(i, 4)));
  scores.neutral = chunks.map((_, i) => clamp(0.62 - 0.45 * bump(i, 10.5, 4.2) + 0.08 * bump(i, 17, 2) + noise(i, 5)));
  const waveform = Array.from({ length: 720 }, (_, i) => {
    const t = (i / 720) * duration;
    const speaking = chunks.some((c) => t >= c.start && t <= c.end) ? 1 : 0.14;
    const energy = 0.45 + 0.4 * Math.sin(i * 0.27) * Math.sin(i * 0.071 + 1) + 0.15 * Math.sin(i * 1.9);
    return Math.max(0.04, Math.min(1, speaking * Math.abs(energy)));
  });
  const summary = analyzeTension(chunks.map((c) => c.start), chunks.map((c) => c.end), scores.anger, scores.frustration, DEFAULT_SETTINGS);
  const { alerts, ...rest } = summary;
  return {
    id: "demo", title: "Demo", tag: "", created: new Date().toISOString(), filename: "demo", is_sample: true, duration,
    emotions: ["anger", "frustration", "happiness", "sadness", "neutral"], chunks, scores, waveform, summary: rest, alerts,
    escalation_config: DEFAULT_SETTINGS, notes: [],
  };
}
