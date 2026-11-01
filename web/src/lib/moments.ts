import { dominantEmotion } from "./emotions";
import { slug } from "./format";
import type { EmotionKey, SessionDetail } from "./types";

export interface Moment {
  t: number;
  tension: number;
  emotion: EmotionKey;
}

/** The most intense passages, spaced apart so the list doesn't repeat the same peak. */
export function pickMoments(session: SessionDetail, count = 5, spacing = 8): Moment[] {
  const tension = session.summary.negative;
  const order = tension.map((_, i) => i).sort((a, b) => tension[b] - tension[a]);
  const picked: Moment[] = [];
  for (const i of order) {
    const chunk = session.chunks[i];
    if (!chunk) continue;
    const t = chunk.start;
    if (picked.some((m) => Math.abs(m.t - t) < spacing)) continue;
    const values = {} as Record<EmotionKey, number>;
    (Object.keys(session.scores) as EmotionKey[]).forEach((k) => (values[k] = session.scores[k][i]));
    picked.push({ t, tension: tension[i], emotion: dominantEmotion(values) });
    if (picked.length >= count) break;
  }
  return picked.sort((a, b) => a.t - b.t);
}

function download(name: string, text: string, mime: string) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export function toCsv(session: SessionDetail): string {
  const keys = Object.keys(session.scores) as EmotionKey[];
  const rows = ["start_s,end_s,tension," + keys.join(",")];
  session.chunks.forEach((c, i) => {
    rows.push([c.start.toFixed(2), c.end.toFixed(2), (session.summary.negative[i] ?? 0).toFixed(3), ...keys.map((k) => session.scores[k][i].toFixed(3))].join(","));
  });
  return rows.join("\n");
}

export const exportCsv = (s: SessionDetail) => download(`${slug(s.title)}.csv`, toCsv(s), "text/csv");
export const exportJson = (s: SessionDetail) => download(`${slug(s.title)}.json`, JSON.stringify(s, null, 2), "application/json");
