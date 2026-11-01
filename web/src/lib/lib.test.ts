import { describe, expect, it } from "vitest";
import { blendColor, dominantEmotion, parseHex } from "./emotions";
import { DEFAULT_SETTINGS, analyzeTension, classifyTrend, detectAlerts, smooth } from "./escalation";
import { fmtDuration, fmtTime } from "./format";
import { buildCurves, buildSeries, evalSeries, peakAt, sampleAll, seriesPath, speechPresence } from "./series";
import { encodeWav, extensionOf } from "./wav";

const steps = (n: number, step = 5) => Array.from({ length: n }, (_, i) => i * step);

describe("tension and alerts (mirror of the Python tests)", () => {
  it("classifies the three example arcs", () => {
    expect(classifyTrend(steps(3), [0.2, 0.22, 0.21])).toBe("stable");
    expect(classifyTrend(steps(3), [0.18, 0.47, 0.91])).toBe("escalating");
    expect(classifyTrend(steps(3), [0.78, 0.56, 0.28])).toBe("de-escalating");
    expect(classifyTrend([0, 1], [0, 1])).toBe("stable");
  });

  it("raises a threshold alert with the right peak", () => {
    const alerts = detectAlerts(steps(6), [0.1, 0.2, 0.7, 0.9, 0.3, 0.1], DEFAULT_SETTINGS);
    expect(alerts).toHaveLength(1);
    expect(alerts[0]).toMatchObject({ kind: "threshold", peak: 0.9, peak_time: 15, start: 10 });
  });

  it("raises a rise alert below the threshold, and none for a calm session", () => {
    const rise = detectAlerts(steps(4), [0.1, 0.1, 0.45, 0.5], DEFAULT_SETTINGS);
    expect(rise).toHaveLength(1);
    expect(rise[0].kind).toBe("rise");
    expect(detectAlerts(steps(5), [0.2, 0.25, 0.22, 0.3, 0.2], DEFAULT_SETTINGS)).toEqual([]);
  });

  it("merges alerts inside the gap and splits beyond it", () => {
    const merged = detectAlerts([0, 5, 10, 15, 20], [0.9, 0.2, 0.9, 0.2, 0.2], { ...DEFAULT_SETTINGS, merge_gap_s: 10 });
    expect(merged).toHaveLength(1);
    const split = detectAlerts([0, 30, 60], [0.9, 0.2, 0.9], { ...DEFAULT_SETTINGS, merge_gap_s: 5 });
    expect(split).toHaveLength(2);
  });

  it("smooths with a centred window", () => {
    const s = smooth([0, 0, 1, 0, 0], 3);
    expect(s[1]).toBeCloseTo(1 / 3);
    expect(s[3]).toBeCloseTo(1 / 3);
    expect(smooth([1, 2], 1)).toEqual([1, 2]);
  });

  it("summarises a session and handles an empty one", () => {
    const out = analyzeTension([0, 4], [4, 8], [0.1, 0.8], [0.5, 0.2], { ...DEFAULT_SETTINGS, smooth_window: 1 });
    expect(out.peak).toBe(0.8);
    expect(out.peak_time).toBe(4);
    expect(out.n_alerts).toBe(1);
    expect(analyzeTension([], [], [], [], DEFAULT_SETTINGS).n_alerts).toBe(0);
  });
});

describe("smooth curves", () => {
  const xs = [0, 2, 4, 6, 8];
  const ys = [0.1, 0.9, 0.2, 0.2, 0.8];
  const series = buildSeries(xs, ys);

  it("passes exactly through the data points", () => {
    xs.forEach((x, i) => expect(evalSeries(series, x)).toBeCloseTo(ys[i], 9));
  });

  it("never overshoots the data range (monotone)", () => {
    for (let x = 0; x <= 8; x += 0.05) {
      const v = evalSeries(series, x);
      expect(v).toBeGreaterThanOrEqual(0.1 - 1e-9);
      expect(v).toBeLessThanOrEqual(0.9 + 1e-9);
    }
  });

  it("holds the end values outside the range and renders a path", () => {
    expect(evalSeries(series, -5)).toBe(0.1);
    expect(evalSeries(series, 99)).toBe(0.8);
    const d = seriesPath(series, (x) => x * 10, (y) => 100 - y * 100);
    expect(d.startsWith("M0.00,90.00")).toBe(true);
    expect(d.match(/C/g)).toHaveLength(4);
    expect(seriesPath(series, (x) => x, (y) => y, 100).endsWith("Z")).toBe(true);
  });

  it("builds curves per emotion and samples them together", () => {
    const chunks = [{ start: 0, end: 2 }, { start: 3, end: 5 }];
    const scores = { anger: [0, 1], frustration: [0.5, 0.5], happiness: [1, 0], sadness: [0, 0], neutral: [0.2, 0.4] };
    const curves = buildCurves(chunks, scores);
    expect(curves.times).toEqual([1, 4]);
    const mid = sampleAll(curves, 2.5);
    expect(mid.anger).toBeCloseTo(0.5, 5);
    expect(mid.frustration).toBeCloseTo(0.5, 5);
  });

  it("knows when speech is present and reads waveform peaks", () => {
    const chunks = [{ start: 2, end: 4 }];
    expect(speechPresence(chunks, 3)).toBe(1);
    expect(speechPresence(chunks, 4.6)).toBeLessThan(1);
    expect(speechPresence(chunks, 9)).toBe(0);
    expect(peakAt([0.1, 0.5, 0.9, 0.3], 7.9, 8)).toBe(0.3);
    expect(peakAt([], 1, 8)).toBe(0);
  });
});

describe("helpers", () => {
  it("formats times", () => {
    expect(fmtTime(0)).toBe("0:00");
    expect(fmtTime(75.9)).toBe("1:15");
    expect(fmtTime(3725)).toBe("1:02:05");
    expect(fmtDuration(42)).toBe("42 s");
    expect(fmtDuration(120)).toBe("2 min");
  });

  it("encodes a valid 16-bit mono WAV", () => {
    const buf = encodeWav(new Float32Array([0, 1, -1, 0.5]), 16000);
    const view = new DataView(buf);
    const text = (o: number, n: number) => String.fromCharCode(...new Uint8Array(buf, o, n));
    expect(text(0, 4)).toBe("RIFF");
    expect(text(8, 4)).toBe("WAVE");
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(40, true)).toBe(8);
    expect(view.getInt16(46, true)).toBe(0x7fff);
    expect(view.getInt16(48, true)).toBe(-0x8000);
    expect(buf.byteLength).toBe(52);
  });

  it("reads extensions and blends emotion colours toward the dominant one", () => {
    expect(extensionOf("Voice Note.MP3")).toBe("mp3");
    expect(extensionOf("noext")).toBe("");
    const colors = {
      anger: parseHex("#ff0000"), frustration: parseHex("#ff8800"), happiness: parseHex("#00ff00"),
      sadness: parseHex("#0000ff"), neutral: parseHex("#888888"),
    };
    const values = { anger: 0.9, frustration: 0.1, happiness: 0.0, sadness: 0.0, neutral: 0.1 };
    expect(dominantEmotion(values)).toBe("anger");
    const [r, g, b] = blendColor(values, colors);
    expect(r).toBeGreaterThan(g);
    expect(r).toBeGreaterThan(b);
    expect(parseHex("#fff")).toEqual([255, 255, 255]);
  });
});
