import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import type { Player, PlayerState } from "../hooks/usePlayer";
import { EMOTIONS, EMOTION_BY_KEY, blendColor, dominantEmotion, type RGB } from "../lib/emotions";
import { fmtTime, pct } from "../lib/format";
import { peakAt, sampleAll, speechPresence, type Curves } from "../lib/series";
import type { Chunk, EmotionKey } from "../lib/types";
import { IconBack, IconForward, IconNote, IconPause, IconPlay, IconVolume } from "./Icons";

interface OrbProps {
  chunks: Chunk[];
  waveform: number[];
  duration: number;
  curves: Curves;
  player: Player;
  colors: Record<EmotionKey, RGB>;
  size?: number;
}

/** A soft sphere of light that takes the colour of the dominant feeling and breathes with the sound. */
export function MoodOrb({ chunks, waveform, duration, curves, player, colors, size = 280 }: OrbProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const coreRef = useRef<HTMLDivElement>(null);
  const haloRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const [label, setLabel] = useState<{ key: EmotionKey; value: number }>({ key: "neutral", value: 0 });
  const lastKey = useRef<EmotionKey>("neutral");

  useEffect(() => {
    return player.subscribe((t) => {
      const values = sampleAll(curves, t);
      const presence = speechPresence(chunks, t);
      const amp = peakAt(waveform, t, duration) * presence;
      const [r, g, b] = blendColor(values, colors);
      const root = rootRef.current;
      if (root) root.style.setProperty("--orb", `${r}, ${g}, ${b}`);
      if (coreRef.current) coreRef.current.style.transform = `scale(${(0.86 + 0.1 * presence + 0.22 * amp).toFixed(3)})`;
      if (haloRef.current) {
        haloRef.current.style.opacity = (0.3 + 0.55 * presence + 0.15 * amp).toFixed(3);
        haloRef.current.style.transform = `scale(${(0.95 + 0.25 * amp).toFixed(3)})`;
      }
      if (ringRef.current) {
        ringRef.current.style.opacity = (0.15 + 0.5 * amp).toFixed(3);
        ringRef.current.style.transform = `scale(${(1.02 + 0.2 * amp).toFixed(3)})`;
      }
      const dom = dominantEmotion(values);
      if (presence > 0.25 && dom !== lastKey.current) {
        lastKey.current = dom;
        setLabel({ key: dom, value: values[dom] });
      }
    });
  }, [player, curves, chunks, waveform, duration, colors]);

  return (
    <div className="orb" ref={rootRef} style={{ width: size, height: size }} aria-hidden="true">
      <div className="orb-halo" ref={haloRef} />
      <div className="orb-ring" ref={ringRef} />
      <div className="orb-core" ref={coreRef} />
      <div className="orb-label">
        <span key={label.key}>{EMOTION_BY_KEY[label.key].label}</span>
      </div>
    </div>
  );
}

interface LegendProps {
  visible: Set<EmotionKey>;
  onToggle: (k: EmotionKey) => void;
  curves: Curves;
  player: Player;
}

/** The five emotions as toggles; each row's bar and number follow the playhead. */
export function EmotionLegend({ visible, onToggle, curves, player }: LegendProps) {
  const fills = useRef<Partial<Record<EmotionKey, HTMLSpanElement | null>>>({});
  const texts = useRef<Partial<Record<EmotionKey, HTMLSpanElement | null>>>({});
  const shown = useRef<Partial<Record<EmotionKey, number>>>({});

  useEffect(() => {
    return player.subscribe((t) => {
      const values = sampleAll(curves, t);
      EMOTIONS.forEach((e) => {
        const v = values[e.key];
        const fill = fills.current[e.key];
        if (fill) fill.style.transform = `scaleX(${v.toFixed(3)})`;
        const rounded = Math.round(v * 100);
        if (shown.current[e.key] !== rounded) {
          shown.current[e.key] = rounded;
          const text = texts.current[e.key];
          if (text) text.textContent = `${rounded}%`;
        }
      });
    });
  }, [player, curves]);

  return (
    <div className="legend" role="group" aria-label="Emotions">
      {EMOTIONS.map((e) => (
        <button key={e.key} className={`legend-row ${visible.has(e.key) ? "on" : ""}`} onClick={() => onToggle(e.key)} aria-pressed={visible.has(e.key)} title={e.blurb}>
          <span className="legend-dot" style={{ background: `var(${e.cssVar})` }} />
          <span className="legend-name">{e.label}</span>
          <span className="legend-track">
            <span className="legend-fill" ref={(el) => { fills.current[e.key] = el; }} style={{ background: `var(${e.cssVar})` }} />
          </span>
          <span className="legend-value mono tabular" ref={(el) => { texts.current[e.key] = el; }}>0%</span>
        </button>
      ))}
    </div>
  );
}

const RATES = [0.75, 1, 1.25, 1.5, 2];

interface TransportProps {
  player: Player;
  state: PlayerState;
  duration: number;
  onAddNote?: () => void;
}

export function Transport({ player, state, duration, onAddNote }: TransportProps) {
  const progressRef = useRef<HTMLSpanElement>(null);
  useEffect(() => player.subscribe((t) => {
    if (progressRef.current) progressRef.current.style.transform = `scaleX(${Math.min(1, t / Math.max(duration, 0.01)).toFixed(4)})`;
  }), [player, duration]);

  return (
    <div className="transport glass">
      <div className="transport-main">
        <button className="icon-btn" onClick={() => player.skip(-5)} aria-label="Back 5 seconds" title="Back 5 s  (←)"><IconBack /></button>
        <button className="play-btn" onClick={() => player.toggle()} aria-label={state.playing ? "Pause" : "Play"} disabled={!state.ready && !state.error}>
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={state.playing ? "pause" : "play"} initial={{ scale: 0.4, opacity: 0, rotate: -40 }} animate={{ scale: 1, opacity: 1, rotate: 0 }} exit={{ scale: 0.4, opacity: 0, rotate: 40 }} transition={{ duration: 0.25 }} style={{ display: "grid" }}>
              {state.playing ? <IconPause /> : <IconPlay />}
            </motion.span>
          </AnimatePresence>
        </button>
        <button className="icon-btn" onClick={() => player.skip(5)} aria-label="Forward 5 seconds" title="Forward 5 s  (→)"><IconForward /></button>
        <div className="transport-time mono tabular">
          <b>{fmtTime(state.displayTime)}</b>
          <span> / {fmtTime(duration)}</span>
        </div>
      </div>

      <div className="transport-bar" aria-hidden="true"><span ref={progressRef} /></div>

      <div className="transport-side">
        <div className="rate-chips" role="group" aria-label="Playback speed">
          {RATES.map((r) => (
            <button key={r} className={`chip ${state.rate === r ? "on" : ""}`} onClick={() => player.setRate(r)}>{r}×</button>
          ))}
        </div>
        <label className="volume" title="Volume">
          <IconVolume />
          <input
            className="slider"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={state.volume}
            style={{ ["--fill" as string]: `${state.volume * 100}%` }}
            onChange={(e) => player.setVolume(Number(e.target.value))}
            aria-label="Volume"
          />
        </label>
        {onAddNote && (
          <button className="btn btn-ghost btn-sm" onClick={onAddNote} title="Add a note at the playhead  (N)">
            <IconNote /> Note
          </button>
        )}
      </div>
      {state.error && <p className="transport-error">{state.error}</p>}
    </div>
  );
}

export { pct };
