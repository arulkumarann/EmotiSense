import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Sparkline } from "../components/Bits";
import { PageTransition } from "../components/Chrome";
import { IconPause, IconPlay, IconRepeat } from "../components/Icons";
import { useSessions } from "../hooks/useData";
import { useTheme } from "../hooks/useTheme";
import { api } from "../lib/api";
import { EMOTION_BY_KEY, blendColor, dominantEmotion } from "../lib/emotions";
import { fmtTime, pct } from "../lib/format";
import { buildCurves, buildSeries, evalSeries, sampleAll, speechPresence, type Curves, type Series } from "../lib/series";
import type { SessionDetail } from "../lib/types";

interface Tile {
  session: SessionDetail;
  curves: Curves;
  tension: Series;
  offset: number;
}

const SPEEDS = [1, 4, 10, 20];
const PAUSE_AT_END = 4;

function LiveTile({ tile, register }: { tile: Tile; register: (id: string, fn: (clock: number) => void) => () => void }) {
  const { colors } = useTheme();
  const rootRef = useRef<HTMLDivElement>(null);
  const orbRef = useRef<HTMLDivElement>(null);
  const coreRef = useRef<HTMLSpanElement>(null);
  const gaugeRef = useRef<HTMLSpanElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const timeRef = useRef<HTMLSpanElement>(null);
  const tensionRef = useRef<HTMLSpanElement>(null);
  const [label, setLabel] = useState("Calm");
  const last = useRef("neutral");
  const { session, curves, tension } = tile;
  const threshold = session.escalation_config.alert_threshold;

  useEffect(() => register(session.id, (clock) => {
    const cycle = session.duration + PAUSE_AT_END;
    const t = Math.min(((clock + tile.offset) % cycle), session.duration);
    const values = sampleAll(curves, t);
    const presence = speechPresence(session.chunks, t);
    const tens = Math.min(1, Math.max(0, evalSeries(tension, t))) * (0.35 + 0.65 * presence);
    const [r, g, b] = blendColor(values, colors);
    orbRef.current?.style.setProperty("--orb", `${r}, ${g}, ${b}`);
    if (coreRef.current) coreRef.current.style.transform = `scale(${(0.82 + 0.2 * presence + 0.2 * tens).toFixed(3)})`;
    if (gaugeRef.current) gaugeRef.current.style.transform = `scaleX(${tens.toFixed(3)})`;
    if (cursorRef.current) cursorRef.current.style.transform = `translateX(${(t / session.duration) * (cursorRef.current.parentElement?.clientWidth ?? 0)}px)`;
    if (timeRef.current) timeRef.current.textContent = fmtTime(t);
    if (tensionRef.current) tensionRef.current.textContent = pct(tens);
    rootRef.current?.classList.toggle("hot", tens >= threshold);
    const dom = dominantEmotion(values);
    if (presence > 0.3 && dom !== last.current) {
      last.current = dom;
      setLabel(EMOTION_BY_KEY[dom].label);
    }
  }), [register, session, curves, tension, colors, threshold, tile.offset]);

  const spark = useMemo(() => session.summary.negative, [session]);
  return (
    <div className="glass live-tile" ref={rootRef}>
      <div className="live-top">
        <h3 title={session.title}>{session.title}</h3>
        <span className="chip"><span className="dot" style={{ width: 6, height: 6 }} />Replay</span>
      </div>
      <div className="live-body">
        <div className="live-orb" ref={orbRef}><span ref={coreRef} /></div>
        <div className="live-meta">
          <span className="display" style={{ fontSize: "1.8rem", lineHeight: 1 }}>{label}</span>
          <div className="gauge"><span ref={gaugeRef} /></div>
          <span className="muted small mono">Tension <span ref={tensionRef}>0%</span></span>
        </div>
      </div>
      <div className="live-spark">
        <Sparkline values={spark} height={54} />
        <div className="cursor" ref={cursorRef} />
      </div>
      <div className="live-time mono"><span ref={timeRef}>0:00</span><span>{fmtTime(session.duration)}</span></div>
    </div>
  );
}

export default function Live() {
  const { sessions } = useSessions();
  const [tiles, setTiles] = useState<Tile[] | null>(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(4);
  const handlers = useRef(new Map<string, (clock: number) => void>());
  const clock = useRef(0);
  const speedRef = useRef(speed);
  const playingRef = useRef(playing);
  speedRef.current = speed;
  playingRef.current = playing;

  useEffect(() => {
    if (!sessions) return;
    const picks = [...sessions].sort((a, b) => Number(a.is_sample) - Number(b.is_sample) || b.created.localeCompare(a.created)).slice(0, 6);
    let cancelled = false;
    Promise.all(picks.map((s) => api.session(s.id))).then((full) => {
      if (cancelled) return;
      setTiles(full.filter((s) => s.chunks.length).map((s, i) => {
        const curves = buildCurves(s.chunks, s.scores);
        return { session: s, curves, tension: buildSeries(curves.times, s.summary.negative), offset: i * 11 };
      }));
    });
    return () => { cancelled = true; };
  }, [sessions]);

  useEffect(() => {
    let raf = 0;
    let prev = performance.now();
    const tick = (now: number) => {
      const dt = (now - prev) / 1000;
      prev = now;
      if (playingRef.current) clock.current += dt * speedRef.current;
      handlers.current.forEach((fn) => fn(clock.current));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const register = useMemo(() => (id: string, fn: (c: number) => void) => {
    handlers.current.set(id, fn);
    return () => { handlers.current.delete(id); };
  }, []);

  return (
    <PageTransition className="page">
      <header className="page-head">
        <div className="page-head-main">
          <span className="eyebrow">Live replay</span>
          <h1 className="display page-title">Watch them <em>unfold.</em></h1>
          <p className="page-sub">Several analyzed sessions replayed side by side, the way a live monitor would show them.</p>
        </div>
        <span className="sim-banner">Simulation · replaying saved sessions</span>
      </header>

      <div className="live-controls">
        <button className="btn btn-primary btn-sm" onClick={() => setPlaying((p) => !p)}>{playing ? <><IconPause /> Pause</> : <><IconPlay /> Play</>}</button>
        <button className="btn btn-ghost btn-sm" onClick={() => { clock.current = 0; }}><IconRepeat /> Restart</button>
        <div className="chips">{SPEEDS.map((s) => <button key={s} className={`chip ${speed === s ? "on" : ""}`} onClick={() => setSpeed(s)}>{s}×</button>)}</div>
      </div>

      {tiles && tiles.length === 0 ? (
        <div className="empty glass glass-pad">
          <h2 className="display" style={{ fontSize: "2.2rem" }}>Nothing to replay yet</h2>
          <p className="lead">Analyze a recording and it will appear here.</p>
          <Link to="/app" className="btn btn-primary">Analyze a recording</Link>
        </div>
      ) : (
        <div className="live-grid">
          {!tiles && [0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 300 }} />)}
          {tiles?.map((t) => <LiveTile key={t.session.id} tile={t} register={register} />)}
        </div>
      )}
    </PageTransition>
  );
}
