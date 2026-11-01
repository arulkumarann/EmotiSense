import { memo, useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as RPointerEvent } from "react";
import { EMOTIONS, dominantEmotion, rgb, type RGB } from "../lib/emotions";
import { fmtTime, pct } from "../lib/format";
import { evalSeries, sampleAll, seriesPath, type Curves, buildSeries } from "../lib/series";
import type { EmotionKey, SessionDetail } from "../lib/types";
import type { Player } from "../hooks/usePlayer";

const PAD_TOP = 16;
const WAVE_H = 54;
const AXIS_H = 30;
const TICKS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800, 3600];

interface Props {
  session: SessionDetail;
  curves: Curves;
  visible: Set<EmotionKey>;
  showTension: boolean;
  tension: number[];
  player: Player;
  colors: Record<EmotionKey, RGB>;
  height?: number;
  onAddNote?: (t: number) => void;
  interactive?: boolean;
}

function TimelineImpl({ session, curves, visible, showTension, tension, player, colors, height = 380, onAddNote, interactive = true }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const [intro, setIntro] = useState(true);
  const duration = session.duration;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    let raf = 0;
    const ro = new ResizeObserver((entries) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setWidth(Math.max(280, Math.floor(entries[0].contentRect.width))));
    });
    ro.observe(el);
    setWidth(Math.max(280, Math.floor(el.clientWidth)));
    return () => {
      ro.disconnect();
      cancelAnimationFrame(raf);
    };
  }, []);

  useEffect(() => {
    const id = window.setTimeout(() => setIntro(false), 2600);
    return () => window.clearTimeout(id);
  }, []);

  // ── geometry ────────────────────────────────────────────────────────────
  const curvesH = Math.max(120, height - PAD_TOP - 8 - 12 - WAVE_H - AXIS_H - 12);
  const ribbonY = PAD_TOP + curvesH + 12;
  const ribbonH = 7;
  const waveTop = ribbonY + ribbonH + 12;
  const waveMid = waveTop + WAVE_H / 2;
  const totalH = waveTop + WAVE_H + AXIS_H;
  const px = useCallback((t: number) => (t / Math.max(duration, 0.001)) * width, [duration, width]);
  const py = useCallback((v: number) => PAD_TOP + (1 - v) * curvesH, [curvesH]);

  const playheadRef = useRef<SVGGElement>(null);
  const playedRectRef = useRef<SVGRectElement>(null);
  const dotRefs = useRef<Partial<Record<EmotionKey, SVGGElement | null>>>({});
  const hoverLineRef = useRef<SVGLineElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const tipTimeRef = useRef<HTMLSpanElement>(null);
  const tipValueRefs = useRef<Partial<Record<EmotionKey, HTMLSpanElement | null>>>({});
  const dragging = useRef(false);

  const tensionSeries = useMemo(() => buildSeries(curves.times, tension.length ? tension : curves.times.map(() => 0)), [curves.times, tension]);

  // ── static layers (rebuilt only when data, size, theme or visibility changes) ──
  const layers = useMemo(() => {
    const interval = TICKS.find((s) => duration / s <= Math.max(4, width / 110)) ?? 3600;
    const ticks: number[] = [];
    for (let t = 0; t <= duration + 0.001; t += interval) ticks.push(t);

    const bars = Math.max(40, Math.floor(width / 4.2));
    const peaks = session.waveform;
    let base = "";
    for (let i = 0; i < bars; i++) {
      const a = Math.floor((i / bars) * peaks.length);
      const b = Math.max(a + 1, Math.floor(((i + 1) / bars) * peaks.length));
      let peak = 0;
      for (let k = a; k < b; k++) peak = Math.max(peak, peaks[k] ?? 0);
      const h = Math.max(2.5, Math.pow(peak, 0.8) * (WAVE_H - 4));
      const x = ((i + 0.5) / bars) * width;
      base += `M${x.toFixed(1)},${(waveMid - h / 2).toFixed(1)}V${(waveMid + h / 2).toFixed(1)}`;
    }

    const ribbon = session.chunks.map((c, i) => {
      const values = {} as Record<EmotionKey, number>;
      (Object.keys(session.scores) as EmotionKey[]).forEach((k) => (values[k] = session.scores[k][i]));
      const dom = dominantEmotion(values);
      const x0 = px(c.start);
      return { x: x0, w: Math.max(3, px(c.end) - x0), color: colors[dom], key: dom, i };
    });

    return { ticks, base, ribbon };
  }, [session, width, duration, waveMid, px, colors]);

  const emotionPaths = useMemo(
    () =>
      EMOTIONS.filter((e) => visible.has(e.key)).map((e) => {
        const s = curves.series[e.key];
        return { key: e.key, line: seriesPath(s, px, py), area: seriesPath(s, px, py, py(0)) };
      }),
    [curves, visible, px, py],
  );
  const tensionPath = useMemo(() => seriesPath(tensionSeries, px, py), [tensionSeries, px, py]);

  // ── per-frame updates: touch the DOM directly, never React state ─────────
  useEffect(() => {
    const render = (t: number) => {
      const x = px(t);
      playheadRef.current?.setAttribute("transform", `translate(${x.toFixed(2)},0)`);
      playedRectRef.current?.setAttribute("width", Math.max(0, x).toFixed(2));
      (Object.keys(dotRefs.current) as EmotionKey[]).forEach((k) => {
        const dot = dotRefs.current[k];
        if (dot && curves.series[k]) dot.setAttribute("transform", `translate(0,${py(Math.min(1, Math.max(0, evalSeries(curves.series[k], t)))).toFixed(2)})`);
      });
    };
    return player.subscribe(render);
  }, [player, px, py, curves]);

  // ── pointer interaction: hover read-out, click to seek, drag to scrub ───
  const timeFromEvent = (e: { clientX: number }) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const x = Math.min(Math.max(e.clientX - rect.left, 0), rect.width);
    return { x, t: (x / rect.width) * duration, w: rect.width };
  };

  const updateHover = (e: { clientX: number }) => {
    const { x, t, w } = timeFromEvent(e);
    const line = hoverLineRef.current;
    const tip = tipRef.current;
    if (line) {
      line.setAttribute("x1", x.toFixed(1));
      line.setAttribute("x2", x.toFixed(1));
      line.style.opacity = "1";
    }
    if (tip) {
      const values = sampleAll(curves, t);
      if (tipTimeRef.current) tipTimeRef.current.textContent = fmtTime(t);
      (Object.keys(values) as EmotionKey[]).forEach((k) => {
        const el = tipValueRefs.current[k];
        if (el) el.textContent = pct(values[k]);
      });
      const flip = x > w - 210;
      tip.style.transform = `translate3d(${flip ? x - 196 : x + 16}px, 8px, 0)`;
      tip.style.opacity = "1";
    }
  };
  const hideHover = () => {
    if (hoverLineRef.current) hoverLineRef.current.style.opacity = "0";
    if (tipRef.current) tipRef.current.style.opacity = "0";
  };

  const onPointerDown = (e: RPointerEvent) => {
    if (!interactive || e.button !== 0) return;
    dragging.current = true;
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    player.seek(timeFromEvent(e).t);
  };
  const onPointerMove = (e: RPointerEvent) => {
    updateHover(e);
    if (dragging.current) player.seek(timeFromEvent(e).t);
  };
  const onPointerUp = (e: RPointerEvent) => {
    dragging.current = false;
    (e.currentTarget as Element).releasePointerCapture?.(e.pointerId);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft") player.skip(-5);
    else if (e.key === "ArrowRight") player.skip(5);
    else return;
    e.preventDefault();
  };

  if (!session.chunks.length) return <div className="timeline-empty">No speech was detected in this recording.</div>;

  return (
    <div className="timeline" ref={wrapRef} style={{ height: totalH }}>
      <svg width={width} height={totalH} viewBox={`0 0 ${width} ${totalH}`} role="img" aria-label="Emotion timeline">
        <defs>
          {EMOTIONS.map((e) => (
            <linearGradient key={e.key} id={`area-${e.key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stopColor={rgb(colors[e.key], 0.34)} />
              <stop offset="1" stopColor={rgb(colors[e.key], 0)} />
            </linearGradient>
          ))}
          <linearGradient id="alert-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--c-anger)" stopOpacity="0.13" />
            <stop offset="1" stopColor="var(--c-anger)" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="play-line" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--accent-2)" />
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0.2" />
          </linearGradient>
          <linearGradient id="wave-played" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="var(--accent)" />
            <stop offset="1" stopColor="var(--accent-2)" />
          </linearGradient>
          <filter id="glow-blur" x="-10%" y="-30%" width="120%" height="160%">
            <feGaussianBlur stdDeviation="5" />
          </filter>
          <clipPath id="played-clip">
            <rect ref={playedRectRef} x="0" y="0" width="0" height={totalH} />
          </clipPath>
        </defs>

        {/* grid */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1="0" x2={width} y1={py(v)} y2={py(v)} className="tl-grid" strokeDasharray={v === 0 ? undefined : "2 6"} />
            {(v === 0.5 || v === 1) && (
              <text x="2" y={py(v) - 5} className="tl-axis-label">
                {pct(v)}
              </text>
            )}
          </g>
        ))}

        {/* tension peaks */}
        {session.alerts.map((a, i) => {
          const x0 = px(a.start) - 4;
          const w = Math.max(14, px(a.end) - px(a.start) + 8);
          return (
            <g key={i} className="tl-alert">
              <rect x={x0} y={PAD_TOP} width={w} height={curvesH} rx="10" fill="url(#alert-fill)" />
              <line x1={x0 + 5} x2={x0 + w - 5} y1={PAD_TOP + 0.5} y2={PAD_TOP + 0.5} stroke="var(--c-anger)" strokeWidth="2" strokeLinecap="round" opacity="0.8" />
            </g>
          );
        })}

        {/* emotion curves */}
        {emotionPaths.map((p, idx) => (
          <g key={p.key}>
            <path d={p.area} fill={`url(#area-${p.key})`} className={intro ? "tl-fade" : undefined} style={{ animationDelay: `${0.5 + idx * 0.12}s` }} />
            <path d={p.line} fill="none" stroke={rgb(colors[p.key], 0.9)} strokeWidth="5" filter="url(#glow-blur)" opacity="0.55" pathLength={1} className={intro ? "tl-draw" : undefined} style={{ animationDelay: `${idx * 0.12}s` }} />
            <path d={p.line} fill="none" stroke={rgb(colors[p.key])} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" pathLength={1} className={intro ? "tl-draw" : undefined} style={{ animationDelay: `${idx * 0.12}s` }} />
          </g>
        ))}
        {showTension && (
          <path d={tensionPath} fill="none" stroke="var(--text)" strokeWidth="1.6" strokeDasharray="1 5" strokeLinecap="round" opacity="0.75" />
        )}

        {/* dominant-emotion ribbon */}
        <g>
          {layers.ribbon.map((r) => (
            <rect key={r.i} x={r.x} y={ribbonY} width={r.w} height={ribbonH} rx="3.5" fill={rgb(r.color, 0.92)} />
          ))}
        </g>

        {/* waveform: muted underneath, accent where it has played */}
        <path d={layers.base} className="tl-wave" strokeWidth="2.4" strokeLinecap="round" />
        <path d={layers.base} stroke="url(#wave-played)" strokeWidth="2.4" strokeLinecap="round" clipPath="url(#played-clip)" fill="none" />

        {/* axis */}
        {layers.ticks.map((t) => (
          <g key={t}>
            <line x1={px(t)} x2={px(t)} y1={waveTop + WAVE_H + 6} y2={waveTop + WAVE_H + 11} className="tl-grid" />
            <text x={Math.min(Math.max(px(t), 14), width - 14)} y={waveTop + WAVE_H + 26} textAnchor="middle" className="tl-axis-label">
              {fmtTime(t)}
            </text>
          </g>
        ))}

        {/* notes */}
        {session.notes.map((n) => (
          <g key={n.id} className="tl-note">
            <title>{`${fmtTime(n.t)}: ${n.text}`}</title>
            <circle cx={px(n.t)} cy={ribbonY + ribbonH / 2} r="5.5" fill="var(--surface-solid)" stroke="var(--accent-2)" strokeWidth="2" />
          </g>
        ))}

        {/* hover guide */}
        <line ref={hoverLineRef} x1="0" x2="0" y1={PAD_TOP} y2={waveTop + WAVE_H} className="tl-hover" style={{ opacity: 0 }} />

        {/* playhead */}
        <g ref={playheadRef} transform="translate(0,0)" style={{ pointerEvents: "none" }}>
          <line x1="0" x2="0" y1={PAD_TOP - 8} y2={waveTop + WAVE_H} stroke="url(#play-line)" strokeWidth="1.6" />
          <circle cx="0" cy={PAD_TOP - 8} r="9" fill="var(--accent)" opacity="0.28" />
          <circle cx="0" cy={PAD_TOP - 8} r="4.5" fill="#fff" />
          {EMOTIONS.filter((e) => visible.has(e.key)).map((e) => (
            <g key={e.key} ref={(el) => { dotRefs.current[e.key] = el; }} transform={`translate(0,${py(0)})`}>
              <circle r="9" fill={rgb(colors[e.key], 0.22)} />
              <circle r="4" fill={rgb(colors[e.key])} stroke="#fff" strokeWidth="1.4" />
            </g>
          ))}
        </g>

        {/* input surface */}
        <rect
          x="0"
          y="0"
          width={width}
          height={totalH - AXIS_H + 8}
          fill="transparent"
          style={{ cursor: interactive ? "pointer" : "default", touchAction: "pan-y" }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={hideHover}
          onPointerCancel={onPointerUp}
          onDoubleClick={(e) => onAddNote?.(timeFromEvent(e).t)}
        />
      </svg>

      <div className="tl-focus" tabIndex={interactive ? 0 : -1} onKeyDown={onKey} aria-label="Timeline. Use left and right arrow keys to move through the recording." />

      <div className="tl-tip glass" ref={tipRef} style={{ opacity: 0 }}>
        <span className="tl-tip-time mono" ref={tipTimeRef}>0:00</span>
        {EMOTIONS.map((e) => (
          <div key={e.key} className="tl-tip-row">
            <i style={{ background: `var(${e.cssVar})` }} />
            <span>{e.label}</span>
            <span className="mono tabular" ref={(el) => { tipValueRefs.current[e.key] = el; }}>0%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export const Timeline = memo(TimelineImpl);
