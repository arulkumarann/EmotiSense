import { useMemo } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { EMOTIONS } from "../lib/emotions";
import { fmtDate, fmtDuration, pct } from "../lib/format";
import { TREND_LABEL } from "../lib/escalation";
import type { EmotionKey, SessionCard } from "../lib/types";
import { EASE } from "./Chrome";

/** Tiny area line of tension over a session. */
export function Sparkline({ values, width = 240, height = 54, color = "var(--accent-2)" }: { values: number[]; width?: number; height?: number; color?: string }) {
  const { line, area } = useMemo(() => {
    if (values.length < 2) return { line: "", area: "" };
    const pts = values.map((v, i) => [(i / (values.length - 1)) * width, height - 4 - Math.min(1, Math.max(0, v)) * (height - 10)] as const);
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      const mx = (x0 + x1) / 2;
      d += ` C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
    }
    return { line: d, area: `${d} L${width},${height} L0,${height} Z` };
  }, [values, width, height]);
  const id = useMemo(() => `sp-${Math.random().toString(36).slice(2, 8)}`, []);
  if (!line) return <div style={{ height }} />;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.4" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** One thin bar showing the average share of each emotion. */
export function MixBar({ means, height = 6 }: { means: Record<EmotionKey, number>; height?: number }) {
  const total = EMOTIONS.reduce((a, e) => a + (means[e.key] ?? 0), 0) || 1;
  return (
    <div className="mixbar" style={{ height }} role="img" aria-label="Average emotion mix">
      {EMOTIONS.map((e) => (
        <span key={e.key} style={{ flexGrow: Math.max((means[e.key] ?? 0) / total, 0.01), background: `var(${e.cssVar})` }} title={`${e.label} ${pct(means[e.key] ?? 0)}`} />
      ))}
    </div>
  );
}

export function SessionTile({ s, index = 0 }: { s: SessionCard; index?: number }) {
  const tone = s.summary.n_alerts > 0 ? "alert" : s.summary.trend === "escalating" ? "warn" : "calm";
  return (
    <motion.div layout initial={{ opacity: 0, y: 26, filter: "blur(8px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} exit={{ opacity: 0, scale: 0.96, filter: "blur(6px)" }} transition={{ duration: 0.65, ease: EASE, delay: Math.min(index * 0.05, 0.4) }}>
      <Link to={`/app/session/${s.id}`} className="glass session-tile">
        <div className="tile-top">
          <div style={{ minWidth: 0 }}>
            <h3 className="tile-title">{s.title}</h3>
            <p className="muted small mono tile-meta"><span>{fmtDate(s.created)}</span><span>{fmtDuration(s.duration)}</span></p>
          </div>
          {s.is_sample ? <span className="chip badge-sample">Sample</span> : s.tag ? <span className="chip">{s.tag}</span> : null}
        </div>
        <div className="tile-spark"><Sparkline values={s.summary.sparkline} /></div>
        <MixBar means={s.summary.emotion_means} />
        <div className="tile-foot">
          <span className={`tone tone-${tone}`}><i />{s.summary.n_alerts === 0 ? "No alerts" : `${s.summary.n_alerts} alert${s.summary.n_alerts === 1 ? "" : "s"}`}</span>
          <span className="muted small">{TREND_LABEL[s.summary.trend]}</span>
          <span className="muted small mono">peak {pct(s.summary.peak)}</span>
        </div>
      </Link>
    </motion.div>
  );
}

/** A circular progress ring with a percentage inside. */
export function ProgressRing({ value, size = 168, children }: { value: number; size?: number; children?: React.ReactNode }) {
  const r = size / 2 - 8;
  const c = 2 * Math.PI * r;
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: "rotate(-90deg)" }}>
        <defs>
          <linearGradient id="ring-g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--accent-2)" />
            <stop offset="1" stopColor="var(--accent)" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--line)" strokeWidth="3" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="url(#ring-g)" strokeWidth="3.5" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - Math.min(1, Math.max(0, value)))} style={{ transition: "stroke-dashoffset 0.6s cubic-bezier(0.22, 1, 0.36, 1)", filter: "drop-shadow(0 0 8px var(--accent-glow))" }} />
      </svg>
      <div className="ring-inner">{children}</div>
    </div>
  );
}
