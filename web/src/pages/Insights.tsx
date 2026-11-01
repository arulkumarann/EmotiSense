import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { EASE, PageTransition, CountUp } from "../components/Chrome";
import { EMOTIONS } from "../lib/emotions";
import { fmtDuration, pct } from "../lib/format";
import { useSessions } from "../hooks/useData";
import type { SessionCard } from "../lib/types";

function Donut({ values }: { values: { key: string; color: string; value: number; label: string }[] }) {
  const total = values.reduce((a, v) => a + v.value, 0) || 1;
  const r = 70;
  const circ = 2 * Math.PI * r;
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setReady(true));
    return () => cancelAnimationFrame(id);
  }, []);
  let offset = 0;
  return (
    <div className="donut-wrap">
      <svg width="190" height="190" viewBox="0 0 190 190" role="img" aria-label="Average emotion mix">
        <g transform="rotate(-90 95 95)">
          <circle cx="95" cy="95" r={r} fill="none" stroke="var(--line)" strokeWidth="18" />
          {values.map((v) => {
            const len = (v.value / total) * circ;
            const el = (
              <circle key={v.key} cx="95" cy="95" r={r} fill="none" stroke={v.color} strokeWidth="18" strokeDasharray={`${ready ? Math.max(len - 3, 0) : 0} ${circ}`} strokeDashoffset={-offset} strokeLinecap="round" style={{ transition: "stroke-dasharray 1.4s cubic-bezier(0.22, 1, 0.36, 1)" }} />
            );
            offset += len;
            return el;
          })}
        </g>
      </svg>
      <div className="donut-legend">
        {values.map((v) => (
          <div key={v.key}><i style={{ background: v.color }} /><span>{v.label}</span><b className="mono tabular">{pct(v.value / total)}</b></div>
        ))}
      </div>
    </div>
  );
}

function TensionTrend({ items }: { items: SessionCard[] }) {
  const W = 560;
  const H = 190;
  const pad = 26;
  const sorted = [...items].sort((a, b) => a.created.localeCompare(b.created));
  if (sorted.length < 2) return <p className="muted">Analyze at least two sessions to see how tension changes over time.</p>;
  const x = (i: number) => pad + (i / (sorted.length - 1)) * (W - pad * 2);
  const y = (v: number) => H - pad - v * (H - pad * 2);
  const path = (get: (s: SessionCard) => number) => sorted.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(get(s)).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Peak and average tension across sessions">
      {[0, 0.5, 1].map((v) => <line key={v} x1={pad} x2={W - pad} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeDasharray={v ? "2 6" : undefined} />)}
      <path d={path((s) => s.summary.peak)} fill="none" stroke="var(--c-anger)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" pathLength={1} className="tl-draw" />
      <path d={path((s) => s.summary.mean)} fill="none" stroke="var(--accent-2)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" pathLength={1} className="tl-draw" style={{ animationDelay: "0.2s" }} />
      {sorted.map((s, i) => (
        <g key={s.id}>
          <circle cx={x(i)} cy={y(s.summary.peak)} r="4" fill="var(--c-anger)"><title>{`${s.title}: peak ${pct(s.summary.peak)}`}</title></circle>
          <circle cx={x(i)} cy={y(s.summary.mean)} r="4" fill="var(--accent-2)"><title>{`${s.title}: average ${pct(s.summary.mean)}`}</title></circle>
        </g>
      ))}
      <g fontFamily="var(--font-mono)" fontSize="10" fill="var(--muted)" letterSpacing="0.1em">
        <text x={pad} y={H - 6}>OLDEST</text>
        <text x={W - pad} y={H - 6} textAnchor="end">NEWEST</text>
        <circle cx={W - 150} cy="10" r="4" fill="var(--c-anger)" /><text x={W - 140} y="14">PEAK</text>
        <circle cx={W - 88} cy="10" r="4" fill="var(--accent-2)" /><text x={W - 78} y="14">AVERAGE</text>
      </g>
    </svg>
  );
}

export default function Insights() {
  const { sessions } = useSessions();
  const [withSamples, setWithSamples] = useState(false);
  const hasOwn = (sessions ?? []).some((s) => !s.is_sample);
  const items = useMemo(() => (sessions ?? []).filter((s) => withSamples || !hasOwn || !s.is_sample), [sessions, withSamples, hasOwn]);

  const stats = useMemo(() => {
    const total = items.reduce((a, s) => a + s.duration, 0);
    const weighted = (get: (s: SessionCard) => number) => (total ? items.reduce((a, s) => a + get(s) * s.duration, 0) / total : 0);
    const mix = EMOTIONS.map((e) => ({ key: e.key, label: e.label, color: `var(${e.cssVar})`, value: weighted((s) => s.summary.emotion_means[e.key] ?? 0) }));
    const byTag = new Map<string, { n: number; sum: number }>();
    items.forEach((s) => {
      const t = s.is_sample ? "Sample" : s.tag || "Unlabelled";
      const cur = byTag.get(t) ?? { n: 0, sum: 0 };
      byTag.set(t, { n: cur.n + 1, sum: cur.sum + s.summary.mean });
    });
    return {
      total, mix,
      avgTension: weighted((s) => s.summary.mean),
      alerts: items.reduce((a, s) => a + s.summary.n_alerts, 0),
      top: [...items].sort((a, b) => b.summary.peak - a.summary.peak).slice(0, 5),
      tags: [...byTag.entries()].map(([tag, v]) => ({ tag, n: v.n, mean: v.sum / v.n })).sort((a, b) => b.mean - a.mean),
    };
  }, [items]);

  const kpis = [
    { label: "Sessions", node: <CountUp to={items.length} /> },
    { label: "Audio analyzed", node: <>{fmtDuration(stats.total)}</> },
    { label: "Average tension", node: <CountUp to={stats.avgTension * 100} suffix="%" /> },
    { label: "Tension alerts", node: <CountUp to={stats.alerts} /> },
  ];

  return (
    <PageTransition className="page">
      <header className="page-head">
        <div className="page-head-main">
          <span className="eyebrow">Insights</span>
          <h1 className="display page-title">The <em>bigger</em> picture.</h1>
          <p className="page-sub">What your sessions have in common: how the feeling is distributed, and where tension tends to appear.</p>
        </div>
        {hasOwn && <button className={`chip ${withSamples ? "on" : ""}`} onClick={() => setWithSamples((v) => !v)}>Include samples</button>}
      </header>

      {sessions && items.length === 0 ? (
        <div className="empty glass glass-pad">
          <h2 className="display" style={{ fontSize: "2.2rem" }}>Nothing to summarize yet</h2>
          <p className="lead">Insights appear once you've analyzed a few recordings.</p>
          <Link to="/app" className="btn btn-primary">Analyze a recording</Link>
        </div>
      ) : (
        <>
          <section className="kpis">
            {kpis.map((k, i) => (
              <motion.div key={k.label} className="glass kpi" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: i * 0.07 }}>
                <span className="eyebrow">{k.label}</span>
                <b>{k.node}</b>
              </motion.div>
            ))}
          </section>
          <section className="charts">
            <motion.div className="glass chart-card" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.25 }}>
              <h3>Emotion mix</h3>
              <Donut values={stats.mix} />
            </motion.div>
            <motion.div className="glass chart-card" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.32 }}>
              <h3>Tension across sessions</h3>
              <TensionTrend items={items} />
            </motion.div>
          </section>
          <section className="charts" style={{ gridTemplateColumns: "minmax(0,1.4fr) minmax(0,1fr)" }}>
            <motion.div className="glass chart-card" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.4 }}>
              <h3>Most intense sessions</h3>
              <ul className="rank-list">
                {stats.top.map((s, i) => (
                  <li key={s.id}>
                    <Link to={`/app/session/${s.id}`} className="rank-item">
                      <span className="rank-n">0{i + 1}</span>
                      <span className="rank-title">{s.title}</span>
                      <span className="rank-bar"><span style={{ width: `${s.summary.peak * 100}%` }} /></span>
                      <span className="mono tabular">{pct(s.summary.peak)}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </motion.div>
            <motion.div className="glass chart-card" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.47 }}>
              <h3>By label</h3>
              <div style={{ display: "grid", gap: "0.85rem" }}>
                {stats.tags.map((t) => (
                  <div className="bar-row" key={t.tag}>
                    <span>{t.tag} <span className="muted small">({t.n})</span></span>
                    <span className="track"><motion.span initial={{ width: 0 }} animate={{ width: `${t.mean * 100}%` }} transition={{ duration: 1.2, ease: EASE, delay: 0.6 }} /></span>
                    <span className="mono tabular">{pct(t.mean)}</span>
                  </div>
                ))}
              </div>
              <p className="muted small">Average tension per label.</p>
            </motion.div>
          </section>
        </>
      )}
    </PageTransition>
  );
}
