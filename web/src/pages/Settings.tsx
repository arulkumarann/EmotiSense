import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { EASE, PageTransition } from "../components/Chrome";
import { useSessions } from "../hooks/useData";
import { useTheme, type ThemePref } from "../hooks/useTheme";
import { useToast } from "../hooks/useToasts";
import { api } from "../lib/api";
import { DEFAULT_SETTINGS, TREND_LABEL, analyzeTension } from "../lib/escalation";
import { fmtTime, pct } from "../lib/format";
import type { EscalationSettings, SessionDetail } from "../lib/types";

const THEMES: { key: ThemePref; label: string; swatch: string }[] = [
  { key: "dark", label: "Dark", swatch: "swatch-dark" },
  { key: "light", label: "Light", swatch: "swatch-light" },
  { key: "system", label: "Match system", swatch: "swatch-system" },
];

const CONTROLS: { key: keyof EscalationSettings; label: string; help: string; min: number; max: number; step: number; fmt: (v: number) => string }[] = [
  { key: "alert_threshold", label: "Alert level", help: "Tension above this line raises an alert.", min: 0.3, max: 0.95, step: 0.01, fmt: pct },
  { key: "rise_delta", label: "Sudden rise", help: "A jump this big within the window also raises an alert.", min: 0.1, max: 0.7, step: 0.01, fmt: pct },
  { key: "rise_window_s", label: "Rise window", help: "How far back a sudden rise is measured.", min: 5, max: 60, step: 1, fmt: (v) => `${v} s` },
  { key: "rise_floor", label: "Rise floor", help: "Ignore sudden rises that stay below this level.", min: 0.1, max: 0.7, step: 0.01, fmt: pct },
  { key: "smooth_window", label: "Smoothing", help: "Passages averaged together before judging tension.", min: 1, max: 9, step: 1, fmt: (v) => String(v) },
  { key: "merge_gap_s", label: "Merge alerts", help: "Alerts closer than this become one.", min: 0, max: 30, step: 1, fmt: (v) => `${v} s` },
];

function Preview({ session, cfg }: { session: SessionDetail; cfg: EscalationSettings }) {
  const W = 620, H = 200, pad = 18;
  const result = useMemo(() => analyzeTension(
    session.chunks.map((c) => c.start), session.chunks.map((c) => c.end), session.scores.anger, session.scores.frustration, cfg,
  ), [session, cfg]);
  const x = (t: number) => pad + (t / session.duration) * (W - pad * 2);
  const y = (v: number) => H - pad - v * (H - pad * 2);
  const pts = result.negative.map((v, i) => [x((session.chunks[i].start + session.chunks[i].end) / 2), y(v)] as const);
  const line = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
  return (
    <div className="preview-box glass">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Tension with alert level">
        <defs>
          <linearGradient id="pv-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="var(--accent-2)" stopOpacity="0.35" /><stop offset="1" stopColor="var(--accent-2)" stopOpacity="0" /></linearGradient>
        </defs>
        {result.alerts.map((a, i) => <rect key={i} x={x(a.start) - 3} y={pad} width={Math.max(10, x(a.end) - x(a.start) + 6)} height={H - pad * 2} rx="8" fill="var(--c-anger)" opacity="0.16" />)}
        <line x1={pad} x2={W - pad} y1={y(cfg.alert_threshold)} y2={y(cfg.alert_threshold)} stroke="var(--c-anger)" strokeDasharray="4 5" opacity="0.9" />
        <text x={W - pad} y={y(cfg.alert_threshold) - 6} textAnchor="end" fontFamily="var(--font-mono)" fontSize="10" fill="var(--c-anger)" letterSpacing="0.1em">ALERT LEVEL</text>
        <path d={`${line} L${pts[pts.length - 1][0]},${H - pad} L${pts[0][0]},${H - pad} Z`} fill="url(#pv-fill)" />
        <path d={line} fill="none" stroke="var(--accent-2)" strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" style={{ transition: "d 0.4s" }} />
      </svg>
      <div className="meta-row">
        <span className="chip"><b>{result.n_alerts}</b> alert{result.n_alerts === 1 ? "" : "s"}</span>
        <span className="chip">Peak {pct(result.peak)} at {fmtTime(result.peak_time)}</span>
        <span className="chip">{TREND_LABEL[result.trend]}</span>
      </div>
    </div>
  );
}

export default function Settings() {
  const { pref, setTheme } = useTheme();
  const toast = useToast();
  const { sessions } = useSessions();
  const [saved, setSaved] = useState<EscalationSettings | null>(null);
  const [draft, setDraft] = useState<EscalationSettings>(DEFAULT_SETTINGS);
  const [defaults, setDefaults] = useState<EscalationSettings>(DEFAULT_SETTINGS);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [preview, setPreview] = useState<SessionDetail | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.settings().then((r) => { setSaved(r.values); setDraft(r.values); setDefaults(r.defaults); }).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!previewId && sessions?.length) setPreviewId([...sessions].sort((a, b) => b.summary.peak - a.summary.peak)[0].id);
  }, [sessions, previewId]);
  useEffect(() => {
    if (!previewId) return;
    let cancelled = false;
    api.session(previewId).then((s) => !cancelled && setPreview(s)).catch(() => undefined);
    return () => { cancelled = true; };
  }, [previewId]);

  const dirty = saved !== null && JSON.stringify(saved) !== JSON.stringify(draft);
  const save = async () => {
    setSaving(true);
    try {
      const r = await api.saveSettings(draft);
      setSaved(r.values);
      setDraft(r.values);
      toast("Settings saved. Every session now uses them.");
    } catch (e) {
      toast((e as Error).message, "error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <PageTransition className="page">
      <header className="page-head">
        <div className="page-head-main">
          <span className="eyebrow">Settings</span>
          <h1 className="display page-title">Make it <em>yours.</em></h1>
        </div>
        <div className="page-actions">
          <button className="btn btn-ghost btn-sm" onClick={() => setDraft(defaults)} disabled={JSON.stringify(draft) === JSON.stringify(defaults)}>Reset</button>
          <button className="btn btn-primary btn-sm" onClick={save} disabled={!dirty || saving}>{saving ? "Saving…" : dirty ? "Save changes" : "Saved"}</button>
        </div>
      </header>

      <div className="settings-grid">
        <div style={{ display: "grid", gap: "1.25rem" }}>
          <motion.section className="glass glass-pad" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE }}>
            <div className="section-head"><h3>Appearance</h3></div>
            <div className="theme-options">
              {THEMES.map((t) => (
                <button key={t.key} className={`theme-opt ${pref === t.key ? "on" : ""}`} onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); setTheme(t.key, { x: r.left + r.width / 2, y: r.top + r.height / 2 }); }}>
                  <span className={`theme-swatch ${t.swatch}`} />
                  <b style={{ fontSize: 14 }}>{t.label}</b>
                </button>
              ))}
            </div>
          </motion.section>

          <motion.section className="glass glass-pad" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.1 }}>
            <div className="section-head"><h3>Tension alerts</h3></div>
            <p className="muted small" style={{ marginBottom: "1rem" }}>Tension is the stronger of anger and frustration at each moment. Move a slider and the preview responds instantly.</p>
            {CONTROLS.map((c) => (
              <div className="setting" key={c.key}>
                <div className="setting-head"><b>{c.label}</b><span className="mono">{c.fmt(draft[c.key])}</span></div>
                <input className="slider" type="range" min={c.min} max={c.max} step={c.step} value={draft[c.key]} aria-label={c.label}
                  style={{ ["--fill" as string]: `${((draft[c.key] - c.min) / (c.max - c.min)) * 100}%` }}
                  onChange={(e) => setDraft({ ...draft, [c.key]: Number(e.target.value) })} />
                <span className="muted small">{c.help}</span>
              </div>
            ))}
          </motion.section>
        </div>

        <motion.section className="glass glass-pad" style={{ position: "sticky", top: 84 }} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.2 }}>
          <div className="section-head"><h3>Live preview</h3></div>
          {!sessions?.length ? <p className="muted">Analyze a recording to preview alert settings on it.</p> : (
            <div style={{ display: "grid", gap: "1rem" }}>
              <div className="field">
                <label htmlFor="pv">Preview on</label>
                <select id="pv" className="input" value={previewId ?? ""} onChange={(e) => setPreviewId(e.target.value)}>
                  {sessions.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
                </select>
              </div>
              {preview ? <Preview session={preview} cfg={draft} /> : <div className="skeleton" style={{ height: 260 }} />}
            </div>
          )}
        </motion.section>
      </div>
    </PageTransition>
  );
}
