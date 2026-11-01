import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { EASE, PageTransition, Reveal } from "../components/Chrome";
import { IconArrow, IconCheck, IconFlag } from "../components/Icons";
import { useModels } from "../hooks/useModels";
import { EMOTIONS } from "../lib/emotions";
import type { ModelCard, ModelStats } from "../lib/types";

const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);

function MetricBar({ label, value, shown, max = 1, color }: { label: string; value: number; shown: string; max?: number; color?: string }) {
  return (
    <div className="metric-row">
      <span>{label}</span>
      <span className="metric-track">
        <motion.span initial={{ width: 0 }} whileInView={{ width: `${Math.min(1, Math.max(0, value / max)) * 100}%` }} viewport={{ once: true }}
          transition={{ duration: 1.1, ease: EASE, delay: 0.15 }} style={color ? { background: color } : undefined} />
      </span>
      <b className="mono tabular">{shown}</b>
    </div>
  );
}

function Bullets({ items, kind }: { items: string[]; kind: "good" | "warn" }) {
  return (
    <ul className={`bullets ${kind}`}>
      {items.map((t) => <li key={t}>{kind === "good" ? <IconCheck /> : <IconFlag />}<span>{t}</span></li>)}
    </ul>
  );
}

function Details({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="details">
      {rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
    </dl>
  );
}

function TimelineStats({ s }: { s: ModelStats }) {
  return (
    <div className="stat-block">
      {s.ccc && (
        <div className="headline-metric">
          <span className="eyebrow">Tracks the emotional arc</span>
          <b className="display">{f2(s.ccc.mean)}<small> ± {f2(s.ccc.std)}</small></b>
          <MetricBar label="CCC" value={s.ccc.mean} shown={f3(s.ccc.mean)} />
        </div>
      )}
      <div className="mini-metrics">
        {s.rmse !== undefined && <div><span className="eyebrow">RMSE</span><b className="mono">{f3(s.rmse)}</b></div>}
        {s.mae !== undefined && <div><span className="eyebrow">MAE</span><b className="mono">{f3(s.mae)}</b></div>}
        {s.alert_f1 !== undefined && <div><span className="eyebrow">Alert F1</span><b className="mono">{f2(s.alert_f1)}</b></div>}
        {s.trend_accuracy !== undefined && <div><span className="eyebrow">Trend</span><b className="mono">{Math.round(s.trend_accuracy * 100)}%</b></div>}
      </div>
      {s.per_emotion_ccc && (
        <div className="per-emotion">
          <span className="eyebrow">CCC by emotion</span>
          {EMOTIONS.map((e) => <MetricBar key={e.key} label={e.label} value={s.per_emotion_ccc![e.key]} shown={f2(s.per_emotion_ccc![e.key])} color={`var(${e.cssVar})`} />)}
        </div>
      )}
    </div>
  );
}

function TimelineModelCard({ m, onUse, delay }: { m: ModelCard; onUse: (id: string) => void; delay: number }) {
  return (
    <Reveal delay={delay} className={`glass model-card ${m.active ? "is-active" : ""}`}>
      <div className="model-top">
        <div>
          <h3 className="display">{m.name}</h3>
          <p className="muted">{m.tagline}</p>
        </div>
        {m.active ? <span className="chip on"><IconCheck /> In use</span> : <span className="chip">Available</span>}
      </div>
      {m.stats ? <TimelineStats s={m.stats} /> : <p className="muted small">Not trained here, so there are no measured results.</p>}
      <p className="model-desc">{m.description}</p>
      <div className="pros-cons">
        <div><span className="eyebrow">Good at</span><Bullets items={m.strengths} kind="good" /></div>
        <div><span className="eyebrow">Keep in mind</span><Bullets items={m.limits} kind="warn" /></div>
      </div>
      <Details rows={m.details} />
      <button className={`btn ${m.active ? "btn-ghost" : "btn-primary"} btn-sm`} onClick={() => onUse(m.id)} disabled={m.active}>
        {m.active ? "Using this model" : <>Use this model <IconArrow className="arrow" /></>}
      </button>
    </Reveal>
  );
}

function Comparison({ models }: { models: ModelCard[] }) {
  const rows = models.filter((m) => m.stats?.ccc);
  const best = (get: (m: ModelCard) => number, higher: boolean) => {
    const vals = rows.map(get);
    return higher ? Math.max(...vals) : Math.min(...vals);
  };
  const cols: { label: string; get: (m: ModelCard) => number; fmt: (v: number) => string; higher: boolean }[] = [
    { label: "CCC", get: (m) => m.stats!.ccc!.mean, fmt: f3, higher: true },
    { label: "RMSE", get: (m) => m.stats!.rmse ?? 0, fmt: f3, higher: false },
    { label: "MAE", get: (m) => m.stats!.mae ?? 0, fmt: f3, higher: false },
    { label: "Alert F1", get: (m) => m.stats!.alert_f1 ?? 0, fmt: f2, higher: true },
    { label: "Trend", get: (m) => (m.stats!.trend_accuracy ?? 0) * 100, fmt: (v) => `${Math.round(v)}%`, higher: true },
  ];
  return (
    <div className="table-wrap">
      <table className="results-table compare">
        <thead><tr><th>Model</th>{cols.map((c) => <th key={c.label}>{c.label}</th>)}<th>Parameters</th><th>Size</th></tr></thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.id} className={m.active ? "row-active" : ""}>
              <td>{m.name}{m.active && <span className="chip on tiny">In use</span>}</td>
              {cols.map((c) => {
                const v = c.get(m);
                return <td key={c.label} className={`mono tabular ${v === best(c.get, c.higher) ? "best" : ""}`}>{c.fmt(v)}</td>;
              })}
              <td className="mono tabular">{(m.params_m ?? 0).toFixed(2)} M</td>
              <td className="mono tabular">{m.size_mb} MB</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">Highlighted cells are the best in each column. All figures come from five-fold testing on sessions the models never saw during training.</p>
    </div>
  );
}

function PipelineFlow({ models, active, encoder }: { models: ModelCard[]; active: ModelCard | null; encoder: ModelCard }) {
  const steps = [
    { k: "Recording", v: "Any audio file or microphone take" },
    { k: "Speech passages", v: "Silence trimmed, cut into short passages" },
    { k: encoder.name, v: "Stage 1 · frozen encoder" },
    { k: active?.name ?? "Timeline model", v: `Stage 2 · ${models.length} to choose from`, hot: true },
    { k: "Five curves + alerts", v: "Anger, frustration, joy, sadness, calm" },
  ];
  return (
    <div className="flow">
      {steps.map((s, i) => (
        <Reveal key={s.k} delay={i * 0.08} className={`flow-step glass ${s.hot ? "hot" : ""}`}>
          <b>{s.k}</b>
          <span className="muted small">{s.v}</span>
        </Reveal>
      ))}
    </div>
  );
}

export default function Models() {
  const { data, error, active, choose } = useModels();

  return (
    <PageTransition className="page models-page">
      <header className="page-head">
        <div className="page-head-main">
          <span className="eyebrow">Models</span>
          <h1 className="display page-title">What's <em>under the hood.</em></h1>
          <p className="page-sub">EmotiSense works in two stages: a large pretrained model listens, then a small trained model draws the timeline. Here is each one, how well it did, and how to choose.</p>
        </div>
        <Link to="/app" className="btn btn-primary">Analyze a recording</Link>
      </header>

      {error && <p className="error-note">{error}</p>}
      {!data && !error && <div className="skeleton" style={{ height: 420 }} />}

      {data && (
        <>
          <PipelineFlow models={data.timeline} active={active} encoder={data.encoder} />

          <section className="m-section">
            <Reveal className="m-head">
              <span className="eyebrow">Stage 2 · choose one</span>
              <h2 className="display">Timeline models</h2>
              <p className="muted">All of these read the same descriptions of the audio, so you can switch freely, even between analyses. Your choice is remembered.</p>
            </Reveal>
            {data.timeline.length === 0 ? (
              <div className="glass glass-pad"><p className="muted">No timeline models were found in the models folder. Train one with <span className="kbd">make timeline-all</span> and place the file in <span className="kbd">models/</span>.</p></div>
            ) : (
              <div className="model-grid">
                {data.timeline.map((m, i) => <TimelineModelCard key={m.id} m={m} onUse={choose} delay={i * 0.08} />)}
              </div>
            )}
          </section>

          {data.timeline.some((m) => m.stats?.ccc) && (
            <section className="m-section">
              <Reveal className="m-head">
                <span className="eyebrow">Side by side</span>
                <h2 className="display">How they compare</h2>
              </Reveal>
              <Reveal className="glass glass-pad"><Comparison models={data.timeline} /></Reveal>
            </section>
          )}

          <section className="m-section">
            <Reveal className="m-head">
              <span className="eyebrow">Stage 1 · always the same</span>
              <h2 className="display">The encoder</h2>
            </Reveal>
            <Reveal className="glass model-card wide">
              <div className="model-top"><div><h3 className="display">{data.encoder.name}</h3><p className="muted">{data.encoder.tagline}</p></div><span className="chip">Fixed</span></div>
              <p className="model-desc">{data.encoder.description}</p>
              <div className="pros-cons">
                <div><span className="eyebrow">Good at</span><Bullets items={data.encoder.strengths} kind="good" /></div>
                <div><span className="eyebrow">Keep in mind</span><Bullets items={data.encoder.limits} kind="warn" /></div>
              </div>
              <Details rows={data.encoder.details} />
            </Reveal>
          </section>

          <section className="m-section">
            <Reveal className="m-head">
              <span className="eyebrow">For reference</span>
              <h2 className="display">The fine-tuned classifier</h2>
              <p className="muted">A separate model that the app does not use for timelines. It shows what training the encoder itself achieves.</p>
            </Reveal>
            <Reveal className="glass model-card wide">
              <div className="model-top">
                <div><h3 className="display">{data.classifier.name}</h3><p className="muted">{data.classifier.tagline}</p></div>
                <span className={`chip ${data.classifier.available ? "on" : ""}`}>{data.classifier.available ? "Found on this computer" : "Weights not on this computer"}</span>
              </div>
              <p className="model-desc">{data.classifier.description}</p>
              {data.classifier.stats?.folds && (
                <div className="clf-grid">
                  <div>
                    <span className="eyebrow">Accuracy on each held-out session</span>
                    <div className="clf-bars">
                      {data.classifier.stats.folds.map((f) => (
                        <MetricBar key={f.name} label={f.name} value={f.ua} max={100} shown={`${f.ua.toFixed(1)}%`} />
                      ))}
                      <MetricBar label="Average" value={data.classifier.stats.mean!.ua} max={100} shown={`${data.classifier.stats.mean!.ua.toFixed(1)}%`} color="linear-gradient(90deg, var(--c-joy), var(--accent-2))" />
                    </div>
                    <p className="muted small">Unweighted accuracy over four emotions. Guessing would score 25%. Weighted average: {data.classifier.stats.mean!.wa.toFixed(1)}%.</p>
                  </div>
                  {data.classifier.stats.recipe_comparison && (
                    <div>
                      <span className="eyebrow">{data.classifier.stats.recipe_comparison.label}</span>
                      <table className="results-table">
                        <thead><tr><th>Recipe</th><th>Weighted</th><th>Unweighted</th></tr></thead>
                        <tbody>
                          <tr><td>Plain paper recipe</td><td className="mono tabular">{data.classifier.stats.recipe_comparison.baseline.wa.toFixed(1)}%</td><td className="mono tabular">{data.classifier.stats.recipe_comparison.baseline.ua.toFixed(1)}%</td></tr>
                          <tr><td>With improvements</td><td className="mono tabular best">{data.classifier.stats.recipe_comparison.improved.wa.toFixed(1)}%</td><td className="mono tabular best">{data.classifier.stats.recipe_comparison.improved.ua.toFixed(1)}%</td></tr>
                        </tbody>
                      </table>
                      <p className="muted small">One session and one run, so treat the gap as a hint, not proof.</p>
                    </div>
                  )}
                </div>
              )}
              <div className="pros-cons">
                <div><span className="eyebrow">Good at</span><Bullets items={data.classifier.strengths} kind="good" /></div>
                <div><span className="eyebrow">Keep in mind</span><Bullets items={data.classifier.limits} kind="warn" /></div>
              </div>
              <Details rows={data.classifier.details} />
              {data.classifier.files && data.classifier.files.length > 0 && (
                <p className="muted small">Found: {data.classifier.files.map((f) => `${f.file} (${f.size_mb} MB)`).join(", ")}</p>
              )}
            </Reveal>
          </section>

          <section className="m-section">
            <Reveal className="m-head"><span className="eyebrow">Plain English</span><h2 className="display">What the numbers mean</h2></Reveal>
            <div className="glossary">
              {data.glossary.map((g, i) => (
                <Reveal key={g.term} delay={(i % 3) * 0.07} className="glass gloss"><h4>{g.term}</h4><p className="muted">{g.plain}</p></Reveal>
              ))}
            </div>
          </section>

          <section className="m-section">
            <Reveal className="m-head"><span className="eyebrow">More</span><h2 className="display">Also supported</h2></Reveal>
            <div className="supported">
              {data.supported.map((s, i) => (
                <Reveal key={s.name} delay={(i % 2) * 0.08} className="glass pillar"><h3>{s.name}</h3><p className="muted">{s.note}</p></Reveal>
              ))}
            </div>
            <Reveal className="data-note"><IconFlag /><p className="muted">{data.data_note}</p></Reveal>
          </section>
        </>
      )}
    </PageTransition>
  );
}
