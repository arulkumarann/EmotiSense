import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { EASE, PageTransition, Reveal } from "../components/Chrome";
import { IconFlag } from "../components/Icons";

const FOLDS = [
  ["Session 1", 72.72, 73.13], ["Session 2", 73.24, 71.73], ["Session 3", 68.14, 67.16], ["Session 4", 67.09, 62.71], ["Session 5", 64.49, 62.31],
] as const;

const TIMELINE = [
  { name: "No context (per-moment only)", ccc: 0.464 },
  { name: "BiLSTM (used here)", ccc: 0.513 },
  { name: "Transformer", ccc: 0.467 },
];

const STEPS = [
  { title: "Find the speech", body: "A voice-activity step trims silence and cuts the recording into passages of a few seconds." },
  { title: "Listen", body: "WavLM-Large, a self-supervised speech model, turns each passage into a rich description of how it sounds." },
  { title: "Follow the arc", body: "A small bidirectional LSTM reads the passages in order, so each moment is judged in the context of the ones around it." },
  { title: "Draw the timeline", body: "Scores for five emotions are smoothed into curves. Tension, the stronger of anger and frustration, drives the alerts." },
];

export default function Method() {
  return (
    <PageTransition className="page">
      <header className="page-head">
        <div className="page-head-main method-hero">
          <span className="eyebrow">Method</span>
          <h1 className="display page-title">Measured, <em>not marketed.</em></h1>
          <p className="lead">How EmotiSense turns sound into a timeline, how well that works, and where it falls short. Everything on this page comes from runs we actually made. <Link to="/app/models" className="link-arrow">Browse every model →</Link></p>
        </div>
      </header>

      <section className="pipeline">
        {STEPS.map((s, i) => (
          <Reveal key={s.title} delay={i * 0.08} className="glass pipe">
            <h4>{s.title}</h4>
            <p className="muted" style={{ fontSize: 14.5 }}>{s.body}</p>
          </Reveal>
        ))}
      </section>

      <div className="charts">
        <Reveal className="glass chart-card">
          <h3>Recognising the emotion</h3>
          <p className="muted small">WavLM-Large fine-tuned on acted dialogues (IEMOCAP, public copy), tested on a session never seen in training. Accuracy over four emotions: neutral, happy, angry, sad.</p>
          <table className="results-table">
            <thead><tr><th>Held-out</th><th>Weighted</th><th>Unweighted</th></tr></thead>
            <tbody>
              {FOLDS.map(([n, wa, ua]) => <tr key={n}><td>{n}</td><td className="mono tabular">{wa.toFixed(1)}%</td><td className="mono tabular">{ua.toFixed(1)}%</td></tr>)}
              <tr><td><b>Average</b></td><td className="mono tabular"><b>69.1%</b></td><td className="mono tabular"><b>67.4%</b></td></tr>
            </tbody>
          </table>
          <p className="muted small">This copy of the dataset has one label per clip and more clips than the standard benchmark, so these figures should not be compared with published numbers.</p>
        </Reveal>

        <Reveal className="glass chart-card" delay={0.1}>
          <h3>Following the arc</h3>
          <p className="muted small">How closely the timeline tracks the labelled emotion over a conversation (concordance correlation, higher is better; 1 is perfect).</p>
          <div style={{ display: "grid", gap: "0.9rem" }}>
            {TIMELINE.map((t, i) => (
              <div className="bar-row" key={t.name} style={{ gridTemplateColumns: "190px minmax(0,1fr) 52px" }}>
                <span>{t.name}</span>
                <span className="track"><motion.span initial={{ width: 0 }} whileInView={{ width: `${t.ccc * 100 * 1.6}%` }} viewport={{ once: true }} transition={{ duration: 1.3, ease: EASE, delay: 0.2 + i * 0.12 }} /></span>
                <span className="mono tabular">{t.ccc.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <p className="muted small">Reading context helps: the BiLSTM beats judging each moment alone. The Transformer did not, on this amount of data.</p>
        </Reveal>
      </div>

      <Reveal className="glass glass-pad" style={{ marginTop: "1.25rem" }}>
        <div className="section-head"><IconFlag /><h3>What to keep in mind</h3></div>
        <ul className="limits">
          {[
            "Scores are indicative. They describe how a voice sounds, not what a person truly feels.",
            "The model learned from acted speech. Everyday recordings, accents and noisy rooms can behave differently.",
            "Speakers are not separated: if two people talk, their voices share one timeline.",
            "Analysis is not live. A two-minute recording takes about 40 seconds on a laptop CPU.",
            "Frustration is the hardest feeling to tell apart from anger; expect it to be the least certain curve.",
          ].map((t) => <li key={t}><IconFlag /><span>{t}</span></li>)}
        </ul>
      </Reveal>
    </PageTransition>
  );
}
