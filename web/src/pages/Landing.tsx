import { motion, useInView, useScroll, useSpring, useTransform } from "framer-motion";
import Lenis from "lenis";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Atmosphere, CountUp, EASE, Logo, Reveal, ThemeToggle } from "../components/Chrome";
import { HeroRibbons } from "../components/HeroRibbons";
import { IconArrow, IconChart, IconFlag, IconLibrary, IconLive, IconLock, IconMic, IconNote, IconShield, IconWave } from "../components/Icons";
import { EmotionLegend, MoodOrb } from "../components/PlayerParts";
import { Timeline } from "../components/Timeline";
import { useDemoPlayer } from "../hooks/useDemoPlayer";
import { useTheme } from "../hooks/useTheme";
import { makeDemoSession } from "../lib/demo";
import { EMOTION_KEYS } from "../lib/emotions";
import { buildCurves, seriesPath } from "../lib/series";
import type { EmotionKey } from "../lib/types";

const NAV = [["how", "How it works"], ["features", "Features"], ["privacy", "Privacy"]] as const;

function useLenis() {
  const ref = useRef<Lenis | null>(null);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return undefined;
    const lenis = new Lenis({ duration: 1.2, easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), smoothWheel: true });
    ref.current = lenis;
    let raf = 0;
    const tick = (time: number) => {
      lenis.raf(time);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      lenis.destroy();
      ref.current = null;
    };
  }, []);
  return ref;
}

function Nav({ onJump }: { onJump: (id: string) => void }) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    let raf = 0;
    const on = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setScrolled(window.scrollY > 24));
    };
    window.addEventListener("scroll", on, { passive: true });
    on();
    return () => {
      window.removeEventListener("scroll", on);
      cancelAnimationFrame(raf);
    };
  }, []);
  return (
    <motion.header className={`l-nav ${scrolled ? "scrolled" : ""}`} initial={{ y: -30, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 1, ease: EASE, delay: 0.1 }}>
      <Logo />
      <nav className="l-links" aria-label="Sections">
        {NAV.map(([id, label]) => <button key={id} onClick={() => onJump(id)}>{label}</button>)}
        <Link to="/app/method">Method</Link>
      </nav>
      <div className="l-nav-end">
        <ThemeToggle />
        <Link to="/app" className="btn btn-primary btn-sm">Open the app <IconArrow className="arrow" /></Link>
      </div>
    </motion.header>
  );
}

function ProductPeek() {
  const session = useMemo(makeDemoSession, []);
  const curves = useMemo(() => buildCurves(session.chunks, session.scores), [session]);
  const { colors } = useTheme();
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: "0px 0px -10% 0px" });
  const player = useDemoPlayer(session.duration, inView, 1.1);
  const visible = useMemo(() => new Set<EmotionKey>(EMOTION_KEYS), []);
  return (
    <motion.div ref={ref} className="peek glass" initial={{ opacity: 0, y: 50, scale: 0.96, filter: "blur(16px)" }} animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }} transition={{ duration: 1.3, ease: EASE, delay: 0.55 }}>
      <div className="peek-bar">
        <span className="peek-dots"><i /><i /><i /></span>
        <span className="eyebrow">Live demo · not a real recording</span>
      </div>
      <div className="peek-body">
        <div className="peek-side">
          <MoodOrb chunks={session.chunks} waveform={session.waveform} duration={session.duration} curves={curves} player={player} colors={colors} size={150} />
          <EmotionLegend visible={visible} onToggle={() => undefined} curves={curves} player={player} />
        </div>
        <div className="peek-chart">
          <Timeline session={session} curves={curves} visible={visible} showTension={false} tension={session.summary.negative} player={player} colors={colors} height={310} interactive={false} />
        </div>
      </div>
    </motion.div>
  );
}

function Hero({ onJump }: { onJump: (id: string) => void }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    let raf = 0;
    const on = (e: PointerEvent) => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        el.style.setProperty("--mx", `${e.clientX}px`);
        el.style.setProperty("--my", `${e.clientY}px`);
      });
    };
    window.addEventListener("pointermove", on, { passive: true });
    return () => window.removeEventListener("pointermove", on);
  }, []);
  const word = (text: string, delay: number, cls?: string) => (
    <span className="word-mask"><motion.span className={cls} initial={{ y: "110%", rotate: 3 }} animate={{ y: 0, rotate: 0 }} transition={{ duration: 1.2, ease: EASE, delay }}>{text}</motion.span></span>
  );
  return (
    <section className="hero" ref={ref}>
      <div className="hero-spot" aria-hidden="true" />
      <HeroRibbons className="hero-ribbons" />
      <div className="hero-copy">
        <motion.span className="eyebrow" initial={{ opacity: 0, x: -14 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 1, ease: EASE, delay: 0.3 }}>Speech emotion timeline</motion.span>
        <h1 className="hero-title">
          <em>{word("Feel", 0.4, "grad-text")}</em>
          <span className="line-serif">{word("the shape of", 0.52)}</span>
          <span className="line-sans">{word("every voice.", 0.64)}</span>
        </h1>
        <motion.p className="lead" initial={{ opacity: 0, y: 20, filter: "blur(8px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} transition={{ duration: 1, ease: EASE, delay: 0.95 }}>
          Drop in a recording and watch anger, frustration, joy, sadness and calm rise and fall, second by second, in step with the sound.
        </motion.p>
        <motion.div className="hero-cta" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1, ease: EASE, delay: 1.1 }}>
          <Link to="/app" className="btn btn-primary">Analyze a recording <IconArrow className="arrow" /></Link>
          <button className="btn btn-ghost" onClick={() => onJump("how")}>See how it works</button>
        </motion.div>
      </div>
      <div className="hero-visual"><ProductPeek /></div>
      <motion.div className="hero-stats" initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1, ease: EASE, delay: 1.3 }}>
        {[["5", "Emotions tracked"], ["~40 s", "For two minutes of audio"], ["100%", "On your own machine"], ["6", "Audio formats"]].map(([n, l]) => (
          <div key={l}><b className="display">{n}</b><span className="eyebrow">{l}</span></div>
        ))}
      </motion.div>
    </section>
  );
}

function Contrast() {
  const session = useMemo(makeDemoSession, []);
  const curves = useMemo(() => buildCurves(session.chunks, session.scores), [session]);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -20% 0px" });
  const paths = useMemo(() => {
    const px = (x: number) => (x / session.duration) * 520;
    const py = (v: number) => 200 - v * 180;
    return (["frustration", "anger", "happiness", "neutral"] as EmotionKey[]).map((k) => ({ k, d: seriesPath(curves.series[k], px, py) }));
  }, [curves, session.duration]);
  return (
    <section className="l-section" id="why">
      <Reveal className="l-head">
        <span className="eyebrow">01 · Why a timeline</span>
        <h2 className="display">One label <em>flattens</em> a conversation.</h2>
        <p className="lead">A person can start steady, tighten, and soften again inside a minute. A single verdict says "frustrated" and hides all of it. EmotiSense keeps the whole arc.</p>
      </Reveal>
      <div className="contrast" ref={ref}>
        <Reveal className="glass contrast-card">
          <span className="eyebrow">A single label</span>
          <div className="verdict"><span className="dot" style={{ background: "var(--c-frustration)", boxShadow: "0 0 18px var(--c-frustration)" }} /><b className="display">Frustrated</b></div>
          <p className="muted">One word for sixty-four seconds. No start, no turning point, no way to tell if it got better.</p>
        </Reveal>
        <Reveal className="glass contrast-card wide" delay={0.12}>
          <span className="eyebrow">The same minute, as a timeline</span>
          <svg viewBox="0 0 520 220" width="100%" role="img" aria-label="Example emotion curves">
            {[0.25, 0.5, 0.75].map((v) => <line key={v} x1="0" x2="520" y1={200 - v * 180} y2={200 - v * 180} stroke="var(--line)" strokeDasharray="2 6" />)}
            {paths.map(({ k, d }, i) => (
              <g key={k}>
                <path d={d} fill="none" stroke={`var(--c-${k === "happiness" ? "joy" : k === "neutral" ? "calm" : k})`} strokeWidth="5" opacity="0.4" style={{ filter: "blur(5px)" }} pathLength={1} className={inView ? "tl-draw" : ""} />
                <path d={d} fill="none" stroke={`var(--c-${k === "happiness" ? "joy" : k === "neutral" ? "calm" : k})`} strokeWidth="2.2" strokeLinecap="round" pathLength={1} className={inView ? "tl-draw" : ""} style={{ animationDelay: `${i * 0.18}s`, opacity: inView ? 1 : 0 }} />
              </g>
            ))}
          </svg>
          <p className="muted">Calm at the start, tension building, a sharp peak, then ease and warmth. That is the story.</p>
        </Reveal>
      </div>
    </section>
  );
}

const STEPS = [
  ["Find the speech", "Silence is trimmed and the recording is cut into passages a few seconds long."],
  ["Listen", "A self-supervised speech model, WavLM-Large, describes how each passage sounds."],
  ["Follow the arc", "A small recurrent network reads the passages in order, so every moment is judged in context."],
  ["Draw the timeline", "Five emotions become smooth curves, synced to the audio, with tension alerts on top."],
];

function How() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 75%", "end 55%"] });
  const progress = useSpring(scrollYProgress, { stiffness: 90, damping: 24, mass: 0.4 });
  const width = useTransform(progress, [0, 1], ["0%", "100%"]);
  return (
    <section className="l-section" id="how">
      <Reveal className="l-head">
        <span className="eyebrow">02 · How it works</span>
        <h2 className="display">From sound to <em>story,</em> in four steps.</h2>
      </Reveal>
      <div className="steps" ref={ref}>
        <div className="steps-line" aria-hidden="true"><motion.span style={{ width }} /></div>
        {STEPS.map(([title, body], i) => (
          <Reveal key={title} delay={i * 0.1} className="step">
            <span className="step-n mono">0{i + 1}</span>
            <h3 className="display">{title}</h3>
            <p className="muted">{body}</p>
          </Reveal>
        ))}
      </div>
    </section>
  );
}

const FEATURES = [
  { icon: IconMic, title: "Record or drop", body: "Speak into your microphone, or drop WAV, MP3, OGG, FLAC or M4A. Files of any length.", span: "s-3", chips: ["WAV", "MP3", "OGG", "FLAC", "M4A"] },
  { icon: IconWave, title: "A timeline that moves with the sound", body: "Press play and a glowing playhead glides across curves, waveform and a colour ribbon. Click or drag anywhere to jump.", span: "s-3 tall" },
  { icon: IconFlag, title: "Tension alerts", body: "Rising anger or frustration is flagged where it happens. Tune how sensitive it is and see the effect instantly.", span: "s-2" },
  { icon: IconNote, title: "Moments and notes", body: "Jump straight to the most intense passages, and pin your own notes to the exact second.", span: "s-2" },
  { icon: IconLibrary, title: "A library you can search", body: "Every session, with a little graph of its mood. Filter by label, sort by tension.", span: "s-2" },
  { icon: IconChart, title: "Insights", body: "See the overall mix of feelings and how tension shifts across everything you've analyzed.", span: "s-3" },
  { icon: IconLive, title: "Live replay", body: "Watch several sessions unfold side by side, orbs shifting colour as the feeling changes.", span: "s-3" },
];

function Features() {
  return (
    <section className="l-section" id="features">
      <Reveal className="l-head">
        <span className="eyebrow">03 · Everything in one place</span>
        <h2 className="display">Made to be <em>used,</em> not admired.</h2>
      </Reveal>
      <div className="bento">
        {FEATURES.map((f, i) => (
          <Reveal key={f.title} delay={(i % 3) * 0.08} className={`glass feature ${f.span}`}>
            <span className="feature-icon"><f.icon /></span>
            <h3 className="display">{f.title}</h3>
            <p className="muted">{f.body}</p>
            {f.chips && <div className="format-chips" style={{ justifyContent: "flex-start" }}>{f.chips.map((c) => <span className="chip" key={c}>{c}</span>)}</div>}
          </Reveal>
        ))}
      </div>
    </section>
  );
}

function Privacy() {
  return (
    <section className="l-section privacy" id="privacy">
      <Reveal className="l-head center">
        <span className="eyebrow">04 · Private by design</span>
        <h2 className="display big">Your voice never <em>leaves the room.</em></h2>
        <p className="lead">The model runs on your own computer. There are no accounts, no uploads and no servers to trust.</p>
      </Reveal>
      <div className="pillars">
        {[[IconShield, "Runs locally", "Analysis happens on your machine, on your CPU or GPU."], [IconLock, "Nothing uploaded", "Recordings and results are stored in a folder you control."], [IconMic, "Yours to delete", "Remove any session and its audio in one click."]].map(([Icon, t, b], i) => {
          const I = Icon as typeof IconShield;
          return (
            <Reveal key={t as string} delay={i * 0.1} className="glass pillar">
              <I /><h3>{t as string}</h3><p className="muted">{b as string}</p>
            </Reveal>
          );
        })}
      </div>
    </section>
  );
}

function Proof() {
  return (
    <section className="l-section" id="proof">
      <Reveal className="l-head">
        <span className="eyebrow">05 · Measured, not marketed</span>
        <h2 className="display">Honest numbers, <em>from real runs.</em></h2>
      </Reveal>
      <div className="proof">
        {[
          { n: 67, s: "%", d: "Balanced accuracy recognising four emotions on sessions the model had never heard." },
          { n: 0.51, s: "", dec: 2, d: "How closely the timeline follows the labelled arc, against 0.46 without reading context." },
          { n: 40, s: " s", d: "To analyze two minutes of audio on an ordinary laptop processor." },
        ].map((p, i) => (
          <Reveal key={p.d} delay={i * 0.1} className="glass proof-card">
            <b className="display"><CountUp to={p.n} decimals={p.dec ?? 0} suffix={p.s} /></b>
            <p className="muted">{p.d}</p>
          </Reveal>
        ))}
      </div>
      <Reveal className="proof-foot"><Link to="/app/method" className="btn btn-ghost">Read the full method <IconArrow className="arrow" /></Link></Reveal>
    </section>
  );
}

function Closer() {
  return (
    <section className="closer">
      <Reveal>
        <h2 className="display big">Bring a <em className="grad-text">recording.</em></h2>
        <p className="lead" style={{ margin: "1.2rem auto 2.2rem", textAlign: "center" }}>It takes under a minute to see how a conversation really moved.</p>
        <Link to="/app" className="btn btn-primary">Start analyzing <IconArrow className="arrow" /></Link>
      </Reveal>
    </section>
  );
}

export default function Landing() {
  const lenis = useLenis();
  const jump = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (lenis.current) lenis.current.scrollTo(el, { offset: -70, duration: 1.4 });
    else el.scrollIntoView({ behavior: "smooth" });
  };
  return (
    <div className="landing">
      <Atmosphere />
      <Nav onJump={jump} />
      <main>
        <Hero onJump={jump} />
        <Contrast />
        <How />
        <Features />
        <Privacy />
        <Proof />
        <Closer />
      </main>
      <footer className="l-footer">
        <Logo />
        <p className="muted small">A speech emotion timeline, built on WavLM. Scores describe how a voice sounds, not what a person feels.</p>
        <nav><Link to="/app">Analyze</Link><Link to="/app/library">Library</Link><Link to="/app/method">Method</Link></nav>
      </footer>
    </div>
  );
}
