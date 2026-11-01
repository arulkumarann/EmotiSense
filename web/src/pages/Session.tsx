import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { EASE, PageTransition } from "../components/Chrome";
import { IconArrow, IconClose, IconDownload, IconFlag, IconNote, IconPlay, IconTrash } from "../components/Icons";
import { EmotionLegend, MoodOrb, Transport } from "../components/PlayerParts";
import { Timeline } from "../components/Timeline";
import { usePlayer } from "../hooks/usePlayer";
import { useTheme } from "../hooks/useTheme";
import { useToast } from "../hooks/useToasts";
import { api, ApiError } from "../lib/api";
import { EMOTION_BY_KEY, EMOTION_KEYS } from "../lib/emotions";
import { fmtDate, fmtDuration, fmtTime, pct } from "../lib/format";
import { modelLabel } from "../lib/models";
import { exportCsv, exportJson, pickMoments } from "../lib/moments";
import { TREND_LABEL } from "../lib/escalation";
import { buildCurves } from "../lib/series";
import type { EmotionKey, SessionDetail } from "../lib/types";

function useSession(id: string | undefined) {
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setSession(null);
    setError(null);
    api.session(id).then((s) => !cancelled && setSession(s)).catch((e: ApiError) => !cancelled && setError(e.status === 404 ? "This session no longer exists." : e.message));
    return () => {
      cancelled = true;
    };
  }, [id]);
  return { session, setSession, error };
}

export default function SessionPage() {
  const { id } = useParams();
  const { session, setSession, error } = useSession(id);

  if (error) {
    return (
      <PageTransition className="page">
        <div className="empty glass glass-pad">
          <h2 className="display" style={{ fontSize: "2.4rem" }}>Nothing here</h2>
          <p className="lead">{error}</p>
          <Link className="btn btn-primary" to="/app/library">Back to the library</Link>
        </div>
      </PageTransition>
    );
  }
  if (!session) {
    return (
      <PageTransition className="page">
        <div className="skeleton" style={{ height: 56, width: "40%", marginBottom: 24 }} />
        <div className="stage">
          <div className="skeleton" style={{ height: 460 }} />
          <div className="skeleton" style={{ height: 460 }} />
        </div>
      </PageTransition>
    );
  }
  return <SessionView key={session.id} session={session} setSession={setSession} />;
}

function SessionView({ session, setSession }: { session: SessionDetail; setSession: (s: SessionDetail) => void }) {
  const navigate = useNavigate();
  const toast = useToast();
  const { colors } = useTheme();
  const { player, state } = usePlayer(api.audioUrl(session.id), session.duration);
  const [smoothing, setSmoothing] = useState<1 | 3 | 5>(3);
  const curves = useMemo(() => buildCurves(session.chunks, session.scores, smoothing), [session.chunks, session.scores, smoothing]);
  const [visible, setVisible] = useState<Set<EmotionKey>>(() => new Set(EMOTION_KEYS));
  const [showTension, setShowTension] = useState(true);
  const [title, setTitle] = useState(session.title);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [noteText, setNoteText] = useState("");
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const titleRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = titleRef.current;
    if (!el) return undefined;
    const fit = () => { el.style.height = "auto"; el.style.height = `${el.scrollHeight}px`; };
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [title]);
  const moments = useMemo(() => pickMoments(session), [session]);
  const s = session.summary;

  const toggle = useCallback((k: EmotionKey) => {
    setVisible((cur) => {
      const next = new Set(cur);
      if (next.has(k)) {
        if (next.size > 1) next.delete(k);
      } else next.add(k);
      return next;
    });
  }, []);

  const focusNote = useCallback((t?: number) => {
    if (t !== undefined) player.seek(t);
    noteRef.current?.focus();
    noteRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [player]);

  // keyboard: space plays, arrows seek, N writes a note
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest("input, textarea, select, [contenteditable]")) return;
      if (e.code === "Space") { e.preventDefault(); player.toggle(); }
      else if (e.key === "ArrowRight") player.skip(e.shiftKey ? 15 : 5);
      else if (e.key === "ArrowLeft") player.skip(e.shiftKey ? -15 : -5);
      else if (e.key.toLowerCase() === "n") { e.preventDefault(); focusNote(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player, focusNote]);

  const saveTitle = async () => {
    const next = title.trim() || "Untitled";
    setTitle(next);
    if (next === session.title) return;
    try {
      await api.rename(session.id, { title: next });
      setSession({ ...session, title: next });
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const addNote = async () => {
    const text = noteText.trim();
    if (!text) return;
    try {
      const note = await api.addNote(session.id, player.getTime(), text);
      setSession({ ...session, notes: [...session.notes, note].sort((a, b) => a.t - b.t) });
      setNoteText("");
      toast(`Note saved at ${fmtTime(note.t)}`);
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const removeNote = async (noteId: number) => {
    await api.removeNote(noteId).catch((e: Error) => toast(e.message, "error"));
    setSession({ ...session, notes: session.notes.filter((n) => n.id !== noteId) });
  };

  const remove = async () => {
    try {
      await api.remove(session.id);
      toast("Session deleted");
      navigate("/app/library");
    } catch (e) {
      toast((e as Error).message, "error");
    }
  };

  const playMoment = (t: number) => {
    player.seek(Math.max(0, t - 1.5));
    player.play();
  };

  const stats = [
    { label: "Peak tension", value: pct(s.peak), sub: `at ${fmtTime(s.peak_time)}` },
    { label: "Average tension", value: pct(s.mean), sub: `${pct(s.fraction_above)} of the time above the line` },
    { label: "Direction", value: TREND_LABEL[s.trend], sub: s.trend === "escalating" ? "Tension builds over time" : s.trend === "de-escalating" ? "Tension settles over time" : "No strong drift" },
    { label: "Tension alerts", value: String(s.n_alerts), sub: s.n_alerts === 0 ? "A calm session" : `${s.n_alerts === 1 ? "One moment" : `${s.n_alerts} moments`} to revisit` },
  ];

  return (
    <PageTransition className="page session-page">
      <header className="page-head">
        <div className="page-head-main">
          <Link to="/app/library" className="eyebrow back-link">← Library</Link>
          <textarea
            ref={titleRef}
            rows={1}
            className="title-input display"
            value={title}
            onChange={(e) => setTitle(e.target.value.replace(/\n/g, " "))}
            onBlur={saveTitle}
            onKeyDown={(e) => e.key === "Enter" && (e.currentTarget as HTMLTextAreaElement).blur()}
            aria-label="Session title"
            maxLength={120}
          />
          <div className="meta-row">
            {session.tag && !session.is_sample && <span className="chip">{session.tag}</span>}
            {session.is_sample && <span className="chip badge-sample">Sample</span>}
            {session.model && <Link to="/app/models" className="chip" title="The timeline model that analyzed this recording">Model · {modelLabel(session.model.kind)}</Link>}
            <span className="muted mono small">{fmtDuration(session.duration)}</span>
            <span className="muted mono small">{fmtDate(session.created)}</span>
            <span className="muted mono small">{session.filename}</span>
          </div>
        </div>
        <div className="page-actions">
          <div className="menu">
            <button className="btn btn-ghost btn-sm"><IconDownload /> Export</button>
            <div className="menu-pop glass">
              <button onClick={() => window.print()}>Report (print / PDF)</button>
              <button onClick={() => exportCsv(session)}>Timeline (CSV)</button>
              <button onClick={() => exportJson(session)}>Everything (JSON)</button>
            </div>
          </div>
          <AnimatePresence mode="wait" initial={false}>
            {confirmDelete ? (
              <motion.div key="c" className="confirm" initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 10 }}>
                <button className="btn btn-danger btn-sm" onClick={remove}>Delete for good</button>
                <button className="icon-btn" onClick={() => setConfirmDelete(false)} aria-label="Cancel"><IconClose /></button>
              </motion.div>
            ) : (
              <motion.button key="d" className="icon-btn" onClick={() => setConfirmDelete(true)} aria-label="Delete session" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <IconTrash />
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </header>

      <div className="stage">
        <motion.aside className="glass orb-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: EASE, delay: 0.1 }}>
          <span className="eyebrow">Right now</span>
          <MoodOrb chunks={session.chunks} waveform={session.waveform} duration={session.duration} curves={curves} player={player} colors={colors} size={236} />
          <EmotionLegend visible={visible} onToggle={toggle} curves={curves} player={player} />
          <button className={`chip tension-toggle ${showTension ? "on" : ""}`} onClick={() => setShowTension((v) => !v)} aria-pressed={showTension}>
            <span className="tension-swatch" /> Tension line
          </button>
        </motion.aside>

        <motion.section className="glass timeline-card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: EASE, delay: 0.2 }}>
          <div className="timeline-head">
            <span className="eyebrow">Emotion over time</span>
            <div className="chips smooth-chips" role="group" aria-label="Curve smoothing">
              {([[1, "Raw"], [3, "Soft"], [5, "Smooth"]] as const).map(([w, label]) => (
                <button key={w} className={`chip ${smoothing === w ? "on" : ""}`} onClick={() => setSmoothing(w)}>{label}</button>
              ))}
            </div>
            <span className="muted small hint">Click or drag to move · double-click to add a note · <span className="kbd">Space</span> to play</span>
          </div>
          <Timeline session={session} curves={curves} visible={visible} showTension={showTension} tension={s.negative} player={player} colors={colors} onAddNote={(t) => focusNote(t)} />
          <Transport player={player} state={state} duration={session.duration} onAddNote={() => focusNote()} />
        </motion.section>
      </div>

      <section className="stats">
        {stats.map((x, i) => (
          <motion.div key={x.label} className="glass stat" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.35 + i * 0.07 }}>
            <span className="eyebrow">{x.label}</span>
            <b className="display stat-value">{x.value}</b>
            <span className="muted small">{x.sub}</span>
          </motion.div>
        ))}
      </section>

      <div className="two-col">
        <motion.section className="glass glass-pad" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.5 }}>
          <div className="section-head"><IconFlag /><h3>Moments to revisit</h3></div>
          {moments.length === 0 ? <p className="muted">Nothing stands out.</p> : (
            <ul className="moments">
              {moments.map((m) => (
                <li key={m.t}>
                  <button className="moment" onClick={() => playMoment(m.t)}>
                    <span className="moment-play"><IconPlay /></span>
                    <span className="mono tabular">{fmtTime(m.t)}</span>
                    <span className="moment-emotion"><i style={{ background: `var(${EMOTION_BY_KEY[m.emotion].cssVar})` }} />{EMOTION_BY_KEY[m.emotion].label}</span>
                    <span className="moment-bar"><span style={{ width: `${m.tension * 100}%` }} /></span>
                    <span className="mono tabular">{pct(m.tension)}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </motion.section>

        <motion.section className="glass glass-pad" initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, ease: EASE, delay: 0.58 }}>
          <div className="section-head"><IconNote /><h3>Notes</h3></div>
          <div className="note-form">
            <textarea ref={noteRef} className="input" rows={2} placeholder="Write a note for the current moment…" value={noteText} maxLength={500}
              onChange={(e) => setNoteText(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void addNote(); }} />
            <button className="btn btn-primary btn-sm" onClick={addNote} disabled={!noteText.trim()}>Save at <span className="tabular">{fmtTime(state.displayTime)}</span> <IconArrow className="arrow" /></button>
          </div>
          <ul className="notes">
            <AnimatePresence initial={false}>
              {session.notes.map((n) => (
                <motion.li key={n.id} layout initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.35, ease: EASE }}>
                  <button className="note-time mono" onClick={() => player.seek(n.t)}>{fmtTime(n.t)}</button>
                  <p>{n.text}</p>
                  <button className="icon-btn small" onClick={() => removeNote(n.id)} aria-label="Delete note"><IconClose /></button>
                </motion.li>
              ))}
            </AnimatePresence>
            {session.notes.length === 0 && <li className="muted empty-note">No notes yet. Press <span className="kbd">N</span> while listening.</li>}
          </ul>
        </motion.section>
      </div>
    </PageTransition>
  );
}
