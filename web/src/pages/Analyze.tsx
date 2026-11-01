import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ProgressRing, SessionTile } from "../components/Bits";
import { EASE, PageTransition } from "../components/Chrome";
import { IconArrow, IconClose, IconMic, IconRecord, IconStop, IconUpload, IconWave } from "../components/Icons";
import { useHealth, useSessions } from "../hooks/useData";
import { useModels } from "../hooks/useModels";
import { ModelPicker } from "../components/ModelPicker";
import { useRecorder } from "../hooks/useRecorder";
import { useToast } from "../hooks/useToasts";
import { api, watchJob, ApiError } from "../lib/api";
import { fmtTime } from "../lib/format";
import { ACCEPTED_EXTENSIONS, isAcceptable } from "../lib/wav";
import type { JobState } from "../lib/types";

const STAGES = [
  { key: "decode", label: "Reading the audio" },
  { key: "speech", label: "Finding the speech" },
  { key: "listen", label: "Listening to each passage" },
  { key: "timeline", label: "Tracing the timeline" },
];

const mb = (n: number) => (n / 1048576 >= 1 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

function RecorderOrb({ level, active }: { level: React.MutableRefObject<number>; active: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!active) return undefined;
    let raf = 0;
    const tick = () => {
      const l = level.current;
      if (ref.current) {
        ref.current.style.setProperty("--lv", l.toFixed(3));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active, level]);
  return (
    <div className="rec-orb" ref={ref} aria-hidden="true">
      <i className="r1" /><i className="r2" /><i className="r3" />
      <span className="rec-core"><IconMic /></span>
    </div>
  );
}

export default function Analyze() {
  const navigate = useNavigate();
  const toast = useToast();
  const { health, offline } = useHealth();
  const { sessions } = useSessions();
  const recorder = useRecorder();
  const models = useModels();

  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [tag, setTag] = useState("");
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState<JobState | null>(null);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);

  const tags = useMemo(() => Array.from(new Set((sessions ?? []).map((s) => s.tag).filter((t) => t && t !== "Sample"))).slice(0, 6), [sessions]);
  const samples = (sessions ?? []).filter((s) => s.is_sample);
  const recent = (sessions ?? []).filter((s) => !s.is_sample).slice(0, 3);

  const accept = useCallback((f: File | undefined) => {
    if (!f) return;
    if (!isAcceptable(f)) {
      toast(`That doesn't look like an audio file. Try ${ACCEPTED_EXTENSIONS.slice(0, 6).join(", ")}…`, "error");
      return;
    }
    setFile(f);
    setFailure(null);
    setTitle((t) => t || f.name.replace(/\.[^.]+$/, ""));
  }, [toast]);

  // whole-window drag and drop
  useEffect(() => {
    const over = (e: DragEvent) => { if (e.dataTransfer?.types.includes("Files")) e.preventDefault(); };
    const enter = (e: DragEvent) => { if (e.dataTransfer?.types.includes("Files")) { dragDepth.current++; setDragging(true); } };
    const leave = () => { dragDepth.current = Math.max(0, dragDepth.current - 1); if (dragDepth.current === 0) setDragging(false); };
    const drop = (e: DragEvent) => {
      e.preventDefault();
      dragDepth.current = 0;
      setDragging(false);
      accept(e.dataTransfer?.files?.[0]);
    };
    window.addEventListener("dragover", over);
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, [accept]);

  // a finished recording becomes the file under review
  useEffect(() => {
    if (recorder.status === "done" && recorder.file) {
      accept(recorder.file);
      setTitle((t) => t || `Recording ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`);
    }
  }, [recorder.status, recorder.file, accept]);

  const analyze = async () => {
    if (!file || busy) return;
    setBusy(true);
    setFailure(null);
    setJob(null);
    try {
      const { job_id, session_id } = await api.upload(file, { title: title.trim(), tag: tag.trim(), model: models.data?.active ?? "" });
      const finished = await watchJob(job_id, setJob);
      if (finished.status === "error") throw new Error(finished.error ?? "The analysis failed");
      toast("Analysis complete");
      navigate(`/app/session/${session_id}`);
    } catch (e) {
      setFailure(e instanceof ApiError || e instanceof Error ? e.message : "Something went wrong");
      setBusy(false);
      setJob(null);
    }
  };

  const reset = () => {
    setFile(null);
    setTitle("");
    setFailure(null);
    recorder.reset();
  };

  const modelNotReady = offline || (health && health.model !== "ready");
  const recording = recorder.status === "recording" || recorder.status === "requesting" || recorder.status === "processing";
  const stageIndex = job ? Math.max(0, STAGES.findIndex((s) => s.key === job.stage)) : 0;

  return (
    <PageTransition className="page analyze-page">
      <header className="page-head">
        <div className="page-head-main">
          <span className="eyebrow">Analyze</span>
          <h1 className="display page-title">Bring a <em>recording.</em></h1>
          <p className="page-sub">Drop in a file or record your voice. A few moments later you'll see how the feeling moved, second by second.</p>
        </div>
      </header>

      {models.data && models.data.timeline.length > 0 && !busy && (
        <div className="model-bar">
          <ModelPicker models={models.data.timeline} activeId={models.data.active} onChange={models.choose} label="Model" />
          <p className="muted small">{models.active?.tagline}. <Link to="/app/models" className="link-arrow">How the models differ →</Link></p>
        </div>
      )}

      <AnimatePresence>
        {dragging && (
          <motion.div className="drop-overlay" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}>
            <motion.div initial={{ scale: 0.9, y: 20 }} animate={{ scale: 1, y: 0 }} transition={{ duration: 0.5, ease: EASE }}>
              <IconUpload />
              <h2 className="display">Let go to analyze</h2>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence mode="wait" initial={false}>
        {busy ? (
          <motion.section key="busy" className="glass glass-pad processing" initial={{ opacity: 0, scale: 0.97, filter: "blur(10px)" }} animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }} exit={{ opacity: 0, scale: 0.97, filter: "blur(10px)" }} transition={{ duration: 0.6, ease: EASE }}>
            <ProgressRing value={job?.progress ?? 0.02}>
              <b className="display ring-pct">{Math.round((job?.progress ?? 0) * 100)}<small>%</small></b>
            </ProgressRing>
            <div className="process-copy">
              <span className="eyebrow">Analyzing · {file?.name}</span>
              <h2 className="display">{job?.detail || "Starting"}</h2>
              <ol className="stages">
                {STAGES.map((s, i) => (
                  <li key={s.key} className={i < stageIndex ? "done" : i === stageIndex ? "now" : ""}>
                    <span className="stage-dot" />{s.label}
                  </li>
                ))}
              </ol>
              <p className="muted small">First analysis after starting can take a little longer while the model warms up.</p>
            </div>
          </motion.section>
        ) : file ? (
          <motion.section key="review" className="glass glass-pad review" initial={{ opacity: 0, y: 20, filter: "blur(10px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} exit={{ opacity: 0, y: -10, filter: "blur(8px)" }} transition={{ duration: 0.6, ease: EASE }}>
            <div className="file-card">
              <span className="file-icon"><IconWave /></span>
              <div style={{ minWidth: 0 }}>
                <b className="file-name">{file.name}</b>
                <span className="muted small mono">{mb(file.size)}</span>
              </div>
              <button className="icon-btn" onClick={reset} aria-label="Choose a different file"><IconClose /></button>
            </div>
            {recorder.previewUrl && file === recorder.file && <audio className="preview-audio" controls src={recorder.previewUrl} />}
            <div className="review-fields">
              <div className="field"><label htmlFor="title">Title</label><input id="title" className="input" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Name this session" maxLength={120} onKeyDown={(e) => e.key === "Enter" && void analyze()} /></div>
              <div className="field">
                <label htmlFor="tag">Label (optional)</label>
                <input id="tag" className="input" value={tag} onChange={(e) => setTag(e.target.value)} placeholder="Interview, Meeting, Voice note…" maxLength={40} list="tag-list" />
                <datalist id="tag-list">{tags.map((t) => <option key={t} value={t} />)}</datalist>
              </div>
            </div>
            {failure && <p className="error-note" role="alert">{failure}</p>}
            {modelNotReady && <p className="muted small">The model is still {health?.model === "error" ? "unavailable" : "warming up"}. Your analysis will start as soon as it is ready.</p>}
            <div className="review-actions">
              <button className="btn btn-primary" onClick={analyze}>Analyze <IconArrow className="arrow" /></button>
              <button className="btn btn-ghost" onClick={reset}>Cancel</button>
            </div>
          </motion.section>
        ) : (
          <motion.section key="pick" className="pick-grid" initial={{ opacity: 0, y: 20, filter: "blur(10px)" }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} exit={{ opacity: 0, y: -10, filter: "blur(8px)" }} transition={{ duration: 0.6, ease: EASE }}>
            <div className={`glass dropzone ${dragging ? "over" : ""}`} onClick={() => inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}>
              <input ref={inputRef} type="file" hidden accept={`audio/*,${ACCEPTED_EXTENSIONS.map((e) => `.${e}`).join(",")}`} onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }} />
              <div className="dz-icon"><IconUpload /></div>
              <h2 className="display">Drop audio here</h2>
              <p className="muted">or click to browse your files</p>
              <div className="format-chips">{["WAV", "MP3", "OGG", "FLAC", "M4A", "WEBM"].map((f) => <span className="chip" key={f}>{f}</span>)}</div>
            </div>

            <div className="glass recorder">
              <span className="eyebrow">Or record</span>
              <RecorderOrb level={recorder.level} active={recorder.status === "recording"} />
              <div className="rec-time mono tabular">{fmtTime(recorder.elapsed)}</div>
              {recorder.status === "recording" ? (
                <button className="btn btn-primary" onClick={recorder.stop}><IconStop /> Stop</button>
              ) : (
                <button className="btn btn-ghost" onClick={recorder.start} disabled={recording}><IconRecord style={{ color: "var(--c-anger)" }} /> {recorder.status === "requesting" ? "Waiting for permission…" : recorder.status === "processing" ? "Preparing…" : "Start recording"}</button>
              )}
              {recorder.error && <p className="error-note" role="alert">{recorder.error}</p>}
              <p className="muted small rec-hint">Recording stays on this computer. Speak naturally for a few seconds or longer.</p>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      {!busy && !file && (samples.length > 0 || recent.length > 0) && (
        <section className="strip">
          {recent.length > 0 && (
            <>
              <div className="strip-head"><h3 className="display">Recent</h3><Link to="/app/library" className="eyebrow link-arrow">All sessions →</Link></div>
              <div className="tile-grid">{recent.map((s, i) => <SessionTile key={s.id} s={s} index={i} />)}</div>
            </>
          )}
          {samples.length > 0 && (
            <>
              <div className="strip-head"><h3 className="display">Try a sample</h3><span className="muted small">Real acted dialogues, already analyzed</span></div>
              <div className="tile-grid">{samples.map((s, i) => <SessionTile key={s.id} s={s} index={i} />)}</div>
            </>
          )}
        </section>
      )}
    </PageTransition>
  );
}
