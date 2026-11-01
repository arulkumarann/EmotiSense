import { AnimatePresence } from "framer-motion";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { SessionTile } from "../components/Bits";
import { PageTransition } from "../components/Chrome";
import { IconSearch } from "../components/Icons";
import { useSessions } from "../hooks/useData";

type Sort = "new" | "long" | "tense";

export default function Library() {
  const { sessions, error } = useSessions();
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState<string | null>(null);
  const [sort, setSort] = useState<Sort>("new");

  const tags = useMemo(() => Array.from(new Set((sessions ?? []).map((s) => (s.is_sample ? "Sample" : s.tag)).filter(Boolean))), [sessions]);
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = (sessions ?? []).filter((s) => {
      const label = s.is_sample ? "Sample" : s.tag;
      return (!tag || label === tag) && (!q || s.title.toLowerCase().includes(q) || label.toLowerCase().includes(q));
    });
    const by: Record<Sort, (a: (typeof list)[number], b: (typeof list)[number]) => number> = {
      new: (a, b) => b.created.localeCompare(a.created),
      long: (a, b) => b.duration - a.duration,
      tense: (a, b) => b.summary.peak - a.summary.peak,
    };
    return [...list].sort(by[sort]);
  }, [sessions, query, tag, sort]);

  return (
    <PageTransition className="page">
      <header className="page-head">
        <div className="page-head-main">
          <span className="eyebrow">Library</span>
          <h1 className="display page-title">Every <em>session,</em> in one place.</h1>
        </div>
        <Link to="/app" className="btn btn-primary">New analysis</Link>
      </header>

      <div className="toolbar">
        <label className="search">
          <IconSearch />
          <input className="input" placeholder="Search by title or label" value={query} onChange={(e) => setQuery(e.target.value)} />
        </label>
        <div className="chips">
          <button className={`chip ${tag === null ? "on" : ""}`} onClick={() => setTag(null)}>All</button>
          {tags.map((t) => <button key={t} className={`chip ${tag === t ? "on" : ""}`} onClick={() => setTag(tag === t ? null : t)}>{t}</button>)}
        </div>
        <div className="chips sort">
          {([["new", "Newest"], ["long", "Longest"], ["tense", "Most tense"]] as const).map(([k, label]) => (
            <button key={k} className={`chip ${sort === k ? "on" : ""}`} onClick={() => setSort(k)}>{label}</button>
          ))}
        </div>
      </div>

      {error && <p className="error-note">{error}</p>}
      {!sessions && !error && <div className="tile-grid">{[0, 1, 2].map((i) => <div key={i} className="skeleton" style={{ height: 236 }} />)}</div>}
      {sessions && shown.length === 0 && (
        <div className="empty glass glass-pad">
          <h2 className="display" style={{ fontSize: "2.2rem" }}>{sessions.length === 0 ? "No sessions yet" : "No matches"}</h2>
          <p className="lead">{sessions.length === 0 ? "Analyze a recording and it will appear here." : "Try a different search or label."}</p>
          {sessions.length === 0 && <Link to="/app" className="btn btn-primary">Analyze a recording</Link>}
        </div>
      )}
      <div className="tile-grid">
        <AnimatePresence mode="popLayout">
          {shown.map((s, i) => <SessionTile key={s.id} s={s} index={i} />)}
        </AnimatePresence>
      </div>
    </PageTransition>
  );
}
