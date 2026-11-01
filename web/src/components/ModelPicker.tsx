import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { ModelCard } from "../lib/types";
import { IconCheck } from "./Icons";

const Chevron = ({ open }: { open: boolean }) => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ transition: "transform 0.4s cubic-bezier(0.22,1,0.36,1)", transform: open ? "rotate(180deg)" : "none" }}>
    <path d="m6 9 6 6 6-6" />
  </svg>
);

interface Props {
  models: ModelCard[];
  activeId: string | null;
  onChange: (id: string) => void;
  label?: string;
  disabled?: boolean;
}

/** A dropdown that shows each model's name, one-line description and measured accuracy. */
export function ModelPicker({ models, activeId, onChange, label = "Model", disabled }: Props) {
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const active = models.find((m) => m.id === activeId) ?? models[0];

  useEffect(() => {
    if (!open) return undefined;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
      else if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => (c + 1) % models.length); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => (c - 1 + models.length) % models.length); }
      else if (e.key === "Enter") { e.preventDefault(); onChange(models[cursor].id); setOpen(false); }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, cursor, models, onChange]);

  if (!active) return null;
  const toggle = () => {
    if (disabled) return;
    setCursor(Math.max(0, models.findIndex((m) => m.id === active.id)));
    setOpen((o) => !o);
  };

  return (
    <div className="picker" ref={rootRef}>
      <button className={`picker-btn glass ${open ? "open" : ""}`} onClick={toggle} aria-haspopup="listbox" aria-expanded={open} disabled={disabled}>
        <span className="eyebrow">{label}</span>
        <b>{active.name}</b>
        {active.stats?.ccc && <span className="picker-score mono">CCC {active.stats.ccc.mean.toFixed(2)}</span>}
        <Chevron open={open} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div className="picker-pop glass" role="listbox" aria-label="Choose a model"
            initial={{ opacity: 0, y: -8, scale: 0.97, filter: "blur(8px)" }} animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
            exit={{ opacity: 0, y: -6, scale: 0.98, filter: "blur(6px)" }} transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}>
            {models.map((m, i) => (
              <button key={m.id} role="option" aria-selected={m.id === active.id} className={`picker-opt ${i === cursor ? "cursor" : ""} ${m.id === active.id ? "on" : ""}`}
                onMouseEnter={() => setCursor(i)} onClick={() => { onChange(m.id); setOpen(false); }}>
                <span className="picker-opt-main">
                  <b>{m.name}</b>
                  <span className="muted small">{m.tagline}</span>
                </span>
                <span className="picker-opt-meta mono">
                  {m.stats?.ccc && <span>CCC {m.stats.ccc.mean.toFixed(2)}</span>}
                  <span>{m.size_mb} MB</span>
                </span>
                <span className="picker-check">{m.id === active.id && <IconCheck />}</span>
              </button>
            ))}
            <Link to="/app/models" className="picker-foot eyebrow">Compare all models →</Link>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
