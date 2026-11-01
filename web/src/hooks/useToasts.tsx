import { AnimatePresence, motion } from "framer-motion";
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";

interface Toast {
  id: number;
  text: string;
  kind: "info" | "error";
}

const Ctx = createContext<(text: string, kind?: Toast["kind"]) => void>(() => undefined);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const next = useRef(1);
  const push = useCallback((text: string, kind: Toast["kind"] = "info") => {
    const id = next.current++;
    setItems((cur) => [...cur.slice(-3), { id, text, kind }]);
    window.setTimeout(() => setItems((cur) => cur.filter((t) => t.id !== id)), kind === "error" ? 6500 : 3800);
  }, []);
  const value = useMemo(() => push, [push]);
  return (
    <Ctx.Provider value={value}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        <AnimatePresence>
          {items.map((t) => (
            <motion.div
              key={t.id}
              className={`toast ${t.kind}`}
              layout
              initial={{ opacity: 0, y: 16, scale: 0.96, filter: "blur(6px)" }}
              animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
              exit={{ opacity: 0, x: 24, filter: "blur(6px)" }}
              transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
            >
              <span className="dot" style={t.kind === "error" ? { background: "var(--danger)", boxShadow: "0 0 12px var(--danger)" } : undefined} />
              {t.text}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </Ctx.Provider>
  );
}

export const useToast = () => useContext(Ctx);
