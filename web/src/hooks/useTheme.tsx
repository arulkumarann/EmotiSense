import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { readEmotionColors, type RGB } from "../lib/emotions";
import type { EmotionKey } from "../lib/types";

export type ThemePref = "system" | "light" | "dark";
const KEY = "emotisense-theme";

interface ThemeCtx {
  pref: ThemePref;
  resolved: "light" | "dark";
  setTheme: (pref: ThemePref, origin?: { x: number; y: number }) => void;
  colors: Record<EmotionKey, RGB>;
}

const Ctx = createContext<ThemeCtx | null>(null);

const prefersDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function readPref(): ThemePref {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark" || v === "system") return v;
  } catch {
    /* storage unavailable */
  }
  return "system";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [pref, setPref] = useState<ThemePref>(readPref);
  const [systemDark, setSystemDark] = useState(prefersDark);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setSystemDark(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const resolved: "light" | "dark" = pref === "system" ? (systemDark ? "dark" : "light") : pref;

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", resolved);
    const meta = document.querySelector('meta[name="theme-color"]');
    meta?.setAttribute("content", resolved === "dark" ? "#04060e" : "#f1f5ff");
  }, [resolved]);

  const setTheme = useCallback((next: ThemePref, origin?: { x: number; y: number }) => {
    const nextResolved = next === "system" ? (prefersDark() ? "dark" : "light") : next;
    const apply = () => {
      document.documentElement.setAttribute("data-theme", nextResolved);
      setPref(next);
      try {
        localStorage.setItem(KEY, next);
      } catch {
        /* ignore */
      }
    };
    const root = document.documentElement;
    const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    if (doc.startViewTransition && !reducedMotion() && nextResolved !== root.getAttribute("data-theme")) {
      root.style.setProperty("--tx", `${origin?.x ?? window.innerWidth - 60}px`);
      root.style.setProperty("--ty", `${origin?.y ?? 40}px`);
      doc.startViewTransition(apply);
    } else {
      root.classList.add("theme-fade");
      apply();
      window.setTimeout(() => root.classList.remove("theme-fade"), 600);
    }
  }, []);

  const colors = useMemo(() => readEmotionColors(), [resolved]); // eslint-disable-line react-hooks/exhaustive-deps

  const value = useMemo(() => ({ pref, resolved, setTheme, colors }), [pref, resolved, setTheme, colors]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTheme(): ThemeCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}
